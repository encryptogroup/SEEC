use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::executor::{Executor, Input, Message};
use crate::mul_triple::storage::MTStorage;
use crate::mul_triple::{boolean, MTProvider};
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{Gate, Protocol, Share, ShareStorage};
use crate::utils::{BoxError, ErasedError};
use crate::CircuitBuilder;
use anyhow::{anyhow, Context};
use mpc_channel::util::{Phase, RunResult, Statistics};
use mpc_channel::{sub_channels_for, Channel, Receiver};
use rand::distributions::{Distribution, Standard};
use rand::rngs::OsRng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::io::BufReader;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;
use zappot::ot_ext::ExtOTMsg;

type DynMTP<P> =
    Box<dyn MTProvider<Output = <P as Protocol>::SetupStorage, Error = BoxError> + Send + 'static>;

pub trait BenchProtocol: Protocol + Default + Debug {
    fn insecure_setup() -> DynMTP<Self>;
    fn ot_setup(ch: Channel<Receiver<ExtOTMsg>>) -> DynMTP<Self>;
    fn stored(path: &Path) -> DynMTP<Self>;
}

impl BenchProtocol for BooleanGmw {
    fn insecure_setup() -> DynMTP<Self> {
        Box::new(ErasedError(boolean::InsecureMTProvider))
    }

    fn ot_setup(ch: Channel<Receiver<ExtOTMsg>>) -> DynMTP<Self> {
        let ot_sender = zappot::ot_ext::Sender::default();
        let ot_recv = zappot::ot_ext::Receiver::default();
        let mtp = boolean::OtMTProvider::new(OsRng, ot_sender, ot_recv, ch.0, ch.1);
        Box::new(ErasedError(mtp))
    }

    fn stored(path: &Path) -> DynMTP<Self> {
        let file = BufReader::new(File::open(path).expect("opening MT file"));
        MTStorage::new(file).into_dyn()
    }
}

pub struct BenchParty<P: Protocol, Idx> {
    id: usize,
    circ: Option<ExecutableCircuit<P::Gate, Idx>>,
    server: Option<SocketAddr>,
    meta: String,
    insecure_setup: bool,
    stored_mts: Option<PathBuf>,
    sleep_after_phase: Duration,
    precompute_layers: bool,
    interleave_setup: bool,
    repeat: usize,
}

#[derive(Debug, Serialize)]
pub struct BenchResult {
    protocol: String,
    metadata: String,
    data: Vec<RunResult>,
}

impl<P, Idx> BenchParty<P, Idx>
where
    P: BenchProtocol,
    Standard: Distribution<<<P as Protocol>::Gate as Gate>::Share>,
    Idx: GateIdx,
    <P::Gate as Gate>::Share: Share<SimdShare = P::ShareStorage>,
{
    pub fn new(id: usize) -> Self {
        Self {
            id,
            circ: None,
            server: None,
            meta: String::new(),
            insecure_setup: false,
            stored_mts: None,
            sleep_after_phase: Duration::from_millis(200),
            precompute_layers: true,
            interleave_setup: false,
            repeat: 1,
        }
    }

    pub fn server(mut self, server: SocketAddr) -> Self {
        self.server = Some(server);
        self
    }

    pub fn explicit_circuit(mut self, circuit: ExecutableCircuit<P::Gate, Idx>) -> Self {
        self.circ = Some(circuit);
        self
    }

    pub fn insecure_setup(mut self, insecure: bool) -> Self {
        assert_eq!(None, self.stored_mts);
        self.insecure_setup = insecure;
        self
    }

    pub fn interleave_setup(mut self, interleave_setup: bool) -> Self {
        self.interleave_setup = interleave_setup;
        self
    }

    /// Default is true
    pub fn precompute_layers(mut self, precompute_layers: bool) -> Self {
        self.precompute_layers = precompute_layers;
        self
    }

    /// Sets the metadata of the `BenchResult` that is returned by `bench()`
    pub fn metadata(mut self, meta: String) -> Self {
        self.meta = meta;
        self
    }

    pub fn sleep_after_phase(mut self, sleep: Duration) -> Self {
        self.sleep_after_phase = sleep;
        self
    }

    pub fn repeat(mut self, repeat: usize) -> Self {
        self.repeat = repeat;
        self
    }

    pub fn stored_mts(mut self, path: &Path) -> Self {
        assert!(!self.insecure_setup);
        self.stored_mts = Some(path.to_path_buf());
        self
    }

    #[tracing::instrument(level = "debug", skip(self))]
    #[allow(clippy::manual_async_fn)] // I want ot force the Send bound here
    pub fn bench(self) -> impl Future<Output = anyhow::Result<BenchResult>> + Send {
        async move {
            let server = self.server.unwrap_or("127.0.0.1:7744".parse().unwrap());
            let (mut sender, bytes_written, mut receiver, bytes_read) = match self.id {
                0 => mpc_channel::tcp::listen(&server).await?,
                1 => {
                    mpc_channel::tcp::connect_with_timeout(&server, Duration::from_secs(120))
                        .await?
                }
                illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
            };

            let mut res = vec![];
            let mut owned_circ;
            for run in 0..self.repeat {
                tracing::debug!(run, "Performing bench run");
                let mut statistics = Statistics::new(bytes_written.clone(), bytes_read.clone())
                    .with_sleep(self.sleep_after_phase)
                    .without_unaccounted(true);

                let (ot_ch, mut exec_ch) = sub_channels_for!(
                    &mut sender,
                    &mut receiver,
                    128,
                    Receiver<ExtOTMsg>,
                    Message<P>
                )
                .await
                .context("Establishing sub channels")?;

                let circ = match &self.circ {
                    Some(circ) => circ,
                    None => {
                        let circ = CircuitBuilder::<P::Gate, Idx>::global_into_circuit();
                        if self.precompute_layers {
                            owned_circ = ExecutableCircuit::DynLayers(circ).precompute_layers();
                            &owned_circ
                        } else {
                            owned_circ = ExecutableCircuit::DynLayers(circ);
                            &owned_circ
                        }
                    }
                };

                let mut mtp = match (self.insecure_setup, &self.stored_mts) {
                    (false, None) => P::ot_setup(ot_ch),
                    (true, None) => P::insecure_setup(),
                    (false, Some(path)) => P::stored(path),
                    (true, Some(_)) => unreachable!("ensure via setters"),
                };
                let mts_needed = circ.interactive_count_times_simd();
                if !self.interleave_setup {
                    statistics
                        .record(Phase::Mts, mtp.precompute_mts(mts_needed))
                        .await
                        .map_err(|err| anyhow!(err))
                        .context("MT precomputation failed")?;
                }

                let mut executor = statistics
                    .record(
                        Phase::FunctionDependentSetup,
                        Executor::<P, Idx>::new(circ, self.id, mtp),
                    )
                    .await
                    .context("Failed to create executor")?;

                let mut rng = ChaCha8Rng::seed_from_u64(42 * self.id as u64);
                let fake_inp = match circ.simd_size(0) {
                    None => Input::Scalar(P::ShareStorage::random(circ.input_count(), &mut rng)),
                    Some(size) => Input::Simd(vec![
                        P::ShareStorage::random(size.get(), &mut rng);
                        circ.input_count()
                    ]),
                };

                let output = statistics
                    .record(
                        Phase::Online,
                        executor.execute(fake_inp, &mut exec_ch.0, &mut exec_ch.1),
                    )
                    .await
                    .context("Failed to execute circuit")?;

                tracing::debug!(id = self.id, ?output);

                res.push(statistics.into_run_result());
            }

            Ok(BenchResult {
                protocol: format!("{:?}", P::default()),
                metadata: self.meta,
                data: res,
            })
        }
    }
}
