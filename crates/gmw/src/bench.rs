use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::executor::{Executor, Message};
use crate::mul_triple::{boolean, ErasedError, MTProvider};
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{Gate, Protocol, Share, ShareStorage};
use crate::CircuitBuilder;
use anyhow::{anyhow, Context};
use mpc_channel::util::{Phase, RunResult, Statistics};
use mpc_channel::{sub_channels_for, Channel, Receiver};
use rand::rngs::OsRng;
use serde::Serialize;
use std::error::Error;
use std::fmt::Debug;
use std::net::SocketAddr;
use std::time::Duration;
use zappot::ot_ext::ExtOTMsg;

pub trait BenchProtocol: Protocol + Default + Debug {
    fn insecure_setup() -> Box<
        dyn MTProvider<Output = Self::SetupStorage, Error = Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    >;
    fn ot_setup(
        ch: Channel<Receiver<ExtOTMsg>>,
    ) -> Box<
        dyn MTProvider<Output = Self::SetupStorage, Error = Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    >;
}

impl BenchProtocol for BooleanGmw {
    fn insecure_setup() -> Box<
        dyn MTProvider<Output = Self::SetupStorage, Error = Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    > {
        Box::new(ErasedError(boolean::InsecureMTProvider))
    }

    fn ot_setup(
        ch: Channel<Receiver<ExtOTMsg>>,
    ) -> Box<
        dyn MTProvider<Output = Self::SetupStorage, Error = Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    > {
        let ot_sender = zappot::ot_ext::Sender::default();
        let ot_recv = zappot::ot_ext::Receiver::default();
        let mtp = boolean::OtMTProvider::new(OsRng, ot_sender, ot_recv, ch.0, ch.1);
        Box::new(ErasedError(mtp))
    }
}

pub struct BenchParty<P: Protocol, Idx> {
    id: usize,
    circ: Option<ExecutableCircuit<P::Gate, Idx>>,
    server: Option<SocketAddr>,
    meta: String,
    insecure_setup: bool,
    sleep_after_phase: Duration,
    precompute_layers: bool,
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
            sleep_after_phase: Duration::from_millis(200),
            precompute_layers: true,
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
        self.insecure_setup = insecure;
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

    pub fn repeat(mut self, repeat: usize) -> Self {
        self.repeat = repeat;
        self
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub async fn bench(self) -> anyhow::Result<BenchResult> {
        let server = self.server.unwrap_or("127.0.0.1:7744".parse().unwrap());
        let (mut sender, bytes_written, mut receiver, bytes_read) = match self.id {
            0 => mpc_channel::tcp::listen(&server).await?,
            1 => mpc_channel::tcp::connect_with_timeout(&server, Duration::from_secs(120)).await?,
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

            let mut mtp = if self.insecure_setup {
                P::insecure_setup()
            } else {
                P::ot_setup(ot_ch)
            };
            let mts_needed = circ.interactive_count_times_simd();
            statistics
                .record(Phase::Mts, mtp.precompute_mts(mts_needed))
                .await
                .map_err(|err| anyhow!(err))
                .context("MT precomputation failed")?;

            let mut executor = statistics
                .record(
                    Phase::FunctionDependentSetup,
                    Executor::<P, Idx>::new(circ, self.id, mtp),
                )
                .await
                .context("Failed to create executor")?;

            let fake_inp = P::ShareStorage::repeat(Default::default(), circ.input_count());

            statistics
                .record(
                    Phase::Online,
                    executor.execute(fake_inp, &mut exec_ch.0, &mut exec_ch.1),
                )
                .await
                .context("Failed to execute circuit")?;

            res.push(statistics.into_run_result());
        }

        Ok(BenchResult {
            protocol: format!("{:?}", P::default()),
            metadata: self.meta,
            data: res,
        })
    }
}
