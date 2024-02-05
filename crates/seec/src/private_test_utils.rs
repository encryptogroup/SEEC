//! Private test utilities - Do Not Use!
//!
//! This module is activated by the "_integration_tests" feature and should not be used by
//! downstream code. It can change in any version.
use std::convert::Infallible;
use std::env;
use std::fmt::Debug;
use std::path::Path;

use anyhow::Result;
use bitvec::field::BitField;
use bitvec::order::Lsb0;
use bitvec::prelude::BitSlice;
use bitvec::vec;
use bitvec::view::BitViewSized;
use itertools::Itertools;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use seec_channel::sub_channel;
use tokio::task::spawn_blocking;
use tokio::time::Instant;
use tracing::{debug, info};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::circuit::base_circuit::{BaseGate, Load};
use crate::circuit::ExecutableCircuit;
use crate::circuit::{BaseCircuit, BooleanGate, GateIdx};
use crate::common::BitVec;
use crate::executor::{Executor, Input};
use crate::mul_triple::MTProvider;
use crate::mul_triple::{arithmetic, boolean};
use crate::protocols::arithmetic_gmw::{AdditiveSharing, ArithmeticGmw};
use crate::protocols::boolean_gmw::{BooleanGmw, XorSharing};
use crate::protocols::mixed_gmw::{MixedGmw, MixedShareStorage, MixedSharing};
use crate::protocols::{mixed_gmw, Gate, Protocol, Ring, ScalarDim, Share, Sharing};

pub trait ProtocolTestExt: Protocol + Default {
    type InsecureSetup: MTProvider<Output = Self::SetupStorage, Error = Infallible>
        + Default
        + Clone
        + Send
        + Sync;
}

impl ProtocolTestExt for BooleanGmw {
    type InsecureSetup = boolean::insecure_provider::InsecureMTProvider;
}

impl<R: Ring> ProtocolTestExt for ArithmeticGmw<R> {
    type InsecureSetup = arithmetic::insecure_provider::InsecureMTProvider<R>;
}

impl<R> ProtocolTestExt for MixedGmw<R>
where
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
    type InsecureSetup = mixed_gmw::InsecureMixedSetup<R>;
}

pub fn create_and_tree(depth: u32) -> BaseCircuit {
    let total_nodes = 2_u32.pow(depth);
    let mut layer_count = total_nodes / 2;
    let mut circuit = BaseCircuit::new();

    let mut previous_layer: Vec<_> = (0..layer_count)
        .map(|_| circuit.add_gate(BooleanGate::Base(BaseGate::Input(ScalarDim))))
        .collect();
    while layer_count > 1 {
        layer_count /= 2;
        previous_layer = previous_layer
            .into_iter()
            .tuples()
            .map(|(from_a, from_b)| circuit.add_wired_gate(BooleanGate::And, &[from_a, from_b]))
            .collect();
    }
    debug_assert_eq!(1, previous_layer.len());
    circuit.add_wired_gate(
        BooleanGate::Base(BaseGate::Output(ScalarDim)),
        &[previous_layer[0]],
    );
    circuit
}

/// Initializes tracing subscriber with EnvFilter for usage in tests. This should be the first call
/// in each test, with the returned value being assigned to a variable to prevent dropping.
/// Output can be configured via RUST_LOG env variable as explained
/// [here](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/struct.EnvFilter.html)
///
/// ```ignore
/// use seec::private_test_utils::init_tracing;
/// fn some_test() {
///     let _guard = init_tracing();
/// }
/// ```
pub fn init_tracing() -> tracing::dispatcher::DefaultGuard {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .set_default()
}

#[derive(Debug)]
pub enum TestChannel {
    InMemory,
    Tcp,
}

pub trait IntoShares<S: Sharing> {
    fn into_shares(self) -> (S::Shared, S::Shared);
}

pub trait IntoInput<S: Sharing> {
    fn into_input(self) -> (S::Shared, S::Shared);
}

pub struct ToBool<R>(pub R);

macro_rules! impl_into_shares {
    ($($typ:ty),+) => {
        $(
            impl IntoShares<XorSharing<ThreadRng>> for $typ {
                fn into_shares(self) -> (BitVec<usize>, BitVec<usize>) {
                    let mut a = vec::BitVec::repeat(false, <$typ>::BITS as usize);
                    a.store(self);
                    let [a, b] = XorSharing::new(thread_rng()).share(a);
                    (a, b)
                }
            }

            impl IntoShares<AdditiveSharing<$typ, ThreadRng>> for $typ {
                fn into_shares(self) -> (Vec<$typ>, Vec<$typ>) {
                    let [a, b] = AdditiveSharing::new(thread_rng()).share(vec![self]);
                    (a, b)
                }
            }

            impl IntoShares<MixedSharing<XorSharing<ThreadRng>, AdditiveSharing<$typ, ThreadRng>, $typ>>
                for $typ
            {
                fn into_shares(self) -> (MixedShareStorage<$typ>, MixedShareStorage<$typ>) {
                    static RNG: Lazy<Mutex<ChaCha8Rng>> = Lazy::new(|| {
                        let seed = match env::var("RNG_SEED") {
                            Ok(seed) => seed.parse().expect("failed to parse RNG_SEED env var as u64"),
                            Err(_) => thread_rng().gen()
                        };
                        debug!(seed, "Input sharing rng seed");
                        Mutex::new(ChaCha8Rng::seed_from_u64(seed))
                    });
                    let mut rng = RNG.lock();
                    // let [a, b] = AdditiveSharing::new(ChaCha8Rng::seed_from_u64(65432)).share(vec![self]);
                    let [a, b] = AdditiveSharing::new(&mut *rng).share(vec![self]);
                    (MixedShareStorage::Arith(a), MixedShareStorage::Arith(b))
                }
            }

            impl IntoShares<MixedSharing<XorSharing<ThreadRng>, AdditiveSharing<$typ, ThreadRng>, $typ>> for ToBool<$typ> {
                fn into_shares(self) -> (MixedShareStorage<$typ>, MixedShareStorage<$typ>) {
                    // use xor bool sharing
                    let (a, b) = IntoShares::<XorSharing<ThreadRng>>::into_shares(self.0);
                    (MixedShareStorage::Bool(a), MixedShareStorage::Bool(b))
                }
            }


            impl<T: IntoShares<AdditiveSharing<$typ, ThreadRng>>> IntoInput<AdditiveSharing<$typ, ThreadRng>>
                for T
            {
                fn into_input(self) -> (Vec<$typ>, Vec<$typ>) {
                    self.into_shares()
                }
            }
        )*
    };
}

impl_into_shares!(u8, u16, u32, u64, u128);

impl IntoShares<XorSharing<ThreadRng>> for bool {
    fn into_shares(self) -> (BitVec<usize>, BitVec<usize>)
    where
        BitSlice<u8, Lsb0>: BitField,
    {
        let a = BitVec::repeat(false, 1);
        let b = BitVec::repeat(self, 1);
        (a, b)
    }
}

impl<R> IntoShares<MixedSharing<XorSharing<ThreadRng>, AdditiveSharing<R, ThreadRng>, R>> for bool
where
    R: Ring,
    Standard: Distribution<R>,
{
    fn into_shares(self) -> (MixedShareStorage<R>, MixedShareStorage<R>)
    where
        BitSlice<u8, Lsb0>: BitField,
    {
        let a = BitVec::repeat(false, 1);
        let b = BitVec::repeat(self, 1);
        (MixedShareStorage::Bool(a), MixedShareStorage::Bool(b))
    }
}

impl<T: IntoShares<XorSharing<ThreadRng>>> IntoInput<XorSharing<ThreadRng>> for T {
    fn into_input(self) -> (BitVec<usize>, BitVec<usize>) {
        self.into_shares()
    }
}

impl<S: Sharing, T: IntoShares<S>> IntoInput<S> for (T,) {
    fn into_input(self) -> (S::Shared, S::Shared) {
        self.0.into_shares()
    }
}

impl<S, T1, T2> IntoInput<S> for (T1, T2)
where
    S: Sharing,
    T1: IntoShares<S>,
    T2: IntoShares<S>,
    S::Shared: Extend<S::Plain>,
    S::Shared: IntoIterator<Item = S::Plain>,
{
    fn into_input(self) -> (S::Shared, S::Shared) {
        let (mut p1, mut p2) = self.0.into_shares();
        let second_input = self.1.into_shares();
        p1.extend(second_input.0);
        p2.extend(second_input.1);
        (p1, p2)
    }
}

impl<S, T1, T2, T3> IntoInput<S> for (T1, T2, T3)
where
    S: Sharing,
    T1: IntoShares<S>,
    T2: IntoShares<S>,
    T3: IntoShares<S>,
    S::Shared: Extend<S::Plain>,
    S::Shared: IntoIterator<Item = S::Plain>,
{
    fn into_input(self) -> (S::Shared, S::Shared) {
        let (mut p1, mut p2) = self.0.into_shares();
        let second_input = self.1.into_shares();
        let third_input = self.2.into_shares();
        p1.extend(second_input.0);
        p1.extend(third_input.0);
        p2.extend(second_input.1);
        p2.extend(third_input.1);
        (p1, p2)
    }
}

impl<S, T> IntoInput<S> for Vec<T>
where
    S: Sharing,
    T: IntoShares<S>,
    S::Shared: Extend<S::Plain>,
    S::Shared: IntoIterator<Item = S::Plain>,
{
    fn into_input(self) -> (S::Shared, S::Shared) {
        self.into_iter().fold(
            Default::default(),
            |(mut p1, mut p2): (S::Shared, S::Shared), inp| {
                let (s1, s2) = inp.into_shares();
                p1.extend(s1);
                p2.extend(s2);
                (p1, p2)
            },
        )
    }
}

/// This is kind of cursed...
impl IntoInput<XorSharing<ThreadRng>> for [BitVec<usize>; 2] {
    fn into_input(self) -> (BitVec<usize>, BitVec<usize>) {
        let [a, b] = self;
        (a, b)
    }
}

#[tracing::instrument(skip(inputs))]
pub async fn execute_bristol<I: IntoInput<XorSharing<ThreadRng>>>(
    bristol_file: impl AsRef<Path> + Debug,
    inputs: I,
    channel: TestChannel,
) -> Result<BitVec<usize>> {
    let path = bristol_file.as_ref().to_path_buf();
    let now = Instant::now();
    let bc =
        spawn_blocking(move || BaseCircuit::<BooleanGate, u32>::load_bristol(path, Load::Circuit))
            .await??;
    info!(
        parsing_time = %now.elapsed().as_millis(),
        "Parsing bristol time (ms)"
    );
    let circuit = ExecutableCircuit::DynLayers(bc.into());
    execute_circuit::<BooleanGmw, _, _>(&circuit, inputs, channel).await
}

#[tracing::instrument(skip(circuit, inputs))]
pub async fn execute_circuit<P, Idx, S: Sharing>(
    circuit: &ExecutableCircuit<P::Gate, Idx>,
    inputs: impl IntoInput<S>,
    channel: TestChannel,
) -> Result<S::Shared>
where
    P: ProtocolTestExt<ShareStorage = S::Shared>,
    <P::Gate as Gate>::Share: Share<SimdShare = P::ShareStorage>,
    Idx: GateIdx,
    <P::InsecureSetup as MTProvider>::Error: Debug,
    <P as Protocol>::ShareStorage: Send + Sync,
{
    let mt_provider = P::InsecureSetup::default();
    let (input_a, input_b) = inputs.into_input();
    let mut ex1: Executor<P, Idx> = Executor::new(circuit, 0, mt_provider.clone())
        .await
        .unwrap();
    let mut ex2: Executor<P, Idx> = Executor::new(circuit, 1, mt_provider).await.unwrap();
    let now = Instant::now();
    let (out1, out2) = match channel {
        TestChannel::InMemory => {
            let (mut t1, mut t2) = seec_channel::in_memory::new_pair(2);
            let h1 = ex1.execute(Input::Scalar(input_a), &mut t1.0, &mut t1.1);
            let h2 = ex2.execute(Input::Scalar(input_b), &mut t2.0, &mut t2.1);
            futures::try_join!(h1, h2)?
        }
        TestChannel::Tcp => {
            let (mut t1, mut t2) =
                seec_channel::tcp::new_local_pair::<seec_channel::Receiver<_>>(None).await?;
            let (mut sub_t1, mut sub_t2) = tokio::try_join!(
                sub_channel(&mut t1.0, &mut t1.2, 2),
                sub_channel(&mut t2.0, &mut t2.2, 2)
            )?;
            let h1 = ex1.execute(Input::Scalar(input_a), &mut sub_t1.0, &mut sub_t1.1);
            let h2 = ex2.execute(Input::Scalar(input_b), &mut sub_t2.0, &mut sub_t2.1);
            let out = futures::try_join!(h1, h2)?;
            info!(
                bytes_sent = t1.1.get(),
                bytes_received = t1.3.get(),
                "Tcp communication"
            );
            out
        }
    };
    info!(exec_time = %now.elapsed().as_millis(), "Execution time (ms)");
    let out1 = out1.into_scalar().unwrap();
    let out2 = out2.into_scalar().unwrap();
    Ok(S::reconstruct([out1, out2]))
}
