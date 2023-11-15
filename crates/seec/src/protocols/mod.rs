use crate::circuit::base_circuit::BaseGate;
use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::common::{BitSlice, BitVec};
use crate::executor::{GateOutputs, Input};
use crate::utils::BoxError;
use async_trait::async_trait;
use bitvec::store::BitStore;
use futures::TryFutureExt;
use num_traits::{Pow, WrappingAdd, WrappingMul, WrappingSub};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, RngCore};
use remoc::RemoteSend;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ops::{BitAnd, BitXor, Shl, Shr};
use zappot::util::Block;

#[cfg(feature = "aby2")]
pub mod aby2;
pub mod arithmetic_gmw;
pub mod boolean_gmw;

pub mod mixed_gmw;
#[cfg(feature = "aby2")]
pub mod tensor_aby2;

pub type ShareOf<Gate> = <Gate as self::Gate>::Share;
pub type SimdShareOf<Gate> = <ShareOf<Gate> as Share>::SimdShare;

pub trait Protocol: Send + Sync {
    const SIMD_SUPPORT: bool = false;
    type Msg: RemoteSend + Clone;
    type SimdMsg: RemoteSend + Clone;
    type Gate: Gate;
    type Wire;
    type ShareStorage: ShareStorage<ShareOf<Self::Gate>>;
    /// The type which provides the data needed to evaluate interactive gate.
    /// In the case of normal GMW, this data is multiplication triples.
    type SetupStorage: SetupStorage;

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = Self::Gate>,
        gate_outputs: impl Iterator<Item = ShareOf<Self::Gate>>,
        inputs: impl Iterator<Item = ShareOf<Self::Gate>>,
        preprocessing_data: &mut Self::SetupStorage,
    ) -> Self::Msg;

    fn compute_msg_simd<'e>(
        &self,
        _party_id: usize,
        _interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = &'e SimdShareOf<Self::Gate>>,
        _inputs: impl Iterator<Item = &'e SimdShareOf<Self::Gate>>,
        _preprocessing_data: &mut Self::SetupStorage,
    ) -> Self::SimdMsg {
        unimplemented!("SIMD evaluation not implemented for this protocol")
    }

    fn evaluate_interactive(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = Self::Gate>,
        gate_outputs: impl Iterator<Item = ShareOf<Self::Gate>>,
        own_msg: Self::Msg,
        other_msg: Self::Msg,
        preprocessing_data: &mut Self::SetupStorage,
    ) -> Self::ShareStorage;

    fn evaluate_interactive_simd<'e>(
        &self,
        _party_id: usize,
        _interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = &'e SimdShareOf<Self::Gate>>,
        _own_msg: Self::SimdMsg,
        _other_msg: Self::SimdMsg,
        _preprocessing_data: &mut Self::SetupStorage,
    ) -> Vec<Self::ShareStorage> {
        unimplemented!("SIMD evaluation not implemented for this protocol")
    }

    // TODO i'm not sure if party_id is needed here
    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self,
        _party_id: usize,
        circuit: &ExecutableCircuit<Self::Gate, Idx>,
    ) -> GateOutputs<Self::ShareStorage> {
        let data = circuit
            .gate_counts()
            .map(|(count, simd_size)| match simd_size {
                None => Input::Scalar(Self::ShareStorage::repeat(Default::default(), count)),
                Some(_simd_size) => Input::Simd(vec![Default::default(); count]),
            })
            .collect();
        GateOutputs::new(data)
    }
}

pub trait Gate: Clone + Hash + PartialOrd + PartialEq + Send + Sync + Debug + 'static {
    type Share: Share;
    type DimTy: Dimension;

    fn is_interactive(&self) -> bool;

    fn input_size(&self) -> usize;

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share, Self::DimTy>>;

    fn wrap_base_gate(base_gate: BaseGate<Self::Share, Self::DimTy>) -> Self;

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        inputs: impl Iterator<Item = Self::Share>,
    ) -> Self::Share;

    fn evaluate_non_interactive_simd<'e>(
        &self,
        party_id: usize,
        inputs: impl Iterator<Item = &'e <Self::Share as Share>::SimdShare>,
    ) -> <Self::Share as Share>::SimdShare {
        let inputs: Vec<_> = inputs.collect();
        let simd_len = inputs.first().map(|v| v.len()).unwrap_or(0);
        (0..simd_len)
            .map(|idx| self.evaluate_non_interactive(party_id, inputs.iter().map(|v| v.get(idx))))
            .collect()
    }

    fn is_non_interactive(&self) -> bool {
        !self.is_interactive()
    }
}

pub trait Share:
    Clone + Default + Debug + PartialEq + PartialOrd + Hash + Send + Sync + 'static
{
    type SimdShare: ShareStorage<Self> + Clone + Default + Debug + PartialEq + PartialOrd + Hash;
}

pub trait Wire: Clone + Debug + Send + Sync + 'static {}

impl<W: Clone + Debug + Send + Sync + 'static> Wire for W {}

pub trait ShareStorage<Share>:
    IntoIterator<Item = Share> + FromIterator<Share> + Clone + Default + Debug + Send + Sync
{
    fn len(&self) -> usize;
    fn repeat(val: Share, len: usize) -> Self;
    fn set(&mut self, idx: usize, val: Share);
    fn get(&self, idx: usize) -> Share;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn random<Rng: RngCore>(size: usize, rng: &mut Rng) -> Self
    where
        Standard: Distribution<Share>,
    {
        rng.sample_iter(Standard).take(size).collect()
    }
}

impl<T: BitStore> ShareStorage<bool> for BitVec<T> {
    fn len(&self) -> usize {
        BitVec::len(self)
    }

    fn repeat(val: bool, len: usize) -> Self {
        BitVec::repeat(val, len)
    }

    fn set(&mut self, idx: usize, val: bool) {
        BitSlice::set(self, idx, val)
    }

    fn get(&self, idx: usize) -> bool {
        self[idx]
    }
}

impl<R: Ring> ShareStorage<R> for Vec<R> {
    fn len(&self) -> usize {
        self.len()
    }

    fn repeat(val: R, len: usize) -> Self {
        vec![val; len]
    }

    fn set(&mut self, idx: usize, val: R) {
        self[idx] = val;
    }

    fn get(&self, idx: usize) -> R {
        self[idx].clone()
    }
}

// TODO I'm not sure if this trait is really needed anymore with the current design
pub trait SetupStorage: Default + Sized + Send + Sync {
    fn len(&self) -> usize;
    /// Split of the last `count` mul triples.
    fn split_off_last(&mut self, count: usize) -> Self;

    fn append(&mut self, other: Self);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes the first `count` elements from self and returns them.
    fn remove_first(&mut self, count: usize) -> Self {
        let removed = self.split_off_last(self.len() - count);
        mem::replace(self, removed)
    }
}

pub trait Sharing {
    type Plain: Clone + Default + Debug;
    type Shared: ShareStorage<Self::Plain>;

    fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2];

    fn reconstruct(shares: [Self::Shared; 2]) -> Self::Shared;
}

pub trait Dimension:
    Clone + PartialOrd + Ord + PartialEq + Eq + Hash + Debug + Send + Sync + 'static
{
    /// Size of dimensions.
    fn as_slice(&self) -> &[usize];

    /// Number of dimensions. Zero for scalar values.
    fn ndim(&self) -> usize {
        self.as_slice().len()
    }
}

#[async_trait]
pub trait FunctionDependentSetup<ShareStorage, G, Idx> {
    type Output;
    type Error;

    /// Called in the function-dependent setup phase
    async fn setup(
        &mut self,
        shares: &GateOutputs<ShareStorage>,
        circuit: &ExecutableCircuit<G, Idx>,
    ) -> Result<(), Self::Error>;

    /// Called in the online phase for each layer. This method can be used to interleave
    /// the setup and online phase.
    async fn request_setup_output(&mut self, count: usize) -> Result<Self::Output, Self::Error>;
}

pub struct ErasedError<FDS>(pub FDS);

#[async_trait]
impl<FDS, ShareStorage, G, Idx> FunctionDependentSetup<ShareStorage, G, Idx> for ErasedError<FDS>
where
    ShareStorage: Sync,
    G: Send + Sync,
    Idx: Sync,
    FDS: FunctionDependentSetup<ShareStorage, G, Idx> + Send,
    FDS::Error: Error + Send + Sync + 'static,
{
    type Output = FDS::Output;
    type Error = BoxError;

    async fn setup(
        &mut self,
        shares: &GateOutputs<ShareStorage>,
        circuit: &ExecutableCircuit<G, Idx>,
    ) -> Result<(), Self::Error> {
        self.0
            .setup(shares, circuit)
            .map_err(BoxError::from_err)
            .await
    }

    async fn request_setup_output(&mut self, count: usize) -> Result<Self::Output, Self::Error> {
        self.0
            .request_setup_output(count)
            .map_err(BoxError::from_err)
            .await
    }
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ScalarDim;

impl Dimension for ScalarDim {
    fn as_slice(&self) -> &[usize] {
        &[]
    }
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub struct DynDim {
    dimensions: Vec<usize>,
}

impl Dimension for DynDim {
    fn as_slice(&self) -> &[usize] {
        &self.dimensions
    }
}

// This doesn't really capture a Ring in the mathematic sense, but is enough for our purposes
pub trait Ring:
    WrappingAdd
    + WrappingSub
    + WrappingMul
    + Pow<u32, Output = Self>
    + BitAnd<Output = Self>
    + BitXor<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Share
    + Ord
    + Eq
    + RemoteSend
    + 'static
    + Sized
{
    const BITS: usize;
    const BYTES: usize;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;

    fn from_block(b: Block) -> Self;

    fn get_bit(&self, idx: usize) -> bool {
        let mask = Self::ONE << idx;
        self.clone() & mask != Self::ZERO
    }
}

impl DynDim {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dimensions: dims.to_vec(),
        }
    }
}

macro_rules! impl_ring {
    ($($typ:ty),+) => {
        $(
        impl Ring for $typ {
            const BITS: usize = { Self::BYTES * 8 };
            const BYTES: usize = { mem::size_of::<Self>() };
            const MAX: Self = <$typ>::MAX;
            const ZERO: Self = 0;
            const ONE: Self = 1;

            fn from_block(b: Block) -> Self {
                let bytes = b.to_le_bytes();
                let truncated: [u8; Self::BYTES] = bytes[..Self::BYTES].try_into().unwrap();
                Self::from_le_bytes(truncated)
            }
        }
        )*
    };
}

impl_ring!(u8, u16, u32, u64, u128);
