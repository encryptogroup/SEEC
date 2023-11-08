use crate::circuit::base_circuit::BaseGate;
use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::executor::{GateOutputs, Input};
use crate::mul_triple::{arithmetic, boolean, MTProvider};
use crate::protocols::arithmetic_gmw::ArithmeticGmw;
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{
    arithmetic_gmw, boolean_gmw, Gate, Protocol, Ring, ScalarDim, SetupStorage, Share,
    ShareStorage, Sharing,
};
use crate::utils::BitVecExt;
use async_trait::async_trait;
use bitvec::vec::BitVec;
use itertools::Itertools;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::iter;
use std::marker::PhantomData;
use tracing::trace;

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct MixedGmw<R>(PhantomData<R>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Msg<R> {
    bool: boolean_gmw::Msg,
    arith: arithmetic_gmw::Msg<R>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MixedShare<R> {
    Bool(bool),
    Arith(R),
}

impl<R> MixedShare<R> {
    pub fn into_bool(self) -> Option<bool> {
        match self {
            MixedShare::Bool(b) => Some(b),
            MixedShare::Arith(_) => None,
        }
    }
    pub fn into_arith(self) -> Option<R> {
        match self {
            MixedShare::Bool(_) => None,
            MixedShare::Arith(r) => Some(r),
        }
    }
}

// TODO Default here is prob wrong
impl<R> Default for MixedShare<R> {
    fn default() -> Self {
        Self::Bool(false)
    }
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum MixedGate<R> {
    Base(BaseGate<MixedShare<R>>),
    Bool(boolean_gmw::BooleanGate),
    Arith(arithmetic_gmw::ArithmeticGate<R>),
}

#[derive(Debug)]
pub struct XorSharing<R: CryptoRng + Rng> {
    rng: R,
}

// TODO, mhh Default will be complicated, iirc it is used during executor setup but the
//  method has no access to whether bool or arith is needed
#[derive(Clone, Debug, Hash, PartialOrd, PartialEq)]
pub enum MixedShareStorage<R: Ring> {
    Bool(<boolean_gmw::BooleanGmw as Protocol>::ShareStorage),
    Arith(<arithmetic_gmw::ArithmeticGmw<R> as Protocol>::ShareStorage),
    Mixed(Vec<MixedShare<R>>),
}

impl<R: Ring> Extend<MixedShare<R>> for MixedShareStorage<R> {
    fn extend<T: IntoIterator<Item = MixedShare<R>>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();
        match self {
            MixedShareStorage::Bool(bv) => {
                bv.extend(iter.map(|e| e.into_bool().unwrap()));
            }
            MixedShareStorage::Arith(v) => {
                v.extend(iter.map(|e| e.into_arith().unwrap()));
            }
            MixedShareStorage::Mixed(v) => {
                v.extend(iter);
            }
        }
    }
}

impl<R: Ring> MixedShareStorage<R> {
    pub fn try_push(&mut self, s: MixedShare<R>) {
        match (self, s) {
            (Self::Bool(bv), MixedShare::Bool(b)) => {
                bv.push(b);
            }
            (Self::Arith(v), MixedShare::Arith(r)) => {
                v.push(r);
            }
            (Self::Mixed(v), s) => {
                v.push(s);
            }
            (Self::Bool(_), MixedShare::Arith(_)) => {
                panic!("Can't push Arith share on Bool storage")
            }
            (Self::Arith(_), MixedShare::Bool(_)) => {
                panic!("Can't push Bool share on Arith storage")
            }
        }
    }
}

impl<R: Ring> IntoIterator for MixedShareStorage<R> {
    type Item = MixedShare<R>;
    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            MixedShareStorage::Bool(s) => Box::new(s.into_iter().map(MixedShare::Bool)),
            MixedShareStorage::Arith(s) => Box::new(s.into_iter().map(MixedShare::Arith)),
            MixedShareStorage::Mixed(s) => Box::new(s.into_iter()),
        }
    }
}

impl<R: Ring> FromIterator<MixedShare<R>> for MixedShareStorage<R> {
    fn from_iter<T: IntoIterator<Item = MixedShare<R>>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let mut acc = match iter.next() {
            None => return MixedShareStorage::default(),
            Some(MixedShare::Bool(b)) => MixedShareStorage::Bool(BitVec::repeat(b, 1)),
            Some(MixedShare::Arith(r)) => MixedShareStorage::Arith(vec![r]),
        };
        iter.for_each(|el| acc.try_push(el));
        acc
    }
}

// TODO does this make sense? Where do I use this default and how would it break for an arith SC?
impl<R: Ring> Default for MixedShareStorage<R> {
    fn default() -> Self {
        MixedShareStorage::Mixed(Default::default())
    }
}

impl<R: Ring> ShareStorage<MixedShare<R>> for MixedShareStorage<R> {
    fn len(&self) -> usize {
        match self {
            MixedShareStorage::Bool(s) => s.len(),
            MixedShareStorage::Arith(s) => s.len(),
            MixedShareStorage::Mixed(s) => s.len(),
        }
    }

    fn repeat(val: MixedShare<R>, len: usize) -> Self {
        match val {
            MixedShare::Bool(b) => Self::Bool(BitVec::repeat(b, len)),
            MixedShare::Arith(r) => Self::Arith(vec![r; len]),
        }
    }

    fn set(&mut self, idx: usize, val: MixedShare<R>) {
        match (self, val) {
            (Self::Bool(bv), MixedShare::Bool(b)) => {
                bv.set(idx, b);
            }
            (Self::Arith(v), MixedShare::Arith(r)) => {
                v[idx] = r;
            }
            (Self::Mixed(v), s) => {
                v[idx] = s;
            }
            // TODO, what if i just convert to Self::Mixed in these two cases?
            (Self::Bool(_), MixedShare::Arith(_)) => {
                panic!("Can't set Arith share on Bool storage")
            }
            (Self::Arith(_), MixedShare::Bool(_)) => {
                panic!("Can't set Bool share on Arith storage")
            }
        }
    }

    fn get(&self, idx: usize) -> MixedShare<R> {
        match self {
            MixedShareStorage::Bool(bv) => MixedShare::Bool(bv[idx]),
            MixedShareStorage::Arith(v) => MixedShare::Arith(v[idx].clone()),
            MixedShareStorage::Mixed(v) => v[idx].clone(),
        }
    }
}

pub struct MixedSetupStorage<R: Ring> {
    bool: <boolean_gmw::BooleanGmw as Protocol>::SetupStorage,
    arith: <arithmetic_gmw::ArithmeticGmw<R> as Protocol>::SetupStorage,
}

// TODO is this really needed as a bound?
impl<R: Ring> Default for MixedSetupStorage<R> {
    fn default() -> Self {
        todo!()
    }
}

// TODO mhh, I can't really impl this trait for the setup storage consisting of two parts...
//  maybe I need to change the trait or the struct
impl<R: Ring> SetupStorage for MixedSetupStorage<R> {
    fn len(&self) -> usize {
        todo!()
    }

    fn split_off_last(&mut self, count: usize) -> Self {
        todo!()
    }

    fn append(&mut self, other: Self) {
        todo!()
    }
}

#[derive(Default, Clone)]
pub struct InsecureMixedSetup<R>(PhantomData<R>);

#[async_trait]
impl<R: Ring> MTProvider for InsecureMixedSetup<R> {
    type Output = MixedSetupStorage<R>;
    type Error = Infallible;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        Ok(MixedSetupStorage {
            bool: boolean::InsecureMTProvider
                .request_mts(amount)
                .await
                .unwrap(),
            arith: arithmetic::InsecureMTProvider::default()
                .request_mts(amount)
                .await
                .unwrap(),
        })
    }
}

impl<R: Ring> Protocol for MixedGmw<R> {
    const SIMD_SUPPORT: bool = false;
    type Msg = Msg<R>;
    type SimdMsg = ();
    type Gate = MixedGate<R>;
    type Wire = ();
    type ShareStorage = MixedShareStorage<R>;
    type SetupStorage = MixedSetupStorage<R>;

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = MixedGate<R>>,
        _gate_outputs: impl Iterator<Item = MixedShare<R>>,
        mut inputs: impl Iterator<Item = MixedShare<R>>,
        preprocessing_data: &mut MixedSetupStorage<R>,
    ) -> Self::Msg {
        trace!("compute_msg");
        // split the iterators according to share type into two
        // call corresponding compute_msg on split iterators
        let (bool_gates, arith_gates) =
            interactive_gates.fold((vec![], vec![]), |(mut bgates, mut agates), mgate| {
                match mgate {
                    MixedGate::Bool(g) => bgates.push(g),
                    MixedGate::Arith(g) => agates.push(g),
                    MixedGate::Base(g) => {
                        panic!("Encountered base gate {g:?} in compute_msg");
                    }
                };
                (bgates, agates)
            });
        let (b_inputs, a_inputs) = inputs.fold((vec![], vec![]), |(mut binps, mut ainps), inp| {
            match inp {
                MixedShare::Bool(b) => binps.push(b),
                MixedShare::Arith(r) => ainps.push(r),
            }
            (binps, ainps)
        });
        let b_msg = BooleanGmw.compute_msg(
            party_id,
            bool_gates.into_iter(),
            iter::empty(),
            b_inputs.into_iter(),
            &mut preprocessing_data.bool,
        );
        let a_msg = ArithmeticGmw::default().compute_msg(
            party_id,
            arith_gates.into_iter(),
            iter::empty(),
            a_inputs.into_iter(),
            &mut preprocessing_data.arith,
        );
        Msg {
            bool: b_msg,
            arith: a_msg,
        }
    }

    fn evaluate_interactive(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = MixedShare<R>>,
        own_msg: Self::Msg,
        other_msg: Self::Msg,
        preprocessing_data: &mut MixedSetupStorage<R>,
    ) -> Self::ShareStorage {
        let mut b_storage = BooleanGmw
            .evaluate_interactive(
                party_id,
                iter::empty(),
                iter::empty(),
                own_msg.bool,
                other_msg.bool,
                &mut preprocessing_data.bool,
            )
            .into_iter();
        let mut a_storage = ArithmeticGmw::default()
            .evaluate_interactive(
                party_id,
                iter::empty(),
                iter::empty(),
                own_msg.arith,
                other_msg.arith,
                &mut preprocessing_data.arith,
            )
            .into_iter();
        let mut ret = Vec::with_capacity(b_storage.len() + a_storage.len());
        for g in interactive_gates {
            match g {
                MixedGate::Base(g) => panic!("Unexpected base gate {g:?}"),
                MixedGate::Bool(_) => ret.push(MixedShare::Bool(
                    b_storage.next().expect("Insufficient Bool outputs"),
                )),
                MixedGate::Arith(_) => ret.push(MixedShare::Arith(
                    a_storage.next().expect("Insufficient Arith outputs"),
                )),
            };
        }
        MixedShareStorage::Mixed(ret)
    }

    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self,
        _party_id: usize,
        circuit: &ExecutableCircuit<Self::Gate, Idx>,
    ) -> GateOutputs<Self::ShareStorage> {
        let data = circuit
            .gate_counts()
            .map(|(count, simd_size)| match simd_size {
                None => Input::Scalar(MixedShareStorage::Mixed(vec![Default::default(); count])),
                Some(_simd_size) => Input::Simd(vec![Default::default(); count]),
            })
            .collect();
        GateOutputs::new(data)
    }
}

impl<R: Ring> Share for MixedShare<R> {
    type SimdShare = MixedShareStorage<R>;
}

impl<R: Ring> Gate for MixedGate<R> {
    type Share = MixedShare<R>;
    type DimTy = ScalarDim;

    fn is_interactive(&self) -> bool {
        match self {
            MixedGate::Bool(g) => g.is_interactive(),
            MixedGate::Arith(g) => g.is_interactive(),
            MixedGate::Base(_) => false,
        }
    }

    fn input_size(&self) -> usize {
        match self {
            MixedGate::Bool(g) => g.input_size(),
            MixedGate::Arith(g) => g.input_size(),
            MixedGate::Base(g) => g.input_size(),
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share>> {
        // TODO, what about base gates inside Self::Bool or Arith?
        match self {
            MixedGate::Base(base_gate) => Some(base_gate),
            _ => None,
        }
    }

    fn wrap_base_gate(base_gate: BaseGate<Self::Share, Self::DimTy>) -> Self {
        Self::Base(base_gate)
    }

    #[inline]
    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = Self::Share>,
    ) -> Self::Share {
        match self {
            MixedGate::Base(base) => base.evaluate_non_interactive(party_id, inputs),
            MixedGate::Bool(g) => {
                let inputs = inputs.map(|i| match i {
                    MixedShare::Bool(b) => b,
                    MixedShare::Arith(_) => {
                        panic!("Received arithmetic share as input for Boolean gate")
                    }
                });
                let out = g.evaluate_non_interactive(party_id, inputs);
                MixedShare::Bool(out)
            }
            MixedGate::Arith(g) => {
                let inputs = inputs.map(|i| match i {
                    MixedShare::Arith(e) => e,
                    MixedShare::Bool(_) => {
                        panic!("Received Boolean share as input for arithmetic gate")
                    }
                });
                let out = g.evaluate_non_interactive(party_id, inputs);
                MixedShare::Arith(out)
            }
        }
    }
}

// impl From<BaseGate<bool>> for BooleanGate {
//     fn from(base_gate: BaseGate<bool>) -> Self {
//         BooleanGate::Base(base_gate)
//     }
// }

// impl<R: CryptoRng + Rng> XorSharing<R> {
//     pub fn new(rng: R) -> Self {
//         Self { rng }
//     }
// }
//
// impl<R: CryptoRng + Rng> Sharing for XorSharing<R> {
//     type Plain = bool;
//     type Shared = BitVec<usize>;
//
//     fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2] {
//         let rand = rand_bitvec(input.len(), &mut self.rng);
//         let masked_input = input ^ &rand;
//         [rand, masked_input]
//     }
//
//     fn reconstruct(shares: [Self::Shared; 2]) -> Self::Shared {
//         let [a, b] = shares;
//         a ^ b
//     }
// }

#[derive(Debug)]
pub struct MixedSharing<B, A, R> {
    bool: B,
    arith: A,
    ring: PhantomData<R>,
}

impl<B, A, R> Sharing for MixedSharing<B, A, R>
where
    B: Sharing<Plain = bool, Shared = BitVec<usize>>,
    R: Ring,
    A: Sharing<Plain = R, Shared = Vec<R>>,
{
    type Plain = MixedShare<R>;
    type Shared = MixedShareStorage<R>;

    fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2] {
        match input {
            MixedShareStorage::Bool(bv) => self.bool.share(bv).map(MixedShareStorage::Bool),
            MixedShareStorage::Arith(v) => self.arith.share(v).map(MixedShareStorage::Arith),
            MixedShareStorage::Mixed(_) => {
                todo!()
            }
        }
    }

    fn reconstruct(shares: [Self::Shared; 2]) -> Self::Shared {
        match shares {
            [MixedShareStorage::Bool(bv0), MixedShareStorage::Bool(bv1)] => {
                MixedShareStorage::Bool(B::reconstruct([bv0, bv1]))
            }
            [MixedShareStorage::Arith(v0), MixedShareStorage::Arith(v1)] => {
                MixedShareStorage::Arith(A::reconstruct([v0, v1]))
            }
            _ => {
                todo!("how to handle this case")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::base_circuit::BaseGate;
    use crate::circuit::{BaseCircuit, ExecutableCircuit};
    use crate::private_test_utils::{execute_circuit, TestChannel};
    use crate::protocols::arithmetic_gmw::ArithmeticGate;
    use crate::protocols::mixed_gmw::{MixedGate, MixedGmw, MixedShareStorage, MixedSharing};
    use crate::protocols::ScalarDim;
    use crate::BooleanGate;
    use bitvec::vec::BitVec;

    #[tokio::test]
    async fn simple_bool_logic() -> anyhow::Result<()> {
        let mut bc: BaseCircuit<MixedGate<u32>> = BaseCircuit::new();

        let in0 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let in1 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let and = bc.add_wired_gate(MixedGate::Bool(BooleanGate::And), &[in0, in1]);
        let out = bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[and]);
        let ec = ExecutableCircuit::DynLayers(bc.into());

        let out: MixedShareStorage<u32> = execute_circuit::<
            MixedGmw<u32>,
            u32,
            MixedSharing<_, _, u32>,
        >(&ec, (true, true), TestChannel::InMemory)
        .await?;

        let exp = MixedShareStorage::Bool(BitVec::repeat(true, 1));
        assert_eq!(out, exp);
        Ok(())
    }

    #[tokio::test]
    async fn simple_arith_circ() -> anyhow::Result<()> {
        let mut bc = BaseCircuit::<MixedGate<u32>, _>::new();
        let inp1 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let inp2 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let inp3 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let add = bc.add_wired_gate(MixedGate::Arith(ArithmeticGate::Add), &[inp1, inp2]);
        let mul = bc.add_wired_gate(MixedGate::Arith(ArithmeticGate::Mul), &[inp3, add]);
        bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[mul]);

        let ec: ExecutableCircuit<MixedGate<u32>, _> = ExecutableCircuit::DynLayers(bc.into());

        let out = execute_circuit::<MixedGmw<u32>, u32, MixedSharing<_, _, u32>>(
            &ec,
            (5, 5, 10),
            TestChannel::InMemory,
        )
        .await?;
        let exp = MixedShareStorage::Arith(vec![100]);
        assert_eq!(out, exp);

        Ok(())
    }
}
