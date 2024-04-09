use crate::bristol;
use crate::circuit::base_circuit::BaseGate;
use crate::common::{BitSlice, BitVec};
use crate::evaluate::and;
use crate::mul_triple::boolean::MulTriple;
use crate::mul_triple::boolean::MulTriples;
use crate::protocols::{Gate, MsgBuf, Protocol, ScalarDim, SetupStorage, Share, Sharing};
use crate::utils::{rand_bitvec, BitVecExt};
use itertools::Itertools;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::mem;
use std::ops::Range;
use tracing::trace;

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct BooleanGmw;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Msg {
    // TODO ser/de the BitVecs or Vecs? Or maybe a single Vec? e and d have the same length
    e: Vec<usize>,
    d: Vec<usize>,
}

pub struct MsgIdx {
    pub start_byte: usize,
    pub bit_len: usize,
}

impl MsgBuf for Msg {
    // Range over **bit** positions
    type Index = MsgIdx;
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimdMsg {
    packed_data: Msg,
    simd_sizes: Vec<u32>,
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum BooleanGate {
    Base(BaseGate<bool>),
    And,
    Xor,
    Inv,
}

#[derive(Debug)]
pub struct XorSharing<R: CryptoRng + Rng> {
    rng: R,
}

impl Protocol for BooleanGmw {
    const SIMD_SUPPORT: bool = true;
    type Msg = Msg;
    type SimdMsg = SimdMsg;
    type Gate = BooleanGate;
    type Wire = ();
    type ShareStorage = BitVec<usize>;
    type SetupStorage = MulTriples;

    fn compute_msg(
        &self,
        _party_id: usize,
        interactive_gates: impl Iterator<Item = BooleanGate>,
        _gate_outputs: impl Iterator<Item = bool>,
        mut inputs: impl Iterator<Item = bool>,
        msg_buf: &mut Self::Msg,
        mul_triples: &mut MulTriples,
    ) -> MsgIdx {
        trace!("compute_msg");
        let (d, e): (BitVec<usize>, BitVec<usize>) = interactive_gates
            .zip(mul_triples.iter().rev())
            .map(|(gate, mt): (BooleanGate, MulTriple)| {
                // TODO debug_assert?
                assert!(matches!(gate, BooleanGate::And));
                let mut inputs = inputs.by_ref().take(gate.input_size());
                let (x, y) = (inputs.next().unwrap(), inputs.next().unwrap());
                debug_assert!(
                    inputs.next().is_none(),
                    "Currently only support AND gates with 2 inputs"
                );
                and::compute_shares(x, y, &mt)
            })
            .unzip();
        let idx = MsgIdx {
            start_byte: msg_buf.d.len(),
            bit_len: d.len(),
        };
        msg_buf.d.extend_from_slice(&d.into_vec());
        msg_buf.e.extend_from_slice(&e.into_vec());
        idx
    }

    fn compute_msg_simd<'e>(
        &self,
        _party_id: usize,
        _interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = &'e BitVec<usize>>,
        inputs: impl Iterator<Item = &'e BitVec<usize>>,
        mul_triples: &mut Self::SetupStorage,
    ) -> Self::SimdMsg {
        let mut simd_sizes = vec![];
        let mut d = BitVec::new();
        let mut e = BitVec::new();
        inputs.chunks(2).into_iter().for_each(|mut chunk| {
            let x = chunk.next().unwrap();
            let y = chunk.next().unwrap();
            simd_sizes.push(x.len() as u32);
            debug_assert_eq!(x.len(), y.len(), "Unequal SIMD sizes");
            e.extend_from_bitslice(x);
            d.extend_from_bitslice(y);
        });
        let mts = mul_triples.slice(mul_triples.len() - d.len()..);

        d.fast_bit_xor_mut(&mts.a().to_bitvec());
        e.fast_bit_xor_mut(&mts.b().to_bitvec());

        SimdMsg {
            packed_data: Msg {
                e: e.into_vec(),
                d: d.into_vec(),
            },
            simd_sizes,
        }
    }

    fn evaluate_interactive(
        &self,
        party_id: usize,
        _interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = bool>,
        msg_idx: MsgIdx,
        own_msg: &mut Self::Msg,
        other_msg: &mut Self::Msg,
        mul_triples: &mut MulTriples,
    ) -> Self::ShareStorage {
        let d = &BitSlice::from_slice(&own_msg.d[msg_idx.start_byte..])[..msg_idx.bit_len];
        let e = &BitSlice::from_slice(&own_msg.e[msg_idx.start_byte..])[..msg_idx.bit_len];
        let resp_d = &BitSlice::from_slice(&other_msg.d[msg_idx.start_byte..])[..msg_idx.bit_len];
        let resp_e = &BitSlice::from_slice(&other_msg.e[msg_idx.start_byte..])[..msg_idx.bit_len];
        // Remove the mul_triples used in compute_msg
        let mul_triples = mul_triples.split_off_last(msg_idx.bit_len);
        // TODO can this operate on the raw elements of the msg and not individual bits?
        //  perf should be much better
        d.into_iter()
            .by_vals()
            .zip(e.iter().by_vals())
            .zip(resp_d.iter().by_vals())
            .zip(resp_e.iter().by_vals())
            .zip(mul_triples.iter().rev())
            .map(|((((d, e), d_resp), e_resp), mt)| {
                let d = [d, d_resp];
                let e = [e, e_resp];
                and::evaluate(d, e, mt, party_id)
            })
            .collect()
    }

    fn evaluate_interactive_simd<'e>(
        &self,
        party_id: usize,
        _interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = &'e BitVec<usize>>,
        own_msg: Self::SimdMsg,
        other_msg: Self::SimdMsg,
        mul_triples: &mut Self::SetupStorage,
    ) -> Vec<Self::ShareStorage> {
        let SimdMsg {
            packed_data: Msg { d, e },
            simd_sizes,
        } = own_msg;
        let own_d = BitVec::from_vec(d);
        let own_e = BitVec::from_vec(e);
        let SimdMsg {
            packed_data: Msg {
                d: resp_d,
                e: resp_e,
            },
            simd_sizes: resp_simd_sizes,
        } = other_msg;
        let resp_d = BitVec::from_vec(resp_d);
        let resp_e = BitVec::from_vec(resp_e);

        assert_eq!(
            simd_sizes, resp_simd_sizes,
            "Message have unequal simd sizes"
        );
        let total_size = simd_sizes.iter().copied().sum::<u32>() as usize;
        // Remove the mul_triples used in compute_msg
        let mts = mul_triples.split_off_last(total_size);
        let d = own_d.fast_bit_xor(&resp_d);
        let e = own_e.fast_bit_xor(&resp_e);
        let mut res = if party_id == 0 {
            d.clone().fast_bit_and(&e)
        } else {
            BitVec::repeat(false, total_size)
        };
        res.fast_bit_xor_mut(&d.fast_bit_and(mts.b()))
            .fast_bit_xor_mut(&e.fast_bit_and(mts.a()))
            .fast_bit_xor_mut(mts.c());
        // res ^= d & mts.b ^ e & mts.a ^ mts.c;
        let mut res = res.as_bitslice();
        simd_sizes
            .iter()
            .map(|size| {
                let simd_share = res[..*size as usize].to_bitvec();
                res = &res[*size as usize..];
                simd_share
            })
            .collect()
    }
}

impl Share for bool {
    type SimdShare = BitVec<usize>;
}

impl Gate for BooleanGate {
    type Share = bool;
    type DimTy = ScalarDim;

    fn is_interactive(&self) -> bool {
        matches!(self, BooleanGate::And)
    }

    fn input_size(&self) -> usize {
        match self {
            BooleanGate::Base(base_gate) => base_gate.input_size(),
            BooleanGate::Inv => 1,
            BooleanGate::And | BooleanGate::Xor => 2,
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share>> {
        match self {
            BooleanGate::Base(base_gate) => Some(base_gate),
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
            BooleanGate::Base(base) => base.evaluate_non_interactive(party_id, inputs),
            BooleanGate::And => panic!("Called evaluate_non_interactive on Gate::AND"),
            BooleanGate::Xor => {
                inputs.next().expect("Missing inputs") ^ inputs.next().expect("Missing inputs")
            }
            BooleanGate::Inv => {
                let inp = inputs.next().expect("Empty input");
                if party_id == 0 {
                    !inp
                } else {
                    inp
                }
            }
        }
    }

    fn evaluate_non_interactive_simd<'this, 'e>(
        &'this self,
        party_id: usize,
        mut inputs: impl Iterator<Item = &'e <Self::Share as Share>::SimdShare>,
    ) -> <Self::Share as Share>::SimdShare {
        match self {
            BooleanGate::Base(base) => base.evaluate_non_interactive_simd(party_id, inputs),
            BooleanGate::And => panic!("Called evaluate_non_interactive on Gate::AND"),
            BooleanGate::Xor => inputs
                .next()
                .expect("Missing inputs")
                .clone()
                .fast_bit_xor(inputs.next().expect("Missing inputs")),
            BooleanGate::Inv => {
                let inp = inputs.next().expect("Empty input").clone();
                if party_id == 0 {
                    !inp
                } else {
                    inp
                }
            }
        }
    }
}

impl From<&bristol::Gate> for BooleanGate {
    fn from(gate: &bristol::Gate) -> Self {
        match gate {
            bristol::Gate::And(_) => BooleanGate::And,
            bristol::Gate::Xor(_) => BooleanGate::Xor,
            bristol::Gate::Inv(_) => BooleanGate::Inv,
        }
    }
}

impl From<BaseGate<bool>> for BooleanGate {
    fn from(base_gate: BaseGate<bool>) -> Self {
        BooleanGate::Base(base_gate)
    }
}

impl<R: CryptoRng + Rng> XorSharing<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: CryptoRng + Rng> Sharing for XorSharing<R> {
    type Plain = bool;
    type Shared = BitVec<usize>;

    fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2] {
        let rand = rand_bitvec(input.len(), &mut self.rng);
        let masked_input = input ^ &rand;
        [rand, masked_input]
    }

    fn reconstruct(shares: [Self::Shared; 2]) -> Self::Shared {
        let [a, b] = shares;
        a ^ b
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::ExecutableCircuit;
    use crate::common::BitVec;
    use crate::private_test_utils::{execute_circuit, TestChannel};
    use crate::protocols::boolean_gmw::BooleanGmw;
    use crate::secret::Secret;
    use crate::{BooleanGate, CircuitBuilder};

    #[tokio::test]
    async fn simple_logic() -> anyhow::Result<()> {
        let a = Secret::<_, u32>::input(0);
        (a & false).output();

        let circ = CircuitBuilder::<BooleanGate, u32>::global_into_circuit();
        let out = execute_circuit::<BooleanGmw, _, _>(
            &ExecutableCircuit::DynLayers(circ),
            true,
            TestChannel::InMemory,
        )
        .await?;
        assert_eq!(out, BitVec::<u8>::repeat(false, 1));
        Ok(())
    }
}
