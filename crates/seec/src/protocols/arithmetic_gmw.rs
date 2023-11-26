use crate::circuit::base_circuit::BaseGate;
use crate::mul_triple::arithmetic::MulTriple;
use crate::mul_triple::arithmetic::MulTriples;
use crate::protocols::{Gate, Protocol, Ring, ScalarDim, SetupStorage, Share, Sharing};
use itertools::izip;
use rand::distributions::{Distribution, Standard};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct ArithmeticGmw<R>(PhantomData<R>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg<R> {
    MulLayer { e: Vec<R>, d: Vec<R> },
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum ArithmeticGate<R> {
    Base(BaseGate<R>),
    Mul,
    Add,
    Sub,
}

#[derive(Debug)]
pub struct AdditiveSharing<RING, RNG: CryptoRng + Rng> {
    rng: RNG,
    phantom: PhantomData<RING>,
}

impl<R: Ring> Protocol for ArithmeticGmw<R> {
    type Msg = Msg<R>;
    type SimdMsg = ();
    type Gate = ArithmeticGate<R>;
    type Wire = ();
    type ShareStorage = Vec<R>;
    type SetupStorage = MulTriples<R>;

    fn compute_msg(
        &self,
        _party_id: usize,
        interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = R>,
        mut inputs: impl Iterator<Item = R>,
        mul_triples: &mut MulTriples<R>,
    ) -> Self::Msg {
        let (d, e) = interactive_gates
            .zip(mul_triples.iter())
            .map(|(gate, mt): (ArithmeticGate<R>, MulTriple<R>)| {
                // TODO debug_assert?
                assert!(matches!(gate, ArithmeticGate::Mul));
                let mut inputs = inputs.by_ref().take(gate.input_size());
                let (x, y): (R, R) = (inputs.next().unwrap(), inputs.next().unwrap());
                debug_assert!(
                    inputs.next().is_none(),
                    "Currently only support AND gates with 2 inputs"
                );
                (x.wrapping_sub(mt.a()), y.wrapping_sub(mt.c()))
            })
            .unzip();
        Msg::MulLayer { e, d }
    }

    fn evaluate_interactive(
        &self,
        party_id: usize,
        _interactive_gates: impl Iterator<Item = Self::Gate>,
        _gate_outputs: impl Iterator<Item = R>,
        own_msg: Self::Msg,
        other_msg: Self::Msg,
        mul_triples: &mut MulTriples<R>,
    ) -> Self::ShareStorage {
        let Msg::MulLayer { e, d } = own_msg;
        let Msg::MulLayer {
            e: resp_e,
            d: resp_d,
        } = other_msg;

        let mts = mul_triples.remove_first(e.len());

        izip!(d, e, resp_d, resp_e, mts.iter())
            .map(|(own_d, own_e, resp_d, resp_e, mt)| {
                let d = own_d.wrapping_add(&resp_d);
                let e = own_e.wrapping_add(&resp_e);
                let da = d.wrapping_mul(mt.a());
                let eb = e.wrapping_mul(mt.b());
                let da_eb_c = da.wrapping_add(&eb).wrapping_add(mt.c());
                if party_id == 0 {
                    let de = d.wrapping_mul(&e);
                    da_eb_c.wrapping_add(&de)
                } else {
                    da_eb_c
                }
            })
            .collect()
    }
}

impl<R: Ring + Share> Gate for ArithmeticGate<R> {
    type Share = R;
    type DimTy = ScalarDim;

    fn is_interactive(&self) -> bool {
        matches!(self, ArithmeticGate::Mul)
    }

    fn input_size(&self) -> usize {
        match self {
            ArithmeticGate::Base(base_gate) => base_gate.input_size(),
            ArithmeticGate::Mul | ArithmeticGate::Add | ArithmeticGate::Sub => 2,
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share>> {
        match self {
            ArithmeticGate::Base(base_gate) => Some(base_gate),
            _ => None,
        }
    }

    fn wrap_base_gate(base_gate: BaseGate<Self::Share, Self::DimTy>) -> Self {
        Self::Base(base_gate)
    }

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Share {
        let mut inputs = inputs.into_iter();
        match self {
            ArithmeticGate::Base(base) => base.evaluate_non_interactive(party_id, inputs.by_ref()),
            ArithmeticGate::Mul => panic!("Called evaluate_non_interactive on Gate::AND"),
            ArithmeticGate::Add => {
                let x = inputs.next().expect("Empty input");
                let y = inputs.next().expect("Empty input");
                x.wrapping_add(&y)
            }
            ArithmeticGate::Sub => {
                // TODO is this the correct order?
                let x = inputs.next().expect("Empty input");
                let y = inputs.next().expect("Empty input");
                x.wrapping_sub(&y)
            }
        }
    }
}

impl<R> From<BaseGate<R>> for ArithmeticGate<R> {
    fn from(base_gate: BaseGate<R>) -> Self {
        ArithmeticGate::Base(base_gate)
    }
}

impl<RING, RNG: CryptoRng + Rng> AdditiveSharing<RING, RNG> {
    pub fn new(rng: RNG) -> Self {
        Self {
            rng,
            phantom: PhantomData,
        }
    }
}

impl<RING, RNG> Sharing for AdditiveSharing<RING, RNG>
where
    RING: Ring,
    RNG: CryptoRng + Rng,
    Standard: Distribution<RING>,
{
    type Plain = RING;
    type Shared = Vec<RING>;

    fn share(&mut self, input: Self::Shared) -> [Self::Shared; 2] {
        let rand: Vec<_> = (&mut self.rng)
            .sample_iter(Standard)
            .take(input.len())
            .collect();
        let masked_input = rand
            .iter()
            .zip(input)
            .map(|(rand, inp)| inp.wrapping_sub(rand))
            .collect();
        [rand, masked_input]
    }

    fn reconstruct(shares: [Self::Shared; 2]) -> Self::Shared {
        let [a, b] = shares;
        a.into_iter()
            .zip(b)
            .map(|(a, b)| a.wrapping_add(&b))
            .collect()
    }
}

impl Share for u8 {
    type SimdShare = Vec<u8>;
}

impl Share for u16 {
    type SimdShare = Vec<u16>;
}

impl Share for u32 {
    type SimdShare = Vec<u32>;
}

impl Share for u64 {
    type SimdShare = Vec<u64>;
}

impl Share for u128 {
    type SimdShare = Vec<u128>;
}

#[cfg(test)]
mod tests {
    use crate::circuit::base_circuit::BaseGate;
    use crate::circuit::{BaseCircuit, DefaultIdx, ExecutableCircuit};
    use crate::private_test_utils::{execute_circuit, TestChannel};
    use crate::protocols::arithmetic_gmw::{AdditiveSharing, ArithmeticGate, ArithmeticGmw};
    use crate::protocols::ScalarDim;

    #[tokio::test]
    async fn simple_circ() -> anyhow::Result<()> {
        let mut bc = BaseCircuit::<_, DefaultIdx>::new();
        let inp1 = bc.add_gate(ArithmeticGate::Base(BaseGate::Input(ScalarDim)));
        let inp2 = bc.add_gate(ArithmeticGate::Base(BaseGate::Input(ScalarDim)));
        let inp3 = bc.add_gate(ArithmeticGate::Base(BaseGate::Input(ScalarDim)));
        let add = bc.add_wired_gate(ArithmeticGate::Add, &[inp1, inp2]);
        let mul = bc.add_wired_gate(ArithmeticGate::Mul, &[inp3, add]);
        bc.add_wired_gate(ArithmeticGate::Base(BaseGate::Output(ScalarDim)), &[mul]);

        let circ = ExecutableCircuit::DynLayers(bc.into());

        let out = execute_circuit::<ArithmeticGmw<u32>, _, AdditiveSharing<u32, _>>(
            &circ,
            (5, 5, 10),
            TestChannel::InMemory,
        )
        .await?;

        assert_eq!(100, out[0]);

        Ok(())
    }
}
