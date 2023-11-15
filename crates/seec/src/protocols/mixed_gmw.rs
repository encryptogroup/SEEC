use crate::circuit::base_circuit::BaseGate;
use crate::circuit::{BaseCircuit, ExecutableCircuit, GateIdx};
use crate::common::BitVec;
use crate::executor::{GateOutputs, Input};
use crate::mul_triple::{arithmetic, boolean, MTProvider};
use crate::protocols::arithmetic_gmw::ArithmeticGmw;
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{
    arithmetic_gmw, boolean_gmw, Gate, Protocol, Ring, ScalarDim, SetupStorage, Share,
    ShareStorage, Sharing,
};
use crate::GateId;
use async_trait::async_trait;
use bitvec::array::BitArray;
use bitvec::order::Lsb0;
use bitvec::view::BitViewSized;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::marker::PhantomData;
use std::{iter, mem};
use tracing::trace;

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct MixedGmw<R>(PhantomData<R>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Msg<R> {
    bool: boolean_gmw::Msg,
    arith: arithmetic_gmw::Msg<R>,
    b2a_rec: Vec<R>,
    // TODO i should be able to eliminate this via shared prngs, this would mean that resharing
    //  a value incurs no communication
    bool_reshares: Vec<R>,
    /// IMPORTANT: this field must not be sent to the other party. It is currently needed for
    /// correctness to let information flow between compute_msg and evaluate_interactive for
    /// the A2BBoolShareSnd gate
    #[serde(skip)]
    own_bool_reshares: Vec<R>,
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

    pub fn unwrap_bool(self) -> bool {
        match self {
            MixedShare::Bool(b) => b,
            MixedShare::Arith(_) => panic!("called unwrap_bool on Arith"),
        }
    }
    pub fn unwrap_arith(self) -> R {
        match self {
            MixedShare::Bool(_) => panic!("called unwrap_arith on Bool"),
            MixedShare::Arith(r) => r,
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
    Conv(ConvGate),
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum ConvGate {
    // Selects the first input for party 0 and the second for party 1
    Select,
    // A2B needs an addition circuit, so it can't be represented as a single gate
    A2BBoolShareSnd,
    // resharing a value needs two gates, mhhh, or maybe it's just not a symm gate?
    A2BBoolShareRcv,
    // after xor sharing arith value, we need to split it into individual bits in the bool domain
    A2BSelectBit(usize),
    // B2A can be done in a single round with SBs, so we can have a single interactive gate
    B2A,
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
        let rec_extend = |this: &mut Self, sh, iter| {
            this.as_mixed();
            this.try_push(sh);
            this.extend(iter);
        };
        match self {
            MixedShareStorage::Bool(bv) => {
                while let Some(val) = iter.next() {
                    match val {
                        MixedShare::Bool(b) => {
                            bv.push(b);
                        }
                        a @ MixedShare::Arith(_) => {
                            rec_extend(self, a, iter);
                            return;
                        }
                    }
                }
            }
            MixedShareStorage::Arith(v) => {
                while let Some(val) = iter.next() {
                    match val {
                        b @ MixedShare::Bool(_) => {
                            rec_extend(self, b, iter);
                            return;
                        }
                        MixedShare::Arith(a) => {
                            v.push(a);
                        }
                    }
                }
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

    pub fn as_mixed(&mut self) {
        *self = match self {
            MixedShareStorage::Bool(bv) => {
                MixedShareStorage::Mixed(bv.iter().by_vals().map(MixedShare::Bool).collect())
            }
            MixedShareStorage::Arith(av) => {
                MixedShareStorage::Mixed(mem::take(av).into_iter().map(MixedShare::Arith).collect())
            }
            MixedShareStorage::Mixed(_) => return,
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
    shared_bits: SharedBits<R>,
}

pub struct SharedBits<R> {
    // each element represents R::BITS boolean shared bits. So for every r in self.bool
    // we have R::BITS corresponding values in self.arith
    bool: Vec<R>,
    arith: Vec<R>,
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
impl<R> MTProvider for InsecureMixedSetup<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    type Output = MixedSetupStorage<R>;
    type Error = Infallible;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        Ok(())
    }

    // TODO this method is bogus at the moment...
    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        let rng = ChaCha8Rng::seed_from_u64(42);
        // both parties sample the same bits, so th plain sample bit is always zero
        let bool = rng.sample_iter(Standard).take(amount).collect();
        let shared_bits = SharedBits {
            bool,
            arith: vec![R::ZERO; amount * R::BITS],
        };
        Ok(MixedSetupStorage {
            bool: boolean::InsecureMTProvider
                .request_mts(amount)
                .await
                .unwrap(),
            arith: arithmetic::InsecureMTProvider::default()
                .request_mts(amount)
                .await
                .unwrap(),
            shared_bits,
        })
    }
}

impl<R> Protocol for MixedGmw<R>
where
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
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

        let mut b_inputs = BitVec::<usize>::new();
        let mut a_inputs = vec![];
        let mut conv_inputs = vec![];
        // split the iterators according to share type into two
        // call corresponding compute_msg on split iterators
        let (bool_gates, arith_gates, conv_gates) = interactive_gates.fold(
            (vec![], vec![], vec![]),
            |(mut bgates, mut agates, mut conv_gates), mgate| {
                match mgate {
                    MixedGate::Bool(g) => {
                        b_inputs.extend(
                            inputs
                                .by_ref()
                                .take(g.input_size())
                                .map(MixedShare::unwrap_bool),
                        );
                        bgates.push(g);
                    }
                    MixedGate::Arith(g) => {
                        a_inputs.extend(
                            inputs
                                .by_ref()
                                .take(g.input_size())
                                .map(MixedShare::unwrap_arith),
                        );
                        agates.push(g);
                    }
                    ref g @ MixedGate::Conv(g_conf) => {
                        conv_inputs.extend(inputs.by_ref().take(g.input_size()));
                        conv_gates.push(g_conf)
                    }
                    MixedGate::Base(g) => {
                        panic!("Encountered base gate {g:?} in compute_msg");
                    }
                };
                (bgates, agates, conv_gates)
            },
        );
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
        let mut conv_inputs = conv_inputs.into_iter();
        let mut own_bool_reshares = vec![];
        let mut bool_reshares = vec![];
        let mut b2a_rec = vec![];
        for g in conv_gates {
            match g {
                ConvGate::A2BBoolShareSnd => {
                    let MixedShare::Arith(r) = conv_inputs.next().expect("Missing input") else {
                        panic!("expected Arith input but got Bool");
                    };
                    let rand: R = random();
                    own_bool_reshares.push(r ^ rand.clone());
                    bool_reshares.push(rand);
                }
                ConvGate::B2A => {
                    let mut buf = BitArray::<_, Lsb0>::new([R::ZERO]);
                    for (mut dest, inp) in buf.iter_mut().zip(conv_inputs.by_ref()) {
                        dest.set(inp.unwrap_bool());
                    }
                    let [xi] = buf.data;
                    let ti = xi
                        ^ preprocessing_data
                            .shared_bits
                            .bool
                            .pop()
                            .expect("Insufficient bool shared bits");
                    b2a_rec.push(ti);
                }
                ConvGate::A2BBoolShareRcv => {
                    // nothing to do, I think?
                }
                ConvGate::Select | ConvGate::A2BSelectBit(_) => {
                    panic!("non-interactive gate in evaluate_interactive")
                }
            }
        }

        Msg {
            bool: b_msg,
            arith: a_msg,
            b2a_rec,
            bool_reshares,
            own_bool_reshares,
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
        let mut own_reshares = own_msg.own_bool_reshares.into_iter();
        let mut other_reshares = other_msg.bool_reshares.into_iter();
        let mut own_b2a_rec = own_msg.b2a_rec.into_iter();
        let mut other_b2a_rec = other_msg.b2a_rec.into_iter();
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
                MixedGate::Conv(ConvGate::A2BBoolShareSnd) => ret.push(MixedShare::Arith(
                    own_reshares.next().expect("Missing own bool reshare"),
                )),
                MixedGate::Conv(ConvGate::A2BBoolShareRcv) => ret.push(MixedShare::Arith(
                    other_reshares.next().expect("Missing other bool reshare"),
                )),
                MixedGate::Conv(ConvGate::B2A) => {
                    let t_rec = own_b2a_rec.next().unwrap() ^ other_b2a_rec.next().unwrap();
                    let l = preprocessing_data.shared_bits.arith.len();
                    let ri_a = preprocessing_data.shared_bits.arith[l - R::BITS..]
                        .iter()
                        // TODO is rev correct here? Must be consistent with the order of the
                        //  bool shared bits...
                        .rev();
                    dbg!(&t_rec);
                    let xa: R = (0..R::BITS)
                        .zip(ri_a)
                        .map(|(i, ri)| {
                            let ti: R = if t_rec.get_bit(i) { R::ONE } else { R::ZERO };
                            let two: R = R::ONE.wrapping_add(&R::ONE);
                            let lhs = if party_id == 0 {
                                ti.wrapping_add(&ri)
                            } else {
                                ri.clone()
                            };
                            let rhs = two.wrapping_mul(&ti).wrapping_mul(ri);
                            let xi = lhs.wrapping_sub(&rhs);
                            two.pow(i as u32).wrapping_mul(&xi)
                        })
                        .reduce(|a, b| a.wrapping_add(&b))
                        .unwrap();
                    ret.push(MixedShare::Arith(xa));
                }
                MixedGate::Conv(ConvGate::A2BSelectBit(_) | ConvGate::Select) => {
                    panic!("Non-interactive gate in evaluate_interactive")
                }
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
            MixedGate::Conv(
                ConvGate::A2BBoolShareSnd | ConvGate::A2BBoolShareRcv | ConvGate::B2A,
            ) => true,
            MixedGate::Conv(ConvGate::A2BSelectBit(_) | ConvGate::Select) => false,
            MixedGate::Base(_) => false,
        }
    }

    fn input_size(&self) -> usize {
        match self {
            MixedGate::Bool(g) => g.input_size(),
            MixedGate::Arith(g) => g.input_size(),
            MixedGate::Base(g) => g.input_size(),
            // TODO mhh, Rcv doesn't really need an input, but maybe it's better to have it symm
            MixedGate::Conv(
                ConvGate::A2BBoolShareSnd | ConvGate::A2BBoolShareRcv | ConvGate::A2BSelectBit(_),
            ) => 1,
            MixedGate::Conv(ConvGate::Select) => 2,
            MixedGate::Conv(ConvGate::B2A) => R::BITS,
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
        let mut unwrap_inp = || inputs.next().expect("Missing input");

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
            MixedGate::Conv(ConvGate::A2BSelectBit(idx)) => {
                let r = match unwrap_inp() {
                    MixedShare::Arith(r) => r,
                    MixedShare::Bool(_) => {
                        panic!("Received Boolean share as input for ConvGate::A2BSelectBit")
                    }
                };
                MixedShare::Bool(r.get_bit(*idx))
            }
            MixedGate::Conv(ConvGate::Select) => {
                let [a, b] = [unwrap_inp(), unwrap_inp()];
                if party_id == 0 {
                    a
                } else {
                    b
                }
            }
            int @ MixedGate::Conv(
                ConvGate::A2BBoolShareSnd | ConvGate::A2BBoolShareRcv | ConvGate::B2A,
            ) => {
                panic!("Got interactive gate {int:?} in evaluate_non_interactive")
            }
        }
    }
}

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

pub fn a2b<R: Ring>(bc: &mut BaseCircuit<MixedGate<R>>, a: GateId) -> Vec<GateId> {
    let a2b0 = bc.add_wired_gate(MixedGate::Conv(ConvGate::A2BBoolShareSnd), &[a]);
    let a2b1 = bc.add_wired_gate(MixedGate::Conv(ConvGate::A2BBoolShareRcv), &[a]);
    // potentially switch gates, depending on party_id of executing party
    let a2b0_sw = bc.add_wired_gate(MixedGate::Conv(ConvGate::Select), &[a2b0, a2b1]);
    let a2b1_sw = bc.add_wired_gate(MixedGate::Conv(ConvGate::Select), &[a2b1, a2b0]);
    let mut split = |a: GateId| {
        (0..R::BITS)
            .map(|idx| {
                bc.add_wired_gate(MixedGate::Conv(ConvGate::A2BSelectBit(idx as usize)), &[a])
            })
            .collect::<Vec<GateId>>()
    };
    let split_a2b0 = split(a2b0_sw);
    let split_a2b1 = split(a2b1_sw);
    basic_add(bc, &split_a2b0, &split_a2b1)
}

fn basic_add<R: Ring>(
    bc: &mut BaseCircuit<MixedGate<R>>,
    a: &[GateId],
    b: &[GateId],
) -> Vec<GateId> {
    use boolean_gmw::BooleanGate;
    macro_rules! xor {
        ($a:expr, $b:expr) => {
            bc.add_wired_gate(MixedGate::Bool(BooleanGate::Xor), &[$a, $b])
        };
    }
    macro_rules! and {
        ($a:expr, $b:expr) => {
            bc.add_wired_gate(MixedGate::Bool(BooleanGate::And), &[$a, $b])
        };
    }

    let mut carry = bc.add_gate(MixedGate::Base(BaseGate::Constant(MixedShare::Bool(false))));

    let mut full_adder = |a, b, c| {
        let xab = xor!(a, b);
        let s = xor!(xab, c);
        let aab = and!(a, b);
        let axab = and!(c, xab);
        let c_out = xor!(aab, axab);
        (s, c_out)
    };
    let mut out = vec![];
    for (a_i, b_i) in a.iter().zip(b) {
        let (s, c) = full_adder(*a_i, *b_i, carry);
        out.push(s);
        carry = c;
    }
    bc.add_wired_gate(MixedGate::Base(BaseGate::Debug), &[carry]);
    out
}

#[cfg(test)]
mod tests {
    use super::basic_add;
    use crate::circuit::base_circuit::BaseGate;
    use crate::circuit::{BaseCircuit, DefaultIdx, ExecutableCircuit};
    use crate::private_test_utils::{execute_circuit, init_tracing, TestChannel, ToBool};
    use crate::protocols::arithmetic_gmw::ArithmeticGate;
    use crate::protocols::mixed_gmw::{
        a2b, ConvGate, MixedGate, MixedGmw, MixedShare, MixedShareStorage, MixedSharing,
    };
    use crate::protocols::ScalarDim;
    use crate::{BooleanGate, GateId};
    use bitvec::vec::BitVec;

    #[tokio::test]
    async fn simple_bool_logic() -> anyhow::Result<()> {
        let mut bc: BaseCircuit<MixedGate<u32>> = BaseCircuit::new();

        let in0 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let in1 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let and = bc.add_wired_gate(MixedGate::Bool(BooleanGate::And), &[in0, in1]);
        bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[and]);
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

    #[tokio::test]
    async fn basic_add_test() -> anyhow::Result<()> {
        let _g = init_tracing();
        let mut bc = BaseCircuit::<MixedGate<u8>, _>::new();

        let inp1: Vec<_> = (0..8)
            .map(|_| bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim))))
            .collect();
        let inp2: Vec<_> = (0..8)
            .map(|_| bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim))))
            .collect();
        let added = basic_add(&mut bc, &inp1, &inp2);
        for g in added {
            bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[g]);
        }

        let ec: ExecutableCircuit<MixedGate<u8>, _> = ExecutableCircuit::DynLayers(bc.into());

        let out = execute_circuit::<MixedGmw<u8>, DefaultIdx, MixedSharing<_, _, u8>>(
            &ec,
            (ToBool(200), ToBool(100)),
            TestChannel::InMemory,
        )
        .await?;
        let mut exp = BitVec::from_element(200_u8.wrapping_add(100) as usize);
        exp.truncate(8);
        assert_eq!(out, MixedShareStorage::Bool(exp));

        Ok(())
    }

    #[tokio::test]
    async fn basic_mixed_circ_a2b() -> anyhow::Result<()> {
        let _g = init_tracing();
        let mut bc = BaseCircuit::<MixedGate<u8>, _>::new();

        let inp1 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));

        let converted_to_b = a2b(&mut bc, inp1);
        for g in converted_to_b {
            bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[g]);
        }

        let ec: ExecutableCircuit<MixedGate<u8>, _> = ExecutableCircuit::DynLayers(bc.into());

        let out = execute_circuit::<MixedGmw<u8>, DefaultIdx, MixedSharing<_, _, u8>>(
            &ec,
            (42,),
            TestChannel::InMemory,
        )
        .await?;
        let mut exp = BitVec::from_element(42);
        exp.truncate(8);
        let exp = MixedShareStorage::Bool(exp);
        assert_eq!(out, exp);
        Ok(())
    }

    #[tokio::test]
    async fn basic_mixed_circ_b2a() -> anyhow::Result<()> {
        let _g = init_tracing();
        let mut bc = BaseCircuit::<MixedGate<u8>, _>::new();

        let inps: Vec<_> = (0..8)
            .map(|_| bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim))))
            .collect();
        let b2a = bc.add_wired_gate(MixedGate::Conv(ConvGate::B2A), &inps);
        bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[b2a]);

        let ec: ExecutableCircuit<MixedGate<u8>, _> = ExecutableCircuit::DynLayers(bc.into());

        let out = execute_circuit::<MixedGmw<u8>, DefaultIdx, MixedSharing<_, _, u8>>(
            &ec,
            (ToBool(67),),
            TestChannel::InMemory,
        )
        .await?;
        let exp = MixedShareStorage::Arith(vec![67]);
        assert_eq!(out, exp);
        Ok(())
    }

    #[tokio::test]
    async fn complex_mixed_circ() -> anyhow::Result<()> {
        let _g = init_tracing();
        let mut bc = BaseCircuit::<MixedGate<u16>, _>::new();

        let binps: Vec<_> = (0..16)
            .map(|_| bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim))))
            .collect();
        let ainp1 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let ainp2 = bc.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));

        let mul = bc.add_wired_gate(MixedGate::Arith(ArithmeticGate::Mul), &[ainp1, ainp2]);
        let mul_b = a2b(&mut bc, mul);

        let added = basic_add(&mut bc, &binps, &mul_b);
        let res_a = bc.add_wired_gate(MixedGate::Conv(ConvGate::B2A), &added);
        bc.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[res_a]);

        let ec: ExecutableCircuit<MixedGate<u16>, _> = ExecutableCircuit::DynLayers(bc.into());

        let out = execute_circuit::<MixedGmw<u16>, DefaultIdx, MixedSharing<_, _, u16>>(
            &ec,
            (ToBool(665), 75, 160),
            TestChannel::InMemory,
        )
        .await?;
        let exp = MixedShareStorage::Arith(vec![12665]);
        assert_eq!(out, exp);
        Ok(())
    }
}
