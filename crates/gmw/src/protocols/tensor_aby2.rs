use crate::circuit::base_circuit::BaseGate;
use crate::circuit::{DefaultIdx, ExecutableCircuit, GateIdx};
use crate::common::BitVec;
use crate::executor::{Executor, GateOutputs, Message, ScGateOutputs};
use crate::mul_triple::{MTProvider, MulTriples};
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{
    boolean_gmw, Dimension, DynDim, FunctionDependentSetup, Gate, Protocol, SetupStorage,
    ShareStorage,
};
use crate::secret::{inputs, Secret};
use crate::utils::rand_bitvec;
use crate::{bristol, CircuitBuilder};
use ahash::AHashMap;
use async_trait::async_trait;
use mpc_bitmatrix::BitMatrix;
use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Debug;
use std::iter::repeat;
use std::ops::{BitXor, Not};
use std::{iter, vec};

#[derive(Clone)]
pub struct BoolTensorAby2 {
    delta_sharing_state: DeltaSharing,
}

#[derive(Clone)]
pub struct DeltaSharing {
    private_rng: ChaChaRng,
    local_joint_rng: ChaChaRng,
    remote_joint_rng: ChaChaRng,
    // TODO ughh
    input_position_share_type_map: HashMap<usize, ShareType>,
}

#[derive(Copy, Clone, Debug)]
pub enum ShareType {
    Local,
    Remote,
}

#[derive(Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct Share {
    public: bool,
    private: bool,
}

#[derive(Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct ShareVec {
    public: BitVec,
    private: BitVec,
}

#[derive(Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct ShareMatrix {
    public: BitMatrix<u8>,
    private: BitMatrix<u8>,
}

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum TensorShare {
    Scalar(Share),
    Vec(ShareVec),
    Matrix(ShareMatrix),
}

#[derive(Clone, Debug, Default, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct DeltaShareStorage {
    shares: Vec<TensorShare>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
// TODO this message has a huge overhead for scalar operations. Do struct of Vecs for each enum
//  variant
pub enum Msg {
    DeltaShares { shares: Vec<PartialShare> },
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum BooleanGate {
    Base(BaseGate<TensorShare, DynDim>),
    And2,
    And3,
    And4,
    Xor,
    Inv,
    Tensor(TensorGate),
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum TensorGate {
    Combine { shape: Shape },
    MatMult { rows: usize, cols: usize },
    Select { idx: SelIdx },
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum Shape {
    Scalar,
    OneDim(u32),
    TwoDim(u32, u32),
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum SelIdx {
    OneDim(u32),
    TwoDim { row: u32, col: u32 },
    Row(u32),
    Col(u32),
}

/// Contains preprocessing data ([\delta_ab]_i for interactive gates in
/// **reverse** topological order. This is needed to evaluate interactive gates.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct SetupData {
    eval_shares: Vec<PartialShare>,
}

/// Either the private or public part of a TensorShare
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum PartialShare {
    Scalar(bool),
    Vec(BitVec),
    Matrix(BitMatrix<u8>),
}

impl BoolTensorAby2 {
    pub fn new(sharing_state: DeltaSharing) -> Self {
        Self {
            delta_sharing_state: sharing_state,
        }
    }
}

pub type AbySetupMsg = Message<BooleanGmw>;

pub struct AbySetupProvider<Mtp> {
    party_id: usize,
    mt_provider: Mtp,
    sender: mpc_channel::Sender<AbySetupMsg>,
    receiver: mpc_channel::Receiver<AbySetupMsg>,
}

impl Protocol for BoolTensorAby2 {
    type Msg = Msg;
    type SimdMsg = ();
    type Gate = BooleanGate;
    type Wire = ();
    type ShareStorage = DeltaShareStorage;
    type SetupStorage = SetupData;

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = BooleanGate>,
        gate_outputs: impl Iterator<Item = TensorShare>,
        mut inputs: impl Iterator<Item = TensorShare>,
        preprocessing_data: &mut SetupData,
    ) -> Self::Msg {
        let shares = interactive_gates
            .zip(gate_outputs)
            .map(|(gate, output)| {
                let inputs = inputs.by_ref().take(gate.input_size());
                gate.compute_delta_share(party_id, inputs, preprocessing_data, output)
            })
            .collect();
        Msg::DeltaShares { shares }
    }

    fn evaluate_interactive(
        &self,
        _party_id: usize,
        _interactive_gates: impl Iterator<Item = BooleanGate>,
        gate_outputs: impl Iterator<Item = TensorShare>,
        Msg::DeltaShares { shares }: Msg,
        Msg::DeltaShares {
            shares: other_shares,
        }: Msg,
        _preprocessing_data: &mut SetupData,
    ) -> Self::ShareStorage {
        gate_outputs
            .zip(shares)
            .zip(other_shares)
            .map(|((mut out_share, my_delta), other_delta)| {
                out_share.set_public(my_delta ^ other_delta);
                out_share
            })
            .collect()
    }

    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self,
        _party_id: usize,
        circuit: &ExecutableCircuit<Self::Gate, Idx>,
    ) -> GateOutputs<Self::ShareStorage> {
        let storage: Vec<_> = circuit
            .gate_counts()
            .map(|(gate_count, simd_size)| {
                assert_eq!(None, simd_size, "SIMD not supported for tensor_aby2");
                ScGateOutputs::Scalar(DeltaShareStorage::repeat(Default::default(), gate_count))
            })
            .collect();
        let mut storage = GateOutputs::new(storage);

        for (gate, sc_gate_id, parents) in circuit.iter_with_parents() {
            let gate_input_iter = parents.map(|parent| storage.get(parent));
            let rng = match self
                .delta_sharing_state
                .input_position_share_type_map
                .get(&sc_gate_id.gate_id.as_usize())
            {
                // The first case doesn't really matter, as the rng won't be used
                None => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Local) => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Remote) => &mut self.delta_sharing_state.remote_joint_rng,
            };
            let output = gate.setup_output_share(gate_input_iter, rng);
            storage.set(sc_gate_id, output);
        }

        storage
    }
}

impl BooleanGate {
    /// output_share contains the previously randomly generated private share needed for the
    /// evaluation
    fn compute_delta_share(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = TensorShare>,
        preprocessing_data: &mut SetupData,
        output_share: TensorShare,
    ) -> PartialShare {
        assert!(matches!(party_id, 0 | 1));
        match self {
            BooleanGate::Base(_) | BooleanGate::Xor | BooleanGate::Inv => {
                panic!("called on non interactive gate")
            }
            BooleanGate::And2 => {
                let TensorShare::Scalar(a) = inputs.next().expect("Empty input") else { 
                    panic!("non-scalar input to scalar gate");
                };
                let TensorShare::Scalar(b) = inputs.next().expect("Insufficient input") else {
                    panic!("non-scalar input to scalar gate");
                };
                let plain_ab = a.public & b.public;
                let PartialShare::Scalar(priv_delta) = preprocessing_data
                    .eval_shares
                    .pop()
                    .expect("Missing delta_ab_share") else {
                    panic!("non-scalar EvalShare to scalar gate");

                };
                let TensorShare::Scalar(output_share) = output_share else {
                    panic!("non-scalar input to scalar gate");
                };
                let share = (party_id == 1) & plain_ab
                    ^ a.public & b.private
                    ^ b.public & a.private
                    ^ priv_delta
                    ^ output_share.private;
                PartialShare::Scalar(share)
            }
            BooleanGate::And3 => {
                todo!()
            }
            BooleanGate::And4 => {
                todo!()
            }
            BooleanGate::Tensor(TensorGate::MatMult { rows, cols }) => {
                let TensorShare::Matrix(a) = inputs.next().expect("Empty input") else {
                    panic!("non-matrix input to MatMult gate");
                };
                let TensorShare::Matrix(b) = inputs.next().expect("Insufficient input") else {
                    panic!("non-matrix input to MatMult gate");
                };
                let PartialShare::Matrix(gamma_delta) = preprocessing_data
                    .eval_shares
                    .pop()
                    .expect("Missing gamma_delta share") else {
                    panic!("non-matrix EvalShare to MatMul gate");
                };
                let TensorShare::Matrix(output_share) = output_share else {
                    panic!("non-matrix input to MatMul gate");
                };

                let mut out = a.public.clone().mat_mul(&b.private)
                    ^ a.private.mat_mul(&b.public)
                    ^ gamma_delta
                    ^ &output_share.private;
                if party_id == 0 {
                    out = out ^ a.public.mat_mul(&b.public);
                }
                assert_eq!((*rows, *cols), out.dim(), "Out Matrix has wrong dimensions");
                PartialShare::Matrix(out)
            }
            BooleanGate::Tensor(_) => todo!(),
        }
    }

    fn setup_output_share(
        &self,
        mut inputs: impl Iterator<Item = TensorShare>,
        mut rng: impl Rng + CryptoRng,
    ) -> TensorShare {
        match self {
            BooleanGate::Base(base_gate) => match base_gate {
                // TODO handle dim
                BaseGate::Input(dim) => {
                    // TODO this needs to randomly generate the private part of the share
                    //  however, there is a problem. This private part needs to match the private
                    //  part of the input which is supplied to Executor::execute
                    //  one option to ensure this would be to use two PRNGs seeded with the same
                    //  seed for this method and for the Sharing of the input
                    //  Or maybe the setup gate outputs of the input gates can be passed to
                    //  the share() method?
                    match dim.as_slice() {
                        [] => TensorShare::Scalar(Share {
                            public: Default::default(),
                            private: rng.gen(),
                        }),
                        &[size] => TensorShare::Vec(ShareVec {
                            public: Default::default(),
                            private: rand_bitvec(size, &mut rng),
                        }),
                        &[rows, cols] => TensorShare::Matrix(ShareMatrix {
                            public: Default::default(),
                            private: BitMatrix::random(rng, rows, cols),
                        }),
                        _ => panic!("Illegal dimensions"),
                    }
                }
                BaseGate::Output(_)
                | BaseGate::SubCircuitInput(_)
                | BaseGate::SubCircuitOutput(_)
                | BaseGate::ConnectToMain(_) => inputs.next().expect("Empty input"),
                BaseGate::Constant(_) => todo!(),
                BaseGate::ConnectToMainFromSimd(_) => {
                    panic!("No SIMD support for BoolTensorAby2")
                }
            },
            BooleanGate::And2 => {
                // input is not actually needed at this stage
                TensorShare::Scalar(Share {
                    // Todo this should really use a configurable Rng
                    private: rng.gen(), //thread_rng().gen(),
                    public: Default::default(),
                })
            }
            BooleanGate::And3 => {
                todo!("Rng val")
            }
            BooleanGate::And4 => {
                todo!("Rng val")
            }
            BooleanGate::Xor => {
                let TensorShare::Scalar(mut a) = inputs.next().expect("Empty input") else {
                    panic!("non-scalar input to scalar gate");
                };
                let TensorShare::Scalar(b) = inputs.next().expect("Empty input") else {
                    panic!("non-scalar input to scalar gate");
                };
                // TODO change this
                a.private ^= b.private;
                TensorShare::Scalar(a)
            }
            BooleanGate::Inv => {
                todo!("I think just return input")
            }
            BooleanGate::Tensor(TensorGate::MatMult { rows, cols }) => {
                let TensorShare::Matrix(a) = inputs.next().expect("Empty input") else {
                    panic!("non-matrix input to MatMult gate");
                };
                let TensorShare::Matrix(b) = inputs.next().expect("Empty input") else {
                    panic!("non-matrix input to MatMult gate");
                };
                let (in_rows, _) = a.private.dim();
                let (_, in_cols) = b.private.dim();
                assert_eq!(in_rows, *rows);
                assert_eq!(in_cols, *cols);
                let private = BitMatrix::random(rng, *rows, *cols);
                TensorShare::Matrix(ShareMatrix {
                    public: Default::default(),
                    private,
                })
            }
            BooleanGate::Tensor(_) => todo!(),
        }
    }

    fn setup_data_circ(
        &self,
        input_shares: &[Vec<Secret>],
        _setup_sub_circ_cache: &mut AHashMap<Vec<Secret>, Secret>,
    ) -> Vec<Secret> {
        // TODO return SmallVec here?
        match self {
            BooleanGate::And2 => {
                let out = input_shares[0][0].clone() & &input_shares[1][0];
                vec![out]
                // TODO do the caching!
                // // skip the empty and single elem sets
                // let ab: Vec<_> = input_shares.take(2).cloned().collect();
                //
                // match setup_sub_circ_cache.get(&ab) {
                //     None => match &ab[..] {
                //         [a, b] => {
                //             let sh = a.clone() & b;
                //             setup_sub_circ_cache.insert(ab, sh.clone());
                //             vec![sh]
                //         }
                //         _ => unreachable!("Insufficient input_shares"),
                //     },
                //     Some(processed_set) => vec![processed_set.clone()],
                // }
            }
            BooleanGate::And3 | BooleanGate::And4 => todo!("not impled"),
            BooleanGate::Tensor(TensorGate::MatMult { rows, cols }) => {
                let [a, b] = input_shares else {
                    panic!("2 inputs needed for MatMult")
                };
                let r = *cols;
                let a_cols = a.len() / *rows;
                let b_rows = b.len() / *cols;
                // cols iterators with a stride of cols iterates b in a transposed way
                let b_trans: Vec<_> = (0..r).flat_map(|k| b.iter().skip(k).step_by(r)).collect();

                a.chunks_exact(a_cols)
                    .flat_map(|row| iter::repeat(row).take(*cols))
                    .zip(b_trans.chunks_exact(b_rows).cycle())
                    .map(|(a_row, b_col)| {
                        a_row
                            .iter()
                            .zip(b_col)
                            .map(|(a_el, b_el)| a_el.clone() & *b_el)
                            .reduce(|acc, el| acc ^ el)
                            .expect("Empty row/col")
                    })
                    .collect()
            }
            non_interactive => {
                assert!(non_interactive.is_non_interactive());
                panic!("Called setup_data_circ on non_interactive gate")
            }
        }
    }
}

impl Gate for BooleanGate {
    type Share = TensorShare;
    type DimTy = DynDim;

    fn is_interactive(&self) -> bool {
        matches!(
            self,
            Self::And2 | Self::And3 | Self::And4 | Self::Tensor(TensorGate::MatMult { .. })
        )
    }

    fn input_size(&self) -> usize {
        match self {
            Self::Base(base_gate) => base_gate.input_size(),
            Self::Inv => 1,
            Self::And2 | Self::Xor => 2,
            Self::And3 => 3,
            Self::And4 => 4,
            Self::Tensor(TensorGate::MatMult { .. }) => 2,
            Self::Tensor(_) => todo!(),
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share, Self::DimTy>> {
        match self {
            Self::Base(base_gate) => Some(base_gate),
            _ => None,
        }
    }

    fn wrap_base_gate(base_gate: BaseGate<Self::Share, Self::DimTy>) -> Self {
        Self::Base(base_gate)
    }

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = Self::Share>,
    ) -> Self::Share {
        match self {
            Self::Base(base) => base.evaluate_non_interactive(party_id, inputs),
            Self::And2 | Self::And3 | Self::And4 => {
                panic!("Called evaluate_non_interactive on Gate::And<N>")
            }
            Self::Xor => {
                let a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");
                a ^ b
            }
            Self::Inv => {
                let inp = inputs.next().expect("Empty input");
                if party_id == 0 {
                    !inp
                } else {
                    inp
                }
            }
            Self::Tensor(_) => todo!(),
        }
    }
}

impl TensorShare {
    pub fn get_private(&self) -> PartialShare {
        match self {
            TensorShare::Scalar(Share { private, .. }) => PartialShare::Scalar(*private),
            TensorShare::Vec(share_vec) => PartialShare::Vec(share_vec.private.clone()),
            TensorShare::Matrix(share_mat) => PartialShare::Matrix(share_mat.private.clone()),
        }
    }
    pub fn get_public(&self) -> PartialShare {
        match self {
            TensorShare::Scalar(Share { public, .. }) => PartialShare::Scalar(*public),
            TensorShare::Vec(share_vec) => PartialShare::Vec(share_vec.public.clone()),
            TensorShare::Matrix(share_mat) => PartialShare::Matrix(share_mat.public.clone()),
        }
    }
}

impl super::Share for TensorShare {
    // TODO this doesn't really make sense at the moment, but is necessary so that
    //  the executor can use these shares since it current requires that the
    //  Gate::SimdShare = Protocol::ShareStorage ... Yea, not ideal
    type SimdShare = DeltaShareStorage;
}

impl Default for TensorShare {
    fn default() -> Self {
        Self::Scalar(Share::default())
    }
}

impl From<BaseGate<TensorShare, DynDim>> for BooleanGate {
    fn from(base_gate: BaseGate<TensorShare, DynDim>) -> Self {
        Self::Base(base_gate)
    }
}

impl From<&bristol::Gate> for BooleanGate {
    fn from(gate: &bristol::Gate) -> Self {
        match gate {
            bristol::Gate::And(_) => Self::And2,
            bristol::Gate::Xor(_) => Self::Xor,
            bristol::Gate::Inv(_) => Self::Inv,
        }
    }
}

impl ShareStorage<TensorShare> for DeltaShareStorage {
    fn len(&self) -> usize {
        self.shares.len()
    }

    fn repeat(val: TensorShare, len: usize) -> Self {
        repeat(val).take(len).collect()
    }

    fn set(&mut self, idx: usize, val: TensorShare) {
        self.shares[idx] = val;
    }

    fn get(&self, idx: usize) -> TensorShare {
        self.shares[idx].clone()
    }
}

impl IntoIterator for DeltaShareStorage {
    type Item = TensorShare;
    type IntoIter = vec::IntoIter<TensorShare>;

    fn into_iter(self) -> Self::IntoIter {
        self.shares.into_iter()
    }
}

impl FromIterator<TensorShare> for DeltaShareStorage {
    fn from_iter<T: IntoIterator<Item = TensorShare>>(iter: T) -> Self {
        Self {
            shares: iter.into_iter().collect(),
        }
    }
}

impl Extend<TensorShare> for DeltaShareStorage {
    fn extend<T: IntoIterator<Item = TensorShare>>(&mut self, iter: T) {
        self.shares.extend(iter);
    }
}

impl SetupData {
    /// Evaluation shares for interactive gates in **topological** order
    pub fn from_raw(mut eval_shares: Vec<PartialShare>) -> Self {
        eval_shares.reverse();
        Self { eval_shares }
    }

    pub fn len(&self) -> usize {
        self.eval_shares.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SetupStorage for SetupData {
    fn len(&self) -> usize {
        self.eval_shares.len()
    }

    fn split_off_last(&mut self, count: usize) -> Self {
        Self {
            eval_shares: self.eval_shares.split_off(self.len() - count),
        }
    }
}

impl Share {
    pub fn new(private: bool, public: bool) -> Self {
        Self { public, private }
    }

    pub fn get_public(&self) -> bool {
        self.public
    }

    pub fn get_private(&self) -> bool {
        self.private
    }
}

impl Not for Share {
    type Output = Share;

    fn not(self) -> Self::Output {
        Self {
            public: self.public,
            private: !self.private,
        }
    }
}

impl TensorShare {
    #[track_caller]
    fn set_public(&mut self, public: PartialShare) {
        match (self, public) {
            (TensorShare::Scalar(share), PartialShare::Scalar(delta)) => {
                share.public = delta;
            }
            (TensorShare::Vec(ShareVec { public, .. }), PartialShare::Vec(delta)) => {
                *public = delta;
            }
            (TensorShare::Matrix(ShareMatrix { public, .. }), PartialShare::Matrix(delta)) => {
                *public = delta;
            }
            _ => {
                panic!("set_public called on TensorShares with different variants");
            }
        }
    }
}

impl BitXor for TensorShare {
    type Output = TensorShare;

    fn bitxor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (TensorShare::Scalar(this), TensorShare::Scalar(rhs)) => TensorShare::Scalar(Share {
                public: this.public ^ rhs.public,
                private: this.private ^ rhs.private,
            }),
            (TensorShare::Matrix(this), TensorShare::Matrix(rhs)) => {
                TensorShare::Matrix(ShareMatrix {
                    public: this.public ^ rhs.public,
                    private: this.private ^ rhs.private,
                })
            }
            _ => todo!(),
        }
    }
}

impl Not for TensorShare {
    type Output = TensorShare;

    fn not(self) -> Self::Output {
        match self {
            TensorShare::Scalar(share) => TensorShare::Scalar(!share),
            TensorShare::Vec(_) => {
                todo!()
            }
            TensorShare::Matrix(_) => {
                todo!()
            }
        }
    }
}

impl PartialShare {
    pub fn into_scalar(self) -> Option<bool> {
        match self {
            Self::Scalar(val) => Some(val),
            _ => None,
        }
    }

    pub fn into_vec(self) -> Option<BitVec> {
        match self {
            Self::Vec(val) => Some(val),
            _ => None,
        }
    }

    pub fn into_matrix(self) -> Option<BitMatrix<u8>> {
        match self {
            Self::Matrix(val) => Some(val),
            _ => None,
        }
    }

    pub fn flatten(self) -> BitVec {
        match self {
            PartialShare::Scalar(val) => BitVec::repeat(val, 1),
            PartialShare::Vec(val) => val,
            PartialShare::Matrix(val) => BitVec::from_vec(val.into_vec()),
        }
    }
}

impl BitXor for PartialShare {
    type Output = PartialShare;

    fn bitxor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (PartialShare::Scalar(this), PartialShare::Scalar(rhs)) => {
                PartialShare::Scalar(this ^ rhs)
            }
            (PartialShare::Vec(this), PartialShare::Vec(rhs)) => {
                assert_eq!(
                    this.len(),
                    rhs.len(),
                    "Xor on PartialShare::Vec of unequal length"
                );
                PartialShare::Vec(this ^ rhs)
            }
            (PartialShare::Matrix(this), PartialShare::Matrix(rhs)) => {
                assert_eq!(
                    this.dim(),
                    rhs.dim(),
                    "Xor on PartialShare::Matrix of unequal dimensions"
                );
                PartialShare::Matrix(this ^ rhs)
            }
            _ => {
                panic!("BitXor called on PartialShare with different variants");
            }
        }
    }
}

impl DeltaSharing {
    pub fn new(
        priv_seed: [u8; 32],
        local_joint_seed: [u8; 32],
        remote_joint_seed: [u8; 32],
        input_position_share_type_map: HashMap<usize, ShareType>,
    ) -> Self {
        Self {
            private_rng: ChaChaRng::from_seed(priv_seed),
            local_joint_rng: ChaChaRng::from_seed(local_joint_seed),
            remote_joint_rng: ChaChaRng::from_seed(remote_joint_seed),
            input_position_share_type_map,
        }
    }

    /// # Warning - Insercure
    /// Insecurely initialize DeltaSharing RNGs with default value. No input_position_share_type_map
    /// is needed when all the RNGs are the same.
    pub fn insecure_default() -> Self {
        Self {
            private_rng: ChaChaRng::seed_from_u64(0),
            local_joint_rng: ChaChaRng::seed_from_u64(0),
            remote_joint_rng: ChaChaRng::seed_from_u64(0),
            input_position_share_type_map: HashMap::new(),
        }
    }

    pub fn share(&mut self, input: Vec<PartialShare>) -> (DeltaShareStorage, Vec<PartialShare>) {
        input
            .into_iter()
            .map(|public| match public {
                PartialShare::Scalar(bit) => {
                    let my_delta = self.private_rng.gen();
                    let other_delta: bool = self.local_joint_rng.gen();
                    let plain_delta = bit ^ my_delta ^ other_delta;
                    let my_share = TensorShare::Scalar(Share::new(my_delta, plain_delta));
                    (my_share, PartialShare::Scalar(plain_delta))
                }
                PartialShare::Vec(_) => {
                    todo!()
                }
                PartialShare::Matrix(plain) => {
                    let (rows, cols) = plain.dim();
                    let my_delta = BitMatrix::random(&mut self.private_rng, rows, cols);
                    let other_delta = BitMatrix::random(&mut self.local_joint_rng, rows, cols);
                    let plain_delta = plain ^ &my_delta ^ &other_delta;
                    let my_share = TensorShare::Matrix(ShareMatrix {
                        public: plain_delta.clone(),
                        private: my_delta,
                    });
                    let other_partial = PartialShare::Matrix(plain_delta);
                    (my_share, other_partial)
                }
            })
            .unzip()
    }

    pub fn plain_delta_to_share(&mut self, plain_deltas: Vec<PartialShare>) -> DeltaShareStorage {
        plain_deltas
            .into_iter()
            .map(|plain_delta| match plain_delta {
                PartialShare::Scalar(bit) => {
                    TensorShare::Scalar(Share::new(self.remote_joint_rng.gen(), bit))
                }
                PartialShare::Vec(_) => {
                    todo!()
                }
                PartialShare::Matrix(mat) => {
                    let (rows, cols) = mat.dim();
                    let private = BitMatrix::random(&mut self.remote_joint_rng, rows, cols);
                    TensorShare::Matrix(ShareMatrix {
                        public: mat,
                        private,
                    })
                }
            })
            .collect()
    }

    pub fn reconstruct(a: DeltaShareStorage, b: DeltaShareStorage) -> Vec<PartialShare> {
        a.into_iter()
            .zip(b)
            .map(|(sh1, sh2)| {
                assert_eq!(
                    sh1.get_public(),
                    sh2.get_public(),
                    "Public shares of outputs can't differ"
                );
                match (sh1, sh2) {
                    (TensorShare::Scalar(sh1), TensorShare::Scalar(sh2)) => PartialShare::Scalar(
                        sh1.get_public() ^ sh1.get_private() ^ sh2.get_private(),
                    ),
                    (TensorShare::Matrix(sh1), TensorShare::Matrix(sh2)) => {
                        PartialShare::Matrix(sh1.public ^ sh1.private ^ sh2.private)
                    }
                    _ => unreachable!(),
                }
            })
            .collect()
    }
}

impl<Mtp> AbySetupProvider<Mtp> {
    pub fn new(
        party_id: usize,
        mt_provider: Mtp,
        sender: mpc_channel::Sender<AbySetupMsg>,
        receiver: mpc_channel::Receiver<AbySetupMsg>,
    ) -> Self {
        Self {
            party_id,
            mt_provider,
            sender,
            receiver,
        }
    }
}

#[async_trait]
impl<MtpErr, Mtp> FunctionDependentSetup<DeltaShareStorage, BooleanGate, usize>
    for AbySetupProvider<Mtp>
where
    MtpErr: Debug,
    Mtp: MTProvider<Output = MulTriples, Error = MtpErr> + Send,
{
    type Output = SetupData;
    type Error = Infallible;

    async fn setup(
        &mut self,
        shares: &GateOutputs<DeltaShareStorage>,
        circuit: &ExecutableCircuit<BooleanGate, usize>,
    ) -> Result<Self::Output, Self::Error> {
        let circ_builder: CircuitBuilder<boolean_gmw::BooleanGate> = CircuitBuilder::new();
        let old = circ_builder.install();
        let total_inputs = circuit
            .interactive_iter()
            .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
            .sum();

        let mut circ_inputs = BitVec::with_capacity(total_inputs);
        // Block is needed as otherwise !Send types are held over .await
        let setup_outputs: Vec<Vec<_>> = {
            let mut input_sw_map: AHashMap<_, Vec<Secret>> = AHashMap::with_capacity(total_inputs);
            let mut setup_outputs = Vec::with_capacity(circuit.interactive_count());
            let mut setup_sub_circ_cache = AHashMap::with_capacity(total_inputs);
            for (gate, _gate_id, parents) in circuit.interactive_with_parents_iter() {
                let mut gate_input_shares = vec![];
                // TODO order of parent gates is relevant for some gate types!
                parents.for_each(|parent| match input_sw_map.entry(parent) {
                    Entry::Vacant(vacant) => {
                        let bit_inputs = shares.get(parent).get_private().flatten();
                        let sh = inputs(bit_inputs.len());
                        gate_input_shares.push(sh.clone());
                        circ_inputs.extend_from_bitslice(&bit_inputs);
                        vacant.insert(sh);
                    }
                    Entry::Occupied(occupied) => {
                        gate_input_shares.push(occupied.get().clone());
                    }
                });
                // TODO This was here before, I think it was because of the caching, but this
                //  could introduce a bug because the circ_inputs.push is not in the same order
                // gate_input_shares.sort();
                let t = gate.setup_data_circ(&gate_input_shares, &mut setup_sub_circ_cache);
                setup_outputs.push(t);
            }
            setup_outputs
                .into_iter()
                .map(|v| v.into_iter().map(|opt_sh| opt_sh.output()).collect())
                .collect()
        };

        let setup_data_circ: ExecutableCircuit<boolean_gmw::BooleanGate, _> =
            ExecutableCircuit::DynLayers(CircuitBuilder::global_into_circuit());
        old.install();
        let mut executor: Executor<BooleanGmw, DefaultIdx> =
            Executor::new(&setup_data_circ, self.party_id, &mut self.mt_provider)
                .await
                .expect("Executor::new in AbySetupProvider");
        executor
            .execute(circ_inputs, &mut self.sender, &mut self.receiver)
            .await
            .unwrap();
        let ScGateOutputs::Scalar(executor_gate_outputs) = executor.gate_outputs().get_sc(0) else {
           panic!("No SIMD support for BoolTensorAby2") 
        };
        let eval_shares = circuit
            .interactive_iter()
            .zip(setup_outputs)
            .map(|((gate, _gate_id), mut setup_out)| match gate {
                BooleanGate::And2 => {
                    assert_eq!(setup_out.len(), 1);
                    let out_id = setup_out.pop().unwrap();
                    let share = executor_gate_outputs.get(out_id.as_usize());
                    PartialShare::Scalar(share)
                }
                BooleanGate::And3 | BooleanGate::And4 => {
                    todo!("not impled")
                }
                BooleanGate::Tensor(TensorGate::MatMult { rows, cols }) => {
                    let bv: BitVec = setup_out
                        .into_iter()
                        .map(|out_id| executor_gate_outputs.get(out_id.as_usize()))
                        .collect();
                    PartialShare::Matrix(BitMatrix::from_bits(&bv, rows, cols))
                }
                _ => unreachable!(),
            })
            .collect();
        Ok(SetupData::from_raw(eval_shares))
    }
}

impl Default for Msg {
    fn default() -> Self {
        Self::DeltaShares { shares: vec![] }
    }
}
