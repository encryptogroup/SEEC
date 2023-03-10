use crate::circuit::base_circuit::BaseGate;
use crate::circuit::{DefaultIdx, ExecutableCircuit, GateIdx};
use crate::common::BitVec;
use crate::executor::{Executor, GateOutputs, ScGateOutputs};
use crate::mul_triple::boolean::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{
    boolean_gmw, FunctionDependentSetup, Gate, Protocol, ScalarDim, SetupStorage, ShareStorage,
};
use crate::secret::Secret;
use crate::{bristol, executor, CircuitBuilder};
use ahash::AHashMap;
use async_trait::async_trait;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Debug;
use std::ops::Not;

pub struct BooleanAby2 {
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

#[derive(Copy, Clone)]
pub enum ShareType {
    Local,
    Remote,
}

#[derive(Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct Share {
    public: bool,
    private: bool,
}

#[derive(Clone, Debug, Default, Hash, PartialOrd, PartialEq)]
pub struct DeltaShareStorage {
    public: BitVec,
    private: BitVec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg {
    Delta { delta: Vec<u8> },
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum BooleanGate {
    Base(BaseGate<Share, ScalarDim>),
    And2,
    And3,
    And4,
    Xor,
    Inv,
}

/// Contains preprocessing data ([\delta_ab]_i for interactive gates in
/// **reverse** topological order. This is needed to evaluate interactive gates.
#[derive(Clone, Default)]
pub struct SetupData {
    eval_shares: Vec<EvalShares>,
}

#[derive(Clone)]
pub struct EvalShares {
    shares: BitVec,
}

impl BooleanAby2 {
    pub fn new(sharing_state: DeltaSharing) -> Self {
        Self {
            delta_sharing_state: sharing_state,
        }
    }
}

pub type AbySetupMsg = executor::Message<BooleanGmw>;

pub struct AbySetupProvider<Mtp> {
    party_id: usize,
    mt_provider: Mtp,
    sender: mpc_channel::Sender<AbySetupMsg>,
    receiver: mpc_channel::Receiver<AbySetupMsg>,
}

impl Protocol for BooleanAby2 {
    const SIMD_SUPPORT: bool = false;
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
        gate_outputs: impl Iterator<Item = Share>,
        mut inputs: impl Iterator<Item = Share>,
        preprocessing_data: &mut SetupData,
    ) -> Self::Msg {
        let delta: BitVec = interactive_gates
            .zip(gate_outputs)
            .map(|(gate, output)| {
                assert!(matches!(gate, BooleanGate::And2));
                let inputs = inputs.by_ref().take(gate.input_size());
                gate.compute_delta_share(party_id, inputs, preprocessing_data, output)
            })
            .collect();
        Msg::Delta {
            delta: delta.into_vec(),
        }
    }

    fn evaluate_interactive(
        &self,
        _party_id: usize,
        _interactive_gates: impl Iterator<Item = BooleanGate>,
        gate_outputs: impl Iterator<Item = Share>,
        Msg::Delta { delta }: Msg,
        Msg::Delta { delta: other_delta }: Msg,
        _preprocessing_data: &mut SetupData,
    ) -> Self::ShareStorage {
        let delta = BitVec::from_vec(delta);
        let other_delta = BitVec::from_vec(other_delta);
        gate_outputs
            .zip(delta)
            .zip(other_delta)
            .map(|((mut out_share, my_delta), other_delta)| {
                out_share.public = my_delta ^ other_delta;
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
            // Okay to use the default value, as they will be overwritten in the next step
            .map(|(gate_count, simd_size)| {
                assert_eq!(None, simd_size, "SIMD not supported for ABY2 protocol");
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
        mut inputs: impl Iterator<Item = Share>,
        preprocessing_data: &mut SetupData,
        output_share: Share,
    ) -> bool {
        assert!(matches!(party_id, 0 | 1));
        assert!(matches!(self, BooleanGate::And2));
        let a = inputs.next().expect("Empty input");
        let b = inputs.next().expect("Insufficient input");
        let plain_ab = a.public & b.public;
        let mut priv_delta = preprocessing_data
            .eval_shares
            .pop()
            .expect("Missing delta_ab_share");
        (party_id == 1) & plain_ab
            ^ a.public & b.private
            ^ b.public & a.private
            ^ priv_delta.shares.pop().expect("Missing eval share")
            ^ output_share.private
    }

    fn setup_output_share(
        &self,
        mut inputs: impl Iterator<Item = Share>,
        mut rng: impl Rng,
    ) -> Share {
        match self {
            BooleanGate::Base(base_gate) => match base_gate {
                BaseGate::Input(_) => {
                    // TODO this needs to randomly generate the private part of the share
                    //  however, there is a problem. This private part needs to match the private
                    //  part of the input which is supplied to Executor::execute
                    //  one option to ensure this would be to use two PRNGs seeded with the same
                    //  seed for this method and for the Sharing of the input
                    //  Or maybe the setup gate outputs of the input gates can be passed to
                    //  the share() method?
                    Share {
                        public: Default::default(),
                        private: rng.gen(),
                    }
                }
                BaseGate::Output(_)
                | BaseGate::SubCircuitInput(_)
                | BaseGate::SubCircuitOutput(_)
                | BaseGate::ConnectToMain(_) => inputs.next().expect("Empty input"),
                BaseGate::Constant(_) => todo!(),
                BaseGate::ConnectToMainFromSimd(_) => {
                    unimplemented!("SIMD currently not supported for ABY2")
                }
            },
            BooleanGate::And2 => {
                // input is not actually needed at this stage
                Share {
                    // Todo this should really use a configurable Rng
                    private: rng.gen(), //thread_rng().gen(),
                    public: Default::default(),
                }
            }
            BooleanGate::And3 => {
                todo!("Rng val")
            }
            BooleanGate::And4 => {
                todo!("Rng val")
            }
            BooleanGate::Xor => {
                let mut a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");
                // TODO change this
                a.private ^= b.private;
                a
            }
            BooleanGate::Inv => {
                todo!("I think just return input")
            }
        }
    }

    fn setup_data_circ<'a>(
        &self,
        input_shares: impl Iterator<Item = &'a Secret>,
        setup_sub_circ_cache: &mut AHashMap<Vec<Secret>, Secret>,
    ) -> Vec<Secret> {
        // TODO return SmallVec here?
        match self {
            BooleanGate::And2 => {
                // skip the empty and single elem sets
                let ab: Vec<_> = input_shares.take(2).cloned().collect();

                match setup_sub_circ_cache.get(&ab) {
                    None => match &ab[..] {
                        [a, b] => {
                            let sh = a.clone() & b;
                            setup_sub_circ_cache.insert(ab, sh.clone());
                            vec![sh]
                        }
                        _ => unreachable!("Insufficient input_shares"),
                    },
                    Some(processed_set) => vec![processed_set.clone()],
                }
            }
            BooleanGate::And3 | BooleanGate::And4 => todo!("not impled"),
            non_interactive => {
                assert!(non_interactive.is_non_interactive());
                panic!("Called setup_data_circ on non_interactive gate")
            }
        }
    }
}

impl Gate for BooleanGate {
    type Share = Share;
    type DimTy = ScalarDim;

    fn is_interactive(&self) -> bool {
        matches!(
            self,
            BooleanGate::And2 | BooleanGate::And3 | BooleanGate::And4
        )
    }

    fn input_size(&self) -> usize {
        match self {
            BooleanGate::Base(base_gate) => base_gate.input_size(),
            BooleanGate::Inv => 1,
            BooleanGate::And2 | BooleanGate::Xor => 2,
            BooleanGate::And3 => 3,
            BooleanGate::And4 => 4,
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

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = Self::Share>,
    ) -> Self::Share {
        match self {
            BooleanGate::Base(base) => base.evaluate_non_interactive(party_id, inputs.by_ref()),
            BooleanGate::And2 | BooleanGate::And3 | BooleanGate::And4 => {
                panic!("Called evaluate_non_interactive on Gate::And<N>")
            }
            BooleanGate::Xor => {
                let a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");
                // TODO change this
                a.xor(b)
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
}

impl Default for Msg {
    fn default() -> Self {
        Msg::Delta { delta: vec![] }
    }
}

impl super::Share for Share {
    type SimdShare = DeltaShareStorage;
}

impl From<BaseGate<Share>> for BooleanGate {
    fn from(base_gate: BaseGate<Share>) -> Self {
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

impl ShareStorage<Share> for DeltaShareStorage {
    fn len(&self) -> usize {
        debug_assert_eq!(self.private.len(), self.public.len());
        self.private.len()
    }

    fn repeat(val: Share, len: usize) -> Self {
        Self {
            private: BitVec::repeat(val.private, len),
            public: BitVec::repeat(val.public, len),
        }
    }

    fn set(&mut self, idx: usize, val: Share) {
        self.public.set(idx, val.public);
        self.private.set(idx, val.private);
    }

    fn get(&self, idx: usize) -> Share {
        Share {
            public: self.public[idx],
            private: self.private[idx],
        }
    }
}

pub struct ShareIter {
    public: <BitVec as IntoIterator>::IntoIter,
    private: <BitVec as IntoIterator>::IntoIter,
}

impl IntoIterator for DeltaShareStorage {
    type Item = Share;
    type IntoIter = ShareIter;

    fn into_iter(self) -> Self::IntoIter {
        ShareIter {
            public: self.public.into_iter(),
            private: self.private.into_iter(),
        }
    }
}

impl Iterator for ShareIter {
    type Item = Share;

    fn next(&mut self) -> Option<Self::Item> {
        let public = self.public.next()?;
        let private = self.private.next()?;
        Some(Share { public, private })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.public.size_hint()
    }
}

impl ExactSizeIterator for ShareIter {}

impl FromIterator<Share> for DeltaShareStorage {
    fn from_iter<T: IntoIterator<Item = Share>>(iter: T) -> Self {
        let (public, private) = iter
            .into_iter()
            .map(|share| (share.public, share.private))
            .unzip();
        Self { public, private }
    }
}

impl Extend<Share> for DeltaShareStorage {
    fn extend<T: IntoIterator<Item = Share>>(&mut self, iter: T) {
        for share in iter {
            self.private.push(share.private);
            self.public.push(share.public);
        }
    }
}

impl SetupData {
    /// Evaluation shares for interactive gates in **topological** order
    pub fn from_raw(mut eval_shares: Vec<EvalShares>) -> Self {
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

    fn xor(&self, other: Share) -> Share {
        Share {
            public: self.public ^ other.public,
            private: self.private ^ other.private,
        }
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

    pub fn share(&mut self, input: BitVec) -> (DeltaShareStorage, BitVec) {
        input
            .into_iter()
            .map(|bit| {
                let my_delta = self.private_rng.gen();
                let other_delta: bool = self.local_joint_rng.gen();
                let plain_delta = bit ^ my_delta ^ other_delta;
                let my_share = Share::new(my_delta, plain_delta);
                (my_share, plain_delta)
            })
            .unzip()
    }

    pub fn plain_delta_to_share(&mut self, plain_deltas: BitVec) -> DeltaShareStorage {
        plain_deltas
            .into_iter()
            .map(|plain_delta| Share::new(self.remote_joint_rng.gen(), plain_delta))
            .collect()
    }

    pub fn reconstruct(a: DeltaShareStorage, b: DeltaShareStorage) -> BitVec {
        a.into_iter()
            .zip(b)
            .map(|(sh1, sh2)| {
                assert_eq!(
                    sh1.get_public(),
                    sh2.get_public(),
                    "Public shares of outputs can't differ"
                );
                sh1.get_public() ^ sh1.get_private() ^ sh2.get_private()
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
        let total_inputs: usize = circuit
            .interactive_iter()
            .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
            .sum();

        let mut circ_inputs = BitVec::<usize>::with_capacity(total_inputs);
        // Block is needed as otherwise !Send types are held over .await
        let setup_outputs: Vec<Vec<_>> = {
            let mut input_sw_map: AHashMap<_, Secret> = AHashMap::with_capacity(total_inputs);
            let mut setup_outputs = Vec::with_capacity(circuit.interactive_count());
            let mut setup_sub_circ_cache = AHashMap::with_capacity(total_inputs);
            for (gate, _gate_id, parents) in circuit.interactive_with_parents_iter() {
                let mut gate_input_shares = vec![];
                parents.for_each(|parent| match input_sw_map.entry(parent) {
                    Entry::Vacant(vacant) => {
                        let sh = Secret::input(0);
                        gate_input_shares.push(sh.clone());
                        circ_inputs.push(shares.get(parent).get_private());
                        vacant.insert(sh);
                    }
                    Entry::Occupied(occupied) => {
                        gate_input_shares.push(occupied.get().clone());
                    }
                });
                gate_input_shares.sort();

                let t = gate.setup_data_circ(gate_input_shares.iter(), &mut setup_sub_circ_cache);
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
            panic!("SIMD not supported for ABY2");
        };

        let eval_shares = circuit
            .interactive_iter()
            .zip(setup_outputs)
            .map(|((gate, _gate_id), setup_out)| match gate {
                BooleanGate::And2 => {
                    let shares = setup_out
                        .into_iter()
                        .map(|out_id| executor_gate_outputs.get(out_id.as_usize()))
                        .collect();
                    EvalShares { shares }
                }
                BooleanGate::And3 | BooleanGate::And4 => {
                    todo!("not impled")
                }
                _ => unreachable!(),
            })
            .collect();
        Ok(SetupData::from_raw(eval_shares))
    }
}
