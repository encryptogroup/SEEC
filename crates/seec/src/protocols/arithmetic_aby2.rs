use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;
use std::error::Error;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Sub};
use ahash::AHashMap;
use async_trait::async_trait;
use itertools::Itertools;
use num_traits::{WrappingAdd, WrappingSub};
use rand::{distributions, Rng, SeedableRng};
use rand::distributions::Distribution;
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};
use seec_channel::multi::{MultiReceiver, MultiSender};
use crate::{executor, protocols, CircuitBuilder};
use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::executor::{Executor, GateOutputs, Input};
use crate::gate::base::BaseGate;
use crate::mul_triple::arithmetic::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::{arithmetic_gmw, FunctionDependentSetup, Gate, Protocol, Ring, ScalarDim, SetupStorage, ShareStorage};
use crate::protocols::aby2::ShareType;
use crate::protocols::arithmetic_gmw::ArithmeticGmw;
use crate::secret::Secret;
use crate::utils::take_arr;

#[derive(Clone, Debug)]
pub struct DeltaSharing<R> {
    pub(crate) private_rng: ChaChaRng,
    pub(crate) local_joint_rng: ChaChaRng,
    pub(crate) remote_joint_rng: ChaChaRng,
    // TODO ughh
    pub(crate) input_position_share_type_map: HashMap<usize, ShareType>,
    phantom: PhantomData<R>,
}

#[derive(Clone, Debug)]
pub struct ArithmeticAby2<R> {
    delta_sharing_state: DeltaSharing<R>,
}

#[derive(Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Debug, Default)]
pub struct Share<R> {
    pub(crate) public: R,
    pub(crate) private: R,
}

impl<R: Ring> Share<R> {
    pub fn get_public(&self) -> R {
        self.public.clone()
    }
    pub fn get_private(&self) -> R {
        self.private.clone()
    }
}

#[derive(Clone, Debug, Default, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct ShareVec<R> {
    pub(crate) public: Vec<R>,
    pub(crate) private: Vec<R>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg<R> {
    Delta { delta: Vec<R> },
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum ArithmeticGate<R> {
    Base(BaseGate<R, ScalarDim>),
    Mul,
    Add,
    Sub,
}

#[derive(Clone, Default)]
pub struct SetupData<R> {
    eval_shares: Vec<EvalShares<R>>,
}

#[derive(Clone)]
pub struct EvalShares<R> {
    shares: Vec<R>,
}

impl<R> ArithmeticAby2<R> {
    pub fn new(sharing_state: DeltaSharing<R>) -> Self {
        Self { delta_sharing_state: sharing_state }
    }
}

pub type AbySetupMsg<R> = executor::Message<ArithmeticGmw<R>>;

pub struct AbySetupProvider<Mtp, R: Ring> {
    party_id: usize,
    mt_provider: Mtp,
    sender: seec_channel::Sender<AbySetupMsg<R>>,
    receiver: seec_channel::Receiver<AbySetupMsg<R>>,
    setup_data: Option<SetupData<R>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstraSetupMsg<R>(Vec<R>);

type InputBy = crate::protocols::aby2::InputBy;

pub struct AstraSetupHelper<R> {
    sender: MultiSender<AstraSetupMsg<R>>,
    // shared rng with p0
    priv_seed_p0: [u8; 32],
    // shared rng with p1
    priv_seed_p1: [u8; 32],
    // shared between p0 and p1
    joint_seed: [u8; 32],
}

pub struct AstraSetupProvider<R> {
    party_id: usize,
    receiver: MultiReceiver<AstraSetupMsg<R>>,
    rng: ChaChaRng,
    setup_data: Option<SetupData<R>>,
}

impl<R> Protocol for ArithmeticAby2<R>
where
    R: Ring,
    distributions::Standard: Distribution<R>,
{
    const SIMD_SUPPORT: bool = false;
    type Plain = R;
    type Share = Share<R>;
    type Msg = Msg<R>;
    type SimdMsg = ();
    type Gate = ArithmeticGate<R>;
    type Wire = ();
    type ShareStorage = ShareVec<R>;
    type SetupStorage = SetupData<R>;

    fn share_constant(
        &self,
        _party_id: usize,
        output_share: Self::Share,
        val: Self::Plain,
    ) -> Self::Share {
        assert_ne!(output_share.private, R::ZERO, "Private part of constant share must be 0");
        Share {
            public: val,
            private: R::ZERO,
        }
    }

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        gate: &Self::Gate,
        mut inputs: impl Iterator<Item=Self::Share>
    ) -> Self::Share {
        match gate {
            ArithmeticGate::Base(base) =>
                base.default_evaluate(party_id, inputs.by_ref()),
            ArithmeticGate::Mul => panic!("Called evaluate_non_interactive on Gate::Mul"),
            ArithmeticGate::Add => {
                let a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");
                a.wrapping_add(&b)
            }
            ArithmeticGate::Sub => {
                let a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");
                a.wrapping_sub(&b)
            }
        }
    }

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item=Self::Gate>,
        gate_outputs: impl Iterator<Item=Self::Share>,
        mut inputs: impl Iterator<Item=Self::Share>,
        preprocessing_data: &mut Self::SetupStorage
    ) -> Self::Msg {
        let delta: Vec<R> = interactive_gates
            .zip(gate_outputs)
            .map(|(gate, output)| {
                assert!(matches!(gate, ArithmeticGate::Mul));
                let inputs = inputs.by_ref().take(gate.input_size());
                gate.compute_delta_share(party_id, inputs, preprocessing_data, output)
            }).collect();
        Msg::Delta { delta }
    }

    fn evaluate_interactive(
        &self,
        _party_id: usize,
        _interactive_gates: impl Iterator<Item=Self::Gate>,
        gate_outputs: impl Iterator<Item=Self::Share>,
        Msg::Delta { delta }: Self::Msg,
        Msg::Delta { delta: other_delta }: Self::Msg,
        _preprocessing_data: &mut Self::SetupStorage
    ) -> Self::ShareStorage {
        gate_outputs
            .zip(delta).zip(other_delta)
            .map(|((mut out_share, my_delta), other_delta)| {
                out_share.public = my_delta.wrapping_add(&other_delta);
                out_share
            }).collect()
    }

    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self, 
        _party_id: usize,
        circuit: &ExecutableCircuit<R, ArithmeticGate<R>, Idx>
    ) -> GateOutputs<Self::ShareStorage> {
        let storage: Vec<_> = circuit
            .gate_counts()
            .map(|(gate_count, simd_size)| {
                assert_eq!(None, simd_size, "SIMD not supported for arithmetic ABY2 protocol");
                Input::Scalar(ShareVec::<R>::repeat(Default::default(), gate_count))
            }).collect();
        let mut storage = GateOutputs::new(storage);
        
        for (gate, sc_gate_id, parents) in circuit.iter_with_parents() {
            let gate_input_iter = parents.map(|parent| storage.get(parent));
            let rng = match self
                .delta_sharing_state
                .input_position_share_type_map
                .get(&sc_gate_id.gate_id.as_usize()) {
                None => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Local) => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Remote) => &mut self.delta_sharing_state.remote_joint_rng,
            };
            let output = gate.setup_output_share(gate_input_iter, rng);
            storage.set(sc_gate_id, output);
        }
        println!("{_party_id}: {storage:?}");
        
        storage
    }
}

impl<R: Ring> Add<Self> for Share<R> {
    type Output = Share<R>;

    fn add(self, rhs: Self) -> Self::Output {
        Share {
            public: self.public + rhs.public,
            private: self.private + rhs.private,
        }
    }
}

impl<R: Ring> WrappingAdd for Share<R> {
    fn wrapping_add(&self, v: &Self) -> Self {
        Share {
            public: self.public.wrapping_add(&v.public),
            private: self.private.wrapping_add(&v.private),
        }
    }
}

impl<R: Ring> Sub<Self> for Share<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            public: self.public.sub(rhs.public),
            private: self.private.sub(rhs.private),
        }
    }
}

impl<R: Ring> WrappingSub for Share<R> {
    fn wrapping_sub(&self, v: &Self) -> Self {
        Share {
            public: self.public.wrapping_sub(&v.public),
            private: self.private.wrapping_sub(&v.private),
        }
    }
}


impl<R: Ring> protocols::Share for Share<R> {
    type Plain = R;
    type SimdShare = ShareVec<R>;
}

impl<R> Gate<R> for ArithmeticGate<R>
where
    R: Ring,
    distributions::Standard: Distribution<R>,
{
    type DimTy = ScalarDim;

    fn is_interactive(&self) -> bool {matches!(self, ArithmeticGate::Mul)}

    fn input_size(&self) -> usize {
        match self {
            ArithmeticGate::Base(base_gate) => base_gate.input_size(),
            _ => 2,
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<R, Self::DimTy>> {
        match self {
            ArithmeticGate::Base(base_gate) => Some(base_gate),
            _ => None,
        }
    }

    fn wrap_base_gate(base_gate: BaseGate<R, Self::DimTy>) -> Self { Self::Base(base_gate) }
}

pub struct ShareIter<R> {
    public: <Vec<R> as IntoIterator>::IntoIter,
    private: <Vec<R> as IntoIterator>::IntoIter,
}

impl<R> FromIterator<Share<R>> for ShareVec<R> {
    fn from_iter<T: IntoIterator<Item=Share<R>>>(iter: T) -> Self {
        let (public, private) = iter
            .into_iter()
            .map(|share| (share.public, share.private))
            .unzip();
        Self { public, private }
    }
}

impl<R> IntoIterator for ShareVec<R> {
    type Item = Share<R>;
    type IntoIter = ShareIter<R>;

    fn into_iter(self) -> Self::IntoIter {
        ShareIter { public: self.public.into_iter(), private: self.private.into_iter() }
    }
}

impl<R> Iterator for ShareIter<R> {
    type Item = Share<R>;

    fn next(&mut self) -> Option<Self::Item> {
        let public = self.public.next()?;
        let private = self.private.next()?;
        Some(Share { public, private } )
    }

    fn size_hint(&self) -> (usize, Option<usize>) { self.public.size_hint() }
}

impl<R: Ring> ShareStorage<Share<R>> for ShareVec<R> {
    fn len(&self) -> usize {
        debug_assert_eq!(self.private.len(), self.public.len());
        self.private.len()
    }

    fn repeat(val: Share<R>, len: usize) -> Self {
        Self {
            private: Vec::repeat(val.private, len),
            public: Vec::repeat(val.public, len),
        }
    }

    fn set(&mut self, idx: usize, val: Share<R>) {
        self.public.set(idx, val.public);
        self.private.set(idx, val.private);
    }

    fn get(&self, idx: usize) -> Share<R> {
        Share {
            public: self.public[idx].clone(),
            private: self.private[idx].clone(),
        }
    }
}

impl<R: Ring> SetupStorage for SetupData<R> {
    fn len(&self) -> usize { self.eval_shares.len() }

    fn split_off_last(&mut self, count: usize) -> Self {
        Self {
            eval_shares: self.eval_shares.split_off(self.len() - count),
        }
    }

    fn reserve(&mut self, additional: usize) {
        self.eval_shares.reserve(additional);
    }

    fn append(&mut self, mut other: Self) { self.eval_shares.append(&mut other.eval_shares); }
}

impl<Mtp, R: Ring> AbySetupProvider<Mtp, R> {
    pub fn new(
        party_id: usize,
        mt_provider: Mtp,
        sender: seec_channel::Sender<AbySetupMsg<R>>,
        receiver: seec_channel::Receiver<AbySetupMsg<R>>,
    ) -> Self {
        Self { party_id, mt_provider, sender, receiver, setup_data: None }
    }
}

impl<R> ArithmeticGate<R>
where
    R: Ring,
    distributions::Standard: Distribution<R>,
{
    fn setup_data_circ<'a, Idx: GateIdx>(
        &self,
        input_shares: impl Iterator<Item = &'a Secret<ArithmeticGmw<R>, Idx>>,
        setup_sub_circ_cache: &mut AHashMap<Vec<Secret<ArithmeticGmw<R>, Idx>>, Secret<ArithmeticGmw<R>, Idx>>,
    ) -> Vec<Secret<ArithmeticGmw<R>, Idx>> {
        let &ArithmeticGate::Mul = self else {
            assert!(self.is_non_interactive(), "Unhandled interactive gate");
            panic!("Called setup_data_circ on non_interactive gate")
        };
        let inputs = 2;

        let inputs_pset = input_shares
            .take(inputs)
            .cloned()
            .powerset()
            .skip(inputs + 1);

        inputs_pset
            .map(|set| match setup_sub_circ_cache.get(&set) {
                None => match &set[..] {
                    [] => unreachable!("Empty set is fitered"),
                    [a, b] => {
                        let sh = a.clone() * b;
                        setup_sub_circ_cache.insert(set, sh.clone());
                        sh
                    }
                    [processed_subset @ .., last] => {
                        assert!(processed_subset.len() >= 2, "Smaller sets are filtered");
                        let subset_out = setup_sub_circ_cache
                            .get(processed_subset)
                            .expect("Subset not present in cache");
                        let sh = last.clone() * subset_out;
                        setup_sub_circ_cache.insert(set, sh.clone());
                        sh
                    }
                },
                Some(processed_set) => processed_set.clone(),
            }).collect()
    }

    fn compute_delta_share(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = Share<R>>,
        preprocessing_delta: &mut SetupData<R>,
        output_share: Share<R>
    ) -> R {
        assert!(matches!(party_id, 0 | 1));
        assert!(matches!(self, ArithmeticGate::Mul));
        let mut priv_delta = preprocessing_delta
            .eval_shares
            .pop()
            .expect("Missing eval_share");
        match self {
            ArithmeticGate::Mul => {
                let a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Insufficient input");
                let plain_ab = a.public.wrapping_mul(&b.public);
                let i = if party_id == 1 { R::ONE } else { R::ZERO };
                i.wrapping_mul(&plain_ab)
                    .wrapping_sub(&a.public.wrapping_mul(&b.private))
                    .wrapping_sub(&b.public.wrapping_mul(&a.private))
                    .wrapping_add(&priv_delta.shares.pop().expect("Missing eval share"))
                    .wrapping_add(&output_share.private)
            }
            _ => unreachable!(),
        }
    }

    fn setup_output_share(
        &self,
        mut inputs: impl Iterator<Item = Share<R>>,
        mut rng: impl Rng,
    ) -> Share<R> {
        match self {
            ArithmeticGate::Base(base_gate) => match base_gate {
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
                | BaseGate::ConnectToMain(_)
                | BaseGate::Debug
                | BaseGate::Identity => inputs.next().expect("Empty input"),
                BaseGate::Constant(_) => Share::default(),
                BaseGate::ConnectToMainFromSimd(_) => {
                    unimplemented!("SIMD currently not supported for ABY2")
                }
            }
            ArithmeticGate::Mul => { 
                Share {
                    private: rng.gen(),
                    public: Default::default(),
                }
            }
            ArithmeticGate::Add => { 
                let mut a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");

                a.private = a.private.wrapping_add(&b.private);
                a
            }
            ArithmeticGate::Sub => {
                let mut a = inputs.next().expect("Empty input");
                let b = inputs.next().expect("Empty input");

                a.private = a.private.wrapping_sub(&b.private);
                a
            }
        }
    }

}

#[async_trait]
impl<MtpErr, Mtp, Idx, R>
FunctionDependentSetup<ArithmeticAby2<R>, Idx> for AbySetupProvider<Mtp, R>
where
    MtpErr: Error + Send + Sync + Debug + 'static,
    Mtp: MTProvider<Output = MulTriples<R>, Error = MtpErr> + Send,
    Idx: GateIdx,
    R: Ring + protocols::Share<SimdShare = <ArithmeticGmw<R> as Protocol>::ShareStorage>,
    distributions::Standard: Distribution<R>,
{
    type Error = Infallible;

    async fn setup(
        &mut self,
        shares: &GateOutputs<ShareVec<R>>,
        circuit: &ExecutableCircuit<R, ArithmeticGate<R>, Idx>
    ) -> Result<(), Self::Error> {
        let circ_builder: CircuitBuilder<R, arithmetic_gmw::ArithmeticGate<R>, Idx> =
            CircuitBuilder::new();
        let old = circ_builder.install();
        let total_inputs: usize = circuit
            .interactive_iter()
            .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
            .sum();

        let mut circ_inputs: Vec<R> = Vec::with_capacity(total_inputs);
        // Block is needed as otherwise !Send types are held over .await
        let setup_outputs: Vec<Vec<_>> = {
            let mut input_sw_map: AHashMap<_, Secret<ArithmeticGmw<R>, Idx>> =
                AHashMap::with_capacity(total_inputs);
            let mut setup_outputs = Vec::with_capacity(circuit.interactive_count());
            let mut setup_sub_circ_cache = AHashMap::with_capacity(total_inputs);
            for (gate, _gate_id, parents) in circuit.interactive_with_parents_iter() {
                let mut gate_input_shares = vec![];
                parents.for_each(|parent| match input_sw_map.entry(parent) {
                    Entry::Vacant(vacant) => {
                        let sh = Secret::<ArithmeticGmw<R>, Idx>::input(0);
                        gate_input_shares.push(sh.clone());
                        circ_inputs.push(shares.get(parent).get_private());
                        vacant.insert(sh);
                    }
                    Entry::Occupied(occupied) => {
                        gate_input_shares.push(occupied.get().clone());
                    }
                });

                // TODO does this impact correctness??
                gate_input_shares.sort();

                let t = gate.setup_data_circ(gate_input_shares.iter(), &mut setup_sub_circ_cache);
                setup_outputs.push(t);
            }

            setup_outputs
                .into_iter()
                .map(|v: Vec<Secret<ArithmeticGmw<R>, Idx>>| v.into_iter().map(|opt_sh| opt_sh.output()).collect())
                .collect()
        };

        let setup_data_circ: ExecutableCircuit<R, arithmetic_gmw::ArithmeticGate<R>, Idx> =
            ExecutableCircuit::DynLayers(CircuitBuilder::global_into_circuit());
        old.install();
        let mut executor: Executor<ArithmeticGmw<R>, Idx> =
            Executor::new(&setup_data_circ, self.party_id, &mut self.mt_provider)
                .await
                .expect("Executor::new in AbySetupProvider");
        executor
            .execute(
                Input::Scalar(circ_inputs),
                &mut self.sender,
                &mut self.receiver,
            ).await
            .unwrap();
        let Input::Scalar(executor_gate_outputs) = executor.gate_outputs().get_sc(0) else {
            panic!("SIMD not supported for arithmetic ABY2");
        };

        let eval_shares = circuit
            .interactive_iter()
            .zip(setup_outputs)
            .map(|((gate, _gate_id), setup_out)| match gate {
                ArithmeticGate::Mul => {
                    let shares = setup_out
                        .into_iter()
                        .map(|out_id| ShareStorage::get(executor_gate_outputs, out_id.as_usize()))
                        .collect();
                    EvalShares { shares }
                }
                _ => unreachable!(),
            })
            .collect();
        self.setup_data = Some(SetupData::from_raw(eval_shares));
        Ok(())
    }

    async fn request_setup_output(
        &mut self,
        count: usize
    ) -> Result<SetupData<R>, Self::Error> {
        Ok(self.setup_data
            .as_mut()
            .expect("setup must be called before request_setup_output")
            .split_off_last(count)
        )
    }
}

impl<R> DeltaSharing<R>
where
    R: Ring,
    distributions::Standard: Distribution<R>,
{  // todo combine with aby2::DeltaSharing
    pub fn new(
        party_id: usize,
        priv_seed: [u8; 32],
        joint_seed: [u8; 32],
        input_position_share_type_map: HashMap<usize, ShareType>,
    ) -> Self {
        assert!(matches!(party_id, 0 | 1), "party_id must be 0 or 1");
        let party_id = party_id as u64;
        let mut local_joint_rng = ChaChaRng::from_seed(joint_seed);
        local_joint_rng.set_stream(party_id);
        let mut remote_joint_rng = local_joint_rng.clone();
        remote_joint_rng.set_stream(party_id ^ 1); // equal to local_joint_rng of the other party
        Self {
            private_rng: ChaChaRng::from_seed(priv_seed),
            local_joint_rng,
            remote_joint_rng,
            input_position_share_type_map,
            phantom: Default::default(),
        }
    }

    /// # Warning - Insecure
    /// Insecurely initialize DeltaSharing RNGs with default value. No input_position_share_type_map
    /// is needed when all the RNGs are the same.
    pub fn insecure_default() -> Self {
        Self {
            private_rng: ChaChaRng::seed_from_u64(0),
            local_joint_rng: ChaChaRng::seed_from_u64(0),
            remote_joint_rng: ChaChaRng::seed_from_u64(0),
            input_position_share_type_map: HashMap::new(),
            phantom: Default::default(),
        }
    }

    pub fn share(&mut self, input: Vec<R>) -> (ShareVec<R>, Vec<R>)
    {
        input
            .into_iter()
            .map(|num| {
                let my_delta = self.private_rng.gen();
                let other_delta = self.local_joint_rng.gen();
                let plain_delta = num.wrapping_add(&my_delta).wrapping_add(&other_delta);
                let my_share = Share { private: my_delta, public: plain_delta.clone() };
                (my_share, plain_delta)
            })
            .unzip()
    }

    pub fn plain_delta_to_share(&mut self, plaint_deltas: Vec<R>) -> ShareVec<R> {
        plaint_deltas
            .into_iter()
            .map(|plain_delta| Share { private: self.remote_joint_rng.gen(), public: plain_delta })
            .collect()
    }

    pub fn reconstruct(a: ShareVec<R>, b: ShareVec<R>) -> Vec<R> {
        a.into_iter()
            .zip(b)
            .map(|(sh1, sh2)|{
                assert_eq!(sh1.get_public(), sh2.get_public(),
                           "Public shares of outputs can't differ");
                sh1.get_public().wrapping_sub(&sh1.get_private()).wrapping_sub(&sh2.get_private())
            }).collect()
    }

}

impl<R> SetupData<R> {
    /// Evaluation shares for interactive gates in **topological** order
    pub fn from_raw(mut eval_shares: Vec<EvalShares<R>>) -> Self {
        eval_shares.reverse();
        Self { eval_shares }
    }

    pub fn len(&self) -> usize { self.eval_shares.len() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<R> Extend<Share<R>> for ShareVec<R> {
    fn extend<T: IntoIterator<Item=Share<R>>>(&mut self, iter: T) {
        for share in iter {
            self.private.push(share.private);
            self.public.push(share.public);
        }
    }
}

impl<R> AstraSetupHelper<R>
where
    R: Ring,
    distributions::Standard: Distribution<R>,
{
    pub fn new(
        sender: MultiSender<AstraSetupMsg<R>>,
        priv_seed_p0: [u8; 32],
        priv_seed_p1: [u8; 32],
        joint_seed: [u8; 32],
    ) -> Self {
        Self {
            sender,
            priv_seed_p0,
            priv_seed_p1,
            joint_seed,
        }
    }

    pub async fn setup<Idx: GateIdx>(
        self,
        circuit: &ExecutableCircuit<R, ArithmeticGate<R>, Idx>,
        input_map: HashMap<usize, InputBy>
    ) {
        let p0_gate_outputs = self.setup_gate_outputs(
            0, circuit, self.priv_seed_p0, self.joint_seed, &input_map);
        let p1_gate_outputs = self.setup_gate_outputs(
            1, circuit, self.priv_seed_p1, self.joint_seed, &input_map);

        let mut rng_p0 = ChaChaRng::from_seed(self.priv_seed_p0);
        rng_p0.set_stream(1);

        let rec_gate_outputs: Vec<Vec<R>> = p0_gate_outputs
            .into_iter()
            .zip(p1_gate_outputs.into_iter())
            .map (|(p0_out, p1_out)| {
                let p0_storage = p0_out.into_scalar().expect("SIMD unsupported");
                let p1_storage = p1_out.into_scalar().expect("SIMD unsupported");

                p0_storage.private.iter()
                    .zip(&p1_storage.private)
                    .map(|(p0, p1)| p0.wrapping_add(p1))
                    .collect()
            }).collect();

        let mut msg = Vec::with_capacity(circuit.interactive_count());

        for (gate, _gate_id, parents) in circuit.interactive_with_parents_iter() {
            match gate {
                ArithmeticGate::Mul => {
                    let inputs: [R; 2] = take_arr(&mut parents.take(2).map(|scg| {
                        rec_gate_outputs[scg.circuit_id as usize][scg.gate_id.as_usize()].clone()
                    }));
                    let lambda_xy = inputs[0].wrapping_mul(&inputs[1]);
                    let lambda_xy_0: R = rng_p0.gen();
                    let lambda_xy_1 = lambda_xy.wrapping_sub(&lambda_xy_0); // todo add?
                    msg.push(lambda_xy_1);
                }
                ni => unreachable!("non interactive gate {ni:?}"),
            }
        }
        self.sender
            .send_to([1], AstraSetupMsg(msg))
            .await
            .expect("failed to send setup message")
    }

    fn setup_gate_outputs<Idx: GateIdx>(
        &self,
        party_id: usize,
        circuit: &ExecutableCircuit<R, ArithmeticGate<R>, Idx>,
        local_seed: [u8; 32],
        joint_seed: [u8; 32],
        input_map: &HashMap<usize, InputBy>,
    ) -> GateOutputs<ShareVec<R>> {
        let input_position_share_type_map = input_map
            .iter()
            .map(|(&pos, by)| {
                let st = match (party_id, by) {
                    (0, InputBy::P0) | (1, InputBy::P1) => ShareType::Local,
                    (0, InputBy::P1) | (1, InputBy::P0) => ShareType::Remote,
                    (id, _) => panic!("Unsupported party id {id}"),
                };
                (pos, st)
            }).collect();

        let mut p = ArithmeticAby2 {
            delta_sharing_state: DeltaSharing::new(
                party_id, local_seed, joint_seed, input_position_share_type_map
            )
        };
        p.setup_gate_outputs(party_id, circuit)
    }
}

impl<R: Ring> AstraSetupProvider<R> {
    pub fn new(party_id: usize, receiver: MultiReceiver<AstraSetupMsg<R>>, seed: [u8; 32]) -> Self {
        let mut rng = ChaChaRng::from_seed(seed);
        rng.set_stream(1);
        Self {
            party_id,
            receiver,
            rng,
            setup_data: None,
        }
    }
}

#[async_trait]
impl<Idx, R> FunctionDependentSetup<ArithmeticAby2<R>, Idx> for AstraSetupProvider<R>
where
    Idx: GateIdx,
    R: Ring,
    distributions::Standard: Distribution<R>,
{
    type Error = Infallible;

    async fn setup(
        &mut self,
        _shares: &GateOutputs<ShareVec<R>>,
        circuit: &ExecutableCircuit<R, ArithmeticGate<R>, Idx>
    ) -> Result<(), Self::Error> {
        if self.party_id == 0 {
            let lambda_values: Vec<_> = (0..circuit.interactive_count())
                .map(|_| EvalShares {
                    shares: Vec::repeat(self.rng.gen(), 1),
                })
                .collect();
            self.setup_data = Some(SetupData::from_raw(lambda_values));
        } else if self.party_id == 1 {
            let msg = self
                .receiver
                .recv_from_single(2)
                .await
                .expect("Recv message from helper");
            let setup_data = msg.0
                .into_iter()
                .map(|eval_share| EvalShares {
                    shares: Vec::repeat(eval_share, 1),
                }).collect();
            self.setup_data = Some(SetupData::from_raw(setup_data));
        } else {
            panic!("Illegal party id {}", self.party_id);
        }
        Ok(())
    }

    async fn request_setup_output(&mut self, count: usize) -> Result<SetupData<R>, Self::Error> {
        Ok(self
            .setup_data
            .as_mut()
            .expect("setup must be called before request_setup_output")
            .split_off_last(count))
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{BaseCircuit, ExecutableCircuit};
    use crate::executor::Executor;
    use crate::gate::base::BaseGate;
    use crate::mul_triple::arithmetic::InsecureMTProvider;
    use crate::protocols::ScalarDim;
    use super::ArithmeticAby2;
    use super::*;
    use super::ArithmeticGate as AG;

    #[tokio::test]
    async fn arithaby() {
        type R = u32;

        let mut c = BaseCircuit::<R, AG<R>>::new();
        let i0 = c.add_gate(AG::Base(BaseGate::Input(ScalarDim)));
        let i1 = c.add_gate(AG::Base(BaseGate::Input(ScalarDim)));
        let i3 = c.add_gate(AG::Base(BaseGate::Input(ScalarDim)));

        let a = c.add_wired_gate(AG::Mul, &[i0, i1]);
        let b = c.add_wired_gate(AG::Mul, &[a, i3]);
        let add = c.add_wired_gate(AG::Add, &[a, b]);
        let sub = c.add_wired_gate(AG::Sub, &[i3, b]);

        let _out = c.add_wired_gate(AG::Base(BaseGate::Output(ScalarDim)), &[a]);
        let _out = c.add_wired_gate(AG::Base(BaseGate::Output(ScalarDim)), &[b]);
        let _out = c.add_wired_gate(AG::Base(BaseGate::Output(ScalarDim)), &[add]);
        let _out = c.add_wired_gate(AG::Base(BaseGate::Output(ScalarDim)), &[sub]);

        let c = ExecutableCircuit::DynLayers(c.into());

        let (ch0, ch1) = seec_channel::in_memory::new_pair(16);
        let setup0: AbySetupProvider<InsecureMTProvider<R>, R> = AbySetupProvider::new(0, InsecureMTProvider::default(), ch0.0, ch0.1);
        let setup1: AbySetupProvider<InsecureMTProvider<R>, R> = AbySetupProvider::new(1, InsecureMTProvider::default(), ch1.0, ch1.1);
        let p_state = ArithmeticAby2::new(DeltaSharing::insecure_default());
        let (mut ex1, mut ex2) = tokio::try_join!(
            Executor::new_with_state(p_state.clone(), &c, 0, setup0),
            Executor::new_with_state(p_state, &c, 1, setup1),
        ).unwrap();

        let (inp0, mask) = DeltaSharing::insecure_default().share(vec![5, 4, 18]);
        let inp1 = DeltaSharing::insecure_default().plain_delta_to_share(mask);
        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(2);

        let h1 = ex1.execute(Input::Scalar(inp0), &mut ch1.0, &mut ch1.1);
        let h2 = ex2.execute(Input::Scalar(inp1), &mut ch2.0, &mut ch2.1);
        let (res1, res2) = tokio::try_join!(h1, h2).unwrap();
        let res =
            DeltaSharing::reconstruct(res1.into_scalar().unwrap(), res2.into_scalar().unwrap());

        assert_eq!(vec![20, 360, 380, 4294966954u32], res);
    }
}