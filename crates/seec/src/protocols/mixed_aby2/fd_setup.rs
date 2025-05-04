use super::MixedAby2;
use super::MixedAbySetupMsg;
use crate::protocols::mixed_aby2::gate::MixedGate;
use crate::protocols::mixed_aby2::setup_data::MixedSetupData;
use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::executor::{Executor, GateOutputs, Input};
use crate::mul_triple::MTProvider;
use crate::protocols::arithmetic_aby2::{ArithmeticAby2, ArithmeticGate, EvalShares};
pub use crate::protocols::mixed::Mixed as MixedPlain;
use crate::protocols::mixed_aby2::gate::ConvGate;
use crate::protocols::mixed_gmw::MixedGmw;
use crate::protocols::{arithmetic_aby2, FunctionDependentSetup};
use crate::protocols::{
    aby2, mixed_gmw, Gate, Protocol, Ring, ShareStorage,
};
use crate::secret::Secret;
use crate::{executor, protocols, CircuitBuilder};
use ahash::AHashMap;
use async_trait::async_trait;
use bitvec::view::BitViewSized;
use either::Either;
use itertools::Itertools;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::collections::hash_map::Entry;
use std::convert::Infallible;
use std::error::Error;
use std::fmt::Debug;
use std::iter;
use crate::protocols::mixed_aby2::share::MixedShare;
use crate::protocols::mixed_aby2::share_storage::MixedShareStorage;

pub struct MixedAbySetupProvider<Mtp, R: Ring>
where
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
    Mtp: MTProvider,
{
    pub(super) party_id: usize,
    pub(super) mt_provider: Mtp,
    pub(super) sender: seec_channel::Sender<MixedAbySetupMsg<R>>,
    pub(super) receiver: seec_channel::Receiver<MixedAbySetupMsg<R>>,
    pub(super) setup_data: Option<
        <Vec<MixedEvalShare<R>> as IntoIterator>::IntoIter,
    >,
}


pub enum MixedEvalShare<R>
where
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
    Arith(EvalShares<R>),
    Bool(aby2::EvalShares),
    Conv(EvalShares<MixedPlain<R>>),
}

#[async_trait]
impl<MtpErr, Mtp, Idx, R> FunctionDependentSetup<MixedAby2<R>, Idx>
    for MixedAbySetupProvider<Mtp, R>
where
    MtpErr: Error + Send + Sync + Debug + 'static,
    Mtp: MTProvider<Output = <MixedGmw<R> as Protocol>::SetupStorage, Error = MtpErr> + Send,
    Idx: GateIdx,
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
    type Error = Infallible;

    async fn setup(
        &mut self,
        shares: &GateOutputs<MixedShareStorage<R>>,
        circuit: &ExecutableCircuit<MixedPlain<R>, MixedGate<R>, Idx>,
    ) -> Result<(), Self::Error> {
        let circ_builder: CircuitBuilder<MixedPlain<R>, mixed_gmw::MixedGate<R>, Idx> =
            CircuitBuilder::new();
        let old = circ_builder.install();

        let total_inputs: usize = circuit
            .interactive_iter()
            .map(|(gate, _)| 2_usize.pow(gate.input_size() as u32))
            .sum();

        let mut circ_inputs: Vec<MixedPlain<R>> = Vec::with_capacity(total_inputs);
        // Block is needed as otherwise !Send types are held over .await
        let setup_outputs: Vec<Vec<_>> = {
            let mut input_sw_map: AHashMap<_, Secret<MixedGmw<R>, Idx>> =
                AHashMap::with_capacity(total_inputs);
            let mut setup_outputs = Vec::with_capacity(circuit.interactive_count());
            let mut setup_sub_circ_cache = AHashMap::with_capacity(total_inputs);
            for (gate, _gate_id, parents) in circuit.interactive_with_parents_iter() {
                let mut gate_input_shares = vec![];
                parents.for_each(|parent| match input_sw_map.entry(parent) {
                    Entry::Vacant(vacant) => {
                        let sh = Secret::<MixedGmw<R>, Idx>::input(0);
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
                .map(|v: Vec<Secret<MixedGmw<R>, Idx>>| {
                    v.into_iter().map(|opt_sh| opt_sh.output()).collect()
                })
                .collect()
        };

        let setup_data_circ: ExecutableCircuit<MixedPlain<R>, mixed_gmw::MixedGate<R>, Idx> =
            ExecutableCircuit::DynLayers(CircuitBuilder::global_into_circuit());
        old.install();

        let mut executor: Executor<MixedGmw<R>, Idx> =
            Executor::new(&setup_data_circ, self.party_id, &mut self.mt_provider)
                .await
                .unwrap();

        executor
            .execute(
                Input::Scalar(mixed_gmw::MixedShareStorage::Mixed(circ_inputs)),
                &mut self.sender,
                &mut self.receiver,
            )
            .await
            .unwrap();
        let Input::Scalar(executor_gate_outputs) = executor.gate_outputs().get_sc(0) else {
            panic!("SIMD not supported for mixed ABY2");
        };

        let eval_shares: Vec<_> = circuit
            .interactive_iter()
            .zip(setup_outputs)
            .map(|((gate, _gate_id), setup_out)| match gate {
                MixedGate::Bool(aby2::BooleanGate::And { .. }) => {
                    let shares = setup_out
                        .into_iter()
                        .map(|out_id| executor_gate_outputs.get(out_id.as_usize()).unwrap_bool())
                        .collect();
                    MixedEvalShare::Bool(aby2::EvalShares { shares })
                }
                MixedGate::Arith(ArithmeticGate::Mul) => {
                    let shares = setup_out
                        .into_iter()
                        .map(|out_id| executor_gate_outputs.get(out_id.as_usize()).unwrap_arith())
                        .collect();
                    MixedEvalShare::Arith(arithmetic_aby2::EvalShares { shares })
                }
                MixedGate::Conv(_) => {
                    let shares: Vec<_> = setup_out
                        .into_iter()
                        .map(|out_id| executor_gate_outputs.get(out_id.as_usize()))
                        .collect();
                    MixedEvalShare::Conv(EvalShares { shares })
                }
                _ => unreachable!(),
            })
            .collect();

        self.setup_data = Some(eval_shares.into_iter());
        Ok(())
    }

    async fn request_setup_output(
        &mut self,
        count: usize,
    ) -> Result<MixedSetupData<R>, Self::Error> {
        let mut arith = vec![];
        let mut bool = vec![];
        let mut conv = vec![];

        self.setup_data
            .as_mut()
            .expect("setup must be called before request_setup_output")
            .take(count)
            .for_each(|x| match x {
                MixedEvalShare::Arith(a) => arith.push(a),
                MixedEvalShare::Bool(b) => bool.push(b),
                MixedEvalShare::Conv(c) => conv.push(c),
            });
        Ok(MixedSetupData {
            bool: aby2::SetupData::from_raw(bool),
            arith: arithmetic_aby2::SetupData::from_raw(arith),
            conv,
        })
    }
}
