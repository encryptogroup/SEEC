use serde::{Deserialize, Serialize};
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::fmt::Debug;
use std::time::Instant;
use std::{iter, mem};

use mpc_channel::{Receiver, Sender};
use tracing::{debug, info, trace};

use crate::circuit::base_circuit::BaseGate;
use crate::circuit::builder::SubCircuitGate;
use crate::circuit::{CircuitId, DefaultIdx, ExecutableCircuit, GateIdx};
use crate::errors::{CircuitError, ExecutorError};
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{
    FunctionDependentSetup, Gate, Protocol, Share, ShareOf, ShareStorage, SimdShareOf,
};

pub type BoolGmwExecutor<'c> = Executor<'c, BooleanGmw, DefaultIdx>;

pub struct Executor<'c, P: Protocol, Idx> {
    circuit: &'c ExecutableCircuit<P::Gate, Idx>,
    protocol_state: P,
    gate_outputs: GateOutputs<P::ShareStorage>,
    party_id: usize,
    setup_storage: P::SetupStorage,
}

pub struct GateOutputs<Shares> {
    data: Vec<ScGateOutputs<Shares>>,
    // Used as a sanity check in debug builds. Stores for which gates we have set the output,
    // so that we can check if an output is set before accessing it.
    #[cfg(debug_assertions)]
    output_set: HashSet<SubCircuitGate<usize>>,
}

pub enum ScGateOutputs<Shares> {
    Scalar(Shares),
    Simd(Vec<Shares>),
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ExecutorMsg<Msg, SimdMsg> {
    scalar: Msg,
    simd: Option<SimdMsg>,
}

pub type Message<P> = ExecutorMsg<<P as Protocol>::Msg, <P as Protocol>::SimdMsg>;

impl<'c, P, Idx> Executor<'c, P, Idx>
where
    P: Protocol + Default,
    Idx: GateIdx,
    <P::Gate as Gate>::Share: Share<SimdShare = P::ShareStorage>,
    P::Msg: Default,
    P::SimdMsg: Default,
{
    pub async fn new<
        FDSetup: FunctionDependentSetup<P::ShareStorage, P::Gate, Idx, Output = P::SetupStorage>,
    >(
        circuit: &'c ExecutableCircuit<P::Gate, Idx>,
        party_id: usize,
        setup: FDSetup,
    ) -> Result<Executor<'c, P, Idx>, CircuitError>
    where
        FDSetup::Error: Debug,
    {
        Self::new_with_state(P::default(), circuit, party_id, setup).await
    }
}

// TODO Bug: There is currently a synchronization bug which can result in the last message
//  being dropped before sending. This occurs if the whole program terminates before the last
//  message is sent. One option to fix this would be to add an explicit synchronization msg
//  whose absence only results in a warning but no incorrect behaviour
impl<'c, P, Idx> Executor<'c, P, Idx>
where
    P: Protocol,
    Idx: GateIdx,
    <P::Gate as Gate>::Share: Share<SimdShare = P::ShareStorage>,
    P::Msg: Default,
    P::SimdMsg: Default,
{
    pub async fn new_with_state<
        FDSetup: FunctionDependentSetup<P::ShareStorage, P::Gate, Idx, Output = P::SetupStorage>,
    >(
        mut protocol_state: P,
        circuit: &'c ExecutableCircuit<P::Gate, Idx>,
        party_id: usize,
        mut setup: FDSetup,
    ) -> Result<Executor<'c, P, Idx>, CircuitError>
    where
        FDSetup::Error: Debug,
    {
        let gate_outputs = protocol_state.setup_gate_outputs(party_id, circuit);
        let setup_storage = setup.setup(&gate_outputs, circuit).await.unwrap();
        Ok(Self {
            circuit,
            protocol_state,
            gate_outputs,
            party_id,
            setup_storage,
        })
    }

    #[tracing::instrument(skip_all, fields(party_id = self.party_id), err)]
    pub async fn execute(
        &mut self,
        inputs: P::ShareStorage,
        sender: &mut Sender<Message<P>>,
        receiver: &mut Receiver<Message<P>>,
    ) -> Result<P::ShareStorage, ExecutorError> {
        info!("Executing circuit");
        let now = Instant::now();
        assert_eq!(
            self.circuit.input_count(),
            inputs.len(),
            "Length of inputs must be equal to circuit input size"
        );
        let mut setup_storage = mem::take(&mut self.setup_storage);
        let mut layer_count = 0;
        let mut interactive_count = 0;
        // TODO provide the option to calculate next layer  during and communication
        //  take care to not block tokio threads -> use tokio rayon
        for layer in self.circuit.layer_iter() {
            for ((gate, sc_gate_id), mut parents) in
                layer.non_interactive_with_parents_iter(self.circuit)
            {
                trace!(?gate, ?sc_gate_id, "Evaluating");

                match gate.as_base_gate() {
                    Some(base_gate @ BaseGate::Input(_)) => {
                        assert_eq!(
                            sc_gate_id.circuit_id, 0,
                            "Input gate in SubCircuit. Use SubCircuitInput"
                        );
                        // TODO, ugh log(n) in loop... and i'm not even sure if this is correct
                        let inp_idx = self
                            .circuit
                            .input_gates()
                            .binary_search(&sc_gate_id.gate_id)
                            .expect("Input gate not contained in input_gates");
                        let output = base_gate.evaluate_non_interactive(
                            self.party_id,
                            iter::once(inputs.get(inp_idx)),
                        );
                        trace!(
                            ?output,
                            sc_gate_id = %sc_gate_id,
                            "Evaluated {:?} gate",
                            gate
                        );
                        self.set_gate_output(sc_gate_id, output);
                    }
                    Some(base_gate @ BaseGate::SubCircuitInput(_)) => {
                        let simd_size = self.circuit.simd_size(sc_gate_id.circuit_id);
                        if simd_size.is_none() {
                            let inputs = self.inputs(parents);
                            let output = base_gate.evaluate_non_interactive(self.party_id, inputs);
                            trace!(
                                ?output,
                                sc_gate_id = %sc_gate_id,
                                "Evaluated {:?} gate",
                                gate
                            );
                            self.set_gate_output(sc_gate_id, output);
                        } else {
                            let inputs = self.inputs(parents);
                            let simd_output = base_gate.evaluate_sc_input_simd(inputs);
                            trace!(
                                ?simd_output,
                                sc_gate_id = %sc_gate_id,
                                "Evaluated SIMD {:?} gate",
                                gate
                            );
                            self.gate_outputs.set_simd(sc_gate_id, simd_output);
                            continue;
                        }
                    }
                    Some(base_gate @ BaseGate::ConnectToMainFromSimd(_)) => {
                        let input = self
                            .gate_outputs
                            .get_simd(parents.next().expect("Missing input"));
                        let output = base_gate.evaluate_connect_to_main_simd(input);
                        trace!(
                            ?output,
                            sc_gate_id = %sc_gate_id,
                            "Evaluated {:?} gate",
                            gate
                        );
                        self.gate_outputs.set(sc_gate_id, output);
                        continue;
                    }
                    _other => {
                        let simd_size = self.circuit.simd_size(sc_gate_id.circuit_id);
                        if simd_size.is_none() {
                            let inputs = self.inputs(parents);
                            let output = gate.evaluate_non_interactive(self.party_id, inputs);
                            trace!(
                                ?output,
                                sc_gate_id = %sc_gate_id,
                                "Evaluated {:?} gate",
                                gate
                            );
                            self.set_gate_output(sc_gate_id, output);
                        } else {
                            let inputs = self.simd_inputs(parents);
                            let output = gate.evaluate_non_interactive_simd(self.party_id, inputs);
                            trace!(
                                ?output,
                                sc_gate_id = %sc_gate_id,
                                "Evaluated SIMD {:?} gate",
                                gate
                            );
                            self.gate_outputs.set_simd(sc_gate_id, output);
                        }
                    }
                };
            }

            let layer_int_cnt = layer.interactive_count();
            if layer_int_cnt == 0 {
                trace!("Layer has no interactive gates. Current layer count {layer_count:?}");
                // If the layer does not contain and gates we continue
                continue;
            }
            // Only count layers with and gates
            layer_count += 1;
            interactive_count += layer_int_cnt;

            let (scalar, simd) = layer.split_simd();

            let scalar_gate_iter = layer.interactive_gates().flatten().cloned();
            // interactive_parents_iter is !Send so we introduce a block s.t. it is not hold
            // over .await
            let scalar_msg = {
                let scalar_output_iter = scalar.interactive_indices().flat_map(|(sc, gate_ids)| {
                    let this = &*self;
                    gate_ids.iter().map(move |gate_id| {
                        this.gate_outputs
                            .get_unchecked(SubCircuitGate::new(sc, *gate_id))
                    })
                });
                let input_iter = scalar
                    .interactive_parents_iter(self.circuit)
                    .flat_map(|parents| self.inputs(parents));
                self.protocol_state.compute_msg(
                    self.party_id,
                    scalar_gate_iter.clone(),
                    scalar_output_iter,
                    input_iter,
                    &mut setup_storage,
                )
            };

            let simd_gate_iter = simd.interactive_gates().flatten().cloned();
            // interactive_parents_iter is !Send so we introduce a block s.t. it is not hold
            // over .await
            let simd_msg = P::SIMD_SUPPORT.then(|| {
                let simd_output_iter = simd.interactive_indices().flat_map(|(sc, gate_ids)| {
                    let this = &*self;
                    gate_ids.iter().map(move |gate_id| {
                        this.gate_outputs
                            .get_simd_unchecked(SubCircuitGate::new(sc, *gate_id))
                    })
                });
                let input_iter = simd
                    .interactive_parents_iter(self.circuit)
                    .flat_map(|parents| parents.map(|parent| self.gate_outputs.get_simd(parent)));
                self.protocol_state.compute_msg_simd(
                    self.party_id,
                    simd_gate_iter.clone(),
                    simd_output_iter,
                    input_iter,
                    &mut setup_storage,
                )
            });

            let msg = ExecutorMsg {
                scalar: scalar_msg.clone(),
                simd: simd_msg.clone(),
            };
            sender.send(msg).await.ok().unwrap();
            debug!("Sending interactive gates layer");
            let ExecutorMsg {
                scalar: resp_scalar,
                simd: resp_simd,
            } = receiver.recv().await.ok().unwrap().unwrap();

            // recreate iters afer .await point, for some reason holding them over that
            // results in a weird compile error. The iterators are Send, but if held over .await
            // the future is not... Seems like a compiler bug
            let scalar_output_iter = scalar.interactive_indices().flat_map(|(sc, gate_ids)| {
                let this = &*self;
                gate_ids.iter().map(move |gate_id| {
                    this.gate_outputs
                        .get_unchecked(SubCircuitGate::new(sc, *gate_id))
                })
            });
            let simd_output_iter = simd.interactive_indices().flat_map(|(sc, gate_ids)| {
                let this = &*self;
                gate_ids.iter().map(move |gate_id| {
                    this.gate_outputs
                        .get_simd_unchecked(SubCircuitGate::new(sc, *gate_id))
                })
            });

            let scalar_interactive_outputs = self.protocol_state.evaluate_interactive(
                self.party_id,
                scalar_gate_iter,
                scalar_output_iter,
                scalar_msg,
                resp_scalar,
                &mut setup_storage,
            );
            let simd_interactive_outputs = match (simd_msg, resp_simd) {
                (Some(simd_msg), Some(resp_simd)) => {
                    Some(self.protocol_state.evaluate_interactive_simd(
                        self.party_id,
                        simd_gate_iter,
                        simd_output_iter,
                        simd_msg,
                        resp_simd,
                        &mut setup_storage,
                    ))
                }
                (Some(_), None) | (None, Some(_)) => panic!("Sent and received simd msg differ"),
                (None, None) => None,
            };

            scalar
                .interactive_iter()
                .zip(scalar_interactive_outputs)
                .for_each(|((_, id), out)| {
                    self.gate_outputs.set(id, out.clone());
                    trace!(?out, gate_id = %id, "Evaluated interactive gate");
                });
            if let Some(simd_interactive_outputs) = simd_interactive_outputs {
                simd.interactive_iter()
                    .zip(simd_interactive_outputs)
                    .for_each(|((_, id), out)| {
                        self.gate_outputs.set_simd(id, out.clone());
                        trace!(?out, gate_id = %id, "Evaluated SIMD interactive gate");
                    });
            }
        }
        info!(
            layer_count,
            interactive_count,
            execution_time_s = now.elapsed().as_secs_f32()
        );
        let output_iter = self
            .circuit
            .output_gates()
            .iter()
            .map(|id| self.gate_outputs.get(SubCircuitGate::new(0, *id)));
        Ok(FromIterator::from_iter(output_iter))
    }

    pub fn gate_outputs(&self) -> &GateOutputs<P::ShareStorage> {
        &self.gate_outputs
    }

    pub fn setup_storage(&self) -> &P::SetupStorage {
        &self.setup_storage
    }

    fn inputs<'s, 'p>(
        &'s self,
        parent_ids: impl Iterator<Item = SubCircuitGate<Idx>> + 'p,
    ) -> impl Iterator<Item = ShareOf<P::Gate>> + 'p
    where
        's: 'p,
    {
        parent_ids.map(move |parent_id| self.gate_outputs.get(parent_id))
    }

    fn simd_inputs<'s, 'p>(
        &'s self,
        parent_ids: impl Iterator<Item = SubCircuitGate<Idx>> + 'p,
    ) -> impl Iterator<Item = &SimdShareOf<P::Gate>> + 'p
    where
        's: 'p,
    {
        parent_ids.map(move |parent_id| self.gate_outputs.get_simd(parent_id))
    }

    fn set_gate_output(&mut self, id: SubCircuitGate<Idx>, output: <P::Gate as Gate>::Share) {
        self.gate_outputs.set(id, output);
    }
}

impl<Shares> GateOutputs<Shares> {
    pub fn get_sc(&self, id: CircuitId) -> &ScGateOutputs<Shares> {
        &self.data[id as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = &ScGateOutputs<Shares>> {
        self.data.iter()
    }
}

impl<Shares> ScGateOutputs<Shares> {
    pub fn as_scalar(&self) -> Option<&Shares> {
        match self {
            ScGateOutputs::Scalar(scalar) => Some(scalar),
            ScGateOutputs::Simd(_) => None,
        }
    }
}

impl<Shares: Clone> GateOutputs<Shares> {
    pub fn new(data: Vec<ScGateOutputs<Shares>>) -> Self {
        Self {
            data,
            #[cfg(debug_assertions)]
            output_set: HashSet::new(),
        }
    }

    pub fn get<Share, Idx: GateIdx>(&self, id: SubCircuitGate<Idx>) -> Share
    where
        Shares: ShareStorage<Share>,
    {
        #[cfg(debug_assertions)]
        assert!(
            self.output_set.contains(&id.into_usize()),
            "parent {id} not set",
        );
        self.get_unchecked(id)
    }

    pub fn get_unchecked<Share, Idx: GateIdx>(&self, id: SubCircuitGate<Idx>) -> Share
    where
        Shares: ShareStorage<Share>,
    {
        match &self.data[id.circuit_id as usize] {
            ScGateOutputs::Scalar(data) => data.get(id.gate_id.as_usize()),
            ScGateOutputs::Simd(_) => {
                panic!("Called GateOutputs::get({id:?}) for Simd circ")
            }
        }
    }

    pub fn get_simd<Idx: GateIdx>(&self, id: SubCircuitGate<Idx>) -> &Shares {
        #[cfg(debug_assertions)]
        assert!(
            self.output_set.contains(&id.into_usize()),
            "SIMD parent {id} not set",
        );
        self.get_simd_unchecked(id)
    }

    pub fn get_simd_unchecked<Idx: GateIdx>(&self, id: SubCircuitGate<Idx>) -> &Shares {
        match &self.data[id.circuit_id as usize] {
            ScGateOutputs::Scalar(_) => {
                panic!("Called GateOutputs::get_simd({id:?}) for scalar circ")
            }
            ScGateOutputs::Simd(data) => &data[id.gate_id.as_usize()],
        }
    }

    pub fn set<Share, Idx: GateIdx>(&mut self, id: SubCircuitGate<Idx>, val: Share)
    where
        Shares: ShareStorage<Share>,
    {
        #[cfg(debug_assertions)]
        self.output_set.insert(id.into_usize());
        match &mut self.data[id.circuit_id as usize] {
            ScGateOutputs::Scalar(data) => data.set(id.gate_id.as_usize(), val),
            ScGateOutputs::Simd(_) => {
                panic!("Called GateOutputs::set for Simd circ")
            }
        }
    }

    pub fn set_simd<Idx: GateIdx>(&mut self, id: SubCircuitGate<Idx>, val: Shares) {
        #[cfg(debug_assertions)]
        self.output_set.insert(id.into_usize());
        match &mut self.data[id.circuit_id as usize] {
            ScGateOutputs::Scalar(_) => {
                panic!("Called GateOutputs::set_simd for scalar circ")
            }
            ScGateOutputs::Simd(data) => {
                data[id.gate_id.as_usize()] = val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::base_circuit::BaseGate;
    use anyhow::Result;
    use bitvec::{bitvec, prelude::Lsb0};
    use tracing::debug;

    use crate::circuit::{BaseCircuit, DefaultIdx, ExecutableCircuit};
    use crate::common::BitVec;
    use crate::private_test_utils::{create_and_tree, execute_circuit, init_tracing, TestChannel};
    use crate::protocols::ScalarDim;
    use crate::secret::{inputs, Secret};
    use crate::{BooleanGate, Circuit, CircuitBuilder};

    #[tokio::test]
    async fn execute_simple_circuit() -> Result<()> {
        let _guard = init_tracing();
        use crate::protocols::boolean_gmw::BooleanGate::*;
        let mut circuit: BaseCircuit = BaseCircuit::new();
        let in_1 = circuit.add_gate(Base(BaseGate::Input(ScalarDim)));
        let in_2 = circuit.add_gate(Base(BaseGate::Input(ScalarDim)));
        let and_1 = circuit.add_wired_gate(And, &[in_1, in_2]);
        let xor_1 = circuit.add_wired_gate(Xor, &[in_2, and_1]);
        let and_2 = circuit.add_wired_gate(And, &[and_1, xor_1]);
        circuit.add_wired_gate(Base(BaseGate::Output(ScalarDim)), &[and_2]);

        let inputs = [BitVec::repeat(true, 2), BitVec::repeat(false, 2)];
        let out = execute_circuit(
            &ExecutableCircuit::DynLayers(circuit.into()),
            inputs,
            TestChannel::InMemory,
        )
        .await?;
        assert_eq!(1, out.len());
        assert!(!out[0]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_and_tree() -> Result<()> {
        let _guard = init_tracing();
        let and_tree = create_and_tree(10);
        let inputs_0 = {
            let mut bits = BitVec::new();
            bits.resize(and_tree.input_count(), true);
            bits
        };
        let inputs_1 = !inputs_0.clone();
        let out = execute_circuit(
            &ExecutableCircuit::DynLayers(and_tree.into()),
            [inputs_0, inputs_1],
            TestChannel::InMemory,
        )
        .await?;
        assert_eq!(1, out.len());
        assert!(out[0]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_2_bit_adder() -> Result<()> {
        let _guard = init_tracing();
        debug!("Test start");
        let inputs = inputs(4);
        debug!("Inputs");
        let [a0, a1, b0, b1]: [Secret; 4] = inputs.try_into().unwrap();
        let xor1 = a0.clone() ^ b0.clone();
        let and1 = a0 & b0;
        let xor2 = a1.clone() ^ b1.clone();
        let and2 = a1 & b1;
        let xor3 = xor2.clone() ^ and1.clone();
        let and3 = xor2 & and1;
        let or = and2 | and3;
        for share in [xor1, xor3, or] {
            share.output();
        }

        debug!("End Secret ops");
        let inputs_0 = bitvec![usize, Lsb0; 1, 1, 0, 0];
        let inputs_1 = bitvec![usize, Lsb0; 0, 1, 0, 1];
        let exp_output = bitvec![usize, Lsb0; 1, 1, 0];
        let adder: Circuit<BooleanGate, DefaultIdx> = CircuitBuilder::global_into_circuit();

        debug!("Into circuit");
        let out = execute_circuit(
            &ExecutableCircuit::StaticLayers(adder.precompute_layers()),
            [inputs_0, inputs_1],
            TestChannel::InMemory,
        )
        .await?;
        debug!("Executed");
        assert_eq!(exp_output, out);
        Ok(())
    }
}
