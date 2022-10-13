#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::fmt::Debug;
use std::iter;
use std::time::Instant;

use mpc_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, trace};

use crate::circuit::builder::SubCircuitGate;
use crate::circuit::{Circuit, CircuitLayerIter, Gate, GateIdx};
use crate::common::BitVec;
use crate::errors::{CircuitError, ExecutorError};
use crate::evaluate::and;
use crate::executor::ExecutorMsg::AndLayer;
use crate::mul_triple::{MTProvider, MulTriples};

pub struct Executor<'c, Idx> {
    circuit: &'c Circuit<Idx>,
    gate_outputs: Vec<BitVec>,
    party_id: usize,
    mts: MulTriples,
    // Used as a sanity check in debug builds. Stores for which gates we have set the output,
    // so that we can check if an ouput is set before accessing it.
    #[cfg(debug_assertions)]
    output_set: HashSet<SubCircuitGate<Idx>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutorMsg {
    // TODO ser/de the BitVecs or Vecs? Or maybe a single Vec? e and d have the same length
    AndLayer { e: Vec<u8>, d: Vec<u8> },
}

impl<'c, Idx: GateIdx> Executor<'c, Idx> {
    pub async fn new<P: MTProvider>(
        circuit: &'c Circuit<Idx>,
        party_id: usize,
        mut mt_provider: P,
    ) -> Result<Executor<'c, Idx>, CircuitError>
    where
        P::Error: Debug,
    {
        let gate_outputs = vec![BitVec::new(); circuit.circuits.len()];

        let mts = mt_provider.request_mts(circuit.and_count()).await.unwrap();
        Ok(Self {
            circuit,
            gate_outputs,
            party_id,
            mts,
            #[cfg(debug_assertions)]
            output_set: HashSet::new(),
        })
    }

    #[tracing::instrument(skip_all, fields(party_id = self.party_id), ret)]
    pub async fn execute(
        &mut self,
        inputs: BitVec,
        sender: &mut Sender<ExecutorMsg>,
        receiver: &mut Receiver<ExecutorMsg>,
    ) -> Result<BitVec, ExecutorError> {
        info!("Executing circuit");
        let now = Instant::now();
        assert_eq!(
            self.circuit.input_count(),
            inputs.len(),
            "Length of inputs must be equal to circuit input size"
        );
        // TODO only count layers if there are and gates
        let mut layer_count = 0;
        let party_id = self.party_id;
        let mut and_cnt = 0;
        // TODO provide the option to calculate next layer  during and communication
        //  take care to not block tokio threads -> use tokio rayon
        for layer in CircuitLayerIter::new(self.circuit) {
            for (gate, sc_gate_id) in layer.non_interactive_iter() {
                let output = match gate {
                    Gate::Input => {
                        assert_eq!(
                            sc_gate_id.circuit_id, 0,
                            "Input gate in SubCircuit. Use SubCircuitInput"
                        );
                        gate.evaluate_non_interactive(
                            iter::once(inputs[sc_gate_id.gate_id.as_usize()]),
                            self.party_id,
                        )
                    }
                    _ => {
                        let inputs = self.gate_inputs(sc_gate_id);
                        gate.evaluate_non_interactive(inputs, self.party_id)
                    }
                };
                trace!(
                    output,
                    sc_gate_id = %sc_gate_id,
                    "Evaluated {:?} gate",
                    gate
                );

                self.set_gate_output(sc_gate_id, output);
            }

            // TODO count() there should be a more efficient option
            let layer_mts = self.mts.split_off_last(layer.and_iter().count());

            // TODO ugh, the AND handling is ugly and brittle
            let (d, e): (BitVec, BitVec) = layer
                .and_iter()
                .zip(layer_mts.iter())
                .map(|(id, mt)| {
                    let mut inputs = self.gate_inputs(id);
                    let (x, y) = (inputs.next().unwrap(), inputs.next().unwrap());
                    debug_assert!(
                        inputs.next().is_none(),
                        "Currently only support AND gates with 2 inputs"
                    );
                    and_cnt += 1;
                    and::compute_shares(x, y, &mt)
                })
                .unzip();

            if d.is_empty() {
                // If the layer does not contain and gates we continue
                continue;
            }
            // Only count layers with and gates
            layer_count += 1;
            // TODO unnecessary clone
            // TODO join send and receive
            sender
                .send(AndLayer {
                    d: d.as_raw_slice().to_owned(),
                    e: e.as_raw_slice().to_owned(),
                })
                .await
                .unwrap();
            debug!(size = d.len(), "Sending And layer");
            let response = receiver.recv().await.unwrap().unwrap();
            let AndLayer {
                d: resp_d,
                e: resp_e,
            } = response;

            let and_outputs = layer
                .and_iter()
                .zip(d)
                .zip(e)
                .zip(BitVec::from_vec(resp_d))
                .zip(BitVec::from_vec(resp_e))
                .zip(layer_mts.iter())
                .map(|(((((gate_id, d), e), d_resp), e_resp), mt)| {
                    let d = [d, d_resp];
                    let e = [e, e_resp];
                    (and::evaluate(d, e, mt, party_id), gate_id)
                });

            for (output, id) in and_outputs {
                self.set_gate_output(id, output);
                trace!(output, gate_id = %id, "Evaluated And gate");
            }
        }
        info!(
            layer_count,
            and_cnt,
            execution_time_s = now.elapsed().as_secs_f32()
        );
        let output_iter = self.circuit.circuits[0].output_gates().iter().map(|id| {
            #[cfg(debug_assertions)]
            assert!(
                self.output_set.contains(&SubCircuitGate::new(0, *id)),
                "output gate with id {id:?} is not set",
            );
            self.gate_outputs[0][id.as_usize()]
        });
        Ok(BitVec::from_iter(output_iter))
    }

    fn gate_inputs(&self, id: SubCircuitGate<Idx>) -> impl Iterator<Item = bool> + '_ {
        self.circuit.parent_gates(id).map(move |parent_id| {
            #[cfg(debug_assertions)]
            assert!(
                self.output_set.contains(&parent_id),
                "parent {} of {} not set",
                parent_id,
                id
            );
            self.gate_outputs[parent_id.circuit_id as usize][parent_id.gate_id.as_usize()]
        })
    }

    fn set_gate_output(&mut self, id: SubCircuitGate<Idx>, output: bool) {
        let sc_outputs = &mut self.gate_outputs[id.circuit_id as usize];
        if sc_outputs.is_empty() {
            *sc_outputs = BitVec::repeat(
                false,
                self.circuit.circuits[id.circuit_id as usize].gate_count(),
            )
        }
        sc_outputs.set(id.gate_id.as_usize(), output);
        #[cfg(debug_assertions)]
        self.output_set.insert(id);
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use bitvec::{bitvec, prelude::Lsb0};

    use crate::circuit::{BaseCircuit, Gate};
    use crate::common::BitVec;
    use crate::private_test_utils::{create_and_tree, execute_circuit, init_tracing, TestChannel};
    use crate::share_wrapper::{inputs, ShareWrapper};
    use crate::CircuitBuilder;

    #[tokio::test]
    async fn execute_simple_circuit() -> Result<()> {
        let _guard = init_tracing();
        use Gate::*;
        let mut circuit = BaseCircuit::<u32>::new();
        let in_1 = circuit.add_gate(Input);
        let in_2 = circuit.add_gate(Input);
        let and_1 = circuit.add_wired_gate(And, &[in_1, in_2]);
        let xor_1 = circuit.add_wired_gate(Xor, &[in_2, and_1]);
        let and_2 = circuit.add_wired_gate(And, &[and_1, xor_1]);
        circuit.add_wired_gate(Output, &[and_2]);

        let inputs = (BitVec::repeat(true, 2), BitVec::repeat(false, 2));
        let out = execute_circuit(&circuit.into(), inputs, TestChannel::InMemory).await?;
        assert_eq!(1, out.len());
        assert_eq!(false, out[0]);
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
            &and_tree.into(),
            (inputs_0, inputs_1),
            TestChannel::InMemory,
        )
        .await?;
        assert_eq!(1, out.len());
        assert_eq!(true, out[0]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_2_bit_adder() -> Result<()> {
        let _guard = init_tracing();
        let inputs = inputs(4);
        let [a0, a1, b0, b1]: [ShareWrapper; 4] = inputs.try_into().unwrap();
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

        let inputs_0 = bitvec![u8, Lsb0; 1, 1, 0, 0];
        let inputs_1 = bitvec![u8, Lsb0; 0, 1, 0, 1];
        let exp_output = bitvec![u8, Lsb0; 1, 1, 0];

        let adder = CircuitBuilder::global_into_circuit();
        let out = execute_circuit(&adder, (inputs_0, inputs_1), TestChannel::InMemory).await?;
        assert_eq!(exp_output, out);
        Ok(())
    }
}
