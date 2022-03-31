use crate::circuit::{Circuit, CircuitLayerIter, Gate, GateId};
use crate::common::BitVec;
use crate::errors::ExecutorError;
use crate::evaluate::and;
use crate::executor::ExecutorMsg::AndLayer;
use crate::mult_triple::MultTriples;
use crate::transport::Transport;

use petgraph::adj::IndexType;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter;
use tracing::{info, trace};

pub struct Executor<'c, Idx> {
    circuit: &'c Circuit<Idx>,
    gate_outputs: BitVec,
    party_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutorMsg {
    AndLayer { e: BitVec, d: BitVec },
}

impl<'c, Idx: IndexType> Executor<'c, Idx> {
    pub fn new(circuit: &'c Circuit<Idx>, party_id: usize) -> Self {
        let mut gate_outputs = BitVec::new();
        gate_outputs.resize(circuit.gate_count(), false);
        Self {
            circuit,
            gate_outputs,
            party_id,
        }
    }

    #[tracing::instrument(skip_all, fields(party_id = self.party_id), ret)]
    pub async fn execute<
        SinkErr: Debug,
        StreamErr: Debug,
        C: Transport<ExecutorMsg, SinkErr, StreamErr>,
    >(
        &mut self,
        inputs: BitVec,
        mut channel: C,
    ) -> Result<BitVec, ExecutorError> {
        info!(?inputs, "Executing circuit");
        assert_eq!(
            self.circuit.input_count(),
            inputs.len(),
            "Length of inputs must be equal to circuit input size"
        );
        let mut mts = MultTriples::zeros(self.circuit.and_count());
        let mut layer_count = 0;
        for layer in CircuitLayerIter::new(self.circuit) {
            layer_count += 1;
            for (gate, id) in layer.non_interactive {
                let output = match gate {
                    Gate::Input => gate
                        .evaluate_non_interactive(iter::once(inputs[id.as_usize()]), self.party_id),
                    _ => {
                        let inputs = self.gate_inputs(id);
                        gate.evaluate_non_interactive(inputs, self.party_id)
                    }
                };
                trace!(
                    output,
                    gate_id = %id,
                    "Evaluated {:?} gate",
                    gate
                );

                self.gate_outputs.set(id.as_usize(), output);
            }

            let layer_mts = mts.split_off_last(layer.and_gates.len());

            // TODO ugh, the AND handling is ugly and brittle
            let (d, e): (BitVec, BitVec) = layer
                .and_gates
                .iter()
                .zip(layer_mts.iter())
                .map(|(id, mt)| {
                    let mut inputs = self.gate_inputs(*id);
                    let (x, y) = (inputs.next().unwrap(), inputs.next().unwrap());
                    debug_assert!(
                        inputs.next().is_none(),
                        "Currently only support AND gates with 2 inputs"
                    );
                    and::compute_shares(x, y, &mt)
                })
                .unzip();

            // TODO unnecessary clone
            channel
                .send(AndLayer {
                    d: d.clone(),
                    e: e.clone(),
                })
                .await
                .unwrap();
            let response = channel.next().await.unwrap().unwrap();
            let ExecutorMsg::AndLayer {
                d: resp_d,
                e: resp_e,
            } = response;

            let and_outputs = layer
                .and_gates
                .iter()
                .zip(d)
                .zip(e)
                .zip(resp_d)
                .zip(resp_e)
                .zip(layer_mts.iter())
                .map(|(((((gate_id, d), e), d_resp), e_resp), mt)| {
                    let d = [d, d_resp];
                    let e = [e, e_resp];
                    (and::evaluate(d, e, mt, self.party_id), *gate_id)
                });

            for (output, id) in and_outputs {
                self.gate_outputs.set(id.as_usize(), output);
                trace!(output, gate_id = %id, "Evaluated And gate");
            }
        }
        info!(layer_count);
        // TODO this assumes that the Output gates are the ones with the highest ids
        let output_range =
            self.circuit.gate_count() - self.circuit.output_count()..self.circuit.gate_count();
        Ok(BitVec::from(&self.gate_outputs[output_range]))
    }

    fn gate_inputs(&self, id: GateId<Idx>) -> impl Iterator<Item = bool> + '_ {
        self.circuit
            .parent_gates(id)
            .map(move |parent_id| self.gate_outputs[parent_id.as_usize()])
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{Circuit, Gate};
    use crate::common::BitVec;
    use crate::private_test_utils::{
        create_and_tree, execute_circuit, init_tracing, TestTransport,
    };
    use crate::share_wrapper::{inputs, ShareWrapper};
    use anyhow::Result;
    use bitvec::{bitvec, prelude::Lsb0};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[tokio::test]
    async fn execute_simple_circuit() -> Result<()> {
        let _guard = init_tracing();
        use Gate::*;
        let mut circuit = Circuit::<u32>::new();
        let in_1 = circuit.add_gate(Input);
        let in_2 = circuit.add_gate(Input);
        let and_1 = circuit.add_wired_gate(And, &[in_1, in_2]);
        let xor_1 = circuit.add_wired_gate(Xor, &[in_2, and_1]);
        let and_2 = circuit.add_wired_gate(And, &[and_1, xor_1]);
        circuit.add_wired_gate(Output, &[and_2]);

        let inputs = (BitVec::repeat(true, 2), BitVec::repeat(false, 2));
        let out = execute_circuit(&circuit, inputs, TestTransport::InMemory).await?;
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
        let out = execute_circuit(&and_tree, (inputs_0, inputs_1), TestTransport::InMemory).await?;
        assert_eq!(1, out.len());
        assert_eq!(true, out[0]);
        Ok(())
    }

    #[tokio::test]
    async fn eval_2_bit_adder() -> Result<()> {
        let _guard = init_tracing();
        let adder = Rc::new(RefCell::new(Circuit::<u16>::new()));
        let inputs = inputs(adder.clone(), 4);
        let [a0, a1, b0, b1]: [ShareWrapper<_>; 4] = inputs.try_into().unwrap();
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

        let inputs_0 = {
            let mut bits = bitvec![u8, Lsb0; 1, 1, 0, 0];
            bits.resize(adder.borrow().input_count(), false);
            bits
        };
        let inputs_1 = {
            let mut bits = bitvec![u8, Lsb0; 0, 1, 0, 1];
            bits.resize(adder.borrow().input_count(), false);
            bits
        };

        let exp_output: BitVec = {
            let mut bits = bitvec![u8, Lsb0; 1, 1, 0];
            bits.resize(adder.borrow().output_count(), false);
            bits
        };
        let adder = &adder.borrow();
        let out = execute_circuit(adder, (inputs_0, inputs_1), TestTransport::InMemory).await?;
        assert_eq!(exp_output, out);
        Ok(())
    }
}
