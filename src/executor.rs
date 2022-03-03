use crate::circuit::{Circuit, CircuitLayerIter, Gate, GateId};
use crate::common::BitVec;
use crate::errors::ExecutorError;
use crate::evaluate::and;
use crate::executor::ExecutorMsg::AndLayer;
use crate::mult_triple::MultTriple;
use crate::transport::Transport;

use petgraph::adj::IndexType;
use std::fmt::Debug;
use std::iter;

pub struct Executor<Idx> {
    circuit: Circuit<Idx>,
    gate_outputs: BitVec,
    party_id: usize,
}

#[derive(Debug, Clone)]
pub enum ExecutorMsg {
    AndLayer(Vec<AndMessage>),
}

#[derive(Debug, Clone)]
// Todo: optimize this, data can probably be hoisted into BitVec, should reduce size by 7/8
pub struct AndMessage {
    data: (bool, bool),
}

impl<Idx: IndexType> Executor<Idx> {
    pub fn new(circuit: Circuit<Idx>, party_id: usize) -> Self {
        let mut gate_outputs = BitVec::new();
        gate_outputs.resize(circuit.gate_count(), false);
        Self {
            circuit,
            gate_outputs,
            party_id,
        }
    }

    pub async fn execute<Err: Debug, C: Transport<ExecutorMsg, Err>>(
        &mut self,
        inputs: BitVec,
        mut channel: C,
    ) -> Result<BitVec, ExecutorError> {
        assert_eq!(
            self.circuit.input_count(),
            inputs.len(),
            "Length of inputs must be equal to circuit input size"
        );
        let mut mts: Vec<_> = (0..self.circuit.and_count())
            .map(|_| MultTriple::zeroes())
            .collect();

        for layer in CircuitLayerIter::new(&self.circuit) {
            for (gate, id) in layer.non_interactive {
                let output = match gate {
                    Gate::Input => gate
                        .evaluate_non_interactive(iter::once(inputs[id.as_usize()]), self.party_id),
                    _ => {
                        let inputs = self.gate_inputs(id);
                        gate.evaluate_non_interactive(inputs, self.party_id)
                    }
                };

                self.gate_outputs.set(id.as_usize(), output);
            }

            // TODO ugh, the and handling is ugly and brittle
            let (and_messages, mts): (Vec<_>, Vec<_>) = layer
                .and_gates
                .iter()
                .map(|id| {
                    let mut inputs = self.gate_inputs(*id);
                    let (x, y) = (inputs.next().unwrap(), inputs.next().unwrap());
                    debug_assert!(
                        inputs.next().is_none(),
                        "Currently only support AND gates with 2 inputs"
                    );
                    let mt = mts.pop().expect("Out of mts");
                    let msg = AndMessage {
                        data: and::compute_shares(x, y, &mt),
                    };
                    (msg, mt)
                })
                .unzip();

            // TODO unnecessary clone
            channel.send(AndLayer(and_messages.clone())).await.unwrap();
            let response = channel.next().await.unwrap();
            let ExecutorMsg::AndLayer(response_and_messages) = response;

            let and_outputs = layer
                .and_gates
                .iter()
                .zip(and_messages)
                .zip(response_and_messages)
                .zip(mts)
                .map(|(((id, msg_1), msg_2), mt)| {
                    let d = [msg_1.data.0, msg_2.data.0];
                    let e = [msg_1.data.1, msg_2.data.1];
                    (and::evaluate(d, e, mt, self.party_id), *id)
                });

            for (output, id) in and_outputs {
                self.gate_outputs.set(id.as_usize(), output);
            }
        }
        let output_range =
            self.circuit.gate_count() - self.circuit.output_count..self.circuit.gate_count();
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
    use crate::executor::Executor;
    use crate::test_utils::create_and_tree;
    use crate::transport::InMemory;
    use bitvec::bitvec;
    use bitvec::prelude::*;

    #[tokio::test]
    async fn execute_simple_circuit() {
        use Gate::*;
        let mut circuit = Circuit::<u32>::new();
        let in_1 = circuit.add_gate(Input);
        let in_2 = circuit.add_gate(Input);
        let and_1 = circuit.add_gate(And);
        let xor_1 = circuit.add_gate(Xor);
        let and_2 = circuit.add_gate(And);
        let out_1 = circuit.add_gate(Output);
        circuit.add_wire(in_1, and_1);
        circuit.add_wire(in_2, and_1);
        circuit.add_wire(in_2, xor_1);
        circuit.add_wire(and_1, xor_1);
        circuit.add_wire(and_1, and_2);
        circuit.add_wire(xor_1, and_2);
        circuit.add_wire(and_2, out_1);
        let mut ex1 = Executor::new(circuit.clone(), 0);
        let mut ex2 = Executor::new(circuit, 1);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(bitvec![u8, Lsb0; 1, 1], t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(bitvec![u8, Lsb0; 0, 0], t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(false, out1[0] ^ out2[0]);
    }

    #[tokio::test]
    async fn eval_and_tree() {
        let and_tree = create_and_tree(10);
        let inputs_0 = {
            let mut bits = BitVec::new();
            bits.resize(and_tree.input_count(), true);
            bits
        };
        let inputs_1 = !inputs_0.clone();
        let mut ex1 = Executor::new(and_tree.clone(), 0);
        let mut ex2 = Executor::new(and_tree, 1);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(inputs_0, t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(inputs_1, t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(true, out1[0] ^ out2[0]);
    }
}
