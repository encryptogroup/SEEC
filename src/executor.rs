use crate::circuit::{Circuit, CircuitLayerIter, Gate, GateId};
use crate::common::BitVec;
use crate::errors::ExecutorError;
use crate::evaluate::and;
use crate::executor::ExecutorMsg::AndLayer;
use crate::mult_triple::MultTriple;
use crate::transport::Transport;

use petgraph::adj::IndexType;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter;
use tracing::{info, trace};

pub struct Executor<Idx> {
    circuit: Circuit<Idx>,
    gate_outputs: BitVec,
    party_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutorMsg {
    AndLayer { e: BitVec, d: BitVec },
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
                trace!(
                    output,
                    gate_id = %id,
                    "Evaluated {:?} gate",
                    gate
                );

                self.gate_outputs.set(id.as_usize(), output);
            }

            // TODO ugh, the AND handling is ugly and brittle
            let ((d, e), mts): ((BitVec, BitVec), Vec<_>) = layer
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
                    let msg = and::compute_shares(x, y, &mt);
                    (msg, mt)
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
                .zip(mts)
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
    use crate::executor::Executor;
    use crate::test_utils::{create_and_tree, init_tracing};
    use crate::transport::InMemory;
    use aes::cipher::generic_array::GenericArray;
    use aes::cipher::{BlockEncrypt, KeyInit};
    use aes::Aes128;
    use bitvec::prelude::Msb0;
    use bitvec::{bitvec, prelude::Lsb0};
    use hex_literal::hex;

    #[tokio::test]
    async fn execute_simple_circuit() {
        let _guard = init_tracing();
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
        let _guard = init_tracing();
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

    #[tokio::test]
    async fn eval_2_bit_adder() {
        let _guard = init_tracing();
        let mut adder = Circuit::<u16>::new();
        let in_a_0 = adder.add_gate(Gate::Input);
        let in_a_1 = adder.add_gate(Gate::Input);
        let in_b_0 = adder.add_gate(Gate::Input);
        let in_b_1 = adder.add_gate(Gate::Input);
        let xor1 = adder.add_gate(Gate::Xor);
        adder.add_wire(in_a_0, xor1);
        adder.add_wire(in_b_0, xor1);
        let and1 = adder.add_gate(Gate::And);
        adder.add_wire(in_a_0, and1);
        adder.add_wire(in_b_0, and1);

        let xor2 = adder.add_gate(Gate::Xor);
        adder.add_wire(in_a_1, xor2);
        adder.add_wire(in_b_1, xor2);
        let and2 = adder.add_gate(Gate::And);
        adder.add_wire(in_a_1, and2);
        adder.add_wire(in_b_1, and2);

        let xor3 = adder.add_gate(Gate::Xor);
        adder.add_wire(xor2, xor3);
        adder.add_wire(and1, xor3);
        let and3 = adder.add_gate(Gate::And);
        adder.add_wire(xor2, and3);
        adder.add_wire(and1, and3);

        // OR gate
        let xor4 = adder.add_gate(Gate::Xor);
        let and4 = adder.add_gate(Gate::And);
        let xor5 = adder.add_gate(Gate::Xor);
        adder.add_wire(and3, xor4);
        adder.add_wire(and2, xor4);
        adder.add_wire(and3, and4);
        adder.add_wire(and2, and4);
        adder.add_wire(xor4, xor5);
        adder.add_wire(and4, xor5);

        let out0 = adder.add_gate(Gate::Output);
        let out1 = adder.add_gate(Gate::Output);
        let out2 = adder.add_gate(Gate::Output);

        adder.add_wire(xor1, out0);
        adder.add_wire(xor3, out1);
        adder.add_wire(xor5, out2);

        let inputs_0 = {
            let mut bits = bitvec![u8, Lsb0; 1, 1, 0, 0];
            bits.resize(adder.input_count(), false);
            bits
        };
        let inputs_1 = {
            let mut bits = bitvec![u8, Lsb0; 0, 1, 0, 1];
            bits.resize(adder.input_count(), false);
            bits
        };

        let exp_output: BitVec = {
            let mut bits = bitvec![u8, Lsb0; 1, 1, 0];
            bits.resize(adder.output_count(), false);
            bits
        };
        let mut ex1 = Executor::new(adder.clone(), 0);
        let mut ex2 = Executor::new(adder, 1);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(inputs_0, t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(inputs_1, t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(exp_output, out1 ^ out2);
    }

    #[tokio::test]
    async fn eval_8_bit_adder() {
        let _guard = init_tracing();
        let adder = Circuit::load_bristol("test_resources/bristol-circuits/int_add8_depth.bristol")
            .unwrap();
        let inputs_0 = {
            let mut bits = bitvec![u8, Lsb0;
                1, 1, 1, 1, 0, 0, 0, 0,
                1, 0, 1, 0, 0, 0, 0, 0,
            ];
            bits.resize(adder.input_count(), false);
            bits
        };
        let inputs_1 = {
            let mut bits = bitvec![u8, Lsb0; 0];
            bits.resize(adder.input_count(), false);
            bits
        };

        let exp_output: BitVec = {
            let mut bits = bitvec![u8, Lsb0; 0, 0, 1, 0, 1, 0, 0, 0];
            bits.resize(adder.output_count(), false);
            bits
        };
        let mut ex1 = Executor::new(adder.clone(), 0);
        let mut ex2 = Executor::new(adder, 1);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(inputs_0, t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(inputs_1, t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(exp_output, out1 ^ out2);
    }

    #[tokio::test]
    async fn eval_aes_circuit() {
        let _guard = init_tracing();
        let adder =
            Circuit::load_bristol("test_resources/bristol-circuits/AES-non-expanded.txt").unwrap();
        let inputs_0 = {
            let mut bits = bitvec![u8, Lsb0;0];
            bits.resize(adder.input_count(), false);
            bits
        };
        let inputs_1 = {
            let mut bits = bitvec![u8, Lsb0; 0];
            bits.resize(adder.input_count(), false);
            bits
        };

        // It seems that the output of the circuit and the aes crate use different bit orderings
        // for the output.
        let exp_output: bitvec::vec::BitVec<u8, Msb0> = {
            let key = GenericArray::from([0u8; 16]);
            let mut block = GenericArray::from([0u8; 16]);
            let cipher = Aes128::new(&key);
            cipher.encrypt_block(&mut block);

            bitvec::vec::BitVec::from_slice(block.as_slice())
        };
        let mut ex1 = Executor::new(adder.clone(), 0);
        let mut ex2 = Executor::new(adder, 1);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(inputs_0, t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(inputs_1, t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(exp_output, out1 ^ out2);
    }

    #[tokio::test]
    async fn eval_sha_256_circuit() {
        let _guard = init_tracing();
        let adder = Circuit::load_bristol("test_resources/bristol-circuits/sha-256.txt").unwrap();
        let inputs_0 = {
            let mut bits = bitvec![u8, Lsb0;0];
            bits.resize(adder.input_count(), false);
            bits
        };
        let inputs_1 = {
            let mut bits = bitvec![u8, Lsb0; 0];
            bits.resize(adder.input_count(), false);
            bits
        };

        // The output of the circuit is apparently in Msb order
        let exp_output: bitvec::vec::BitVec<u8, Msb0> = bitvec::vec::BitVec::from_slice(&hex!(
            // From: https://homes.esat.kuleuven.be/~nsmart/MPC/sha-256-test.txt
            // The output of the circuit is not the *normal* sha256 output
            "da5698be17b9b46962335799779fbeca8ce5d491c0d26243bafef9ea1837a9d8"
        ));
        let mut ex1 = Executor::new(adder.clone(), 0);
        let mut ex2 = Executor::new(adder, 1);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(inputs_0, t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(inputs_1, t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(exp_output, out1 ^ out2);
    }
}
