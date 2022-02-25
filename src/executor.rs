use crate::circuit;
use crate::circuit::{Circuit, Gate, Output, Xor};
use crate::common::BitVec;
use crate::errors::ExecutorError;
use crate::executor::ExecutorMsg::AndLayer;
use crate::mult_triple::MultTriple;
use crate::transport::Transport;
use futures::{Sink, SinkExt, Stream, StreamExt};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::ops::BitXor;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

pub struct Executor {
    circuit: Circuit,
}
#[derive(Debug, Clone)]
pub enum ExecutorMsg {
    AndLayer(Vec<AndMessage>),
}

#[derive(Debug, Clone)]
pub struct AndMessage {
    data: (bool, bool),
    gate_id: u64,
}

impl Executor {
    pub fn new(circuit: Circuit) -> Self {
        Self { circuit }
    }

    pub async fn execute<Err: Debug, C: Transport<ExecutorMsg, Err>>(
        &mut self,
        inputs: BitVec,
        mut channel: C,
    ) -> Result<BitVec, ExecutorError> {
        assert_eq!(
            self.circuit.input_size(),
            inputs.len(),
            "Length of inputs must be equal to circuit input size"
        );
        let mut mts: Vec<_> = (0..self.circuit.and_size())
            .map(|_| MultTriple::zeroes())
            .collect();
        // TODO maybe smallvec
        let mut and_layer: HashMap<Arc<Gate>, Vec<bool>> = HashMap::new();
        let mut xor_inputs: HashMap<Arc<Gate>, bool> = HashMap::new();
        let mut outputs: Vec<(Output, bool)> = Vec::new();

        let mut gate_stack: VecDeque<_> = self
            .circuit
            .input_gates()
            .iter()
            .map(Clone::clone)
            .map(Arc::new)
            .zip(inputs.iter().by_vals())
            .collect();
        loop {
            if gate_stack.is_empty() {
                break;
            }
            while let Some((gate, input)) = gate_stack.pop_front() {
                let children = match gate.as_ref() {
                    Gate::Input(gate) => gate.get_children(),
                    Gate::Xor(gate) => gate.get_children(),
                    Gate::And(_) => {
                        todo!("And in gate stack")
                    }
                    Gate::Output(gate) => {
                        outputs.push((gate.clone(), input));
                        continue;
                    }
                };
                for child in children {
                    match child.as_ref() {
                        Gate::Xor(xor) => match xor_inputs.entry(child.clone()) {
                            Entry::Occupied(entry) => {
                                let gate_result = circuit::Xor::evaluate(*entry.get(), input);
                                gate_stack.extend(
                                    xor.get_children()
                                        .iter()
                                        .map(|gate| (child.clone(), gate_result)),
                                );
                            }
                            Entry::Vacant(entry) => {
                                entry.insert(input);
                            }
                        },
                        Gate::And(_) => and_layer.entry(child.clone()).or_default().push(input),
                        Gate::Output(_) => {
                            panic!("Input leads to output")
                        }
                        Gate::Input(_) => {
                            panic!("Input leads to input gate")
                        }
                    }
                }
            }

            let and_messages: Vec<_> = and_layer
                .iter()
                .zip(&mts)
                .map(|((gate, inputs), mt)| {
                    if let Gate::And(and) = gate.as_ref() {
                        let de = circuit::And::compute_shares(inputs[0], inputs[1], mt);
                        AndMessage {
                            data: de,
                            gate_id: and.get_id(),
                        }
                    } else {
                        panic!("BUG: Non and gate in AND layer")
                    }
                })
                .collect();
            channel.send(AndLayer(and_messages.clone())).await.unwrap();
            let response = channel.next().await.unwrap();
            let response_and_messages = if let ExecutorMsg::AndLayer(msgs) = response {
                msgs
            } else {
                unreachable!()
            };
            let and_msg_cnt = and_messages.len();
            let and_outputs = and_messages
                .into_iter()
                .zip(response_and_messages)
                .zip(mts.drain(..and_msg_cnt))
                .map(|((msg_1, msg_2), mt)| {
                    let d = [msg_1.data.0, msg_2.data.0];
                    let e = [msg_1.data.1, msg_2.data.1];
                    circuit::And::evaluate(d, e, mt)
                });

            // TODO below
            for ((gate, _), input) in std::mem::take(&mut and_layer).drain().zip(and_outputs) {
                for child in gate.get_children() {
                    match child.as_ref() {
                        Gate::And(_) => and_layer.entry(child.clone()).or_default().push(input),
                        Gate::Xor(_) | Gate::Output(_) => {
                            gate_stack.push_back((child.clone(), input));
                        }
                        Gate::Input(_) => {
                            panic!("Illegal input gate")
                        }
                    }
                }
            }
        }
        outputs.sort_by_key(|(gate, _)| gate.id);
        Ok(outputs.into_iter().map(|(_, val)| val).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{And, Circuit, Gate, GateCommon, Input, Output, Xor};
    use crate::executor::Executor;
    use crate::transport::InMemory;
    use bitvec::bitvec;
    use bitvec::prelude::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn execute_simple_circuit() {
        let out = Arc::new(Gate::Output(Output { id: 6 }));
        let and_2 = Arc::new(Gate::And(And {
            common: GateCommon {
                children: vec![out],
                id: 5,
            },
        }));
        let xor_1 = Arc::new(Gate::Xor(Xor {
            common: GateCommon {
                children: vec![and_2.clone()],
                id: 4,
            },
        }));
        let and_1 = Arc::new(Gate::And(And {
            common: GateCommon {
                children: vec![xor_1.clone(), and_2.clone()],
                id: 3,
            },
        }));
        let in_1 = Input {
            common: GateCommon {
                children: vec![and_1.clone()],
                id: 1,
            },
        };
        let in_2 = Input {
            common: GateCommon {
                children: vec![and_1, xor_1],
                id: 2,
            },
        };
        let circuit = Circuit::new(vec![in_1, in_2]);
        let mut ex1 = Executor::new(circuit.clone());
        let mut ex2 = Executor::new(circuit);

        let (t1, t2) = InMemory::new_pair();
        let h1 = tokio::spawn(async move { ex1.execute(bitvec![u8, Lsb0; 1, 1], t1).await });
        let h2 = tokio::spawn(async move { ex2.execute(bitvec![u8, Lsb0; 0, 0], t2).await });
        let (out1, out2) = futures::join!(h1, h2);
        let (out1, out2) = (out1.unwrap().unwrap(), out2.unwrap().unwrap());
        assert_eq!(false, dbg!(out1[0]) ^ dbg!(out2[0]));
    }
}
