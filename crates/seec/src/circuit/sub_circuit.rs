use crate::circuit::circuit_connections::CrossCircuitConnections;
use crate::circuit::{base_circuit, BaseCircuit, GateIdx};
use crate::protocols::Gate;
use crate::GateId;
use ahash::HashMap;
use parking_lot::{Mutex, MutexGuard};
use std::collections::hash_map;
use std::fmt::{Debug, Formatter};
use std::slice;
use std::sync::{mpsc, Arc};

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum CallId {
    Ret,
    Call(u32),
}
#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct CircId(u32);

#[derive(Debug)]
pub struct Circuit<G, Idx> {
    base_circuits: Vec<BaseCircuit<G, Idx>>,
    main: SubCircuit<Idx>,
}

#[derive(Clone, Debug)]
pub struct SubCircuit<Idx>(Arc<Mutex<SubCircuitInner<Idx>>>);

#[derive(Debug)]
struct SubCircuitInner<Idx> {
    base_circ_id: CircId, // this corresponds to the position in Circuits.base_circuits Vec
    // TODO how to handle parent sub-circuit? If this is stored in sub-circuits, this likely
    //  leads to a cycle... Do we even need a ref to the parent?
    sub_circuits: Vec<SubCircuit<Idx>>, // position in this Vec is CallId::Call(id)
    call_map: CallMap<Idx>,
}

impl<Idx> SubCircuit<Idx> {
    pub fn lock(&self) -> MutexGuard<SubCircuitInner<Idx>> {
        self.0.lock()
    }
}

#[derive(Default, Debug)]
struct CallMap<Idx> {
    map: HashMap<GateId<Idx>, Vec<(CallId, GateId<Idx>)>>,
}

impl<Idx: GateIdx> CallMap<Idx> {
    pub fn outgoing_gates(
        &self,
        g: GateId<Idx>,
    ) -> impl Iterator<Item = &(CallId, GateId<Idx>)> + '_ {
        self.map.get(&g).map(|v| v.iter()).unwrap_or([].iter())
    }
}

#[derive(Debug)]
struct SubCircIter<'base, G, Idx: GateIdx> {
    circ: &'base Circuit<G, Idx>,
    base_iter: base_circuit::BaseLayerIter<'base, G, Idx, ()>,
    sub_circ_iters: HashMap<CallId, ChildOrParent<Self>>,
    sub_circuit: SubCircuit<Idx>,
    // This is passed to child sub circ iters
    own_returned_gates_tx: mpsc::Sender<GateId<Idx>>,
    // This receives messages from child sub circ iters
    own_returned_gates_rx: mpsc::Receiver<GateId<Idx>>,
    // Notify parent sub circ iter of returned gates
    parent_returned_gates_tx: Option<mpsc::Sender<GateId<Idx>>>,
}

#[derive(Debug)]
enum ChildOrParent<T> {
    Child(T),
    IsParent,
}

impl<'base, G: Gate, Idx: GateIdx> ChildOrParent<SubCircIter<'base, G, Idx>> {
    // Adds gate to next layer of child of sends it into returned queue of parent
    fn add_to_next_layer(
        &mut self,
        gate_id: GateId<Idx>,
        parent_returned_gates_tx: Option<&mpsc::Sender<GateId<Idx>>>,
    ) {
        match self {
            ChildOrParent::Child(iter) => iter.base_iter.add_to_next_layer(gate_id.into()),
            ChildOrParent::IsParent => parent_returned_gates_tx
                .expect("Iter is child but has no parent")
                .send(gate_id)
                .expect("returned gates receiver dropped prematurely"),
        }
    }
}

struct SubCircLayer<'sc_iter, 'base, G, Idx: GateIdx> {
    call_id: CallId,
    base_layer: base_circuit::CircuitLayer<G, Idx>,
    sub_circ_iter: &'sc_iter mut SubCircIter<'base, G, Idx>,
}

impl<'base, G: Gate, Idx: GateIdx> SubCircIter<'base, G, Idx> {
    fn next(&mut self) -> Option<SubCircLayer<'_, 'base, G, Idx>> {
        // Update own base layer iter with returned gate ids
        while let Ok(ret_gate_id) = self.own_returned_gates_rx.try_recv() {
            self.base_iter.add_to_next_layer(ret_gate_id.into());
        }
        // TODO what to do when empty?
        let base_layer = self.base_iter.next().unwrap_or_default();
        let sc = self.sub_circuit.lock();
        for gate_id in base_layer.iter_ids() {
            // TODO, what to do about outgoing gates that return to the calling circuit?
            for &(call_id, out_gate) in sc.call_map.outgoing_gates(gate_id) {
                let sc_iter = self.sub_circ_iters.entry(call_id).or_insert_with(|| {
                    let call_id = match call_id {
                        CallId::Ret => return ChildOrParent::IsParent,
                        CallId::Call(id) => id,
                    };
                    let called_sc = sc.sub_circuits[call_id as usize].clone();
                    let called_circ_id = called_sc.lock().base_circ_id;
                    let called_base_circ = &self.circ.base_circuits[called_circ_id.0 as usize];
                    let mut base_iter = base_circuit::BaseLayerIter::new_uninit(called_base_circ);
                    let (tx, rx) = mpsc::channel();
                    println!("Inserting sub circ iter for {:?}", called_circ_id);
                    ChildOrParent::Child(SubCircIter {
                        circ: self.circ,
                        base_iter,
                        sub_circ_iters: Default::default(),
                        sub_circuit: called_sc,
                        own_returned_gates_tx: tx,
                        own_returned_gates_rx: rx,
                        parent_returned_gates_tx: Some(self.own_returned_gates_tx.clone()),
                    })
                });
                // TODO either add to next or current layer depending on from gate
                //  is_non_interactive(from_gate) -> current_layer
                //  is_interactive(from_gate) -> next_layer
                //  does this make sense?
                sc_iter.add_to_next_layer(out_gate, self.parent_returned_gates_tx.as_ref());
            }
        }
        drop(sc);
        // drive own base iter to get gates for this base circuit
        // check own gates in self.sub_circ call_map to see if they are outgoing
        // if yes, add new SubCircIter to self.sub_circ_iters or update iter state
        // returns None when?? If own base_iter and all sub_circ_iters are exhausted?
        Some(SubCircLayer {
            call_id: CallId::Call(0),
            base_layer,
            sub_circ_iter: self,
        })
    }
}

impl<'sc_iter, 'base, G, Idx: GateIdx> SubCircLayer<'sc_iter, 'base, G, Idx> {
    fn sub_layer_iter(&mut self) -> SubLayerIter<'_, 'base, G, Idx> {
        SubLayerIter {
            iter: self.sub_circ_iter.sub_circ_iters.iter_mut(),
        }
    }
}

struct SubLayerIter<'a, 'base, G, Idx: GateIdx> {
    iter: hash_map::IterMut<'a, CallId, ChildOrParent<SubCircIter<'base, G, Idx>>>,
}

impl<'a, 'base, G, Idx> Iterator for SubLayerIter<'a, 'base, G, Idx>
where
    G: Gate,
    Idx: GateIdx,
{
    type Item = SubCircLayer<'a, 'base, G, Idx>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((call_id, sub_iter)) = self.iter.next() {
            let sub_iter = match sub_iter {
                ChildOrParent::Child(child) => child,
                // skip connections to parents
                ChildOrParent::IsParent => continue,
            };
            let mut sub_circ_layer = match sub_iter.next() {
                None => continue,
                Some(layer) => layer,
            };
            sub_circ_layer.call_id = *call_id;
            return Some(sub_circ_layer);
        }
        None
    }
}

impl<'sc_iter, 'base, G: Debug, Idx: GateIdx> Debug for SubCircLayer<'sc_iter, 'base, G, Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubCircLayer")
            .field("call_id", &self.call_id)
            .field("base_layer", &self.base_layer)
            .finish()
    }
}

// enum LayerExecutor {
//     BeforeInteractive {
//         layer: Layer
//         sub_executors: Vec<LayerExecutor>
//     },
//     AfterInteractive {
//         interactive_gates: InteractiveLayer
//         msg_range: Range<usize>
//         sub_executors: Vec<LayerExecutor>
//     }
// }
//
//
// impl LayerExecutor {
//     fn handle_before() {
//         handle_non_interactive(layer)
//         let msg_idx = handle_interactive(layer, &mut msg_buf)
//         for sub_layer in layer:
//             sub_executor = LayerExecutor::BeforeInteractive { layer: sub_layer, ... }
//             sub_executor.handle_before()
//             self.sub_executors.push(sub_executor)
//     }

// TODO: How is the interface for evaluating interactive gates?
//  for layer in sc_iter:
//      handle_non_interactive(layer)
//      compute_msg(layer, &mut msg_buf)
//      for sub_layer in layer:
//          handle_layer(layer)
//

mod exp {
    struct ScCallGraph {
        g: petgraph::Graph<SubCircuit, ()>,
    }

    struct SubCircuit {
        call_map: CallMap,
    }

    struct CallMap;
}

#[cfg(test)]
mod tests {
    use crate::circuit::sub_circuit::{
        CallId, CallMap, CircId, SubCircIter, SubCircLayer, SubCircuit, SubCircuitInner,
    };
    use crate::circuit::{GateIdx, LayerIterable};
    use crate::protocols::{Gate, ScalarDim};
    use crate::BaseGate::*;
    use crate::BooleanGate;
    use crate::BooleanGate::*;
    use parking_lot::Mutex;
    use std::sync::{mpsc, Arc};

    type BaseCircuit = crate::circuit::BaseCircuit<BooleanGate, u32>;

    #[test]
    fn simple_layer_iter() {
        let mut main = BaseCircuit::new();
        let inp1_0 = main.add_gate(Base(Input(ScalarDim)));
        let inp2_0 = main.add_gate(Base(Input(ScalarDim)));
        let and_0 = main.add_wired_gate(And, &[inp1_0, inp2_0]);
        let sub_in0 = main.add_gate(Base(SubCircuitInput(ScalarDim)));

        let mut bc1 = BaseCircuit::new();
        let scin1_1 = bc1.add_gate(Base(SubCircuitInput(ScalarDim)));
        let scin2_1 = bc1.add_gate(Base(SubCircuitInput(ScalarDim)));
        let xor_1 = bc1.add_wired_gate(Xor, &[scin1_1, scin2_1]);
        let and_1 = bc1.add_wired_gate(And, &[scin1_1, xor_1]);
        let sub_in1 = bc1.add_gate(Base(SubCircuitInput(ScalarDim)));
        let out_1 = bc1.add_wired_gate(Base(SubCircuitOutput(ScalarDim)), &[sub_in1]);

        let mut bc2 = BaseCircuit::new();
        let scin1_2 = bc2.add_gate(Base(SubCircuitInput(ScalarDim)));
        let scin2_2 = bc2.add_gate(Base(SubCircuitInput(ScalarDim)));
        let and_2 = bc2.add_wired_gate(And, &[scin1_2, scin2_2]);
        let and_2 = bc2.add_wired_gate(And, &[scin1_2, and_2]);
        let out_2 = bc2.add_wired_gate(Base(SubCircuitOutput(ScalarDim)), &[and_2]);

        let mut main_sc = SubCircuit(Arc::new(Mutex::new(SubCircuitInner {
            base_circ_id: CircId(0),
            sub_circuits: vec![],
            call_map: CallMap::default(),
        })));

        let mut sc1 = SubCircuit(Arc::new(Mutex::new(SubCircuitInner {
            base_circ_id: CircId(1),
            sub_circuits: vec![],
            call_map: CallMap::default(),
        })));

        let mut sc2 = SubCircuit(Arc::new(Mutex::new(SubCircuitInner {
            base_circ_id: CircId(2),
            sub_circuits: vec![],
            call_map: CallMap::default(),
        })));

        {
            let mut main_sc = main_sc.lock();
            let call_id = CallId::Call(0);
            main_sc.sub_circuits.push(sc1.clone());
            main_sc
                .call_map
                .map
                .insert(inp1_0, vec![(call_id, scin1_1)]);
            main_sc.call_map.map.insert(and_0, vec![(call_id, and_0)]);
        }
        {
            let mut sc1 = sc1.lock();
            sc1.sub_circuits.push(sc2.clone());
            sc1.call_map.map.insert(out_1, vec![(CallId::Ret, sub_in0)]);
            let call_id = CallId::Call(0);
            sc1.call_map.map.insert(xor_1, vec![(call_id, scin1_2)]);
            sc1.call_map.map.insert(and_1, vec![(call_id, scin2_2)]);
        }
        {
            let mut sc2 = sc2.lock();
            sc2.call_map.map.insert(out_2, vec![(CallId::Ret, sub_in1)]);
        }

        let circ = super::Circuit {
            base_circuits: vec![main, bc1, bc2],
            main: main_sc,
        };
        let main_base_iter = circ.base_circuits[0].layer_iter();
        let (tx, rx) = mpsc::channel();
        let mut sc_iter = SubCircIter {
            circ: &circ,
            base_iter: main_base_iter,
            sub_circ_iters: Default::default(),
            sub_circuit: circ.main.clone(),
            own_returned_gates_tx: tx,
            own_returned_gates_rx: rx,
            parent_returned_gates_tx: None,
        };

        dbg!(sc_iter.next());
        let layer = sc_iter.next().unwrap();
        fn handle_layer<'sc_iter, 'base, G: Gate, Idx: GateIdx>(
            mut layer: SubCircLayer<'sc_iter, 'base, G, Idx>,
        ) {
            dbg!(&layer);
            for sub_layer in layer.sub_layer_iter() {
                handle_layer(sub_layer)
            }
        };
        handle_layer(layer);
        todo!()
    }
}
