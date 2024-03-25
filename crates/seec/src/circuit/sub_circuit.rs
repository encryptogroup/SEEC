use crate::circuit::circuit_connections::CrossCircuitConnections;
use crate::circuit::{base_circuit, BaseCircuit, GateIdx};
use crate::protocols::Gate;
use crate::GateId;
use ahash::HashMap;
use std::collections::hash_map;
use std::slice;
use std::sync::Arc;

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct CallId(u32);
#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct CircId(u32);

pub struct Circuit<G, Idx> {
    base_circuits: Vec<BaseCircuit<G, Idx>>,
    main: SubCircuit<Idx>,
}

#[derive(Clone)]
pub struct SubCircuit<Idx>(Arc<SubCircuitInner<Idx>>);

struct SubCircuitInner<Idx> {
    base_circ_id: CircId, // this corresponds to the position in Circuits.base_circuits Vec
    sub_circuits: Vec<SubCircuit<Idx>>, // position in this Vec is CallId
    call_map: CallMap<Idx>,
}

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

struct SubCircIter<'base, G, Idx: GateIdx> {
    circ: &'base Circuit<G, Idx>,
    base_iter: base_circuit::BaseLayerIter<'base, G, Idx, ()>,
    sub_circ_iters: HashMap<CallId, Self>,
    sub_circuit: SubCircuit<Idx>,
}

struct SubCircLayer<'sc_iter, 'base, G, Idx: GateIdx> {
    call_id: CallId,
    base_layer: base_circuit::CircuitLayer<G, Idx>,
    sub_circ_iter: &'sc_iter mut SubCircIter<'base, G, Idx>,
}

impl<'base, G: Gate, Idx: GateIdx> SubCircIter<'base, G, Idx> {
    fn next(&mut self) -> Option<SubCircLayer<'_, 'base, G, Idx>> {
        let base_layer = self.base_iter.next().unwrap();
        for gate_id in base_layer.iter_ids() {
            for &(call_id, out_gate) in self.sub_circuit.0.call_map.outgoing_gates(gate_id) {
                let sc_iter = self.sub_circ_iters.entry(call_id).or_insert_with(|| {
                    let called_sc = self.sub_circuit.0.sub_circuits[call_id.0 as usize].clone();
                    let called_circ_id = called_sc.0.base_circ_id;
                    let called_base_circ = &self.circ.base_circuits[called_circ_id.0 as usize];
                    let mut base_iter = base_circuit::BaseLayerIter::new_uninit(called_base_circ);
                    SubCircIter {
                        circ: self.circ,
                        base_iter,
                        sub_circ_iters: Default::default(),
                        sub_circuit: called_sc,
                    }
                });
                // TODO
                sc_iter.base_iter.add_to_next_layer(out_gate.into());
            }
        }
        // drive own base iter to get gates for this base circuit
        // check own gates in self.sub_circ call_map to see if they are outgoing
        // if yes, add new SubCircIter to self.sub_circ_iters or update iter state
        // returns None when?? If own base_iter and all sub_circ_iters are exhausted?
        todo!()
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
    iter: hash_map::IterMut<'a, CallId, SubCircIter<'base, G, Idx>>,
}


/// enum LayerExecutor {
///     BeforeInteractive {
///         layer: Layer
///         sub_executors: Vec<LayerExecutor>
///     },
///     AfterInteractive {
///         interactive_gates: InteractiveLayer
///         msg_range: Range<usize>
///         sub_executors: Vec<LayerExecutor>
///     }
/// }
///
///
/// impl LayerExecutor {
///     fn handle_before() {
///         handle_non_interactive(layer)
///         let msg_idx = handle_interactive(layer, &mut msg_buf)
///         for sub_layer in layer:
///             sub_executor = LayerExecutor::BeforeInteractive { layer: sub_layer, ... }
///             sub_executor.handle_before()
///             self.sub_executors.push(sub_executor)
///     }



// TODO: How is the interface for evaluating interactive gates?
//  for layer in sc_iter:
//      handle_non_interactive(layer)
//      compute_msg(layer, &mut msg_buf)
//      for sub_layer in layer:
//          handle_layer(layer)
//
