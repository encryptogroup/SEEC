use crate::circuit::base_circuit::{BaseGate, Load};
use crate::circuit::circuit_connections::CrossCircuitConnections;
use crate::circuit::{base_circuit, BaseCircuit, CircuitId, DefaultIdx, GateIdx, LayerIterable};
use crate::errors::CircuitError;
use crate::protocols::{Gate, Wire};
use crate::{bristol, BooleanGate, SharedCircuit, SubCircuitGate};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::mem;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use tracing::trace;

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "\
    G: serde::Serialize + serde::de::DeserializeOwned,\
    Idx: GateIdx + Ord + Eq + Hash + serde::Serialize + serde::de::DeserializeOwned,\
    W: serde::Serialize + serde::de::DeserializeOwned")]
pub struct Circuit<G = BooleanGate, Idx = DefaultIdx, W = ()> {
    pub(crate) circuits: Vec<BaseCircuit<G, Idx, W>>,
    pub(crate) circ_map: HashMap<CircuitId, usize>,
    pub(crate) connections: CrossCircuitConnections<Idx>,
}

impl<G, Idx, W> Circuit<G, Idx, W> {
    pub fn get_circ(&self, id: CircuitId) -> &BaseCircuit<G, Idx, W> {
        &self.circuits[self.circ_map[&id]]
    }

    pub fn iter_circs(&self) -> impl Iterator<Item = &BaseCircuit<G, Idx, W>> + '_ {
        (0..self.circ_map.len()).map(|circ_id| self.get_circ(circ_id as CircuitId))
    }
}

impl<G: Gate, Idx: GateIdx, W: Wire> Circuit<G, Idx, W> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_gate(&self, id: SubCircuitGate<Idx>) -> G {
        self.get_circ(id.circuit_id).get_gate(id.gate_id)
    }

    // TODO optimization!
    pub fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        let same_circuit = self
            .get_circ(id.circuit_id)
            .parent_gates(id.gate_id)
            .map(move |parent_gate| SubCircuitGate::new(id.circuit_id, parent_gate));

        same_circuit.chain(self.connections.parent_gates(id))
    }

    pub fn gate_count(&self) -> usize {
        self.iter_circs().map(|circ| circ.gate_count()).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        // Reuse the the CircuitLayerIter api to get an iterator over individual gates. This
        // is maybe a little more inefficient than necessary, but probably fine for the moment,
        // as this method is expected to be called in the preprocessing phase
        let layer_iter = CircuitLayerIter::new(self);
        layer_iter.flat_map(|layer| {
            layer
                .sc_layers
                .into_iter()
                .flat_map(|(sc_id, _, base_layer)| base_layer.into_sc_iter(sc_id))
        })
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        // TODO this can be optimized in the future
        self.iter().filter(|(gate, _)| gate.is_interactive())
    }

    // pub fn into_base_circuit(self) -> BaseCircuit<Idx> {
    //     let mut res = BaseCircuit::new();
    //     let mut new_ids: Vec<Vec<_>> = vec![];
    //     for (c_id, circ) in self.circuits.into_iter().enumerate() {
    //         let g = circ.as_graph();
    //         let (nodes, edges) = (g.raw_nodes(), g.raw_edges());
    //
    //         new_ids.push(nodes.into_iter().map(|n| res.add_gate(n.weight)).collect());
    //         for edge in edges {
    //             let from = new_ids[c_id][edge.source().index()];
    //             let to = new_ids[c_id][edge.target().index()];
    //             res.add_wire(from, to);
    //         }
    //         // TODO DRY
    //         // TODO to convert this to the map implementation, maybe not iterate over input gates
    //         //  of circ but iterate over all edges in self.circuit_connectioctions at the end
    //         //  to link up circuits
    //         for input_gate in circ.sub_circuit_input_gates() {
    //             let to = (c_id.try_into().unwrap(), *input_gate);
    //             for from in self
    //                 .circuit_connections
    //                 .neighbors_directed(to, Direction::Incoming)
    //             {
    //                 if from.0 as usize >= new_ids.len() {
    //                     continue;
    //                 }
    //                 let from = new_ids[from.0 as usize][from.1.as_usize()];
    //                 res.add_wire(from, new_ids[c_id][to.1.as_usize()]);
    //             }
    //         }
    //         // TODO remove following when removing interleaving of SCs
    //         for output_gate in circ.sub_circuit_output_gates() {
    //             let from = (c_id.try_into().unwrap(), *output_gate);
    //             for to in self
    //                 .circuit_connections
    //                 .neighbors_directed(from, Direction::Outgoing)
    //             {
    //                 if to.0 as usize >= new_ids.len() {
    //                     continue;
    //                 }
    //                 let from = new_ids[from.0 as usize][from.1.as_usize()];
    //                 res.add_wire(from, new_ids[to.0 as usize][to.1.as_usize()]);
    //             }
    //         }
    //     }
    //
    //     res
    // }
}

impl<G, Idx, W> Circuit<G, Idx, W> {
    pub fn interactive_count(&self) -> usize {
        self.iter_circs().map(|circ| circ.interactive_count()).sum()
    }

    pub fn interactive_count_times_simd(&self) -> usize {
        self.iter_circs()
            .map(|circ| {
                circ.interactive_count() * circ.simd_size().map(NonZeroUsize::get).unwrap_or(1)
            })
            .sum()
    }

    /// Returns the input count of the **main circuit**.
    pub fn input_count(&self) -> usize {
        self.get_circ(0).input_count()
    }

    /// Returns the output count of the **main circuit**.
    pub fn output_count(&self) -> usize {
        self.get_circ(0).output_count()
    }
}

impl<Share, G> Circuit<G, u32>
where
    Share: Clone,
    G: Gate<Share = Share> + From<BaseGate<Share>> + for<'a> From<&'a bristol::Gate>,
{
    pub fn load_bristol(path: impl AsRef<Path>) -> Result<Self, CircuitError> {
        BaseCircuit::load_bristol(path, Load::Circuit).map(Into::into)
    }
}

impl<G: Clone, Idx: GateIdx, W: Clone> Clone for Circuit<G, Idx, W> {
    fn clone(&self) -> Self {
        Self {
            circuits: self.circuits.clone(),
            circ_map: self.circ_map.clone(),
            connections: self.connections.clone(),
        }
    }
}

#[derive(Clone)]
pub struct CircuitLayerIter<'a, G, Idx: GateIdx, W> {
    circuit: &'a Circuit<G, Idx, W>,
    layer_iters: HashMap<CircuitId, base_circuit::BaseLayerIter<'a, G, Idx, W>>,
}

impl<'a, G: Gate, Idx: GateIdx, W: Wire> CircuitLayerIter<'a, G, Idx, W> {
    pub fn new(circuit: &'a Circuit<G, Idx, W>) -> Self {
        let first_iter = circuit.get_circ(0).layer_iter();
        Self {
            circuit,
            layer_iters: [(0, first_iter)].into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitLayer<G, Idx: Hash + PartialEq + Eq> {
    pub(crate) sc_layers: Vec<(
        CircuitId,
        Option<NonZeroUsize>,
        base_circuit::CircuitLayer<G, Idx>,
    )>,
}

impl<'a, G: Gate, Idx: GateIdx, W: Wire> Iterator for CircuitLayerIter<'a, G, Idx, W> {
    type Item = CircuitLayer<G, Idx>;

    // TODO optimize this method, it makes up a big part of the runtime
    // TODO remove BaseLayerIters when they are not use anymore
    fn next(&mut self) -> Option<Self::Item> {
        trace!("layer_iters: {:#?}", &self.layer_iters);
        let mut sc_layers: Vec<_> = self
            .layer_iters
            .iter_mut()
            .filter_map(|(&sc_id, iter)| {
                iter.next().map(|layer| {
                    let simd_size = self.circuit.get_circ(sc_id).simd_size();
                    (sc_id, simd_size, layer)
                })
            })
            .collect();
        // Only retain iters which can yield elements
        self.layer_iters.retain(|_sc_id, iter| !iter.is_exhausted());

        // This is crucial as the executor depends on the and gates being in the same order for
        // both parties
        sc_layers.sort_unstable_by_key(|sc_id| sc_id.0);
        for (sc_id, _simd_size, layer) in &sc_layers {
            // TODO this iterates over all range connections for circuit in sc_layers for every
            //  layer. This could slow down the layer generation. Use hashmaps?
            for potential_out in layer.iter_ids() {
                let from = SubCircuitGate::new(*sc_id, potential_out);
                let outgoing = self.circuit.connections.outgoing_gates(from);

                for sc_gate in outgoing {
                    let to_layer_iter =
                        self.layer_iters
                            .entry(sc_gate.circuit_id)
                            .or_insert_with(|| {
                                base_circuit::BaseLayerIter::new_uninit(
                                    self.circuit.get_circ(sc_gate.circuit_id),
                                )
                            });
                    to_layer_iter.add_to_next_layer(sc_gate.gate_id.into());
                }
            }
        }
        // Todo can there be circuits where a layer is empty, but a later call to next returns
        //  a layer?
        if sc_layers.is_empty() {
            None
        } else {
            Some(CircuitLayer { sc_layers })
        }
    }
}

impl<G: Gate, Idx: GateIdx + Hash + PartialEq + Eq + Copy> CircuitLayer<G, Idx> {
    pub(crate) fn interactive_count_times_simd(&self) -> usize {
        self.sc_layers
            .iter()
            .map(|(_, simd, layer)| {
                let simd = simd.map(|v| v.get()).unwrap_or(1);
                simd * layer.interactive_len()
            })
            .sum()
    }

    pub(crate) fn split_simd(mut self) -> (Self, Self) {
        let mut simd = vec![];
        self.sc_layers.retain_mut(|(sc_id, simd_size, layer)| {
            if simd_size.is_some() {
                simd.push((*sc_id, *simd_size, mem::take(layer)));
                false
            } else {
                true
            }
        });
        (self, Self { sc_layers: simd })
    }

    pub(crate) fn non_interactive_iter(
        &self,
    ) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        self.sc_layers.iter().flat_map(|(sc_id, _, layer)| {
            layer
                .non_interactive_iter()
                .map(|(gate, gate_idx)| (gate, SubCircuitGate::new(*sc_id, gate_idx)))
        })
    }

    pub(crate) fn interactive_iter(
        &self,
    ) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        self.sc_layers.iter().flat_map(|(sc_id, _, layer)| {
            layer
                .interactive_iter()
                .map(|(gate, gate_idx)| (gate, SubCircuitGate::new(*sc_id, gate_idx)))
        })
    }

    pub(crate) fn freeable_simd_gates(&self) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.sc_layers.iter().flat_map(|(sc_id, _, layer)| {
            layer
                .freeable_gates
                .iter()
                .map(|gate_idx| SubCircuitGate::new(*sc_id, *gate_idx))
        })
    }
}

impl<G, Idx: GateIdx, W> Default for Circuit<G, Idx, W> {
    fn default() -> Self {
        Self {
            circuits: vec![],
            circ_map: Default::default(),
            connections: Default::default(),
        }
    }
}

impl<G, Idx: GateIdx + Default, W> From<BaseCircuit<G, Idx, W>> for Circuit<G, Idx, W> {
    fn from(bc: BaseCircuit<G, Idx, W>) -> Self {
        Self {
            circuits: vec![bc],
            circ_map: [(0, 0)].into_iter().collect(),
            connections: Default::default(),
        }
    }
}

impl<G, Idx: GateIdx, W> TryFrom<SharedCircuit<G, Idx, W>> for Circuit<G, Idx, W> {
    type Error = SharedCircuit<G, Idx, W>;

    fn try_from(circuit: SharedCircuit<G, Idx, W>) -> Result<Self, Self::Error> {
        Arc::try_unwrap(circuit).map(|mutex| mutex.into_inner().into())
    }
}
