use crate::circuit::circuit_connections::CrossCircuitConnections;
use crate::circuit::{base_circuit, dyn_layers::CircuitLayerIter, CircuitId, GateIdx};
use crate::common::BitVec;
use crate::protocols::Gate;
use crate::{GateId, SubCircuitGate};
use either::Either;
use serde::{Deserialize, Serialize};
use std::cmp::Eq;
use std::cmp::Ord;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ops::Range;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(bound = "G: serde::Serialize + serde::de::DeserializeOwned,\
    Idx: Ord + Eq + Hash + serde::Serialize + serde::de::DeserializeOwned")]
pub struct Circuit<G, Idx> {
    main_input_gates: Vec<GateId<Idx>>,
    main_output_gates: Vec<GateId<Idx>>,
    sub_circuits: Vec<SubCircuit<G, Idx>>,
    sc_map: HashMap<CircuitId, usize>,
    layers: Vec<Layer>,
    cross_circuit_incoming: CrossCircuitConnections<Idx>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Layer {
    sub_circuit_layers: Vec<CircuitId>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SubCircuit<G, Idx> {
    simd_size: Option<NonZeroUsize>,
    gate_count: usize,
    interactive_count: usize,
    layers: Vec<ScLayer<G, Idx>>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScLayer<G, Idx> {
    non_interactive: ScLayerGates<G, Idx>,
    interactive: ScLayerGates<G, Idx>,
    incoming_idx: Vec<GateId<Idx>>,
}

#[derive(Debug, Clone)]
pub struct ExecutableScLayer<'c, G, Idx> {
    pub(crate) sc_id: CircuitId,
    layer: &'c ScLayer<G, Idx>,
}

#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScLayerGates<G, Idx> {
    gates: Vec<G>,
    idx: Vec<GateId<Idx>>,
    incoming: Vec<Range<Idx>>,
    potential_cross_circ: BitVec,
}

#[derive(Debug, Clone)]
pub struct LayerIterator<'a, G, Idx> {
    circ: &'a Circuit<G, Idx>,
    layer_iter_state: Vec<usize>,
    current_layer: usize,
}

// TODO: Ramblings, damn, the approach I want to take is a little more complicated than I though,
//  I think i need to include an ID for the base circuit so that I can match up layers with
//  different sc id's to the same underlying sub circuit. Alternatively I could maybe use ptr
//  values of the base scs, like I do in the builder -> circuit conversion

impl<G, Idx> Circuit<G, Idx> {
    pub fn gate_count(&self) -> usize {
        self.sc_map
            .values()
            .map(|mapped_id| self.sub_circuits[*mapped_id].gate_count)
            .sum()
    }

    pub fn interactive_count(&self) -> usize {
        self.sc_map
            .values()
            .map(|mapped_id| self.sub_circuits[*mapped_id].interactive_count)
            .sum()
    }

    pub fn interactive_count_times_simd(&self) -> usize {
        self.sc_map
            .values()
            .map(|mapped_id| {
                let sc = &self.sub_circuits[*mapped_id];
                sc.interactive_count * sc.simd_size.map(NonZeroUsize::get).unwrap_or(1)
            })
            .sum()
    }

    pub fn sub_circuit_count(&self) -> usize {
        self.sc_map.len()
    }

    pub fn gate_counts(&self) -> impl Iterator<Item = (usize, Option<NonZeroUsize>)> + '_ {
        // Return gate counts in correct order
        (0..self.sc_map.len() as CircuitId).map(|sc_id| {
            let sc = self.get_circ(sc_id);
            (sc.gate_count, sc.simd_size)
        })
    }

    pub fn input_count(&self) -> usize {
        self.main_input_gates.len()
    }

    pub fn output_count(&self) -> usize {
        self.main_output_gates.len()
    }

    pub fn input_gates(&self) -> &[GateId<Idx>] {
        &self.main_input_gates
    }

    pub fn output_gates(&self) -> &[GateId<Idx>] {
        &self.main_output_gates
    }

    pub(crate) fn get_circ(&self, sc_id: CircuitId) -> &SubCircuit<G, Idx> {
        let mapped_id = self.sc_map[&sc_id];
        &self.sub_circuits[mapped_id]
    }
}

impl<G: Gate, Idx: GateIdx> Circuit<G, Idx> {
    fn add_layer(
        &mut self,
        (sc_id, simd_size, sc_layer): (
            CircuitId,
            Option<NonZeroUsize>,
            base_circuit::CircuitLayer<G, Idx>,
        ),
        circ: &super::dyn_layers::Circuit<G, Idx>,
        layer_ptrs: &[usize],
        splits: &mut HashMap<usize, Vec<usize>>,
    ) {
        let mapped_sc_id = self.sc_map[&sc_id];
        let new_sc_layer = ScLayer::from_base_layer(sc_id, sc_layer, circ);
        match self.sub_circuits.get_mut(mapped_sc_id) {
            None => {
                self.sub_circuits
                    .push(SubCircuit::new(new_sc_layer, simd_size));
            }
            Some(sc) => {
                let layer = layer_ptrs[sc_id as usize];
                if let Some(precomp_layer) = sc.layers.get(layer) {
                    if &new_sc_layer != precomp_layer {
                        let mut split_sc = sc.clone();
                        for split in splits.get(&mapped_sc_id).unwrap_or(&vec![]) {
                            let split_sc = &self.sub_circuits[*split];
                            let split_precomp_layer = &split_sc.layers[layer];
                            if &new_sc_layer == split_precomp_layer {
                                self.sc_map.insert(sc_id, *split);
                                return;
                            }
                        }
                        // truncate everything beginning from precomp_layer which is not identical
                        // to sc_layer
                        split_sc.truncate(layer);
                        split_sc.push_layer(new_sc_layer);
                        // Update sc_map to so sc_id points to new sc
                        self.sc_map.insert(sc_id, self.sub_circuits.len());
                        splits
                            .entry(mapped_sc_id)
                            .or_default()
                            .push(self.sub_circuits.len());
                        self.sub_circuits.push(split_sc);
                    }
                } else {
                    sc.push_layer(new_sc_layer);
                }
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        let layer_iter = LayerIterator::new(self);
        layer_iter.flatten().flat_map(|layer| {
            layer
                .iter()
                .map(move |(g, id)| (g, SubCircuitGate::new(layer.sc_id, id)))
        })
    }

    pub fn iter_with_parents(
        &self,
    ) -> impl Iterator<
        Item = (
            G,
            SubCircuitGate<Idx>,
            impl Iterator<Item = SubCircuitGate<Idx>> + '_,
        ),
    > + '_ {
        let layer_iter = LayerIterator::new(self);
        layer_iter
            .flatten()
            .flat_map(move |layer| layer.iter_with_parents(self))
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        let layer_iter = LayerIterator::new(self);
        layer_iter.flatten().flat_map(|layer| {
            layer
                .layer
                .interactive
                .iter()
                .map(move |(g, id)| (g, SubCircuitGate::new(layer.sc_id, id)))
        })
    }

    pub fn interactive_with_parents_iter(
        &self,
    ) -> impl Iterator<
        Item = (
            G,
            SubCircuitGate<Idx>,
            impl Iterator<Item = SubCircuitGate<Idx>> + '_,
        ),
    > + '_ {
        let layer_iter = LayerIterator::new(self);
        layer_iter.flatten().flat_map(|layer| {
            let sc_id = layer.sc_id;
            layer
                .clone()
                .interactive_iter()
                .zip(layer.interactive_parents_iter(self))
                .map(move |((g, id), parents)| (g, SubCircuitGate::new(sc_id, id), parents))
        })
    }
}

impl<G: Gate, Idx: GateIdx> super::dyn_layers::Circuit<G, Idx> {
    pub fn precompute_layers(self) -> Circuit<G, Idx> {
        let layer_iter = CircuitLayerIter::new(&self);
        let mut res_circ = Circuit {
            main_input_gates: self.get_circ(0).input_gates().to_vec(),
            main_output_gates: self.get_circ(0).output_gates().to_vec(),
            sub_circuits: vec![],
            sc_map: self.circ_map.clone(),
            layers: vec![],
            cross_circuit_incoming: self.connections.clone_incoming(),
        };
        let mut layer_ptrs = vec![0; self.circ_map.len()];
        let mut splits = HashMap::default();

        for layer in layer_iter {
            let mut layer_ids = Vec::with_capacity(layer.sc_layers.len());
            for sc_layer in layer.sc_layers {
                let sc_id = sc_layer.0;
                res_circ.add_layer(sc_layer, &self, &layer_ptrs, &mut splits);
                layer_ptrs[sc_id as usize] += 1;
                layer_ids.push(sc_id);
            }
            res_circ.layers.push(Layer {
                sub_circuit_layers: layer_ids,
            });
        }
        res_circ
    }
}

impl<G, Idx> SubCircuit<G, Idx> {
    pub(crate) fn simd_size(&self) -> Option<NonZeroUsize> {
        self.simd_size
    }
}

impl<G: Gate, Idx: GateIdx> SubCircuit<G, Idx> {
    fn new(layer: ScLayer<G, Idx>, simd_size: Option<NonZeroUsize>) -> Self {
        Self {
            simd_size,
            gate_count: layer.len(),
            interactive_count: layer.interactive.len(),
            layers: vec![layer],
        }
    }

    fn push_layer(&mut self, layer: ScLayer<G, Idx>) {
        self.gate_count += layer.len();
        self.interactive_count += layer.interactive.len();
        self.layers.push(layer);
    }

    fn truncate(&mut self, len: usize) {
        let removed_gates: usize = self.layers[len..].iter().map(|layer| layer.len()).sum();
        let removed_interactive_gates: usize = self.layers[len..]
            .iter()
            .map(|layer| layer.interactive_count())
            .sum();
        self.gate_count -= removed_gates;
        self.interactive_count -= removed_interactive_gates;
        self.layers.truncate(len);
    }
}

impl<'c, G: Gate, Idx: GateIdx> ExecutableScLayer<'c, G, Idx> {
    pub fn len(&self) -> usize {
        self.layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn interactive_count(&self) -> usize {
        self.layer.interactive_count()
    }

    fn iter<'this>(&'this self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone + 'c
    where
        'c: 'this,
    {
        self.non_interactive_iter().chain(self.interactive_iter())
    }

    fn iter_with_parents(
        self,
        circ: &'c Circuit<G, Idx>,
    ) -> impl Iterator<
        Item = (
            G,
            SubCircuitGate<Idx>,
            impl Iterator<Item = SubCircuitGate<Idx>> + 'c,
        ),
    > + 'c {
        let sc_id = self.sc_id;
        let non_interactive = self.non_interactive_iter().zip(
            self.clone()
                .non_interactive_parents_iter(circ)
                .map(Either::Left),
        );
        let interactive = self
            .interactive_iter()
            .zip(self.interactive_parents_iter(circ).map(Either::Right));
        non_interactive
            .chain(interactive)
            .map(move |((g, idx), parents)| (g, SubCircuitGate::new(sc_id, idx), parents))
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone + 'c {
        self.layer
            .interactive
            .gates
            .iter()
            .cloned()
            .zip(self.layer.interactive.idx.iter().cloned())
    }

    pub fn interactive_gates(&self) -> &'c [G] {
        &self.layer.interactive.gates
    }

    pub fn non_interactive_gates(&self) -> &'c [G] {
        &self.layer.non_interactive.gates
    }

    pub fn interactive_indices(&self) -> &'c [GateId<Idx>] {
        bytemuck::cast_slice(&self.layer.interactive.idx[..])
    }

    pub fn non_interactive_indices(&self) -> &'c [GateId<Idx>] {
        bytemuck::cast_slice(&self.layer.non_interactive.idx[..])
    }

    pub fn interactive_parents_iter(
        self,
        circ: &'c Circuit<G, Idx>,
    ) -> impl Iterator<Item = impl Iterator<Item = SubCircuitGate<Idx>> + 'c> + 'c {
        let sc_id = self.sc_id;
        self.layer
            .interactive
            .incoming
            .iter()
            .zip(self.layer.interactive.potential_cross_circ.iter().by_vals())
            .enumerate()
            .map(move |(idx, (inc_range, cross_circuit))| {
                if cross_circuit {
                    let gate = SubCircuitGate::new(sc_id, self.layer.interactive.idx[idx]);
                    Either::Left(circ.cross_circuit_incoming.parent_gates(gate))
                } else {
                    Either::Right(
                        self.layer.incoming_idx[inc_range.start.index()..inc_range.end.index()]
                            .iter()
                            .map(move |id| SubCircuitGate::new(sc_id, *id)),
                    )
                }
            })
    }

    #[inline]
    pub fn non_interactive_iter(&self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone + 'c {
        self.layer
            .non_interactive
            .gates
            .iter()
            .cloned()
            .zip(self.layer.non_interactive.idx.iter().cloned())
    }

    #[inline]
    pub fn non_interactive_parents_iter(
        self,
        circ: &'c Circuit<G, Idx>,
    ) -> impl Iterator<Item = impl Iterator<Item = SubCircuitGate<Idx>> + 'c> + 'c {
        let sc_id = self.sc_id;
        self.layer
            .non_interactive
            .incoming
            .iter()
            .zip(
                self.layer
                    .non_interactive
                    .potential_cross_circ
                    .iter()
                    .by_vals(),
            )
            .enumerate()
            .map(move |(idx, (inc_range, cross_circuit))| {
                if cross_circuit {
                    let gate = SubCircuitGate::new(sc_id, self.layer.non_interactive.idx[idx]);
                    Either::Left(circ.cross_circuit_incoming.parent_gates(gate))
                } else {
                    Either::Right(
                        self.layer.incoming_idx[inc_range.start.index()..inc_range.end.index()]
                            .iter()
                            .map(move |id| SubCircuitGate::new(sc_id, *id)),
                    )
                }
            })
    }
}

impl<G: Gate, Idx: GateIdx> ScLayer<G, Idx> {
    pub fn len(&self) -> usize {
        self.interactive.len() + self.non_interactive.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn interactive_count(&self) -> usize {
        self.interactive.len()
    }

    fn from_base_layer(
        sc_id: CircuitId,
        base_layer: base_circuit::CircuitLayer<G, Idx>,
        circ: &super::dyn_layers::Circuit<G, Idx>,
    ) -> Self {
        let incoming_cnt_guess =
            (base_layer.interactive_len() + base_layer.non_interactive_len()) * 2;
        let mut incoming_idx = Vec::with_capacity(incoming_cnt_guess);
        let non_interactive = ScLayerGates::from_base(
            sc_id,
            (
                base_layer.non_interactive_gates,
                base_layer.non_interactive_ids,
            ),
            &mut incoming_idx,
            circ,
        );
        let interactive = ScLayerGates::from_base(
            sc_id,
            (base_layer.interactive_gates, base_layer.interactive_ids),
            &mut incoming_idx,
            circ,
        );

        Self {
            non_interactive,
            interactive,
            incoming_idx,
        }
    }
}

impl<G: Gate, Idx: GateIdx> ScLayerGates<G, Idx> {
    fn len(&self) -> usize {
        self.gates.len()
    }

    fn from_base(
        sc_id: CircuitId,
        (gates, idx): (Vec<G>, Vec<GateId<Idx>>),
        incoming_idx: &mut Vec<GateId<Idx>>,
        circ: &super::dyn_layers::Circuit<G, Idx>,
    ) -> Self {
        let mut incoming = Vec::with_capacity(gates.len());
        let mut potential_cross_circ = BitVec::with_capacity(gates.len());

        for id in &idx {
            let start = Idx::new(incoming_idx.len());
            incoming_idx.extend(circ.get_circ(sc_id).parent_gates(*id));
            let end = Idx::new(incoming_idx.len());
            incoming.push(start..end);
            let has_cross_connection = circ
                .connections
                .parent_gates(SubCircuitGate::new(sc_id, *id))
                .next()
                .is_some();
            potential_cross_circ.push(has_cross_connection);
        }

        Self {
            gates,
            idx,
            incoming,
            potential_cross_circ,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone + '_ {
        self.gates.iter().cloned().zip(self.idx.iter().cloned())
    }
}

impl<'a, G, Idx> LayerIterator<'a, G, Idx> {
    pub fn new(circ: &'a Circuit<G, Idx>) -> Self {
        let layer_iter_state = vec![0; circ.sc_map.len()];
        Self {
            circ,
            layer_iter_state,
            current_layer: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScLayerIterator<'c, G, Idx> {
    circ: &'c Circuit<G, Idx>,
    sc_layer: &'c [CircuitId],
    layer_iter_state: Vec<usize>,
    skip_simd: bool,
    skip_scalar: bool,
}

impl<'c, G: Clone, Idx: Clone> ScLayerIterator<'c, G, Idx> {
    /// Returns two iterators (scalar, simd) which only return layers from either scalar or simd
    /// sc's
    pub(crate) fn split_simd(self) -> (Self, Self) {
        let mut scalar = self.clone();
        scalar.skip_simd = true;
        let mut simd = self;
        simd.skip_scalar = true;
        (scalar, simd)
    }
}

impl<'c, G: Gate, Idx: GateIdx> ScLayerIterator<'c, G, Idx> {
    pub(crate) fn interactive_count_times_simd(&self) -> usize {
        self.clone()
            .map(|layer| {
                let simd_size = self
                    .circ
                    .get_circ(layer.sc_id)
                    .simd_size
                    .map(|s| s.get())
                    .unwrap_or(1);
                layer.interactive_count() * simd_size
            })
            .sum()
    }
}

impl<'c, G: Gate, Idx: GateIdx> Iterator for ScLayerIterator<'c, G, Idx> {
    type Item = ExecutableScLayer<'c, G, Idx>;

    fn next(&mut self) -> Option<Self::Item> {
        let sc_idx = *self.sc_layer.first()?;
        let sc = self.circ.get_circ(sc_idx);
        self.sc_layer = &self.sc_layer[1..];
        // TODO this recursion could lead to a stack overflow for many parallel sc's
        match (sc.simd_size(), self.skip_scalar, self.skip_simd) {
            (None, true, _) => return self.next(),
            (Some(_), _, true) => return self.next(),
            _ => (),
        }
        let layer = sc.layers.get(self.layer_iter_state[sc_idx as usize]);
        layer.map(|layer| ExecutableScLayer {
            sc_id: sc_idx,
            layer,
        })
    }
}

impl<'c, G: Gate, Idx: GateIdx> Iterator for LayerIterator<'c, G, Idx> {
    type Item = ScLayerIterator<'c, G, Idx>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_layer >= self.circ.layers.len() {
            return None;
        }
        let sc_layer = &self.circ.layers[self.current_layer].sub_circuit_layers;
        let layer_iter_state = self.layer_iter_state.clone();
        for sc in sc_layer {
            self.layer_iter_state[*sc as usize] += 1;
        }
        self.current_layer += 1;
        Some(ScLayerIterator {
            circ: self.circ,
            sc_layer,
            layer_iter_state,
            skip_simd: false,
            skip_scalar: false,
        })
    }
}
