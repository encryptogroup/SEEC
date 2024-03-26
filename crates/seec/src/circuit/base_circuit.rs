#![allow(clippy::extra_unused_type_parameters)] // false positive in current nightly

use ahash::HashMap;
use parking_lot::lock_api::Mutex;
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, Direction, Graph};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, trace};

use crate::circuit::{CircuitId, DefaultIdx, GateIdx, LayerIterable};
use crate::errors::CircuitError;
use crate::protocols::boolean_gmw::BooleanGate;
use crate::protocols::{Dimension, Gate, ScalarDim, Share, ShareStorage, Wire};
use crate::{bristol, SharedCircuit, SubCircuitGate};

type CircuitGraph<Gate, Idx, Wire> = Graph<Gate, Wire, Directed, Idx>;

#[derive(Serialize, Deserialize)]
#[serde(bound = "Gate: serde::Serialize + serde::de::DeserializeOwned,\
    Idx: GateIdx + serde::Serialize + serde::de::DeserializeOwned,\
    Wire: serde::Serialize + serde::de::DeserializeOwned")]
pub struct BaseCircuit<Gate = BooleanGate, Idx = u32, Wire = ()> {
    graph: CircuitGraph<Gate, Idx, Wire>,
    is_main: bool,
    simd_size: Option<NonZeroUsize>,
    interactive_gate_count: usize,
    input_gates: Vec<GateId<Idx>>,
    output_gates: Vec<GateId<Idx>>,
    constant_gates: Vec<GateId<Idx>>,
    sub_circuit_output_gates: Vec<GateId<Idx>>,
    pub(crate) sub_circuit_input_gates: Vec<GateId<Idx>>,
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum BaseGate<T, D = ScalarDim> {
    Output(D),
    Input(D),
    /// Input from a sub circuit called within a circuit.
    SubCircuitInput(D),
    /// Output from this circuit into another sub circuit
    SubCircuitOutput(D),
    ConnectToMain(D),
    /// Connects a sub circuit to the main circuit and selects the i'th individual value from
    /// the SIMD output
    ConnectToMainFromSimd((D, u32)),
    Constant(T),
    Debug,
}

#[derive(
    Debug,
    Default,
    Copy,
    Clone,
    Ord,
    PartialOrd,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Pod,
    Zeroable,
)]
#[repr(transparent)]
pub struct GateId<Idx = DefaultIdx>(pub(crate) Idx);

impl<G: Gate, Idx: GateIdx, W: Wire> BaseCircuit<G, Idx, W> {
    pub fn new() -> Self {
        Self {
            graph: Default::default(),
            is_main: false,
            simd_size: None,
            interactive_gate_count: 0,
            input_gates: vec![],
            output_gates: vec![],
            constant_gates: vec![],
            sub_circuit_output_gates: vec![],
            sub_circuit_input_gates: vec![],
        }
    }

    pub fn new_main() -> Self {
        let mut new = Self::new();
        new.is_main = true;
        new
    }

    pub fn with_capacity(gates: usize, wires: usize) -> Self {
        let mut new = Self::new();
        new.graph = Graph::with_capacity(gates, wires);
        new
    }

    #[tracing::instrument(level = "trace", skip(self))]
    pub fn add_gate(&mut self, gate: G) -> GateId<Idx> {
        let gate_id = self.graph.add_node(gate.clone()).into();

        if let Some(base_gate) = gate.as_base_gate() {
            match base_gate {
                BaseGate::Input(_) => self.input_gates.push(gate_id),
                BaseGate::Constant(_) => self.constant_gates.push(gate_id),
                BaseGate::Output(_) => self.output_gates.push(gate_id),
                BaseGate::SubCircuitOutput(_) => self.sub_circuit_output_gates.push(gate_id),
                BaseGate::SubCircuitInput(_)
                | BaseGate::ConnectToMain(_)
                | BaseGate::ConnectToMainFromSimd(_) => self.sub_circuit_input_gates.push(gate_id),
                BaseGate::Debug => (/* nothing special to do */),
            }
        }
        if gate.is_interactive() {
            self.interactive_gate_count += 1;
        }
        trace!(%gate_id, "Added gate");
        gate_id
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn add_sc_input_gate(&mut self, gate: G) -> GateId<Idx> {
        let gate_id = self.add_gate(gate.clone());
        // explicitly add sc input id if gate is **not** a sc input
        if !matches!(gate.as_base_gate(), Some(BaseGate::SubCircuitInput(_))) {
            self.sub_circuit_input_gates.push(gate_id);
        }
        debug!(%gate_id, ?gate, "Added sub circuit input gate");
        gate_id
    }

    pub fn get_gate(&self, id: impl Into<GateId<Idx>>) -> G {
        let id = id.into();
        self.graph[NodeIndex::from(id)].clone()
    }

    pub fn parent_gates(
        &self,
        id: impl Into<GateId<Idx>>,
    ) -> impl Iterator<Item = GateId<Idx>> + '_ {
        self.graph
            .neighbors_directed(id.into().into(), Direction::Incoming)
            .map(GateId::from)
    }

    pub fn gate_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn wire_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn save_dot(&self, path: impl AsRef<Path>) -> Result<(), CircuitError> {
        let path = {
            let mut p = path.as_ref().to_path_buf();
            p.set_extension("dot");
            p
        };
        // TODO it would be nice to display the gate type AND id, however this doesn't seem to be
        //  possible with the current api of petgraph
        let dot_content = Dot::with_config(&self.graph, &[Config::EdgeNoLabel]);
        fs::write(path, format!("{dot_content:?}")).map_err(CircuitError::SaveAsDot)?;
        Ok(())
    }

    pub fn as_graph(&self) -> &CircuitGraph<G, Idx, W> {
        &self.graph
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, GateId<Idx>)> + '_ {
        self.layer_iter()
            .visit_sc_inputs()
            .flat_map(|layer| layer.into_interactive_iter())
    }

    pub fn iter(&self) -> impl Iterator<Item = (G, GateId<Idx>)> + '_ {
        self.layer_iter()
            .visit_sc_inputs()
            .flat_map(|layer| layer.into_iter())
    }
}

impl<G: Gate, Idx: GateIdx> BaseCircuit<G, Idx, ()> {
    #[tracing::instrument(level="trace", skip(self), fields(%from, %to))]
    pub fn add_wire(&mut self, from: GateId<Idx>, to: GateId<Idx>) {
        self.graph.add_edge(from.into(), to.into(), ());
        trace!("Added wire");
    }

    pub fn add_wired_gate(&mut self, gate: G, from: &[GateId<Idx>]) -> GateId<Idx> {
        let added = self.add_gate(gate);
        // reverse so that connections during online phase are yielded in the same order as passed
        // here in from
        for from_id in from.iter().rev() {
            self.add_wire(*from_id, added);
        }
        added
    }

    /// Adds another SubCircuit into `self`. The gates and wires of `circuit` are added to
    /// `self` and the `inputs` gates in `self` are connected to the [`BaseCircuit::sub_circuit_input_gates`]
    /// of the provided `circuit`.
    #[instrument(level = "debug", ret, skip_all)]
    pub fn add_sub_circuit(
        &mut self,
        circuit: &Self,
        inputs: impl IntoIterator<Item = GateId<Idx>>,
    ) -> Vec<GateId<Idx>> {
        assert!(!circuit.is_main, "Can't add main circuit as sub circuit");
        assert!(
            circuit.input_gates().is_empty(),
            "Added circuit can't have Input gates. Must have SubCircuitInput gates"
        );
        let mut gate_id_map = vec![GateId::default(); circuit.gate_count()];
        for (gate, id) in circuit.iter() {
            let new_id = self.add_gate(gate);
            gate_id_map[id.as_usize()] = new_id;
            for parent in circuit.parent_gates(id) {
                let new_parent_id = gate_id_map[parent.as_usize()];
                self.add_wire(new_parent_id, new_id);
            }
        }
        let mut inp_cnt = 0;
        for (from, to) in inputs.into_iter().zip(circuit.sub_circuit_input_gates()) {
            self.add_wire(from, gate_id_map[to.as_usize()]);
            inp_cnt += 1;
        }
        assert_eq!(
            inp_cnt,
            circuit.sub_circuit_input_gates().len(),
            "inputs needs to have same length as circuit.sub_circuit_input_gates()"
        );
        circuit
            .sub_circuit_output_gates()
            .iter()
            .map(|out_id| gate_id_map[out_id.as_usize()])
            .collect()
    }
}

impl<G, Idx, W> BaseCircuit<G, Idx, W> {
    pub fn is_simd(&self) -> bool {
        self.simd_size.is_some()
    }

    pub fn simd_size(&self) -> Option<NonZeroUsize> {
        self.simd_size
    }

    pub fn interactive_count(&self) -> usize {
        self.interactive_gate_count
    }

    pub fn input_count(&self) -> usize {
        self.input_gates.len()
    }

    pub fn input_gates(&self) -> &[GateId<Idx>] {
        &self.input_gates
    }

    pub fn sub_circuit_input_gates(&self) -> &[GateId<Idx>] {
        &self.sub_circuit_input_gates
    }

    pub fn sub_circuit_input_count(&self) -> usize {
        self.sub_circuit_input_gates.len()
    }

    pub fn output_count(&self) -> usize {
        self.output_gates.len()
    }

    pub fn output_gates(&self) -> &[GateId<Idx>] {
        &self.output_gates
    }

    pub fn sub_circuit_output_gates(&self) -> &[GateId<Idx>] {
        &self.sub_circuit_output_gates
    }

    pub fn into_shared(self) -> SharedCircuit<G, Idx, W> {
        SharedCircuit::new(Mutex::new(self))
    }

    pub fn set_simd_size(&mut self, size: NonZeroUsize) {
        self.simd_size = Some(size);
    }
}

#[derive(Debug)]
pub enum Load {
    Circuit,
    SubCircuit,
}

impl<Share, G, Idx: GateIdx> BaseCircuit<G, Idx>
where
    Share: Clone,
    G: Gate<Share = Share> + From<BaseGate<Share>> + for<'a> From<&'a bristol::Gate>,
{
    #[tracing::instrument(skip(bristol))]
    pub fn from_bristol(bristol: bristol::Circuit, load: Load) -> Result<Self, CircuitError> {
        info!(
            "Converting bristol circuit with header: {:?}",
            bristol.header
        );
        let mut circuit = Self::with_capacity(bristol.header.gates, bristol.header.wires);
        let total_input_wires = bristol.total_input_wires();
        // We treat the output wires of the bristol::Gates as their GateIds. Unfortunately,
        // the output wires are not given in ascending order, so we need to save a mapping
        // of wire ids to GateIds
        let mut wire_mapping = vec![GateId::default(); bristol.header.wires];
        let (input_gate, output_gate) = match load {
            Load::Circuit => (BaseGate::Input(ScalarDim), BaseGate::Output(ScalarDim)),
            Load::SubCircuit => (
                BaseGate::SubCircuitInput(ScalarDim),
                BaseGate::SubCircuitOutput(ScalarDim),
            ),
        };
        for mapping in &mut wire_mapping[0..total_input_wires] {
            let added_id = circuit.add_gate(input_gate.clone().into());
            *mapping = added_id;
        }
        for gate in &bristol.gates {
            let gate_data = gate.get_data();
            let added_id = circuit.add_gate(gate.into());
            for out_wire in &gate_data.output_wires {
                match wire_mapping.get_mut(*out_wire) {
                    None => return Err(CircuitError::ConversionError),
                    Some(mapped) => *mapped = added_id,
                }
            }
            for input_wire in &gate_data.input_wires {
                let mapped_input = wire_mapping
                    .get(*input_wire)
                    .ok_or(CircuitError::ConversionError)?;
                circuit.add_wire(*mapped_input, added_id);
            }
        }
        let output_gates =
            bristol.header.wires - bristol.total_output_wires()..bristol.header.wires;
        for output_id in &wire_mapping[output_gates] {
            let added_id = circuit.add_gate(output_gate.clone().into());
            circuit.add_wire(*output_id, added_id);
        }
        Ok(circuit)
    }

    pub fn load_bristol(path: impl AsRef<Path>, load: Load) -> Result<Self, CircuitError> {
        let parsed = bristol::Circuit::load(path)?;
        BaseCircuit::from_bristol(parsed, load)
    }
}

impl<G: Clone, Idx: GateIdx, W: Clone> Clone for BaseCircuit<G, Idx, W> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            is_main: self.is_main,
            simd_size: self.simd_size,
            interactive_gate_count: self.interactive_gate_count,
            input_gates: self.input_gates.clone(),
            output_gates: self.output_gates.clone(),
            constant_gates: self.constant_gates.clone(),
            sub_circuit_output_gates: self.sub_circuit_output_gates.clone(),
            sub_circuit_input_gates: self.sub_circuit_input_gates.clone(),
        }
    }
}

impl<G: Gate, Idx: GateIdx> Default for BaseCircuit<G, Idx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G, Idx, W> Debug for BaseCircuit<G, Idx, W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseCircuit")
            .field("input_count", &self.input_count())
            .field(
                "sub_circuit_input_gates",
                &self.sub_circuit_input_gates().len(),
            )
            .field("interactive_count", &self.interactive_count())
            .field("output_count", &self.output_count())
            .field(
                "sub_circuit_output_gates",
                &self.sub_circuit_output_gates().len(),
            )
            .field("constant_gates", &self.constant_gates.len())
            .field("simd_size", &self.simd_size)
            .finish()
    }
}

impl<T: Share, D: Dimension> BaseGate<T, D> {
    pub(crate) fn evaluate_sc_input_simd(
        &self,
        inputs: impl Iterator<Item = <Self as Gate>::Share>,
    ) -> <<Self as Gate>::Share as Share>::SimdShare {
        let Self::SubCircuitInput(_) = self else {
            panic!("Called evaluate_sc_input_simd on wrong gate {self:?}");
        };
        inputs.collect()
    }

    pub(crate) fn evaluate_connect_to_main_simd(
        &self,
        input: &<<Self as Gate>::Share as Share>::SimdShare,
    ) -> <Self as Gate>::Share {
        let Self::ConnectToMainFromSimd((_, at)) = self else {
            panic!("Called evaluate_connect_to_main_simd on wrong gate {self:?}");
        };
        input.get(*at as usize)
    }
}

impl<T: Share, D: Dimension> Gate for BaseGate<T, D> {
    type Share = T;
    type DimTy = D;

    fn is_interactive(&self) -> bool {
        false
    }

    fn input_size(&self) -> usize {
        1
    }

    fn as_base_gate(&self) -> Option<&BaseGate<Self::Share, D>> {
        Some(self)
    }

    fn wrap_base_gate(base_gate: BaseGate<Self::Share, Self::DimTy>) -> Self {
        base_gate
    }

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = Self::Share>,
    ) -> Self::Share {
        match self {
            Self::Constant(constant) => {
                if party_id == 0 {
                    constant.clone()
                } else {
                    constant.zero()
                }
            }
            Self::Output(_)
            | Self::Input(_)
            | Self::SubCircuitInput(_)
            | Self::SubCircuitOutput(_)
            | Self::ConnectToMain(_) => inputs
                .next()
                .unwrap_or_else(|| panic!("Empty input for {self:?}")),
            Self::ConnectToMainFromSimd(_) => {
                panic!("BaseGate::evaluate_non_interactive called on SIMD gates")
            }
            Self::Debug => {
                let inp = inputs.next().expect("Empty input");
                debug!("BaseGate::Debug party_id={party_id}: {inp:?}");
                inp
            }
        }
    }

    fn evaluate_non_interactive_simd<'e>(
        &self,
        _party_id: usize,
        mut inputs: impl Iterator<Item = &'e <Self::Share as Share>::SimdShare>,
    ) -> <Self::Share as Share>::SimdShare {
        match self {
            BaseGate::Output(_)
            | BaseGate::Input(_)
            | BaseGate::ConnectToMain(_)
            | BaseGate::SubCircuitInput(_)
            | BaseGate::ConnectToMainFromSimd(_) => {
                inputs.next().expect("Missing input to {self:?}").clone()
            }
            BaseGate::SubCircuitOutput(_) => {
                inputs.next().expect("Missing input to {self:?}").clone()
            }
            BaseGate::Constant(_constant) => {
                todo!("SimdShare from constant")
            }
            BaseGate::Debug => {
                todo!("Debug SIMD gate not impld")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BaseLayerIter<'a, G, Idx: GateIdx, W> {
    circuit: &'a BaseCircuit<G, Idx, W>,
    inputs_needed_cnt: Vec<u32>,
    to_visit: VecDeque<NodeIndex<Idx>>,
    next_layer: VecDeque<NodeIndex<Idx>>,
    visited: <CircuitGraph<G, Idx, W> as Visitable>::Map,
    added_to_next: <CircuitGraph<G, Idx, W> as Visitable>::Map,
    // only used for SIMD circuits
    inputs_left_to_provide: HashMap<NodeIndex<Idx>, u32>,
    // (non_interactive, interactive)
    last_layer_size: (usize, usize),
    gates_produced: usize,
}

impl<'a, Idx: GateIdx, G: Gate, W: Wire> BaseLayerIter<'a, G, Idx, W> {
    pub fn new(circuit: &'a BaseCircuit<G, Idx, W>) -> Self {
        let mut uninit = Self::new_uninit(circuit);

        uninit.next_layer.extend(
            circuit
                .input_gates
                .iter()
                .copied()
                .map(Into::<NodeIndex<Idx>>::into),
        );
        uninit
    }

    pub fn new_uninit(circuit: &'a BaseCircuit<G, Idx, W>) -> Self {
        let inputs_needed_cnt = circuit
            .as_graph()
            .node_identifiers()
            .map(|idx| {
                circuit
                    .graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .count()
                    .try_into()
                    .expect("u32::MAX is max input for gate")
            })
            .collect();
        let to_visit = VecDeque::new();
        let next_layer = circuit.constant_gates.iter().map(|&g| g.into()).collect();
        let visited = circuit.graph.visit_map();
        let added_to_next = circuit.graph.visit_map();
        Self {
            circuit,
            inputs_needed_cnt,
            to_visit,
            next_layer,
            visited,
            added_to_next,
            inputs_left_to_provide: Default::default(),
            last_layer_size: (0, 0),
            gates_produced: 0,
        }
    }

    pub fn visit_sc_inputs(mut self) -> Self {
        self.next_layer.extend(
            self.circuit
                .sub_circuit_input_gates
                .iter()
                .copied()
                .map(Into::<NodeIndex<Idx>>::into),
        );
        self
    }

    pub fn add_to_visit(&mut self, idx: NodeIndex<Idx>) {
        self.to_visit.push_back(idx);
    }

    /// Adds idx to the next layer if it has not been visited
    pub fn add_to_next_layer(&mut self, idx: NodeIndex<Idx>) {
        if !self.added_to_next.is_visited(&idx) {
            self.next_layer.push_back(idx);
            self.added_to_next.visit(idx);
        }
    }

    pub fn is_exhausted(&self) -> bool {
        self.gates_produced == self.circuit.gate_count()
    }

    pub fn swap_next_layer(&mut self) {
        std::mem::swap(&mut self.to_visit, &mut self.next_layer);
    }

    pub fn process_to_visit(&mut self) -> Option<CircuitLayer<G, Idx>> {
        // TODO this current implementation is confusing -> Refactor
        let graph = self.circuit.as_graph();
        let mut layer = CircuitLayer::with_capacity(self.last_layer_size);

        while let Some(node_idx) = self.to_visit.pop_front() {
            // This case handles the interactive gates at the front of to_visit that
            // are here because they were `add_to_next_layer` but whose neighbours have not
            // had their counts decreased
            if self.visited.is_visited(&node_idx) {
                let mut neigh_cnt = 0;
                for neigh in graph.neighbors(node_idx) {
                    neigh_cnt += 1;
                    {
                        let count = self.inputs_needed_cnt[neigh.index()];
                        trace!("Node: {node_idx:?} -> Neigh {neigh:?}: count {count}")
                    }
                    self.inputs_needed_cnt[neigh.index()] -= 1;
                    let inputs_needed = self.inputs_needed_cnt[neigh.index()];
                    if inputs_needed == 0 {
                        self.add_to_visit(neigh);
                    }
                }
                if self.circuit.is_simd() {
                    self.inputs_left_to_provide
                        .entry(node_idx)
                        .or_insert(neigh_cnt);
                }
                continue;
            }
            self.visited.visit(node_idx);

            if self.circuit.is_simd() {
                for neigh in graph.neighbors_directed(node_idx, Direction::Incoming) {
                    let cnt = self
                        .inputs_left_to_provide
                        .get_mut(&neigh)
                        .expect("inputs_left_to_provide must be initialize");
                    *cnt -= 1;
                    if *cnt == 0 {
                        layer.freeable_gates.push(neigh.into());
                    }
                }
            }

            let gate = graph[node_idx].clone();
            if gate.is_interactive() {
                self.add_to_next_layer(node_idx);
                layer.push_interactive((gate.clone(), node_idx.into()));
            } else {
                layer.push_non_interactive((gate.clone(), node_idx.into()));
                let mut neigh_cnt = 0;
                for neigh in graph.neighbors(node_idx) {
                    neigh_cnt += 1;
                    self.inputs_needed_cnt[neigh.index()] -= 1;
                    let inputs_needed = self.inputs_needed_cnt[neigh.index()];
                    if inputs_needed == 0 {
                        self.add_to_visit(neigh)
                    }
                }
                if self.circuit.is_simd() {
                    self.inputs_left_to_provide
                        .entry(node_idx)
                        .or_insert(neigh_cnt);
                }
            }
        }
        if layer.is_empty() {
            None
        } else {
            self.gates_produced += layer.interactive_len() + layer.non_interactive_len();
            Some(layer)
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CircuitLayer<G, Idx> {
    pub(crate) non_interactive_gates: Vec<G>,
    pub(crate) non_interactive_ids: Vec<GateId<Idx>>,
    pub(crate) interactive_gates: Vec<G>,
    pub(crate) interactive_ids: Vec<GateId<Idx>>,
    /// SIMD Gates that can be freed after this layer
    pub(crate) freeable_gates: Vec<GateId<Idx>>, // TODO add output gates here so that the CircuitLayerIter::next doesn't need to iterate
                                                 //  over all potential outs
}

impl<G, Idx> CircuitLayer<G, Idx> {
    fn with_capacity((non_interactive, interactive): (usize, usize)) -> Self {
        Self {
            non_interactive_gates: Vec::with_capacity(non_interactive),
            non_interactive_ids: Vec::with_capacity(non_interactive),
            interactive_gates: Vec::with_capacity(interactive),
            interactive_ids: Vec::with_capacity(interactive),
            freeable_gates: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.non_interactive_gates.is_empty() && self.interactive_gates.is_empty()
    }

    fn push_interactive(&mut self, (gate, id): (G, GateId<Idx>)) {
        self.interactive_gates.push(gate);
        self.interactive_ids.push(id);
    }

    fn push_non_interactive(&mut self, (gate, id): (G, GateId<Idx>)) {
        self.non_interactive_gates.push(gate);
        self.non_interactive_ids.push(id);
    }

    pub fn interactive_len(&self) -> usize {
        self.interactive_gates.len()
    }

    pub fn non_interactive_len(&self) -> usize {
        self.non_interactive_gates.len()
    }
}

impl<G: Clone, Idx: GateIdx> CircuitLayer<G, Idx> {
    /// If idx is < self.non_interactive_gates.len(), returns an non-interactive gate,
    /// otherwise an interactive one. Panics if out of bound. Intended for usage with an
    /// enumerated self.iter_ids call
    pub(crate) fn get_gate(&self, idx: usize) -> &G {
        let ni_len = self.non_interactive_len();
        if idx < ni_len {
            &self.non_interactive_gates[idx]
        } else {
            &self.interactive_gates[idx - ni_len]
        }
    }

    pub(crate) fn iter_ids(&self) -> impl Iterator<Item = GateId<Idx>> + '_ {
        self.non_interactive_ids
            .iter()
            .chain(&self.interactive_ids)
            .copied()
    }

    pub(crate) fn into_interactive_iter(self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone {
        self.interactive_gates.into_iter().zip(self.interactive_ids)
    }

    #[allow(unused)]
    pub(crate) fn into_non_interactive_iter(
        self,
    ) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone {
        self.non_interactive_gates
            .into_iter()
            .zip(self.non_interactive_ids)
    }

    pub(crate) fn interactive_iter(&self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone + '_ {
        self.interactive_gates
            .clone()
            .into_iter()
            .zip(self.interactive_ids.clone())
    }

    pub(crate) fn non_interactive_iter(
        &self,
    ) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone + '_ {
        self.non_interactive_gates
            .clone()
            .into_iter()
            .zip(self.non_interactive_ids.clone())
    }
}

impl<G: Clone, Idx: GateIdx> CircuitLayer<G, Idx> {
    pub(crate) fn into_iter(self) -> impl Iterator<Item = (G, GateId<Idx>)> + Clone {
        let ni = self
            .non_interactive_gates
            .into_iter()
            .zip(self.non_interactive_ids);
        let i = self.interactive_gates.into_iter().zip(self.interactive_ids);
        ni.chain(i)
    }

    pub(crate) fn into_sc_iter(
        self,
        sc_id: CircuitId,
    ) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone {
        self.into_iter()
            .map(move |(g, gate_id)| (g, SubCircuitGate::new(sc_id, gate_id)))
    }
}

impl<'a, G: Gate, Idx: GateIdx, W: Wire> Iterator for BaseLayerIter<'a, G, Idx, W> {
    type Item = CircuitLayer<G, Idx>;

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn next(&mut self) -> Option<Self::Item> {
        self.swap_next_layer();
        self.process_to_visit()
    }
}

impl<G: Gate, Idx: GateIdx, W: Wire> LayerIterable for BaseCircuit<G, Idx, W> {
    type Layer = CircuitLayer<G, Idx>;
    type LayerIter<'this> = BaseLayerIter<'this, G, Idx, W> where Self: 'this;

    fn layer_iter(&self) -> Self::LayerIter<'_> {
        BaseLayerIter::new(self)
    }
}

impl<G, Idx: GateIdx> Default for CircuitLayer<G, Idx> {
    fn default() -> Self {
        Self {
            non_interactive_gates: vec![],
            non_interactive_ids: vec![],
            interactive_gates: vec![],
            interactive_ids: vec![],
            freeable_gates: vec![],
        }
    }
}

impl<Idx: GateIdx> GateId<Idx> {
    pub fn as_usize(&self) -> usize {
        self.0.index()
    }
}

impl<Idx> From<NodeIndex<Idx>> for GateId<Idx>
where
    Idx: GateIdx,
{
    fn from(idx: NodeIndex<Idx>) -> Self {
        Self(
            idx.index()
                .try_into()
                .map_err(|_| ())
                .expect("idx must fit into Idx"),
        )
    }
}

impl<Idx: GateIdx> From<GateId<Idx>> for NodeIndex<Idx> {
    fn from(id: GateId<Idx>) -> Self {
        NodeIndex::from(id.0)
    }
}

impl<Idx> From<u16> for GateId<Idx>
where
    Idx: TryFrom<u16>,
{
    fn from(val: u16) -> Self {
        GateId(
            val.try_into()
                .map_err(|_| ())
                .expect("val must fit into Idx"),
        )
    }
}

impl<Idx> From<u32> for GateId<Idx>
where
    Idx: TryFrom<u32>,
{
    fn from(val: u32) -> Self {
        GateId(
            val.try_into()
                .map_err(|_| ())
                .expect("val must fit into Idx"),
        )
    }
}

impl<Idx> From<usize> for GateId<Idx>
where
    Idx: TryFrom<usize>,
{
    fn from(val: usize) -> Self {
        GateId(
            val.try_into()
                .map_err(|_| ())
                .expect("val must fit into Idx"),
        )
    }
}

impl<Idx: GateIdx> Display for GateId<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(itoa::Buffer::new().format(self.0.index()))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, mem};

    use crate::bristol;
    use crate::circuit::base_circuit::{BaseGate, BaseLayerIter, CircuitLayer, Load};
    use crate::circuit::{BaseCircuit, GateId};

    use crate::protocols::boolean_gmw::BooleanGate;
    use crate::protocols::ScalarDim;

    #[test]
    fn gate_size() {
        // Assert that the gate size stays at 8 bytes (might change in the future)
        assert_eq!(8, mem::size_of::<BooleanGate>());
    }

    #[test]
    fn circuit_layer_iter() {
        let mut circuit: BaseCircuit = BaseCircuit::new();
        let inp = || BooleanGate::Base(BaseGate::Input(ScalarDim));
        let and = || BooleanGate::And;
        let xor = || BooleanGate::Xor;
        let out = || BooleanGate::Base(BaseGate::Output(ScalarDim));
        let in_1 = circuit.add_gate(inp());
        let in_2 = circuit.add_gate(inp());
        let out_1 = circuit.add_gate(out());
        let and_1 = circuit.add_wired_gate(and(), &[in_1, in_2]);
        let in_3 = circuit.add_gate(inp());
        let xor_1 = circuit.add_wired_gate(xor(), &[in_3, and_1]);
        let and_2 = circuit.add_wired_gate(and(), &[and_1, xor_1]);
        circuit.add_wire(and_2, out_1);

        let mut cl_iter = BaseLayerIter::new(&circuit);

        let first_layer = CircuitLayer {
            non_interactive_gates: vec![inp(), inp(), inp()],
            non_interactive_ids: vec![0_u32.into(), 1_u32.into(), 4_u32.into()],
            interactive_gates: vec![BooleanGate::And],
            interactive_ids: vec![3_u32.into()],
            freeable_gates: vec![],
        };

        let snd_layer = CircuitLayer {
            non_interactive_gates: vec![xor()],
            non_interactive_ids: vec![5_u32.into()],
            interactive_gates: vec![BooleanGate::And],
            interactive_ids: vec![6_u32.into()],
            freeable_gates: vec![],
        };

        let third_layer = CircuitLayer {
            non_interactive_gates: vec![out()],
            non_interactive_ids: vec![2_u32.into()],
            interactive_gates: vec![],
            interactive_ids: vec![],
            freeable_gates: vec![],
        };

        assert_eq!(Some(first_layer), cl_iter.next());
        assert_eq!(Some(snd_layer), cl_iter.next());
        assert_eq!(Some(third_layer), cl_iter.next());
        assert_eq!(None, cl_iter.next());
    }

    #[test]
    fn parent_gates() {
        let mut circuit: BaseCircuit = BaseCircuit::new();
        let from_0 = circuit.add_gate(BooleanGate::Base(BaseGate::Input(ScalarDim)));
        let from_1 = circuit.add_gate(BooleanGate::Base(BaseGate::Input(ScalarDim)));
        let to = circuit.add_gate(BooleanGate::And);
        circuit.add_wire(from_0, to);
        circuit.add_wire(from_1, to);
        assert_eq!(
            vec![from_1, from_0],
            circuit.parent_gates(to).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore]
    fn big_circuit() {
        // create a circuit with 10_000 layer of 10_000 nodes each (100_000_000 nodes)
        // this test currently allocates 3.5 GB of memory for a graph idx type of u32
        let mut circuit: BaseCircuit = BaseCircuit::with_capacity(100_000_000, 100_000_000);
        for i in 0_u32..10_000 {
            for j in 0_u32..10_000 {
                if i == 0 {
                    circuit.add_gate(BooleanGate::Base(BaseGate::Input(ScalarDim)));
                    continue;
                }
                let to_id = circuit.add_gate(BooleanGate::And);
                let from_id = (i - 1) * 10_000 + j;
                circuit.add_wire(GateId::from(from_id), to_id);
            }
        }
    }

    #[test]
    fn convert_bristol_aes_circuit() {
        let aes_text =
            fs::read_to_string("test_resources/bristol-circuits/AES-non-expanded.txt").unwrap();
        let parsed = bristol::circuit(&aes_text).unwrap();
        let converted: BaseCircuit<BooleanGate, u32> =
            BaseCircuit::from_bristol(parsed.clone(), Load::Circuit).unwrap();
        assert_eq!(
            parsed.header.gates + converted.input_count() + converted.output_count(),
            converted.gate_count()
        );
        assert_eq!(converted.input_count(), parsed.total_input_wires());
        assert_eq!(parsed.total_output_wires(), converted.output_count());
        // TODO comparing the wire counts is a little tricky since we have a slightly different
        //  view of what a wire is
        //  assert_eq!(parsed.header.wires, converted.wire_count());
    }
}
