use std::collections::{HashSet, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::path::Path;
use std::{fs, ops};

use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{IntoNeighbors, Reversed};
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, Direction, Graph};
use tracing::{debug, info, trace};

use crate::bristol;
use crate::circuit::{DefaultIdx, GateIdx, LayerIterable};
use crate::errors::CircuitError;

type CircuitGraph<Idx> = Graph<Gate, Wire, Directed, Idx>;

pub struct BaseCircuit<Idx = u32> {
    graph: CircuitGraph<Idx>,
    and_count: usize,
    input_gates: Vec<GateId<Idx>>,
    output_gates: Vec<GateId<Idx>>,
    constant_gates: Vec<GateId<Idx>>,
    // TODO I don't think this field is needed anymore
    sub_circuit_output_gates: Vec<GateId<Idx>>,
    pub(crate) sub_circuit_input_gates: Vec<GateId<Idx>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Gate {
    And,
    Xor,
    Inv,
    Output,
    Input,
    // Input from a sub circuit called within a circuit.
    SubCircuitInput,
    // Output from this circuit into another sub circuit
    SubCircuitOutput,
    Constant(bool),
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct GateId<Idx = DefaultIdx>(pub(crate) Idx);

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Wire;

impl<Idx: GateIdx> BaseCircuit<Idx> {
    pub fn new() -> Self {
        Self {
            graph: Default::default(),
            and_count: 0,
            input_gates: vec![],
            output_gates: vec![],
            constant_gates: vec![],
            sub_circuit_output_gates: vec![],
            sub_circuit_input_gates: vec![],
        }
    }

    pub fn with_capacity(gates: usize, wires: usize) -> Self {
        let mut new = Self::new();
        new.graph = Graph::with_capacity(gates, wires);
        new
    }

    #[tracing::instrument(level = "trace", skip(self))]
    pub fn add_gate(&mut self, gate: Gate) -> GateId<Idx> {
        let gate_id = self.graph.add_node(gate).into();
        match gate {
            Gate::Input => self.input_gates.push(gate_id),
            Gate::Constant(_) => self.constant_gates.push(gate_id),
            Gate::Output => self.output_gates.push(gate_id),
            Gate::SubCircuitOutput => self.sub_circuit_output_gates.push(gate_id),
            Gate::SubCircuitInput => self.sub_circuit_input_gates.push(gate_id),
            Gate::And => self.and_count += 1,
            _ => (),
        }
        trace!(%gate_id, "Added gate");
        gate_id
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn add_sc_input_gate(&mut self, gate: Gate) -> GateId<Idx> {
        let gate_id = self.graph.add_node(gate).into();
        match gate {
            Gate::Input => self.input_gates.push(gate_id),
            Gate::Constant(_) => self.constant_gates.push(gate_id),
            Gate::Output => self.output_gates.push(gate_id),
            Gate::SubCircuitOutput => self.sub_circuit_output_gates.push(gate_id),
            Gate::And => self.and_count += 1,
            _ => (),
        }
        self.sub_circuit_input_gates.push(gate_id);
        debug!(%gate_id, ?gate, "Added sub circuit input gate");
        gate_id
    }

    #[tracing::instrument(level="trace", skip(self), fields(%from, %to))]
    pub fn add_wire(&mut self, from: GateId<Idx>, to: GateId<Idx>) {
        self.graph.add_edge(from.into(), to.into(), Wire);
        trace!("Added wire");
    }

    pub fn add_wired_gate(&mut self, gate: Gate, from: &[GateId<Idx>]) -> GateId<Idx> {
        let added = self.add_gate(gate);
        for from_id in from {
            self.add_wire(*from_id, added);
        }
        added
    }

    pub fn get_gate(&self, id: impl Into<GateId<Idx>>) -> Gate {
        let id = id.into();
        self.graph[NodeIndex::from(id)]
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

    pub fn as_graph(&self) -> &CircuitGraph<Idx> {
        &self.graph
    }
}

impl<Idx> BaseCircuit<Idx> {
    pub fn and_count(&self) -> usize {
        self.and_count
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

    pub fn output_count(&self) -> usize {
        self.output_gates.len()
    }

    pub fn output_gates(&self) -> &[GateId<Idx>] {
        &self.output_gates
    }

    pub fn sub_circuit_output_gates(&self) -> &[GateId<Idx>] {
        &self.sub_circuit_output_gates
    }
}

impl BaseCircuit<usize> {
    #[tracing::instrument(skip(bristol))]
    pub fn from_bristol(bristol: bristol::Circuit) -> Result<Self, CircuitError> {
        info!(
            "Converting bristol circuit with header: {:?}",
            bristol.header
        );
        let mut circuit = Self::with_capacity(bristol.header.gates, bristol.header.wires);
        let total_input_wires = bristol.header.input_wires.iter().sum();
        // We treat the output wires of the bristol::Gates as their GateIds. Unfortunately,
        // the output wires are not given in ascending order, so we need to save a mapping
        // of wire ids to GateIds
        let mut wire_mapping = vec![GateId::from(0_u32); bristol.header.wires];
        for mapping in &mut wire_mapping[0..total_input_wires] {
            let added_id = circuit.add_gate(Gate::Input);
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
        let output_gates = bristol.header.wires - bristol.header.output_wires..bristol.header.wires;
        for output_id in &wire_mapping[output_gates] {
            let added_id = circuit.add_gate(Gate::Output);
            circuit.add_wire(*output_id, added_id);
        }
        Ok(circuit)
    }

    pub fn load_bristol(path: impl AsRef<Path>) -> Result<Self, CircuitError> {
        let parsed = bristol::Circuit::load(path)?;
        BaseCircuit::from_bristol(parsed)
    }
}

impl<Idx: GateIdx> Clone for BaseCircuit<Idx> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            and_count: self.and_count,
            input_gates: self.input_gates.clone(),
            output_gates: self.output_gates.clone(),
            constant_gates: self.constant_gates.clone(),
            sub_circuit_output_gates: self.sub_circuit_output_gates.clone(),
            sub_circuit_input_gates: self.sub_circuit_input_gates.clone(),
        }
    }
}

impl<Idx: GateIdx> Default for BaseCircuit<Idx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Idx> Debug for BaseCircuit<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit")
            .field("input_count", &self.input_count())
            .field(
                "sub_circuit_input_gates",
                &self.sub_circuit_input_gates().len(),
            )
            .field("and_count", &self.and_count())
            .field("output_count", &self.output_count())
            .field(
                "sub_circuit_output_gates",
                &self.sub_circuit_output_gates().len(),
            )
            .finish()
    }
}

impl Gate {
    pub fn input_count(&self) -> usize {
        match self {
            Gate::Input | Gate::Constant(_) => 0,
            Gate::Inv | Gate::Output | Gate::SubCircuitInput | Gate::SubCircuitOutput => 1,
            Gate::And | Gate::Xor => 2,
        }
    }

    // TODO maybe have more generic evaluation context?
    pub(crate) fn evaluate_non_interactive(
        &self,
        mut input: impl Iterator<Item = bool>,
        party_id: usize,
    ) -> bool {
        match self {
            Gate::And => panic!("Called evaluate_non_interactive on Gate::AND"),
            Gate::Xor => input.fold(false, ops::BitXor::bitxor),
            Gate::Inv => {
                let inp = input.next().expect("Empty input");
                assert!(
                    input.next().is_none(),
                    "Gate::Output, Gate::Input and Gate::Inv must have single input"
                );
                if party_id == 0 {
                    !inp
                } else {
                    inp
                }
            }
            &Gate::Constant(constant) => {
                if party_id == 0 {
                    constant
                } else {
                    !constant
                }
            }
            Gate::Output | Gate::Input | Gate::SubCircuitInput | Gate::SubCircuitOutput => {
                let inp = input
                    .next()
                    .unwrap_or_else(|| panic!("Empty input for {self:?}"));
                assert!(
                    input.next().is_none(),
                    "Gate::Output, Gate::Input and Gate::Inv must have single input"
                );
                inp
            }
        }
    }
}

impl From<&bristol::Gate> for Gate {
    fn from(gate: &bristol::Gate) -> Self {
        match gate {
            bristol::Gate::And(_) => Gate::And,
            bristol::Gate::Xor(_) => Gate::Xor,
            bristol::Gate::Inv(_) => Gate::Inv,
        }
    }
}

#[derive(Debug)]
pub struct BaseLayerIter<'a, Idx: GateIdx> {
    circuit: &'a BaseCircuit<Idx>,
    to_visit: VecDeque<NodeIndex<Idx>>,
    next_layer: VecDeque<NodeIndex<Idx>>,
    visited: <CircuitGraph<Idx> as Visitable>::Map,
}

impl<'a, Idx: GateIdx> BaseLayerIter<'a, Idx> {
    pub fn new(circuit: &'a BaseCircuit<Idx>) -> Self {
        let mut uninit = Self::new_uninit(circuit);
        uninit.next_layer.extend(
            circuit
                .input_gates
                .iter()
                .chain(&circuit.constant_gates)
                .copied()
                .map(Into::<NodeIndex<Idx>>::into),
        );
        uninit
    }

    pub fn new_uninit(circuit: &'a BaseCircuit<Idx>) -> Self {
        let to_visit = VecDeque::new();
        let next_layer = VecDeque::new();
        let visited = circuit.graph.visit_map();
        Self {
            circuit,
            to_visit,
            next_layer,
            visited,
        }
    }

    pub fn add_to_visit(&mut self, idx: NodeIndex<Idx>) {
        self.to_visit.push_back(idx);
    }

    pub fn add_to_next_layer(&mut self, idx: NodeIndex<Idx>) {
        self.next_layer.push_back(idx);
    }
}

#[derive(Default, Debug, Eq, PartialEq, Clone)]
pub struct CircuitLayer<Idx: Hash + PartialEq + Eq> {
    // TODO alignment of the tuple? Check this and maybe split it into two vecs
    pub(crate) non_interactive: Vec<(Gate, GateId<Idx>)>,
    pub(crate) and_gates: Vec<GateId<Idx>>,
}

impl<Idx: GateIdx> CircuitLayer<Idx> {
    fn new() -> Self {
        Default::default()
    }

    fn add_non_interactive(&mut self, (gate, gate_id): (Gate, GateId<Idx>)) {
        match gate {
            Gate::And => {
                panic!("Called add_non_interactive() on And gate")
            }
            non_interactive => self.non_interactive.push((non_interactive, gate_id)),
        }
    }

    fn is_empty(&self) -> bool {
        self.non_interactive.is_empty() && self.and_gates.is_empty()
    }
}

impl<'a, Idx: GateIdx> Iterator for BaseLayerIter<'a, Idx> {
    type Item = CircuitLayer<Idx>;

    #[tracing::instrument(level = "trace", skip(self), ret)]
    fn next(&mut self) -> Option<Self::Item> {
        // TODO clean this method up
        let graph = self.circuit.as_graph();
        let mut layer = CircuitLayer::new();
        let mut and_gates: HashSet<GateId<Idx>> = HashSet::new();
        std::mem::swap(&mut self.to_visit, &mut self.next_layer);
        while let Some(node_idx) = self.to_visit.pop_front() {
            if self.visited.is_visited(&node_idx) {
                continue;
            }
            let gate = &graph[node_idx];
            match gate {
                Gate::And => {
                    and_gates.insert(node_idx.into());
                    self.next_layer
                        .extend(graph.neighbors(node_idx).filter(|neigh| {
                            Reversed(graph).neighbors(*neigh).all(|b| {
                                self.visited.is_visited(&b) || and_gates.contains(&b.into())
                            })
                        }));
                }
                _non_interactive => {
                    self.visited.visit(node_idx);
                    layer.add_non_interactive((*gate, node_idx.into()));
                    for neigh in graph.neighbors(node_idx) {
                        if Reversed(graph)
                            .neighbors(neigh)
                            .all(|b| self.visited.is_visited(&b))
                        {
                            self.to_visit.push_back(neigh);
                        } else if Reversed(graph)
                            .neighbors(neigh)
                            .all(|b| self.visited.is_visited(&b) || and_gates.contains(&b.into()))
                        {
                            self.next_layer.push_back(neigh);
                        }
                    }
                }
            }
        }
        for and_id in &and_gates {
            self.visited.visit(and_id.0);
        }
        layer.and_gates = and_gates.into_iter().collect();
        layer.and_gates.sort_unstable();
        if layer.is_empty() {
            None
        } else {
            Some(layer)
        }
    }
}

impl<Idx: GateIdx> LayerIterable for BaseCircuit<Idx> {
    type Layer = CircuitLayer<Idx>;
    type LayerIter<'this> = BaseLayerIter<'this, Idx> where Self: 'this;

    fn layer_iter(&self) -> Self::LayerIter<'_> {
        BaseLayerIter::new(self)
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
    use crate::circuit::base_circuit::{BaseLayerIter, CircuitLayer};
    use crate::circuit::{BaseCircuit, Gate, GateId};

    #[test]
    fn gate_size() {
        // Assert that the gate size stays at 1 byte (might change in the future)
        assert_eq!(1, mem::size_of::<Gate>());
    }

    #[test]
    fn circuit_layer_iter() {
        let mut circuit = BaseCircuit::<u32>::new();
        let inp = || Gate::Input;
        let and = || Gate::And;
        let xor = || Gate::Xor;
        let out = || Gate::Output;
        let in_1 = circuit.add_gate(inp());
        let in_2 = circuit.add_gate(inp());
        let out_1 = circuit.add_gate(out());
        let and_1 = circuit.add_wired_gate(and(), &[in_1, in_2]);
        let in_3 = circuit.add_gate(inp());
        let xor_1 = circuit.add_wired_gate(xor(), &[in_3, and_1]);
        let and_2 = circuit.add_wired_gate(and(), &[and_1, xor_1]);
        let _ = circuit.add_wire(and_2, out_1);

        let mut cl_iter = BaseLayerIter::new(&circuit);

        let first_layer = CircuitLayer {
            non_interactive: [
                (inp(), 0_u32.into()),
                (inp(), 1_u32.into()),
                (inp(), 4_u32.into()),
            ]
            .into_iter()
            .collect(),
            and_gates: [3_u32.into()].into_iter().collect(),
        };

        let snd_layer = CircuitLayer {
            non_interactive: [(xor(), 5_u32.into())].into_iter().collect(),
            and_gates: [6_u32.into()].into_iter().collect(),
        };

        let third_layer = CircuitLayer {
            non_interactive: [(out(), 2_u32.into())].into_iter().collect(),
            and_gates: [].into_iter().collect(),
        };

        assert_eq!(Some(first_layer), cl_iter.next());
        assert_eq!(Some(snd_layer), cl_iter.next());
        assert_eq!(Some(third_layer), cl_iter.next());
        assert_eq!(None, cl_iter.next());
    }

    #[test]
    fn parent_gates() {
        let mut circuit = BaseCircuit::<u32>::new();
        let from_0 = circuit.add_gate(Gate::Input);
        let from_1 = circuit.add_gate(Gate::Input);
        let to = circuit.add_gate(Gate::And);
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
        let mut circuit = BaseCircuit::<u32>::with_capacity(100_000_000, 100_000_000);
        for i in 0_u32..10_000 {
            for j in 0_u32..10_000 {
                if i == 0 {
                    circuit.add_gate(Gate::Input);
                    continue;
                }
                let to_id = circuit.add_gate(Gate::And);
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
        let converted = BaseCircuit::from_bristol(parsed.clone()).unwrap();
        assert_eq!(
            parsed.header.gates + converted.input_count() + converted.output_count(),
            converted.gate_count()
        );
        assert_eq!(
            converted.input_count(),
            parsed.header.input_wires.iter().sum::<usize>()
        );
        assert_eq!(parsed.header.output_wires, converted.output_count());
        // TODO comparing the wire counts is a little tricky since we have a slightly different
        //  view of what a wire is
        //  assert_eq!(parsed.header.wires, converted.wire_count());
    }
}
