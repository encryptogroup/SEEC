use crate::errors::CircuitError;

use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, Reversed};
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, Direction, Graph};

use crate::bristol;
use petgraph::adj::IndexType;
use std::collections::{HashSet, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::path::Path;
use std::{fs, ops};
use tracing::{debug, info};

type CircuitGraph<Idx> = Graph<Gate, Wire, Directed, Idx>;
type DefaultIdx = u32;

pub struct Circuit<Idx = u32> {
    pub(crate) graph: CircuitGraph<Idx>,
    pub(crate) input_count: usize,
    pub(crate) and_count: usize,
    pub(crate) output_count: usize,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Gate {
    And,
    Xor,
    Inv,
    Output,
    Input,
}

#[derive(Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Debug)]
pub struct GateId<Idx = DefaultIdx>(NodeIndex<Idx>);

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Wire;

impl<Idx: IndexType> Circuit<Idx> {
    pub fn new() -> Self {
        Self {
            graph: Default::default(),
            input_count: 0,
            and_count: 0,
            output_count: 0,
        }
    }

    pub fn with_capacity(gates: usize, wires: usize) -> Self {
        let graph = Graph::with_capacity(gates, wires);
        Self {
            graph,
            input_count: 0,
            and_count: 0,
            output_count: 0,
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn add_gate(&mut self, gate: Gate) -> GateId<Idx> {
        match &gate {
            Gate::And => self.and_count += 1,
            Gate::Output => self.output_count += 1,
            Gate::Input => self.input_count += 1,
            _ => (),
        }
        let gate_id = self.graph.add_node(gate).into();
        debug!(%gate_id, "Added gate");
        gate_id
    }

    #[tracing::instrument(skip(self), fields(%from, %to))]
    pub fn add_wire(&mut self, from: GateId<Idx>, to: GateId<Idx>) {
        self.graph.add_edge(from.0, to.0, Wire);
        debug!("Added wire");
    }

    pub fn add_wired_gate(&mut self, gate: Gate, from: &[GateId<Idx>]) -> GateId<Idx> {
        let added = self.add_gate(gate);
        for from_id in from {
            self.add_wire(*from_id, added);
        }
        added
    }

    pub fn and_count(&self) -> usize {
        self.and_count
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    pub fn gate_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn wire_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn get_gate(&self, id: impl Into<GateId<Idx>>) -> &Gate {
        let id = id.into().0;
        &self.graph[id]
    }

    pub fn parent_gates(
        &self,
        id: impl Into<GateId<Idx>>,
    ) -> impl Iterator<Item = GateId<Idx>> + '_ {
        self.graph
            .neighbors_directed(id.into().0, Direction::Incoming)
            .map(GateId::from)
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

    pub(crate) fn as_graph(&self) -> &CircuitGraph<Idx> {
        &self.graph
    }
}

impl Circuit<usize> {
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
        let mut wire_mapping = vec![GateId::from(0); bristol.header.wires];
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
        let bristol_text = fs::read_to_string(path)?;
        let parsed = bristol::circuit(&bristol_text).map_err(|err| err.to_owned())?;
        Circuit::from_bristol(parsed)
    }
}

impl<Idx: IndexType> Clone for Circuit<Idx> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            input_count: self.input_count,
            and_count: self.and_count,
            output_count: self.output_count,
        }
    }
}

impl<Idx: IndexType> Default for Circuit<Idx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Idx: IndexType> Debug for Circuit<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit")
            .field("input_count", &self.input_count)
            .field("and_count", &self.and_count)
            .field("output_count", &self.output_count)
            .finish()
    }
}

impl Gate {
    // TODO mauby have more generic evaluation context?
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
                debug_assert!(
                    input.next().is_none(),
                    "Gate::Output, Gate::Input and Gate::Inv must have single input"
                );
                if party_id == 0 {
                    !inp
                } else {
                    inp
                }
            }
            Gate::Output | Gate::Input => {
                let inp = input.next().expect("Empty input");
                debug_assert!(
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

pub struct CircuitLayerIter<'a, Idx: IndexType> {
    graph: &'a CircuitGraph<Idx>,
    to_visit: VecDeque<NodeIndex<Idx>>,
    next_layer: VecDeque<NodeIndex<Idx>>,
    visited: <CircuitGraph<Idx> as Visitable>::Map,
}

impl<'a, Idx: IndexType> CircuitLayerIter<'a, Idx> {
    pub fn new(circuit: &'a Circuit<Idx>) -> Self {
        // this assumes that the firs circuit.input_count nodes are the input nodes
        let graph = circuit.as_graph();
        let to_visit = VecDeque::new();
        let next_layer = graph.node_identifiers().take(circuit.input_count).collect();
        let visited = graph.visit_map();
        Self {
            graph,
            to_visit,
            next_layer,
            visited,
        }
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
pub struct CircuitLayer<Idx: Hash + PartialEq + Eq> {
    pub(crate) non_interactive: Vec<(Gate, GateId<Idx>)>,
    pub(crate) and_gates: Vec<GateId<Idx>>,
}

impl<Idx: IndexType> CircuitLayer<Idx> {
    fn new() -> Self {
        Default::default()
    }

    fn add_non_interactive(&mut self, data: (Gate, GateId<Idx>)) {
        self.non_interactive.push(data);
    }

    fn is_empty(&self) -> bool {
        self.non_interactive.is_empty() && self.and_gates.is_empty()
    }
}

impl<'a, Idx: IndexType> Iterator for CircuitLayerIter<'a, Idx> {
    type Item = CircuitLayer<Idx>;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO clean this method up
        let mut layer = CircuitLayer::new();
        let mut and_gates: HashSet<GateId<Idx>> = HashSet::new();
        std::mem::swap(&mut self.to_visit, &mut self.next_layer);
        while let Some(node_idx) = self.to_visit.pop_front() {
            if self.visited.is_visited(&node_idx) {
                continue;
            }
            let gate = &self.graph[node_idx];
            match gate {
                Gate::And => {
                    and_gates.insert(node_idx.into());
                    self.next_layer
                        .extend(self.graph.neighbors(node_idx).filter(|neigh| {
                            Reversed(self.graph).neighbors(*neigh).all(|b| {
                                self.visited.is_visited(&b) || and_gates.contains(&b.into())
                            })
                        }));
                }
                Gate::Input | Gate::Output | Gate::Xor | Gate::Inv => {
                    self.visited.visit(node_idx);
                    layer.add_non_interactive((gate.clone(), node_idx.into()));
                    for neigh in self.graph.neighbors(node_idx) {
                        if Reversed(self.graph)
                            .neighbors(neigh)
                            .all(|b| self.visited.is_visited(&b))
                        {
                            self.to_visit.push_back(neigh);
                        } else if Reversed(self.graph)
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

impl<Idx: IndexType> GateId<Idx> {
    pub fn as_usize(&self) -> usize {
        self.0.index()
    }
}

impl<Idx: IndexType> From<NodeIndex<Idx>> for GateId<Idx> {
    fn from(idx: NodeIndex<Idx>) -> Self {
        Self(idx)
    }
}

impl<Idx> From<u16> for GateId<Idx>
where
    NodeIndex<Idx>: From<u16>,
{
    fn from(val: u16) -> Self {
        GateId(val.into())
    }
}

impl<Idx> From<u32> for GateId<Idx>
where
    NodeIndex<Idx>: From<u32>,
{
    fn from(val: u32) -> Self {
        GateId(val.into())
    }
}

impl<Idx> From<usize> for GateId<Idx>
where
    NodeIndex<Idx>: From<usize>,
{
    fn from(val: usize) -> Self {
        GateId(val.into())
    }
}

impl<Idx: IndexType> Display for GateId<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(itoa::Buffer::new().format(self.0.index()))
    }
}

#[cfg(test)]
mod tests {
    use crate::bristol;
    use crate::circuit::{Circuit, CircuitLayer, CircuitLayerIter, Gate, GateId};
    use std::fs;

    #[test]
    fn circuit_layer_iter() {
        let mut circuit = Circuit::<u32>::new();
        let inp = || Gate::Input;
        let and = || Gate::And;
        let xor = || Gate::Xor;
        let out = || Gate::Output;
        let in_1 = circuit.add_gate(inp());
        let in_2 = circuit.add_gate(inp());
        let and_1 = circuit.add_gate(and());
        let xor_1 = circuit.add_gate(xor());
        let and_2 = circuit.add_gate(and());
        let out_1 = circuit.add_gate(out());
        circuit.add_wire(in_1, and_1);
        circuit.add_wire(in_2, and_1);
        circuit.add_wire(in_2, xor_1);
        circuit.add_wire(and_1, xor_1);
        circuit.add_wire(and_1, and_2);
        circuit.add_wire(xor_1, and_2);
        circuit.add_wire(and_2, out_1);

        let mut cl_iter = CircuitLayerIter::new(&circuit);

        let first_layer = CircuitLayer {
            non_interactive: [(inp(), 0.into()), (inp(), 1.into())].into_iter().collect(),
            and_gates: [2.into()].into_iter().collect(),
        };

        let snd_layer = CircuitLayer {
            non_interactive: [(xor(), 3.into())].into_iter().collect(),
            and_gates: [4.into()].into_iter().collect(),
        };

        let third_layer = CircuitLayer {
            non_interactive: [(out(), 5.into())].into_iter().collect(),
            and_gates: [].into_iter().collect(),
        };

        assert_eq!(Some(first_layer), cl_iter.next());
        assert_eq!(Some(snd_layer), cl_iter.next());
        assert_eq!(Some(third_layer), cl_iter.next());
        assert_eq!(None, cl_iter.next());
    }

    #[test]
    fn parent_gates() {
        let mut circuit = Circuit::<u32>::new();
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
        let mut circuit = Circuit::<u32>::with_capacity(100_000_000, 100_000_000);
        for i in 0..10_000 {
            for j in 0..10_000 {
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
        let converted = Circuit::from_bristol(parsed.clone()).unwrap();
        assert_eq!(
            parsed.header.gates + converted.input_count() + converted.output_count(),
            converted.gate_count()
        );
        assert_eq!(
            converted.input_count(),
            parsed.header.input_wires.iter().sum()
        );
        assert_eq!(parsed.header.output_wires, converted.output_count());
        // TODO comparing the wire counts is a little tricky since we have a slightly different
        // view of what a wire is
        // assert_eq!(parsed.header.wires, converted.wire_count());
    }
}
