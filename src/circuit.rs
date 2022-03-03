use crate::errors::CircuitError;

use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, Reversed};
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, Direction, Graph};

use std::collections::VecDeque;
use std::path::Path;
use std::{fs, ops};

#[derive(Clone)]
pub struct Circuit {
    pub(crate) graph: CircuitGraph,
    pub(crate) input_count: usize,
    pub(crate) and_count: usize,
    pub(crate) output_count: usize,
}

type CircuitGraph = Graph<Gate, Wire, Directed, u32>;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Gate {
    And,
    Xor,
    Inv,
    Output,
    Input,
}

#[derive(Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Debug)]
pub struct GateId(NodeIndex<u32>);

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Wire;

impl Circuit {
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

    pub fn add_gate(&mut self, gate: Gate) -> GateId {
        match &gate {
            Gate::And => self.and_count += 1,
            Gate::Output => self.output_count += 1,
            Gate::Input => self.input_count += 1,
            _ => (),
        }
        self.graph.add_node(gate).into()
    }

    pub fn add_wire(&mut self, from: GateId, to: GateId) {
        self.graph.add_edge(from.0, to.0, Wire);
    }

    pub fn and_count(&self) -> usize {
        self.and_count
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn gate_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn wire_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn get_gate(&self, id: impl Into<GateId>) -> &Gate {
        let id = id.into().0;
        &self.graph[id]
    }

    pub fn parent_gates(&self, id: impl Into<GateId>) -> impl Iterator<Item = GateId> + '_ {
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
        let dot_content = Dot::with_config(&self.graph, &[Config::EdgeNoLabel]);
        fs::write(path, format!("{dot_content:?}"))?;
        Ok(())
    }

    pub(crate) fn as_graph(&self) -> &CircuitGraph {
        &self.graph
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

pub struct CircuitLayerIter<'a> {
    graph: &'a CircuitGraph,
    to_visit: VecDeque<NodeIndex<u32>>,
    next_layer: VecDeque<NodeIndex<u32>>,
    visited: <CircuitGraph as Visitable>::Map,
}

impl<'a> CircuitLayerIter<'a> {
    pub fn new(circuit: &'a Circuit) -> Self {
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
pub struct CircuitLayer {
    pub(crate) non_interactive: Vec<(Gate, GateId)>,
    pub(crate) and_gates: Vec<GateId>,
}

impl CircuitLayer {
    fn new() -> Self {
        Default::default()
    }

    fn add_and_gate(&mut self, id: GateId) {
        self.and_gates.push(id);
    }

    fn add_non_interactive(&mut self, data: (Gate, GateId)) {
        self.non_interactive.push(data);
    }

    fn is_empty(&self) -> bool {
        self.non_interactive.is_empty() && self.and_gates.is_empty()
    }
}

impl<'a> Iterator for CircuitLayerIter<'a> {
    type Item = CircuitLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let mut layer = CircuitLayer::new();
        std::mem::swap(&mut self.to_visit, &mut self.next_layer);
        while let Some(node_idx) = self.to_visit.pop_front() {
            if self.visited.is_visited(&node_idx) {
                continue;
            }
            self.visited.visit(node_idx);
            let gate = &self.graph[node_idx];
            match gate {
                Gate::And => {
                    layer.add_and_gate(node_idx.into());
                    self.next_layer.extend(self.graph.neighbors(node_idx));
                }
                Gate::Input | Gate::Output | Gate::Xor | Gate::Inv => {
                    layer.add_non_interactive((gate.clone(), node_idx.into()));
                    for neigh in self.graph.neighbors(node_idx) {
                        // Look at each neighbor, and those that only have incoming edges
                        // from the already ordered list, they are the next to visit.
                        if Reversed(self.graph)
                            .neighbors(neigh)
                            .all(|b| self.visited.is_visited(&b))
                        {
                            self.to_visit.push_back(neigh);
                        }
                    }
                }
            }
        }
        if layer.is_empty() {
            None
        } else {
            Some(layer)
        }
    }
}

impl GateId {
    pub fn as_usize(&self) -> usize {
        self.0.index().into()
    }
}

impl From<NodeIndex<u32>> for GateId {
    fn from(idx: NodeIndex<u32>) -> Self {
        Self(idx)
    }
}

impl From<usize> for GateId {
    fn from(id: usize) -> Self {
        let id: u32 = id.try_into().expect("Id to big for u32");
        GateId(NodeIndex::from(id))
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{Circuit, CircuitLayer, CircuitLayerIter, Gate, GateId};

    #[test]
    fn circuit_layer_iter() {
        let mut circuit = Circuit::new();
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
            non_interactive: vec![(inp(), 0.into()), (inp(), 1.into())],
            and_gates: vec![2.into()],
        };

        let snd_layer = CircuitLayer {
            non_interactive: vec![(xor(), 3.into())],
            and_gates: vec![4.into()],
        };

        let third_layer = CircuitLayer {
            non_interactive: vec![(out(), 5.into())],
            and_gates: vec![],
        };

        assert_eq!(Some(first_layer), cl_iter.next());
        assert_eq!(Some(snd_layer), cl_iter.next());
        assert_eq!(Some(third_layer), cl_iter.next());
        assert_eq!(None, cl_iter.next());
    }

    #[test]
    fn parent_gates() {
        let mut circuit = Circuit::new();
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
        let mut circuit = Circuit::new();
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
}
