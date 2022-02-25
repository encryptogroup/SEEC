use crate::traits::GateMeta;
use futures::StreamExt;
use std::collections::{HashSet, VecDeque};
use std::ops::Not;
use std::sync::Arc;

#[derive(Clone)]
pub struct Circuit {
    pub(crate) input_gates: Vec<Gate>,
    pub(crate) and_size: usize,
    pub(crate) size: usize,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum Gate {
    And(And),
    Xor(Xor),
    Output(Output),
    Input(Input),
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Input {
    pub(crate) common: GateCommon,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct And {
    pub(crate) common: GateCommon,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Xor {
    pub(crate) common: GateCommon,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Output {
    pub(crate) id: u64,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub(crate) struct GateCommon {
    pub(crate) children: Vec<Arc<Gate>>,
    pub(crate) id: u64,
}

impl Circuit {
    pub fn new(input_gates: Vec<Input>) -> Self {
        let input_gates = input_gates.into_iter().map(Gate::Input).collect();
        let mut circuit = Self {
            input_gates,
            and_size: 0,
            size: 0,
        };
        let mut and_gates = 0;
        let mut total_gates = 0;
        circuit.visit_with(|gate| match gate {
            Gate::And(_) => {
                and_gates += 1;
                total_gates += 1;
            }
            _ => total_gates += 1,
        });
        circuit.and_size = and_gates;
        circuit.size = total_gates;
        circuit
    }

    pub fn input_gates(&self) -> &[Gate] {
        &self.input_gates
    }

    pub fn input_size(&self) -> usize {
        self.input_gates.len()
    }

    pub fn and_size(&self) -> usize {
        self.and_size
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn visit_with(&self, mut visit_fn: impl FnMut(&Gate)) {
        let mut visited: HashSet<&Gate> = HashSet::new();
        let mut to_visit: VecDeque<&Gate> = VecDeque::from_iter(self.input_gates.iter());
        while let Some(gate) = to_visit.pop_front() {
            visit_fn(gate);
            visited.insert(gate);
            to_visit.extend(
                gate.get_children()
                    .iter()
                    .map(AsRef::as_ref)
                    .filter(|child| !visited.contains(child)),
            );
        }
    }
}

impl Gate {
    pub fn get_children(&self) -> &[Arc<Gate>] {
        match self {
            Gate::And(gate) => gate.get_children(),
            Gate::Xor(gate) => gate.get_children(),
            Gate::Input(gate) => gate.get_children(),
            Gate::Output(_) => &[],
        }
    }
}

// TODO: better way to do this? Maybe trait based or with macro?
impl Input {
    pub fn get_children(&self) -> &[Arc<Gate>] {
        &self.common.children
    }

    pub fn get_id(&self) -> u64 {
        self.common.id
    }
}

impl And {
    pub fn get_children(&self) -> &[Arc<Gate>] {
        &self.common.children
    }

    pub fn get_id(&self) -> u64 {
        self.common.id
    }
}

impl Xor {
    pub fn get_children(&self) -> &[Arc<Gate>] {
        &self.common.children
    }

    pub fn get_id(&self) -> u64 {
        self.common.id
    }
}
