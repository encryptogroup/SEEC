use crate::circuit::{Circuit, Gate, GateId};
use petgraph::graph::IndexType;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use std::rc::Rc;

#[derive(Clone)]
pub struct ShareWrapper<Idx> {
    circuit: Rc<RefCell<Circuit<Idx>>>,
    output_of: GateId<Idx>,
}

impl<Idx: IndexType> ShareWrapper<Idx> {
    pub fn input(circuit: Rc<RefCell<Circuit<Idx>>>) -> Self {
        let output_of = circuit.borrow_mut().add_gate(Gate::Input);
        Self { circuit, output_of }
    }

    /// Consumes this ShareWrapper and constructs a `Gate::Output` in the circuit with its value
    pub fn output(self) -> GateId<Idx> {
        let mut circuit = self.circuit.borrow_mut();
        circuit.add_wired_gate(Gate::Output, &[self.output_of])
    }
}

impl<Idx: IndexType> BitXor for ShareWrapper<Idx> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let output_of = {
            let mut circuit = self.circuit.borrow_mut();
            circuit.add_wired_gate(Gate::Xor, &[self.output_of, rhs.output_of])
        };
        Self {
            circuit: self.circuit,
            output_of,
        }
    }
}

impl<Idx: IndexType> BitXorAssign for ShareWrapper<Idx> {
    fn bitxor_assign(&mut self, rhs: Self) {
        let mut circuit = self.circuit.borrow_mut();
        self.output_of = circuit.add_wired_gate(Gate::Xor, &[self.output_of, rhs.output_of]);
    }
}

impl<Idx: IndexType> BitAnd for ShareWrapper<Idx> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let output_of = {
            let mut circuit = self.circuit.borrow_mut();
            circuit.add_wired_gate(Gate::And, &[self.output_of, rhs.output_of])
        };
        Self {
            circuit: self.circuit,
            output_of,
        }
    }
}

impl<Idx: IndexType> BitAndAssign for ShareWrapper<Idx> {
    fn bitand_assign(&mut self, rhs: Self) {
        let mut circuit = self.circuit.borrow_mut();
        self.output_of = circuit.add_wired_gate(Gate::And, &[self.output_of, rhs.output_of]);
    }
}

impl<Idx: Debug> Debug for ShareWrapper<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShareWrapper for output of gate {:?}", self.output_of)
    }
}

impl<Idx: IndexType> BitOr for ShareWrapper<Idx> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: IndexType> BitOrAssign for ShareWrapper<Idx> {
    fn bitor_assign(&mut self, rhs: Self) {
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl<Idx: IndexType> Not for ShareWrapper<Idx> {
    type Output = Self;

    fn not(self) -> Self::Output {
        let output_of = {
            let mut circuit = self.circuit.borrow_mut();
            circuit.add_wired_gate(Gate::Inv, &[self.output_of])
        };
        Self {
            circuit: self.circuit,
            output_of,
        }
    }
}

// TODO placeholder function until I can think of a nice place to put this
pub fn inputs<Idx: IndexType>(
    circuit: Rc<RefCell<Circuit<Idx>>>,
    inputs: usize,
) -> Vec<ShareWrapper<Idx>> {
    (0..inputs)
        .map(|_| ShareWrapper::input(circuit.clone()))
        .collect()
}
