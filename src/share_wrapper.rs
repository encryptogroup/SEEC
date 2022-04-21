use crate::circuit::{Circuit, Gate, GateId};
use itertools::Itertools;
use petgraph::graph::IndexType;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use std::rc::Rc;

#[derive(Clone)]
pub struct ShareWrapper<Idx> {
    pub(crate) circuit: Rc<RefCell<Circuit<Idx>>>,
    output_of: GateId<Idx>,
}

impl<Idx: IndexType> ShareWrapper<Idx> {
    /// Note: Do not use while holding mutable borrow of self.circuit as it will panic!
    pub fn from_const(circuit: Rc<RefCell<Circuit<Idx>>>, constant: bool) -> Self {
        let output_of = {
            let mut circuit = circuit.borrow_mut();
            circuit.add_gate(Gate::Constant(constant))
        };
        Self { circuit, output_of }
    }

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

// TODO currently the std::ops traits are only implemented for Rhs = Self, this means that
//  users of the Api need to explicitly clone ShareWrapper if they want to use the output of a gate
//  multiple times. We could implement the traits also for Rhs = &Self which might make it more
//  ergonomic

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

impl<Idx: IndexType> BitXor<bool> for ShareWrapper<Idx> {
    type Output = Self;

    fn bitxor(self, rhs: bool) -> Self::Output {
        let output_of = {
            let mut circuit = self.circuit.borrow_mut();
            let const_gate = circuit.add_gate(Gate::Constant(rhs));
            circuit.add_wired_gate(Gate::Xor, &[self.output_of, const_gate])
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

impl<Idx: IndexType> BitXorAssign<bool> for ShareWrapper<Idx> {
    fn bitxor_assign(&mut self, rhs: bool) {
        let mut circuit = self.circuit.borrow_mut();
        let const_gate = circuit.add_gate(Gate::Constant(rhs));
        self.output_of = circuit.add_wired_gate(Gate::Xor, &[self.output_of, const_gate]);
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

impl<Idx: IndexType> BitAnd<bool> for ShareWrapper<Idx> {
    type Output = Self;

    fn bitand(self, rhs: bool) -> Self::Output {
        let output_of = {
            let mut circuit = self.circuit.borrow_mut();
            let const_gate = circuit.add_gate(Gate::Constant(rhs));
            circuit.add_wired_gate(Gate::And, &[self.output_of, const_gate])
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

impl<Idx: IndexType> BitAndAssign<bool> for ShareWrapper<Idx> {
    fn bitand_assign(&mut self, rhs: bool) {
        let mut circuit = self.circuit.borrow_mut();
        let const_gate = circuit.add_gate(Gate::Constant(rhs));
        self.output_of = circuit.add_wired_gate(Gate::And, &[self.output_of, const_gate]);
    }
}

impl<Idx: IndexType> BitOr for ShareWrapper<Idx> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: IndexType> BitOr<bool> for ShareWrapper<Idx> {
    type Output = Self;

    fn bitor(self, rhs: bool) -> Self::Output {
        let rhs = ShareWrapper::from_const(Rc::clone(&self.circuit), rhs);
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: IndexType> BitOrAssign for ShareWrapper<Idx> {
    fn bitor_assign(&mut self, rhs: Self) {
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl<Idx: IndexType> BitOrAssign<bool> for ShareWrapper<Idx> {
    fn bitor_assign(&mut self, rhs: bool) {
        let rhs = ShareWrapper::from_const(Rc::clone(&self.circuit), rhs);
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

impl<Idx: Debug> Debug for ShareWrapper<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShareWrapper for output of gate {:?}", self.output_of)
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

/// Reduce the slice of ShareWrappers with the provided operation. The operation can be a closure
/// or simply one of the operations implemented on [`ShareWrapper`]s, like [`std::ops::BitAnd`].  
/// The circuit will be constructed such that the depth is minimal.
///
/// ```rust
///# use std::cell::RefCell;
///# use std::rc::Rc;
///# use gmw_rs::circuit::Circuit;
///# use gmw_rs::share_wrapper::{inputs, low_depth_reduce};
///#
/// let and_tree = Rc::new(RefCell::new(Circuit::<u16>::new()));
/// let inputs = inputs(and_tree.clone(), 23);
/// low_depth_reduce(&inputs, std::ops::BitAnd::bitand)
///     .unwrap()
///     .output();
/// ```
///
pub fn low_depth_reduce<
    Idx: IndexType,
    F: FnMut(ShareWrapper<Idx>, ShareWrapper<Idx>) -> ShareWrapper<Idx>,
>(
    shares: &[ShareWrapper<Idx>],
    mut f: F,
) -> Option<ShareWrapper<Idx>> {
    // Todo: This implementation is probably a little bit inefficient. It might be possible to use
    //  the lower level api to construct the circuit faster. This should be benchmarked however.
    let mut old_buf = Vec::with_capacity(shares.len() / 2);
    let mut buf = shares.to_owned();
    while buf.len() > 1 {
        mem::swap(&mut buf, &mut old_buf);
        let mut iter = old_buf.drain(..).tuples();
        for (s1, s2) in iter.by_ref() {
            buf.push(f(s1, s2));
        }
        for odd in iter.into_buffer() {
            buf.push(odd)
        }
    }
    debug_assert!(buf.len() <= 1);
    buf.pop()
}
