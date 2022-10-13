use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use itertools::Itertools;

use crate::circuit::builder::SharedCircuit;
use crate::circuit::{CircuitId, DefaultIdx, Gate, GateId};
use crate::CircuitBuilder;

// TODO ShareWrappe can now implement Clone, but should it?
#[derive(Clone)]
pub struct ShareWrapper {
    pub(crate) circuit_id: CircuitId,
    pub(crate) output_of: GateId<DefaultIdx>,
    // The current ShareWrapper API has some significant limitations when used in a multi-threaded
    // context. Better to forbid it for now so that we can maybe change to non-thread safe
    // primitives
    not_thread_safe: PhantomData<*const ()>,
}

impl ShareWrapper {
    /// Note: Do not use while holding mutable borrow of self.circuit as it will panic!
    pub fn from_const(circuit_id: CircuitId, constant: bool) -> Self {
        let circuit = CircuitBuilder::get_global_circuit(circuit_id).unwrap_or_else(|| {
            panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
        });
        let output_of = {
            let mut circuit = circuit.lock();
            circuit.add_gate(Gate::Constant(constant))
        };
        Self {
            circuit_id,
            output_of,
            not_thread_safe: PhantomData,
        }
    }

    pub fn input(circuit_id: CircuitId) -> Self {
        let circuit = CircuitBuilder::get_global_circuit(circuit_id).unwrap_or_else(|| {
            panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
        });
        let output_of = circuit.lock().add_gate(Gate::Input);
        Self {
            circuit_id,
            output_of,
            not_thread_safe: PhantomData,
        }
    }

    pub fn sub_circuit_input(circuit_id: CircuitId, gate: Gate) -> Self {
        let circuit = CircuitBuilder::get_global_circuit(circuit_id).unwrap_or_else(|| {
            panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
        });
        let output_of = {
            let mut circ = circuit.lock();
            circ.add_sc_input_gate(gate)
        };
        Self {
            circuit_id,
            output_of,
            not_thread_safe: PhantomData,
        }
    }

    pub fn sub_circuit_output(&self) -> ShareWrapper {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let output_of = circuit.add_wired_gate(Gate::SubCircuitOutput, &[self.output_of]);
        Self {
            circuit_id: self.circuit_id,
            output_of,
            not_thread_safe: PhantomData,
        }
    }

    /// Consumes this ShareWrapper and constructs a `Gate::Output` in the circuit with its value
    pub fn output(self) -> GateId {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        circuit.add_wired_gate(Gate::Output, &[self.output_of])
    }

    pub fn connect_to_main_circuit(self) -> ShareWrapper {
        assert_ne!(
            self.circuit_id, 0,
            "Can't connect ShareWrapper of main circuit to main circuit"
        );
        let out = self.sub_circuit_output();
        CircuitBuilder::with_global(|builder| {
            let input_to_main = ShareWrapper::sub_circuit_input(0, Gate::SubCircuitInput);
            builder.connect_circuits([(out, input_to_main.clone())]);
            input_to_main
        })
    }

    pub fn gate_id(&self) -> GateId {
        self.output_of
    }

    pub fn get_circuit(&self) -> SharedCircuit {
        CircuitBuilder::get_global_circuit(self.circuit_id)
            .expect("circuit_id is not stored in global CircuitBuilder")
    }
}

impl<Rhs: Borrow<Self>> BitXor<Rhs> for ShareWrapper {
    type Output = Self;

    fn bitxor(mut self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(Gate::Xor, &[self.output_of, rhs.output_of])
        };
        self
    }
}

impl BitXor<bool> for ShareWrapper {
    type Output = Self;

    fn bitxor(mut self, rhs: bool) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            let const_gate = circuit.add_gate(Gate::Constant(rhs));
            circuit.add_wired_gate(Gate::Xor, &[self.output_of, const_gate])
        };
        self
    }
}

impl<Rhs: Borrow<Self>> BitXorAssign<Rhs> for ShareWrapper {
    fn bitxor_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        self.output_of = circuit.add_wired_gate(Gate::Xor, &[self.output_of, rhs.output_of]);
    }
}

impl BitXorAssign<bool> for ShareWrapper {
    fn bitxor_assign(&mut self, rhs: bool) {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let const_gate = circuit.add_gate(Gate::Constant(rhs));
        self.output_of = circuit.add_wired_gate(Gate::Xor, &[self.output_of, const_gate]);
    }
}

impl<Rhs: Borrow<Self>> BitAnd<Rhs> for ShareWrapper {
    type Output = Self;

    fn bitand(mut self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(Gate::And, &[self.output_of, rhs.output_of])
        };
        self
    }
}

impl BitAnd<bool> for ShareWrapper {
    type Output = Self;

    fn bitand(mut self, rhs: bool) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            let const_gate = circuit.add_gate(Gate::Constant(rhs));
            circuit.add_wired_gate(Gate::And, &[self.output_of, const_gate])
        };
        self
    }
}

impl<Rhs: Borrow<Self>> BitAndAssign<Rhs> for ShareWrapper {
    fn bitand_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        self.output_of = circuit.add_wired_gate(Gate::And, &[self.output_of, rhs.output_of]);
    }
}

impl BitAndAssign<bool> for ShareWrapper {
    fn bitand_assign(&mut self, rhs: bool) {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let const_gate = circuit.add_gate(Gate::Constant(rhs));
        self.output_of = circuit.add_wired_gate(Gate::And, &[self.output_of, const_gate]);
    }
}

impl<Rhs: Borrow<Self>> BitOr<Rhs> for ShareWrapper {
    type Output = Self;

    fn bitor(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "ShareWrapper operations are only defined on Wrappers for the same circuit"
        );
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl BitOr<bool> for ShareWrapper {
    type Output = Self;

    fn bitor(self, rhs: bool) -> Self::Output {
        let rhs = ShareWrapper::from_const(self.circuit_id, rhs);
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Rhs: Borrow<Self>> BitOrAssign<Rhs> for ShareWrapper {
    fn bitor_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl BitOrAssign<bool> for ShareWrapper {
    fn bitor_assign(&mut self, rhs: bool) {
        let rhs = ShareWrapper::from_const(self.circuit_id, rhs);
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl Not for ShareWrapper {
    type Output = Self;

    fn not(mut self) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(Gate::Inv, &[self.output_of])
        };
        self
    }
}

impl<'a> Not for &'a ShareWrapper {
    type Output = ShareWrapper;

    fn not(self) -> Self::Output {
        let output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(Gate::Inv, &[self.output_of])
        };
        ShareWrapper {
            circuit_id: self.circuit_id,
            output_of,
            not_thread_safe: PhantomData,
        }
    }
}

impl Debug for ShareWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShareWrapper for output of gate {:?}", self.output_of)
    }
}

// TODO placeholder function until I can think of a nice place to put this
// Creates inputs for the main circuit (id = 0)
pub fn inputs(inputs: usize) -> Vec<ShareWrapper> {
    (0..inputs).map(|_| ShareWrapper::input(0)).collect()
}

// TODO placeholder function until I can think of a nice place to put this
// TODO this needs to have a generic return type to support more complex sub circuit input functions
pub(crate) fn sub_circuit_inputs(
    circuit_id: CircuitId,
    inputs: usize,
    gate: Gate,
) -> Vec<ShareWrapper> {
    (0..inputs)
        .map(|_| ShareWrapper::sub_circuit_input(circuit_id, gate))
        .collect()
}

/// Reduce the slice of ShareWrappers with the provided operation. The operation can be a closure
/// or simply one of the operations implemented on [`ShareWrapper`]s, like [`std::ops::BitAnd`].  
/// The circuit will be constructed such that the depth is minimal.
///
/// ```rust
///# use std::sync::Arc;
///# use gmw::circuit::BaseCircuit;
///# use gmw::share_wrapper::{inputs, low_depth_reduce};
///# use parking_lot::Mutex;
///# use gmw::CircuitBuilder;
///#
/// let inputs = inputs(23);
/// low_depth_reduce(inputs, std::ops::BitAnd::bitand)
///     .unwrap()
///     .output();
/// let and_tree = CircuitBuilder::global_into_circuit();
/// assert_eq!(and_tree.and_count(), 22)
/// ```
///
pub fn low_depth_reduce<F>(
    shares: impl IntoIterator<Item = ShareWrapper>,
    mut f: F,
) -> Option<ShareWrapper>
where
    F: FnMut(ShareWrapper, ShareWrapper) -> ShareWrapper,
{
    // Todo: This implementation is probably a little bit inefficient. It might be possible to use
    //  the lower level api to construct the circuit faster. This should be benchmarked however.
    let mut buf: Vec<_> = shares.into_iter().collect();
    let mut old_buf = Vec::with_capacity(buf.len() / 2);
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
