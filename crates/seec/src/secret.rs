//! High-level [`Secret`] API to construct a circuit.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use itertools::Itertools;

use crate::circuit::builder::SharedCircuit;
use crate::circuit::{BooleanGate, CircuitId, DefaultIdx, GateId, GateIdx};
use crate::gate::base::BaseGate;
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::ScalarDim;
use crate::CircuitBuilder;

pub struct Secret<P = BooleanGmw, Idx = DefaultIdx> {
    pub(crate) circuit_id: CircuitId,
    pub(crate) output_of: GateId<Idx>,
    // The current Secret API has some significant limitations when used in a multi-threaded
    // context. Better to forbid it for now so that we can maybe change to non-thread safe
    // primitives
    not_thread_safe: PhantomData<*const ()>,
    protocol: PhantomData<P>,
}

impl<P, Idx: GateIdx> Secret<P, Idx> {
    /// Create a Secret from raw parts. This method should not be needed in most cases.
    /// The user needs to ensure that the global CircuitBuilder has a circuit and gate with the
    /// corresponding id's.
    pub fn from_parts(circuit_id: CircuitId, output_of: GateId<Idx>) -> Self {
        Self {
            circuit_id,
            output_of,
            not_thread_safe: PhantomData,
            protocol: PhantomData,
        }
    }
}

impl<Idx: GateIdx> Secret<BooleanGmw, Idx> {
    /// Note: Do not use while holding mutable borrow of self.circuit as it will panic!
    pub fn from_const(circuit_id: CircuitId, constant: bool) -> Self {
        let circuit = CircuitBuilder::get_global_circuit(circuit_id).unwrap_or_else(|| {
            panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
        });
        let output_of = {
            let mut circuit = circuit.lock();
            circuit.add_gate(BooleanGate::Base(BaseGate::Constant(constant)))
        };
        Self::from_parts(circuit_id, output_of)
    }

    pub fn input(circuit_id: CircuitId) -> Self {
        let circuit = CircuitBuilder::get_global_circuit(circuit_id).unwrap_or_else(|| {
            panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
        });
        let output_of = circuit
            .lock()
            .add_gate(BooleanGate::Base(BaseGate::Input(ScalarDim)));
        Self::from_parts(circuit_id, output_of)
    }

    pub fn sub_circuit_input(circuit_id: CircuitId, gate: BooleanGate) -> Self {
        let circuit = CircuitBuilder::get_global_circuit(circuit_id).unwrap_or_else(|| {
            panic!("circuit_id {circuit_id} is not stored in global CircuitBuilder")
        });
        let output_of = {
            let mut circ = circuit.lock();
            circ.add_sc_input_gate(gate)
        };
        Self::from_parts(circuit_id, output_of)
    }

    pub fn sub_circuit_output(&self) -> Self {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let output_of = circuit.add_wired_gate(
            BooleanGate::Base(BaseGate::SubCircuitOutput(ScalarDim)),
            &[self.output_of],
        );
        Self::from_parts(self.circuit_id, output_of)
    }

    /// Constructs a `Gate::Output` in the circuit with this secret's value
    pub fn output(&self) -> GateId<Idx> {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        circuit.add_wired_gate(
            BooleanGate::Base(BaseGate::Output(ScalarDim)),
            &[self.output_of],
        )
    }

    pub fn into_output(self) -> GateId<Idx> {
        self.output()
    }

    pub fn connect_to_main_circuit(self) -> Self {
        assert_ne!(
            self.circuit_id, 0,
            "Can't connect Secret of main circuit to main circuit"
        );
        let out = self.sub_circuit_output();
        CircuitBuilder::<bool, BooleanGate, Idx>::with_global(|builder| {
            let input_to_main = Secret::sub_circuit_input(
                0,
                BooleanGate::Base(BaseGate::SubCircuitInput(ScalarDim)),
            );
            builder.connect_circuits([(out, input_to_main.clone())]);
            input_to_main
        })
    }

    pub fn gate_id(&self) -> GateId<Idx> {
        self.output_of
    }

    pub fn circuit_id(&self) -> CircuitId {
        self.circuit_id
    }

    pub(crate) fn get_circuit(&self) -> SharedCircuit<bool, BooleanGate, Idx> {
        CircuitBuilder::get_global_circuit(self.circuit_id)
            .expect("circuit_id is not stored in global CircuitBuilder")
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitXor<Rhs> for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn bitxor(mut self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "Secret operations are only defined on Wrappers for the same circuit"
        );
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, rhs.output_of])
        };
        self
    }
}

impl<Idx: GateIdx> BitXor<bool> for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn bitxor(mut self, rhs: bool) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
            circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, const_gate])
        };
        self
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitXorAssign<Rhs> for Secret<BooleanGmw, Idx> {
    fn bitxor_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "Secret operations are only defined on Wrappers for the same circuit"
        );
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        self.output_of = circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, rhs.output_of]);
    }
}

impl<Idx: GateIdx> BitXorAssign<bool> for Secret<BooleanGmw, Idx> {
    fn bitxor_assign(&mut self, rhs: bool) {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
        self.output_of = circuit.add_wired_gate(BooleanGate::Xor, &[self.output_of, const_gate]);
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitAnd<Rhs> for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn bitand(mut self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "Secret operations are only defined on Wrappers for the same circuit"
        );
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::And, &[self.output_of, rhs.output_of])
        };
        self
    }
}

impl<Idx: GateIdx> BitAnd<bool> for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn bitand(mut self, rhs: bool) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
            circuit.add_wired_gate(BooleanGate::And, &[self.output_of, const_gate])
        };
        self
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitAndAssign<Rhs> for Secret<BooleanGmw, Idx> {
    fn bitand_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "Secret operations are only defined on Wrappers for the same circuit"
        );
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        self.output_of = circuit.add_wired_gate(BooleanGate::And, &[self.output_of, rhs.output_of]);
    }
}

impl<Idx: GateIdx> BitAndAssign<bool> for Secret<BooleanGmw, Idx> {
    fn bitand_assign(&mut self, rhs: bool) {
        let circuit = self.get_circuit();
        let mut circuit = circuit.lock();
        let const_gate = circuit.add_gate(BooleanGate::Base(BaseGate::Constant(rhs)));
        self.output_of = circuit.add_wired_gate(BooleanGate::And, &[self.output_of, const_gate]);
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitOr<Rhs> for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn bitor(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.borrow();
        assert_eq!(
            self.circuit_id, rhs.circuit_id,
            "Secret operations are only defined on Wrappers for the same circuit"
        );
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: GateIdx> BitOr<bool> for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn bitor(self, rhs: bool) -> Self::Output {
        let rhs = Secret::from_const(self.circuit_id, rhs);
        // a | b <=> (a ^ b) ^ (a & b)
        self.clone() ^ rhs.clone() ^ (self & rhs)
    }
}

impl<Idx: GateIdx, Rhs: Borrow<Self>> BitOrAssign<Rhs> for Secret<BooleanGmw, Idx> {
    fn bitor_assign(&mut self, rhs: Rhs) {
        let rhs = rhs.borrow();
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl<Idx: GateIdx> BitOrAssign<bool> for Secret<BooleanGmw, Idx> {
    fn bitor_assign(&mut self, rhs: bool) {
        let rhs = Secret::from_const(self.circuit_id, rhs);
        *self ^= rhs.clone() ^ (self.clone() & rhs);
    }
}

impl<Idx: GateIdx> Not for Secret<BooleanGmw, Idx> {
    type Output = Self;

    fn not(mut self) -> Self::Output {
        self.output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::Inv, &[self.output_of])
        };
        self
    }
}

impl<'a, Idx: GateIdx> Not for &'a Secret<BooleanGmw, Idx> {
    type Output = Secret<BooleanGmw, Idx>;

    fn not(self) -> Self::Output {
        let output_of = {
            let circuit = self.get_circuit();
            let mut circuit = circuit.lock();
            circuit.add_wired_gate(BooleanGate::Inv, &[self.output_of])
        };
        Secret::from_parts(self.circuit_id, output_of)
    }
}

impl<P, Idx: GateIdx> Debug for Secret<P, Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Secret")
            .field("circuit_id", &self.circuit_id)
            .field("output_of", &self.output_of)
            .finish()
    }
}

impl<P, Idx: Hash> Hash for Secret<P, Idx> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.circuit_id.hash(state);
        self.output_of.hash(state);
    }
}

impl<P, Idx: PartialEq> PartialEq for Secret<P, Idx> {
    fn eq(&self, other: &Self) -> bool {
        self.circuit_id == other.circuit_id && self.output_of == other.output_of
    }
}

impl<P, Idx: Eq> Eq for Secret<P, Idx> {}

impl<P, Idx: PartialOrd> PartialOrd for Secret<P, Idx> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.circuit_id.partial_cmp(&other.circuit_id) {
            Some(Ordering::Equal) => self.output_of.partial_cmp(&other.output_of),
            other => other,
        }
    }
}

impl<P, Idx: Ord> Ord for Secret<P, Idx> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("Inconsistent partial_cmp")
    }
}

impl<P, Idx: Clone> Clone for Secret<P, Idx> {
    fn clone(&self) -> Self {
        Self {
            circuit_id: self.circuit_id,
            output_of: self.output_of.clone(),
            not_thread_safe: PhantomData,
            protocol: PhantomData,
        }
    }
}

// TODO placeholder function until I can think of a nice place to put this
// Creates inputs for the main circuit (id = 0)
pub fn inputs<Idx: GateIdx>(inputs: usize) -> Vec<Secret<BooleanGmw, Idx>> {
    (0..inputs).map(|_| Secret::input(0)).collect()
}

// TODO placeholder function until I can think of a nice place to put this
// TODO this needs to have a generic return type to support more complex sub circuit input functions
pub(crate) fn sub_circuit_inputs<Idx: GateIdx>(
    circuit_id: CircuitId,
    inputs: usize,
    gate: BooleanGate,
) -> Vec<Secret<BooleanGmw, Idx>> {
    (0..inputs)
        .map(|_| Secret::sub_circuit_input(circuit_id, gate))
        .collect()
}

/// Reduce the slice of Secrets with the provided operation. The operation can be a closure
/// or simply one of the operations implemented on [`Secret`]s, like [`BitAnd`].  
/// The circuit will be constructed such that the depth is minimal.
///
/// ```rust
///# use std::sync::Arc;
///# use seec::circuit::{BaseCircuit, DefaultIdx};
///# use seec::secret::{inputs, low_depth_reduce};
///# use parking_lot::Mutex;
///# use seec::{BooleanGate, Circuit, CircuitBuilder};
///#
/// let inputs = inputs::<DefaultIdx>(23);
/// low_depth_reduce(inputs, std::ops::BitAnd::bitand)
///     .unwrap()
///     .output();
/// // It is important that the Gate and Idx type of the circuit match up with those of the
/// // Secrets, as otherwise an empty circuit will be returned
/// let and_tree: Circuit<bool, BooleanGate, DefaultIdx> = CircuitBuilder::global_into_circuit();
/// assert_eq!(and_tree.interactive_count(), 22)
/// ```
///
pub fn low_depth_reduce<F, Idx: GateIdx>(
    shares: impl IntoIterator<Item = Secret<BooleanGmw, Idx>>,
    mut f: F,
) -> Option<Secret<BooleanGmw, Idx>>
where
    F: FnMut(Secret<BooleanGmw, Idx>, Secret<BooleanGmw, Idx>) -> Secret<BooleanGmw, Idx>,
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
