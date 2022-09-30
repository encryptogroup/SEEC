use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::ops::RangeInclusive;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::{iter, mem};

use itertools::{EitherOrBoth, GroupBy, Itertools};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use smallvec::SmallVec;
use tracing::{debug, trace};

use crate::circuit::{
    BaseCircuit, Circuit, CircuitId, CrossCircuitConnections, DefaultIdx, Gate, GateId, GateIdx,
};
use crate::share_wrapper::{sub_circuit_inputs, ShareWrapper};
use crate::utils::ByAddress;

pub type SharedCircuit<Idx = DefaultIdx> = Arc<Mutex<BaseCircuit<Idx>>>;

/// Lazily initialized global CircuitBuilder. This is used by the ShareWrapper API's and the
/// #[sub_circuit] macro to construct a circuit without having direct access to the circuits.
pub(crate) static CIRCUIT_BUILDER: Lazy<Mutex<CircuitBuilder>> =
    Lazy::new(|| Mutex::new(CircuitBuilder::new()));
#[doc(hidden)]
/// Used by the sub_circuit attr proc macro
pub static EVALUATING_SUB_CIRCUIT: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct SubCircuitGate<Idx = DefaultIdx> {
    pub circuit_id: CircuitId,
    pub gate_id: GateId<Idx>,
}

pub struct CircuitBuilder<Idx = DefaultIdx> {
    pub(crate) circuits: Vec<SharedCircuit<Idx>>,
    connections: CrossCircuitConnections<Idx>,
    caches: HashSet<ByAddress<'static, dyn SubCircCache + Send + Sync>>,
}

impl<Idx> SubCircuitGate<Idx> {
    pub fn new(circuit_id: CircuitId, gate_id: GateId<Idx>) -> Self {
        Self {
            circuit_id,
            gate_id,
        }
    }

    pub fn new_main_gate(gate_id: GateId<Idx>) -> Self {
        Self::new(0, gate_id)
    }
}

impl<Idx: GateIdx> SubCircuitGate<Idx> {
    pub fn successor(&self) -> SubCircuitGate<Idx> {
        Self {
            circuit_id: self.circuit_id,
            gate_id: GateId(self.gate_id.0 + Idx::one()),
        }
    }
}

impl CircuitBuilder {
    pub fn install(self) -> CircuitBuilder {
        mem::replace(&mut *CIRCUIT_BUILDER.lock(), self)
    }

    pub fn get_global_circuit(id: CircuitId) -> Option<SharedCircuit> {
        CircuitBuilder::with_global(|builder| builder.get_circuit(id))
    }

    pub fn push_global_circuit(circuit: SharedCircuit) -> CircuitId {
        CircuitBuilder::with_global(|builder| builder.push_circuit(circuit))
    }

    pub fn with_global<R, F>(op: F) -> R
    where
        F: FnOnce(&mut CircuitBuilder) -> R,
    {
        op(&mut CIRCUIT_BUILDER.lock())
    }

    pub fn global_into_circuit() -> Circuit<DefaultIdx> {
        let mut global_builder = mem::take(&mut *CIRCUIT_BUILDER.lock());
        global_builder.clear_caches();
        global_builder.into_circuit()
    }

    #[tracing::instrument(level = "trace", skip(self, inputs))]
    pub fn connect_sub_circuit(&mut self, inputs: &[ShareWrapper], sc_id: CircuitId) {
        trace!(to = sc_id, ?inputs);
        let circuit = self
            .circuits
            .get(sc_id as usize)
            .expect("SubCircuit not added")
            .clone();
        let circuit = circuit.lock();
        let to_circuit_id = sc_id;
        let inputs: Vec<_> = inputs.iter().cloned().map(Into::into).collect();
        let inputs = &inputs[..];

        let from_ranges = match group_gates_iter(inputs) {
            None => return,
            Some(iter) => iter,
        };

        let to_input_ids = circuit.sub_circuit_input_gates();
        let to_are_consecutive = to_input_ids
            .iter()
            .zip(to_input_ids.iter().skip(1))
            .all(|(prev, next)| prev.0 + 1 == next.0);

        let mut inputs = inputs.iter().copied();
        // TODO this will use range connections even when the groups are only one elem big
        //  we should probably iterate over the groups, check their size and then decide
        //  if it should be a range conn or one-to-one
        if to_are_consecutive && !to_input_ids.is_empty() {
            // to an from are consecutive gates
            debug!("Adding range sub circuit");

            let _to_ids_iter = to_input_ids.iter();
            let mut inputs_connected = 0;

            // type inference issue
            for (first_from, group) in from_ranges.into_iter() {
                let (count, last_in_grp) = group.fold((0, None), |(count, _last), eob| {
                    let left = match eob {
                        EitherOrBoth::Both(left, _) | EitherOrBoth::Left(left) => *left,
                        EitherOrBoth::Right(_) => unreachable!("Left is always longer"),
                    };
                    (count + 1, Some(left))
                });

                let last_in_grp = last_in_grp.expect("Empty from group");

                let from_range = RangeInclusive::new(first_from, last_in_grp);
                let to_ids = &to_input_ids[inputs_connected..inputs_connected + count];
                inputs_connected += count;
                let to_range = RangeInclusive::new(
                    SubCircuitGate::new(to_circuit_id, to_ids[0]),
                    SubCircuitGate::new(to_circuit_id, *to_ids.last().unwrap()),
                );
                assert_eq!(
                    from_range.end().gate_id.0 - from_range.start().gate_id.0,
                    to_range.end().gate_id.0 - to_range.start().gate_id.0
                );
                trace!(?from_range, ?to_range, "connecting ranges");
                self.connections
                    .range_connections
                    .outgoing
                    .insert(from_range.clone(), to_range.clone());
                self.connections
                    .range_connections
                    .incoming
                    .insert(to_range, from_range);
            }
        } else {
            debug!(
                // from_are_consecutive,
                to_are_consecutive,
                "Adding one to one connections"
            );
            let connections = to_input_ids.iter().flat_map(|to_input_id| {
                let input_count = circuit.get_gate(*to_input_id).input_count();
                let inputs_for_gate: SmallVec<[_; 2]> =
                    (0..input_count).map(|_| inputs.next().unwrap()).collect();
                inputs_for_gate
                    .into_iter()
                    .map(move |sc_gate| (sc_gate, SubCircuitGate::new(to_circuit_id, *to_input_id)))
            });
            self.connect_circuits_unchecked(connections);
        }
    }

    /// Connections maps tuples of (SubCircuit, In/OutGate) to (SubCircuit, InputGate)
    ///
    /// # Panics
    /// TODO
    pub fn connect_circuits(
        &mut self,
        connections: impl IntoIterator<Item = (ShareWrapper, ShareWrapper)>,
    ) {
        for (from, to) in connections {
            let from: SubCircuitGate = from.into();
            let to: SubCircuitGate = to.into();
            assert!(
                matches!(
                    self.circuits[to.circuit_id as usize]
                        .lock()
                        .get_gate(to.gate_id),
                    Gate::SubCircuitInput
                ),
                "Can only connect to a SubCircuitInput gate"
            );
            self.connections
                .one_to_one
                .outgoing
                .entry(from)
                .or_default()
                .push(to);
            self.connections
                .one_to_one
                .incoming
                .entry(to)
                .or_default()
                .push(from);
        }
    }
}

impl<Idx: GateIdx> CircuitBuilder<Idx> {
    pub fn new() -> Self {
        Self {
            circuits: vec![Default::default()],
            connections: CrossCircuitConnections::default(),
            caches: HashSet::new(),
        }
    }

    pub fn circuits_count(&self) -> usize {
        self.circuits.len()
    }

    pub fn get_main_circuit(&self) -> SharedCircuit<Idx> {
        self.circuits[0].clone()
    }

    pub fn get_circuit(&self, id: CircuitId) -> Option<SharedCircuit<Idx>> {
        self.circuits.get(id as usize).cloned()
    }

    pub fn push_circuit(&mut self, circuit: SharedCircuit<Idx>) -> CircuitId {
        let id = self.circuits.len().try_into().expect("Too many circuits");
        self.circuits.push(circuit);
        id
    }

    /// Connections maps tuples of (SubCircuit, In/OutGate) to (SubCircuit, InputGate)
    ///
    /// TODO: This currently only uses the one_to_one mapping, it should use a range mapping
    ///     where applicable
    fn connect_circuits_unchecked(
        &mut self,
        connections: impl IntoIterator<Item = (SubCircuitGate<Idx>, SubCircuitGate<Idx>)>,
    ) {
        for (from, to) in connections {
            self.connections
                .one_to_one
                .outgoing
                .entry(from)
                .or_default()
                .push(to);
            self.connections
                .one_to_one
                .incoming
                .entry(to)
                .or_default()
                .push(from);
        }
    }

    pub fn into_circuit(self) -> Circuit<Idx> {
        // Converts the Vec<Arc<Mutex<Circuit>>> into Vec<Arc<Circuit>> as the Mutex is not
        // needed when evaluating the circuit. This prevents unnecessary locking during the
        // evaluation
        let mut circuits = Vec::with_capacity(self.circuits.len());
        let mut unwrapped_circuits_map = HashMap::new();
        for c in self.circuits {
            let c_ptr = Arc::as_ptr(&c);
            let entry = unwrapped_circuits_map.entry(c_ptr);
            match entry {
                // If there is already an entry in the map, we clone the Arc at the specified
                // position in the circuits Vec
                Entry::Occupied(occ) => {
                    circuits.push(Arc::clone(&circuits[*occ.get()]));
                }
                Entry::Vacant(vac) => {
                    // If there is no entry in the map, we mem::take the locked circuit and push it
                    // into circuits in a new Arc allocation
                    let c = std::mem::take(&mut *c.lock());
                    circuits.push(Arc::new(c));
                    // Add the index of the added circuit into the map with the original circuits
                    // addr as key
                    vac.insert(circuits.len() - 1);
                }
            }
        }
        Circuit {
            circuits,
            connections: self.connections,
        }
    }

    pub fn add_cache<C: SubCircCache + Send + Sync>(&mut self, cache: &'static C) {
        self.caches.insert(ByAddress(cache));
    }

    pub fn clear_caches(&mut self) {
        for c in &self.caches {
            c.0.clear();
        }
    }
}

impl<Idx: GateIdx> Default for CircuitBuilder<Idx> {
    fn default() -> Self {
        Self::new()
    }
}

fn group_gates_iter<'a, Idx>(
    gates: &'a [SubCircuitGate<Idx>],
) -> Option<
    GroupBy<
        SubCircuitGate<Idx>,
        impl Iterator<Item = EitherOrBoth<&'a SubCircuitGate<Idx>, &'a SubCircuitGate<Idx>>>,
        impl FnMut(
            &EitherOrBoth<&'a SubCircuitGate<Idx>, &'a SubCircuitGate<Idx>>,
        ) -> SubCircuitGate<Idx>,
    >,
>
where
    Idx: GateIdx,
{
    let mut curr_range_start = *gates.first()?;
    Some(
        gates
            .iter()
            .zip_longest(gates.iter().skip(1))
            .group_by(move |eob| match *eob {
                EitherOrBoth::Both(&curr, &next) => {
                    if curr.successor() == next {
                        curr_range_start
                    } else {
                        mem::replace(&mut curr_range_start, next)
                    }
                }
                EitherOrBoth::Left(_) => curr_range_start,
                EitherOrBoth::Right(_) => {
                    unreachable!("The right iter is always one shorter")
                }
            }),
    )
}

pub trait SubCircuitOutput {
    fn create_output_gates(self) -> Self;
    fn connect_to_main(self, circuit_id: CircuitId) -> Self;
}

pub trait SubCircuitInput {
    // This is the Self type but with a generic lifetime. Needed for the implementation on
    // &[ShareWrappe]
    type Input<'a>;
    type Size: Clone;

    fn with_input<F, R>(self, for_circuit: CircuitId, f: F) -> R
    where
        for<'a> F: FnOnce(Self::Input<'a>) -> R;

    fn size(&self) -> Self::Size;

    fn flatten(self) -> Vec<ShareWrapper>;
}

pub trait SubCircCache {
    fn clear(&self);
}

impl<K, V> SubCircCache for Mutex<HashMap<K, V>> {
    fn clear(&self) {
        self.lock().clear()
    }
}

impl<const N: usize> SubCircuitOutput for [ShareWrapper; N] {
    fn create_output_gates(self) -> Self {
        self.map(|sh| sh.sub_circuit_output())
    }

    fn connect_to_main(mut self, circuit_id: CircuitId) -> Self {
        let main_inputs = sub_circuit_inputs(0, self.len(), Gate::SubCircuitInput);
        self.iter_mut().for_each(|sh| sh.circuit_id = circuit_id);
        CircuitBuilder::with_global(|global| {
            global.connect_circuits(self.into_iter().zip(main_inputs.iter().cloned()));
        });
        main_inputs.try_into().expect("Bug: self.len() != N")
    }
}

impl SubCircuitOutput for Vec<ShareWrapper> {
    fn create_output_gates(self) -> Self {
        self.iter().map(ShareWrapper::sub_circuit_output).collect()
    }

    fn connect_to_main(mut self, circuit_id: CircuitId) -> Self {
        let main_inputs = sub_circuit_inputs(0, self.len(), Gate::SubCircuitInput);
        self.iter_mut().for_each(|sh| sh.circuit_id = circuit_id);
        CircuitBuilder::with_global(|global| {
            global.connect_circuits(self.into_iter().zip(main_inputs.iter().cloned()));
        });
        main_inputs
    }
}

impl SubCircuitOutput for ShareWrapper {
    fn create_output_gates(self) -> Self {
        self.sub_circuit_output()
    }

    fn connect_to_main(mut self, circuit_id: CircuitId) -> Self {
        let main_input = ShareWrapper::sub_circuit_input(0, Gate::SubCircuitInput);
        self.circuit_id = circuit_id;
        CircuitBuilder::with_global(|global| {
            global.connect_circuits(iter::once((self, main_input.clone())));
        });
        main_input
    }
}

impl SubCircuitInput for &[ShareWrapper] {
    type Input<'a> = &'a [ShareWrapper];
    type Size = usize;

    fn with_input<F, R>(self, for_circuit: CircuitId, f: F) -> R
    where
        for<'a> F: FnOnce(Self::Input<'a>) -> R,
    {
        let inputs: Vec<_> = self
            .iter()
            .map(|_| ShareWrapper::sub_circuit_input(for_circuit, Gate::SubCircuitInput))
            .collect();
        f(&inputs[..])
    }

    fn size(&self) -> Self::Size {
        self.len()
    }

    fn flatten(self) -> Vec<ShareWrapper> {
        self.to_vec()
    }
}

impl<const N: usize> SubCircuitInput for &[[ShareWrapper; N]] {
    type Input<'a> = &'a [[ShareWrapper; N]];
    type Size = usize;

    fn with_input<F, R>(self, for_circuit: CircuitId, f: F) -> R
    where
        for<'a> F: FnOnce(Self::Input<'a>) -> R,
    {
        let inputs: Vec<_> = self
            .iter()
            .map(|_| {
                [(); N].map(|_| ShareWrapper::sub_circuit_input(for_circuit, Gate::SubCircuitInput))
            })
            .collect();
        f(&inputs[..])
    }

    fn size(&self) -> Self::Size {
        self.len()
    }

    fn flatten(self) -> Vec<ShareWrapper> {
        self.iter().flatten().cloned().collect()
    }
}

impl From<ShareWrapper> for SubCircuitGate {
    fn from(share: ShareWrapper) -> Self {
        SubCircuitGate {
            circuit_id: share.circuit_id,
            gate_id: share.output_of,
        }
    }
}

impl<Idx: GateIdx> Display for SubCircuitGate<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScGate(sc: {}, gate: {})", self.circuit_id, self.gate_id)
    }
}

#[cfg(test)]
mod tests {
    use itertools::EitherOrBoth;

    use crate::circuit::builder::group_gates_iter;
    use crate::{GateId, SubCircuitGate};

    fn scg(gate_id: u32) -> SubCircuitGate {
        SubCircuitGate::new(0, GateId(gate_id))
    }

    fn collect_groups(input: &[SubCircuitGate]) -> Vec<Vec<SubCircuitGate>> {
        group_gates_iter(&input)
            .unwrap()
            .into_iter()
            .map(|(k, group)| {
                let group: Vec<_> = group
                    .map(|eob| match eob {
                        EitherOrBoth::Both(left, _) | EitherOrBoth::Left(left) => *left,
                        EitherOrBoth::Right(_) => {
                            unreachable!()
                        }
                    })
                    .collect();
                assert_eq!(k, group[0], "{:?}", group);
                group
            })
            .collect()
    }

    #[test]
    fn test_group_gates() {
        let grp1 = [scg(1), scg(2), scg(3), scg(4)];
        let grp2 = [scg(6), scg(7), scg(8)];
        let inp = [&grp1[..], &grp2[..]].concat();
        let grps: Vec<_> = collect_groups(&inp);
        assert_eq!(&grps[0], &grp1[..]);
        assert_eq!(&grps[1], &grp2[..]);
    }

    #[test]
    fn test_group_gates_single_elem() {
        let inp = [scg(1), scg(3), scg(5), scg(7)];
        let grps: Vec<_> = collect_groups(&inp);
        assert_eq!(&grps[0], &[scg(1)]);
        assert_eq!(&grps[1], &[scg(3)]);
        assert_eq!(&grps[2], &[scg(5)]);
        assert_eq!(&grps[3], &[scg(7)]);
    }
}
