//! [`CircuitBuilder`] is used to build an aggregate circuit of [`BaseCircuit`]s.
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::{iter, mem};

use itertools::{EitherOrBoth, GroupBy, Itertools};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use tracing::{debug, trace};
use typemap::{Key, ShareMap};

use crate::circuit::circuit_connections::CrossCircuitConnections;
use crate::circuit::{
    dyn_layers::Circuit, BaseCircuit, BooleanGate, CircuitId, DefaultIdx, GateId, GateIdx,
};
use crate::gate::base::BaseGate;
use crate::protocols::boolean_gmw::BooleanGmw;
use crate::protocols::{Gate, Plain, Protocol, ScalarDim};
use crate::secret::{sub_circuit_inputs, Secret};
use crate::utils::ByAddress;

pub type SharedCircuit<P = bool, G = BooleanGate, Idx = DefaultIdx, W = ()> =
    Arc<Mutex<BaseCircuit<P, G, Idx, W>>>;

/// Lazily initialized global CircuitBuilders. This is used by the Secret API's and the
/// #[sub_circuit] macro to construct a circuit without having direct access to the circuits.
/// The ShareMap stores one builder per Gate type.
pub(crate) static CIRCUIT_BUILDER_MAP: Lazy<Mutex<ShareMap>> =
    Lazy::new(|| Mutex::new(ShareMap::custom()));

/// Needed for the ShareMap.
struct KeyWrapper<P, G, Idx>(PhantomData<(P, G, Idx)>);

impl<P: 'static, G: 'static, Idx: 'static> Key for KeyWrapper<P, G, Idx> {
    type Value = CircuitBuilder<P, G, Idx>;
}

/// Used by the sub_circuit attr proc macro
#[doc(hidden)]
pub static EVALUATING_SUB_CIRCUIT: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct SubCircuitGate<Idx = DefaultIdx> {
    pub circuit_id: CircuitId,
    pub gate_id: GateId<Idx>,
}

pub struct CircuitBuilder<P = bool, G = BooleanGate, Idx = DefaultIdx> {
    pub(crate) circuits: Vec<SharedCircuit<P, G, Idx>>,
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

    pub fn into_usize(self) -> SubCircuitGate<usize> {
        SubCircuitGate {
            circuit_id: self.circuit_id,
            gate_id: GateId(self.gate_id.as_usize()),
        }
    }
}

impl<P: Plain, G: Gate<P>, Idx: GateIdx> CircuitBuilder<P, G, Idx> {
    pub fn install(self) -> Self {
        Self::with_global(|builder| mem::replace(builder, self))
    }

    pub fn get_global_circuit(id: CircuitId) -> Option<SharedCircuit<P, G, Idx>> {
        Self::with_global(|builder| builder.get_circuit(id))
    }

    pub fn push_global_circuit(circuit: SharedCircuit<P, G, Idx>) -> CircuitId {
        Self::with_global(|builder| builder.push_circuit(circuit))
    }

    #[allow(clippy::unwrap_or_default)]
    pub fn with_global<R, F>(op: F) -> R
    where
        F: FnOnce(&mut CircuitBuilder<P, G, Idx>) -> R,
    {
        let mut map = CIRCUIT_BUILDER_MAP.lock();
        let builder = map
            .entry::<KeyWrapper<P, G, Idx>>()
            .or_insert_with(CircuitBuilder::<P, G, Idx>::default);
        op(builder)
    }

    pub fn with_global_main_circ_mut<R, F>(f: F) -> R
    where
        F: FnOnce(&mut BaseCircuit<P, G, Idx>) -> R,
    {
        Self::with_global(|builder| {
            let main_circ = builder.get_main_circuit();
            let mut main_circ = main_circ.lock();
            f(&mut *main_circ)
        })
    }

    pub fn global_into_circuit() -> Circuit<P, G, Idx> {
        let mut global_builder = Self::with_global(mem::take);
        global_builder.clear_caches();
        global_builder.into_circuit()
    }

    #[tracing::instrument(level = "trace", skip(self, to_circuit, to_gate_ids))]
    pub(crate) fn connect_sub_circuit_gates<Prot>(
        &mut self,
        from: &[Secret<Prot, Idx>],
        to_circuit: &mut BaseCircuit<P, G, Idx>,
        to_circuit_id: CircuitId,
        to_gate_ids: &[GateId<Idx>],
    ) where
        Prot: Protocol<Gate = G>,
    {
        let inputs: Vec<_> = from.iter().cloned().map(Into::into).collect();

        let from_ranges = match group_gates_iter(&inputs) {
            None => return,
            Some(iter) => iter,
        };

        let to_are_consecutive = to_gate_ids
            .iter()
            .zip(to_gate_ids.iter().skip(1))
            .all(|(prev, next)| prev.0 + Idx::one() == next.0);

        // TODO this will use range connections even when the groups are only one elem big
        //  we should probably iterate over the groups, check their size and then decide
        //  if it should be a range conn or one-to-one
        if to_are_consecutive && !to_gate_ids.is_empty() {
            // to an from are consecutive gates
            debug!("Adding range sub circuit connection");

            let _to_ids_iter = to_gate_ids.iter();
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
                let to_ids = &to_gate_ids[inputs_connected..inputs_connected + count];
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
            let mut input_it = inputs.iter();
            let connections = to_gate_ids.iter().flat_map(move |to_input_id| {
                let input_count = to_circuit.get_gate(*to_input_id).input_size();

                // Unfortunately necessary, as captured variable input_it can't escpae
                // FnMut body
                let inputs_for_gate: SmallVec<[_; 2]> =
                    input_it.by_ref().cloned().take(input_count).collect();
                inputs_for_gate
                    .into_iter()
                    .map(move |sc_gate| (sc_gate, SubCircuitGate::new(to_circuit_id, *to_input_id)))
            });
            self.connect_circuits_unchecked(connections);
        };
    }

    #[tracing::instrument(level = "trace", skip(self, inputs))]
    pub fn connect_sub_circuit<Prot>(
        &mut self,
        inputs: &[Secret<Prot, Idx>],
        sc_id: CircuitId,
    ) -> Vec<Secret<Prot, Idx>>
    where
        Prot: Protocol<Gate = G>,
    {
        trace!(to = sc_id, ?inputs);
        let circuit = self
            .circuits
            .get(sc_id as usize)
            .expect("SubCircuit not added")
            .clone();
        let mut circuit = circuit.lock();
        let to_input_ids = circuit.sub_circuit_input_gates().to_vec();
        self.connect_sub_circuit_gates(inputs, &mut *circuit, sc_id, &to_input_ids);

        circuit
            .sub_circuit_output_gates()
            .iter()
            .map(|gate_id| Secret::from_parts(sc_id, *gate_id))
            .collect()
    }

    /// Connections maps tuples of (SubCircuit, In/OutGate) to (SubCircuit, InputGate)
    ///
    /// # Panics
    /// TODO
    pub fn connect_circuits(
        &mut self,
        connections: impl IntoIterator<Item = (Secret<BooleanGmw, Idx>, Secret<BooleanGmw, Idx>)>,
    ) {
        for (from, to) in connections {
            let from: SubCircuitGate<Idx> = from.into();
            let to: SubCircuitGate<Idx> = to.into();
            if from.circuit_id == to.circuit_id {
                self.circuits[from.circuit_id as usize]
                    .lock()
                    .add_wire(from.gate_id, to.gate_id);
            } else {
                assert!(
                    matches!(
                        self.circuits[to.circuit_id as usize]
                            .lock()
                            .get_gate(to.gate_id)
                            .as_base_gate(),
                        Some(BaseGate::SubCircuitInput(_))
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
}

impl<P, G: Gate<P>, Idx: GateIdx> CircuitBuilder<P, G, Idx> {
    pub fn new() -> Self {
        let main_circ = BaseCircuit::new();
        Self {
            circuits: vec![main_circ.into_shared()],
            connections: CrossCircuitConnections::default(),
            caches: HashSet::new(),
        }
    }

    pub fn circuits_count(&self) -> usize {
        self.circuits.len()
    }

    pub fn get_main_circuit(&self) -> SharedCircuit<P, G, Idx> {
        self.circuits[0].clone()
    }

    pub fn get_circuit(&self, id: CircuitId) -> Option<SharedCircuit<P, G, Idx>> {
        self.circuits.get(id as usize).cloned()
    }

    pub fn push_circuit(&mut self, circuit: SharedCircuit<P, G, Idx>) -> CircuitId {
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

    pub fn into_circuit(self) -> Circuit<P, G, Idx> {
        // Converts the Vec<Arc<Mutex<Circuit>>> into Vec<Circuit> and HashMap<CircuitId, usize>
        // to reduce locking overhead and ptr indirection + enable Ser/De
        let mut circuits = vec![];
        let mut unwrapped_circuits_map = HashMap::new();
        let mut circ_map = HashMap::new();
        for (idx, c) in self.circuits.into_iter().enumerate() {
            let c_ptr = Arc::as_ptr(&c);
            let entry = unwrapped_circuits_map.entry(c_ptr);
            match entry {
                // If there is already an entry in the map, we clone the Arc at the specified
                // position in the circuits Vec
                Entry::Occupied(occ) => {
                    circ_map.insert(idx as CircuitId, *occ.get());
                }
                Entry::Vacant(vac) => {
                    // If there is no entry in the map, we mem::take the locked circuit and push it
                    // into circuits
                    let dedup_id = circuits.len();
                    let c = mem::take(&mut *c.lock());
                    circuits.push(c);
                    // Add the index of the added circuit into the map with the original circuits
                    // addr as key
                    vac.insert(dedup_id);
                    circ_map.insert(idx as CircuitId, dedup_id);
                }
            }
        }
        Circuit {
            circuits,
            circ_map,
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

impl<P, G: Gate<P>, Idx: GateIdx> Default for CircuitBuilder<P, G, Idx> {
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

pub trait SubCircuitOutput: Sized {
    fn create_output_gates(self) -> Self;
    fn connect_to_main(self, circuit_id: CircuitId) -> Self;
    fn connect_simd_to_main(self, circuit_id: CircuitId, simd_size: usize) -> Vec<Self>;
}

pub trait SubCircuitInput {
    // This is the Self type but with a generic lifetime. Needed for the implementation on
    // &[Secret]
    type Input<'a>;
    type Size: Clone;
    type Protocol: Protocol;
    type Idx: GateIdx;

    fn with_input<F, R>(self, for_circuit: CircuitId, f: F) -> R
    where
        for<'a> F: FnOnce(Self::Input<'a>) -> R;

    fn size(&self) -> Self::Size;

    fn flatten(self) -> Vec<Secret<Self::Protocol, Self::Idx>>;
}

pub trait SubCircCache {
    fn clear(&self);
}

impl<K, V> SubCircCache for Mutex<HashMap<K, V>> {
    fn clear(&self) {
        self.lock().clear()
    }
}

impl<Idx: GateIdx, const N: usize> SubCircuitOutput for [Secret<BooleanGmw, Idx>; N] {
    fn create_output_gates(self) -> Self {
        self.map(|sh| sh.sub_circuit_output())
    }

    fn connect_to_main(mut self, circuit_id: CircuitId) -> Self {
        let main_inputs = sub_circuit_inputs(
            0,
            self.len(),
            BooleanGate::Base(BaseGate::SubCircuitInput(ScalarDim)),
        );
        let main_inputs_gates: Vec<_> = main_inputs.iter().map(|sw| sw.output_of).collect();
        self.iter_mut().for_each(|sh| sh.circuit_id = circuit_id);
        CircuitBuilder::<bool, BooleanGate, Idx>::with_global(|global| {
            let mut main_circ = global.get_main_circuit().lock_arc();
            global.connect_sub_circuit_gates(&self, &mut *main_circ, 0, &main_inputs_gates);
        });
        main_inputs.try_into().expect("Bug: self.len() != N")
    }

    fn connect_simd_to_main(self, _circuit_id: CircuitId, _simd_size: usize) -> Vec<Self> {
        todo!()
    }
}

impl<Idx: GateIdx> SubCircuitOutput for Vec<Secret<BooleanGmw, Idx>> {
    fn create_output_gates(self) -> Self {
        self.iter().map(Secret::sub_circuit_output).collect()
    }

    fn connect_to_main(mut self, circuit_id: CircuitId) -> Self {
        let main_inputs = sub_circuit_inputs(
            0,
            self.len(),
            BooleanGate::Base(BaseGate::SubCircuitInput(ScalarDim)),
        );
        let main_inputs_gates: Vec<_> = main_inputs.iter().map(|sw| sw.output_of).collect();
        self.iter_mut().for_each(|sh| sh.circuit_id = circuit_id);
        CircuitBuilder::<bool, BooleanGate, Idx>::with_global(|global| {
            let mut main_circ = global.get_main_circuit().lock_arc();
            global.connect_sub_circuit_gates(&self, &mut *main_circ, 0, &main_inputs_gates);
        });
        main_inputs
    }

    fn connect_simd_to_main(mut self, circuit_id: CircuitId, simd_size: usize) -> Vec<Self> {
        self.iter_mut().for_each(|sh| sh.circuit_id = circuit_id);
        (0..simd_size)
            .map(|simd_idx| {
                let main_inputs = sub_circuit_inputs(
                    0,
                    self.len(),
                    BooleanGate::Base(BaseGate::ConnectToMainFromSimd((
                        ScalarDim,
                        simd_idx.try_into().expect("Max SIMD size is u32::MAX"),
                    ))),
                );
                let main_input_gates: Vec<_> = main_inputs.iter().map(|sw| sw.output_of).collect();
                CircuitBuilder::<bool, BooleanGate, Idx>::with_global(|global| {
                    let mut main_circ = global.get_main_circuit().lock_arc();
                    global.connect_sub_circuit_gates(&self, &mut *main_circ, 0, &main_input_gates);
                });
                main_inputs
            })
            .collect()
    }
}

impl<Idx: GateIdx> SubCircuitOutput for Secret<BooleanGmw, Idx> {
    fn create_output_gates(self) -> Self {
        self.sub_circuit_output()
    }

    fn connect_to_main(mut self, circuit_id: CircuitId) -> Self {
        let main_input =
            Secret::sub_circuit_input(0, BooleanGate::Base(BaseGate::SubCircuitInput(ScalarDim)));
        self.circuit_id = circuit_id;
        CircuitBuilder::<bool, BooleanGate, Idx>::with_global(|global| {
            global.connect_circuits(iter::once((self, main_input.clone())));
        });
        main_input
    }

    fn connect_simd_to_main(self, _circuit_id: CircuitId, _simd_size: usize) -> Vec<Self> {
        todo!()
    }
}

impl<Idx: GateIdx> SubCircuitInput for &[Secret<BooleanGmw, Idx>] {
    type Input<'a> = &'a [Secret<BooleanGmw, Idx>];
    type Size = usize;
    type Protocol = BooleanGmw;
    type Idx = Idx;

    fn with_input<F, R>(self, for_circuit: CircuitId, f: F) -> R
    where
        for<'a> F: FnOnce(Self::Input<'a>) -> R,
    {
        let inputs: Vec<_> = self
            .iter()
            .map(|_| {
                Secret::sub_circuit_input(
                    for_circuit,
                    BooleanGate::Base(BaseGate::SubCircuitInput(ScalarDim)),
                )
            })
            .collect();
        f(&inputs[..])
    }

    fn size(&self) -> Self::Size {
        self.len()
    }

    fn flatten(self) -> Vec<Secret<BooleanGmw, Idx>> {
        self.to_vec()
    }
}

impl<Idx: GateIdx, const N: usize> SubCircuitInput for &[[Secret<BooleanGmw, Idx>; N]] {
    type Input<'a> = &'a [[Secret<BooleanGmw, Idx>; N]];
    type Size = usize;
    type Protocol = BooleanGmw;
    type Idx = Idx;

    fn with_input<F, R>(self, for_circuit: CircuitId, f: F) -> R
    where
        for<'a> F: FnOnce(Self::Input<'a>) -> R,
    {
        let inputs: Vec<_> = self
            .iter()
            .map(|_| {
                [(); N].map(|_| {
                    Secret::sub_circuit_input(
                        for_circuit,
                        BooleanGate::Base(BaseGate::SubCircuitInput(ScalarDim)),
                    )
                })
            })
            .collect();
        f(&inputs[..])
    }

    fn size(&self) -> Self::Size {
        self.len()
    }

    fn flatten(self) -> Vec<Secret<BooleanGmw, Idx>> {
        self.iter().flatten().cloned().collect()
    }
}

impl<P: Protocol, Idx: GateIdx> From<Secret<P, Idx>> for SubCircuitGate<Idx> {
    fn from(share: Secret<P, Idx>) -> Self {
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
        group_gates_iter(input)
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
