use crate::circuit::{CircuitId, GateIdx};
use crate::utils::RangeInclusiveStartWrapper;
use crate::{GateId, SubCircuitGate};
use petgraph::adj::IndexType;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::{Eq, Ord};
use std::collections::{BTreeMap, Bound, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::RangeInclusive;
use tracing::trace;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound = "Idx: Ord + Eq + Hash + serde::Serialize + serde::de::DeserializeOwned")]
pub(crate) struct CrossCircuitConnections<Idx> {
    pub(crate) one_to_one: OneToOneConnections<Idx>,
    pub(crate) range_connections: RangeConnections<Idx>,
}

type OneToOneMap<Idx, const BUF_SIZE: usize> =
    HashMap<SubCircuitGate<Idx>, SmallVec<[SubCircuitGate<Idx>; BUF_SIZE]>>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound = "Idx: Eq + Hash + serde::Serialize + serde::de::DeserializeOwned")]
pub(crate) struct OneToOneConnections<Idx> {
    pub(crate) outgoing: OneToOneMap<Idx, 1>,
    pub(crate) incoming: OneToOneMap<Idx, 2>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound = "Idx: Ord + Eq + Hash + serde::Serialize + serde::de::DeserializeOwned")]
pub(crate) struct RangeConnections<Idx> {
    pub(crate) incoming: RangeSubCircuitConnections<Idx>,
    pub(crate) outgoing: RangeSubCircuitConnections<Idx>,
}

type FromRange<Idx> = RangeInclusiveStartWrapper<SubCircuitGate<Idx>>;
type ToRanges<Idx> = SmallVec<[RangeInclusive<SubCircuitGate<Idx>>; 1]>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound = "Idx: Ord + Eq + Hash + serde::Serialize + serde::de::DeserializeOwned")]
pub struct RangeSubCircuitConnections<Idx> {
    map: HashMap<CircuitId, BTreeMap<FromRange<Idx>, ToRanges<Idx>>>,
}

impl<Idx: GateIdx> CrossCircuitConnections<Idx> {
    pub(crate) fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.one_to_one
            .parent_gates(id)
            .chain(self.range_connections.parent_gates(id))
    }

    pub(crate) fn outgoing_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.one_to_one
            .outgoing_gates(id)
            .chain(self.range_connections.outgoing_gates(id))
    }

    pub(crate) fn clone_incoming(&self) -> Self {
        Self {
            one_to_one: OneToOneConnections {
                outgoing: Default::default(),
                incoming: self.one_to_one.incoming.clone(),
            },
            range_connections: RangeConnections {
                incoming: self.range_connections.incoming.clone(),
                outgoing: Default::default(),
            },
        }
    }
}

impl<Idx: GateIdx> OneToOneConnections<Idx> {
    pub(crate) fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.incoming
            .get(&id)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }

    pub(crate) fn outgoing_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.outgoing
            .get(&id)
            .map(|sv| sv.iter().copied())
            .into_iter()
            .flatten()
    }
}

impl<Idx> RangeConnections<Idx>
where
    Idx: GateIdx,
{
    pub(crate) fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        let range_conns = self.incoming.get_mapped_ranges(id);
        range_conns.flat_map(move |(to_range, from_ranges)| {
            if !to_range.contains(&id) {
                unreachable!("to_range is wrong");
            }
            let offset = id.gate_id.0 - to_range.start().gate_id.0;

            from_ranges.iter().map(move |from_range| {
                let from_range_start = from_range.start();
                let from_gate_id = (from_range_start.gate_id.0 + offset).into();
                SubCircuitGate::new(from_range_start.circuit_id, from_gate_id)
            })
        })
    }

    #[tracing::instrument(level = "trace", skip(self))]
    #[allow(clippy::let_with_type_underscore)] // false positive https://github.com/rust-lang/rust-clippy/issues/10498
    pub(crate) fn outgoing_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        let range_conns = self.outgoing.get_mapped_ranges(id);
        range_conns.flat_map(move |(from_range, to_ranges)| {
            trace!(?from_range, ?to_ranges, "range_conns_outgoing_gates");
            if !from_range.contains(&id) {
                unreachable!("from_range does not contain id")
            }
            let offset = id.gate_id.0 - from_range.start().gate_id.0;
            to_ranges.iter().map(move |to_range| {
                let to_gate_id = (to_range.start().gate_id.0 + offset).into();
                SubCircuitGate::new(to_range.start().circuit_id, to_gate_id)
            })
        })
    }
}

impl<Idx: Ord + Copy + IndexType + Debug> RangeSubCircuitConnections<Idx> {
    /// # Panics
    /// This function asserts that the from range is not a strictly contained in an
    /// already stored range.
    /// E.g. When (3..=9) is stored, it is illegal to store (4..=6)
    /// It **is** allowed, to store ranges with the same start but differing lengths, e.g. (3..=5)
    pub(crate) fn insert(
        &mut self,
        from: RangeInclusive<SubCircuitGate<Idx>>,
        to: RangeInclusive<SubCircuitGate<Idx>>,
    ) {
        assert!(
            from.start() <= from.end(),
            "from.start() must be <= than end()"
        );
        let from_circuit_id = from.start().circuit_id;
        let new_from_start_wrapper = RangeInclusiveStartWrapper::new(from.clone());
        let bmap = self.map.entry(from_circuit_id).or_default();
        let potential_conflict = bmap
            .range((
                Bound::Unbounded,
                Bound::Included(new_from_start_wrapper.clone()),
            ))
            .next_back();
        if let Some((potential_conflict, _)) = potential_conflict {
            assert!(
                !(potential_conflict.range.start() < from.start()
                    && from.end() <= potential_conflict.range.end()),
                "RangeSubCircuitConnections can't store a range which is a \
                strict sub range of an already stored one"
            );
        }

        bmap.entry(new_from_start_wrapper).or_default().push(to);
    }

    // TODO is it possible to provide an API that takes multiple (sorted?) gates and more
    //  efficiently returns the mapped ranges?
    pub(crate) fn get_mapped_ranges(
        &self,
        gate: SubCircuitGate<Idx>,
    ) -> impl Iterator<
        Item = (
            RangeInclusive<SubCircuitGate<Idx>>,
            &[RangeInclusive<SubCircuitGate<Idx>>],
        ),
    > {
        let start_wrapper = RangeInclusiveStartWrapper::new(
            gate..=SubCircuitGate::new(gate.circuit_id, GateId(<Idx as IndexType>::max())),
        );
        self.map
            .get(&gate.circuit_id)
            .into_iter()
            .flat_map(move |bmap| {
                bmap.range((Bound::Unbounded, Bound::Included(start_wrapper.clone())))
                    .rev()
                    .take_while(move |(range_wrapper, _to_ranges)| {
                        *range_wrapper.range.start() <= gate && gate <= *range_wrapper.range.end()
                    })
                    .map(|(from_range_wrapper, to_vec)| {
                        (from_range_wrapper.range.clone(), to_vec.as_slice())
                    })
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::circuit_connections::RangeSubCircuitConnections;
    use crate::{GateId, SubCircuitGate};

    #[test]
    fn test_range_connections_simple() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let to_range =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));

        rc.insert(from_range.clone(), to_range.clone());

        for (ret_from_range, ret_to_ranges) in
            rc.get_mapped_ranges(SubCircuitGate::new(0, GateId(0_u32)))
        {
            assert_eq!(ret_from_range, from_range);
            assert_eq!(ret_to_ranges[0], to_range);
            assert_eq!(ret_to_ranges.len(), 1);
        }
    }

    #[test]
    fn test_range_connections_overlapping() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(50_u32))..=SubCircuitGate::new(0, GateId(100_u32));
        let to_range =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));

        rc.insert(from_range_0.clone(), to_range.clone());
        rc.insert(from_range_1.clone(), to_range.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(50_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 2);
        assert_eq!(mapped_ranges[1], (from_range_0, &[to_range.clone()][..]));
        assert_eq!(mapped_ranges[0], (from_range_1, &[to_range][..]));
    }

    #[test]
    fn test_range_connections_inside_each_other_same_start() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(20_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(120_u32));

        rc.insert(from_range_0.clone(), to_range_0.clone());
        rc.insert(from_range_1.clone(), to_range_1.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(20_u32)))
            .collect();
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 2);
        assert_eq!(mapped_ranges[0], (from_range_0, &[to_range_0][..]));
        assert_eq!(mapped_ranges[1], (from_range_1, &[to_range_1][..]));
    }

    #[test]
    fn test_range_connections_map_to_multiple() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(2, GateId(100_u32))..=SubCircuitGate::new(2, GateId(150_u32));

        rc.insert(from_range_0.clone(), to_range_0.clone());
        rc.insert(from_range_0.clone(), to_range_1.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(20_u32)))
            .collect();
        assert_eq!(mapped_ranges.len(), 1);
        assert_eq!(
            mapped_ranges[0],
            (from_range_0, &[to_range_0, to_range_1][..])
        );
    }

    #[test]
    fn test_range_connections_regression() {
        let mut rc = RangeSubCircuitConnections::default();

        let from_range_0 = SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(8_u32),
        }..=SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(167_u32),
        };
        let to_range_0 = SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(0_u32),
        }..=SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(159_u32),
        };
        rc.insert(from_range_0.clone(), to_range_0.clone());
        let from_range_1 = SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(280_u32),
        }..=SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(339_u32),
        };
        let to_range_1 = SubCircuitGate {
            circuit_id: 2,
            gate_id: GateId(0_u32),
        }..=SubCircuitGate {
            circuit_id: 2,
            gate_id: GateId(59_u32),
        };
        rc.insert(from_range_1, to_range_1);
        let from_range_2 = SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(8_u32),
        }..=SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(87_u32),
        };
        let to_range_2 = SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(0_u32),
        }..=SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(79_u32),
        };
        rc.insert(from_range_2.clone(), to_range_2.clone());
        let from_range_4 = SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(96_u32),
        }..=SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(175_u32),
        };
        let to_range_4 = SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(80_u32),
        }..=SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(159_u32),
        };
        rc.insert(from_range_4, to_range_4);

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(87_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 2);
        assert_eq!(
            mapped_ranges[0],
            (from_range_0.clone(), &[to_range_0.clone()][..])
        );
        assert_eq!(mapped_ranges[1], (from_range_2, &[to_range_2][..]));

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(88_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 1);
        assert_eq!(mapped_ranges[0], (from_range_0, &[to_range_0][..]));
    }

    #[test]
    #[should_panic]
    fn test_range_connections_inside_each_other_illegal() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(10_u32))..=SubCircuitGate::new(0, GateId(20_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(1, GateId(110_u32))..=SubCircuitGate::new(1, GateId(120_u32));

        rc.insert(from_range_0, to_range_0);
        rc.insert(from_range_1, to_range_1);
    }

    #[test]
    fn test_range_connections_simd() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));

        rc.insert(from_range_0.clone(), to_range_0.clone());
        rc.insert(from_range_1, to_range_1.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(40_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 1);
        assert_eq!(
            mapped_ranges[0],
            (from_range_0, &[to_range_0, to_range_1][..])
        );
    }
}
