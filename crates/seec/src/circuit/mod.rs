use crate::SubCircuitGate;
pub use builder::SubCircCache;
use bytemuck::Pod;
use either::Either;
use num_integer::Integer;
use petgraph::adj::IndexType;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;
use std::num::NonZeroUsize;

pub mod base_circuit;
pub mod builder;
pub(crate) mod circuit_connections;
pub mod dyn_layers;
pub mod static_layers;
mod sub_circuit;

pub use crate::protocols::boolean_gmw::BooleanGate;
use crate::protocols::Gate;
pub use base_circuit::{BaseCircuit, GateId};
pub use builder::{CircuitBuilder, SharedCircuit};

pub type CircuitId = u32;

pub trait GateIdx:
    IndexType
    + Integer
    + Copy
    + Send
    + Sync
    + TryFrom<usize>
    + TryFrom<u32>
    + TryFrom<u16>
    + Into<GateId<Self>>
    + Pod
{
}

impl<T> GateIdx for T where
    T: IndexType
        + Integer
        + Copy
        + Send
        + Sync
        + TryFrom<usize>
        + TryFrom<u32>
        + TryFrom<u16>
        + Into<GateId<Self>>
        + Pod
{
}

pub type DefaultIdx = u32;

pub trait LayerIterable {
    type Layer;
    type LayerIter<'this>: Iterator<Item = Self::Layer>
    where
        Self: 'this;

    fn layer_iter(&self) -> Self::LayerIter<'_>;
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "\
    G: serde::Serialize + serde::de::DeserializeOwned,\
    Idx: GateIdx + Ord + Eq + Hash + serde::Serialize + serde::de::DeserializeOwned")]
pub enum ExecutableCircuit<G, Idx> {
    DynLayers(dyn_layers::Circuit<G, Idx>),
    StaticLayers(static_layers::Circuit<G, Idx>),
}

pub enum ExecutableLayer<'c, G, Idx: Hash + PartialEq + Eq> {
    DynLayer(dyn_layers::CircuitLayer<G, Idx>),
    StaticLayer(static_layers::ScLayerIterator<'c, G, Idx>),
}

impl<G, Idx> ExecutableCircuit<G, Idx> {
    pub fn interactive_count(&self) -> usize {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.interactive_count(),
            ExecutableCircuit::StaticLayers(circ) => circ.interactive_count(),
        }
    }

    // Returns interactive count and treats each interactive gate in a SIMD circuit as <simd_size>
    // interactive gates.
    pub fn interactive_count_times_simd(&self) -> usize {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.interactive_count_times_simd(),
            ExecutableCircuit::StaticLayers(circ) => circ.interactive_count_times_simd(),
        }
    }

    pub fn input_count(&self) -> usize {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.input_count(),
            ExecutableCircuit::StaticLayers(circ) => circ.input_count(),
        }
    }

    pub fn output_count(&self) -> usize {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.output_count(),
            ExecutableCircuit::StaticLayers(circ) => circ.output_count(),
        }
    }

    pub fn input_gates(&self) -> &[GateId<Idx>] {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.get_circ(0).input_gates(),
            ExecutableCircuit::StaticLayers(circ) => circ.input_gates(),
        }
    }

    pub fn output_gates(&self) -> &[GateId<Idx>] {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.get_circ(0).output_gates(),
            ExecutableCircuit::StaticLayers(circ) => circ.output_gates(),
        }
    }

    pub fn simd_size(&self, circ_id: CircuitId) -> Option<NonZeroUsize> {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.get_circ(circ_id).simd_size(),
            ExecutableCircuit::StaticLayers(circ) => circ.get_circ(circ_id).simd_size(),
        }
    }
}

impl<G: Gate, Idx: GateIdx> ExecutableCircuit<G, Idx> {
    pub fn precompute_layers(self) -> Self {
        match self {
            ExecutableCircuit::DynLayers(circ) => {
                ExecutableCircuit::StaticLayers(circ.precompute_layers())
            }
            compressed @ ExecutableCircuit::StaticLayers(_) => compressed,
        }
    }

    pub fn gate_count(&self) -> usize {
        match self {
            ExecutableCircuit::DynLayers(circ) => circ.gate_count(),
            ExecutableCircuit::StaticLayers(circ) => circ.gate_count(),
        }
    }

    /// Returns iterator over tuples of (gate_count, simd_size) for each sub_circuit
    pub fn gate_counts(&self) -> impl Iterator<Item = (usize, Option<NonZeroUsize>)> + '_ {
        match self {
            ExecutableCircuit::DynLayers(circ) => Either::Left(
                circ.iter_circs()
                    .map(|bc| (bc.gate_count(), bc.simd_size())),
            ),
            ExecutableCircuit::StaticLayers(circ) => Either::Right(circ.gate_counts()),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        match self {
            ExecutableCircuit::DynLayers(circ) => Either::Left(circ.iter()),
            ExecutableCircuit::StaticLayers(circ) => Either::Right(circ.iter()),
        }
    }

    pub fn iter_with_parents(
        &self,
    ) -> impl Iterator<
        Item = (
            G,
            SubCircuitGate<Idx>,
            impl Iterator<Item = SubCircuitGate<Idx>> + '_,
        ),
    > + '_ {
        match self {
            ExecutableCircuit::DynLayers(circ) => {
                let iter = circ
                    .iter()
                    .map(|(g, idx)| (g, idx, Either::Left(circ.parent_gates(idx))));
                Either::Left(iter)
            }
            ExecutableCircuit::StaticLayers(circ) => Either::Right(
                circ.iter_with_parents()
                    .map(|(g, idx, parents)| (g, idx, Either::Right(parents))),
            ),
        }
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        match self {
            ExecutableCircuit::DynLayers(circ) => Either::Left(circ.interactive_iter()),
            ExecutableCircuit::StaticLayers(circ) => Either::Right(circ.interactive_iter()),
        }
    }

    pub fn interactive_with_parents_iter(
        &self,
    ) -> impl Iterator<
        Item = (
            G,
            SubCircuitGate<Idx>,
            impl Iterator<Item = SubCircuitGate<Idx>> + '_,
        ),
    > + '_ {
        match self {
            ExecutableCircuit::DynLayers(circ) => Either::Left(
                circ.interactive_iter()
                    .map(|(g, idx)| (g, idx, Either::Left(circ.parent_gates(idx)))),
            ),
            ExecutableCircuit::StaticLayers(circ) => Either::Right(
                circ.interactive_with_parents_iter()
                    .map(|(g, idx, parents)| (g, idx, Either::Right(parents))),
            ),
        }
    }

    pub fn layer_iter(&self) -> impl Iterator<Item = ExecutableLayer<'_, G, Idx>> + '_ {
        match self {
            ExecutableCircuit::DynLayers(circ) => {
                Either::Left(dyn_layers::CircuitLayerIter::new(circ).map(ExecutableLayer::DynLayer))
            }
            ExecutableCircuit::StaticLayers(circ) => Either::Right(
                static_layers::LayerIterator::new(circ).map(ExecutableLayer::StaticLayer),
            ),
        }
    }
}

impl<G: Clone, Idx: GateIdx> Clone for ExecutableCircuit<G, Idx> {
    fn clone(&self) -> Self {
        match self {
            Self::DynLayers(circ) => Self::DynLayers(circ.clone()),
            Self::StaticLayers(circ) => Self::StaticLayers(circ.clone()),
        }
    }
}

impl<'c, G: Gate, Idx: GateIdx> ExecutableLayer<'c, G, Idx> {
    pub fn interactive_count(&self) -> usize {
        match self {
            ExecutableLayer::DynLayer(layer) => layer.interactive_iter().count(),
            ExecutableLayer::StaticLayer(layer_iter) => layer_iter
                .clone()
                .map(|layer| layer.interactive_count())
                .sum(),
        }
    }

    pub fn interactive_count_times_simd(&self) -> usize {
        match self {
            ExecutableLayer::DynLayer(layer) => layer.interactive_count_times_simd(),
            ExecutableLayer::StaticLayer(layer_iter) => layer_iter.interactive_count_times_simd(),
        }
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        match self {
            ExecutableLayer::DynLayer(layer) => Either::Left(layer.interactive_iter()),
            ExecutableLayer::StaticLayer(layer_iter) => {
                Either::Right(layer_iter.clone().flat_map(|layer| {
                    layer
                        .interactive_iter()
                        .map(move |(g, id)| (g, SubCircuitGate::new(layer.sc_id, id)))
                }))
            }
        }
    }

    pub fn interactive_gates<'s>(&'s self) -> impl Iterator<Item = &'s [G]> + Clone
    where
        'c: 's,
    {
        match self {
            ExecutableLayer::DynLayer(layer) => Either::Left(
                layer
                    .sc_layers
                    .iter()
                    .map(|(_, _, base_layer)| &base_layer.interactive_gates[..]),
            ),
            ExecutableLayer::StaticLayer(layer_iter) => {
                Either::Right(layer_iter.clone().map(|layer| layer.interactive_gates()))
            }
        }
    }

    pub fn interactive_indices<'s>(
        &'s self,
    ) -> impl Iterator<Item = (CircuitId, &'s [GateId<Idx>])> + Clone
    where
        'c: 's,
    {
        match self {
            ExecutableLayer::DynLayer(layer) => Either::Left(
                layer
                    .sc_layers
                    .iter()
                    .map(|(sc_id, _, base_layer)| (*sc_id, &base_layer.interactive_ids[..])),
            ),
            ExecutableLayer::StaticLayer(layer_iter) => Either::Right(
                layer_iter
                    .clone()
                    .map(|layer| (layer.sc_id, layer.interactive_indices())),
            ),
        }
    }

    pub fn interactive_parents_iter<'s>(
        &'s self,
        circ: &'c ExecutableCircuit<G, Idx>,
    ) -> impl Iterator<Item = impl Iterator<Item = SubCircuitGate<Idx>> + 'c> + 's
    where
        'c: 's,
    {
        match (self, circ) {
            (ExecutableLayer::DynLayer(layer), ExecutableCircuit::DynLayers(circ)) => Either::Left(
                layer
                    .interactive_iter()
                    .map(|(_, id)| Either::Left(circ.parent_gates(id))),
            ),
            (ExecutableLayer::StaticLayer(layer_iter), ExecutableCircuit::StaticLayers(circ)) => {
                Either::Right(
                    layer_iter
                        .clone()
                        .flat_map(|layer| layer.interactive_parents_iter(circ).map(Either::Right)),
                )
            }
            _ => panic!("Non matching ExecutableLayer and ExecutableCircuit"),
        }
    }

    pub fn non_interactive_iter(
        &self,
    ) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        match self {
            ExecutableLayer::DynLayer(layer) => Either::Left(layer.non_interactive_iter()),
            ExecutableLayer::StaticLayer(layer_iter) => {
                Either::Right(layer_iter.clone().flat_map(|layer| {
                    layer
                        .non_interactive_iter()
                        .map(move |(g, id)| (g, SubCircuitGate::new(layer.sc_id, id)))
                }))
            }
        }
    }

    pub fn non_interactive_with_parents_iter<'s>(
        &'s self,
        circ: &'c ExecutableCircuit<G, Idx>,
    ) -> impl Iterator<
        Item = (
            (G, SubCircuitGate<Idx>),
            impl Iterator<Item = SubCircuitGate<Idx>> + 'c,
        ),
    > + 's
    where
        'c: 's,
    {
        match (self, circ) {
            (ExecutableLayer::DynLayer(layer), ExecutableCircuit::DynLayers(circ)) => Either::Left(
                layer
                    .non_interactive_iter()
                    .map(|(g, id)| ((g, id), Either::Left(circ.parent_gates(id)))),
            ),
            (ExecutableLayer::StaticLayer(layer_iter), ExecutableCircuit::StaticLayers(circ)) => {
                Either::Right(layer_iter.clone().flat_map(|layer| {
                    let non_inter_active_iter = layer
                        .non_interactive_iter()
                        .map(move |(g, id)| (g, SubCircuitGate::new(layer.sc_id, id)));

                    iter::zip(
                        non_inter_active_iter,
                        layer.non_interactive_parents_iter(circ).map(Either::Right),
                    )
                }))
            }
            _ => panic!("Non matching ExecutableLayer and ExecutableCircuit"),
        }
    }

    // Returns freeable SIMD gates
    pub(crate) fn freeable_simd_gates(&self) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        match self {
            ExecutableLayer::DynLayer(layer) => Either::Left(layer.freeable_simd_gates()),
            ExecutableLayer::StaticLayer(_) => Either::Right(iter::empty()),
        }
    }

    /// Split layer into (scalar, simd) gates
    pub(crate) fn split_simd(self) -> (Self, Self) {
        match self {
            Self::DynLayer(layer) => {
                let (scalar, simd) = layer.split_simd();
                (Self::DynLayer(scalar), Self::DynLayer(simd))
            }
            Self::StaticLayer(layer_iter) => {
                let (scalar, simd) = layer_iter.split_simd();
                (Self::StaticLayer(scalar), Self::StaticLayer(simd))
            }
        }
    }
}
