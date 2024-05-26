use crate::protocols::{Dimension, Gate, Plain, ScalarDim};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use tracing::debug;

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum BaseGate<Plain, Dim = ScalarDim> {
    Output(Dim),
    Input(Dim),
    /// Input from a sub circuit called within a circuit.
    SubCircuitInput(Dim),
    /// Output from this circuit into another sub circuit
    SubCircuitOutput(Dim),
    ConnectToMain(Dim),
    /// Connects a sub circuit to the main circuit and selects the i'th individual value from
    /// the SIMD output
    ConnectToMainFromSimd((Dim, u32)),
    /// Identity gate, simply outputs its input
    Identity,
    Constant(Plain),
    Debug,
}

impl<P: Plain, Dim: Dimension> Gate<P> for BaseGate<P, Dim> {
    type DimTy = Dim;

    fn is_interactive(&self) -> bool {
        false
    }

    fn input_size(&self) -> usize {
        1
    }

    fn as_base_gate(&self) -> Option<&BaseGate<P, Self::DimTy>> {
        Some(self)
    }

    fn wrap_base_gate(base_gate: BaseGate<P, Self::DimTy>) -> Self {
        base_gate
    }
}

impl<Plain: Debug, Dim: Debug> BaseGate<Plain, Dim> {
    pub fn default_evaluate<Share: Debug>(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = Share>,
    ) -> Share {
        match self {
            Self::Constant(_) => {
                panic!("Constant base must be handled in executor")
            }
            Self::Output(_)
            | Self::Input(_)
            | Self::SubCircuitInput(_)
            | Self::SubCircuitOutput(_)
            | Self::ConnectToMain(_)
            | Self::Identity => inputs
                .next()
                .unwrap_or_else(|| panic!("Empty input for {self:?}")),
            Self::ConnectToMainFromSimd(_) => {
                panic!("BaseGate::evaluate_non_interactive called on SIMD gates")
            }
            Self::Debug => {
                let inp = inputs.next().expect("Empty input");
                debug!("BaseGate::Debug party_id={party_id}: {inp:?}");
                inp
            }
        }
    }

    pub fn default_evaluate_simd<'a, SimdShare: Clone + Debug + 'a>(
        &self,
        party_id: usize,
        mut inputs: impl Iterator<Item = &'a SimdShare>,
    ) -> SimdShare {
        match self {
            BaseGate::Constant(_constant) => {
                panic!("Constant base gate is handled in executor")
            }
            BaseGate::Output(_)
            | BaseGate::Input(_)
            | BaseGate::ConnectToMain(_)
            | BaseGate::SubCircuitInput(_)
            | BaseGate::SubCircuitOutput(_)
            | BaseGate::ConnectToMainFromSimd(_)
            | BaseGate::Identity => inputs.next().expect("Missing input to {self:?}").clone(),
            BaseGate::Debug => {
                let inp = inputs.next().expect("Empty input");
                debug!("BaseGate::Debug party_id={party_id}: {inp:?}");
                inp.clone()
            }
        }
    }
}
