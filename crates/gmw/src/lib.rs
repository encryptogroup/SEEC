pub use sub_circuit_impl::sub_circuit;

pub use circuit::builder::{
    CircuitBuilder, SharedCircuit, SubCircuitGate, SubCircuitInput, SubCircuitOutput,
};
pub use circuit::Circuit;
pub use circuit::{Gate, GateId};

pub mod bristol;
pub mod circuit;
pub mod common;
pub mod errors;
pub mod evaluate;
pub mod executor;
pub mod mul_triple;
#[cfg(feature = "_integration_tests")]
#[doc(hidden)]
/// Do **not** use items from this module. They are intended for integration tests and must
/// therefore be public.
pub mod private_test_utils;
pub mod share_wrapper;
pub(crate) mod utils;

pub type Sender<T> = remoc::rch::mpsc::Sender<T, remoc::codec::Bincode>;
pub type Receiver<T> = remoc::rch::mpsc::Receiver<T, remoc::codec::Bincode>;
