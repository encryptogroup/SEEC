//! # SEEC Executes Enormous Circuits
//!
//! This framework implements secure 2-party secret-sharing-based multi party computation protocols.
//! Currently, we implement the Boolean and arithmetic versions of GMW87 with multiplication triple preprocessing.
//! Additionally, we implement the Boolean part of the ABY2.0 protocol.
//!
//! ## Secure Multi-Party Computation
//! In secure multi-party computation (MPC), there are n parties, each with their private input x_i.
//! Given a public function f(x_1, ..., x_n), the parties execute a protocol π that correctly and
//! securely realizes the functionality f. In other words, at the end of the protocol, the parties know
//! the output of f, but have no information about the input of the other parties other than what is revealed
//! by the output itself. Currently, SEEC is limited to the n = 2 party case, also known as secure
//! two-party computation. We hope to extend this in the future to n parties.
//!
//! ### Security
//! The two most prevalent security models are
//! - semi-honest security, where an attacker can corrupt parties, but they follow the protocol as specified.
//! - malicious security, where corrupted parties can arbitrarily deviate from the protocol.
//!
//! SEEC currently only implements semi-honestly secure protocols (GMW, ABY2.0).
//!
//! ## Using SEEC
//! To use SEEC, you need a recent stable Rust toolchain (nightly on ARM[^nightly]). The easiest way to install Rust
//! is via the official toolchain installer [rustup](https://rustup.rs/).
//!
//! Create a new Rust project using cargo `cargo new --bin seec-test`, and add SEEC as a dependency to your
//! `Cargo.toml`
//! ```shell
//! // seec-test/Cargo.toml
//! [dependencies]
//! seec = { git = "ssh://git@git.rwth-aachen.de/encrypto/code/rnieminen/rust-framework.git" }
//! ```
//! (Note: currently SEEC is only available as a private git repository. Make sure that your
//! [ssh-agent](https://linux.die.net/man/1/ssh-agent)) is started. More information [here](https://doc.rust-lang.org/cargo/appendix/git-authentication.html)
//!
//! The following annotated example of using SEEC is also located at `crates/seec/examples/simple.rs`.
//! You can run it using `cargo run --example simple`.
//!```ignore,toml
#![doc = include_str!("../examples/simple.rs")]
//!```
//!
//! [^nightly]: If you are on an ARM platform, you need to install a recent nightly toolchain. After
//!             installing, we advise to issue `rustup override set nightly` in the top-level directory.
pub use circuit::builder::{
    CircuitBuilder, SharedCircuit, SubCircuitGate, SubCircuitInput, SubCircuitOutput,
};
pub use circuit::dyn_layers::Circuit;
pub use circuit::GateId;
pub use parse::bristol;
pub use protocols::boolean_gmw::BooleanGate;
pub use seec_channel as channel;
pub use seec_macros::sub_circuit;

#[cfg(feature = "bench-api")]
pub mod bench;
pub mod circuit;
pub mod common;
pub mod errors;
pub mod evaluate;
pub mod executor;
pub mod mul_triple;
pub mod parse;
#[cfg(feature = "_integration_tests")]
#[doc(hidden)]
/// Do **not** use items from this module. They are intended for integration tests and must
/// therefore be public.
pub mod private_test_utils;
pub mod protocols;
pub mod secret;
pub(crate) mod utils;
