use crate::circuit::{Circuit, Gate};
use itertools::Itertools;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub(crate) fn create_and_tree(depth: u32) -> Circuit {
    let total_nodes = 2_u32.pow(depth);
    let mut layer_count = total_nodes / 2;
    let mut circuit = Circuit::new();

    let mut previous_layer: Vec<_> = (0..layer_count)
        .map(|_| circuit.add_gate(Gate::Input))
        .collect();
    while layer_count > 1 {
        layer_count /= 2;
        previous_layer = previous_layer
            .into_iter()
            .tuples()
            .map(|(from_a, from_b)| circuit.add_wired_gate(Gate::And, &[from_a, from_b]))
            .collect();
    }
    debug_assert_eq!(1, previous_layer.len());
    circuit.add_wired_gate(Gate::Output, &[previous_layer[0]]);
    circuit
}

/// Initializes tracing subscriber with EnvFilter for usage in tests. This should be the first call
/// in each test, with the returned value being assigned to a variable to prevent dropping.  
/// Output can be configured via RUST_LOG env variable as explained
/// [here](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/struct.EnvFilter.html)
///
/// ```
/// fn some_test() {
///     let _guard = init_tracing();
/// }
/// ```
pub(crate) fn init_tracing() -> tracing::dispatcher::DefaultGuard {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .set_default()
}
