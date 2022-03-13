use crate::circuit::{Circuit, Gate};
use itertools::Itertools;

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
            .map(|(from_a, from_b)| {
                let to = circuit.add_gate(Gate::And);
                circuit.add_wire(from_a, to);
                circuit.add_wire(from_b, to);
                to
            })
            .collect();
    }
    debug_assert_eq!(1, previous_layer.len());
    let out = circuit.add_gate(Gate::Output);
    circuit.add_wire(previous_layer[0], out);
    circuit
}
