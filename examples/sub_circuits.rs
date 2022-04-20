use std::ops;

use tracing_subscriber::EnvFilter;

use gmw_rs::circuit::builder::CircuitBuilder;
use gmw_rs::circuit::CircuitLayerIter;
use gmw_rs::share_wrapper::{inputs, low_depth_reduce, ShareWrapper};
use gmw_rs::sub_circuit;

#[sub_circuit]
fn and_sc(input: &[ShareWrapper]) -> ShareWrapper {
    low_depth_reduce(input.iter().cloned(), ops::BitAnd::bitand).expect("Empty input")
}

#[sub_circuit]
fn or_sc(input: &[ShareWrapper]) -> ShareWrapper {
    low_depth_reduce(input.iter().cloned(), ops::BitOr::bitor).expect("Empty input")
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    CircuitBuilder::new().install();
    let input_shares = inputs(8);

    let and_outputs = input_shares
        .chunks_exact(4)
        .fold(vec![], |mut acc, input_chunk| {
            let output = and_sc(input_chunk);
            acc.push(output);
            acc
        });

    let or_out = or_sc(&and_outputs);

    (or_out ^ false).output();

    let circuit = CircuitBuilder::global_into_circuit();
    let layer_iter = CircuitLayerIter::new(&circuit);
    for layer in layer_iter {
        dbg!(layer);
    }

    // let circuit = circuit.into_base_circuit();
    // eprintln!("into base");
    // circuit.save_dot("sub_circuits.dot").unwrap();
}
