use std::ops;

use criterion::{criterion_group, BenchmarkId, Criterion};

use gmw::circuit::dyn_layers::{Circuit, CircuitLayerIter};
use gmw::secret::{inputs, low_depth_reduce, Secret};
use gmw::{sub_circuit, BooleanGate, CircuitBuilder};

fn build_circuit(keyword_size: usize, target_text_size: usize) -> Circuit {
    CircuitBuilder::<BooleanGate, u32>::new().install();
    let keyword: Vec<_> = (0..keyword_size)
        .map(|_| inputs(8).try_into().unwrap())
        .collect();
    let target_text: Vec<_> = (0..target_text_size)
        .map(|_| inputs(8).try_into().unwrap())
        .collect();

    create_search_circuit(&keyword, &target_text);

    let circ = CircuitBuilder::<BooleanGate, u32>::global_into_circuit();
    circ
}

fn create_search_circuit(keyword: &[[Secret; 8]], target_text: &[[Secret; 8]]) -> Secret {
    /*
     * Calculate the number of positions we need to compare. E.g., if search_keyword
     * is "key" and target_text is "target", we must do 4 comparison:
     *
     * target , target , target , target
     * ^^^       ^^^       ^^^       ^^^
     * key       key       key       key
     */
    let result_bits: Vec<_> = target_text
        .windows(keyword.len())
        .map(|target| comparison_circuit(keyword, target))
        .collect();
    // Finally, use OR tree to get the final answer of whether any of the comparisons was a match
    or_sc(&result_bits)
}

#[sub_circuit]
fn comparison_circuit(keyword: &[[Secret; 8]], target_text: &[[Secret; 8]]) -> Secret {
    const CHARACTER_BIT_LEN: usize = 6; // Follows from the special PrivMail encoding
    let splitted_keyword: Vec<_> = keyword
        .iter()
        .map(|c| c.iter().cloned().take(CHARACTER_BIT_LEN))
        .flatten()
        .collect();
    let splitted_text: Vec<_> = target_text
        .iter()
        .map(|c| c.iter().cloned().take(CHARACTER_BIT_LEN))
        .flatten()
        .collect();

    let res: Vec<_> = splitted_keyword
        .into_iter()
        .zip(splitted_text)
        .map(|(k, t)| !(k.clone() ^ t))
        .collect();

    low_depth_reduce(res, ops::BitAnd::bitand).expect("Empty input")
}

#[sub_circuit]
fn or_sc(input: &[Secret]) -> Secret {
    low_depth_reduce(input.to_owned(), ops::BitOr::bitor).expect("Empty input")
}

fn bench_circuit_layer_iter(c: &mut Criterion) {
    let mut grp = c.benchmark_group("circuit_layer_iter");
    const BASE_SIZE: usize = 128;
    for target_size in [
        BASE_SIZE,
        BASE_SIZE * 2,
        BASE_SIZE * 4,
        BASE_SIZE * 8,
        BASE_SIZE * 16,
    ] {
        let circ = build_circuit(8, target_size);
        grp.bench_with_input(
            BenchmarkId::from_parameter(target_size),
            &target_size,
            |b, _| {
                b.iter(|| {
                    CircuitLayerIter::new(&circ).for_each(|_| ());
                });
            },
        );
    }
    grp.finish();
}

criterion_group!(benches, bench_circuit_layer_iter);
