use criterion::criterion_main;

use benchmarks::*;

mod benchmarks;
criterion_main!(layer_iter::benches,);
