use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use seec_channel::Channel;
use seec_channel_macros::sub_channels_for;
use tokio::runtime::Runtime;

// Here we have an async function to benchmark
async fn layer(layers: usize, ch: &mut Channel<Vec<u8>>) {
    let data = vec![0xFF_u8; 1];

    for _layer in 0..layers {
        ch.0.send(data.clone()).await.unwrap();
        ch.1.recv().await.unwrap().unwrap();
    }
}

fn bench_layers(c: &mut Criterion) {
    let layers: usize = 1000;

    let rt = Runtime::new().unwrap();

    let (mut ch0, mut ch1) = rt.block_on(async {
        let (mut base_ch0, mut base_ch1) = seec_channel::tcp::new_local_pair(None)
            .await
            .expect("unable to create local channel");

        let f0 = sub_channels_for!(&mut base_ch0.0, &mut base_ch0.2, 16, Vec<u8>);
        let f1 = sub_channels_for!(&mut base_ch1.0, &mut base_ch1.2, 16, Vec<u8>);
        let (ch0, ch1) = tokio::try_join!(f0, f1).unwrap();
        (ch0, ch1)
    });

    c.bench_with_input(
        BenchmarkId::new("bench_layers", layers),
        &layers,
        |b, &layers| {
            b.iter(|| {
                rt.block_on(async {
                    tokio::join!(layer(layers, &mut ch0), layer(layers, &mut ch1));
                })
            });
        },
    );
}

criterion_group!(benches, bench_layers);
criterion_main!(benches);
