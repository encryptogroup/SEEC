use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use quic_serde_stream::testing::local_conn;
use quic_serde_stream::Id;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::join;

fn criterion_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let (mut server, _client) = rt.block_on(local_conn()).unwrap();
    let id = AtomicU64::new(0);
    let id = &id;
    c.bench_function("create byte sub stream", |b| {
        b.to_async(&rt).iter_batched(
            || server.clone(),
            |mut server| async move {
                let id = id.fetch_add(1, Ordering::Relaxed);
                server.byte_sub_stream(Id::new(id)).await;
            },
            BatchSize::SmallInput,
        )
    });

    id.store(0, Ordering::Relaxed);
    let (server, client) = rt.block_on(local_conn()).unwrap();

    c.bench_function("byte ping pong", |b| {
        let mut server = &server;
        let mut client = &client;
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut server = server.clone();
            let mut client = client.clone();
            let id = id.fetch_add(1, Ordering::Relaxed);
            let (mut snd_s, mut rcv_s) = server.byte_sub_stream(Id::new(id)).await;
            let (mut snd_c, mut rcv_c) = client.byte_sub_stream(Id::new(id)).await;
            let now = Instant::now();

            for _ in 0..iters {
                join!(
                    async {
                        snd_s.write_all(b"hello").await.unwrap();
                    },
                    async {
                        snd_c.write_all(b"hello").await.unwrap();
                    }
                );
                join!(
                    async {
                        let mut buf = [0; 5];
                        rcv_s.read_exact(&mut buf).await.unwrap();
                    },
                    async {
                        let mut buf = [0; 5];
                        rcv_c.read_exact(&mut buf).await.unwrap();
                    }
                );
            }
            now.elapsed()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
