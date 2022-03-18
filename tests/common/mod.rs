use anyhow::Result;
use gmw_rs::circuit::Circuit;
use gmw_rs::common::BitVec;
use gmw_rs::executor::Executor;
use gmw_rs::transport::Tcp;
use petgraph::graph::IndexType;
use std::fmt::Debug;
use std::path::Path;
use tokio::task::spawn_blocking;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

// Code duplication due to https://github.com/rust-lang/cargo/issues/8379
pub fn init_tracing() -> tracing::dispatcher::DefaultGuard {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .set_default()
}

// TODO provide execute_bristol helper function

#[tracing::instrument(skip(inputs))]
pub async fn execute_bristol(
    bristol_file: impl AsRef<Path> + Debug,
    inputs: (BitVec, BitVec),
) -> Result<BitVec> {
    let path = bristol_file.as_ref().to_path_buf();
    let circuit = spawn_blocking(move || Circuit::load_bristol(path)).await??;
    execute_circuit(&circuit, inputs).await
}

// TODO provide execute_circuit helper function

#[tracing::instrument(skip_all)]
pub async fn execute_circuit<Idx: IndexType>(
    circuit: &Circuit<Idx>,
    (input_a, input_b): (BitVec, BitVec),
) -> Result<BitVec> {
    let (t1, t2) = Tcp::new_local_pair(None).await?;
    let mut ex1 = Executor::new(circuit, 0);
    let mut ex2 = Executor::new(circuit, 1);

    let h1 = ex1.execute(input_a, t1);
    let h2 = ex2.execute(input_b, t2);
    let (out1, out2) = futures::try_join!(h1, h2)?;
    Ok(out1 ^ out2)
}
