use anyhow::Context;
use clap::Parser;
use gmw::mul_triple;
use gmw::mul_triple::storage::MTStorage;
use mpc_channel::sub_channels_for;
use rand::rngs::OsRng;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::Instant;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Clone, Debug)]
struct Args {
    #[clap(short, long)]
    num: usize,
    #[clap(short, long)]
    batch_size: usize,
    #[clap(short, long)]
    output: PathBuf,
    #[clap(long, default_value = "127.0.0.1:7742")]
    server: SocketAddr,
    #[clap(long)]
    id: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging, see top of file for instructions on how to get output.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let (mut sender, bytes_written, mut receiver, bytes_read) = match args.id {
        0 => mpc_channel::tcp::listen(&args.server).await?,
        1 => mpc_channel::tcp::connect_with_timeout(&args.server, Duration::from_secs(120)).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    let (mt_ch, mut sync_ch) = sub_channels_for!(
        &mut sender,
        &mut receiver,
        128,
        mul_triple::boolean::ot_ext::DefaultMsg,
        mpc_channel::SyncMsg
    )
    .await
    .context("sub-channel establishment")?;

    let mtp =
        mul_triple::boolean::ot_ext::OtMTProvider::new_with_default_ot_ext(OsRng, mt_ch.0, mt_ch.1);

    let mut mt_storage = MTStorage::create(&args.output).context("create mt storage")?;
    mt_storage.set_batch_size(args.batch_size);
    let now = Instant::now();
    mt_storage
        .store_mts(args.num, mtp)
        .await
        .context("unable to precompute and store MTs")?;
    info!(
        elapsed_ms = now.elapsed().as_millis(),
        bytes_written = bytes_written.get(),
        bytes_received = bytes_read.get(),
        "Finished precomputing MTs"
    );

    mpc_channel::sync(&mut sync_ch.0, &mut sync_ch.1)
        .await
        .context("unable to sync")?;

    Ok(())
}
