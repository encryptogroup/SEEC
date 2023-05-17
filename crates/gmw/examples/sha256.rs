//! This example shows how to load a Bristol circuit (sha256) and execute it, optionally generating
//! the multiplication triples via a third trusted party (see the `trusted_party_mts.rs` example).
//!
//! It also demonstrates how to use the [`Statistics`] API to track the communication of
//! different phases and write it to a file.

use std::fs::File;
use std::io::{stdout, BufWriter, Write};
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use gmw::circuit::base_circuit::Load;
use gmw::circuit::{BaseCircuit, ExecutableCircuit};
use gmw::common::BitVec;
use gmw::executor::{Executor, Message};
use gmw::mul_triple::boolean::insecure_provider::InsecureMTProvider;
use gmw::mul_triple::boolean::trusted_seed_provider::TrustedMTProviderClient;
use gmw::protocols::boolean_gmw::BooleanGmw;
use gmw::BooleanGate;
use mpc_channel::sub_channels_for;
use mpc_channel::util::{Phase, Statistics};

#[derive(Parser, Debug)]
struct Args {
    /// Id of this party
    #[clap(long)]
    id: usize,

    /// Address of server to bind or connect to
    #[clap(long)]
    server: SocketAddr,

    /// Optional address of trusted server providing MTs
    #[clap(long)]
    mt_provider: Option<SocketAddr>,

    /// Sha256 as a bristol circuit
    #[clap(
        long,
        default_value = "test_resources/bristol-circuits/sha-256-low_depth.txt"
    )]
    circuit: PathBuf,
    /// File path for the communication statistics. Will overwrite existing files.
    #[clap(long)]
    stats: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = init_tracing()?;
    let args = Args::parse();
    let circuit: ExecutableCircuit<BooleanGate, u32> = ExecutableCircuit::DynLayers(
        BaseCircuit::load_bristol(args.circuit, Load::Circuit)?.into(),
    );

    let (mut sender, bytes_written, mut receiver, bytes_read) = match args.id {
        0 => mpc_channel::tcp::listen(args.server).await?,
        1 => mpc_channel::tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    // Initialize the communication statistics tracker with the counters for the main channel
    let mut comm_stats = Statistics::new(bytes_written, bytes_read).without_unaccounted(true);

    let (mut sender, mut receiver) =
        sub_channels_for!(&mut sender, &mut receiver, 8, Message<BooleanGmw>).await?;

    let mut executor: Executor<BooleanGmw, _> = if let Some(addr) = args.mt_provider {
        let (mt_sender, bytes_written, mt_receiver, bytes_read) =
            mpc_channel::tcp::connect(addr).await?;
        // Set the counters for the helper channel
        comm_stats.set_helper(bytes_written, bytes_read);
        let mt_provider = TrustedMTProviderClient::new("unique-id".into(), mt_sender, mt_receiver);
        // As the MTs are generated when the Executor is created, we record the communication
        // with the `record_helper` method and a custom category
        comm_stats
            .record_helper(
                Phase::Custom("Helper-Mts"),
                Executor::new(&circuit, args.id, mt_provider),
            )
            .await?
    } else {
        let mt_provider = InsecureMTProvider;
        comm_stats
            .record(
                Phase::FunctionDependentSetup,
                Executor::new(&circuit, args.id, mt_provider),
            )
            .await?
    };
    let input = BitVec::repeat(false, 768);
    let _out = comm_stats
        .record(
            Phase::Online,
            executor.execute(input, &mut sender, &mut receiver),
        )
        .await?;

    // Depending on whether a --stats file is set, create a file writer or stdout
    let mut writer: Box<dyn Write> = match args.stats {
        Some(path) => {
            let file = File::create(path)?;
            Box::new(file)
        }
        None => Box::new(stdout()),
    };
    // serde_json is used to write the statistics in json format. `.csv` is currently not
    // supported.
    let mut res = comm_stats.into_run_result();
    res.add_metadata("circuit", "sha256.rs");
    serde_json::to_writer_pretty(&mut writer, &res)?;
    writeln!(writer)?;

    Ok(())
}

pub fn init_tracing() -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_writer = BufWriter::new(File::create("sha256.log")?);
    let (non_blocking, appender_guard) = tracing_appender::non_blocking(log_writer);
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(non_blocking)
        .init();
    Ok(appender_guard)
}
