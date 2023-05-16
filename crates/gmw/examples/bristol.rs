//! This example shows how to load a Bristol circuit (default sha256) and execute it,
//! optionally generating the multiplication triples via a third trusted party
//! (see the `trusted_party_mts.rs` example).
//!
//! It also demonstrates how to use the [`Statistics`] API to track the communication of
//! different phases and write it to a file.

use anyhow::{Context, Result};
use clap::Parser;
use gmw::bench::BenchParty;
use gmw::circuit::base_circuit::Load;
use gmw::circuit::{BaseCircuit, ExecutableCircuit};
use gmw::protocols::boolean_gmw::BooleanGmw;
use gmw::BooleanGate;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{stdout, BufWriter, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
struct Args {
    /// Id of this party. If not provided, both parties will be spawned within this process.
    #[clap(long)]
    id: Option<usize>,

    /// Address of server to bind or connect to. Localhost if not provided
    #[clap(long)]
    server: Option<SocketAddr>,

    /// Skips the MT generation
    #[clap(long)]
    skip_setup: bool,

    #[clap(long, default_value = "1")]
    repeat: usize,

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
    let args = Args::parse();
    let _guard = init_tracing(&args.circuit)?;
    let circ_name = args
        .circuit
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let circuit: ExecutableCircuit<BooleanGate, _> = ExecutableCircuit::DynLayers(
        BaseCircuit::load_bristol(args.circuit, Load::Circuit)?.into(),
    )
    .precompute_layers();

    let create_party = |id, circ| {
        BenchParty::<BooleanGmw, u32>::new(id)
            .explicit_circuit(circ)
            .repeat(args.repeat)
            .insecure_setup(args.skip_setup)
            .metadata(circ_name.clone())
    };

    let results = if let Some(id) = args.id {
        let party = create_party(id, circuit);
        party.bench().await.context("Failed to run benchmark")?
    } else {
        let party0 = create_party(0, circuit.clone());
        let party1 = create_party(1, circuit);
        let bench0 = tokio::spawn(party0.bench());
        let bench1 = tokio::spawn(party1.bench());
        let (res0, _res1) = tokio::try_join!(bench0, bench1).context("Failed to join parties")?;
        res0.context("Failed to run benchmark")?
    };

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
    serde_json::to_writer_pretty(&mut writer, &results)?;
    writeln!(writer)?;

    Ok(())
}

pub fn init_tracing(circ: &Path) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let mut log_file = circ
        .file_stem()
        .unwrap_or(OsStr::new("bristol_circ"))
        .to_os_string();
    log_file.push(".log");
    let log_writer = BufWriter::new(File::create(log_file)?);
    let (non_blocking, appender_guard) = tracing_appender::non_blocking(log_writer);
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(non_blocking)
        .init();
    Ok(appender_guard)
}
