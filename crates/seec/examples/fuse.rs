//! FUSE Mixed GMW executor.

use anyhow::{Context, Result};
use clap::{Args, Parser};
use seec::bench::BenchParty;
use seec::circuit::ExecutableCircuit;
use seec::parse::fuse::{CallMode, FuseConverter};
use seec::protocols::mixed_gmw::{Mixed, MixedGate, MixedGmw};
use std::fs::File;
use std::io;
use std::io::{stdout, BufReader, BufWriter, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
enum ProgArgs {
    Compile(CompileArgs),
    Execute(ExecuteArgs),
}

#[derive(Args, Debug)]
/// Precompile a FUSE circuit for faster execution
struct CompileArgs {
    // TODO Currently not implemented
    // #[arg(long)]
    // simd: Option<NonZeroUsize>,
    /// Output path of the compile circuit.
    #[arg(short, long)]
    output: PathBuf,

    #[clap(short, long)]
    log: Option<PathBuf>,

    /// Use dynamic layers instead of static
    #[clap(short, long)]
    dyn_layers: bool,

    /// Inline sub-circuit calls into the main circuit.
    #[clap(short, long)]
    inline_circuits: bool,

    /// Circuit in FUSE format
    circuit: PathBuf,
}

#[derive(Args, Debug)]
struct ExecuteArgs {
    /// Id of this party. If not provided, both parties will be spawned within this process.
    #[clap(long)]
    id: Option<usize>,

    /// Address of server to bind or connect to. Localhost if not provided
    #[clap(long)]
    server: Option<SocketAddr>,

    /// Performs insecure setup by randomly generating MTs based on fixed seed (no OTs)
    #[clap(long)]
    insecure_setup: bool,

    // TODO /// Use MTs stored in <FILE> generated via precompute_mts.rs
    // #[clap(long)]
    // stored_mts: Option<PathBuf>,
    /// Perform setup interleaved with the online phase
    #[clap(long)]
    interleave_setup: bool,

    #[clap(long, default_value = "1")]
    repeat: usize,

    /// File path for the communication statistics. Will overwrite existing files.
    #[clap(long)]
    stats: Option<PathBuf>,

    #[clap(short, long)]
    log: Option<PathBuf>,

    /// Circuit to execute. Must be compiled beforehand
    circuit: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let prog_args = ProgArgs::parse();
    init_tracing(&prog_args).context("failed to init logging")?;
    match prog_args {
        ProgArgs::Compile(args) => compile(args).context("failed to compile circuit"),
        ProgArgs::Execute(args) => execute(args).await.context("failed to execute circuit"),
    }
}

fn compile(compile_args: CompileArgs) -> Result<()> {
    let call_mode = match compile_args.inline_circuits {
        true => CallMode::InlineCircuits,
        false => CallMode::CallCircuits,
    };
    let converter = FuseConverter::<u32>::new(call_mode);
    let circ = converter
        .convert(&compile_args.circuit)
        .ok()
        .context("Unable to load and convert FUSE circuit")?;
    let mut circ = ExecutableCircuit::DynLayers(circ);
    if !compile_args.dyn_layers {
        circ = circ.precompute_layers();
    }
    let out =
        BufWriter::new(File::create(compile_args.output).context("failed to create output file")?);
    bincode::serialize_into(out, &circ).context("failed to serialize circuit")?;
    Ok(())
}

impl ProgArgs {
    fn log(&self) -> Option<&PathBuf> {
        match self {
            ProgArgs::Compile(args) => args.log.as_ref(),
            ProgArgs::Execute(args) => args.log.as_ref(),
        }
    }
}

async fn execute(execute_args: ExecuteArgs) -> Result<()> {
    let circ_name = execute_args
        .circuit
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let circuit = load_circ(&execute_args).context("failed to load circuit")?;

    let create_party = |id, circ| {
        let mut party = BenchParty::<MixedGmw<u32>, u32>::new(id)
            .explicit_circuit(circ)
            .repeat(execute_args.repeat)
            .insecure_setup(execute_args.insecure_setup)
            .metadata(circ_name.clone());
        if let Some(server) = execute_args.server {
            party = party.server(server);
        }
        party
    };

    let results = if let Some(id) = execute_args.id {
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
    let mut writer: Box<dyn Write> = match execute_args.stats {
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

fn load_circ(args: &ExecuteArgs) -> Result<ExecutableCircuit<Mixed<u32>, MixedGate<u32>, u32>> {
    bincode::deserialize_from(BufReader::new(
        File::open(&args.circuit).context("Failed to open circuit file")?,
    ))
    .context("Failed to deserialize circuit")
}

fn init_tracing(args: &ProgArgs) -> Result<Option<tracing_appender::non_blocking::WorkerGuard>> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env()
        .context("Invalid log directives")?;
    match args.log() {
        Some(path) => {
            let log_writer =
                BufWriter::new(File::create(path).context("failed to create log file")?);
            let (non_blocking, appender_guard) = tracing_appender::non_blocking(log_writer);
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(env_filter)
                .with_writer(non_blocking)
                .init();
            Ok(Some(appender_guard))
        }
        None => {
            tracing_subscriber::fmt()
                .with_writer(io::stderr)
                .with_env_filter(env_filter)
                .init();
            Ok(None)
        }
    }
}
