//! This example shows how to load a Bristol circuit (default sha256) and execute it,
//! optionally generating the multiplication triples via a third trusted party
//! (see the `trusted_party_mts.rs` example).
//!
//! It also demonstrates how to use the [`Statistics`] API to track the communication of
//! different phases and write it to a file.

use anyhow::{Context, Result};
use clap::{Args, Parser};
use seec::bench::BenchParty;
use seec::circuit::base_circuit::Load;
use seec::circuit::{BaseCircuit, ExecutableCircuit};
use seec::protocols::boolean_gmw::BooleanGmw;
use seec::secret::inputs;
use seec::SubCircuitOutput;
use seec::{BooleanGate, CircuitBuilder};
use std::fs::File;
use std::io;
use std::io::{stdout, BufReader, BufWriter, Write};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
enum ProgArgs {
    Compile(CompileArgs),
    Execute(ExecuteArgs),
}

#[derive(Args, Debug)]
/// Precompile a bristol circuit for faster execution
struct CompileArgs {
    #[arg(long)]
    simd: Option<NonZeroUsize>,

    /// Output path of the compile circuit.
    #[arg(short, long)]
    output: PathBuf,

    #[clap(short, long)]
    log: Option<PathBuf>,

    /// Use dynamic layers instead of static
    #[clap(short, long)]
    dyn_layers: bool,

    /// Circuit in bristol format
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

    /// Use MTs stored in <FILE> generated via precompute_mts.rs
    #[clap(long)]
    stored_mts: Option<PathBuf>,

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
    let load = match compile_args.simd {
        Some(_) => Load::SubCircuit,
        None => Load::Circuit,
    };
    let mut bc: BaseCircuit = BaseCircuit::load_bristol(&compile_args.circuit, load)
        .expect("failed to load bristol circuit");

    let mut circ = match compile_args.simd {
        Some(size) => {
            bc.set_simd_size(size);
            let circ_input_size = bc.sub_circuit_input_count();
            let inputs = inputs::<u32>(circ_input_size);
            let bc = bc.into_shared();

            let (output, circ_id) = CircuitBuilder::with_global(|builder| {
                builder.get_main_circuit().lock().set_simd_size(size);
                let circ_id = builder.push_circuit(bc);
                let output = builder.connect_sub_circuit(&inputs, circ_id);
                (output, circ_id)
            });
            let main = output.connect_to_main(circ_id);
            main.iter().for_each(|s| {
                s.output();
            });
            let circ = CircuitBuilder::global_into_circuit();
            ExecutableCircuit::DynLayers(circ)
        }
        None => ExecutableCircuit::DynLayers(bc.into()),
    };
    if !compile_args.dyn_layers {
        circ = circ.precompute_layers();
    }
    let out =
        BufWriter::new(File::create(&compile_args.output).context("failed to create output file")?);
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
        let mut party = BenchParty::<BooleanGmw, u32>::new(id)
            .explicit_circuit(circ)
            .repeat(execute_args.repeat)
            .insecure_setup(execute_args.insecure_setup)
            .interleave_setup(execute_args.interleave_setup)
            .metadata(circ_name.clone());
        if let Some(path) = &execute_args.stored_mts {
            party = party.stored_mts(path);
        }
        if let Some(server) = execute_args.server {
            party = party.server(server)
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

fn load_circ(args: &ExecuteArgs) -> Result<ExecutableCircuit<BooleanGate, u32>> {
    let res = bincode::deserialize_from(BufReader::new(
        File::open(&args.circuit).context("Failed to open circuit file")?,
    ));
    match res {
        Ok(circ) => Ok(circ),
        Err(_) => {
            // try to load as bristol
            Ok(ExecutableCircuit::DynLayers(
                BaseCircuit::load_bristol(&args.circuit, Load::Circuit)
                    .context("Circuit is neither .seec file or bristol")?
                    .into(),
            )
            .precompute_layers())
        }
    }
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
