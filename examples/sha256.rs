use anyhow::Result;
use clap::Parser;
use gmw_rs::circuit::Circuit;
use gmw_rs::common::BitVec;
use gmw_rs::executor::Executor;
use gmw_rs::mult_triple::insecure_provider::InsecureMTProvider;
use gmw_rs::mult_triple::trusted_provider::TrustedMTProviderClient;
use gmw_rs::transport::Tcp;
use std::fs::File;
use std::io::BufWriter;
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
struct Args {
    /// Id of this party
    #[clap(long)]
    id: usize,

    /// Address of server to bind or connect to
    #[clap(long)]
    server: SocketAddr,

    /// Optional address of trusted server providing MT's
    #[clap(long)]
    mt_provider: Option<SocketAddr>,

    /// Sha256 as a bristol circuit
    #[clap(
        long,
        default_value = "test_resources/bristol-circuits/sha-256-low_depth.txt"
    )]
    circuit: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = init_tracing()?;
    let args = Args::parse();
    let circuit = Circuit::load_bristol(args.circuit)?;
    let mut transport = match args.id {
        0 => Tcp::listen(args.server).await?,
        1 => Tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };
    let mut executor = if let Some(addr) = args.mt_provider {
        let transport = Tcp::connect(addr).await?;
        let mt_provider = TrustedMTProviderClient::new("unique-id".into(), transport);
        Executor::new(&circuit, args.id, mt_provider).await?
    } else {
        let mt_provider = InsecureMTProvider::default();
        Executor::new(&circuit, args.id, mt_provider).await?
    };
    let input = BitVec::repeat(false, 768);
    let out = executor.execute(input, &mut transport).await?;
    println!("{out:?}");
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
