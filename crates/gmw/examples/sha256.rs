use std::fs::File;
use std::io::BufWriter;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use mpc_channel::Tcp;
use remoc::rch::mpsc::channel;
use tokio::net::{TcpListener, TcpStream};
use tracing_subscriber::EnvFilter;

use gmw::circuit::BaseCircuit;
use gmw::common::BitVec;
use gmw::executor::Executor;
use gmw::mul_triple::insecure_provider::InsecureMTProvider;
use gmw::mul_triple::trusted_seed_provider::TrustedMTProviderClient;

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
}

#[tokio::main]
async fn main() -> Result<()> {
    let _guard = init_tracing()?;
    let args = Args::parse();
    let circuit = BaseCircuit::load_bristol(args.circuit)?.into();
    // let mut transport = match args.id {
    //     0 => Tcp::listen(args.server).await?,
    //     1 => Tcp::connect(args.server).await?,
    //     illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    // };

    let (mut base_sender, mut base_receiver) = match args.id {
        0 => {
            let listener = TcpListener::bind(args.server).await.unwrap();
            let (socket, _) = listener.accept().await.unwrap();
            socket.set_nodelay(true).unwrap();
            let (socket_rx, socket_tx) = socket.into_split();

            // Establish Remoc connection over TCP.
            let (conn, tx, rx) = remoc::Connect::io::<_, _, _, _, remoc::codec::Bincode>(
                remoc::Cfg::default(),
                socket_rx,
                socket_tx,
            )
            .await
            .unwrap();
            tokio::spawn(conn);
            (tx, rx)
        }
        1 => {
            let socket = TcpStream::connect(args.server).await.unwrap();
            socket.set_nodelay(true).unwrap();
            let (socket_rx, socket_tx) = socket.into_split();

            // Establish Remoc connection over TCP.
            let (conn, tx, rx) = remoc::Connect::io(remoc::Cfg::default(), socket_rx, socket_tx)
                .await
                .unwrap();
            tokio::spawn(conn);
            (tx, rx)
        }
        _ => panic!("Only my-id 0 or 1 are supported"),
    };

    let (mut sender, remote_receiver) = channel(2);
    base_sender.send(remote_receiver).await.unwrap();
    let mut receiver = base_receiver.recv().await.unwrap().unwrap();

    let mut executor = /*if let Some(addr) = args.mt_provider*/ {
    //     let transport = Tcp::connect(addr).await?;
    //     let mt_provider = TrustedMTProviderClient::new("unique-id".into(), transport);
    //     Executor::new(&circuit, args.id, mt_provider).await?
    // } else {
        let mt_provider = InsecureMTProvider::default();
        Executor::new(&circuit, args.id, mt_provider).await?
    };
    let input = BitVec::repeat(false, 768);
    let out = executor.execute(input, &mut sender, &mut receiver).await?;
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
