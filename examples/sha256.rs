use anyhow::Result;
use clap::Parser;
use gmw_rs::circuit::Circuit;
use gmw_rs::common::BitVec;
use gmw_rs::executor::Executor;
use gmw_rs::transport::Tcp;
use std::net::SocketAddr;

#[derive(Parser, Debug)]
struct Args {
    /// Id of this party
    #[clap(long)]
    id: usize,

    /// Addresses of parties in in order of their id.
    /// Example: --parties 127.0.0.1:7744 127.0.0.1:7744
    #[clap(long, value_delimiter = ' ', number_of_values = 2)]
    parties: Vec<SocketAddr>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let circuit = Circuit::load_bristol("test_resources/bristol-circuits/sha-256-low_depth.txt")?;
    let mut transport = match args.id {
        0 => Tcp::listen(args.parties[0]).await?,
        1 => Tcp::connect(args.parties[1]).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };
    let mut executor = Executor::new(&circuit, args.id);
    let input = BitVec::repeat(false, 768);
    let out = executor.execute(input, &mut transport).await?;
    println!("{out:?}");
    Ok(())
}
