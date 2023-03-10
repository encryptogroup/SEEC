use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use gmw::mul_triple::boolean::trusted_seed_provider::TrustedMTProviderServer;

/// Example usage of the `TrustedMTProviderServer`.
#[derive(Parser, Debug)]
struct Args {
    /// <IP>:<Port> the trusted party MT provider should listen on
    #[clap(long)]
    address: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let args = Args::parse();

    TrustedMTProviderServer::start(args.address).await?;
    Ok(())
}
