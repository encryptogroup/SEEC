use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

// Code duplication due to https://github.com/rust-lang/cargo/issues/8379
pub fn init_tracing() -> tracing::dispatcher::DefaultGuard {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .set_default()
}
