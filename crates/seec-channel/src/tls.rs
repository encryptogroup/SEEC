use crate::util::{Counter, TrackingReadWrite};
use crate::{BaseReceiver, BaseSender, TrackingChannel};
use remoc::{ConnectError, RemoteSend};
use rustls::pki_types::{CertificateDer, InvalidDnsNameError, PrivateKeyDer, ServerName};
use rustls::version::TLS13;
use rustls_native_certs::load_native_certs;
use rustls_pemfile::{certs, private_key};
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{split, AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tracing::info;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Encountered io error when establishing TLS connection")]
    Io(#[from] io::Error),
    #[error("TLS error")]
    TlsError(#[from] rustls::Error),
    #[error("Invalid DNS name")]
    InvalidDnsNameError(#[from] InvalidDnsNameError),
    #[error("Missing private key file")]
    MissingKey,
    #[error("Error in establishing remoc connection")]
    RemocConnect(#[from] ConnectError<io::Error, io::Error>),
}

fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, io::Error> {
    certs(&mut BufReader::new(File::open(path)?)).collect()
}

fn load_key(path: &Path) -> Result<PrivateKeyDer<'static>, Error> {
    private_key(&mut BufReader::new(File::open(path)?))?.ok_or(Error::MissingKey)
}

#[tracing::instrument(err)]
pub async fn listen<T: RemoteSend>(
    addr: impl ToSocketAddrs + Debug,
    private_key_file: impl AsRef<Path> + Debug,
    certificate_chain_file: impl AsRef<Path> + Debug,
) -> Result<TrackingChannel<T>, Error> {
    info!("Listening for connections");
    let listener = TcpListener::bind(addr).await?;
    let (stream, remote_addr) = listener.accept().await?;
    info!(?remote_addr, "Accepted TCP connection to remote");
    let (tracking_stream, write_counter, read_counter) = tracking_stream(stream)?;
    let (sender, receiver) =
        tls_accept(tracking_stream, private_key_file, certificate_chain_file).await?;
    // return the counters that include tls overhead
    // TODO it might be nice to have both counters
    Ok((sender, write_counter, receiver, read_counter))
}

#[tracing::instrument(err)]
pub async fn connect<T: RemoteSend>(
    domain: &str,
    remote_addr: impl ToSocketAddrs + Debug,
) -> Result<TrackingChannel<T>, Error> {
    info!("Connecting to remote");
    let stream = TcpStream::connect(remote_addr).await?;
    info!("Established TCP connection to server");
    let (tracking_stream, write_counter, read_counter) = tracking_stream(stream)?;
    let (sender, receiver) = tls_connect(domain, tracking_stream).await?;
    Ok((sender, write_counter, receiver, read_counter))
}

fn tracking_stream(
    tcp_stream: TcpStream,
) -> Result<
    (
        impl AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
        Counter,
        Counter,
    ),
    Error,
> {
    tcp_stream.set_nodelay(true)?;
    let (socket_read, socket_write) = tcp_stream.into_split();
    let tracking_channel = TrackingReadWrite::new(socket_read, socket_write);
    let write_counter = tracking_channel.bytes_written();
    let read_counter = tracking_channel.bytes_read();
    Ok((tracking_channel, write_counter, read_counter))
}

async fn tls_accept<T, IO>(
    tcp_stream: IO,
    private_key_file: impl AsRef<Path> + Debug,
    certificate_chain_file: impl AsRef<Path> + Debug,
) -> Result<(BaseSender<T>, BaseReceiver<T>), Error>
where
    T: RemoteSend,
    IO: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
{
    let certs = load_certs(certificate_chain_file.as_ref())?;
    let key = load_key(private_key_file.as_ref())?;
    let config = rustls::ServerConfig::builder_with_protocol_versions(&[&TLS13])
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    let acceptor = TlsAcceptor::from(Arc::new(config));
    let tls_stream = acceptor.accept(tcp_stream).await?;
    info!("Established TLS connection to remote");
    let (tls_reader, tls_writer) = split(tls_stream);
    let (sender, _, receiver, _) =
        super::establish_remoc_connection(tls_reader, tls_writer).await?;
    Ok((sender, receiver))
}

async fn tls_connect<T, IO>(
    domain: &str,
    tcp_stream: IO,
) -> Result<(BaseSender<T>, BaseReceiver<T>), Error>
where
    T: RemoteSend,
    IO: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
{
    let domain = ServerName::try_from(domain.to_string())?;
    let mut root_cert_store = rustls::RootCertStore::empty();
    let (added, ignored) = root_cert_store.add_parsable_certificates(load_native_certs()?);
    info!("Added {added} certificates to store. Ignored {ignored}");

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_cert_store)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));
    let tls_stream = connector.connect(domain, tcp_stream).await?;
    info!("Established TLS connection to server");
    let (tls_reader, tls_writer) = split(tls_stream);
    let (sender, _, receiver, _) =
        super::establish_remoc_connection(tls_reader, tls_writer).await?;
    Ok((sender, receiver))
}
