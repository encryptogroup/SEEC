use crate::util::TrackingReadWrite;
use crate::TrackingChannel;
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
use tokio::io::split;
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
    let certs = load_certs(certificate_chain_file.as_ref())?;
    let key = load_key(private_key_file.as_ref())?;
    let config = rustls::ServerConfig::builder_with_protocol_versions(&[&TLS13])
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    let acceptor = TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind(addr).await?;
    let (socket, remote_addr) = listener.accept().await?;
    info!(?remote_addr, "Accepted TCP connection to remote");
    socket.set_nodelay(true)?;
    let (socket_read, socket_write) = socket.into_split();
    let tracking_channel = TrackingReadWrite::new(socket_read, socket_write);
    let write_counter = tracking_channel.bytes_written();
    let read_counter = tracking_channel.bytes_read();
    let tls_stream = acceptor.accept(tracking_channel).await?;
    info!(?remote_addr, "Established TLS connection to remote");
    let (tls_reader, tls_writer) = split(tls_stream);
    let (sender, _, receiver, _) =
        super::establish_remoc_connection(tls_reader, tls_writer).await?;
    // return the counters that include tls overhead
    // TODO it might be nice to have both counters
    Ok((sender, write_counter, receiver, read_counter))
}

#[tracing::instrument(err)]
pub async fn connect<T: RemoteSend>(
    domain: &str,
    remote_addr: impl ToSocketAddrs + Debug,
) -> Result<TrackingChannel<T>, Error> {
    let domain = ServerName::try_from(domain.to_string())?;
    let mut root_cert_store = rustls::RootCertStore::empty();
    let (added, ignored) = root_cert_store.add_parsable_certificates(load_native_certs()?);
    info!("Added {added} certificates to store. Ignored {ignored}");

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_cert_store)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));

    info!("Connecting to remote");
    let stream = TcpStream::connect(remote_addr).await?;
    stream.set_nodelay(true)?;
    let (socket_read, socket_write) = stream.into_split();
    let tracking_channel = TrackingReadWrite::new(socket_read, socket_write);
    let write_counter = tracking_channel.bytes_written();
    let read_counter = tracking_channel.bytes_read();
    info!("Established TCP connection to server");
    let tls_stream = connector.connect(domain, tracking_channel).await?;
    info!("Established TLS connection to server");
    let (tls_reader, tls_writer) = split(tls_stream);
    let (sender, _, receiver, _) =
        super::establish_remoc_connection(tls_reader, tls_writer).await?;
    Ok((sender, write_counter, receiver, read_counter))
}

// #[tracing::instrument(err)]
// pub async fn server<T: RemoteSend>(
//     addr: impl ToSocketAddrs + Debug,
// ) -> Result<impl Stream<Item = Result<TrackingChannel<T>, crate::tcp::Error>>, io::Error> {
//     info!("Starting Tcp Server");
//     let listener = TcpListener::bind(addr).await?;
//     let s = stream! {
//         loop {
//             let (socket, _) = listener.accept().await?;
//             yield establish_remoc_connection_tls(socket).await;
//
//         }
//     };
//     Ok(s)
// }
