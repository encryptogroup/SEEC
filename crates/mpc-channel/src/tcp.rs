// //! TCP implementation of a channel.

use super::util::{TrackingReader, TrackingWriter};
use crate::util::Counter;
use crate::{channel, Channel, CommunicationError, Receiver, Sender};
use async_stream::stream;
use futures::Stream;
use remoc::{ConnectError, RemoteSend};

use std::fmt::Debug;
use std::io;
use std::net::Ipv4Addr;

use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tracing::info;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Encountered io error when establishing TCP connection")]
    Io(#[from] io::Error),
    #[error("Error in establishing remoc connection")]
    RemocConnect(#[from] ConnectError<io::Error, io::Error>),
    #[error("Error exchanging initial Receiver")]
    ExchangeReceiver(#[from] CommunicationError),
}

#[tracing::instrument(err)]
pub async fn listen<T: RemoteSend>(
    addr: impl ToSocketAddrs + Debug,
    local_buffer: usize,
) -> Result<Channel<T>, Error> {
    info!("Listening for connections");
    let listener = TcpListener::bind(addr).await?;
    let (socket, remote_addr) = listener.accept().await?;
    info!(?remote_addr, "Established connection to remote");
    establish_remoc_connection(socket, local_buffer).await
}

#[tracing::instrument(err)]
pub async fn connect<T: RemoteSend>(
    remote_addr: impl ToSocketAddrs + Debug,
    local_buffer: usize,
) -> Result<Channel<T>, Error> {
    info!("Connecting to remote");
    let stream = TcpStream::connect(remote_addr).await?;
    info!("Established connection to remote");
    establish_remoc_connection(stream, local_buffer).await
}

#[tracing::instrument(err)]
pub async fn server<T: RemoteSend>(
    addr: impl ToSocketAddrs + Debug,
    local_buffer: usize,
) -> Result<impl Stream<Item = Result<Channel<T>, Error>>, io::Error> {
    info!("Starting Tcp Server");
    let listener = TcpListener::bind(addr).await?;
    let s = stream! {
        loop {
            let (socket, _) = listener.accept().await?;
            yield establish_remoc_connection(socket, local_buffer).await;

        }
    };
    Ok(s)
}

/// For testing purposes. Create two parties communicating via TcpStreams on localhost:port
/// If None is supplied, a random available port is selected
pub async fn new_local_pair<T: RemoteSend>(
    port: Option<u16>,
    local_buffer: usize,
) -> Result<(Channel<T>, Channel<T>), Error> {
    // use port 0 to bind to available random one
    let mut port = port.unwrap_or(0);
    let addr = (Ipv4Addr::LOCALHOST, port);
    let listener = TcpListener::bind(addr).await?;
    if port == 0 {
        // get the actual port bound to
        port = listener.local_addr()?.port();
    }
    let addr = (Ipv4Addr::LOCALHOST, port);
    let accept = async {
        let (socket, _) = listener.accept().await?;
        Ok(socket)
    };
    let (server, client) = tokio::try_join!(accept, TcpStream::connect(addr))?;

    let (ch1, ch2) = tokio::try_join!(
        establish_remoc_connection(server, local_buffer),
        establish_remoc_connection(client, local_buffer),
    )?;

    Ok((ch1, ch2))
}

// TODO provide way of passing remoc::Cfg to method
async fn establish_remoc_connection<T: RemoteSend>(
    socket: TcpStream,
    local_buffer: usize,
) -> Result<(Sender<T>, Counter, Receiver<T>, Counter), Error> {
    // send data ASAP
    socket.set_nodelay(true)?;
    let (socket_rx, socket_tx) = socket.into_split();
    let tracking_rx = TrackingReader::new(socket_rx);
    let tracking_tx = TrackingWriter::new(socket_tx);
    let bytes_read = tracking_rx.bytes_read();
    let bytes_written = tracking_tx.bytes_written();

    let mut cfg = remoc::Cfg::balanced();
    cfg.receive_buffer = 16 * 1024 * 1024;
    cfg.chunk_size = 1024 * 1024;

    // Establish Remoc connection over TCP.
    let (conn, mut tx, mut rx) =
        remoc::Connect::io::<_, _, _, _, remoc::codec::Bincode>(cfg, tracking_rx, tracking_tx)
            .await?;
    tokio::spawn(conn);

    let (sender, remote_receiver) = channel(local_buffer);
    tx.send(remote_receiver)
        .await
        .map_err(|err| CommunicationError::BaseSend(err.kind))?;
    let receiver = rx
        .recv()
        .await
        .map_err(CommunicationError::BaseRecv)?
        .ok_or(CommunicationError::RemoteClosed)?;

    Ok((sender, bytes_written, receiver, bytes_read))
}

#[cfg(test)]
mod tests {
    use crate::tcp::new_local_pair;
    use remoc::codec;
    use remoc::rch::mpsc::channel;

    #[tokio::test]
    async fn establish_connection() {
        let (ch1, ch2) = new_local_pair::<()>(None, 2).await.unwrap();

        let (_tx1, bytes_written1, _rx1, bytes_read1) = ch1;
        let (_tx2, bytes_written2, _rx2, bytes_read2) = ch2;
        assert_eq!(bytes_written1.get(), bytes_read2.get());
        assert_eq!(bytes_written2.get(), bytes_read1.get());
    }

    #[tokio::test]
    async fn send_channel_via_channel() {
        let (ch1, ch2) = new_local_pair(None, 2).await.unwrap();

        let (tx1, _, _rx1, _) = ch1;
        let (_tx2, _, mut rx2, _) = ch2;

        let (new_tx, remote_new_rx) = channel::<_, codec::Bincode>(10);
        tx1.send(remote_new_rx).await.unwrap();
        let mut new_rx = rx2.recv().await.unwrap().unwrap();
        new_tx.send(42).await.unwrap();
        new_tx.send(42).await.unwrap();
        new_tx.send(42).await.unwrap();
        drop(new_tx);
        let mut items_received = 0;
        while let Some(item) = new_rx.recv().await.transpose() {
            let item = item.unwrap();
            assert_eq!(item, 42);
            items_received += 1;
        }
        assert_eq!(items_received, 3);
    }
}
