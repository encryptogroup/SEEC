// //! TCP implementation of a channel.

use async_stream::stream;
use std::fmt::Debug;
use std::io;
use std::net::Ipv4Addr;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};

use futures::Stream;
use remoc::{ConnectError, RemoteSend};
use tokio::time::Instant;
use tracing::info;

use crate::TrackingChannel;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Encountered io error when establishing TCP connection")]
    Io(#[from] io::Error),
    #[error("Error in establishing remoc connection")]
    RemocConnect(#[from] ConnectError<io::Error, io::Error>),
}

#[tracing::instrument(err)]
pub async fn listen<T: RemoteSend>(
    addr: impl ToSocketAddrs + Debug,
) -> Result<TrackingChannel<T>, Error> {
    info!("Listening for connections");
    let listener = TcpListener::bind(addr).await?;
    let (socket, remote_addr) = listener.accept().await?;
    info!(?remote_addr, "Established connection to remote");
    establish_remoc_connection_tcp(socket).await
}

#[tracing::instrument(err)]
pub async fn connect<T: RemoteSend>(
    remote_addr: impl ToSocketAddrs + Debug,
) -> Result<TrackingChannel<T>, Error> {
    info!("Connecting to remote");
    let stream = TcpStream::connect(remote_addr).await?;
    info!("Established connection to server");
    establish_remoc_connection_tcp(stream).await
}

#[tracing::instrument(err)]
/// Connect to remote and retry upon failure for `timout` time
pub async fn connect_with_timeout<T: RemoteSend>(
    remote_addr: impl ToSocketAddrs + Debug,
    timeout: Duration,
) -> Result<TrackingChannel<T>, Error> {
    info!("Connecting to remote with timeout {timeout:?}");
    let mut wait = Duration::from_millis(10);
    let exp_wait_factor = 1.2;
    let start = Instant::now();
    let mut last_err = None;
    while start.elapsed() < timeout {
        let stream = match TcpStream::connect(&remote_addr).await {
            Ok(stream) => stream,
            Err(err) => {
                last_err = Some(err.into());
                tokio::time::sleep(wait).await;
                wait = Duration::from_millis((wait.as_millis() as f64 * exp_wait_factor) as u64);
                continue;
            }
        };
        info!("Established connection to remote");
        match establish_remoc_connection_tcp(stream).await {
            Ok(ch) => return Ok(ch),
            Err(err) => {
                last_err = Some(err);
                continue;
            }
        }
    }
    Err(last_err.unwrap())
}

#[tracing::instrument(err)]
pub async fn server<T: RemoteSend>(
    addr: impl ToSocketAddrs + Debug,
) -> Result<impl Stream<Item = Result<TrackingChannel<T>, Error>>, io::Error> {
    info!("Starting Tcp Server");
    let listener = TcpListener::bind(addr).await?;
    let s = stream! {
        loop {
            let (socket, _) = listener.accept().await?;
            yield establish_remoc_connection_tcp(socket).await;

        }
    };
    Ok(s)
}

/// For testing purposes. Create two parties communicating via TcpStreams on localhost:port
/// If None is supplied, a random available port is selected
pub async fn new_local_pair<T: RemoteSend>(
    port: Option<u16>,
) -> Result<(TrackingChannel<T>, TrackingChannel<T>), Error> {
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
        establish_remoc_connection_tcp(server),
        establish_remoc_connection_tcp(client),
    )?;

    Ok((ch1, ch2))
}

pub(crate) async fn establish_remoc_connection_tcp<T: RemoteSend>(
    socket: TcpStream,
) -> Result<TrackingChannel<T>, Error> {
    // send data ASAP
    socket.set_nodelay(true)?;
    let (socket_rx, socket_tx) = socket.into_split();
    Ok(super::establish_remoc_connection(socket_rx, socket_tx).await?)
}

#[cfg(test)]
mod tests {
    use crate::tcp::new_local_pair;
    use remoc::codec;
    use remoc::rch::mpsc::channel;
    use std::time::Duration;

    #[tokio::test]
    async fn establish_connection() {
        let (ch1, ch2) = new_local_pair::<()>(None).await.unwrap();

        // Sleep to ensure values have been actually sent and counters are correct
        tokio::time::sleep(Duration::from_millis(10)).await;
        let (_tx1, bytes_written1, _rx1, bytes_read1) = ch1;
        let (_tx2, bytes_written2, _rx2, bytes_read2) = ch2;
        assert_eq!(bytes_written1.get(), bytes_read2.get());
        assert_eq!(bytes_written2.get(), bytes_read1.get());
    }

    #[tokio::test]
    async fn send_channel_via_channel() {
        let (ch1, ch2) = new_local_pair(None).await.unwrap();

        let (mut tx1, _, _rx1, _) = ch1;
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
