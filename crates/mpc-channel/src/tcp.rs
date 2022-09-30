//! TCP implementation of a channel.
use super::util::{TrackingReader, TrackingWriter};
use super::Channel;
use async_stream::stream;
use futures::{Sink, Stream};
use pin_project::pin_project;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio_serde::formats::SymmetricalBincode;
use tokio_serde::SymmetricallyFramed;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
use tracing::info;

type SinkPart<Item> = SymmetricallyFramed<
    FramedWrite<TrackingWriter<OwnedWriteHalf>, LengthDelimitedCodec>,
    Item,
    SymmetricalBincode<Item>,
>;

type StreamPart<Item> = SymmetricallyFramed<
    FramedRead<TrackingReader<OwnedReadHalf>, LengthDelimitedCodec>,
    Item,
    SymmetricalBincode<Item>,
>;

/// A [`Channel`](`Channel`) which sends [bincode](https://docs.rs/bincode/1.3.1/bincode/)
/// serialized values over a TCP connection, tracking the number of bytes sent and received.
#[pin_project]
pub struct Tcp<Item> {
    #[pin]
    sender: SinkPart<Item>,
    #[pin]
    receiver: StreamPart<Item>,
}

impl<Item: DeserializeOwned> Stream for Tcp<Item> {
    type Item = Result<Item, io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.receiver.poll_next(cx)
    }
}

impl<Item: Serialize> Sink<Item> for Tcp<Item> {
    type Error = io::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sender.poll_ready(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Item) -> Result<(), Self::Error> {
        let this = self.project();
        this.sender.start_send(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sender.poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sender.poll_close(cx)
    }
}

impl<Item> Channel<Item> for Tcp<Item>
where
    Item: Serialize + DeserializeOwned + Unpin + Send,
{
    type StreamPart = StreamPart<Item>;
    type SinkPart = SinkPart<Item>;

    fn split_mut(&mut self) -> (&mut Self::StreamPart, &mut Self::SinkPart) {
        (&mut self.receiver, &mut self.sender)
    }
}

impl<Item> Tcp<Item> {
    #[tracing::instrument(err)]
    pub async fn listen(addr: impl ToSocketAddrs + Debug) -> Result<Self, io::Error> {
        info!("Listening for connections");
        let listener = TcpListener::bind(addr).await?;
        let (socket, _) = listener.accept().await?;
        // send data ASAP
        socket.set_nodelay(true)?;
        Ok(Self::from_tcp_stream(socket))
    }

    #[tracing::instrument(err)]
    pub async fn connect(addr: impl ToSocketAddrs + Debug) -> Result<Self, io::Error> {
        info!("Connecting to remote");
        let socket = TcpStream::connect(addr).await?;
        // send data ASAP
        socket.set_nodelay(true)?;
        Ok(Self::from_tcp_stream(socket))
    }

    #[tracing::instrument(err)]
    pub async fn server(
        addr: impl ToSocketAddrs + Debug,
    ) -> Result<impl Stream<Item = Self>, io::Error> {
        info!("Starting Tcp Server");
        let listener = TcpListener::bind(addr).await?;
        let s = stream! {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                // send data ASAP
                socket.set_nodelay(true).unwrap();
                yield Self::from_tcp_stream(socket);

            }
        };
        Ok(s)
    }

    /// For testing purposes. Create two parties communicating via TcpStreams on localhost:port
    /// If None is supplied, a random available port is selected
    pub async fn new_local_pair(port: Option<u16>) -> Result<(Self, Self), io::Error> {
        // use port 0 to bind to available random one
        let mut port = port.unwrap_or(0);
        let addr = ("127.0.0.1", port);
        let listener = TcpListener::bind(addr).await?;
        if port == 0 {
            // get the actual port bound to
            port = listener.local_addr()?.port();
        }
        let addr = ("127.0.0.1", port);
        let accept = async {
            let (socket, _) = listener.accept().await?;
            Ok(Self::from_tcp_stream(socket))
        };
        let (server, client) = tokio::try_join!(accept, Self::connect(addr))?;
        Ok((server, client))
    }

    fn from_tcp_stream(socket: TcpStream) -> Self {
        let (read_half, write_half) = socket.into_split();
        let framed_read =
            FramedRead::new(TrackingReader::new(read_half), LengthDelimitedCodec::new());
        let framed_write =
            FramedWrite::new(TrackingWriter::new(write_half), LengthDelimitedCodec::new());
        // Deserialize frames
        let receiver = tokio_serde::SymmetricallyFramed::new(
            framed_read,
            SymmetricalBincode::<Item>::default(),
        );
        let sender = tokio_serde::SymmetricallyFramed::new(
            framed_write,
            SymmetricalBincode::<Item>::default(),
        );
        Self { sender, receiver }
    }

    /// Return the number of bytes written since creation.
    pub fn bytes_written(&self) -> usize {
        self.sender.get_ref().get_ref().bytes_written()
    }

    /// Return the number of bytes read since creation.
    pub fn bytes_read(&self) -> usize {
        self.receiver.get_ref().get_ref().bytes_read()
    }

    /// Returns the total number of bytes read and written.
    pub fn bytes_total(&self) -> usize {
        self.bytes_read() + self.bytes_written()
    }

    pub fn reset_bytes_total(&mut self) {
        self.sender.get_mut().get_mut().reset();
        self.receiver.get_mut().get_mut().reset();
    }
}

#[cfg(test)]
mod tests {
    use super::Tcp;
    use futures::{SinkExt, StreamExt};
    use serde::{Deserialize, Serialize};

    #[tokio::test]
    async fn tcp_transport() {
        #[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
        struct Dummy([u8; 32]);

        let (mut t1, mut t2) = Tcp::new_local_pair(None).await.unwrap();

        t1.send(Dummy::default()).await.unwrap();
        assert_eq!(t2.next().await.unwrap().unwrap(), Dummy::default());
        // + 4 bytes because the value is prefixed by its size, see FramedWrite/Read
        assert_eq!(t1.bytes_written(), 32 + 4);
        assert_eq!(t2.bytes_read(), 32 + 4);
    }
}
