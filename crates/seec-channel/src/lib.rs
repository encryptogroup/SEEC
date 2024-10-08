//! Channel abstraction for communication
use crate::util::{Counter, TrackingReader, TrackingWriter};
use async_trait::async_trait;

use remoc::rch::{base, mpsc};
use remoc::{codec, ConnectError, RemoteSend};
use serde::{Deserialize, Serialize};

use tokio::io;
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::debug;

pub use seec_channel_macros::sub_channels_for;

pub mod in_memory;
pub mod multi;
pub mod tcp;
pub mod tls;
pub mod util;

pub type BaseSender<T> = base::Sender<T, codec::Bincode>;
pub type BaseReceiver<T> = base::Receiver<T, codec::Bincode>;

pub type Sender<T> = mpsc::Sender<T, codec::Bincode, 128>;
pub type Receiver<T> = mpsc::Receiver<T, codec::Bincode, 128>;

pub type TrackingChannel<T> = (BaseSender<T>, Counter, BaseReceiver<T>, Counter);
pub type Channel<T> = (Sender<T>, Receiver<T>);

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SyncMsg;

#[async_trait]
pub trait SenderT<T> {
    type Error;
    async fn send(&mut self, item: T) -> Result<(), Self::Error>;
}

#[async_trait]
pub trait ReceiverT<T> {
    type Error;
    async fn recv(&mut self) -> Result<Option<T>, Self::Error>;
}

#[derive(thiserror::Error, Debug)]
pub enum CommunicationError {
    #[error("Error sending initial value")]
    BaseSend(#[source] base::SendError<()>),
    #[error("Error receiving value on base channel")]
    BaseRecv(#[from] base::RecvError),
    #[error("Error sending value on mpsc channel")]
    Send(#[source] mpsc::SendError<()>),
    #[error("Error receiving value on mpsc channel")]
    Recv(#[from] mpsc::RecvError),
    #[error("Error in Multi-Sender/Receiver")]
    Multi(#[from] multi::Error),
    #[error("Unexpected termination. Remote is closed.")]
    RemoteClosed,
    #[error("Received out of order message")]
    UnexpectedMessage,
    #[error("Unabel to establish multi-sub-channel with party {0}")]
    MultiSubChannel(u32, #[source] Box<CommunicationError>),
}

pub fn channel<T: RemoteSend, const BUFFER: usize>(
    local_buffer: usize,
) -> (
    mpsc::Sender<T, codec::Bincode, BUFFER>,
    mpsc::Receiver<T, codec::Bincode, BUFFER>,
) {
    let (sender, receiver) = mpsc::channel(local_buffer);
    let sender = sender.set_buffer::<BUFFER>();
    let receiver = receiver.set_buffer::<BUFFER>();
    (sender, receiver)
}

#[tracing::instrument(skip_all)]
pub async fn sub_channel<S, R, Msg, SubMsg>(
    sender: &mut S,
    receiver: &mut R,
    local_buffer: usize,
) -> Result<(Sender<SubMsg>, Receiver<SubMsg>), CommunicationError>
where
    S: SenderT<Msg>,
    R: ReceiverT<Msg>,
    Sender<SubMsg>: Into<Msg>,
    Msg: Into<Option<Sender<SubMsg>>> + RemoteSend,
    SubMsg: RemoteSend,
    CommunicationError: From<S::Error> + From<R::Error>,
{
    debug!("Establishing new sub_channel");
    let (remote_sub_sender, sub_receiver) = channel(local_buffer);
    sender.send(remote_sub_sender.into()).await?;
    debug!("Sent remote_sub_receiver");
    let msg = receiver
        .recv()
        .await?
        .ok_or(CommunicationError::RemoteClosed)?;
    let sub_sender = msg.into().ok_or(CommunicationError::UnexpectedMessage)?;
    debug!("Received sub_receiver");
    Ok((sub_sender, sub_receiver))
}

#[tracing::instrument(skip_all)]
pub async fn sub_channel_with<S, R, Msg, SubMsg>(
    sender: &mut S,
    receiver: &mut R,
    local_buffer: usize,
    wrap_fn: impl FnOnce(Sender<SubMsg>) -> Msg,
    extract_fn: impl FnOnce(Msg) -> Option<Sender<SubMsg>>,
) -> Result<(Sender<SubMsg>, Receiver<SubMsg>), CommunicationError>
where
    S: SenderT<Msg>,
    R: ReceiverT<Msg>,
    Msg: RemoteSend,
    SubMsg: RemoteSend,
    CommunicationError: From<S::Error> + From<R::Error>,
{
    debug!("Establishing new sub_channel");
    let (remote_sub_sender, sub_receiver) = channel(local_buffer);
    sender.send(wrap_fn(remote_sub_sender)).await?;
    debug!("Sent remote_sub_receiver");
    let msg = receiver
        .recv()
        .await?
        .ok_or(CommunicationError::RemoteClosed)?;
    let sub_sender = extract_fn(msg).ok_or(CommunicationError::UnexpectedMessage)?;
    debug!("Received sub_receiver");
    Ok((sub_sender, sub_receiver))
}

pub async fn sync<S, R>(sender: &mut S, receiver: &mut R) -> Result<(), CommunicationError>
where
    S: SenderT<SyncMsg>,
    R: ReceiverT<SyncMsg>,
    CommunicationError: From<S::Error> + From<R::Error>,
{
    sender.send(SyncMsg).await?;
    // ignore receiving a None
    receiver.recv().await?;
    sender.send(SyncMsg).await?;
    // ignore receiving a None
    let _err = receiver.recv().await;
    Ok(())
}

#[async_trait]
impl<T, Codec> SenderT<T> for base::Sender<T, Codec>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    type Error = base::SendError<T>;
    async fn send(&mut self, item: T) -> Result<(), Self::Error> {
        base::Sender::send(self, item).await
    }
}

#[async_trait]
impl<T, Codec> ReceiverT<T> for base::Receiver<T, Codec>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    type Error = base::RecvError;
    async fn recv(&mut self) -> Result<Option<T>, Self::Error> {
        base::Receiver::recv(self).await
    }
}

#[async_trait]
impl<T, Codec, const BUFFER: usize> SenderT<T> for mpsc::Sender<T, Codec, BUFFER>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    type Error = mpsc::SendError<T>;
    async fn send(&mut self, item: T) -> Result<(), Self::Error> {
        mpsc::Sender::send(self, item).await
    }
}

#[async_trait]
impl<T, Codec, const BUFFER: usize> ReceiverT<T> for mpsc::Receiver<T, Codec, BUFFER>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    type Error = mpsc::RecvError;
    async fn recv(&mut self) -> Result<Option<T>, Self::Error> {
        mpsc::Receiver::recv(self).await
    }
}

impl<T> From<base::SendError<T>> for CommunicationError {
    fn from(err: base::SendError<T>) -> Self {
        CommunicationError::BaseSend(err.without_item())
    }
}

impl<T> From<mpsc::SendError<T>> for CommunicationError {
    fn from(err: mpsc::SendError<T>) -> Self {
        CommunicationError::Send(err.without_item())
    }
}

// TODO provide way of passing remoc::Cfg to method
async fn establish_remoc_connection<R, W, T>(
    reader: R,
    writer: W,
) -> Result<TrackingChannel<T>, ConnectError<io::Error, io::Error>>
where
    R: AsyncRead + Send + Sync + Unpin + 'static,
    W: AsyncWrite + Send + Sync + Unpin + 'static,
    T: RemoteSend,
{
    let tracking_rx = TrackingReader::new(reader);
    let tracking_tx = TrackingWriter::new(writer);
    let bytes_read = tracking_rx.bytes_read();
    let bytes_written = tracking_tx.bytes_written();

    let mut cfg = remoc::Cfg::balanced();
    cfg.receive_buffer = 16 * 1024 * 1024;
    cfg.chunk_size = 1024 * 1024;

    // Establish Remoc connection over TCP.
    let (conn, tx, rx) = remoc::Connect::io_buffered::<_, _, _, _, remoc::codec::Bincode>(
        cfg,
        tracking_rx,
        tracking_tx,
        8096,
    )
    .await?;

    tokio::spawn(conn);

    debug!("Established remoc connection");

    Ok((tx, bytes_written, rx, bytes_read))
}
