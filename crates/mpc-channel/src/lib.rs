//! Channel abstraction for communication
use crate::util::Counter;
use remoc::rch::base::SendErrorKind;
use remoc::rch::{base, mpsc};
use remoc::RemoteSend;

pub mod in_memory;
pub mod tcp;
pub mod util;

pub type Channel<T> = (Sender<T>, Counter, Receiver<T>, Counter);

pub type Sender<T> = mpsc::Sender<T, remoc::codec::Bincode, 128>;
pub type Receiver<T> = mpsc::Receiver<T, remoc::codec::Bincode, 128>;

#[derive(thiserror::Error, Debug)]
pub enum CommunicationError {
    #[error("Error sending initial value")]
    BaseSend(SendErrorKind),
    #[error("Error receiving value on base channel")]
    BaseRecv(#[from] base::RecvError),
    #[error("Error sending value on mpsc channel")]
    Send(#[from] mpsc::SendError<()>),
    #[error("Error receiving value on mpsc channel")]
    Recv(#[from] mpsc::RecvError),
    #[error("Unexpected termination. Remote is closed.")]
    RemoteClosed,
    #[error("Received out of order message")]
    UnexpectedMessage,
}

pub fn channel<T: RemoteSend, const BUFFER: usize>(
    local_buffer: usize,
) -> (
    mpsc::Sender<T, remoc::codec::Bincode, BUFFER>,
    mpsc::Receiver<T, remoc::codec::Bincode, BUFFER>,
) {
    let (sender, receiver) = mpsc::channel(local_buffer);
    let sender = sender.set_buffer::<BUFFER>();
    let receiver = receiver.set_buffer::<BUFFER>();
    (sender, receiver)
}

pub async fn establish_sub_channel<Msg, SubMsg, W, E>(
    sender: &mut Sender<Msg>,
    receiver: &mut Receiver<Msg>,
    local_buffer: usize,
    wrap_fn: W,
    extract_fn: E,
) -> Result<(Sender<SubMsg>, Receiver<SubMsg>), CommunicationError>
where
    Msg: RemoteSend,
    SubMsg: RemoteSend,
    W: FnOnce(Receiver<SubMsg>) -> Msg,
    E: FnOnce(Msg) -> Option<Receiver<SubMsg>>,
{
    let (sub_sender, remote_sub_receiver) = channel(local_buffer);
    sender
        .send(wrap_fn(remote_sub_receiver))
        .await
        .map_err(|err| err.without_item())?;
    let msg = receiver
        .recv()
        .await?
        .ok_or(CommunicationError::RemoteClosed)?;
    let sub_receiver = extract_fn(msg).ok_or(CommunicationError::UnexpectedMessage)?;
    Ok((sub_sender, sub_receiver))
}
