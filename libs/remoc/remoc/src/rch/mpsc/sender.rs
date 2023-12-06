use bytes::Buf;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use std::{
    convert::TryFrom,
    error::Error,
    fmt,
    marker::PhantomData,
    sync::{Arc, Weak},
};

use super::{
    super::{
        base::{self, PortDeserializer, PortSerializer},
        ClosedReason, RemoteSendError, SendErrorExt, BACKCHANNEL_MSG_CLOSE, BACKCHANNEL_MSG_ERROR,
        DEFAULT_BUFFER,
    },
    receiver::RecvError,
    recv_impl, send_impl,
};
use crate::{chmux, codec, RemoteSend};

/// An error occurred during sending over an mpsc channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SendError<T> {
    /// The remote end closed the channel.
    Closed(T),
    /// Sending to a remote endpoint failed.
    RemoteSend(base::SendErrorKind),
    /// Connecting a sent channel failed.
    RemoteConnect(chmux::ConnectError),
    /// Listening for a received channel failed.
    RemoteListen(chmux::ListenerError),
    /// Forwarding at a remote endpoint to another remote endpoint failed.
    RemoteForward,
}

impl<T> SendError<T> {
    /// True, if the remote endpoint closed the channel.
    pub fn is_closed(&self) -> bool {
        matches!(self, Self::Closed(_))
    }

    /// Returns the reason for why the channel has been disconnected.
    ///
    /// Returns [None] if the error is not due to the channel being disconnected.
    /// Currently this can only happen if a serialization error occurred.
    pub fn closed_reason(&self) -> Option<ClosedReason> {
        match self {
            Self::RemoteSend(base::SendErrorKind::Serialize(_)) => None,
            Self::RemoteSend(base::SendErrorKind::Send(chmux::SendError::Closed { .. })) => {
                Some(ClosedReason::Dropped)
            }
            Self::Closed(_) => Some(ClosedReason::Closed),
            _ => Some(ClosedReason::Failed),
        }
    }

    /// True, if the remote endpoint closed the channel, was dropped or the connection failed.
    pub fn is_disconnected(&self) -> bool {
        !matches!(self, Self::RemoteSend(base::SendErrorKind::Serialize(_)))
    }

    /// Returns whether the error is final, i.e. no further send operation can succeed.
    #[deprecated = "a remoc::rch::mpsc::SendError is always final"]
    pub fn is_final(&self) -> bool {
        true
    }

    /// Returns the error without the contained item.
    pub fn without_item(self) -> SendError<()> {
        match self {
            Self::Closed(_) => SendError::Closed(()),
            Self::RemoteSend(err) => SendError::RemoteSend(err),
            Self::RemoteConnect(err) => SendError::RemoteConnect(err),
            Self::RemoteListen(err) => SendError::RemoteListen(err),
            Self::RemoteForward => SendError::RemoteForward,
        }
    }
}

impl<T> SendErrorExt for SendError<T> {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn is_disconnected(&self) -> bool {
        self.is_disconnected()
    }

    fn is_final(&self) -> bool {
        #[allow(deprecated)]
        self.is_final()
    }
}

impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Closed(_) => write!(f, "channel is closed"),
            Self::RemoteSend(err) => write!(f, "send error: {}", err),
            Self::RemoteConnect(err) => write!(f, "connect error: {}", err),
            Self::RemoteListen(err) => write!(f, "listen error: {}", err),
            Self::RemoteForward => write!(f, "forwarding error"),
        }
    }
}

impl<T> Error for SendError<T> where T: fmt::Debug {}

impl<T> SendError<T> {
    fn from_remote_send_error(err: RemoteSendError, value: T) -> Self {
        match err {
            RemoteSendError::Send(err) => Self::RemoteSend(err),
            RemoteSendError::Connect(err) => Self::RemoteConnect(err),
            RemoteSendError::Listen(err) => Self::RemoteListen(err),
            RemoteSendError::Forward => Self::RemoteForward,
            RemoteSendError::Closed => Self::Closed(value),
        }
    }
}

/// An error occurred during trying to send over an mpsc channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TrySendError<T> {
    /// The remote end closed the channel.
    Closed(T),
    /// The data could not be sent on the channel because the channel
    /// is currently full and sending would require blocking.
    Full(T),
    /// Sending to a remote endpoint failed.
    RemoteSend(base::SendErrorKind),
    /// Connecting a sent channel failed.
    RemoteConnect(chmux::ConnectError),
    /// Listening for a received channel failed.
    RemoteListen(chmux::ListenerError),
    /// Forwarding at a remote endpoint to another remote endpoint failed.
    RemoteForward,
}

impl<T> TrySendError<T> {
    /// True, if the remote endpoint closed the channel.
    pub fn is_closed(&self) -> bool {
        matches!(self, Self::Closed(_))
    }

    /// True, if the remote endpoint closed the channel, was dropped or the connection failed.
    pub fn is_disconnected(&self) -> bool {
        !matches!(self, Self::RemoteSend(base::SendErrorKind::Serialize(_)) | Self::Full(_))
    }

    /// Returns whether the error is final, i.e. no further send operation can succeed.
    pub fn is_final(&self) -> bool {
        !matches!(self, Self::Full(_))
    }
}

impl<T> SendErrorExt for TrySendError<T> {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn is_disconnected(&self) -> bool {
        self.is_disconnected()
    }

    fn is_final(&self) -> bool {
        self.is_final()
    }
}

impl<T> fmt::Display for TrySendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Closed(_) => write!(f, "channel is closed"),
            Self::Full(_) => write!(f, "channel is full"),
            Self::RemoteSend(err) => write!(f, "send error: {}", err),
            Self::RemoteConnect(err) => write!(f, "connect error: {}", err),
            Self::RemoteListen(err) => write!(f, "listen error: {}", err),
            Self::RemoteForward => write!(f, "forwarding error"),
        }
    }
}

impl<T> TrySendError<T> {
    fn from_remote_send_error(err: RemoteSendError, value: T) -> Self {
        match err {
            RemoteSendError::Send(err) => Self::RemoteSend(err),
            RemoteSendError::Connect(err) => Self::RemoteConnect(err),
            RemoteSendError::Listen(err) => Self::RemoteListen(err),
            RemoteSendError::Forward => Self::RemoteForward,
            RemoteSendError::Closed => Self::Closed(value),
        }
    }
}

impl<T> From<SendError<T>> for TrySendError<T> {
    fn from(err: SendError<T>) -> Self {
        match err {
            SendError::Closed(v) => Self::Closed(v),
            SendError::RemoteSend(err) => Self::RemoteSend(err),
            SendError::RemoteConnect(err) => Self::RemoteConnect(err),
            SendError::RemoteListen(err) => Self::RemoteListen(err),
            SendError::RemoteForward => Self::RemoteForward,
        }
    }
}

impl<T> TryFrom<TrySendError<T>> for SendError<T> {
    type Error = TrySendError<T>;

    fn try_from(err: TrySendError<T>) -> Result<Self, Self::Error> {
        match err {
            TrySendError::Closed(v) => Ok(Self::Closed(v)),
            TrySendError::RemoteSend(err) => Ok(Self::RemoteSend(err)),
            TrySendError::RemoteConnect(err) => Ok(Self::RemoteConnect(err)),
            TrySendError::RemoteForward => Ok(Self::RemoteForward),
            other => Err(other),
        }
    }
}

impl<T> Error for TrySendError<T> where T: fmt::Debug {}

/// Send values to the associated [Receiver](super::Receiver), which may be located on a remote endpoint.
///
/// Instances are created by the [channel](super::channel) function.
pub struct Sender<T, Codec = codec::Default, const BUFFER: usize = DEFAULT_BUFFER> {
    tx: Weak<tokio::sync::mpsc::Sender<Result<T, RecvError>>>,
    closed_rx: tokio::sync::watch::Receiver<Option<ClosedReason>>,
    remote_send_err_rx: tokio::sync::watch::Receiver<Option<RemoteSendError>>,
    dropped_tx: tokio::sync::mpsc::Sender<()>,
    _codec: PhantomData<Codec>,
}

impl<T, Codec, const BUFFER: usize> fmt::Debug for Sender<T, Codec, BUFFER> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Sender").finish()
    }
}

impl<T, Codec, const BUFFER: usize> Clone for Sender<T, Codec, BUFFER> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            closed_rx: self.closed_rx.clone(),
            remote_send_err_rx: self.remote_send_err_rx.clone(),
            dropped_tx: self.dropped_tx.clone(),
            _codec: PhantomData,
        }
    }
}

/// Mpsc sender in transport.
#[derive(Serialize, Deserialize)]
pub(crate) struct TransportedSender<T, Codec> {
    /// chmux port number. `None` if closed.
    port: Option<u32>,
    /// Data type.
    data: PhantomData<T>,
    /// Data codec.
    codec: PhantomData<Codec>,
}

impl<T, Codec, const BUFFER: usize> Sender<T, Codec, BUFFER>
where
    T: Send + 'static,
{
    /// Creates a new sender.
    pub(crate) fn new(
        tx: tokio::sync::mpsc::Sender<Result<T, RecvError>>,
        mut closed_rx: tokio::sync::watch::Receiver<Option<ClosedReason>>,
        remote_send_err_rx: tokio::sync::watch::Receiver<Option<RemoteSendError>>,
    ) -> Self {
        let tx = Arc::new(tx);
        let (dropped_tx, mut dropped_rx) = tokio::sync::mpsc::channel(1);

        let this = Self {
            tx: Arc::downgrade(&tx),
            closed_rx: closed_rx.clone(),
            remote_send_err_rx,
            dropped_tx,
            _codec: PhantomData,
        };

        // Drop strong reference to sender when channel is closed.
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    res = closed_rx.changed() => {
                        match res {
                            Ok(()) if closed_rx.borrow().is_some() => break,
                            Ok(()) => (),
                            Err(_) => break,
                        }
                    },
                    _ = dropped_rx.recv() => break,
                }
            }

            drop(tx);
        });

        this
    }

    /// Creates a new sender that is closed.
    pub(crate) fn new_closed() -> Self {
        Self {
            tx: Weak::new(),
            closed_rx: tokio::sync::watch::channel(Some(ClosedReason::Closed)).1,
            remote_send_err_rx: tokio::sync::watch::channel(None).1,
            dropped_tx: tokio::sync::mpsc::channel(1).0,
            _codec: PhantomData,
        }
    }

    /// Sends a value over this channel.
    #[inline]
    pub async fn send(&self, value: T) -> Result<(), SendError<T>> {
        if let Some(err) = self.remote_send_err_rx.borrow().as_ref() {
            return Err(SendError::from_remote_send_error(err.clone(), value));
        }

        if let Some(tx) = self.tx.upgrade() {
            if let Err(err) = tx.send(Ok(value)).await {
                return Err(SendError::Closed(err.0.expect("unreachable")));
            }
        } else {
            return Err(SendError::Closed(value));
        }

        Ok(())
    }

    /// Attempts to immediately send a message over this channel.
    #[inline]
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        if let Some(err) = self.remote_send_err_rx.borrow().as_ref() {
            return Err(TrySendError::from_remote_send_error(err.clone(), value));
        }

        match self.tx.upgrade() {
            Some(tx) => match tx.try_send(Ok(value)) {
                Ok(()) => Ok(()),
                Err(tokio::sync::mpsc::error::TrySendError::Full(err)) => {
                    Err(TrySendError::Full(err.expect("unreachable")))
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(err)) => {
                    Err(TrySendError::Closed(err.expect("unreachable")))
                }
            },
            None => Err(TrySendError::Closed(value)),
        }
    }

    /// Blocking send to call outside of asynchronous contexts.
    ///
    /// # Panics
    /// This function panics if called within an asynchronous execution context.
    #[inline]
    pub fn blocking_send(&self, value: T) -> Result<(), SendError<T>> {
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(self.send(value))
    }

    /// Wait for channel capacity, returning an owned permit.
    /// Once capacity to send one message is available, it is reserved for the caller.
    #[inline]
    pub async fn reserve(&self) -> Result<Permit<T>, SendError<()>> {
        if let Some(err) = self.remote_send_err_rx.borrow().as_ref() {
            return Err(SendError::from_remote_send_error(err.clone(), ()));
        }

        if let Some(tx) = self.tx.upgrade() {
            let tx = (*tx).clone();
            match tx.reserve_owned().await {
                Ok(permit) => Ok(Permit(permit)),
                Err(_) => Err(SendError::Closed(())),
            }
        } else {
            Err(SendError::Closed(()))
        }
    }

    /// Returns the current capacity of the channel.
    ///
    /// Zero is returned when the channel has been closed or an error has occurred.
    #[inline]
    pub fn capacity(&self) -> usize {
        match self.tx.upgrade() {
            Some(tx) => tx.capacity(),
            None => 0,
        }
    }

    /// Completes when the receiver has been closed, dropped or the connection failed.
    ///
    /// Use [closed_reason](Self::closed_reason) to obtain the cause for closure.
    #[inline]
    pub async fn closed(&self) {
        let mut closed = self.closed_rx.clone();
        while closed.borrow().is_none() {
            if closed.changed().await.is_err() {
                break;
            }
        }
    }

    /// Returns the reason for why the channel has been closed.
    ///
    /// Returns [None] if the channel is not closed.
    #[inline]
    pub fn closed_reason(&self) -> Option<ClosedReason> {
        match (self.closed_rx.borrow().clone(), self.remote_send_err_rx.borrow().as_ref()) {
            (Some(reason), _) => Some(reason),
            (None, Some(_)) => Some(ClosedReason::Failed),
            (None, None) => None,
        }
    }

    /// Returns whether the receiver has been closed, dropped or the connection failed.
    ///
    /// Use [closed_reason](Self::closed_reason) to obtain the cause for closure.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.closed_reason().is_some()
    }

    /// Sets the codec that will be used when sending this sender to a remote endpoint.
    pub fn set_codec<NewCodec>(self) -> Sender<T, NewCodec, BUFFER> {
        Sender {
            tx: self.tx.clone(),
            closed_rx: self.closed_rx.clone(),
            remote_send_err_rx: self.remote_send_err_rx.clone(),
            dropped_tx: self.dropped_tx.clone(),
            _codec: PhantomData,
        }
    }

    /// Sets the buffer size that will be used when sending this sender to a remote endpoint.
    pub fn set_buffer<const NEW_BUFFER: usize>(self) -> Sender<T, Codec, NEW_BUFFER> {
        assert!(NEW_BUFFER > 0, "buffer size must not be zero");
        Sender {
            tx: self.tx.clone(),
            closed_rx: self.closed_rx.clone(),
            remote_send_err_rx: self.remote_send_err_rx.clone(),
            dropped_tx: self.dropped_tx.clone(),
            _codec: PhantomData,
        }
    }
}

/// Owned permit to send one value into the channel.
pub struct Permit<T>(tokio::sync::mpsc::OwnedPermit<Result<T, RecvError>>);

impl<T> Permit<T>
where
    T: Send,
{
    /// Sends a value using the reserved capacity.
    #[inline]
    pub fn send(self, value: T) {
        self.0.send(Ok(value));
    }
}

impl<T, Codec, const BUFFER: usize> Drop for Sender<T, Codec, BUFFER> {
    fn drop(&mut self) {
        // empty
    }
}

impl<T, Codec, const BUFFER: usize> Serialize for Sender<T, Codec, BUFFER>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    /// Serializes this sender for sending over a chmux channel.
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let port = match self.tx.upgrade() {
            // Channel is open.
            Some(tx) => {
                // Prepare channel for takeover.
                let mut closed_rx = self.closed_rx.clone();
                let mut remote_send_err_rx = self.remote_send_err_rx.clone();

                Some(PortSerializer::connect(|connect| {
                    async move {
                        // Establish chmux channel.
                        let (mut raw_tx, raw_rx) = match connect.await {
                            Ok(tx_rx) => tx_rx,
                            Err(err) => {
                                let _ = tx.send(Err(RecvError::RemoteConnect(err))).await;
                                return;
                            }
                        };

                        recv_impl!(T, tx, raw_tx, raw_rx, remote_send_err_rx, closed_rx);
                    }
                    .boxed()
                })?)
            }
            None => {
                // Channel is closed.
                None
            }
        };

        // Encode chmux port number in transport type and serialize it.
        let transported = TransportedSender::<T, Codec> { port, data: PhantomData, codec: PhantomData };
        transported.serialize(serializer)
    }
}

impl<'de, T, Codec, const BUFFER: usize> Deserialize<'de> for Sender<T, Codec, BUFFER>
where
    T: RemoteSend,
    Codec: codec::Codec,
{
    /// Deserializes this sender after it has been received over a chmux channel.
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        assert!(BUFFER > 0, "BUFFER must not be zero");

        // Get chmux port number from deserialized transport type.
        let TransportedSender { port, .. } = TransportedSender::<T, Codec>::deserialize(deserializer)?;

        match port {
            // Received channel is open.
            Some(port) => {
                // Create internal communication channels.
                let (tx, mut rx) = tokio::sync::mpsc::channel(BUFFER);
                let (closed_tx, closed_rx) = tokio::sync::watch::channel(None);
                let (remote_send_err_tx, remote_send_err_rx) = tokio::sync::watch::channel(None);

                // Accept chmux port request.
                PortDeserializer::accept(port, |local_port, request| {
                    async move {
                        // Accept chmux connection request.
                        let (raw_tx, mut raw_rx) = match request.accept_from(local_port).await {
                            Ok(tx_rx) => tx_rx,
                            Err(err) => {
                                let _ = remote_send_err_tx.send(Some(RemoteSendError::Listen(err)));
                                return;
                            }
                        };

                        send_impl!(T, rx, raw_tx, raw_rx, remote_send_err_tx, closed_tx);
                    }
                    .boxed()
                })?;

                Ok(Self::new(tx, closed_rx, remote_send_err_rx))
            }

            // Received closed channel.
            None => Ok(Self::new_closed()),
        }
    }
}
