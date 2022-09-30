//! Channel abstraction for communication
use futures::{Sink, SinkExt, Stream, TryStream, TryStreamExt};
pub use in_memory::InMemory;
use pin_project::pin_project;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};
pub use tcp::Tcp;
use thiserror::Error;

pub mod in_memory;
pub mod tcp;
pub mod util;

/// Represents a point-to-point two-way communication channel
pub trait Channel<Item>: SinkExt<Item> + TryStreamExt<Ok = Item> + Unpin {
    type StreamPart: TryStreamExt<Ok = Item, Error = <Self as TryStream>::Error> + Send + Unpin;
    type SinkPart: SinkExt<Item, Error = <Self as Sink<Item>>::Error> + Send + Unpin;

    /// Temporarily "constrict" the channel. This intended to be used for a message of type
    /// `SubItem` which is a variant of the `Item` msg.
    ///
    /// # Panics
    /// Should the constricted channel receive an `Item` which can't be converted to a `SubItem`
    /// via [TryInto](`TryInto`), polling it will panic.
    #[allow(clippy::type_complexity)]
    fn constrict<SubItem>(
        &mut self,
    ) -> RwChannel<
        StreamMapTryInto<&mut Self::StreamPart, SubItem>,
        SinkWithInto<&mut Self::SinkPart, Item>,
    >
    where
        Self: Sized,
        SubItem: IsSubMsg<Item>,
    {
        let (stream, sink) = self.split_mut();
        let wrapped_sink = SinkWithInto {
            sink,
            _item: PhantomData::default(),
        };
        let mapped_stream = StreamMapTryInto {
            stream,
            _sub_item: PhantomData::default(),
        };
        RwChannel {
            sink: wrapped_sink,
            stream: mapped_stream,
        }
    }

    /// Split the channel into mutable references to its parts.
    fn split_mut(&mut self) -> (&mut Self::StreamPart, &mut Self::SinkPart);
}

impl<'a, Item, C> Channel<Item> for &'a mut C
where
    C: Channel<Item>,
    &'a mut C: SinkExt<Item, Error = <C as Sink<Item>>::Error>
        + TryStreamExt<Ok = Item, Error = <C as TryStream>::Error>,
{
    type StreamPart = C::StreamPart;
    type SinkPart = C::SinkPart;

    fn split_mut(&mut self) -> (&mut Self::StreamPart, &mut Self::SinkPart) {
        (*self).split_mut()
    }
}

/// Intended for enum member conversion and used for the implementation of
/// [`Channel::constrict`](`Channel::constrict`).
pub trait IsSubMsg<Msg>: Sized {
    type Error: Send + 'static;

    fn into_msg(self) -> Msg;
    fn try_from_msg(msg: Msg) -> Result<Self, Self::Error>;
}

impl<M, SM> IsSubMsg<M> for SM
where
    M: TryInto<Self>,
    M::Error: Send + 'static,
    Self: Into<M> + Sized,
{
    type Error = <M as TryInto<Self>>::Error;

    fn into_msg(self) -> M {
        self.into()
    }

    fn try_from_msg(msg: M) -> Result<Self, Self::Error> {
        msg.try_into()
    }
}

#[pin_project]
/// A generic Channel wrapping a sink and stream.
#[derive(Debug)]
pub struct RwChannel<R, W> {
    #[pin]
    stream: R,
    #[pin]
    sink: W,
}

impl<Item, R, W> Sink<Item> for RwChannel<R, W>
where
    W: Sink<Item>,
{
    type Error = W::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sink.poll_ready(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Item) -> Result<(), Self::Error> {
        let this = self.project();
        this.sink.start_send(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sink.poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sink.poll_close(cx)
    }
}

impl<Item, Err, R, W> Stream for RwChannel<R, W>
where
    R: Stream<Item = Result<Item, Err>>,
{
    type Item = Result<Item, Err>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.stream.poll_next(cx)
    }
}

impl<Item, Err, R, W> Channel<Item> for RwChannel<R, W>
where
    R: Stream<Item = Result<Item, Err>> + Unpin + Send,
    W: Sink<Item> + Unpin + Send,
{
    type StreamPart = R;
    type SinkPart = W;

    fn split_mut(&mut self) -> (&mut Self::StreamPart, &mut Self::SinkPart) {
        (&mut self.stream, &mut self.sink)
    }
}

/// Transforms items sent into Sink by applying [`Into::into`](Into::into)
#[pin_project]
#[derive(Debug)]
pub struct SinkWithInto<S, I> {
    #[pin]
    sink: S,
    _item: PhantomData<I>,
}

/// Transforms items received from Stream by applying [`TryInto::try_into`](TryInto::try_into).
/// If the conversion fails, `poll_next` will panic, as this likely indicates a bug.
#[pin_project]
#[derive(Debug)]
pub struct StreamMapTryInto<S, I> {
    #[pin]
    stream: S,
    _sub_item: PhantomData<I>,
}

impl<Item, SubItem, S> Sink<SubItem> for SinkWithInto<S, Item>
where
    S: Sink<Item>,
    S::Error: 'static,
    SubItem: IsSubMsg<Item>,
{
    type Error = S::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sink.poll_ready(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: SubItem) -> Result<(), Self::Error> {
        let this = self.project();
        let item = item.into_msg();
        this.sink.start_send(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sink.poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.sink.poll_close(cx)
    }
}

impl<Item, SubItem, Err, S> Stream for StreamMapTryInto<S, SubItem>
where
    S: Stream<Item = Result<Item, Err>>,
    SubItem: IsSubMsg<Item>,
    SubItem::Error: Debug,
{
    type Item = Result<SubItem, ConstrictError<SubItem::Error, Err>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.stream.poll_next(cx).map(|val| match val {
            Some(Ok(msg)) => Some(SubItem::try_from_msg(msg).map_err(ConstrictError::TryFrom)),
            Some(Err(err)) => Some(Err(ConstrictError::Stream(err))),
            None => None,
        })
    }
}

#[derive(Error, Debug)]
pub enum ConstrictError<TryFromErr, StreamErr> {
    #[error("TryFrom for Msg failed")]
    TryFrom(TryFromErr),
    #[error("Stream returned error")]
    Stream(StreamErr),
}
