//! In-memory implementation of a channel.
use super::Channel;
use futures::channel::mpsc;
use futures::channel::mpsc::{unbounded, SendError};
use futures::{Sink, Stream};
use pin_project::pin_project;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};

#[pin_project]
/// In-memory channel using unbounded channels. Intended for testing purposes.
pub struct InMemory<Item> {
    #[pin]
    sender: mpsc::UnboundedSender<Item>,
    #[pin]
    receiver: InfallibleStream<mpsc::UnboundedReceiver<Item>>,
}

// Helper struct until https://github.com/rust-lang/rust/issues/63063 is stabilized
#[pin_project]
pub struct InfallibleStream<S> {
    #[pin]
    stream: S,
}

impl<Item, S> Stream for InfallibleStream<S>
where
    S: Stream<Item = Item>,
{
    type Item = Result<Item, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.stream.poll_next(cx).map(|val| val.map(Ok))
    }
}

impl<Item> Stream for InMemory<Item> {
    type Item = Result<Item, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.receiver.poll_next(cx)
    }
}

impl<Item> Sink<Item> for InMemory<Item> {
    type Error = SendError;

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

impl<Item> InMemory<Item> {
    pub fn new_pair() -> (InMemory<Item>, InMemory<Item>) {
        let (s1, r1) = unbounded();
        let (s2, r2) = unbounded();
        let t1 = InMemory {
            sender: s1,
            receiver: InfallibleStream { stream: r2 },
        };
        let t2 = InMemory {
            sender: s2,
            receiver: InfallibleStream { stream: r1 },
        };
        (t1, t2)
    }
}

impl<Item> Channel<Item> for InMemory<Item>
where
    Item: Send,
{
    type StreamPart = InfallibleStream<mpsc::UnboundedReceiver<Item>>;
    type SinkPart = mpsc::UnboundedSender<Item>;

    fn split_mut(&mut self) -> (&mut Self::StreamPart, &mut Self::SinkPart) {
        (&mut self.receiver, &mut self.sender)
    }
}
