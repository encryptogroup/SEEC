use futures::channel::mpsc;
use futures::channel::mpsc::{unbounded, SendError};
use futures::{Sink, SinkExt, Stream, StreamExt};
use pin_project::pin_project;
use std::fmt::Debug;
use std::pin::Pin;
use std::task::{Context, Poll};

pub trait Transport<Item, Err: Debug>:
    SinkExt<Item, Error = Err> + StreamExt<Item = Item> + Unpin
{
}
impl<Item, Err: Debug, C: SinkExt<Item, Error = Err> + StreamExt<Item = Item> + Unpin>
    Transport<Item, Err> for C
{
}

#[pin_project]
pub struct InMemory<Item> {
    #[pin]
    sender: mpsc::UnboundedSender<Item>,
    #[pin]
    receiver: mpsc::UnboundedReceiver<Item>,
}

impl<Item> Stream for InMemory<Item> {
    type Item = Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        this.receiver.poll_next(cx)
    }
}

impl<Item> Sink<Item> for InMemory<Item> {
    type Error = SendError;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let mut this = self.project();
        this.sender.poll_ready(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Item) -> Result<(), Self::Error> {
        let mut this = self.project();
        this.sender.start_send(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let mut this = self.project();
        this.sender.poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let mut this = self.project();
        this.sender.poll_close(cx)
    }
}

impl<Item> InMemory<Item> {
    pub fn new_pair() -> (InMemory<Item>, InMemory<Item>) {
        let (s1, r1) = unbounded();
        let (s2, r2) = unbounded();
        let t1 = InMemory {
            sender: s1,
            receiver: r2,
        };
        let t2 = InMemory {
            sender: s2,
            receiver: r1,
        };
        (t1, t2)
    }
}
