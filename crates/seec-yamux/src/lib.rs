use std::collections::VecDeque;
use std::future::{Future, poll_fn};
use std::task::Poll;
use futures::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot};
use yamux::Stream;

pub struct Connection<T> {
    inner: yamux::Connection<T>,
    cmd_recv: mpsc::Receiver<Cmd>,
    buffered_inbound: VecDeque<Stream>,
    requested_inbound: VecDeque<oneshot::Sender<Stream>>,
    requested_outbound: VecDeque<oneshot::Sender<Stream>>,
}

pub enum Cmd {
    NewChannel((oneshot::Sender<Stream>, oneshot::Sender<Stream>))
}


impl<T: AsyncRead + AsyncWrite + Unpin> Connection<T> {
    pub fn spawn(mut self) -> impl Future<Output = ()> {
        poll_fn(move |cx| {
            match self.inner.poll_next_inbound(cx) {
                Poll::Ready(Some(Ok(stream))) => self.buffered_inbound.push_back(stream),
                _ => ()
            }
            match self.cmd_recv.poll_recv(cx) {
                Poll::Ready(Some(Cmd::NewChannel((outbound, inbound)))) => {
                    if let Some(stream) = self.buffered_inbound.pop_front() {
                        inbound.send(stream).unwrap();
                        self.requested_outbound.push_back(outbound);
                    } else {
                        self.requested_inbound.push_back(inbound);
                        self.requested_outbound.push_back(outbound);
                    }
                },
                _ => ()
            }
            if !self.requested_outbound.is_empty() {
                match self.inner.poll_new_outbound(cx) {
                    Poll::Ready(Ok(stream)) => {
                        if let Some(outbound) = self.requested_outbound.pop_front() {
                            outbound.send(stream).unwrap();
                        }
                    }
                    _ => ()
                }
            }
            Poll::Pending
        })
    }
}