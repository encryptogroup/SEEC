//! Networking utilities.
use bytes::Bytes;
use futures::{Sink, Stream};
use pin_project::pin_project;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use std::collections::HashMap;
use std::future::Future;
use std::io::{Error, IoSlice};
use std::ops::AddAssign;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::{io, mem};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// [AsyncWriter](`AsyncWrite`) that tracks the number of bytes written.
#[pin_project]
pub struct TrackingWriter<AsyncWriter> {
    #[pin]
    writer: AsyncWriter,
    bytes_written: Counter,
}

/// [AsyncReader](`AsyncRead`) that tracks the number of bytes read.
#[pin_project]
pub struct TrackingReader<AsyncReader> {
    #[pin]
    reader: AsyncReader,
    bytes_read: Counter,
}

#[derive(Clone, Default, Debug)]
/// A counter that tracks communication in bytes sent or received. Can be used with the
/// [`CommStatistics`] struct to track communication in different phases.
pub struct Counter(Arc<AtomicUsize>);

/// A utility struct which is used to track communication on a **best-effort basis**. When created, it is initialized with
/// the [`Counter`]s of the main channel. Optionally, a counter pair for a helper channel can be
/// set.
///
/// # Caveat
/// As the actual sending of values transmitted via channels is done in an asynchronous background
/// task, `CommStatistics` can only record the communication on a best-effort basis. It is possible
/// for values that are sent to a channel within a [`CommStatistics::record`] call to not be
/// tracked as the specified [`Communication`], but rather as `Unaccounted`. If the amount of
/// unaccounted communication is higher than desired, adding a [`tokio::time::sleep`] at the end
/// of the `record` call might reduce it.
#[derive(Default)]
pub struct CommStatistics {
    main: CounterPair,
    helper: Option<CounterPair>,
    recorded: Mutex<HashMap<Communication, CountPair>>,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize)]
/// Categories for recorded communication. The `Custom` variant can be used to label the
/// communication with a user chosen string.
pub enum Communication {
    FunctionIndependentSetup,
    FunctionDependentSetup,
    Ots,
    Mts,
    Online,
    Unaccounted,
    Custom(&'static str),
}

#[derive(Default, Debug, Clone)]
struct CounterPair {
    send: Counter,
    recv: Counter,
}

#[derive(Default, Debug, Clone, Serialize)]
struct CountPair {
    sent: usize,
    recv: usize,
}

impl<AsyncWriter> TrackingWriter<AsyncWriter> {
    pub fn new(writer: AsyncWriter) -> Self {
        Self {
            writer,
            bytes_written: Counter::default(),
        }
    }

    #[inline]
    pub fn bytes_written(&self) -> Counter {
        self.bytes_written.clone()
    }

    pub fn reset(&mut self) {
        self.bytes_written.reset();
    }
}

impl<AsyncReader> TrackingReader<AsyncReader> {
    pub fn new(reader: AsyncReader) -> Self {
        Self {
            reader,
            bytes_read: Counter::default(),
        }
    }

    #[inline]
    pub fn bytes_read(&self) -> Counter {
        self.bytes_read.clone()
    }

    pub fn reset(&mut self) {
        self.bytes_read.reset();
    }
}

impl<AW: AsyncWrite> AsyncWrite for TrackingWriter<AW> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Error>> {
        let this = self.project();
        let poll = this.writer.poll_write(cx, buf);
        if let Poll::Ready(Ok(bytes_written)) = &poll {
            *this.bytes_written += *bytes_written;
        }
        poll
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        let this = self.project();
        this.writer.poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        let this = self.project();
        this.writer.poll_shutdown(cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<Result<usize, Error>> {
        let this = self.project();
        let poll = this.writer.poll_write_vectored(cx, bufs);
        if let Poll::Ready(Ok(bytes_written)) = &poll {
            *this.bytes_written += *bytes_written;
        }
        poll
    }

    fn is_write_vectored(&self) -> bool {
        self.writer.is_write_vectored()
    }
}

impl<AR: AsyncRead> AsyncRead for TrackingReader<AR> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let bytes_before = buf.filled().len();
        let this = self.project();
        let poll = this.reader.poll_read(cx, buf);
        *this.bytes_read += buf.filled().len() - bytes_before;
        poll
    }
}

impl<S: Sink<Bytes>> Sink<Bytes> for TrackingWriter<S> {
    type Error = S::Error;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.writer.poll_ready(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: Bytes) -> Result<(), Self::Error> {
        // The size_of<u32> adds the size of the length tag which we'd use when actually
        // using a framed transport
        let this = self.project();
        *this.bytes_written += item.len() + mem::size_of::<u32>();
        this.writer.start_send(item)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.writer.poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.project();
        this.writer.poll_close(cx)
    }
}

impl<S: Stream<Item = Bytes>> Stream for TrackingReader<S> {
    type Item = Bytes;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let poll = this.reader.poll_next(cx);
        if let Poll::Ready(Some(bytes)) = &poll {
            // The size_of<u32> adds the size of the length tag which we'd use when actually
            // using a framed transport
            *this.bytes_read += bytes.len() + mem::size_of::<u32>();
        }
        poll
    }
}

impl Counter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    pub fn reset(&self) -> usize {
        self.0.swap(0, Ordering::SeqCst)
    }
}

impl AddAssign<usize> for Counter {
    fn add_assign(&mut self, rhs: usize) {
        self.0.fetch_add(rhs, Ordering::SeqCst);
    }
}

impl CommStatistics {
    /// Create a new [`CommStatistics`] with the counters for the main channel.
    pub fn new(send_counter: Counter, recv_counter: Counter) -> Self {
        Self {
            main: CounterPair {
                send: send_counter,
                recv: recv_counter,
            },
            helper: None,
            recorded: Default::default(),
        }
    }

    /// Add the helper counters. This might be used to track the communication with a trusted third
    /// party.
    pub fn with_helper(
        mut self,
        helper_send_counter: Counter,
        helper_recv_counter: Counter,
    ) -> Self {
        self.set_helper(helper_send_counter, helper_recv_counter);
        self
    }

    /// Set the helper counters. This might be used to track the communication with a trusted third
    /// party.
    pub fn set_helper(&mut self, helper_send_counter: Counter, helper_recv_counter: Counter) {
        self.helper = Some(CounterPair {
            send: helper_send_counter,
            recv: helper_recv_counter,
        });
    }

    /// Record the main channel communication that happens within the future `f` on a
    /// best-effort basis.
    pub async fn record<F, R>(&mut self, comm: Communication, f: F) -> R
    where
        F: Future<Output = R>,
    {
        self.record_for(self.main.clone(), comm, f).await
    }

    /// Record the helper channel communication that happens within the future `f` on a
    /// best-effort basis.
    pub async fn record_helper<F, R>(&mut self, comm: Communication, f: F) -> R
    where
        F: Future<Output = R>,
    {
        let helper = self
            .helper
            .clone()
            .expect("Helper counter must be set to record helper communication");
        self.record_for(helper, comm, f).await
    }

    async fn record_for<F, R>(&mut self, cnt_pair: CounterPair, comm: Communication, f: F) -> R
    where
        F: Future<Output = R>,
    {
        self.record_unaccounted();
        let ret = f.await;
        let accounted = cnt_pair.reset();
        let mut recorded = self.recorded.lock().unwrap();
        let entry = recorded.entry(comm).or_default();
        *entry += accounted;

        ret
    }

    fn record_unaccounted(&self) {
        let mut unaccounted = self.main.reset();
        if let Some(helper) = &self.helper {
            unaccounted += helper.reset();
        }
        let mut recorded = self.recorded.lock().unwrap();
        let unaccounted_entry = recorded.entry(Communication::Unaccounted).or_default();
        *unaccounted_entry += unaccounted;
    }
}

impl CounterPair {
    fn reset(&self) -> CountPair {
        CountPair {
            sent: self.send.reset(),
            recv: self.recv.reset(),
        }
    }
}

impl AddAssign for CountPair {
    fn add_assign(&mut self, rhs: Self) {
        self.sent += rhs.sent;
        self.recv += rhs.recv;
    }
}

// Custom serialization implementation which records the unaccounted communication before
// serialization. The Communication::Custom variant is serialized as the internal string.
impl Serialize for CommStatistics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.record_unaccounted();
        let recorded = self.recorded.lock().unwrap();

        let mut map = serializer.serialize_map(Some(recorded.len()))?;
        for (k, v) in recorded.iter() {
            match k {
                Communication::Custom(custom) => {
                    map.serialize_entry(custom, v)?;
                }
                k => {
                    map.serialize_entry(k, v)?;
                }
            }
        }
        map.end()
    }
}
