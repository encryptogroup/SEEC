//! Networking utilities.
use bytes::Bytes;
use futures::{Sink, Stream};
use indexmap::IndexMap;
use pin_project::pin_project;
use serde::ser::{Impossible, SerializeMap};
use serde::{Serialize, Serializer};
use std::future::Future;
use std::io::{Error, IoSlice};
use std::ops::AddAssign;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
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
/// [`Statistics`] struct to track communication in different phases.
pub struct Counter(Arc<AtomicUsize>);

/// A utility struct which is used to track communication on a **best-effort basis**.
///
/// When created, it is initialized with the [`Counter`]s of the main channel.
/// Optionally, a counter pair for a helper channel can be set.
///
/// # Serialization
/// The `Statistics` struct can be serialized into a variety of formats via
/// [serde](https://serde.rs/#data-formats).
///
/// Example json output:
/// ```json
///{
///   "helper-mts_comm_bytes_sent": 94,
///   "helper-mts_comm_bytes_recvd": 94,
///   "helper-mts_time_ms": 0,
///   "Unaccounted_comm_bytes_sent": 194,
///   "Unaccounted_comm_bytes_recvd": 194,
///   "Unaccounted_time_ms": 0,
///   "Online_comm_bytes_sent": 68938,
///   "Online_comm_bytes_recvd": 68938,
///   "Online_time_ms": 243
/// }
/// ```
///
/// # Caveat
/// As the actual sending of values transmitted via channels is done in an asynchronous background
/// task, `Statistics` can only record the communication on a best-effort basis. It is possible
/// for values that are sent to a channel within a [`Statistics::record`] call to not be
/// tracked as the specified [`Phase`], but rather as `Unaccounted`. If the amount of
/// unaccounted communication is higher than desired, adding a [`tokio::time::sleep`] at the end
/// of the `record` call might reduce it.
#[derive(Default)]
pub struct Statistics {
    main: CounterPair,
    helper: Option<CounterPair>,
    // IndexMap so that iteration order is not random
    recorded: Mutex<IndexMap<Phase, (CountPair, Duration)>>,
    prev_phase: Option<Phase>,
    // record unaccounted communication as the last phase
    unaccounted_as_previous: bool,
    // sleep duration after every `record`. Can reduce unaccounted comm
    sleep_after_phase: Duration,
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, Serialize)]
/// Categories for recorded communication. The `Custom` variant can be used to label the
/// communication with a user chosen string.
pub enum Phase {
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

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct CountPair {
    sent: usize,
    rcvd: usize,
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

impl Statistics {
    /// Create a new [`Statistics`] with the counters for the main channel.
    pub fn new(send_counter: Counter, recv_counter: Counter) -> Self {
        Self {
            main: CounterPair {
                send: send_counter,
                recv: recv_counter,
            },
            ..Default::default()
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

    /// Add a sleep duration at the end of a `record` or `record_helper` call to reduce
    /// unaccounted communication. This time is not part of the tracked statistics.
    pub fn with_sleep(mut self, sleep: Duration) -> Self {
        self.sleep_after_phase = sleep;
        self
    }

    /// If `record_as_last` is set to true (default is false), at the beginning of each phase,
    /// every unaccounted communication is recorded as the previous phase. If communication
    /// occurs before the first phase, it is still recorded as unaccounted.
    pub fn without_unaccounted(mut self, record_as_prev: bool) -> Self {
        self.unaccounted_as_previous = record_as_prev;
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

    /// Set the sleep duration at the end of a `record` or `record_helper` call to reduce
    /// unaccounted communication. This time is not part of the tracked statistics.
    pub fn set_sleep(&mut self, sleep: Duration) {
        self.sleep_after_phase = sleep;
    }

    pub fn set_without_unaccounted(&mut self, record_as_prev: bool) {
        self.unaccounted_as_previous = record_as_prev;
    }

    /// Record the main channel communication that happens within the future `f` on a
    /// best-effort basis.
    pub async fn record<F, R>(&mut self, comm: Phase, f: F) -> R
    where
        F: Future<Output = R>,
    {
        self.record_for(self.main.clone(), comm, f).await
    }

    /// Record the helper channel communication that happens within the future `f` on a
    /// best-effort basis.
    pub async fn record_helper<F, R>(&mut self, phase: Phase, f: F) -> R
    where
        F: Future<Output = R>,
    {
        let helper = self
            .helper
            .clone()
            .expect("Helper counter must be set to record helper communication");
        self.record_for(helper, phase, f).await
    }

    async fn record_for<F, R>(&mut self, cnt_pair: CounterPair, phase: Phase, f: F) -> R
    where
        F: Future<Output = R>,
    {
        self.record_unaccounted();
        let now = Instant::now();
        let ret = f.await;
        let elapsed = now.elapsed();
        tokio::time::sleep(self.sleep_after_phase).await;
        let comm = cnt_pair.reset();
        let mut recorded = self.recorded.lock().unwrap();
        let entry = recorded.entry(phase).or_default();
        entry.0 += comm;
        entry.1 += elapsed;

        ret
    }

    fn record_unaccounted(&self) {
        let mut unaccounted = self.main.reset();
        if let Some(helper) = &self.helper {
            unaccounted += helper.reset();
        }
        let mut recorded = self.recorded.lock().unwrap();
        let phase = match (self.unaccounted_as_previous, self.prev_phase) {
            (true, Some(phase)) => phase,
            _ => Phase::Unaccounted,
        };
        let phase_entry = recorded.entry(phase).or_default();
        phase_entry.0 += unaccounted;
    }

    /// Get the statistics for a phase.
    pub fn get(self, phase: &Phase) -> Option<(CountPair, Duration)> {
        self.record_unaccounted();
        self.recorded.lock().unwrap().get(phase).copied()
    }
}

impl CounterPair {
    fn reset(&self) -> CountPair {
        CountPair {
            sent: self.send.reset(),
            rcvd: self.recv.reset(),
        }
    }
}

impl AddAssign for CountPair {
    fn add_assign(&mut self, rhs: Self) {
        self.sent += rhs.sent;
        self.rcvd += rhs.rcvd;
    }
}

// Custom serialization implementation which records the unaccounted communication before
// serialization. The Communication::Custom variant is serialized as the internal string.
impl Serialize for Statistics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.record_unaccounted();
        let recorded = self.recorded.lock().unwrap();

        let mut map = serializer.serialize_map(Some(recorded.len()))?;
        for (k, v) in recorded.iter() {
            map.serialize_entry(
                &WithPostfix {
                    delegate: k,
                    postfix: "_comm_bytes_sent",
                },
                &v.0.sent,
            )?;

            map.serialize_entry(
                &WithPostfix {
                    delegate: k,
                    postfix: "_comm_bytes_rcvd",
                },
                &v.0.rcvd,
            )?;

            map.serialize_entry(
                &WithPostfix {
                    delegate: k,
                    postfix: "_time_ms",
                },
                &v.1.as_millis(),
            )?;
        }
        map.end()
    }
}

// The below is a fairly verbose implementation of a helper struct which is used to
// deserialize the Phase variants with a chosen postfix. The majority of the Serializer methods
// are not needed and can be ignored. The important implementations are at the top.

struct WithPostfix<T> {
    delegate: T,
    postfix: &'static str,
}

impl<T: Serialize> Serialize for WithPostfix<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.delegate.serialize(WithPostfix {
            delegate: serializer,
            postfix: self.postfix,
        })
    }
}

#[allow(unused_variables)]
impl<S: Serializer> Serializer for WithPostfix<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = Impossible<S::Ok, S::Error>;
    type SerializeTuple = Impossible<S::Ok, S::Error>;
    type SerializeTupleStruct = Impossible<S::Ok, S::Error>;
    type SerializeTupleVariant = Impossible<S::Ok, S::Error>;
    type SerializeMap = Impossible<S::Ok, S::Error>;
    type SerializeStruct = Impossible<S::Ok, S::Error>;
    type SerializeStructVariant = Impossible<S::Ok, S::Error>;

    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        self.delegate
            .collect_str(&format_args!("{}{}", v, self.postfix))
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        value.serialize(WithPostfix {
            delegate: self.delegate,
            postfix: self.postfix,
        })
    }

    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.serialize_str(variant)
    }

    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_some<T: ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_unit_struct(self, name: &'static str) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        todo!()
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        todo!()
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        todo!()
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        todo!()
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        todo!()
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        todo!()
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        todo!()
    }
}
