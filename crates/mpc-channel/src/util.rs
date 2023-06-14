//! Networking utilities.
use bytes::Bytes;
use futures::{Sink, Stream};
use indexmap::IndexMap;
use pin_project::pin_project;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::io::{Error, IoSlice};
use std::ops::{AddAssign, DivAssign};
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
/// The `Statistics` struct can be serialized into a serializable form via the [`Statistics::into_run_result`]
/// method. The result can be serialized into a variety of formats via
/// [serde](https://serde.rs/#data-formats) (Note: `.csv` output is currently not supported, `json`
/// is recommended).
///
/// Example json output:
/// ```json
///{
///   "meta": {
///     "custom": {
///       "circuit": "sha256.rs"
///     }
///   },
///   "communication": {
///     "Unaccounted": {
///       "sent": 58,
///       "rcvd": 120
///     },
///     "FunctionDependentSetup": {
///       "sent": 75,
///       "rcvd": 13
///     },
///     "Online": {
///       "sent": 36776,
///       "rcvd": 36776
///     }
///   },
///   "time": {
///     "Unaccounted": 0,
///     "FunctionDependentSetup": 0,
///     "Online": 291
///   }
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

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RunResult {
    #[serde(skip_deserializing)]
    pub meta: Metadata,
    pub communication_bytes: IndexMap<Phase, CountPair>,
    pub time_ms: IndexMap<Phase, u128>,
}

trait SerializableMetadata: erased_serde::Serialize + Debug + Send {}
impl<T: ?Sized + erased_serde::Serialize + Debug + Send> SerializableMetadata for T {}

erased_serde::serialize_trait_object!(SerializableMetadata);

#[derive(Debug, Default, Serialize)]
pub struct Metadata {
    data: IndexMap<String, Box<dyn SerializableMetadata>>,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
/// Categories for recorded communication. The `Custom` variant can be used to label the
/// communication with a user chosen string.
pub enum Phase {
    FunctionIndependentSetup,
    FunctionDependentSetup,
    Ots,
    Mts,
    Online,
    Unaccounted,
    Custom(String),
}

#[derive(Default, Debug, Clone)]
struct CounterPair {
    send: Counter,
    recv: Counter,
}

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CountPair {
    pub sent: usize,
    pub rcvd: usize,
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

    /// If `record_as_prev` is set to true (default is false), at the beginning of each phase,
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
        let phase = match (self.unaccounted_as_previous, &self.prev_phase) {
            (true, Some(phase)) => phase.clone(),
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

    /// Convert into a [`RunResult`] which can be serialized via [serde](serde.rs/).
    pub fn into_run_result(self) -> RunResult {
        self.record_unaccounted();
        let recorded = self.recorded.into_inner().unwrap();
        let (communication, time) = recorded
            .into_iter()
            .map(|(phase, (comm, time))| ((phase.clone(), comm), (phase, time.as_millis())))
            .unzip();
        RunResult {
            meta: Default::default(),
            communication_bytes: communication,
            time_ms: time,
        }
    }
}

impl RunResult {
    /// Add any metadata to the `{ "meta": { "custom": { .. }, .. } }` map inside the result.
    /// The value can be of any type that implements [`Serialize`], [`Debug`] and is `'static'.
    ///
    /// ## Example
    /// ```
    ///# use mpc_channel::util::Statistics;
    /// let statistics = Statistics::default();
    /// let mut run_res = statistics.into_run_result();
    /// run_res.add_metadata("Description", "Lorem Ipsum");
    /// run_res.add_metadata("other-data", vec![1, 2, 3]);
    /// ```
    pub fn add_metadata<V: Serialize + Debug + Send + 'static>(&mut self, key: &str, value: V) {
        self.meta.data.insert(key.to_string(), Box::new(value));
    }

    pub fn total_bytes_sent(&self) -> usize {
        self.communication_bytes.values().map(|val| val.sent).sum()
    }

    pub fn total_bytes_recv(&self) -> usize {
        self.communication_bytes.values().map(|val| val.rcvd).sum()
    }

    pub fn setup_ms(&self) -> u128 {
        let phases = [
            Phase::Ots,
            Phase::Mts,
            Phase::FunctionIndependentSetup,
            Phase::FunctionDependentSetup,
        ];
        phases
            .into_iter()
            .map(|phase| self.time_ms.get(&phase).copied().unwrap_or_default())
            .sum()
    }

    pub fn online_ms(&self) -> u128 {
        self.time_ms
            .get(&Phase::Online)
            .copied()
            .unwrap_or_default()
    }

    /// Calculate mean, loses metadata information.
    pub fn mean(data: &[Self]) -> Self {
        let mut res = Self::default();
        for val in data {
            for (k, v) in &val.communication_bytes {
                *res.communication_bytes.entry(k.clone()).or_default() += *v;
            }
            for (k, v) in &val.time_ms {
                *res.time_ms.entry(k.clone()).or_default() += *v;
            }
        }
        for comm in res.communication_bytes.values_mut() {
            *comm /= data.len();
        }
        for time in res.time_ms.values_mut() {
            *time /= data.len() as u128;
        }
        res
    }
}

impl Clone for RunResult {
    fn clone(&self) -> Self {
        let meta = self
            .meta
            .data
            .iter()
            .map(|(k, v)| {
                let json = serde_json::to_string(v)?;
                let json_val: serde_json::Value = serde_json::from_str(&json)?;
                Ok((
                    k.clone(),
                    Box::new(json_val) as Box<dyn SerializableMetadata>,
                ))
            })
            .collect::<Result<IndexMap<_, _>, serde_json::Error>>()
            .unwrap_or_default();
        Self {
            meta: Metadata { data: meta },
            communication_bytes: self.communication_bytes.clone(),
            time_ms: self.time_ms.clone(),
        }
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

impl DivAssign<usize> for CountPair {
    fn div_assign(&mut self, rhs: usize) {
        self.sent /= rhs;
        self.rcvd /= rhs;
    }
}
