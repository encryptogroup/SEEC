use pin_project::pin_project;
use std::io;
use std::io::{Error, IoSlice};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

#[pin_project]
pub(super) struct TrackingWriter<AsyncWriter> {
    #[pin]
    writer: AsyncWriter,
    bytes_written: usize,
}

#[pin_project]
pub(super) struct TrackingReader<AsyncReader> {
    #[pin]
    reader: AsyncReader,
    bytes_read: usize,
}

impl<AsyncWriter> TrackingWriter<AsyncWriter> {
    pub fn new(writer: AsyncWriter) -> Self {
        Self {
            writer,
            bytes_written: 0,
        }
    }

    #[inline]
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }
}

impl<AsyncReader> TrackingReader<AsyncReader> {
    pub fn new(reader: AsyncReader) -> Self {
        Self {
            reader,
            bytes_read: 0,
        }
    }

    #[inline]
    pub fn bytes_read(&self) -> usize {
        self.bytes_read
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
            *this.bytes_written += bytes_written;
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
            *this.bytes_written += bytes_written;
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
