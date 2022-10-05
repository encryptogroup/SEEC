use remoc::rch::mpsc::{RecvError, SendError};
use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {}

#[derive(Error, Debug)]
pub enum CircuitError {
    #[error("Unable to save circuit as dot file")]
    SaveAsDot(#[source] io::Error),
    #[error("Unable to load bristol file")]
    LoadBristol(#[from] BristolError),
    #[error("Unable to convert bristol circuit")]
    ConversionError,
}

#[derive(Debug, Error)]
pub enum BristolError {
    #[error("Unable to read bristol file")]
    ReadFailed(#[from] io::Error),
    #[error("Unable to parse bristol file")]
    ParseFailed(#[from] nom::Err<nom::error::Error<String>>),
}

#[derive(Debug, Error)]
pub enum MTProviderError<Request> {
    #[error("Sending MT request failed")]
    RequestFailed(#[from] SendError<Request>),
    #[error("Receiving MTs failed")]
    ReceiveFailed(#[from] RecvError),
    #[error("Remote unexpectedly closed")]
    RemoteClosed,
    #[error("Received illegal message from provided")]
    IllegalMessage,
}
