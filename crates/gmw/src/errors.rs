use remoc::rch::base;
use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("Received out of order message during execution")]
    OutOfOrderMessage,
    #[error("Unable to perform function dependent setup")]
    Setup,
}

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
    RequestFailed(#[from] base::SendError<Request>),
    #[error("Receiving MTs failed")]
    ReceiveFailed(#[from] base::RecvError),
    #[error("Remote unexpectedly closed")]
    RemoteClosed,
    #[error("Received illegal message from provided")]
    IllegalMessage,
}
