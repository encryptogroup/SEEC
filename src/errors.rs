use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {}

#[derive(Error, Debug)]
pub enum CircuitError {
    #[error("Unable to save circuit as dot file")]
    SaveAsDot(#[source] io::Error),
    #[error("Unable to load bristol file")]
    LoadBristol(#[from] io::Error),
    #[error("Unable to load parse file")]
    ParseBristol(#[from] nom::Err<nom::error::Error<String>>),
    #[error("Unable to convert bristol circuit")]
    ConversionError,
}
