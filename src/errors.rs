use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {}

#[derive(Error, Debug)]
pub enum CircuitError {
    #[error("Unable to safe circuit as dot file")]
    SaveAsDot(#[from] io::Error),
}
