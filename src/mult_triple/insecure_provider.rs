//! Insecure MTProvider, intended for testing
use crate::mult_triple::{MTProvider, MultTriples};
use async_trait::async_trait;
use std::convert::Infallible;

/// An insecure [`MTProvider`] which simply returns [`MultTriples::zeros`]. **Do not use in
/// production!**.
#[derive(Clone, Default)]
pub struct InsecureMTProvider;

#[async_trait]
impl MTProvider for InsecureMTProvider {
    type Error = Infallible;

    async fn request_mts(&mut self, amount: usize) -> Result<MultTriples, Self::Error> {
        Ok(MultTriples::zeros(amount))
    }
}
