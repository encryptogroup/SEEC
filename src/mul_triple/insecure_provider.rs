//! Insecure MTProvider, intended for testing.
use std::convert::Infallible;

use async_trait::async_trait;

use crate::mul_triple::{MTProvider, MulTriples};

/// An insecure [`MTProvider`] which simply returns [`MulTriples::zeros`]. **Do not use in
/// production!**.
#[derive(Clone, Default)]
pub struct InsecureMTProvider;

#[async_trait]
impl MTProvider for InsecureMTProvider {
    type Error = Infallible;

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        Ok(MulTriples::zeros(amount))
    }
}
