//! Insecure MTProvider, intended for testing.
use crate::mul_triple::boolean::MulTriples;
use crate::mul_triple::MTProvider;
use async_trait::async_trait;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::convert::Infallible;

/// An insecure [`MTProvider`] which simply returns [`MulTriples::zeros`]. **Do not use in
/// production!**.
#[derive(Clone, Default)]
pub struct InsecureMTProvider;

#[async_trait]
impl MTProvider for InsecureMTProvider {
    type Output = MulTriples;
    type Error = Infallible;

    async fn precompute_mts(&mut self, _amount: usize) -> Result<(), Infallible> {
        // Nothing to do
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        Ok(MulTriples::random(
            amount,
            &mut ChaCha8Rng::seed_from_u64(42),
        ))
    }
}
