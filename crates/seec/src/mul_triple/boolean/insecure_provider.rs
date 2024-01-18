//! Insecure MTProvider, intended for testing.
use crate::mul_triple::boolean::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::SetupStorage;
use async_trait::async_trait;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::convert::Infallible;

/// An insecure [`MTProvider`] which simply returns [`MulTriples::zeros`]. **Do not use in
/// production!**.
#[derive(Clone, Default)]
pub struct InsecureMTProvider {
    mts: Option<MulTriples>,
}

#[async_trait]
impl MTProvider for InsecureMTProvider {
    type Output = MulTriples;
    type Error = Infallible;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Infallible> {
        self.mts = Some(gen_mts(amount));
        // Nothing to do
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        let mts = match &mut self.mts {
            None => gen_mts(amount),
            Some(mts) if mts.len() >= amount => mts.split_off_last(amount),
            _ => gen_mts(amount),
        };
        Ok(mts)
    }
}

fn gen_mts(amount: usize) -> MulTriples {
    MulTriples::random(amount, &mut ChaCha8Rng::seed_from_u64(42))
}
