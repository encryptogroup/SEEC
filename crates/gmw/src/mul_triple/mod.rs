//! Multiplication triples.

use crate::circuit::ExecutableCircuit;
use crate::executor::GateOutputs;
use crate::protocols::FunctionDependentSetup;
use async_trait::async_trait;
use std::error::Error;

pub mod arithmetic;
pub mod boolean;

/// Provides a source of multiplication triples.
#[async_trait]
pub trait MTProvider {
    type Output;
    type Error;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error>;

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error>;
}

pub struct ErasedError<MTP>(pub MTP);

#[async_trait]
impl<Mtp: MTProvider + Send> MTProvider for &mut Mtp {
    type Output = Mtp::Output;
    type Error = Mtp::Error;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        (**self).precompute_mts(amount).await
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        (**self).request_mts(amount).await
    }
}

#[async_trait]
impl<Out, Err> MTProvider for Box<dyn MTProvider<Output = Out, Error = Err> + Send> {
    type Output = Out;
    type Error = Err;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        (**self).precompute_mts(amount).await
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        (**self).request_mts(amount).await
    }
}

// TODO I think this impl would disallow downstream crates to impl FunctionDependentSetup
#[async_trait]
impl<ShareStorage: Sync, G: Send + Sync, Idx: Send + Sync, Mtp: MTProvider + Send>
    FunctionDependentSetup<ShareStorage, G, Idx> for Mtp
{
    type Output = Mtp::Output;
    type Error = Mtp::Error;

    async fn setup(
        &mut self,
        _shares: &GateOutputs<ShareStorage>,
        circuit: &ExecutableCircuit<G, Idx>,
    ) -> Result<Self::Output, Self::Error> {
        self.request_mts(circuit.interactive_count_times_simd())
            .await
    }
}

#[async_trait]
impl<MTP> MTProvider for ErasedError<MTP>
where
    MTP: MTProvider + Send,
    <MTP as MTProvider>::Error: Error + Send + Sync + 'static,
{
    type Output = MTP::Output;
    type Error = Box<dyn Error + Send + Sync>;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        self.0
            .precompute_mts(amount)
            .await
            .map_err(|err| Box::new(err) as Box<dyn Error + Send + Sync>)
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        self.0
            .request_mts(amount)
            .await
            .map_err(|err| Box::new(err) as Box<dyn Error + Send + Sync>)
    }
}
