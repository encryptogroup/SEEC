//! Multiplication triples and providers.

use crate::circuit::ExecutableCircuit;
use crate::executor::GateOutputs;
use crate::protocols::FunctionDependentSetup;
use crate::utils::{BoxError, ErasedError};
use async_trait::async_trait;
use std::error::Error;

pub mod arithmetic;
pub mod boolean;
pub mod storage;

/// Provides a source of multiplication triples.
#[async_trait]
pub trait MTProvider {
    type Output;
    type Error;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error>;

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error>;

    fn into_dyn(
        self,
    ) -> Box<dyn MTProvider<Output = Self::Output, Error = BoxError> + Send + 'static>
    where
        Self: Sized + Send + 'static,
        Self::Error: Error + Send + Sync + 'static,
    {
        Box::new(ErasedError(self))
    }
}

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
//  on second thought, this might not be the case
#[async_trait]
impl<ShareStorage: Sync, G: Send + Sync, Idx: Send + Sync, Mtp: MTProvider + Send>
    FunctionDependentSetup<ShareStorage, G, Idx> for Mtp
{
    type Output = Mtp::Output;
    type Error = Mtp::Error;

    async fn setup(
        &mut self,
        _shares: &GateOutputs<ShareStorage>,
        _circuit: &ExecutableCircuit<G, Idx>,
    ) -> Result<(), Self::Error> {
        // TODO, should this call precompute_mts ? Mhh, currently I call it explicitly
        //  when I want to perform FIP
        Ok(())
    }

    async fn request_setup_output(&mut self, count: usize) -> Result<Self::Output, Self::Error> {
        self.request_mts(count).await
    }
}

#[async_trait]
impl<MTP> MTProvider for ErasedError<MTP>
where
    MTP: MTProvider + Send,
    <MTP as MTProvider>::Error: Error + Send + Sync + 'static,
{
    type Output = MTP::Output;
    type Error = BoxError;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        self.0
            .precompute_mts(amount)
            .await
            .map_err(BoxError::from_err)
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        self.0.request_mts(amount).await.map_err(BoxError::from_err)
    }
}
