//! Multiplication triples.

use crate::circuit::ExecutableCircuit;
use crate::executor::GateOutputs;
use crate::protocols::FunctionDependentSetup;
use async_trait::async_trait;

pub mod arithmetic;
pub mod boolean;

/// Provides a source of multiplication triples.
#[async_trait]
pub trait MTProvider {
    type Output;
    type Error;
    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error>;
}

#[async_trait]
impl<Mtp: MTProvider + Send> MTProvider for &mut Mtp {
    type Output = Mtp::Output;
    type Error = Mtp::Error;

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        (*self).request_mts(amount).await
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
