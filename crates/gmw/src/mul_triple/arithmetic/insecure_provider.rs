//! Insecure MTProvider, intended for testing.
use crate::mul_triple::arithmetic::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::Ring;
use async_trait::async_trait;
use std::convert::Infallible;
use std::marker::PhantomData;

/// An insecure [`MTProvider`] which simply returns [`MulTriples::zeros`]. **Do not use in
/// production!**.
#[derive(Clone, Default)]
pub struct InsecureMTProvider<R>(PhantomData<R>);

#[async_trait]
impl<R: Ring> MTProvider for InsecureMTProvider<R> {
    type Output = MulTriples<R>;
    type Error = Infallible;

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        Ok(MulTriples::zeros(amount))
    }
}
