//! Multiplication triples.
use crate::circuit::ExecutableCircuit;
use crate::common::{BitSlice, BitVec};
use crate::executor::GateOutputs;
use crate::protocols::{FunctionDependentSetup, SetupStorage};
use crate::utils;
use async_trait::async_trait;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::ops::Index;

pub mod insecure_provider;
pub mod ot_ext;
pub mod trusted_provider;
pub mod trusted_seed_provider;

/// Provides a source of multiplication triples.
#[async_trait]
pub trait MTProvider {
    type Output;
    type Error;
    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error>;
}

/// Efficient storage of multiple triples.
///
/// This struct is a container for multiple multiplication triples, where the components
/// are efficiently stored in [`BitVec`]s. For a large amount, a single multiplication triple
/// will only take up 3 bits of storage, compared to the 3 bytes needed for a single
/// [`MulTriple`]. Prefer this type over `Vec<MulTriple>`.
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct MulTriples {
    a: BitVec<usize>,
    b: BitVec<usize>,
    c: BitVec<usize>,
}

#[derive(Debug, Clone)]
pub struct MulTriplesSlice<'a> {
    a: &'a BitSlice<usize>,
    b: &'a BitSlice<usize>,
    c: &'a BitSlice<usize>,
}

impl MulTriples {
    /// Create `size` multiplication triples where a,b,c are set to zero. Intended for testing
    /// purposes.
    pub fn zeros(size: usize) -> Self {
        let zeros = BitVec::repeat(false, size);
        Self {
            a: zeros.clone(),
            b: zeros.clone(),
            c: zeros,
        }
    }

    /// Create a random pair of multiplication triples `[(a1, b1, c1), (a2, b2, c2)]` where
    /// `c1 ^ c2 = (a1 ^ a2) & (b1 ^ b2)`.
    pub fn random_pair<R: CryptoRng + Rng>(size: usize, rng: &mut R) -> [Self; 2] {
        let [a1, a2, b1, b2, c1] = utils::rand_bitvecs(size, rng);
        let mts1 = Self::from_raw(a1, b1, c1);
        let c2 = compute_c(&mts1, &a2, &b2);
        let mts2 = Self::from_raw(a2, b2, c2);
        [mts1, mts2]
    }

    /// Create multiplication triples with random values, without any structure.
    pub fn random<R: Rng + CryptoRng>(size: usize, rng: &mut R) -> Self {
        let [a, b, c] = utils::rand_bitvecs(size, rng);
        Self::from_raw(a, b, c)
    }

    /// Create multiplication triples with the provided `c` values and random `a,b` values.
    pub fn random_with_fixed_c<R: Rng + CryptoRng>(c: BitVec<usize>, rng: &mut R) -> Self {
        let [a, b] = utils::rand_bitvecs(c.len(), rng);
        Self::from_raw(a, b, c)
    }

    /// Construct multiplication triples from their components.
    ///
    /// # Panics
    /// Panics if the provided bitvectors don't have an equal length.
    pub fn from_raw(a: BitVec<usize>, b: BitVec<usize>, c: BitVec<usize>) -> Self {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), c.len());
        Self { a, b, c }
    }

    /// Return the amount of multiplication triples.
    pub fn len(&self) -> usize {
        self.a.len()
    }

    /// Returns true if there are no multiplication triples stored.
    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    /// Provides an iterator over the multiplication triples in the form of [`MulTriple`]s.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = MulTriple> + '_ {
        self.slice(..).iter()
    }

    pub fn pop(&mut self) -> Option<MulTriple> {
        Some(MulTriple {
            a: self.a.pop()?,
            b: self.b.pop()?,
            c: self.c.pop()?,
        })
    }

    pub fn slice<Idx>(&self, range: Idx) -> MulTriplesSlice<'_>
    where
        BitSlice<usize>: Index<Idx, Output = BitSlice<usize>>,
        Idx: Clone,
    {
        MulTriplesSlice {
            a: &self.a[range.clone()],
            b: &self.b[range.clone()],
            c: &self.c[range],
        }
    }

    pub fn a(&self) -> &BitSlice<usize> {
        &self.a
    }

    pub fn b(&self) -> &BitSlice<usize> {
        &self.b
    }

    pub fn c(&self) -> &BitSlice<usize> {
        &self.c
    }
}

impl<'a> MulTriplesSlice<'a> {
    /// Provides an iterator over the multiplication triples in the form of [`MulTriple`]s.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = MulTriple> + 'a {
        self.a
            .iter()
            .by_vals()
            .zip(self.b.iter().by_vals())
            .zip(self.c.iter().by_vals())
            .map(|((a, b), c)| MulTriple { a, b, c })
    }

    pub fn slice<Idx>(&self, range: Idx) -> Self
    where
        BitSlice<usize>: Index<Idx, Output = BitSlice<usize>>,
        Idx: Clone,
    {
        Self {
            a: &self.a[range.clone()],
            b: &self.b[range.clone()],
            c: &self.c[range],
        }
    }

    pub fn a(&self) -> &BitSlice<usize> {
        self.a
    }

    pub fn b(&self) -> &BitSlice<usize> {
        self.b
    }

    pub fn c(&self) -> &BitSlice<usize> {
        self.c
    }
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

fn compute_c(mts: &MulTriples, a: &BitVec<usize>, b: &BitVec<usize>) -> BitVec<usize> {
    (a.clone() ^ &mts.a) & (b.clone() ^ &mts.b) ^ &mts.c
}

fn compute_c_owned(mts: MulTriples, a: BitVec<usize>, b: BitVec<usize>) -> BitVec<usize> {
    (a ^ mts.a) & (b ^ mts.b) ^ mts.c
}

#[derive(Debug, Default)]
/// A single multiplication triple.
pub struct MulTriple {
    a: bool,
    b: bool,
    c: bool,
}

impl MulTriple {
    /// Create a multiplication triple with every component set to 0.
    pub fn zeros() -> Self {
        // Default of bool is false
        Self::default()
    }

    #[inline]
    pub fn a(&self) -> bool {
        self.a
    }

    #[inline]
    pub fn b(&self) -> bool {
        self.b
    }

    #[inline]
    pub fn c(&self) -> bool {
        self.c
    }
}

impl SetupStorage for MulTriples {
    fn len(&self) -> usize {
        self.len()
    }

    /// Split of the last `count` many multiplication triples into a new `MulTriples`.
    fn split_off_last(&mut self, count: usize) -> Self {
        let len = self.len();
        let a = self.a.split_off(len - count);
        let b = self.b.split_off(len - count);
        let c = self.c.split_off(len - count);
        Self { a, b, c }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::mul_triple::MulTriples;

    #[test]
    fn random_triple() {
        let [p1, p2] = MulTriples::random_pair(512, &mut thread_rng());
        let left = p1.c ^ p2.c;
        let right = (p1.a ^ p2.a) & (p1.b ^ p2.b);
        assert_eq!(left, right);
    }
}
