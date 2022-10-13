//! Multiplication triples.
use async_trait::async_trait;
use bitvec::store::BitStore;
use num_integer::div_ceil;
use rand::{CryptoRng, Fill, Rng};
use serde::{Deserialize, Serialize};
use std::mem;

use crate::common::BitVec;

pub mod insecure_provider;
pub mod ot_ext;
pub mod trusted_provider;
pub mod trusted_seed_provider;

/// Provides a source of multiplication triples.
#[async_trait]
pub trait MTProvider {
    type Error;
    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error>;
}

/// Efficient storage of multiple triples.
///
/// This struct is a container for multiple multiplication triples, where the components
/// are efficiently stored in [`BitVec`]s. For a large amount, a single multiplication triple
/// will only take up 3 bits of storage, compared to the 3 bytes needed for a single
/// [`MulTriple`]. Prefer this type over `Vec<MulTriple>`.
#[derive(Serialize, Deserialize, Debug)]
pub struct MulTriples {
    a: BitVec<usize>,
    b: BitVec<usize>,
    c: BitVec<usize>,
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
        let [a1, a2, b1, b2, c1] = rand_bitvecs(size, rng);
        let mts1 = Self::from_raw(a1, b1, c1);
        let c2 = compute_c(&mts1, &a2, &b2);
        let mts2 = Self::from_raw(a2, b2, c2);
        [mts1, mts2]
    }

    /// Create multiplication triples with random values, without any structure.
    pub fn random<R: Rng + CryptoRng>(size: usize, rng: &mut R) -> Self {
        let [a, b, c] = rand_bitvecs(size, rng);
        Self::from_raw(a, b, c)
    }

    /// Create multiplication triples with the provided `c` values and random `a,b` values.
    pub fn random_with_fixed_c<R: Rng + CryptoRng>(c: BitVec<usize>, rng: &mut R) -> Self {
        let [a, b] = rand_bitvecs(c.len(), rng);
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

    /// Split of the last `count` many multiplication triples into a new `MulTriples`.
    pub fn split_off_last(&mut self, count: usize) -> Self {
        let len = self.len();
        let a = self.a.split_off(len - count);
        let b = self.b.split_off(len - count);
        let c = self.c.split_off(len - count);
        Self { a, b, c }
    }

    /// Provides an iterator over the multiplication triples in the form of [`MulTriple`]s.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = MulTriple> + ExactSizeIterator<Item = MulTriple> + '_ {
        self.a
            .iter()
            .by_vals()
            .zip(self.b.iter().by_vals())
            .zip(self.c.iter().by_vals())
            .map(|((a, b), c)| MulTriple { a, b, c })
    }

    // pub fn drain_mts(
    //     &mut self,
    // ) -> impl Iterator<Item = MulTriple> + ExactSizeIterator<Item = MulTriple> + '_ {
    //     self.a
    //         .drain(..)
    //         .rev()
    //         .zip(self.b.drain(..).rev().self.c.drain(..).rev())
    //         .map(|((a, b), c)| MulTriple { a, b, c })
    // }
}

fn compute_c(mts: &MulTriples, a: &BitVec<usize>, b: &BitVec<usize>) -> BitVec<usize> {
    (a.clone() ^ &mts.a) & (b.clone() ^ &mts.b) ^ &mts.c
}

fn compute_c_owned(mts: MulTriples, a: BitVec<usize>, b: BitVec<usize>) -> BitVec<usize> {
    (a ^ mts.a) & (b ^ mts.b) ^ mts.c
}

/// Helper method to quickly create an array of random BitVecs.
fn rand_bitvecs<R: CryptoRng + Rng, const N: usize>(
    size: usize,
    rng: &mut R,
) -> [BitVec<usize>; N] {
    // Workaround until array::from_fn https://github.com/rust-lang/rust/issues/89379 is stabilized
    [(); N].map(|_| rand_bitvec(size, rng))
}

fn rand_bitvec<T, R>(size: usize, rng: &mut R) -> BitVec<T>
where
    T: BitStore + Copy,
    [T]: Fill,
    R: CryptoRng + Rng,
{
    let bitstore_items = div_ceil(size, mem::size_of::<T>());
    let mut buf = vec![T::ZERO; bitstore_items];
    rng.fill(&mut buf[..]);
    let mut bv = BitVec::from_vec(buf);
    bv.truncate(size);
    bv
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

    pub fn a(&self) -> bool {
        self.a
    }

    pub fn b(&self) -> bool {
        self.b
    }

    pub fn c(&self) -> bool {
        self.c
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
