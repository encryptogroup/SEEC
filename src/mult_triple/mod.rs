use crate::common::BitVec;
use async_trait::async_trait;
use num_integer::div_ceil;
use rand::{CryptoRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

pub mod insecure_provider;
pub mod trusted_provider;
pub mod trusted_seed_provider;

#[async_trait]
pub trait MTProvider {
    type Error;
    async fn request_mts(&mut self, amount: usize) -> Result<MultTriples, Self::Error>;
}

#[derive(Serialize, Deserialize)]
pub struct MultTriples {
    a: BitVec,
    b: BitVec,
    c: BitVec,
}

impl MultTriples {
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

    pub fn random_pair<R: CryptoRng + Rng>(size: usize, rng: &mut R) -> [Self; 2] {
        let [a1, a2, b1, b2, c1] = rand_mt_bufs(size, rng);
        let mts1 = Self::from_raw(a1, b1, c1);
        let c2 = compute_c(&mts1, &a2, &b2);
        let mts2 = Self::from_raw(a2, b2, c2);
        [mts1, mts2]
    }

    pub fn random<R: Rng + CryptoRng + SeedableRng>(size: usize, rng: &mut R) -> Self {
        let [a, b, c] = rand_mt_bufs(size, rng);
        Self::from_raw(a, b, c)
    }

    pub fn random_with_fixed_c<R: Rng + CryptoRng + SeedableRng>(c: BitVec, rng: &mut R) -> Self {
        let [a, b] = rand_mt_bufs(c.len(), rng);
        Self::from_raw(a, b, c)
    }

    fn from_raw(a: BitVec, b: BitVec, c: BitVec) -> Self {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), c.len());
        Self { a, b, c }
    }

    /// Return the amount of multiplication triples.
    pub fn len(&self) -> usize {
        self.a.len()
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    /// Split of the last `count` many multiplication triples into a new `MultTriples`.
    pub fn split_off_last(&mut self, count: usize) -> Self {
        let len = self.len();
        let a = self.a.split_off(len - count);
        let b = self.b.split_off(len - count);
        let c = self.c.split_off(len - count);
        Self { a, b, c }
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = MultTriple> + ExactSizeIterator<Item = MultTriple> + '_ {
        self.a
            .iter()
            .by_vals()
            .zip(self.b.iter().by_vals())
            .zip(self.c.iter().by_vals())
            .map(|((a, b), c)| MultTriple { a, b, c })
    }
}

fn compute_c(mts: &MultTriples, a: &BitVec, b: &BitVec) -> BitVec {
    (a.clone() ^ &mts.a) & (b.clone() ^ &mts.b) ^ &mts.c
}

fn compute_c_owned(mts: MultTriples, a: BitVec, b: BitVec) -> BitVec {
    (a ^ mts.a) & (b ^ mts.b) ^ mts.c
}

fn rand_mt_bufs<R: CryptoRng + Rng, const N: usize>(size: usize, rng: &mut R) -> [BitVec; N] {
    let bytes = div_ceil(size, 8);
    let mut bufs = [(); N].map(|_| vec![0_u8; bytes]);
    for buf in &mut bufs {
        rng.fill_bytes(buf);
    }
    bufs.map(BitVec::from_vec)
}

#[derive(Debug, Default)]
pub struct MultTriple {
    a: bool,
    b: bool,
    c: bool,
}

impl MultTriple {
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
    use crate::mult_triple::MultTriples;
    use rand::thread_rng;

    #[test]
    fn random_triple() {
        let [p1, p2] = MultTriples::random_pair(512, &mut thread_rng());
        let left = p1.c ^ p2.c;
        let right = (p1.a ^ p2.a) & (p1.b ^ p2.b);
        assert_eq!(left, right);
    }
}
