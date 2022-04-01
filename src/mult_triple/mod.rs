use crate::common::BitVec;
use async_trait::async_trait;
use num_integer::div_ceil;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};

pub mod insecure_provider;
pub mod trusted_provider;

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

    pub fn random<RNG: CryptoRng + Rng>(size: usize, rng: &mut RNG) -> [Self; 2] {
        let bytes = div_ceil(size, 8);
        let mut bufs = [(); 5].map(|_| vec![0_u8; bytes]);
        for buf in &mut bufs {
            rng.fill_bytes(buf);
        }
        let bit_bufs = bufs.map(BitVec::from_vec);
        let [a1, a2, b1, b2, c1] = bit_bufs;
        let mut c2 = c1.clone();
        c2 ^= (a1.clone() ^ &a2) & (b1.clone() ^ &b2);
        [Self::from_raw(a1, b1, c1), Self::from_raw(a2, b2, c2)]
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
        let [p1, p2] = MultTriples::random(512, &mut thread_rng());
        let left = p1.c ^ p2.c;
        let right = (p1.a ^ p2.a) & (p1.b ^ p2.b);
        assert_eq!(left, right);
    }
}
