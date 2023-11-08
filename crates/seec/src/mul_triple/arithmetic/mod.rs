use crate::protocols::{Ring, SetupStorage};
use serde::{Deserialize, Serialize};

pub mod insecure_provider;
pub mod ot_ext;

pub use insecure_provider::InsecureMTProvider;
pub use ot_ext::OtMTProvider;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct MulTriples<R> {
    a: Vec<R>,
    b: Vec<R>,
    c: Vec<R>,
}

pub struct MulTriple<R> {
    a: R,
    b: R,
    c: R,
}

impl<R> MulTriples<R> {
    pub fn len(&self) -> usize {
        debug_assert!(self.a.len() == self.b.len() && self.a.len() == self.c.len());
        self.a.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<R: Ring> MulTriples<R> {
    pub fn zeros(size: usize) -> Self {
        Self {
            a: vec![R::default(); size],
            b: vec![R::default(); size],
            c: vec![R::default(); size],
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = MulTriple<R>> + '_ {
        self.a
            .iter()
            .zip(&self.b)
            .zip(&self.c)
            .map(|((a, b), c)| MulTriple {
                a: a.clone(),
                b: b.clone(),
                c: c.clone(),
            })
    }

    pub fn extend_from_mts(&mut self, other: &Self) {
        self.a.extend_from_slice(&other.a);
        self.b.extend_from_slice(&other.b);
        self.c.extend_from_slice(&other.c);
    }
}

impl<R> MulTriple<R> {
    pub fn a(&self) -> &R {
        &self.a
    }

    pub fn b(&self) -> &R {
        &self.b
    }

    pub fn c(&self) -> &R {
        &self.c
    }
}

impl<R: Default + Send + Sync> SetupStorage for MulTriples<R> {
    fn len(&self) -> usize {
        self.a.len()
    }

    fn split_off_last(&mut self, count: usize) -> Self {
        let split_at = self.len() - count;
        let a = self.a.split_off(split_at);
        let b = self.b.split_off(split_at);
        let c = self.c.split_off(split_at);
        Self { a, b, c }
    }

    fn append(&mut self, mut other: Self) {
        self.a.append(&mut other.a);
        self.b.append(&mut other.b);
        self.c.append(&mut other.c);
    }
}
