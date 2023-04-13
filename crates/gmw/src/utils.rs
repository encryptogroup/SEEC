use crate::common::BitVec;
use bitvec::prelude::BitStore;
use num_integer::div_ceil;
use rand::{CryptoRng, Fill, Rng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::RangeInclusive;
use std::{array, mem};

pub(crate) struct ByAddress<'a, T: ?Sized>(pub(crate) &'a T);

impl<'a, T: ?Sized> Hash for ByAddress<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.0, state)
    }
}

impl<'a, T: ?Sized> PartialEq for ByAddress<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl<'a, T: ?Sized> Eq for ByAddress<'a, T> {}

//
// RangeInclusive start wrapper
//

#[derive(Eq, Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RangeInclusiveStartWrapper<T> {
    pub(crate) range: RangeInclusive<T>,
}

impl<T> RangeInclusiveStartWrapper<T> {
    pub(crate) fn new(range: RangeInclusive<T>) -> RangeInclusiveStartWrapper<T> {
        RangeInclusiveStartWrapper { range }
    }
}

impl<T> PartialEq for RangeInclusiveStartWrapper<T>
where
    T: Eq,
{
    #[inline]
    fn eq(&self, other: &RangeInclusiveStartWrapper<T>) -> bool {
        self.range.start() == other.range.start() && self.range.end() == other.range.end()
    }
}

impl<T> Ord for RangeInclusiveStartWrapper<T>
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &RangeInclusiveStartWrapper<T>) -> Ordering {
        match self.range.start().cmp(other.range.start()) {
            Ordering::Equal => self.range.end().cmp(other.range.end()),
            not_eq => not_eq,
        }
    }
}

impl<T> PartialOrd for RangeInclusiveStartWrapper<T>
where
    T: Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &RangeInclusiveStartWrapper<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Helper method to quickly create an array of random BitVecs.
pub(crate) fn rand_bitvecs<R: CryptoRng + Rng, const N: usize>(
    size: usize,
    rng: &mut R,
) -> [BitVec<usize>; N] {
    array::from_fn(|_| rand_bitvec(size, rng))
}

pub(crate) fn rand_bitvec<T, R>(size: usize, rng: &mut R) -> BitVec<T>
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
