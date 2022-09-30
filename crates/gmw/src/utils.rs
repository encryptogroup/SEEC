use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::RangeInclusive;

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

#[derive(Eq, Debug, Clone)]
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
