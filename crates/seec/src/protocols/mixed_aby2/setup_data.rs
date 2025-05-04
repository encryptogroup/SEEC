use rand::distributions::{Distribution, Standard};
use crate::protocols::{Protocol, Ring, SetupStorage};
use crate::protocols::aby2::BooleanAby2;
use crate::protocols::arithmetic_aby2::{ArithmeticAby2, EvalShares};
use crate::protocols::mixed::Mixed as MixedPlain;

#[derive(Clone, Default)]
pub struct MixedSetupData<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    pub(super) bool: <BooleanAby2 as Protocol>::SetupStorage,
    pub(super) arith: <ArithmeticAby2<R> as Protocol>::SetupStorage,
    pub(super) conv: Vec<EvalShares<MixedPlain<R>>>,
}

// This trait is required by bound but is meaningless in this case
impl<R> SetupStorage for MixedSetupData<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    fn len(&self) -> usize {
        panic!("not implemented")
    }

    fn split_off_last(&mut self, _count: usize) -> Self {
        panic!("not implemented")
    }

    fn reserve(&mut self, _additional: usize) {
        panic!("not implemented")
    }

    fn append(&mut self, _other: Self) {
        panic!("not implemented")
    }
}