use rand::distributions::{Distribution, Standard};
use std::iter;
use crate::protocols::mixed_aby2::{ArithmeticShareStorage, BooleanShareStorage};
use crate::protocols::mixed_aby2::share::MixedShare;
use crate::protocols::{Ring, ShareStorage};


#[derive(Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    Bool(BooleanShareStorage),
    Arith(ArithmeticShareStorage<R>),
    Mixed(Vec<MixedShare<R>>),
}


impl<R> Extend<MixedShare<R>> for MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    fn extend<T: IntoIterator<Item = MixedShare<R>>>(&mut self, iter: T) {
        match self {
            MixedShareStorage::Bool(b) => {
                let boolean_shares = iter.into_iter().map(|el| match el {
                    MixedShare::Bool(b) => b,
                    MixedShare::Arith(_) => panic!("Cannot extend with arithmetic share"),
                });
                b.extend(boolean_shares);
            }
            MixedShareStorage::Arith(a) => {
                let arith_shares = iter.into_iter().map(|el| match el {
                    MixedShare::Bool(_) => panic!("Cannot extend with boolean share"),
                    MixedShare::Arith(a) => a,
                });
                a.extend(arith_shares);
            }
            MixedShareStorage::Mixed(m) => m.extend(iter),
        }
    }
}

impl<R> IntoIterator for MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    type Item = MixedShare<R>;
    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            MixedShareStorage::Bool(s) => Box::new(s.into_iter().map(MixedShare::Bool)),
            MixedShareStorage::Arith(s) => Box::new(s.into_iter().map(MixedShare::Arith)),
            MixedShareStorage::Mixed(s) => Box::new(s.into_iter()),
        }
    }
}

impl<R> MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    pub fn try_push(&mut self, s: MixedShare<R>) {
        match (self, s) {
            (Self::Bool(v), MixedShare::Bool(b)) => {
                v.public.push(b.public);
                v.private.push(b.private);
            }
            (Self::Arith(v), MixedShare::Arith(a)) => {
                v.public.push(a.public);
                v.private.push(a.private);
            }
            (Self::Mixed(v), s) => v.push(s),
            (Self::Bool(_), MixedShare::Arith(_)) | (Self::Arith(_), MixedShare::Bool(_)) => {
                panic!("Mismatch between MixedShareStorageType and MixedShare!")
            }
        }
    }
}

impl<R> FromIterator<MixedShare<R>> for MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    fn from_iter<T: IntoIterator<Item = MixedShare<R>>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let mut acc = match iter.next() {
            None => return MixedShareStorage::default(),
            Some(MixedShare::Bool(b)) => MixedShareStorage::Bool(BooleanShareStorage::repeat(b, 1)),
            Some(MixedShare::Arith(a)) => {
                MixedShareStorage::Arith(ArithmeticShareStorage::repeat(a, 1))
            }
        };

        iter.for_each(|el| acc.try_push(el));
        acc
    }
}

impl<R> Default for MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    fn default() -> Self {
        Self::Mixed(Default::default())
    }
}

impl<R> ShareStorage<MixedShare<R>> for MixedShareStorage<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    fn len(&self) -> usize {
        match self {
            MixedShareStorage::Bool(s) => s.len(),
            MixedShareStorage::Arith(s) => s.len(),
            MixedShareStorage::Mixed(s) => s.len(),
        }
    }

    fn repeat(val: MixedShare<R>, len: usize) -> Self {
        Self::Mixed(iter::repeat(val).take(len).collect())
    }

    fn set(&mut self, idx: usize, val: MixedShare<R>) {
        match (self, val) {
            (Self::Bool(bv), MixedShare::Bool(b)) => bv.set(idx, b),
            (MixedShareStorage::Arith(av), MixedShare::Arith(a)) => av.set(idx, a),
            (MixedShareStorage::Mixed(v), s) => v[idx] = s,
            (Self::Bool(_), MixedShare::Arith(_)) | (Self::Arith(_), MixedShare::Bool(_)) => {
                panic!("Mismatch between MixedShareStorage type and value type");
            }
        }
    }

    fn get(&self, idx: usize) -> MixedShare<R> {
        match self {
            MixedShareStorage::Bool(v) => MixedShare::Bool(v.get(idx)),
            MixedShareStorage::Arith(v) => MixedShare::Arith(v.get(idx)),
            MixedShareStorage::Mixed(v) => v[idx].clone(),
        }
    }
}