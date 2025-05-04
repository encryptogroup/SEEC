use serde::{Deserialize, Serialize};
use crate::protocols::{Plain, Ring, Share};
use crate::protocols::mixed_gmw::MixedShareStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Mixed<R> {
    Bool(bool),
    Arith(R),
}


impl<R: Ring> Mixed<R> {
    pub fn into_bool(self) -> Option<bool> {
        match self {
            Mixed::Bool(b) => Some(b),
            Mixed::Arith(_) => None,
        }
    }
    pub fn into_arith(self) -> Option<R> {
        match self {
            Mixed::Bool(_) => None,
            Mixed::Arith(r) => Some(r),
        }
    }

    pub fn unwrap_bool(self) -> bool {
        match self {
            Mixed::Bool(b) => b,
            Mixed::Arith(_) => panic!("called unwrap_bool on Arith"),
        }
    }
    pub fn unwrap_arith(self) -> R {
        match self {
            Mixed::Bool(_) => panic!("called unwrap_arith on Bool"),
            Mixed::Arith(r) => r,
        }
    }

    pub fn convert_into_bool(self) -> Mixed<R> {
        match self {
            Mixed::Arith(a) => Mixed::Bool(a & R::ONE == R::ONE),
            s => s,
        }
    }
}

// TODO Default here is prob wrong
impl<R> Default for Mixed<R> {
    fn default() -> Self {
        Self::Bool(false)
    }
}

impl<R: Ring> Plain for Mixed<R> {}


impl<R: Ring> Share for Mixed<R> {
    type Plain = Mixed<R>;
    type SimdShare = MixedShareStorage<R>;

    fn zero(&self) -> Self {
        match self {
            Mixed::Bool(_) => Mixed::Bool(false),
            Mixed::Arith(_) => Mixed::Arith(R::ZERO),
        }
    }
}
