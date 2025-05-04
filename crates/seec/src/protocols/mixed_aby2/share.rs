use rand::distributions::{Distribution, Standard};
use crate::protocols;
use crate::protocols::mixed::Mixed as MixedPlain;
use crate::protocols::mixed_aby2::{ArithmeticShare, BooleanShare};
use crate::protocols::mixed_aby2::share_storage::MixedShareStorage;
use crate::protocols::Ring;

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum MixedShare<R> {
    Bool(BooleanShare),
    Arith(ArithmeticShare<R>),
}

impl<R> Default for MixedShare<R> {
    fn default() -> Self {
        Self::Bool(BooleanShare::default())
    }
}

impl<R> protocols::Share for MixedShare<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    type Plain = MixedPlain<R>;
    type SimdShare = MixedShareStorage<R>;
}

impl<R: Ring> MixedShare<R> {
    pub fn new_bool(private: bool, public: bool) -> Self {
        Self::Bool(BooleanShare::new(private, public))
    }

    pub fn new_arith(private: R, public: R) -> Self {
        Self::Arith(ArithmeticShare { private, public })
    }

    pub fn into_bool(self) -> Option<BooleanShare> {
        match self {
            MixedShare::Bool(b) => Some(b),
            MixedShare::Arith(_) => None,
        }
    }

    pub fn into_arith(self) -> Option<ArithmeticShare<R>> {
        match self {
            MixedShare::Bool(_) => None,
            MixedShare::Arith(a) => Some(a),
        }
    }

    pub fn unwrap_bool(self) -> BooleanShare {
        self.into_bool().expect("Expected boolean share but got arithmetic share.")
    }

    pub fn unwrap_arith(self) -> ArithmeticShare<R> {
        self.into_arith().expect("Expected arithmetic share but got boolean share.")
    }

    pub fn get_public(self) -> MixedPlain<R> {
        match self {
            MixedShare::Bool(b) => MixedPlain::Bool(b.public),
            MixedShare::Arith(a) => MixedPlain::Arith(a.public),
        }
    }

    pub fn get_private(self) -> MixedPlain<R> {
        match self {
            MixedShare::Bool(b) => MixedPlain::Bool(b.private),
            MixedShare::Arith(a) => MixedPlain::Arith(a.private),
        }
    }
}