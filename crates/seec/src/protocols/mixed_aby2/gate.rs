use rand::distributions::{Distribution, Standard};
use bitvec::view::BitViewSized;
use rand::Rng;
use ahash::AHashMap;
use itertools::Itertools;
use crate::circuit::GateIdx;
use crate::gate::base::BaseGate;
use crate::protocols::aby2::BooleanGate;
use crate::protocols::arithmetic_aby2::ArithmeticGate;
use crate::protocols::mixed_aby2::{ArithmeticShare, BooleanShare};
use crate::protocols::mixed_aby2::share::MixedShare;
use crate::protocols::mixed_gmw::MixedGmw;
use crate::protocols::{aby2, arithmetic_aby2, mixed_gmw, Gate, Ring, ScalarDim};
use crate::protocols::mixed::Mixed as MixedPlain;
use crate::secret::Secret;

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum ConvGate {
    Bit2A,
    // A2BInitSend,
    // A2BInitRecv,
    // A2BSelectBit(usize) ,
}


#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub enum MixedGate<R> {
    Base(BaseGate<MixedPlain<R>, ScalarDim>),
    Bool(BooleanGate),
    Arith(ArithmeticGate<R>),
    Conv(ConvGate),
}

impl<R> MixedGate<R>
where
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized
{
    pub(crate) fn setup_output_share(
        &self,
        mut input: impl Iterator<Item = MixedShare<R>>,
        mut rng: impl Rng,
    ) -> MixedShare<R> {
        match self {
            MixedGate::Base(base_gate) => {
                match base_gate {
                    BaseGate::Input(_) =>
                        // TODO how do we get the input type here?
                        //  this is really bad for aby2.0 using the gmw-based
                        //  setup. The workaround atm is that the mixed gmw
                        //  uses convert_into_bool which just takes the lsb.
                        MixedShare::Arith(ArithmeticShare {
                            public: Default::default(),
                            private: rng.gen(),
                        }),
                    BaseGate::Output(_)
                    | BaseGate::SubCircuitInput(_)
                    | BaseGate::SubCircuitOutput(_)
                    | BaseGate::Identity
                    | BaseGate::ConnectToMain(_)
                    | BaseGate::Debug => input.next().expect("Empty input"),
                    BaseGate::ConnectToMainFromSimd(_) =>
                        unimplemented!("SIMD currently not supported for ABY2"),
                    BaseGate::Constant(_) => MixedShare::default(),
                }

            }
            MixedGate::Bool(bg) => MixedShare::Bool(bg.setup_output_share(
                input.map(MixedShare::unwrap_bool), rng
            )),
            MixedGate::Arith(ag) => MixedShare::Arith(ag.setup_output_share(
                input.map(MixedShare::unwrap_arith), rng
            )),
            MixedGate::Conv(ConvGate::Bit2A) => {
                MixedShare::Arith(ArithmeticShare {
                    public: Default::default(),
                    private: rng.gen(),
                })
            }

            /* MixedGate::Conv(ConvGate::A2BInit) => {
                MixedShare::Arith(ArithmeticShare {
                    public: Default::default(),
                    private: rng.gen(),
                })
            },
            MixedGate::Conv(ConvGate::A2BSelectBit(_)) => {
                MixedShare::Bool(BooleanShare {
                    public: Default::default(),
                    private: rng.gen::<bool>(),
                })
            }*/
        }
    }

    pub(crate) fn setup_data_circ<'a, Idx: GateIdx>(
        &self,
        mut input_shares: impl Iterator<Item = &'a Secret<MixedGmw<R>, Idx>>,
        setup_sub_circ_cache: &mut AHashMap<Vec<Secret<MixedGmw<R>, Idx>>, Secret<MixedGmw<R>, Idx>>,
    ) -> Vec<Secret<MixedGmw<R>, Idx>> {
        let (inputs, is_bool) = match self {
            MixedGate::Bool(BooleanGate::And { n }) => (*n as usize, true),
            MixedGate::Arith(ArithmeticGate::Mul) => (2, false),
            MixedGate::Conv(ConvGate::Bit2A) => {
                let i: &Secret<MixedGmw<R>, Idx> = input_shares.next().expect("insufficient input");
                return vec![i.clone().bit_to_a()]
            },
            _ => {
                assert!(self.is_non_interactive(), "Unhandled interactive gate");
                panic!("Called setup_data_circ on non_interactive gate")
            }
        };

        let inputs_pset = input_shares
            .take(inputs)
            .cloned()
            .powerset()
            .skip(inputs + 1);

        let mul = |a: &Secret<MixedGmw<R>, Idx>, b: &Secret<MixedGmw<R>, Idx>| if is_bool {
            a.clone() & b
        } else {
            a.clone() * b
        };

        inputs_pset
            .map(|set| match setup_sub_circ_cache.get(&set) {
                None => match &set[..] {
                    [] => unreachable!("Empty set is filtered"),
                    [a, b] => {
                        mul(a, b)
                    }
                    [processed_subset @ .., last] => {
                        assert!(processed_subset.len() >= 2, "Smaller sets are filtered");
                        let subset_out = setup_sub_circ_cache
                            .get(processed_subset)
                            .expect("Subset not present in cache");
                        let sh = mul(last, subset_out);
                        setup_sub_circ_cache.insert(set, sh.clone());
                        sh
                    }
                }
                Some(processed_set) => processed_set.clone(),
            })
            .collect()
    }

    pub fn from_gmw_gate(g: mixed_gmw::MixedGate<R>) -> Self {
        match g {
            mixed_gmw::MixedGate::Base(base_gate) => Self::Base(base_gate),
            mixed_gmw::MixedGate::Bool(boolean_gate) => Self::Bool(BooleanGate::from_gmw_gate(boolean_gate)),
            mixed_gmw::MixedGate::Arith(arith_gate) => Self::Arith(ArithmeticGate::from_gmw_gate(arith_gate)),
            mixed_gmw::MixedGate::Conv(conv_gate) => Self::Conv(ConvGate::from_gmw_gate(conv_gate)),
        }
    }
}



impl ConvGate {
    pub fn from_gmw_gate(g: mixed_gmw::ConvGate) -> Self {
        let _ = g;
        loop { println!("todo") }
    }
}


impl<R> Gate<MixedPlain<R>> for MixedGate<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    type DimTy = ScalarDim;

    fn is_interactive(&self) -> bool {
        match self {
            MixedGate::Base(_) => false,
            MixedGate::Bool(g) => g.is_interactive(),
            MixedGate::Arith(g) => g.is_interactive(),
            MixedGate::Conv(c) => match c {
                ConvGate::Bit2A => true,
                // ConvGate::A2BInit => true,
                // ConvGate::A2BSelectBit(_) => false,
            },
        }
    }

    fn input_size(&self) -> usize {
        match self {
            MixedGate::Base(g) => g.input_size(),
            MixedGate::Bool(g) => g.input_size(),
            MixedGate::Arith(g) => g.input_size(),
            MixedGate::Conv(c) => match c {
                ConvGate::Bit2A => 1,
                // ConvGate::A2BInit => 1,
                // ConvGate::A2BSelectBit(_) => 1,
            },
        }
    }

    fn as_base_gate(&self) -> Option<&BaseGate<MixedPlain<R>, Self::DimTy>> {
        match self {
            MixedGate::Base(g) => Some(g),
            _ => None,
        }
    }

    fn wrap_base_gate(base_gate: BaseGate<MixedPlain<R>, Self::DimTy>) -> Self {
        Self::Base(base_gate)
    }
}