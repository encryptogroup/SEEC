use crate::executor;
use crate::protocols::aby2::{BooleanAby2, DeltaSharing, ShareType};
use crate::protocols::arithmetic_aby2::{ArithmeticAby2, EvalShares};
use crate::protocols::{aby2, arithmetic_aby2, Gate, Protocol, Ring, ScalarDim, ShareStorage};
use bitvec::view::BitViewSized;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter;
use crate::circuit::{ExecutableCircuit, GateIdx};
use crate::executor::{GateOutputs, Input};
use crate::protocols::mixed_gmw::MixedGmw;
pub use crate::protocols::mixed::Mixed as MixedPlain;
use share::MixedShare;
use setup_data::MixedSetupData;
use super::aby2::Share;
mod fd_setup;
pub use fd_setup::MixedAbySetupProvider;
use gate::{ConvGate, MixedGate};
use share_storage::MixedShareStorage;
use crate::protocols::mixed::Mixed;

mod setup_data;
mod share_storage;
mod share;
mod gate;

pub type MixedAbySetupMsg<R> = executor::Message<MixedGmw<R>>;
type BooleanShareStorage = <BooleanAby2 as Protocol>::ShareStorage;
type ArithmeticShareStorage<R> = <ArithmeticAby2<R> as Protocol>::ShareStorage;

type ArithmeticShare<R> = arithmetic_aby2::Share<R>;
type BooleanShare = aby2::Share;

type ConvMsg<R> = Vec<MixedPlain<R>>;

#[derive(Clone, Debug)]
pub struct MixedAby2<R>
where
    R: Ring,
    Standard: Distribution<R>,
{
    delta_sharing_state: DeltaSharing,
    a: ArithmeticAby2<R>,
    b: BooleanAby2,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Msg<R> {
    bool: aby2::Msg,
    arith: arithmetic_aby2::Msg<R>,

    conv: ConvMsg<R>,
}


impl<R> Protocol for MixedAby2<R>
where
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
    const SIMD_SUPPORT: bool = false;
    type Plain = MixedPlain<R>;
    type Share = MixedShare<R>;
    type Msg = Msg<R>;
    type SimdMsg = ();
    type Gate = MixedGate<R>;
    type Wire = ();
    type ShareStorage = MixedShareStorage<R>;
    type SetupStorage = MixedSetupData<R>;

    fn share_constant(
        &self,
        _party_id: usize,
        output_share: Self::Share,
        val: Self::Plain,
    ) -> Self::Share {
        match (output_share, val) {
            (MixedShare::Bool(o), MixedPlain::Bool(b)) => {
                MixedShare::Bool(self.b.share_constant(_party_id, o, b))
            }
            (MixedShare::Arith(o), MixedPlain::Arith(a)) => {
                MixedShare::Arith(self.a.share_constant(_party_id, o, a))
            }
            (_, _) => panic!("mismatch between output_share and val!"),
        }
    }

    fn evaluate_non_interactive(
        &self,
        party_id: usize,
        gate: &Self::Gate,
        mut inputs: impl Iterator<Item = Self::Share>,
    ) -> Self::Share {
        match gate {
            MixedGate::Base(b) => b.default_evaluate(party_id, inputs),
            MixedGate::Bool(g) => {
                let inputs = inputs.map(MixedShare::unwrap_bool);
                MixedShare::Bool(self.b.evaluate_non_interactive(party_id, g, inputs))
            }
            MixedGate::Arith(g) => {
                let inputs = inputs.map(MixedShare::unwrap_arith);
                MixedShare::Arith(self.a.evaluate_non_interactive(party_id, g, inputs))
            }
            MixedGate::Conv(ConvGate::Bit2A) =>
                panic!("called evaluate_non_interactive on interactive gate"),
            /* &MixedGate::Conv(ConvGate::A2BSelectBit(i)) => {
                assert!(i < R::BITS);
                let a = inputs.next().expect("Insufficient inputs").unwrap_arith();
                let get_bit = move |x: R| (x >> i) & R::ONE == R::ONE;
                MixedShare::Bool(BooleanShare {
                    public: get_bit(a.public),
                    private: get_bit(a.private),
                })
            } */
        }
    }

    fn compute_msg(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = Self::Gate>,
        gate_outputs: impl Iterator<Item = Self::Share>,
        mut inputs: impl Iterator<Item = Self::Share>,
        preprocessing_data: &mut Self::SetupStorage,
    ) -> Self::Msg {
        let mut b_inputs = BooleanShareStorage::default();
        let mut b_outs = vec![];
        let mut a_inputs = ArithmeticShareStorage::default();
        let mut a_outs = vec![];
        let mut conv_inputs = MixedShareStorage::default();
        let mut conv_outs = vec![];
        let (bool_gates, arith_gates, conv_gates) = interactive_gates.zip(gate_outputs).fold(
            (vec![], vec![], vec![]),
            |(mut bgates, mut agates, mut conv_gates), (mgate, mout)| {
                let input_size = mgate.input_size();
                let exact_inputs = inputs.by_ref().take(input_size);

                match mgate {
                    MixedGate::Bool(g) => {
                        b_inputs.extend(exact_inputs.map(MixedShare::unwrap_bool));
                        b_outs.push(mout.unwrap_bool());
                        bgates.push(g);
                    }
                    MixedGate::Arith(g) => {
                        a_inputs.extend(exact_inputs.map(MixedShare::unwrap_arith));
                        a_outs.push(mout.unwrap_arith());
                        agates.push(g);
                    }
                    MixedGate::Conv(g_conv) => {
                        conv_inputs.extend(exact_inputs);
                        conv_outs.push(mout);
                        conv_gates.push(g_conv)
                    }
                    MixedGate::Base(g) => {
                        panic!("Encountered base gate {g:?} in compute_msg");
                    }
                };
                (bgates, agates, conv_gates)
            },
        );

        let b_msg = self.b.compute_msg(
            party_id,
            bool_gates.into_iter(),
            b_outs.into_iter(),
            b_inputs.into_iter(),
            &mut preprocessing_data.bool,
        );
        let a_msg = self.a.compute_msg(
            party_id,
            arith_gates.into_iter(),
            a_outs.into_iter(),
            a_inputs.into_iter(),
            &mut preprocessing_data.arith,
        );



        let preprocessing_data = &mut preprocessing_data.conv;
        let mut conv_inputs = conv_inputs.into_iter();
        let mut conv_outs = conv_outs.into_iter();

        let reshares = compute_conv_gate_msg(
            party_id,
            conv_gates.into_iter(),
            conv_inputs.into_iter(),
            conv_outs.into_iter(),
            preprocessing_data,
        );

        Msg {
            bool: b_msg,
            arith: a_msg,
            conv: reshares,
        }        
    }

    fn evaluate_interactive(
        &self,
        party_id: usize,
        interactive_gates: impl Iterator<Item = Self::Gate>,
        gate_outputs: impl Iterator<Item = Self::Share>,
        own_msg: Self::Msg,
        other_msg: Self::Msg,
        preprocessing_data: &mut Self::SetupStorage,
    ) -> Self::ShareStorage {
        let mut a_shares = vec![];
        let mut b_shares = vec![];
        let mut c_shares = vec![];

        let mut c_gates = vec![];

        let gates: Vec<_> = interactive_gates.collect();
        gates.iter().zip(gate_outputs).for_each(|(gate, share)| match gate {
            MixedGate::Conv(_) => {
                c_shares.push(share);
                c_gates.push(gate.clone());
            },
            MixedGate::Arith(_) | MixedGate::Bool(_) | MixedGate::Base(_) => match share {
                MixedShare::Bool(b) => b_shares.push(b),
                MixedShare::Arith(a) => a_shares.push(a),
            },
        });

        let b_storage = self.b.evaluate_interactive(
            party_id,
            iter::empty(),
            b_shares.into_iter(),
            own_msg.bool,
            other_msg.bool,
            &mut preprocessing_data.bool,
        );
        let a_storage = self.a.evaluate_interactive(
            party_id,
            iter::empty(),
            a_shares.into_iter(),
            own_msg.arith,
            other_msg.arith,
            &mut preprocessing_data.arith,
        );
        let c_storage: Vec<MixedShare<R>> = conv_evaluate_interactive(
            c_gates.into_iter(),
            c_shares.into_iter(),
            own_msg.conv,
            other_msg.conv,
        );

        let mut ret = Vec::with_capacity(b_storage.len() + a_storage.len());
        let mut b_storage = b_storage.into_iter();
        let mut a_storage = a_storage.into_iter();
        let mut c_storage = c_storage.into_iter();
        for g in gates {
            ret.push(match g {
                MixedGate::Base(_) => panic!("Unexpected base gate {g:?}"),
                MixedGate::Bool(_) => {
                    MixedShare::Bool(b_storage.next().expect("Insufficient Bool outputs"))
                }
                MixedGate::Arith(_) => {
                    MixedShare::Arith(a_storage.next().expect("Insufficient Arith outputs"))
                }
                MixedGate::Conv(_) => {
                    c_storage.next().expect("Insufficient outputs for conv")
                },
            });
        }
        MixedShareStorage::Mixed(ret)
    }

    fn setup_gate_outputs<Idx: GateIdx>(
        &mut self,
        _party_id: usize,
        circuit: &ExecutableCircuit<MixedPlain<R>, MixedGate<R>, Idx>,
    ) -> GateOutputs<Self::ShareStorage> {
        let storage: Vec<_> = circuit
            .gate_counts()
            .map(|(gate_count, simd_size)| {
                assert_eq!(None, simd_size);
                Input::Scalar(MixedShareStorage::repeat(Default::default(), gate_count))
            })
            .collect();
        let mut storage = GateOutputs::new(storage);

        for (gate, sc_gate_id, parents) in circuit.iter_with_parents() {
            let gate_output_iter = parents.map(|parent| storage.get(parent));
            let rng = match self
                .delta_sharing_state
                .input_position_share_type_map
                .get(&sc_gate_id.gate_id.as_usize())
            {
                None => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Local) => &mut self.delta_sharing_state.private_rng,
                Some(ShareType::Remote) => &mut self.delta_sharing_state.remote_joint_rng,
            };
            let output = gate.setup_output_share(gate_output_iter, rng);
            storage.set(sc_gate_id, output);
        }
        storage
    }
}

fn conv_evaluate_interactive<R>(
    gates: impl Iterator<Item = MixedGate<R>>,
    gate_outputs: impl Iterator<Item = MixedShare<R>>,
    own_msg: ConvMsg<R>,
    other_msg: ConvMsg<R>,
) -> Vec<MixedShare<R>>
where
    R: Ring,
    [R; 1]: BitViewSized,
{
    let own_msg = own_msg.into_iter();
    let other_msg = other_msg.into_iter();

    let interactive_gates = gates.flat_map(|g| match g {
        MixedGate::Conv(c) => Some(c),
        _ => None,
    });

    itertools::izip!(interactive_gates, gate_outputs, own_msg, other_msg).map(
        |(gate, gate_out, public_v_i, public_v_i_min_1)| match gate {
            ConvGate::Bit2A => {
                let public_v_i = public_v_i.unwrap_arith();
                let public_v_i_min_1 = public_v_i_min_1.unwrap_arith();

                let mut gate_out = gate_out.unwrap_arith();
                gate_out.public = public_v_i.wrapping_add(&public_v_i_min_1);
                MixedShare::Arith(gate_out)
            },
        }
    ).collect()
}

fn compute_conv_gate_msg<R>(
    party_id: usize,
    conv_gates: impl Iterator<Item=ConvGate>,
    mut conv_inputs: impl Iterator<Item=MixedShare<R>>,
    mut conv_outs: impl Iterator<Item=MixedShare<R>>,
    preprocessing_data: &mut Vec<EvalShares<Mixed<R>>>,
) -> Vec<Mixed<R>>
where
    R: Ring,
    [R; 1]: BitViewSized,
{
    let mut reshares: Vec<_> = vec![];
    for conv_gate in conv_gates { match conv_gate {
        ConvGate::Bit2A => {
            let Share {
                public: input_public,
                private: _,
            } = conv_inputs.next().expect("Empty input").unwrap_bool();
            let input_public: R = if input_public { R::ONE } else { R::ZERO };

            let generated_additive_sharing = preprocessing_data
                .pop()
                .expect("missing preprocessing data")
                .shares
                .pop()
                .expect("missing eval share")
                .unwrap_arith();

            let two = R::ONE.wrapping_add(&R::ONE);
            let output_additive =
                R::ONE.wrapping_sub(&two.wrapping_mul(&input_public))
                    .wrapping_mul(&generated_additive_sharing);
            let output_additive = if party_id == 1 {
                input_public.wrapping_add(&output_additive)
            } else {
                output_additive
            };

            // reshare: I don't think this is right atm.
            let ArithmeticShare {
                public: _,
                private: delta,
            } = conv_outs.next().expect("Missing output").unwrap_arith();
            reshares.push(MixedPlain::Arith(
                output_additive
                    .wrapping_add(&delta)
            ));
        }
    }}
    reshares
}

impl DeltaSharing
{
    pub fn mixed_share<R: Ring>(&mut self, input: Vec<MixedPlain<R>>) -> (MixedShareStorage<R>, Vec<MixedPlain<R>>)
    where
        Standard: Distribution<R>,
    {
        input
            .into_iter()
            .map(|plain| match plain {
                MixedPlain::Bool(bit) => {
                    let my_delta = self.private_rng.gen::<bool>();
                    let other_delta = self.local_joint_rng.gen::<bool>();
                    let plain_delta = bit ^ my_delta ^ other_delta;
                    let my_share = MixedShare::new_bool(my_delta, plain_delta);
                    (my_share, MixedPlain::Bool(plain_delta))
                }
                MixedPlain::Arith(num) => {
                    let my_delta: R = self.private_rng.gen();
                    let other_delta: R = self.local_joint_rng.gen();
                    let plain_delta = num.wrapping_add(&my_delta).wrapping_add(&other_delta);
                    let my_share = MixedShare::new_arith(my_delta, plain_delta.clone());
                    (my_share, MixedPlain::Arith(plain_delta))
                }
            })
            .unzip()
    }

    pub fn mixed_plain_delta_to_share<R: Ring>(&mut self, plain_deltas: Vec<MixedPlain<R>>) -> MixedShareStorage<R>
    where
        Standard: Distribution<R>,
    {
        plain_deltas
            .into_iter()
            .map(|plain_delta| match plain_delta {
                MixedPlain::Bool(plain_bool) => {
                    MixedShare::new_bool(self.remote_joint_rng.gen::<bool>(), plain_bool)
                }
                MixedPlain::Arith(plain_arith) => {
                    MixedShare::new_arith(self.remote_joint_rng.gen(), plain_arith)
                }
            })
            .collect()
    }

    pub fn mixed_reconstruct<R: Ring>(a: MixedShareStorage<R>, b: MixedShareStorage<R>) -> Vec<MixedPlain<R>>
    where
        Standard: Distribution<R>,
    {
        a.into_iter()
            .zip(b)
            .map(|(sh1, sh2)| match (sh1, sh2) {
                (MixedShare::Bool(sh1), MixedShare::Bool(sh2)) => {
                    assert_eq!(
                        sh1.get_public(),
                        sh2.get_public(),
                        "Public shares of outputs can't differ"
                    );
                    let res = sh1.get_public() ^ sh1.get_private() ^ sh2.get_private();
                    MixedPlain::Bool(res)
                }
                (MixedShare::Arith(sh1), MixedShare::Arith(sh2)) => {
                    assert_eq!(
                        sh1.get_public(),
                        sh2.get_public(),
                        "Public shares of outputs can't differ"
                    );
                    let res = sh1
                        .get_public()
                        .wrapping_sub(&sh1.get_private())
                        .wrapping_sub(&sh2.get_private());
                    MixedPlain::Arith(res)
                }
                (MixedShare::Bool(_), MixedShare::Arith(_))
                | (MixedShare::Arith(_), MixedShare::Bool(_)) => panic!("Mismatch in share type!"),
            })
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::aby2::BooleanGate;
    use executor::Executor;
    use executor::Input;
    use base::BaseGate;
    use crate::circuit::BaseCircuit;
    use crate::gate::base;
    use crate::protocols::mixed_gmw;

    type R = u32;
    type AG = arithmetic_aby2::ArithmeticGate<R>;


    #[tokio::test]
    async fn test_exclusively_arithmetic() {
        let mut c = BaseCircuit::<MixedPlain<R>, MixedGate<R>>::new();
        let i0 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let i1 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let i3 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));

        let a = c.add_wired_gate(MixedGate::Arith(AG::Mul), &[i0, i1]);
        let b = c.add_wired_gate(MixedGate::Arith(AG::Mul), &[a, i3]);
        let add = c.add_wired_gate(MixedGate::Arith(AG::Add), &[a, b]);
        let sub = c.add_wired_gate(MixedGate::Arith(AG::Sub), &[i3, b]);

        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[a]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[b]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[add]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[sub]);

        let c = ExecutableCircuit::DynLayers(c.into());

        let (ch0, ch1) = seec_channel::in_memory::new_pair(2);
        let setup0 = MixedAbySetupProvider::<_, R> {
            party_id: 0,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch0.0,
            receiver: ch0.1,
            setup_data: None,
        };
        let setup1 = MixedAbySetupProvider::<_, R> {
            party_id: 1,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch1.0,
            receiver: ch1.1,
            setup_data: None,
        };

        let p_state = MixedAby2::<R>{
            delta_sharing_state: DeltaSharing::insecure_default(),
            b: BooleanAby2::new(DeltaSharing::insecure_default()),
            a: ArithmeticAby2::new(DeltaSharing::insecure_default()),
        };
        let (mut ex1, mut ex2) = tokio::try_join!(
            Executor::new_with_state(p_state.clone(), &c, 0, setup0),
            Executor::new_with_state(p_state, &c, 1, setup1),
        ).unwrap();

        let (inp0, mask) = DeltaSharing::insecure_default().arith_share(vec![5, 4, 18]);

        let inp1 = DeltaSharing::insecure_default().arith_plain_delta_to_share(mask);

        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(2);

        let inp0 = MixedShareStorage::<R>::Arith(inp0);
        let inp1 = MixedShareStorage::<R>::Arith(inp1);

        let h1 = ex1.execute(Input::Scalar(inp0), &mut ch1.0, &mut ch1.1);
        let h2 = ex2.execute(Input::Scalar(inp1), &mut ch2.0, &mut ch2.1);
        let (res1, res2) = tokio::try_join!(h1, h2).unwrap();
        let res =
            DeltaSharing::arith_reconstruct(
                res1
                    .into_scalar()
                    .unwrap()
                    .into_iter()
                    .map(MixedShare::unwrap_arith)
                    .collect(),
                res2
                    .into_scalar()
                    .unwrap()
                    .into_iter()
                    .map(MixedShare::unwrap_arith)
                    .collect(),
            );

        assert_eq!(vec![20, 360, 380, 4294966954u32], res);
    }

    #[tokio::test]
    async fn test_arith_addition() {
        let mut c = BaseCircuit::<MixedPlain<R>, MixedGate<R>>::new();
        let i0 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let i1 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));

        let a = c.add_wired_gate(MixedGate::Arith(AG::Add), &[i0, i1]);

        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[a]);

        let c = ExecutableCircuit::DynLayers(c.into());

        let (ch0, ch1) = seec_channel::in_memory::new_pair(2);
        let setup0 = MixedAbySetupProvider::<_, R> {
            party_id: 0,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch0.0,
            receiver: ch0.1,
            setup_data: None,
        };
        let setup1 = MixedAbySetupProvider::<_, R> {
            party_id: 1,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch1.0,
            receiver: ch1.1,
            setup_data: None,
        };

        let p_state = MixedAby2::<R>{
            delta_sharing_state: DeltaSharing::insecure_default(),
            b: BooleanAby2::new(DeltaSharing::insecure_default()),
            a: ArithmeticAby2::new(DeltaSharing::insecure_default()),
        };
        let (mut ex1, mut ex2) = tokio::try_join!(
        Executor::new_with_state(p_state.clone(), &c, 0, setup0),
        Executor::new_with_state(p_state, &c, 1, setup1),
        ).unwrap();

        let (inp0, mask) = DeltaSharing::insecure_default().arith_share(vec![5, 7]);

        let inp1 = DeltaSharing::insecure_default().arith_plain_delta_to_share(mask);

        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(2);

        let inp0 = MixedShareStorage::<R>::Arith(inp0);
        let inp1 = MixedShareStorage::<R>::Arith(inp1);

        let h1 = ex1.execute(Input::Scalar(inp0), &mut ch1.0, &mut ch1.1);
        let h2 = ex2.execute(Input::Scalar(inp1), &mut ch2.0, &mut ch2.1);
        let (res1, res2) = tokio::try_join!(h1, h2).unwrap();
        let res =
            DeltaSharing::arith_reconstruct(
                res1
                    .into_scalar()
                    .unwrap()
                    .into_iter()
                    .map(MixedShare::unwrap_arith)
                    .collect(),
                res2
                    .into_scalar()
                    .unwrap()
                    .into_iter()
                    .map(MixedShare::unwrap_arith)
                    .collect(),
            );

        assert_eq!(vec![12], res);
    }

    #[tokio::test]
    async fn test_arith_multiplication() {
        let mut c = BaseCircuit::<MixedPlain<R>, MixedGate<R>>::new();
        let i0 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let i1 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let i3 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));

        let a = c.add_wired_gate(MixedGate::Arith(AG::Mul), &[i0, i1]);
        let b = c.add_wired_gate(MixedGate::Arith(AG::Mul), &[i1, i3]);

        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[a]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[b]);

        let c = ExecutableCircuit::DynLayers(c.into());

        let (ch0, ch1) = seec_channel::in_memory::new_pair(2);
        let setup0 = MixedAbySetupProvider::<_, R> {
            party_id: 0,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch0.0,
            receiver: ch0.1,
            setup_data: None,
        };
        let setup1 = MixedAbySetupProvider::<_, R> {
            party_id: 1,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch1.0,
            receiver: ch1.1,
            setup_data: None,
        };

        let p_state = MixedAby2::<R>{
            delta_sharing_state: DeltaSharing::insecure_default(),
            b: BooleanAby2::new(DeltaSharing::insecure_default()),
            a: ArithmeticAby2::new(DeltaSharing::insecure_default()),
        };
        let (mut ex1, mut ex2) = tokio::try_join!(
        Executor::new_with_state(p_state.clone(), &c, 0, setup0),
        Executor::new_with_state(p_state, &c, 1, setup1),
        ).unwrap();

        let inputs = vec![5, 7, 2];
        let (inp0, mask) = DeltaSharing::insecure_default().arith_share(inputs.clone());

        let inp1 = DeltaSharing::insecure_default().arith_plain_delta_to_share(mask);

        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(2);

        let inp0 = MixedShareStorage::<R>::Arith(inp0);
        let inp1 = MixedShareStorage::<R>::Arith(inp1);

        let h1 = ex1.execute(Input::Scalar(inp0), &mut ch1.0, &mut ch1.1);
        let h2 = ex2.execute(Input::Scalar(inp1), &mut ch2.0, &mut ch2.1);
        let (res1, res2) = tokio::try_join!(h1, h2).unwrap();
        let res =
            DeltaSharing::arith_reconstruct(
                res1
                    .into_scalar()
                    .unwrap()
                    .into_iter()
                    .map(MixedShare::unwrap_arith)
                    .collect(),
                res2
                    .into_scalar()
                    .unwrap()
                    .into_iter()
                    .map(MixedShare::unwrap_arith)
                    .collect(),
            );

        assert_eq!(vec![35, 14], res);
    }

    #[tokio::test]
    async fn test_bin() {
        let mut c = BaseCircuit::<MixedPlain<R>, MixedGate<R>>::new();
        let i0 = c.add_gate(MixedGate::Base(BaseGate::Constant(MixedPlain::Bool(true))));
        let i1 = c.add_gate(MixedGate::Base(BaseGate::Constant(MixedPlain::Bool(false))));
        let i2 = c.add_gate(MixedGate::Base(BaseGate::Constant(MixedPlain::Bool(true))));
        let a = c.add_wired_gate(MixedGate::Bool(BooleanGate::And { n: 2 }), &[i0, i1]);
        let b = c.add_wired_gate(MixedGate::Bool(BooleanGate::And { n: 2 }), &[i0, i2]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[a]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[b]);

        let inputs = vec![];
        let expected = vec![MixedPlain::Bool(false), MixedPlain::Bool(true)];

        let c = ExecutableCircuit::DynLayers(c.into());
        let (ch0, ch1) = seec_channel::in_memory::new_pair(2);
        let setup0 = MixedAbySetupProvider::<_, R> {
            party_id: 0,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch0.0,
            receiver: ch0.1,
            setup_data: None,
        };
        let setup1 = MixedAbySetupProvider::<_, R> {
            party_id: 1,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch1.0,
            receiver: ch1.1,
            setup_data: None,
        };

        let p_state = MixedAby2::<R>{
            delta_sharing_state: DeltaSharing::insecure_default(),
            b: BooleanAby2::new(DeltaSharing::insecure_default()),
            a: ArithmeticAby2::new(DeltaSharing::insecure_default()),
        };
        let (mut ex1, mut ex2) = tokio::try_join!(
            Executor::new_with_state(p_state.clone(), &c, 0, setup0),
            Executor::new_with_state(p_state, &c, 1, setup1),
        ).unwrap();

        let (inp0, mask) = DeltaSharing::insecure_default().mixed_share(inputs.clone());
        let inp1 = DeltaSharing::insecure_default().mixed_plain_delta_to_share(mask);

        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(2);

        let h1 = ex1.execute(Input::Scalar(inp0), &mut ch1.0, &mut ch1.1);
        let h2 = ex2.execute(Input::Scalar(inp1), &mut ch2.0, &mut ch2.1);
        let (res1, res2) = tokio::try_join!(h1, h2).unwrap();
        let res = DeltaSharing::mixed_reconstruct(res1.into_scalar().unwrap(), res2.into_scalar().unwrap());

        assert_eq!(expected, res);

    }

    #[tokio::test]
    async fn test_bit_to_a() {
        let mut c = BaseCircuit::<MixedPlain<R>, MixedGate<R>>::new();
        let i0 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let i1 = c.add_gate(MixedGate::Base(BaseGate::Input(ScalarDim)));
        let a = c.add_wired_gate(MixedGate::Conv(ConvGate::Bit2A), &[i0]);
        let b = c.add_wired_gate(MixedGate::Conv(ConvGate::Bit2A), &[i1]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[a]);
        let _out = c.add_wired_gate(MixedGate::Base(BaseGate::Output(ScalarDim)), &[b]);

        let inputs = vec![MixedPlain::Bool(true), MixedPlain::Bool(false)];
        let expected = vec![MixedPlain::Arith(1u32), MixedPlain::Arith(0u32)];

        let c = ExecutableCircuit::DynLayers(c.into());

        simple_test_suite(inputs, expected, c).await;
    }

    async fn simple_test_suite<R, Idx>(
        inputs: Vec<Mixed<R>>,
        expected: Vec<Mixed<R>>,
        c: ExecutableCircuit<Mixed<R>, MixedGate<R>, Idx>)
    where
        R: Ring,
        Standard: Distribution<R>,
        [R; 1]: BitViewSized,
        Idx: GateIdx,
    {
        let (ch0, ch1) = seec_channel::in_memory::new_pair(2);
        let setup0 = MixedAbySetupProvider::<_, R> {
            party_id: 0,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch0.0,
            receiver: ch0.1,
            setup_data: None,
        };
        let setup1 = MixedAbySetupProvider::<_, R> {
            party_id: 1,
            mt_provider: mixed_gmw::InsecureMixedSetup::default(),
            sender: ch1.0,
            receiver: ch1.1,
            setup_data: None,
        };

        let p_state = MixedAby2::<R> {
            delta_sharing_state: DeltaSharing::insecure_default(),
            b: BooleanAby2::new(DeltaSharing::insecure_default()),
            a: ArithmeticAby2::new(DeltaSharing::insecure_default()),
        };
        let (mut ex1, mut ex2) = tokio::try_join!(
            Executor::new_with_state(p_state.clone(), &c, 0, setup0),
            Executor::new_with_state(p_state, &c, 1, setup1),
        ).unwrap();

        let (inp0, mask) = DeltaSharing::insecure_default().mixed_share(inputs.clone());
        let inp1 = DeltaSharing::insecure_default().mixed_plain_delta_to_share(mask);

        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(2);

        let h1 = ex1.execute(Input::Scalar(inp0), &mut ch1.0, &mut ch1.1);
        let h2 = ex2.execute(Input::Scalar(inp1), &mut ch2.0, &mut ch2.1);
        let (res1, res2) = tokio::try_join!(h1, h2).unwrap();
        let res = DeltaSharing::mixed_reconstruct(res1.into_scalar().unwrap(), res2.into_scalar().unwrap());

        assert_eq!(expected, res);
    }
}
