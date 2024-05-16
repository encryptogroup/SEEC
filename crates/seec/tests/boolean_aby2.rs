use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use seec::circuit::ExecutableCircuit;
use seec::common::BitVec;
use seec::executor::{Executor, GateOutputs, Input};
use seec::mul_triple::boolean::insecure_provider::InsecureMTProvider;
use seec::private_test_utils::init_tracing;
use seec::protocols::aby2::{
    AbySetupProvider, AstraSetupHelper, AstraSetupProvider, BooleanAby2, DeltaSharing, InputBy,
    ShareType,
};
use seec::Circuit;
use seec_channel::multi;

#[tokio::test(flavor = "multi_thread")]
async fn eval_8_bit_adder() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let circ = ExecutableCircuit::DynLayers(Circuit::load_bristol(
        "test_resources/bristol-circuits/int_add8_depth.bristol",
    )?);

    let priv_seed1 = thread_rng().gen();
    let priv_seed2 = thread_rng().gen();
    let joint_seed = thread_rng().gen();

    let share_map1 = (0..8)
        .map(|pos| (pos, ShareType::Local))
        .chain((8..16).map(|pos| (pos, ShareType::Remote)))
        .collect();
    let share_map2 = (0..8)
        .map(|pos| (pos, ShareType::Remote))
        .chain((8..16).map(|pos| (pos, ShareType::Local)))
        .collect();
    let mut sharing_state1 = DeltaSharing::new(0, priv_seed1, joint_seed, share_map1);
    let mut sharing_state2 = DeltaSharing::new(1, priv_seed2, joint_seed, share_map2);
    let state1 = BooleanAby2::new(sharing_state1.clone());
    let state2 = BooleanAby2::new(sharing_state2.clone());

    let (ch1, ch2) = seec_channel::in_memory::new_pair(16);
    let delta_provider1 = AbySetupProvider::new(0, InsecureMTProvider::default(), ch1.0, ch1.1);
    let delta_provider2 = AbySetupProvider::new(1, InsecureMTProvider::default(), ch2.0, ch2.1);

    let (mut ex1, mut ex2): (Executor<BooleanAby2, u32>, Executor<BooleanAby2, u32>) =
        tokio::try_join!(
            Executor::new_with_state(state1, &circ, 0, delta_provider1),
            Executor::new_with_state(state2, &circ, 1, delta_provider2)
        )
        .unwrap();
    let (shared_30, plain_delta_30) = sharing_state1.share(BitVec::from_element(30_u8));
    let (shared_12, plain_delta_12) = sharing_state2.share(BitVec::from_element(12_u8));

    let inp1 = {
        let mut inp = shared_30;
        inp.extend(sharing_state1.plain_delta_to_share(plain_delta_12));
        inp
    };
    let inp2 = {
        let mut inp = sharing_state2.plain_delta_to_share(plain_delta_30);
        inp.extend(shared_12);
        inp
    };

    let reconstruct: BitVec = inp1
        .clone()
        .into_iter()
        .zip(inp2.clone())
        .map(|(sh1, sh2)| {
            assert_eq!(sh1.get_public(), sh2.get_public());
            sh1.get_public() ^ sh1.get_private() ^ sh2.get_private()
        })
        .collect();
    assert_eq!(BitVec::from_slice(&[30_u8, 12]), reconstruct);

    let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(16);

    let (out0, out1) = tokio::try_join!(
        ex1.execute(Input::Scalar(inp1), &mut ch1.0, &mut ch1.1),
        ex2.execute(Input::Scalar(inp2), &mut ch2.0, &mut ch2.1),
    )?;

    let out0 = out0.into_scalar().unwrap();
    let out1 = out1.into_scalar().unwrap();
    let out_bits: BitVec = DeltaSharing::reconstruct(out0, out1);
    assert_eq!(BitVec::from_element(42_u8), out_bits);

    Ok(())
}

#[tokio::test]
async fn astra_setup() -> anyhow::Result<()> {
    let _g = init_tracing();
    let mut channels = multi::new_local(3);
    let helper_ch = channels.pop().unwrap();
    let p1_ch = channels.pop().unwrap();
    let p0_ch = channels.pop().unwrap();
    let priv_seed_p0: [u8; 32] = thread_rng().gen();
    let priv_seed_p1: [u8; 32] = thread_rng().gen();
    let joint_seed: [u8; 32] = thread_rng().gen();
    let helper = AstraSetupHelper::new(
        helper_ch.0,
        helper_ch.1,
        priv_seed_p0,
        priv_seed_p1,
        joint_seed,
    );

    let astra_setup0 = AstraSetupProvider::new(0, p0_ch.0, p0_ch.1, priv_seed_p0);
    let astra_setup1 = AstraSetupProvider::new(1, p1_ch.0, p1_ch.1, priv_seed_p1);

    let circ = ExecutableCircuit::DynLayers(
        Circuit::load_bristol("test_resources/bristol-circuits/int_add8_depth.bristol").unwrap(),
    );

    let input_map = (0..8)
        .map(|i| (i, InputBy::P0))
        .chain((8..16).map(|i| (i, InputBy::P1)))
        .collect();
    let circ_clone = circ.clone();
    let jh = tokio::spawn(async move { helper.setup(&circ_clone, input_map).await });

    let share_map1 = (0..8)
        .map(|pos| (pos, ShareType::Local))
        .chain((8..16).map(|pos| (pos, ShareType::Remote)))
        .collect();
    let share_map2 = (0..8)
        .map(|pos| (pos, ShareType::Remote))
        .chain((8..16).map(|pos| (pos, ShareType::Local)))
        .collect();

    let mut sharing_state1 = DeltaSharing::new(0, priv_seed_p0, joint_seed, share_map1);
    let mut sharing_state2 = DeltaSharing::new(1, priv_seed_p1, joint_seed, share_map2);
    let state1 = BooleanAby2::new(sharing_state1.clone());
    let state2 = BooleanAby2::new(sharing_state2.clone());

    let (mut ex1, mut ex2): (Executor<BooleanAby2, u32>, Executor<BooleanAby2, u32>) =
        tokio::try_join!(
            Executor::new_with_state(state1, &circ, 0, astra_setup0),
            Executor::new_with_state(state2, &circ, 1, astra_setup1)
        )
        .unwrap();
    let (shared_30, plain_delta_30) = sharing_state1.share(BitVec::from_element(30_u8));
    let (shared_12, plain_delta_12) = sharing_state2.share(BitVec::from_element(12_u8));

    let inp1 = {
        let mut inp = shared_30;
        inp.extend(sharing_state1.plain_delta_to_share(plain_delta_12));
        inp
    };
    let inp2 = {
        let mut inp = sharing_state2.plain_delta_to_share(plain_delta_30);
        inp.extend(shared_12);
        inp
    };

    let reconstruct: BitVec = inp1
        .clone()
        .into_iter()
        .zip(inp2.clone())
        .map(|(sh1, sh2)| {
            assert_eq!(sh1.get_public(), sh2.get_public());
            sh1.get_public() ^ sh1.get_private() ^ sh2.get_private()
        })
        .collect();
    assert_eq!(BitVec::from_slice(&[30_u8, 12]), reconstruct);

    let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(16);

    let (out0, out1) = tokio::try_join!(
        ex1.execute(Input::Scalar(inp1), &mut ch1.0, &mut ch1.1),
        ex2.execute(Input::Scalar(inp2), &mut ch2.0, &mut ch2.1),
    )?;

    let out0 = out0.into_scalar().unwrap();
    let out1 = out1.into_scalar().unwrap();
    let out_bits: BitVec = DeltaSharing::reconstruct(out0, out1);
    assert_eq!(BitVec::from_element(42_u8), out_bits);

    Ok(())
}
