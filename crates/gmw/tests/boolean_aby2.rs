use gmw::common::BitVec;
use gmw::executor::Executor;

use gmw::mul_triple::insecure_provider::InsecureMTProvider;
use gmw::private_test_utils::init_tracing;
use gmw::protocols::aby2::{AbySetupProvider, BooleanAby2, DeltaSharing, ShareType};
use gmw::Circuit;
use rand::{thread_rng, Rng};

#[tokio::test(flavor = "multi_thread")]
async fn eval_8_bit_adder() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let circ = Circuit::load_bristol("test_resources/bristol-circuits/int_add8_depth.bristol")?;

    let priv_seed1 = thread_rng().gen();
    let priv_seed2 = thread_rng().gen();
    let joint_seed1 = thread_rng().gen();
    let joint_seed2 = thread_rng().gen();

    let share_map1 = (0..8)
        .map(|pos| (pos, ShareType::Local))
        .chain((8..16).map(|pos| (pos, ShareType::Remote)))
        .collect();
    let share_map2 = (0..8)
        .map(|pos| (pos, ShareType::Remote))
        .chain((8..16).map(|pos| (pos, ShareType::Local)))
        .collect();
    let mut sharing_state1 = DeltaSharing::new(priv_seed1, joint_seed1, joint_seed2, share_map1);
    let mut sharing_state2 = DeltaSharing::new(priv_seed2, joint_seed2, joint_seed1, share_map2);
    let state1 = BooleanAby2::new(sharing_state1.clone());
    let state2 = BooleanAby2::new(sharing_state2.clone());

    let (ch1, ch2) = mpc_channel::in_memory::new_pair(16);
    let delta_provider1 = AbySetupProvider::new(0, InsecureMTProvider, ch1.0, ch1.1);
    let delta_provider2 = AbySetupProvider::new(1, InsecureMTProvider, ch2.0, ch2.1);

    let (mut ex1, mut ex2): (Executor<BooleanAby2, usize>, Executor<BooleanAby2, usize>) =
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

    let (mut ch1, mut ch2) = mpc_channel::in_memory::new_pair(16);

    let (out1, out2) = tokio::try_join!(
        ex1.execute(inp1, &mut ch1.0, &mut ch1.1),
        ex2.execute(inp2, &mut ch2.0, &mut ch2.1),
    )?;

    let out_bits: BitVec = DeltaSharing::reconstruct(out1, out2);
    assert_eq!(BitVec::from_element(42_u8), out_bits);

    Ok(())
}
