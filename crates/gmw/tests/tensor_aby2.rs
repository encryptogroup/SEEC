use async_trait::async_trait;
use gmw::circuit::base_circuit::BaseGate;
use gmw::circuit::{BaseCircuit, ExecutableCircuit};
use gmw::executor::{Executor, GateOutputs, Input};
use gmw::mul_triple::boolean::insecure_provider::InsecureMTProvider;
use gmw::private_test_utils::init_tracing;
use gmw::protocols::tensor_aby2::{
    AbySetupProvider, BoolTensorAby2, BooleanGate, DeltaShareStorage, DeltaSharing, PartialShare,
    SetupData, ShareType, TensorGate,
};
use gmw::protocols::{DynDim, FunctionDependentSetup, Protocol};
use gmw::Circuit;
use mpc_bitmatrix::BitMatrix;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

#[derive(Clone, Debug)]
struct MockSetupProvider {
    party_id: usize,
    shares: Vec<Vec<PartialShare>>,
}

impl MockSetupProvider {
    fn new(party_id: usize, shares: &[GateOutputs<DeltaShareStorage>]) -> Self {
        let shares = shares[0]
            .iter()
            .zip(shares[1].iter())
            .map(|(a, b)| {
                let a = a.as_scalar().unwrap().clone();
                let b = b.as_scalar().unwrap().clone();
                a.into_iter()
                    .zip(b)
                    .map(|(a, b)| a.get_private() ^ b.get_private())
                    .collect()
            })
            .collect();
        Self { party_id, shares }
    }
}

#[async_trait]
impl FunctionDependentSetup<DeltaShareStorage, BooleanGate, usize> for MockSetupProvider {
    type Output = SetupData;
    type Error = ();

    async fn setup(
        &mut self,
        _shares: &GateOutputs<DeltaShareStorage>,
        circuit: &ExecutableCircuit<BooleanGate, usize>,
    ) -> Result<Self::Output, Self::Error> {
        let res = circuit
            .interactive_with_parents_iter()
            .map(|(gate, _gate_id, parents)| {
                let mut gate_inp: Vec<_> = parents
                    .map(|parent| {
                        self.shares[parent.circuit_id as usize]
                            .get(parent.gate_id.as_usize())
                            .unwrap()
                            .clone()
                    })
                    .collect();
                match gate {
                    BooleanGate::Tensor(TensorGate::MatMult { rows, cols }) => {
                        let PartialShare::Matrix(b) = gate_inp.pop().unwrap() else {panic!()};
                        let PartialShare::Matrix(a) = gate_inp.pop().unwrap() else {panic!()};
                        assert_eq!(a.dim().0, rows);
                        assert_eq!(b.dim().1, cols);
                        if self.party_id == 0 {
                            let a_mul_b = a.mat_mul(&b);
                            PartialShare::Matrix(a_mul_b)
                        } else {
                            PartialShare::Matrix(BitMatrix::zeros(rows, cols))
                        }
                    }
                    _other => unreachable!("illegal gate encounterd"),
                }
            })
            .collect();
        Ok(SetupData::from_raw(res))
    }
}

#[tokio::test]
#[ignore]
async fn simple_matmul() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let circ = ExecutableCircuit::DynLayers(build_circ());
    let mut rng = ChaChaRng::seed_from_u64(4269);
    let priv_seed1 = rng.gen();
    let priv_seed2 = rng.gen(); //rng.gen();
    let joint_seed1 = rng.gen(); //rng.gen();
    let joint_seed2 = rng.gen(); //rng.gen();

    let share_map1 = (0..1)
        .map(|pos| (pos, ShareType::Local))
        .chain((1..2).map(|pos| (pos, ShareType::Remote)))
        .collect();
    let share_map2 = (0..1)
        .map(|pos| (pos, ShareType::Remote))
        .chain((1..2).map(|pos| (pos, ShareType::Local)))
        .collect();
    dbg!(&share_map1);
    let mut sharing_state1 = DeltaSharing::new(priv_seed1, joint_seed1, joint_seed2, share_map1);
    let mut sharing_state2 = DeltaSharing::new(priv_seed2, joint_seed2, joint_seed1, share_map2);
    let state1 = BoolTensorAby2::new(sharing_state1.clone());
    let state2 = BoolTensorAby2::new(sharing_state2.clone());

    let (ch1, ch2) = mpc_channel::in_memory::new_pair(16);
    let delta_provider1 = AbySetupProvider::new(0, InsecureMTProvider, ch1.0, ch1.1);
    let delta_provider2 = AbySetupProvider::new(1, InsecureMTProvider, ch2.0, ch2.1);
    let gate_output = [
        state1.clone().setup_gate_outputs(0, &circ),
        state2.clone().setup_gate_outputs(1, &circ),
    ];
    let mut mock_delta_provider1 = MockSetupProvider::new(0, &gate_output[..]);
    let mut mock_delta_provider2 = MockSetupProvider::new(1, &gate_output[..]);

    let (mut ex1, mut ex2): (Executor<BoolTensorAby2, _>, Executor<BoolTensorAby2, _>) =
        tokio::try_join!(
            Executor::new_with_state(state1, &circ, 0, delta_provider1),
            Executor::new_with_state(state2, &circ, 1, delta_provider2)
        )
        .unwrap();

    let mock_setup1 = mock_delta_provider1
        .setup(ex1.gate_outputs(), &circ)
        .await
        .unwrap();
    let mock_setup2 = mock_delta_provider2
        .setup(ex2.gate_outputs(), &circ)
        .await
        .unwrap();

    assert_eq!(&mock_setup1, ex1.setup_storage());
    assert_eq!(&mock_setup2, ex2.setup_storage());

    let id_mat = PartialShare::Matrix(BitMatrix::identity(4));

    let mat = PartialShare::Matrix(BitMatrix::random(&mut rng, 4, 4));

    let (shared_id_mat, plain_delta_id_mat) = sharing_state1.share(vec![id_mat.clone()]);
    let mut inp2 = sharing_state2.plain_delta_to_share(plain_delta_id_mat);
    let (shared_mat, plain_delta_mat) = sharing_state2.share(vec![mat.clone()]);
    inp2.extend(shared_mat);

    let inp1 = {
        let mut inp = shared_id_mat;
        inp.extend(sharing_state1.plain_delta_to_share(plain_delta_mat));
        inp
    };

    let reconst = DeltaSharing::reconstruct(inp1.clone(), inp2.clone());
    assert_eq!(reconst, [id_mat, mat.clone()]);

    let (mut ch1, mut ch2) = mpc_channel::in_memory::new_pair(16);

    let (out0, out1) = tokio::try_join!(
        ex1.execute(Input::Scalar(inp1), &mut ch1.0, &mut ch1.1),
        ex2.execute(Input::Scalar(inp2), &mut ch2.0, &mut ch2.1),
    )?;

    let out0 = out0.into_scalar().unwrap();
    let out1 = out1.into_scalar().unwrap();
    let out = DeltaSharing::reconstruct(out0, out1);
    eprintln!("{:b}", out[0].clone().into_matrix().unwrap());
    assert_eq!(out[0], mat, "Wrong output");

    Ok(())
}

fn build_circ() -> Circuit<BooleanGate, usize> {
    let mut circ = BaseCircuit::new();

    let in1 = circ.add_gate(BooleanGate::Base(BaseGate::Input(DynDim::new(&[4, 4]))));
    let in2 = circ.add_gate(BooleanGate::Base(BaseGate::Input(DynDim::new(&[4, 4]))));

    let matmul = circ.add_wired_gate(
        BooleanGate::Tensor(TensorGate::MatMult { rows: 4, cols: 4 }),
        &[in2, in1],
    );

    circ.add_wired_gate(
        BooleanGate::Base(BaseGate::Output(DynDim::new(&[4, 4]))),
        &[matmul],
    );

    circ.into()
}
