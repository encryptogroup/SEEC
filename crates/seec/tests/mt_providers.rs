use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use bitvec::order::Msb0;
use seec::circuit::dyn_layers::Circuit;
use seec::circuit::ExecutableCircuit;
use seec::common::BitVec;
use seec::executor::{Executor, Input};
use seec::mul_triple::boolean;
use seec::mul_triple::boolean::trusted_provider::{
    TrustedMTProviderClient, TrustedMTProviderServer,
};
use seec::private_test_utils::init_tracing;
use seec::protocols::boolean_gmw::BooleanGmw;
use seec_channel::{sub_channel, tcp};
use tokio::task::spawn_blocking;

// Test the TrustedMTProvider by executing the aes circuit with the provided mts
#[tokio::test]
async fn trusted_mt_provider() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let tp_addr = ("127.0.0.1", 7750);
    let _mt_server =
        tokio::spawn(async move { TrustedMTProviderServer::start(tp_addr).await.unwrap() });
    let circuit = ExecutableCircuit::DynLayers(
        spawn_blocking(move || {
            Circuit::load_bristol("test_resources/bristol-circuits/AES-non-expanded.txt")
        })
        .await??,
    );
    let (sender, _, receiver, _) = tcp::connect(tp_addr).await?;
    let mt_provider_1 = TrustedMTProviderClient::new("some_id".into(), sender, receiver);
    let (sender, _, receiver, _) = tcp::connect(tp_addr).await?;
    let mt_provider_2 = TrustedMTProviderClient::new("some_id".into(), sender, receiver);
    let mut ex1 = Executor::<BooleanGmw, _>::new(&circuit, 0, mt_provider_1).await?;
    let mut ex2 = Executor::<BooleanGmw, _>::new(&circuit, 1, mt_provider_2).await?;
    let input_a = BitVec::repeat(false, 256);
    let input_b = BitVec::repeat(false, 256);
    let (mut t1, mut t2) = tcp::new_local_pair::<seec_channel::Receiver<_>>(None).await?;
    let (mut t1, mut t2) = tokio::try_join!(
        sub_channel(&mut t1.0, &mut t1.2, 8),
        sub_channel(&mut t2.0, &mut t2.2, 8)
    )?;
    let h1 = ex1.execute(Input::Scalar(input_a), &mut t1.0, &mut t1.1);
    let h2 = ex2.execute(Input::Scalar(input_b), &mut t2.0, &mut t2.1);
    let out = futures::try_join!(h1, h2)?;

    let exp_output: bitvec::vec::BitVec<u8, Msb0> = {
        let key = GenericArray::from([0u8; 16]);
        let mut block = GenericArray::from([0u8; 16]);
        let cipher = Aes128::new(&key);
        cipher.encrypt_block(&mut block);

        bitvec::vec::BitVec::from_slice(block.as_slice())
    };
    let out0 = out.0.into_scalar().unwrap();
    let out1 = out.1.into_scalar().unwrap();
    assert_eq!(exp_output, out0 ^ out1);

    Ok(())
}

// Test the TrustedMTProvider by executing the aes circuit with the provided mts
#[tokio::test]
async fn trusted_seed_mt_provider() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let tp_addr = ("127.0.0.1", 7749);
    let _mt_server = tokio::spawn(async move {
        boolean::trusted_seed_provider::TrustedMTProviderServer::start(tp_addr)
            .await
            .unwrap()
    });
    let circuit = ExecutableCircuit::DynLayers(
        spawn_blocking(move || {
            Circuit::load_bristol("test_resources/bristol-circuits/AES-non-expanded.txt")
        })
        .await??,
    );
    let (sender, _, receiver, _) = tcp::connect(tp_addr).await?;
    let mt_provider_1 = boolean::trusted_seed_provider::TrustedMTProviderClient::new(
        "some_id".into(),
        sender,
        receiver,
    );
    let (sender, _, receiver, _) = tcp::connect(tp_addr).await?;
    let mt_provider_2 = boolean::trusted_seed_provider::TrustedMTProviderClient::new(
        "some_id".into(),
        sender,
        receiver,
    );
    let mut ex1 = Executor::<BooleanGmw, _>::new(&circuit, 0, mt_provider_1).await?;
    let mut ex2 = Executor::<BooleanGmw, _>::new(&circuit, 1, mt_provider_2).await?;
    let input_a = BitVec::repeat(false, 256);
    let input_b = BitVec::repeat(false, 256);
    let (mut t1, mut t2) = tcp::new_local_pair::<seec_channel::Receiver<_>>(None).await?;
    let (mut t1, mut t2) = tokio::try_join!(
        sub_channel(&mut t1.0, &mut t1.2, 8),
        sub_channel(&mut t2.0, &mut t2.2, 8)
    )?;
    let h1 = ex1.execute(Input::Scalar(input_a), &mut t1.0, &mut t1.1);
    let h2 = ex2.execute(Input::Scalar(input_b), &mut t2.0, &mut t2.1);
    let out = futures::try_join!(h1, h2)?;

    let exp_output: bitvec::vec::BitVec<u8, Msb0> = {
        let key = GenericArray::from([0u8; 16]);
        let mut block = GenericArray::from([0u8; 16]);
        let cipher = Aes128::new(&key);
        cipher.encrypt_block(&mut block);

        bitvec::vec::BitVec::from_slice(block.as_slice())
    };

    let out0 = out.0.into_scalar().unwrap();
    let out1 = out.1.into_scalar().unwrap();
    assert_eq!(exp_output, out0 ^ out1);

    Ok(())
}
