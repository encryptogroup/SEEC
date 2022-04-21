use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use bitvec::order::Msb0;
use gmw_rs::circuit::Circuit;
use gmw_rs::common::BitVec;
use gmw_rs::executor::Executor;
use gmw_rs::mul_triple::trusted_provider::{TrustedMTProviderClient, TrustedMTProviderServer};
use gmw_rs::mul_triple::trusted_seed_provider;
use gmw_rs::private_test_utils::init_tracing;
use gmw_rs::transport::Tcp;
use tokio::task::spawn_blocking;

// Test the TrustedMTProvider by executing the aes circuit with the provided mts
#[tokio::test]
async fn trusted_mt_provider() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let tp_addr = ("127.0.0.1", 7750);
    let _mt_server =
        tokio::spawn(async move { TrustedMTProviderServer::start(tp_addr).await.unwrap() });
    let circuit = spawn_blocking(move || {
        Circuit::load_bristol("test_resources/bristol-circuits/AES-non-expanded.txt")
    })
    .await??;
    let mt_provider_1 =
        TrustedMTProviderClient::new("some_id".into(), Tcp::connect(tp_addr).await?);
    let mt_provider_2 =
        TrustedMTProviderClient::new("some_id".into(), Tcp::connect(tp_addr).await?);
    let mut ex1 = Executor::new(&circuit, 0, mt_provider_1).await?;
    let mut ex2 = Executor::new(&circuit, 1, mt_provider_2).await?;
    let input_a = BitVec::repeat(false, 256);
    let input_b = BitVec::repeat(false, 256);
    let (mut t1, mut t2) = Tcp::new_local_pair(None).await?;
    let h1 = ex1.execute(input_a, &mut t1);
    let h2 = ex2.execute(input_b, &mut t2);
    let out = futures::try_join!(h1, h2)?;

    let exp_output: bitvec::vec::BitVec<u8, Msb0> = {
        let key = GenericArray::from([0u8; 16]);
        let mut block = GenericArray::from([0u8; 16]);
        let cipher = Aes128::new(&key);
        cipher.encrypt_block(&mut block);

        bitvec::vec::BitVec::from_slice(block.as_slice())
    };

    assert_eq!(exp_output, out.0 ^ out.1);

    Ok(())
}

// Test the TrustedMTProvider by executing the aes circuit with the provided mts
#[tokio::test]
async fn trusted_seed_mt_provider() -> anyhow::Result<()> {
    let _guard = init_tracing();
    let tp_addr = ("127.0.0.1", 7749);
    let _mt_server = tokio::spawn(async move {
        trusted_seed_provider::TrustedMTProviderServer::start(tp_addr)
            .await
            .unwrap()
    });
    let circuit = spawn_blocking(move || {
        Circuit::load_bristol("test_resources/bristol-circuits/AES-non-expanded.txt")
    })
    .await??;
    let mt_provider_1 = trusted_seed_provider::TrustedMTProviderClient::new(
        "some_id".into(),
        Tcp::connect(tp_addr).await?,
    );
    let mt_provider_2 = trusted_seed_provider::TrustedMTProviderClient::new(
        "some_id".into(),
        Tcp::connect(tp_addr).await?,
    );
    let mut ex1 = Executor::new(&circuit, 0, mt_provider_1).await?;
    let mut ex2 = Executor::new(&circuit, 1, mt_provider_2).await?;
    let input_a = BitVec::repeat(false, 256);
    let input_b = BitVec::repeat(false, 256);
    let (mut t1, mut t2) = Tcp::new_local_pair(None).await?;
    let h1 = ex1.execute(input_a, &mut t1);
    let h2 = ex2.execute(input_b, &mut t2);
    let out = futures::try_join!(h1, h2)?;

    let exp_output: bitvec::vec::BitVec<u8, Msb0> = {
        let key = GenericArray::from([0u8; 16]);
        let mut block = GenericArray::from([0u8; 16]);
        let cipher = Aes128::new(&key);
        cipher.encrypt_block(&mut block);

        bitvec::vec::BitVec::from_slice(block.as_slice())
    };

    assert_eq!(exp_output, out.0 ^ out.1);

    Ok(())
}
