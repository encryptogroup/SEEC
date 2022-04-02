use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use anyhow::Result;
use bitvec::bitvec;
use bitvec::order::{Lsb0, Msb0};
use gmw_rs::common::BitVec;
use gmw_rs::private_test_utils::{execute_bristol, init_tracing, TestTransport};
use hex_literal::hex;

#[tokio::test]
async fn eval_8_bit_adder() -> Result<()> {
    let _guard = init_tracing();
    let exp_output = BitVec::from_element(42);

    let out = execute_bristol(
        "test_resources/bristol-circuits/int_add8_depth.bristol",
        (12_u8, 30_u8),
        TestTransport::Tcp,
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}

#[tokio::test]
async fn eval_aes_circuit() -> Result<()> {
    let _guard = init_tracing();
    // It seems that the output of the circuit and the aes crate use different bit orderings
    // for the output.
    let exp_output: bitvec::vec::BitVec<u8, Msb0> = {
        let key = GenericArray::from([0u8; 16]);
        let mut block = GenericArray::from([0u8; 16]);
        let cipher = Aes128::new(&key);
        cipher.encrypt_block(&mut block);

        bitvec::vec::BitVec::from_slice(block.as_slice())
    };
    let out = execute_bristol(
        "test_resources/bristol-circuits/AES-non-expanded.txt",
        (0_u128, 0_u128),
        TestTransport::Tcp,
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}

#[tokio::test]
async fn eval_sha_256_circuit_zeros() -> Result<()> {
    let _guard = init_tracing();
    let inputs_0 = BitVec::repeat(false, 512);
    let inputs_1 = BitVec::repeat(false, 512);

    // The output of the circuit is apparently in Msb order
    let exp_output: bitvec::vec::BitVec<u8, Msb0> = bitvec::vec::BitVec::from_slice(&hex!(
        // From: https://homes.esat.kuleuven.be/~nsmart/MPC/sha-256-test.txt
        // The output of the circuit is not the *normal* sha256 output
        "da5698be17b9b46962335799779fbeca8ce5d491c0d26243bafef9ea1837a9d8"
    ));
    let out = execute_bristol(
        "test_resources/bristol-circuits/sha-256.txt",
        [inputs_0, inputs_1],
        TestTransport::Tcp,
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}

#[tokio::test]
async fn eval_sha_256_circuit_ones() -> Result<()> {
    let _guard = init_tracing();
    let inputs_0 = BitVec::repeat(true, 512);
    let inputs_1 = BitVec::repeat(false, 512);

    // The output of the circuit is apparently in Msb order
    let exp_output: bitvec::vec::BitVec<u8, Msb0> = bitvec::vec::BitVec::from_slice(&hex!(
        // From: https://homes.esat.kuleuven.be/~nsmart/MPC/sha-256-test.txt
        // The output of the circuit is not the *normal* sha256 output
        "ef0c748df4da50a8d6c43c013edc3ce76c9d9fa9a1458ade56eb86c0a64492d2"
    ));
    let out = execute_bristol(
        "test_resources/bristol-circuits/sha-256.txt",
        [inputs_0, inputs_1],
        TestTransport::Tcp,
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}

#[tokio::test]
async fn eval_sha_256_low_depth() -> Result<()> {
    let _guard = init_tracing();
    let inputs_0 = BitVec::repeat(false, 768);
    let inputs_1 = BitVec::repeat(false, 768);

    // Expected output computed via Motion
    let exp_output = bitvec![u8, Lsb0; 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0];
    let out = execute_bristol(
        "test_resources/bristol-circuits/sha-256-low_depth.txt",
        [inputs_0, inputs_1],
        TestTransport::Tcp,
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}
