use crate::common::{execute_bristol, init_tracing};
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use anyhow::Result;
use bitvec::bitvec;
use bitvec::order::{Lsb0, Msb0};
use gmw_rs::common::BitVec;
use hex_literal::hex;

mod common;

#[tokio::test]
async fn eval_8_bit_adder() -> Result<()> {
    let _guard = init_tracing();
    let inputs_0 = bitvec![u8, Lsb0;
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 0, 0,
    ];
    let inputs_1 = BitVec::repeat(false, 16);
    let exp_output: BitVec = {
        let mut bits = bitvec![u8, Lsb0; 0, 0, 1, 0, 1, 0, 0, 0];
        bits.resize(8, false);
        bits
    };

    let out = execute_bristol(
        "test_resources/bristol-circuits/int_add8_depth.bristol",
        (inputs_0, inputs_1),
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}

#[tokio::test]
async fn eval_aes_circuit() -> Result<()> {
    let _guard = init_tracing();
    let inputs_0 = BitVec::repeat(false, 256);
    let inputs_1 = BitVec::repeat(false, 256);
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
        (inputs_0, inputs_1),
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}

#[tokio::test]
async fn eval_sha_256_circuit() -> Result<()> {
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
        (inputs_0, inputs_1),
    )
    .await?;
    assert_eq!(exp_output, out);
    Ok(())
}
