use aes::cipher::KeyIvInit;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::{fs, iter};

use anyhow::{ensure, Context, Result};
use bitvec::order::Msb0;
use bitvec::view::BitView;
use cbc::cipher::{block_padding::Pkcs7, BlockEncryptMut};
use clap::Parser;
use once_cell::sync::Lazy;

use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};

use gmw::circuit::base_circuit::Load;
use gmw::{BooleanGate, Circuit, CircuitBuilder, SharedCircuit, SubCircuitOutput};
use tracing::info;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

use gmw::circuit::BaseCircuit;
use gmw::common::{BitSlice, BitVec};
use gmw::executor::Executor;
use gmw::mul_triple::insecure_provider::InsecureMTProvider;
use gmw::protocols::boolean_gmw::{BooleanGmw, XorSharing};
use gmw::protocols::{boolean_gmw, Sharing};
use gmw::secret::{inputs, Secret};
use mpc_channel::{sub_channels_for, Channel};

#[derive(Parser, Debug)]
struct Args {
    /// Id of this party
    #[clap(long)]
    id: usize,

    /// Address of server to bind or connect to
    #[clap(long)]
    server: SocketAddr,

    /// 16 byte key in Hex format
    #[clap(long, default_value = "00112233445566778899aabbccddeeff")]
    key: String,

    /// File to encrypt (required if --id 1)
    #[clap(conflicts_with("input_bytes"))]
    file: Option<PathBuf>,

    /// Size of artificial input to encrypt
    #[clap(long)]
    input_bytes: Option<usize>,

    /// Validate the encryption by sending key and iv in plain
    #[clap(long)]
    validate: bool,

    /// When enabled, a sub-circuit will be used to store the AES circuit.
    #[clap(long)]
    use_sc: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    let (mut sender, bytes_written, mut receiver, bytes_read) = match args.id {
        0 => mpc_channel::tcp::listen(args.server).await?,
        1 => mpc_channel::tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    let (mut sharing_channel, executor_channel) =
        sub_channels_for!(&mut sender, &mut receiver, 8, Msg, boolean_gmw::Msg).await?;
    let mut sharing = XorSharing::new(ChaChaRng::from_rng(thread_rng()).context("Sharing RNG")?);

    match args.id {
        0 => {
            let key: [u8; 16] = hex::decode(&args.key)
                .context("Decoding key")?
                .try_into()
                .ok()
                .context("Key must be 16 bytes long")?;
            // Apparently, the AES circuit wants it's arguments in Msb format. However,
            // the executor currently expects arguments to be in a BitVec with Lsb order.
            // The following changes the order of the bits manyally
            let msb_key: BitVec = key.iter().map(|byte| byte.reverse_bits()).collect();
            let iv = thread_rng().gen::<[u8; 16]>();
            let msb_iv: BitVec = iv.iter().map(|byte| byte.reverse_bits()).collect();

            let [key_share0, key_share1] = sharing.share(msb_key);
            let [iv_share0, iv_share1] = sharing.share(msb_iv);
            sharing_channel
                .0
                .send(Msg::ShareIvKey {
                    iv: iv_share1,
                    key: key_share1,
                })
                .await?;
            let Msg::ShareInput(input_share) = sharing_channel.1.recv().await?.ok_or(anyhow::anyhow!("Remote closed"))? else {
                anyhow::bail!("Received wrong message. Expected ShareInput")
            };

            let out = encrypt(
                &args,
                executor_channel,
                &input_share,
                &key_share0,
                &iv_share0,
            )
            .await?;

            sharing_channel
                .0
                .send(Msg::ReconstructAesCiphertext(out.clone()))
                .await?;
            if args.validate {
                sharing_channel.0.send(Msg::PlainIvKey { iv, key }).await?;
            }
            let Msg::Ack = sharing_channel.1.recv().await.context("Receiving Ack")?.ok_or(anyhow::anyhow!("Unexpected None"))? else {
                anyhow::bail!("Expected Ack message")
            };
            info!(
                bytes_written = bytes_written.get(),
                bytes_read = bytes_read.get(),
            );
        }
        1 => {
            let (data, padded_data) = get_data(&args).context("Loading data to encrypt")?;
            let padded_file_data = BitVec::from_vec(padded_data);
            let [input_share0, input_share1] = sharing.share(padded_file_data);
            sharing_channel
                .0
                .send(Msg::ShareInput(input_share1))
                .await?;
            let Msg::ShareIvKey {iv: iv_share, key: key_share } = sharing_channel.1.recv().await
                .context("Receiving IvKeyShare")?
                .ok_or(anyhow::anyhow!("Remote closed"))? else {
                    anyhow::bail!("Received wrong message. Expected IvKeyShare")
            };
            let out = encrypt(
                &args,
                executor_channel,
                &input_share0,
                &key_share,
                &iv_share,
            )
            .await?;

            let Msg::ReconstructAesCiphertext(shared_out) = sharing_channel.1.recv().await
                .context("Receiving ciphertext share")?
                .ok_or(anyhow::anyhow!("Remote closed"))? else {
                    anyhow::bail!("Received wrong message. Expected IvKeyShare")
            };

            let ciphertext = sharing.reconstruct([out, shared_out]);

            if args.validate {
                let Msg::PlainIvKey {iv, key} = sharing_channel.1.recv().await
                    .context("Reconstructing Iv/Key")?
                    .ok_or(anyhow::anyhow!("Remote closed"))? else {
                        anyhow::bail!("Received wrong message. Expected ReconstructIvKey")
                };
                validate(iv, key, &data, &ciphertext)?;
            }

            sharing_channel.0.send(Msg::Ack).await?;

            let encoded = hex::encode(ciphertext.as_raw_slice());
            info!(
                bytes_written = bytes_written.get(),
                bytes_read = bytes_read.get(),
                ciphertext = encoded
            );
        }
        _ => unreachable!(),
    };

    Ok(())
}

fn get_data(args: &Args) -> Result<(Vec<u8>, Vec<u8>)> {
    assert_eq!(1, args.id, "Function must be called by id 1");
    let data = match (&args.file, args.input_bytes) {
        (None, None) => {
            anyhow::bail!("Either <file> or --input-bytes must be set for --id 1")
        }
        (Some(path), None) => fs::read(path)?,
        (None, Some(bytes)) => {
            vec![0; bytes]
        }
        _ => unreachable!("Both can't be set"),
    };
    // Pad the file data with
    // simple PKCS#7 padding (https://www.rfc-editor.org/rfc/rfc5652#section-6.3)
    let padding_bytes_needed: u8 = (16 - data.len() % 16).try_into().unwrap();
    let mut padded_file_data = data.clone();
    padded_file_data.extend(iter::repeat(padding_bytes_needed).take(padding_bytes_needed.into()));
    //
    padded_file_data.iter_mut().for_each(|bit| {
        *bit = bit.reverse_bits();
    });
    Ok((data, padded_file_data))
}

async fn encrypt(
    args: &Args,
    mut executor_channel: Channel<boolean_gmw::Msg>,
    shared_file: &BitSlice,
    shared_key: &BitSlice,
    shared_iv: &BitSlice,
) -> Result<BitVec> {
    let circuit = build_enc_circuit(shared_file.len(), args.use_sc)?;

    let mut input = shared_key.to_bitvec();
    input.extend_from_bitslice(shared_iv);
    input.extend_from_bitslice(shared_file);

    let mut executor: Executor<BooleanGmw, usize> =
        Executor::new(&circuit, args.id, InsecureMTProvider::default()).await?;
    Ok(executor
        .execute(input, &mut executor_channel.0, &mut executor_channel.1)
        .await?)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum Msg {
    ShareIvKey { iv: BitVec, key: BitVec },
    ShareInput(BitVec),
    ReconstructAesCiphertext(BitVec),
    PlainIvKey { iv: [u8; 16], key: [u8; 16] },
    Ack,
}

fn build_enc_circuit(data_size_bits: usize, use_sc: bool) -> Result<Circuit<BooleanGate, usize>> {
    assert_eq!(
        data_size_bits % 128,
        0,
        "data_size must be multiple of 128 bits"
    );
    // let aes_circ = BaseCircuit::<BooleanGate, usize>::load_bristol(aes_bristol, Load::SubCircuit)?
    //     .into_shared();
    let key_size = 128;
    let iv_size = 128;
    let key = inputs(key_size);
    let iv = inputs(iv_size);
    let data = inputs(data_size_bits);

    let mut chaining_state = iv;
    data.chunks_exact(128)
        .for_each(|chunk| aes_circ(&key, chunk, &mut chaining_state, use_sc));

    Ok(CircuitBuilder::<BooleanGate, usize>::global_into_circuit())
}

fn aes_circ(
    key: &[Secret<BooleanGmw, usize>],
    chunk: &[Secret<BooleanGmw, usize>],
    chaining_state: &mut [Secret<BooleanGmw, usize>],
    use_sc: bool,
) {
    static AES_CIRC: Lazy<SharedCircuit<BooleanGate, usize>> = Lazy::new(|| {
        BaseCircuit::load_bristol(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test_resources/bristol-circuits/AES-non-expanded.txt"),
            Load::SubCircuit,
        )
        .expect("Loading Aes circuit")
        .into_shared()
    });
    let inp: Vec<_> = chunk
        .iter()
        .zip(chaining_state.iter())
        .map(|(d, c)| d.clone() ^ c)
        .chain(key.iter().cloned())
        .collect();
    let output = if use_sc {
        let (output, circ_id) = CircuitBuilder::with_global(|builder| {
            let circ_id = builder.push_circuit(AES_CIRC.clone());
            (builder.connect_sub_circuit(&inp, circ_id), circ_id)
        });
        output.connect_to_main(circ_id)
    } else {
        CircuitBuilder::with_global(|builder| {
            let inp = inp.into_iter().map(|sh| {
                assert_eq!(0, sh.circuit_id());
                sh.gate_id()
            });
            let out = builder
                .get_main_circuit()
                .lock()
                .add_sub_circuit(&AES_CIRC.lock(), inp);
            out.into_iter()
                .map(|gate_id| Secret::from_parts(0, gate_id))
                .collect()
        })
    };
    chaining_state.clone_from_slice(&output);
    for sh in output {
        sh.output();
    }
}

fn validate(
    plain_iv: [u8; 16],
    plain_key: [u8; 16],
    plaintext: &[u8],
    ciphertext: &BitSlice,
) -> Result<()> {
    type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;
    let expected: Vec<_> = Aes128CbcEnc::new(&plain_key.into(), &plain_iv.into())
        .encrypt_padded_vec_mut::<Pkcs7>(plaintext);
    let expected = expected.view_bits::<Msb0>();
    ensure!(
        expected == ciphertext,
        "Ciphertext does not match expected.\nExpected: {:?},\nActual:\t{:?}",
        expected,
        ciphertext
    );
    Ok(())
}

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
}
