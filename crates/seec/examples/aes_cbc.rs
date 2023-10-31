use aes::cipher::KeyIvInit;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::net::SocketAddr;
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::{io, iter, mem};

use anyhow::{ensure, Context, Result};
use bitvec::order::Msb0;
use bitvec::view::BitView;
use cbc::cipher::{block_padding::Pkcs7, BlockEncryptMut};
use clap::{Args, Parser};
use once_cell::sync::Lazy;
use rand::rngs::{OsRng, ThreadRng};
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use seec::bench::BenchParty;
use seec::circuit::base_circuit::Load;
use seec::circuit::{BaseCircuit, ExecutableCircuit};
use seec::common::{BitSlice, BitVec};
use seec::executor::{Executor, Input, Message, Output};
use seec::mul_triple::boolean::insecure_provider::InsecureMTProvider;
use seec::mul_triple::MTProvider;
use seec::protocols::boolean_gmw::{BooleanGmw, XorSharing};
use seec::protocols::Sharing;
use seec::secret::{inputs, Secret};
use seec::{mul_triple, BooleanGate, CircuitBuilder, SharedCircuit, SubCircuitOutput};
use seec_channel::{sub_channel_with, sub_channels_for, Channel};
use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
enum ProgArgs {
    Compile(CompileArgs),
    Execute(ExecuteArgs),
}

#[derive(Args, Debug)]
/// Compile an AES-CBC circuit to encrypt a specific number of blocks
struct CompileArgs {
    /// Size of artificial input to encrypt
    #[arg(long, default_value = "1")]
    input_blocks: usize,

    /// When enabled, a sub-circuit will be used to store the AES circuit.
    #[arg(long)]
    use_sc: bool,

    /// If set, circuit is stored with precomputed static layers
    #[arg(long)]
    static_layers: bool,

    /// Output path of the compile circuit
    #[arg(default_value = "aes_circ.seec")]
    output: PathBuf,
}

#[derive(Args, Debug)]
/// Execute the provided circuit
struct ExecuteArgs {
    /// Id of this party
    #[arg(long)]
    id: usize,

    /// Address of server to bind or connect to
    #[arg(long, default_value = "127.0.0.1:7745")]
    server: SocketAddr,

    /// 16 byte key in Hex format. Ignored when using --bench
    #[arg(long, default_value = "00112233445566778899aabbccddeeff")]
    key: String,

    /// Size of artificial input to encrypt in blocks of 16 bytes = 128 bits
    #[arg(long, conflicts_with = "bench")]
    input_blocks: Option<usize>,

    /// Validate the encryption by sending key and iv in plain
    #[arg(long, conflicts_with = "bench")]
    validate: bool,

    /// Benchmark execution (ignores key)
    #[arg(long)]
    bench: bool,

    /// Insecure setup
    #[arg(long)]
    insecure_setup: bool,

    #[arg(default_value = "aes_circ.seec")]
    circuit: PathBuf,
}

#[derive(Serialize, Deserialize, Debug)]
enum Msg {
    ShareIvKey {
        iv: BitVec<usize>,
        key: BitVec<usize>,
    },
    ShareInput(BitVec<usize>),
    ReconstructAesCiphertext(Output<BitVec<usize>>),
    PlainIvKey {
        iv: [usize; 2],
        key: [usize; 2],
    },
    OtChannel(seec_channel::Receiver<mul_triple::boolean::ot_ext::DefaultMsg>),
    Ack,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = ProgArgs::parse();
    match args {
        ProgArgs::Compile(compile_args) => {
            compile(&compile_args).context("Failed to compile circuit")?
        }
        ProgArgs::Execute(exec_args) if exec_args.bench => bench_execute(&exec_args)
            .await
            .context("failed to execute benchmark")?,
        ProgArgs::Execute(exec_args) => execute(&exec_args)
            .await
            .context("failed to execute circuit")?,
    };

    Ok(())
}

fn compile(args: &CompileArgs) -> Result<()> {
    let input_bits = args.input_blocks * 128;
    let circ = build_enc_circuit(input_bits, args.use_sc).context("failed to construct circuit")?;
    let circ = if args.static_layers {
        circ.precompute_layers()
    } else {
        circ
    };
    let out = BufWriter::new(File::create(&args.output).context("failed to create output file")?);
    bincode::serialize_into(out, &circ).context("failed to serialize circuit")?;
    Ok(())
}

async fn execute(args: &ExecuteArgs) -> Result<()> {
    let (mut sender, bytes_written, mut receiver, bytes_read) = match args.id {
        0 => seec_channel::tcp::listen(args.server).await?,
        1 => seec_channel::tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    let (mut main_channel, executor_channel) =
        sub_channels_for!(&mut sender, &mut receiver, 8, Msg, Message<BooleanGmw>).await?;
    let mut sharing = XorSharing::new(ChaChaRng::from_rng(thread_rng()).context("Sharing RNG")?);

    let ot_channel = if args.insecure_setup.not() {
        let sub_channel = sub_channel_with(
            &mut main_channel.0,
            &mut main_channel.1,
            128,
            Msg::OtChannel,
            |msg| match msg {
                Msg::OtChannel(receiver) => Some(receiver),
                _ => None,
            },
        )
        .await
        .context("failed to create ot sub_channel")?;
        Some(sub_channel)
    } else {
        None
    };

    match args.id {
        0 => {
            let key: [u8; 16] = hex::decode(&args.key)
                .context("Decoding key")?
                .try_into()
                .ok()
                .context("Key must be 16 bytes long")?;
            let key: [usize; 2] = bytemuck::cast(key);
            // Apparently, the AES circuit wants it's arguments in Msb format. However,
            // the executor currently expects arguments to be in a BitVec with Lsb order.
            // The following changes the order of the bits manyally
            // TODO the following is likely wrong with the change to BitVec<usize>
            let msb_key: BitVec<usize> = key.iter().map(|bytes| bytes.reverse_bits()).collect();
            let iv = thread_rng().gen::<[usize; 2]>();
            let msb_iv: BitVec<usize> = iv.iter().map(|bytes| bytes.reverse_bits()).collect();

            let [key_share0, key_share1] = sharing.share(msb_key);
            let [iv_share0, iv_share1] = sharing.share(msb_iv);
            main_channel
                .0
                .send(Msg::ShareIvKey {
                    iv: iv_share1,
                    key: key_share1,
                })
                .await?;
            let Msg::ShareInput(input_share) = main_channel
                .1
                .recv()
                .await?
                .ok_or(anyhow::anyhow!("Remote closed"))?
            else {
                anyhow::bail!("Received wrong message. Expected ShareInput")
            };

            let out = encrypt(
                args,
                executor_channel,
                ot_channel,
                &input_share,
                &key_share0,
                &iv_share0,
            )
            .await?;

            main_channel
                .0
                .send(Msg::ReconstructAesCiphertext(out.clone()))
                .await?;
            if args.validate {
                main_channel.0.send(Msg::PlainIvKey { iv, key }).await?;
            }
            // Try to recv Ack but ignore errors
            let _ = main_channel.1.recv().await;

            info!(
                bytes_written = bytes_written.get(),
                bytes_read = bytes_read.get(),
            );
        }
        1 => {
            let (data, padded_data) = get_data(args).context("Loading data to encrypt")?;
            let mut padded_data_usize = vec![0_usize; padded_data.len() / mem::size_of::<usize>()];
            bytemuck::cast_slice_mut(&mut padded_data_usize).clone_from_slice(&padded_data);
            let padded_file_data = BitVec::from_vec(padded_data_usize);
            let [input_share0, input_share1] = sharing.share(padded_file_data);
            main_channel.0.send(Msg::ShareInput(input_share1)).await?;
            let Msg::ShareIvKey {
                iv: iv_share,
                key: key_share,
            } = main_channel
                .1
                .recv()
                .await
                .context("Receiving IvKeyShare")?
                .ok_or(anyhow::anyhow!("Remote closed"))?
            else {
                anyhow::bail!("Received wrong message. Expected IvKeyShare")
            };
            let out = encrypt(
                args,
                executor_channel,
                ot_channel,
                &input_share0,
                &key_share,
                &iv_share,
            )
            .await?;

            let Msg::ReconstructAesCiphertext(shared_out) = main_channel
                .1
                .recv()
                .await
                .context("Receiving ciphertext share")?
                .ok_or(anyhow::anyhow!("Remote closed"))?
            else {
                anyhow::bail!("Received wrong message. Expected IvKeyShare")
            };

            let ciphertext = match (out, shared_out) {
                (Output::Scalar(out), Output::Scalar(shared_out)) => {
                    XorSharing::<ThreadRng>::reconstruct([out, shared_out])
                }
                (Output::Simd(out), Output::Simd(shared_out)) => out
                    .into_iter()
                    .zip(shared_out)
                    .flat_map(|(a, b)| XorSharing::<ThreadRng>::reconstruct([a, b]))
                    .collect(),
                _ => unreachable!("Non compatible output"),
            };

            if args.validate {
                let Msg::PlainIvKey { iv, key } = main_channel
                    .1
                    .recv()
                    .await
                    .context("Reconstructing Iv/Key")?
                    .ok_or(anyhow::anyhow!("Remote closed"))?
                else {
                    anyhow::bail!("Received wrong message. Expected ReconstructIvKey")
                };
                validate(iv, key, &data, &ciphertext)?;
            }

            main_channel.0.send(Msg::Ack).await?;

            let encoded = hex::encode(bytemuck::cast_slice(ciphertext.as_raw_slice()));
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

async fn bench_execute(args: &ExecuteArgs) -> Result<()> {
    let exec_circ: ExecutableCircuit<BooleanGate, usize> = bincode::deserialize_from(
        BufReader::new(File::open(&args.circuit).context("Failed to open circuit file")?),
    )?;

    let p = BenchParty::<BooleanGmw, usize>::new(args.id)
        .explicit_circuit(exec_circ)
        .precompute_layers(false)
        .server(args.server)
        .insecure_setup(args.insecure_setup);
    let res = p
        .bench()
        .await
        .context("failed to run benchmark for aes-cbc")?;
    println!(
        "{}",
        serde_json::to_string(&res).context("failed to serialize bench results to json")?
    );
    Ok(())
}

fn get_data(args: &ExecuteArgs) -> Result<(Vec<u8>, Vec<u8>)> {
    assert_eq!(1, args.id, "Function must be called by id 1");
    let data = vec![0; args.input_blocks.unwrap() * 16 - 5];
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
    args: &ExecuteArgs,
    mut executor_channel: Channel<Message<BooleanGmw>>,
    ot_channel: Option<Channel<mul_triple::boolean::ot_ext::DefaultMsg>>,
    shared_file: &BitSlice<usize>,
    shared_key: &BitSlice<usize>,
    shared_iv: &BitSlice<usize>,
) -> Result<Output<BitVec<usize>>> {
    let exec_circ: ExecutableCircuit<BooleanGate, usize> = bincode::deserialize_from(
        BufReader::new(File::open(&args.circuit).context("Failed to open circuit file")?),
    )?;

    let mut input = shared_key.to_bitvec();
    input.extend_from_bitslice(shared_iv);
    input.extend_from_bitslice(shared_file);

    let mtp = match ot_channel {
        None => InsecureMTProvider.into_dyn(),
        Some(ot_channel) => mul_triple::boolean::ot_ext::OtMTProvider::new_with_default_ot_ext(
            OsRng,
            ot_channel.0,
            ot_channel.1,
        )
        .into_dyn(),
    };

    let mut executor: Executor<BooleanGmw, usize> = Executor::new(&exec_circ, args.id, mtp).await?;
    Ok(executor
        .execute(
            Input::Scalar(input),
            &mut executor_channel.0,
            &mut executor_channel.1,
        )
        .await?)
}

fn build_enc_circuit(
    data_size_bits: usize,
    use_sc: bool,
) -> Result<ExecutableCircuit<BooleanGate, usize>> {
    assert_eq!(
        data_size_bits % 128,
        0,
        "data_size must be multiple of 128 bits"
    );
    let key_size = 128;
    let iv_size = 128;
    let key = inputs(key_size);
    let iv = inputs(iv_size);
    let data = inputs(data_size_bits);

    let mut chaining_state = iv;
    data.chunks_exact(128)
        .for_each(|chunk| aes_cbc_chunk(&key, chunk, &mut chaining_state, use_sc));

    Ok(ExecutableCircuit::DynLayers(CircuitBuilder::<
        BooleanGate,
        usize,
    >::global_into_circuit()))
}

fn aes_cbc_chunk(
    key: &[Secret<BooleanGmw, usize>],
    chunk: &[Secret<BooleanGmw, usize>],
    chaining_state: &mut [Secret<BooleanGmw, usize>],
    use_sc: bool,
) {
    let inp: Vec<_> = chunk
        .iter()
        .zip(chaining_state.iter())
        .map(|(d, c)| d.clone() ^ c)
        .collect();
    let output = aes128(key, &inp, use_sc);
    chaining_state.clone_from_slice(&output);
    for sh in output {
        sh.output();
    }
}

fn aes128(
    key: &[Secret<BooleanGmw, usize>],
    chunk: &[Secret<BooleanGmw, usize>],
    use_sc: bool,
) -> Vec<Secret<BooleanGmw, usize>> {
    static AES_CIRC: Lazy<SharedCircuit<BooleanGate, usize>> = Lazy::new(|| {
        BaseCircuit::load_bristol(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test_resources/bristol-circuits/AES-non-expanded.txt"),
            Load::SubCircuit,
        )
        .expect("Loading Aes circuit")
        .into_shared()
    });

    let inp = [chunk, key].concat();

    if use_sc {
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
    }
}

fn validate(
    plain_iv: [usize; 2],
    plain_key: [usize; 2],
    plaintext: &[u8],
    ciphertext: &BitSlice<usize>,
) -> Result<()> {
    type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;
    let plain_key: [u8; 16] = bytemuck::cast(plain_key);
    let plain_iv: [u8; 16] = bytemuck::cast(plain_iv);
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
        .with_writer(io::stderr)
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
}
