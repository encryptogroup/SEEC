use std::fs::File;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Instant;
use std::{fs, ops};

use base64::prelude::{Engine, BASE64_STANDARD};
use clap::Parser;
use rand::rngs::OsRng;
use serde::Deserialize;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

use seec::circuit::builder::CircuitBuilder;
use seec::circuit::{ExecutableCircuit, GateId};
use seec::common::BitVec;
use seec::executor::{BoolGmwExecutor, Input, Message};
use seec::mul_triple::boolean::ot_ext::OtMTProvider;
use seec::protocols::boolean_gmw::BooleanGmw;
use seec::secret::{inputs, low_depth_reduce, Secret};
use seec::sub_circuit;
use seec_channel::sub_channels_for;
use zappot::ot_ext;

#[derive(Parser, Debug)]
struct Args {
    #[clap(long)]
    my_id: usize,
    #[clap(long)]
    server: SocketAddr,
    #[clap(long)]
    query_file_path: PathBuf,
    #[clap(long)]
    mail_dir_path: PathBuf,
    /// Save the circuit to privmail.dot
    #[clap(long)]
    save_circuit: bool,
    #[clap(long, default_value_t = 1)]
    duplication_factor: usize,
}

#[derive(Deserialize)]
struct SearchQuery {
    keywords: Vec<SearchQueryKeywords>,
    modifier_chain_share: String,
}

#[derive(Deserialize)]
struct SearchQueryKeywords {
    keyword_truncated: String,
}

#[derive(Deserialize)]
struct Mail {
    secret_share_truncated_block: String,
}

fn priv_mail_search(
    search_queries: &[SearchQueryKeywords],
    modifier_chain_share: &str,
    mails: &[Mail],
    duplication_factor: usize,
) -> (BitVec<usize>, Vec<GateId>) {
    debug!(%modifier_chain_share);
    let (mut input, modifier_chain_input) = base64_string_to_input(modifier_chain_share, 1);
    let modifier_chain_share_input: Vec<_> = modifier_chain_input.into_iter().flatten().collect();

    // Decode and initialize the search keywords
    let search_keywords: Vec<_> = search_queries
        .iter()
        .map(|search_query| {
            debug!(keyword = %search_query.keyword_truncated);
            let (query_input, shares) = base64_string_to_input(&search_query.keyword_truncated, 1);
            input.extend_from_bitslice(&query_input);
            shares
        })
        .collect();
    assert!(modifier_chain_share_input.len() >= 2 * search_keywords.len() - 1);

    // Decode and initialize the target text
    let target_texts: Vec<_> = mails
        .iter()
        .map(|mail| {
            debug!(target_text = %mail.secret_share_truncated_block);
            let (mail_input, shares) =
                base64_string_to_input(&mail.secret_share_truncated_block, duplication_factor);
            input.extend_from_bitslice(&mail_input);
            shares
        })
        .collect();

    // Search with the keywords over the target texts
    let mut search_results = Vec::with_capacity(target_texts.len());
    for (j, keyword) in search_keywords.iter().enumerate() {
        for (i, target_text) in target_texts.iter().enumerate() {
            let search_result_per_mail = create_search_circuit(keyword, target_text);
            if j == 0 {
                search_results.push(search_result_per_mail ^ &modifier_chain_share_input[0]);
            } else {
                search_results[i] = create_chaining_circuit(
                    &search_results[i],
                    &search_result_per_mail,
                    &modifier_chain_share_input[2 * j - 1],
                    &modifier_chain_share_input[2 * j],
                );
            }
        }
    }
    let out_ids = search_results
        .into_iter()
        .map(Secret::into_output)
        .collect();
    (input, out_ids)
}

fn create_chaining_circuit(
    previous_search_result: &Secret,
    new_search_result: &Secret,
    or_bit: &Secret,
    not_bit: &Secret,
) -> Secret {
    ((previous_search_result.clone() ^ or_bit) & ((new_search_result.clone() ^ not_bit) ^ or_bit))
        ^ or_bit
}

fn create_search_circuit(keyword: &[[Secret; 8]], target_text: &[[Secret; 8]]) -> Secret {
    /*
     * Calculate the number of positions we need to compare. E.g., if search_keyword
     * is "key" and target_text is "target", we must do 4 comparison:
     *
     * target , target , target , target
     * ^^^       ^^^       ^^^       ^^^
     * key       key       key       key
     */
    let result_bits: Vec<_> = target_text
        .windows(keyword.len())
        .map(|target| comparison_circuit(keyword, target))
        .collect();
    // Finally, use OR tree to get the final answer of whether any of the comparisons was a match
    debug!("OR reduce for {} bits", result_bits.len());
    or_sc(&result_bits)
}

#[sub_circuit]
fn comparison_circuit(keyword: &[[Secret; 8]], target_text: &[[Secret; 8]]) -> Secret {
    const CHARACTER_BIT_LEN: usize = 6; // Follows from the special PrivMail encoding
    let splitted_keyword: Vec<_> = keyword
        .iter()
        .flat_map(|c| c.iter().take(CHARACTER_BIT_LEN).cloned())
        .collect();
    let splitted_text: Vec<_> = target_text
        .iter()
        .flat_map(|c| c.iter().take(CHARACTER_BIT_LEN).cloned())
        .collect();

    let res: Vec<_> = splitted_keyword
        .into_iter()
        .zip(splitted_text)
        .map(|(k, t)| !(k ^ t))
        .collect();

    low_depth_reduce(res, ops::BitAnd::bitand).expect("Empty input")
}

#[sub_circuit]
fn or_sc(input: &[Secret]) -> Secret {
    low_depth_reduce(input.to_owned(), ops::BitOr::bitor)
        .unwrap_or_else(|| Secret::from_const(0, false))
}

fn base64_string_to_input(
    input: &str,
    duplication_factor: usize,
) -> (BitVec<usize>, Vec<[Secret; 8]>) {
    let decoded = BASE64_STANDARD.decode(input).expect("Decode base64 input");
    let duplicated = decoded.repeat(duplication_factor);
    let shares = (0..duplicated.len())
        .map(|_| inputs(8).try_into().unwrap())
        .collect();
    let input = BitVec::from_vec(duplicated);
    let input = BitVec::from_iter(input);
    (input, shares)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let search_query: SearchQuery =
        serde_yaml::from_reader(File::open(args.query_file_path).expect("Opening query file"))
            .expect("Deserializing query file");
    let mails: Vec<Mail> = fs::read_dir(args.mail_dir_path)
        .expect("Reading mail dir")
        .map(|entry| {
            let entry = entry.expect("Mail dir iteration");
            serde_yaml::from_reader(File::open(entry.path()).expect("Opening mail file"))
                .expect("Deserializing mail file")
        })
        .collect();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let now = Instant::now();
    let (input, _) = priv_mail_search(
        &search_query.keywords,
        &search_query.modifier_chain_share,
        &mails,
        args.duplication_factor,
    );
    let circuit = ExecutableCircuit::DynLayers(CircuitBuilder::global_into_circuit());
    info!("Building circuit took: {}", now.elapsed().as_secs_f32());
    // circuit = circuit.clone().into_base_circuit().into();
    // if args.save_circuit {
    //     bc.save_dot("privmail.dot")?;
    // }
    // dbg!(&circuit);
    let (mut sender, bytes_written, mut receiver, bytes_read) = match args.my_id {
        0 => seec_channel::tcp::listen(args.server).await?,
        1 => seec_channel::tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    let (ch1, mut ch2) = sub_channels_for!(
        &mut sender,
        &mut receiver,
        64,
        seec_channel::Receiver<ot_ext::ExtOTMsg>,
        Message<BooleanGmw>
    )
    .await?;

    let mut executor = {
        let mt_provider = OtMTProvider::new(
            OsRng,
            ot_ext::Sender::default(),
            ot_ext::Receiver::default(),
            ch1.0,
            ch1.1,
        );
        BoolGmwExecutor::new(&circuit, args.my_id, mt_provider).await?
    };

    let output = executor
        .execute(Input::Scalar(input), &mut ch2.0, &mut ch2.1)
        .await?;
    info!(
        my_id = %args.my_id,
        output = ?output,
        bytes_written = bytes_written.get(),
        bytes_read = bytes_read.get(),
        gate_count = circuit.gate_count(),
    );
    Ok(())
}
