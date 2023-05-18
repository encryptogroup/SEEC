use std::fs::File;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::{fs, ops};

use clap::Parser;
use serde::Deserialize;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

use gmw::circuit::ExecutableCircuit;
use gmw::circuit::GateId;
use gmw::common::BitVec;
use gmw::executor::{BoolGmwExecutor, Input, Message};
use gmw::mul_triple::boolean::insecure_provider::InsecureMTProvider;
use gmw::protocols::boolean_gmw::BooleanGmw;
use gmw::secret::{inputs, low_depth_reduce, Secret};
use gmw::CircuitBuilder;
use mpc_channel::sub_channels_for;

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
            let mut search_result_per_mail = Secret::from_const(0, false);
            if keyword.len() <= target_text.len() {
                search_result_per_mail = create_search_circuit(keyword, target_text);
            }
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
    let num_positions = target_text.len() - keyword.len() + 1;
    let search_results_per_position: Vec<_> = (0..num_positions)
        .map(|k| create_comparison_circuit(keyword, target_text, k))
        .collect();

    let result_bits = search_results_per_position.into_iter().map(|result| {
        debug!("AND reduce for {} bits", result.len());
        low_depth_reduce(result, ops::BitAnd::bitand).expect("Reduce returned None")
    });
    // Finally, use OR tree to get the final answer of whether any of the comparisons was a match
    low_depth_reduce(result_bits, ops::BitOr::bitor).expect("Reduce returned None")
}

fn create_comparison_circuit(
    keyword: &[[Secret; 8]],
    target_text: &[[Secret; 8]],
    text_position: usize,
) -> Vec<Secret> {
    const CHARACTER_BIT_LEN: usize = 6; // Follows from the special PrivMail encoding
    let splitted_keyword: Vec<_> = keyword
        .iter()
        .flat_map(|c| c.iter().take(CHARACTER_BIT_LEN).cloned())
        .collect();
    let splitted_text: Vec<_> = target_text[text_position..text_position + keyword.len()]
        .iter()
        .flat_map(|c| c.iter().take(CHARACTER_BIT_LEN).cloned())
        .collect();

    let mut res = splitted_keyword;
    xor_inplace(&mut res, &splitted_text);
    res.iter_mut().for_each(|c| *c = !&*c);
    res
}

fn base64_string_to_input(
    input: &str,
    duplication_factor: usize,
) -> (BitVec<usize>, Vec<[Secret; 8]>) {
    let decoded = base64::decode(input).expect("Decode base64 input");
    let duplicated = decoded.repeat(duplication_factor);
    let shares = (0..duplicated.len())
        .map(|_| inputs(8).try_into().unwrap())
        .collect();
    let input = BitVec::from_vec(duplicated);
    let input = BitVec::from_iter(input);
    (input, shares)
}

fn xor_inplace(dst: &mut [Secret], other: &[Secret]) {
    assert_eq!(
        dst.len(),
        other.len(),
        "xor_inplace must be called with slices of equal length"
    );
    dst.iter_mut().zip(other).for_each(|(a, b)| {
        *a ^= b;
    })
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

    let (input, _) = priv_mail_search(
        &search_query.keywords,
        &search_query.modifier_chain_share,
        &mails,
        args.duplication_factor,
    );

    let circuit: ExecutableCircuit<_, _> =
        ExecutableCircuit::DynLayers(CircuitBuilder::global_into_circuit());
    // if args.save_circuit {
    //     circuit.save_dot("privmail.dot")?;
    // }

    let (mut sender, bytes_written, mut receiver, bytes_read) = match args.my_id {
        0 => mpc_channel::tcp::listen(args.server).await?,
        1 => mpc_channel::tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    let mut executor = {
        let mt_provider = InsecureMTProvider;
        BoolGmwExecutor::new(&circuit, args.my_id, mt_provider).await?
    };

    let (mut sender, mut receiver) =
        sub_channels_for!(&mut sender, &mut receiver, 16, Message<BooleanGmw>).await?;

    let output = executor
        .execute(Input::Scalar(input), &mut sender, &mut receiver)
        .await?;
    info!(
        my_id = %args.my_id,
        output = ?output,
        bytes_written = bytes_written.get(),
        bytes_read = bytes_read.get(),
        gate_count = circuit.gate_count()
    );
    Ok(())
}
