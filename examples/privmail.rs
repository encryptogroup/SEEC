use clap::Parser;
use gmw_rs::circuit::{Circuit, GateId};
use gmw_rs::common::BitVec;
use gmw_rs::executor::Executor;
use gmw_rs::mul_triple::insecure_provider::InsecureMTProvider;
use gmw_rs::share_wrapper::{inputs, low_depth_reduce, ShareWrapper};
use gmw_rs::transport::Tcp;
use serde::Deserialize;
use std::cell::RefCell;
use std::fs::File;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::rc::Rc;
use std::{fs, ops};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

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
}

type SharedCircuit = Rc<RefCell<Circuit>>;

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
    circuit: SharedCircuit,
    search_queries: &[SearchQueryKeywords],
    modifier_chain_share: &str,
    mails: &[Mail],
) -> (BitVec, Vec<GateId>) {
    debug!(%modifier_chain_share);
    let (mut input, modifier_chain_input) =
        base64_string_to_input(circuit.clone(), modifier_chain_share);
    let modifier_chain_share_input: Vec<_> = modifier_chain_input.into_iter().flatten().collect();

    // Decode and initialize the search keywords
    let search_keywords: Vec<_> = search_queries
        .iter()
        .map(|search_query| {
            debug!(keyword = %search_query.keyword_truncated);
            let (query_input, shares) =
                base64_string_to_input(circuit.clone(), &search_query.keyword_truncated);
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
                base64_string_to_input(circuit.clone(), &mail.secret_share_truncated_block);
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
        .map(ShareWrapper::output)
        .collect();
    (input, out_ids)
}

fn create_chaining_circuit(
    previous_search_result: &ShareWrapper,
    new_search_result: &ShareWrapper,
    or_bit: &ShareWrapper,
    not_bit: &ShareWrapper,
) -> ShareWrapper {
    ((previous_search_result.clone() ^ or_bit) & ((new_search_result.clone() ^ not_bit) ^ or_bit))
        ^ or_bit
}

fn create_search_circuit(
    keyword: &[[ShareWrapper; 8]],
    target_text: &[[ShareWrapper; 8]],
) -> ShareWrapper {
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

    // Combine the bits (basically a zip (transpose?) operation: [[a,b],[c,d]] to [[a,c],[b,d]])
    let mut xor_combined = vec![vec![]; search_results_per_position[0].len()];
    for xor_position_splitted in search_results_per_position {
        assert_eq!(xor_position_splitted.len(), xor_combined.len());
        xor_combined
            .iter_mut()
            .zip(xor_position_splitted)
            .for_each(|(v, xor_pos)| v.push(xor_pos));
    }

    let result_bits: Vec<_> = xor_combined
        .into_iter()
        .map(|combined| {
            debug!("AND reduce for {} bits", combined.len());
            low_depth_reduce(&combined, ops::BitAnd::bitand).expect("Reduce returned None")
        })
        .collect();
    // Finally, use OR tree to get the final answer of whether any of the comparisons was a match
    low_depth_reduce(&result_bits, ops::BitOr::bitor).expect("Reduce returned None")
}

fn create_comparison_circuit(
    keyword: &[[ShareWrapper; 8]],
    target_text: &[[ShareWrapper; 8]],
    text_position: usize,
) -> Vec<ShareWrapper> {
    const CHARACTER_BIT_LEN: usize = 6; // Follows from the special PrivMail encoding
    let splitted_keyword: Vec<_> = keyword
        .iter()
        .map(|c| c.iter().cloned().take(CHARACTER_BIT_LEN))
        .flatten()
        .collect();
    let splitted_text: Vec<_> = target_text[text_position..text_position + keyword.len()]
        .iter()
        .map(|c| c.iter().cloned().take(CHARACTER_BIT_LEN))
        .flatten()
        .collect();

    let mut res = splitted_keyword;
    xor_inplace(&mut res, &splitted_text);
    res.iter_mut().for_each(|c| *c = !&*c);
    res
}

fn base64_string_to_input(circuit: SharedCircuit, input: &str) -> (BitVec, Vec<[ShareWrapper; 8]>) {
    let decoded = base64::decode(input).expect("Decode base64 input");
    let shares = (0..decoded.len())
        .map(|_| inputs(circuit.clone(), 8).try_into().unwrap())
        .collect();
    let input = BitVec::from_vec(decoded);
    (input, shares)
}

fn xor_inplace(dst: &mut [ShareWrapper], other: &[ShareWrapper]) {
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

    let circuit = Circuit::new();
    let circuit = Rc::new(RefCell::new(circuit));
    let (input, _) = priv_mail_search(
        circuit.clone(),
        &search_query.keywords,
        &search_query.modifier_chain_share,
        &mails,
    );
    let mut transport = match args.my_id {
        0 => Tcp::listen(args.server).await?,
        1 => Tcp::connect(args.server).await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };
    let circuit = Rc::try_unwrap(circuit).unwrap().into_inner();
    if args.save_circuit {
        circuit.save_dot("privmail.dot")?;
    }

    let mut executor = {
        let mt_provider = InsecureMTProvider::default();
        Executor::new(&circuit, args.my_id, mt_provider).await?
    };

    let output = executor.execute(input, &mut transport).await?;
    info!(
        my_id = %args.my_id,
        output = ?output,
        bytes_written = transport.bytes_written(),
        bytes_read = transport.bytes_read()
    );
    Ok(())
}
