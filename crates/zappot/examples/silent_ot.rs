//! Example - SilentOT extension
//!
//! This example shows how to use the zappot crate to execute the SilentOT extension protocol
//! to generate random OTs.
use bitvec::vec::BitVec;
use clap::Parser;
use rand_core::OsRng;
use std::time::Duration;
use tokio::time::Instant;
use zappot::base_ot;
use zappot::silent_ot::{Receiver, Sender};
use zappot::util::Block;

#[derive(Parser, Debug, Clone)]
struct Args {
    /// Number of OTs to execute
    #[clap(short, long, default_value_t = 1000)]
    num_ots: usize,
    #[clap(long, default_value_t = 2)]
    scaler: usize,
    /// Number of threads per party
    #[clap(short, long, default_value_t = 1)]
    threads: usize,
    /// The port to bind to on localhost
    #[clap(short, long, default_value_t = 8066)]
    port: u16,
}

/// Example of the sender side
async fn sender(args: Args) -> (Vec<[Block; 2]>, usize, usize) {
    // Create a secure RNG to use in the protocol
    let mut rng = OsRng::default();
    // Create the ot extension sender. A base OT **receiver** is passed as an argument and used
    // to create the base_ots
    // Create a channel by listening on a socket address. Once another party connect, this
    // returns the channel
    let (mut ch_sender, bytes_sent, mut ch_receiver, bytes_rcv) =
        mpc_channel::tcp::listen(("127.0.0.1", args.port), 128)
            .await
            .expect("Error listening for channel connection");
    let base_sender = base_ot::Sender;
    let sender = Sender::new_with_base_ot_sender(
        base_sender,
        &mut rng,
        args.num_ots,
        args.scaler,
        args.threads,
        &mut ch_sender,
        &mut ch_receiver,
    )
    .await;
    // Perform the random ots
    let ots = sender
        .random_silent_send(&mut rng, ch_sender, ch_receiver)
        .await;
    (ots, bytes_sent.get(), bytes_rcv.get())
}

/// Example of the receiver side
async fn receiver(args: Args) -> (Vec<Block>, BitVec) {
    // Create a secure RNG to use in the protocol
    let mut rng = OsRng::default();
    let (mut ch_sender, _, mut ch_receiver, _) =
        mpc_channel::tcp::connect(("127.0.0.1", args.port), 128)
            .await
            .expect("Error listening for channel connection");
    // Create the SilentOT extension receiver. A base OT receiver is passed as an argument and used
    // to create the base_ots
    let base_receiver = base_ot::Receiver;
    let receiver = Receiver::new_with_base_ot_receiver(
        base_receiver,
        &mut rng,
        args.num_ots,
        args.scaler,
        args.threads,
        &mut ch_sender,
        &mut ch_receiver,
    )
    .await;

    // Perform the random ot extension
    receiver.random_silent_receive(ch_sender, ch_receiver).await
}

#[tokio::main]
async fn main() {
    let args: Args = Args::parse();
    let now = Instant::now();
    // Spawn the sender future
    let sender_fut = tokio::spawn(sender(args.clone()));
    // Ensure that the sender is listening for connections, in a real setting, the receiver
    // might try to reconnect if the sender is not listening yet
    tokio::time::sleep(Duration::from_millis(50)).await;
    // Spawn the receiver future
    let (receiver_ots, choices) = tokio::spawn(receiver(args.clone()))
        .await
        .expect("Error await receiver");
    let (sender_ots, bytes_sent, bytes_recv) = sender_fut.await.expect("Error awaiting sender");

    println!(
        "Executed {} ots in {} ms. Sent bytes: {}, Recv bytes: {}",
        args.num_ots,
        now.elapsed().as_millis(),
        bytes_sent,
        bytes_recv
    );

    // Assert that the random OTs have been generated correctly
    for ((recv, choice), [send1, send2]) in receiver_ots.into_iter().zip(choices).zip(sender_ots) {
        let [chosen, not_chosen] = if choice {
            [send2, send1]
        } else {
            [send1, send2]
        };
        assert_eq!(recv, chosen);
        assert_ne!(recv, not_chosen);
    }
}
