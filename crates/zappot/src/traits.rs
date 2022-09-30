//! Oblivious transfer traits.
use crate::util::Block;
use async_trait::async_trait;
use bitvec::slice::BitSlice;
use futures::{Sink, TryStream};
use mpc_channel::Channel;
use rand::{CryptoRng, RngCore};
use std::fmt::{Debug, Formatter};
use thiserror::Error;

pub type ProtocolError<Msg, Ch> = Error<Msg, <Ch as Sink<Msg>>::Error, <Ch as TryStream>::Error>;

#[derive(Error, Debug)]
pub enum Error<Msg, SinkErr, StreamErr> {
    #[error("Error sending value")]
    Send(SinkErr),
    #[error("Error receiving value")]
    Receive(StreamErr),
    #[error("Received out of order message")]
    WrongOrder(Msg),
    #[error("The other party terminated the protocol")]
    UnexpectedTermination,
    #[error("The other party deviated from the protocol")]
    ProtocolDeviation,
    #[error("Error in base OT execution")]
    BaseOT(Box<dyn std::error::Error + Send>),
}

/// Sender of base random OTs.
#[async_trait]
pub trait BaseROTSender {
    type Msg;

    /// Send `count` number of random OTs via the provided channel.
    async fn send_random<RNG, CH>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<[Block; 2]>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Unpin + Send;
}

/// Receiver of base random OTs.
#[async_trait]
pub trait BaseROTReceiver {
    type Msg;

    /// Receive `count` number of random OTs via the provided channel.
    async fn receive_random<RNG, CH>(
        &mut self,
        choices: &BitSlice,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<Block>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Unpin + Send;
}

/// OT extension sender.
#[async_trait]
pub trait ExtROTSender {
    type Msg;

    async fn send_random<RNG, CH>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<[Block; 2]>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Unpin + Send;
}

/// OT extension receiver.
#[async_trait]
pub trait ExtROTReceiver {
    type Msg;

    async fn receive_random<RNG, CH>(
        &mut self,
        choices: &BitSlice,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<Block>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Unpin + Send;
}

// impl<Msg, Ch: Channel<Msg>> Debug for ProtocolError<Msg, Ch> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         f.debug_tuple("test").finish()
//     }
// }
