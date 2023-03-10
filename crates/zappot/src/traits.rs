//! Oblivious transfer traits.
use crate::util::Block;
use async_trait::async_trait;
use bitvec::slice::BitSlice;
use bitvec::store::BitStore;
use bytemuck::Pod;
use rand::{CryptoRng, RngCore};
use remoc::rch::mpsc::{RecvError, SendError};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<Msg> {
    #[error("Error sending value")]
    Send(#[from] SendError<Msg>),
    #[error("Error receiving value")]
    Receive(#[from] RecvError),
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
    async fn send_random<RNG>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[Block; 2]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send;
}

/// Receiver of base random OTs.
#[async_trait]
pub trait BaseROTReceiver {
    type Msg;

    /// Receive `count` number of random OTs via the provided channel.
    async fn receive_random<RNG>(
        &mut self,
        choices: &BitSlice,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send;
}

/// OT extension sender.
#[async_trait]
pub trait ExtROTSender {
    type Msg;

    async fn send_random<RNG>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[Block; 2]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send;

    async fn send_correlated<RNG>(
        &mut self,
        count: usize,
        correlation: impl Fn(usize, Block) -> Block + Send,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send;

    async fn send_correlated_bytes<const LEN: usize, RNG>(
        &mut self,
        count: usize,
        correlation: impl Fn(usize, [u8; LEN]) -> [u8; LEN] + Send,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[u8; LEN]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        [u8; LEN]: Pod;
}

/// OT extension receiver.
#[async_trait]
pub trait ExtROTReceiver {
    type Msg;

    async fn receive_random<C, RNG>(
        &mut self,
        choices: &BitSlice<C>,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        C: Pod + BitStore + Sync,
        <C as BitStore>::Unalias: Pod;

    async fn receive_correlated<C, RNG>(
        &mut self,
        choices: &BitSlice<C>,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        C: Pod + BitStore + Sync,
        <C as BitStore>::Unalias: Pod;

    async fn receive_correlated_bytes<const LEN: usize, C, RNG>(
        &mut self,
        choices: &BitSlice<C>,
        rng: &mut RNG,
        sender: &mpc_channel::Sender<Self::Msg>,
        receiver: &mut mpc_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[u8; LEN]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        [u8; LEN]: Pod,
        C: Pod + BitStore + Sync,
        <C as BitStore>::Unalias: Pod;
}
