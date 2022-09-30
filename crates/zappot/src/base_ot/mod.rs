//! Chou Orlandi base OT protocol.
use crate::traits::{BaseROTReceiver, BaseROTSender, Error, ProtocolError};
use crate::util::Block;
use crate::{DefaultRom, Rom128};
use async_trait::async_trait;
use bitvec::macros::internal::funty::Fundamental;
use bitvec::slice::BitSlice;
use blake2::digest::Output;
use blake2::Digest;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use futures::{SinkExt, StreamExt, TryStreamExt};
use mpc_channel::Channel;
use rand::{CryptoRng, Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Debug, Default, Clone)]
pub struct Sender;

#[derive(Debug, Default, Clone)]
pub struct Receiver;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BaseOTMsg {
    First(RistrettoPoint, Output<DefaultRom>),
    Second(Vec<RistrettoPoint>),
    Third(Block),
}

impl Sender {
    pub fn new() -> Self {
        Sender
    }
}

impl Receiver {
    pub fn new() -> Self {
        Receiver
    }
}

#[async_trait]
impl BaseROTSender for Sender {
    type Msg = BaseOTMsg;

    #[allow(non_snake_case)]
    async fn send_random<RNG, CH>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<[Block; 2]>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Unpin + Send,
    {
        let (stream, sink) = channel.split_mut();
        let a = Scalar::random(rng);
        let mut A = &RISTRETTO_BASEPOINT_TABLE * &a;
        let seed: Block = rng.gen();
        // TODO: libOTE uses fixedKeyAES hash here, using Blake should be fine and not really
        //  impact performance
        let seed_comm = seed.rom_hash();
        sink.send(BaseOTMsg::First(A, seed_comm))
            .await
            .map_err(Error::Send)?;
        let msg = stream
            .try_next()
            .await
            .map_err(Error::Receive)?
            .ok_or(Error::UnexpectedTermination)?;
        let points = match msg {
            BaseOTMsg::Second(points) => points,
            msg => return Err(Error::WrongOrder(msg)),
        };
        if count != points.len() {
            return Err(Error::UnexpectedTermination);
        }
        sink.send(BaseOTMsg::Third(seed))
            .await
            .map_err(Error::Send)?;
        A *= a;
        let ots = points
            .into_iter()
            .enumerate()
            .map(|(i, mut B)| {
                B *= a;
                let k0 = rom_hash_point(&B, i, seed);
                B -= A;
                let k1 = rom_hash_point(&B, i, seed);
                [k0, k1]
            })
            .collect();
        Ok(ots)
    }
}

#[async_trait]
impl BaseROTReceiver for Receiver {
    type Msg = BaseOTMsg;

    #[allow(non_snake_case)]
    async fn receive_random<RNG, CH>(
        &mut self,
        choices: &BitSlice,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<Block>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Unpin + Send,
    {
        let (stream, sink) = channel.split_mut();
        let msg = stream
            .try_next()
            .await
            .map_err(Error::Receive)?
            .ok_or(Error::UnexpectedTermination)?;
        let (A, comm) = match msg {
            BaseOTMsg::First(A, comm) => (A, comm),
            msg => return Err(Error::WrongOrder(msg)),
        };
        let (bs, Bs): (Vec<_>, Vec<_>) = choices
            .iter()
            .map(|choice| {
                let b = Scalar::random(rng);
                let B_0 = &RISTRETTO_BASEPOINT_TABLE * &b;
                let B = [B_0, A + B_0];
                (b, B[choice.as_usize()])
            })
            .unzip();
        sink.send(BaseOTMsg::Second(Bs))
            .await
            .map_err(Error::Send)?;
        let msg = stream
            .try_next()
            .await
            .map_err(Error::Receive)?
            .ok_or(Error::UnexpectedTermination)?;
        let seed = match msg {
            BaseOTMsg::Third(seed) => seed,
            msg => return Err(Error::WrongOrder(msg)),
        };
        if comm != seed.rom_hash() {
            return Err(Error::ProtocolDeviation);
        }
        let ots = bs
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                let B = A * b;
                rom_hash_point(&B, i, seed)
            })
            .collect();
        Ok(ots)
    }
}

/// Hash a point and counter using the ROM.
fn rom_hash_point(point: &RistrettoPoint, counter: usize, seed: Block) -> Block {
    let mut rom = Rom128::new();
    rom.update(point.compress().as_bytes());
    rom.update(&counter.to_le_bytes());
    rom.update(&seed.to_le_bytes());
    let out = rom.finalize();
    Block::from_le_bytes(out.into())
}

#[cfg(test)]
mod tests {
    use crate::base_ot::{Receiver, Sender};
    use crate::traits::{BaseROTReceiver, BaseROTSender};
    use bitvec::bitvec;
    use mpc_channel::in_memory::InMemory;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[tokio::test]
    async fn base_rot() {
        let (mut ch1, mut ch2) = InMemory::new_pair();
        let mut rng_send = StdRng::seed_from_u64(42);
        let mut rng_recv = StdRng::seed_from_u64(42 * 42);
        let mut sender = Sender;
        let mut receiver = Receiver;
        let send = sender.send_random(128, &mut rng_send, &mut ch1);
        let choices = bitvec![0;128];
        let receive = receiver.receive_random(&choices, &mut rng_recv, &mut ch2);

        let (sender_out, receiver_out) = tokio::try_join!(send, receive).unwrap();
        for (recv, [send, _]) in receiver_out.into_iter().zip(sender_out) {
            assert_eq!(recv, send);
        }
    }
}
