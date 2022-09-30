//! Trusted Seed MT Provider.
//!
//! This module implements a trusted third party MT provider according to the
//! [Chameleon paper](https://dl.acm.org/doi/pdf/10.1145/3196494.3196522). The third party
//! generates two random seeds and derives the MTs from them. The first party gets the first seed,
//! while the second party receives the second seed and the `c` values for their MTs.  
//! For a visualization of the protocol, look at Figure 1 of the linked paper.
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Sink, StreamExt, TryStream};
use rand::{random, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use tokio::net::ToSocketAddrs;
use tokio::sync::Mutex;
use tracing::error;

use crate::common::BitVec;
use crate::errors::MTProviderError;
use crate::mul_triple::{compute_c_owned, rand_bitvecs, MTProvider, MulTriples};
use mpc_channel::{Channel, Tcp};

pub struct TrustedMTProviderClient<T> {
    id: String,
    channel: T,
}

// TODO: Which prng to choose? Context: https://github.com/rust-random/rand/issues/932
//  using ChaCha with 8 rounds is likely to be secure enough and would provide a little more
//  performance. This should be benchmarked however
type MtRng = ChaCha12Rng;
pub type MtRngSeed = <MtRng as SeedableRng>::Seed;

#[derive(Clone)]
pub struct TrustedMTProviderServer<T> {
    channel: T,
    seeds: Arc<Mutex<HashMap<String, MtRngSeed>>>,
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    RequestTriples { id: String, amount: usize },
    Seed(MtRngSeed),
    SeedAndC { seed: MtRngSeed, c: BitVec },
}

impl<T> TrustedMTProviderClient<T> {
    pub fn new(id: String, channel: T) -> Self {
        Self { id, channel }
    }
}

#[async_trait]
impl<C: Channel<Message> + Send> MTProvider for TrustedMTProviderClient<C> {
    type Error = MTProviderError<<C as TryStream>::Error, <C as Sink<Message>>::Error>;

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        self.channel
            .send(Message::RequestTriples {
                id: self.id.clone(),
                amount,
            })
            .await
            .map_err(Self::Error::RequestFailed)?;
        let msg: Message = self
            .channel
            .try_next()
            .await
            .map_err(|err| Self::Error::ReceiveFailed(Some(err)))?
            .ok_or(Self::Error::ReceiveFailed(None))?;
        match msg {
            Message::Seed(seed) => {
                let mut rng = MtRng::from_seed(seed);
                Ok(MulTriples::random(amount, &mut rng))
            }
            Message::SeedAndC { seed, c } => {
                let mut rng = MtRng::from_seed(seed);
                Ok(MulTriples::random_with_fixed_c(c, &mut rng))
            }
            _ => Err(Self::Error::IllegalMessage),
        }
    }
}

impl<C> TrustedMTProviderServer<C> {
    pub fn new(channel: C) -> Self {
        Self {
            channel,
            seeds: Default::default(),
        }
    }
}

impl<C> TrustedMTProviderServer<C>
where
    C: Channel<Message> + Send,
    <C as Sink<Message>>::Error: Error,
    <C as TryStream>::Error: Error,
{
    #[tracing::instrument(skip(self), err(Debug))]
    async fn handle_request(
        &mut self,
        id: String,
        amount: usize,
    ) -> Result<(), <C as Sink<Message>>::Error> {
        let mut seeds = self.seeds.lock().await;
        match seeds.entry(id) {
            Entry::Vacant(vacant) => {
                let seed1 = random();
                let seed2 = random();
                vacant.insert(seed1);
                let mut rng1 = MtRng::from_seed(seed1);
                let mut rng2 = MtRng::from_seed(seed2);
                let mts = MulTriples::random(amount, &mut rng1);
                let [a, b] = rand_bitvecs(amount, &mut rng2);
                let c = compute_c_owned(mts, a, b);
                self.channel
                    .send(Message::SeedAndC { seed: seed2, c })
                    .await?;
            }
            Entry::Occupied(occupied) => {
                let seed = occupied.remove();
                self.channel.send(Message::Seed(seed)).await?;
            }
        };
        Ok(())
    }

    async fn handle_conn(mut self) {
        loop {
            match self.channel.try_next().await {
                Ok(Some(Message::RequestTriples { id, amount })) => {
                    let _ = self.handle_request(id, amount).await;
                }
                Ok(None) => break,
                Ok(_other) => error!("Server received illegal msg"),
                Err(err) => {
                    error!(%err, "Error handling connection");
                }
            }
        }
    }
}

impl TrustedMTProviderServer<Tcp<Message>> {
    pub async fn start(addr: impl ToSocketAddrs + Debug) -> Result<(), io::Error> {
        let data = Default::default();
        Tcp::server(addr)
            .await?
            .for_each(|conn| async {
                let data = Arc::clone(&data);
                let mt_server = Self {
                    seeds: data,
                    channel: conn,
                };
                tokio::spawn(async {
                    mt_server.handle_conn().await;
                });
            })
            .await;
        Ok(())
    }
}
