//! Trusted Seed MT Provider.
//!
//! This module implements a trusted third party MT provider according to the
//! [Chameleon paper](https://dl.acm.org/doi/pdf/10.1145/3196494.3196522). The third party
//! generates two random seeds and derives the MTs from them. The first party gets the first seed,
//! while the second party receives the second seed and the `c` values for their MTs.  
//! For a visualization of the protocol, look at Figure 1 of the linked paper.
use crate::common::BitVec;
use crate::errors::MTProviderError;
use crate::mul_triple::{compute_c_owned, rand_bitvecs, MTProvider, MulTriples};
use crate::transport::{Tcp, Transport};
use async_trait::async_trait;
use futures::StreamExt;
use rand::{random, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::sync::Arc;
use tokio::net::ToSocketAddrs;
use tokio::sync::Mutex;

pub struct TrustedMTProviderClient<T> {
    id: String,
    transport: T,
}

// TODO: Which prng to choose? Context: https://github.com/rust-random/rand/issues/932
//  using ChaCha with 8 rounds is likely to be secure enough and would provide a little more
//  performance. This should be benchmarked however
type MtRng = ChaCha12Rng;
pub type MtRngSeed = <MtRng as SeedableRng>::Seed;

#[derive(Clone)]
pub struct TrustedMTProviderServer<T> {
    transport: T,
    seeds: Arc<Mutex<HashMap<String, MtRngSeed>>>,
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    RequestTriples { id: String, amount: usize },
    Seed(MtRngSeed),
    SeedAndC { seed: MtRngSeed, c: BitVec },
}

impl<T> TrustedMTProviderClient<T> {
    pub fn new(id: String, transport: T) -> Self {
        Self { id, transport }
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }
}

#[async_trait]
impl<T: Transport<Message> + Send> MTProvider for TrustedMTProviderClient<T> {
    type Error = MTProviderError<T::StreamError, T::SinkError>;
    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
        self.transport
            .send(Message::RequestTriples {
                id: self.id.clone(),
                amount,
            })
            .await
            .map_err(Self::Error::RequestFailed)?;
        let msg: Message = self
            .transport
            .next()
            .await
            .ok_or(Self::Error::ReceiveFailed(None))?
            .map_err(|err| Self::Error::ReceiveFailed(Some(err)))?;
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

impl<T> TrustedMTProviderServer<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            seeds: Default::default(),
        }
    }
}

impl<T: Transport<Message> + Send> TrustedMTProviderServer<T> {
    #[tracing::instrument(skip(self), err(Debug))]
    async fn handle_request(&mut self, id: String, amount: usize) -> Result<(), T::SinkError>
    where
        T::SinkError: Debug,
    {
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
                self.transport
                    .send(Message::SeedAndC { seed: seed2, c })
                    .await?;
            }
            Entry::Occupied(occupied) => {
                let seed = occupied.remove();
                self.transport.send(Message::Seed(seed)).await?;
            }
        };
        Ok(())
    }

    async fn handle_conn(mut self)
    where
        T::SinkError: Debug,
    {
        while let Some(msg) = self.transport.next().await {
            if let Ok(Message::RequestTriples { id, amount }) = msg {
                let _ = self.handle_request(id, amount).await;
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
                    transport: conn,
                };
                tokio::spawn(async {
                    mt_server.handle_conn().await;
                });
            })
            .await;
        Ok(())
    }
}
