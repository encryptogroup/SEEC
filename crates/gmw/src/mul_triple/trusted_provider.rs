//! Trusted Multiplication Triple Provider.
//!
//! This module implements a very basic trusted multiplication provider client/server.
//! The [`TrustedMTProviderClient`] is used to connect to a [`TrustedMTProviderServer`]. When
//! [`MTProvider::request_mts`] is called on the client, a request is sent to the server. Upon
//! receiving it, the server generates random multiplication triples by calling
//! [`MulTriples::random_pair`] and returns one [`MulTriples`] struct to each party.
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::io;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Sink, StreamExt, TryStream};
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use tokio::net::ToSocketAddrs;
use tokio::sync::Mutex;
use tracing::error;

use crate::errors::MTProviderError;
use crate::mul_triple::{MTProvider, MulTriples};
use mpc_channel::{Channel, Tcp};

pub struct TrustedMTProviderClient<T> {
    id: String,
    channel: T,
}

#[derive(Clone)]
pub struct TrustedMTProviderServer<T> {
    channel: T,
    mts: Arc<Mutex<HashMap<String, MulTriples>>>,
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    RequestTriples { id: String, amount: usize },
    MulTriples(MulTriples),
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
            Message::MulTriples(mts) => Ok(mts),
            _ => Err(Self::Error::IllegalMessage),
        }
    }
}

impl<C> TrustedMTProviderServer<C> {
    pub fn new(channel: C) -> Self {
        Self {
            channel,
            mts: Default::default(),
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
        let mut mts = self.mts.lock().await;
        let mt = match mts.entry(id) {
            Entry::Vacant(vacant) => {
                // TODO `random` call might be blocking, better use rayon here
                //  Note: It's fine for the moment, as the Server is not really able to utilize
                //  parallelization anyway, due to the lock
                let [mt1, mt2] = MulTriples::random_pair(amount, &mut thread_rng());
                vacant.insert(mt1);
                mt2
            }
            Entry::Occupied(occupied) => occupied.remove(),
        };
        self.channel.send(Message::MulTriples(mt)).await?;
        Ok(())
    }

    #[tracing::instrument(skip(self))]
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
                    mts: data,
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
