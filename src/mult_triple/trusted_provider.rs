use crate::errors::MTProviderError;
use crate::mult_triple::{MTProvider, MultTriples};
use crate::transport::{Tcp, Transport};
use async_trait::async_trait;
use futures::StreamExt;
use rand::thread_rng;
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

#[derive(Clone)]
pub struct TrustedMTProviderServer<T> {
    transport: T,
    mts: Arc<Mutex<HashMap<String, MultTriples>>>,
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    RequestTriples { id: String, amount: usize },
    MultTriples(MultTriples),
}

impl<T> TrustedMTProviderClient<T> {
    pub fn new(id: String, transport: T) -> Self {
        Self { id, transport }
    }
}

#[async_trait]
impl<T: Transport<Message> + Send> MTProvider for TrustedMTProviderClient<T> {
    type Error = MTProviderError<T::StreamError, T::SinkError>;
    async fn request_mts(&mut self, amount: usize) -> Result<MultTriples, Self::Error> {
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
            Message::MultTriples(mts) => Ok(mts),
            _ => Err(Self::Error::IllegalMessage),
        }
    }
}

impl<T> TrustedMTProviderServer<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            mts: Default::default(),
        }
    }
}

impl<T: Transport<Message> + Send> TrustedMTProviderServer<T> {
    #[tracing::instrument(skip(self), err(Debug))]
    async fn handle_request(&mut self, id: String, amount: usize) -> Result<(), T::SinkError>
    where
        T::SinkError: Debug,
    {
        let mut mts = self.mts.lock().await;
        let mt = match mts.entry(id) {
            Entry::Vacant(vacant) => {
                // TODO `random` call might be blocking, better use rayon here
                //  Note: It's fine for the moment, as the Server is not really able to utilize
                //  parallelization anyway, due to the lock
                let [mt1, mt2] = MultTriples::random(amount, &mut thread_rng());
                vacant.insert(mt1);
                mt2
            }
            Entry::Occupied(occupied) => occupied.remove(),
        };
        self.transport.send(Message::MultTriples(mt)).await?;
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
                    mts: data,
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
