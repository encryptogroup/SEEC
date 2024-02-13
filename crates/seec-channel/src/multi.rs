use crate::{tcp, BaseReceiver, BaseSender, Receiver, Sender};
use futures::future::join;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};
use futures::{Stream, TryFutureExt};
use remoc::rch::mpsc::SendError;
use remoc::rch::{base, mpsc};
use remoc::RemoteSend;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};
use tokio::task::JoinSet;
use tracing::{debug, error, info, instrument, Instrument};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unable to establish TCP remoc connection")]
    Tcp(#[from] tcp::Error),
    #[error("internal error in multi-party connection establishment")]
    Internal(#[from] tokio::task::JoinError),
    #[error("error when sending initial message")]
    InitialMessageFailed(#[from] base::SendError<()>),
    #[error("missing initial message")]
    MissingInitialMsg,
    #[error("received multiple initial message with equal party id")]
    DuplicateInitialMsg,
    #[error("unable to multi-send message")]
    MultiSend(Vec<mpsc::SendError<()>>),
    #[error("unable to multi-recv message")]
    MultiRecv(Option<mpsc::RecvError>),
    #[error("unknown party id")]
    UnknownParty(u32),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "T: RemoteSend")]
pub struct InitialMsg<T> {
    party_id: u32,
    sender: Sender<T>,
}

#[derive(Debug)]
pub struct MultiSender<T> {
    senders: HashMap<u32, Sender<T>>,
}

#[derive(Debug)]
pub struct MultiReceiver<T> {
    receivers: HashMap<u32, Receiver<T>>,
}

impl<T: RemoteSend + Clone> MultiSender<T> {
    pub async fn send_to(&self, to: impl IntoIterator<Item = &u32>, msg: T) -> Result<(), Error> {
        let mut fu = FuturesUnordered::new();
        for to in to {
            debug!(to, "Sending");
            let sender = self
                .senders
                .get(to)
                .ok_or_else(|| Error::UnknownParty(*to))?;
            fu.push(sender.send(msg.clone()));
        }
        let mut errors = vec![];
        loop {
            match fu.next().await {
                None => break,
                Some(Ok(())) => continue,
                Some(Err(err)) => errors.push(err.without_item()),
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::MultiSend(errors))
        }
    }

    #[instrument(level = "debug", skip(self, msg), ret)]
    pub async fn send_all(&self, msg: T) -> Result<(), Error> {
        self.send_to(self.senders.keys(), msg).await
    }
}

#[derive(Debug, PartialEq)]
pub struct From<T> {
    from: u32,
    msg: T,
}

impl<T: RemoteSend> MultiReceiver<T> {
    pub fn recv_from<'a>(
        &mut self,
        from: &HashSet<u32>,
    ) -> impl Stream<Item = Result<From<T>, Error>> + '_ {
        // this is unfortunately O(|receivers|) instead of O(|from|), but I doubt,
        // that this has a noticeable perf impact
        self.receivers
            .iter_mut()
            .filter(|(id, _)| from.contains(*id))
            .map(map_recv_fut)
            .collect::<FuturesUnordered<_>>()
    }

    pub fn recv_all(&mut self) -> impl Stream<Item = Result<From<T>, Error>> + '_ {
        self.receivers
            .iter_mut()
            .map(map_recv_fut)
            .collect::<FuturesUnordered<_>>()
    }
}

#[inline]
async fn map_recv_fut<T: RemoteSend>(
    (from, receiver): (&u32, &mut Receiver<T>),
) -> Result<From<T>, Error> {
    debug!(from);
    match receiver.recv().await {
        Ok(Some(msg)) => {
            debug!(from, "Received msg");
            Ok(From { from: *from, msg })
        }
        Ok(None) => Err(Error::MultiRecv(None)),
        Err(err) => Err(Error::MultiRecv(Some(err))),
    }
}

/// remotes must include local_addr
#[instrument]
pub async fn connect<T: RemoteSend>(
    local_addr: SocketAddr,
    remotes: &[SocketAddr],
    timeout: Duration,
) -> Result<(MultiSender<T>, MultiReceiver<T>), Error> {
    let listener = TcpListener::bind(local_addr)
        .await
        .map_err(tcp::Error::Io)?;
    let mut my_party_id = None;
    let remotes: Vec<_> = remotes
        .iter()
        .cloned()
        .enumerate()
        .filter(|(id, addr)| {
            let is_local = addr == &local_addr;
            if is_local {
                my_party_id = Some(*id as u32);
            }
            // filter out local address
            !is_local
        })
        .collect();
    let Some(my_party_id) = my_party_id else {
        panic!("remotes must contain local_addr to ensure correct party ids");
    };
    let (listen, connect) = join(
        listen_for_remotes::<T>(listener, remotes.len()),
        connect_to_remotes(my_party_id, remotes, timeout),
    )
    .await;
    Ok((listen?, connect?))
}

#[instrument(level = "debug", skip_all)]
async fn listen_for_remotes<T: RemoteSend>(
    listener: TcpListener,
    mut num_remotes: usize,
) -> Result<MultiSender<T>, Error> {
    let mut senders = HashMap::new();
    loop {
        // local addr is part of remotes
        if num_remotes == 0 {
            break;
        }
        match listener.accept().await {
            Ok((stream, addr)) => {
                // we're not interested in the base channel, as we use the ones from
                // connecting to the remotes. We only need to spawn the connectio
                // which is done internally in this method
                let (_, _, mut base_receiver, _) =
                    tcp::establish_remoc_connection_tcp::<InitialMsg<T>>(stream).await?;
                debug!(%addr, "Established connection to remote");
                match base_receiver.recv().await {
                    Ok(Some(InitialMsg { party_id, sender })) => {
                        if let Some(_) = senders.insert(party_id, sender) {
                            return Err(Error::DuplicateInitialMsg);
                        }
                    }
                    _ => return Err(Error::MissingInitialMsg),
                }
                // TODO how to get the correct party id??? Can't use the addr for sockets on
                //  localhost :/
                num_remotes -= 1;
            }
            Err(err) => {
                error!("Error during TCP connection establishment. {err:#?}")
            }
        }
    }
    debug!("Listened for all remotes");
    Ok(MultiSender { senders })
}

#[instrument(level = "debug", skip(remotes, timeout))]
async fn connect_to_remotes<T, S>(
    my_party_id: u32,
    remotes: impl IntoIterator<Item = (usize, S)>,
    timeout: Duration,
) -> Result<MultiReceiver<T>, Error>
where
    T: RemoteSend,
    S: ToSocketAddrs + Debug + Sync + Send + 'static,
{
    let mut join_set = JoinSet::new();
    for (id, remote) in remotes {
        join_set.spawn(
            async move {
                let ch = tcp::connect_with_timeout(remote, timeout).await;
                ch.map(|ch| (id, ch))
            }
            .in_current_span(),
        );
    }
    let mut receivers = HashMap::new();
    while let Some(conn_res) = join_set.join_next().await {
        match conn_res {
            Ok(Ok((id, (mut base_sender, _, _, _)))) => {
                let id = id as u32;
                let (sender, receiver) = super::channel(128);
                receivers.insert(id, receiver);
                // we send our party id over the channel, so the other side knows with
                // which party it communicates
                base_sender
                    .send(InitialMsg {
                        party_id: my_party_id,
                        sender,
                    })
                    .await
                    .map_err(|err| err.without_item())?;
            }
            Ok(Err(err)) => {
                return Err(err.into());
            }
            Err(err) => return Err(err.into()),
        }
    }
    debug!("Connected to all remotes");
    Ok(MultiReceiver { receivers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::init_tracing;
    use std::net::{IpAddr, Ipv4Addr};
    use tokio::task::JoinError;
    use tracing::{debug_span, Instrument};

    #[tokio::test]
    async fn create_multi_channel() {
        let _g = init_tracing();
        let base_port: u16 = 7712;
        let parties = 10;
        let parties_addrs: Vec<_> = (0..parties)
            .map(|p| SocketAddr::from((Ipv4Addr::LOCALHOST, base_port + p)))
            .collect();
        let mut join_set = JoinSet::new();
        for party in 0..parties {
            let parties_addrs = parties_addrs.clone();
            join_set.spawn(async move {
                connect::<()>(
                    parties_addrs[party as usize],
                    &parties_addrs,
                    Duration::from_millis(200),
                )
                .await
            });
        }
        let mut cnt = 0;
        loop {
            match join_set.join_next().await {
                None => break,
                Some(Ok(Ok(_))) => {
                    cnt += 1;
                }
                Some(Ok(Err(err))) => {
                    panic!("{err:?}");
                }
                Some(Err(err)) => {
                    panic!("{err:?}");
                }
            }
        }
        assert_eq!(cnt, parties);
    }

    #[tokio::test]
    async fn send_receive_via_multi_channel() {
        let _g = init_tracing();
        let base_port: u16 = 7612;
        let parties = 5;
        let parties_addrs: Vec<_> = (0..parties)
            .map(|p| SocketAddr::from((Ipv4Addr::LOCALHOST, base_port + p)))
            .collect();
        let mut join_set = JoinSet::new();
        for party in 0..parties {
            let parties_addrs = parties_addrs.clone();
            join_set.spawn(async move {
                let ch = connect::<String>(
                    parties_addrs[party as usize],
                    &parties_addrs,
                    Duration::from_millis(200),
                )
                .await;
                (party, ch)
            });
        }
        let mut multi_channels = HashMap::new();
        loop {
            match join_set.join_next().await {
                Some(Ok((id, Ok(ch)))) => {
                    multi_channels.insert(id, ch);
                }
                None => break,
                Some(err) => {
                    panic!("{err:#?}")
                }
            }
        }

        multi_channels
            .get(&0)
            .unwrap()
            .0
            .send_all("hello there".to_string())
            .await
            .unwrap();

        for (id, (_, mreceiver)) in multi_channels.iter_mut().filter(|(id, _)| **id != 0) {
            debug!(id, "Listening on");
            let res = mreceiver
                .recv_from(&FromIterator::from_iter([0]))
                .next()
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                From {
                    from: 0,
                    msg: String::from("hello there")
                },
                res
            );
        }
    }
}
