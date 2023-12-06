//! ALSZ13 OT extension protocol.
use crate::traits::{BaseROTReceiver, BaseROTSender, Error, ExtROTReceiver, ExtROTSender};
use crate::util::aes_hash::FIXED_KEY_HASH;
use crate::util::aes_rng::AesRng;
use crate::util::tokio_rayon::spawn_compute;
use crate::util::transpose::transpose;
use crate::util::Block;
use crate::{base_ot, BASE_OT_COUNT};
use async_trait::async_trait;
use bitvec::bitvec;
use bitvec::slice::BitSlice;
use bitvec::store::BitStore;
use bitvec::vec::BitVec;
use bytemuck::{cast_slice, Pod};
use rand::{CryptoRng, Rng, RngCore};
use rand_core::SeedableRng;
use rayon::prelude::*;
use remoc::RemoteSend;
use seec_channel::channel;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::{iter, mem};

pub struct Sender<BaseOT = base_ot::Receiver> {
    base_ot: BaseOT,
    base_rngs: Option<Arc<Mutex<Vec<AesRng>>>>,
    base_choices: Option<BitVec>,
}

pub struct Receiver<BaseOT = base_ot::Sender> {
    base_ot: BaseOT,
    base_rngs: Option<Arc<Mutex<Vec<[AesRng; 2]>>>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ExtOTMsg<BaseOTMsg: RemoteSend = base_ot::BaseOTMsg> {
    // Workaround for compiler bug,
    // see https://github.com/serde-rs/serde/issues/1296#issuecomment-394056188
    #[serde(bound = "")]
    BaseOTChannel(seec_channel::Receiver<BaseOTMsg>),
    URow(usize, Vec<u8>),
    Correlated(Vec<u8>),
}

impl<BaseOT> Sender<BaseOT>
where
    BaseOT: BaseROTReceiver + Send,
    BaseOT::Msg: RemoteSend + Debug,
{
    pub async fn perform_base_ots<RNG: RngCore + CryptoRng + Send>(
        &mut self,
        rng: &mut RNG,
        sender: &seec_channel::Sender<<Self as ExtROTSender>::Msg>,
        receiver: &mut seec_channel::Receiver<<Self as ExtROTSender>::Msg>,
    ) -> Result<(), Error<<Self as ExtROTSender>::Msg>> {
        if let (Some(_), Some(_)) = (&self.base_rngs, &self.base_choices) {
            return Ok(());
        }

        let (base_sender, base_remote_receiver) = channel(BASE_OT_COUNT);
        sender
            .send(ExtOTMsg::BaseOTChannel(base_remote_receiver))
            .await?;
        let msg = receiver.recv().await?.ok_or(Error::UnexpectedTermination)?;
        let mut base_receiver = match msg {
            ExtOTMsg::BaseOTChannel(receiver) => receiver,
            _ => return Err(Error::WrongOrder(msg)),
        };
        let rand_choices: BitVec = {
            let mut bv = bitvec![0; BASE_OT_COUNT];
            rng.fill(bv.as_raw_mut_slice());
            bv
        };
        let base_ots = self
            .base_ot
            .receive_random(&rand_choices, rng, &base_sender, &mut base_receiver)
            .await
            .map_err(|err| Error::BaseOT(Box::new(err)))?;
        let base_rngs = base_ots.into_iter().map(AesRng::from_seed).collect();
        self.base_rngs = Some(Arc::new(Mutex::new(base_rngs)));
        self.base_choices = Some(rand_choices);
        Ok(())
    }
}

#[async_trait]
impl<BaseOT> ExtROTSender for Sender<BaseOT>
where
    BaseOT: BaseROTReceiver + Send,
    BaseOT::Msg: RemoteSend + Debug,
{
    type Msg = ExtOTMsg<BaseOT::Msg>;

    #[allow(non_snake_case)]
    async fn send_random<RNG>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        sender: &seec_channel::Sender<Self::Msg>,
        receiver: &mut seec_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[Block; 2]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
    {
        assert_eq!(
            count % 8,
            0,
            "Number of OT extensions must be multiple of 8"
        );
        self.perform_base_ots(rng, sender, receiver).await?;
        let base_ot_rngs = self.base_rngs.clone().unwrap();
        let choices = self.base_choices.clone().unwrap();

        let delta: Block = (&choices)
            .try_into()
            .expect("BASE_OT_COUNT must be size of a Block");

        let rows = BASE_OT_COUNT;
        let cols = count / 8; // div by 8 because of u8
        let mut v_mat = spawn_compute(move || {
            let mut base_ot_rngs = base_ot_rngs.lock().unwrap();
            let mut v_mat = vec![0_u8; rows * cols];
            v_mat
                .chunks_exact_mut(cols)
                .zip(&mut base_ot_rngs[..])
                .for_each(|(row, prg)| {
                    prg.fill_bytes(row);
                });
            v_mat
        })
        .await;
        let mut rows_received = 0;
        while let Some(msg) = receiver.recv().await.transpose() {
            let (idx, mut u_row) = match msg.map_err(Error::Receive)? {
                ExtOTMsg::URow(idx, row) => (idx, row),
                msg => return Err(Error::WrongOrder(msg)),
            };
            let r = choices[idx];
            let v_row = &mut v_mat[idx * cols..(idx + 1) * cols];
            for el in &mut u_row {
                // computes r_j * u_j
                // TODO cleanup, also const time?
                *el = if r { *el } else { 0 };
            }
            v_row.iter_mut().zip(u_row).for_each(|(v, u)| {
                *v ^= u;
            });
            rows_received += 1;
            if rows_received == rows {
                break;
            }
        }

        let ots = spawn_compute(move || {
            let v_mat = transpose(&v_mat, rows, count);
            v_mat
                // TODO benchmark parallelization
                .par_chunks_exact(BASE_OT_COUNT / u8::BITS as usize)
                .map(|row| {
                    let block = row
                        .try_into()
                        .expect("message size must be block length (128 bits)");
                    let x_0 = FIXED_KEY_HASH.cr_hash_block(block);
                    let x_1 = FIXED_KEY_HASH.cr_hash_block(block ^ delta);
                    [x_0, x_1]
                })
                .collect()
        })
        .await;
        Ok(ots)
    }

    async fn send_correlated<RNG>(
        &mut self,
        count: usize,
        correlation: impl Fn(usize, Block) -> Block + Send,
        rng: &mut RNG,
        sender: &seec_channel::Sender<Self::Msg>,
        receiver: &mut seec_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
    {
        let correlation_wrapper =
            move |i, bytes| correlation(i, Block::from_le_bytes(bytes)).to_le_bytes();
        let out = self
            .send_correlated_bytes::<{ mem::size_of::<Block>() }, _>(
                count,
                correlation_wrapper,
                rng,
                sender,
                receiver,
            )
            .await;
        out.map(|bytes| bytes.into_iter().map(Block::from_le_bytes).collect())
    }

    async fn send_correlated_bytes<const LEN: usize, RNG>(
        &mut self,
        count: usize,
        correlation: impl Fn(usize, [u8; LEN]) -> [u8; LEN] + Send,
        rng: &mut RNG,
        sender: &seec_channel::Sender<Self::Msg>,
        receiver: &mut seec_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[u8; LEN]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        [u8; LEN]: Pod,
    {
        let r_ot = self.send_random(count, rng, sender, receiver).await?;

        let (ret, correlated) = r_ot
            .into_iter()
            .enumerate()
            .map(|(idx, [ot0, ot1])| {
                let ot0 = split_arr(ot0.to_le_bytes());
                let ot1 = split_arr(ot1.to_le_bytes());
                let correlated = xor_arr(ot1, correlation(idx, ot0));
                (ot0, correlated)
            })
            .unzip();
        let correlated = bytemuck::cast_vec(correlated);
        // TODO this sends a block for each c-ot, for an l bit c-ot with l < kappa,
        //  this wastes communication
        sender.send(ExtOTMsg::Correlated(correlated)).await?;
        Ok(ret)
    }
}

impl<BaseOT> Receiver<BaseOT>
where
    BaseOT: BaseROTSender + Send,
    BaseOT::Msg: RemoteSend + Debug,
{
    pub async fn perform_base_ots<RNG: RngCore + CryptoRng + Send>(
        &mut self,
        rng: &mut RNG,
        sender: &seec_channel::Sender<<Self as ExtROTReceiver>::Msg>,
        receiver: &mut seec_channel::Receiver<<Self as ExtROTReceiver>::Msg>,
    ) -> Result<(), Error<<Self as ExtROTReceiver>::Msg>> {
        if self.base_rngs.is_some() {
            return Ok(());
        }

        let (base_sender, base_remote_receiver) = channel(BASE_OT_COUNT);
        sender
            .send(ExtOTMsg::BaseOTChannel(base_remote_receiver))
            .await?;
        let msg = receiver.recv().await?.ok_or(Error::UnexpectedTermination)?;
        let mut base_receiver = match msg {
            ExtOTMsg::BaseOTChannel(receiver) => receiver,
            _ => return Err(Error::WrongOrder(msg)),
        };
        let base_ots = self
            .base_ot
            .send_random(BASE_OT_COUNT, rng, &base_sender, &mut base_receiver)
            .await
            .map_err(|err| Error::BaseOT(Box::new(err)))?;

        let base_rngs = base_ots
            .into_iter()
            .map(|[ot0, ot1]| [AesRng::from_seed(ot0), AesRng::from_seed(ot1)])
            .collect();
        self.base_rngs = Some(Arc::new(Mutex::new(base_rngs)));
        Ok(())
    }
}

#[async_trait]
impl<BaseOT> ExtROTReceiver for Receiver<BaseOT>
where
    BaseOT: BaseROTSender + Send,
    BaseOT::Msg: RemoteSend + Debug,
{
    type Msg = ExtOTMsg<BaseOT::Msg>;

    #[allow(non_snake_case)]
    async fn receive_random<C, RNG>(
        &mut self,
        choices: &BitSlice<C>,
        rng: &mut RNG,
        sender: &seec_channel::Sender<Self::Msg>,
        receiver: &mut seec_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        C: Pod + BitStore + Sync,
        <C as BitStore>::Unalias: Pod,
    {
        assert_eq!(
            choices.len() % 8,
            0,
            "Number of OT extensions must be multiple of 8"
        );
        let count = choices.len();
        self.perform_base_ots(rng, sender, receiver).await?;
        let base_ot_rngs = self.base_rngs.clone().unwrap();

        let rows = BASE_OT_COUNT;
        let cols = count / 8; // div by 8 because of u8

        let choices = choices.to_bitvec();
        let sender = sender.clone();
        let t_mat = spawn_compute(move || {
            let mut base_ot_rngs = base_ot_rngs.lock().unwrap();
            let choices = cast_slice::<_, u8>(choices.as_raw_slice());
            let mut t_mat = vec![0_u8; rows * cols];
            t_mat
                .par_chunks_exact_mut(cols)
                .enumerate()
                .zip(&mut base_ot_rngs[..])
                .for_each(|((idx, t_row), [prg0, prg1])| {
                    prg0.fill_bytes(t_row);
                    let u_row = {
                        let mut row = vec![0_u8; cols];
                        prg1.fill_bytes(&mut row);
                        row.iter_mut().zip(t_row).zip(choices).for_each(
                            |((val, rand_val), choice)| {
                                *val ^= *rand_val ^ choice;
                            },
                        );
                        row
                    };
                    sender
                        .blocking_send(ExtOTMsg::URow(idx, u_row))
                        .expect("URow send failed");
                });
            t_mat
        })
        .await;

        let ots = spawn_compute(move || {
            let t_mat = transpose(&t_mat, rows, count);
            t_mat
                // TODO parallelize this code
                .par_chunks_exact(BASE_OT_COUNT / u8::BITS as usize)
                .map(|rows| {
                    let block = rows
                        .try_into()
                        .expect("message size must be block length (128 bits)");
                    FIXED_KEY_HASH.cr_hash_block(block)
                })
                .collect()
        })
        .await;
        Ok(ots)
    }

    async fn receive_correlated<C, RNG>(
        &mut self,
        choices: &BitSlice<C>,
        rng: &mut RNG,
        sender: &seec_channel::Sender<Self::Msg>,
        receiver: &mut seec_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<Block>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        C: Pod + BitStore + Sync,
        <C as BitStore>::Unalias: Pod,
    {
        let ret = self
            .receive_correlated_bytes::<{ mem::size_of::<Block>() }, _, _>(
                choices, rng, sender, receiver,
            )
            .await;
        ret.map(|bytes| bytes.into_iter().map(Block::from_le_bytes).collect())
    }

    async fn receive_correlated_bytes<const LEN: usize, C, RNG>(
        &mut self,
        choices: &BitSlice<C>,
        rng: &mut RNG,
        sender: &seec_channel::Sender<Self::Msg>,
        receiver: &mut seec_channel::Receiver<Self::Msg>,
    ) -> Result<Vec<[u8; LEN]>, Error<Self::Msg>>
    where
        RNG: RngCore + CryptoRng + Send,
        [u8; LEN]: Pod,
        C: Pod + BitStore + Sync,
        <C as BitStore>::Unalias: Pod,
    {
        let r_ot = self.receive_random(choices, rng, sender, receiver).await?;
        let correlated = match receiver.recv().await? {
            Some(ExtOTMsg::Correlated(correlated)) => correlated,
            Some(other) => Err(Error::WrongOrder(other))?,
            None => Err(Error::UnexpectedTermination)?,
        };

        let correlated = bytemuck::cast_vec(correlated);
        let ret = iter::zip(r_ot, correlated)
            .zip(choices)
            .map(|((rot, correlated), choice)| {
                let rot = split_arr(rot.to_le_bytes());
                if *choice {
                    xor_arr(rot, correlated)
                } else {
                    rot
                }
            })
            .collect();
        Ok(ret)
    }
}

impl<BaseOt> Sender<BaseOt> {
    pub fn new(base_ot_receiver: BaseOt) -> Self {
        Self {
            base_ot: base_ot_receiver,
            base_rngs: None,
            base_choices: None,
        }
    }
}

impl<BaseOt> Receiver<BaseOt> {
    pub fn new(base_ot_sender: BaseOt) -> Self {
        Self {
            base_ot: base_ot_sender,
            base_rngs: None,
        }
    }
}

impl Default for Sender {
    fn default() -> Self {
        Sender::new(base_ot::Receiver)
    }
}

impl Default for Receiver {
    fn default() -> Self {
        Receiver::new(base_ot::Sender)
    }
}

fn split_arr<const N: usize, const M: usize, T: Copy>(arr: [T; N]) -> [T; M] {
    assert!(M <= N, "Length of array prefix must be less than array");
    arr[..M].try_into().unwrap()
}

fn xor_arr<const N: usize>(mut a: [u8; N], b: [u8; N]) -> [u8; N] {
    a.iter_mut().zip(b).for_each(|(a, b)| *a ^= b);
    a
}

#[cfg(test)]
mod tests {
    use crate::base_ot;
    use crate::ot_ext::{Receiver, Sender};
    use crate::traits::{ExtROTReceiver, ExtROTSender};
    use bitvec::bitvec;
    use bitvec::order::Lsb0;
    use bitvec::vec::BitVec;

    use rand::distributions::Standard;
    use rand::rngs::StdRng;
    use rand::{thread_rng, Rng, SeedableRng};
    use tokio::time::Instant;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn random_ot_ext() {
        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(128);
        let num_ots: usize = 1000;
        let now = Instant::now();
        let send = tokio::spawn(async move {
            let mut sender = Sender::new(base_ot::Receiver {});
            let mut rng_send = StdRng::seed_from_u64(42);
            sender
                .send_random(num_ots, &mut rng_send, &ch1.0, &mut ch1.1)
                .await
                .unwrap()
        });
        let choices = bitvec![usize, Lsb0; 0; num_ots];
        let receive = tokio::spawn(async move {
            let mut receiver = Receiver::new(base_ot::Sender {});
            let mut rng_recv = StdRng::seed_from_u64(42 * 42);
            receiver
                .receive_random(&choices, &mut rng_recv, &ch2.0, &mut ch2.1)
                .await
                .unwrap()
        });
        let (recv, sent) = tokio::try_join!(receive, send).unwrap();
        println!("Total time: {}", now.elapsed().as_secs_f32());
        for (r, [s, _]) in recv.into_iter().zip(sent) {
            assert_eq!(r, s)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn correlated_ot_ext() {
        let (mut ch1, mut ch2) = seec_channel::in_memory::new_pair(128);
        let num_ots: usize = 1000;
        let now = Instant::now();
        let send = tokio::spawn(async move {
            let mut sender = Sender::new(base_ot::Receiver {});
            let mut rng_send = StdRng::seed_from_u64(42);
            sender
                .send_correlated(num_ots, |_i, x| x, &mut rng_send, &ch1.0, &mut ch1.1)
                .await
                .unwrap()
        });
        let choices: BitVec = thread_rng()
            .sample_iter::<bool, _>(Standard)
            .take(num_ots)
            .collect();
        let receive = tokio::spawn(async move {
            let mut receiver = Receiver::new(base_ot::Sender {});
            let mut rng_recv = StdRng::seed_from_u64(42 * 42);
            receiver
                .receive_correlated(&choices, &mut rng_recv, &ch2.0, &mut ch2.1)
                .await
                .unwrap()
        });
        let (recv, sent) = tokio::try_join!(receive, send).unwrap();
        println!("Total time: {}", now.elapsed().as_secs_f32());
        for (r, s) in recv.into_iter().zip(sent) {
            assert_eq!(r, s)
        }
    }
}
