//! ALSZ13 OT extension protocol.
use crate::base_ot;
use crate::traits::{
    BaseROTReceiver, BaseROTSender, Error, ExtROTReceiver, ExtROTSender, ProtocolError,
};
use crate::util::aes_hash::FIXED_KEY_HASH;
use crate::util::aes_rng::AesRng;
use crate::util::tokio_rayon::spawn_compute;
use crate::util::transpose::transpose;
use crate::util::Block;
use async_trait::async_trait;
use bitvec::bitvec;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bytemuck::cast_slice;
use futures::{Sink, SinkExt, StreamExt, TryStream};
use mpc_channel::{Channel, ConstrictError, IsSubMsg};
use rand::{CryptoRng, Rng, RngCore};
use rand_core::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;
use tokio::sync::mpsc;

const BASE_OT_COUNT: usize = 128;

pub struct Sender<BaseOT> {
    base_ot: BaseOT,
}

pub struct Receiver<BaseOT> {
    base_ot: BaseOT,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ExtOTMsg<BaseOTMsg = base_ot::BaseOTMsg> {
    BaseOT(BaseOTMsg),
    URow(usize, Vec<u8>),
}

#[async_trait]
impl<BaseOT> ExtROTSender for Sender<BaseOT>
where
    BaseOT: BaseROTReceiver + Send,
    BaseOT::Msg: IsSubMsg<ExtOTMsg<BaseOT::Msg>> + Send + Debug + 'static,
    <BaseOT::Msg as IsSubMsg<ExtOTMsg<BaseOT::Msg>>>::Error: Debug + Sized,
{
    type Msg = ExtOTMsg<BaseOT::Msg>;

    #[allow(non_snake_case)]
    async fn send_random<RNG, CH>(
        &mut self,
        count: usize,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<[Block; 2]>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Send + Unpin,
        <CH as Sink<Self::Msg>>::Error: std::error::Error + Send,
        <CH as TryStream>::Error: std::error::Error + Send,
    {
        // assert_eq!(
        //     count % 8,
        //     0,
        //     "Number of OT extensions must be multiple of 8"
        // );
        // let (base_ots, choices) = {
        //     let mut channel = channel.constrict::<BaseOT::Msg>();
        //     let rand_choices: BitVec = {
        //         let mut bv = bitvec![0; BASE_OT_COUNT];
        //         rng.fill(bv.as_raw_mut_slice());
        //         bv
        //     };
        //     let base_ots = self
        //         .base_ot
        //         .receive_random(&rand_choices, rng, &mut channel)
        //         .await
        //         .map_err(|err| Error::BaseOT(Box::new(err)))?;
        //     (base_ots, rand_choices)
        // };
        // todo!()
        // // let (stream, sink) = channel.split_mut();
        // //
        // // let delta: Block = (&choices)
        // //     .try_into()
        // //     .expect("BASE_OT_COUNT must be size of a Block");
        // // let rows = BASE_OT_COUNT;
        // // let cols = count / 8; // div by 8 because of u8
        // // let mut v_mat = spawn_compute(move || {
        // //     let mut v_mat = vec![0_u8; rows * cols];
        // //     v_mat
        // //         .chunks_exact_mut(cols)
        // //         .zip(base_ots)
        // //         .for_each(|(row, seed)| {
        // //             let mut prg = AesRng::from_seed(seed);
        // //             prg.fill_bytes(row);
        // //         });
        // //     v_mat
        // // })
        // // .await;
        // // let mut rows_received = 0;
        // // while let Some(msg) = stream.next().await {
        // //     let (idx, mut u_row) = match msg.map_err(ProtocolError::<Self::Msg, CH>::Receive)? {
        // //         ExtOTMsg::URow(idx, row) => (idx, row),
        // //         msg => return Err(ProtocolError::<Self::Msg, CH>::WrongOrder(msg)),
        // //     };
        // //     let r = choices[idx];
        // //     let v_row = &mut v_mat[idx * cols..(idx + 1) * cols];
        // //     for el in &mut u_row {
        // //         // computes r_j * u_j
        // //         // TODO cleanup, also const time?
        // //         *el = if r { *el } else { 0 };
        // //     }
        // //     v_row.iter_mut().zip(u_row).for_each(|(v, u)| {
        // //         *v ^= u;
        // //     });
        // //     rows_received += 1;
        // //     if rows_received == rows {
        // //         break;
        // //     }
        // // }
        // //
        // // let ots = spawn_compute(move || {
        // //     let v_mat = transpose(&v_mat, rows, count);
        // //     v_mat
        // //         // TODO benchmark parallelization
        // //         .par_chunks_exact(BASE_OT_COUNT / u8::BITS as usize)
        // //         .map(|row| {
        // //             let block = row
        // //                 .try_into()
        // //                 .expect("message size must be block length (128 bits)");
        // //             let x_0 = FIXED_KEY_HASH.cr_hash_block(block);
        // //             let x_1 = FIXED_KEY_HASH.cr_hash_block(block ^ delta);
        // //             [x_0, x_1]
        // //         })
        // //         .collect()
        // // })
        // // .await;
        // // Ok(ots)
    }
}

fn assert_static<T: 'static>(val: &T) {}

#[async_trait]
impl<BaseOT> ExtROTReceiver for Receiver<BaseOT>
where
    BaseOT: BaseROTSender + Send,
    BaseOT::Msg: IsSubMsg<ExtOTMsg<BaseOT::Msg>> + Send + Debug + 'static,
    <BaseOT::Msg as IsSubMsg<ExtOTMsg<BaseOT::Msg>>>::Error: Debug + Sized,
{
    type Msg = ExtOTMsg<BaseOT::Msg>;

    #[allow(non_snake_case)]
    async fn receive_random<RNG, CH>(
        &mut self,
        choices: &BitSlice,
        rng: &mut RNG,
        channel: &mut CH,
    ) -> Result<Vec<Block>, ProtocolError<Self::Msg, CH>>
    where
        RNG: RngCore + CryptoRng + Send,
        CH: Channel<Self::Msg> + Send + Unpin,
    {
        todo!()
        // assert_eq!(
        //     choices.len() % 8,
        //     0,
        //     "Number of OT extensions must be multiple of 8"
        // );
        // let count = choices.len();
        // let base_ots = {
        //     let mut channel = channel.constrict::<BaseOT::Msg>();
        //     self.base_ot
        //         .send_random(BASE_OT_COUNT, rng, &mut channel)
        //         .await
        //         .map_err(|err| Error::BaseOT(Box::new(err)))?
        // };
        // let (stream, sink) = channel.split_mut();
        //
        // let rows = BASE_OT_COUNT;
        // let cols = count / 8; // div by 8 because of u8
        //
        // let (mtx, mut mrx) = mpsc::channel(BASE_OT_COUNT);
        // let choices = choices.to_bitvec();
        //
        // let send_task = async {
        //     while let Some((idx, u_row)) = mrx.recv().await {
        //         sink.send(ExtOTMsg::URow(idx, u_row))
        //             .await
        //             .map_err(ProtocolError::<Self::Msg, CH>::Send)?;
        //     }
        //     Ok::<_, ProtocolError<Self::Msg, CH>>(())
        // };
        //
        // let t_mat_fut = spawn_compute(move || {
        //     let choices = cast_slice::<_, u8>(choices.as_raw_slice());
        //     let mut t_mat = vec![0_u8; rows * cols];
        //     t_mat
        //         .par_chunks_exact_mut(cols)
        //         .enumerate()
        //         .zip(base_ots)
        //         .for_each(|((idx, t_row), [s0, s1])| {
        //             let mut prg0 = AesRng::from_seed(s0);
        //             let mut prg1 = AesRng::from_seed(s1);
        //             prg0.fill_bytes(t_row);
        //             let u_row = {
        //                 let mut row = vec![0_u8; cols];
        //                 prg1.fill_bytes(&mut row);
        //                 row.iter_mut().zip(t_row).zip(choices).for_each(
        //                     |((val, rand_val), choice)| {
        //                         *val ^= *rand_val ^ choice;
        //                     },
        //                 );
        //                 row
        //             };
        //             mtx.blocking_send((idx, u_row))
        //                 .expect("Called in async context");
        //         });
        //     Ok(t_mat)
        // });
        //
        // // TODO this could be optimized a little by using select and continuing with the following
        // //  compute task while sand_task has not finished
        // let (_, t_mat) = tokio::try_join!(send_task, t_mat_fut)?;
        //
        // let ots = spawn_compute(move || {
        //     let t_mat = transpose(&t_mat, rows, count);
        //     t_mat
        //         // TODO parallelize this code
        //         .par_chunks_exact(BASE_OT_COUNT / u8::BITS as usize)
        //         .map(|rows| {
        //             let block = rows
        //                 .try_into()
        //                 .expect("message size must be block length (128 bits)");
        //             FIXED_KEY_HASH.cr_hash_block(block)
        //         })
        //         .collect()
        // })
        // .await;
        // Ok(ots)
    }
}

impl<BaseOt> Sender<BaseOt> {
    pub fn new(base_ot_receiver: BaseOt) -> Self {
        Self {
            base_ot: base_ot_receiver,
        }
    }
}

impl<BaseOt> Receiver<BaseOt> {
    pub fn new(base_ot_sender: BaseOt) -> Self {
        Self {
            base_ot: base_ot_sender,
        }
    }
}

impl Default for Sender<base_ot::Receiver> {
    fn default() -> Self {
        Sender::new(base_ot::Receiver)
    }
}

impl Default for Receiver<base_ot::Sender> {
    fn default() -> Self {
        Receiver::new(base_ot::Sender)
    }
}

impl<BaseOTMsg> From<BaseOTMsg> for ExtOTMsg<BaseOTMsg> {
    fn from(msg: BaseOTMsg) -> Self {
        Self::BaseOT(msg)
    }
}

impl TryFrom<ExtOTMsg<base_ot::BaseOTMsg>> for base_ot::BaseOTMsg {
    type Error = ExtOTMsg<base_ot::BaseOTMsg>;

    fn try_from(value: ExtOTMsg<base_ot::BaseOTMsg>) -> Result<Self, Self::Error> {
        match value {
            ExtOTMsg::BaseOT(msg) => Ok(msg),
            value => Err(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::base_ot;
    use crate::ot_ext::{Receiver, Sender};
    use crate::traits::{ExtROTReceiver, ExtROTSender};
    use bitvec::bitvec;
    use bitvec::order::Lsb0;
    use mpc_channel::in_memory::InMemory;

    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tokio::time::Instant;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ot_ext() {
        let (ch1, mut ch2) = InMemory::new_pair();
        let num_ots: usize = 1000;
        let now = Instant::now();
        let send = tokio::spawn(async move {
            let mut sender = Sender::new(base_ot::Receiver {});
            let mut rng_send = StdRng::seed_from_u64(42);
            let mut ch1 = ch1;
            sender
                .send_random(num_ots, &mut rng_send, &mut ch1)
                .await
                .unwrap()
        });
        let choices = bitvec![usize, Lsb0; 0;num_ots];
        let receive = tokio::spawn(async move {
            let mut receiver = Receiver::new(base_ot::Sender {});
            let mut rng_recv = StdRng::seed_from_u64(42 * 42);
            receiver
                .receive_random(&choices, &mut rng_recv, &mut ch2)
                .await
                .unwrap()
        });
        let (recv, sent) = tokio::try_join!(receive, send).unwrap();
        println!("Total time: {}", now.elapsed().as_secs_f32());
        for (r, [s, _]) in recv.into_iter().zip(sent) {
            assert_eq!(r, s)
        }
    }
}
