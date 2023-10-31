use crate::mul_triple::arithmetic::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::{Ring, SetupStorage};
use async_trait::async_trait;
use bitvec::slice::BitSlice;
use bitvec::store::BitStore;
use bytemuck::Pod;
use itertools::izip;
use num_integer::Integer;
use rand::distributions::{Distribution, Standard};
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use remoc::RemoteSend;
use std::fmt::Debug;
use std::mem;
use zappot::traits::{ExtROTReceiver, ExtROTSender};
use zappot::util::aes_rng::AesRng;
use zappot::util::Block;

pub struct OtMTProvider<R, RNG, OtS: ExtROTSender, OtR: ExtROTReceiver> {
    rng: RNG,
    ot_sender: OtS,
    ot_receiver: OtR,
    ch_sender: seec_channel::Sender<seec_channel::Receiver<OtS::Msg>>,
    ch_receiver: seec_channel::Receiver<seec_channel::Receiver<OtS::Msg>>,
    precomputed_mts: Option<MulTriples<R>>,
}

impl<R: Ring, RNG: RngCore + CryptoRng + Send, OtS: ExtROTSender, OtR: ExtROTReceiver>
    OtMTProvider<R, RNG, OtS, OtR>
{
    pub fn new(
        rng: RNG,
        ot_sender: OtS,
        ot_receiver: OtR,
        ch_sender: seec_channel::Sender<seec_channel::Receiver<OtS::Msg>>,
        ch_receiver: seec_channel::Receiver<seec_channel::Receiver<OtS::Msg>>,
    ) -> Self {
        Self {
            rng,
            ot_sender,
            ot_receiver,
            ch_sender,
            ch_receiver,
            precomputed_mts: None,
        }
    }
}

impl<R, RNG, OtS, OtR> OtMTProvider<R, RNG, OtS, OtR>
where
    R: Ring + From<u8> + BitStore + Pod,
    <R as BitStore>::Unalias: Pod,
    Block: From<R>,
    Standard: Distribution<R>,
    RNG: RngCore + CryptoRng + Send,
    OtS: ExtROTSender<Msg = OtR::Msg> + Send,
    OtS::Msg: RemoteSend + Debug,
    OtR: ExtROTReceiver + Send,
    OtR::Msg: RemoteSend + Debug,
{
    async fn compute_mts(&mut self, mut amount: usize) -> Result<MulTriples<R>, ()> {
        let mut sender_rng = AesRng::from_rng(&mut self.rng).unwrap();
        let mut receiver_rng = AesRng::from_rng(&mut self.rng).unwrap();

        amount *= R::BITS;
        let amount = Integer::next_multiple_of(&amount, &8);

        let (ch_sender1, mut ch_receiver1) =
            seec_channel::sub_channel(&mut self.ch_sender, &mut self.ch_receiver, 128)
                .await
                .unwrap();
        let (ch_sender2, mut ch_receiver2) =
            seec_channel::sub_channel(&mut self.ch_sender, &mut self.ch_receiver, 128)
                .await
                .unwrap();

        let a_i: Vec<R> = (&mut self.rng).sample_iter(Standard).take(amount).collect();
        let b_i: Vec<R> = (&mut self.rng).sample_iter(Standard).take(amount).collect();

        // Correlation function is slightly different from that given in ABY paper
        // we do a wrapping_add of x and then later do wrapping_sub to compute u. The way given in
        // the paper is slightly wrong
        let correlation = |idx: usize, x: Block| {
            let mt_idx = idx / R::BITS;
            let i = idx % R::BITS;
            let a = &a_i[mt_idx];
            let x = R::from_block(x);
            let res = a.wrapping_mul(&R::from(2).pow(i as u32)).wrapping_add(&x);
            Block::from(res)
        };

        let send = self.ot_sender.send_correlated(
            amount,
            correlation,
            &mut sender_rng,
            &ch_sender1,
            &mut ch_receiver2,
        );

        let choices = BitSlice::from_slice(&b_i);

        let receive = self.ot_receiver.receive_correlated(
            choices,
            &mut receiver_rng,
            &ch_sender2,
            &mut ch_receiver1,
        );

        let (send_ots, recv_ots) = tokio::try_join!(send, receive).unwrap();

        let reduce = |corr_ots: Vec<Block>, op: fn(&R, &R) -> R| -> Vec<_> {
            corr_ots
                .chunks_exact(R::BITS)
                .map(|chunk| {
                    chunk
                        .iter()
                        .fold(R::from(0), |acc, b| op(&acc, &R::from_block(*b)))
                })
                .collect()
        };

        let u: Vec<_> = reduce(send_ots, R::wrapping_sub);
        let v = reduce(recv_ots, R::wrapping_add);

        let c_i = izip!(&a_i, &b_i, u, v)
            .map(|(a, b, u, v)| a.wrapping_mul(b).wrapping_add(&u).wrapping_add(&v))
            .collect();

        Ok(MulTriples {
            a: a_i,
            b: b_i,
            c: c_i,
        })
    }
}

#[async_trait]
impl<R, RNG, OtS, OtR> MTProvider for OtMTProvider<R, RNG, OtS, OtR>
where
    R: Ring + From<u8> + BitStore + Pod,
    <R as BitStore>::Unalias: Pod,
    Block: From<R>,
    Standard: Distribution<R>,
    RNG: RngCore + CryptoRng + Send,
    OtS: ExtROTSender<Msg = OtR::Msg> + Send,
    OtS::Msg: RemoteSend + Debug,
    OtR: ExtROTReceiver + Send,
    OtR::Msg: RemoteSend + Debug,
{
    type Output = MulTriples<R>;
    type Error = ();

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), ()> {
        let mts = self.compute_mts(amount).await?;
        self.precomputed_mts = Some(mts);
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        match &mut self.precomputed_mts {
            Some(mts) if mts.len() >= amount => Ok(mts.split_off_last(amount)),
            Some(mts) => {
                let additional_needed = amount - mts.len();
                let mut precomputed = mem::take(mts);
                let additional_mts = self.compute_mts(additional_needed).await?;
                precomputed.extend_from_mts(&additional_mts);
                Ok(precomputed)
            }
            None => self.compute_mts(amount).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mul_triple::arithmetic::ot_ext::OtMTProvider;
    use crate::mul_triple::MTProvider;
    use crate::private_test_utils::init_tracing;
    use rand::rngs::OsRng;
    use zappot::ot_ext;

    #[tokio::test]
    async fn ot_ext_provider() {
        let _guard = init_tracing();
        let ((ch_sender1, ch_receiver1), (ch_sender2, ch_receiver2)) =
            seec_channel::in_memory::new_pair(8);

        let party = |ch_sender, ch_receiver| async {
            let ot_sender = ot_ext::Sender::default();
            let ot_receiver = ot_ext::Receiver::default();

            let mut mtp = OtMTProvider::<u64, _, _, _>::new(
                OsRng,
                ot_sender,
                ot_receiver,
                ch_sender,
                ch_receiver,
            );
            mtp.request_mts(32).await.unwrap()
        };

        let (mts1, mts2) = tokio::join!(
            party(ch_sender1, ch_receiver1),
            party(ch_sender2, ch_receiver2)
        );

        mts1.iter()
            .enumerate()
            .zip(mts2.iter())
            .for_each(|((idx, mt1), mt2)| {
                assert_eq!(
                    mt1.c.wrapping_add(mt2.c),
                    (mt1.a.wrapping_add(mt2.a)).wrapping_mul(mt1.b.wrapping_add(mt2.b)),
                    "Wrong MTs for idx {idx}"
                )
            });
    }
}
