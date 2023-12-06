use crate::common::BitVec;
use crate::mul_triple::boolean::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::SetupStorage;
use crate::utils::rand_bitvec;
use async_trait::async_trait;
use num_integer::Integer;
use rand::{CryptoRng, RngCore, SeedableRng};
use remoc::RemoteSend;
use std::fmt::Debug;
use std::mem;
use thiserror::Error;
use zappot::traits::{ExtROTReceiver, ExtROTSender};
use zappot::util::aes_rng::AesRng;

pub type Msg<Msg> = seec_channel::Receiver<Msg>;
/// Message for default ot ext
pub type DefaultMsg = Msg<<zappot::ot_ext::Sender as ExtROTSender>::Msg>;

pub struct OtMTProvider<RNG, S: ExtROTSender, R: ExtROTReceiver> {
    rng: RNG,
    ot_sender: S,
    ot_receiver: R,
    ch_sender: seec_channel::Sender<Msg<S::Msg>>,
    ch_receiver: seec_channel::Receiver<Msg<S::Msg>>,
    precomputed_mts: Option<MulTriples>,
}

#[derive(Error, Debug)]
pub enum Error {
    // TODO
}

impl<RNG: RngCore + CryptoRng + Send, S: ExtROTSender, R: ExtROTReceiver> OtMTProvider<RNG, S, R> {
    pub fn new(
        rng: RNG,
        ot_sender: S,
        ot_receiver: R,
        ch_sender: seec_channel::Sender<Msg<S::Msg>>,
        ch_receiver: seec_channel::Receiver<Msg<S::Msg>>,
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

impl<RNG: RngCore + CryptoRng + Send>
    OtMTProvider<RNG, zappot::ot_ext::Sender, zappot::ot_ext::Receiver>
{
    pub fn new_with_default_ot_ext(
        rng: RNG,
        ch_sender: seec_channel::Sender<DefaultMsg>,
        ch_receiver: seec_channel::Receiver<DefaultMsg>,
    ) -> Self {
        Self {
            rng,
            ot_sender: Default::default(),
            ot_receiver: Default::default(),
            ch_sender,
            ch_receiver,
            precomputed_mts: None,
        }
    }
}

impl<RNG, S, R> OtMTProvider<RNG, S, R>
where
    RNG: RngCore + CryptoRng + Send,
    S: ExtROTSender<Msg = R::Msg> + Send,
    S::Msg: RemoteSend + Debug,
    R: ExtROTReceiver + Send,
    R::Msg: RemoteSend + Debug,
{
    #[tracing::instrument(level = "debug", skip(self))]
    async fn compute_mts(&mut self, amount: usize) -> Result<MulTriples, Error> {
        tracing::debug!("Computing MTs via OT");
        let mut sender_rng = AesRng::from_rng(&mut self.rng).unwrap();
        let mut receiver_rng = AesRng::from_rng(&mut self.rng).unwrap();

        let amount = Integer::next_multiple_of(&amount, &8);

        let (ch_sender1, mut ch_receiver1) =
            seec_channel::sub_channel(&mut self.ch_sender, &mut self.ch_receiver, 128)
                .await
                .unwrap();
        let (ch_sender2, mut ch_receiver2) =
            seec_channel::sub_channel(&mut self.ch_sender, &mut self.ch_receiver, 128)
                .await
                .unwrap();

        let send =
            self.ot_sender
                .send_random(amount, &mut sender_rng, &ch_sender1, &mut ch_receiver2);

        let a_i = rand_bitvec(amount, &mut receiver_rng);
        let receive = self.ot_receiver.receive_random(
            a_i.as_bitslice(),
            &mut receiver_rng,
            &ch_sender2,
            &mut ch_receiver1,
        );

        let (send_ots, recv_ots) = tokio::try_join!(send, receive).unwrap();

        let mut b_i = BitVec::with_capacity(amount);
        let mut v_i: BitVec<usize> = BitVec::with_capacity(amount);

        send_ots
            .into_iter()
            .map(|arr| arr.map(|b| b.lsb()))
            .for_each(|[m0, m1]| {
                b_i.push(m0 ^ m1);
                v_i.push(m0);
            });
        let u_i = recv_ots.into_iter().map(|b| b.lsb());
        let c_i = a_i
            .iter()
            .by_vals()
            .zip(b_i.iter().by_vals())
            .zip(u_i)
            .zip(v_i)
            .map(|(((a, b), u), v)| a & b ^ u ^ v)
            .collect();

        Ok(MulTriples::from_raw(a_i, b_i, c_i))
    }
}

#[async_trait]
impl<RNG, S, R> MTProvider for OtMTProvider<RNG, S, R>
where
    RNG: RngCore + CryptoRng + Send,
    S: ExtROTSender<Msg = R::Msg> + Send,
    S::Msg: RemoteSend + Debug,
    R: ExtROTReceiver + Send,
    R::Msg: RemoteSend + Debug,
{
    type Output = MulTriples;
    type Error = Error;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        let mts = self.compute_mts(amount).await?;
        self.precomputed_mts = Some(mts);
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<MulTriples, Self::Error> {
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
    use crate::mul_triple::boolean::ot_ext::OtMTProvider;
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

            let mut mtp = OtMTProvider::new(OsRng, ot_sender, ot_receiver, ch_sender, ch_receiver);
            mtp.request_mts(1024).await.unwrap()
        };

        let (mts1, mts2) = tokio::join!(
            party(ch_sender1, ch_receiver1),
            party(ch_sender2, ch_receiver2)
        );

        assert_eq!(mts1.c ^ mts2.c, (mts1.a ^ mts2.a) & (mts1.b ^ mts2.b))
    }
}
