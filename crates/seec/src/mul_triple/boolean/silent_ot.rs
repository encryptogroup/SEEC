use crate::common::BitVec;
use crate::mul_triple::boolean::MulTriples;
use crate::mul_triple::MTProvider;
use crate::protocols::SetupStorage;
use async_trait::async_trait;
use rand::rngs::OsRng;
use rand::SeedableRng;
use rand_chacha::rand_core::{CryptoRng, RngCore};
use rand_chacha::ChaChaRng;
use std::thread::available_parallelism;
use zappot::silent_ot;
use zappot::silent_ot::MultType;

pub type Msg = silent_ot::Msg;

pub struct SilentMtProvider<Rng> {
    rng: Rng,
    configured_ots: usize,
    stored_mts: Option<MulTriples>,
    silent_sender: Option<silent_ot::Sender>,
    silent_receiver: Option<silent_ot::Receiver>,
    ch1: Option<seec_channel::Channel<silent_ot::Msg>>,
    ch2: Option<seec_channel::Channel<silent_ot::Msg>>,
}

impl<Rng: RngCore + CryptoRng + Send> SilentMtProvider<Rng> {
    /// Executes base OTs for silent OT but not num_ots silentOT itself. `Rng` is used to seed
    /// ChaChaRng's.
    /// When the `silent-ot` feature is enabled, the [`MultType::ExConv7x25`] code from libOTe is used by default.
    /// If only `silent-ot-quasi-cyclic` is enabled, ZappOT wil not depend on libOTe and the [`MultType::QuasiCyclic`]
    /// code is used. Note that this code depends on AVX2 support, and therefore can't be use on ARM.
    pub async fn new(
        num_ots: usize,
        rng: Rng,
        ch1: seec_channel::Channel<silent_ot::Msg>,
        ch2: seec_channel::Channel<silent_ot::Msg>,
    ) -> Self {
        #[cfg(feature = "silent-ot")]
        {
            Self::new_with_mult_type(num_ots, MultType::ExConv7x24, rng, ch1, ch2).await
        }
        // not silent_ot implies silent-ot-quasi-cyclic
        #[cfg(not(feature = "silent-ot"))]
        {
            Self::new_with_mult_type(num_ots, MultType::QuasiCyclic { scaler: 2 }, rng, ch1, ch2)
                .await
        }
    }

    pub async fn new_with_mult_type(
        num_ots: usize,
        mul_type: MultType,
        mut rng: Rng,
        mut ch1: seec_channel::Channel<silent_ot::Msg>,
        mut ch2: seec_channel::Channel<silent_ot::Msg>,
    ) -> Self {
        let mut rng1 = ChaChaRng::from_rng(&mut rng).expect("Seeding Rng in SilentMtProvider::new");
        let mut rng2 = ChaChaRng::from_rng(&mut rng).expect("Seeding Rng in SilentMtProvider::new");
        let threads_per_ot = available_parallelism().map(usize::from).unwrap_or(2) / 2;
        let (silent_sender, silent_receiver) = tokio::join!(
            silent_ot::Sender::new_with_base_ot_sender(
                zappot::base_ot::Sender::new(),
                &mut rng1,
                num_ots,
                mul_type,
                threads_per_ot,
                &mut ch1.0,
                &mut ch1.1
            ),
            silent_ot::Receiver::new_with_base_ot_receiver(
                zappot::base_ot::Receiver::new(),
                &mut rng2,
                num_ots,
                mul_type,
                threads_per_ot,
                &mut ch2.0,
                &mut ch2.1
            ),
        );
        Self {
            rng,
            configured_ots: num_ots,
            stored_mts: None,
            silent_sender: Some(silent_sender),
            silent_receiver: Some(silent_receiver),
            ch1: Some(ch1),
            ch2: Some(ch2),
        }
    }

    pub async fn precompute_mts(&mut self) {
        let silent_sender = self
            .silent_sender
            .take()
            .expect("precompute_mts can only be called once");
        let silent_receiver = self.silent_receiver.take().unwrap();
        let ch1 = self.ch1.take().unwrap();
        let ch2 = self.ch2.take().unwrap();
        let send = silent_sender.random_silent_send(&mut self.rng, ch1.0, ch1.1);

        let receive = silent_receiver.random_silent_receive(ch2.0, ch2.1);

        let (send_ots, (recv_ots, a_i)) = tokio::join!(send, receive);
        eprintln!("{send_ots:?}");
        eprintln!("{recv_ots:?}");
        eprintln!("{a_i:?}");

        let mut b_i = BitVec::with_capacity(self.configured_ots);
        let mut v_i: BitVec<usize> = BitVec::with_capacity(self.configured_ots);

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

        self.stored_mts = Some(MulTriples::from_raw(a_i, b_i, c_i));
    }

    pub fn mts_available(&self) -> usize {
        self.stored_mts.as_ref().map(|mts| mts.len()).unwrap_or(0)
    }
}

impl SilentMtProvider<OsRng> {
    pub fn from_raw_mts(mts: MulTriples) -> Self {
        Self {
            rng: OsRng,
            configured_ots: mts.len(),
            stored_mts: Some(mts),
            silent_sender: None,
            silent_receiver: None,
            ch1: None,
            ch2: None,
        }
    }
}

#[async_trait]
impl<Rng: RngCore + CryptoRng + Send> MTProvider for SilentMtProvider<Rng> {
    type Output = MulTriples;
    type Error = ();

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        assert_eq!(
            amount, self.configured_ots,
            "Requested OTs must be equal to configured OTs of SilentMtProvider"
        );
        self.precompute_mts().await;
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        if let Some(stored_mts) = &mut self.stored_mts {
            return Ok(stored_mts.split_off_last(amount));
        }
        self.precompute_mts().await;
        self.request_mts(amount).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn silent_mts() {
        let (ch11, ch21) = seec_channel::in_memory::new_pair(128);
        let (ch12, ch22) = seec_channel::in_memory::new_pair(128);
        let (mut mtp1, mut mtp2) = tokio::join!(
            SilentMtProvider::new(100, OsRng, ch11, ch22),
            SilentMtProvider::new(100, OsRng, ch12, ch21)
        );

        let (mts1, mts2) = tokio::try_join!(mtp1.request_mts(100), mtp2.request_mts(100),).unwrap();
        let left = mts1.c ^ mts2.c;
        let right = (mts1.a ^ mts2.a) & (mts1.b ^ mts2.b);
        assert_eq!(left, right);
    }
}
