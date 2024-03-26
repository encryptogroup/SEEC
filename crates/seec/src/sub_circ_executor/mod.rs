use crate::circuit::sub_circuit::SubCircIter;
use crate::circuit::{sub_circuit, GateIdx};
use crate::errors::ExecutorError;
use crate::executor::DynFDSetup;
use crate::protocols::Protocol;
use crate::sub_circ_executor::storage::{ScalarOrSimd, Storage};
use seec_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

mod layer_executor;
mod storage;

pub struct PartyId(pub usize);

pub struct Executor<'c, P: Protocol, Idx> {
    circuit: &'c sub_circuit::Circuit<P::Gate, Idx>,
    setup: DynFDSetup<'c, P, Idx>,
    layer_executor: LayerExecutor<P>,
}

struct LayerExecutor<P: Protocol> {
    protocol_state: P,
    storage: Storage<P::ShareStorage>,
    party_id: PartyId,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ExecutorMsg<Msg, SimdMsg> {
    scalar: Msg,
    simd: Option<SimdMsg>,
}

pub type Message<P> = ExecutorMsg<<P as Protocol>::Msg, <P as Protocol>::SimdMsg>;

impl<'c, P: Protocol, Idx: GateIdx> Executor<'c, P, Idx> {
    #[tracing::instrument(skip_all, fields(party_id = %self.layer_executor.party_id), err)]
    pub async fn execute(
        &mut self,
        inputs: ScalarOrSimd<P::ShareStorage>,
        sender: &mut Sender<Message<P>>,
        receiver: &mut Receiver<Message<P>>,
    ) -> Result<ScalarOrSimd<P::ShareStorage>, ExecutorError> {
        let mut main_iter = SubCircIter::new_main(&self.circuit);

        while let Some(layer) = main_iter.next() {}
        todo!()
    }
}

impl<P: Protocol> LayerExecutor<P> {
    fn handle_layer(&mut self) {}
}

impl Display for PartyId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
