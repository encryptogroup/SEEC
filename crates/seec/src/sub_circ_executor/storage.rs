use crate::circuit::sub_circuit::CallId;
use crate::circuit::GateIdx;
use crate::GateId;
use serde::{Deserialize, Serialize};
use std::ops::Index;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalarOrSimd<Shares> {
    Scalar(Shares),
    Simd(Vec<Shares>),
}

pub struct Storage<Shares> {
    own: ScalarOrSimd<Shares>,
    sub_circuit: Vec<Storage<Shares>>,
}

impl<Shares> Storage<Shares> {
    pub(crate) fn sub_storage_mut(&mut self, call_id: CallId) -> &mut Self {
        let CallId::Call(id) = call_id else {
            panic!("Can't index storage with CallId::Ret")
        };
        &mut self.sub_circuit[id as usize]
    }
}
