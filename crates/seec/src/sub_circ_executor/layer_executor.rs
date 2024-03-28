use crate::circuit::sub_circuit::{SubCircIter, SubCircLayer};
use crate::circuit::{base_circuit, GateIdx};
use crate::executor::DynFDSetup;
use crate::protocols::{Protocol, ShareStorage};
use crate::sub_circ_executor::storage::Storage;
use std::mem;
use std::ops::Range;

pub enum LayerExecutor<P: Protocol, Idx: GateIdx> {
    ConstructMsg {
        base_layer: base_circuit::CircuitLayer<P::Gate, Idx>,
        sub_executors: Vec<Self>,
    },
    ProcessMsg {
        base_layer: base_circuit::CircuitLayer<P::Gate, Idx>,
        setup: P::SetupStorage,
        sub_executors: Vec<Self>,
        msg_range: Range<usize>,
    },
}

impl<P: Protocol, Idx: GateIdx> LayerExecutor<P, Idx> {
    async fn construct_msg(
        &mut self,
        layer: &mut SubCircLayer<'_, '_, P::Gate, Idx>,
        storage: &mut Storage<P::ShareStorage>,
        setup: &mut DynFDSetup<'_, P, Idx>,
        msg_buf: &mut (),
    ) {
        match self {
            LayerExecutor::ConstructMsg {
                base_layer,
                sub_executors,
            } => {
                // handle the non-interactive part of base_layer
                Self::handle_non_interactive_base_layer(base_layer, storage);
                base_layer.drop_non_interactive();

                // drop non-interactive part
                let setup_data = setup.request_setup_output(42).await.unwrap();
                // handle interactive part, write msg to msg_buf -> record msg offset
                // change self type to process_msg

                for mut sub_layer in layer.sub_layer_iter() {
                    let mut sub_executor = LayerExecutor::ConstructMsg {
                        base_layer: mem::take(&mut sub_layer.base_layer),
                        sub_executors: vec![],
                    };
                    let sub_storage = storage.sub_storage_mut(sub_layer.call_id);
                    sub_executor
                        .construct_msg(&mut sub_layer, sub_storage, setup, msg_buf)
                        .await;
                    sub_executors.push(sub_executor);
                }
            }
            LayerExecutor::ProcessMsg { .. } => {
                panic!("Call construct_msg before process_msg")
            }
        }
        todo!()
    }

    fn handle_non_interactive_base_layer(
        base_layer: &base_circuit::CircuitLayer<P::Gate, Idx>,
        storage: &mut Storage<P::ShareStorage>,
    ) {
        todo!()
    }

    fn process_msg(
        &mut self,
        storage: &mut Storage<P::ShareStorage>,
        own_msg_buf: &mut (),
        other_msg_buf: &mut (),
    ) {
        todo!()
    }
}

// impl LayerExecutor {
//     fn handle_before() {
//         handle_non_interactive(layer)
//         let msg_idx = handle_interactive(layer, &mut msg_buf)
//         for sub_layer in layer:
//             sub_executor = LayerExecutor::BeforeInteractive { layer: sub_layer, ... }
//             sub_executor.handle_before()
//             self.sub_executors.push(sub_executor)
//     }
