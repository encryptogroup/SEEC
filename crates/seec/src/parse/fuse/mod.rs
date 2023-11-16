use crate::circuit::base_circuit::BaseGate;
use crate::circuit::BaseCircuit;
use crate::parse::fuse::module_generated::fuse::ir::{
    CircuitTable, ModuleTable, PrimitiveOperation,
};
use crate::protocols::{arithmetic_gmw, boolean_gmw, mixed_gmw, Ring, ScalarDim};
use crate::{Circuit, CircuitBuilder, GateId};
use ahash::{HashMap, HashMapExt};

mod module_generated;

/// Plan:
/// - read fb file
///

impl<'a, R: Ring> TryFrom<ModuleTable<'a>> for Circuit<mixed_gmw::MixedGate<R>> {
    type Error = ();

    fn try_from(module: ModuleTable<'a>) -> Result<Self, Self::Error> {
        let mut builder = CircuitBuilder::<mixed_gmw::MixedGate<R>>::new();
        let ep = module.entry_point().unwrap_or("main");
        let main_circ = module
            .circuits()
            .expect("Missing circs")
            .iter()
            .filter(|c| {
                let c = c
                    .circuit_buffer_nested_flatbuffer()
                    .expect("Missing nested circuit_table");
                c.name() == ep
            })
            .next()
            .expect("Missing main circ");
        let main_circ = main_circ
            .circuit_buffer_nested_flatbuffer()
            .unwrap()
            .try_into()
            .unwrap();
        *builder.get_main_circuit().lock() = main_circ;
        Ok(builder.into_circuit())
    }
}

// TODO this is potentially not expressible using From as we might need state
impl<'a, R: Ring> TryFrom<CircuitTable<'a>> for BaseCircuit<mixed_gmw::MixedGate<R>> {
    type Error = ();

    fn try_from(circ: CircuitTable<'a>) -> Result<Self, Self::Error> {
        let mut res_c = BaseCircuit::new();
        let mut key_map = HashMap::with_capacity(circ.nodes().unwrap().len());
        for (idx, node) in circ.nodes().expect("No nodes").iter().enumerate() {
            key_map.entry(node.id()).or_insert(idx);
            let gate = node.operation().try_into()?;
            if let Some(inps) = node.input_identifiers() {
                let mapped_inps: Vec<_> = inps
                    .iter()
                    .map(|inp_k| GateId::from(key_map[&inp_k]))
                    .collect();
                res_c.add_wired_gate(gate, &mapped_inps);
            } else {
                res_c.add_gate(gate);
            }
        }
        Ok(res_c)
    }
}

impl<R> TryFrom<PrimitiveOperation> for mixed_gmw::MixedGate<R> {
    type Error = ();

    fn try_from(value: PrimitiveOperation) -> Result<Self, Self::Error> {
        use arithmetic_gmw::ArithmeticGate as AG;
        use boolean_gmw::BooleanGate as BG;
        use mixed_gmw::MixedGate::*;
        use PrimitiveOperation as PO;
        Ok(match value {
            PO::Custom => {
                todo!()
            }
            PO::And => Bool(BG::And),
            PO::Xor => Bool(BG::Xor),
            PO::Not => {
                todo!()
            }
            PO::Or => {
                todo!()
            }
            PO::Nand => {
                todo!()
            }
            PO::Nor => {
                todo!()
            }
            PO::Xnor => {
                todo!()
            }
            PO::Gt => {
                todo!()
            }
            PO::Ge => {
                todo!()
            }
            PO::Lt => {
                todo!()
            }
            PO::Le => {
                todo!()
            }
            PO::Eq => {
                todo!()
            }
            PO::Add => Arith(AG::Add),
            PO::Mul => Arith(AG::Mul),
            PO::Div => {
                todo!()
            }
            PO::Neg => {
                todo!()
            }
            PO::Sub => {
                todo!()
            }
            PO::Loop => {
                todo!()
            }
            PO::CallSubcircuit => {
                todo!()
            }
            PO::Merge => {
                todo!()
            }
            PO::Input => Base(BaseGate::Input(ScalarDim)),
            PO::Output => Base(BaseGate::Output(ScalarDim)),
            PO::Constant => {
                todo!()
            }
            PO::Split => {
                todo!()
            }
            PO::Mux => {
                todo!()
            }
            PO::SelectOffset => {
                todo!()
            }
            PO::Square => {
                todo!()
            }
            other => panic!("Illegal PrimitveOperation {other:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{DefaultIdx, ExecutableCircuit};
    use crate::common::BitVec;
    use crate::parse::fuse::module_generated::fuse::ir::root_as_module_table;
    use crate::private_test_utils::{execute_circuit, TestChannel, ToBool};
    use crate::protocols::mixed_gmw;
    use crate::protocols::mixed_gmw::{MixedGmw, MixedShareStorage, MixedSharing};
    use crate::Circuit;
    use std::fs;

    #[test]
    fn read_simple_fuse_fb() {
        let data = fs::read("test_resources/fuse-circuits/tutorial_addition.mfs").unwrap();
        let mod_table = root_as_module_table(&data).expect("Deser fuse fb");
    }

    #[test]
    fn convert_simple_fuse() {
        let data = fs::read("test_resources/fuse-circuits/tutorial_addition.mfs").unwrap();
        let mod_table = root_as_module_table(&data).expect("Deser fuse fb");
        let circ: Circuit<mixed_gmw::MixedGate<u32>> = mod_table.try_into().unwrap();
    }

    #[tokio::test]
    async fn convert_and_execute_simple_fuse() {
        let data = fs::read("test_resources/fuse-circuits/tutorial_addition.mfs").unwrap();
        let mod_table = root_as_module_table(&data).expect("Deser fuse fb");
        let circ: Circuit<mixed_gmw::MixedGate<u32>> = mod_table.try_into().unwrap();
        let ec = ExecutableCircuit::DynLayers(circ);
        let out = execute_circuit::<MixedGmw<u32>, DefaultIdx, MixedSharing<_, _, u32>>(
            &ec,
            (ToBool(42), ToBool(666)),
            TestChannel::InMemory,
        )
        .await
        .unwrap();
        let mut exp = BitVec::from_element(42 + 666);
        exp.truncate(ec.output_count());
        let exp = MixedShareStorage::Bool(exp);
        assert_eq!(out, exp)
    }
}
