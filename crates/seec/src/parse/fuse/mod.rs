use crate::circuit::base_circuit::BaseGate;
use crate::parse::fuse::module_generated::fuse::ir::{
    root_as_module_table, CircuitTable, ModuleTable, NodeTable, PrimitiveOperation, PrimitiveType,
};
use crate::protocols::mixed_gmw::{ConvGate, MixedGate, MixedGmw, MixedShare};
use crate::protocols::{arithmetic_gmw, boolean_gmw, mixed_gmw, Ring, ScalarDim};
use crate::secret::Secret;
use crate::{Circuit, CircuitBuilder, GateId, SharedCircuit};
use ahash::{HashMap, HashMapExt};
use bitvec::view::BitViewSized;
use flatbuffers::InvalidFlatbuffer;
use rand::distributions::{Distribution, Standard};
use std::path::Path;
use std::{fs, io};
use thiserror::Error;
use tracing::{debug, instrument};

#[allow(unused_imports, dead_code, clippy::all)]
#[rustfmt::skip]
mod module_generated;

/// Plan:
/// - read fb file
/// - identify main circuit
/// - parse all **other** circuits, which should be sub-circuits, error on SCCall
/// - add SCs to builder, storing mapping of sc name -> shared_circ
/// - parse main circ, if sccall is encountered, use connect_sub_circuit and connect_to_main on sc

type BaseCircuit<R> = crate::circuit::BaseCircuit<MixedGate<R>>;

pub struct FuseConverter<R> {
    builder: CircuitBuilder<MixedGate<R>>,
    sc_map: HashMap<String, SharedCircuit<MixedGate<R>>>,
    call_mode: CallMode,
}

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("reading circuit file failed")]
    Read(#[from] io::Error),
    #[error("FUSE flatbuffer is invalid")]
    InvalidFlatbuffer(#[from] InvalidFlatbuffer),
    #[error("missing: {0}")]
    MissingField(&'static str),
}

impl<'a, R> TryFrom<ModuleTable<'a>> for Circuit<MixedGate<R>>
where
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
    type Error = ConversionError;

    fn try_from(module: ModuleTable<'a>) -> Result<Self, Self::Error> {
        let converter = FuseConverter::new(CallMode::CallCircuits);
        converter.convert_module(module)
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, PartialEq, Eq)]
pub enum CallMode {
    InlineCircuits,
    CallCircuits,
}

impl<R> FuseConverter<R>
where
    R: Ring,
    Standard: Distribution<R>,
    [R; 1]: BitViewSized,
{
    pub fn new(call_mode: CallMode) -> Self {
        Self {
            builder: CircuitBuilder::<MixedGate<R>>::new(),
            sc_map: HashMap::new(),
            call_mode,
        }
    }

    pub fn convert(self, path: impl AsRef<Path>) -> Result<Circuit<MixedGate<R>>, ConversionError> {
        let data = fs::read(path)?;
        let mod_table = root_as_module_table(&data)?;
        self.convert_module(mod_table)
    }

    fn convert_module(
        mut self,
        module: ModuleTable<'_>,
    ) -> Result<Circuit<MixedGate<R>>, ConversionError> {
        let ep = module.entry_point().unwrap_or("main");

        for c in module
            .circuits()
            .ok_or(ConversionError::MissingField("circuits"))?
            .iter()
        {
            let circ_table =
                c.circuit_buffer_nested_flatbuffer()
                    .ok_or(ConversionError::MissingField(
                        "circuit_buffer_nested_flatbuffer",
                    ))?;
            // filter out main circ from sub_circs
            if circ_table.name() == ep {
                continue;
            }
            self.add_fuse_sub_circ(circ_table, false)?;
        }

        // mostly duplicating this loop is a little ugly, but i was not able to save
        // the main circuit table buffer in the above loop due to lifetime errors
        for c in module.circuits().unwrap() {
            let circ_table =
                c.circuit_buffer_nested_flatbuffer()
                    .ok_or(ConversionError::MissingField(
                        "circuit_buffer_nested_flatbuffer",
                    ))?;

            if circ_table.name() == ep {
                self.add_fuse_sub_circ(circ_table, true)?;
                break;
            }
        }

        Ok(self.builder.into_circuit())
    }

    #[instrument(level = "debug", skip(self, circ), fields(name = circ.name()))]
    fn add_fuse_sub_circ(
        &mut self,
        circ: CircuitTable<'_>,
        is_main: bool,
    ) -> Result<(), ConversionError> {
        let mut res_c = BaseCircuit::new();
        let mut key_map = HashMap::with_capacity(circ.nodes().unwrap().len());
        for node in circ
            .nodes()
            .ok_or(ConversionError::MissingField("circuit.nodes"))?
            .iter()
        {
            let mapped_inps: Vec<_> = node
                .input_identifiers()
                .map(|inps| inps.iter().map(|inp_k| key_map[&inp_k]).collect())
                .unwrap_or_default();
            let gate_id = self.add_node(&mut res_c, node, &mapped_inps, is_main);
            key_map.entry(node.id()).or_insert(gate_id);
        }
        if is_main {
            *self.builder.get_main_circuit().lock() = res_c;
            return Ok(());
        }
        let shared = res_c.into_shared();
        self.sc_map.insert(circ.name().to_string(), shared.clone());
        Ok(())
    }

    fn add_node(
        &mut self,
        bc: &mut BaseCircuit<R>,
        node: NodeTable<'_>,
        inputs: &[GateId],
        in_main: bool,
    ) -> GateId {
        use arithmetic_gmw::ArithmeticGate as AG;
        use boolean_gmw::BooleanGate as BG;
        use mixed_gmw::MixedGate::*;
        use PrimitiveOperation as PO;
        let prim_op = node.operation();
        if prim_op.has_one_to_one_mapping() {
            let gate = match prim_op {
                PO::And => Bool(BG::And),
                PO::Xor => Bool(BG::Xor),
                PO::Not => Bool(BG::Inv),
                PO::Add => Arith(AG::Add),
                PO::Mul => Arith(AG::Mul),
                PO::Sub => Arith(AG::Sub),
                PO::Input if in_main => Base(BaseGate::Input(ScalarDim)),
                PO::Input if !in_main => Base(BaseGate::SubCircuitInput(ScalarDim)),
                PO::Output if in_main => Base(BaseGate::Output(ScalarDim)),
                PO::Output if !in_main => Base(BaseGate::SubCircuitOutput(ScalarDim)),
                PO::Constant => {
                    // TODO properly decode node.payload
                    match node.output_datatypes().map(|v| v.get(0)) {
                        None => Base(BaseGate::Constant(MixedShare::Bool(true))),
                        Some(dt) if dt.primitive_type() == PrimitiveType::Bool => {
                            Base(BaseGate::Constant(MixedShare::Bool(true)))
                        }
                        Some(dt) if dt.primitive_type() == PrimitiveType::Int32 => {
                            Base(BaseGate::Constant(MixedShare::Arith(R::ONE)))
                        }
                        Some(dt) => {
                            panic!("Unsupported constant datatype {dt:?}")
                        }
                    }
                }
                PO::Merge => Conv(ConvGate::B2A),
                other => panic!("One to one operation {other:?} without impl"),
            };
            return bc.add_wired_gate(gate, inputs);
        }

        match prim_op {
            PO::Split => {
                assert_eq!(1, inputs.len(), "Expecting 1 input for Split gate");
                let mut b_shares = mixed_gmw::a2b(bc, inputs[0]);
                // Insert identity gates so that output of a2b has contiguous gate ids
                for sh in &mut b_shares {
                    *sh = bc.add_wired_gate(Base(BaseGate::Identity), &[*sh]);
                }
                // we return the first one, if we encounter a node with an input offset, we
                // can retrieve this gate_id from the map and add the offset
                b_shares[0]
            }
            PO::CallSubcircuit if self.call_mode == CallMode::CallCircuits => {
                assert!(
                    in_main,
                    "Can only process CallSubcircuit nodes in main circuit"
                );
                let circ = self
                    .sc_map
                    .get(
                        node.subcircuit_name()
                            .expect("Missing subcircuit_name in CallSubcircuit Node"),
                    )
                    .expect("Missing subcircuit")
                    .clone();
                // TODO uaarrgh, really not sure if this is correct... this is basically
                //  a connect_to_main impl as in the SubcircuitOutput trait
                let sc_id = self.builder.push_circuit(circ);
                let inputs: Vec<_> = inputs
                    .iter()
                    .map(|gid| Secret::<MixedGmw<R>>::from_parts(0, *gid))
                    .collect();
                let outputs = self.builder.connect_sub_circuit(&inputs, sc_id);
                let main_inputs: Vec<_> = (0..outputs.len())
                    .map(|_| bc.add_gate(Base(BaseGate::SubCircuitInput(ScalarDim))))
                    .collect();
                self.builder
                    .connect_sub_circuit_gates(&outputs, bc, 0, &main_inputs);
                assert_eq!(
                    1,
                    main_inputs.len(),
                    "Currently only supporting CallSubcircuits with one output"
                );
                main_inputs[0]
            }
            PO::CallSubcircuit if self.call_mode == CallMode::InlineCircuits => {
                debug!(name = node.subcircuit_name(), "CallSubcircuit");
                let circ = self
                    .sc_map
                    .get(
                        node.subcircuit_name()
                            .expect("Missing subcircuit_name in CallSubcircuit Node"),
                    )
                    .expect("Missing subcircuit")
                    .clone();
                let circ = circ.lock();
                let outputs = bc.add_sub_circuit(&*circ, inputs.iter().copied());

                outputs[0]
            }
            other => unimplemented!("{other:?}"),
        }
    }
}

impl PrimitiveOperation {
    fn has_one_to_one_mapping(&self) -> bool {
        use PrimitiveOperation as PO;
        matches!(
            *self,
            PO::And
                | PO::Xor
                | PO::Not
                | PO::Add
                | PO::Sub
                | PO::Mul
                | PO::Input
                | PO::Output
                | PO::Constant
                | PO::Merge
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{DefaultIdx, ExecutableCircuit};
    use crate::common::BitVec;
    use crate::parse::fuse::module_generated::fuse::ir::root_as_module_table;
    use crate::parse::fuse::{CallMode, FuseConverter};
    use crate::private_test_utils::{execute_circuit, init_tracing, TestChannel, ToBool};
    use crate::protocols::mixed_gmw;
    use crate::protocols::mixed_gmw::{MixedGmw, MixedShareStorage, MixedSharing};
    use crate::Circuit;
    use rand::distributions::Standard;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;
    use std::fs;

    #[test]
    fn read_simple_fuse_fb() {
        let data = fs::read("test_resources/fuse-circuits/tutorial_addition.mfs").unwrap();
        let _mod_table = root_as_module_table(&data).expect("Deser fuse fb");
    }

    #[test]
    fn convert_simple_fuse() {
        let data = fs::read("test_resources/fuse-circuits/tutorial_addition.mfs").unwrap();
        let mod_table = root_as_module_table(&data).expect("Deser fuse fb");
        let _circ: Circuit<mixed_gmw::MixedGate<u32>> = mod_table.try_into().unwrap();
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

    #[tokio::test]
    async fn convert_and_execute_fuse_mnist() {
        let _g = init_tracing();
        let data = fs::read("test_resources/fuse-circuits/mnist_fuse.mfs").unwrap();
        let mod_table = root_as_module_table(&data).expect("Deser fuse fb");
        let converter = FuseConverter::<u32>::new(CallMode::InlineCircuits);
        let circ: Circuit<mixed_gmw::MixedGate<u32>> = converter
            .convert_module(mod_table)
            .expect("Fuse conversion");
        let ec = ExecutableCircuit::DynLayers(circ);
        let inputs: Vec<u32> = ChaChaRng::seed_from_u64(46456315)
            .sample_iter(Standard)
            .take(ec.input_count())
            .collect();
        let out = execute_circuit::<MixedGmw<u32>, DefaultIdx, MixedSharing<_, _, u32>>(
            &ec,
            inputs,
            TestChannel::InMemory,
        )
        .await
        .unwrap();
        // TODO, this is very likely not the correct output
        let exp = vec![u32::MAX; 10];
        let exp = MixedShareStorage::Arith(exp);
        assert_eq!(out, exp)
    }
}
