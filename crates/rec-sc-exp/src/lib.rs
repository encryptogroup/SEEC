use std::collections::{hash_map, HashMap};
use std::sync::Arc;

struct BaseCircuit;

struct Circuit {
    base_circuits: Vec<BaseCircuit>,
    main: SubCircuit,
}

struct SubCircuit(Arc<SubCircInner>);

struct SubCircInner {
    base_circ_id: CircId, // this corresponds to the position in Circuits.base_circuits Vec
    sub_circuits: Vec<SubCircuit>, // position in this Vec is CallId
    call_map: CallMap,
}

struct CallId(u32);
struct CircId(u32);
struct GateId(u32);

struct CallGateId {
    call: CallId,
    gate: GateId,
}
struct CircGateId {
    circ: CircId,
    gate: GateId,
}

struct CallMap {
    map: HashMap<GateId, CallGateId>,
}

struct Storage<T> {
    own: Vec<T>,
    sub_circuits: Vec<Storage<T>>, // indexed by the call Id
}

impl Storage<()> {
    // clear storage and free associated memory. Should be called
    // once a sub-circuit is evaluated
    fn free() {}
}

struct CircuitLayer<'a, G> {
    call_id: CallId,
    gates: Vec<G>,
    gate_ids: Vec<GateId>,
    sub_circ_iter: &'a mut SubCircIter,
}

impl<'a, G> CircuitLayer<'a, G> {
    fn sub_layer_iter(&mut self) -> SubLayerIter<'_> {
        SubLayerIter {
            sub_circ_iter: self.sub_circ_iter.sub_circ_iters.iter_mut(),
        }
    }
}

struct SubLayerIter<'a> {
    sub_circ_iter: hash_map::IterMut<'a, CallId, SubCircIter>,
}

fn handle_layer<'a>(mut layer: CircuitLayer<'a, ()>, storage: &mut Storage<()>) {
    // do stuff with layer.gates
    // how do I associate a CircuitLayer with its correct storage destination?
    // Storage also needs to be a recursive datastructure
    let iter = layer.sub_layer_iter();
    for layer in iter {
        let storage = &mut storage.sub_circuits[layer.call_id.0 as usize];
        handle_layer(layer, storage);
    }
}

struct SubCircIter {
    own_base_iter: BaseIter,
    sub_circ_iters: HashMap<CallId, SubCircIter>,
    sub_circ: SubCircuit,
}

impl SubCircIter {
    fn next(&mut self) -> Option<CircuitLayer<'_, ()>> {
        // drive own base iter to get gates for this base circuit
        // check own gates in self.sub_circ call_map to see if they are outgoing
        // if yes, add new SubCircIter to self.sub_circ_iters or update iter state
        // with connected to gates
        Some(CircuitLayer {
            call_id: CallId(0),
            gates: vec![],
            gate_ids: vec![],
            sub_circ_iter: self,
        })
    }
}

impl<'a> Iterator for SubLayerIter<'a> {
    type Item = CircuitLayer<'a, ()>;
    fn next(&mut self) -> Option<Self::Item> {
        self.sub_circ_iter
            .next()
            .map(|(call_id, sc_iter)| sc_iter.next().map(|layer| layer))
            .flatten()
    }
}

struct BaseIter {
    // the same?
}

impl BaseIter {
    // mhhh, these methods are only needed if we can have a sub-circuit call directly from an
    // interactive gate. If we add a SubCircuitCall gate, which **must** be the from gate of a
    // call map, then we only need add to current layer.
    fn add_to_current_layer(&mut self, gate_id: GateId) {}
    fn add_to_next_layer(&mut self, gate_id: GateId) {}
}
