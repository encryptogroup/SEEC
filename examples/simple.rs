//! This example shows how to use the high-level ShareWrapper api to create a circuit and then
//! execute it with the messages being exchanged via Tcp.
//! Run the example via `cargo run --example`.
//!
//! To view the logging output, set the environment variable `RUST_LOG` as specified
//! [here](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/struct.EnvFilter.html).
use anyhow::Result;
use bitvec::prelude::*;

use bitvec::bitvec;
use gmw_rs::circuit::Circuit;
use gmw_rs::executor::Executor;
use gmw_rs::mul_triple::insecure_provider::InsecureMTProvider;
use gmw_rs::share_wrapper::{inputs, ShareWrapper};
use gmw_rs::transport::Tcp;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use tokio::time::sleep;
use tracing_subscriber::EnvFilter;

fn build_circuit(circuit: Rc<RefCell<Circuit>>) {
    // The `inputs` method is a convenience method to create n input gates for the circuit.
    // It returns a Vec<ShareWrapper>. In the following, we use try_into() to convert it into
    // an array to destructure it
    let [a, b, c, d]: [ShareWrapper<_>; 4] = inputs(circuit, 4).try_into().unwrap();
    // a,b,c,d are `ShareWrapper`s representing the output share of a gate. They support
    // the standard std::ops traits like BitAnd and BitXor (and their Assign variants) which
    // are used to implicitly build the circuit.

    // Creates a new Xor gate with the input of a and b. The output is a new ShareWrapper
    // representing the output of the new gate.
    let xor = a ^ b;
    // To use a ShareWrapper multiple times (connect to gate represented by it to multiple
    // different ones), it must be cloned.
    let and = c & d.clone();
    // we can still use d but not c
    let mut tmp = and ^ d;
    // BitAnd and BitXor are also supported, as is Not. See the ShareWrapper documentation for
    // all operations.
    tmp &= !xor;
    // `output()` consumes the ShareWrapper and creates a new Output gate with its output.
    // It returns the gate_id of the Output gate (this is usually not needed).
    let _out_gate_id = tmp.output();
}

#[tracing::instrument(skip(circuit), err)]
async fn party(circuit: Circuit, party_id: usize) -> Result<bool> {
    // Create a new insecure MTProvider. This will simply provide both parties with multiplication
    // triples consisting of zeros. The sha256 and trusted_party_mts examples show how to use
    // a trusted provider.
    let mt_provider = InsecureMTProvider::default();
    // Create a new Executor for the circuit. It's important that the party_id is either 0 or 1,
    // as otherwise wrong results might be computed. In the future, there might be support for more
    // than two parties
    let mut executor = Executor::new(&circuit, party_id, mt_provider).await?;

    // Create inputs as a BitVector. The inputs will be used for the input gate with the
    // corresponding index.
    let inputs = bitvec![u8, Lsb0; 0, 1, 1, 0];
    // `circuit.input_count()` input count can be used to determine how big the input should be
    assert_eq!(circuit.input_count(), inputs.len());

    // When using the Tcp transport, one party is essentially a server and needs to `listen` for
    // new connections. The other party then `connect`s to it. If party 1 connects before party 0
    // listens, an error will be returned.
    let tcp_transport = if party_id == 0 {
        Tcp::listen(("127.0.0.1", 7766)).await?
    } else {
        Tcp::connect(("127.0.0.1", 7766)).await?
    };

    // Execute the circuit and await its result (in the form of a BitVec)
    let output = executor.execute(inputs, tcp_transport).await?;

    assert_eq!(circuit.output_count(), output.len());
    // As there is only one output gate, we simply return the first element
    Ok(output[0])
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging, see top of file for instructions on how to get output.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let circuit = Circuit::new();
    // Currently you need to explicitly place the Circuit in a Rc<RefCell<>> to use the
    // ShareWrappers. This will likely change in the future.
    let circuit = Rc::new(RefCell::new(circuit));
    // Use the build_circuit method to construct the circuit. Note that because of the usage of
    // Rc, a reference counted smart pointer to express shared ownership, the operations on the
    // ShareWrappers in the build_circuit method change the circuit defined above
    build_circuit(Rc::clone(&circuit));
    // Save the circuit in .dot format for easy inspection and debugging
    circuit.borrow().save_dot("examples/simple-circuit.dot")?;
    // In an actual setting, we would have two parties on different hosts, each constructing the
    // same circuit and evaluating it. To simulate that, we convert the shared Circuit into
    // an owned Circuit and clone it. Each party has their own but identical circuit.
    let circuit_party_0 = Rc::try_unwrap(circuit).unwrap().into_inner();
    let circuit_party_1 = circuit_party_0.clone();
    // Spawn separate tasks for each party
    let party0 = tokio::spawn(async { party(circuit_party_0, 0).await.unwrap() });
    // Sleep a little to ensure the server is started. In practice party 1 should retry the
    // connection
    sleep(Duration::from_millis(100)).await;
    let party1 = tokio::spawn(async { party(circuit_party_1, 1).await.unwrap() });
    // Join the handles returned by tokio::spawn to wait for the completion of the protocol
    let (out_0, out_1) = tokio::try_join!(party0, party1)?;
    // Xor the individual shares to get the final output
    let out = out_0 ^ out_1;
    println!("Output of the circuit: {out}");
    Ok(())
}
