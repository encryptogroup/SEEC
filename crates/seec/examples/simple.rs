//! This example shows how to use the high-level Secret api to create a circuit and then
//! execute it with the messages being exchanged via Tcp.
//! Run the example via `cargo run --example`.
//!
//! To view the logging output, set the environment variable `RUST_LOG` as specified
//! [here](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/struct.EnvFilter.html).
use std::time::Duration;

use anyhow::Result;
use bitvec::bitvec;
use bitvec::prelude::*;
use tokio::time::sleep;
use tracing_subscriber::EnvFilter;

use seec::channel::sub_channels_for;
use seec::circuit::{dyn_layers::Circuit, DefaultIdx, ExecutableCircuit};
use seec::executor::{Executor, Input, Message};
use seec::mul_triple::boolean::insecure_provider::InsecureMTProvider;
use seec::mul_triple::MTProvider;
use seec::protocols::boolean_gmw::BooleanGmw;
use seec::secret::inputs;
use seec::CircuitBuilder;

fn build_circuit() {
    // The `inputs` method is a convenience method to create n input gates for the circuit.
    // It returns a Vec<Secret>. In the following, we use try_into() to convert it into
    // an array to destructure it
    let [a, b, c, d]: [_; 4] = inputs::<DefaultIdx>(4).try_into().unwrap();
    // a,b,c,d are `Secret`s representing the output share of a gate. They support
    // the standard std::ops traits like BitAnd and BitXor (and their Assign variants) which
    // are used to implicitly build the circuit.

    // Creates a new Xor gate with the input of a and b. The output is a new Secret
    // representing the output of the new gate.
    let xor = a ^ b;
    // To use a Secret multiple times (connect to gate represented by it to multiple
    // different ones), simply use a reference to it (only possibile on the rhs).
    let and = c & &d;
    // we can still use d but not c
    let mut tmp = and ^ d;
    // BitAnd and BitXor are also supported, as is Not. See the Secret documentation for
    // all operations.
    tmp &= !xor;
    // `output()` consumes the Secret and creates a new Output gate with its output.
    // It returns the gate_id of the Output gate (this is usually not needed).
    let _out_gate_id = tmp.output();
}

#[tracing::instrument(skip(circuit), err)]
async fn party(circuit: Circuit, party_id: usize) -> Result<bool> {
    // Create a new insecure MTProvider. This will simply provide both parties with multiplication
    // triples consisting of zeros. The sha256 and trusted_party_mts examples show how to use
    // a trusted provider. This example also shows how to use dynamic dispatch for the MTProvider.
    let mt_provider: Box<dyn MTProvider<Output = _, Error = _> + Send> =
        InsecureMTProvider::default().into_dyn();
    // Create a new Executor for the circuit. It's important that the party_id is either 0 or 1,
    // as otherwise wrong results might be computed. In the future, there might be support for more
    // than two parties
    let exec_circuit = ExecutableCircuit::DynLayers(circuit);
    let mut executor = Executor::<BooleanGmw, _>::new(&exec_circuit, party_id, mt_provider).await?;

    // Create inputs as a BitVector. The inputs will be used for the input gate with the
    // corresponding index.
    let inputs = bitvec![usize, Lsb0; 0, 1, 1, 0];
    // `circuit.input_count()` input count can be used to determine how big the input should be
    assert_eq!(exec_circuit.input_count(), inputs.len());

    // When using the Tcp transport, one party is essentially a server and needs to `listen` for
    // new connections. The other party then `connect`s to it. If party 1 connects before party 0
    // listens, an error will be returned.
    let (mut sender, _bytes_written, mut receiver, _bytes_read) = match party_id {
        0 => seec_channel::tcp::listen("127.0.0.1:7766").await?,
        1 => seec_channel::tcp::connect("127.0.0.1:7766").await?,
        illegal => anyhow::bail!("Illegal party id {illegal}. Must be 0 or 1."),
    };

    let (mut sender, mut receiver) =
        sub_channels_for!(&mut sender, &mut receiver, 8, Message<BooleanGmw>).await?;
    // Execute the circuit and await its result (in the form of a BitVec)
    let output = executor
        .execute(Input::Scalar(inputs), &mut sender, &mut receiver)
        .await?
        .into_scalar()
        .unwrap();

    assert_eq!(exec_circuit.output_count(), output.len());
    // As there is only one output gate, we simply return the first element
    Ok(output[0])
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging, see top of file for instructions on how to get output.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Create the Circuit. The operations on the Secret will use a lazily initialized
    // global CircuitBuilder from which we can get the constructed Circuit
    build_circuit();
    // Save the circuit in .dot format for easy inspection and debugging
    // circuit.lock().save_dot("examples/simple-circuit.dot")?;
    // In an actual setting, we would have two parties on different hosts, each constructing the
    // same circuit and evaluating it. To simulate that, we convert the shared Circuit into
    // an owned Circuit and clone it. The conversion will fail if there are any outstanding clones
    // of the SharedCircuit, e.g. when there are Secrets still alive.
    // Each party has their own but identical circuit.
    let circuit_party_0: Circuit = CircuitBuilder::global_into_circuit();
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
