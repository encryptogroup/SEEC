use anyhow::Result;
use gmw_rs::circuit::Circuit;
use gmw_rs::common::BitVec;
use gmw_rs::private_test_utils::{execute_circuit, init_tracing, TestTransport};
use gmw_rs::share_wrapper::{inputs, low_depth_reduce};
use std::cell::RefCell;
use std::rc::Rc;

#[tokio::test]
async fn and_tree() -> Result<()> {
    // TODO assert depth
    let _guard = init_tracing();
    let and_tree = Rc::new(RefCell::new(Circuit::<u16>::new()));
    let inputs = inputs(and_tree.clone(), 23);
    low_depth_reduce(&inputs, std::ops::BitAnd::bitand)
        .unwrap()
        .output();
    and_tree
        .borrow()
        .save_dot("tests/circuit-graphs/and_tree.dot")
        .unwrap();

    let input_count = and_tree.borrow().input_count();
    let inputs_0 = BitVec::repeat(false, input_count);
    let inputs_1 = BitVec::repeat(true, input_count);

    let exp_output: BitVec = BitVec::repeat(true, 1);
    let and_tree = &and_tree.borrow();
    let out = execute_circuit(and_tree, (inputs_0, inputs_1), TestTransport::Tcp).await?;
    assert_eq!(exp_output, out);
    Ok(())
}
