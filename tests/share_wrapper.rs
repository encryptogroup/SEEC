use crate::common::init_tracing;
use gmw_rs::circuit::Circuit;
use gmw_rs::common::BitVec;
use gmw_rs::executor::Executor;
use gmw_rs::share_wrapper::{inputs, low_depth_reduce};
use gmw_rs::transport::InMemory;
use std::cell::RefCell;
use std::rc::Rc;

mod common;

#[tokio::test]
async fn and_tree() {
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
    let mut ex1 = Executor::new(and_tree, 0);
    let mut ex2 = Executor::new(and_tree, 1);

    let (t1, t2) = InMemory::new_pair();
    let h1 = async move { ex1.execute(inputs_0, t1).await };
    let h2 = async move { ex2.execute(inputs_1, t2).await };
    let (out1, out2) = futures::join!(h1, h2);
    let (out1, out2) = (out1.unwrap(), out2.unwrap());
    assert_eq!(exp_output, out1 ^ out2);
}
