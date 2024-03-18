use std::marker::PhantomData;

trait Share {}

trait SimdShare: Share {
    type Scalar: Share;
}

trait Gate {}

struct Sender<T>(PhantomData<T>);

struct Receiver<T>(PhantomData<T>);

struct Context;

// impl Context {
//     async fn get_setup_data(&self)
// }

// TODO: what about preprocessing information?
//  in original, this is passed to the methods, but I'm not super happy with it
//  maybe I can have a more general ExecutionContext, which allows access to preprocessing data?
trait Protocol<G: Gate> {
    type Share: Share;
    type SimdShare: SimdShare<Scalar = Self::Share>;

    type SetupData;

    type Msg;

    fn eval_non_interactive(
        &self,
        gate: G,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Share;

    fn eval_interactive_msg(
        &self,
        gate: G,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Msg;
    fn eval_interactive(
        &self,
        gate: G,
        inputs: impl IntoIterator<Item = Self::Share>,
        msg: Self::Msg,
        other_msg: Self::Msg,
    ) -> Self::Share;

    // todo replicate for SIMD, what about async?
    async fn simd_eval_interactive_msg(
        &self,
        gate: G,
        inputs: impl IntoIterator<Item = Self::Share>,
        sender: &mut Sender<Self::Msg>,
    );
    // TODO: what about access to own sent message?
    async fn simd_eval_interactive(
        &self,
        gate: G,
        inputs: impl IntoIterator<Item = Self::Share>,
        receiver: &mut Sender<Self::Msg>,
    );
}

struct Gmw<G>(PhantomData<G>);

struct BoolGate;

impl Share for bool {}

struct BitVec;

impl Share for BitVec {}

impl SimdShare for BitVec {
    type Scalar = bool;
}

impl Gate for BoolGate {}

impl Protocol<BoolGate> for Gmw<BoolGate> {
    type Share = bool;
    type SimdShare = BitVec;
    type Msg = ();

    fn eval_non_interactive(
        &self,
        gate: BoolGate,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Share {
        todo!()
    }

    fn eval_interactive_msg(
        &self,
        gate: BoolGate,
        inputs: impl IntoIterator<Item = Self::Share>,
    ) -> Self::Msg {
        todo!()
    }

    fn eval_interactive(
        &self,
        gate: BoolGate,
        inputs: impl IntoIterator<Item = Self::Share>,
        msg: Self::Msg,
        other_msg: Self::Msg,
    ) -> Self::Share {
        todo!()
    }
}
