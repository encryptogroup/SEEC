use crate::Storage;
use bitvec::order::Lsb0;
use bitvec::store::BitStore;
use bitvec::vec::BitVec;
use bitvec::view::BitView;
use std::mem;

/// Simple transpose implementation using BitVec that is used to check other impls
pub(crate) fn transpose<T: Storage + BitStore>(input: &[T], rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(
        rows * cols,
        mem::size_of_val(input) * 8,
        "input has wrong length"
    );
    let input = input.view_bits::<Lsb0>();
    let mut output: BitVec<T, Lsb0> = BitVec::repeat(false, rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let bit = input[idx];
            let t_idx = col * rows + row;
            output.set(t_idx, bit);
        }
    }
    output.into_vec()
}
