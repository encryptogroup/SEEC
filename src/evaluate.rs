use crate::circuit;
use crate::common::{BitSlice, BitVec};
use crate::mult_triple::MultTriple;
use bitvec::ptr::BitRef;
use std::ops;

impl circuit::Xor {
    pub fn evaluate(x: bool, y: bool) -> bool {
        x ^ y
    }
}

impl circuit::And {
    pub fn compute_shares(x: bool, y: bool, mt: &MultTriple) -> (bool, bool) {
        let d = x ^ mt.get_a();
        let e = y ^ mt.get_b();
        (d, e)
    }

    pub fn evaluate(d: [bool; 2], e: [bool; 2], mt: MultTriple) -> bool {
        let d = d[0] ^ d[1];
        let e = e[0] ^ e[1];
        d & mt.get_b() ^ e & mt.get_a() ^ mt.get_c() ^ d & e
    }
}
