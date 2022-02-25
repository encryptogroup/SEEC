use crate::common::BitVec;
use bitvec::prelude::*;
use bitvec::{bitarr, bitvec};

pub struct MultTriple {
    // holds a single u8, since we only need 3 bits (Note: 5 bits are wasted here, this could
    // maybe be optimized by not having individual MultTriple structs but storing multiple triples
    // continuously
    /// Stores data as [c, a, b]
    data: BitArray<[u8; 1]>,
}

impl MultTriple {
    pub fn zeroes() -> Self {
        let zeroes = bitarr![u8, Lsb0; 0;3];
        Self { data: zeroes }
    }

    pub fn get_c(&self) -> bool {
        self.data[0]
    }

    pub fn get_a(&self) -> bool {
        self.data[1]
    }

    pub fn get_b(&self) -> bool {
        self.data[2]
    }
}
