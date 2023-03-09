use crate::Storage;
use std::arch::x86_64::{__m128i, _mm_movemask_epi8, _mm_set_epi16, _mm_setr_epi8, _mm_slli_epi64};
use std::mem;

#[repr(C)]
union __U128 {
    vector: __m128i,
    bytes: [u8; 16],
}

impl Default for __U128 {
    #[inline]
    fn default() -> Self {
        __U128 { bytes: [0u8; 16] }
    }
}

#[inline]
#[allow(unused)]
pub(crate) fn transpose<T: Storage>(input: &[T], nrows: usize, ncols: usize) -> Vec<T> {
    assert_eq!(nrows % 8, 0);
    assert_eq!(ncols % 8, 0);
    assert!(ncols >= 16, "ncols must be >= 16");
    assert!(nrows >= 16, "nrows must be >= 16");
    assert_eq!(
        ncols * nrows,
        input.len() * mem::size_of::<T>() * 8,
        "input.len() does not match nrows/ncols"
    );
    let mut output = vec![T::zero(); input.len()];
    let input = bytemuck::cast_slice(input);
    let byte_output = bytemuck::cast_slice_mut(output.as_mut_slice());

    let inp = |x: usize, y: usize| -> usize { x * ncols / 8 + y / 8 };
    let out = |x: usize, y: usize| -> usize { y * nrows / 8 + x / 8 };

    unsafe {
        let mut h = [0_u8; 4];
        let mut rr: usize = 0;
        while rr <= nrows - 16 {
            let mut cc = 0;
            while cc < ncols {
                let mut v = _mm_setr_epi8(
                    *input.get_unchecked(inp(rr, cc)) as i8,
                    *input.get_unchecked(inp(rr + 1, cc)) as i8,
                    *input.get_unchecked(inp(rr + 2, cc)) as i8,
                    *input.get_unchecked(inp(rr + 3, cc)) as i8,
                    *input.get_unchecked(inp(rr + 4, cc)) as i8,
                    *input.get_unchecked(inp(rr + 5, cc)) as i8,
                    *input.get_unchecked(inp(rr + 6, cc)) as i8,
                    *input.get_unchecked(inp(rr + 7, cc)) as i8,
                    *input.get_unchecked(inp(rr + 8, cc)) as i8,
                    *input.get_unchecked(inp(rr + 9, cc)) as i8,
                    *input.get_unchecked(inp(rr + 10, cc)) as i8,
                    *input.get_unchecked(inp(rr + 11, cc)) as i8,
                    *input.get_unchecked(inp(rr + 12, cc)) as i8,
                    *input.get_unchecked(inp(rr + 13, cc)) as i8,
                    *input.get_unchecked(inp(rr + 14, cc)) as i8,
                    *input.get_unchecked(inp(rr + 15, cc)) as i8,
                );
                (0..8).rev().for_each(|i| {
                    h = _mm_movemask_epi8(v).to_le_bytes();
                    // TODO maybe this can be optimized by directly writing the
                    // output of the movemask
                    *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
                    *byte_output.get_unchecked_mut(out(rr, cc + i) + 1) = h[1];

                    v = _mm_slli_epi64::<1>(v);
                });
                cc += 8;
            }
            rr += 16;
        }
        if rr == nrows {
            return output;
        }

        // The remainder is a block of 8x(16n+8) bits (n may be 0).
        //  Do a PAIR of 8x8 blocks in each step:
        let mut tmp = __U128::default();
        let mut cc = 0;
        if ncols % 16 != 0 || nrows % 16 != 0 {
            // The fancy optimizations in the else-branch don't work if the above if-condition
            // holds, so we use the simpler non-simd variant for that case.
            while cc <= ncols - 16 {
                for i in 0..8 {
                    let h = *(input.get_unchecked(inp(rr + i, cc)) as *const _ as *const [u8; 2]);
                    tmp.bytes[i] = h[0];
                    tmp.bytes[i + 8] = h[1];
                }
                for i in (0..8).rev() {
                    h = _mm_movemask_epi8(tmp.vector).to_le_bytes();
                    // TODO maybe this can be optimized by directly writing the
                    // output of the movemask
                    *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
                    *byte_output.get_unchecked_mut(out(rr, cc + i + 8)) = h[1];

                    tmp.vector = _mm_slli_epi64::<1>(tmp.vector);
                }
                cc += 16;
            }
        } else {
            while cc <= ncols - 16 {
                let mut v = _mm_set_epi16(
                    *(input.get_unchecked(inp(rr + 7, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr + 6, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr + 5, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr + 4, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr + 3, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr + 2, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr + 1, cc)) as *const _ as *const i16),
                    *(input.get_unchecked(inp(rr, cc)) as *const _ as *const i16),
                );
                for i in (0..8).rev() {
                    h = _mm_movemask_epi8(v).to_le_bytes();
                    *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
                    *byte_output.get_unchecked_mut(out(rr, cc + i + 8)) = h[1];
                    v = _mm_slli_epi64::<1>(v);
                }
                cc += 16;
            }
        }
        if cc == ncols {
            return output;
        }
        for i in 0..8 {
            tmp.bytes[i] = *input.get_unchecked(inp(rr + i, cc));
        }
        for i in (0..8).rev() {
            h = _mm_movemask_epi8(tmp.vector).to_le_bytes();
            *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
            tmp.vector = _mm_slli_epi64::<1>(tmp.vector);
        }
    };
    output
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    fn arbitrary_bitmat<T: Arbitrary + Clone>(
        max_row: usize,
        max_col: usize,
    ) -> BoxedStrategy<(Vec<T>, usize, usize)>
    where
        T::Strategy: Clone,
    {
        (
            (16..max_row).prop_map(|row| row / 8 * 8),
            (16..max_col).prop_map(|col| col / 8 * 8),
        )
            .prop_flat_map(|(rows, cols)| {
                (
                    vec![any::<T>(); rows * cols / (std::mem::size_of::<T>() * 8)],
                    Just(rows),
                    Just(cols),
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn test_byte_transpose((v, rows, cols) in arbitrary_bitmat::<u8>(16 * 30, 16 * 30)) {
            let transposed = super::transpose(v.as_slice(), rows, cols);
            let expected = crate::simple::transpose(v.as_slice(), rows, cols);
            prop_assert_eq!(transposed, expected);
        }

        #[test]
        fn test_u64_transpose((v, rows, cols) in arbitrary_bitmat::<u64>(16 * 30, 16 * 30)) {
            let transposed = super::transpose(v.as_slice(), rows, cols);
            let expected = crate::simple::transpose(v.as_slice(), rows, cols);
            prop_assert_eq!(transposed, expected);
        }
    }
}
