use crate::Storage;
use std::mem;
use std::simd::{u16x8, u8x16, Simd};

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
        let mut h = [0_u8; 2];
        let mut rr: usize = 0;
        while rr <= nrows - 16 {
            let mut cc = 0;
            while cc < ncols {
                let mut v = Simd::<u8, 16>::from_array([
                    *input.get_unchecked(inp(rr, cc)),
                    *input.get_unchecked(inp(rr + 1, cc)),
                    *input.get_unchecked(inp(rr + 2, cc)),
                    *input.get_unchecked(inp(rr + 3, cc)),
                    *input.get_unchecked(inp(rr + 4, cc)),
                    *input.get_unchecked(inp(rr + 5, cc)),
                    *input.get_unchecked(inp(rr + 6, cc)),
                    *input.get_unchecked(inp(rr + 7, cc)),
                    *input.get_unchecked(inp(rr + 8, cc)),
                    *input.get_unchecked(inp(rr + 9, cc)),
                    *input.get_unchecked(inp(rr + 10, cc)),
                    *input.get_unchecked(inp(rr + 11, cc)),
                    *input.get_unchecked(inp(rr + 12, cc)),
                    *input.get_unchecked(inp(rr + 13, cc)),
                    *input.get_unchecked(inp(rr + 14, cc)),
                    *input.get_unchecked(inp(rr + 15, cc)),
                ]);
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
        let mut tmp = u8x16::default();
        let mut cc = 0;
        if ncols % 16 != 0 || nrows % 16 != 0 {
            // The fancy optimizations in the else-branch don't work if the above if-condition
            // holds, so we use the simpler non-simd variant for that case.
            while cc <= ncols - 16 {
                for i in 0..8 {
                    let h = *(input.get_unchecked(inp(rr + i, cc)) as *const _ as *const [u8; 2]);
                    tmp[i] = h[0];
                    tmp[i + 8] = h[1];
                }
                for i in (0..8).rev() {
                    h = _mm_movemask_epi8(tmp).to_le_bytes();
                    // TODO maybe this can be optimized by directly writing the
                    // output of the movemask
                    *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
                    *byte_output.get_unchecked_mut(out(rr, cc + i + 8)) = h[1];

                    tmp = _mm_slli_epi64::<1>(tmp);
                }
                cc += 16;
            }
        } else {
            while cc <= ncols - 16 {
                let mut v = u16x8::from_array([
                    *(input.get_unchecked(inp(rr + 7, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr + 6, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr + 5, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr + 4, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr + 3, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr + 2, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr + 1, cc)) as *const _ as *const u16),
                    *(input.get_unchecked(inp(rr, cc)) as *const _ as *const u16),
                ]);
                for i in (0..8).rev() {
                    h = _mm_movemask_epi8(mem::transmute(v)).to_le_bytes();
                    *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
                    *byte_output.get_unchecked_mut(out(rr, cc + i + 8)) = h[1];
                    v = mem::transmute(_mm_slli_epi64::<1>(mem::transmute(v)));
                }
                cc += 16;
            }
        }
        if cc == ncols {
            return output;
        }
        for i in 0..8 {
            tmp[i] = *input.get_unchecked(inp(rr + i, cc));
        }
        for i in (0..8).rev() {
            h = _mm_movemask_epi8(tmp).to_le_bytes();
            *byte_output.get_unchecked_mut(out(rr, cc + i)) = h[0];
            tmp = _mm_slli_epi64::<1>(tmp);
        }
    };
    output
}

fn _mm_movemask_epi8(a: Simd<u8, 16>) -> u16 {
    #[cfg(target_feature = "sse2")]
    {
        use std::arch::x86_64::_mm_movemask_epi8 as op;
        unsafe { op(a.into()) as u16 }
    }
    #[cfg(not(target_feature = "sse2"))]
    {
        use std::simd::{i8x16, Mask, ToBitMask};
        let a: i8x16 = a.cast();
        let msb = a >> i8x16::splat(7);
        let mask = unsafe { Mask::from_int_unchecked(msb) };
        mask.to_bitmask()
    }
}

fn _mm_slli_epi64<const IMM8: i32>(a: Simd<u8, 16>) -> Simd<u8, 16> {
    #[cfg(target_feature = "sse2")]
    {
        use std::arch::x86_64::_mm_slli_epi64 as op;
        unsafe { op::<IMM8>(a.into()).into() }
    }
    #[cfg(not(target_feature = "sse2"))]
    {
        use std::simd::u64x2;
        let mut a: u64x2 = unsafe { mem::transmute(a) };
        a <<= u64x2::splat(IMM8 as u64);
        unsafe { mem::transmute(a) }
    }
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
        fn test_u32_transpose((v, rows, cols) in arbitrary_bitmat::<u32>(16 * 30, 16 * 30)) {
            let transposed = super::transpose(v.as_slice(), rows, cols);
            let expected = crate::simple::transpose(v.as_slice(), rows, cols);
            prop_assert_eq!(transposed, expected);
        }
    }
}
