#![cfg_attr(is_nightly, feature(portable_simd))]
//! SEEC-BitMatrix
//!
//! A library for fast bitmatrix transpose intended for MPC implementations.

use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bitvec::store::BitStore;
use bitvec::vec::BitVec;
use cfg_if::cfg_if;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
#[cfg(feature = "rayon")]
use rayon::iter::IndexedParallelIterator;
#[cfg(feature = "rayon")]
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use serde::{Deserialize, Serialize};
use std::fmt::{Binary, Debug, Formatter};
use std::ops::{BitAnd, BitXor, Range};
use std::slice::{ChunksExact, ChunksExactMut};

#[cfg(is_nightly)]
mod portable_transpose;
mod simple;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod sse2_transpose;

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BitMatrix<T: Storage> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BitMatrixView<'a, T> {
    rows: usize,
    cols: usize,
    data: &'a [T],
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BitMatrixViewMut<'a, T> {
    rows: usize,
    cols: usize,
    data: &'a mut [T],
}

pub trait Storage:
    bytemuck::Pod + BitXor<Output = Self> + BitAnd<Output = Self> + Send + Sync
{
    const BITS: usize;

    fn zero() -> Self;
}

impl<T: Storage> BitMatrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::zeros(rows, cols)
    }

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Self {
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        check_dim::<T>(rows, cols);
        Self {
            data: vec![T::zero(); rows * cols / T::BITS],
            rows,
            cols,
        }
    }

    pub fn random<R: Rng>(rng: R, rows: usize, cols: usize) -> Self
    where
        Standard: Distribution<T>,
    {
        check_dim::<T>(rows, cols);
        let data = rng
            .sample_iter(Standard)
            .take(rows * cols / T::BITS)
            .collect();
        Self { data, rows, cols }
    }

    pub fn view(&self) -> BitMatrixView<'_, T> {
        BitMatrixView {
            rows: self.rows,
            cols: self.cols,
            data: self.data.as_slice(),
        }
    }

    pub fn view_mut(&mut self) -> BitMatrixViewMut<'_, T> {
        BitMatrixViewMut {
            rows: self.rows,
            cols: self.cols,
            data: self.data.as_mut_slice(),
        }
    }

    // Returns dimensions (rows, columns).
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn storage_len(&self) -> usize {
        self.data.len()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn iter_rows(&self) -> Rows<'_, T> {
        Rows::new(self.view())
    }

    pub fn iter_raw_rows(&self) -> RawRows<'_, T> {
        RawRows::new(self.view())
    }

    pub fn iter_raw_rows_mut(&mut self) -> RawRowsMut<'_, T> {
        RawRowsMut::new(self.view_mut())
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_raw_rows(&self) -> impl IndexedParallelIterator<Item = &[T]> {
        assert_eq!(
            0,
            self.cols % T::BITS,
            "cols must be divisable by bits for raw iterator"
        );
        self.data.par_chunks_exact(self.cols / T::BITS)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter_raw_rows_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [T]> {
        assert_eq!(
            0,
            self.cols % T::BITS,
            "cols must be divisable by bits for raw iterator"
        );
        self.data.par_chunks_exact_mut(self.cols / T::BITS)
    }

    pub fn scalar_and(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.dim(),
            rhs.dim(),
            "Dimensions must be identical for scalar_and"
        );
        let data = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| *a & *b)
            .collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> BitMatrix<T>
where
    T: Storage + BitStore<Unalias = T>,
{
    pub fn from_bits(bits: &BitSlice<T, Lsb0>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            bits.len() % T::BITS,
            0,
            "Length of bits must be  multiple of T::BITS"
        );
        assert_eq!(bits.len(), rows * cols, "bits.len() != rows * cols");
        let data = bits.to_bitvec().into_vec();
        Self { rows, cols, data }
    }

    pub fn identity(size: usize) -> Self {
        check_dim::<T>(size, size);
        let mut bv: BitVec<T> = BitVec::repeat(false, size * size);
        let mut idx = 0;
        while idx < size * size {
            bv.set(idx, true);
            idx += size + 1;
        }
        Self::from_vec(bv.into_vec(), size, size)
    }

    pub fn mat_mul(self, rhs: &Self) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "Illegal dimensions for matrix multiplication"
        );
        // TODO this can likely be heavily optimized. One option is to use the raw_rows iterator
        //  if possible to do the dotp on the raw elements. This should be significantly faster.
        let dotp = |l_row: &BitSlice<T>, r_row| -> bool {
            let and = l_row.to_bitvec() & r_row;
            and.iter().by_vals().reduce(BitXor::bitxor).unwrap()
        };

        let rhs = rhs.view().transpose();
        let bits = self
            .iter_rows()
            .flat_map(|l_row| rhs.iter_rows().map(|r_row| dotp(l_row, r_row)));
        let mut bv: BitVec<T> = BitVec::with_capacity(self.rows * rhs.rows);
        bv.extend(bits);
        Self::from_vec(bv.into_vec(), self.rows, rhs.rows)
    }

    pub fn into_bitvec(self) -> BitVec<T> {
        BitVec::from_vec(self.data)
    }
}

impl<'a, T: Storage> BitMatrixView<'a, T> {
    pub fn from_slice(data: &'a [T], rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len() * T::BITS,
            rows * cols,
            "data.len() does not match rows * cols"
        );
        Self { rows, cols, data }
    }

    pub fn fast_transpose(&self) -> BitMatrix<T> {
        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] {
                let transposed = sse2_transpose::transpose(self.data, self.rows, self.cols);
            } else if #[cfg(is_nightly)] {
                let transposed = portable_transpose::transpose(self.data, self.rows, self.cols);
            } else {
                use std::compile_error;

                compile_error!("Target must either be x86_64 with sse2 enabled or crate \
                feature \"portable_transpose\" must be enabled (requires nightly)")
            }
        }

        BitMatrix::from_vec(transposed, self.cols, self.rows)
    }

    #[inline]
    pub fn raw_row(&self, row: usize) -> Option<&'a [T]> {
        let idx = raw_row_idx::<T>(row, self.cols);
        self.data.get(idx)
    }

    fn can_do_sse_trans(&self) -> bool {
        self.rows % 8 == 0 && self.cols % 8 == 0 && self.rows >= 16 && self.cols >= 16
    }
}

impl<'a, T: BitStore<Unalias = T>> BitMatrixView<'a, T> {
    pub fn as_bitslice(&self) -> &'a BitSlice<T> {
        BitSlice::from_slice(self.data)
    }

    #[inline]
    pub fn row(&self, row: usize) -> Option<&'a BitSlice<T>> {
        let data = self.as_bitslice();
        let start_idx = row * self.cols;
        let end_idx = (row + 1) * self.cols;
        data.get(start_idx..end_idx)
    }
}

impl<'a, T: Storage + BitStore<Unalias = T>> BitMatrixView<'a, T> {
    pub fn transpose(&self) -> BitMatrix<T> {
        let transposed = if self.can_do_sse_trans()
            && (cfg!(is_nightly) || cfg!(all(target_arch = "x86_64", target_feature = "sse2")))
        {
            cfg_if! {
                if #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] {
                    sse2_transpose::transpose(self.data, self.rows, self.cols)
                } else if #[cfg(is_nightly)] {
                    portable_transpose::transpose(self.data, self.rows, self.cols)
                } else {
                    simple::transpose(self.data, self.rows, self.cols)
                }
            }
        } else {
            simple::transpose(self.data, self.rows, self.cols)
        };
        BitMatrix::from_vec(transposed, self.cols, self.rows)
    }
}

impl<'a, T: Storage> BitMatrixViewMut<'a, T> {
    pub fn raw_row_mut(&'a mut self, row: usize) -> Option<&'a mut [T]> {
        let idx = raw_row_idx::<T>(row, self.cols);
        self.data.get_mut(idx)
    }
}

#[inline]
fn raw_row_idx<T: Storage>(row: usize, cols: usize) -> Range<usize> {
    assert_eq!(
        0,
        cols % T::BITS,
        "cols must be divisable by T::BITS for raw_row. Use row() instead."
    );
    let cols_el = cols / T::BITS;
    let start_idx = row * cols_el;
    let end_idx = (row + 1) * cols_el;
    start_idx..end_idx
}

impl<T: Storage> BitXor for BitMatrix<T> {
    type Output = BitMatrix<T>;

    fn bitxor(mut self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dim(),
            rhs.dim(),
            "BitXor on matrices with different dimensions"
        );
        self.data.iter_mut().zip(rhs.data).for_each(|(a, b)| {
            *a = *a ^ b;
        });
        self
    }
}

impl<T: Storage> BitXor<&Self> for BitMatrix<T> {
    type Output = BitMatrix<T>;

    fn bitxor(mut self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.dim(),
            rhs.dim(),
            "BitXor on matrices with different dimensions"
        );
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, b)| {
            *a = *a ^ *b;
        });
        self
    }
}

impl<T: Binary + Storage> Binary for BitMatrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let fmt_bin: Vec<_> = self.data.iter().map(|el| format!("{el:b}")).collect();

        f.debug_struct("BitMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &fmt_bin)
            .finish()
    }
}

impl Storage for u8 {
    const BITS: usize = 8;
    fn zero() -> Self {
        0
    }
}
impl Storage for u16 {
    const BITS: usize = 16;
    fn zero() -> Self {
        0
    }
}
impl Storage for u32 {
    const BITS: usize = 32;
    fn zero() -> Self {
        0
    }
}
impl Storage for u64 {
    const BITS: usize = 64;
    fn zero() -> Self {
        0
    }
}
impl Storage for u128 {
    const BITS: usize = 128;
    fn zero() -> Self {
        0
    }
}

#[derive(Clone, Debug)]
pub struct Rows<'a, T> {
    view: BitMatrixView<'a, T>,
    row: usize,
}

impl<'a, T> Rows<'a, T> {
    pub fn new(view: BitMatrixView<'a, T>) -> Self {
        Self { view, row: 0 }
    }
}

impl<'a, T: BitStore<Unalias = T>> Iterator for Rows<'a, T> {
    type Item = &'a BitSlice<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.view.row(self.row);
        self.row += 1;
        ret
    }
}

#[derive(Clone, Debug)]
pub struct RawRows<'a, T> {
    chunks: ChunksExact<'a, T>,
}

impl<'a, T: Storage> RawRows<'a, T> {
    pub fn new(view: BitMatrixView<'a, T>) -> Self {
        assert_eq!(
            0,
            view.cols % T::BITS,
            "cols of BitMatrix must be multiple of T::BITS for raw rows iterator"
        );
        let chunks = view.data.chunks_exact(view.cols / T::BITS);
        Self { chunks }
    }
}

impl<'a, T: Storage> Iterator for RawRows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }
}

#[derive(Debug)]
pub struct RawRowsMut<'a, T> {
    chunks: ChunksExactMut<'a, T>,
}

impl<'a, T: Storage> RawRowsMut<'a, T> {
    pub fn new(view: BitMatrixViewMut<'a, T>) -> Self {
        assert_eq!(
            0,
            view.cols % T::BITS,
            "cols of BitMatrix must be multiple of T::BITS for raw rows iterator"
        );
        let chunks = view.data.chunks_exact_mut(view.cols / T::BITS);
        Self { chunks }
    }
}

impl<'a, T: Storage> Iterator for RawRowsMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }
}

fn check_dim<T: Storage>(rows: usize, cols: usize) {
    assert_eq!(
        (rows * cols) % T::BITS,
        0,
        "rows * cols must be divisable by T::BITS"
    );
}

#[cfg(test)]
mod tests {
    use crate::BitMatrix;
    use bitvec::vec::BitVec;
    use ndarray::Array2;
    use num_traits::{One, Zero};
    use rand::{thread_rng, Rng};
    use std::fmt::{Debug, Formatter};
    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Copy, Clone)]
    struct Z2(u8);

    #[test]
    fn mul() {
        let id: BitMatrix<u8> = BitMatrix::identity(128);
        let other = BitMatrix::random(thread_rng(), 128, 256);
        let mul = id.mat_mul(&other);
        assert_eq!(other, mul)
    }

    #[test]
    fn mul_ndarray() {
        let cols = 128;
        let rows = 128;
        let mut rng = thread_rng();
        let nd_arr1: Vec<_> = (0..rows * cols).map(|_| Z2(rng.gen_range(0..2))).collect();
        let bitmat1 = BitMatrix::from_bits(
            &nd_arr1.iter().map(|bit| bit.0 == 1).collect::<BitVec<u8>>(),
            rows,
            cols,
        );
        let nd_arr1 = Array2::from_shape_vec((rows, cols), nd_arr1).unwrap();

        let nd_arr2: Vec<_> = (0..rows * cols).map(|_| Z2(rng.gen_range(0..2))).collect();
        let bitmat2 = BitMatrix::from_bits(
            &nd_arr2.iter().map(|bit| bit.0 == 1).collect::<BitVec<u8>>(),
            rows,
            cols,
        );
        let nd_arr2 = Array2::from_shape_vec((rows, cols), nd_arr2).unwrap();

        let res_nd_arr = nd_arr1.dot(&nd_arr2);
        let res_bit_mat = bitmat1.mat_mul(&bitmat2);

        for (nd_row, bit_mat_row) in res_nd_arr.rows().into_iter().zip(res_bit_mat.iter_rows()) {
            for (nd_bit, bit_mat_bit) in nd_row.iter().zip(bit_mat_row) {
                assert_eq!(
                    nd_bit.0 == 1,
                    *bit_mat_bit,
                    "BitMatrix::mat_mult differs from nd_array"
                );
            }
        }
    }

    impl Add for Z2 {
        type Output = Z2;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0 % 2)
        }
    }

    impl Sub for Z2 {
        type Output = Z2;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0 % 2)
        }
    }

    impl Mul for Z2 {
        type Output = Z2;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0 * rhs.0 % 2)
        }
    }

    impl Div for Z2 {
        type Output = Z2;

        fn div(self, rhs: Self) -> Self::Output {
            Self(self.0 / rhs.0 % 2)
        }
    }

    impl Zero for Z2 {
        fn zero() -> Self {
            Self(0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }

    impl One for Z2 {
        fn one() -> Self {
            Self(1)
        }
    }

    impl Debug for Z2 {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }

    impl PartialEq<Array2<Z2>> for BitMatrix<u8> {
        fn eq(&self, other: &Array2<Z2>) -> bool {
            other
                .rows()
                .into_iter()
                .zip(self.iter_rows())
                .all(|(r1, r2)| {
                    r1.iter()
                        .zip(r2)
                        .all(|(el1, el2)| matches!((el1, *el2), (Z2(0), false) | (Z2(1), true)))
                })
        }
    }
}
