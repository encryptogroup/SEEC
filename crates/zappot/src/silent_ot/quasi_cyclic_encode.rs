use std::cmp::{max, min};
use std::fmt::Debug;

use bitpolymul::{DecodeCache, FftPoly};
use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bytemuck::{cast_slice, cast_slice_mut};
use ndarray::Array2;
use num_integer::Integer;
use num_prime::nt_funcs::next_prime;
use rand::Rng;
use rand_core::SeedableRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use seec_bitmatrix::BitMatrixView;
use std::time::Instant;

use crate::silent_ot::get_reg_noise_weight;
use crate::silent_ot::pprf::PprfConfig;
use crate::util::aes_rng::AesRng;
use crate::util::Block;

#[derive(Debug, Clone)]
pub struct QuasiCyclicEncoder {
    pub(crate) conf: QuasiCyclicConf,
    a_polynomials: Vec<FftPoly>,
}

impl QuasiCyclicEncoder {
    pub(crate) fn new(conf: QuasiCyclicConf) -> Self {
        let a = init_a_polynomials(conf);
        Self {
            conf,
            a_polynomials: a,
        }
    }

    pub(crate) fn dual_encode(&self, rT: Array2<Block>) -> Vec<Block> {
        let conf = self.conf;
        let mut c_mod_p1: Array2<Block> = Array2::zeros((QuasiCyclicConf::ROWS, conf.n_blocks()));
        let mut B = vec![Block::zero(); conf.N2];
        let reducer = MultAddReducer::new(conf, &self.a_polynomials);
        c_mod_p1
            .outer_iter_mut()
            .into_par_iter()
            .zip(rT.outer_iter())
            .for_each_init(
                || reducer.clone(),
                |reducer, (mut cmod_row, rt_row)| {
                    let cmod_row = cmod_row.as_slice_mut().unwrap();
                    let rt_row = rt_row.as_slice().unwrap();
                    reducer.reduce(cmod_row, rt_row);
                },
            );

        let num_blocks = Integer::next_multiple_of(&conf.requested_num_ots, &128);
        copy_out(&mut B[..num_blocks], &c_mod_p1);
        B.truncate(self.conf.requested_num_ots);
        B
    }

    pub(crate) fn dual_encode_choice(&self, sb: &[Block]) -> Vec<u8> {
        let mut c128 = vec![Block::zero(); self.conf.n_blocks()];
        let mut reducer = MultAddReducer::new(self.conf, &self.a_polynomials);
        reducer.reduce(&mut c128, sb);

        let mut C = vec![0; self.conf.requested_num_ots];

        let c128_bits: &BitSlice<usize, Lsb0> = BitSlice::from_slice(cast_slice(&c128));
        C.iter_mut()
            .zip(c128_bits.iter().by_vals())
            .for_each(|(c, bit)| {
                *c = bit as u8;
            });
        C
    }
}

fn init_a_polynomials(conf: QuasiCyclicConf) -> Vec<FftPoly> {
    let mut temp = vec![0_u64; 2 * conf.n_blocks()];
    (0..conf.scaler - 1)
        .map(|s| {
            let mut fft_poly = FftPoly::new();
            let mut pub_rng = AesRng::from_seed((s + 1).into());
            pub_rng.fill(&mut temp[..]);
            fft_poly.encode(&temp);
            fft_poly
        })
        .collect()
}

fn copy_out(dest: &mut [Block], c_mod_p1: &Array2<Block>) {
    assert_eq!(dest.len() % 128, 0, "Dest must have a length of 128");
    dest.par_chunks_exact_mut(128)
        .enumerate()
        .for_each(|(i, chunk)| {
            // TODO maybe it's faster to transpose c_mod_p1 and then access rows,
            //  should have better cache efficiency
            chunk
                .iter_mut()
                .zip(c_mod_p1.column(i))
                .for_each(|(block, cmod)| *block = *cmod);
            // TODO don't allocate in loop...
            let transposed = BitMatrixView::from_slice(chunk, 128, 128)
                .fast_transpose()
                .into_vec();
            chunk.copy_from_slice(&transposed);
        });
}

#[derive(Copy, Clone, Debug)]
/// Configuration options  for the quasi cyclic silent OT implementation. Is created by
/// calling the [configure()](`configure`) function.
pub struct QuasiCyclicConf {
    /// The prime for QuasiCyclic encoding
    pub(crate) P: usize,
    /// the number of OTs being requested.
    pub(crate) requested_num_ots: usize,
    /// The dense vector size, this will be at least as big as mRequestedNumOts.
    pub(crate) N: usize,
    /// The sparse vector size, this will be mN * mScaler.
    pub(crate) N2: usize,
    /// The scaling factor that the sparse vector will be compressed by.
    pub(crate) scaler: usize,
    /// The size of each regular section of the sparse vector.
    pub(crate) size_per: usize,
    /// The number of regular section of the sparse vector.
    pub(crate) num_partitions: usize,
}

impl QuasiCyclicConf {
    pub const ROWS: usize = 128;

    /// Create a new [QuasiCyclicConf](`QuasiCyclicConf`) given the provided values.
    pub fn configure(num_ots: usize, scaler: usize, sec_param: usize) -> Self {
        let P = next_prime(&max(num_ots, 128 * 128), None).unwrap();
        let num_partitions = get_reg_noise_weight(0.2, sec_param) as usize;
        let ss = (P * scaler + num_partitions - 1) / num_partitions;
        let size_per = Integer::next_multiple_of(&ss, &8);
        let N2 = size_per * num_partitions;
        let N = N2 / scaler;
        Self {
            P,
            num_partitions,
            size_per,
            N2,
            N,
            scaler,
            requested_num_ots: num_ots,
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.N / Self::ROWS
    }

    pub fn n2_blocks(&self) -> usize {
        self.N2 / Self::ROWS
    }

    pub fn n64(self) -> usize {
        self.n_blocks() * 2
    }

    pub fn P(&self) -> usize {
        self.P
    }
    pub fn requested_num_ots(&self) -> usize {
        self.requested_num_ots
    }
    pub fn N(&self) -> usize {
        self.N
    }
    pub fn N2(&self) -> usize {
        self.N2
    }
    pub fn scaler(&self) -> usize {
        self.scaler
    }
    pub fn size_per(&self) -> usize {
        self.size_per
    }
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
    /// Returns the amount of base OTs needed for this configuration.
    pub fn base_ot_count(&self) -> usize {
        let pprf_conf = PprfConfig::from(*self);
        pprf_conf.base_ot_count()
    }
}

impl From<QuasiCyclicConf> for PprfConfig {
    fn from(conf: QuasiCyclicConf) -> Self {
        PprfConfig::new(conf.size_per, conf.num_partitions)
    }
}

#[derive(Clone)]
/// Helper struct which manages parameters and cached values for the mult_add_reduce operation
pub struct MultAddReducer<'a> {
    a_polynomials: &'a [FftPoly],
    conf: QuasiCyclicConf,
    b_poly: FftPoly,
    temp128: Vec<Block>,
    cache: DecodeCache,
}

impl<'a> MultAddReducer<'a> {
    pub(crate) fn new(conf: QuasiCyclicConf, a_polynomials: &'a [FftPoly]) -> Self {
        Self {
            a_polynomials,
            conf,
            b_poly: FftPoly::new(),
            temp128: vec![Block::zero(); 2 * conf.n_blocks()],
            cache: DecodeCache::default(),
        }
    }

    pub(crate) fn reduce(&mut self, dest: &mut [Block], b128: &[Block]) {
        let n64 = self.conf.n64();
        let mut c_poly = FftPoly::new();
        for s in 1..self.conf.scaler {
            let a_poly = &self.a_polynomials[s - 1];
            let b64 = &cast_slice(b128)[s * n64..(s + 1) * n64];
            let _now = Instant::now();
            self.b_poly.encode(b64);
            if s == 1 {
                c_poly.mult(a_poly, &self.b_poly);
            } else {
                self.b_poly.mult_eq(a_poly);
                c_poly.add_eq(&self.b_poly);
            }
        }
        c_poly.decode_with_cache(&mut self.cache, cast_slice_mut(&mut self.temp128));

        self.temp128
            .iter_mut()
            .zip(b128)
            .take(self.conf.n_blocks())
            .for_each(|(t, b)| *t ^= *b);

        modp(dest, &self.temp128, self.conf.P);
    }
}

pub fn modp(dest: &mut [Block], inp: &[Block], prime: usize) {
    let p: usize = prime;

    let p_blocks = (p + 127) / 128;
    let p_bytes = (p + 7) / 8;
    let dest_len = dest.len();
    assert!(dest_len >= p_blocks);
    assert!(inp.len() >= p_blocks);
    let count = (inp.len() * 128 + p - 1) / p;
    {
        let dest_bytes = cast_slice_mut::<_, u8>(dest);
        let inp_bytes = cast_slice::<_, u8>(inp);
        dest_bytes[..p_bytes].copy_from_slice(&inp_bytes[..p_bytes]);
    }

    for i in 1..count {
        let begin = i * p;
        let begin_block = begin / 128;
        let end_block = min(i * p + p, inp.len() * 128);
        let end_block = (end_block + 127) / 128;
        assert!(end_block <= inp.len());
        // TODO the above calculations seem redundant
        let in_i = &inp[begin_block..end_block];
        let shift = begin & 127;
        bit_shift_xor(dest, in_i, shift as u8);
    }
    let dest_bytes = cast_slice_mut::<_, u8>(dest);

    let offset = p & 7;
    if offset != 0 {
        let mask = ((1 << offset) - 1) as u8;
        let idx = p / 8;
        dest_bytes[idx] &= mask;
    }
    let rem = dest_len * 16 - p_bytes;
    if rem != 0 {
        dest_bytes[p_bytes..p_bytes + rem].fill(0);
    }
}

pub fn bit_shift_xor(dest: &mut [Block], inp: &[Block], bit_shift: u8) {
    assert!(bit_shift <= 127, "bit_shift must be less than 127");

    dest.iter_mut()
        .zip(inp)
        .zip(&inp[1..])
        .for_each(|((d, inp), inp_off)| {
            let mut shifted = *inp >> bit_shift;
            shifted |= *inp_off << (128 - bit_shift);
            *d ^= shifted;
        });
    if dest.len() >= inp.len() {
        let inp_last = *inp.last().expect("empty input");
        dest[inp.len() - 1] ^= inp_last >> bit_shift;
    }
}

#[cfg(test)]
mod tests {

    use crate::silent_ot::quasi_cyclic_encode::{bit_shift_xor, modp};
    use crate::util::Block;
    use bitvec::order::Lsb0;
    use bitvec::prelude::{BitSlice, BitVec};
    use std::cmp::min;

    #[test]
    fn basic_bit_shift_xor() {
        let dest = &mut [Block::zero(), Block::zero()];
        let inp = &[Block::all_ones(), Block::all_ones()];
        let bit_shift = 10;
        bit_shift_xor(dest, inp, bit_shift);
        assert_eq!(Block::all_ones(), dest[0]);
        let exp = Block::from(u128::MAX >> bit_shift);
        assert_eq!(exp, dest[1]);
    }

    #[test]
    fn basic_modp() {
        let i_bits = 1026;
        let n_bits = 223;
        let n = (n_bits + 127) / 128;
        let c = (i_bits + n_bits - 1) / n_bits;
        let mut dest = vec![Block::zero(); n];
        let mut inp = vec![Block::all_ones(); (i_bits + 127) / 128];
        let p = n_bits;
        let inp_bits: &mut BitSlice<usize, Lsb0> =
            BitSlice::from_slice_mut(bytemuck::cast_slice_mut(&mut inp));
        inp_bits[i_bits..].fill(false);
        let mut dv: BitVec<usize, Lsb0> = BitVec::repeat(true, p);
        let mut iv: BitVec<usize, Lsb0> = BitVec::new();
        for j in 1..c {
            let rem = min(p, i_bits - j * p);
            iv.clear();
            let inp = &inp_bits[j * p..(j * p) + rem];
            iv.extend_from_bitslice(inp);
            iv.resize(p, false);
            dv ^= &iv;
        }
        modp(&mut dest, &inp, p);
        let dest_bits: &BitSlice<usize, Lsb0> = BitSlice::from_slice(bytemuck::cast_slice(&dest));
        let dv2 = &dest_bits[..p];
        assert_eq!(dv, dv2);
    }
}
