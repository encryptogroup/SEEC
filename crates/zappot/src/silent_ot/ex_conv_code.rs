use crate::silent_ot::pprf::PprfConfig;
use crate::silent_ot::{get_reg_noise_weight, MultType};
use num_integer::Integer;

#[derive(Debug)]
pub struct ExConvEncoder {
    pub(crate) enc: libote::ExConvCode,
    pub(crate) conf: ExConvConf,
}

#[derive(Debug, Copy, Clone)]
#[allow(unused)]
pub struct ExConvConf {
    pub(crate) scaler: usize,
    pub(crate) weight: usize,
    pub(crate) accumulator_size: usize,
    /// the number of OTs being requested.
    pub(crate) requested_num_ots: usize,
    /// The dense vector size, this will be at least as big as mRequestedNumOts.
    pub(crate) N: usize,
    /// The sparse vector size, this will be mN * mScaler.
    pub(crate) N2: usize,
    /// The size of each regular section of the sparse vector.
    pub(crate) size_per: usize,
    /// The number of regular section of the sparse vector.
    pub(crate) num_partitions: usize,
    pub(crate) code_size: usize,
}

impl ExConvConf {
    pub fn configure(num_ots: usize, mult_type: MultType, sec_param: usize) -> Self {
        let scaler = 2;
        let (weight, min_dist) = match mult_type {
            MultType::ExConv7x24 => (7, 0.1),
            MultType::ExConv21x24 => (21, 0.15),
            other => panic!("Unsupported mult_type {other:?}"),
        };
        let num_partitions = get_reg_noise_weight(min_dist, sec_param) as usize;
        let size_per = Integer::next_multiple_of(
            &((num_ots * scaler + num_partitions - 1) / num_partitions),
            &8,
        );
        let N2 = size_per * num_partitions;
        let N = N2 / scaler;
        let code_size = num_ots * scaler;
        Self {
            scaler,
            weight,
            accumulator_size: 24,
            requested_num_ots: num_ots,
            N,
            N2,
            size_per,
            num_partitions,
            code_size,
        }
    }
}

impl From<ExConvConf> for PprfConfig {
    fn from(value: ExConvConf) -> Self {
        PprfConfig::new(value.size_per, value.num_partitions)
    }
}
