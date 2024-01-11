use cxx::{type_id, ExternType, UniquePtr};
use std::fmt::{Debug, Formatter};

pub struct SilverEncoder {
    inner: UniquePtr<ffi::SilverEncBridge>,
}

// TODO: Safety...
unsafe impl Send for ffi::SilverEncBridge {}
unsafe impl Sync for ffi::SilverEncBridge {}

#[derive(Copy, Clone, Debug)]
pub enum SilverCode {
    Weight5 = 5,
    Weight11 = 11,
}

impl SilverEncoder {
    pub fn new(code: SilverCode, rows: u64) -> Self {
        let mut inner = ffi::new_enc();
        inner.pin_mut().init(rows, code.into());
        Self { inner }
    }

    pub fn dual_encode(&mut self, c: &mut [Block]) {
        self.inner.pin_mut().dual_encode_block(c);
    }

    pub fn dual_encode2(&mut self, c0: &mut [Block], c1: &mut [u8]) {
        self.inner.pin_mut().dual_encode2_block(c0, c1);
    }
}

impl Debug for SilverEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SilverEncoder").finish()
    }
}

impl SilverCode {
    pub fn gap(&self) -> u64 {
        ffi::SilverCode::from(*self).gap()
    }
}

impl From<SilverCode> for ffi::SilverCode {
    fn from(value: SilverCode) -> Self {
        match value {
            SilverCode::Weight5 => ffi::SilverCode {
                weight: ffi::SilverCodeWeight::Weight5,
            },
            SilverCode::Weight11 => ffi::SilverCode {
                weight: ffi::SilverCodeWeight::Weight11,
            },
        }
    }
}

pub struct EACode {
    inner: UniquePtr<ffi::EACodeBridge>,
}

// TODO: Safety...
unsafe impl Send for ffi::EACodeBridge {}
unsafe impl Sync for ffi::EACodeBridge {}

impl EACode {
    pub fn new(message_size: u64, code_size: u64, expander_weight: u64) -> Self {
        let mut inner = ffi::new_ea_code();
        inner
            .pin_mut()
            .config(message_size, code_size, expander_weight);
        Self { inner }
    }

    pub fn dual_encode_block(&mut self, e: &mut [Block], w: &mut [Block]) {
        self.inner.pin_mut().dual_encode_block(e, w);
    }

    pub fn dual_encode2_block(
        &mut self,
        e0: &mut [Block],
        w0: &mut [Block],
        e1: &mut [u8],
        w1: &mut [u8],
    ) {
        self.inner.pin_mut().dual_encode2_block(e0, w0, e1, w1);
    }
}

impl Debug for EACode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EACode").finish()
    }
}

pub struct ExConvCode {
    inner: UniquePtr<ffi::ExConvCodeBridge>,
}

// TODO: Safety...
unsafe impl Send for ffi::ExConvCodeBridge {}
unsafe impl Sync for ffi::ExConvCodeBridge {}

impl ExConvCode {
    pub fn new(
        message_size: u64,
        code_size: u64,
        expander_weight: u64,
        accumulator_weight: u64,
    ) -> Self {
        let mut inner = ffi::new_ex_conv_code();
        inner
            .pin_mut()
            .config(message_size, code_size, expander_weight, accumulator_weight);
        Self { inner }
    }

    pub fn dual_encode_block(&mut self, e: &mut [Block]) {
        self.inner.pin_mut().dual_encode_block(e);
    }

    ///
    /// # Panics
    /// If e1 is not aligned to a 16-byte boundary.
    pub fn dual_encode2_block(&mut self, e0: &mut [Block], e1: &mut [u8]) {
        assert_eq!(
            0,
            e1.as_ptr() as usize % 16,
            "e1 must be 16-byte aligned. Allocate buffer with aligned-vec"
        );
        self.inner.pin_mut().dual_encode2_block(e0, e1);
    }
}

impl Debug for ExConvCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExConvCode").finish()
    }
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug, Default, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct Block {
    data: u128,
}

unsafe impl ExternType for Block {
    type Id = type_id!("osuCrypto::block");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge(namespace = "osuCryptoBridge")]
mod ffi {

    struct SilverCode {
        weight: SilverCodeWeight,
    }

    #[repr(u32)]
    enum SilverCodeWeight {
        Weight5 = 5,
        Weight11 = 11,
    }

    unsafe extern "C++" {
        include!("libote/src/SilentEncoderBridge.h");

        type SilverEncBridge;

        type SilverCodeWeight;
        #[namespace = "osuCrypto"]
        type SilverCode;

        #[namespace = "osuCrypto"]
        #[cxx_name = "block"]
        type Block = super::Block;

        #[cxx_name = "newEnc"]
        fn new_enc() -> UniquePtr<SilverEncBridge>;

        fn init(self: Pin<&mut SilverEncBridge>, rows: u64, code: SilverCode);

        #[cxx_name = "dualEncodeBlock"]
        fn dual_encode_block(self: Pin<&mut SilverEncBridge>, c: &mut [Block]);

        #[cxx_name = "dualEncode2Block"]
        fn dual_encode2_block(self: Pin<&mut SilverEncBridge>, c0: &mut [Block], c1: &mut [u8]);

        #[namespace = "osuCrypto"]
        fn gap(self: &mut SilverCode) -> u64;

        type EACodeBridge;

        #[cxx_name = "newEACode"]
        fn new_ea_code() -> UniquePtr<EACodeBridge>;

        fn config(
            self: Pin<&mut EACodeBridge>,
            message_size: u64,
            code_size: u64,
            expander_weight: u64,
        );

        #[cxx_name = "dualEncodeBlock"]
        fn dual_encode_block(self: Pin<&mut EACodeBridge>, e: &mut [Block], w: &mut [Block]);

        #[cxx_name = "dualEncode2Block"]
        fn dual_encode2_block(
            self: Pin<&mut EACodeBridge>,
            e0: &mut [Block],
            w0: &mut [Block],
            e1: &mut [u8],
            w1: &mut [u8],
        );

        type ExConvCodeBridge;
        #[cxx_name = "newExConvCode"]
        fn new_ex_conv_code() -> UniquePtr<ExConvCodeBridge>;

        fn config(
            self: Pin<&mut ExConvCodeBridge>,
            message_size: u64,
            code_size: u64,
            expander_weight: u64,
            accumulator_size: u64,
        );

        #[cxx_name = "dualEncodeBlock"]
        fn dual_encode_block(self: Pin<&mut ExConvCodeBridge>, e: &mut [Block]);

        #[cxx_name = "dualEncode2Block"]
        fn dual_encode2_block(self: Pin<&mut ExConvCodeBridge>, e0: &mut [Block], e1: &mut [u8]);
    }
}

#[cfg(test)]
mod tests {
    use crate::ffi::{SilverCode, SilverCodeWeight};
    use crate::{ffi, Block, EACode, ExConvCode};
    use std::slice;

    #[test]
    fn create_silver_encoder() {
        let _enc = ffi::new_enc();
    }

    #[test]
    fn init_encoder() {
        let mut enc = ffi::new_enc();
        enc.pin_mut().init(
            50,
            SilverCode {
                weight: SilverCodeWeight::Weight5,
            },
        );
    }

    #[test]
    fn dual_encode() {
        let mut enc = ffi::new_enc();
        enc.pin_mut().init(
            50,
            SilverCode {
                weight: SilverCodeWeight::Weight5,
            },
        );
        let mut c = vec![Block::default(); 100];
        enc.pin_mut().dual_encode_block(&mut c);
    }

    #[test]
    fn dual_encode2() {
        let mut enc = ffi::new_enc();
        enc.pin_mut().init(
            50,
            SilverCode {
                weight: SilverCodeWeight::Weight5,
            },
        );
        let mut c0 = vec![Block::default(); 100];
        let mut c1 = vec![0; 100];
        enc.pin_mut().dual_encode2_block(&mut c0, &mut c1);
    }

    #[test]
    fn new_ea_code() {
        let _code = EACode::new(100, 200, 7);
    }

    #[test]
    fn eac_dual_encode() {
        let mut code = EACode::new(100, 200, 7);
        let mut e = vec![Block::default(); 200];
        let mut w = vec![Block::default(); 100];
        code.dual_encode_block(&mut e, &mut w)
    }

    #[test]
    fn eac_dual_encode2() {
        let mut code = EACode::new(100, 200, 7);
        let mut e0 = vec![Block::default(); 200];
        let mut e1 = vec![0; 200];
        let mut w0 = vec![Block::default(); 100];
        let mut w1 = vec![0; 100];
        code.dual_encode2_block(&mut e0, &mut w0, &mut e1, &mut w1);
    }

    #[test]
    fn new_ex_conv_code() {
        let _code = ExConvCode::new(100, 200, 7, 16);
    }

    #[test]
    fn ex_conv_dual_encode() {
        let mut code = ExConvCode::new(100, 200, 7, 16);
        let mut e = vec![Block::default(); 200];
        code.dual_encode_block(&mut e)
    }

    #[test]
    fn ex_conv_dual_encode2() {
        let mut code = ExConvCode::new(256, 512, 7, 24);
        let mut e0 = vec![Block::default(); 512];
        // create aligned u8 slice
        let mut e1 = vec![Block::default(); 512 / 16];
        let e1 = unsafe { slice::from_raw_parts_mut(e1.as_mut_ptr() as *mut u8, 512) };
        code.dual_encode2_block(&mut e0, e1);
    }
}
