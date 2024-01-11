use std::path::Path;
use std::{env, fs};

/// This build script builds the required parts of libOTe and cryptotools necessary
/// for the Silver/EaCode/ExConvCode codes for silent OT.
fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=libOTe");
    println!("cargo:rerun-if-changed=thirdparty");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    fs::create_dir_all(out_dir.join("libOTe/libOTe")).unwrap();
    fs::create_dir_all(out_dir.join("cryptoTools/cryptoTools/Common")).unwrap();
    fs::copy(
        "src/libOTe_config.h",
        out_dir.join("libOTe/libOTe/config.h"),
    )
    .unwrap();
    fs::copy(
        "src/cryptoTools_config.h",
        out_dir.join("cryptoTools/cryptoTools/Common/config.h"),
    )
    .unwrap();

    let (sse_enabled, aes_enabled, avx_enabled) = {
        let env = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        let target_features: Vec<_> = env.split(',').collect();
        let sse = target_features.contains(&"sse2");
        let aes = target_features.contains(&"aes");
        let avx2 = target_features.contains(&"avx2");
        (sse, aes, avx2)
    };

    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .file("libOTe/libOTe/Tools/LDPC/LdpcEncoder.cpp")
        .file("libOTe/libOTe/Tools/EACode/EACode.cpp")
        .file("libOTe/libOTe/Tools/EACode/Expander.cpp")
        .file("libOTe/libOTe/Tools/EACode/EACodeInstantiations.cpp")
        .file("libOTe/libOTe/Tools/EACode/ExpanderInstantiations.cpp")
        .file("libOTe/libOTe/Tools/ExConvCode/ExConvCode.cpp")
        .file("libOTe/libOTe/Tools/ExConvCode/ExConvCodeInstantiations.cpp")
        .file("libOTe/cryptoTools/cryptoTools/Common/Timer.cpp")
        .file("libOTe/cryptoTools/cryptoTools/Common/Log.cpp")
        .file("libOTe/cryptoTools/cryptoTools/Common/block.cpp")
        .file("libOTe/cryptoTools/cryptoTools/Crypto/PRNG.cpp")
        .file("libOTe/cryptoTools/cryptoTools/Crypto/AES.cpp")
        .includes(&[
            Path::new("src"),
            Path::new("libOTe"),
            Path::new("libOTe/cryptoTools"),
            Path::new("thirdparty/libdivide"),
            Path::new("thirdparty/span-lite/include"),
            out_dir.join("libOTe").as_path(),
            out_dir.join("cryptoTools").as_path(),
        ])
        .warnings(false)
        .flag("-std=c++17")
        .flag("-march=native")
        .define("ENABLE_INSECURE_SILVER", "ON");

    if sse_enabled {
        build
            .define("ENABLE_SSE", None)
            .define("OC_ENABLE_SSE2", None);
    }
    if avx_enabled {
        build.define("ENABLE_AVX", None);
    }
    if aes_enabled {
        build.flag("-mpclmul");
        build
            .define("OC_ENABLE_PCLMUL", None)
            .define("OC_ENABLE_AESNI", None);
    } else {
        build.define("OC_ENABLE_PORTABLE_AES", None);
    }

    build.compile("silent_encoder_bridge");
}
