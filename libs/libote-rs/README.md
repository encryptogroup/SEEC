# libOTe-rs

This library builds and provides bindings to partial functionality of [libOTe](https://github.com/osu-crypto/libOTe).

Currently, we offer bindings to the Silver, EACode, and ExConvCode codes of libOTe. 


## Development
To compile this project, you need to have the git submodules at `libote` and those in `thirdparty/`cloned.

libOTe-rs can be compiled, tested, and added as dependency using the normal Cargo tools.


## SIMD Intrinsics
libOTe can be compiled with or without usage of SIMD intrinsics such as SSE2, AVX, or AES-NI. We detect the target architecture and available features in our [build script](./build.rs), and pass the appropriate flags to the libOTe build. To ensure these intrinsics are used if supported by your CPU, set the `RUSTFLAGS` environment variable to
```shell
export RUSTFLAGS="-Ctarget-cpu=native"
```
. This is not needed if you're within the SEEC workspace, as this option is set in the top-level `.cargo/config.toml`.
