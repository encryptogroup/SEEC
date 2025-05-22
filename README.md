# SEEC Executes Enormous Circuits

![ci badge](https://github.com/encryptogroup/SEEC/actions/workflows/push.yml/badge.svg?branch=main) [![rustdoc](https://github.com/encryptogroup/SEEC/actions/workflows/rustdoc.yml/badge.svg)](https://encryptogroup.github.io/SEEC/seec/)

This framework implements secure 2-party secret-sharing-based multi party computation protocols. Currently, we implement
the Boolean and arithmetic versions of GMW87 with multiplication triple preprocessing. Additionally, we implement the
Boolean part of the ABY2.0 protocol. 

## Citing SEEC
If you use SEEC for your academic projects, please cite as follows:
```
@inproceedings{HKS24,
    author={Henri Dohmen and Robin Hundt and Nora Khayata and Thomas Schneider},
    title={{SEEC} — {M}emory Safety Meets Efficiency in Secure Two-Party Computation},
    booktitle={ASIACCS},
    year={2025},
}
```
Our poster abstract is available [here](https://encrypto.de/papers/HKS24Poster.pdf).

## Secure Multi-Party Computation

In secure multi-party computation (MPC), there are n parties, each with their private input x_i. Given a public function
f(x_1, ..., x_n), the parties execute a protocol π that correctly and securely realizes the functionality f. In other
words, at the end of the protocol, the parties know the output of f, but have no information about the input of the
other parties other than what is revealed by the output itself. Currently, SEEC is limited to the n = 2 party case, also
known as secure two-party computation. We hope to extend this in the future to n parties.

### Security

The two most prevalent security models are

- semi-honest security, where an attacker can corrupt parties, but they follow the protocol as specified.
- malicious security, where corrupted parties can arbitrarily deviate from the protocol.

SEEC currently only implements semi-honestly secure protocols (GMW, ABY2.0).

## Using SEEC

SEEC can be used as a library by adding it to the `Cargo.toml` file of an existing project.

```toml
[dependencies]
seec = { git = "https://github.com/encryptogroup/SEEC.git", features = ["..."] }
```

## Documentation

Documentation for the main branch is hosted [here](https://encryptogroup.github.io/SEEC/seec/).

## Development

### Installing Rust

The project is implemented in the [Rust](https://www.rust-lang.org/) programming language. To compile it, the latest
stable toolchain is needed (older toolchains might work but are not guaranteed). The recommended way to install it, is
via the toolchain manager [rustup](https://rustup.rs/).

One way of installing `rustup`:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

As a starting point to learn Rust, have a look at the superb
official [learning material](https://www.rust-lang.org/learn). To quickly look up syntax and idioms, we
recommend https://cheats.rs/.

### Checking for compilation errors

To simply check the code for error and warnings, execute:

```shell
cargo check
```

### Testing

The tests can be run with the following command:

```shell
cargo test [--release] [--all-features]
```

The `--release` flag is optional, but can decrease the runtime of the tests, at the cost of increased compilation time.
The `--all-features` flag enables all optional features.

### Formatting

This project uses `rustfmt` as its formatter. Code can be formatted with:

```shell
cargo fmt
```

## ARM Support

SEEC has (WIP) ARM support. Because we use the unstable `std::simd` feature of Rust for portable SIMD transport, a
recent nightly toolchain is needed. The easiest way to install this is via `rustup` (https://rustup.rs/).

```shell
rustup toolchain install nightly
```

Then set the toolchain to use for this project to nightly.

```shell
rustup override set nightly
```

And verify that nightly is used.

```shell
cargo --version
# output should include nightly
```

If on an ARM platform, building and testing all crates in this repository won't work, as some (
e.g. `crates/bitpolymul`), require x86_64 intrinsics. Offending packages can be `--exclude`d or you can simply change
into the main `crates/seec` directory and run cargo there.

## Silent-OT

Our OT library [ZappOT](./crates/zappot) has optional support for Silent-OT.
Using the quasi-cyclic code (https://eprint.iacr.org/2019/1159.pdf) requires an x86-64 CPU with AVX2 support. ZappOT has
WIP support for newer codes offered by libOTe (https://github.com/osu-crypto/libOTe). For these, we build and link to
the code implementations in libOTe. These should work on other architectures, e.g., ARM, but we currently do not test
this in CI.
Concretely, via libOTe, we support

- Silver (INSECURE! https://eprint.iacr.org/2021/1150, see https://eprint.iacr.org/2023/882 for attack)
- ExpandAccumulate (https://eprint.iacr.org/2022/1014)
- ExpandConvolute (https://eprint.iacr.org/2023/882).

SEEC currently supports generating Boolean MTs with Silent-OT when enabling the `silent-ot` feature.

> [!NOTE]  
> Silent-OT with the quasi-cyclic code (`--feature silent-ot-quasi-cyclic` in SEEC) only works on x86_64 linux with AVX2
> support. The libOTe codes (`--feature silent-ot`) currently work on x86_64 linux and aarch64 ARM (M1 Macs). Other
> targets might work, but are not tested.

## Organization

This project is organized as a Cargo workspace with multiple crates in the `crates/` directory. The main crate is
located at `crates/seec` and it depends on most of the other crates.

Also of interest is the `crates/zappot` library, which implements several oblivious transfer (OT) protocols. These are
used by SEEC to compute setup data such as Beaver multiplication triples, but they can also be used independently.

Main Crates:

- seec: The main library which implements several MPC protocols.
- seec-macros: Offers the `#[sub_circuit]` proc-macro that turns functions into reusable sub-circuits.
- seec-channel: A convenient wrapper over a fork of [remoc](https://github.com/ENQT-GmbH/remoc).
- seec-bitmatrix: A bitmatrix implementation including portable SIMD matrix transpose (needs Rust nightly).
- zappot: Our OT library, including support for Silent-OT.

We also provide an additional library at `libs/libote-rs` which builds and provides bindings to the codes used in libOTe
for its implementation of Silent-OT. These can be optionally used by ZappOT.

### Architecture

The figure below shows a simplified version of the main traits and types of SEEC.
![](figures/architecture.svg)

## Benchmarking

Alongside SEEC, we're developing an MPC [benchmarking tool](https://github.com/encryptogroup/mpc-bench).
