# SEEC Executes Enormous Circuits

This framework implements secure 2-party secret-sharing-based multi party computation protocols. Currently, we implement the Boolean and arithmetic versions of GMW87 with multiplication triple preprocessing. Additionally, we implement the Boolean part of the ABY2.0 protocol.

## Secure Multi-Party Computation
In secure multi-party computation (MPC), there are n parties, each with their private input x_i. Given a public function f(x_1, ..., x_n), the parties execute a protocol π that correctly and securely realizes the functionality f. In other words, at the end of the protocol, the parties know the output of f, but have no information about the input of the other parties other than what is revealed by the output itself. Currently, SEEC is limited to the n = 2 party case, also known as secure two-party computation. We hope to extend this in the future to n parties.

### Security
The two most prevalent security models are
- semi-honest security, where an attacker can corrupt parties, but they follow the protocol as specified.
- malicious security, where corrupted parties can arbitrarily deviate from the protocol.

SEEC currently only implements semi-honestly secure protocols (GMW, ABY2.0). 

## Development
### Installing Rust

The project is implemented in the [Rust](https://www.rust-lang.org/) programming language. To compile it, the latest stable toolchain is needed (older toolchains might work but are not guaranteed). The recommended way to install it, is via the toolchain manager [rustup](https://rustup.rs/).

One way of installing `rustup`:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Checking for compilation errors

To simply check the code for error and warnings, execute:

```shell
cargo check
```

### Testing

The tests can be run with the following command:

```shell
cargo test [--release]
```

The `--release` flag is optional, but can decrease the runtime of the tests, at the cost of increased compilation time.

### Formatting

This project uses `rustfmt` as its formatter. Code can be formatted with:

```shell
cargo fmt
```

## ARM Support
SEEC has (WIP) ARM support. Because we use the unstable `std::simd` feature of Rust for portable SIMD transport, a recent nightly toolchain is needed. The easiest way to install this is via `rustup` (https://rustup.rs/).
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

## Silent-OT
Our OT library [ZappOT](./crates/zappot) has optional support for Silent-OT. 
Using the quasi-cyclic code (https://eprint.iacr.org/2019/1159.pdf) requires an x86-64 CPU with AVX2 support. ZappOT has WIP support for newer codes offered by libOTe (https://github.com/osu-crypto/libOTe). For these, we build and link to the code implementations in libOTe. These should work on other architectures, e.g., ARM, but we currently do not test this in CI.
 Concretely, via libOTe, we support
- Silver (INSECURE! https://eprint.iacr.org/2021/1150, see https://eprint.iacr.org/2023/882 for attack)
- ExpandAccumulate (https://eprint.iacr.org/2022/1014)
- ExpandConvolute (https://eprint.iacr.org/2023/882).

SEEC currently supports generating Boolean MTs with Silent-OT when enabling the `silent-ot` feature.