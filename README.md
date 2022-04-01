# GMW

This framework implements the GMW secure multi party computation protocol.

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
