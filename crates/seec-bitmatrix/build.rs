#[rustversion::nightly]
fn main() {
    println!("cargo::rustc-check-cfg=cfg(is_nightly)");
    println!("cargo:rustc-cfg=is_nightly");
}

#[rustversion::not(nightly)]
fn main() {
    println!("cargo::rustc-check-cfg=cfg(is_nightly)");
}
