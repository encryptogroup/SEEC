use crate::config::BenchTarget;
use crate::{cmake, extract_targets, Optimization};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn build(
    aby_dir: &Path,
    targets: &[BenchTarget],
    opt: Optimization,
    clean_build: bool,
) -> Result<PathBuf> {
    let targets = extract_targets(targets);
    let build_dir = cmake::Build::new(aby_dir)
        .optimization(opt)
        .clean(clean_build)
        .cm_arg("-DABY_BUILD_EXE=On")
        .targets(targets)
        .build()
        .context("Failed building ABY")?;
    Ok(build_dir.join("bin"))
}
