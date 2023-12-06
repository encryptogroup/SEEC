use crate::config::BenchTarget;
use crate::{cmake, extract_targets, Optimization};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn build(
    motion_dir: &Path,
    targets: &[BenchTarget],
    opt: Optimization,
    clean_build: bool,
) -> Result<PathBuf> {
    let targets = extract_targets(targets);
    let build_dir = cmake::Build::new(motion_dir)
        .optimization(opt)
        .cm_arg("-DMOTION_BUILD_EXE=On")
        .cm_arg("-DBOOST_USE_SEGMENTED_STACKS=On")
        .targets(targets)
        .clean(clean_build)
        .build()
        .context("Failed building MOTION")?;

    Ok(build_dir.join("bin"))
}
