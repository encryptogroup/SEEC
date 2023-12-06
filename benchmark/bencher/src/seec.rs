use crate::config::BenchTarget;
use crate::{extract_targets, CommandExt, Optimization};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn build(
    seec_dir: &Path,
    targets: &[BenchTarget],
    opt: Optimization,
    clean_build: bool,
) -> Result<PathBuf> {
    let examples = extract_targets(targets);
    if clean_build {
        Command::new("cargo")
            .arg("clean")
            .current_dir(seec_dir)
            .run_output()
            .context("Failed to clean Seec dir")?;
    }
    let mut cargo_cmd = Command::new("cargo");
    cargo_cmd.arg("build").arg("--message-format=json");
    cargo_cmd.arg("--all-features");
    if let Some(opt) = opt.cargo_opt() {
        cargo_cmd.arg(opt);
    }

    for example in examples {
        cargo_cmd.arg("--example").arg(example);
    }

    let output = cargo_cmd
        .current_dir(seec_dir)
        .run_output()
        .context("Failed to run cargo for SEEC")?;

    let build_dir = cargo_metadata::Message::parse_stream(&output.stdout[..])
        .filter_map(Result::ok)
        .find_map(|msg| match msg {
            cargo_metadata::Message::CompilerArtifact(artifact) => {
                artifact.executable.map(|path| {
                    let mut exec_path = path.into_std_path_buf();
                    exec_path.pop();
                    // return the dir containing the executables
                    exec_path
                })
            }
            _ => None,
        })
        .context("Unable to parse build dir")?;
    Ok(build_dir)
}
