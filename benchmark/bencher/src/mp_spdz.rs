use crate::config::BenchTarget;
use crate::{in_dir, CommandExt};
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn build(mp_spdz_dir: &Path, clean_build: bool) -> Result<PathBuf> {
    if clean_build {
        in_dir(mp_spdz_dir, || {
            let _ = Command::new("make").arg("clean").run();
            Command::new("make")
                .arg("setup")
                .arg("Programs/Circuits")
                .arg("-j")
                .run()
                .context("Failed MP-SPDZ setup")?;
            Command::new("make")
                .arg("semi-bin-party.x")
                .arg("-j")
                .run()
                .context("Failed to build MP-SPDZ binaries")?;

            Ok::<_, anyhow::Error>(())
        })??;
    }

    // binaries are placed in the src dir
    Ok(mp_spdz_dir.to_path_buf())
}

pub fn compile(mp_spdz_dir: &Path, targets: &[BenchTarget]) -> Result<()> {
    let create_cmd = |target: &BenchTarget, compile_arg: Option<&String>| {
        let mut cmd = Command::new("./compile.py");
        cmd.current_dir(mp_spdz_dir)
            .args(["-B", "1"])
            .arg(&target.target);
        if let Some(arg) = compile_arg {
            cmd.arg(arg);
        }
        cmd
    };

    for target in targets {
        ensure!(
            target.compile_args.is_none(),
            "compile_args not permitted for MP-SPDZ"
        );
        match &target.compile_flags {
            Some(flags) => {
                for flag in flags {
                    create_cmd(target, Some(flag))
                        .run_output()
                        .with_context(|| format!("Failed to compile program: {}", target.target))?;
                }
            }
            None => {
                create_cmd(target, None)
                    .run_output()
                    .with_context(|| format!("Failed to compile program: {}", target.target))?;
            }
        }
    }
    Ok(())
}
