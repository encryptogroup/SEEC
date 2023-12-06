use crate::{CommandExt, Optimization};
use anyhow::{anyhow, Context};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::available_parallelism;

#[derive(Debug, Clone)]
pub struct Build {
    source_dir: PathBuf,
    build_dir: PathBuf,
    opt: Optimization,
    targets: Vec<OsString>,
    cmake_args: Vec<OsString>,
    jobs: usize,
    clean: bool,
}

impl Build {
    pub fn new(source_dir: &Path) -> Self {
        let jobs = available_parallelism().map(usize::from).unwrap_or(1);
        let opt = Optimization::Release;
        Self {
            source_dir: source_dir.to_path_buf(),
            build_dir: opt.build_dir(source_dir),
            opt,
            targets: vec![],
            cmake_args: vec![],
            jobs,
            clean: false,
        }
    }

    pub fn optimization(&mut self, opt: Optimization) -> &mut Self {
        self.opt = opt;
        self.build_dir = opt.build_dir(&self.source_dir);
        self
    }

    pub fn cm_arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
        self.cmake_args.push(arg.as_ref().to_os_string());
        self
    }

    pub fn cm_args(&mut self, args: impl IntoIterator<Item = impl AsRef<OsStr>>) -> &mut Self {
        for arg in args {
            self.cm_arg(arg.as_ref());
        }
        self
    }

    pub fn target(&mut self, target: impl AsRef<OsStr>) -> &mut Self {
        self.targets.push(target.as_ref().to_os_string());
        self
    }

    pub fn targets(&mut self, targets: impl IntoIterator<Item = impl AsRef<OsStr>>) -> &mut Self {
        for target in targets {
            self.target(target);
        }
        self
    }

    pub fn clean(&mut self, clean: bool) -> &mut Self {
        self.clean = clean;
        self
    }

    pub fn build(&mut self) -> anyhow::Result<PathBuf> {
        if self.clean {
            // we don't care about the return, as this will fail if there is nothing to clean
            // make clean doesn't work for a remote build
            let _ = fs::remove_dir_all(&self.build_dir);
        }

        Command::new("cmake")
            .arg(self.opt.cmake_opt())
            .args(&self.cmake_args)
            .arg("-S")
            .arg(&self.source_dir)
            .arg("-B")
            .arg(&self.build_dir)
            .run()
            .with_context(|| {
                format!("Executing cmake to configure {}", self.source_dir.display())
            })?;

        Command::new("make")
            .arg("-j")
            .arg(&self.jobs.to_string())
            .args(&self.targets)
            .current_dir(&self.build_dir)
            .run()
            .with_context(|| anyhow!("Failed make targets {:?}", &self.targets))?;

        Ok(self.build_dir.clone())
    }
}
