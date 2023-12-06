use crate::config::BenchTarget;
use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use regex::Regex;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Duration;
use std::{env, fs, io, iter, thread};
use tempfile::NamedTempFile;
use tracing::info;

pub mod aby;
pub mod bench;
pub mod cmake;
pub mod config;
pub mod motion;
pub mod mp_spdz;
pub mod net;
pub mod rsync;
pub mod seec;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
pub enum Optimization {
    Debug,
    Release,
}

impl Optimization {
    pub fn cmake_opt(&self) -> &'static str {
        match self {
            Optimization::Debug => "-DCMAKE_BUILD_TYPE=Debug",
            Optimization::Release => "-DCMAKE_BUILD_TYPE=Release",
        }
    }

    pub fn cargo_opt(&self) -> Option<&'static str> {
        match self {
            Optimization::Debug => None,
            Optimization::Release => Some("--release"),
        }
    }

    pub fn build_dir(&self, base: &Path) -> PathBuf {
        base.join(format!("build-{:?}", self))
    }
}

impl Display for Optimization {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Optimization::Debug => write!(f, "debug"),
            Optimization::Release => write!(f, "release"),
        }
    }
}

/// Execute f in dir and change back to previous working dir. f should not panic
pub fn in_dir<F, R>(dir: impl AsRef<Path>, mut f: F) -> Result<R>
where
    F: FnMut() -> R,
{
    let cd = env::current_dir().context("Unable to get current dir")?;
    env::set_current_dir(dir).context("Unable to set to temporary dir")?;
    let ret = f();
    env::set_current_dir(&cd).context("Unable to reset to original dir")?;
    Ok(ret)
}

pub fn create_build_dir(base_dir: &Path, opt: Optimization) -> Result<PathBuf> {
    let path = opt.build_dir(base_dir);

    match fs::create_dir(&path) {
        Ok(_) => Ok(path),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Ok(path),
        Err(err) => Err(err.into()),
    }
}

pub trait CommandExt {
    fn run(&mut self) -> Result<()>;
    fn run_output(&mut self) -> Result<Output>;
}

impl CommandExt for Command {
    fn run(&mut self) -> Result<()> {
        info!("Running: {self:?}");
        let status = self
            .status()
            .with_context(|| anyhow!("Failed to execute command: {:?}", self))?;
        if status.success() {
            Ok(())
        } else {
            Err(anyhow!(
                "Command failed. Executed: {self:?}\nExit code: {}",
                status
            ))
        }
    }

    fn run_output(&mut self) -> Result<Output> {
        info!("Running: {self:?}");
        let output = self
            .output()
            .with_context(|| anyhow!("Failed to execute command: {:?}", self))?;
        if output.status.success() {
            Ok(output)
        } else {
            let stdout = String::from_utf8_lossy(&output.stdout).to_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).to_owned();
            Err(anyhow!(
                "Command failed. Executed: {self:?}\nStdout:\n{}\nStderr:\n{}\nExitcode:{}",
                stdout,
                stderr,
                output.status
            ))
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct MemoryData {
    debuggee: String,
    runtime: String,
    max_heap: String,
    allocations: usize,
}

pub fn record_memory(
    cmd: &Command,
    persist_heaptrack_file: bool,
    heaptrack_dir: Option<&Path>,
) -> Result<MemoryData> {
    let (heaptrack, heaptrack_print) = if let Some(path) = heaptrack_dir {
        (
            path.join("build/bin/heaptrack").into_os_string(),
            path.join("build/bin/heaptrack_print").into_os_string(),
        )
    } else {
        ("heaptrack".into(), "heaptrack_print".into())
    };

    let target = Path::new(cmd.get_program())
        .components()
        .last()
        .context("Missing program")?
        .as_os_str();

    let out_file = NamedTempFile::new()?.into_temp_path();

    let mut heaptrack_cmd = Command::new(&heaptrack);
    if let Some(dir) = cmd.get_current_dir() {
        heaptrack_cmd.current_dir(dir);
    }
    heaptrack_cmd
        .arg("--record-only")
        .arg("-o")
        .arg(&out_file)
        .arg(cmd.get_program())
        .args(cmd.get_args())
        .run_output()
        .with_context(|| anyhow!("Failed memory profiling for {}", target.to_string_lossy()))?;

    let mut out_file = out_file.to_path_buf();
    out_file.set_extension("zst");

    thread::sleep(Duration::from_secs(1));
    let output = Command::new(&heaptrack_print)
        .args([
            "-m", "0", "-t", "0", "-p", "0", "-a", "0", "-l", "0", "-s", "0", "-T", "0", "-n", "0",
        ])
        .arg(&out_file)
        .run_output()
        .context("Failed to read heaptrack file")?;
    let output = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_heaptrack_out(output).context("Failed to parse heaptrack_print output");
    if persist_heaptrack_file {
        let renamed = Path::new(out_file.file_name().unwrap());
        info!("Persisting heaptrack file: {}", renamed.display());
        fs::copy(&out_file, renamed).context("Unable to persists heaptrack file")?;
    }
    parsed
}

fn parse_heaptrack_out(stdout: Cow<'_, str>) -> Result<MemoryData> {
    let regexes = [
        r"Debuggee command was:\s(.*)",
        r"total runtime:\s((?:\d|\.)+\p{Alphabetic}+)",
        r"calls to allocation functions:\s(\d+)",
        r"peak heap memory consumption:\s((?:\d|\.)+\p{Alphabetic}+)",
    ]
    .map(|re| Regex::new(re).unwrap());

    let [debugee, time, allocs, peak] = regexes.map(|re| {
        let grp = re
            .captures(&stdout)
            .with_context(|| anyhow!("Regex \"{re:?}\" did not match in:\n{}", &stdout))?
            .get(1)
            .with_context(|| anyhow!("Missing capture group for {re:?}"))?
            .as_str()
            .to_owned();
        Ok::<_, anyhow::Error>(grp)
    });

    let allocations = allocs?
        .parse()
        .context("call to allocations not a number")?;

    Ok(MemoryData {
        debuggee: debugee?,
        runtime: time?,
        max_heap: peak?,
        allocations,
    })
}

fn extract_targets(bench_targets: &[BenchTarget]) -> HashSet<String> {
    bench_targets
        .iter()
        .map(|bench| bench.target.clone())
        .collect()
}

#[derive(Debug, Default, Clone)]
/// Restricted but clonable version of a Command
pub struct CloneCommand {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub cwd: Option<PathBuf>,
}

impl CloneCommand {
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Self {
            program: program.as_ref().to_os_string(),
            ..Default::default()
        }
    }

    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.args.push(arg.as_ref().to_os_string());
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.arg(arg);
        }
        self
    }

    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
        self.cwd = Some(dir.as_ref().to_path_buf());
        self
    }

    pub fn opt_current_dir<P: AsRef<Path>>(&mut self, dir: Option<P>) -> &mut Self {
        self.cwd = dir.map(|p| p.as_ref().to_path_buf());
        self
    }

    pub fn into_cmd(self) -> Command {
        let mut cmd = Command::new(self.program);
        cmd.args(self.args);
        if let Some(cwd) = self.cwd {
            cmd.current_dir(cwd);
        }
        cmd
    }
}

pub trait IterExt: Iterator {
    fn duplicate<'a>(self, n: usize) -> Box<dyn Iterator<Item = Self::Item> + 'a>
    where
        Self::Item: Clone + 'a,
        Self: Sized + 'a,
    {
        Box::new(self.flat_map(move |el| iter::repeat(el).take(n)))
    }
}

impl<I: Iterator> IterExt for I {}

#[cfg(test)]
mod tests {
    use crate::{parse_heaptrack_out, MemoryData};
    use anyhow::Result;
    use std::borrow::Cow;

    #[test]
    fn heaptrack_out() -> Result<()> {
        let inp = Cow::Borrowed(
            r#"
reading file "heaptrack.semi-bin-party.x.1138739.zst" - please wait, this might take some time...
Debuggee command was: ./semi-bin-party.x 0 aescbc_circuit-10 -F -pn 16499 -h localhost -N 2
finished reading file, now analyzing data:

total runtime: 23.955000s.
calls to allocation functions: 85286 (3560/s)
temporary memory allocations: 1303 (54/s)
peak heap memory consumption: 7.53M
peak RSS (including heaptrack overhead): 15.02M
total memory leaked: 13.89K
        "#,
        );
        let mem_data = parse_heaptrack_out(inp)?;
        assert_eq!(
            mem_data,
            MemoryData {
                debuggee: "./semi-bin-party.x 0 aescbc_circuit-10 -F -pn 16499 -h localhost -N 2"
                    .to_string(),
                runtime: "23.955000s".to_string(),
                max_heap: "7.53M".to_string(),
                allocations: 85286,
            }
        );
        Ok(())
    }
}
