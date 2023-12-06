use crate::net::NetSetting;
use anyhow::{Context, Result};
use clap::ValueEnum;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

pub type BuildPaths = HashMap<Framework, PathBuf>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteConfig {
    pub server: SocketAddr,
    pub client: SocketAddr,
    pub jump: String,
    pub remote_path: PathBuf,
    pub remote_hosts: [String; 2],
    pub heaptrack_dir: PathBuf,
    pub lan_cmd: String,
    pub wan_cmd: String,
    pub reset_net_cmd: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub net_settings: Option<Vec<NetSetting>>,
    pub repeat: Option<NonZeroUsize>,
    pub bench: Vec<BenchTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchTarget {
    pub framework: Framework,
    pub target: String,
    pub tag: String,
    pub args: Option<HashMap<String, Vec<String>>>,
    pub flags: Option<Vec<String>>,
    #[serde(default)]
    pub pass_compile_args_to_execute: bool,
    pub compile_args: Option<HashMap<String, Vec<String>>>,
    pub compile_flags: Option<Vec<String>>,
    #[serde(default)]
    pub persist_heaptrack: bool,
    pub net_settings: Option<Vec<NetSetting>>,
    pub repeat: Option<NonZeroUsize>,
    /// 0 will be treated as all cores
    pub cores: Option<Vec<usize>>,
    #[serde(default)]
    pub ignore_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ValueEnum, Hash, Eq, PartialEq)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum Framework {
    Aby,
    Motion,
    MpSpdz,
    Seec,
}

impl RemoteConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let file_content = fs::read_to_string(path).context("Failed to read remote config file")?;
        toml::from_str(&file_content).map_err(Into::into)
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let file_content = fs::read_to_string(path).context("Failed to read config file")?;
        let mut conf: Config =
            toml::from_str(&file_content).context("failed to deserialize bench config")?;
        // if global net_settings, set those that don't have an explicit one to global
        if let Some(net_settings) = &conf.net_settings {
            for bench_target in &mut conf.bench {
                bench_target
                    .net_settings
                    .get_or_insert_with(|| net_settings.clone());
            }
        }
        // same with repeat
        if let Some(repeat) = conf.repeat {
            for bench_target in &mut conf.bench {
                bench_target.repeat.get_or_insert(repeat);
            }
        }

        Ok(conf)
    }

    // If tags is empty, all bench targets are retained
    pub fn filter_by_tags(&mut self, tags: &Vec<String>) {
        if tags.is_empty() {
            return;
        }
        self.bench.retain(|b| tags.contains(&b.tag));
    }

    pub fn used_frameworks(&self) -> HashSet<Framework> {
        self.bench
            .iter()
            .map(|target| target.framework.clone())
            .unique()
            .collect()
    }

    pub fn targets_for(&self, framework: Framework) -> Vec<BenchTarget> {
        self.bench
            .iter()
            .filter_map(|bench| {
                if &bench.framework == &framework {
                    Some(bench.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::config::Config;
    use std::fs;

    #[test]
    fn parse_config() {
        let conf = fs::read_to_string("bench_config.toml").unwrap();
        let conf: Config = toml::from_str(&conf).expect("Deserializing conf failed");
    }
}
