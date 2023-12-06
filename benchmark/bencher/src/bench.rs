use crate::config::{BenchTarget, Framework, RemoteConfig};
use crate::net::NetSetting;
use crate::{record_memory, CommandExt, MemoryData};
use crate::{CloneCommand, IterExt};
use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use regex::Regex;
use seec_channel::Channel;
use serde::{Deserialize, Serialize};
use std::iter;
use std::net::SocketAddr;
use std::path::Path;
use std::process::Output;
use std::sync::Mutex;
use std::time::Instant;
use tempfile::NamedTempFile;
use tracing::{error, info};

// TODO, make enum to capture failed benchmarks

#[derive(Debug, Serialize)]
pub struct BenchData {
    args: Vec<String>,
    net_setting: Option<NetSetting>,
    tag: String,
    mem: Option<MemoryData>,
    runtime: RuntimeData,
    comm: CommData,
}

#[derive(Debug, Serialize)]
pub struct RuntimeData {
    wall_time_ms: u128,
    setup_time_ms: u128,
    online_time_ms: u128,
    original: OrigRuntimeData,
}

impl RuntimeData {
    pub fn from_orig(orig: OrigRuntimeData, wall_time_ms: u128) -> Self {
        match orig {
            OrigRuntimeData::Aby(ref inner) => Self {
                wall_time_ms,
                setup_time_ms: (inner.base_ots + inner.setup) as u128,
                online_time_ms: inner.online as u128,
                original: orig,
            },
            OrigRuntimeData::Motion(ref inner) => Self {
                wall_time_ms,
                setup_time_ms: (inner.preprocessing.mean + inner.gates_setup.mean) as u128,
                online_time_ms: inner.gates_online.mean as u128,
                original: orig,
            },
            OrigRuntimeData::MpSpdz(ref inner) => {
                // TODO how to handle setup time?
                Self {
                    wall_time_ms,
                    setup_time_ms: 0,
                    online_time_ms: (inner.time_s * 1000.0) as u128,
                    original: orig,
                }
            }
            OrigRuntimeData::Seec(ref inner) => {
                let mean = seec_channel::util::RunResult::mean(&inner.data);
                Self {
                    wall_time_ms,
                    setup_time_ms: mean.setup_ms(),
                    online_time_ms: mean.online_ms(),
                    original: orig,
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub enum OrigRuntimeData {
    Aby(AbyRuntimeData),
    Motion(MotionBenchRuntimeData),
    MpSpdz(MpSpdzRuntimeData),
    Seec(seec::bench::BenchResult),
}

#[derive(Debug, Serialize)]
pub struct CommData {
    bytes_sent: usize,
    bytes_received: usize,
    original: OrigCommData,
}

impl CommData {
    pub fn from_orig(orig: OrigCommData) -> Self {
        match orig {
            OrigCommData::Aby(ref inner) => Self {
                bytes_sent: inner.sent_bytes.total,
                bytes_received: inner.recv_bytes.total,
                original: orig,
            },
            OrigCommData::Motion(ref inner) => Self {
                bytes_sent: inner.bytes_sent,
                bytes_received: inner.bytes_received,
                original: orig,
            },
            OrigCommData::MpSpdz(ref inner) => {
                const MB: f64 = 1000.0 * 1000.0;
                Self {
                    bytes_sent: (inner.data_sent_mb * MB) as usize,
                    bytes_received: ((inner.global_data_sent_mb - inner.data_sent_mb) * MB)
                        as usize,
                    original: orig,
                }
            }
            OrigCommData::Seec(ref inner) => {
                let mean = seec_channel::util::RunResult::mean(&inner.data);
                Self {
                    bytes_sent: mean.total_bytes_sent(),
                    bytes_received: mean.total_bytes_recv(),
                    original: orig,
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub enum OrigCommData {
    Aby(AbyCommData),
    Motion(MotionBenchCommData),
    MpSpdz(MpSpdzCommData),
    Seec(seec::bench::BenchResult),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AbyRuntimeData {
    total: f64,
    init: f64,
    circuit_gen: f64,
    network: f64,
    base_ots: f64,
    setup: f64,
    ot_extension: f64,
    garbling: f64,
    online: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AbyCommData {
    sent_bytes: AbyComm,
    recv_bytes: AbyComm,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AbyComm {
    total: usize,
    base_ot: usize,
    setup: usize,
    ot_extension: usize,
    garbling: usize,
    online: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(unused)]
pub struct MotionBenchRuntimeData {
    repetitions: usize,
    mt_presetup: Aggr,
    mt_setup: Aggr,
    sp_presetup: Aggr,
    sp_setup: Aggr,
    sb_presetup: Aggr,
    sb_setup: Aggr,
    base_ots: Aggr,
    ot_extension_setup: Aggr,
    kk13_ot_extension_setup: Aggr,
    preprocessing: Aggr,
    gates_setup: Aggr,
    gates_online: Aggr,
    evaluate: Aggr,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Aggr {
    mean: f64,
    median: f64,
    stddev: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MotionBenchCommData {
    bytes_sent: usize,
    num_messages_sent: usize,
    bytes_received: usize,
    num_messages_received: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MpSpdzRuntimeData {
    time_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MpSpdzCommData {
    data_sent_mb: f64,
    global_data_sent_mb: f64,
}

impl Framework {
    pub fn party_args(&self, party: usize, parties: &[SocketAddr]) -> Vec<String> {
        assert_eq!(2, parties.len(), "Two parties needed");
        let server = &parties[0];
        match self {
            Framework::Seec => {
                vec![
                    "--id".to_string(),
                    party.to_string(),
                    "--server".to_string(),
                    server.to_string(),
                ]
            }
            Framework::Aby => {
                vec![
                    "-r".to_string(),
                    party.to_string(),
                    "-a".to_string(),
                    parties[0].ip().to_string(),
                    "-p".to_string(),
                    parties[0].port().to_string(),
                ]
            }
            Framework::Motion => {
                let fmt_party =
                    |id: usize| format!("{id},{},{}", parties[id].ip(), parties[id].port());
                vec![
                    "--my-id".to_string(),
                    party.to_string(),
                    "--parties".to_string(),
                    fmt_party(0),
                    fmt_party(1),
                ]
            }
            Framework::MpSpdz => {
                let host = parties[0];
                vec![
                    "-p".into(),
                    party.to_string(),
                    "-N".into(),
                    "2".into(),
                    "-h".into(),
                    host.ip().to_string(),
                    "-pn".into(),
                    host.port().to_string(),
                ]
            }
        }
    }

    fn parse_output(&self, out: &Output) -> Result<(OrigRuntimeData, OrigCommData)> {
        match self {
            Framework::Aby => {
                let out =
                    String::from_utf8(out.stdout.clone()).context("Motion stdout not UTF-8")?;
                let (runtime, comm) = out.split_once('\n').context("ABY out missing data")?;
                let runtime_data = serde_json::from_str(runtime).with_context(|| {
                    anyhow!(
                        "Failed to parse ABY runtime data. Input: {}",
                        runtime.clone()
                    )
                })?;
                let comm_data =
                    serde_json::from_str(comm).context("Failed to parse ABY comm data")?;
                Ok((
                    OrigRuntimeData::Aby(runtime_data),
                    OrigCommData::Aby(comm_data),
                ))
            }
            Framework::Motion => {
                let out =
                    String::from_utf8(out.stdout.clone()).context("Motion stdout not UTF-8")?;
                let (runtime, comm) = out.split_once('\n').context("Motion out missing data")?;
                let runtime_data =
                    serde_json::from_str(runtime).context("Failed to parse Motion runtime data")?;
                let comm_data =
                    serde_json::from_str(comm).context("Failed to parse Motion comm data")?;
                Ok((
                    OrigRuntimeData::Motion(runtime_data),
                    OrigCommData::Motion(comm_data),
                ))
            }
            Framework::MpSpdz => {
                let out =
                    String::from_utf8(out.stderr.clone()).context("MP-SPDZ stderr not UTF-8")?;
                let regexes = [
                    r"Time = ((?:\d|\.)+) seconds",
                    r"Data sent = ((?:\d|\.)+) MB",
                    r"Global data sent = ((?:\d|\.)+) MB",
                ]
                .map(|re| Regex::new(re).unwrap());

                let [time, data_sent, global_data_sent] = regexes.map(|re| {
                    let val = re
                        .captures(&out)
                        .with_context(|| anyhow!("Regex {re:?} did not match"))?
                        .get(1)
                        .with_context(|| anyhow!("Missing capture group for {re:?}"))?
                        .as_str()
                        .parse::<f64>()
                        .with_context(|| {
                            anyhow!("Unable to parse as f64 for capture of regex {re:?}")
                        })?;
                    Ok::<_, anyhow::Error>(val)
                });
                let time = time.context("Unable to parse MP-SPDZ time")?;
                let data_sent = data_sent.context("Unable to parse MP-SPDZ data_sent")?;
                let global_data_sent =
                    global_data_sent.context("Unable to parse MP-SPDZ global_data_sent")?;
                Ok((
                    OrigRuntimeData::MpSpdz(MpSpdzRuntimeData { time_s: time }),
                    OrigCommData::MpSpdz(MpSpdzCommData {
                        data_sent_mb: data_sent,
                        global_data_sent_mb: global_data_sent,
                    }),
                ))
            }
            Framework::Seec => {
                let bench_data: seec::bench::BenchResult = serde_json::from_slice(&out.stdout)
                    .context("Unable to parse seec bench output")?;
                Ok((
                    OrigRuntimeData::Seec(bench_data.clone()),
                    OrigCommData::Seec(bench_data),
                ))
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RunBenchMsg {
    Sync,
    Error(String),
}

impl BenchTarget {
    pub fn run(
        &self,
        party: usize,
        parties: &[SocketAddr],
        build_dir: &Path,
        heaptrack_dir: &Path,
        remote_conf: &Option<RemoteConfig>,
        quick_run: bool,
        sync_channel: &mut Channel<RunBenchMsg>,
    ) -> Result<Vec<Result<BenchData>>> {
        let synchronize = |sync_channel: &mut Channel<RunBenchMsg>| {
            let send_sync = || {
                sync_channel
                    .0
                    .blocking_send(RunBenchMsg::Sync)
                    .context("unable to send sync message")
            };

            let mut expect_sync = || {
                if let RunBenchMsg::Error(err) = sync_channel
                    .1
                    .blocking_recv()
                    .context("no sync message received")?
                    .context("missing sync msg")?
                {
                    anyhow::bail!("Received remote error: {err}")
                } else {
                    Ok::<(), anyhow::Error>(())
                }
            };

            match party {
                0 => {
                    send_sync()?;
                    expect_sync()?;
                    send_sync()?;
                    Ok::<(), anyhow::Error>(())
                }
                1 => {
                    expect_sync()?;
                    send_sync()?;
                    expect_sync()?;
                    Ok::<(), anyhow::Error>(())
                }
                _ => panic!("invalid party id"),
            }
        };

        let mut run_for_net_setting = |net_setting: Option<NetSetting>| {
            match (net_setting, remote_conf) {
                (Some(setting), Some(conf)) => {
                    setting
                        .configure(&conf)
                        .context(anyhow!("failed to configure net setting {setting:?}"))?;
                }
                (Some(_), None) => {
                    panic!("must provide remote conf to use net setting");
                }
                _ => (),
            };

            let mut results = vec![];

            let mut args: Vec<_> = self
                .prepare_commands(build_dir, party, parties)
                .context("unable to prepare cmds")?;
            if quick_run {
                args.truncate(1);
            }
            const MAX_RETRY: usize = 3;
            for (cmd, args) in args {
                let mut run_bench = |cmd: CloneCommand, args: Vec<String>| {
                    let mut cmd = cmd.into_cmd();
                    cmd.current_dir(build_dir);
                    synchronize(sync_channel)?;
                    let now = Instant::now();
                    let out = match cmd.run_output() {
                        Ok(out) => out,
                        Err(err) => {
                            let err = Err(err.context("Failed to run bench target"));
                            sync_channel
                                .0
                                .blocking_send(RunBenchMsg::Error(format!("{err:?}")))
                                .expect("Sync channel broken");
                            synchronize(sync_channel)?;
                            return err;
                        }
                    };
                    let time = now.elapsed();

                    let (runtime, comm) = match self.framework.parse_output(&out) {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            let err =
                                Err(err.context(anyhow!("Failed to parse bench output. {out:?}")));
                            sync_channel
                                .0
                                .blocking_send(RunBenchMsg::Error(format!("{err:?}")))
                                .expect("Sync channel broken");
                            synchronize(sync_channel)?;
                            return err;
                        }
                    };
                    let runtime = RuntimeData::from_orig(runtime, time.as_millis());
                    let comm = CommData::from_orig(comm);

                    synchronize(sync_channel)?;

                    // Only try to bench memory if cores are not restricted via taskset, as this
                    // will inevitably fail
                    let mem = if cmd.get_program() != "taskset" {
                        match record_memory(&cmd, self.persist_heaptrack, Some(heaptrack_dir)) {
                            Ok(mem) => Some(mem),
                            Err(err) => {
                                let err = err.context(anyhow!("failed to record memory"));
                                sync_channel
                                    .0
                                    .blocking_send(RunBenchMsg::Error(format!("{err:?}")))
                                    .expect("Sync channel broken");
                                synchronize(sync_channel)?;
                                error!(?err);
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let bench_data = BenchData {
                        args,
                        net_setting,
                        tag: self.tag.clone(),
                        mem,
                        runtime,
                        comm,
                    };
                    info!(?bench_data);

                    Ok(bench_data)
                };
                for _ in 0..MAX_RETRY {
                    match run_bench(cmd.clone(), args.clone()) {
                        res @ Ok(_) => {
                            results.push(res);
                            break;
                        }
                        Err(_err) if self.ignore_error => {
                            break;
                        }
                        Err(err) => {
                            error!(?err, "Encountered error, retrying.");
                        }
                    }
                }
            }
            Ok(results)
        };
        match &self.net_settings {
            None => run_for_net_setting(None),
            Some(settings) => {
                let mut results = vec![];
                for setting in settings {
                    results.extend(run_for_net_setting(Some(*setting))?);
                }
                Ok(results)
            }
        }
    }

    fn prepare_commands(
        &self,
        build_path: &Path,
        party: usize,
        parties: &[SocketAddr],
    ) -> Result<Vec<(CloneCommand, Vec<String>)>> {
        static TMP_FILES: Mutex<Vec<NamedTempFile>> = Mutex::new(Vec::new());

        let party_args = self.framework.party_args(party, parties);
        let flags: Vec<_> = self.flags.iter().flatten().cloned().collect();
        let repeat = self.repeat.map(|n| n.get()).unwrap_or(1);
        let mut commands = match &self.framework {
            Framework::Seec if self.compile_args.is_some() || self.compile_flags.is_some() => {
                let comp_arg_combinations = self
                    .compile_args
                    .iter()
                    .flatten()
                    .map(|(k, v)| v.iter().map(|arg| [k.clone(), arg.clone()]))
                    .multi_cartesian_product()
                    // Add empty args vec if self.compile args is empty
                    // so we have at least one bench target
                    .chain(if self.compile_args.is_none() {
                        Box::new(iter::once(vec![])) as Box<dyn Iterator<Item = Vec<[String; 2]>>>
                    } else {
                        Box::new(iter::empty())
                    });

                comp_arg_combinations
                    .map(|args| {
                        let mut cmd = CloneCommand::new(Path::new(".").join(&self.target));
                        cmd.current_dir(build_path);
                        cmd.arg("compile");
                        for arg_p in &args {
                            cmd.args(arg_p);
                        }
                        let out_file = NamedTempFile::new().context("unable to create tmp file")?;
                        cmd.arg("--output").arg(out_file.path());
                        cmd.args(self.compile_flags.iter().flatten());
                        cmd.into_cmd()
                            .run_output()
                            .context("failed to compile seec circuit")?;

                        let mut cmd = CloneCommand::new(Path::new(".").join(&self.target));
                        cmd.arg("execute");
                        cmd.args(party_args.clone());
                        cmd.args(self.flags.iter().flatten());
                        if self.pass_compile_args_to_execute {
                            cmd.args(args.iter().flatten());
                        }

                        cmd.arg(out_file.path());
                        // push the out_file to the static so that it is dropped at the end of the
                        // program
                        TMP_FILES.lock().unwrap().push(out_file);
                        let used_args = args
                            .into_iter()
                            .flatten()
                            .chain(self.compile_flags.iter().flatten().cloned())
                            .chain(self.flags.iter().flatten().cloned())
                            .collect();
                        Ok::<_, anyhow::Error>((cmd, used_args))
                    })
                    .collect()
            }
            Framework::Aby | Framework::Motion | Framework::Seec => {
                let arg_combinations = self
                    .args
                    .iter()
                    .flatten()
                    .map(|(k, v)| v.iter().map(|arg| [k.clone(), arg.clone()]))
                    .multi_cartesian_product();
                Ok(arg_combinations
                    .map(|args| {
                        let mut cmd = CloneCommand::new(Path::new(".").join(&self.target));
                        cmd.args(party_args.clone());
                        for arg in &args {
                            cmd.args(arg);
                        }
                        cmd.args(&flags);
                        (
                            cmd,
                            args.into_iter().flatten().chain(flags.clone()).collect(),
                        )
                    })
                    .collect())
            }
            Framework::MpSpdz => match &self.compile_flags {
                None => {
                    let mut cmd = CloneCommand::new("./semi-bin-party.x");
                    cmd.args(&party_args).arg(&self.target);
                    cmd.args(&flags);
                    Ok(vec![(cmd, flags.clone())])
                }
                Some(comp_flags) => Ok(comp_flags
                    .iter()
                    .map(|comp_flag| {
                        let mut cmd = CloneCommand::new("./semi-bin-party.x");
                        let target = format!("{}-{}", &self.target, comp_flag);
                        cmd.args(&party_args).args(&flags).arg(target);
                        (cmd, [flags.clone(), vec![comp_flag.clone()]].concat())
                    })
                    .collect()),
            },
        }?;
        if let Some(cores @ [_not_empty, ..]) = self.cores.as_deref() {
            commands = commands
                .into_iter()
                .flat_map(|(cmd, args)| {
                    cores.iter().map(move |count| {
                        if *count == 0 {
                            return (cmd.clone(), args.clone());
                        }
                        let mut reduced_cores_cmd = CloneCommand::new("taskset");
                        reduced_cores_cmd
                            .arg("--cpu-list")
                            .arg(format!("0-{}", *count - 1))
                            .arg(&cmd.program)
                            .args(&cmd.args)
                            .opt_current_dir(cmd.cwd.clone());
                        let mut args = args.clone();
                        args.push("cores".to_string());
                        args.push(format!("{count}"));
                        (reduced_cores_cmd, args)
                    })
                })
                .collect();
        }
        Ok(commands.into_iter().duplicate(repeat).collect())
    }
}
