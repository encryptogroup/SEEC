use anyhow::{anyhow, Context, Result};
use bencher::bench::RunBenchMsg;
use bencher::config::{BenchTarget, Config, Framework, RemoteConfig};
use bencher::rsync::do_rsync;
use bencher::{aby, motion, mp_spdz, net, seec, CommandExt, Optimization};
use chrono::Local;
use clap::Parser;
use itertools::Itertools;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use std::{fs, thread};
use tracing::level_filters::LevelFilter;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug, Clone)]
struct ProgArgs {
    #[arg(long, default_value = "bench_config.toml")]
    conf: PathBuf,
    #[arg(long)]
    clean_build: bool,
    #[arg(long, value_enum, default_value = "release")]
    opt: Optimization,
    #[arg(long, requires_all = ["server", "client"])]
    id: Option<usize>,
    #[arg(long, requires = "id")]
    server: Option<SocketAddr>,
    #[arg(long, requires = "id")]
    client: Option<SocketAddr>,
    #[arg(short, long)]
    out: Option<PathBuf>,
    #[arg(long)]
    remote_conf: Option<PathBuf>,
    #[arg(long)]
    is_remote: bool,
    #[arg(long)]
    build_on_remote: bool,
    #[arg(long, default_value = "../heaptrack")]
    heaptrack_dir: PathBuf,
    #[arg(long, default_value = "../../")]
    project_root: PathBuf,
    // Only run the first configuration of each target
    #[arg(long)]
    quick_run: bool,
    #[arg(short, long)]
    tags: Vec<String>,
}

#[tokio::main(worker_threads = 4)]
async fn main() -> Result<()> {
    let args = ProgArgs::parse();
    init_tracing();
    let start_t = Instant::now();

    let mut config = Config::load(&args.conf)?;
    config.filter_by_tags(&args.tags);
    let build_paths = build(&args, &config).context("failed to build bench targets")?;
    info!("Build took {:?}", start_t.elapsed());

    let remote_conf = match (args.is_remote, &args.remote_conf) {
        (false, Some(remote_conf_path)) => {
            let conf =
                RemoteConfig::load(remote_conf_path).context("failed to load remote conf")?;
            write_build_paths(&build_paths).context("Unable to save build paths")?;
            do_rsync(&conf)?;
            thread::scope(|s| {
                s.spawn(|| execute_remotely(0, &conf, &args));
                s.spawn(|| execute_remotely(1, &conf, &args));
            });
            return Ok(());
        }
        (true, Some(remote_conf_path)) => {
            Some(RemoteConfig::load(remote_conf_path).context("failed to load remote conf")?)
        }
        (false, None) => None,
        (true, None) => {
            panic!("If --is-remote, --remote-conf needs to be provided");
        }
    };
    let mut out_file = args
        .out
        .map(|p| File::create(p).expect("Unable to create out file"));

    let parties = match (args.server, args.client) {
        (Some(server), Some(client)) => [server, client],
        (None, None) => ["127.0.0.1:7744".parse()?, "127.0.0.1:7745".parse()?],
        _ => anyhow::bail!("either both --server and --client must be provided or neither"),
    };

    let mut base_sync_channel = match args.id {
        Some(0) => {
            let (sender, _, receiver, _) = seec_channel::tcp::listen(&parties[0])
                .await
                .context("unable to to create sync server channel")?;
            Some((sender, receiver))
        }
        Some(1) => {
            let (sender, _, receiver, _) =
                seec_channel::tcp::connect_with_timeout(&parties[0], Duration::from_secs(120))
                    .await
                    .context("unable to connect to sync channel")?;
            Some((sender, receiver))
        }
        Some(illegal) => {
            anyhow::bail!("Illegal id {illegal}")
        }
        None => None,
    };

    let mut sync_channel = match &mut base_sync_channel {
        Some(base_ch) => Some(
            seec_channel::sub_channels_for!(&mut base_ch.0, &mut base_ch.1, 8, RunBenchMsg)
                .await
                .context("unable to establish sync channel")?,
        ),
        None => None,
    };

    for bench in config.bench {
        info!(?bench, "Executing bench target");
        let build_path = build_paths.get(&bench.framework).context(anyhow!(
            "Bench target for framework {:?} defined but not built.",
            &bench.framework
        ))?;
        let res = thread::scope(|s| match args.id {
            Some(id) => {
                let bench = &bench;
                let heaptrack_dir = &args.heaptrack_dir;
                let remote_conf = &remote_conf;
                let ch = sync_channel.as_mut().unwrap();
                s.spawn(move || {
                    bench.run(
                        id,
                        &parties,
                        &build_path,
                        heaptrack_dir,
                        &remote_conf,
                        args.quick_run,
                        ch,
                    )
                })
            }
            .join(),
            None => {
                let (ch0, ch1) = seec_channel::in_memory::new_pair(8);

                let handle0 = s.spawn(|| {
                    let mut ch0 = ch0;
                    bench.run(
                        0,
                        &parties,
                        &build_path,
                        &args.heaptrack_dir,
                        &remote_conf,
                        args.quick_run,
                        &mut ch0,
                    )
                });
                let handle1 = s.spawn(|| {
                    let mut ch1 = ch1;

                    bench.run(
                        1,
                        &parties,
                        &build_path,
                        &args.heaptrack_dir,
                        &remote_conf,
                        args.quick_run,
                        &mut ch1,
                    )
                });
                let _ = handle0.join();
                handle1.join()
            }
        });
        let res: Vec<_> = match res {
            Ok(Ok(data)) => data
                .into_iter()
                .filter_map(|bench_data| match bench_data {
                    Ok(val) => Some(val),
                    Err(err) => {
                        error!(?err);
                        None
                    }
                })
                .collect(),
            err => {
                error!(?err);
                continue;
            }
        };
        let serialized = match serde_json::to_string(&(bench, res)) {
            Ok(data) => data,
            Err(err) => {
                error!("{err:#?}");
                continue;
            }
        };
        if let Some(out) = &mut out_file {
            writeln!(out, "{serialized}").context("failed to write to out file")?;
        } else {
            println!("{serialized}");
        }
    }

    if let (Some(remote_conf), false) = (&remote_conf, args.is_remote) {
        net::NetSetting::reset(remote_conf).context("failed to reset net setting")?;
    }

    info!("Total time: {:?}", start_t.elapsed());

    Ok(())
}

fn build(args: &ProgArgs, config: &Config) -> Result<HashMap<Framework, PathBuf>> {
    match (args.is_remote, args.build_on_remote) {
        (true, false) => {
            let build_paths = fs::read_to_string("./benchmark/bencher/target/build-paths.toml")
                .context("failed to read build-paths.toml")?;
            let mut pre_built: HashMap<_, _> =
                toml::from_str(&build_paths).context("failed to deserialize build-paths.toml")?;
            let used_frameworks = config.used_frameworks();
            // MP-SPDZ needs to be built remotely everytime
            if used_frameworks.contains(&Framework::MpSpdz) {
                let mp_spdz_targets = config.targets_for(Framework::MpSpdz);
                let mp_spdz_path =
                    build_mp_spdz(&args, &mp_spdz_targets).context("Unable to build MpSpdz")?;
                pre_built.insert(Framework::MpSpdz, mp_spdz_path);
            }
            Ok(pre_built)
        }
        (false, true) => Ok(HashMap::new()),
        (false, false) | (true, true) => {
            let mut build_paths = HashMap::new();
            let used_frameworks = config.used_frameworks();

            for framework in &used_frameworks {
                info!("Building {framework:?}");
                let build_path = match framework {
                    Framework::Aby => {
                        let aby_targets = config.targets_for(Framework::Aby);
                        build_aby(&args, &aby_targets).context("Unable to build Aby")?
                    }
                    Framework::Motion => {
                        let motion_targets = config.targets_for(Framework::Motion);
                        build_motion(&args, &motion_targets).context("Unable to build Motion")?
                    }
                    Framework::MpSpdz if args.remote_conf.is_none() => {
                        let mp_spdz_targets = config.targets_for(Framework::MpSpdz);
                        build_mp_spdz(&args, &mp_spdz_targets).context("Unable to build MpSpdz")?
                    }
                    Framework::MpSpdz => {
                        // remote_conf is Some and mp-spdz needs to be built on remote always
                        continue;
                    }
                    Framework::Seec => {
                        let seec_targets = config.targets_for(Framework::Seec);
                        build_seec(&args, &seec_targets).context("Unable to build Seec")?
                    }
                };
                build_paths.insert(framework.clone(), build_path);
            }
            Ok(build_paths)
        }
    }
}

fn build_aby(args: &ProgArgs, targets: &[BenchTarget]) -> Result<PathBuf> {
    let aby_path = args.project_root.join("benchmark/ABY");
    aby::build(&aby_path, targets, args.opt, args.clean_build)
}

fn build_motion(args: &ProgArgs, targets: &[BenchTarget]) -> Result<PathBuf> {
    let motion_path = args.project_root.join("benchmark/MOTION");
    motion::build(&motion_path, targets, args.opt, args.clean_build)
}

fn build_mp_spdz(args: &ProgArgs, targets: &[BenchTarget]) -> Result<PathBuf> {
    let mp_spdz_path = args.project_root.join("benchmark/MP-SPDZ");
    // MP-SPDZ needs to always be clean build on remote
    let clean_build = if args.is_remote {
        true
    } else {
        args.clean_build
    };
    let build_path = mp_spdz::build(&mp_spdz_path, clean_build)?;
    mp_spdz::compile(&mp_spdz_path, targets).context("Compiling mp-spdz programs")?;
    Ok(build_path)
}

fn build_seec(args: &ProgArgs, targets: &[BenchTarget]) -> Result<PathBuf> {
    let seec_path = args.project_root.join("crates/seec");
    seec::build(&seec_path, targets, args.opt, args.clean_build)
}

fn execute_remotely(id: usize, remote_conf: &RemoteConfig, args: &ProgArgs) -> Result<()> {
    let mut remote_args = args.clone();

    remote_args.is_remote = true;
    remote_args.id = Some(id);
    remote_args.client = Some(remote_conf.client);
    remote_args.server = Some(remote_conf.server);
    remote_args.conf = remote_conf
        .remote_path
        .join(make_path_rel(&args.conf).context("failed to get remote conf path")?);
    remote_args.remote_conf = Some(
        remote_conf
            .remote_path
            .join(make_path_rel(args.remote_conf.as_ref().unwrap())?),
    );
    remote_args.project_root = remote_conf.remote_path.clone();
    remote_args.heaptrack_dir = remote_conf.heaptrack_dir.clone();
    let timestamp = Local::now().to_rfc3339();
    if remote_args.out.is_none() {
        remote_args.out = Some(PathBuf::from(format!("mpc-bench-{}.jsonl", timestamp)));
    }

    // force clean build if building on remote
    if args.build_on_remote {
        remote_args.clean_build = true;
    }

    let remote = &remote_conf.remote_hosts[id];

    let log_name = format!("mpc-bench-{}.log", timestamp);

    let remote_cmd = format!(
        "cd {};\
        RUST_LOG=info nohup ./benchmark/bencher/target/debug/bencher \
        {remote_args} > {} 2>&1",
        &remote_conf.remote_path.display(),
        log_name
    );
    Command::new("ssh")
        .arg("-J")
        .arg(&remote_conf.jump)
        .arg(remote)
        .arg(remote_cmd)
        .run()
        .context("failed to execute bencher remotely")
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
}

fn write_build_paths(paths: &HashMap<Framework, PathBuf>) -> Result<()> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let paths: HashMap<_, _> = paths
        .iter()
        .map(|(framework, path)| {
            let stripped: PathBuf = make_path_rel(path)?;
            Ok((framework.clone(), Path::new("./").join(stripped)))
        })
        .collect::<Result<_, anyhow::Error>>()?;
    let ser = toml::to_string(&paths).context("failed to serialize build paths")?;
    fs::write(manifest_dir.join("target/build-paths.toml"), &ser)
        .context("unable to create build-paths.toml")?;
    Ok(())
}

fn make_path_rel(path: &Path) -> Result<PathBuf> {
    Ok(path
        .canonicalize()?
        .components()
        .skip_while(|c| c.as_os_str() != OsStr::new("gmw-rs"))
        .skip(1)
        .collect())
}

impl Display for ProgArgs {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        fn fmt_opt<T: Display>(opt: &str, val: &Option<T>) -> String {
            val.as_ref()
                .map(|val| format!("{opt} {val}"))
                .unwrap_or_default()
        }

        fn fmt_flag(display: &str, val: bool) -> &str {
            if val {
                display
            } else {
                ""
            }
        }
        fn fmt_multi(opt: &str, vals: impl IntoIterator<Item = impl Display>) -> String {
            vals.into_iter().map(|val| format!("{opt} {val}")).join(" ")
        }

        write!(
            f,
            "--conf {conf} \
            --opt {opt} \
            --heaptrack-dir {heaptrack_dir} \
            --project-root {project_root} \
            {id} \
            {server} \
            {client} \
            {out} \
            {remote_conf} \
            {is_remote} \
            {build_on_remote} \
            {clean_build} \
            {quick_run} \
            {tags}",
            conf = self.conf.display(),
            opt = self.opt,
            heaptrack_dir = self.heaptrack_dir.display(),
            project_root = self.project_root.display(),
            id = fmt_opt("--id", &self.id),
            server = fmt_opt("--server", &self.server),
            client = fmt_opt("--client", &self.client),
            out = fmt_opt("--out", &self.out.as_deref().map(Path::display)),
            remote_conf = fmt_opt(
                "--remote-conf",
                &self.remote_conf.as_deref().map(Path::display)
            ),
            is_remote = fmt_flag("--is-remote", self.is_remote),
            build_on_remote = fmt_flag("--build-on-remote", self.build_on_remote),
            clean_build = fmt_flag("--clean-build", self.clean_build),
            quick_run = fmt_flag("--quick-run", self.quick_run),
            tags = fmt_multi("--tags", &self.tags),
        )
    }
}
