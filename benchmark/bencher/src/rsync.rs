use crate::config::RemoteConfig;
use crate::net::NetSetting;
use crate::CommandExt;
use anyhow::{anyhow, Context, Result};
use std::panic::resume_unwind;
use std::path::Path;
use std::process::Command;
use std::thread;

pub fn do_rsync(remote_conf: &RemoteConfig) -> Result<()> {
    NetSetting::reset(remote_conf)?;
    let sync = |host| {
        Command::new("rsync")
            .arg("-az")
            .arg("-e")
            .arg(format!("ssh -J {}", &remote_conf.jump))
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .arg(format!("{host}:{}", remote_conf.remote_path.display()))
            .run()
            .context(anyhow!("unable to rsync to {host}"))
    };

    thread::scope(|s| {
        let h0 = s.spawn(|| sync(&remote_conf.remote_hosts[0]).context("unable to sync to host 0"));
        let h1 = s.spawn(|| sync(&remote_conf.remote_hosts[1]).context("unable to sync to host 1"));
        h0.join().map_err(|err| resume_unwind(err)).unwrap()?;
        h1.join().map_err(|err| resume_unwind(err)).unwrap()?;
        Ok(())
    })
}
