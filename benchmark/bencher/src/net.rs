use crate::config::RemoteConfig;
use crate::CommandExt;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::process::{Command, Output};
use tracing::info;

#[derive(Debug, Copy, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum NetSetting {
    Reset,
    Lan,
    Wan,
}

impl NetSetting {
    pub fn reset(remote_conf: &RemoteConfig) -> Result<()> {
        let cmd = format!("eval \"{}\"", remote_conf.reset_net_cmd);
        let conf = |host| {
            info!("Resetting net setting for {host}");
            let output = Command::new("ssh")
                .arg("-J")
                .arg(&remote_conf.jump)
                .arg(host)
                .arg(&cmd)
                .output()
                .context(anyhow!("unable to reset net setting for {}", host))?;
            handle_tc_output(host, output)
        };
        conf(&remote_conf.remote_hosts[0])?;
        conf(&remote_conf.remote_hosts[1])?;
        Ok(())
    }

    pub fn configure_remote(&self, remote_conf: &RemoteConfig) -> Result<()> {
        let cmd = match self {
            NetSetting::Lan => format!("eval \"{}\"", remote_conf.lan_cmd),
            NetSetting::Wan => format!("eval \"{}\"", remote_conf.wan_cmd),
            NetSetting::Reset => format!("eval \"{}\"", remote_conf.reset_net_cmd),
        };
        let conf = |host| {
            info!("Configuring net setting for {host} to {self:?}");
            Command::new("ssh")
                .arg("-J")
                .arg(&remote_conf.jump)
                .arg(host)
                .arg(&cmd)
                .run()
                .context(anyhow!("unable to configure net setting for {}", host))
        };
        conf(&remote_conf.remote_hosts[0])?;
        conf(&remote_conf.remote_hosts[1])?;
        Ok(())
    }

    pub fn configure(&self, remote_conf: &RemoteConfig) -> Result<()> {
        info!("Configuring local net setting to {self:?}");
        let mut cmd = Command::new("bash");
        cmd.arg("-c");
        match self {
            NetSetting::Lan => cmd.arg(&remote_conf.lan_cmd),
            NetSetting::Wan => cmd.arg(&remote_conf.wan_cmd),
            NetSetting::Reset => cmd.arg(&remote_conf.reset_net_cmd),
        };
        let output = cmd.output()?;
        handle_tc_output("localhost", output)
    }
}

fn handle_tc_output(host: &str, output: Output) -> Result<()> {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let expected_err = stderr.contains("Cannot delete qdisc with handle of zero");
    if !(output.status.success() || expected_err) {
        Err(anyhow!(
            "unable to reset net setting for {}. Command failed unexpectedly. Output: {}",
            host,
            stderr
        ))
    } else {
        Ok(())
    }
}
