//! Where are we, what accelerator is here, and is its SDK installed?
//!
//! Requirement (4). Detection is runtime probing — vendor CLIs and, later, `dlopen` of
//! vendor libraries — never a build-time link. That distinction is what makes
//! requirement (3) achievable: no C compiles in our build graph, and a vendor runtime
//! is consulted if and only if it is present.
//!
//! **The probing and the parsing are separate on purpose.** `detect()` runs commands;
//! the `parse_*` functions are pure `&str -> Option<Accel>`. So the Linux and Windows
//! probes are testable on a Mac against canned fixtures, which is the only way
//! requirement (2) has an observable criterion before a CI matrix exists.

use std::fmt;
use std::process::Command;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as host;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as host;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod other;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
use other as host;

/// An accelerator family. `amd` is deliberately two families: a ROCm GPU is not a
/// Ryzen AI NPU, and there is no HIP/ROCm target in the registry — so defaulting a
/// Radeon to the `aie` form would emit IRON Python for a GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accel {
    pub family: &'static str,
    pub device: String,
    /// How this was detected, so a user can check the tool's reasoning.
    pub via: &'static str,
}

/// An SDK/compiler for a family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sdk {
    pub family: &'static str,
    pub name: &'static str,
    pub version: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Platform {
    pub os: &'static str,
    pub arch: &'static str,
    pub triple: String,
    pub accels: Vec<Accel>,
    pub sdks: Vec<Sdk>,
}

impl Platform {
    /// The family whose form is the default output target.
    ///
    /// With no accelerator at all this is "none", which maps to the `linalg` CPU
    /// bridge — so the default never produces an artifact the user cannot run.
    pub fn primary_family(&self) -> &'static str {
        // Order of preference when a box has more than one: a real compute accelerator
        // beats an NPU beats nothing. An amd-gpu is deliberately NOT preferred, because
        // tile-rs has no target for it.
        for want in [
            "apple-gpu",
            "nvidia",
            "ascend",
            "trainium",
            "tpu",
            "amd-npu",
        ] {
            if self.accels.iter().any(|a| a.family == want) {
                return want;
            }
        }
        "none"
    }

    pub fn sdk_for(&self, family: &str) -> Option<&Sdk> {
        self.sdks.iter().find(|s| s.family == family)
    }

    /// A family that is present but whose target tile-rs does not have.
    pub fn unsupported_present(&self) -> Vec<&Accel> {
        self.accels
            .iter()
            .filter(|a| a.family == "amd-gpu")
            .collect()
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "platform: {}", self.triple)?;
        if self.accels.is_empty() {
            writeln!(f, "accelerator: none detected")?;
        }
        for a in &self.accels {
            let sdk = match self.sdk_for(a.family) {
                Some(s) => match &s.version {
                    Some(v) => format!("{} {v}", s.name),
                    None => format!("{} (version unknown)", s.name),
                },
                // "device present, SDK missing" is a different state from "absent", and
                // conflating them is how a user ends up debugging the wrong thing.
                None => "SDK missing".to_string(),
            };
            writeln!(
                f,
                "accelerator: {} [{}] via {} — {}",
                a.device, a.family, a.via, sdk
            )?;
        }
        Ok(())
    }
}

/// Run a command and capture stdout, or `None` if it is not installed / failed.
///
/// Never an error path: a missing vendor CLI is the normal case on most machines.
pub fn probe(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).into_owned();
    if s.trim().is_empty() {
        None
    } else {
        Some(s)
    }
}

/// The host triple as this binary sees it.
pub fn triple() -> String {
    let os = match std::env::consts::OS {
        "macos" => "apple-darwin",
        "linux" => "unknown-linux-gnu",
        "windows" => "pc-windows-msvc",
        other => other,
    };
    format!("{}-{}", std::env::consts::ARCH, os)
}

pub fn detect() -> Platform {
    detect_in(&crate::simenv::Env::detect())
}

/// Detect within an environment.
///
/// `env.vendor_clis == false` models a machine with no vendor tooling at all — which is
/// also exactly what a stripped `PATH` produces, so the simulated and the real version
/// of this situation take the same code path.
pub fn detect_in(env: &crate::simenv::Env) -> Platform {
    // `no-devices` must actually remove the devices.
    //
    // This only honoured `no-vendor-clis`, so `TILE_SIMULATE=no-devices` printed
    // "SIMULATED ENVIRONMENT — forced absent: no-devices" and then `doctor` reported the
    // M1 Ultra two lines below it. A simulation that announces an absence it did not
    // apply is worse than no simulation: every test written against it passes for the
    // wrong reason, and `-O4`'s "refuses without a device" was one of them.
    if env.forced.contains(&"no-devices") {
        return Platform {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            triple: triple(),
            accels: Vec::new(),
            // The SDKs stay: a machine can have CUDA installed and no GPU in it, and
            // that is exactly the state this flag is for.
            sdks: host::sdks(),
        };
    }
    if !env.vendor_clis {
        return Platform {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            triple: triple(),
            accels: Vec::new(),
            sdks: Vec::new(),
        };
    }
    let mut accels = host::accelerators();
    let mut sdks = host::sdks();
    // Vendor CLIs that exist on every OS are probed once, here, rather than three times.
    if let Some(out) = probe("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]) {
        if let Some(a) = parse_nvidia_smi(&out) {
            accels.push(a);
        }
    }
    if let Some(out) = probe("npu-smi", &["info"]) {
        if let Some(a) = parse_npu_smi(&out) {
            accels.push(a);
        }
    }
    if let Some(out) = probe("nvcc", &["--version"]) {
        if let Some(s) = parse_nvcc(&out) {
            sdks.push(s);
        }
    }
    Platform {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        triple: triple(),
        accels,
        sdks,
    }
}

// ── Pure parsers. Fixture-tested; this is what makes the Linux path checkable on a Mac.

pub fn parse_nvidia_smi(out: &str) -> Option<Accel> {
    let name = out.lines().map(str::trim).find(|l| !l.is_empty())?;
    Some(Accel {
        family: "nvidia",
        device: name.to_string(),
        via: "nvidia-smi",
    })
}

pub fn parse_npu_smi(out: &str) -> Option<Accel> {
    // `npu-smi info` prints a table; the chip name appears in a row like
    // "| 0     910B2               | OK  ...".
    for line in out.lines() {
        if let Some(idx) = line.find("910") {
            let name: String = line[idx..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
            if !name.is_empty() {
                return Some(Accel {
                    family: "ascend",
                    device: format!("Ascend {name}"),
                    via: "npu-smi",
                });
            }
        }
    }
    None
}

pub fn parse_nvcc(out: &str) -> Option<Sdk> {
    let v = out
        .lines()
        .find(|l| l.contains("release"))
        .and_then(|l| l.split("release").nth(1))
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string());
    Some(Sdk {
        family: "nvidia",
        name: "nvcc",
        version: v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triple_is_well_formed() {
        let t = triple();
        assert!(t.contains('-'), "triple {t} has no separator");
        assert!(t.starts_with(std::env::consts::ARCH));
    }

    #[test]
    fn nvidia_smi_names_the_device() {
        let a = parse_nvidia_smi("Tesla T4\n").expect("parsed");
        assert_eq!(a.family, "nvidia");
        assert_eq!(a.device, "Tesla T4");
    }

    #[test]
    fn npu_smi_finds_the_ascend_chip() {
        let out = "\
+------------------------------------------------------------+
| npu-smi 23.0.0                                             |
+-------+-----------------+---------------------------------+
| 0     910B2             | OK    | 43.5  58   0/0  0/32768 |
+-------+-----------------+---------------------------------+";
        let a = parse_npu_smi(out).expect("parsed");
        assert_eq!(a.family, "ascend");
        assert_eq!(a.device, "Ascend 910B2");
    }

    #[test]
    fn nvcc_version_is_extracted() {
        let out = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.4, V12.4.131\n";
        let s = parse_nvcc(out).expect("parsed");
        assert_eq!(s.version.as_deref(), Some("12.4"));
    }

    #[test]
    fn absent_vendor_tools_are_not_an_error() {
        assert!(probe("definitely-not-a-real-binary-xyz", &[]).is_none());
    }

    #[test]
    fn no_accelerator_means_family_none() {
        let p = Platform {
            os: "linux",
            arch: "x86_64",
            triple: "x86_64-unknown-linux-gnu".into(),
            accels: vec![],
            sdks: vec![],
        };
        assert_eq!(p.primary_family(), "none");
    }

    #[test]
    fn an_amd_gpu_is_never_the_primary_family() {
        // There is no HIP/ROCm target in the registry. Preferring it would emit IRON
        // Python (the Ryzen AI NPU form) for a Radeon.
        let p = Platform {
            os: "linux",
            arch: "x86_64",
            triple: "x86_64-unknown-linux-gnu".into(),
            accels: vec![Accel {
                family: "amd-gpu",
                device: "Radeon RX 7900".into(),
                via: "rocm-smi",
            }],
            sdks: vec![],
        };
        assert_eq!(p.primary_family(), "none");
        assert_eq!(p.unsupported_present().len(), 1);
    }

    #[test]
    fn an_amd_npu_is_a_different_family_from_an_amd_gpu() {
        let p = Platform {
            os: "linux",
            arch: "x86_64",
            triple: "x86_64-unknown-linux-gnu".into(),
            accels: vec![Accel {
                family: "amd-npu",
                device: "Ryzen AI".into(),
                via: "xrt-smi",
            }],
            sdks: vec![],
        };
        assert_eq!(p.primary_family(), "amd-npu");
        assert!(p.unsupported_present().is_empty());
    }
}
