//! What is present on this machine — and how to pretend that something isn't.
//!
//! Almost every interesting failure in this tool is an *absence*: no emitter compiled
//! in, no toolchain installed, no vendor SDK, no accelerator, no license, no network.
//! Those are exactly the paths that never run on a developer's machine, because a
//! developer's machine has everything. So they rot, and the first person to meet them is
//! a user on a bare box.
//!
//! [`Env`] makes absence a value. [`Env::detect`] reports the real machine;
//! [`Env::from_spec`] builds one from a description, so a test — or a CI job, or a user
//! reproducing a bug report — can run the whole tool as if a dependency were missing.
//!
//! ## The one rule
//!
//! **A simulated environment announces itself.** Every run under `TILE_SIMULATE` prints
//! a banner to stderr naming what was forced absent. A testing hook that can silently
//! change behaviour is a bug generator: someone leaves the variable set, gets a refusal
//! they cannot explain, and files it against the wrong component.
//!
//! ```text
//! TILE_SIMULATE=no-toolchains tile softmax.rs -o out.cu
//! TILE_SIMULATE=bare          tile doctor
//! ```

use std::fmt;

/// Build features that provide an MLIR->source emitter.
const EMITTER_FEATURES: &[&str] = &["emitters", "pico", "ascend"];

/// Everything the tool's behaviour depends on that might not be there.
#[derive(Clone, Debug)]
pub struct Env {
    /// Build features compiled into this binary.
    pub features: Vec<String>,
    /// Toolchains provisioned under `~/.tile-rs`.
    pub toolchains: Vec<String>,
    /// Accelerator families with a device present.
    pub devices: Vec<String>,
    /// May we shell out to vendor CLIs at all? False models a machine with none, and is
    /// also what a stripped `PATH` produces for real.
    pub vendor_clis: bool,
    pub network: bool,
    pub license: bool,
    /// Absences that were FORCED rather than observed. Non-empty means simulated.
    pub forced: Vec<&'static str>,
}

impl Default for Env {
    fn default() -> Self {
        Env {
            features: Vec::new(),
            toolchains: Vec::new(),
            devices: Vec::new(),
            vendor_clis: true,
            network: true,
            license: false,
            forced: Vec::new(),
        }
    }
}

/// The absences a spec can name. Kept as data so `--help` and the error message for a
/// bad spec both stay correct without being written twice.
pub const ABSENCES: &[(&str, &str)] = &[
    (
        "no-emitters",
        "no MLIR->source emitter is compiled into this build",
    ),
    ("no-toolchains", "no toolchain is provisioned"),
    ("no-devices", "no accelerator is present"),
    (
        "no-vendor-clis",
        "no vendor CLI is on PATH (nvidia-smi, npu-smi, xcrun...)",
    ),
    ("no-network", "nothing may be fetched"),
    ("no-license", "no knowledge-base license key"),
    (
        "bare",
        "all of the above — the emptiest machine that can still run the tool",
    ),
];

#[derive(Debug)]
pub struct BadSpec(pub String);

impl fmt::Display for BadSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "unknown TILE_SIMULATE term {:?}. Known terms:", self.0)?;
        for (name, what) in ABSENCES {
            writeln!(f, "  {name:<16} {what}")?;
        }
        Ok(())
    }
}

impl Env {
    /// The real machine.
    pub fn detect() -> Env {
        // Every build feature the route graph can ask for. A feature compiled in but
        // missing from this list makes the tool refuse a route it could actually take --
        // which is exactly what happened when `pico` was added and this was not.
        let mut features = Vec::new();
        for (name, on) in [
            ("emitters", cfg!(feature = "emitters")),
            ("pico", cfg!(feature = "pico")),
            ("stats", cfg!(feature = "stats")),
        ] {
            if on {
                features.push(name.to_string());
            }
        }
        Env {
            features,
            // M4 fills these in from ~/.tile-rs; until then nothing is provisioned, which
            // is itself the honest answer rather than a placeholder.
            toolchains: Vec::new(),
            devices: Vec::new(),
            vendor_clis: true,
            network: true,
            license: false,
            forced: Vec::new(),
        }
    }

    /// Apply a `TILE_SIMULATE` spec to a real environment.
    ///
    /// A spec can only ever REMOVE. Letting it add would let a test claim a capability
    /// the machine does not have, and the run would then fail somewhere less honest.
    pub fn simulate(mut self, spec: &str) -> Result<Env, BadSpec> {
        for term in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let all = term == "bare";
            if !all && !ABSENCES.iter().any(|(n, _)| *n == term) {
                return Err(BadSpec(term.to_string()));
            }
            if all || term == "no-emitters" {
                // Every feature that provides an emitter, not just the one named
                // `emitters`: `pico` is one too, and a "no-emitters" machine that still
                // had pico would be a machine nobody has.
                self.features
                    .retain(|f| !EMITTER_FEATURES.contains(&f.as_str()));
                self.forced.push("no-emitters");
            }
            if all || term == "no-toolchains" {
                self.toolchains.clear();
                self.forced.push("no-toolchains");
            }
            if all || term == "no-devices" {
                self.devices.clear();
                self.forced.push("no-devices");
            }
            if all || term == "no-vendor-clis" {
                self.vendor_clis = false;
                self.forced.push("no-vendor-clis");
            }
            if all || term == "no-network" {
                self.network = false;
                self.forced.push("no-network");
            }
            if all {
                // `bare` is the emptiest machine that can still run the tool, so it
                // clears every build feature rather than the emitting ones only.
                self.features.clear();
            }
            if all || term == "no-license" {
                self.license = false;
                self.forced.push("no-license");
            }
        }
        self.forced.sort_unstable();
        self.forced.dedup();
        Ok(self)
    }

    /// Detect, then apply `TILE_SIMULATE` if it is set.
    pub fn from_process() -> Result<Env, BadSpec> {
        let base = Env::detect();
        match std::env::var("TILE_SIMULATE") {
            Ok(spec) if !spec.trim().is_empty() => base.simulate(&spec),
            _ => Ok(base),
        }
    }

    pub fn is_simulated(&self) -> bool {
        !self.forced.is_empty()
    }

    /// The banner. A simulated run says so, every time, on stderr — never on stdout,
    /// which may be carrying a kernel.
    pub fn banner(&self) -> Option<String> {
        if !self.is_simulated() {
            return None;
        }
        Some(format!(
            "tile: SIMULATED ENVIRONMENT — forced absent: {}\n\
             tile: unset TILE_SIMULATE for this machine's real capabilities",
            self.forced.join(", ")
        ))
    }

    pub fn has_feature(&self, f: &str) -> bool {
        self.features.iter().any(|x| x == f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_build_feature_the_graph_can_ask_for_is_reported() {
        // A feature compiled in but absent from `detect` makes the tool refuse a route it
        // could take, and the refusal blames the build rather than the list.
        let e = Env::detect();
        assert_eq!(e.has_feature("emitters"), cfg!(feature = "emitters"));
        assert_eq!(e.has_feature("pico"), cfg!(feature = "pico"));
        assert_eq!(e.has_feature("stats"), cfg!(feature = "stats"));
    }

    #[test]
    fn a_plain_environment_is_not_simulated_and_says_nothing() {
        let e = Env::detect();
        assert!(!e.is_simulated());
        assert!(e.banner().is_none());
    }

    #[test]
    fn no_emitters_removes_every_feature_that_provides_one() {
        // A "no-emitters" machine that still had `pico` compiled in would be a machine
        // nobody has, and the scenario it is meant to model would not be modelled.
        let mut e = Env::detect();
        e.features = vec!["emitters".into(), "pico".into(), "stats".into()];
        let e = e.simulate("no-emitters").unwrap();
        assert_eq!(e.features, vec!["stats".to_string()], "an emitter survived");
    }

    #[test]
    fn bare_removes_everything_at_once() {
        let e = Env::detect().simulate("bare").unwrap();
        assert!(e.features.is_empty());
        assert!(e.devices.is_empty());
        assert!(!e.vendor_clis);
        assert!(!e.network);
        assert!(!e.license);
        assert_eq!(e.forced.len(), 6, "bare must name every absence it caused");
    }

    #[test]
    fn a_simulated_environment_always_announces_itself() {
        // The rule that keeps this from being a bug generator.
        let e = Env::detect().simulate("no-network").unwrap();
        let b = e.banner().expect("a banner");
        assert!(b.contains("SIMULATED"));
        assert!(b.contains("no-network"));
        assert!(b.contains("unset TILE_SIMULATE"));
    }

    #[test]
    fn a_spec_can_only_remove_never_add() {
        let mut base = Env::detect();
        base.devices.push("nvidia".into());
        let e = base.clone().simulate("no-devices").unwrap();
        assert!(e.devices.is_empty());
        // And nothing in the vocabulary grants anything.
        for (name, _) in ABSENCES {
            assert!(
                name.starts_with("no-") || *name == "bare",
                "{name} is not an absence"
            );
        }
    }

    #[test]
    fn an_unknown_term_is_refused_with_the_vocabulary_printed() {
        let e = Env::detect().simulate("no-gpus").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("no-gpus"));
        for (name, _) in ABSENCES {
            assert!(msg.contains(name), "the refusal must list {name}");
        }
    }

    #[test]
    fn terms_compose_and_are_reported_once_each() {
        let e = Env::detect()
            .simulate("no-network,no-network,no-devices")
            .unwrap();
        assert_eq!(e.forced, vec!["no-devices", "no-network"]);
    }

    #[test]
    fn an_empty_spec_is_not_a_simulation() {
        let e = Env::detect().simulate("  ").unwrap();
        assert!(!e.is_simulated());
    }
}
