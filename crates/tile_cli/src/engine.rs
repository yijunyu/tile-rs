//! `-m`: the inference engine that consumes the kernels this tool produces.
//!
//! Requirement (8). Each accelerator family has its own `ds4-rs-*` engine, and the
//! mapping is the part worth getting exactly right: an engine built for one backend does
//! not run another's kernels, so picking the wrong one fails at load time with an error
//! about the model rather than about the machine.
//!
//! ## What is here and what is not
//!
//! Locating an engine, mapping a family to one, and the rules around `-m` are here and
//! tested. **Cloning, building and supervising one is not**: those need the sibling
//! repositories and a machine with the accelerator, and writing them blind would produce
//! a supervisor whose failure modes nobody has seen. The refusal names exactly what is
//! missing, which is worth more than a launcher that has never launched anything.

use std::fmt;
use std::path::{Path, PathBuf};

/// The engine for an accelerator family.
///
/// `amd-gpu` maps to `ds4-rs-amd` even though tile-rs has no ROCm *codegen* target: the
/// engine and the codegen backend are different questions, and a machine can perfectly
/// well serve a model on a Radeon while tile-rs has nothing to lower to it.
pub fn engine_for(family: &str) -> Option<&'static str> {
    Some(match family {
        "apple-gpu" => "ds4-rs-metal",
        "nvidia" => "ds4-rs-cuda",
        "amd-gpu" | "amd-npu" => "ds4-rs-amd",
        "ascend" => "ds4-rs-ascend",
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    /// No engine exists for this family.
    NoEngine { family: String },
    /// The engine is named but not on this machine.
    NotPresent {
        engine: &'static str,
        looked: Vec<String>,
    },
    /// The engine checkout is there but its server would not build.
    BuildFailed { engine: &'static str, why: String },
    /// No weights for this model, and none may be fetched without a pinned entry.
    NoWeights { model: String, looked: Vec<String> },
    /// The server binary started and then exited.
    Exited { engine: &'static str, code: String },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::NoEngine { family } => write!(
                f,
                "no ds4-rs engine is mapped to the \"{family}\" family, so there is nothing \
                 to provision beside the daemon"
            ),
            EngineError::NotPresent { engine, looked } => write!(
                f,
                "{engine} is not on this machine. Looked in:\n  {}\n  \
                 Clone it there, or point TILE_ENGINE_PATH at it. Nothing is fetched for \
                 you: an engine checkout is a repository with its own build, and \
                 acquiring one silently is not something this tool will do.",
                looked.join("\n  ")
            ),
            EngineError::BuildFailed { engine, why } => write!(
                f,
                "{engine} is present but its server did not build:\n{why}\n  \
                 The daemon keeps serving the kernel tools; nothing about your kernels \
                 depends on this engine."
            ),
            EngineError::NoWeights { model, looked } => write!(
                f,
                "no weights for \"{model}\". Looked in:\n  {}\n  \
                 Weights are NOT fetched for you: a multi-gigabyte download needs a \
                 pinned entry with a digest, the same rule every toolchain here obeys, \
                 and there is no such entry for this model. Put a .gguf at one of those \
                 paths, or pass -m with a path to one.",
                looked.join("\n  ")
            ),
            EngineError::Exited { engine, code } => write!(
                f,
                "{engine} started and then exited ({code}). The daemon keeps serving the \
                 kernel tools."
            ),
        }
    }
}

/// Where an engine checkout might be. `TILE_ENGINE_PATH` wins, then the home directory,
/// then a sibling of the current directory — the three places one actually ends up.
pub fn search_paths(engine: &str) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("TILE_ENGINE_PATH") {
        v.push(PathBuf::from(p).join(engine));
    }
    if let Some(home) = std::env::var_os("HOME") {
        v.push(PathBuf::from(home).join(engine));
    }
    v.push(PathBuf::from("..").join(engine));
    v
}

/// A checkout looks like one when it has a manifest. Anything less is a directory that
/// happens to share a name.
pub fn looks_like_checkout(p: &Path) -> bool {
    p.join("Cargo.toml").exists()
}

pub fn locate(engine: &str) -> Result<PathBuf, EngineError> {
    let paths = search_paths(engine);
    for p in &paths {
        if looks_like_checkout(p) {
            return Ok(p.clone());
        }
    }
    Err(EngineError::NotPresent {
        engine: Box::leak(engine.to_string().into_boxed_str()),
        looked: paths.iter().map(|p| p.display().to_string()).collect(),
    })
}

/// What `-m <model>` on this machine would do.
pub fn plan(family: &str, _model: &str) -> Result<PathBuf, EngineError> {
    let Some(engine) = engine_for(family) else {
        return Err(EngineError::NoEngine {
            family: family.to_string(),
        });
    };
    locate(engine)
}

/// Where weights for `model` might already be. Nothing is downloaded.
pub fn weight_paths(model: &str) -> Vec<PathBuf> {
    // A path the user gave outright wins: `-m ./qwen3-8b.gguf` is unambiguous.
    let mut v = Vec::new();
    let direct = PathBuf::from(model);
    if direct.extension().is_some_and(|e| e == "gguf") {
        v.push(direct);
    }
    let home = crate::provision::home();
    v.push(home.join("models").join(format!("{model}.gguf")));
    if let Some(h) = std::env::var_os("HOME") {
        v.push(
            PathBuf::from(&h)
                .join("models")
                .join(format!("{model}.gguf")),
        );
    }
    v
}

/// Build the engine's server binary, reporting before it starts.
///
/// Announced first, because a build that takes minutes and says nothing is
/// indistinguishable from a hang -- the same rule the toolchain fetcher follows.
/// The cargo feature that selects this family's backend in the ds4 workspace.
///
/// The engines are one codebase with a backend chosen at build time -- `ds4_metal` and
/// `ds4_cuda` are optional path dependencies, and the OTHER one's directory need not
/// exist. Building without naming the feature makes cargo resolve the whole workspace and
/// fail on a manifest that was never meant to be there: "failed to load manifest for
/// dependency `ds4_cuda`" on a Mac. The repo's own README says
/// `-p ds4_server --features metal`, and that is what this passes.
pub fn backend_feature(family: &str) -> &'static str {
    match family {
        "nvidia" => "cuda",
        _ => "metal",
    }
}

pub fn build(
    engine: &'static str,
    at: &Path,
    feature: &str,
    report: &mut dyn FnMut(&str),
) -> Result<PathBuf, EngineError> {
    let bin = at.join("target/release/ds4-server");
    if bin.exists() {
        report(&format!(
            "{engine}: server already built at {}",
            bin.display()
        ));
        return Ok(bin);
    }
    report(&format!(
        "{engine}: building ds4-server in {} with --features {feature} (release; a few \
         minutes the first time)",
        at.display()
    ));
    let out = std::process::Command::new("cargo")
        .args([
            "build",
            "--release",
            "-p",
            "ds4_server",
            "--features",
            feature,
        ])
        .current_dir(at)
        .output()
        .map_err(|e| EngineError::BuildFailed {
            engine,
            why: format!("could not run cargo: {e}"),
        })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        // Translate the one cargo error that is not about the code at all.
        //
        // An optional path dependency still needs its manifest to EXIST for cargo to
        // resolve the workspace, so a checkout missing the other backend's crate fails
        // before compiling a line -- with a message about `ds4_cuda` that reads like a
        // build error on a machine that has no CUDA and wants none. Naming it as an
        // incomplete checkout is the difference between an actionable report and a wild
        // goose chase.
        if let Some(dep) = err
            .lines()
            .find_map(|l| {
                l.trim()
                    .strip_prefix("failed to load manifest for dependency `")
            })
            .and_then(|r| r.split('`').next())
        {
            return Err(EngineError::BuildFailed {
                engine,
                why: format!(
                    "    the checkout is incomplete: crates/{dep} is missing, and cargo \n\
                     \x20   needs an OPTIONAL path dependency's manifest to exist before it \n\
                     \x20   can resolve the workspace — so this fails before compiling \n\
                     \x20   anything, and not because of the backend you asked for.\n\
                     \x20   Fix it in {}, not here.",
                    at.display()
                ),
            });
        }
        // Otherwise the last few lines only: a full cargo log buries the reason.
        let tail: Vec<&str> = err.lines().rev().take(8).collect();
        return Err(EngineError::BuildFailed {
            engine,
            why: tail
                .into_iter()
                .rev()
                .map(|l| format!("    {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
        });
    }
    if !bin.exists() {
        return Err(EngineError::BuildFailed {
            engine,
            why: format!("    cargo succeeded but {} is not there", bin.display()),
        });
    }
    report(&format!("{engine}: built {}", bin.display()));
    Ok(bin)
}

/// A size a person can read. Weights run from megabytes to tens of gigabytes, and "0.0
/// GB" for a real file is the kind of report that makes someone doubt the whole line.
pub fn human_bytes(n: u64) -> String {
    // Bytes are whole things; a fractional one reads as a rounding artefact.
    const UNITS: [(&str, f64); 3] = [("GB", 1e9), ("MB", 1e6), ("kB", 1e3)];
    for (name, div) in UNITS {
        if n as f64 >= div {
            return format!("{:.1} {name}", n as f64 / div);
        }
    }
    format!("{n} B")
}

/// A running engine, supervised by this process.
#[derive(Debug)]
pub struct Running {
    pub engine: &'static str,
    pub bin: PathBuf,
    pub weights: PathBuf,
    child: std::process::Child,
}

impl Running {
    /// Has it exited? `None` means still running.
    pub fn exited(&mut self) -> Option<String> {
        match self.child.try_wait() {
            Ok(Some(st)) => Some(
                st.code()
                    .map(|c| format!("exit {c}"))
                    .unwrap_or_else(|| "killed by a signal".into()),
            ),
            _ => None,
        }
    }

    /// Stop it. Called when the daemon goes down, so the engine does not outlive it.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Build the engine if needed, find the weights, and start it.
///
/// Every failure here is NON-fatal to the caller by construction: it returns an error and
/// the daemon goes on serving the kernel tools. Losing the kernel tools because a model
/// could not be provisioned would be the wrong trade -- they do not depend on it.
pub fn start(
    family: &str,
    model: &str,
    report: &mut dyn FnMut(&str),
) -> Result<Running, EngineError> {
    let Some(engine) = engine_for(family) else {
        return Err(EngineError::NoEngine {
            family: family.to_string(),
        });
    };
    let at = locate(engine)?;
    let bin = build(engine, &at, backend_feature(family), report)?;

    let candidates = weight_paths(model);
    let Some(weights) = candidates.iter().find(|p| p.exists()).cloned() else {
        return Err(EngineError::NoWeights {
            model: model.to_string(),
            looked: candidates.iter().map(|p| p.display().to_string()).collect(),
        });
    };
    let bytes = std::fs::metadata(&weights).map(|m| m.len()).unwrap_or(0);
    report(&format!(
        "{engine}: starting with {} ({})",
        weights.display(),
        human_bytes(bytes)
    ));
    let child = std::process::Command::new(&bin)
        .arg("-m")
        .arg(&weights)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| EngineError::BuildFailed {
            engine,
            why: format!("    could not start {}: {e}", bin.display()),
        })?;
    let mut r = Running {
        engine,
        bin,
        weights,
        child,
    };
    // Give it a moment and check it is still alive: a server that dies immediately is a
    // configuration error, and reporting "started" for it would be a lie the user only
    // discovers at their first request.
    std::thread::sleep(std::time::Duration::from_millis(400));
    if let Some(code) = r.exited() {
        return Err(EngineError::Exited { engine, code });
    }
    report(&format!("{engine}: serving alongside the daemon"));
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_accelerator_family_maps_to_its_own_engine() {
        // An engine built for one backend does not run another's kernels, so a wrong
        // mapping fails at load time with an error about the model rather than the
        // machine — which is the hardest kind to trace back to here.
        assert_eq!(engine_for("apple-gpu"), Some("ds4-rs-metal"));
        assert_eq!(engine_for("nvidia"), Some("ds4-rs-cuda"));
        assert_eq!(engine_for("ascend"), Some("ds4-rs-ascend"));
        assert_eq!(engine_for("amd-gpu"), Some("ds4-rs-amd"));
        assert_eq!(engine_for("amd-npu"), Some("ds4-rs-amd"));
    }

    #[test]
    fn no_two_families_share_an_engine_by_accident() {
        let pairs = [
            ("apple-gpu", "nvidia"),
            ("nvidia", "ascend"),
            ("ascend", "apple-gpu"),
        ];
        for (a, b) in pairs {
            assert_ne!(engine_for(a), engine_for(b), "{a} and {b} share an engine");
        }
    }

    #[test]
    fn a_family_with_no_engine_says_so_rather_than_guessing() {
        assert_eq!(engine_for("none"), None);
        assert_eq!(engine_for("tpu"), None);
        let e = plan("tpu", "qwen3").unwrap_err();
        assert!(e.to_string().contains("no ds4-rs engine"), "{e}");
    }

    #[test]
    fn the_amd_engine_exists_even_though_the_amd_gpu_codegen_target_does_not() {
        // The engine and the codegen backend are different questions: a machine can serve
        // a model on a Radeon while tile-rs has nothing to lower to it.
        assert_eq!(engine_for("amd-gpu"), Some("ds4-rs-amd"));
        assert_eq!(crate::default_form_for("amd-gpu").id, "linalg");
    }

    #[test]
    fn an_absent_engine_names_every_place_it_looked() {
        let e = locate("ds4-rs-nonesuch").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("Looked in"), "{msg}");
        assert!(msg.contains("ds4-rs-nonesuch"), "{msg}");
        // And says plainly that nothing is fetched: a repository with its own build is
        // not something to acquire silently.
        assert!(msg.contains("Nothing is fetched for you"), "{msg}");
    }

    #[test]
    fn a_directory_that_merely_shares_the_name_is_not_a_checkout() {
        let d = std::env::temp_dir().join(format!("tile-eng-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert!(
            !looks_like_checkout(&d),
            "an empty directory passed as a checkout"
        );
        std::fs::write(d.join("Cargo.toml"), "[package]\n").unwrap();
        assert!(looks_like_checkout(&d));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_present_engine_is_located_and_missing_weights_are_refused_not_invented() {
        // This used to assert that a present engine was REFUSED -- "a supervisor whose
        // failure modes nobody has watched is worse than none". The supervisor is written
        // now, so the honest half moved: the checkout is found, and the thing that stops
        // the launch is the absence of weights, which are never fetched.
        let root = std::env::temp_dir().join(format!("tile-engroot-{}", std::process::id()));
        let engine = root.join("ds4-rs-metal");
        std::fs::create_dir_all(engine.join("target/release")).unwrap();
        std::fs::write(engine.join("Cargo.toml"), "[package]\n").unwrap();
        // A stand-in binary so `build` takes its already-built path. Without it this test
        // spawned a real `cargo build` against a bare manifest -- seconds of nothing, in
        // the unit tests, on every run. A test suite slow enough to skip is one that does
        // not run.
        std::fs::write(
            engine.join("target/release/ds4-server"),
            "#!/bin/sh\nexit 0\n",
        )
        .unwrap();
        std::env::set_var("TILE_ENGINE_PATH", &root);

        assert_eq!(plan("apple-gpu", "qwen3").unwrap(), engine);

        // No .gguf anywhere it looks, so it stops -- and says why, naming the paths.
        //
        // Deliberately does NOT set TILE_HOME. Each module here keeps its own ENV_LOCK,
        // so a lock in this module does not exclude `provision`'s tests, and setting a
        // process-global TILE_HOME from here made
        // `a_corrupted_artifact_is_rejected_before_anything_is_unpacked` see a home that
        // already had the artifact and report AlreadyPresent. A model name nobody has
        // reaches the same NoWeights branch without touching shared state.
        let mut said = Vec::new();
        let e = start("apple-gpu", "a-model-nobody-has", &mut |m: &str| {
            said.push(m.to_string())
        })
        .unwrap_err();
        match &e {
            EngineError::NoWeights { looked, .. } => {
                assert!(!looked.is_empty(), "refused without saying where it looked")
            }
            // A checkout with a bare [package] has no server to build, so a build failure
            // here is equally honest -- what must never happen is a silent download.
            EngineError::BuildFailed { .. } => {}
            other => panic!("unexpected: {other}"),
        }
        assert!(
            !e.to_string().contains("download") || e.to_string().contains("NOT fetched"),
            "{e}"
        );

        std::env::remove_var("TILE_ENGINE_PATH");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_size_a_person_can_read() {
        // "0.0 GB" for a real file is the kind of line that makes someone doubt the rest
        // of the report.
        assert_eq!(human_bytes(2_000_000), "2.0 MB");
        assert_eq!(human_bytes(8_000_000_000), "8.0 GB");
        assert_eq!(human_bytes(512), "512 B");
    }
}
