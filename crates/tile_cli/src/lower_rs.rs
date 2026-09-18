//! `.rs` in, MLIR and target source out — through the real codegen backend.
//!
//! This is the one edge in the graph that needs a toolchain, and it is the front door of
//! the whole tool: `tile softmax.rs -o softmax.metal` is what a user reaches for first.
//!
//! ## Three things that fail silently, and are refused here instead
//!
//! Each of these cost a debugging session, and each fails in a way that points at the
//! wrong component:
//!
//! * **`RUSTFLAGS` set.** Cargo replaces `build.rustflags` wholesale rather than merging,
//!   so the backend never loads and the kernel builds with the stock LLVM backend —
//!   quietly, with no error and no emitted source. The release notes warn about it; this
//!   refuses before the build rather than after.
//! * **`--target` omitted.** Without it the backend flag reaches build scripts and
//!   proc-macro dependencies, which then fail with `found invalid metadata files for
//!   crate core` — a message about `core` for a mistake about scope. Passed always.
//! * **`TILERS_CODEGEN_PATH` given a form id.** An unrecognised value is not rejected; it
//!   falls through to the Ascend default and fails with `Could not determine
//!   ASCEND_HOME_PATH`. `forms::codegen_path` is the translation.

use crate::forms::{self, Form};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum LowerError {
    /// The backend has not been provisioned.
    NoBackend {
        looked: String,
    },
    /// `tile_std` is not on disk, so a kernel crate cannot depend on it.
    NoTileStd {
        looked: Vec<String>,
    },
    /// `RUSTFLAGS` is set and would silently disable the backend.
    RustflagsSet {
        value: String,
    },
    /// This form is not a codegen target.
    NotATarget {
        form: &'static str,
    },
    /// The pinned toolchain is not installed.
    NoToolchain {
        want: String,
    },
    /// The build ran and the compiler rejected the kernel. Its own words.
    Rejected {
        diagnostics: String,
    },
    /// The build succeeded but produced nothing, which means the backend did not run.
    NoOutput {
        hint: String,
    },
    Io(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::NoBackend { looked } => write!(
                f,
                "the codegen backend is not installed (looked in {looked}).\n  \
                 Run `tile install rustc_codegen_tile` — it is ~97 MB and verified \
                 against a pinned sha256."
            ),
            LowerError::NoTileStd { looked } => write!(
                f,
                "tile_std is not on disk, and a kernel crate must depend on it. Looked \
                 in:\n  {}\n  Point TILE_STD_PATH at a checkout of the tile-rs \
                 repository.",
                looked.join("\n  ")
            ),
            LowerError::RustflagsSet { value } => write!(
                f,
                "RUSTFLAGS is set ({value:?}), and cargo replaces build.rustflags \
                 wholesale rather than merging.\n  The codegen backend would never load \
                 and your kernel would build with the stock backend — quietly, with no \
                 error and no emitted source.\n  Re-run with RUSTFLAGS unset: \
                 `env -u RUSTFLAGS tile ...`"
            ),
            LowerError::NotATarget { form } => {
                write!(
                    f,
                    "\"{form}\" is not a codegen target; there is nothing to emit it as"
                )
            }
            LowerError::NoToolchain { want } => write!(
                f,
                "the pinned toolchain {want} is not installed. The backend dlopens rustc \
                 internals and does not load into another.\n  \
                 rustup toolchain install {want} --component rustc-dev --component \
                 llvm-tools --component rust-src"
            ),
            LowerError::Rejected { diagnostics } => {
                write!(f, "the kernel did not compile:\n{diagnostics}")
            }
            LowerError::NoOutput { hint } => write!(
                f,
                "the build succeeded but emitted nothing, which means the backend did not \
                 run. {hint}"
            ),
            LowerError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// What a lowering produced.
#[derive(Debug)]
pub struct Lowered {
    pub mlir: String,
    /// The target source, when the backend emitted one.
    pub target_source: Option<String>,
}

pub const PINNED_TOOLCHAIN: &str = "nightly-2025-08-04";

/// The provisioned backend's shared library.
pub fn backend_lib() -> Result<PathBuf, LowerError> {
    let root = crate::provision::home().join("toolchains/rustc_codegen_tile");
    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Err(LowerError::NoBackend {
            looked: root.display().to_string(),
        });
    };
    for e in rd.flatten() {
        let lib = e.path().join(format!("lib/librustc_codegen_tile.{ext}"));
        if lib.exists() {
            return Ok(lib);
        }
    }
    Err(LowerError::NoBackend {
        looked: root.display().to_string(),
    })
}

/// A `tile_std` checkout for the kernel crate to depend on.
pub fn tile_std_path() -> Result<PathBuf, LowerError> {
    let mut looked = Vec::new();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("TILE_STD_PATH") {
        candidates.push(PathBuf::from(p));
    }
    // Beside this crate, when running from the repository.
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tile_std"));
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join("tile-rs/crates/tile_std"));
    }
    for c in &candidates {
        looked.push(c.display().to_string());
        if c.join("Cargo.toml").exists() {
            return Ok(c.clone());
        }
    }
    Err(LowerError::NoTileStd { looked })
}

fn host_triple() -> String {
    crate::platform::triple()
}

/// Lower `source` to MLIR, and to `target`'s source when one is asked for.
///
/// Assembles a scratch crate around the kernel, builds it with the pinned toolchain and
/// the provisioned backend, and reads back what the backend wrote. The kernel source is
/// used verbatim: if it is missing `#![no_core]` or an attribute, the compiler says so
/// and its words are passed through rather than paraphrased.
pub fn lower(
    source: &str,
    target: &Form,
    report: &mut dyn FnMut(&str),
) -> Result<Lowered, LowerError> {
    // Refuse before spending a minute on a build that cannot work.
    if let Some(v) = std::env::var_os("RUSTFLAGS") {
        let v = v.to_string_lossy().to_string();
        if !v.trim().is_empty() {
            return Err(LowerError::RustflagsSet { value: v });
        }
    }
    let Some(codegen_path) = forms::codegen_path(target.id) else {
        return Err(LowerError::NotATarget { form: target.id });
    };
    let lib = backend_lib()?;
    let tile_std = tile_std_path()?;

    let work = std::env::temp_dir().join(format!("tile-lower-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(work.join("src")).map_err(|e| LowerError::Io(e.to_string()))?;
    std::fs::create_dir_all(work.join(".cargo")).map_err(|e| LowerError::Io(e.to_string()))?;

    // crate-type must include cdylib: the backend emits from the linked artifact.
    std::fs::write(
        work.join("Cargo.toml"),
        format!(
            "[package]\nname = \"tile_kernel\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
             [lib]\ncrate-type = [\"cdylib\", \"lib\"]\ntest = false\n\
             [dependencies]\ntile_std = {{ path = {:?} }}\n",
            tile_std.display().to_string()
        ),
    )
    .map_err(|e| LowerError::Io(e.to_string()))?;
    std::fs::write(work.join("src/lib.rs"), source).map_err(|e| LowerError::Io(e.to_string()))?;
    std::fs::write(
        work.join(".cargo/config.toml"),
        format!(
            "[build]\nrustflags = [\n  \"-Zcodegen-backend={}\",\n  \
             \"-Zcrate-attr=feature(register_tool)\",\n  \
             \"-Zcrate-attr=register_tool(tile)\",\n  \"-Cpanic=abort\",\n  \
             \"-Clto=off\",\n]\n[unstable]\n",
            lib.display()
        ),
    )
    .map_err(|e| LowerError::Io(e.to_string()))?;

    let triple = host_triple();
    report(&format!(
        "building the kernel for {codegen_path} with {PINNED_TOOLCHAIN}"
    ));

    let out = std::process::Command::new("cargo")
        .arg(format!("+{PINNED_TOOLCHAIN}"))
        .args(["build", "--target", &triple])
        .current_dir(&work)
        // --target is not optional. Without it the backend flag reaches build scripts and
        // proc-macro dependencies, which fail with "found invalid metadata files for
        // crate core" -- a message about `core` for a mistake about scope.
        .env_remove("RUSTFLAGS")
        .env("TILERS_CODEGEN_SO", &lib)
        .env("TILERS_CODEGEN_PATH", codegen_path)
        .output()
        .map_err(|e| LowerError::Io(format!("cargo is not available: {e}")))?;

    let diagnostics = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        if diagnostics.contains("is not installed") || diagnostics.contains("no such command") {
            return Err(LowerError::NoToolchain {
                want: PINNED_TOOLCHAIN.into(),
            });
        }
        return Err(LowerError::Rejected { diagnostics });
    }

    // The backend writes beside the linked artifact.
    let deps = work.join("target").join(&triple).join("debug/deps");
    let mut mlir = None;
    let mut target_source = None;
    if let Ok(rd) = std::fs::read_dir(&deps) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".mlir") && mlir.is_none() {
                mlir = std::fs::read_to_string(e.path()).ok();
            } else if name.contains(".tile.") && target_source.is_none() {
                target_source = std::fs::read_to_string(e.path()).ok();
            }
        }
    }
    let result = match mlir {
        Some(mlir) => Ok(Lowered {
            mlir,
            target_source,
        }),
        None => Err(LowerError::NoOutput {
            hint: format!(
                "nothing matching *.mlir under {}. The usual cause is RUSTFLAGS being set \
                 in a parent process.",
                deps.display()
            ),
        }),
    };
    let _ = std::fs::remove_dir_all(&work);
    result
}

/// Is everything needed for a `.rs` lowering present?
pub fn available() -> bool {
    backend_lib().is_ok() && tile_std_path().is_ok()
}

pub fn toolchain_installed() -> bool {
    std::process::Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(PINNED_TOOLCHAIN))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RUSTFLAGS and TILE_HOME are process-global, and cargo's runner is parallel. A test
    /// that sets one is briefly changing the world for every other test in this binary.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_set_rustflags_is_refused_before_the_build_rather_than_after() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Cargo replaces build.rustflags wholesale, so the backend never loads and the
        // kernel builds with the stock backend -- quietly, with no error and no output.
        // Discovering that after a two-minute build is the failure this prevents.
        let msl = forms::by_id("msl").unwrap();
        let prev = std::env::var_os("RUSTFLAGS");
        std::env::set_var("RUSTFLAGS", "-C target-cpu=native");
        let e = lower("", msl, &mut |_| {}).unwrap_err();
        match prev {
            Some(v) => std::env::set_var("RUSTFLAGS", v),
            None => std::env::remove_var("RUSTFLAGS"),
        }
        assert!(matches!(e, LowerError::RustflagsSet { .. }), "{e}");
        assert!(e.to_string().contains("env -u RUSTFLAGS"), "{e}");
        assert!(
            e.to_string().contains("quietly"),
            "the silence is the point"
        );
    }

    #[test]
    fn a_form_that_is_not_a_codegen_target_is_refused_by_name() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mlir = forms::by_id("mlir").unwrap();
        let e = lower("", mlir, &mut |_| {}).unwrap_err();
        assert!(matches!(e, LowerError::NotATarget { form: "mlir" }), "{e}");
    }

    #[test]
    fn an_absent_backend_names_the_command_that_installs_it() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("TILE_HOME");
        let empty = std::env::temp_dir().join(format!("tile-nobk-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        std::env::set_var("TILE_HOME", &empty);
        let e = backend_lib().unwrap_err();
        match prev {
            Some(v) => std::env::set_var("TILE_HOME", v),
            None => std::env::remove_var("TILE_HOME"),
        }
        assert!(
            e.to_string().contains("tile install rustc_codegen_tile"),
            "{e}"
        );
        // And says what it costs, because 97 MB is worth warning about.
        assert!(e.to_string().contains("MB"), "{e}");
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn tile_std_is_found_beside_this_crate_when_running_from_the_repository() {
        let p = tile_std_path().expect("tile_std should be a sibling");
        assert!(p.join("Cargo.toml").exists());
    }

    #[test]
    fn the_pinned_toolchain_is_named_in_the_refusal_with_its_components() {
        let e = LowerError::NoToolchain {
            want: PINNED_TOOLCHAIN.into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("rustc-dev"), "{msg}");
        assert!(msg.contains("rust-src"), "{msg}");
        // And says WHY it has to be that one.
        assert!(msg.contains("dlopens rustc internals"), "{msg}");
    }
}
