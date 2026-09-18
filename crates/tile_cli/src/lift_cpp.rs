//! Lifting AscendC (`cpp`, `.cce`) back to the tile-rs DSL.
//!
//! This is the first lift the CLI can actually take. The route graph has carried a
//! `pto -> tile` edge since the beginning with nothing behind it — `how_to_get("lift")`
//! answers `Unwritten`, and the binary says so in as many words. That edge stays
//! unwritten; this one is a different claim, and it is justified individually the way
//! the comment above the lifts in `routes.rs` asks for.
//!
//! **Why a toolchain and not a compiled feature.** The lifter is 1.36 MB of source and
//! needs `syn`, while this crate's whole discipline is one dependency and a hand-parsed
//! argv. Compiling it in would spend that budget; vendoring it would put a second copy
//! of a 728 KB `lower.rs` in a second repository, and the two would drift — which is
//! precisely the class of defect the lifter's own history is made of (a table read from
//! the wrong copy, twice). So `ascendc-to-rs` is treated as what it is: an external
//! tool, found the way every vendor compiler is found, and named in the route graph as
//! `Need::Toolchain`.
//!
//! The consequence is honest in both directions. A machine without the binary gets exit
//! 4 and the command that would fix it, not a silent failure; and there is exactly one
//! implementation of the lift, so what `tile` produces is what the lifter produces.

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum LiftError {
    /// The binary is not on this machine.
    NotInstalled,
    /// It ran and refused the input.
    Rejected {
        code: i32,
        stderr: String,
    },
    /// It ran, said nothing, and wrote nothing.
    NoOutput,
    Io(String),
}

impl std::fmt::Display for LiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiftError::NotInstalled => write!(
                f,
                "ascendc-to-rs is not on this machine.\n  \
                 cargo install --path crates/ascendc_to_rs   (in the ascend-rs checkout)"
            ),
            LiftError::Rejected { code, stderr } => {
                write!(f, "ascendc-to-rs refused the input (exit {code})")?;
                if !stderr.trim().is_empty() {
                    write!(f, ":\n{}", stderr.trim())?;
                }
                Ok(())
            }
            LiftError::NoOutput => write!(
                f,
                "ascendc-to-rs produced no output. A kernel it cannot find is not an \
                 error to it; here it is, because the route promised a file."
            ),
            LiftError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Where the binary might be, in the order a person would look.
///
/// `PATH` first, because an explicitly installed tool should win over a checkout that
/// happens to be lying around. `TILE_ASCENDC_TO_RS` overrides everything, which is what
/// the tests use rather than installing into the user's cargo bin.
pub fn locate() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TILE_ASCENDC_TO_RS") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let c = dir.join("ascendc-to-rs");
            if c.is_file() {
                return Some(c);
            }
        }
    }
    // A cargo-installed binary, and then a sibling checkout's release build. Both are
    // "on disk IS the toolchain" in the sense `Caps::from_env` already uses for the
    // codegen backend.
    let home = std::env::var("HOME").ok()?;
    [PathBuf::from(&home).join(".cargo/bin/ascendc-to-rs")]
        .into_iter()
        .find(|c| c.is_file())
}

/// Is the lift takeable on this machine?
pub fn available() -> bool {
    locate().is_some()
}

/// Lift one AscendC source to tile-rs.
///
/// `input` is the path on disk rather than the text already read, because the lifter
/// resolves `#include`s relative to the source: a tiling struct routinely arrives as
/// `"utils/xxx_tiling.h"`, and handing over the text alone loses the directory that
/// makes it findable. Both the kernel directory and its parent go on the include path,
/// which is what the lifter's own corpus tooling does.
pub fn lift(input: &Path) -> Result<String, LiftError> {
    let bin = locate().ok_or(LiftError::NotInstalled)?;
    let dir = input.parent().unwrap_or(Path::new("."));
    let parent = dir.parent().unwrap_or(dir);
    let out = std::env::temp_dir().join(format!(
        "tile-lift-{}-{}.rs",
        std::process::id(),
        input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "kernel".into())
    ));
    let _ = std::fs::remove_file(&out);
    let r = std::process::Command::new(&bin)
        .arg("-I")
        .arg(dir)
        .arg("-I")
        .arg(parent)
        .arg("-o")
        .arg(&out)
        .arg(input)
        .output()
        .map_err(|e| LiftError::Io(format!("could not run {}: {e}", bin.display())))?;
    if !r.status.success() {
        let _ = std::fs::remove_file(&out);
        return Err(LiftError::Rejected {
            code: r.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&r.stderr).to_string(),
        });
    }
    let text = std::fs::read_to_string(&out)
        .map_err(|e| LiftError::Io(format!("{}: {e}", out.display())))?;
    let _ = std::fs::remove_file(&out);
    if text.trim().is_empty() {
        return Err(LiftError::NoOutput);
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The override is what the CLI's own tests use, so it has to actually win.
    #[test]
    fn env_override_beats_path() {
        let f = std::env::temp_dir().join("tile-lift-probe-bin");
        std::fs::write(&f, b"#!/bin/sh\n").unwrap();
        std::env::set_var("TILE_ASCENDC_TO_RS", &f);
        assert_eq!(locate().as_deref(), Some(f.as_path()));
        std::env::remove_var("TILE_ASCENDC_TO_RS");
        let _ = std::fs::remove_file(&f);
    }

    /// A path that does not exist is not a location. Returning it anyway would turn
    /// "not installed" into a confusing exec failure at the moment of use.
    #[test]
    fn override_must_exist() {
        std::env::set_var("TILE_ASCENDC_TO_RS", "/nonexistent/ascendc-to-rs");
        assert!(locate().is_none());
        std::env::remove_var("TILE_ASCENDC_TO_RS");
    }
}
