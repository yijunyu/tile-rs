//! `-O3`: compile the emitted source with the target's own compiler and report what it
//! says.
//!
//! This is the whole difference between O2 and O3. O2 reasons about the kernel; O3 asks
//! the vendor's compiler, which knows things no amount of reasoning here will. On a Mac
//! with the Metal toolchain that is a real answer today: `xcrun metal -c` compiles the
//! MSL this tool just wrote and reports its diagnostics.
//!
//! Two rules:
//!
//! * **Absence is not failure.** A target whose compiler is not installed is reported as
//!   unavailable, and the run continues at the level that needed nothing — with the
//!   degradation named. Refusing the whole conversion because a *check* could not run
//!   would be worse than not offering the check.
//! * **The compiler's word is the compiler's.** Its diagnostics are passed through, not
//!   summarised into a verdict. A tool that paraphrases a compiler error is a tool that
//!   eventually paraphrases it wrongly.

use crate::forms::Form;
use std::fmt;
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub enum Feedback {
    /// The compiler ran and accepted it.
    Ok { compiler: String },
    /// The compiler ran and rejected it. Its own words.
    Rejected {
        compiler: String,
        diagnostics: String,
    },
    /// No compiler for this form on this machine.
    Unavailable { want: &'static str, why: String },
    /// tile-rs knows no compiler for this form at all.
    NoCompilerKnown { form: &'static str },
}

impl fmt::Display for Feedback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Feedback::Ok { compiler } => write!(f, "{compiler} accepted the generated source"),
            Feedback::Rejected {
                compiler,
                diagnostics,
            } => {
                write!(
                    f,
                    "{compiler} rejected the generated source:\n{diagnostics}"
                )
            }
            Feedback::Unavailable { want, why } => write!(
                f,
                "-O3 wanted {want} to check the generated source; {why}. \
                 The conversion still ran, at the level that needs nothing."
            ),
            Feedback::NoCompilerKnown { form } => write!(
                f,
                "-O3 has no compiler to consult for \"{form}\"; the conversion still ran."
            ),
        }
    }
}

impl Feedback {
    /// Did the check actually happen? Used to decide whether O3 delivered anything.
    pub fn ran(&self) -> bool {
        matches!(self, Feedback::Ok { .. } | Feedback::Rejected { .. })
    }
}

/// The compiler for a form: the program, the arguments before the file, and whether it
/// wants the source on stdin or as a path.
fn compiler_for(form: &str) -> Option<(&'static str, Vec<&'static str>, &'static str)> {
    match form {
        // `xcrun metal` is the Metal front end; -c stops before linking, which is all a
        // syntax and semantics check needs.
        "msl" => Some((
            "xcrun",
            vec!["metal", "-x", "metal", "-c", "-o", "/dev/null"],
            "metal",
        )),
        "gpu" => Some(("nvcc", vec!["-x", "cu", "-c", "-o", "/dev/null"], "nvcc")),
        "spirv" => Some(("glslangValidator", vec!["-S", "comp"], "glslang")),
        _ => None,
    }
}

/// Compile `source` as `form` and report what the compiler said.
pub fn check(form: &Form, source: &str) -> Feedback {
    let Some((prog, args, label)) = compiler_for(form.id) else {
        return Feedback::NoCompilerKnown { form: form.id };
    };

    // Write beside the process, not into the user's tree: a check must not litter.
    //
    // The name carries a per-call sequence as well as the pid. Two concurrent checks of
    // the same form in one process -- an MCP daemon serving two clients, or just a
    // parallel test run -- would otherwise stage onto the same path and compile each
    // other's source.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "tile-o3-{}-{}-{}.{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        form.id,
        form.primary_ext()
    ));
    if let Err(e) = std::fs::write(&path, source) {
        return Feedback::Unavailable {
            want: label,
            why: format!("could not stage it: {e}"),
        };
    }

    let out = Command::new(prog).args(&args).arg(&path).output();
    let _ = std::fs::remove_file(&path);

    match out {
        Err(e) => Feedback::Unavailable {
            want: label,
            why: format!("{prog} is not on PATH ({e})"),
        },
        Ok(o) if o.status.success() => Feedback::Ok {
            compiler: label.to_string(),
        },
        Ok(o) => {
            let diag = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stderr),
                String::from_utf8_lossy(&o.stdout)
            );
            // A non-zero exit with no output is the toolchain being broken or absent
            // rather than the source being wrong; saying "rejected" would blame the
            // kernel for the machine.
            // A missing toolchain component is a machine problem, not a kernel
            // problem: `xcrun metal` exists but was built without MetalToolchain.
            // Blaming the generated source for that would be the same lie as
            // treating an absent compiler as a rejection.
            let missing_component = diag.contains("missing Metal Toolchain")
                || diag.contains("missing toolchain")
                || diag.contains("cannot execute tool");
            if diag.trim().is_empty() || missing_component {
                Feedback::Unavailable {
                    want: label,
                    why: if missing_component {
                        format!("{prog}: {diag}", diag = diag.trim())
                    } else {
                        format!("{prog} exited {} with no diagnostics", o.status)
                    },
                }
            } else {
                Feedback::Rejected {
                    compiler: label.to_string(),
                    diagnostics: diag.trim().into(),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms;

    #[test]
    fn a_form_with_no_known_compiler_says_so_rather_than_failing() {
        let f = forms::by_id("csl").unwrap();
        assert_eq!(
            check(f, "comptime {}"),
            Feedback::NoCompilerKnown { form: "csl" }
        );
    }

    #[test]
    fn an_absent_compiler_is_unavailable_not_rejected() {
        let _g = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        // Blaming the kernel for the machine is the failure mode this guards.
        let f = forms::by_id("gpu").unwrap();
        let fb = check(f, "__global__ void k() {}");
        // nvcc is not on a Mac; on a CUDA box this compiles. Both are fine; what must
        // never happen is "rejected" for a machine problem.
        match fb {
            Feedback::Unavailable { .. } | Feedback::Ok { .. } => {}
            other => panic!("expected unavailable or ok, got {other:?}"),
        }
    }

    /// The staging counter is process-wide, so a test that counts staged files has to be
    /// the only one staging any.
    static STAGING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_check_never_leaves_a_file_behind() {
        let _g = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let before = staged_files();
        let _ = check(forms::by_id("msl").unwrap(), "kernel void k() {}");
        assert_eq!(staged_files(), before, "-O3 littered the temp directory");
    }

    fn staged_files() -> usize {
        std::fs::read_dir(std::env::temp_dir())
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with("tile-o3-"))
                    .count()
            })
            .unwrap_or(0)
    }

    #[test]
    fn unavailable_explains_that_the_conversion_still_ran() {
        // Refusing a conversion because a CHECK could not run would be worse than not
        // offering the check.
        let fb = Feedback::Unavailable {
            want: "metal",
            why: "not on PATH".into(),
        };
        assert!(fb.to_string().contains("still ran"), "{fb}");
        assert!(!fb.ran());
    }

    #[test]
    fn where_a_real_compiler_exists_o3_is_a_real_check() {
        // No `cfg(target_os)` guard: the check already skips when the compiler is not
        // there, and a cfg here would be platform behaviour leaking out of
        // `platform/` — which the specification forbids for a reason.
        let _g = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        // The point of O3: a real compiler, not our reasoning about one.
        let f = forms::by_id("msl").unwrap();
        let good = check(f, "#include <metal_stdlib>\nkernel void k() {}\n");
        if !good.ran() {
            eprintln!("skipped: no Metal toolchain here ({good})");
            return;
        }
        assert!(matches!(good, Feedback::Ok { .. }), "{good}");

        let bad = check(
            f,
            "#include <metal_stdlib>\nkernel void k( { syntax error\n",
        );
        match bad {
            Feedback::Rejected { diagnostics, .. } => {
                assert!(!diagnostics.is_empty(), "a rejection with no diagnostics");
            }
            other => panic!("the compiler should have rejected that: {other}"),
        }
    }
}
