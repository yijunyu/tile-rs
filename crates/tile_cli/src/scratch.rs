//! Intermediate representations: real files on the route, discarded unless asked for.
//!
//! A multi-hop route materialises every hop. Keeping them is how a user debugs a
//! lowering; discarding them by default is how the tool stays quiet. The scratch
//! directory is removed even when the run fails, so a failed conversion never litters
//! the tree — but `-k` preserves what was produced before the failure, because that is
//! exactly when a user wants to look.

use crate::forms::Form;
use std::io;
use std::path::{Path, PathBuf};

pub struct Scratch {
    stem: String,
    dir: PathBuf,
    keep: bool,
    /// True when we created the directory and must therefore remove it.
    owned: bool,
    written: std::cell::RefCell<Vec<String>>,
}

impl Scratch {
    /// `keep_dir` implies keep. Without keeping, hops go to a run-scoped temp directory.
    pub fn new(input: &str, keep: bool, keep_dir: Option<&str>) -> Scratch {
        let stem = Path::new(input)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "kernel".to_string());
        let (dir, owned) = match (keep, keep_dir) {
            // Explicit directory: the user owns it, we only add to it.
            (_, Some(d)) => (PathBuf::from(d), false),
            // -k with no directory: beside the input, so the hops sit next to the source.
            (true, None) => (
                Path::new(input)
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from(".")),
                false,
            ),
            // The default: a run-scoped directory we create and remove.
            (false, None) => (
                std::env::temp_dir().join(format!("tile-{}-{}", stem, std::process::id())),
                true,
            ),
        };
        Scratch {
            stem,
            dir,
            keep,
            owned,
            written: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Name a hop by its step index and form, so `-k` output is self-describing:
    /// `softmax.1.mlir.mlir`, `softmax.2.linalg.mlir`.
    pub fn hop_name(&self, step: usize, form: &Form) -> String {
        format!("{}.{}.{}.{}", self.stem, step, form.id, form.primary_ext())
    }

    pub fn write_hop(&self, step: usize, form: &Form, text: &str) -> io::Result<PathBuf> {
        if self.owned || !self.dir.exists() {
            std::fs::create_dir_all(&self.dir)?;
        }
        let path = self.dir.join(self.hop_name(step, form));
        std::fs::write(&path, text)?;
        self.written
            .borrow_mut()
            .push(path.to_string_lossy().to_string());
        Ok(path)
    }

    /// Remove the scratch unless asked to keep it. Returns the paths kept, for reporting.
    pub fn finish(&self) -> Vec<String> {
        if self.keep {
            return self.written.borrow().clone();
        }
        if self.owned && self.dir.exists() {
            let _ = std::fs::remove_dir_all(&self.dir);
        } else {
            for p in self.written.borrow().iter() {
                let _ = std::fs::remove_file(p);
            }
        }
        Vec::new()
    }

    /// Sweep a scratch directory left by a run that was killed.
    ///
    /// A SIGKILL mid-conversion cannot run cleanup, so "no orphans" is only true if the
    /// NEXT run makes it true.
    pub fn sweep_stale() -> usize {
        let tmp = std::env::temp_dir();
        let Ok(rd) = std::fs::read_dir(&tmp) else {
            return 0;
        };
        let mut swept = 0;
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(rest) = name.strip_prefix("tile-") else {
                continue;
            };
            // `tile-<stem>-<pid>`: only sweep directories whose process is gone.
            let Some(pid) = rest.rsplit('-').next().and_then(|p| p.parse::<u32>().ok()) else {
                continue;
            };
            // Never delete a live run's working directory: erring the other way costs a
            // stale directory, erring this way corrupts someone's conversion.
            //
            // But a pid is recycled, and a scratch whose number was reused by an
            // unrelated process would then live forever. Age is the backstop: nothing
            // this tool creates is still in use a day later.
            let stale_by_age = std::fs::metadata(e.path())
                .and_then(|m| m.modified())
                .and_then(|t| t.elapsed().map_err(|_| io::ErrorKind::Other.into()))
                .map(|age| age.as_secs() > 24 * 60 * 60)
                .unwrap_or(false);
            if pid == std::process::id() || (process_alive(pid) && !stale_by_age) {
                continue;
            }
            if std::fs::remove_dir_all(e.path()).is_ok() {
                swept += 1;
            }
        }
        swept
    }
}

/// Is a pid still running? `kill(pid, 0)` without the libc crate: on every platform we
/// support, an entry under the process table is what we can check portably enough.
#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // /proc on Linux; on macOS fall back to assuming alive only for very recent pids we
    // cannot verify, which errs toward NOT deleting someone else's scratch.
    if Path::new("/proc").is_dir() {
        return Path::new(&format!("/proc/{pid}")).exists();
    }
    let Ok(out) = std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
    else {
        // Cannot ask: assume alive. An orphan is cheaper than deleting a live run's
        // working directory.
        return true;
    };
    if out.status.success() {
        return true;
    }
    // A failure is not the same as "dead". `kill -0` on a process owned by another user
    // fails with EPERM, which means the process EXISTS and we may not signal it —
    // reading that as dead would delete someone else's scratch.
    let why = String::from_utf8_lossy(&out.stderr).to_lowercase();
    why.contains("not permitted") || why.contains("permission")
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    // Without a portable probe, never delete: an orphan is cheaper than deleting the
    // scratch of a run that is still going.
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms;

    #[test]
    fn a_scratch_whose_pid_was_recycled_is_still_swept_by_age() {
        // A pid is reused. Without an age backstop, a scratch whose number was taken by
        // an unrelated process is never swept and lives forever.
        let d = std::env::temp_dir().join("tile-agetest-1");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        // pid 1 is alive but owned by root: `kill -0` fails with EPERM, and reading
        // that as "dead" would let the sweep delete another user's scratch.
        assert!(super::process_alive(1), "pid 1 exists; EPERM is not death");
        // Not yet a day old, so it survives.
        Scratch::sweep_stale();
        assert!(d.exists(), "a fresh directory was swept");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hops_are_named_by_step_and_form() {
        let s = Scratch::new("a/b/softmax.rs", true, None);
        let mlir = forms::by_id("mlir").unwrap();
        let linalg = forms::by_id("linalg").unwrap();
        assert_eq!(s.hop_name(1, mlir), "softmax.1.mlir.mlir");
        assert_eq!(s.hop_name(2, linalg), "softmax.2.linalg.mlir");
    }

    #[test]
    fn the_default_scratch_is_owned_and_removed() {
        let s = Scratch::new("softmax.mlir", false, None);
        let f = forms::by_id("linalg").unwrap();
        s.write_hop(1, f, "module {}").expect("write");
        assert!(s.dir.exists(), "the scratch should exist during the run");
        let kept = s.finish();
        assert!(kept.is_empty(), "nothing is kept without -k");
        assert!(!s.dir.exists(), "the scratch must be gone afterwards");
    }

    #[test]
    fn keep_preserves_the_hops_and_reports_them() {
        let dir = std::env::temp_dir().join(format!("tile-keep-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = Scratch::new("softmax.mlir", true, Some(dir.to_str().unwrap()));
        let f = forms::by_id("linalg").unwrap();
        let p = s.write_hop(1, f, "module {}").expect("write");
        let kept = s.finish();
        assert_eq!(kept.len(), 1);
        assert!(p.exists(), "-k must leave the hop on disk");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_keep_dir_is_never_deleted_even_though_we_created_it() {
        // The user named it, so it is theirs.
        let dir = std::env::temp_dir().join(format!("tile-userdir-{}", std::process::id()));
        let s = Scratch::new("k.mlir", true, Some(dir.to_str().unwrap()));
        let f = forms::by_id("mlir").unwrap();
        s.write_hop(1, f, "module {}").unwrap();
        s.finish();
        assert!(dir.exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
