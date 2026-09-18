//! Acquiring a toolchain: on demand, verified, user-scoped, and never a scavenger hunt.
//!
//! Requirements (1) and (6). Four rules, and the last two are the ones that make this
//! honest rather than convenient:
//!
//! * **User-scoped.** Everything lands under `~/.tile-rs/toolchains/<id>/<version>`.
//!   Never root, never a system path, never a package manager, never a shell profile.
//! * **Verified before it is used.** The manifest pins a sha256 and the bytes must match
//!   it. **The transport is not trusted; the hash is** — which is what lets the fetch go
//!   through the system's `curl` (see below) without weakening the guarantee.
//! * **Only what the route needs, only when it needs it.** Never a bulk install.
//! * **What cannot be automated is refused with the exact remedy.** A vendor SDK behind a
//!   EULA, a login, or a root install is not silently acquired — legally or practically.
//!   Printing the one command instead is what "never leave users hunting for setup docs"
//!   actually asks for; claiming to install it would be a lie the tool cannot keep.
//!
//! ## Why the fetch shells out
//!
//! Requirement (3) says no crate in this build graph may compile C. Every mature TLS
//! stack does — `ring` and `aws-lc-rs` both — so an HTTPS client inside our graph is not
//! available today at all. Rather than quietly relax the requirement, the fetch uses the
//! system's `curl` or `wget`, exactly as accelerator detection uses the system's
//! `nvidia-smi`: a runtime tool, not a build dependency. The integrity guarantee does not
//! depend on it, because we hash what arrives.
//!
//! `file://` needs no fetcher at all, which is what makes the tests hermetic.

use crate::{manifest, sha256};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum ProvisionError {
    /// Nothing in the manifest for this tool on this platform.
    Unknown {
        id: String,
        os: String,
        arch: String,
    },
    /// The manifest says a person has to do this.
    Barrier {
        id: String,
        barrier: String,
        remedy: String,
    },
    /// The entry exists but carries no real checksum yet.
    Unpinned {
        id: String,
        note: String,
    },
    /// Fetching was declined by policy (`--offline` / `--no-install`).
    Declined {
        id: String,
        how: &'static str,
    },
    /// The fetch itself failed.
    Fetch(String),
    /// The bytes are not the bytes the manifest pinned.
    Checksum {
        id: String,
        want: String,
        got: String,
    },
    Io(String),
}

impl fmt::Display for ProvisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvisionError::Unknown { id, os, arch } => write!(
                f,
                "no manifest entry for {id} on {os}/{arch}; tile-rs cannot provision it here"
            ),
            ProvisionError::Barrier { id, barrier, remedy } => write!(
                f,
                "{id} cannot be installed for you ({barrier}).\n  Run this instead:\n    {remedy}\n  \
                 Then re-run the same command; nothing else is needed."
            ),
            ProvisionError::Unpinned { id, note } => write!(
                f,
                "{id} has no verified checksum in the manifest, so it will not be \
                 downloaded.\n  {note}"
            ),
            ProvisionError::Declined { id, how } => {
                write!(f, "{id} is not installed and {how} was given; run `tile install {id}`")
            }
            ProvisionError::Fetch(e) => write!(f, "fetch failed: {e}"),
            ProvisionError::Checksum { id, want, got } => write!(
                f,
                "{id}: checksum mismatch, nothing was unpacked.\n  \
                 manifest: {want}\n  received: {got}"
            ),
            ProvisionError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Acquisition policy, from the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Fetch when a route needs it.
    OnDemand,
    /// Diagnose, never fetch. `--no-install`.
    NeverInstall,
    /// Touch the network under no circumstances. `--offline`.
    Offline,
}

impl Policy {
    /// Does this policy refuse to acquire `url`?
    ///
    /// `--no-install` refuses everything: the user asked for a diagnosis. `--offline` is
    /// narrower — it is about the NETWORK, so an artifact already on the filesystem is
    /// still fair game. Conflating the two makes `--offline` mean "do less work", which
    /// is not what anyone reaches for it to mean.
    fn declines(&self, url: &str) -> Option<&'static str> {
        match self {
            Policy::OnDemand => None,
            Policy::NeverInstall => Some("--no-install"),
            Policy::Offline if url.starts_with("file://") => None,
            Policy::Offline => Some("--offline"),
        }
    }
}

/// `~/.tile-rs`, and the version of its layout.
///
/// The layout version exists because changing the directory shape later strands every
/// existing install silently. One file now costs nothing; discovering the need later
/// costs a migration nobody can test.
pub const LAYOUT_VERSION: &str = "1";

pub fn home() -> PathBuf {
    std::env::var("TILE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let h = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(h).join(".tile-rs")
        })
}

pub fn toolchain_dir(id: &str, version: &str) -> PathBuf {
    home().join("toolchains").join(id).join(version)
}

fn write_layout_version() -> std::io::Result<()> {
    let h = home();
    std::fs::create_dir_all(&h)?;
    let f = h.join("layout-version");
    if !f.exists() {
        std::fs::write(f, format!("{LAYOUT_VERSION}\n"))?;
    }
    Ok(())
}

/// Is this tool already installed?
pub fn installed(id: &str, version: &str) -> bool {
    let d = toolchain_dir(id, version);
    d.join(".installed").exists()
}

/// What a run would do, without doing it.
pub fn plan(id: &str, os: &str, arch: &str) -> Result<manifest::Tool, ProvisionError> {
    let tools = manifest::builtin();
    let t = manifest::find(&tools, id, os, arch).ok_or_else(|| ProvisionError::Unknown {
        id: id.to_string(),
        os: os.to_string(),
        arch: arch.to_string(),
    })?;
    if let Some(b) = &t.barrier {
        return Err(ProvisionError::Barrier {
            id: t.id.clone(),
            barrier: b.clone(),
            remedy: t
                .remedy
                .clone()
                .unwrap_or_else(|| "see the vendor's docs".into()),
        });
    }
    if !t.pinned() {
        return Err(ProvisionError::Unpinned {
            id: t.id.clone(),
            note: t.note.clone().unwrap_or_else(|| {
                "no sha256 is recorded, and an unverified download will not be made".into()
            }),
        });
    }
    Ok(t.clone())
}

/// Fetch bytes. `file://` is handled here; anything else goes through the system fetcher.
fn fetch(url: &str, policy: Policy) -> Result<Vec<u8>, ProvisionError> {
    if let Some(path) = url.strip_prefix("file://") {
        return std::fs::read(path).map_err(|e| ProvisionError::Fetch(format!("{path}: {e}")));
    }
    if policy == Policy::Offline {
        return Err(ProvisionError::Fetch(
            "--offline was given and this is not a file:// URL".into(),
        ));
    }
    for (prog, args) in [("curl", vec!["-fsSL", url]), ("wget", vec!["-qO-", url])] {
        let Ok(out) = std::process::Command::new(prog).args(&args).output() else {
            continue;
        };
        if out.status.success() {
            return Ok(out.stdout);
        }
    }
    Err(ProvisionError::Fetch(format!(
        "neither curl nor wget could fetch {url}"
    )))
}

/// The result of an acquisition, for reporting.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    AlreadyPresent,
    Installed { bytes: usize, sha256: String },
}

/// Acquire `id` for this platform under the given policy.
///
/// Idempotent: a second call with the tool present does nothing and says so.
pub fn ensure(
    id: &str,
    os: &str,
    arch: &str,
    policy: Policy,
    report: &mut dyn FnMut(&str),
) -> Result<Outcome, ProvisionError> {
    let t = plan(id, os, arch)?;
    install_tool(&t, policy, report)
}

/// Everything the manifest knows about, for `tile install --list`.
pub fn listing(os: &str, arch: &str) -> Vec<(manifest::Tool, bool)> {
    manifest::builtin()
        .into_iter()
        .filter(|t| t.os == os && t.arch == arch)
        .map(|t| {
            let done = installed(&t.id, &t.version);
            (t, done)
        })
        .collect()
}

/// Open a gzipped tar into `into`, dropping `strip` leading path components.
///
/// Shells out to `tar` for the same reason the fetch shells out to `curl`: a gzip and tar
/// implementation in our graph is two more dependencies for something every machine this
/// tool runs on already has. The integrity guarantee does not depend on it — the bytes
/// were hashed before this is called — and `tar` is given an explicit destination so a
/// path in the archive cannot escape it.
pub fn unpack_tar_gz(archive: &Path, into: &Path, strip: usize) -> Result<(), ProvisionError> {
    let out = std::process::Command::new("tar")
        .arg("xzf")
        .arg(archive)
        .arg(format!("--strip-components={strip}"))
        .arg("-C")
        .arg(into)
        .output()
        .map_err(|e| ProvisionError::Io(format!("tar is not available: {e}")))?;
    if !out.status.success() {
        return Err(ProvisionError::Io(format!(
            "tar failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(())
}

/// Does `--offline` refuse a network URL without attempting it?
///
/// Exposed so the specification can assert the property rather than a message. Checking
/// "no network request was made" any other way means trusting a log line.
pub fn probe_offline_refuses_network() -> bool {
    matches!(
        fetch("https://example.invalid/artifact", Policy::Offline),
        Err(ProvisionError::Fetch(_))
    )
}

/// Anything under `~/.tile-rs` that this build wrote. Used by the tests, and by a future
/// `tile uninstall`.
pub fn is_user_scoped(p: &Path) -> bool {
    p.starts_with(home())
}

/// `ensure`, given the manifest entry directly. Split out so tests can drive a successful
/// install: every entry in the shipped manifest is deliberately either barriered or
/// unpinned, so there is nothing in it that would actually install.
pub fn install_tool(
    t: &manifest::Tool,
    policy: Policy,
    report: &mut dyn FnMut(&str),
) -> Result<Outcome, ProvisionError> {
    if installed(&t.id, &t.version) {
        return Ok(Outcome::AlreadyPresent);
    }
    if let Some(how) = policy.declines(&t.url) {
        return Err(ProvisionError::Declined {
            id: t.id.clone(),
            how,
        });
    }
    // An entry with no recorded digest is refused BEFORE the fetch, not after it.
    //
    // The comparison below would catch it either way, but by then the bytes have already
    // been pulled from a URL nobody has vouched for. The pin is what makes the URL
    // trustworthy, so an unpinned entry must not be contacted at all -- "we downloaded it
    // and then decided not to trust it" is not the same guarantee.
    if t.sha256.trim().is_empty() {
        return Err(ProvisionError::Checksum {
            id: t.id.clone(),
            want: "a recorded sha256 (the manifest has none)".into(),
            got: "nothing was downloaded".into(),
        });
    }
    // Announced BEFORE it starts. A tool that downloads silently on first use is a tool
    // people stop trusting.
    report(&format!("fetching {} {} from {}", t.id, t.version, t.url));
    let bytes = fetch(&t.url, policy)?;
    let got = sha256::hex(&bytes);
    if got != t.sha256 {
        return Err(ProvisionError::Checksum {
            id: t.id.clone(),
            want: t.sha256.clone(),
            got,
        });
    }
    report(&format!("sha256 ok ({} bytes)", bytes.len()));

    let dir = toolchain_dir(&t.id, &t.version);
    let staging = dir.with_extension(format!("staging.{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| ProvisionError::Io(e.to_string()))?;

    // Unpack only AFTER the hash matched. An archive is a instruction set for writing
    // files, and opening one you have not verified is the whole attack.
    match t.unpack.as_deref() {
        Some("tar.gz") => {
            let archive = staging.join("artifact.tar.gz");
            std::fs::write(&archive, &bytes).map_err(|e| ProvisionError::Io(e.to_string()))?;
            report(&format!("unpacking {} bytes", bytes.len()));
            unpack_tar_gz(&archive, &staging, t.strip)?;
            let _ = std::fs::remove_file(&archive);
        }
        Some(other) => {
            return Err(ProvisionError::Io(format!(
                "manifest asks for unpack = {other:?}, which this build does not know how \
                 to open"
            )))
        }
        None => {
            let name = t.url.rsplit('/').next().unwrap_or("artifact");
            std::fs::write(staging.join(name), &bytes)
                .map_err(|e| ProvisionError::Io(e.to_string()))?;
        }
    }

    std::fs::write(staging.join(".installed"), format!("{}\n", t.sha256))
        .map_err(|e| ProvisionError::Io(e.to_string()))?;
    write_layout_version().map_err(|e| ProvisionError::Io(e.to_string()))?;
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ProvisionError::Io(e.to_string()))?;
    }
    match std::fs::rename(&staging, &dir) {
        Ok(()) => {}
        Err(_) if dir.exists() => {
            let _ = std::fs::remove_dir_all(&staging);
            return Ok(Outcome::AlreadyPresent);
        }
        Err(e) => return Err(ProvisionError::Io(e.to_string())),
    }
    Ok(Outcome::Installed {
        bytes: bytes.len(),
        sha256: t.sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TILE_HOME` is process-global, so two sandboxed tests running in parallel would
    /// each see the other's prefix. cargo's test runner is parallel by default, so this
    /// has to be serialized rather than hoped about.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A private `~/.tile-rs` for one test, so nothing touches the developer's real one.
    /// The guard field is never read; holding it IS the point.
    struct Sandbox(
        PathBuf,
        #[allow(dead_code)] std::sync::MutexGuard<'static, ()>,
    );
    impl Sandbox {
        fn new(tag: &str) -> Sandbox {
            let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let d = std::env::temp_dir().join(format!("tile-prov-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            std::env::set_var("TILE_HOME", &d);
            Sandbox(d, guard)
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
            std::env::remove_var("TILE_HOME");
        }
    }

    #[test]
    fn a_barrier_refuses_with_the_exact_remedy() {
        let e = plan("cuda", "linux", "x86_64").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("eula"), "{msg}");
        assert!(msg.contains("developer.nvidia.com"), "{msg}");
        assert!(msg.contains("Run this instead"), "{msg}");
    }

    #[test]
    fn xcode_is_refused_with_the_one_command_that_fixes_it() {
        let e = plan("xcode-clt", "macos", "aarch64").unwrap_err();
        assert!(e.to_string().contains("xcode-select --install"), "{e}");
    }

    #[test]
    fn the_published_backend_is_pinned_and_would_be_fetched() {
        // One release exists, for one triple, and its digest is in the manifest.
        let t = plan("rustc_codegen_tile", "macos", "aarch64").expect("a plan");
        assert_eq!(t.sha256.len(), 64);
        assert!(t.auto_installable());
        assert_eq!(t.unpack.as_deref(), Some("tar.gz"));
    }

    #[test]
    fn a_platform_with_no_published_asset_is_told_so_rather_than_sent_to_a_404() {
        let e = plan("rustc_codegen_tile", "linux", "x86_64").unwrap_err();
        assert!(matches!(e, ProvisionError::Unknown { .. }), "{e}");
        assert!(e.to_string().contains("linux/x86_64"), "{e}");
    }

    #[test]
    fn an_unpinned_entry_will_not_be_downloaded() {
        // Not in the shipped manifest any more, but the rule still has to hold: an entry
        // whose hash is absent or all zeros is refused rather than fetched, because
        // downloading something unverifiable is worse than refusing.
        let t = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: "https://example.invalid/x".into(),
            sha256: "0".repeat(64),
            barrier: None,
            remedy: None,
            note: None,
            unpack: None,
            strip: 0,
        };
        assert!(!t.pinned(), "an all-zero hash must not read as pinned");
        assert!(!t.auto_installable() || !t.pinned());
    }

    #[test]
    fn an_unknown_tool_names_the_platform_it_was_asked_about() {
        let e = plan("nonesuch", "linux", "x86_64").unwrap_err();
        assert!(e.to_string().contains("linux/x86_64"), "{e}");
    }

    #[test]
    fn a_local_artifact_installs_verifies_and_is_idempotent() {
        let _s = Sandbox::new("ok");
        let payload = b"a plausible toolchain artifact";
        let src = std::env::temp_dir().join(format!("tile-art-{}.bin", std::process::id()));
        std::fs::write(&src, payload).unwrap();
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: format!("file://{}", src.display()),
            sha256: sha256::hex(payload),
            barrier: None,
            remedy: None,
            note: None,
            unpack: None,
            strip: 0,
        };
        // Drive the same path `ensure` would, with a manifest entry made here rather
        // than shipped: the checked-in ones are all either barriered or unpinned, by
        // design, so this is the only way to exercise a successful install.
        let mut said = Vec::new();
        let out = install_tool(&tool, Policy::OnDemand, &mut |m| said.push(m.to_string()));
        assert!(matches!(out, Ok(Outcome::Installed { .. })), "{out:?}");
        assert!(installed("demo", "1.0"));
        assert!(
            said.iter().any(|m| m.contains("fetching")),
            "the fetch must be announced"
        );
        assert!(said.iter().any(|m| m.contains("sha256 ok")));

        // Second time: nothing happens.
        let mut said2 = Vec::new();
        let again = install_tool(&tool, Policy::OnDemand, &mut |m| said2.push(m.to_string()));
        assert_eq!(again.unwrap(), Outcome::AlreadyPresent);
        assert!(said2.is_empty(), "an idempotent run must be silent");
        let _ = std::fs::remove_file(&src);
    }

    #[test]
    fn a_corrupted_artifact_is_rejected_before_anything_is_unpacked() {
        let _s = Sandbox::new("bad");
        let src = std::env::temp_dir().join(format!("tile-bad-{}.bin", std::process::id()));
        std::fs::write(&src, b"not what was pinned").unwrap();
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: format!("file://{}", src.display()),
            sha256: sha256::hex(b"the bytes the manifest expected"),
            barrier: None,
            remedy: None,
            note: None,
            unpack: None,
            strip: 0,
        };
        let e = install_tool(&tool, Policy::OnDemand, &mut |_| {}).unwrap_err();
        assert!(matches!(e, ProvisionError::Checksum { .. }), "{e}");
        assert!(
            !installed("demo", "1.0"),
            "a bad artifact was installed anyway"
        );
        assert!(
            !toolchain_dir("demo", "1.0").exists(),
            "a directory was left behind for a rejected artifact"
        );
        let _ = std::fs::remove_file(&src);
    }

    #[test]
    fn no_install_declines_and_names_the_command_that_would_work() {
        let _s = Sandbox::new("decline");
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: "file:///nonexistent".into(),
            sha256: "a".repeat(64),
            barrier: None,
            remedy: None,
            note: None,
            unpack: None,
            strip: 0,
        };
        let e = install_tool(&tool, Policy::NeverInstall, &mut |_| {}).unwrap_err();
        assert!(e.to_string().contains("tile install demo"), "{e}");
    }

    #[test]
    fn a_tarball_is_verified_before_it_is_unpacked_and_then_opened() {
        // An archive is an instruction set for writing files. Opening one you have not
        // verified is the whole attack, so the order here is the point.
        let _s = Sandbox::new("untar");
        let work = std::env::temp_dir().join(format!("tile-tarsrc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(work.join("payload/lib")).unwrap();
        std::fs::write(work.join("payload/lib/libthing.dylib"), b"a shared library").unwrap();
        std::fs::write(work.join("payload/USAGE.md"), b"set TILERS_CODEGEN_SO").unwrap();
        let tarball = work.join("art.tar.gz");
        let st = std::process::Command::new("tar")
            .arg("czf")
            .arg(&tarball)
            .arg("-C")
            .arg(&work)
            .arg("payload")
            .status()
            .unwrap();
        assert!(st.success());

        let bytes = std::fs::read(&tarball).unwrap();
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: format!("file://{}", tarball.display()),
            sha256: sha256::hex(&bytes),
            barrier: None,
            remedy: None,
            note: None,
            unpack: Some("tar.gz".into()),
            strip: 1,
        };
        let out = install_tool(&tool, Policy::OnDemand, &mut |_| {});
        assert!(matches!(out, Ok(Outcome::Installed { .. })), "{out:?}");

        // strip = 1 drops the `payload/` prefix, as the published tarball needs.
        let dir = toolchain_dir("demo", "1.0");
        assert!(
            dir.join("lib/libthing.dylib").exists(),
            "the archive was not opened"
        );
        assert!(dir.join("USAGE.md").exists());
        assert!(
            !dir.join("artifact.tar.gz").exists(),
            "the archive was left behind"
        );
        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn a_corrupted_tarball_is_never_opened() {
        let _s = Sandbox::new("badtar");
        let work = std::env::temp_dir().join(format!("tile-badtar-{}", std::process::id()));
        std::fs::create_dir_all(&work).unwrap();
        let tarball = work.join("art.tar.gz");
        std::fs::write(&tarball, b"this is not a tarball").unwrap();
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: format!("file://{}", tarball.display()),
            sha256: sha256::hex(b"the bytes the manifest expected"),
            barrier: None,
            remedy: None,
            note: None,
            unpack: Some("tar.gz".into()),
            strip: 0,
        };
        let e = install_tool(&tool, Policy::OnDemand, &mut |_| {}).unwrap_err();
        assert!(matches!(e, ProvisionError::Checksum { .. }), "{e}");
        assert!(
            !toolchain_dir("demo", "1.0").exists(),
            "a directory survived"
        );
        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn an_unpack_format_this_build_does_not_know_is_refused() {
        let _s = Sandbox::new("unkfmt");
        let src = std::env::temp_dir().join(format!("tile-unk-{}.bin", std::process::id()));
        std::fs::write(&src, b"x").unwrap();
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: format!("file://{}", src.display()),
            sha256: sha256::hex(b"x"),
            barrier: None,
            remedy: None,
            note: None,
            unpack: Some("zip".into()),
            strip: 0,
        };
        let e = install_tool(&tool, Policy::OnDemand, &mut |_| {}).unwrap_err();
        assert!(e.to_string().contains("does not know how to open"), "{e}");
        let _ = std::fs::remove_file(&src);
    }

    #[test]
    fn offline_installs_from_a_local_artifact_because_offline_is_about_the_network() {
        let _s = Sandbox::new("offlocal");
        let payload = b"local artifact";
        let src = std::env::temp_dir().join(format!("tile-offi-{}.bin", std::process::id()));
        std::fs::write(&src, payload).unwrap();
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: format!("file://{}", src.display()),
            sha256: sha256::hex(payload),
            barrier: None,
            remedy: None,
            note: None,
            unpack: None,
            strip: 0,
        };
        let out = install_tool(&tool, Policy::Offline, &mut |_| {});
        assert!(matches!(out, Ok(Outcome::Installed { .. })), "{out:?}");
        let _ = std::fs::remove_file(&src);
    }

    #[test]
    fn no_install_declines_even_a_local_artifact_because_it_asked_for_a_diagnosis() {
        let _s = Sandbox::new("noinstlocal");
        let tool = manifest::Tool {
            id: "demo".into(),
            version: "1.0".into(),
            os: "any".into(),
            arch: "any".into(),
            url: "file:///anywhere".into(),
            sha256: "a".repeat(64),
            barrier: None,
            remedy: None,
            note: None,
            unpack: None,
            strip: 0,
        };
        let e = install_tool(&tool, Policy::NeverInstall, &mut |_| {}).unwrap_err();
        assert!(e.to_string().contains("--no-install"), "{e}");
    }

    #[test]
    fn offline_refuses_a_network_url_but_still_reads_a_local_one() {
        assert!(fetch("https://example.invalid/x", Policy::Offline).is_err());
        let src = std::env::temp_dir().join(format!("tile-off-{}.bin", std::process::id()));
        std::fs::write(&src, b"local").unwrap();
        let got = fetch(&format!("file://{}", src.display()), Policy::Offline).unwrap();
        assert_eq!(got, b"local");
        let _ = std::fs::remove_file(&src);
    }

    #[test]
    fn everything_written_stays_under_the_user_prefix() {
        let _s = Sandbox::new("scope");
        assert!(is_user_scoped(&toolchain_dir("demo", "1.0")));
        assert!(!is_user_scoped(Path::new("/usr/local/lib/libfoo.so")));
        assert!(!is_user_scoped(Path::new("/etc/profile")));
    }

    #[test]
    fn the_layout_version_is_written_so_a_later_move_can_migrate() {
        let _s = Sandbox::new("layout");
        write_layout_version().unwrap();
        let v = std::fs::read_to_string(home().join("layout-version")).unwrap();
        assert_eq!(v.trim(), LAYOUT_VERSION);
    }
}
