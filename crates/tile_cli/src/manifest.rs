//! The provisioning manifest: pinned data, parsed without a dependency.
//!
//! A restricted TOML subset — `[[tool]]` tables of `key = "value"` — because that is all
//! the manifest needs and a TOML crate would cost more than the feature. The parser is
//! deliberately strict: an unknown key or a malformed line is an ERROR, not something to
//! skip. A manifest that silently ignores what it does not understand is how a typo in a
//! `sha256` becomes an unverified download.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub id: String,
    pub version: String,
    pub os: String,
    pub arch: String,
    pub url: String,
    pub sha256: String,
    /// Why tile-rs will not install this for you: `eula`, `vendor-login`, `os-level`.
    pub barrier: Option<String>,
    /// The exact command or step a person must run when there is a barrier.
    pub remedy: Option<String>,
    pub note: Option<String>,
    /// How to open what was downloaded: `tar.gz`, or nothing for a bare file.
    pub unpack: Option<String>,
    /// Leading path components to drop while unpacking, as `tar --strip-components`.
    pub strip: usize,
}

impl Tool {
    /// May the tool acquire this itself?
    pub fn auto_installable(&self) -> bool {
        self.barrier.is_none() && !self.sha256.is_empty()
    }
    /// Has this entry been given a real checksum yet?
    pub fn pinned(&self) -> bool {
        self.sha256.len() == 64 && self.sha256.chars().any(|c| c != '0')
    }
}

#[derive(Debug)]
pub struct ParseError {
    pub line: usize,
    pub what: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provision manifest line {}: {}", self.line, self.what)
    }
}

const KEYS: &[&str] = &[
    "id", "version", "os", "arch", "url", "sha256", "barrier", "remedy", "note", "verify",
    "unpack", "strip",
];

pub fn parse(text: &str) -> Result<Vec<Tool>, ParseError> {
    let mut out = Vec::new();
    let mut cur: Option<BTreeMap<String, String>> = None;

    for (n, raw) in text.lines().enumerate() {
        let line = n + 1;
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t == "[[tool]]" {
            if let Some(map) = cur.take() {
                out.push(finish(map, line)?);
            }
            cur = Some(BTreeMap::new());
            continue;
        }
        let Some((k, v)) = t.split_once('=') else {
            return Err(ParseError {
                line,
                what: format!("not a key = value pair: {t:?}"),
            });
        };
        let k = k.trim().to_string();
        if !KEYS.contains(&k.as_str()) {
            // Strict on purpose: skipping unknown keys is how a typo'd `sha256` becomes
            // an unverified download.
            return Err(ParseError {
                line,
                what: format!("unknown key {k:?}"),
            });
        }
        let v = v.trim();
        let v = v
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .ok_or_else(|| ParseError {
                line,
                what: format!("value must be quoted: {v:?}"),
            })?;
        match cur.as_mut() {
            Some(map) => {
                map.insert(k, v.to_string());
            }
            None => {
                return Err(ParseError {
                    line,
                    what: "key outside any [[tool]]".into(),
                })
            }
        }
    }
    if let Some(map) = cur.take() {
        out.push(finish(map, text.lines().count())?);
    }
    Ok(out)
}

fn finish(map: BTreeMap<String, String>, line: usize) -> Result<Tool, ParseError> {
    let need = |k: &str| -> Result<String, ParseError> {
        map.get(k).cloned().ok_or_else(|| ParseError {
            line,
            what: format!("[[tool]] has no {k}"),
        })
    };
    Ok(Tool {
        id: need("id")?,
        version: need("version")?,
        os: need("os")?,
        arch: need("arch")?,
        url: need("url")?,
        sha256: map.get("sha256").cloned().unwrap_or_default(),
        barrier: map.get("barrier").cloned(),
        remedy: map.get("remedy").cloned(),
        note: map.get("note").cloned(),
        unpack: map.get("unpack").cloned(),
        strip: map.get("strip").and_then(|v| v.parse().ok()).unwrap_or(0),
    })
}

/// The manifest compiled into this binary.
pub fn builtin() -> Vec<Tool> {
    parse(include_str!("../assets/provision.toml")).expect("the checked-in manifest parses")
}

/// The entry for `id` on this (os, arch), if there is one.
pub fn find<'a>(tools: &'a [Tool], id: &str, os: &str, arch: &str) -> Option<&'a Tool> {
    tools
        .iter()
        .find(|t| t.id == id && t.os == os && t.arch == arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checked_in_manifest_parses() {
        let tools = builtin();
        assert!(!tools.is_empty());
    }

    #[test]
    fn every_entry_that_cannot_be_auto_installed_carries_a_remedy() {
        // "Never leave users hunting for setup docs" is only true if the refusal itself
        // contains the answer. An entry with a barrier and no remedy is a dead end.
        for t in builtin() {
            if t.barrier.is_some() {
                assert!(t.remedy.is_some(), "{} has a barrier but no remedy", t.id);
            }
        }
    }

    #[test]
    fn a_barrier_entry_is_never_auto_installable() {
        for t in builtin() {
            if t.barrier.is_some() {
                assert!(
                    !t.auto_installable(),
                    "{} would be silently installed",
                    t.id
                );
            }
        }
    }

    #[test]
    fn the_codegen_backend_is_pinned_to_the_published_digest() {
        let tools = builtin();
        let t = find(&tools, "rustc_codegen_tile", "macos", "aarch64").expect("the entry");
        assert!(t.pinned(), "the entry must carry a real hash");
        assert!(t.auto_installable());
        assert_eq!(t.unpack.as_deref(), Some("tar.gz"));
        assert_eq!(t.strip, 1, "the tarball has a leading directory");
        // The version encodes the nightly, because the backend dlopens rustc internals
        // and does not load into another.
        assert!(t.version.contains("nightly-"), "{}", t.version);
    }

    #[test]
    fn no_entry_exists_for_a_platform_with_no_published_asset() {
        // One release, one asset. A linux entry would be a URL that 404s, and a manifest
        // pointing at nothing is worse than one that says nothing: the tool answers "no
        // manifest entry here", which is true, rather than failing mid-download.
        let tools = builtin();
        assert!(find(&tools, "rustc_codegen_tile", "linux", "x86_64").is_none());
        assert!(find(&tools, "rustc_codegen_tile", "linux", "aarch64").is_none());
    }

    #[test]
    fn an_unknown_key_is_an_error_not_something_to_skip() {
        let e = parse("[[tool]]\nid = \"x\"\nshaa256 = \"deadbeef\"\n").unwrap_err();
        assert!(e.to_string().contains("shaa256"), "{e}");
    }

    #[test]
    fn an_unquoted_value_is_refused() {
        let e = parse("[[tool]]\nid = x\n").unwrap_err();
        assert!(e.to_string().contains("quoted"), "{e}");
    }

    #[test]
    fn a_key_outside_a_table_is_refused() {
        let e = parse("id = \"x\"\n").unwrap_err();
        assert!(e.to_string().contains("outside"), "{e}");
    }

    #[test]
    fn a_table_missing_a_required_key_is_refused() {
        let e = parse("[[tool]]\nid = \"x\"\n").unwrap_err();
        assert!(e.to_string().contains("no version"), "{e}");
    }

    #[test]
    fn lookup_is_by_id_and_platform() {
        let tools = builtin();
        assert!(find(&tools, "rustc_codegen_tile", "macos", "aarch64").is_some());
        assert!(find(&tools, "rustc_codegen_tile", "macos", "riscv64").is_none());
        assert!(find(&tools, "nonesuch", "macos", "aarch64").is_none());
    }
}
