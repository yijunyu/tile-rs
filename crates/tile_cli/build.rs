//! The per-backend intrinsic surface, DERIVED FROM THE TREE at build time.
//!
//! `--list-ops` probes 32 portable ops by lowering a real module for each, which is the
//! stronger claim and the smaller question. This table answers the other one: of the
//! intrinsics `tile_std` actually declares -- 509 of them, against that probe's 32 --
//! which ones does a given backend's emitter have an arm for?
//!
//! It is derived rather than written down for the reason the coverage report is measured
//! rather than tabulated: a hand-maintained list of what `pto` can do is exactly the
//! artefact that sent a reader to four greps across three repositories and a wrong
//! conclusion. Re-derived on every build, so it cannot drift from the sources it
//! describes; `cargo:rerun-if-changed` is emitted for every file read.
//!
//! What it CANNOT say is that the arm works. A name in an emitter is evidence that
//! somebody wrote a case for it, not that the case emits anything -- `SKILL.md` trap 3
//! records 42 of 42 `spirv` intrinsics emitting `p1[gid] = p0[gid]` at exit 0. The
//! report says so in its own footer, and points at the measured instruments.
//!
//! A SOURCE THAT IS NOT THERE IS UNKNOWN, NEVER ZERO. `cpp` (AscendC) is a closed
//! emitter living in a sibling checkout, and `pico`'s is too; a packaged crate has no
//! sibling crates at all. Each such backend is emitted with no source and the report
//! refuses to print a count for it, rather than scoring it 0 of 509.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Backend form id -> its emitter source, relative to `crates/`.
///
/// Mirrors `tile_codegen::emitters`' `#[path]` set plus the two PTO/GPU/linalg modules
/// declared at that crate's root. Keyed on the form id so `forms::by_id` and this table
/// cannot disagree about a name.
const EMITTERS: &[(&str, &str)] = &[
    ("aie", "rustc_codegen_tile/src/mlir_to_aie.rs"),
    ("bang", "rustc_codegen_tile/src/mlir_to_bang.rs"),
    ("csl", "rustc_codegen_tile/src/mlir_to_csl.rs"),
    ("gaudi", "rustc_codegen_tile/src/mlir_to_gaudi.rs"),
    ("gpu", "rustc_codegen_tile/src/mlir_to_gpu.rs"),
    ("hexagon", "rustc_codegen_tile/src/mlir_to_hexagon.rs"),
    ("linalg", "rustc_codegen_tile/src/mlir_to_linalg.rs"),
    ("msl", "rustc_codegen_tile/src/mlir_to_msl.rs"),
    ("musa", "rustc_codegen_tile/src/mlir_to_musa.rs"),
    ("nki", "rustc_codegen_tile/src/mlir_to_nki.rs"),
    ("pto", "rustc_codegen_tile/src/mlir_to_pto.rs"),
    ("rvv", "rustc_codegen_tile/src/mlir_to_rvv.rs"),
    ("spirv", "rustc_codegen_tile/src/mlir_to_spirv.rs"),
    ("tpu", "rustc_codegen_tile/src/mlir_to_tpu.rs"),
    ("ttmetal", "rustc_codegen_tile/src/mlir_to_ttmetal.rs"),
];

/// Where the declarations live. Both files, because `lib.rs` holds 162 of the 509 and a
/// scan of `tile.rs` alone would report a third of the surface as absent.
const DECLS: &[&str] = &["tile_std/src/lib.rs", "tile_std/src/tile.rs"];

/// Every `__tile_*` token in a source, however it is spelled.
///
/// The emitters match on the MLIR callee name, which is the intrinsic with an `@` in
/// front, and several match a FAMILY prefix and dispatch on the suffix -- `mlir_to_cpp`
/// matches `@__tile_compare_scalar_` and then six mode prefixes, so the dtype never
/// appears in the source at all. Collecting tokens rather than full names is what lets
/// the prefix rule below see those arms.
fn tokens(src: &str) -> BTreeSet<String> {
    let src = &strip_comments(src);
    let mut out = BTreeSet::new();
    let b = src.as_bytes();
    let pat = b"__tile_";
    let mut i = 0;
    while i + pat.len() <= b.len() {
        if &b[i..i + pat.len()] == pat {
            let mut j = i + pat.len();
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            out.insert(String::from_utf8_lossy(&b[i..j]).into_owned());
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Rust source with its comments removed, string literals intact.
///
/// A name in a COMMENT is not an arm, and every emitter documents the intrinsics it
/// handles in a header table -- `mlir_to_pto.rs` alone names 10 that appear nowhere in
/// its code. Crediting those would make the report claim arms that do not exist, which
/// is the mirror of the under-report above and just as bad.
///
/// String state is tracked, so a `//` inside a literal does not start a comment: the
/// emitters carry paths and MLIR fragments in strings, and truncating one would drop
/// real tokens. Nested block comments are not handled; Rust allows them and no emitter
/// uses one, so the residue is a known and empty case rather than an unknown one.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let (mut i, mut in_str, mut esc) = (0usize, false, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c as char);
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_str = true;
            out.push('"');
            i += 1;
        } else if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    out
}

/// `pub fn __tile_<name>(` -- the declaration, with the line it is on.
fn declarations(src: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let t = line.trim_start();
        // `pub unsafe fn` is a declaration too, and the first version of this scan
        // matched only `pub fn `. Measured: 22 intrinsics -- the whole `__tile_v_*`
        // family -- are declared `pub unsafe fn` in two of the three trees, so the
        // surface under-reported by 22 while claiming to be the whole of it. An
        // under-report is the one failure this command exists to remove.
        let Some(rest) = t
            .strip_prefix("pub fn __tile_")
            .or_else(|| t.strip_prefix("pub unsafe fn __tile_"))
        else {
            continue;
        };
        let name: String = std::iter::once("__tile_")
            .chain(std::iter::once(
                rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or(""),
            ))
            .collect();
        if name.len() > "__tile_".len() {
            out.push((name, n + 1));
        }
    }
    out
}

/// Does `have` cover `want` as a family prefix?
///
/// `__tile_compare_scalar_` covers `__tile_compare_scalar_eq_bf16`; `__tile_m` must not
/// cover `__tile_matmul`, or every backend "covers" everything. The rule: at least four
/// characters of family name past `__tile_`, and the suffix it would dispatch on must
/// begin at an underscore boundary, so a prefix never cuts a word in half.
fn covers(have: &str, want: &str) -> bool {
    if have == want || have.len() < "__tile_".len() + 4 || have.len() >= want.len() {
        return false;
    }
    if !want.starts_with(have) {
        return false;
    }
    have.ends_with('_') || want.as_bytes()[have.len()] == b'_'
}

/// `use crate::<mod>` in an emitter: the shared modules it dispatches through.
///
/// `mlir_to_csl` names no intrinsic at all and lowers a whole matmul family, because
/// `classify_callee` is in `mlir_emit_context`. Trimmed to the module name only, so
/// `use crate::mlir_emit_context::{classify_callee, Op}` and a bare
/// `use crate::mlir_parse;` both resolve to one name.
fn helper_mods(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let Some(rest) = line.trim_start().strip_prefix("use crate::") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// Each readable backend seeded with itself as the first element of its basis chain.
fn ids_of(sets: &BTreeMap<&'static str, BTreeSet<String>>) -> BTreeMap<&'static str, Vec<&'static str>> {
    sets.keys().map(|k| (*k, vec![*k])).collect()
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("intrinsics_table.rs");

    let mut decls: Vec<(String, String, usize)> = Vec::new(); // name, where, line
    for rel in DECLS {
        let p = root.join(rel);
        println!("cargo:rerun-if-changed={}", p.display());
        let Ok(src) = std::fs::read_to_string(&p) else {
            continue;
        };
        for (name, line) in declarations(&src) {
            decls.push((name, (*rel).to_string(), line));
        }
    }
    decls.sort();
    decls.dedup_by(|a, b| a.0 == b.0);

    // One token set per backend whose source is present. An absent source is omitted
    // entirely rather than recorded as an empty set, because an empty set reads as a
    // backend that lowers nothing.
    let mut sets: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    let mut sets_merged: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    let mut helpers: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    let mut sets_src: BTreeMap<&str, String> = BTreeMap::new();
    let mut srcs: BTreeMap<&str, &str> = BTreeMap::new();
    for (id, rel) in EMITTERS {
        let p = root.join(rel);
        println!("cargo:rerun-if-changed={}", p.display());
        if let Ok(src) = std::fs::read_to_string(&p) {
            sets.insert(id, tokens(&src));
            sets_src.insert(id, src);
            srcs.insert(id, rel);
        }
    }

    // DELEGATION AND SHARED HELPERS, because two kinds of arm live outside the
    // emitter's own file and a name-based scan reported both as absent.
    //
    //  * `mlir_to_rvv` is a thin wrapper over the linalg egress ("this backend has no
    //    translator of its own", says its own header) and `mlir_to_musa` delegates to
    //    `mlir_to_gpu` and rewrites CUDA spellings. The probe lowers 25 ops on rvv and
    //    14 on musa that neither file mentions.
    //  * `classify_callee` lives in the shared `mlir_emit_context`, so `csl`, `hexagon`
    //    and `ttmetal` dispatch a whole family from a file of their own that names no
    //    intrinsic at all -- three more under-reports, on a second tree.
    //
    // Both were caught by the cross-check in `intrinsics.rs` against the MEASURED
    // `--list-ops` probe, which is the failure a derived table has and a measured one
    // does not.
    //
    // Derived rather than declared, and the two kinds are kept apart on purpose:
    //  - a CALL to another emitter's entry point IS delegation, so it counts only when
    //    `convert_mlir_to_X(` appears -- matched with the open paren so a doc link like
    //    [`convert_mlir_to_linalg`] does not count, and iterated to a fixed point so a
    //    chain composes. `mlir_to_msl` IMPORTS `mlir_to_pto` for helpers without ever
    //    calling its entry point, and crediting msl with pto's whole vocabulary would be
    //    the over-claim the vacuity control exists to catch.
    //  - a `use crate::<mod>` where <mod> is NOT an emitter is a shared helper and is
    //    always included. Measured: those modules name 9 and 10 `__tile_*` tokens, so
    //    this adds a dispatch table, not a vocabulary.
    let mut basis: BTreeMap<&str, Vec<&str>> = ids_of(&sets);
    for _ in 0..EMITTERS.len() {
        let mut grew = false;
        let snapshot = basis.clone();
        for (id, chain) in basis.iter_mut() {
            for other in snapshot.keys() {
                if other == id || chain.contains(other) {
                    continue;
                }
                let calls = format!("convert_mlir_to_{other}(");
                if chain
                    .iter()
                    .any(|c| sets_src.get(c).is_some_and(|t: &String| t.contains(&calls)))
                {
                    chain.push(other);
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    for (id, chain) in &basis {
        let mut merged = BTreeSet::new();
        for c in chain {
            if let Some(t) = sets.get(*c) {
                merged.extend(t.iter().cloned());
            }
            // The shared helpers this file imports, read in the same pass. A module that
            // is itself an emitter is excluded here: that is the delegation case above,
            // and it needs a CALL rather than an import.
            for m in helper_mods(sets_src.get(*c).map(String::as_str).unwrap_or("")) {
                if EMITTERS.iter().any(|(e, _)| *e == m) {
                    continue;
                }
                let hp = root.join(format!("rustc_codegen_tile/src/{m}.rs"));
                if let Ok(src) = std::fs::read_to_string(&hp) {
                    merged.extend(tokens(&src));
                    helpers
                        .entry(*id)
                        .or_insert_with(BTreeSet::new)
                        .insert(format!("rustc_codegen_tile/src/{m}.rs"));
                }
            }
        }
        sets_merged.insert(*id, merged);
    }
    let sets = sets_merged;

    let ids: Vec<&str> = sets.keys().copied().collect();
    let mut s = String::new();
    s.push_str("// GENERATED by build.rs from the tree. Do not edit.\n");
    s.push_str("/// Backends whose emitter source this build could read, with the source(s)\n");
    s.push_str("/// the surface was derived from -- its own, plus any it delegates to.\n");
    s.push_str("pub const BACKENDS: &[(&str, &str)] = &[\n");
    for id in &ids {
        let mut chain: Vec<String> = basis[id].iter().map(|c| srcs[c].to_string()).collect();
        if let Some(h) = helpers.get(*id) {
            chain.extend(h.iter().cloned());
        }
        s.push_str(&format!("    ({:?}, {:?}),\n", id, chain.join(" + ")));
    }
    s.push_str("];\n");
    s.push_str("/// Every intrinsic `tile_std` declares: name, declaring file, line, and\n");
    s.push_str("/// two bitmasks over BACKENDS -- an EXACT mention, and a family-prefix one.\n");
    s.push_str("pub const INTRINSICS: &[(&str, &str, usize, u32, u32)] = &[\n");
    for (name, where_, line) in &decls {
        let mut exact = 0u32;
        let mut pref = 0u32;
        for (i, id) in ids.iter().enumerate() {
            let set = &sets[id];
            if set.contains(name) {
                exact |= 1 << i;
            } else if set.iter().any(|t| covers(t, name)) {
                pref |= 1 << i;
            }
        }
        s.push_str(&format!(
            "    ({name:?}, {where_:?}, {line}, {exact}, {pref}),\n"
        ));
    }
    s.push_str("];\n");
    std::fs::write(&out, s).expect("write intrinsics table");
}
