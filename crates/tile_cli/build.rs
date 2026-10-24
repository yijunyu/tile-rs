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

/// `pub fn __tile_<name>(` -- the declaration, with the line it is on.
fn declarations(src: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("pub fn __tile_") else {
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

    // DELEGATION, because two backends have no translator of their own and a
    // name-based scan reported their whole surface as absent. `mlir_to_rvv` is a thin
    // wrapper over the linalg egress ("this backend has no translator of its own",
    // says its own header) and `mlir_to_musa` delegates to `mlir_to_gpu` and rewrites
    // CUDA spellings. The probe lowers 25 ops on rvv and 14 on musa that neither file
    // mentions -- which the cross-check in `intrinsics.rs` caught, and which is exactly
    // the under-report a derived table is prone to and a measured one is not.
    //
    // Derived rather than declared: a call to another emitter's entry point IS the
    // delegation. Matched with the open paren so a doc link like [`convert_mlir_to_linalg`]
    // does not count as one, and iterated to a fixed point so a chain composes.
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
        let chain: Vec<&str> = basis[id].iter().map(|c| srcs[c]).collect();
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
