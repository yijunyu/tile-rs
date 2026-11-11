//! Byte-exact golden files for every emitter.
//!
//! These do two jobs at once, and the second is the one that is easy to overlook.
//!
//! 1. **Correctness.** Emitting the canonical softmax lowering produces exactly the
//!    bytes checked in beside this test. The emitters are pure functions of their input
//!    — the same contract `tile_spec`'s `backend_emit_purity.feature` states — so this
//!    is a legitimate thing to freeze rather than a brittle snapshot.
//!
//! 2. **Drift alarm.** `tile_cli` is defined by files it does NOT own:
//!    `crates/rustc_codegen_tile/src/mlir_to_*.rs` are `#[path]`-included, so there is
//!    no version boundary between them and this crate, and they change on their own
//!    cadence. Without these files, an emitter change would silently change what `tile`
//!    produces and nothing anywhere would notice.
//!
//! To re-bless after an intentional emitter change:
//!
//! ```text
//! TILE_BLESS=1 cargo test --test golden
//! ```
//!
//! and read the diff in the commit. A re-bless that nobody reads is the same as not
//! having the test.

use std::path::{Path, PathBuf};
use tile_cli::{emit, forms};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/golden")
}

fn canonical_mlir() -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms/softmax.mlir");
    std::fs::read_to_string(p).expect("the canonical snippet")
}

/// Every kernel the goldens freeze, by the stem its files are named after.
///
/// It was softmax alone, which exercises one shape. The REDUCTION shape is the one that
/// shipped a miscompile -- `simd_max` across 32 lanes presented as the whole row's -- and
/// it is only pinned here if a kernel of that shape is frozen too. `reduce_sum` is that
/// kernel: a dropped barrier or a lost cross-group fold shows up as a diff in its text,
/// rather than as a wrong answer on hardware nobody has.
fn canonical_kernels() -> Vec<(&'static str, String)> {
    ["softmax", "reduce_sum"]
        .into_iter()
        .map(|stem| {
            let p =
                Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("testdata/forms/{stem}.mlir"));
            let text =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            (stem, text)
        })
        .collect()
}

fn blessing() -> bool {
    std::env::var("TILE_BLESS").is_ok_and(|v| v == "1")
}

/// Every form that has an emitter, and the file its output is frozen in.
fn emittable() -> Vec<&'static forms::Form> {
    forms::FORMS
        .iter()
        .filter(|f| emit::emitter_exists(f.id))
        .collect()
}

#[test]
fn every_emitter_matches_its_golden_file() {
    if !cfg!(feature = "emitters") {
        // A build without the emitters cannot check them. It must still SAY so rather
        // than reporting a green run that checked nothing.
        eprintln!("skipped: built without --features emitters");
        return;
    }
    let mut blessed = Vec::new();
    let mut failures = Vec::new();

    for (stem, mlir) in canonical_kernels() {
        for f in emittable() {
            let path = golden_dir().join(format!("{stem}.{}.{}", f.id, f.primary_ext()));
            let got = match emit::emit(f, &mlir) {
                Ok(s) => s,
                Err(e) => {
                    // A backend that never lowered this kernel has no golden, and that is
                    // not a failure -- reduce_sum is an honest gap in most of them. A
                    // backend that HAS a golden and now refuses has regressed, and without
                    // this arm its file would simply stop being checked with nobody seeing
                    // it go.
                    if path.exists() {
                        failures.push(format!(
                            "{}/{stem}: has a frozen golden and now refuses: {e}",
                            f.id
                        ));
                    }
                    continue;
                }
            };
            if blessing() || !path.exists() {
                std::fs::create_dir_all(golden_dir()).unwrap();
                std::fs::write(&path, &got).unwrap();
                blessed.push(format!("{stem}.{}", f.id));
                continue;
            }
            let want = std::fs::read_to_string(&path).unwrap();
            if want != got {
                failures.push(format!(
                    "{}/{stem}: output differs from {} ({} bytes now, {} frozen).\n  \
                     If an emitter changed on purpose: TILE_BLESS=1 cargo test --test golden",
                    f.id,
                    path.file_name().unwrap().to_string_lossy(),
                    got.len(),
                    want.len()
                ));
            }
        }
    }

    if !blessed.is_empty() {
        eprintln!(
            "blessed {} golden file(s): {}",
            blessed.len(),
            blessed.join(", ")
        );
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn emitting_twice_is_byte_identical() {
    if !cfg!(feature = "emitters") {
        return;
    }
    // The purity contract, checked here as well as in tile_spec — because -O0/-O1/-O2
    // promise byte-identical output to USERS, and that promise is only as good as the
    // emitter underneath it.
    let mlir = canonical_mlir();
    for f in emittable() {
        let a = emit::emit(f, &mlir);
        let b = emit::emit(f, &mlir);
        assert_eq!(a, b, "{} is not a pure function of its input", f.id);
    }
}

#[test]
fn every_emitted_form_carries_its_target_idiom() {
    if !cfg!(feature = "emitters") {
        return;
    }
    // Guards against a golden file that is frozen but wrong: the output must still look
    // like the language it claims to be. Same idiom table as
    // tile_spec/features/backend_emit_purity.feature.
    const IDIOM: &[(&str, &str)] = &[
        ("gpu", "__global__"),
        ("musa", "musa_runtime.h"),
        ("spirv", "layout(set = 0"),
        ("msl", "kernel void"),
        ("nki", "@nki.jit"),
        ("aie", "from aie.iron"),
        ("bang", "__mlu_entry__"),
        ("gaudi", "tpc-clang"),
        ("tpu", "pallas"),
        ("csl", "comptime"),
        ("hexagon", "hvx_"),
        ("ttmetal", "void MAIN"),
        ("linalg", "linalg."),
        ("rvv", "riscv64"),
        ("pto", "module"),
    ];
    let mlir = canonical_mlir();
    for (id, idiom) in IDIOM {
        let f = forms::by_id(id).expect("form");
        let out = emit::emit(f, &mlir).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(
            out.contains(idiom),
            "{id} output does not contain {idiom:?}"
        );
    }
}

#[test]
fn a_golden_file_exists_for_every_emitter() {
    // A new backend cannot land without freezing its output — otherwise the drift alarm
    // has a hole exactly where the newest, least-exercised code is.
    if !cfg!(feature = "emitters") {
        return;
    }
    for f in emittable() {
        let path = golden_dir().join(format!("softmax.{}.{}", f.id, f.primary_ext()));
        assert!(
            path.exists(),
            "no golden file for {} at {}",
            f.id,
            path.display()
        );
    }
}

#[test]
fn an_empty_module_is_refused_by_every_emitter() {
    for f in emittable() {
        assert!(
            emit::emit(f, "").is_err(),
            "{} accepted an empty module",
            f.id
        );
    }
}

// ── Provenance of the emitters this crate does not own ────────────────────────────
//
// The drift alarm above has a blind spot that only shows up with an OUT-OF-TREE emitter.
// `mlir_to_*.rs` live in this repository, so git records which version a golden file was
// blessed against. The PICO emitter does not: it is `#[path]`-included from a sibling
// checkout (`../tile-rs-pico/build/mlir_to_pico.rs`), at whatever revision happens to be
// on this disk. Nothing pins it, so a green `--features pico` run means only "green
// against the emitter that was there", and it does not say which one that was.
//
// This bit me in the honest direction, which is the only reason I noticed. The PICO
// session reported that the manifest had gained a `backend_obligations` field and that my
// golden must therefore be stale -- and my suite was green. Both facts were true. The test
// could not tell the difference between "blessed against the current emitter" and
// "blessed against an older copy sitting in the sibling directory".
//
// So record the digest. A changed emitter now fails with its own cause printed, instead
// of a diff in generated assembly that the reader has to reverse-engineer.
//
// What the digest does NOT tell you, and why the re-bless instructions are specific:
// `build/mlir_to_pico.rs` is `build/`-gitignored in tile-rs-pico and tracked by no commit
// there (verified: `git ls-files --error-unmatch` does not know it). It is WRITTEN by
// `scripts/vendor_to_tile_rs.sh`. So `git pull` cannot update it, a branch merge cannot
// update it, and a digest of it cannot be mapped back to a revision by this crate. My
// first version of this comment claimed the recorded digest "= f491282"; that was an
// inference from the checkout's HEAD, and the file's contents had nothing to do with it.
// The PICO side now stamps a `source revision:` line into the banner for exactly this
// reason, so a copy generated after their c756f5d says what it came from -- read the
// banner, do not infer from the surrounding checkout.
#[test]
#[cfg(feature = "pico")]
fn the_out_of_tree_pico_emitter_is_the_one_the_golden_was_blessed_against() {
    let src =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tile-rs-pico/build/mlir_to_pico.rs");
    let Ok(got) = tile_cli::sha256::hex_file(&src) else {
        // Built with `--features pico` but the sibling is gone: the include would not
        // have compiled, so this is unreachable in practice. Say so rather than pass.
        panic!("pico emitter not readable at {}", src.display());
    };
    let stamp = golden_dir().join("pico-emitter.sha256");
    if blessing() || !stamp.exists() {
        std::fs::write(&stamp, format!("{got}\n")).expect("write provenance stamp");
        return;
    }
    let want = std::fs::read_to_string(&stamp).expect("read provenance stamp");
    assert_eq!(
        got,
        want.trim(),
        "\nThe out-of-tree PICO emitter has changed.\n\
         \n  at:     {}\n  blessed: {}\n  now:     {got}\n\
         \nThis is not necessarily a bug -- tile-rs-pico moves on its own cadence.\n\
         \nThe file is generated and gitignored there, so it does not travel with a pull\n\
         or a merge: regenerate it with `scripts/vendor_to_tile_rs.sh` (no argument) run\n\
         in the tile-rs-pico checkout. Then read its banner -- a copy generated after\n\
         their c756f5d carries a `source revision:` line naming what it was built from --\n\
         and re-bless BOTH this stamp and softmax.pico.pico.s with TILE_BLESS=1, from\n\
         THIS crate's testdata/forms/softmax.mlir rather than from the PICO fixture, so\n\
         the golden keeps its own kernel name.\n",
        src.display(),
        want.trim(),
    );
}
