//! ISSTA campaign — metamorphic testing: surface syntax is not semantics.
//!
//! The emitters are pure functions of their input (`tile_spec`'s
//! `backend_emit_purity.feature` states this), so transformations that cannot change what
//! a module MEANS must not change what `emit` produces. Where there is no oracle for the
//! output — and for a 15-backend lowering there is no oracle short of running every
//! device — the relation between two runs IS the oracle.
//!
//! Three relations, all cheap enough to hold across every emitter:
//!
//! 1. **Determinism.** `emit(a) == emit(a)` byte-for-byte, including the error path:
//!    the same input must be refused with the same message, not a different one per call.
//! 2. **Trailing whitespace.** A file that ends in extra spaces/newlines is the same file;
//!    a line-based reader that sees a phantom empty line and changes output is reading
//!    the wrong thing.
//! 3. **Comments.** `// ...` lines carry no semantics in MLIR. An emitter whose output
//!    depends on them is either copying source text into the kernel (a leak) or losing
//!    track of line structure.
//!
//! A violation is reported with the transform and form so it can be minimized; unlike a
//! fuzz finding it says nothing yet about which side is wrong — that judgment happens
//! when the case is read, the way `dtype_is_not_ignored`'s allowlist records the
//! mechanism that makes an identical pair correct.

use tile_cli::{emit, forms};

fn corpus() -> Vec<(&'static str, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms");
    [("softmax", "softmax"), ("reduce_sum", "reduce_sum")]
        .iter()
        .map(|(_, stem)| {
            let p = dir.join(format!("{stem}.mlir"));
            let text =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            (*stem, text)
        })
        .collect()
}

fn emitter_forms() -> Vec<&'static forms::Form> {
    let list: Vec<&'static forms::Form> = forms::FORMS
        .iter()
        .filter(|f| emit::have_emitter(f.id))
        .collect();
    // Vacuity guard: without emitters every relation below holds trivially — but only
    // a build that ASKED for emitters (default features) can be wrong about it; the
    // --no-default-features CI job legitimately has none.
    if cfg!(feature = "emitters") {
        assert!(
            !list.is_empty(),
            "emitters feature is on but no form has one — the oracles would hold vacuously"
        );
    }
    list
}

/// Determinism across calls: same bytes out, same error out.
#[test]
fn issta_metamorphic_emit_is_deterministic() {
    for (name, src) in corpus() {
        for f in emitter_forms() {
            let a = emit::emit(f, &src);
            let b = emit::emit(f, &src);
            match (&a, &b) {
                (Ok(x), Ok(y)) => assert!(
                    x == y,
                    "form={} kernel={name}: second emit differs\n--- first (200) ---\n{}\n--- second (200) ---\n{}",
                    f.id,
                    &x[..x.len().min(200)],
                    &y[..y.len().min(200)]
                ),
                (Err(x), Err(y)) => assert!(
                    x.to_string() == y.to_string(),
                    "form={} kernel={name}: refusal is not stable: {x:?} vs {y:?}",
                    f.id
                ),
                _ => panic!(
                    "form={} kernel={name}: first {a:?}, second {b:?} — emit is not a pure function",
                    f.id
                ),
            }
        }
    }
}

/// Trailing whitespace is not input.
#[test]
fn issta_metamorphic_trailing_whitespace_is_invisible() {
    for (name, src) in corpus() {
        let dirty = format!("{src}   \n\n\t\n  \n");
        for f in emitter_forms() {
            let clean = emit::emit(f, &src);
            let noisy = emit::emit(f, &dirty);
            compare(f.id, name, "trailing whitespace", &clean, &noisy);
        }
    }
}

/// Comment lines are not input.
#[test]
fn issta_metamorphic_comments_are_invisible() {
    for (name, src) in corpus() {
        // A comment line before every line: maximal positions, no structural change.
        let commented: String = src
            .lines()
            .map(|l| format!("// issta: surface only\n{l}"))
            .collect::<Vec<_>>()
            .join("\n");
        for f in emitter_forms() {
            let clean = emit::emit(f, &src);
            let noisy = emit::emit(f, &commented);
            compare(f.id, name, "comment lines", &clean, &noisy);
        }
    }
}

fn compare(
    form: &str,
    kernel: &str,
    transform: &str,
    a: &Result<String, impl std::fmt::Debug>,
    b: &Result<String, impl std::fmt::Debug>,
) {
    match (a, b) {
        (Ok(x), Ok(y)) => assert!(
            x == y,
            "form={form} kernel={kernel} transform={transform}: output changed\n--- clean (300) ---\n{}\n--- transformed (300) ---\n{}",
            &x[..x.len().min(300)],
            &y[..y.len().min(300)]
        ),
        (Err(x), Err(y)) => assert!(
            format!("{x:?}") == format!("{y:?}"),
            "form={form} kernel={kernel} transform={transform}: refusal changed: {x:?} vs {y:?}"
        ),
        _ => panic!(
            "form={form} kernel={kernel} transform={transform}: clean={a:?} transformed={b:?}"
        ),
    }
}
