//! ISSTA campaign — fuzzing as a robustness oracle: `emit` must never panic.
//!
//! `emit::emit(form, mlir)` is the one door every conversion walks through, and its
//! contract is that an input the tool cannot lower comes back as `EmitError::Rejected` —
//! not as a panic. A panic here is a compiler crash on adversarial or merely truncated
//! IR: the CLI dies with a backtrace instead of naming what it could not read.
//!
//! The oracle is deliberately trivial (no panic = pass), which is what makes it able to
//! find anything: it needs no model of what the emitters SHOULD produce, only the
//! guarantee they must not die. Every family below is either derived from the real
//! corpus (truncation, deletion, delimiter-adjacent edits — the shape of files that
//! arrive half-written) or from a failure class already paid for in this tree's history
//! (unbalanced slices in `mlir_parse`, extents that overflow, CRLF line endings).
//!
//! Findings become regression cases in `issta_fuzz_malformed_snippets_never_panic`,
//! which pins the minimized inputs so the campaign does not have to re-discover them.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;
use tile_cli::{emit, forms};

/// The corpus: real modules, pretty and generic, small enough to mutate exhaustively.
fn corpus() -> Vec<(&'static str, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms");
    let mut out = Vec::new();
    for stem in ["softmax", "reduce_sum", "softmax_linalg", "softmax_pto"] {
        let p = dir.join(format!("{stem}.mlir"));
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        out.push((stem, text));
    }
    out
}

/// Forms that HAVE an emitter in this build. `emit` returns `NoEmitter` for the rest,
/// which proves nothing about robustness.
///
/// Emptiness is only a defect when the build ASKED for emitters: `--no-default-features`
/// (the lean-release artifact, built by CI's no-emitters job) legitimately has none, and
/// the oracles below simply have nothing to drive there.
fn emitter_forms() -> Vec<&'static forms::Form> {
    let list: Vec<&'static forms::Form> = forms::FORMS
        .iter()
        .filter(|f| emit::have_emitter(f.id))
        .collect();
    if cfg!(feature = "emitters") {
        assert!(
            !list.is_empty(),
            "emitters feature is on but no form has one — the fuzz oracle would pass vacuously"
        );
    }
    list
}

/// Deterministic stride sampling: cover the whole input, bounded per family.
fn stride(len: usize, budget: usize) -> usize {
    (len / budget).max(1)
}

/// Corpus-derived mutation families. Labels are stable so a finding can be minimized
/// into a snippet without re-running the campaign.
fn corpus_mutations(src: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();

    // 1. Truncation at char boundaries — the half-written / cat-interrupted file.
    let s: Vec<usize> = src
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(src.len()))
        .collect();
    for i in s.iter().step_by(stride(src.chars().count(), 96).max(1)) {
        out.push((format!("trunc@{i}"), src[..*i].to_string()));
    }

    // 2. Single-char deletion — the off-by-one in what arrived.
    for i in s.iter().step_by(stride(src.chars().count(), 96).max(1)) {
        let mut v = String::new();
        v.push_str(&src[..*i]);
        let rest = &src[*i..];
        // drop the char at i (not the byte — keep UTF-8 valid, this is deletion not corruption)
        let mut r = rest.chars();
        r.next();
        v.push_str(r.as_str());
        out.push((format!("del@{i}"), v));
    }

    // 3. Multibyte insertion at delimiter-adjacent positions. The historical panic class
    //    in this tree was byte-index slicing; CJK beside delimiters is what a byte index
    //    that a char boundary disagrees with looks like.
    for (i, c) in src.char_indices() {
        if matches!(c, '(' | ')' | '[' | ']' | '{' | '}') {
            let mut v = String::new();
            v.push_str(&src[..i]);
            v.push_str("中文");
            v.push_str(&src[i..]);
            out.push((format!("cjk@{i}"), v));
            let mut v2 = String::new();
            v2.push_str(&src[..i]);
            v2.push('é');
            v2.push_str(&src[i..]);
            out.push((format!("accent@{i}"), v2));
        }
    }

    // 4. Delimiter storms appended — depth and imbalance.
    for (label, tail) in [
        ("storm-o", "((((((((("),
        ("storm-c", "))))))))))"),
        ("storm-os", "[[[[[[[[[["),
        ("storm-cs", "]]]]]]]]]]"),
        ("storm-ob", "{{{{{{{{{{"),
        ("storm-cb", "}}}}}}}}}}"),
        ("storm-mix", "([{})"),
        ("crlf", ""),
    ] {
        if label == "crlf" {
            out.push((label.to_string(), src.replace('\n', "\r\n")));
        } else {
            out.push((label.to_string(), format!("{src}{tail}")));
        }
    }

    // 5. Extent corruption — the extents a caller controls. `u32::MAX + 1` and a value
    //    that cannot fit in u32 at all are the two an unguarded `parse().unwrap()` eats.
    for bad in [
        "0",
        "-1",
        "4294967295",
        "4294967296",
        "18446744073709551616",
        "99999999999999999999999999",
    ] {
        let digits: Vec<usize> = src
            .match_indices(|c: char| c.is_ascii_digit())
            .map(|(i, _)| i)
            .collect();
        if digits.is_empty() {
            continue;
        }
        let st = stride(digits.len(), 24).max(1);
        // group contiguous digit runs, then rewrite sampled runs
        let mut runs: Vec<(usize, usize)> = Vec::new(); // (start, end) byte range
        for i in &digits {
            match runs.last_mut() {
                Some(last) if last.1 == *i => last.1 = i + 1,
                _ => runs.push((*i, i + 1)),
            }
        }
        for (a, b) in runs.iter().step_by(st) {
            let mut v = String::new();
            v.push_str(&src[..*a]);
            v.push_str(bad);
            v.push_str(&src[*b..]);
            out.push((format!("extent-{bad}@{a}"), v));
        }
    }

    out
}

/// Hand-written malformed snippets. Each family is a shape a real session produced at
/// least once; any that has ever panicked goes here in minimized form.
fn malformed_snippets() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = [
        ("empty", String::new()),
        ("ws", "   \n\t\n".to_string()),
        ("only-o", "((((((((".to_string()),
        ("only-c", "))))))))".to_string()),
        ("only-brace", "{{{{".to_string()),
        ("module-open", "module {".to_string()),
        ("func-partial", "func.func @f(".to_string()),
        ("func-name-only", "func.func".to_string()),
        ("call-no-args", "%r = call @f() : () -> i32".to_string()),
        (
            "const-overflow",
            "arith.constant 99999999999999999999999 : i32".to_string(),
        ),
        ("const-neg-ext", "arith.constant -1 : index".to_string()),
        ("unterminated-string", "func.func @f() { \"abc".to_string()),
        ("nul", "func\0.func @f() { }".to_string()),
        ("cjk-ident", "func.func @中文() { }".to_string()),
        ("crlf-only", "\r\n\r\n".to_string()),
        ("loc-at-eol", "func.func @f() {\n  loc(unknown)".to_string()),
        (
            "dollar-no-def",
            "%undefined = arith.addi %a, %b : i32".to_string(),
        ),
        (
            "nested-deep",
            format!("({}{})", "(".repeat(4000), ")".repeat(4000)),
        ),
        ("nested-mlir", format!("module {}", "{".repeat(500))),
        (
            "huge-shape",
            "memref<4294967296x4294967296xf32>".to_string(),
        ),
        ("unknown-dialect", "#unk.unknown<garbage> = ()".to_string()),
        (
            "binaryish",
            String::from_utf8_lossy(&[0xFF, 0xFE, 0x00, 0x80, 0x28, 0x29]).into_owned(),
        ),
    ]
    .iter()
    .map(|(l, s)| (l.to_string(), s.clone()))
    .collect();

    // A real corpus file with its braces unbalanced at both ends.
    let (name, src) = &corpus()[0];
    out.push((
        format!("{name}-open-only"),
        src.replace('}', "").to_string(),
    ));
    out.push((
        format!("{name}-close-only"),
        src.replace('{', "").to_string(),
    ));

    // Regression: the u32::MAX extent class found by this campaign (three separate
    // panics — aie's MemTile staging multiply, pto's rowreduce pad-up, bang's
    // rows × cols buffer size). One digit run in a real module is all it took;
    // these two mutations are the exact found inputs, minimized to their single
    // edit. Each must come back as EmitError::Rejected, never a panic.
    let load_cols_max = src.replace("%c1, %c1024)", "%c1, %c4294967295)");
    let softmax_rows_max = src.replace(
        "__tile_softmax_f32(%c0, %t0, %c1,",
        "__tile_softmax_f32(%c0, %t0, %c4294967295,",
    );
    // If the corpus drifts so the edits no longer land, this test must fail rather
    // than silently stop pinning the finding.
    assert_ne!(
        load_cols_max, *src,
        "corpus drift: the extent-u32max load edit no longer lands — repin the finding"
    );
    assert_ne!(
        softmax_rows_max, *src,
        "corpus drift: the extent-u32max softmax edit no longer lands — repin the finding"
    );
    out.push(("extent-u32max-load-cols".to_string(), load_cols_max));
    out.push(("extent-u32max-softmax-rows".to_string(), softmax_rows_max));
    out
}

/// A quieted-but-visible panic hook: each caught panic prints ONE line (location +
/// message) instead of a backtrace, so triage sees where it died. With the harness's
/// captured output, the lines surface in the failure report.
fn quiet_panics() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| std::panic::set_hook(Box::new(|info| eprintln!("{info}"))));
}

fn run_family(label: &str, inputs: &[(String, String)]) {
    quiet_panics();
    let forms_list = emitter_forms();
    if forms_list.is_empty() {
        // No emitters in this build (the --no-default-features CI job): nothing to
        // fuzz, and the vacuity guard above already checked this was legitimate.
        return;
    }
    assert!(!inputs.is_empty(), "empty input family — nothing fuzzed");
    let mut findings: Vec<String> = Vec::new();
    for (mname, mlir) in inputs {
        for f in &forms_list {
            let f_id = f.id;
            let m = mlir.clone();
            let res = catch_unwind(AssertUnwindSafe(|| emit::emit(f, &m)));
            if res.is_err() {
                // char-boundary-safe preview: byte slicing at 400 could itself panic.
                let preview: String = m.chars().take(400).collect();
                findings.push(format!(
                    "{label} / {mname} / form={f_id}: PANIC\n--- input (first 400 chars) ---\n{preview}"
                ));
            }
        }
    }
    if !findings.is_empty() {
        panic!(
            "{} panicking input(s) — an emitter must reject, not die:\n\n{}",
            findings.len(),
            findings.join("\n\n")
        );
    }
}

/// Corpus mutations: truncation, deletion, delimiter-adjacent multibyte inserts,
/// delimiter storms, CRLF, corrupted extents — across every emitter.
#[test]
fn issta_fuzz_corpus_mutations_never_panic() {
    for (name, src) in corpus() {
        let muts = corpus_mutations(&src);
        run_family(name, &muts);
    }
}

/// The u32::MAX extent findings must be REFUSED, not merely panic-free.
///
/// The fuzz oracle only proves "no panic"; a fix that instead emitted a wrapped or
/// defaulted geometry would still pass it. This test pins the stronger contract for
/// the three (form, input) pairs the campaign actually found: `emit` must return an
/// error, because an extent of 4294967295 elements cannot be lowered to any of them.
#[test]
fn issta_regression_u32max_extents_are_refused() {
    quiet_panics();
    let snippets = malformed_snippets();
    let expectations = [
        ("extent-u32max-load-cols", "aie"),
        ("extent-u32max-softmax-rows", "pto"),
        ("extent-u32max-softmax-rows", "bang"),
    ];
    for (label, fid) in expectations {
        let mlir = &snippets
            .iter()
            .find(|(l, _)| l == label)
            .unwrap_or_else(|| panic!("regression snippet {label} missing"))
            .1;
        let f = forms::FORMS
            .iter()
            .find(|f| f.id == fid)
            .unwrap_or_else(|| panic!("form {fid} missing"));
        if !emit::have_emitter(fid) {
            // A build without emitters refuses everything with NoEmitter; the claim
            // under test only exists in a build that compiles this emitter in.
            if cfg!(feature = "emitters") {
                panic!(
                    "form {fid} missing from an emitters build — the refusal claim is untestable"
                );
            }
            continue;
        }
        if let Ok(out) = emit::emit(f, mlir) {
            let preview: String = out.chars().take(300).collect();
            panic!(
                "form={fid} ACCEPTED {label} — an extent of 4294967295 must be refused, \
                 not emitted:\n{preview}"
            );
        }
    }
}

/// The minimized malformed snippets. This is the regression list: when the campaign
/// finds a panic, its minimized input lands here and this test owns it forever.
#[test]
fn issta_fuzz_malformed_snippets_never_panic() {
    run_family("snippet", &malformed_snippets());
}
