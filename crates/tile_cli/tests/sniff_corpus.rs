//! Every form in the table has a specimen on disk, and every specimen resolves to the
//! form it is filed under.
//!
//! This is the test that makes the taxonomy real. Without a corpus the sniffer is
//! validated on whatever file the author happened to have, on one machine — and the
//! four overloaded extensions (`.py` x3, `.mlir` x4, `.c` x2, `.cpp` x2) are precisely
//! where that goes wrong.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tile_cli::forms::{self, How};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/forms")
}
fn negative_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/negative")
}

/// Which form each specimen is filed under. The map is written out longhand on purpose:
/// a specimen that silently changes form must fail here, not surprise someone later.
const EXPECTED: &[(&str, &str)] = &[
    ("softmax.rs", "tile"),
    ("softmax.mlir", "mlir"),
    ("softmax_linalg.mlir", "linalg"),
    ("softmax_rvv.mlir", "rvv"),
    ("softmax_pto.mlir", "pto"),
    ("softmax.metal", "msl"),
    ("softmax.cu", "gpu"),
    ("softmax.mu", "musa"),
    ("softmax.comp", "spirv"),
    ("softmax_nki.py", "nki"),
    ("softmax_aie.py", "aie"),
    ("softmax_tpu.py", "tpu"),
    ("softmax.mlu", "bang"),
    ("softmax_gaudi.c", "gaudi"),
    ("softmax_hexagon.c", "hexagon"),
    ("softmax_ttmetal.cpp", "ttmetal"),
    ("softmax.cce", "cpp"),
    ("softmax.csl", "csl"),
    ("softmax.pico.s", "pico"),
    ("softmax.mlir.txt", "debug"),
];

#[test]
fn every_specimen_resolves_to_its_filed_form() {
    for (file, want) in EXPECTED {
        let path = corpus_dir().join(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("corpus file {file} unreadable: {e}"));
        let r = forms::resolve_input(path.to_str().unwrap(), &text, None)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(
            r.form.id, *want,
            "{file} resolved to {} but is filed as {want}",
            r.form.id
        );
    }
}

#[test]
fn every_form_in_the_table_has_a_specimen() {
    // A 17th backend must not be able to land without one, or the sniffer for it is
    // never exercised.
    let covered: BTreeSet<&str> = EXPECTED.iter().map(|(_, f)| *f).collect();
    let all: BTreeSet<&str> = forms::FORMS.iter().map(|f| f.id).collect();
    let missing: Vec<&&str> = all.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "forms with no corpus specimen: {missing:?}"
    );
}

#[test]
fn the_shared_extensions_are_told_apart_by_content_alone() {
    // The whole reason the table carries magic. Group the specimens by extension and
    // assert that each overloaded one really does produce distinct forms.
    for (ext, files) in [
        (
            "py",
            vec!["softmax_nki.py", "softmax_aie.py", "softmax_tpu.py"],
        ),
        (
            "mlir",
            vec![
                "softmax.mlir",
                "softmax_linalg.mlir",
                "softmax_rvv.mlir",
                "softmax_pto.mlir",
            ],
        ),
        ("c", vec!["softmax_gaudi.c", "softmax_hexagon.c"]),
        ("cpp", vec!["softmax_ttmetal.cpp"]),
    ] {
        let mut seen = BTreeSet::new();
        for f in &files {
            let path = corpus_dir().join(f);
            let text = std::fs::read_to_string(&path).unwrap();
            let r = forms::resolve_input(path.to_str().unwrap(), &text, None).unwrap();
            assert!(
                seen.insert(r.form.id),
                ".{ext}: {f} collided with an earlier specimen on form {}",
                r.form.id
            );
            // The one exception is a form with no magic of its own: `mlir` is the
            // documented fallback for `.mlir`, which is what lets a plain module
            // resolve while an unmarked `.py` correctly refuses.
            let expect_magic = !r.form.magic.is_empty();
            if expect_magic {
                assert_eq!(
                    r.how,
                    How::Magic,
                    ".{ext}: {f} should be decided by content"
                );
            } else {
                assert_eq!(
                    r.how,
                    How::Extension,
                    ".{ext}: {f} is the extension fallback"
                );
            }
        }
    }
}

#[test]
fn rs_is_reserved_and_never_sniffed() {
    let path = corpus_dir().join("softmax.rs");
    let text = std::fs::read_to_string(&path).unwrap();
    // The specimen deliberately contains "__global__" and "kernel void" in a comment.
    assert!(
        text.contains("__global__"),
        "the corpus file must keep this trap"
    );
    assert!(
        text.contains("kernel void"),
        "the corpus file must keep this trap"
    );
    let r = forms::resolve_input(path.to_str().unwrap(), &text, None).unwrap();
    assert_eq!(r.form.id, "tile");
    assert_eq!(r.how, How::Reserved);
}

#[test]
fn an_unmarked_py_is_refused_with_its_three_candidates_named() {
    let path = negative_dir().join("plain.py");
    let text = std::fs::read_to_string(&path).unwrap();
    let e = forms::resolve_input(path.to_str().unwrap(), &text, None).unwrap_err();
    let msg = e.to_string();
    for want in ["nki", "aie", "tpu", "-f"] {
        assert!(msg.contains(want), "refusal must name {want:?}: {msg}");
    }
}

#[test]
fn an_unmarked_cpp_is_refused_rather_than_guessed() {
    let path = negative_dir().join("ambiguous.cpp");
    let text = std::fs::read_to_string(&path).unwrap();
    let e = forms::resolve_input(path.to_str().unwrap(), &text, None).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("ttmetal") && msg.contains("cpp"), "{msg}");
}

#[test]
fn a_non_utf8_input_is_not_even_read_as_text() {
    let path = negative_dir().join("binary.mlir");
    let bytes = std::fs::read(&path).unwrap();
    assert!(
        String::from_utf8(bytes).is_err(),
        "the trap file must stay non-UTF8"
    );
}

#[test]
fn forcing_a_form_overrides_content_but_records_the_contradiction() {
    let path = corpus_dir().join("softmax.metal");
    let text = std::fs::read_to_string(&path).unwrap();
    let r = forms::resolve_input(path.to_str().unwrap(), &text, Some("gpu")).unwrap();
    assert_eq!(r.form.id, "gpu");
    assert_eq!(r.how, How::Forced);
    assert_eq!(
        r.contradicted_by.map(|f| f.id),
        Some("msl"),
        "a forced form that contradicts strong magic must be warned about"
    );
}

#[test]
fn an_output_extension_shared_by_several_forms_demands_dash_t() {
    let default = forms::by_id("linalg").unwrap();
    let e = forms::resolve_output(Some("out.py"), None, default).unwrap_err();
    let msg = e.to_string();
    assert!(
        msg.contains("nki") && msg.contains("aie") && msg.contains("tpu"),
        "{msg}"
    );
    // ...and -t settles it.
    let f = forms::resolve_output(Some("out.py"), Some("nki"), default).unwrap();
    assert_eq!(f.id, "nki");
}

#[test]
fn dash_o_wins_over_any_extension_inference() {
    let default = forms::by_id("linalg").unwrap();
    let f = forms::resolve_output(Some("out.txt"), Some("msl"), default).unwrap();
    assert_eq!(f.id, "msl", "-t must beat the .txt extension");
}

#[test]
fn a_refinement_beats_the_form_it_refines() {
    // An rvv module carries linalg's marker as well as its own triple stamp. Without
    // the refinement rule the general form wins by table order and rvv is unreachable —
    // which is exactly what this corpus caught the first time it ran.
    let path = corpus_dir().join("softmax_rvv.mlir");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("linalg."),
        "the trap must stay: rvv output IS linalg output"
    );
    assert!(text.contains("riscv64"));
    let r = forms::resolve_input(path.to_str().unwrap(), &text, None).unwrap();
    assert_eq!(r.form.id, "rvv");
    assert_eq!(forms::by_id("rvv").unwrap().refines, Some("linalg"));
}
