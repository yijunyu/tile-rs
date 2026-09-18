//! What the MLIR backends DECLARE, checked against what was asked for.
//!
//! `linalg`, `rvv` and `pto` emit MLIR and nothing here can EXECUTE them.
//!
//! This header used to say `mlir-opt` is not on this machine. That was wrong: it is at
//! `/opt/homebrew/opt/llvm/bin/mlir-opt`, absent only from `PATH`, and it parses the
//! emitted modules cleanly -- the emitted files even carry
//! `// Verify: mlir-opt --verify-diagnostics` in their own preamble. Parsing is not
//! execution, so what is below still stands on its own, but the claim that the tool was
//! missing had been sitting here unchecked while a real checker went unused.
//!
//! `emitted_parses.rs` checks that no SSA value is used before it is defined, which is
//! structure. This checks meaning.
//!
//! It is possible only because these backends NAME the operation instead of hand-rolling
//! it — `linalg.reduce … dimensions = [1]` with an `arith.addf` body, rather than a loop.
//! That is the same property that makes them unable to express the partial-reduction bug
//! that `bang` and `gaudi` both had: the width belongs to the operand, not to the code.
//!
//! The property cuts both ways, though. A named operation cannot reduce the wrong NUMBER
//! of elements, but it can perfectly well name the wrong OPERATION or the wrong DIMENSION,
//! and neither a parser nor a use-before-def check would notice:
//!
//!   * `arith.addf` where `arith.maximumf` was meant is a `reduce_max` that sums;
//!   * `dimensions = [0]` instead of `[1]` reduces ACROSS rows rather than along each —
//!     which is precisely the defect found in `bang`, arriving by a different route.
//!
//! Needs no toolchain at all.

use std::process::Command;

/// (op, the primitive its lowering must name, whether it reduces).
const EXPECTED: &[(&str, &str, bool)] = &[
    ("reduce_sum", "arith.addf", true),
    ("reduce_max", "arith.maximumf", true),
    ("absmax", "arith.maximumf", true),
    ("exp", "linalg.exp", false),
    ("log", "linalg.log", false),
    ("sqrt", "linalg.sqrt", false),
    ("tanh", "linalg.tanh", false),
    ("neg", "linalg.negf", false),
    ("abs", "linalg.abs", false),
    ("softmax", "linalg.softmax", false),
];

const SHAPES: &[(usize, usize)] = &[(1, 256), (4, 256), (3, 129), (1, 4096)];

fn kernel(op: &str, rows: usize, cols: usize, reduces: bool) -> String {
    let out_cols = if reduces { 1 } else { cols };
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %o = llvm.mlir.constant({out_cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %o) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

#[test]
fn mlir_backends_declare_the_operation_that_was_asked_for() {
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-declared-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for form in ["linalg", "rvv"] {
        for (op, primitive, reduces) in EXPECTED {
            for &(rows, cols) in SHAPES {
                let src = dir.join(format!("{form}_{op}_{rows}x{cols}.in.mlir"));
                std::fs::write(&src, kernel(op, rows, cols, *reduces)).expect("write");
                // A fresh output path per kernel: a refused emission must not leave the
                // previous one to be checked against this op's expectation, which cost 48
                // false failures when the emulation sweep reused one path.
                let out = dir.join(format!("{form}_{op}_{rows}x{cols}.out.mlir"));
                let emitted = Command::new(exe)
                    .arg(&src)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                if !emitted.status.success() || !out.exists() {
                    continue; // a refusal is an answer
                }
                checked += 1;
                let text = std::fs::read_to_string(&out).unwrap_or_default();

                if !text.contains(primitive) {
                    bad.push(format!(
                        "{form} {op} {rows}x{cols}: does not name `{primitive}`"
                    ));
                }
                if *reduces {
                    // Along each row, not across them. `dimensions = [0]` here would be
                    // the bang defect arriving by a different route -- and a parser, a
                    // use-before-def check and a syntax gate would all pass it.
                    assert_ne!(rows, 0);
                    if !text.contains("dimensions = [1]") {
                        bad.push(format!(
                            "{form} {op} {rows}x{cols}: does not reduce dimension 1 — a \
                             reduction over dimension 0 crosses rows"
                        ));
                    }
                }
                // The declared tensor must be the shape that was asked for. A named
                // operation over the wrong extent is still the wrong computation.
                if !text.contains(&format!("tensor<{rows}x{cols}xf32>")) {
                    bad.push(format!(
                        "{form} {op} {rows}x{cols}: no tensor<{rows}x{cols}xf32> in the \
                         emitted module"
                    ));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("declared_semantics: {checked} emissions checked");
    for b in bad.iter().take(8) {
        eprintln!("  {b}");
    }
    // Guard the vacuous pass: if both backends started refusing, nothing would be checked.
    assert!(
        checked >= 20,
        "only {checked} emissions were produced -- the backends are refusing nearly \
         everything and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} lowerings declare something other than what was asked for. This is not a \
         syntax complaint: the module parses, defines every value it uses, and names the \
         wrong operation.",
        bad.len()
    );
}
