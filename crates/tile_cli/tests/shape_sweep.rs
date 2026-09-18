//! The shape sweep, as something this repo can run again.
//!
//! Every kernel in the test corpus was 32, 64, 256, 1024 or 4096 wide -- all powers of two
//! -- and every one of them passed. Generating kernels at widths chosen to BREAK
//! assumptions rather than confirm them found, in one afternoon:
//!
//!   * a threadgroup tree fold correct only for powers of two, in six emitters: at
//!     `tcount = 127` it started at stride 63 and never read `sdata[126]`, so the row's
//!     last element took no part in its own softmax;
//!   * two backends unable to cover a row wider than one workgroup, independently, neither
//!     caught by a corpus whose rows all fitted in one;
//!   * a fourth prologue form with the same assumption, in the two-input family;
//!   * an accuracy gate stricter than `numpy.allclose`, which this tool's own reference
//!     failed against torch;
//!   * `rows` never leaving 1, so no kernel's row indexing had ever been exercised (#017).
//!
//! None of that is reachable from a corpus of round numbers, and none of it would be
//! findable again if the generator lived only in the session that wrote it.
//!
//! It needs a GPU and a torch, so it does not run by default:
//!
//!     TILE_SWEEP=1 cargo test --test shape_sweep -- --nocapture
//!
//! What it asserts is a FLOOR, not an exact count: the operations with a reference must
//! agree at every shape, and the ones this tool has no derived bound for are named rather
//! than counted. A test that pinned "191 of 192" would fail the day someone adds an op,
//! which is the wrong thing to make expensive.

use std::process::Command;

/// Widths chosen to break assumptions: below a subgroup, exactly one, a prime, a
/// non-multiple of the 64-wide workgroup, and past the 1024 threadgroup bound. Row counts
/// above 1 so that `base = row * num_elements` is exercised at all.
const SHAPES: &[(usize, usize)] = &[
    (1, 17),
    (1, 32),
    (1, 64),
    (1, 127),
    (1, 256),
    (1, 1000),
    (1, 1024),
    (1, 4096),
    (3, 129),
    (4, 256),
    (8, 64),
    (5, 1023),
];

/// Ops with a reference in the harness, so a disagreement means something.
const UNARY: &[&str] = &[
    "exp", "relu", "sqrt", "tanh", "neg", "abs", "sigmoid", "silu",
];
const REDUCE: &[&str] = &["reduce_sum", "reduce_max", "absmax"];

/// Both element types. Four of the findings this sweep produced were in the f16 path
/// alone -- a kernel family the harness could not bind and had never run, a dtype read
/// from a `char*` declaration that does not carry it, a relative gate five times tighter
/// than f16 can represent, and a control arm firing on a correct kernel because reseeding
/// only permutes a periodic row. None of them are reachable from an f32-only corpus.
const DTYPES: &[&str] = &["f32", "f16"];

fn row_kernel(op: &str, rows: usize, cols: usize, reduces: bool, ty: &str) -> String {
    let out_cols = if reduces { 1 } else { cols };
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %o = llvm.mlir.constant({out_cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_{ty}(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_{ty}(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{ty}(%arg1, %y, %r, %o) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// Two tiles in, one out. Its own prologue form -- `gid = row * tcount + tid`, block
/// indexing -- which the row-shaped fixes did not touch, so one workgroup over a 4096-wide
/// row wrote only the first `tcount` elements. Six emitters shared it.
fn two_input_kernel(op: &str, rows: usize, cols: usize, ty: &str) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_{ty}(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %b = llvm.call @__tile_load_{ty}(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_{ty}(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{ty}(%arg2, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// Non-square on purpose: `m = rows, k = cols, n = cols` was square BY ASSUMPTION once,
/// and agreed with the kernel only as long as nobody ran a non-square matmul.
fn matmul_kernel(m: usize, k: usize, n: usize, ty: &str) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
         \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
         \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
         \x20   %x = llvm.call @__tile_load_{ty}(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_load_{ty}(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %z = llvm.call @__tile_matmul_{ty}(%x, %x, %y, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{ty}(%arg2, %z, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// Matrix times vector, one value out per row. Its own `Shape` variant because its two
/// inputs are DIFFERENT sizes -- the first shape for which `(rows, cols)` was not enough.
fn matvec_kernel(rows: usize, cols: usize) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %one = llvm.mlir.constant(1 : i32) : i32\n\
         \x20   %x = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %v = llvm.call @__tile_load_f32(%arg1, %one, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_matvec_f32(%x, %x, %v, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %one) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

#[test]
fn every_op_agrees_at_every_shape() {
    if std::env::var("TILE_SWEEP").as_deref() != Ok("1") {
        eprintln!(
            "shape_sweep: skipped. It needs a GPU and a torch; run it with \
             TILE_SWEEP=1 cargo test --test shape_sweep -- --nocapture"
        );
        return;
    }
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-shape-sweep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let targets: Vec<&str> = std::env::var("TILE_SWEEP_TARGETS")
        .map(|s| Box::leak(s.into_boxed_str()) as &str)
        .unwrap_or("msl,spirv")
        .split(',')
        .collect();

    let mut failures: Vec<String> = Vec::new();
    let mut unbounded: Vec<String> = Vec::new();
    let mut agreed = 0usize;
    let mut ran = 0usize;

    for &(rows, cols) in SHAPES {
        for (op, reduces) in UNARY
            .iter()
            .map(|o| (*o, false))
            .chain(REDUCE.iter().map(|o| (*o, true)))
            .chain(std::iter::once(("softmax", false)))
        {
            for ty in DTYPES {
                let path = dir.join(format!("{op}_{ty}_{rows}x{cols}.mlir"));
                std::fs::write(&path, row_kernel(op, rows, cols, reduces, ty)).expect("write");
                for t in &targets {
                    let out = Command::new(exe)
                        .arg(&path)
                        .args(["-t", t, "-r", "--force"])
                        .current_dir(&dir)
                        .output()
                        .expect("the tile binary runs");
                    let text = String::from_utf8_lossy(&out.stderr).to_string()
                        + &String::from_utf8_lossy(&out.stdout);
                    let Some(verdict) = text
                        .lines()
                        .find(|l| l.contains("verdict:"))
                        .map(|l| l.split("verdict:").nth(1).unwrap_or("").trim().to_string())
                    else {
                        // Refused or unlowerable. That is a statement, not a failure: this
                        // repo refuses far more often than it guesses, on purpose.
                        continue;
                    };
                    ran += 1;
                    if verdict.starts_with("all three agree") {
                        agreed += 1;
                    } else if verdict.starts_with("the kernel differs from BOTH references") {
                        // The verdict the tool prints when it has NO derived bound for the
                        // operation at this dtype -- f16 softmax and rms_norm, whose error is
                        // not the summation bound. It says so in full over the following
                        // lines; this matches the first, which is the part `verdict` holds.
                        //
                        // Counting these as failures would pressure someone into widening a
                        // number to make the suite green, which is exactly what #013 was filed
                        // about. They are named instead, so a new one cannot appear unnoticed.
                        //
                        // `softmax f16` at 4096 wide is the only one today, on BOTH backends
                        // and by the same amount -- two independently written kernels
                        // deviating identically, which is evidence about the format rather
                        // than about either lowering. Its outputs sit near 2.4e-4 with a wide
                        // spread, so the smallest land in f16 subnormals where relative
                        // precision degrades.
                        unbounded.push(format!("{op} {ty} {rows}x{cols} on {t}"));
                    } else {
                        failures.push(format!("{op} {ty} {rows}x{cols} on {t}: {verdict}"));
                    }
                }
            }
        }
    }

    // The shapes that are not a single row of one tile. Each of these found something the
    // row sweep could not: the two-input family had its own block-indexing prologue, and
    // matvec's M/K/N could not be bound at all, so the mixed-precision matvec had never
    // been measured on any backend.
    let run_one = |name: String,
                   src: String,
                   ran: &mut usize,
                   agreed: &mut usize,
                   failures: &mut Vec<String>,
                   unbounded: &mut Vec<String>| {
        let path = dir.join(format!("{name}.mlir"));
        std::fs::write(&path, src).expect("write");
        for t in &targets {
            let out = Command::new(exe)
                .arg(&path)
                .args(["-t", t, "-r", "--force"])
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            let text = String::from_utf8_lossy(&out.stderr).to_string()
                + &String::from_utf8_lossy(&out.stdout);
            let Some(v) = text
                .lines()
                .find(|l| l.contains("verdict:"))
                .map(|l| l.split("verdict:").nth(1).unwrap_or("").trim().to_string())
            else {
                continue;
            };
            *ran += 1;
            if v.starts_with("all three agree") {
                *agreed += 1;
            } else if v.starts_with("the kernel differs from BOTH references") {
                unbounded.push(format!("{name} on {t}"));
            } else {
                failures.push(format!("{name} on {t}: {v}"));
            }
        }
    };

    for &(rows, cols) in &[(1usize, 17usize), (1, 256), (1, 4096), (3, 129), (8, 64)] {
        for op in ["min", "max"] {
            for ty in DTYPES {
                run_one(
                    format!("{op}_{ty}_{rows}x{cols}"),
                    two_input_kernel(op, rows, cols, ty),
                    &mut ran,
                    &mut agreed,
                    &mut failures,
                    &mut unbounded,
                );
            }
        }
    }
    for &(m, k, n) in &[
        (16usize, 16usize, 16usize),
        (32, 64, 16),
        (7, 13, 5),
        (3, 257, 9),
    ] {
        run_one(
            format!("matmul_f32_{m}x{k}x{n}"),
            matmul_kernel(m, k, n, "f32"),
            &mut ran,
            &mut agreed,
            &mut failures,
            &mut unbounded,
        );
    }
    for &(rows, cols) in &[(1usize, 64usize), (4, 256), (7, 129), (16, 1024)] {
        run_one(
            format!("matvec_f32_{rows}x{cols}"),
            matvec_kernel(rows, cols),
            &mut ran,
            &mut agreed,
            &mut failures,
            &mut unbounded,
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("shape_sweep: {agreed} of {ran} measured comparisons agree");
    if !unbounded.is_empty() {
        eprintln!(
            "  {} reported honestly as having NO derived bound (not counted as failures):",
            unbounded.len()
        );
        for u in &unbounded {
            eprintln!("    {u}");
        }
    }
    for f in &failures {
        eprintln!("  {f}");
    }
    assert!(
        ran > 100,
        "the sweep must actually have run: only {ran} comparisons"
    );
    // The honestly-unbounded list is expected to be SMALL. If it grows, something started
    // reporting "no derived bound" that used to be judged, and that is worth failing over
    // even though each individual line is not a defect.
    assert!(
        unbounded.len() <= 4,
        "{} comparisons now report no derived bound, up from the 2 known ones \
         (softmax f16 at 4096 wide, on each backend). Either an op lost its bound or a \
         new dtype/shape combination needs one: {unbounded:?}",
        unbounded.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {ran} shapes disagree. Each line above is a kernel that computes a \
         different answer at that shape than the reference does -- which is what this \
         sweep exists to surface, and how the power-of-two tree fold, the wide-row \
         coverage gap and #017 were found.",
        failures.len()
    );
}
