//! An emitter that accepts an f16 intrinsic and emits an f32 kernel.
//!
//! This is backlog #005 in the DTYPE dimension. That entry was "an unhandled intrinsic
//! emits a plausible copy instead of refusing", and `emit::ignored_intrinsics` catches it
//! by re-emitting with the intrinsic's NAME replaced: if the output does not change, the
//! emitter never looked at that operation.
//!
//! It cannot catch this one. The name IS handled -- there is an arm for `__tile_exp_f16`
//! -- and what the emitter ignores is the WIDTH. So the check is the same idea one level
//! down: emit the same kernel at f16 and at f32 and compare the two outputs. If they are
//! byte-identical, the emitter did not look at the dtype, and the kernel it produced reads
//! four bytes where two were meant: twice the buffer's length, every value a pair of f16s
//! reinterpreted as one f32.
//!
//! WHY A DIFF IS NOT ENOUGH ON ITS OWN, and why this file has an allowlist. Identical text
//! is the CORRECT answer for a backend that obtains its dtype at run time. `tpu` emits
//! `jax.ShapeDtypeStruct(x0.shape, x0.dtype)` -- the dtype comes from the array it is
//! handed. `ttmetal` names no dtype anywhere; its circular buffers carry their own format,
//! set by the host. Both are right to be width-independent, and a bare diff would report
//! them exactly as it reports the four that are wrong.
//!
//! `CLUSTER-lowering.md` records this as a corollary that had to be WITHDRAWN once already:
//! "emit a reduction at two row widths and diff the output; identical text means a
//! hardcoded width. It does not." The same caution applies here, which is why every
//! identical backend must be named with the mechanism that makes it correct rather than
//! merely excused.

use std::process::Command;

/// A backend whose output is legitimately the same at both widths, and why.
///
/// The reason must name the MECHANISM that supplies the dtype at run time. "It does not
/// matter" is not one of these; neither is "not implemented yet", which is the condition
/// this file exists to report.
fn dtype_agnostic(backend: &str, op: &str) -> Option<&'static str> {
    // Per (backend, OPERATION), not per backend. Widening this gate to the two-input ops
    // turned up nki's matmul: `nisa.nc_matmul(t0, t1)` takes no dtype at all, so the
    // operands carry it and identical text is right -- while every OTHER nki arm passes
    // `dtype=` explicitly because `nisa.activation` and friends can change it. One backend
    // can be agnostic for one operation and not for the rest, and a per-backend list
    // cannot say that.
    if backend == "nki" && op == "matmul" {
        return Some(
            "nisa.nc_matmul takes no dtype; the operands carry it, as np.matmul of two \
             f16 tiles yields f16",
        );
    }
    if backend == "nki" && op == "rms_norm" {
        return Some(
            "every operation in the emitted line inherits f16 from `nl.load`, and the one \
             explicit dtype is the ACCUMULATOR, pinned to f32 on purpose -- so the text is \
             width-independent because the width travels in the operands",
        );
    }
    if backend == "nki" && op == "log" {
        return Some(
            "`nl.log(t0)` takes no dtype; t0 comes from `nl.load(p0[...])` whose width is \
             the host array's, so identical text is right — same mechanism as rms_norm",
        );
    }
    match backend {
        "tpu" => Some(
            "emits jax.ShapeDtypeStruct(x0.shape, x0.dtype) -- the dtype comes from the \
             array the kernel is handed, at run time",
        ),
        "ttmetal" => Some(
            "names no dtype at all; the circular buffers carry their own DataFormat, set \
             by the host outside the compute kernel",
        ),
        _ => None,
    }
}

/// The f16 and f32 forms of one kernel, identical but for the intrinsic width.
fn pair(op: &str, reduces: bool, width: &str) -> String {
    let (rows, cols) = (4usize, 256usize);
    let out = if reduces { 1 } else { cols };
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %o = llvm.mlir.constant({out} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_{width}(%arg0, %r, %c) \
         : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_{width}(%a, %a, %r, %c) \
         : (i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{width}(%arg1, %y, %r, %o) \
         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// Backends known to ignore the dtype TODAY, with why each is not a one-line fix.
///
/// A ratchet, not an excuse list. The test fails on a backend that is not here, and it
/// also fails on one that is here and has STOPPED ignoring -- so fixing a backend forces
/// removing it, and the debt can only shrink deliberately.
///
/// `bang` is deliberately absent: it was on this list in every respect until its
/// `prescan_body_bang` learned to read the dtype from the loads and stores, and its whole
/// `half` path had been present the entire time. That is the shape of fix these four do
/// NOT have available.
fn known_debt(backend: &str) -> Option<&'static str> {
    match backend {
        "hexagon" => Some(
            "calls hvx_vexp_f32 and friends by name, with the width in the callee, over \
             `const float*` buffers",
        ),
        "csl" => Some("declares `var input0_tile: [N]f32;` regardless of the intrinsic"),
        _ => None,
    }
}

/// `rms_norm` in its eps-carrying five-argument form.
fn rms_norm_pair(width: &str) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant(4 : i32) : i32\n\
         \x20   %c = llvm.mlir.constant(256 : i32) : i32\n\
         \x20   %e = llvm.mlir.constant(2.500000e-02 : f32) : f32\n\
         \x20   %a = llvm.call @__tile_load_{width}(%arg0, %r, %c) \
         : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_rms_norm_{width}(%a, %a, %e, %r, %c) \
         : (i32, i32, f32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{width}(%arg1, %y, %r, %c) \
         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// `(m x k) . (k x n)` at one width.
fn matmul_pair(width: &str) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %m = llvm.mlir.constant(8 : i32) : i32\n\
         \x20   %k = llvm.mlir.constant(16 : i32) : i32\n\
         \x20   %n = llvm.mlir.constant(4 : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_{width}(%arg0, %m, %k) \
         : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %b = llvm.call @__tile_load_{width}(%arg1, %k, %n) \
         : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_matmul_{width}(%a, %a, %b, %m, %k, %n) \
         : (i32, i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{width}(%arg2, %y, %m, %n) \
         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

/// Two equal-sized inputs at one width.
fn binary_pair(op: &str, width: &str) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant(4 : i32) : i32\n\
         \x20   %c = llvm.mlir.constant(256 : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_{width}(%arg0, %r, %c) \
         : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %b = llvm.call @__tile_load_{width}(%arg1, %r, %c) \
         : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_{width}(%a, %a, %b, %r, %c) \
         : (i32, i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_{width}(%arg2, %y, %r, %c) \
         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

#[test]
fn no_backend_emits_the_same_kernel_for_f16_and_f32() {
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-dtype-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    const BACKENDS: &[&str] = &[
        "linalg", "rvv", "pto", "msl", "gpu", "musa", "spirv", "nki", "aie", "tpu", "bang",
        "gaudi", "hexagon", "ttmetal", "csl",
    ];
    // (intrinsic, reduces)
    const OPS: &[(&str, bool)] = &[
        ("exp", false),
        ("sigmoid", false),
        ("log", false),
        ("rsqrt", false),
        ("silu", false),
        ("softmax", false),
        ("rms_norm", false),
        ("reduce_sum", true),
        ("reduce_max", true),
        ("absmax", true),
    ];
    // The two-input operations, which the first version of this gate did not cover. Their
    // absence made "0 unexplained" an overclaim: nki's matmul was byte-identical at both
    // widths the whole time and no gate could see it. It turned out to be CORRECT, which
    // is the more interesting outcome -- the audit was incomplete either way.
    const TWO_IN: &[(&str, bool)] = &[("min", false), ("max", false), ("matmul", false)];

    let emit = |backend: &str, op: &str, reduces: bool, width: &str| -> Option<Vec<u8>> {
        let src = dir.join(format!("{backend}_{op}_{width}.mlir"));
        let text = if op == "matmul" {
            matmul_pair(width)
        } else if op == "min" || op == "max" {
            binary_pair(op, width)
        } else if op == "rms_norm" {
            // The FIVE-argument form, which is the one that carries eps. The generic
            // builder emits the four-argument one, and nki REFUSES that -- so rms_norm was
            // never compared on nki at all and the gate silently skipped it. Second
            // coverage hole this file has had, same shape as the missing two-input ops:
            // an audit that cannot emit a kernel reports nothing about it, and reports
            // nothing in exactly the same way it reports agreement.
            rms_norm_pair(width)
        } else {
            pair(op, reduces, width)
        };
        std::fs::write(&src, text).ok()?;
        let out = dir.join(format!("{backend}_{op}_{width}.out"));
        let _ = std::fs::remove_file(&out);
        let r = Command::new(exe)
            .arg(&src)
            .args(["-t", backend, "-o"])
            .arg(&out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .ok()?;
        if !r.status.success() {
            return None;
        }
        let bytes = std::fs::read(&out).ok()?;
        (!bytes.is_empty()).then_some(bytes)
    };

    let (mut pairs, mut identical) = (0usize, 0usize);
    let mut ignoring: Vec<String> = Vec::new();
    let mut excused: Vec<String> = Vec::new();
    let mut debt: Vec<String> = Vec::new();

    for backend in BACKENDS {
        for (op, reduces) in OPS.iter().chain(TWO_IN.iter()) {
            let (Some(a), Some(b)) = (
                emit(backend, op, *reduces, "f16"),
                emit(backend, op, *reduces, "f32"),
            ) else {
                continue; // a backend that refuses one width has nothing to compare
            };
            pairs += 1;
            if a != b {
                continue;
            }
            identical += 1;
            match (dtype_agnostic(backend, op), known_debt(backend)) {
                (Some(_), _) => excused.push(format!("{backend}/{op}")),
                (None, Some(_)) => debt.push(format!("{backend}/{op}")),
                (None, None) => ignoring.push(format!("{backend}/{op}")),
            }
        }
    }

    // A backend on the allowlist that has STOPPED being identical is also worth knowing:
    // it means the mechanism named above changed, and the reason recorded here is stale.
    for backend in BACKENDS {
        // Try both an elementwise op and matmul: a backend may be agnostic for only one
        // of them, which is exactly nki's case and would otherwise go unchecked here.
        if let Some(why) =
            dtype_agnostic(backend, "exp").or_else(|| dtype_agnostic(backend, "matmul"))
        {
            let seen = excused
                .iter()
                .any(|e| e.starts_with(&format!("{backend}/")));
            assert!(
                seen,
                "{backend} is recorded as dtype-agnostic ({why}), but every kernel it \
                 emitted differs between f16 and f32 -- the reason is stale"
            );
        }
    }

    // A backend on the debt list that no longer ignores the dtype has been FIXED, and the
    // entry is now a false record of the state of the tree. Failing here is how the list
    // shrinks on purpose rather than rotting.
    for backend in BACKENDS {
        if known_debt(backend).is_some() {
            let still = debt.iter().any(|e| e.starts_with(&format!("{backend}/")));
            assert!(
                still,
                "{backend} is on the dtype debt list but every kernel it emitted now \
                 differs between f16 and f32 -- it was fixed; remove it from `known_debt`"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "dtype_is_not_ignored: {pairs} pairs emitted at both widths, {identical} identical \
         ({} dtype-agnostic by design, {} known debt, {} unexplained)",
        excused.len(),
        debt.len(),
        ignoring.len()
    );
    for d in &debt {
        eprintln!("  known debt: {d}");
    }
    for e in &ignoring {
        eprintln!("  ignores the dtype: {e}");
    }
    assert!(
        pairs >= 40,
        "only {pairs} pairs emitted at both widths -- this test would pass by comparing \
         almost nothing"
    );
    assert!(
        ignoring.is_empty(),
        "{} backend-operation pairs emit a BYTE-IDENTICAL kernel for f16 and f32, and are \
         on neither list. The emitter did not look at the width, so the kernel reads four \
         bytes where two were meant: {}.\n\nA backend that genuinely obtains its dtype at \
         run time belongs in `dtype_agnostic` with the MECHANISM named. One that simply \
         does not support f16 belongs in `known_debt` with backlog #024 -- and should \
         refuse the intrinsic rather than emit a plausible kernel, which is #005 in the \
         dtype dimension.",
        ignoring.len(),
        ignoring.join(", ")
    );
}
