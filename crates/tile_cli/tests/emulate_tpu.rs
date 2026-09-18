//! Running the TPU/Pallas backend with no TPU, to check what it COMPUTES.
//!
//! Fourth in the line after bang, gpu and nki. Pallas makes this the easiest of the four:
//! the emitted kernel is `def k(p0_ref, p1_ref)` with `t0 = p0_ref[...]`, a `jnp` call, and
//! `p1_ref[...] = t1`. A numpy array already supports both of those subscripts, and `jnp`
//! IS numpy for anything a kernel body does, so the shim is a re-export plus an identity
//! `jax.jit`.
//!
//! Like nki and unlike gpu, this reaches the REDUCTIONS, because Pallas names them
//! (`jnp.max(t0, axis=-1, keepdims=True)`) rather than building one out of warp shuffles.
//!
//! A false alarm on the way, worth recording because it looked exactly like a real find: a
//! batched shell loop reported `reduce_max` returning the SUM -- the precise defect
//! `declared_semantics.rs` exists to catch. Reading the emitted code first showed
//! `jnp.max(t0, axis=-1, keepdims=True)`, which is correct, and running that one case alone
//! gave the right answer. The loop had reused one filename across cases. Each case here
//! gets its own path for that reason.
//!
//! Needs a Python with numpy; the interpreter is probed, not assumed.

use std::process::Command;
use tile_cli::run::{error_budget, reference_output, unit_roundoff, RefOp, Shape};
use tile_cli::torchref::{Accuracy, Tolerance};

const JAX_INIT: &str = r#"
import numpy as _np
def jit(fn=None, **kw):
    if fn is None:
        return lambda f: f
    return fn
class ShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

# `jax.nn` and `jax.lax`, taken from what mlir_to_tpu.rs can actually emit rather than
# added one crash at a time: `grep -oE '\bjax\.[a-z_0-9.]+'` over the emitter.
class _NN:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + _np.exp(-x))
    @staticmethod
    def silu(x):
        return x / (1.0 + _np.exp(-x))
    @staticmethod
    def softmax(x, axis=-1):
        m = _np.max(x, axis=axis, keepdims=True)
        e = _np.exp(x - m)
        return e / _np.sum(e, axis=axis, keepdims=True)

class _Lax:
    @staticmethod
    def rsqrt(x):
        # `jax.lax.rsqrt` is ONE primitive. Computing it as 1/sqrt(x) in the input's own
        # dtype rounds TWICE, and in f16 that showed up as 6.4e-4 against a one-rounding
        # bound of 4.883e-4 -- reported as a defect in the lowering, which emits exactly
        # `jax.lax.rsqrt(t0)` and is not at fault. Compute wide, narrow once, which is
        # what the primitive does. Unchanged for f32 inputs.
        a = _np.asarray(x)
        return (1.0 / _np.sqrt(a.astype(_np.float32))).astype(a.dtype)
    @staticmethod
    def dynamic_slice(x, starts, sizes):
        sl = tuple(slice(s, s + n) for s, n in zip(starts, sizes))
        return x[sl]
    @staticmethod
    def top_k(x, k):
        idx = _np.argsort(-x, axis=-1)[..., :k]
        return _np.take_along_axis(x, idx, axis=-1), idx

nn = _NN()
lax = _Lax()
"#;

const JAX_NUMPY: &str = r#"
from numpy import *
import numpy as _np
float32 = _np.float32
float16 = _np.float16
int32 = _np.int32
bfloat16 = _np.float32   # numpy has no bfloat16; f32 is the honest stand-in for a
                         # correctness comparison, and only f32 kernels are driven.
"#;

const PALLAS: &str = r#"
import numpy as _np
def pallas_call(fn, out_shape=None, **kw):
    def run(*xs):
        out = _np.zeros(out_shape.shape, dtype=out_shape.dtype)
        fn(*xs, out)
        return out
    return run
def BlockSpec(*a, **kw):
    return None
def program_id(axis=0):
    return 0
class _Grid:
    def __getitem__(self, k):
        return k
grid = _Grid()
"#;

const DRIVER: &str = r#"
import sys, importlib.util, numpy as np

def input_value(b, i):
    return ((i + 7*b) % 17) * 0.25 - 2.0 + i * 1e-4

def grid(buf, rows, cols, dt=np.float32):
    return np.array([[input_value(buf, r*cols + c) for c in range(cols)] for r in range(rows)],
                    dtype=dt)

def main():
    # path a_rows a_cols out_rows out_cols [b_rows b_cols] [--dtype-half]
    #
    # tpu is DTYPE-AGNOSTIC by construction: the emitted kernel says
    # `jax.ShapeDtypeStruct(x0.shape, x0.dtype)`, so the width travels in the array it is
    # handed and the same emitted text is correct for both. That is why this needed only a
    # driver flag and no emitter change at all -- and why `dtype_is_not_ignored.rs` lists
    # tpu as agnostic with that mechanism named rather than as debt.
    a = [x for x in sys.argv if x != "--dtype-half"]
    dt = np.float16 if len(a) != len(sys.argv) else np.float32
    path = a[1]
    ar, ac, orow, ocol = int(a[2]), int(a[3]), int(a[4]), int(a[5])
    br, bc = (int(a[6]), int(a[7])) if len(a) > 7 else (0, 0)
    spec = importlib.util.spec_from_file_location("kern", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    p0 = grid(0, ar, ac, dt)
    # A Pallas ref IS an array subscript: `p0_ref[...]` reads and `p1_ref[...] = v` writes,
    # both of which a numpy array does natively.
    out = np.zeros((orow, ocol), dtype=dt)
    if br:
        # Generator index 1 for the second buffer, exactly as reference_output does.
        m.k(p0, grid(1, br, bc, dt), out)
    else:
        m.k(p0, out)
    for v in out.reshape(-1):
        print("%.9e" % v)

main()
"#;
fn which(tool: &str) -> Option<String> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .ok()?;
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !p.is_empty()).then_some(p)
}

/// A Python that actually has numpy. PROBED, not assumed: on this machine the default
/// `python3` does not have it and `/usr/bin/python3` does, and assuming the first would
/// have skipped the whole test with a misleading note.
fn python_with_numpy() -> Option<String> {
    let mut candidates: Vec<String> = vec!["/usr/bin/python3".into()];
    for c in ["python3", "python3.13", "python3.12", "python3.11"] {
        if let Some(p) = which(c) {
            candidates.push(p);
        }
    }
    candidates.into_iter().find(|p| {
        Command::new(p)
            .args(["-c", "import numpy"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// The epsilon `rms_norm` carries, deliberately NOT 1e-6.
///
/// `Shape::rms_eps_from_mlir`'s own comment records why: the emitter once hardcoded 1e-6
/// and the reference hardcoded 1e-6 to match, so the two constants agreed with each other
/// and checked nothing -- change the kernel's eps and the comparison would have gone on
/// passing. A value neither side can have baked in makes the agreement mean something,
/// and the reference reads it back out of the generated MLIR rather than being told twice.
const RMS_EPS: f32 = 2.5e-2;

fn kernel_mlir(op: &str, rows: usize, cols: usize, reduces: bool) -> String {
    let out_cols = if reduces { 1 } else { cols };
    // rms_norm is the one operation here whose parameter travels in the CALL rather than
    // in the shape, so it needs the five-argument form.
    if op == "rms_norm" {
        return format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
             attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %e = llvm.mlir.constant({RMS_EPS:e} : f32) : f32\n\
             \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) \
             : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_rms_norm_f32(%a, %a, %e, %r, %c) \
             : (i32, i32, f32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
    }
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

/// Judge an f16 result with the run path's own rule rather than a number chosen here.
fn f16_judge(
    label: String,
    got: Vec<f32>,
    want: &[f32],
    op: RefOp,
    shape: Shape,
    bad: &mut Vec<String>,
) {
    if got.len() != want.len() {
        bad.push(format!(
            "{label}: wrote {} values, {} were due",
            got.len(),
            want.len()
        ));
        return;
    }
    let tol = match error_budget(op, shape, "half") {
        Some(b) => Tolerance::Summation(b),
        None => Tolerance::Fixed {
            abs: 1e-5,
            rel: 1e-4_f32.max(unit_roundoff("half")),
        },
    };
    let acc = Accuracy::of(&got, want);
    if acc.compared == 0 {
        bad.push(format!("{label}: nothing was compared"));
        return;
    }
    if !tol.admits(&acc) {
        bad.push(format!(
            "{label}: max_abs {:.3e} max_rel {:.3e} near_zero_abs {:.3e} non_finite {} \
             over {} compared; {}",
            acc.max_abs,
            acc.max_rel,
            acc.max_abs_near_zero,
            acc.non_finite,
            acc.compared,
            match &tol {
                Tolerance::Fixed { abs, rel } => format!("allowed abs {abs:.3e} rel {rel:.3e}"),
                Tolerance::Summation(b) => format!("allowed abs {:.3e} (summation)", b.guaranteed),
            }
        ));
    }
}

#[test]
fn tpu_computes_what_the_reference_computes() {
    let Some(py) = python_with_numpy() else {
        eprintln!("emulate_tpu: skipped, no Python with numpy");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emulate-tpu-{}", std::process::id()));
    let jax = dir.join("jax").join("experimental");
    std::fs::create_dir_all(&jax).expect("a scratch package");
    std::fs::write(dir.join("jax").join("__init__.py"), JAX_INIT).expect("write");
    std::fs::write(dir.join("jax").join("numpy.py"), JAX_NUMPY).expect("write");
    std::fs::write(jax.join("__init__.py"), "").expect("write");
    std::fs::write(jax.join("pallas.py"), PALLAS).expect("write");
    std::fs::write(dir.join("drv.py"), DRIVER).expect("write");

    let shapes = [(1usize, 64usize), (1, 256), (4, 256), (3, 129), (8, 64)];
    let ops: &[(&str, RefOp, bool)] = &[
        ("exp", RefOp::Exp, false),
        ("neg", RefOp::Neg, false),
        ("abs", RefOp::Abs, false),
        ("sigmoid", RefOp::Sigmoid, false),
        ("log", RefOp::Log, false),
        ("rsqrt", RefOp::Rsqrt, false),
        ("silu", RefOp::Silu, false),
        ("rms_norm", RefOp::RmsNorm, false),
        ("softmax", RefOp::Softmax, false),
        ("reduce_sum", RefOp::ReduceSum, true),
        ("reduce_max", RefOp::ReduceMax, true),
        ("absmax", RefOp::Absmax, true),
    ];

    let (mut checked, mut refused) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    for &(rows, cols) in &shapes {
        for (name, op, reduces) in ops {
            let src = dir.join(format!("{name}_{rows}x{cols}.mlir"));
            let mlir = kernel_mlir(name, rows, cols, *reduces);
            // Read the eps back the way the emitter does, so the reference and the
            // kernel cannot be handed different numbers. Ops without one are
            // unaffected: they have no eps in the call and none in the reference.
            let eps = Shape::rms_eps_from_mlir(&mlir).unwrap_or(1e-6);
            std::fs::write(&src, &mlir).expect("write");
            let py_out = dir.join(format!("{name}_{rows}x{cols}.py"));
            let _ = std::fs::remove_file(&py_out);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "tpu", "-o"])
                .arg(&py_out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !py_out.exists() {
                refused += 1; // a refusal is an answer
                continue;
            }
            let out_cols = if *reduces { 1 } else { cols };
            let run = Command::new(&py)
                .arg(dir.join("drv.py"))
                .arg(&py_out)
                .arg(rows.to_string())
                .arg(cols.to_string())
                .arg(rows.to_string())
                .arg(out_cols.to_string())
                .env("PYTHONPATH", &dir)
                .output()
                .expect("python runs");
            if !run.status.success() {
                bad.push(format!(
                    "{name} {rows}x{cols}: did not run — {}",
                    String::from_utf8_lossy(&run.stderr)
                        .lines()
                        .last()
                        .unwrap_or("")
                        .trim()
                        .chars()
                        .take(90)
                        .collect::<String>()
                ));
                continue;
            }
            let shape = if *reduces {
                Shape::RowReduce { rows, cols }
            } else {
                Shape::Rows { rows, cols }
            };
            let want = reference_output(*op, shape, eps, "float");
            let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
                .lines()
                .filter_map(|l| l.trim().parse::<f32>().ok())
                .collect();
            checked += 1;
            if got.len() != want.len() {
                bad.push(format!(
                    "{name} {rows}x{cols}: wrote {} values, {} were due",
                    got.len(),
                    want.len()
                ));
                continue;
            }
            let mut worst = 0.0f32;
            for (g, w) in got.iter().zip(want.iter()) {
                if (g.is_nan() && w.is_nan()) || (g.is_infinite() && w.is_infinite()) {
                    continue;
                }
                // `f32::max` RETURNS THE OTHER OPERAND when one side is NaN, so folding a
                // one-sided NaN straight into `worst` DISCARDS it: a kernel writing NaN
                // where the reference says 0.5 left worst at 0 and passed. Hoist it to
                // infinity so the disagreement survives the fold.
                let d = (g - w).abs() / w.abs().max(1.0);
                worst = if d.is_nan() {
                    f32::INFINITY
                } else {
                    worst.max(d)
                };
            }
            if worst > 1e-5 {
                bad.push(format!("{name} {rows}x{cols}: max rel {worst:.2e}"));
            }
        }
    }

    // MATMUL: the first shape with two inputs of DIFFERENT sizes, and the one where an
    // indexing mistake is easiest to make and hardest to see. `reference_output` builds
    // both operands itself -- buffer 0 and buffer 1 from the same generator the driver
    // uses -- so the only thing being compared is the lowering.
    for &(m, k, n) in &[(8usize, 16usize, 4usize), (4, 4, 4), (16, 8, 32)] {
        let mlir = format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
             \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
             \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f32(%arg2, %y, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
        let src = dir.join(format!("matmul_{m}x{k}x{n}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let py_out = dir.join(format!("matmul_{m}x{k}x{n}.py"));
        let _ = std::fs::remove_file(&py_out);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "tpu", "-o"])
            .arg(&py_out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !py_out.exists() {
            refused += 1;
            continue;
        }
        let run = Command::new(&py)
            .arg(dir.join("drv.py"))
            .arg(&py_out)
            .arg(m.to_string())
            .arg(k.to_string())
            .arg(m.to_string())
            .arg(n.to_string())
            .arg(k.to_string())
            .arg(n.to_string())
            .env("PYTHONPATH", &dir)
            .output()
            .expect("python runs");
        if !run.status.success() {
            bad.push(format!(
                "matmul {m}x{k}x{n}: did not run — {}",
                String::from_utf8_lossy(&run.stderr)
                    .lines()
                    .last()
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(90)
                    .collect::<String>()
            ));
            continue;
        }
        let want = reference_output(RefOp::Matmul, Shape::Matmul { m, k, n }, 1e-6, "float");
        let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect();
        checked += 1;
        if got.len() != want.len() {
            bad.push(format!(
                "matmul {m}x{k}x{n}: wrote {} values, {} were due",
                got.len(),
                want.len()
            ));
            continue;
        }
        let mut worst = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            // `f32::max` RETURNS THE OTHER OPERAND when one side is NaN, so folding a
            // one-sided NaN straight into `worst` DISCARDS it: a kernel writing NaN
            // where the reference says 0.5 left worst at 0 and passed. Hoist it to
            // infinity so the disagreement survives the fold.
            let d = (g - w).abs() / w.abs().max(1.0);
            worst = if d.is_nan() {
                f32::INFINITY
            } else {
                worst.max(d)
            };
        }
        if worst > 1e-4 {
            bad.push(format!("matmul {m}x{k}x{n}: max rel {worst:.2e}"));
        }
    }

    // MAX: two inputs of EQUAL size. It was held back for want of a sweep, not for want
    // of a driver -- the harness above already takes an optional second buffer of any
    // shape, and the matmul sweep uses it. `min` is not here because this backend has no
    // arm for it; `emulator_coverage.rs` holds that distinction.
    //
    // Equal-size operands are where a swapped pair hides best, since max(a, b) == max(b, a).
    // The second buffer comes from generator index 1, so the VALUES differ even though the
    // operation does not care about order -- which is what makes agreement mean something.
    for &(rows, cols) in &shapes {
        let mlir = format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_max_f32(%a, %a, %b, %r, %c) \
             : (i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %c) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
        let src = dir.join(format!("max_{rows}x{cols}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let py_out = dir.join(format!("max_{rows}x{cols}.py"));
        let _ = std::fs::remove_file(&py_out);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "tpu", "-o"])
            .arg(&py_out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !py_out.exists() {
            refused += 1;
            continue;
        }
        let run = Command::new(&py)
            .arg(dir.join("drv.py"))
            .arg(&py_out)
            .arg(rows.to_string())
            .arg(cols.to_string())
            .arg(rows.to_string())
            .arg(cols.to_string())
            .arg(rows.to_string())
            .arg(cols.to_string())
            .env("PYTHONPATH", &dir)
            .output()
            .expect("python runs");
        if !run.status.success() {
            bad.push(format!(
                "max {rows}x{cols}: did not run — {}",
                String::from_utf8_lossy(&run.stderr)
                    .lines()
                    .last()
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(90)
                    .collect::<String>()
            ));
            continue;
        }
        let want = reference_output(RefOp::Max, Shape::Rows2 { rows, cols }, 1e-6, "float");
        let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect();
        checked += 1;
        if got.len() != want.len() {
            bad.push(format!(
                "max {rows}x{cols}: wrote {} values, {} were due",
                got.len(),
                want.len()
            ));
            continue;
        }
        let mut worst = 0.0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g - w).abs() / w.abs().max(1.0);
            worst = if d.is_nan() {
                f32::INFINITY
            } else {
                worst.max(d)
            };
        }
        if worst > 1e-5 {
            bad.push(format!("max {rows}x{cols}: max rel {worst:.2e}"));
        }
    }

    // ── f16 ──────────────────────────────────────────────────────────────────────
    //
    // tpu is the cheapest of the four to reach and the most interesting reason why: its
    // emitted kernel is DTYPE-AGNOSTIC by construction --
    // `jax.ShapeDtypeStruct(x0.shape, x0.dtype)` -- so the width travels in the array
    // rather than in the text, and driving f16 needed a driver flag and no emitter change.
    // bang needed four accumulator fixes and a shim rewrite to get here; nki needed one.
    //
    // Tolerances come from `error_budget` and `unit_roundoff`, never from a number chosen
    // here.
    let f16_ops: &[(&str, RefOp, bool)] = &[
        ("exp", RefOp::Exp, false),
        ("log", RefOp::Log, false),
        ("rsqrt", RefOp::Rsqrt, false),
        ("reduce_sum", RefOp::ReduceSum, true),
        ("reduce_max", RefOp::ReduceMax, true),
        ("absmax", RefOp::Absmax, true),
    ];
    for &(rows, cols) in &shapes {
        for (name, op, reduces) in f16_ops {
            let out_cols = if *reduces { 1 } else { cols };
            let mlir = format!(
                "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                 attributes {{hacc.entry}} {{\n    ^bb0:\n\
                 \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                 \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                 \x20   %o = llvm.mlir.constant({out_cols} : i32) : i32\n\
                 \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) \
                 : (!llvm.ptr<1>, i32, i32) -> i32\n\
                 \x20   %y = llvm.call @__tile_{name}_f16(%a, %a, %r, %c) \
                 : (i32, i32, i32, i32) -> i32\n\
                 \x20   llvm.call @__tile_store_f16(%arg1, %y, %r, %o) \
                 : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                 \x20   llvm.return\n  }}\n}}\n"
            );
            let stem = format!("f16_{name}_{rows}x{cols}");
            let src = dir.join(format!("{stem}.mlir"));
            std::fs::write(&src, &mlir).expect("write");
            let py_out = dir.join(format!("{stem}.py"));
            let _ = std::fs::remove_file(&py_out);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "tpu", "-o"])
                .arg(&py_out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !py_out.exists() {
                refused += 1;
                continue;
            }
            let shape = if *reduces {
                Shape::RowReduce { rows, cols }
            } else {
                Shape::Rows { rows, cols }
            };
            let want = reference_output(*op, shape, 1e-6, "half");
            let run = Command::new(&py)
                .arg(dir.join("drv.py"))
                .arg(&py_out)
                .arg(rows.to_string())
                .arg(cols.to_string())
                .arg(rows.to_string())
                .arg(out_cols.to_string())
                .arg("--dtype-half")
                .env("PYTHONPATH", &dir)
                .output()
                .expect("python runs");
            if !run.status.success() {
                bad.push(format!(
                    "f16 {name} {rows}x{cols}: did not run — {}",
                    String::from_utf8_lossy(&run.stderr)
                        .lines()
                        .last()
                        .unwrap_or("")
                        .trim()
                        .chars()
                        .take(90)
                        .collect::<String>()
                ));
                continue;
            }
            let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
                .lines()
                .filter_map(|l| l.trim().parse::<f32>().ok())
                .collect();
            checked += 1;
            f16_judge(
                format!("f16 {name} {rows}x{cols}"),
                got,
                &want,
                *op,
                shape,
                &mut bad,
            );
        }
    }

    // The two-input f16 operations, closed the same way bang's and nki's were.
    for &(rows, cols) in &shapes {
        let mlir = format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) \
             : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f16(%arg1, %r, %c) \
             : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_max_f16(%a, %a, %b, %r, %c) \
             : (i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg2, %y, %r, %c) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
        let stem = format!("f16_max_{rows}x{cols}");
        let src = dir.join(format!("{stem}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let py_out = dir.join(format!("{stem}.py"));
        let _ = std::fs::remove_file(&py_out);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "tpu", "-o"])
            .arg(&py_out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !py_out.exists() {
            refused += 1;
            continue;
        }
        let shape = Shape::Rows2 { rows, cols };
        let want = reference_output(RefOp::Max, shape, 1e-6, "half");
        let run = Command::new(&py)
            .arg(dir.join("drv.py"))
            .arg(&py_out)
            .arg(rows.to_string())
            .arg(cols.to_string())
            .arg(rows.to_string())
            .arg(cols.to_string())
            .arg(rows.to_string())
            .arg(cols.to_string())
            .arg("--dtype-half")
            .env("PYTHONPATH", &dir)
            .output()
            .expect("python runs");
        if !run.status.success() {
            bad.push(format!(
                "f16 max {rows}x{cols}: did not run — {}",
                String::from_utf8_lossy(&run.stderr)
                    .lines()
                    .last()
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(90)
                    .collect::<String>()
            ));
            continue;
        }
        let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect();
        checked += 1;
        f16_judge(
            format!("f16 max {rows}x{cols}"),
            got,
            &want,
            RefOp::Max,
            shape,
            &mut bad,
        );
    }

    for &(m, k, n) in &[(8usize, 16usize, 4usize), (4, 4, 4), (16, 8, 32)] {
        let mlir = format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
             \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
             \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f16(%arg0, %m, %k) \
             : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f16(%arg1, %k, %n) \
             : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_matmul_f16(%a, %a, %b, %m, %k, %n) \
             : (i32, i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f16(%arg2, %y, %m, %n) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
        let stem = format!("f16_matmul_{m}x{k}x{n}");
        let src = dir.join(format!("{stem}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let py_out = dir.join(format!("{stem}.py"));
        let _ = std::fs::remove_file(&py_out);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "tpu", "-o"])
            .arg(&py_out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !py_out.exists() {
            refused += 1;
            continue;
        }
        let shape = Shape::Matmul { m, k, n };
        let want = reference_output(RefOp::Matmul, shape, 1e-6, "half");
        let run = Command::new(&py)
            .arg(dir.join("drv.py"))
            .arg(&py_out)
            .arg(m.to_string())
            .arg(k.to_string())
            .arg(m.to_string())
            .arg(n.to_string())
            .arg(k.to_string())
            .arg(n.to_string())
            .arg("--dtype-half")
            .env("PYTHONPATH", &dir)
            .output()
            .expect("python runs");
        if !run.status.success() {
            bad.push(format!(
                "f16 matmul {m}x{k}x{n}: did not run — {}",
                String::from_utf8_lossy(&run.stderr)
                    .lines()
                    .last()
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(90)
                    .collect::<String>()
            ));
            continue;
        }
        let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect();
        checked += 1;
        f16_judge(
            format!("f16 matmul {m}x{k}x{n}"),
            got,
            &want,
            RefOp::Matmul,
            shape,
            &mut bad,
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emulate_tpu: {checked} kernels run and compared, {refused} refused");
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    assert!(
        checked >= 15,
        "only {checked} kernels were emitted and run -- tpu is refusing nearly everything \
         and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emulated Pallas kernels disagree with the reference. This is not a syntax \
         complaint: the kernel ran and computed a different answer.",
        bad.len()
    );
}

/// Which operations tpu is driven for in f16, and why the rest are not.
///
/// Exhaustive over `RefOp`, like msl's, bang's and nki's.
///
/// tpu was the cheapest of the four backends to reach and the reason is structural rather
/// than lucky: its emitted kernel says `jax.ShapeDtypeStruct(x0.shape, x0.dtype)`, so the
/// width travels in the array it is handed and the same text is correct at both. bang
/// needed four accumulator fixes and a shim rewrite to get here; nki needed one accumulator
/// fix; tpu needed a driver flag. **A backend that reads its dtype at run time cannot get
/// this wrong**, which is worth more than any number of gates over one that can.
#[test]
fn every_reference_operation_states_what_tpu_f16_does_with_it() {
    fn not_driven_because(op: RefOp) -> &'static str {
        match op {
            RefOp::Exp
            | RefOp::Log
            | RefOp::Rsqrt
            | RefOp::ReduceSum
            | RefOp::ReduceMax
            | RefOp::Absmax
            | RefOp::Max
            | RefOp::Matmul => "",
            RefOp::Sigmoid | RefOp::Silu | RefOp::Softmax | RefOp::RmsNorm => {
                "a chain of several f16 operations judged against a bound for ONE rounding; \
                 the same class held out on bang and nki, and deriving gamma_k needs a \
                 narrowing count the harness cannot see from outside the emitter"
            }
            RefOp::Neg
            | RefOp::Abs
            | RefOp::Relu
            | RefOp::Sqrt
            | RefOp::Tanh
            | RefOp::Softplus
            | RefOp::Min
            | RefOp::Matvec => "tpu has no arm for this operation in any width",
        }
    }

    const ALL: &[RefOp] = &[
        RefOp::Matmul,
        RefOp::Softmax,
        RefOp::Exp,
        RefOp::Sigmoid,
        RefOp::Relu,
        RefOp::Sqrt,
        RefOp::Log,
        RefOp::Neg,
        RefOp::Abs,
        RefOp::Tanh,
        RefOp::Rsqrt,
        RefOp::Silu,
        RefOp::Softplus,
        RefOp::ReduceMax,
        RefOp::ReduceSum,
        RefOp::Min,
        RefOp::Max,
        RefOp::Matvec,
        RefOp::RmsNorm,
        RefOp::Absmax,
    ];
    assert_eq!(
        ALL.len(),
        20,
        "RefOp changed; say what tpu's f16 does with the new one"
    );
    let driven = ALL
        .iter()
        .filter(|op| not_driven_because(**op).is_empty())
        .count();
    assert!(
        driven >= 8,
        "only {driven} operations are driven in f16 on tpu; the sweep has lost coverage"
    );
    for op in ALL {
        let why = not_driven_because(*op);
        if !why.is_empty() {
            eprintln!("tpu f16 skips {op:?}: {why}");
        }
    }
}
