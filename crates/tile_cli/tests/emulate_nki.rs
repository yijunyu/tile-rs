//! Running the AWS Neuron backend with no Trainium, to check what it COMPUTES.
//!
//! Third in the line that starts with `emulate_bang.rs`. The lever is the same and the
//! reason it works is worth stating: NKI's emitted code is a *small, total* vocabulary —
//! `nl.load`, `nl.store`, `nisa.activation`, `nisa.tensor_scalar`, `nisa.tensor_tensor`,
//! `nisa.tensor_reduce`, `nl.divide` — every one of which is a numpy expression. A shim of
//! forty lines turns the kernel into a function this machine can run, and it is compared
//! against `run::reference_output`, the same reference the Metal, Vulkan, bang and CUDA
//! paths use.
//!
//! Unlike the CUDA emulator this one reaches the REDUCTIONS, because NKI names the
//! reduction (`tensor_reduce(np.add, t, axis=(1,))`) instead of hand-rolling it out of warp
//! shuffles. A named operation can be emulated honestly; a warp cannot be faked one thread
//! at a time. That difference is the same property `declared_semantics.rs` relies on.
//!
//! Needs a Python with numpy. The default `python3` on this machine does NOT have it and
//! `/usr/bin/python3` does, so the interpreter is PROBED rather than assumed; skips with a
//! note if none is found.

use std::process::Command;
use tile_cli::run::{error_budget, reference_output, unit_roundoff, RefOp, Shape};
use tile_cli::torchref::{Accuracy, Tolerance};

const NKI_INIT: &str = "def jit(fn):\n    return fn\n";

const NKI_LANGUAGE: &str = r#"
import numpy as np

class _MGrid:
    """`nl.mgrid[0:R, 0:C]` is used as an INDEX, so a tuple of slices is exactly right."""
    def __getitem__(self, key):
        return key

mgrid = _MGrid()

def load(x):
    return np.array(x, dtype=np.float32, copy=True)

def store(dst, val):
    # `dst` is the sliced view from `p1[nl.mgrid[...]]`; for numpy that would be a copy and
    # the write would be lost. The driver passes a recorder whose __getitem__ keeps the
    # slice, so the store lands where the kernel meant it to.
    dst.assign(val)

def divide(a, b):
    return np.divide(a, b)

def stack(arrs, axis=0):
    return np.stack(arrs, axis=axis)

def arange(n):
    return np.arange(n, dtype=np.float32)

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def zeros(shape, dtype=np.float32, **kw):
    return np.zeros(shape, dtype=dtype)

# The rest of the vocabulary, taken from what mlir_to_nki.rs can actually emit rather than
# guessed at: `grep -oE '\bnl\.[a-z_0-9]+'` over the emitter. Adding these one crash at a
# time would have worked too, but only for the ops this test happens to drive.
sbuf = "sbuf"
psum = "psum"
float32 = np.float32
float16 = np.float16

def ones(shape, dtype=np.float32, **kw):
    return np.ones(shape, dtype=dtype)

def full(shape, fill, dtype=np.float32, **kw):
    return np.full(shape, fill, dtype=dtype)

def ndarray(shape, dtype=np.float32, **kw):
    return np.zeros(shape, dtype=dtype)

def abs(x):
    return np.abs(x)

def add(a, b):
    return np.add(a, b)

def multiply(a, b):
    return np.multiply(a, b)

def maximum(a, b):
    return np.maximum(a, b)

def minimum(a, b):
    return np.minimum(a, b)

def max(x, axis=None, keepdims=False):
    return np.max(x, axis=axis, keepdims=keepdims)

def log(x, dtype=np.float32, **kw):
    return np.log(x)

def sqrt(x):
    return np.sqrt(x)

def rsqrt(x):
    # `nl.rsqrt` is one primitive; 1/sqrt(x) in the input's own dtype rounds TWICE. That
    # cost tpu's f16 rsqrt a false failure at 1.3 unit roundoffs before the same shape was
    # fixed there. It has not tripped a bound here, which is not a reason to keep a shim
    # that models the wrong number of roundings. Unchanged for f32.
    a = np.asarray(x)
    return (1.0 / np.sqrt(a.astype(np.float32))).astype(a.dtype)

def round(x):
    return np.round(x)

def clip(x, lo, hi):
    return np.clip(x, lo, hi)

def cast(x, dtype=np.float32):
    return x.astype(dtype)

def argmax(x, axis=None, keepdims=False):
    return np.argmax(x, axis=axis, keepdims=keepdims)

def argsort(x, axis=-1):
    return np.argsort(x, axis=axis)

def sort(x, axis=-1):
    return np.sort(x, axis=axis)

def concatenate(arrs, axis=0):
    return np.concatenate(arrs, axis=axis)

def program_id(axis=0):
    return 0

def sequential_range(n):
    return range(n)
"#;

const NKI_ISA: &str = r#"
import numpy as np

def activation(fn, x, dtype=np.float32, **kw):
    return fn(x).astype(dtype)

def tensor_scalar(a, op, s, dtype=np.float32, **kw):
    return op(a, s).astype(dtype)

def tensor_tensor(a, b, op, dtype=np.float32, **kw):
    return op(a, b).astype(dtype)

def tensor_reduce(op, x, axis=(1,), dtype=np.float32, negate=False, **kw):
    # keepdims, because the emitted code broadcasts the result straight back against the
    # tile: `tensor_scalar(t0, np.subtract, rowmax)` only works if the axis is kept.
    #
    # And both spellings appear: `np.add` is a ufunc and has `.reduce`; `np.max` is a
    # dispatcher and does not.
    if hasattr(op, "reduce"):
        r = op.reduce(x, axis=tuple(axis), keepdims=True)
    else:
        r = op(x, axis=tuple(axis), keepdims=True)
    return (-r if negate else r).astype(dtype)

def cast(x, dtype=np.float32, **kw):
    return x.astype(dtype)

def nc_matmul(a, b, **kw):
    return np.matmul(a, b)

def nc_transpose(a, **kw):
    return np.transpose(a)
"#;

const DRIVER: &str = r#"
import sys, importlib.util, numpy as np

class _Out:
    def __init__(self, arr):
        self.arr = arr
        self.sl = None
    def __getitem__(self, key):
        self.sl = key
        return self
    def assign(self, val):
        self.arr[self.sl] = val

def input_value(b, i):
    return ((i + 7*b) % 17) * 0.25 - 2.0 + i * 1e-4

def grid(buf, rows, cols, dt=np.float32):
    # NARROWED to the buffer's dtype on the way in, which is what `quantize_inputs`
    # mirrors on the reference side so both are given identical numbers. That was #010:
    # an f16 kernel compared against f32 values it was never handed.
    return np.array([[input_value(buf, r*cols + c) for c in range(cols)] for r in range(rows)],
                    dtype=dt)

def main():
    # path a_rows a_cols out_rows out_cols [b_rows b_cols] [--dtype half]
    a = [x for x in sys.argv if x != "--dtype-half"]
    dt = np.float16 if len(a) != len(sys.argv) else np.float32
    path = a[1]
    ar, ac, orow, ocol = int(a[2]), int(a[3]), int(a[4]), int(a[5])
    br, bc = (int(a[6]), int(a[7])) if len(a) > 7 else (0, 0)
    spec = importlib.util.spec_from_file_location("kern", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    p0 = grid(0, ar, ac, dt)
    out = np.zeros((orow, ocol), dtype=dt)
    if br:
        # The SECOND buffer uses generator index 1, exactly as reference_output does. Using
        # index 0 for both would compare two different problems and agree anyway for a
        # symmetric op, which is the worst kind of pass.
        m.k(p0, grid(1, br, bc, dt), _Out(out))
    else:
        m.k(p0, _Out(out))
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
fn nki_computes_what_the_reference_computes() {
    let Some(py) = python_with_numpy() else {
        eprintln!("emulate_nki: skipped, no Python with numpy");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emulate-nki-{}", std::process::id()));
    let pkg = dir.join("neuronxcc").join("nki");
    std::fs::create_dir_all(&pkg).expect("a scratch package");
    std::fs::write(dir.join("neuronxcc").join("__init__.py"), "").expect("write");
    std::fs::write(pkg.join("__init__.py"), NKI_INIT).expect("write");
    std::fs::write(pkg.join("language.py"), NKI_LANGUAGE).expect("write");
    std::fs::write(pkg.join("isa.py"), NKI_ISA).expect("write");
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
                .args(["-t", "nki", "-o"])
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
            .args(["-t", "nki", "-o"])
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
            .args(["-t", "nki", "-o"])
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
    // Unlike bang, nki needed no emitter fix and no shim work: it already threads
    // `dtype=np.float16` through its arms (eleven of twelve did; the twelfth, `nl.log`,
    // was fixed while auditing #024), and the shim's ops end in `.astype(dtype)` so the
    // narrowing is real rather than nominal. What was missing was only a driver that
    // builds f16 arrays, which is why this is measurement and not repair.
    //
    // The tolerance comes from `error_budget` and `unit_roundoff`. An f16 kernel differs
    // from an f32 reference by up to 2^-11 however good the lowering is, and judging it by
    // the f32 bound reports every correct kernel as broken -- which is the report
    // `unit_roundoff` exists to prevent.
    let f16_ops: &[(&str, RefOp, bool)] = &[
        ("exp", RefOp::Exp, false),
        // sigmoid is NOT here. nki decomposes it into FOUR narrowing steps --
        // multiply, exp, add, divide, each `dtype=np.float16` -- so its bound is
        // gamma_4 = 1.96e-3 and it observes 8.7e-4: correct, and above a fixed tolerance
        // that describes ONE rounding (4.883e-4).
        //
        // The contrast with msl is the interesting part and not a defect in either. msl's
        // f16 sigmoid passes the same bound because its ggml-derived kernel computes in
        // f32 internally and only the BUFFERS are f16; nki narrows at every step. Two
        // correct lowerings of one operation with different rounding behaviour, which is
        // the same distinction `error_budget` draws when it says what accumulator it
        // assumes.
        //
        // Widening the bound to admit it would fit a constant to the one op that failed.
        ("log", RefOp::Log, false),
        ("rsqrt", RefOp::Rsqrt, false),
        ("reduce_sum", RefOp::ReduceSum, true),
        ("reduce_max", RefOp::ReduceMax, true),
        ("absmax", RefOp::Absmax, true),
        // rms_norm narrows ONCE here -- measured, not assumed: the emitted Python contains
        // exactly one `dtype=np.float16`. bang's needed its accumulator fixed before it
        // could be driven; nki's did not.
        ("rms_norm", RefOp::RmsNorm, false),
    ];
    for &(rows, cols) in &shapes {
        for (name, op, reduces) in f16_ops {
            let out_cols = if *reduces { 1 } else { cols };
            let mlir = if *name == "rms_norm" {
                format!(
                    "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                     attributes {{hacc.entry}} {{\n    ^bb0:\n\
                     \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                     \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                     \x20   %e = llvm.mlir.constant(2.500000e-02 : f32) : f32\n\
                     \x20   %a = llvm.call @__tile_load_f16(%arg0, %r, %c) \
                     : (!llvm.ptr<1>, i32, i32) -> i32\n\
                     \x20   %y = llvm.call @__tile_rms_norm_f16(%a, %a, %e, %r, %c) \
                     : (i32, i32, f32, i32, i32) -> i32\n\
                     \x20   llvm.call @__tile_store_f16(%arg1, %y, %r, %c) \
                     : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                     \x20   llvm.return\n  }}\n}}\n"
                )
            } else {
                format!(
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
                )
            };
            let stem = format!("f16_{name}_{rows}x{cols}");
            let src = dir.join(format!("{stem}.mlir"));
            std::fs::write(&src, &mlir).expect("write");
            let py_out = dir.join(format!("{stem}.py"));
            let _ = std::fs::remove_file(&py_out);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "nki", "-o"])
                .arg(&py_out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !py_out.exists() {
                refused += 1;
                continue;
            }
            // No "does it name float16" check here: `dtype_is_not_ignored.rs` owns that
            // property, and owns it with the nuance this one lacked. Pinning rms_norm's
            // accumulator to f32 removed the last explicit `dtype=` from its emitted line,
            // so the text became width-independent -- CORRECTLY, because the operands
            // carry f16 from `nl.load` and the result narrows on the store. This guard
            // called that "the dtype was dropped". A property worth checking in one place,
            // with the allowlist that separates the two cases, is worth NOT checking in a
            // second place that cannot.
            let shape = if *reduces {
                Shape::RowReduce { rows, cols }
            } else {
                Shape::Rows { rows, cols }
            };
            // The eps comes back out of the generated MLIR, never from a constant repeated
            // here -- see Shape::rms_eps_from_mlir's own comment.
            let eps = Shape::rms_eps_from_mlir(&mlir).unwrap_or(1e-6);
            let want = reference_output(*op, shape, eps, "half");
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

    // The two-input f16 operations. Recorded as a harness gap rather than a property
    // until now, which is the kind of entry that should not survive long: the driver has
    // taken an optional second buffer since matmul was added, and it takes `--dtype-half`
    // since the f16 sweep above. Nothing was blocking these but the sweep itself.
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
            .args(["-t", "nki", "-o"])
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
            .args(["-t", "nki", "-o"])
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
    eprintln!("emulate_nki: {checked} kernels run and compared, {refused} refused");
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    assert!(
        checked >= 15,
        "only {checked} kernels were emitted and run -- nki is refusing nearly everything \
         and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emulated NKI kernels disagree with the reference. This is not a syntax \
         complaint: the kernel ran and computed a different answer.",
        bad.len()
    );
}

/// Which operations nki is driven for in f16, and why the rest are not.
///
/// Exhaustive over `RefOp`, like msl's and bang's, so an operation added later cannot
/// compile until someone says what nki's f16 does with it.
///
/// The held-out reasons here are all *measured*, and one of them corrects a measurement:
/// counting `dtype=np.float16` annotations undercounts narrowings, because numpy narrows
/// implicitly wherever an operand is already f16. rms_norm read as one step by that count
/// and behaved like six; the emitted line showed a 256-term sum accumulated into f16, which
/// is now pinned to f32 and drives clean.
#[test]
fn every_reference_operation_states_what_nki_f16_does_with_it() {
    fn not_driven_because(op: RefOp) -> &'static str {
        match op {
            RefOp::Exp
            | RefOp::Log
            | RefOp::Rsqrt
            | RefOp::RmsNorm
            | RefOp::ReduceSum
            | RefOp::ReduceMax
            | RefOp::Absmax
            | RefOp::Max
            | RefOp::Matmul => "",
            RefOp::Sigmoid => {
                "four narrowing steps -- multiply, exp, add, divide -- so gamma_4 = 1.96e-3; \
                 observes 8.7e-4 against a one-rounding bound of 4.883e-4. msl's f16 sigmoid \
                 passes the same bound because it computes in f32 and only its BUFFERS are \
                 f16: two correct lowerings, different rounding"
            }
            RefOp::Silu => "five narrowing steps, same reasoning as sigmoid",
            RefOp::Softmax => "four narrowing steps, same reasoning as sigmoid",
            RefOp::Neg
            | RefOp::Abs
            | RefOp::Relu
            | RefOp::Sqrt
            | RefOp::Tanh
            | RefOp::Softplus
            | RefOp::Min
            | RefOp::Matvec => "nki has no arm for this operation in any width",
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
        "RefOp changed; say what nki's f16 does with the new one"
    );
    let driven = ALL
        .iter()
        .filter(|op| not_driven_because(**op).is_empty())
        .count();
    assert!(
        driven >= 9,
        "only {driven} operations are driven in f16 on nki; the sweep has lost coverage"
    );
    for op in ALL {
        let why = not_driven_because(*op);
        if !why.is_empty() {
            eprintln!("nki f16 skips {op:?}: {why}");
        }
    }
}
