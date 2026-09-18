//! Running the MUSA backend with no MUSA, to check what it COMPUTES.
//!
//! Moore Threads' MUSA emits the same vocabulary as CUDA -- `__global__`, `threadIdx.x`,
//! `expf` -- so the identical stub answers it, under the name its preamble includes:
//! `musa_runtime.h` rather than `cuda_runtime.h`. Both tests read ONE copy of that stub
//! from `testdata/emu/`, because two copies of an emulator drift and a drifted emulator
//! reports a defect in whichever backend was checked against the older one.
//!
//! It is NOT a rename of the CUDA backend, which is the only reason this is worth running:
//! the two differ by seven to nine lines per kernel, so this checks a genuinely different
//! lowering rather than the same text twice.
//!
//! WHAT IT COVERS. The six operations MUSA lowers that do not reduce. `softmax` and
//! `rms_norm` are the other two it lowers, and both call `warp_reduce_*`, which is built
//! out of `__shfl_down_sync` across 32 lanes -- one thread at a time cannot be a warp, and
//! a stub returning the value unchanged would compute a confident wrong answer. They are
//! skipped, not faked. Everything else in the corpus MUSA has no arm for.
//!
//! Needs a host C compiler; skips with a note otherwise.

use std::process::Command;
use tile_cli::run::{reference_output, RefOp, Shape};

/// The same shim `emulate_gpu.rs` uses, written out under MUSA's include name.
const MUSA_EMU: &str = include_str!("../testdata/emu/cuda_runtime.h");

/// The driver INCLUDES the kernel rather than linking it: `threadIdx` and friends are
/// `static` in the header, so two translation units get two copies and the driver's writes
/// never reach the kernel. The symptom was quiet -- element 0 of each block came out right
/// and every other element stayed 0.0, because only thread 0 ever ran.
/// The two-input form: `k(p0, p1, p2)`, with the same block-and-thread walk.
///
/// `max` was listed as undrivable here for want of a second input buffer. That was a limit
/// of this file: the emitted kernel has taken three all along.
const DRIVER2: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "musa_runtime.h"
#include "KERNEL_SRC"
static float input_value(int b, int i) {
  return (float)(((i + 7*b) % 17) * 0.25 - 2.0) + (float)i * 1e-4f;
}
int main(int argc, char** argv) {
  (void)argc;
  int rows = atoi(argv[1]), cols = atoi(argv[2]);
  int n = rows * cols;
  float* p0 = calloc(n, sizeof(float));
  float* p1 = calloc(n, sizeof(float));
  float* p2 = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) p0[i] = input_value(0, i);
  /* Generator index 1 for the second operand, as reference_output uses. */
  for (int i = 0; i < n; i++) p1[i] = input_value(1, i);
  gridDim.x = rows; blockDim.x = cols;
  for (int b = 0; b < rows; b++) {
    blockIdx.x = b;
    for (int t = 0; t < cols; t++) { threadIdx.x = t; k(p0, p1, p2); }
  }
  for (int i = 0; i < n; i++) printf("%.9e\n", p2[i]);
  return 0;
}
"#;

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "musa_runtime.h"
#include "KERNEL_SRC"
static float input_value(int b, int i) {
  return (float)(((i + 7*b) % 17) * 0.25 - 2.0) + (float)i * 1e-4f;
}
int main(int argc, char** argv) {
  (void)argc;
  int rows = atoi(argv[1]), cols = atoi(argv[2]);
  int n = rows * cols;
  float* p0 = calloc(n, sizeof(float));
  float* p1 = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) p0[i] = input_value(0, i);
  gridDim.x = rows; blockDim.x = cols;
  for (int b = 0; b < rows; b++) {
    blockIdx.x = b;
    for (int t = 0; t < cols; t++) { threadIdx.x = t; k(p0, p1); }
  }
  for (int i = 0; i < n; i++) printf("%.9e\n", p1[i]);
  return 0;
}
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

fn kernel_mlir(op: &str, rows: usize, cols: usize) -> String {
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

#[test]
fn musa_computes_what_the_reference_computes() {
    let Some(cc) = which("clang").or_else(|| which("gcc")) else {
        eprintln!("emulate_musa: skipped, no host C compiler");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emulate-musa-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    // The emitted MUSA does `#include <musa_runtime.h>`, so the stub must answer to THAT
    // name on the include path. `musa_fp16.h` is the second include the f16 preamble adds.
    std::fs::write(dir.join("musa_runtime.h"), MUSA_EMU).expect("write stub");
    std::fs::write(
        dir.join("musa_fp16.h"),
        "#pragma once\n#include \"musa_runtime.h\"\n",
    )
    .expect("write stub");

    // Multi-row shapes are the point: a per-row defect is invisible at one row.
    let shapes = [(1usize, 64usize), (1, 256), (4, 256), (3, 129), (8, 64)];
    // ELEMENTWISE ONLY -- see the note at the top on why the reductions are absent.
    // Exactly what MUSA lowers and does not reduce. The list is short on purpose: an op
    // this backend has no arm for would be counted as "refused" and read by nobody, which
    // is the trap `emulator_coverage.rs` exists to close.
    let ops: &[(&str, RefOp)] = &[
        ("exp", RefOp::Exp),
        ("sigmoid", RefOp::Sigmoid),
        ("log", RefOp::Log),
        ("rsqrt", RefOp::Rsqrt),
        ("silu", RefOp::Silu),
    ];

    let (mut checked, mut refused) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    for &(rows, cols) in &shapes {
        for (name, op) in ops {
            let src = dir.join(format!("{name}_{rows}x{cols}.mlir"));
            std::fs::write(&src, kernel_mlir(name, rows, cols)).expect("write");
            // A FRESH path per kernel: a refused emission must not leave the previous one
            // to be compared against this op's reference.
            let cu = dir.join(format!("{name}_{rows}x{cols}.cu"));
            let _ = std::fs::remove_file(&cu);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "musa", "-o"])
                .arg(&cu)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !cu.exists() {
                refused += 1; // a refusal is an answer
                continue;
            }
            let drv = dir.join(format!("{name}_{rows}x{cols}_drv.c"));
            std::fs::write(
                &drv,
                DRIVER.replace("KERNEL_SRC", &cu.file_name().unwrap().to_string_lossy()),
            )
            .expect("write driver");
            let bin = dir.join(format!("{name}_{rows}x{cols}.bin"));
            let built = Command::new(&cc)
                .args(["-std=c11", "-I"])
                .arg(&dir)
                .args(["-x", "c"])
                .arg(&drv)
                .arg("-o")
                .arg(&bin)
                .arg("-lm")
                .output()
                .expect("the C compiler runs");
            if !built.status.success() {
                bad.push(format!(
                    "{name} {rows}x{cols}: did not build — {}",
                    String::from_utf8_lossy(&built.stderr)
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("")
                ));
                continue;
            }
            let want = reference_output(*op, Shape::Rows { rows, cols }, 1e-6, "float");
            let run = Command::new(&bin)
                .arg(rows.to_string())
                .arg(cols.to_string())
                .output()
                .expect("the emulated kernel runs");
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

    // MAX: two operands of EQUAL size. It was listed as undrivable here because this
    // file's driver passed one input buffer -- a limit of the harness, not of CUDA, whose
    // emitted kernel has taken three all along. Equal sizes are where a swapped pair hides
    // best, since max(a, b) == max(b, a); the second operand comes from generator index 1
    // so the values differ even though the operation does not care about order.
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
        let cu = dir.join(format!("max_{rows}x{cols}.cu"));
        let _ = std::fs::remove_file(&cu);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "musa", "-o"])
            .arg(&cu)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !cu.exists() {
            refused += 1;
            continue;
        }
        let drv = dir.join(format!("max_{rows}x{cols}_drv.c"));
        std::fs::write(
            &drv,
            DRIVER2.replace("KERNEL_SRC", &cu.file_name().unwrap().to_string_lossy()),
        )
        .expect("write driver");
        let bin = dir.join(format!("max_{rows}x{cols}.bin"));
        let built = Command::new(&cc)
            .args(["-std=c11", "-I"])
            .arg(&dir)
            .args(["-x", "c"])
            .arg(&drv)
            .arg("-o")
            .arg(&bin)
            .arg("-lm")
            .output()
            .expect("the C compiler runs");
        if !built.status.success() {
            bad.push(format!(
                "max {rows}x{cols}: did not build — {}",
                String::from_utf8_lossy(&built.stderr)
                    .lines()
                    .find(|l| l.contains("error"))
                    .unwrap_or("")
            ));
            continue;
        }
        let want = reference_output(RefOp::Max, Shape::Rows2 { rows, cols }, 1e-6, "float");
        let run = Command::new(&bin)
            .arg(rows.to_string())
            .arg(cols.to_string())
            .output()
            .expect("the emulated kernel runs");
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
            if (g.is_nan() && w.is_nan()) || (g.is_infinite() && w.is_infinite()) {
                continue;
            }
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

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emulate_musa: {checked} kernels run and compared, {refused} refused");
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    // Guard the vacuous pass.
    assert!(
        checked >= 25,
        "only {checked} kernels were emitted and run -- musa is refusing nearly everything \
         and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emulated MUSA kernels disagree with the reference. This is not a syntax \
         complaint: the kernel ran and computed a different answer.",
        bad.len()
    );
}
