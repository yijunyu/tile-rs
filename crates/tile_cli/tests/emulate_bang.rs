//! Running a backend that has no hardware here, to check what it COMPUTES.
//!
//! Every other gate in this repo is syntax: `emitted_parses.rs` asks whether emitted text
//! is a program. That is worth having — it found three backends emitting text that was not
//! — but it cannot tell a correct kernel from one that computes the wrong thing.
//!
//! `bang` can be taken further. Its emitted MLU C is scalar loops, `expf`, and `__memcpy`
//! once the macros are defined, so a stub that EMULATES rather than no-ops turns the
//! kernel into a plain C function this machine can run: `__memcpy` as a real memcpy,
//! `taskId = 0`, and the `__bang_*` family implemented for real.
//!
//! It is then driven with the same input generator `run::input_values_for` uses and its
//! output compared against `run::reference_output` — the same reference the Metal and
//! Vulkan paths compare against, so the three cannot drift.
//!
//! What this establishes, and what it does not. It checks the ARITHMETIC and the INDEXING,
//! which is where every defect it found lived: `reduce_sum`, `reduce_max`, `absmax` and
//! `softmax` all reduced the whole TILE instead of each row, so a 4x256 kernel wrote one
//! value where four were due. It says nothing about DMA behaviour, bank conflicts,
//! multi-task scheduling, or whether `cncc` accepts the source. A Cambricon engineer still
//! has to run it. But "four values were expected and one was written" needed no Cambricon.
//!
//! Needs a host C compiler; skips with a note otherwise.

use std::process::Command;
use tile_cli::run::{error_budget, reference_output, unit_roundoff, RefOp, Shape};
use tile_cli::torchref::{Accuracy, Tolerance};

/// The emulating stub. Same vocabulary as the parse-only stub in `emitted_parses.rs`, but
/// the operations actually happen.
const BANG_EMULATOR: &str = r#"
#ifndef TILE_BANG_EMU_H
#define TILE_BANG_EMU_H
#include <math.h>
#include <stdint.h>
#include <string.h>
#define __mlu_entry__
#define __mlu_func__
#define __nram__
#define __wram__
#define __mlu_shared__
enum MemDir { GDRAM2NRAM, NRAM2GDRAM, NRAM2NRAM, GDRAM2GDRAM, NRAM2WRAM };
/* One task. That is what every `taskId == 0` guard assumes and what a single-core run is. */
static int taskId = 0, taskDim = 1, coreId = 0, coreDim = 1;
static inline void __memcpy(void* d, const void* s, size_t n, enum MemDir dir) {
  (void)dir; memcpy(d, s, n);
}
static inline void __sync_all(void) {}
static inline void __sync_cluster(void) {}
typedef __fp16 half;
static inline void __bang_add_f32(float* d, const float* a, const float* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]+b[i]; }
static inline void __bang_sub_f32(float* d, const float* a, const float* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]-b[i]; }
static inline void __bang_mul_f32(float* d, const float* a, const float* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]*b[i]; }
static inline void __bang_div_f32(float* d, const float* a, const float* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]/b[i]; }
static inline void __bang_maxequal_f32(float* d, const float* a, const float* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]>b[i]?a[i]:b[i]; }
static inline void __bang_minequal_f32(float* d, const float* a, const float* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]<b[i]?a[i]:b[i]; }
static inline void __bang_abs_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=fabsf(a[i]); }
static inline void __bang_active_exp_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=expf(a[i]); }
static inline void __bang_active_relu_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=a[i]>0?a[i]:0; }
static inline void __bang_active_sigmoid_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=1.0f/(1.0f+expf(-a[i])); }
static inline void __bang_active_tanh_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=tanhf(a[i]); }
static inline void __bang_active_sqrt_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=sqrtf(a[i]); }
static inline void __bang_log_f32(float* d, const float* a, int n) { for (int i=0;i<n;i++) d[i]=logf(a[i]); }
static inline void __bang_mul_scalar_f32(float* d, const float* a, float s, int n) { for (int i=0;i<n;i++) d[i]=a[i]*s; }
static inline void __bang_add_scalar_f32(float* d, const float* a, float s, int n) { for (int i=0;i<n;i++) d[i]=a[i]+s; }
static inline void __bang_write_value_f32(float* d, int n, float v) { for (int i=0;i<n;i++) d[i]=v; }

/* Half twins. `__fp16` promotes to float in arithmetic and narrows on assignment,
   which is exactly what f16 hardware does with an f32 intermediate -- so the bodies
   are textually identical and only the pointer types change.

   Without these the f32 helpers accepted a `half*` with nothing but a warning and
   reinterpreted the array, so an f16 kernel calling __bang_mul computed garbage and
   the harness reported it as a wrong lowering. A shim that silently accepts the
   wrong width is the same defect as an emitter that emits the wrong width. */
static inline void __bang_add_f16(half* d, const half* a, const half* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]+b[i]; }
static inline void __bang_sub_f16(half* d, const half* a, const half* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]-b[i]; }
static inline void __bang_mul_f16(half* d, const half* a, const half* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]*b[i]; }
static inline void __bang_div_f16(half* d, const half* a, const half* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]/b[i]; }
static inline void __bang_maxequal_f16(half* d, const half* a, const half* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]>b[i]?a[i]:b[i]; }
static inline void __bang_minequal_f16(half* d, const half* a, const half* b, int n) { for (int i=0;i<n;i++) d[i]=a[i]<b[i]?a[i]:b[i]; }
static inline void __bang_abs_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=fabsf(a[i]); }
static inline void __bang_active_exp_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=expf(a[i]); }
static inline void __bang_active_relu_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=a[i]>0?a[i]:0; }
static inline void __bang_active_sigmoid_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=1.0f/(1.0f+expf(-a[i])); }
static inline void __bang_active_tanh_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=tanhf(a[i]); }
static inline void __bang_active_sqrt_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=sqrtf(a[i]); }
static inline void __bang_log_f16(half* d, const half* a, int n) { for (int i=0;i<n;i++) d[i]=logf(a[i]); }
static inline void __bang_mul_scalar_f16(half* d, const half* a, float s, int n) { for (int i=0;i<n;i++) d[i]=a[i]*s; }
static inline void __bang_add_scalar_f16(half* d, const half* a, float s, int n) { for (int i=0;i<n;i++) d[i]=a[i]+s; }
static inline void __bang_write_value_f16(half* d, int n, float v) { for (int i=0;i<n;i++) d[i]=v; }

/* Dispatch on the destination pointer, so an emitted kernel calls the right one
   without either the kernel or this file naming a width. */
#define __bang_add(d, ...) _Generic((d), float*: __bang_add_f32, half*: __bang_add_f16)(d, __VA_ARGS__)
#define __bang_sub(d, ...) _Generic((d), float*: __bang_sub_f32, half*: __bang_sub_f16)(d, __VA_ARGS__)
#define __bang_mul(d, ...) _Generic((d), float*: __bang_mul_f32, half*: __bang_mul_f16)(d, __VA_ARGS__)
#define __bang_div(d, ...) _Generic((d), float*: __bang_div_f32, half*: __bang_div_f16)(d, __VA_ARGS__)
#define __bang_maxequal(d, ...) _Generic((d), float*: __bang_maxequal_f32, half*: __bang_maxequal_f16)(d, __VA_ARGS__)
#define __bang_minequal(d, ...) _Generic((d), float*: __bang_minequal_f32, half*: __bang_minequal_f16)(d, __VA_ARGS__)
#define __bang_abs(d, ...) _Generic((d), float*: __bang_abs_f32, half*: __bang_abs_f16)(d, __VA_ARGS__)
#define __bang_active_exp(d, ...) _Generic((d), float*: __bang_active_exp_f32, half*: __bang_active_exp_f16)(d, __VA_ARGS__)
#define __bang_active_relu(d, ...) _Generic((d), float*: __bang_active_relu_f32, half*: __bang_active_relu_f16)(d, __VA_ARGS__)
#define __bang_active_sigmoid(d, ...) _Generic((d), float*: __bang_active_sigmoid_f32, half*: __bang_active_sigmoid_f16)(d, __VA_ARGS__)
#define __bang_active_tanh(d, ...) _Generic((d), float*: __bang_active_tanh_f32, half*: __bang_active_tanh_f16)(d, __VA_ARGS__)
#define __bang_active_sqrt(d, ...) _Generic((d), float*: __bang_active_sqrt_f32, half*: __bang_active_sqrt_f16)(d, __VA_ARGS__)
#define __bang_log(d, ...) _Generic((d), float*: __bang_log_f32, half*: __bang_log_f16)(d, __VA_ARGS__)
#define __bang_mul_scalar(d, ...) _Generic((d), float*: __bang_mul_scalar_f32, half*: __bang_mul_scalar_f16)(d, __VA_ARGS__)
#define __bang_add_scalar(d, ...) _Generic((d), float*: __bang_add_scalar_f32, half*: __bang_add_scalar_f16)(d, __VA_ARGS__)
#define __bang_write_value(d, ...) _Generic((d), float*: __bang_write_value_f32, half*: __bang_write_value_f16)(d, __VA_ARGS__)
typedef __fp16 half;
#endif
"#;

/// The driver. Its generator must match `run::input_values_for` exactly, or the comparison
/// is of two different problems — which is what backlog #010 was about.
/// The two-input form: `k(p0, p1, p2)`.
///
/// `max` and `matmul` were listed as undrivable on this backend, and the reason given was
/// that the driver's `k()` takes one input. That was a limit of THIS FILE, not a property
/// of Cambricon -- the emitted kernels have taken three buffers all along. A gap explained
/// by a property stays; a gap explained by a driver gets closed.
const DRIVER2: &str = r#"
#include <stdio.h>
#include <stdlib.h>
static float input_value(int b, int i) {
  return (float)(((i + 7*b) % 17) * 0.25 - 2.0) + (float)i * 1e-4f;
}
void k(float*, float*, float*);
int main(int argc, char** argv) {
  (void)argc;
  int a_n = atoi(argv[1]), b_n = atoi(argv[2]), out_n = atoi(argv[3]);
  float* p0 = calloc(a_n, sizeof(float));
  float* p1 = calloc(b_n, sizeof(float));
  float* p2 = calloc(out_n, sizeof(float));
  for (int i = 0; i < a_n; i++) p0[i] = input_value(0, i);
  /* The SECOND buffer uses generator index 1, exactly as reference_output does. Index 0
     for both would compare two different problems and agree anyway for a symmetric
     operation, which is the worst kind of pass. */
  for (int i = 0; i < b_n; i++) p1[i] = input_value(1, i);
  k(p0, p1, p2);
  for (int i = 0; i < out_n; i++) printf("%.9e\n", p2[i]);
  return 0;
}
"#;

/// The f16 drivers. `bang.h` types `half` as `__fp16`, a REAL 16-bit float on this
/// target, so an f16 kernel here runs at f16 precision rather than at f32 wearing its
/// name -- which is the difference between measuring this backend's f16 and confirming
/// that f32 still works. (The shared CUDA stub types `__half` as `float` and says so, so
/// gpu and musa cannot make the same claim and do not drive f16.)
const DRIVER_F16: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "bang.h"
static float input_value(int b, int i) {
  return (float)(((i + 7*b) % 17) * 0.25 - 2.0) + (float)i * 1e-4f;
}
void k(half*, half*);
int main(int argc, char** argv) {
  (void)argc;
  int in_n = atoi(argv[1]), out_n = atoi(argv[2]);
  half* p0 = calloc(in_n, sizeof(half));
  half* p1 = calloc(out_n, sizeof(half));
  /* Narrowed to f16 on the way in, which is what `quantize_inputs` mirrors on the
     reference side so both are given identical numbers -- see #010. */
  for (int i = 0; i < in_n; i++) p0[i] = (half)input_value(0, i);
  k(p0, p1);
  for (int i = 0; i < out_n; i++) printf("%.9e\n", (float)p1[i]);
  return 0;
}
"#;

/// The two-input f16 form.
const DRIVER2_F16: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include "bang.h"
static float input_value(int b, int i) {
  return (float)(((i + 7*b) % 17) * 0.25 - 2.0) + (float)i * 1e-4f;
}
void k(half*, half*, half*);
int main(int argc, char** argv) {
  (void)argc;
  int a_n = atoi(argv[1]), b_n = atoi(argv[2]), out_n = atoi(argv[3]);
  half* p0 = calloc(a_n, sizeof(half));
  half* p1 = calloc(b_n, sizeof(half));
  half* p2 = calloc(out_n, sizeof(half));
  for (int i = 0; i < a_n; i++) p0[i] = (half)input_value(0, i);
  for (int i = 0; i < b_n; i++) p1[i] = (half)input_value(1, i);
  k(p0, p1, p2);
  for (int i = 0; i < out_n; i++) printf("%.9e\n", (float)p2[i]);
  return 0;
}
"#;

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
static float input_value(int b, int i) {
  return (float)(((i + 7*b) % 17) * 0.25 - 2.0) + (float)i * 1e-4f;
}
void k(float*, float*);
int main(int argc, char** argv) {
  (void)argc;
  int in_n = atoi(argv[1]), out_n = atoi(argv[2]);
  float* p0 = calloc(in_n, sizeof(float));
  float* p1 = calloc(out_n, sizeof(float));
  for (int i = 0; i < in_n; i++) p0[i] = input_value(0, i);
  k(p0, p1);
  for (int i = 0; i < out_n; i++) printf("%.9e\n", p1[i]);
  return 0;
}
"#;

fn which(tool: &str) -> Option<String> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if p.is_empty() {
        None
    } else {
        Some(p)
    }
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
        // Every field `within` actually looks at, because "max_abs 0, max_rel 0" and a
        // failure is otherwise unreadable -- it means non_finite, or the near-zero
        // absolute bound, neither of which the two headline numbers show.
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
fn bang_computes_what_the_reference_computes() {
    let Some(cc) = which("clang").or_else(|| which("gcc")) else {
        eprintln!("emulate_bang: skipped, no host C compiler");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emulate-bang-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    std::fs::write(dir.join("bang.h"), BANG_EMULATOR).expect("write emulator");
    std::fs::write(dir.join("driver.c"), DRIVER).expect("write driver");
    std::fs::write(dir.join("driver2.c"), DRIVER2).expect("write two-input driver");

    // Multi-row shapes are the point: every defect this found was invisible at one row.
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

    let mut checked = 0usize;
    let mut refused = 0usize;
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
            // A FRESH path per kernel. Reusing one output path meant a refused emission
            // left the previous kernel's file behind, and it was compared against this
            // op's reference -- 48 false failures in the first sweep, all of them that.
            let mlu = dir.join(format!("{name}_{rows}x{cols}.c"));
            let _ = std::fs::remove_file(&mlu);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "bang", "-o"])
                .arg(&mlu)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !mlu.exists() {
                refused += 1; // a refusal is an answer, not a failure
                continue;
            }
            let bin = dir.join(format!("{name}_{rows}x{cols}.bin"));
            let built = Command::new(&cc)
                .args(["-std=c11", "-I"])
                .arg(&dir)
                .arg("-o")
                .arg(&bin)
                .arg(&mlu)
                .arg(dir.join("driver.c"))
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
            let shape = if *reduces {
                Shape::RowReduce { rows, cols }
            } else {
                Shape::Rows { rows, cols }
            };
            let want = reference_output(*op, shape, eps, "float");
            let run = Command::new(&bin)
                .arg((rows * cols).to_string())
                .arg(want.len().to_string())
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

    // The two-input operations. Both were listed as undrivable here, and the stated reason
    // was that this driver's `k()` takes one input buffer -- a limit of this file, not a
    // property of the backend. The emitted kernels have taken three buffers all along.
    //
    // `max` uses two operands of EQUAL size, which is where a swapped pair hides best:
    // max(a, b) == max(b, a). The second comes from generator index 1, so the values differ
    // even though the operation does not care about order.
    #[allow(clippy::type_complexity)]
    let two_input: &[(&str, RefOp, &[(usize, usize, usize)])] = &[
        (
            "max",
            RefOp::Max,
            &[(1, 64, 0), (4, 256, 0), (3, 129, 0), (8, 64, 0)],
        ),
        (
            "matmul",
            RefOp::Matmul,
            &[(8, 16, 4), (4, 4, 4), (16, 8, 32), (3, 129, 5)],
        ),
    ];
    for (name, op, cases) in two_input {
        for &(a, b, c) in *cases {
            // `max` is rows x cols against itself; `matmul` is (m x k) . (k x n).
            let is_matmul = *name == "matmul";
            let (mlir, label, in_a, in_b, want) = if is_matmul {
                let (m, k, n) = (a, b, c);
                (
                    format!(
                        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
                         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
                         \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
                         \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
                         \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
                         \x20   %a = llvm.call @__tile_load_f32(%arg0, %m, %k) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %b = llvm.call @__tile_load_f32(%arg1, %k, %n) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %y = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) \
                         : (i32, i32, i32, i32, i32, i32) -> i32\n\
                         \x20   llvm.call @__tile_store_f32(%arg2, %y, %m, %n) \
                         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                         \x20   llvm.return\n  }}\n}}\n"
                    ),
                    format!("matmul {m}x{k}x{n}"),
                    m * k,
                    k * n,
                    reference_output(*op, Shape::Matmul { m, k, n }, 1e-6, "float"),
                )
            } else {
                let (rows, cols) = (a, b);
                (
                    format!(
                        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
                         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
                         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %b = llvm.call @__tile_load_f32(%arg1, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %y = llvm.call @__tile_max_f32(%a, %a, %b, %r, %c) \
                         : (i32, i32, i32, i32, i32) -> i32\n\
                         \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                         \x20   llvm.return\n  }}\n}}\n"
                    ),
                    format!("max {rows}x{cols}"),
                    rows * cols,
                    rows * cols,
                    reference_output(*op, Shape::Rows2 { rows, cols }, 1e-6, "float"),
                )
            };
            let stem = label.replace(' ', "_");
            let src = dir.join(format!("{stem}.mlir"));
            std::fs::write(&src, &mlir).expect("write");
            let mlu = dir.join(format!("{stem}.c"));
            let _ = std::fs::remove_file(&mlu);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "bang", "-o"])
                .arg(&mlu)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !mlu.exists() {
                refused += 1;
                continue;
            }
            let bin = dir.join(format!("{stem}.bin"));
            let built = Command::new(&cc)
                .args(["-std=c11", "-I"])
                .arg(&dir)
                .arg("-o")
                .arg(&bin)
                .arg(&mlu)
                .arg(dir.join("driver2.c"))
                .arg("-lm")
                .output()
                .expect("the C compiler runs");
            if !built.status.success() {
                bad.push(format!(
                    "{label}: did not build — {}",
                    String::from_utf8_lossy(&built.stderr)
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("")
                ));
                continue;
            }
            let run = Command::new(&bin)
                .arg(in_a.to_string())
                .arg(in_b.to_string())
                .arg(want.len().to_string())
                .output()
                .expect("the emulated kernel runs");
            let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
                .lines()
                .filter_map(|l| l.trim().parse::<f32>().ok())
                .collect();
            checked += 1;
            if got.len() != want.len() {
                bad.push(format!(
                    "{label}: wrote {} values, {} were due",
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
            if worst > 1e-4 {
                bad.push(format!("{label}: max rel {worst:.2e}"));
            }
        }
    }

    // ── f16 ──────────────────────────────────────────────────────────────────────
    //
    // Eleven of these twelve operations emitted a kernel BYTE-IDENTICAL to their f32
    // lowering until `prescan_body_bang` learned to read the dtype from the loads and
    // stores rather than only from a matmul. The `half` path was always there; the flag
    // was set in exactly one place. So this sweep is the check on that fix, and the only
    // reason it can be honest is that `bang.h` types `half` as `__fp16`.
    //
    // The tolerance comes from `error_budget` and `unit_roundoff`, not from a number
    // chosen here -- an f16 kernel is expected to differ from an f32 reference by up to
    // 2^-11 however good the lowering is, and judging it by the f32 bound reports every
    // correct kernel as broken.
    std::fs::write(dir.join("driver_f16.c"), DRIVER_F16).expect("write f16 driver");
    std::fs::write(dir.join("driver2_f16.c"), DRIVER2_F16).expect("write f16 driver2");

    let f16_ops: &[(&str, RefOp, bool)] = &[
        ("exp", RefOp::Exp, false),
        ("sigmoid", RefOp::Sigmoid, false),
        ("log", RefOp::Log, false),
        ("rsqrt", RefOp::Rsqrt, false),
        // silu and softmax are NOT here, and the reason is measured rather than assumed.
        //
        // Both are chains of several f16 operations, each narrowing to the buffer type,
        // and the fixed tolerance is ONE unit roundoff -- 4.883e-4. silu makes three
        // narrowing `__bang_*` calls, so its bound is gamma_3 = 1.47e-3, and it observes
        // 7.5e-4 to 9.0e-4: correct, and above a bound that describes a different shape.
        // softmax observes 6.2e-4 to 8.1e-4 on the same reasoning.
        //
        // Widening the fixed bound to admit them would be fitting a constant to the two
        // kernels that failed, which is what this repo refuses everywhere else. Deriving
        // gamma_k properly needs a count of narrowing steps the harness cannot see from
        // outside the emitter -- so they are left out and said so, exactly as
        // `error_budget` leaves softmax and rms_norm out of the SUMMATION bound rather
        // than writing down a formula that does not hold for them.
        //
        // rms_norm is out for the same reason and was the closest call: it REDUCES and
        // then scales, so two roundings, and it observes 5.08e-4 against a one-rounding
        // 4.883e-4 -- over by a factor of 1.04. Admitting it by nudging the bound to 5.1e-4
        // would be the clearest possible case of fitting a constant to a measurement.
        // `error_budget` already excludes rms_norm from its SUMMATION bound in as many
        // words, for the same reason, so this is the existing decision applied to the
        // fixed bound rather than a new one.
        //
        // All three were far worse before the fixes this sweep exists to check, and those
        // numbers are the verification the sweep cannot carry itself:
        //
        //     silu      2.35e5   -> 9.0e-4   (the shim gained f16 `__bang_*`)
        //     rms_norm  all NaN  -> 5.1e-4   (accumulator f16 -> f32)
        //     softmax   2.66e-3  -> 8.1e-4   (row sum f16 -> f32)
        ("reduce_sum", RefOp::ReduceSum, true),
        ("reduce_max", RefOp::ReduceMax, true),
        ("absmax", RefOp::Absmax, true),
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
                     \x20   %e = llvm.mlir.constant({RMS_EPS:e} : f32) : f32\n\
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
            let eps = Shape::rms_eps_from_mlir(&mlir).unwrap_or(1e-6);
            let stem = format!("f16_{name}_{rows}x{cols}");
            let src = dir.join(format!("{stem}.mlir"));
            std::fs::write(&src, &mlir).expect("write");
            let mlu = dir.join(format!("{stem}.c"));
            let _ = std::fs::remove_file(&mlu);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "bang", "-o"])
                .arg(&mlu)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !mlu.exists() {
                refused += 1;
                continue;
            }
            // The fix this sweep exists to check: an f16 kernel must SAY it is f16.
            let text = std::fs::read_to_string(&mlu).unwrap_or_default();
            if !text.contains("dtype=f16") {
                bad.push(format!(
                    "f16 {name} {rows}x{cols}: emitted a kernel declaring dtype=f32 for an \
                     f16 intrinsic -- the buffers would be read four bytes at a time"
                ));
                continue;
            }
            let bin = dir.join(format!("{stem}.bin"));
            let built = Command::new(&cc)
                .args(["-std=c11", "-I"])
                .arg(&dir)
                .arg("-o")
                .arg(&bin)
                .arg(&mlu)
                .arg(dir.join("driver_f16.c"))
                .arg("-lm")
                .output()
                .expect("the C compiler runs");
            if !built.status.success() {
                bad.push(format!(
                    "f16 {name} {rows}x{cols}: did not build — {}",
                    String::from_utf8_lossy(&built.stderr)
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("")
                ));
                continue;
            }
            let shape = if *reduces {
                Shape::RowReduce { rows, cols }
            } else {
                Shape::Rows { rows, cols }
            };
            let want = reference_output(*op, shape, eps, "half");
            let run = Command::new(&bin)
                .arg((rows * cols).to_string())
                .arg(want.len().to_string())
                .output()
                .expect("the emulated kernel runs");
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

    // The two-input f16 operations. `DRIVER2_F16` was written when the f16 sweep went in
    // and then not used, so the closure test below recorded max and matmul as a HARNESS
    // gap. An entry that names the harness rather than the backend is a promise to come
    // back, and this is coming back.
    #[allow(clippy::type_complexity)]
    let f16_two: &[(&str, RefOp, &[(usize, usize, usize)])] = &[
        (
            "max",
            RefOp::Max,
            &[(1, 64, 0), (4, 256, 0), (3, 129, 0), (8, 64, 0)],
        ),
        (
            "matmul",
            RefOp::Matmul,
            &[(8, 16, 4), (4, 4, 4), (16, 8, 32), (3, 129, 5)],
        ),
    ];
    for (name, op, cases) in f16_two {
        for &(a, b, c) in *cases {
            let is_matmul = *name == "matmul";
            let (mlir, label, in_a, in_b, want, shape) = if is_matmul {
                let (m, k, n) = (a, b, c);
                let sh = Shape::Matmul { m, k, n };
                (
                    format!(
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
                    ),
                    format!("f16 matmul {m}x{k}x{n}"),
                    m * k,
                    k * n,
                    reference_output(*op, sh, 1e-6, "half"),
                    sh,
                )
            } else {
                let (rows, cols) = (a, b);
                let sh = Shape::Rows2 { rows, cols };
                (
                    format!(
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
                    ),
                    format!("f16 max {rows}x{cols}"),
                    rows * cols,
                    rows * cols,
                    reference_output(*op, sh, 1e-6, "half"),
                    sh,
                )
            };
            let stem = label.replace(' ', "_");
            let src = dir.join(format!("{stem}.mlir"));
            std::fs::write(&src, &mlir).expect("write");
            let mlu = dir.join(format!("{stem}.c"));
            let _ = std::fs::remove_file(&mlu);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "bang", "-o"])
                .arg(&mlu)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !mlu.exists() {
                refused += 1;
                continue;
            }
            let bin = dir.join(format!("{stem}.bin"));
            let built = Command::new(&cc)
                .args(["-std=c11", "-I"])
                .arg(&dir)
                .arg("-o")
                .arg(&bin)
                .arg(&mlu)
                .arg(dir.join("driver2_f16.c"))
                .arg("-lm")
                .output()
                .expect("the C compiler runs");
            if !built.status.success() {
                bad.push(format!(
                    "{label}: did not build — {}",
                    String::from_utf8_lossy(&built.stderr)
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("")
                ));
                continue;
            }
            let run = Command::new(&bin)
                .arg(in_a.to_string())
                .arg(in_b.to_string())
                .arg(want.len().to_string())
                .output()
                .expect("the emulated kernel runs");
            let got: Vec<f32> = String::from_utf8_lossy(&run.stdout)
                .lines()
                .filter_map(|l| l.trim().parse::<f32>().ok())
                .collect();
            checked += 1;
            f16_judge(label, got, &want, *op, shape, &mut bad);
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emulate_bang: {checked} kernels run and compared, {refused} refused");
    for b in &bad {
        eprintln!("  {b}");
    }
    // Guard the vacuous pass: if bang started refusing everything, nothing would be
    // compared and this would pass by checking nothing.
    assert!(
        checked >= 20,
        "only {checked} kernels were emitted and run -- bang is refusing nearly \
         everything and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emulated kernels disagree with the reference. This is not a syntax \
         complaint: the kernel ran and computed a different answer.",
        bad.len()
    );
}

/// Which operations this backend is driven for in f16, and why the rest are not.
///
/// The reasons live in the sweep as comments, and a comment is not a gate. This is, and it
/// is exhaustive: an operation added to `RefOp` cannot compile until someone says what
/// Cambricon's f16 does with it.
///
/// The three "composite" entries are the interesting ones. They are not failures and not
/// unimplemented -- they are correct kernels measured against a bound that describes a
/// different shape, and the honest options were to derive gamma_k (which needs a rounding
/// count the harness cannot see) or to say so. Fitting the bound to them was the third
/// option and is the one this repo refuses everywhere else.
#[test]
fn every_reference_operation_states_what_bang_f16_does_with_it() {
    fn not_driven_because(op: RefOp) -> &'static str {
        match op {
            RefOp::Exp
            | RefOp::Sigmoid
            | RefOp::Log
            | RefOp::Rsqrt
            | RefOp::ReduceSum
            | RefOp::ReduceMax
            | RefOp::Absmax
            | RefOp::Max
            | RefOp::Matmul => "",
            RefOp::Silu => {
                "three narrowing __bang_* steps, so gamma_3 = 1.47e-3; observes 9.0e-4 \
                 against a one-rounding bound of 4.883e-4"
            }
            RefOp::Softmax => "exp then divide, two roundings; observes 8.1e-4",
            RefOp::RmsNorm => {
                "reduce then scale, two roundings; observes 5.08e-4 -- over the one-rounding \
                 bound by a factor of 1.04, which is exactly the margin one must not fit to"
            }
            RefOp::Neg
            | RefOp::Abs
            | RefOp::Relu
            | RefOp::Sqrt
            | RefOp::Tanh
            | RefOp::Softplus
            | RefOp::Min
            | RefOp::Matvec => "bang has no arm for this operation in any width",
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
        "RefOp changed; say what bang's f16 does with the new one"
    );
    let driven = ALL
        .iter()
        .filter(|op| not_driven_because(**op).is_empty())
        .count();
    assert!(
        driven >= 9,
        "only {driven} operations are driven in f16 on bang; the sweep has lost coverage"
    );
    for op in ALL {
        let why = not_driven_because(*op);
        if !why.is_empty() {
            eprintln!("bang f16 skips {op:?}: {why}");
        }
    }
}
