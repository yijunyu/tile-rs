//! What each emulated backend is checked for, and what it is not — enforced, not described.
//!
//! Seven backends are checked for what they COMPUTE: bang, gpu, musa, nki and tpu through
//! shims, msl on this machine's GPU, and linalg through LLVM's own lowering and JIT. Each drives a SUBSET of the twenty operations the harness can
//! produce a reference for, and until now nothing said which subset or why. The emulators
//! print a "refused" count — `emulate_gpu: 20 kernels run, 30 refused` — and that number
//! was read by nobody. It turned out to mean that six of `emulate_gpu`'s ten listed
//! operations are not lowered by the CUDA backend at all.
//!
//! So the subset is stated here per backend, in three categories, and every entry is
//! CHECKED by emitting the kernel:
//!
//! * `Driven` — that backend's emulator runs it and compares. Emission must succeed, and
//!   the emulator's own source must name it; if either stops being true, that emulator is
//!   silently covering less than it claims.
//! * `LoweredButNot` — the backend lowers it and the emulator still cannot drive it. The
//!   reason must be a PROPERTY, not a plan. Seven entries once sat here saying "this
//!   driver's k() takes one input" — a fact about THIS FILE, not about any backend. All
//!   seven are now closed by giving the drivers a second input buffer, and closing them
//!   turned up a heap overflow in bang's matmul. What is left is the real kind: one thread
//!   cannot be a warp. Emission must succeed, and the emulator must NOT name it.
//! * `NotLowered` — the backend has no arm, so there is nothing to drive. Emission must
//!   FAIL. When a backend gains an arm this test goes red and says so, which is the only
//!   way new capability gets noticed by the thing that measures capability.
//!
//! The match over `RefOp` is exhaustive per backend, so adding an operation cannot compile
//! until someone has decided what all five do with it.

use std::process::Command;
use tile_cli::run::RefOp;

#[derive(Clone, Copy, Debug)]
enum Cover {
    Driven,
    LoweredButNot(&'static str),
    NotLowered,
}
use Cover::*;

/// CUDA. Its reductions are hand-rolled from `__shfl_down_sync` across 32 lanes, and the
/// emulator runs one thread at a time: a stub returning the value unchanged would compute
/// a confident wrong answer, so those are refused rather than faked.
fn gpu(op: RefOp) -> Cover {
    match op {
        RefOp::Exp | RefOp::Sigmoid | RefOp::Log | RefOp::Rsqrt | RefOp::Silu | RefOp::Max => {
            Driven
        }
        RefOp::Softmax | RefOp::RmsNorm | RefOp::ReduceSum | RefOp::ReduceMax | RefOp::Absmax => {
            LoweredButNot("lowers to __shfl_down_sync; one thread cannot be a warp")
        }
        RefOp::Matmul => LoweredButNot(
            "lowers to a 2-D grid with shared-memory tiles; one thread is not a block",
        ),
        RefOp::Neg
        | RefOp::Abs
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Tanh
        | RefOp::Softplus
        | RefOp::Min
        | RefOp::Matvec => NotLowered,
    }
}

/// Cambricon. Emulated through a `bang.h` that implements the `__bang_*` vocabulary, so
/// its named reductions are reachable where CUDA's warp shuffles are not.
fn bang(op: RefOp) -> Cover {
    match op {
        RefOp::Exp
        | RefOp::Neg
        | RefOp::Sigmoid
        | RefOp::Log
        | RefOp::Rsqrt
        | RefOp::Silu
        | RefOp::Softmax
        | RefOp::ReduceSum
        | RefOp::ReduceMax
        | RefOp::Absmax
        | RefOp::RmsNorm
        | RefOp::Max
        | RefOp::Matmul => Driven,
        RefOp::Abs
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Tanh
        | RefOp::Softplus
        | RefOp::Min
        | RefOp::Matvec => NotLowered,
    }
}

/// AWS Neuron. Answered with numpy, and its reductions are NAMED operations
/// (`nisa.tensor_reduce`) rather than hand-rolled lane arithmetic, which is why this
/// reaches reductions and the CUDA emulator does not.
fn nki(op: RefOp) -> Cover {
    match op {
        RefOp::Exp
        | RefOp::Sigmoid
        | RefOp::Log
        | RefOp::Rsqrt
        | RefOp::Silu
        | RefOp::Softmax
        | RefOp::ReduceSum
        | RefOp::ReduceMax
        | RefOp::Absmax
        | RefOp::RmsNorm
        | RefOp::Matmul
        | RefOp::Max => Driven,
        RefOp::Neg
        | RefOp::Abs
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Tanh
        | RefOp::Softplus
        | RefOp::Min
        | RefOp::Matvec => NotLowered,
    }
}

/// TPU via Pallas. Same shape as Neuron: `jnp.max`, `jnp.mean` are named, so reductions
/// are reachable.
fn tpu(op: RefOp) -> Cover {
    match op {
        RefOp::Exp
        | RefOp::Sigmoid
        | RefOp::Log
        | RefOp::Rsqrt
        | RefOp::Silu
        | RefOp::Softmax
        | RefOp::ReduceSum
        | RefOp::ReduceMax
        | RefOp::Absmax
        | RefOp::RmsNorm
        | RefOp::Matmul
        | RefOp::Max => Driven,
        RefOp::Neg
        | RefOp::Abs
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Tanh
        | RefOp::Softplus
        | RefOp::Min
        | RefOp::Matvec => NotLowered,
    }
}

/// Metal, run on the GPU itself rather than emulated. The only backend with nothing left
/// in the other two categories.
fn msl(op: RefOp) -> Cover {
    match op {
        RefOp::Exp
        | RefOp::Neg
        | RefOp::Abs
        | RefOp::Sigmoid
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Log
        | RefOp::Tanh
        | RefOp::Rsqrt
        | RefOp::Silu
        | RefOp::Softplus
        | RefOp::Softmax
        | RefOp::ReduceSum
        | RefOp::ReduceMax
        | RefOp::Absmax
        | RefOp::Min
        | RefOp::Max
        | RefOp::Matvec
        | RefOp::Matmul
        | RefOp::RmsNorm => Driven,
    }
}

/// How one backend classifies every operation.
type Classify = fn(RefOp) -> Cover;

/// Moore Threads' MUSA. Emits the same vocabulary as CUDA and is answered by the same
/// shim under a different include name, but is not a rename -- the two differ by seven to
/// nine lines per kernel. It lowers exactly the set gpu does, and holds back the same two
/// for the same reason.
fn musa(op: RefOp) -> Cover {
    match op {
        RefOp::Exp | RefOp::Sigmoid | RefOp::Log | RefOp::Rsqrt | RefOp::Silu | RefOp::Max => {
            Driven
        }
        RefOp::Softmax | RefOp::RmsNorm | RefOp::ReduceSum | RefOp::ReduceMax | RefOp::Absmax => {
            LoweredButNot(
                "calls warp_reduce_*, built on __shfl_down_sync; one thread is not a warp",
            )
        }
        RefOp::Matmul => LoweredButNot(
            "lowers to a 2-D grid with shared-memory tiles; one thread is not a block",
        ),
        RefOp::Neg
        | RefOp::Abs
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Tanh
        | RefOp::Softplus
        | RefOp::Min
        | RefOp::Matvec => NotLowered,
    }
}

/// linalg, lowered by `mlir-opt` and JITted by `mlir-runner`. Not a shim and not a device:
/// the toolchain that executes it ships with LLVM, so agreement here is between two
/// independent implementations rather than between a kernel and its author's idea of one.
fn linalg(op: RefOp) -> Cover {
    match op {
        RefOp::Exp
        | RefOp::Neg
        | RefOp::Abs
        | RefOp::Sigmoid
        | RefOp::Relu
        | RefOp::Sqrt
        | RefOp::Log
        | RefOp::Tanh
        | RefOp::Rsqrt
        | RefOp::Silu
        | RefOp::RmsNorm
        | RefOp::ReduceSum
        | RefOp::ReduceMax
        | RefOp::Absmax
        | RefOp::Min
        | RefOp::Max
        | RefOp::Matvec
        | RefOp::Matmul => Driven,
        RefOp::Softmax => LoweredButNot(
            "linalg.softmax decomposes through the transform dialect, not through any pass \
             in this pipeline; -convert-linalg-to-loops leaves it standing",
        ),
        RefOp::Softplus => NotLowered,
    }
}

const BACKENDS: &[(&str, Classify)] = &[
    ("gpu", gpu),
    ("bang", bang),
    ("nki", nki),
    ("tpu", tpu),
    ("msl", msl),
    ("linalg", linalg),
    ("musa", musa),
];

const ALL_OPS: &[RefOp] = &[
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

/// The intrinsic suffix, and the call shape it needs.
fn mlir_for(op: RefOp) -> String {
    let name = match op {
        RefOp::Matmul => "matmul",
        RefOp::Softmax => "softmax",
        RefOp::Exp => "exp",
        RefOp::Sigmoid => "sigmoid",
        RefOp::Relu => "relu",
        RefOp::Sqrt => "sqrt",
        RefOp::Log => "log",
        RefOp::Neg => "neg",
        RefOp::Abs => "abs",
        RefOp::Tanh => "tanh",
        RefOp::Rsqrt => "rsqrt",
        RefOp::Silu => "silu",
        RefOp::Softplus => "softplus",
        RefOp::ReduceMax => "reduce_max",
        RefOp::ReduceSum => "reduce_sum",
        RefOp::Min => "min",
        RefOp::Max => "max",
        RefOp::Matvec => "matvec",
        RefOp::RmsNorm => "rms_norm",
        RefOp::Absmax => "absmax",
    };
    let (rows, cols) = (4usize, 256usize);
    match op {
        RefOp::Matmul => "module {\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {hacc.entry} {\n    ^bb0:\n    \
             %m = llvm.mlir.constant(8 : i32) : i32\n    %k = llvm.mlir.constant(16 : i32) : i32\n    \
             %n = llvm.mlir.constant(4 : i32) : i32\n    \
             %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %y = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) \
             : (i32, i32, i32, i32, i32, i32) -> i32\n    \
             llvm.call @__tile_store_f32(%arg2, %y, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n    \
             llvm.return\n  }\n}\n"
            .to_string(),
        RefOp::Matvec => "module {\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {hacc.entry} {\n    ^bb0:\n    \
             %r = llvm.mlir.constant(8 : i32) : i32\n    %c = llvm.mlir.constant(16 : i32) : i32\n    \
             %one = llvm.mlir.constant(1 : i32) : i32\n    \
             %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %b = llvm.call @__tile_load_f32(%arg1, %one, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %y = llvm.call @__tile_matvec_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n    \
             llvm.call @__tile_store_f32(%arg2, %y, %r, %one) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n    \
             llvm.return\n  }\n}\n"
            .to_string(),
        RefOp::Min | RefOp::Max => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n    \
             %r = llvm.mlir.constant({rows} : i32) : i32\n    \
             %c = llvm.mlir.constant({cols} : i32) : i32\n    \
             %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %b = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %y = llvm.call @__tile_{name}_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32\n    \
             llvm.call @__tile_store_f32(%arg2, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n    \
             llvm.return\n  }}\n}}\n"
        ),
        RefOp::RmsNorm => format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
             attributes {{hacc.entry}} {{\n    ^bb0:\n    \
             %r = llvm.mlir.constant({rows} : i32) : i32\n    \
             %c = llvm.mlir.constant({cols} : i32) : i32\n    \
             %e = llvm.mlir.constant(2.500000e-02 : f32) : f32\n    \
             %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
             %y = llvm.call @__tile_rms_norm_f32(%a, %a, %e, %r, %c) : (i32, i32, f32, i32, i32) -> i32\n    \
             llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n    \
             llvm.return\n  }}\n}}\n"
        ),
        _ => {
            let out = if matches!(op, RefOp::ReduceSum | RefOp::ReduceMax | RefOp::Absmax) {
                1
            } else {
                cols
            };
            format!(
                "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                 attributes {{hacc.entry}} {{\n    ^bb0:\n    \
                 %r = llvm.mlir.constant({rows} : i32) : i32\n    \
                 %c = llvm.mlir.constant({cols} : i32) : i32\n    \
                 %o = llvm.mlir.constant({out} : i32) : i32\n    \
                 %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n    \
                 %y = llvm.call @__tile_{name}_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n    \
                 llvm.call @__tile_store_f32(%arg1, %y, %r, %o) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n    \
                 llvm.return\n  }}\n}}\n"
            )
        }
    }
}

/// The `RefOp`s an emulator's own source mentions -- i.e. what its sweeps attempt.
///
/// Emission alone cannot tell `Driven` from `LoweredButNot`: both require the backend to
/// emit, so marking everything `Driven` passed. Checked against that exact mis-declaration
/// -- claiming gpu drives softmax -- and it did NOT catch it until this was added. The
/// emulator's own file is the only source of truth for what its sweep reaches.
fn source_of(backend: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/emulate_{backend}.rs"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn the_declared_coverage_of_every_emulator_is_what_the_backends_actually_lower() {
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-coverage-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut wrong: Vec<String> = Vec::new();
    let (mut driven, mut held_back, mut absent) = (0usize, 0usize, 0usize);

    for (backend, classify) in BACKENDS {
        let source = source_of(backend);
        for op in ALL_OPS {
            let attempted = source.contains(&format!("RefOp::{op:?}"));
            let src = dir.join(format!("{backend}_{op:?}.mlir"));
            std::fs::write(&src, mlir_for(*op)).expect("write");
            let out = dir.join(format!("{backend}_{op:?}.out"));
            let _ = std::fs::remove_file(&out);
            let r = Command::new(exe)
                .arg(&src)
                .args(["-t", backend, "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            let emitted =
                r.status.success() && out.exists() && out.metadata().is_ok_and(|m| m.len() > 0);

            match classify(*op) {
                Driven => {
                    driven += 1;
                    if !emitted {
                        wrong.push(format!(
                            "{backend}/{op:?}: declared DRIVEN, but the backend now refuses it. \
                             That emulator is quietly covering less than it says."
                        ));
                    }
                    if !attempted {
                        wrong.push(format!(
                            "{backend}/{op:?}: declared DRIVEN, but tests/emulate_{backend}.rs \
                             never names it -- nothing runs it and the claim is empty."
                        ));
                    }
                }
                LoweredButNot(reason) => {
                    held_back += 1;
                    if attempted {
                        wrong.push(format!(
                            "{backend}/{op:?}: declared lowered-but-undriven ({reason}), yet \
                             tests/emulate_{backend}.rs names it -- it IS driven; reclassify."
                        ));
                    }
                    if !emitted {
                        wrong.push(format!(
                            "{backend}/{op:?}: declared lowered-but-undriven ({reason}), but the \
                             backend refuses it — so it is NotLowered now."
                        ));
                    }
                }
                NotLowered => {
                    absent += 1;
                    if emitted {
                        wrong.push(format!(
                            "{backend}/{op:?}: declared NOT LOWERED, but the backend emits it. \
                             It gained an arm and nothing is checking what that arm computes — \
                             drive it, or reclassify it with the property that prevents it."
                        ));
                    }
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emulator_coverage: {} backend-operation pairs — {driven} driven, {held_back} lowered \
         but not driven, {absent} not lowered",
        driven + held_back + absent
    );
    for w in wrong.iter().take(12) {
        eprintln!("  {w}");
    }
    assert!(
        wrong.is_empty(),
        "{} declared coverage entries disagree with what the backends emit",
        wrong.len()
    );
}
