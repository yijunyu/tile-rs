//! Running the linalg backend, rather than only parsing it.
//!
//! `linalg` is the one backend whose output is a STANDARD dialect, and the toolchain that
//! executes it ships with LLVM. So this needs no shim at all and no device: `mlir-opt`
//! bufferizes and lowers the emitted module to LLVM, `mlir-runner` JITs it, and the numbers
//! that come back are what the standard MLIR pipeline computed. Nothing between the emitter
//! and the answer is mine -- the same property that makes the Metal gate worth more than
//! the three shims, arriving from the opposite direction.
//!
//! That matters beyond one more backend. Every other gate compares an emitter against
//! `run::reference_output`, which is Rust code written in this repository. Here the answer
//! comes from LLVM's own execution engine, so agreement is between two independent
//! implementations rather than between a kernel and its author's idea of the kernel. The
//! first run made the point on its own: linalg's `matvec` returned -10.6844, -11.874,
//! -12.0008 and the Metal GPU had returned -10.6843624, -11.8740444, -12.0007992 for the
//! same problem -- an Apple GPU and an LLVM JIT agreeing to six digits.
//!
//! WHAT IS OUT, AND WHY. `softmax` alone. `linalg.softmax` is an aggregate named op whose
//! decomposition is exposed through the TRANSFORM dialect rather than through any pass in
//! this pipeline: `-convert-linalg-to-loops` leaves it standing and `mlir-runner` then
//! cannot parse it. That is a property of the op, not a gap in effort, and it is checked by
//! `mlir-opt` and by four other emulators regardless. `softplus` is absent because this
//! backend has no arm for it.
//!
//! Needs `mlir-opt` and `mlir-runner`; skips with a note otherwise. Both ship with LLVM and
//! are present here at `/opt/homebrew/opt/llvm/bin`, just not on `PATH`.

use std::path::{Path, PathBuf};
use std::process::Command;
use tile_cli::run::{input_values_for, reference_output, RefOp, Shape};

/// `mlir-opt` and `mlir-runner` are LLVM tools that Homebrew does not link onto `PATH`.
fn llvm_tool(name: &str) -> Option<PathBuf> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name}"))
        .output()
        .ok()?;
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if out.status.success() && !p.is_empty() {
        return Some(PathBuf::from(p));
    }
    let p = PathBuf::from(format!("/opt/homebrew/opt/llvm/bin/{name}"));
    p.exists().then_some(p)
}

/// An MLIR float literal that round-trips: MLIR rejects `-2` where an f32 is due.
fn f(v: f32) -> String {
    format!("{v:.9e}")
}

/// A `dense<...>` literal of the given shape, filled from the harness's own generator.
fn dense(buf: usize, shape: &[usize]) -> String {
    let n: usize = shape.iter().product();
    let v = input_values_for(buf, n);
    match shape {
        [_] => format!(
            "[{}]",
            v.iter().map(|x| f(*x)).collect::<Vec<_>>().join(", ")
        ),
        [r, c] => {
            let rows: Vec<String> = (0..*r)
                .map(|i| {
                    let cells: Vec<String> = (0..*c).map(|j| f(v[i * c + j])).collect();
                    format!("[{}]", cells.join(", "))
                })
                .collect();
            format!("[{}]", rows.join(", "))
        }
        _ => unreachable!("linalg emits rank 1 or 2"),
    }
}

/// The dimensions inside a `tensor<AxBxf32>`.
fn dims(t: &str) -> Vec<usize> {
    t.trim_start_matches("tensor<")
        .trim_end_matches("xf32>")
        .split('x')
        .filter_map(|d| d.parse().ok())
        .collect()
}

/// Every `tensor<...xf32>` in order, from a signature fragment.
fn tensors_in(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(i) = rest.find("tensor<") {
        let after = &rest[i..];
        if let Some(j) = after.find('>') {
            out.push(after[..=j].to_string());
            rest = &after[j + 1..];
        } else {
            break;
        }
    }
    out
}

/// Wrap the emitted module in a `main` that feeds it and prints what it returned.
///
/// The signature is read back OUT of the emitted file rather than assumed, so a backend
/// that changes an operand's shape produces a harness that no longer matches and says so,
/// instead of one that quietly feeds the wrong thing.
fn with_main(body: &str) -> Option<String> {
    let sig_start = body.find("func.func @k(")?;
    let rest = &body[sig_start..];
    let close = rest.find(") -> ")?;
    let args = tensors_in(&rest[..close]);
    let ret_end = rest[close + 5..].find(' ')?;
    let ret = rest[close + 5..close + 5 + ret_end].to_string();
    if args.is_empty() || !ret.starts_with("tensor<") {
        return None;
    }
    let rshape = dims(&ret);
    let rdims = rshape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");

    let mut m =
        String::from("\nfunc.func private @printMemrefF32(memref<*xf32>)\nfunc.func @main() {\n");
    for (i, a) in args.iter().enumerate() {
        // Operand i comes from generator index i, exactly as reference_output builds it.
        m.push_str(&format!(
            "  %in{i} = arith.constant dense<{}> : {a}\n",
            dense(i, &dims(a))
        ));
    }
    let names: Vec<String> = (0..args.len()).map(|i| format!("%in{i}")).collect();
    m.push_str(&format!(
        "  %r = call @k({}) : ({}) -> {ret}\n",
        names.join(", "),
        args.join(", ")
    ));
    m.push_str(&format!(
        "  %m = bufferization.to_buffer %r : {ret} to memref<{rdims}xf32>\n"
    ));
    m.push_str(&format!(
        "  %u = memref.cast %m : memref<{rdims}xf32> to memref<*xf32>\n"
    ));
    m.push_str("  call @printMemrefF32(%u) : (memref<*xf32>) -> ()\n  return\n}\n");
    Some(format!("{body}{m}"))
}

/// The values `printMemrefF32` wrote, in order.
fn parse_memref(stdout: &str) -> Vec<f32> {
    let Some(i) = stdout.find("data =") else {
        return Vec::new();
    };
    stdout[i + 6..]
        .replace(['[', ']'], " ")
        .split(',')
        .filter_map(|t| {
            let t = t.trim();
            (!t.is_empty()).then(|| t.parse::<f32>().ok()).flatten()
        })
        .collect()
}

#[test]
fn linalg_computes_what_the_reference_computes() {
    let (Some(mlir_opt), Some(mlir_runner)) = (llvm_tool("mlir-opt"), llvm_tool("mlir-runner"))
    else {
        eprintln!("emulate_linalg: skipped, no mlir-opt / mlir-runner");
        return;
    };
    let libdir = mlir_opt
        .parent()
        .and_then(Path::parent)
        .map(|p| p.join("lib"));
    let Some(libdir) = libdir.filter(|d| d.join("libmlir_runner_utils.dylib").exists()) else {
        eprintln!("emulate_linalg: skipped, no MLIR runner utils beside mlir-opt");
        return;
    };

    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emulate-linalg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let shapes = [(1usize, 64usize), (1, 256), (4, 256), (3, 129), (8, 64)];
    // (intrinsic, reference op, how the output is shaped)
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Elementwise,
        Reduce,
        Binary,
    }
    let ops: &[(&str, RefOp, Kind)] = &[
        ("exp", RefOp::Exp, Kind::Elementwise),
        ("neg", RefOp::Neg, Kind::Elementwise),
        ("abs", RefOp::Abs, Kind::Elementwise),
        ("sigmoid", RefOp::Sigmoid, Kind::Elementwise),
        ("relu", RefOp::Relu, Kind::Elementwise),
        ("sqrt", RefOp::Sqrt, Kind::Elementwise),
        ("log", RefOp::Log, Kind::Elementwise),
        ("tanh", RefOp::Tanh, Kind::Elementwise),
        ("rsqrt", RefOp::Rsqrt, Kind::Elementwise),
        ("silu", RefOp::Silu, Kind::Elementwise),
        ("rms_norm", RefOp::RmsNorm, Kind::Elementwise),
        ("reduce_sum", RefOp::ReduceSum, Kind::Reduce),
        ("reduce_max", RefOp::ReduceMax, Kind::Reduce),
        ("absmax", RefOp::Absmax, Kind::Reduce),
        ("min", RefOp::Min, Kind::Binary),
        ("max", RefOp::Max, Kind::Binary),
    ];

    let (mut checked, mut refused) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    let run_case = |label: String,
                    mlir: String,
                    want: &[f32],
                    checked: &mut usize,
                    refused: &mut usize,
                    bad: &mut Vec<String>| {
        let stem = label.replace([' ', 'x'], "_");
        let src = dir.join(format!("{stem}.mlir"));
        std::fs::write(&src, &mlir).expect("write");
        let lin = dir.join(format!("{stem}.linalg.mlir"));
        let _ = std::fs::remove_file(&lin);
        let emitted = Command::new(exe)
            .arg(&src)
            .args(["-t", "linalg", "-o"])
            .arg(&lin)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !lin.exists() {
            *refused += 1;
            return;
        }
        let body = std::fs::read_to_string(&lin).expect("read emitted");
        let Some(wrapped) = with_main(&body) else {
            bad.push(format!(
                "{label}: could not read a @k signature out of the emitted module"
            ));
            return;
        };
        let main_path = dir.join(format!("{stem}.main.mlir"));
        std::fs::write(&main_path, wrapped).expect("write main");
        let low = dir.join(format!("{stem}.llvm.mlir"));
        let lowered = Command::new(&mlir_opt)
            .arg(&main_path)
            .args([
                "-one-shot-bufferize=bufferize-function-boundaries",
                "-buffer-deallocation-pipeline",
                "-convert-linalg-to-loops",
                "-convert-scf-to-cf",
                "-expand-strided-metadata",
                "-lower-affine",
                "-finalize-memref-to-llvm",
                "-convert-arith-to-llvm",
                "-convert-math-to-llvm",
                "-convert-func-to-llvm",
                "-convert-cf-to-llvm",
                "-reconcile-unrealized-casts",
                "-o",
            ])
            .arg(&low)
            .output()
            .expect("mlir-opt runs");
        if !lowered.status.success() {
            bad.push(format!(
                "{label}: did not lower — {}",
                String::from_utf8_lossy(&lowered.stderr)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(100)
                    .collect::<String>()
            ));
            return;
        }
        let run = Command::new(&mlir_runner)
            .arg(&low)
            .args(["-e", "main", "-entry-point-result=void"])
            .arg(format!(
                "-shared-libs={}",
                libdir.join("libmlir_runner_utils.dylib").display()
            ))
            .arg(format!(
                "-shared-libs={}",
                libdir.join("libmlir_c_runner_utils.dylib").display()
            ))
            .output()
            .expect("mlir-runner runs");
        if !run.status.success() {
            bad.push(format!(
                "{label}: did not run — {}",
                String::from_utf8_lossy(&run.stderr)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(100)
                    .collect::<String>()
            ));
            return;
        }
        let got = parse_memref(&String::from_utf8_lossy(&run.stdout));
        *checked += 1;
        if got.len() != want.len() {
            bad.push(format!(
                "{label}: wrote {} values, {} were due",
                got.len(),
                want.len()
            ));
            return;
        }
        let mut worst = 0.0f32;
        let mut compared = 0usize;
        for (g, w) in got.iter().zip(want.iter()) {
            if (g.is_nan() && w.is_nan()) || (g.is_infinite() && w.is_infinite()) {
                continue;
            }
            compared += 1;
            let d = (g - w).abs() / w.abs().max(1.0);
            worst = if d.is_nan() {
                f32::INFINITY
            } else {
                worst.max(d)
            };
        }
        if compared == 0 {
            bad.push(format!(
                "{label}: every one of {} pairs was skipped as NaN or infinite on both sides \
                 -- this comparison checked nothing",
                want.len()
            ));
            return;
        }
        // 1e-4, not the 1e-5 the other gates use, and the limit is the PRINTER rather than
        // the kernel: `printMemrefF32` writes about six significant figures, so a value
        // near 7.40089 cannot be compared more finely than its last printed digit.
        if worst > 1e-4 {
            bad.push(format!(
                "{label}: max rel {worst:.2e} over {compared}/{} compared",
                want.len()
            ));
        }
    };

    for &(rows, cols) in &shapes {
        for (name, op, kind) in ops {
            let (mlir, want) = match kind {
                Kind::Binary => (
                    format!(
                        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
                         %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
                         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %b = llvm.call @__tile_load_f32(%arg1, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %y = llvm.call @__tile_{name}_f32(%a, %a, %b, %r, %c) \
                         : (i32, i32, i32, i32, i32) -> i32\n\
                         \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                         \x20   llvm.return\n  }}\n}}\n"
                    ),
                    reference_output(*op, Shape::Rows2 { rows, cols }, 1e-6, "float"),
                ),
                Kind::Reduce => (
                    format!(
                        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                         attributes {{hacc.entry}} {{\n    ^bb0:\n\
                         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                         \x20   %o = llvm.mlir.constant(1 : i32) : i32\n\
                         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %y = llvm.call @__tile_{name}_f32(%a, %a, %r, %c) \
                         : (i32, i32, i32, i32) -> i32\n\
                         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %o) \
                         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                         \x20   llvm.return\n  }}\n}}\n"
                    ),
                    reference_output(*op, Shape::RowReduce { rows, cols }, 1e-6, "float"),
                ),
                Kind::Elementwise if *name == "rms_norm" => {
                    let mlir = format!(
                        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                         attributes {{hacc.entry}} {{\n    ^bb0:\n\
                         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                         \x20   %e = llvm.mlir.constant(2.500000e-02 : f32) : f32\n\
                         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %y = llvm.call @__tile_rms_norm_f32(%a, %a, %e, %r, %c) \
                         : (i32, i32, f32, i32, i32) -> i32\n\
                         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                         \x20   llvm.return\n  }}\n}}\n"
                    );
                    // The eps comes back out of the generated MLIR, never from a constant
                    // repeated here -- see Shape::rms_eps_from_mlir's own comment.
                    let eps = Shape::rms_eps_from_mlir(&mlir).unwrap_or(1e-6);
                    let want = reference_output(*op, Shape::Rows { rows, cols }, eps, "float");
                    (mlir, want)
                }
                Kind::Elementwise => (
                    format!(
                        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                         attributes {{hacc.entry}} {{\n    ^bb0:\n\
                         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
                         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
                         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32) -> i32\n\
                         \x20   %y = llvm.call @__tile_{name}_f32(%a, %a, %r, %c) \
                         : (i32, i32, i32, i32) -> i32\n\
                         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) \
                         : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                         \x20   llvm.return\n  }}\n}}\n"
                    ),
                    reference_output(*op, Shape::Rows { rows, cols }, 1e-6, "float"),
                ),
            };
            run_case(
                format!("{name} {rows}x{cols}"),
                mlir,
                &want,
                &mut checked,
                &mut refused,
                &mut bad,
            );
        }
    }

    // matvec: a matrix times a cols-long vector. The emitted signature typed that vector by
    // its ROW count until this gate ran, so the module did not verify at all.
    for &(rows, cols) in &[(8usize, 16usize), (4, 4), (3, 129)] {
        let mlir = format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
             \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
             \x20   %one = llvm.mlir.constant(1 : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f32(%arg1, %one, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_matvec_f32(%a, %a, %b, %r, %c) \
             : (i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f32(%arg2, %y, %r, %one) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
        let want = reference_output(RefOp::Matvec, Shape::Matvec { rows, cols }, 1e-6, "float");
        run_case(
            format!("matvec {rows}x{cols}"),
            mlir,
            &want,
            &mut checked,
            &mut refused,
            &mut bad,
        );
    }

    for &(m, k, n) in &[
        (8usize, 16usize, 4usize),
        (4, 4, 4),
        (16, 8, 32),
        (3, 129, 5),
    ] {
        let mlir = format!(
            "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, \
             %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    ^bb0:\n\
             \x20   %m = llvm.mlir.constant({m} : i32) : i32\n\
             \x20   %k = llvm.mlir.constant({k} : i32) : i32\n\
             \x20   %n = llvm.mlir.constant({n} : i32) : i32\n\
             \x20   %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32\n\
             \x20   %y = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) \
             : (i32, i32, i32, i32, i32, i32) -> i32\n\
             \x20   llvm.call @__tile_store_f32(%arg2, %y, %m, %n) \
             : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
             \x20   llvm.return\n  }}\n}}\n"
        );
        let want = reference_output(RefOp::Matmul, Shape::Matmul { m, k, n }, 1e-6, "float");
        run_case(
            format!("matmul {m}x{k}x{n}"),
            mlir,
            &want,
            &mut checked,
            &mut refused,
            &mut bad,
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emulate_linalg: {checked} kernels lowered, JITted and compared, {refused} refused");
    for b in bad.iter().take(12) {
        eprintln!("  {b}");
    }
    assert!(
        checked >= 60,
        "only {checked} kernels ran -- linalg is refusing nearly everything and this test \
         would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} linalg kernels disagree with the reference. The module was lowered by mlir-opt \
         and executed by LLVM's own JIT, so this is a disagreement about arithmetic.",
        bad.len()
    );
}
