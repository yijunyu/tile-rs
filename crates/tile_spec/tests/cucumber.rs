//! Executable Gherkin harness for tile-rs.
//!
//! Loads every `.feature` under `features/`, registers step definitions that
//! drive the REAL open surface:
//!   * the `CodegenTarget` trait + `TargetRegistry` (from `tile_codegen`),
//!   * the 14 open `convert_mlir_to_*` pure emitters (`#[path]`-included from
//!     `rustc_codegen_tile/src/`, NO LLVM dep — same trick the generality-matrix
//!     tests use),
//!   * the tile DSL public signatures (`tile_std/src/tile.rs`, read as text
//!     because that crate is `no_core` and not host-runnable).
//!
//! Each `Scenario` becomes one `#[test]`-equivalent assertion run: the harness
//! reports "<N> scenarios, all green", and any UNDEFINED or failing step fails
//! the test with the scenario name + step text.

use std::fs;
use std::path::Path;
use tile_codegen::{CodegenTarget, EmitOpts, EmitOut, TargetRegistry};
use tile_spec::{Runner, World};

// ── The 14 open emitters, included with NO LLVM dep ──────────────────────────
// Dependency order: mlir_parse <- mlir_to_pto <- (gpu, spirv, msl, nki, aie,
// bang, gaudi, tpu, linalg); musa <- gpu; csl/hexagon/ttmetal standalone.
#[path = "../../rustc_codegen_tile/src/mlir_parse.rs"]
mod mlir_parse;
#[path = "../../rustc_codegen_tile/src/mlir_to_aie.rs"]
mod mlir_to_aie;
#[path = "../../rustc_codegen_tile/src/mlir_to_bang.rs"]
mod mlir_to_bang;
#[path = "../../rustc_codegen_tile/src/mlir_to_csl.rs"]
mod mlir_to_csl;
#[path = "../../rustc_codegen_tile/src/mlir_to_gaudi.rs"]
mod mlir_to_gaudi;
#[path = "../../rustc_codegen_tile/src/mlir_to_gpu.rs"]
mod mlir_to_gpu;
#[path = "../../rustc_codegen_tile/src/mlir_to_hexagon.rs"]
mod mlir_to_hexagon;
#[path = "../../rustc_codegen_tile/src/mlir_to_linalg.rs"]
mod mlir_to_linalg;
#[path = "../../rustc_codegen_tile/src/mlir_to_msl.rs"]
mod mlir_to_msl;
#[path = "../../rustc_codegen_tile/src/mlir_to_musa.rs"]
mod mlir_to_musa;
#[path = "../../rustc_codegen_tile/src/mlir_to_nki.rs"]
mod mlir_to_nki;
#[path = "../../rustc_codegen_tile/src/mlir_to_pto.rs"]
mod mlir_to_pto;
#[path = "../../rustc_codegen_tile/src/mlir_to_spirv.rs"]
mod mlir_to_spirv;
#[path = "../../rustc_codegen_tile/src/mlir_to_tpu.rs"]
mod mlir_to_tpu;
#[path = "../../rustc_codegen_tile/src/mlir_to_ttmetal.rs"]
mod mlir_to_ttmetal;

/// Dispatch a backend `name` to its pure emitter. Returns the emitted source or
/// the emitter's error string. This is the open dispatch surface the registry
/// formalizes.
fn emit(target: &str, mlir: &str) -> Result<String, String> {
    match target {
        "gpu" => mlir_to_gpu::convert_mlir_to_gpu(mlir),
        "musa" => mlir_to_musa::convert_mlir_to_musa(mlir),
        "spirv" => mlir_to_spirv::convert_mlir_to_spirv(mlir),
        "msl" => mlir_to_msl::convert_mlir_to_msl(mlir),
        "nki" => mlir_to_nki::convert_mlir_to_nki(mlir),
        "aie" => mlir_to_aie::convert_mlir_to_aie(mlir),
        "bang" => mlir_to_bang::convert_mlir_to_bang(mlir),
        "gaudi" => mlir_to_gaudi::convert_mlir_to_gaudi(mlir),
        "tpu" => mlir_to_tpu::convert_mlir_to_tpu(mlir),
        "csl" => mlir_to_csl::convert_mlir_to_csl(mlir),
        "hexagon" => mlir_to_hexagon::convert_mlir_to_hexagon(mlir),
        "ttmetal" => mlir_to_ttmetal::convert_mlir_to_ttmetal(mlir),
        "linalg" => mlir_to_linalg::convert_mlir_to_linalg(mlir),
        "pto" => mlir_to_pto::convert_mlir_to_pto(mlir),
        other => Err(format!("unknown target '{other}'")),
    }
}

// ── Canonical MLIR snippets per op (reused from the proven matrix tests) ─────
fn snippet(op: &str) -> &'static str {
    match op {
        "softmax" => SOFTMAX_MLIR,
        "rms_norm" => RMS_NORM_MLIR,
        "matmul" => MATMUL_MLIR,
        "rope" => ROPE_MLIR,
        "add" => ADD_MLIR,
        "silu_mul" => SILU_MUL_MLIR,
        "argmax" => ARGMAX_MLIR,
        "sample_top_p" => SAMPLE_TOP_P_MLIR,
        "token_accept" => TOKEN_ACCEPT_MLIR,
        _ => panic!("no canonical MLIR snippet for op '{op}'"),
    }
}

// ── Phase-6 speculative-decoding + SiLU→Mul fusion snippets ──
// These drive the fused multi-op / spec-decode lowering arms that the single-op
// matrix snippets above never reach.
const SILU_MUL_MLIR: &str = r#"
module {
  llvm.func @swiglu(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(2 : i32) : i32
    %c = llvm.mlir.constant(64 : i32) : i32
    %gate = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @ascend_tile_silu_f32(%gate, %gate, %r, %c) : (i32, i32, i32, i32) -> i32
    %res = llvm.call @ascend_tile_mul_f32(%s, %s, %up, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const ARGMAX_MLIR: &str = r#"
module {
  llvm.func @argmax_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_argmax_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const SAMPLE_TOP_P_MLIR: &str = r#"
module {
  llvm.func @sample_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %temp = llvm.mlir.constant(2 : i32) : i32
    %p = llvm.mlir.constant(1 : i32) : i32
    %seed = llvm.mlir.constant(42 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_sample_top_p_f32(%a, %a, %temp, %p, %seed, %r, %c) : (i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const TOKEN_ACCEPT_MLIR: &str = r#"
module {
  llvm.func @accept_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(1 : i32) : i32
    %thr = llvm.mlir.constant(1 : i32) : i32
    %draft = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %tgt = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %probs = llvm.call @ascend_tile_load_f32(%arg2, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_token_accept_f32(%draft, %draft, %tgt, %probs, %thr, %r) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const SOFTMAX_MLIR: &str = r#"
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c1024 = llvm.mlir.constant(1024 : i32) : i32
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const RMS_NORM_MLIR: &str = r#"
module {
  llvm.func @rms_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %e = llvm.mlir.constant(0 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_rms_norm_f32(%e, %a, %e, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const MATMUL_MLIR: &str = r#"
module {
  llvm.func @tile_matmul_f16(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %m = llvm.mlir.constant(64 : i32) : i32
    %k = llvm.mlir.constant(128 : i32) : i32
    %n = llvm.mlir.constant(64 : i32) : i32
    %a = llvm.call @ascend_tile_load_f16(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f16(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @ascend_tile_matmul_f16(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const ROPE_MLIR: &str = r#"
module {
  llvm.func @rope_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %rows = llvm.mlir.constant(4 : i32) : i32
    %cols = llvm.mlir.constant(64 : i32) : i32
    %pos  = llvm.mlir.constant(7 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %rows, %cols) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_rope_f32(%a, %a, %pos, %rows, %cols) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %rows, %cols) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const ADD_MLIR: &str = r#"
module {
  llvm.func @add_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @ascend_tile_add_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %s, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

// ── A custom CodegenTarget for the registry scenarios ────────────────────────
struct MyAccel;
impl CodegenTarget for MyAccel {
    fn name(&self) -> &'static str {
        "myaccel"
    }
    fn emit(&self, mlir: &str, _o: &EmitOpts) -> Result<EmitOut, String> {
        if mlir.trim().is_empty() {
            return Err("empty".into());
        }
        Ok(EmitOut {
            source: "// myaccel\n".into(),
            ext: "acc",
            meta: Default::default(),
        })
    }
}

/// The plain `(mlir)->Result<String,String>` shape the 14 backends use, lifted
/// through `EmitterTarget` in the "adapter" scenario.
fn fake_cuda_convert(mlir: &str) -> Result<String, String> {
    if mlir.trim().is_empty() {
        return Err("empty".into());
    }
    Ok(format!("// CUDA\n{mlir}"))
}

/// Build the runner with every step definition wired to real code.
fn build_runner() -> Runner {
    let mut r = Runner::new();

    // ---- backend_emit_purity.feature ----
    r.given("the canonical softmax MLIR snippet", |w, _| {
        w.set("mlir", SOFTMAX_MLIR);
    });
    r.when("I emit the snippet to target \"{}\"", |w, a| {
        let target = &a[0];
        w.set("target", target);
        match emit(target, w.get("mlir")) {
            Ok(src) => {
                w.set("emitted", src);
                w.set_flag("emit_ok", true);
            }
            Err(e) => {
                w.set("emit_err", e);
                w.set_flag("emit_ok", false);
            }
        }
    });
    r.when("I emit an empty module to target \"{}\"", |w, a| {
        let target = &a[0];
        w.set("target", target);
        w.set_flag("emit_ok", emit(target, "   \n  ").is_ok());
    });
    r.then("the emit succeeds", |w, _| {
        assert!(
            w.flag("emit_ok"),
            "emit to '{}' failed: {}",
            w.get("target"),
            w.get("emit_err")
        );
    });
    r.then("the emit fails", |w, _| {
        assert!(
            !w.flag("emit_ok"),
            "emit to '{}' unexpectedly succeeded",
            w.get("target")
        );
    });
    r.then("the output contains \"{}\"", |w, a| {
        let needle = &a[0];
        assert!(
            w.get("emitted").contains(needle.as_str()),
            "target '{}' output missing idiom {:?}.\n--- emitted (first 400 chars) ---\n{}",
            w.get("target"),
            needle,
            &w.get("emitted").chars().take(400).collect::<String>()
        );
    });
    r.then(
        "emitting the same snippet again yields byte-identical output",
        |w, _| {
            let again = emit(w.get("target"), w.get("mlir")).expect("re-emit ok");
            assert_eq!(
                again,
                w.get("emitted"),
                "target '{}' emit is NOT deterministic",
                w.get("target")
            );
        },
    );

    // ---- core_op_lowering.feature ----
    r.given("the canonical \"{}\" MLIR snippet", |w, a| {
        w.set("mlir", snippet(&a[0]));
    });

    // ---- registry_add_target.feature ----
    r.given(
        "a registry pre-populated with the built-in targets",
        |w, _| {
            // We cannot stash a registry in World (it's not String), so we record a
            // marker and rebuild the registry in each Then. The registry is cheap.
            w.set("registry", "builtin");
        },
    );
    r.given("an empty registry", |w, _| {
        w.set("registry", "empty");
        w.set_flag("has_myaccel", false);
        w.set_flag("has_cuda", false);
    });
    r.given("a custom target named \"{}\" registered on top", |w, a| {
        assert_eq!(a[0], "myaccel");
        w.set_flag("has_myaccel", true);
    });
    r.when("I register a custom target named \"{}\"", |w, a| {
        assert_eq!(a[0], "myaccel");
        w.set_flag("has_myaccel", true);
    });
    r.when(
        "I register the plain \"{}\" convert fn as an EmitterTarget",
        |w, a| {
            assert_eq!(a[0], "cuda");
            w.set_flag("has_cuda", true);
        },
    );
    r.when("TILERS_CODEGEN_PATH is \"{}\"", |w, a| {
        w.set("codegen_path", &a[0]);
    });
    r.then("the registry is not empty", |w, _| {
        let reg = build_registry(w);
        assert!(!reg.is_empty());
    });
    r.then("selecting \"{}\" routes to a target", |w, a| {
        let reg = build_registry(w);
        assert!(
            reg.select(&a[0]).is_some(),
            "select('{}') routed to nothing",
            a[0]
        );
    });
    r.then("selecting \"{}\" routes to nothing", |w, a| {
        let reg = build_registry(w);
        assert!(
            reg.select(&a[0]).is_none(),
            "select('{}') unexpectedly routed",
            a[0]
        );
    });
    r.then("the registry has {} target", |w, a| {
        let reg = build_registry(w);
        let want: usize = a[0].trim().parse().expect("count");
        assert_eq!(reg.len(), want, "registry size mismatch");
    });
    r.then(
        "the custom target emits its signature for a non-empty module",
        |_w, _| {
            let out = MyAccel
                .emit("module {}", &EmitOpts::default())
                .expect("emit");
            assert!(out.source.contains("myaccel"));
        },
    );
    r.then(
        "the registry selects the matching target by that name",
        |w, _| {
            let reg = build_registry(w);
            let name = w.get("codegen_path");
            let t = reg
                .select(name)
                .unwrap_or_else(|| panic!("no target for '{name}'"));
            assert_eq!(t.name(), name);
        },
    );
    r.then(
        "emitting a non-empty module through it yields the adapter output",
        |_w, _| {
            let out = fake_cuda_convert("module {}").expect("convert ok");
            assert!(out.starts_with("// CUDA"));
        },
    );
    r.then("emitting an empty module through it fails", |_w, _| {
        assert!(fake_cuda_convert("   ").is_err());
    });

    // ---- tile_dsl_shapes.feature ----
    r.given("the tile_std tile DSL source", |w, _| {
        let src = fs::read_to_string("../tile_std/src/tile.rs").expect("read tile_std/src/tile.rs");
        w.set("tile_src", src);
    });
    r.then(
        "a \"{}\" type is declared with const ROWS and COLS dimensions",
        |w, a| {
            let ty = &a[0];
            let src = w.get("tile_src");
            let needle = format!("pub struct {ty}<const ROWS: usize, const COLS: usize");
            assert!(
                src.contains(&needle),
                "missing shape-typed declaration: {needle}"
            );
        },
    );
    r.then("the intrinsic \"{}\" is generic over \"{}\"", |w, a| {
        let (func, dims) = (&a[0], &a[1]);
        let src = w.get("tile_src");
        // Find the `pub fn <func><...>` and assert the generic list contains dims.
        let sig = extract_fn_generics(src, func)
            .unwrap_or_else(|| panic!("intrinsic '{func}' not found in tile DSL"));
        for d in dims.split(", ") {
            assert!(
                sig.contains(d),
                "intrinsic '{func}' generics {{{sig}}} missing dim '{d}'"
            );
        }
    });
    r.then(
        "the intrinsic \"{}\" binds three distinct dims \"{}\", \"{}\", \"{}\"",
        |w, a| {
            let func = &a[0];
            let src = w.get("tile_src");
            let sig = extract_fn_generics(src, func).expect("matmul intrinsic");
            for d in [&a[1], &a[2], &a[3]] {
                assert!(
                    sig.contains(&format!("const {d}")),
                    "matmul missing const {d}"
                );
            }
            // distinctness: M, K, N are three different identifiers.
            assert!(a[1] != a[2] && a[2] != a[3] && a[1] != a[3]);
        },
    );

    r
}

/// Build a fresh registry reflecting the World's registry markers.
fn build_registry(w: &World) -> TargetRegistry {
    let mut reg = if w.get("registry") == "builtin" {
        TargetRegistry::with_builtin()
    } else {
        TargetRegistry::new()
    };
    if w.flag("has_myaccel") {
        reg.register(Box::new(MyAccel));
    }
    if w.flag("has_cuda") {
        reg.register(Box::new(tile_codegen::EmitterTarget::new(
            "cuda",
            "cu",
            fake_cuda_convert,
        )));
    }
    reg
}

/// Pull the `<...>` generic-parameter list of `pub fn <name>` out of source.
fn extract_fn_generics(src: &str, name: &str) -> Option<String> {
    let key = format!("pub fn {name}");
    let start = src.find(&key)?;
    let after = &src[start + key.len()..];
    let lt = after.find('<')?;
    let gt = after.find('>')?;
    if lt < gt {
        Some(after[lt + 1..gt].to_string())
    } else {
        None
    }
}

/// Discover and run every `.feature` file, asserting all scenarios green.
#[test]
fn all_features_green() {
    let runner = build_runner();
    let dir = Path::new("features");
    let mut total_scenarios = 0usize;
    let mut features_run = 0usize;
    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("features/ dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "feature").unwrap_or(false))
        .collect();
    entries.sort();
    assert!(
        !entries.is_empty(),
        "no .feature files found under features/"
    );
    for path in entries {
        let text = fs::read_to_string(&path).unwrap();
        let n = runner.run_str(&text);
        eprintln!("  {:<40} {n} scenario(s) green", path.display());
        total_scenarios += n;
        features_run += 1;
    }
    eprintln!(
        "tile-rs GWT: {features_run} feature file(s), {total_scenarios} scenario(s) all green"
    );
    assert!(total_scenarios >= 1);
}
