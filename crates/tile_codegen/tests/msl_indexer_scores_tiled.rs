//! The DeepSeek-V4 tiled indexer-score kernel, emitted from its intrinsic.
//!
//! Two properties are checked. With the head dimension and tile width left at
//! their defaults, the emitted Metal must stay byte-identical to the committed
//! kernels the engine loads. At other shapes, the emitted kernel must compute
//! the same scores as a CPU reference on the GPU (macOS only).
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const F32_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexer_scores_tiled_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ncomp = llvm.mlir.constant(32 : i32) : i32
    %ntok  = llvm.mlir.constant(8 : i32) : i32
    %nh    = llvm.mlir.constant(64 : i32) : i32
    %p0    = llvm.mlir.constant(0 : i32) : i32
    %r     = llvm.mlir.constant(4 : i32) : i32
    %qts   = llvm.mlir.constant(32768 : i32) : i32
    %qhs   = llvm.mlir.constant(512 : i32) : i32
    %wts   = llvm.mlir.constant(256 : i32) : i32
    %irs   = llvm.mlir.constant(512 : i32) : i32
    %sts   = llvm.mlir.constant(128 : i32) : i32
    %scl   = llvm.mlir.constant(1 : i32) : i32
    %ret = llvm.call @__tile_indexer_scores_tiled_f32(%arg0, %arg1, %arg2, %arg3, %ncomp, %ntok, %nh, %p0, %r, %qts, %qhs, %wts, %irs, %sts, %scl) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const HALF_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexer_scores_tiled(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ncomp = llvm.mlir.constant(32 : i32) : i32
    %ntok  = llvm.mlir.constant(8 : i32) : i32
    %nh    = llvm.mlir.constant(64 : i32) : i32
    %p0    = llvm.mlir.constant(128 : i32) : i32
    %r     = llvm.mlir.constant(4 : i32) : i32
    %qts   = llvm.mlir.constant(32768 : i32) : i32
    %qhs   = llvm.mlir.constant(512 : i32) : i32
    %wts   = llvm.mlir.constant(256 : i32) : i32
    %irs   = llvm.mlir.constant(512 : i32) : i32
    %sts   = llvm.mlir.constant(128 : i32) : i32
    %scl   = llvm.mlir.constant(1 : i32) : i32
    %ret = llvm.call @__tile_indexer_scores_tiled_bf16(%arg0, %arg1, %arg2, %arg3, %ncomp, %ntok, %nh, %p0, %r, %qts, %qhs, %wts, %irs, %sts, %scl) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

fn emit(mlir: &str) -> String {
    let reg = TargetRegistry::with_builtin();
    let msl = reg.select("msl").expect("msl target");
    msl.emit(mlir, &EmitOpts::default()).expect("emit").source
}

fn golden(name: &str) -> String {
    let p = format!("{}/../../benchmarks/ds4_msl/emitted/{name}.metal", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

#[test]
fn default_shape_is_byte_identical_to_the_shipped_kernels() {
    assert_eq!(emit(F32_MLIR), golden("dsv4_indexer_scores_tiled_f32"));
    assert_eq!(emit(HALF_MLIR), golden("dsv4_indexer_scores_tiled"));
}

/// MLIR calling the intrinsic with explicit head_dim and tile_cols operands.
fn shaped_mlir(half: bool, head_dim: &str, tile_cols: &str) -> String {
    let callee = if half { "__tile_indexer_scores_tiled_bf16" } else { "__tile_indexer_scores_tiled_f32" };
    format!(
        r#"
module {{
  llvm.func @indexer_gate(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: i32) attributes {{hacc.entry}} {{
    ^bb0:
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %hd = llvm.mlir.constant({head_dim} : i32) : i32
    %tc = llvm.mlir.constant({tile_cols} : i32) : i32
    %ret = llvm.call @{callee}(%arg0, %arg1, %arg2, %arg3, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %hd, %tc) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }}
}}
"#
    )
}

fn try_emit(mlir: &str) -> Result<String, String> {
    let reg = TargetRegistry::with_builtin();
    reg.select("msl").unwrap().emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

#[test]
fn explicit_shipped_shape_matches_the_shipped_kernel_body() {
    // Same shape, spelled as operands: only the kernel name in the signature differs.
    let explicit = emit(&shaped_mlir(false, "128", "32"));
    let shipped = golden("dsv4_indexer_scores_tiled_f32")
        .replace("ds4_dsv4_indexer_scores_tiled_f32(", "indexer_gate(");
    assert_eq!(explicit, shipped);
}

#[test]
fn shape_operands_reach_the_kernel() {
    let src = emit(&shaped_mlir(true, "64", "16"));
    assert!(src.contains("constexpr uint D  = 64u;"), "{src}");
    assert!(src.contains("constexpr uint TN = 16u;"), "{src}");
    assert!(src.contains("i += 64u)"), "16 key columns need 64 threads per group:\n{src}");
    assert!(src.contains("threadgroup half qtg[TM*D];"), "{src}");
    assert!(!src.contains("128u"), "a shipped-shape constant leaked into a 64/16 kernel:\n{src}");
}

#[test]
fn unusable_shapes_are_refused_not_emitted() {
    let not_multiple = try_emit(&shaped_mlir(false, "12", "32")).unwrap_err();
    assert!(not_multiple.contains("head_dim 12"), "{not_multiple}");
    let too_many_threads = try_emit(&shaped_mlir(false, "128", "512")).unwrap_err();
    assert!(too_many_threads.contains("1024"), "{too_many_threads}");
    // head_dim passed as the kernel's runtime argument instead of a constant.
    let runtime = shaped_mlir(false, "128", "32").replace("%c1, %hd, %tc)", "%c1, %arg4, %tc)");
    assert!(runtime.contains("%arg4, %tc)"));
    let not_constant = try_emit(&runtime).unwrap_err();
    assert!(not_constant.contains("operand 15 (head_dim) must be a positive integer constant"), "{not_constant}");
    let wrong_arity = try_emit(&shaped_mlir(false, "128", "32").replace(", %tc)", ")")).unwrap_err();
    assert!(wrong_arity.contains("got 16"), "{wrong_arity}");
}

/// GPU check at shapes the shipped kernel never had, against a CPU reference.
#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    struct Case {
        half: bool,
        head_dim: usize,
        tile_cols: usize,
        n_tokens: usize,
        n_comp: usize,
        n_head: usize,
        pos0: u32,
        ratio: u32,
    }

    /// Deterministic values in [-1, 1).
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn reference(c: &Case, q: &[f32], w: &[f32], k: &[f32], scale: f32) -> Vec<f32> {
        let (d, h) = (c.head_dim, c.n_head);
        let mut out = vec![0.0f32; c.n_tokens * c.n_comp];
        for t in 0..c.n_tokens {
            let visible = (((c.pos0 as usize) + t + 1) / c.ratio as usize).min(c.n_comp);
            for comp in 0..c.n_comp {
                out[t * c.n_comp + comp] = if comp >= visible {
                    f32::NEG_INFINITY
                } else {
                    (0..h)
                        .map(|head| {
                            let qrow = &q[(t * h + head) * d..][..d];
                            let krow = &k[comp * d..][..d];
                            let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
                            dot.max(0.0) * (w[t * h + head] * scale)
                        })
                        .sum()
                };
            }
        }
        out
    }

    fn run(dev: &Device, c: &Case) {
        let src = emit(&shaped_mlir(c.half, &c.head_dim.to_string(), &c.tile_cols.to_string()));
        let lib = dev
            .new_library_with_source(&src, &CompileOptions::new())
            .unwrap_or_else(|e| panic!("emitted kernel does not compile: {e}\n{src}"));
        let f = lib.get_function("indexer_gate", None).expect("entry point");
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&f));
        let pipe = dev.new_compute_pipeline_state(&desc).expect("pipeline");
        let threads = (c.tile_cols * 4) as u64;
        // The kernel writes two score cells per thread and pairs them across a
        // 32-wide simdgroup; on a GPU with another execution width it is wrong.
        assert_eq!(pipe.thread_execution_width(), 32, "kernel assumes a 32-thread simdgroup");
        assert!(pipe.max_total_threads_per_threadgroup() >= threads);

        let (d, h) = (c.head_dim, c.n_head);
        let q = fill(c.n_tokens * h * d, 1);
        let w = fill(c.n_tokens * h, 2);
        let k = fill(c.n_comp * d, 3);
        let scale = 1.0 / ((d * h) as f32).sqrt();
        let o = MTLResourceOptions::StorageModeShared;
        let buf = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
        let (bq, bw, bk) = (buf(&q), buf(&w), buf(&k));
        let bs = dev.new_buffer((c.n_tokens * c.n_comp * 4) as u64, o);
        let scalars: [u32; 10] = [
            c.n_comp as u32,
            c.n_tokens as u32,
            h as u32,
            c.pos0,
            c.ratio,
            (h * d * 4) as u32,
            (d * 4) as u32,
            (h * 4) as u32,
            (d * 4) as u32,
            (c.n_comp * 4) as u32,
        ];
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [&bq, &bw, &bk, &bs].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, v) in scalars.iter().enumerate() {
            enc.set_bytes(4 + i as u64, 4, v as *const u32 as *const _);
        }
        enc.set_bytes(14, 4, &scale as *const f32 as *const _);
        let groups = MTLSize::new(c.n_comp.div_ceil(c.tile_cols) as u64, c.n_tokens.div_ceil(8) as u64, 1);
        enc.dispatch_thread_groups(groups, MTLSize::new(threads, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let got = unsafe { std::slice::from_raw_parts(bs.contents() as *const f32, c.n_tokens * c.n_comp) };
        let want = reference(c, &q, &w, &k, scale);
        let tol = if c.half { 2e-2 } else { 1e-4 };
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            let ok = if e.is_infinite() { g == e } else { (g - e).abs() <= tol * e.abs().max(1.0) };
            assert!(
                ok,
                "half={} head_dim={} tile_cols={}: token {} comp {}: got {g}, want {e}",
                c.half, c.head_dim, c.tile_cols, i / c.n_comp, i % c.n_comp
            );
        }
    }

    #[test]
    fn emitted_kernels_match_the_cpu_reference_at_several_shapes() {
        let Some(dev) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let shapes = [(false, 128, 32), (true, 128, 32), (false, 64, 16), (true, 64, 16), (false, 16, 64), (false, 72, 8)];
        for (half, head_dim, tile_cols) in shapes {
            // Token and key counts that are not multiples of the tile, so edge
            // tiles and the causal mask are both exercised.
            let c = Case { half, head_dim, tile_cols, n_tokens: 13, n_comp: 2 * tile_cols + 5, n_head: 3, pos0: 40, ratio: 4 };
            run(&dev, &c);
        }
    }
}
