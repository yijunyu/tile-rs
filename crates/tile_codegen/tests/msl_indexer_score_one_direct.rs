//! The DeepSeek-V4 single-token indexer-score kernel, emitted from its intrinsic.
//!
//! With `n_head` left at its default the emitted Metal must stay byte-identical
//! to the committed kernel the engine loads. At other head counts it must
//! compute the same scores as a CPU reference on the GPU (macOS only).
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const SHIPPED_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexer_score_one_direct(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ncomp = llvm.mlir.constant(8 : i32) : i32
    %qhs   = llvm.mlir.constant(512 : i32) : i32
    %irs   = llvm.mlir.constant(512 : i32) : i32
    %scl   = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.call @__tile_indexer_score_one_direct_f32(%arg0, %arg1, %arg2, %arg3, %ncomp, %qhs, %irs, %scl) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

fn try_emit(mlir: &str) -> Result<String, String> {
    let reg = TargetRegistry::with_builtin();
    reg.select("msl").expect("msl target").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

fn emit(mlir: &str) -> String {
    try_emit(mlir).expect("emit")
}

fn golden() -> String {
    let p = format!(
        "{}/../../benchmarks/ds4_msl/emitted/dsv4_indexer_score_one_direct.metal",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// MLIR calling the intrinsic with an explicit n_head operand.
fn with_n_head(n_head: &str) -> String {
    format!(
        r#"
module {{
  llvm.func @score_one_gate(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: i32) attributes {{hacc.entry}} {{
    ^bb0:
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %nh = llvm.mlir.constant({n_head} : i32) : i32
    %r = llvm.call @__tile_indexer_score_one_direct_f32(%arg0, %arg1, %arg2, %arg3, %c1, %c1, %c1, %c1, %nh) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }}
}}
"#
    )
}

#[test]
fn default_head_count_is_byte_identical_to_the_shipped_kernel() {
    assert_eq!(emit(SHIPPED_MLIR), golden());
    let explicit = emit(&with_n_head("64"));
    assert_eq!(explicit, golden().replace("ds4_dsv4_indexer_score_one_direct(", "score_one_gate("));
}

#[test]
fn n_head_reaches_the_kernel_and_bad_values_are_refused() {
    let src = emit(&with_n_head("12"));
    assert!(src.contains("head0 < 12u; head0 += 4u"), "{src}");
    assert!(!src.contains("64u"), "{src}");
    let odd = try_emit(&with_n_head("6")).unwrap_err();
    assert!(odd.contains("n_head 6 must be a positive multiple of 4"), "{odd}");
    let runtime = with_n_head("64").replace("%c1, %nh)", "%c1, %arg4)");
    assert!(runtime.contains("%c1, %arg4)"));
    let not_constant = try_emit(&runtime).unwrap_err();
    assert!(not_constant.contains("operand 8 (n_head) must be a positive integer constant"), "{not_constant}");
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_the_cpu_reference_at_several_head_counts() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};
    const HEAD_DIM: usize = 128;
    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let fill = |n: usize, seed: u64| -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    for n_head in [64usize, 4, 12, 96] {
        let src = emit(&with_n_head(&n_head.to_string()));
        let lib = dev
            .new_library_with_source(&src, &CompileOptions::new())
            .unwrap_or_else(|e| panic!("does not compile: {e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("score_one_gate", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
        assert_eq!(pipe.thread_execution_width(), 32, "kernel assumes a 32-lane simdgroup");

        let n_comp = 37;
        // Buffers carry 64 extra non-zero heads past n_head. Fresh GPU memory is
        // zero, and a zero weight hides a kernel that reads too many heads.
        let q = fill((n_head + 64) * HEAD_DIM, 11);
        let w: Vec<f32> = fill(n_head + 64, 12).iter().map(|v| v.abs() + 0.5).collect();
        let k = fill(n_comp * HEAD_DIM, 13);
        let scale = 1.0 / ((n_head * HEAD_DIM) as f32).sqrt();
        let want: Vec<f32> = (0..n_comp)
            .map(|c| {
                (0..n_head)
                    .map(|h| {
                        let dot: f32 = (0..HEAD_DIM).map(|d| q[h * HEAD_DIM + d] * k[c * HEAD_DIM + d]).sum();
                        dot.max(0.0) * (w[h] * scale)
                    })
                    .sum()
            })
            .collect();

        let o = MTLResourceOptions::StorageModeShared;
        let buf = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
        let (bq, bw, bk) = (buf(&q), buf(&w), buf(&k));
        let bs = dev.new_buffer((n_comp * 4) as u64, o);
        let scalars = [n_comp as u32, (HEAD_DIM * 4) as u32, (HEAD_DIM * 4) as u32];
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
        enc.set_bytes(7, 4, &scale as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(n_comp as u64, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(bs.contents() as *const f32, n_comp) };
        for (c, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!((g - e).abs() <= 1e-4 * e.abs().max(1.0), "n_head {n_head} comp {c}: got {g}, want {e}");
        }
    }
}
