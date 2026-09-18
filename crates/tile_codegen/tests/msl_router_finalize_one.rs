//! The DeepSeek-V4 MoE router finalize kernel, emitted from its intrinsic.
//!
//! At the shipped 256 experts, top 6 the emitted Metal must stay byte-identical
//! to the committed kernel. At other shapes, on a Metal GPU, the selected expert
//! ids must equal a CPU replay of the same bitonic network, in both sort and
//! hash modes.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const SHIPPED_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_router_finalize_one(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %hb = llvm.mlir.constant(1 : i32) : i32
    %hm = llvm.mlir.constant(0 : i32) : i32
    %ub = llvm.mlir.constant(0 : i32) : i32
    %tk = llvm.mlir.constant(0 : i32) : i32
    %hr = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.call @__tile_router_finalize_one_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %hb, %hm, %ub, %tk, %hr) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

fn golden() -> String {
    let p = format!("{}/../../benchmarks/ds4_msl/emitted/dsv4_router_finalize_one.metal", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn shaped(n_expert: &str, top_k: &str) -> String {
    format!(
        r#"
module {{
  llvm.func @router_gate(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: i32) attributes {{hacc.entry}} {{
    ^bb0:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %ne = llvm.mlir.constant({n_expert} : i32) : i32
    %tk = llvm.mlir.constant({top_k} : i32) : i32
    %r = llvm.call @__tile_router_finalize_one_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %c0, %c0, %c0, %c0, %c0, %ne, %tk) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }}
}}
"#
    )
}

#[test]
fn shipped_shape_is_byte_identical_to_the_shipped_kernel() {
    assert_eq!(try_emit(SHIPPED_MLIR).unwrap(), golden());
    let explicit = try_emit(&shaped("256", "6")).unwrap();
    assert_eq!(explicit, golden().replace("ds4_dsv4_router_finalize_one(", "router_gate("));
}

#[test]
fn shape_reaches_the_kernel_and_bad_shapes_are_refused() {
    let src = try_emit(&shaped("64", "8")).unwrap();
    assert!(src.contains("threadgroup float sel_scores[64];") && src.contains("k <= 64u") && src.contains("hrow * 8u"), "{src}");
    assert!(!src.contains("256") && !src.contains("6u;"), "{src}");
    for (ne, tk, why) in [("100", "6", "n_expert 100 must be a power of two"), ("2048", "6", "n_expert 2048"), ("64", "65", "top_k 65 must be from 1 to n_expert (64)")] {
        let e = try_emit(&shaped(ne, tk)).unwrap_err();
        assert!(e.contains(why), "{ne}/{tk}: {e}");
    }
    let runtime = shaped("64", "8").replace("%c0, %ne, %tk)", "%c0, %arg5, %tk)");
    assert!(runtime.contains("%arg5, %tk)"));
    assert!(try_emit(&runtime).unwrap_err().contains("operand 10 (n_expert) must be a positive integer constant"));
    let arity = try_emit(&shaped("64", "8").replace(", %ne, %tk)", ", %ne)")).unwrap_err();
    assert!(arity.contains("expected 10 operands, or 12 with n_expert and top_k; got 11"), "{arity}");
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_a_cpu_replay_of_the_sort() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};
    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // Exact replay of the kernel's bitonic network: every (k, j) stage, with
    // the lower index of each pair deciding the swap.
    fn replay(scores: &[f32], top_k: usize) -> Vec<i32> {
        let n = scores.len();
        let mut idx: Vec<usize> = (0..n).collect();
        let mut k = 2;
        while k <= n {
            let mut j = k >> 1;
            while j > 0 {
                for tid in 0..n {
                    let other = tid ^ j;
                    if other > tid {
                        let (a, b) = (scores[idx[tid]], scores[idx[other]]);
                        if (tid & k == 0 && a < b) || (tid & k != 0 && a > b) {
                            idx.swap(tid, other);
                        }
                    }
                }
                j >>= 1;
            }
            k <<= 1;
        }
        idx[..top_k].iter().map(|&i| i as i32).collect()
    }
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    for (n_expert, top_k) in [(256usize, 6usize), (64, 8), (8, 2), (1024, 6), (16, 16)] {
        let src = try_emit(&shaped(&n_expert.to_string(), &top_k.to_string())).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("router_gate", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();

        let probs: Vec<f32> = (0..n_expert).map(|_| next()).collect();
        let bias: Vec<f32> = (0..n_expert).map(|_| next() * 0.5).collect();
        let hash_rows = 5usize;
        let hash: Vec<i32> = (0..hash_rows * top_k).map(|i| (i * 7 % n_expert) as i32).collect();
        for (has_bias, hash_mode, token) in [(1u32, 0u32, 0u32), (0, 0, 0), (0, 1, 3), (0, 1, 99)] {
            let o = MTLResourceOptions::StorageModeShared;
            let fbuf = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
            let ibuf = |v: &[i32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
            let (bp, bb, bh, bt) = (fbuf(&probs), fbuf(&bias), ibuf(&hash), ibuf(&[0]));
            let bo = ibuf(&vec![-1i32; top_k]);
            let queue = dev.new_command_queue();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bp, &bb, &bh, &bt, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in [has_bias, hash_mode, 0, token, hash_rows as u32].iter().enumerate() {
                enc.set_bytes(5 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(n_expert as u64, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const i32, top_k) }.to_vec();
            let want = if hash_mode != 0 {
                let row = (token as usize).min(hash_rows - 1);
                hash[row * top_k..(row + 1) * top_k].to_vec()
            } else {
                let scores: Vec<f32> = (0..n_expert).map(|i| probs[i] + if has_bias != 0 { bias[i] } else { 0.0 }).collect();
                replay(&scores, top_k)
            };
            assert_eq!(got, want, "n_expert {n_expert} top_k {top_k} has_bias {has_bias} hash_mode {hash_mode} token {token}");
        }
    }
}
