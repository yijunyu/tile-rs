//! The DeepSeek-V4 hyper-connection expand kernel, emitted from its intrinsic.
//!
//! The kernel is the general expand fully unrolled: it preloads one residual per
//! hyper-connection and expands the combine over them. That count was written
//! into the text; it is now a trailing operand, and the 4 in the intrinsic's
//! name is only the default. At 4 the emitted kernel must match the committed
//! one byte for byte; at other counts, on a Metal GPU, it must match a CPU
//! reference of the same expansion.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_hc_expand4(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %n_embd    = llvm.mlir.constant(8 : i32) : i32
    %n_hc      = llvm.mlir.constant(4 : i32) : i32
    %n_tokens  = llvm.mlir.constant(2 : i32) : i32
    %nb_block0 = llvm.mlir.constant(4 : i32) : i32
    %nb_block1 = llvm.mlir.constant(32 : i32) : i32
    %nb_add0   = llvm.mlir.constant(4 : i32) : i32
    %nb_add1   = llvm.mlir.constant(32 : i32) : i32
    %nb_res0   = llvm.mlir.constant(4 : i32) : i32
    %nb_res1   = llvm.mlir.constant(32 : i32) : i32
    %nb_res2   = llvm.mlir.constant(128 : i32) : i32
    %nb_post0  = llvm.mlir.constant(4 : i32) : i32
    %nb_post1  = llvm.mlir.constant(16 : i32) : i32
    %nb_comb0  = llvm.mlir.constant(4 : i32) : i32
    %nb_comb1  = llvm.mlir.constant(16 : i32) : i32
    %nb_comb2  = llvm.mlir.constant(64 : i32) : i32
    %nb0       = llvm.mlir.constant(4 : i32) : i32
    %nb1       = llvm.mlir.constant(32 : i32) : i32
    %nb2       = llvm.mlir.constant(128 : i32) : i32
    %has_add   = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.call @__tile_dsv4_hc_expand4_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %n_embd, %n_hc, %n_tokens, %nb_block0, %nb_block1, %nb_add0, %nb_add1, %nb_res0, %nb_res1, %nb_res2, %nb_post0, %nb_post1, %nb_comb0, %nb_comb1, %nb_comb2, %nb0, %nb1, %nb2, %has_add) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

fn golden() -> String {
    let p = format!("{}/../../benchmarks/ds4_msl/emitted/dsv4_hc_expand4.metal", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// The fixture's call with an unroll operand, and n_hc set to match.
fn with_hc(hc: u32) -> String {
    let call = MLIR.find("llvm.call @__tile_dsv4_hc_expand4_f32").unwrap();
    let open = MLIR[call..].find('(').unwrap() + call;
    let close = MLIR[open..].find(')').unwrap() + open;
    let types_open = MLIR[close..].find('(').unwrap() + close;
    let types_close = MLIR[types_open..].find(')').unwrap() + types_open;
    let spliced = format!("{}, %hcu{}, i32{}", &MLIR[..close], &MLIR[close..types_close], &MLIR[types_close..]);
    let line = spliced[..spliced.find("llvm.call @__tile_dsv4_hc_expand4_f32").unwrap()].rfind('\n').unwrap() + 1;
    format!("{}    %hcu = llvm.mlir.constant({hc} : i32) : i32\n{}", &spliced[..line], &spliced[line..])
}

#[test]
fn shipped_unroll_is_byte_identical() {
    assert_eq!(try_emit(MLIR).unwrap(), golden());
    assert_eq!(try_emit(&with_hc(4)).unwrap(), golden());
}

#[test]
fn unroll_reaches_the_kernel_and_bad_counts_are_refused() {
    let two = try_emit(&with_hc(2)).unwrap();
    assert!(two.contains("if (n_hc != 2u) return;"), "{two}");
    assert!(two.contains("const float r1 =") && !two.contains("const float r2 ="), "{two}");
    assert!(two.contains("dst_hc < 2u"), "{two}");
    let eight = try_emit(&with_hc(8)).unwrap();
    assert!(eight.contains("const float r7 =") && eight.contains("* r7;"), "{eight}");
    assert!(try_emit(&with_hc(17)).unwrap_err().contains("hc_unroll 17 must be from 1 to 16"));
    let runtime = with_hc(4).replace(", %hcu)", ", %n_hc_rt)");
    assert!(try_emit(&runtime).is_err());
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};
    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut seed = 0x5DEE_CE66_D1E5_1234u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    for (hc, n_embd, n_tokens, has_add) in [(4usize, 16usize, 3usize, 1u32), (1, 8, 2, 0), (2, 9, 4, 1), (8, 5, 2, 1)] {
        let src = try_emit(&with_hc(hc as u32)).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("ds4_dsv4_hc_expand4", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();

        let block: Vec<f32> = (0..n_embd * n_tokens).map(|_| next()).collect();
        let add: Vec<f32> = (0..n_embd * n_tokens).map(|_| next()).collect();
        let res: Vec<f32> = (0..n_embd * hc * n_tokens).map(|_| next()).collect();
        let post: Vec<f32> = (0..hc * n_tokens).map(|_| next()).collect();
        let comb: Vec<f32> = (0..hc * hc * n_tokens).map(|_| next()).collect();

        // Element strides in bytes, matching the fixture's layout.
        let (nb_block0, nb_block1) = (4, (n_embd * 4) as u32);
        let (nb_res0, nb_res1, nb_res2) = (4u32, (n_embd * 4) as u32, (n_embd * hc * 4) as u32);
        let (nb_post0, nb_post1) = (4u32, (hc * 4) as u32);
        let (nb_comb0, nb_comb1, nb_comb2) = (4u32, (hc * 4) as u32, (hc * hc * 4) as u32);
        let (nb0, nb1, nb2) = (4u32, (n_embd * 4) as u32, (n_embd * hc * 4) as u32);

        let mut want = vec![0f32; n_embd * hc * n_tokens];
        for t in 0..n_tokens {
            for d in 0..n_embd {
                let mut block_v = block[t * n_embd + d];
                if has_add != 0 {
                    block_v += add[t * n_embd + d];
                }
                for dst in 0..hc {
                    let mut acc = block_v * post[t * hc + dst];
                    for s in 0..hc {
                        acc += comb[t * hc * hc + s * hc + dst] * res[t * n_embd * hc + s * n_embd + d];
                    }
                    want[t * n_embd * hc + dst * n_embd + d] = acc;
                }
            }
        }

        let o = MTLResourceOptions::StorageModeShared;
        let buf = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
        let (bb, br, bp, bc, ba) = (buf(&block), buf(&res), buf(&post), buf(&comb), buf(&add));
        let bo = dev.new_buffer((want.len() * 4) as u64, o);
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [&bb, &br, &bp, &bc, &ba, &bo].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        let scalars: [u32; 19] = [
            n_embd as u32, hc as u32, n_tokens as u32,
            nb_block0, nb_block1, nb_block0, nb_block1,
            nb_res0, nb_res1, nb_res2,
            nb_post0, nb_post1,
            nb_comb0, nb_comb1, nb_comb2,
            nb0, nb1, nb2,
            has_add,
        ];
        for (i, v) in scalars.iter().enumerate() {
            enc.set_bytes(6 + i as u64, 4, v as *const u32 as *const _);
        }
        let threads = 64u64;
        let groups = ((n_embd * n_tokens) as u64).div_ceil(threads);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, want.len()) };
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 * w.abs().max(1e-3),
                "hc {hc} n_embd {n_embd}: index {i}: got {g}, want {w}"
            );
        }
    }
}
