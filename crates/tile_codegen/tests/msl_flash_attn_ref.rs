//! The DeepSeek-V4 flash-attention reference kernels, emitted from their intrinsics.
//!
//! Both walked a 512-wide key head and a 512-wide value head, written into the
//! text and into their names. The two dimensions are now trailing operands, so
//! the names' 512s are only defaults. At 512 the emitted kernels must match the
//! committed ones byte for byte; at other head dimensions, on a Metal GPU, they
//! must match a CPU reference of the same attention.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const PLAIN: &str = r#"
module {
  llvm.func @ds4_kernel_flash_attn_ext_f16_dk512_dv512(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne01 = llvm.mlir.constant(8 : i32) : i32
    %ne02 = llvm.mlir.constant(2 : i32) : i32
    %ne03 = llvm.mlir.constant(1 : i32) : i32
    %nb01 = llvm.mlir.constant(1024 : i32) : i32
    %nb02 = llvm.mlir.constant(8192 : i32) : i32
    %nb03 = llvm.mlir.constant(16384 : i32) : i32
    %ne11 = llvm.mlir.constant(64 : i32) : i32
    %ne122 = llvm.mlir.constant(2 : i32) : i32
    %ne123 = llvm.mlir.constant(1 : i32) : i32
    %nb11 = llvm.mlir.constant(1024 : i32) : i32
    %nb12 = llvm.mlir.constant(65536 : i32) : i32
    %nb13 = llvm.mlir.constant(131072 : i32) : i32
    %nb21 = llvm.mlir.constant(1024 : i32) : i32
    %nb22 = llvm.mlir.constant(65536 : i32) : i32
    %nb23 = llvm.mlir.constant(131072 : i32) : i32
    %ne31 = llvm.mlir.constant(8 : i32) : i32
    %ne32 = llvm.mlir.constant(2 : i32) : i32
    %ne33 = llvm.mlir.constant(1 : i32) : i32
    %nb31 = llvm.mlir.constant(128 : i32) : i32
    %nb32 = llvm.mlir.constant(1024 : i32) : i32
    %nb33 = llvm.mlir.constant(2048 : i32) : i32
    %scale = llvm.mlir.constant(0.0441941738 : f32) : f32
    %maxb = llvm.mlir.constant(0.0 : f32) : f32
    %m0 = llvm.mlir.constant(0.0 : f32) : f32
    %m1 = llvm.mlir.constant(0.0 : f32) : f32
    %nhl2 = llvm.mlir.constant(0 : i32) : i32
    %lsc = llvm.mlir.constant(0.0 : f32) : f32
    %hm = llvm.mlir.constant(0 : i32) : i32
    %hs = llvm.mlir.constant(0 : i32) : i32
    %hb = llvm.mlir.constant(0 : i32) : i32
    %hsc = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_f16_dk512_dv512(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %ne01, %ne02, %ne03, %nb01, %nb02, %nb03, %ne11, %ne122, %ne123, %nb11, %nb12, %nb13, %nb21, %nb22, %nb23, %ne31, %ne32, %ne33, %nb31, %nb32, %nb33, %scale, %maxb, %m0, %m1, %nhl2, %lsc, %hm, %hs, %hb, %hsc) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, i32, f32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const VEC: &str = r#"
module {
  llvm.func @ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne01 = llvm.mlir.constant(1 : i32) : i32
    %ne02 = llvm.mlir.constant(2 : i32) : i32
    %ne03 = llvm.mlir.constant(1 : i32) : i32
    %nb01 = llvm.mlir.constant(1024 : i32) : i32
    %nb02 = llvm.mlir.constant(1024 : i32) : i32
    %nb03 = llvm.mlir.constant(2048 : i32) : i32
    %ne11 = llvm.mlir.constant(8 : i32) : i32
    %ne122 = llvm.mlir.constant(2 : i32) : i32
    %ne123 = llvm.mlir.constant(1 : i32) : i32
    %nb11 = llvm.mlir.constant(1024 : i32) : i32
    %nb12 = llvm.mlir.constant(8192 : i32) : i32
    %nb13 = llvm.mlir.constant(16384 : i32) : i32
    %nb21 = llvm.mlir.constant(1024 : i32) : i32
    %nb22 = llvm.mlir.constant(8192 : i32) : i32
    %nb23 = llvm.mlir.constant(16384 : i32) : i32
    %ne31 = llvm.mlir.constant(8 : i32) : i32
    %ne32 = llvm.mlir.constant(2 : i32) : i32
    %ne33 = llvm.mlir.constant(1 : i32) : i32
    %nb31 = llvm.mlir.constant(16 : i32) : i32
    %nb32 = llvm.mlir.constant(16 : i32) : i32
    %nb33 = llvm.mlir.constant(32 : i32) : i32
    %ne1 = llvm.mlir.constant(1 : i32) : i32
    %ne2 = llvm.mlir.constant(2 : i32) : i32
    %ne3 = llvm.mlir.constant(1 : i32) : i32
    %scale = llvm.mlir.constant(0.0441941738 : f32) : f32
    %maxb = llvm.mlir.constant(0.0 : f32) : f32
    %m0 = llvm.mlir.constant(0.0 : f32) : f32
    %m1 = llvm.mlir.constant(0.0 : f32) : f32
    %nhl2 = llvm.mlir.constant(0 : i32) : i32
    %lsc = llvm.mlir.constant(0.0 : f32) : f32
    %hm = llvm.mlir.constant(0 : i32) : i32
    %hs = llvm.mlir.constant(0 : i32) : i32
    %hb = llvm.mlir.constant(0 : i32) : i32
    %hsc = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_vec_f16_dk512_dv512(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %ne01, %ne02, %ne03, %nb01, %nb02, %nb03, %ne11, %ne122, %ne123, %nb11, %nb12, %nb13, %nb21, %nb22, %nb23, %ne31, %ne32, %ne33, %nb31, %nb32, %nb33, %ne1, %ne2, %ne3, %scale, %maxb, %m0, %m1, %nhl2, %lsc, %hm, %hs, %hb, %hsc) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, i32, f32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

fn golden(name: &str) -> String {
    let p = format!("{}/../../benchmarks/ds4_msl/emitted/{name}.metal", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn with_dims(mlir: &str, callee: &str, dk: u32, dv: u32) -> String {
    let call = mlir.find(&format!("llvm.call @{callee}")).unwrap();
    let open = mlir[call..].find('(').unwrap() + call;
    let close = mlir[open..].find(')').unwrap() + open;
    let types_open = mlir[close..].find('(').unwrap() + close;
    let types_close = mlir[types_open..].find(')').unwrap() + types_open;
    let spliced = format!(
        "{}, %dk, %dv{}, i32, i32{}",
        &mlir[..close],
        &mlir[close..types_close],
        &mlir[types_close..]
    );
    let line = spliced[..spliced.find(&format!("llvm.call @{callee}")).unwrap()].rfind('\n').unwrap() + 1;
    format!(
        "{}    %dk = llvm.mlir.constant({dk} : i32) : i32\n    %dv = llvm.mlir.constant({dv} : i32) : i32\n{}",
        &spliced[..line],
        &spliced[line..]
    )
}

const PLAIN_CALLEE: &str = "__tile_flash_attn_ext_f16_dk512_dv512";
const VEC_CALLEE: &str = "__tile_flash_attn_ext_vec_f16_dk512_dv512";

#[test]
fn shipped_head_dims_are_byte_identical() {
    assert_eq!(try_emit(PLAIN).unwrap(), golden("flash_attn_ext_f16_dk512_dv512"));
    assert_eq!(try_emit(VEC).unwrap(), golden("flash_attn_ext_vec_f16_dk512_dv512"));
    assert_eq!(try_emit(&with_dims(PLAIN, PLAIN_CALLEE, 512, 512)).unwrap(), golden("flash_attn_ext_f16_dk512_dv512"));
    assert_eq!(try_emit(&with_dims(VEC, VEC_CALLEE, 512, 512)).unwrap(), golden("flash_attn_ext_vec_f16_dk512_dv512"));
}

#[test]
fn head_dims_reach_the_kernels_and_bad_ones_are_refused() {
    let src = try_emit(&with_dims(PLAIN, PLAIN_CALLEE, 64, 32)).unwrap();
    assert!(src.contains("constexpr short DK = 64;") && src.contains("constexpr short DV = 32;"), "{src}");
    // The kernel's own name carries 512; only the constants must follow the operands.
    assert!(!src.contains("constexpr short DK = 512;") && !src.contains("constexpr short DV = 512;"), "{src}");
    let v = try_emit(&with_dims(VEC, VEC_CALLEE, 128, 256)).unwrap();
    assert!(v.contains("constexpr short DK = 128;") && v.contains("constexpr short DV = 256;"), "{v}");
    assert!(try_emit(&with_dims(PLAIN, PLAIN_CALLEE, 63, 64)).unwrap_err().contains("dk 63 must be a positive multiple of 4"));
    assert!(try_emit(&with_dims(PLAIN, PLAIN_CALLEE, 64, 8192)).unwrap_err().contains("dv 8192"));
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    /// Mirrors `struct FaUniforms` in the emitted prelude: 32 four-byte fields.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FaUniforms {
        ne01: i32, ne02: i32, ne03: i32,
        nb01: u32, nb02: u32, nb03: u32,
        ne11: i32, ne_12_2: i32, ne_12_3: i32,
        nb11: u32, nb12: u32, nb13: u32,
        nb21: u32, nb22: u32, nb23: u32,
        ne31: i32, ne32: i32, ne33: i32,
        nb31: u32, nb32: u32, nb33: u32,
        scale: f32, max_bias: f32, m0: f32, m1: f32,
        n_head_log2: i32,
        logit_softcap: f32,
        has_mask: u32, has_sinks: u32, has_bias: u32, has_softcap: u32,
    }

    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut seed = 0xA5A5_1234_9876_F00Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let to_half = |v: &[f32]| -> Vec<u16> {
        v.iter()
            .map(|&x| {
                if x == 0.0 {
                    return 0;
                }
                let bits = x.to_bits();
                let sign = ((bits >> 16) & 0x8000) as u16;
                let e = ((bits >> 23) & 0xff) as i32 - 127;
                let mant = ((bits & 0x7f_ffff) >> 13) as u16;
                sign | (((e + 15) as u16) << 10) | mant
            })
            .collect()
    };
    // Values that survive the f32 -> binary16 cast exactly, so the reference and
    // the kernel see identical inputs.
    let quantized = |x: f32| -> f32 {
        let step = 1.0f32 / 256.0;
        (x / step).round() * step
    };

    /// The vector kernel's uniforms carry three more dimensions.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FaVecUniforms {
        ne01: i32, ne02: i32, ne03: i32,
        nb01: u32, nb02: u32, nb03: u32,
        ne11: i32, ne_12_2: i32, ne_12_3: i32,
        nb11: u32, nb12: u32, nb13: u32,
        nb21: u32, nb22: u32, nb23: u32,
        ne31: i32, ne32: i32, ne33: i32,
        nb31: u32, nb32: u32, nb33: u32,
        ne1: i32, ne2: i32, ne3: i32,
        scale: f32, max_bias: f32, m0: f32, m1: f32,
        n_head_log2: i32,
        logit_softcap: f32,
        has_mask: u32, has_sinks: u32, has_bias: u32, has_softcap: u32,
    }

    const N_ROWS: usize = 5;
    const N_KEYS: usize = 7;
    for (dk, dv) in [(512usize, 512usize), (64, 64), (128, 256), (32, 16)] {
        let q: Vec<f32> = (0..N_ROWS * dk).map(|_| quantized(next())).collect();
        let k: Vec<f32> = (0..N_KEYS * dk).map(|_| quantized(next())).collect();
        let v: Vec<f32> = (0..N_KEYS * dv).map(|_| quantized(next())).collect();
        let scale = 1.0f32 / (dk as f32).sqrt();

        let mut want = vec![0f32; N_ROWS * dv];
        for row in 0..N_ROWS {
            let scores: Vec<f64> = (0..N_KEYS)
                .map(|ic| {
                    let dot: f64 = (0..dk).map(|i| (q[row * dk + i] * k[ic * dk + i]) as f64).sum();
                    dot * scale as f64
                })
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores.iter().map(|s| (s - m).exp()).sum();
            for d in 0..dv {
                let num: f64 = (0..N_KEYS).map(|ic| (scores[ic] - m).exp() * v[ic * dv + d] as f64).sum();
                want[row * dv + d] = (num / denom) as f32;
            }
        }

        let o = MTLResourceOptions::StorageModeShared;
        let hbuf = |x: &[u16]| dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 2) as u64, o);
        let (bq, bk, bv) = (hbuf(&to_half(&q)), hbuf(&to_half(&k)), hbuf(&to_half(&v)));
        let empty = dev.new_buffer(4, o);
        let queue = dev.new_command_queue();

        let pipeline = |mlir: &str, callee: &str, entry: &str| {
            let src = try_emit(&with_dims(mlir, callee, dk as u32, dv as u32)).unwrap();
            let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
            dev.new_compute_pipeline_state(&desc).unwrap()
        };
        let check = |got: &[f32], entry: &str| {
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1e-2),
                    "{entry} dk {dk} dv {dv}: row {} d {}: got {g}, want {w}",
                    i / dv,
                    i % dv
                );
            }
        };

        // Plain: 8 query rows per threadgroup, output in buffer 7.
        {
            let entry = "ds4_kernel_flash_attn_ext_f16_dk512_dv512";
            let pipe = pipeline(PLAIN, PLAIN_CALLEE, entry);
            let bo = dev.new_buffer((want.len() * 4) as u64, o);
            let u = FaUniforms {
                ne01: N_ROWS as i32, ne02: 1, ne03: 1,
                nb01: (dk * 2) as u32, nb02: (N_ROWS * dk * 2) as u32, nb03: (N_ROWS * dk * 2) as u32,
                ne11: N_KEYS as i32, ne_12_2: 1, ne_12_3: 1,
                nb11: (dk * 2) as u32, nb12: (N_KEYS * dk * 2) as u32, nb13: (N_KEYS * dk * 2) as u32,
                nb21: (dv * 2) as u32, nb22: (N_KEYS * dv * 2) as u32, nb23: (N_KEYS * dv * 2) as u32,
                scale,
                ..Default::default()
            };
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bq, &bk, &bv, &empty, &empty, &empty, &empty, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_bytes(8, std::mem::size_of::<FaUniforms>() as u64, &u as *const FaUniforms as *const _);
            enc.dispatch_thread_groups(MTLSize::new((N_ROWS as u64).div_ceil(8), 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            check(unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, want.len()) }, entry);
        }

        // Vector: one query row per threadgroup, output in buffer 6, rid layout.
        {
            let entry = "ds4_kernel_flash_attn_ext_vec_f16_dk512_dv512";
            let pipe = pipeline(VEC, VEC_CALLEE, entry);
            let bo = dev.new_buffer((want.len() * 4) as u64, o);
            let u = FaVecUniforms {
                ne01: N_ROWS as i32, ne02: 1, ne03: 1,
                nb01: (dk * 2) as u32, nb02: (N_ROWS * dk * 2) as u32, nb03: (N_ROWS * dk * 2) as u32,
                ne11: N_KEYS as i32, ne_12_2: 1, ne_12_3: 1,
                nb11: (dk * 2) as u32, nb12: (N_KEYS * dk * 2) as u32, nb13: (N_KEYS * dk * 2) as u32,
                nb21: (dv * 2) as u32, nb22: (N_KEYS * dv * 2) as u32, nb23: (N_KEYS * dv * 2) as u32,
                ne1: 1, ne2: 1, ne3: 1,
                scale,
                ..Default::default()
            };
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bq, &bk, &bv, &empty, &empty, &empty, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_bytes(7, std::mem::size_of::<FaVecUniforms>() as u64, &u as *const FaVecUniforms as *const _);
            enc.dispatch_thread_groups(MTLSize::new(N_ROWS as u64, 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            check(unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, want.len()) }, entry);
        }
    }
}
