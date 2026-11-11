//! The staged DeepSeek-V4 flash-attention kernels, emitted from their intrinsics.
//!
//! All four stages sized their shared tiles for a 64-wide key head and a
//! 64-wide value head. Those two widths are now trailing operands. (The eight
//! queries, four simdgroups and thirty-two keys a pass around them are not
//! free: they are the shape of the 8x8 simdgroup tiles the stages move data
//! in.) At the shipped widths the emitted kernels must match the committed
//! ones byte for byte; at other widths, on a Metal GPU, each stage must match
//! a CPU reference of what that stage computes.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const SETUP: &str = r#"
module {
  llvm.func @ds4_flash_attn_ext_setup(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %dk   = llvm.mlir.constant(64 : i32) : i32
    %dv   = llvm.mlir.constant(64 : i32) : i32
    %ne01 = llvm.mlir.constant(8 : i32) : i32
    %nb01 = llvm.mlir.constant(256 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_setup_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %dk, %dv, %ne01, %nb01) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const SCORE: &str = r#"
module {
  llvm.func @ds4_flash_attn_ext_score(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %dk    = llvm.mlir.constant(64 : i32) : i32
    %dv    = llvm.mlir.constant(64 : i32) : i32
    %ne01  = llvm.mlir.constant(8 : i32) : i32
    %ne11  = llvm.mlir.constant(64 : i32) : i32
    %nb01  = llvm.mlir.constant(256 : i32) : i32
    %nb11  = llvm.mlir.constant(128 : i32) : i32
    %scale = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_score_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %dk, %dv, %ne01, %ne11, %nb01, %nb11, %scale) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const OUT: &str = r#"
module {
  llvm.func @ds4_flash_attn_ext_out(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %dk    = llvm.mlir.constant(64 : i32) : i32
    %dv    = llvm.mlir.constant(64 : i32) : i32
    %ne01  = llvm.mlir.constant(8 : i32) : i32
    %ne11  = llvm.mlir.constant(64 : i32) : i32
    %nb01  = llvm.mlir.constant(256 : i32) : i32
    %nb11  = llvm.mlir.constant(128 : i32) : i32
    %nb21  = llvm.mlir.constant(128 : i32) : i32
    %scale = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_out_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %dk, %dv, %ne01, %ne11, %nb01, %nb11, %nb21, %scale) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const OUT_MS: &str = r#"
module {
  llvm.func @ds4_flash_attn_ext_out_ms(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %dk    = llvm.mlir.constant(64 : i32) : i32
    %dv    = llvm.mlir.constant(64 : i32) : i32
    %ne01  = llvm.mlir.constant(8 : i32) : i32
    %ne11  = llvm.mlir.constant(64 : i32) : i32
    %nb01  = llvm.mlir.constant(256 : i32) : i32
    %nb11  = llvm.mlir.constant(128 : i32) : i32
    %nb21  = llvm.mlir.constant(128 : i32) : i32
    %scale = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.call @__tile_flash_attn_ext_out_ms_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %dk, %dv, %ne01, %ne11, %nb01, %nb11, %nb21, %scale) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
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

/// Appends trailing constant operands to the one `llvm.call` of `callee`.
fn with_shape(mlir: &str, callee: &str, vals: &[u32]) -> String {
    let call = mlir.find(&format!("llvm.call @{callee}")).unwrap();
    let open = mlir[call..].find('(').unwrap() + call;
    let close = mlir[open..].find(')').unwrap() + open;
    let types_open = mlir[close..].find('(').unwrap() + close;
    let types_close = mlir[types_open..].find(')').unwrap() + types_open;
    let names: Vec<String> = (0..vals.len()).map(|i| format!("%shape{i}")).collect();
    let args: String = names.iter().map(|n| format!(", {n}")).collect();
    let types: String = vals.iter().map(|_| ", i32".to_string()).collect();
    let spliced = format!(
        "{}{args}{}{types}{}",
        &mlir[..close],
        &mlir[close..types_close],
        &mlir[types_close..]
    );
    let at = spliced.find(&format!("llvm.call @{callee}")).unwrap();
    let line = spliced[..at].rfind('\n').unwrap() + 1;
    let decls: String = names
        .iter()
        .zip(vals)
        .map(|(n, v)| format!("    {n} = llvm.mlir.constant({v} : i32) : i32\n"))
        .collect();
    format!("{}{decls}{}", &spliced[..line], &spliced[line..])
}

const SETUP_CALLEE: &str = "__tile_flash_attn_ext_setup_f32";
const SCORE_CALLEE: &str = "__tile_flash_attn_ext_score_f32";
const OUT_CALLEE: &str = "__tile_flash_attn_ext_out_f32";
const OUT_MS_CALLEE: &str = "__tile_flash_attn_ext_out_ms_f32";

/// dk, dv as the shipped kernels were built.
const SHIPPED: [u32; 2] = [64, 64];

#[test]
fn shipped_staging_shape_is_byte_identical() {
    for (mlir, name) in
        [(SETUP, "flash_attn_ext_setup"), (SCORE, "flash_attn_ext_score"), (OUT, "flash_attn_ext_out"), (OUT_MS, "flash_attn_ext_out_ms")]
    {
        assert_eq!(try_emit(mlir).unwrap(), golden(name), "{name} without operands");
    }
    let [dk, dv] = SHIPPED;
    assert_eq!(try_emit(&with_shape(SETUP, SETUP_CALLEE, &[dk])).unwrap(), golden("flash_attn_ext_setup"));
    assert_eq!(try_emit(&with_shape(SCORE, SCORE_CALLEE, &[dk])).unwrap(), golden("flash_attn_ext_score"));
    assert_eq!(try_emit(&with_shape(OUT, OUT_CALLEE, &[dk, dv])).unwrap(), golden("flash_attn_ext_out"));
    assert_eq!(try_emit(&with_shape(OUT_MS, OUT_MS_CALLEE, &[dk, dv])).unwrap(), golden("flash_attn_ext_out_ms"));
}

#[test]
fn head_widths_reach_the_kernels_and_bad_ones_are_refused() {
    let src = try_emit(&with_shape(OUT_MS, OUT_MS_CALLEE, &[128, 256])).unwrap();
    assert!(src.contains("constexpr ushort DK_FIXED  = 128;"), "{src}");
    assert!(src.contains("constexpr ushort DV_FIXED  = 256;"), "{src}");
    assert!(!src.contains("constexpr ushort DK_FIXED  = 64;"), "{src}");
    // The tile shape around the widths stays as the simdgroup matmuls need it.
    for fixed in ["constexpr ushort NQ    = 8;", "constexpr ushort NSG   = 4;", "constexpr ushort C     = 32;"] {
        assert!(src.contains(fixed), "missing {fixed} in\n{src}");
    }
    // The stages that stop before the value head take one operand, not two.
    let s = try_emit(&with_shape(SCORE, SCORE_CALLEE, &[256])).unwrap();
    assert!(s.contains("constexpr ushort DK_FIXED  = 256;"), "{s}");
    assert!(s.contains("constexpr ushort DK8_FIXED = DK_FIXED / 8;  // 32"), "{s}");

    let err = |m: &str, callee: &str, v: &[u32]| try_emit(&with_shape(m, callee, v)).unwrap_err();
    assert!(err(OUT_MS, OUT_MS_CALLEE, &[8, 64]).contains("dk 8"));
    assert!(err(OUT_MS, OUT_MS_CALLEE, &[64, 16]).contains("dv 16"));
    // 32 divides into four 8-wide tiles, but they are consumed in pairs.
    assert!(err(OUT_MS, OUT_MS_CALLEE, &[64, 32]).contains("dv 32"));
    assert!(err(OUT_MS, OUT_MS_CALLEE, &[2048, 64]).contains("dk 2048"));
    // Threadgroup memory is 32 KiB and these widths want more.
    assert!(err(OUT_MS, OUT_MS_CALLEE, &[1024, 1024]).contains("threadgroup memory"));
    // A stage that takes two operands will not accept one.
    assert!(!err(OUT_MS, OUT_MS_CALLEE, &[64]).is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn staged_kernels_match_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let queue = dev.new_command_queue();
    let mut seed = 0x51ED_2718_3141_C0DEu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Values that survive the cast to binary16 exactly, so the reference and
        // the kernel see identical inputs.
        let x = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        (x * 256.0).round() / 256.0
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

    // The shipped widths, then three that move both of them independently.
    for shape @ [dk, dv] in [SHIPPED, [128, 128], [16, 64], [32, 192]] {
        let (dk, dv) = (dk as usize, dv as usize);
        // These stages walk the keys in whole passes of `c`, so the key count
        // is one every shape here divides; the query count deliberately is not.
        let (n_rows, n_keys) = (5usize, 64usize);
        let q: Vec<f32> = (0..n_rows * dk).map(|_| next()).collect();
        let k: Vec<f32> = (0..n_keys * dk).map(|_| next()).collect();
        let v: Vec<f32> = (0..n_keys * dv).map(|_| next()).collect();
        let scale = 1.0f32 / (dk as f32).sqrt();

        // What each stage should produce, in f64.
        let scores: Vec<Vec<f64>> = (0..n_rows)
            .map(|row| {
                (0..n_keys)
                    .map(|ic| {
                        let dot: f64 = (0..dk).map(|i| (q[row * dk + i] * k[ic * dk + i]) as f64).sum();
                        dot * scale as f64
                    })
                    .collect()
            })
            .collect();
        let want_m: Vec<f64> = scores.iter().map(|s| s.iter().copied().fold(f64::NEG_INFINITY, f64::max)).collect();
        let want_s: Vec<f64> =
            scores.iter().zip(&want_m).map(|(s, m)| s.iter().map(|x| (x - m).exp()).sum()).collect();
        let mut want_o = vec![0f32; n_rows * dv];
        for row in 0..n_rows {
            for d in 0..dv {
                let num: f64 =
                    (0..n_keys).map(|ic| (scores[row][ic] - want_m[row]).exp() * v[ic * dv + d] as f64).sum();
                want_o[row * dv + d] = (num / want_s[row]) as f32;
            }
        }

        let o = MTLResourceOptions::StorageModeShared;
        let bq = dev.new_buffer_with_data(q.as_ptr() as *const _, (q.len() * 4) as u64, o);
        let hk = to_half(&k);
        let hv = to_half(&v);
        let bk = dev.new_buffer_with_data(hk.as_ptr() as *const _, (hk.len() * 2) as u64, o);
        let bv = dev.new_buffer_with_data(hv.as_ptr() as *const _, (hv.len() * 2) as u64, o);
        let empty = dev.new_buffer(4, o);
        // The last stage folds an additive mask and an attention sink in
        // unconditionally: a zero mask and a sink far below every score leave
        // the plain attention the other stages compute.
        let zeros = vec![0u16; n_rows * n_keys];
        let bmask = dev.new_buffer_with_data(zeros.as_ptr() as *const _, (zeros.len() * 2) as u64, o);
        let sink = [-1.0e30f32];
        let bsinks = dev.new_buffer_with_data(sink.as_ptr() as *const _, 4, o);

        let run = |mlir: &str, callee: &str, entry: &str, vals: &[u32], scalars: &[u32], out_len: usize| {
            let src = try_emit(&with_shape(mlir, callee, vals)).unwrap();
            let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
            let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
            let bo = dev.new_buffer((out_len * 4) as u64, o);
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bq, &bk, &bv, &bmask, &bsinks, &empty, &empty, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            // dk, dv, the stage's own dimensions and strides, and, where the
            // stage has one, the scale as its float bits.
            for (i, s) in scalars.iter().enumerate() {
                enc.set_bytes(8 + i as u64, 4, s as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((n_rows as u64).div_ceil(8), 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, out_len) }.to_vec()
        };

        let (dku, dvu) = (dk as u32, dv as u32);
        let (nb01, nb11, nb21) = ((dk * 4) as u32, (dk * 2) as u32, (dv * 2) as u32);
        let near = |got: f32, want: f64, what: &str, i: usize| {
            let w = want as f32;
            assert!(
                (got - w).abs() <= 2e-3 * w.abs().max(1e-2),
                "{what} at shape {shape:?}, index {i}: got {got}, want {w}"
            );
        };

        // M37a: the staged queries are echoed back as floats.
        let got = run(SETUP, SETUP_CALLEE, "ds4_flash_attn_ext_setup", &[dku], &[dku, dvu, n_rows as u32, nb01], n_rows * dk);
        for (i, (g, w)) in got.iter().zip(&q).enumerate() {
            near(*g, *w as f64, "setup echo", i);
        }

        // M37b: the softmax denominator and running maximum per query row.
        let got = run(
            SCORE,
            SCORE_CALLEE,
            "ds4_flash_attn_ext_score",
            &[dku],
            &[dku, dvu, n_rows as u32, n_keys as u32, nb01, nb11, scale.to_bits()],
            n_rows * 2,
        );
        for row in 0..n_rows {
            near(got[row * 2], want_s[row], "score S", row);
            near(got[row * 2 + 1], want_m[row], "score M", row);
        }

        // M37c and M37d: the attention output itself.
        for (mlir, callee, entry) in [
            (OUT, OUT_CALLEE, "ds4_flash_attn_ext_out"),
            (OUT_MS, OUT_MS_CALLEE, "ds4_flash_attn_ext_out_ms"),
        ] {
            let got = run(
                mlir,
                callee,
                entry,
                &[dku, dvu],
                &[dku, dvu, n_rows as u32, n_keys as u32, nb01, nb11, nb21, scale.to_bits()],
                n_rows * dv,
            );
            for (i, (g, w)) in got.iter().zip(&want_o).enumerate() {
                near(*g, *w as f64, entry, i);
            }
        }
    }
}
