//! The four wide matvecs: the paired half-precision one and the three routed
//! IQ2_XXS ones.
//!
//! Each wrote two numbers into its text, the simdgroups a threadgroup runs and
//! the output rows each one owns, and each backed the row count with a literal
//! accumulator list of exactly four slots. Both numbers are now trailing
//! operands and the lists are generated to match.
//!
//! The paired half kernel is checked against a CPU reference. For the IQ2_XXS
//! kernels the reference is the kernel at its shipped split, whose text the
//! first test pins: a result that depends on how the work was divided is a bug
//! in the division, and that is what these three can get wrong.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const PAIR4: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mv_f16_f32_pair_4(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(32 : i32) : i32
    %ne01 = llvm.mlir.constant(64 : i32) : i32
    %ne0  = llvm.mlir.constant(64 : i32) : i32
    %ne1  = llvm.mlir.constant(4  : i32) : i32
    %ne12 = llvm.mlir.constant(1  : i32) : i32
    %r2   = llvm.mlir.constant(1  : i32) : i32
    %r3   = llvm.mlir.constant(1  : i32) : i32
    %nb01 = llvm.mlir.constant(64   : i32) : i32
    %nb02 = llvm.mlir.constant(4096 : i32) : i32
    %nb03 = llvm.mlir.constant(4096 : i32) : i32
    %nb11 = llvm.mlir.constant(128  : i32) : i32
    %nb12 = llvm.mlir.constant(512  : i32) : i32
    %nb13 = llvm.mlir.constant(512  : i32) : i32
    %r = llvm.call @__tile_mul_mv_f16_f32_pair_4(%arg0, %arg1, %arg2, %arg3, %arg4, %ne00, %ne01, %ne0, %ne1, %ne12, %r2, %r3, %nb01, %nb02, %nb03, %nb11, %nb12, %nb13) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const IQ2: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mv_id_iq2_xxs_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(256 : i32) : i32
    %ne01 = llvm.mlir.constant(16  : i32) : i32
    %ne0  = llvm.mlir.constant(16  : i32) : i32
    %ne1  = llvm.mlir.constant(1   : i32) : i32
    %ne11 = llvm.mlir.constant(1   : i32) : i32
    %nei0 = llvm.mlir.constant(2   : i32) : i32
    %nbi1 = llvm.mlir.constant(8   : i32) : i32
    %nb01 = llvm.mlir.constant(66   : i32) : i32
    %nb02 = llvm.mlir.constant(1056 : i32) : i32
    %nb11 = llvm.mlir.constant(1024 : i32) : i32
    %nb12 = llvm.mlir.constant(1024 : i32) : i32
    %r = llvm.call @__tile_mul_mv_id_iq2_xxs_f32(%arg0, %arg1, %arg2, %arg3, %ne00, %ne01, %ne0, %ne1, %ne11, %nei0, %nbi1, %nb01, %nb02, %nb11, %nb12) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const IQ2_PAIR: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mv_id_iq2_xxs_pair_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(256 : i32) : i32
    %ne01 = llvm.mlir.constant(16  : i32) : i32
    %ne0  = llvm.mlir.constant(16  : i32) : i32
    %ne1  = llvm.mlir.constant(1   : i32) : i32
    %ne11 = llvm.mlir.constant(1   : i32) : i32
    %nei0 = llvm.mlir.constant(2   : i32) : i32
    %nbi1 = llvm.mlir.constant(8   : i32) : i32
    %nb01 = llvm.mlir.constant(66   : i32) : i32
    %nb02 = llvm.mlir.constant(1056 : i32) : i32
    %nb11 = llvm.mlir.constant(1024 : i32) : i32
    %nb12 = llvm.mlir.constant(1024 : i32) : i32
    %r = llvm.call @__tile_mul_mv_id_iq2_xxs_pair_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %ne00, %ne01, %ne0, %ne1, %ne11, %nei0, %nbi1, %nb01, %nb02, %nb11, %nb12) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const IQ2_SWIGLU: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(256 : i32) : i32
    %ne01 = llvm.mlir.constant(16  : i32) : i32
    %ne0  = llvm.mlir.constant(16  : i32) : i32
    %ne1  = llvm.mlir.constant(1   : i32) : i32
    %ne11 = llvm.mlir.constant(1   : i32) : i32
    %nei0 = llvm.mlir.constant(2   : i32) : i32
    %nbi1 = llvm.mlir.constant(8   : i32) : i32
    %nb01 = llvm.mlir.constant(66   : i32) : i32
    %nb02 = llvm.mlir.constant(1056 : i32) : i32
    %nb11 = llvm.mlir.constant(1024 : i32) : i32
    %nb12 = llvm.mlir.constant(1024 : i32) : i32
    %mid_row_stride = llvm.mlir.constant(64 : i32) : i32
    %weight_stride  = llvm.mlir.constant(4  : i32) : i32
    %clamp_value    = llvm.mlir.constant(7.0 : f32) : f32
    %r = llvm.call @__tile_mul_mv_id_iq2_xxs_pair_swiglu_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %ne00, %ne01, %ne0, %ne1, %ne11, %nei0, %nbi1, %nb01, %nb02, %nb11, %nb12, %mid_row_stride, %weight_stride, %clamp_value) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32) -> i32
    llvm.return
  }
}
"#;

fn fixture(name: &str) -> String {
    match name {
        "mul_mv_f16_f32_pair_4" => PAIR4,
        "mul_mv_id_iq2_xxs_f32" => IQ2,
        "mul_mv_id_iq2_xxs_pair_f32" => IQ2_PAIR,
        "mul_mv_id_iq2_xxs_pair_swiglu_f32" => IQ2_SWIGLU,
        other => panic!("no fixture for {other}"),
    }
    .to_string()
}

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

/// Appends trailing constant operands to the one `llvm.call` of `callee`.
fn with_split(mlir: &str, callee: &str, vals: &[u32]) -> String {
    let call = mlir.find(&format!("llvm.call @{callee}")).unwrap();
    let open = mlir[call..].find('(').unwrap() + call;
    let close = mlir[open..].find(')').unwrap() + open;
    let types_open = mlir[close..].find('(').unwrap() + close;
    let types_close = mlir[types_open..].find(')').unwrap() + types_open;
    let names: Vec<String> = (0..vals.len()).map(|i| format!("%split{i}")).collect();
    let args: String = names.iter().map(|n| format!(", {n}")).collect();
    let types: String = vals.iter().map(|_| ", i32".to_string()).collect();
    let spliced =
        format!("{}{args}{}{types}{}", &mlir[..close], &mlir[close..types_close], &mlir[types_close..]);
    let at = spliced.find(&format!("llvm.call @{callee}")).unwrap();
    let line = spliced[..at].rfind('\n').unwrap() + 1;
    let decls: String = names
        .iter()
        .zip(vals)
        .map(|(n, v)| format!("    {n} = llvm.mlir.constant({v} : i32) : i32\n"))
        .collect();
    format!("{}{decls}{}", &spliced[..line], &spliced[line..])
}

/// fixture stem, intrinsic, entry point, shipped simdgroup count.
const KERNELS: [(&str, &str, &str, u32); 4] = [
    ("mul_mv_f16_f32_pair_4", "__tile_mul_mv_f16_f32_pair_4", "ds4_kernel_mul_mv_f16_f32_pair_4", 4),
    ("mul_mv_id_iq2_xxs_f32", "__tile_mul_mv_id_iq2_xxs_f32", "ds4_kernel_mul_mv_id_iq2_xxs_f32", 2),
    (
        "mul_mv_id_iq2_xxs_pair_f32",
        "__tile_mul_mv_id_iq2_xxs_pair_f32",
        "ds4_kernel_mul_mv_id_iq2_xxs_pair_f32",
        2,
    ),
    (
        "mul_mv_id_iq2_xxs_pair_swiglu_f32",
        "__tile_mul_mv_id_iq2_xxs_pair_swiglu_f32",
        "ds4_kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32",
        2,
    ),
];

#[test]
fn the_shipped_splits_are_what_the_kernels_had() {
    for (stem, callee, _, nsg) in KERNELS {
        let m = fixture(stem);
        let bare = try_emit(&m).unwrap();
        assert_eq!(bare, try_emit(&with_split(&m, callee, &[nsg, 4])).unwrap(), "{stem}");
        assert!(bare.contains(&format!("= {nsg};")), "{stem} lost its simdgroup count:\n{bare}");
        assert!(bare.contains("0.0f, 0.0f, 0.0f, 0.0f };"), "{stem} lost its four-row accumulator:\n{bare}");
    }
}

#[test]
fn the_split_reaches_the_kernels_and_bad_ones_are_refused() {
    for (stem, callee, _, _) in KERNELS {
        let m = fixture(stem);
        let src = try_emit(&with_split(&m, callee, &[4, 2])).unwrap();
        assert!(src.contains("0.0f, 0.0f };"), "{stem} kept a four-row accumulator at two rows:\n{src}");
        assert!(!src.contains("0.0f, 0.0f, 0.0f, 0.0f };"), "{stem} still has four slots:\n{src}");
        let wide = try_emit(&with_split(&m, callee, &[1, 6])).unwrap();
        assert!(
            wide.contains("0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };"),
            "{stem} did not grow its accumulator to six rows:\n{wide}"
        );
        assert!(try_emit(&with_split(&m, callee, &[64, 4])).unwrap_err().contains("nsg 64"), "{stem}");
        assert!(try_emit(&with_split(&m, callee, &[4, 32])).unwrap_err().contains("nr0 32"), "{stem}");
        if stem != "mul_mv_f16_f32_pair_4" {
            // The routed kernels hand every lane an equal run of the tables they
            // stage, so eight simdgroups would leave the sign table short.
            let e = try_emit(&with_split(&m, callee, &[8, 4])).unwrap_err();
            assert!(e.contains("must be 1, 2 or 4"), "{stem}: {e}");
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_agree_across_splits() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let queue = dev.new_command_queue();

    const NE00: usize = 512; // two IQ2_XXS super-blocks per row
    const ROWS: usize = 48; // divisible by every simdgroups x rows product below
    const EXPERTS: usize = 2;
    const IQ2_BLOCK_BYTES: usize = 66;
    const QK_K: usize = 256;

    let mut state = 0xC0DE_FEED_2024_9001u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    // Weights for the routed kernels are raw IQ2_XXS blocks; the kernel reads
    // them through its own lookup tables, so arbitrary bytes are valid input.
    let blocks_per_row = NE00 / QK_K;
    let iq_bytes: Vec<u8> = (0..ROWS * EXPERTS * blocks_per_row * IQ2_BLOCK_BYTES)
        .map(|i| if i % IQ2_BLOCK_BYTES < 2 { (next() % 0x3c) as u8 } else { (next() & 0xff) as u8 })
        .collect();
    let x: Vec<f32> = (0..NE00).map(|_| ((next() % 512) as f32 - 256.0) / 256.0).collect();
    let ids: Vec<i32> = (0..EXPERTS).map(|i| (EXPERTS - 1 - i) as i32).collect();
    let mid: Vec<f32> = (0..ROWS).map(|_| ((next() % 512) as f32 - 256.0) / 512.0).collect();

    // The paired half kernel gets plain half weights, so a CPU reference is exact.
    let hw: Vec<f32> = (0..2 * ROWS * NE00).map(|_| ((next() % 129) as i64 - 64) as f32 / 256.0).collect();
    let halves: Vec<u16> = hw
        .iter()
        .map(|&v| {
            if v == 0.0 {
                return 0;
            }
            let b = v.to_bits();
            ((b >> 16) & 0x8000) as u16
                | ((((b >> 23) & 0xff) as i32 - 127 + 15) as u16) << 10
                | ((b & 0x7f_ffff) >> 13) as u16
        })
        .collect();

    let o = MTLResourceOptions::StorageModeShared;
    let raw = |b: &[u8]| dev.new_buffer_with_data(b.as_ptr() as *const _, b.len() as u64, o);
    let b_iq = raw(&iq_bytes);
    let b_half_a = dev.new_buffer_with_data(halves.as_ptr() as *const _, (ROWS * NE00 * 2) as u64, o);
    let b_half_b = dev.new_buffer_with_data(
        unsafe { halves.as_ptr().add(ROWS * NE00) } as *const _,
        (ROWS * NE00 * 2) as u64,
        o,
    );
    let b_x = dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as u64, o);
    let b_ids = dev.new_buffer_with_data(ids.as_ptr() as *const _, (ids.len() * 4) as u64, o);
    let b_mid = dev.new_buffer_with_data(mid.as_ptr() as *const _, (mid.len() * 4) as u64, o);

    let iq_nb01 = (blocks_per_row * IQ2_BLOCK_BYTES) as u32;
    let iq_nb02 = iq_nb01 * ROWS as u32;
    let nb11 = (NE00 * 4) as u32;

    let run = |stem: &str, callee: &str, entry: &str, nsg: u32, nr0: u32| -> Vec<Vec<f32>> {
        let m = fixture(stem);
        let src = try_emit(&with_split(&m, callee, &[nsg, nr0])).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();

        let n_out = if stem == "mul_mv_f16_f32_pair_4" { ROWS } else { ROWS * EXPERTS };
        let n_bufs = match stem {
            "mul_mv_f16_f32_pair_4" => 2,
            "mul_mv_id_iq2_xxs_f32" => 1,
            "mul_mv_id_iq2_xxs_pair_f32" => 2,
            _ => 3,
        };
        let outs: Vec<_> = (0..n_bufs).map(|_| dev.new_buffer((n_out * 4) as u64, o)).collect();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);

        let (grid, threads) = match stem {
            // One threadgroup per NR0 rows; the routed kernels take NSG x NR0.
            "mul_mv_f16_f32_pair_4" => (MTLSize::new((ROWS as u64).div_ceil(nr0 as u64), 1, 1), nsg * 32),
            _ => (
                MTLSize::new((ROWS as u64).div_ceil((nsg * nr0) as u64), 1, EXPERTS as u64),
                nsg * 32,
            ),
        };
        match stem {
            "mul_mv_f16_f32_pair_4" => {
                for (i, b) in [&b_half_a, &b_half_b, &b_x, &outs[0], &outs[1]].into_iter().enumerate() {
                    enc.set_buffer(i as u64, Some(b), 0);
                }
                let u: [u32; 13] = [
                    NE00 as u32, ROWS as u32, ROWS as u32, 1, 1, 1, 1, (NE00 * 2) as u32,
                    (ROWS * NE00 * 2) as u32, (ROWS * NE00 * 2) as u32, nb11, nb11, nb11,
                ];
                for (i, v) in u.iter().enumerate() {
                    enc.set_bytes(5 + i as u64, 4, v as *const u32 as *const _);
                }
            }
            "mul_mv_id_iq2_xxs_f32" => {
                for (i, b) in [&b_iq, &b_x, &b_ids, &outs[0]].into_iter().enumerate() {
                    enc.set_buffer(i as u64, Some(b), 0);
                }
                let u: [u32; 11] = [
                    NE00 as u32, ROWS as u32, ROWS as u32, EXPERTS as u32, 1, EXPERTS as u32,
                    (EXPERTS * 4) as u32, iq_nb01, iq_nb02, nb11, nb11,
                ];
                for (i, v) in u.iter().enumerate() {
                    enc.set_bytes(4 + i as u64, 4, v as *const u32 as *const _);
                }
            }
            "mul_mv_id_iq2_xxs_pair_f32" => {
                for (i, b) in [&b_iq, &b_x, &b_ids, &outs[0], &outs[1], &b_iq].into_iter().enumerate() {
                    enc.set_buffer(i as u64, Some(b), 0);
                }
                let u: [u32; 11] = [
                    NE00 as u32, ROWS as u32, ROWS as u32, EXPERTS as u32, 1, EXPERTS as u32,
                    (EXPERTS * 4) as u32, iq_nb01, iq_nb02, nb11, nb11,
                ];
                for (i, v) in u.iter().enumerate() {
                    enc.set_bytes(6 + i as u64, 4, v as *const u32 as *const _);
                }
            }
            _ => {
                for (i, b) in
                    [&b_iq, &b_x, &b_ids, &outs[0], &outs[1], &outs[2], &b_iq, &b_mid].into_iter().enumerate()
                {
                    enc.set_buffer(i as u64, Some(b), 0);
                }
                let u: [u32; 13] = [
                    NE00 as u32, ROWS as u32, ROWS as u32, EXPERTS as u32, 1, EXPERTS as u32,
                    (EXPERTS * 4) as u32, iq_nb01, iq_nb02, nb11, nb11, ROWS as u32, iq_nb01,
                ];
                for (i, v) in u.iter().enumerate() {
                    enc.set_bytes(8 + i as u64, 4, v as *const u32 as *const _);
                }
                let clamp = 30.0f32;
                enc.set_bytes(21, 4, &clamp as *const f32 as *const _);
            }
        }
        enc.dispatch_thread_groups(grid, MTLSize::new(threads as u64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        outs.iter()
            .map(|b| unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n_out) }.to_vec())
            .collect()
    };

    for (stem, callee, entry, shipped_nsg) in KERNELS {
        let base = run(stem, callee, entry, shipped_nsg, 4);

        if stem == "mul_mv_f16_f32_pair_4" {
            // This one has an independent reference: plain half weights.
            for (which, got) in base.iter().enumerate() {
                for row in 0..ROWS {
                    let off = which * ROWS * NE00 + row * NE00;
                    let want: f32 = (0..NE00).map(|k| hw[off + k] * x[k]).sum();
                    assert!(
                        (got[row] - want).abs() <= 2e-3 * want.abs().max(1e-2),
                        "{stem} stream {which} row {row}: got {}, want {want}",
                        got[row]
                    );
                }
            }
        }

        // Eight simdgroups is out of reach for the routed kernels, which stage
        // fixed-size tables; the paired half kernel stages nothing.
        let splits: &[(u32, u32)] = if stem == "mul_mv_f16_f32_pair_4" {
            &[(1, 1), (2, 2), (4, 3), (8, 6), (1, 8)]
        } else {
            &[(1, 1), (2, 2), (4, 3), (1, 6), (4, 8)]
        };
        for &(nsg, nr0) in splits {
            let got = run(stem, callee, entry, nsg, nr0);
            for (which, (g, b)) in got.iter().zip(&base).enumerate() {
                for (row, (a, c)) in g.iter().zip(b).enumerate() {
                    assert!(
                        (a - c).abs() <= 1e-4 * c.abs().max(1e-3),
                        "{stem} stream {which} row {row} moved with the split \
                         ({nsg} simdgroups, {nr0} rows): {a} against {c} at the shipped split"
                    );
                }
            }
        }
    }
}
