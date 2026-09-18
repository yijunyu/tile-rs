//! The q8_0 matvec kernels, dense and expert-routed, from their intrinsics.
//!
//! Both split their work three ways: over simdgroups, over the output rows a
//! simdgroup owns, and over the slice of a 32-element quantization block one
//! lane takes. All three were written into the text, and one of them, the row
//! count, also into a literal initializer list that only had room for two rows.
//! The three are now trailing operands. At the shipped split the emitted text
//! is unchanged; at other splits, on a Metal GPU, both kernels must match a CPU
//! reference of the same dequantized matvec.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const DENSE: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mv_q8_0_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(64 : i32) : i32
    %ne01 = llvm.mlir.constant(64 : i32) : i32
    %ne0  = llvm.mlir.constant(64 : i32) : i32
    %ne1  = llvm.mlir.constant(4  : i32) : i32
    %ne12 = llvm.mlir.constant(1  : i32) : i32
    %r2   = llvm.mlir.constant(1  : i32) : i32
    %r3   = llvm.mlir.constant(1  : i32) : i32
    %nb01 = llvm.mlir.constant(68   : i32) : i32
    %nb02 = llvm.mlir.constant(4352 : i32) : i32
    %nb03 = llvm.mlir.constant(4352 : i32) : i32
    %nb11 = llvm.mlir.constant(256  : i32) : i32
    %nb12 = llvm.mlir.constant(1024 : i32) : i32
    %nb13 = llvm.mlir.constant(1024 : i32) : i32
    %r = llvm.call @__tile_mul_mv_q8_0_f32(%arg0, %arg1, %arg2, %ne00, %ne01, %ne0, %ne1, %ne12, %r2, %r3, %nb01, %nb02, %nb03, %nb11, %nb12, %nb13) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const ROUTED: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mv_id_q8_0_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(64 : i32) : i32
    %ne01 = llvm.mlir.constant(64 : i32) : i32
    %ne0  = llvm.mlir.constant(64 : i32) : i32
    %ne1  = llvm.mlir.constant(4  : i32) : i32
    %ne11 = llvm.mlir.constant(1  : i32) : i32
    %nei0 = llvm.mlir.constant(4  : i32) : i32
    %nbi1 = llvm.mlir.constant(16 : i32) : i32
    %nb01 = llvm.mlir.constant(68   : i32) : i32
    %nb02 = llvm.mlir.constant(4352 : i32) : i32
    %nb11 = llvm.mlir.constant(256  : i32) : i32
    %nb12 = llvm.mlir.constant(1024 : i32) : i32
    %r = llvm.call @__tile_mul_mv_id_q8_0_f32(%arg0, %arg1, %arg2, %arg3, %ne00, %ne01, %ne0, %ne1, %ne11, %nei0, %nbi1, %nb01, %nb02, %nb11, %nb12) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const DENSE_CALLEE: &str = "__tile_mul_mv_q8_0_f32";
const ROUTED_CALLEE: &str = "__tile_mul_mv_id_q8_0_f32";

/// nsg, nr0, nq as the kernels were written.
const SHIPPED: [u32; 3] = [4, 2, 8];

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

#[test]
fn the_shipped_split_is_what_the_kernels_had() {
    for (mlir, callee) in [(DENSE, DENSE_CALLEE), (ROUTED, ROUTED_CALLEE)] {
        let bare = try_emit(mlir).unwrap();
        assert_eq!(bare, try_emit(&with_split(mlir, callee, &SHIPPED)).unwrap());
        for want in [
            "constexpr short NW    = 32;",
            "constexpr short NSG   = 4;",
            "constexpr short NR0   = 2;",
            "constexpr short NQ    = 8;",
            "constexpr short QK8_0 = 32;",
            "float sumf[NR0] = { 0.0f, 0.0f };",
        ] {
            assert!(bare.contains(want), "missing {want} in\n{bare}");
        }
    }
}

#[test]
fn the_split_reaches_the_kernels_and_bad_ones_are_refused() {
    let src = try_emit(&with_split(DENSE, DENSE_CALLEE, &[8, 3, 16])).unwrap();
    for want in [
        "constexpr short NSG   = 8;",
        "constexpr short NR0   = 3;",
        "constexpr short NQ    = 16;",
        // The accumulator list follows the row count rather than assuming two.
        "float sumf[NR0] = { 0.0f, 0.0f, 0.0f };",
    ] {
        assert!(src.contains(want), "missing {want} in\n{src}");
    }
    // The simdgroup width and the q8_0 block size are the hardware and the format.
    assert!(src.contains("constexpr short NW    = 32;") && src.contains("constexpr short QK8_0 = 32;"), "{src}");
    // The routed kernel is a separate copy of the same emitter, so it gets the
    // same reading. Its lane slice changes nothing a result can show, so only
    // the text can hold it to the operand.
    let r = try_emit(&with_split(ROUTED, ROUTED_CALLEE, &[1, 1, 32])).unwrap();
    for want in [
        "constexpr short NSG   = 1;",
        "constexpr short NR0   = 1;",
        "constexpr short NQ    = 32;",
        "float sumf[NR0] = { 0.0f };",
    ] {
        assert!(r.contains(want), "missing {want} in\n{r}");
    }

    let err = |m: &str, c: &str, v: &[u32]| try_emit(&with_split(m, c, v)).unwrap_err();
    // Zero is refused by the operand reader before the kernel sees it.
    assert!(err(DENSE, DENSE_CALLEE, &[0, 2, 8]).contains("(nsg) must be a positive integer constant"));
    assert!(err(DENSE, DENSE_CALLEE, &[4, 0, 8]).contains("(nr0) must be a positive integer constant"));
    assert!(err(DENSE, DENSE_CALLEE, &[64, 2, 8]).contains("nsg 64"));
    assert!(err(DENSE, DENSE_CALLEE, &[4, 32, 8]).contains("nr0 32"));
    assert!(err(DENSE, DENSE_CALLEE, &[4, 2, 12]).contains("nq 12"));
    assert!(!err(DENSE, DENSE_CALLEE, &[4, 2]).is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let queue = dev.new_command_queue();
    const NE00: usize = 256; // 8 q8_0 blocks per row
    const ROWS: usize = 12; // deliberately not a multiple of every row count
    const COLS: usize = 3; // src1 vectors
    const EXPERTS: usize = 4;
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    // A q8_0 row: per block, an f16 scale then 32 signed bytes.
    let quantize_rows = |rows: usize, next: &mut dyn FnMut() -> u64| -> (Vec<u8>, Vec<f32>) {
        let nb = NE00 / BLOCK;
        let mut bytes = vec![0u8; rows * nb * BLOCK_BYTES];
        let mut deq = vec![0f32; rows * NE00];
        for r in 0..rows {
            for b in 0..nb {
                // A scale that is exact in binary16.
                let scale = 1.0f32 / (1 << (1 + (next() % 4))) as f32;
                let half = {
                    let bits = scale.to_bits();
                    let sign = ((bits >> 16) & 0x8000) as u16;
                    let e = ((bits >> 23) & 0xff) as i32 - 127;
                    let mant = ((bits & 0x7f_ffff) >> 13) as u16;
                    sign | (((e + 15) as u16) << 10) | mant
                };
                let off = (r * nb + b) * BLOCK_BYTES;
                bytes[off..off + 2].copy_from_slice(&half.to_le_bytes());
                for i in 0..BLOCK {
                    let q = (next() % 255) as i64 - 127;
                    bytes[off + 2 + i] = (q as i8) as u8;
                    deq[r * NE00 + b * BLOCK + i] = q as f32 * scale;
                }
            }
        }
        (bytes, deq)
    };

    let (w_bytes, w_deq) = quantize_rows(ROWS * EXPERTS, &mut next);
    let x: Vec<f32> = (0..COLS * NE00).map(|_| ((next() % 512) as f32 - 256.0) / 256.0).collect();

    let o = MTLResourceOptions::StorageModeShared;
    let bw = dev.new_buffer_with_data(w_bytes.as_ptr() as *const _, w_bytes.len() as u64, o);
    let bx = dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as u64, o);
    // Slot i takes expert (EXPERTS - 1 - i), so a wrong route shows up.
    let ids: Vec<i32> = (0..EXPERTS).map(|i| (EXPERTS - 1 - i) as i32).collect();
    let bids = dev.new_buffer_with_data(ids.as_ptr() as *const _, (ids.len() * 4) as u64, o);

    let nb01 = (NE00 / BLOCK * BLOCK_BYTES) as u32;
    let nb02 = nb01 * ROWS as u32;
    let nb11 = (NE00 * 4) as u32;

    let dot = |row: usize, col: usize| -> f32 {
        (0..NE00).map(|k| w_deq[row * NE00 + k] * x[col * NE00 + k]).sum()
    };

    for split @ [nsg, nr0, nq] in [SHIPPED, [1, 1, 32], [8, 3, 4], [2, 4, 16], [4, 2, 1]] {
        let build = |mlir: &str, callee: &str, entry: &str| {
            let src = try_emit(&with_split(mlir, callee, &[nsg, nr0, nq])).unwrap();
            let lib =
                dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
            dev.new_compute_pipeline_state(&desc).unwrap()
        };
        let groups = (ROWS as u64).div_ceil(nr0 as u64);

        // Dense: dst[col][row].
        {
            let pipe = build(DENSE, DENSE_CALLEE, "ds4_kernel_mul_mv_q8_0_f32");
            let bo = dev.new_buffer((ROWS * COLS * 4) as u64, o);
            let u: [u32; 13] = [
                NE00 as u32, ROWS as u32, ROWS as u32, COLS as u32, 1, 1, 1, nb01, nb02, nb02, nb11,
                nb11 * COLS as u32, nb11 * COLS as u32,
            ];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bw, &bx, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in u.iter().enumerate() {
                enc.set_bytes(3 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(groups, COLS as u64, 1),
                MTLSize::new((nsg * 32) as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, ROWS * COLS) };
            for col in 0..COLS {
                for row in 0..ROWS {
                    let want = dot(row, col);
                    let g = got[col * ROWS + row];
                    assert!(
                        (g - want).abs() <= 1e-3 * want.abs().max(1e-2),
                        "dense at split {split:?}, col {col} row {row}: got {g}, want {want}"
                    );
                }
            }
        }

        // Routed: slot idx reads expert ids[idx] and writes dst[idx][row].
        {
            let pipe = build(ROUTED, ROUTED_CALLEE, "ds4_kernel_mul_mv_id_q8_0_f32");
            let bo = dev.new_buffer((ROWS * EXPERTS * 4) as u64, o);
            let u: [u32; 11] = [
                NE00 as u32, ROWS as u32, ROWS as u32, EXPERTS as u32, 1, EXPERTS as u32,
                (EXPERTS * 4) as u32, nb01, nb02, nb11, nb11,
            ];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bw, &bx, &bids, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in u.iter().enumerate() {
                enc.set_bytes(4 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(groups, 1, EXPERTS as u64),
                MTLSize::new((nsg * 32) as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, ROWS * EXPERTS) };
            for idx in 0..EXPERTS {
                let expert = ids[idx] as usize;
                for row in 0..ROWS {
                    let want = dot(expert * ROWS + row, 0);
                    let g = got[idx * ROWS + row];
                    assert!(
                        (g - want).abs() <= 1e-3 * want.abs().max(1e-2),
                        "routed at split {split:?}, slot {idx} row {row}: got {g}, want {want}"
                    );
                }
            }
        }
    }
}
