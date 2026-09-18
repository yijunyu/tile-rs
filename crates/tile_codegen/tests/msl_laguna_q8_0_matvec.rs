//! The Laguna q8_0 matvec, the last Metal emitter that still wrote its work
//! split into the kernel text.
//!
//! It carried three numbers: the simdgroups a threadgroup runs, the output rows
//! each one owns, and the contiguous quants a lane takes from a block. The row
//! count was also a four-slot accumulator list. All three are now trailing
//! operands. The committed kernel pins the shipped split byte for byte; at
//! other splits, on a Metal GPU, it must match a dequantized CPU reference.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const MATVEC: &str = r#"
module {
  llvm.func @laguna_q8_0_matvec_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @__tile_laguna_q8_0_matvec_f32(%a, %r, %c) : (i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const CALLEE: &str = "__tile_laguna_q8_0_matvec_f32";

/// nsg, nr0, nq as the committed kernel was built.
const SHIPPED: [u32; 3] = [4, 4, 8];

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

/// Appends trailing constant operands to the one `llvm.call` of `callee`.
fn with_split(vals: &[u32]) -> String {
    let mlir = MATVEC;
    let call = mlir.find(&format!("llvm.call @{CALLEE}")).unwrap();
    let open = mlir[call..].find('(').unwrap() + call;
    let close = mlir[open..].find(')').unwrap() + open;
    let types_open = mlir[close..].find('(').unwrap() + close;
    let types_close = mlir[types_open..].find(')').unwrap() + types_open;
    let names: Vec<String> = (0..vals.len()).map(|i| format!("%split{i}")).collect();
    let args: String = names.iter().map(|n| format!(", {n}")).collect();
    let types: String = vals.iter().map(|_| ", i32".to_string()).collect();
    let spliced =
        format!("{}{args}{}{types}{}", &mlir[..close], &mlir[close..types_close], &mlir[types_close..]);
    let at = spliced.find(&format!("llvm.call @{CALLEE}")).unwrap();
    let line = spliced[..at].rfind('\n').unwrap() + 1;
    let decls: String = names
        .iter()
        .zip(vals)
        .map(|(n, v)| format!("    {n} = llvm.mlir.constant({v} : i32) : i32\n"))
        .collect();
    format!("{}{decls}{}", &spliced[..line], &spliced[line..])
}

#[test]
fn the_shipped_split_is_what_the_kernel_had() {
    let bare = try_emit(MATVEC).unwrap();
    assert_eq!(bare, try_emit(&with_split(&SHIPPED)).unwrap());
    for want in [
        "constexpr uint NR0 = 4u;",
        "constexpr uint NSG = 4u;",
        "constexpr uint NQ  = 8u;",
        "float sumf[NR0] = {0.0f, 0.0f, 0.0f, 0.0f};",
    ] {
        assert!(bare.contains(want), "missing {want} in\n{bare}");
    }
}

#[test]
fn the_split_reaches_the_kernel_and_bad_ones_are_refused() {
    let src = try_emit(&with_split(&[2, 6, 16])).unwrap();
    for want in [
        "constexpr uint NR0 = 6u;",
        "constexpr uint NSG = 2u;",
        "constexpr uint NQ  = 16u;",
        "float sumf[NR0] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};",
    ] {
        assert!(src.contains(want), "missing {want} in\n{src}");
    }
    // The q8_0 block is the format and does not move with the split.
    assert!(src.contains("constexpr uint QK8 = 32u;"), "{src}");
    let err = |v: &[u32]| try_emit(&with_split(v)).unwrap_err();
    assert!(err(&[64, 4, 8]).contains("nsg 64"));
    assert!(err(&[4, 32, 8]).contains("nr0 32"));
    assert!(err(&[4, 4, 12]).contains("nq 12"));
}

#[cfg(target_os = "macos")]
#[test]
fn the_emitted_kernel_matches_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let queue = dev.new_command_queue();

    // Enough blocks per row that the lane stride is actually taken.
    const IN_DIM: usize = 2048;
    const OUT_DIM: usize = 20; // not a multiple of every row count
    const TOKENS: usize = 2;
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let nb = IN_DIM / BLOCK;
    let mut w_bytes = vec![0u8; OUT_DIM * nb * BLOCK_BYTES];
    let mut w_deq = vec![0f32; OUT_DIM * IN_DIM];
    for r in 0..OUT_DIM {
        for b in 0..nb {
            let scale = 1.0f32 / (1 << (5 + (next() % 3))) as f32;
            let bits = scale.to_bits();
            let half = ((bits >> 16) & 0x8000) as u16
                | ((((bits >> 23) & 0xff) as i32 - 127 + 15) as u16) << 10
                | ((bits & 0x7f_ffff) >> 13) as u16;
            let off = (r * nb + b) * BLOCK_BYTES;
            w_bytes[off..off + 2].copy_from_slice(&half.to_le_bytes());
            for i in 0..BLOCK {
                let q = (next() % 255) as i64 - 127;
                w_bytes[off + 2 + i] = (q as i8) as u8;
                w_deq[r * IN_DIM + b * BLOCK + i] = q as f32 * scale;
            }
        }
    }
    let x: Vec<f32> = (0..TOKENS * IN_DIM).map(|_| ((next() % 512) as f32 - 256.0) / 256.0).collect();

    let o = MTLResourceOptions::StorageModeShared;
    let b_w = dev.new_buffer_with_data(w_bytes.as_ptr() as *const _, w_bytes.len() as u64, o);
    let b_x = dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as u64, o);

    for split @ [nsg, nr0, nq] in [SHIPPED, [1, 1, 32], [2, 3, 4], [8, 5, 16], [4, 8, 1]] {
        let src = try_emit(&with_split(&[nsg, nr0, nq])).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("laguna_q8_0_matvec_f32", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
        let b_out = dev.new_buffer((TOKENS * OUT_DIM * 4) as u64, o);

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [&b_w, &b_x, &b_out].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, v) in [IN_DIM as u32, OUT_DIM as u32, TOKENS as u32].iter().enumerate() {
            enc.set_bytes(3 + i as u64, 4, v as *const u32 as *const _);
        }
        let row_bytes = (nb * BLOCK_BYTES) as u64;
        enc.set_bytes(6, 8, &row_bytes as *const u64 as *const _);
        // One threadgroup covers NSG x NR0 rows; the y axis is the token.
        enc.dispatch_thread_groups(
            MTLSize::new((OUT_DIM as u64).div_ceil((nsg * nr0) as u64), TOKENS as u64, 1),
            MTLSize::new((nsg * 32) as u64, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let got = unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, TOKENS * OUT_DIM) };
        for t in 0..TOKENS {
            for row in 0..OUT_DIM {
                let want: f32 =
                    (0..IN_DIM).map(|k| w_deq[row * IN_DIM + k] * x[t * IN_DIM + k]).sum();
                let g = got[t * OUT_DIM + row];
                assert!(
                    (g - want).abs() <= 2e-3 * want.abs().max(1e-2),
                    "at split {split:?}, token {t} row {row}: got {g}, want {want}"
                );
            }
        }
    }
}
