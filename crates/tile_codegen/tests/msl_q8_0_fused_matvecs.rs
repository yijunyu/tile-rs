//! The two fused q8_0 matvecs: the shared gate/up SwiGLU and the low-rank
//! attention output projection.
//!
//! Both are the q8_0 matvec with more streams bolted on, and both carried the
//! same three numbers in their text: the simdgroups a threadgroup runs, the
//! output rows each owns, and the slice of a quantization block a lane takes.
//! The row count was again also a literal accumulator list with room for two.
//! All three are now trailing operands. At the shipped split the emitted text
//! is unchanged; at other splits, on a Metal GPU, both must match a CPU
//! reference of the same dequantized arithmetic.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const SWIGLU: &str = r#"
module {
  llvm.func @ds4_kernel_dsv4_shared_gate_up_swiglu_q8_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(64  : i32) : i32
    %ne01 = llvm.mlir.constant(4   : i32) : i32
    %ne0  = llvm.mlir.constant(4   : i32) : i32
    %ne1  = llvm.mlir.constant(1   : i32) : i32
    %ne12 = llvm.mlir.constant(1   : i32) : i32
    %r2   = llvm.mlir.constant(1   : i32) : i32
    %r3   = llvm.mlir.constant(1   : i32) : i32
    %nb01 = llvm.mlir.constant(68  : i32) : i32
    %nb02 = llvm.mlir.constant(272 : i32) : i32
    %nb03 = llvm.mlir.constant(272 : i32) : i32
    %nb11 = llvm.mlir.constant(256 : i32) : i32
    %nb12 = llvm.mlir.constant(256 : i32) : i32
    %nb13 = llvm.mlir.constant(256 : i32) : i32
    %r = llvm.call @__tile_dsv4_shared_gate_up_swiglu_q8_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %ne00, %ne01, %ne0, %ne1, %ne12, %r2, %r3, %nb01, %nb02, %nb03, %nb11, %nb12, %nb13) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const ATTN_LOW: &str = r#"
module {
  llvm.func @ds4_kernel_dsv4_attn_out_low_q8_0_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(64 : i32) : i32
    %ne01 = llvm.mlir.constant(64 : i32) : i32
    %ne0  = llvm.mlir.constant(64 : i32) : i32
    %ne1  = llvm.mlir.constant(4  : i32) : i32
    %ne11 = llvm.mlir.constant(1  : i32) : i32
    %nei0 = llvm.mlir.constant(4  : i32) : i32
    %nb01 = llvm.mlir.constant(68   : i32) : i32
    %nb02 = llvm.mlir.constant(4352 : i32) : i32
    %nb11 = llvm.mlir.constant(256  : i32) : i32
    %nb12 = llvm.mlir.constant(1024 : i32) : i32
    %r = llvm.call @__tile_dsv4_attn_out_low_q8_0_f32(%arg0, %arg1, %arg2, %ne00, %ne01, %ne0, %ne1, %ne11, %nei0, %nb01, %nb02, %nb11, %nb12) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const SWIGLU_CALLEE: &str = "__tile_dsv4_shared_gate_up_swiglu_q8_0";
const ATTN_LOW_CALLEE: &str = "__tile_dsv4_attn_out_low_q8_0_f32";

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
fn the_shipped_splits_are_what_the_kernels_had() {
    let s = try_emit(SWIGLU).unwrap();
    assert_eq!(s, try_emit(&with_split(SWIGLU, SWIGLU_CALLEE, &[2, 2, 8])).unwrap());
    assert!(s.contains("float sumg[NR0] = { 0.0f, 0.0f };") && s.contains("float sumu[NR0] = { 0.0f, 0.0f };"), "{s}");
    assert!(s.contains("constexpr short NSG   = 2;"), "{s}");

    let a = try_emit(ATTN_LOW).unwrap();
    assert_eq!(a, try_emit(&with_split(ATTN_LOW, ATTN_LOW_CALLEE, &[4, 2, 8])).unwrap());
    assert!(a.contains("float sumf[NR0] = { 0.0f, 0.0f };") && a.contains("constexpr short NSG   = 4;"), "{a}");
}

#[test]
fn the_split_reaches_the_kernels_and_bad_ones_are_refused() {
    let s = try_emit(&with_split(SWIGLU, SWIGLU_CALLEE, &[4, 3, 16])).unwrap();
    for want in [
        "constexpr short NSG   = 4;",
        "constexpr short NR0   = 3;",
        "constexpr short NQ    = 16;",
        // Both accumulator lists follow the row count.
        "float sumg[NR0] = { 0.0f, 0.0f, 0.0f };",
        "float sumu[NR0] = { 0.0f, 0.0f, 0.0f };",
    ] {
        assert!(s.contains(want), "missing {want} in\n{s}");
    }
    let a = try_emit(&with_split(ATTN_LOW, ATTN_LOW_CALLEE, &[1, 4, 2])).unwrap();
    for want in [
        "constexpr short NSG   = 1;",
        "constexpr short NR0   = 4;",
        "constexpr short NQ    = 2;",
        "float sumf[NR0] = { 0.0f, 0.0f, 0.0f, 0.0f };",
    ] {
        assert!(a.contains(want), "missing {want} in\n{a}");
    }
    let err = |m: &str, c: &str, v: &[u32]| try_emit(&with_split(m, c, v)).unwrap_err();
    assert!(err(SWIGLU, SWIGLU_CALLEE, &[64, 2, 8]).contains("nsg 64"));
    assert!(err(SWIGLU, SWIGLU_CALLEE, &[2, 2, 12]).contains("nq 12"));
    assert!(err(ATTN_LOW, ATTN_LOW_CALLEE, &[4, 32, 8]).contains("nr0 32"));
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
    const ROWS: usize = 12; // not a multiple of every row count
    const GROUPS: usize = 4; // routed groups for the attention projection
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let mut state = 0x0BAD_C0FFEE_1234u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut quantize = |rows: usize| -> (Vec<u8>, Vec<f32>) {
        let nb = NE00 / BLOCK;
        let mut bytes = vec![0u8; rows * nb * BLOCK_BYTES];
        let mut deq = vec![0f32; rows * NE00];
        for r in 0..rows {
            for b in 0..nb {
                let scale = 1.0f32 / (1 << (4 + (next() % 3))) as f32;
                let bits = scale.to_bits();
                let half = ((bits >> 16) & 0x8000) as u16
                    | ((((bits >> 23) & 0xff) as i32 - 127 + 15) as u16) << 10
                    | ((bits & 0x7f_ffff) >> 13) as u16;
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

    let (gate_b, gate_d) = quantize(ROWS);
    let (up_b, up_d) = quantize(ROWS);
    let (proj_b, proj_d) = quantize(ROWS * GROUPS);
    let x: Vec<f32> = (0..NE00).map(|_| ((next() % 512) as f32 - 256.0) / 256.0).collect();

    let o = MTLResourceOptions::StorageModeShared;
    let mk = |b: &[u8]| dev.new_buffer_with_data(b.as_ptr() as *const _, b.len() as u64, o);
    let (bg, bu, bp) = (mk(&gate_b), mk(&up_b), mk(&proj_b));
    let bx = dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as u64, o);
    let nb01 = (NE00 / BLOCK * BLOCK_BYTES) as u32;
    let nb02 = nb01 * ROWS as u32;
    let nb11 = (NE00 * 4) as u32;
    let dot = |w: &[f32], row: usize| -> f32 { (0..NE00).map(|k| w[row * NE00 + k] * x[k]).sum() };

    for split @ [nsg, nr0, nq] in [[2u32, 2u32, 8u32], [1, 1, 32], [4, 3, 4], [2, 4, 16], [8, 2, 1]] {
        let build = |mlir: &str, callee: &str, entry: &str| {
            let src = try_emit(&with_split(mlir, callee, &[nsg, nr0, nq])).unwrap();
            let lib =
                dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
            dev.new_compute_pipeline_state(&desc).unwrap()
        };
        let groups = (ROWS as u64).div_ceil(nr0 as u64);

        // Gate and up projections plus their SwiGLU product, three outputs.
        {
            let pipe = build(SWIGLU, SWIGLU_CALLEE, "ds4_kernel_dsv4_shared_gate_up_swiglu_q8_0");
            let outs: Vec<_> = (0..3).map(|_| dev.new_buffer((ROWS * 4) as u64, o)).collect();
            let u: [u32; 13] =
                [NE00 as u32, ROWS as u32, ROWS as u32, 1, 1, 1, 1, nb01, nb02, nb02, nb11, nb11, nb11];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bg, &bu, &bx, &outs[0], &outs[1], &outs[2]].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in u.iter().enumerate() {
                enc.set_bytes(6 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new((nsg * 32) as u64, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let read = |b: &metal::Buffer| unsafe {
                std::slice::from_raw_parts(b.contents() as *const f32, ROWS).to_vec()
            };
            let (g, u_, m) = (read(&outs[0]), read(&outs[1]), read(&outs[2]));
            for row in 0..ROWS {
                let wg = dot(&gate_d, row);
                let wu = dot(&up_d, row);
                let wm = wg / (1.0 + (-wg).exp()) * wu;
                for (got, want, what) in [(g[row], wg, "gate"), (u_[row], wu, "up"), (m[row], wm, "swiglu")] {
                    assert!(
                        (got - want).abs() <= 2e-3 * want.abs().max(1e-2),
                        "{what} at split {split:?}, row {row}: got {got}, want {want}"
                    );
                }
            }
        }

        // Low-rank attention output: group idx reads weight block idx.
        {
            let pipe = build(ATTN_LOW, ATTN_LOW_CALLEE, "ds4_kernel_dsv4_attn_out_low_q8_0_f32");
            let bo = dev.new_buffer((ROWS * GROUPS * 4) as u64, o);
            let u: [u32; 10] =
                [NE00 as u32, ROWS as u32, ROWS as u32, GROUPS as u32, 1, GROUPS as u32, nb01, nb02, nb11, nb11];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bp, &bx, &bo].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in u.iter().enumerate() {
                enc.set_bytes(3 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(groups, 1, GROUPS as u64),
                MTLSize::new((nsg * 32) as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, ROWS * GROUPS) };
            for idx in 0..GROUPS {
                for row in 0..ROWS {
                    let want = dot(&proj_d, idx * ROWS + row);
                    let g = got[idx * ROWS + row];
                    assert!(
                        (g - want).abs() <= 2e-3 * want.abs().max(1e-2),
                        "attention projection at split {split:?}, group {idx} row {row}: got {g}, want {want}"
                    );
                }
            }
        }
    }
}
