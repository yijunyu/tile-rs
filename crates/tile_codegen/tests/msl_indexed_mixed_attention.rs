//! The DeepSeek-V4 indexed mixed-attention kernels, emitted from their intrinsics.
//!
//! Both kernels ship in both staging precisions, as four committed files: the
//! one-row kernel as `_h8` (float) and `_h8_half`, the batched kernel as
//! `_h8_rb4` (half) and `_h8_rb4_float`. Each must be reproduced byte for byte
//! from its MLIR. At other shapes, on a Metal GPU, both kernels must match a CPU
//! reference of the same selection rules and softmax in both precisions.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const ROW_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexed_mixed_attention_h8(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %nt   = llvm.mlir.constant(2 : i32) : i32
    %nh   = llvm.mlir.constant(8 : i32) : i32
    %nr   = llvm.mlir.constant(4 : i32) : i32
    %nc   = llvm.mlir.constant(8 : i32) : i32
    %tk   = llvm.mlir.constant(4 : i32) : i32
    %ra   = llvm.mlir.constant(4 : i32) : i32
    %wn   = llvm.mlir.constant(0 : i32) : i32
    %p0   = llvm.mlir.constant(8 : i32) : i32
    %rs   = llvm.mlir.constant(0 : i32) : i32
    %rc   = llvm.mlir.constant(16 : i32) : i32
    %qts  = llvm.mlir.constant(16384 : i32) : i32
    %qhs  = llvm.mlir.constant(2048 : i32) : i32
    %rrs  = llvm.mlir.constant(2048 : i32) : i32
    %crs  = llvm.mlir.constant(2048 : i32) : i32
    %tts  = llvm.mlir.constant(16 : i32) : i32
    %dts  = llvm.mlir.constant(16384 : i32) : i32
    %dhs  = llvm.mlir.constant(2048 : i32) : i32
    %scl  = llvm.mlir.constant(1 : i32) : i32
    %ret  = llvm.call @__tile_indexed_mixed_attention_h8_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %nt, %nh, %nr, %nc, %tk, %ra, %wn, %p0, %rs, %rc, %qts, %qhs, %rrs, %crs, %tts, %dts, %dhs, %scl) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const BATCHED_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexed_mixed_attention_h8_rb4(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %nt   = llvm.mlir.constant(2 : i32) : i32
    %nh   = llvm.mlir.constant(8 : i32) : i32
    %nr   = llvm.mlir.constant(4 : i32) : i32
    %nc   = llvm.mlir.constant(8 : i32) : i32
    %tk   = llvm.mlir.constant(4 : i32) : i32
    %ra   = llvm.mlir.constant(4 : i32) : i32
    %wn   = llvm.mlir.constant(0 : i32) : i32
    %p0   = llvm.mlir.constant(8 : i32) : i32
    %rs   = llvm.mlir.constant(0 : i32) : i32
    %rc   = llvm.mlir.constant(16 : i32) : i32
    %qts  = llvm.mlir.constant(16384 : i32) : i32
    %qhs  = llvm.mlir.constant(2048 : i32) : i32
    %rrs  = llvm.mlir.constant(2048 : i32) : i32
    %crs  = llvm.mlir.constant(2048 : i32) : i32
    %tts  = llvm.mlir.constant(16 : i32) : i32
    %dts  = llvm.mlir.constant(16384 : i32) : i32
    %dhs  = llvm.mlir.constant(2048 : i32) : i32
    %scl  = llvm.mlir.constant(1 : i32) : i32
    %ret  = llvm.call @__tile_indexed_mixed_attention_h8_rb4_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %nt, %nh, %nr, %nc, %tk, %ra, %wn, %p0, %rs, %rc, %qts, %qhs, %rrs, %crs, %tts, %dts, %dhs, %scl) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const ROW_HALF_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexed_mixed_attention_h8_half(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %nt   = llvm.mlir.constant(2 : i32) : i32
    %nh   = llvm.mlir.constant(8 : i32) : i32
    %nr   = llvm.mlir.constant(4 : i32) : i32
    %nc   = llvm.mlir.constant(8 : i32) : i32
    %tk   = llvm.mlir.constant(4 : i32) : i32
    %ra   = llvm.mlir.constant(4 : i32) : i32
    %wn   = llvm.mlir.constant(0 : i32) : i32
    %p0   = llvm.mlir.constant(8 : i32) : i32
    %rs   = llvm.mlir.constant(0 : i32) : i32
    %rc   = llvm.mlir.constant(16 : i32) : i32
    %qts  = llvm.mlir.constant(16384 : i32) : i32
    %qhs  = llvm.mlir.constant(2048 : i32) : i32
    %rrs  = llvm.mlir.constant(2048 : i32) : i32
    %crs  = llvm.mlir.constant(2048 : i32) : i32
    %tts  = llvm.mlir.constant(16 : i32) : i32
    %dts  = llvm.mlir.constant(16384 : i32) : i32
    %dhs  = llvm.mlir.constant(2048 : i32) : i32
    %scl  = llvm.mlir.constant(1 : i32) : i32
    %hd   = llvm.mlir.constant(512 : i32) : i32
    %hg   = llvm.mlir.constant(8 : i32) : i32
    %sb   = llvm.mlir.constant(16 : i32) : i32
    %ret  = llvm.call @__tile_indexed_mixed_attention_h8_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %nt, %nh, %nr, %nc, %tk, %ra, %wn, %p0, %rs, %rc, %qts, %qhs, %rrs, %crs, %tts, %dts, %dhs, %scl, %hd, %hg, %sb) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const BATCHED_FLOAT_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_indexed_mixed_attention_h8_rb4_float(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %nt   = llvm.mlir.constant(2 : i32) : i32
    %nh   = llvm.mlir.constant(8 : i32) : i32
    %nr   = llvm.mlir.constant(4 : i32) : i32
    %nc   = llvm.mlir.constant(8 : i32) : i32
    %tk   = llvm.mlir.constant(4 : i32) : i32
    %ra   = llvm.mlir.constant(4 : i32) : i32
    %wn   = llvm.mlir.constant(0 : i32) : i32
    %p0   = llvm.mlir.constant(8 : i32) : i32
    %rs   = llvm.mlir.constant(0 : i32) : i32
    %rc   = llvm.mlir.constant(16 : i32) : i32
    %qts  = llvm.mlir.constant(16384 : i32) : i32
    %qhs  = llvm.mlir.constant(2048 : i32) : i32
    %rrs  = llvm.mlir.constant(2048 : i32) : i32
    %crs  = llvm.mlir.constant(2048 : i32) : i32
    %tts  = llvm.mlir.constant(16 : i32) : i32
    %dts  = llvm.mlir.constant(16384 : i32) : i32
    %dhs  = llvm.mlir.constant(2048 : i32) : i32
    %scl  = llvm.mlir.constant(1 : i32) : i32
    %hd   = llvm.mlir.constant(512 : i32) : i32
    %hg   = llvm.mlir.constant(8 : i32) : i32
    %sb   = llvm.mlir.constant(32 : i32) : i32
    %rb   = llvm.mlir.constant(4 : i32) : i32
    %ret  = llvm.call @__tile_indexed_mixed_attention_h8_rb4_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %nt, %nh, %nr, %nc, %tk, %ra, %wn, %p0, %rs, %rc, %qts, %qhs, %rrs, %crs, %tts, %dts, %dhs, %scl, %hd, %hg, %sb, %rb) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
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

/// A call with the shape operands; `rows` selects the batched intrinsic.
fn shaped(head_dim: u32, heads: u32, stage_bits: u32, rows: Option<u32>) -> String {
    let callee = if rows.is_some() { "__tile_indexed_mixed_attention_h8_rb4_f32" } else { "__tile_indexed_mixed_attention_h8_f32" };
    let extra = match rows {
        Some(_) => "%hd, %hg, %sb, %rb".to_string(),
        None => "%hd, %hg, %sb".to_string(),
    };
    let n = if rows.is_some() { 28 } else { 27 };
    let types = ["!llvm.ptr<1>"; 6].into_iter().chain(std::iter::repeat("i32").take(n - 6)).collect::<Vec<_>>().join(", ");
    let consts = "%c1, ".repeat(18);
    format!(
        r#"
module {{
  llvm.func @attn_gate(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: i32) attributes {{hacc.entry}} {{
    ^bb0:
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %hd = llvm.mlir.constant({head_dim} : i32) : i32
    %hg = llvm.mlir.constant({heads} : i32) : i32
    %sb = llvm.mlir.constant({stage_bits} : i32) : i32
    %rb = llvm.mlir.constant({rb} : i32) : i32
    %r = llvm.call @{callee}(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, {consts}{extra}) : ({types}) -> i32
    llvm.return
  }}
}}
"#,
        rb = rows.unwrap_or(1)
    )
}

/// (committed file stem, MLIR it is emitted from, staging it must use)
const SHIPPED: [(&str, &str, &str); 4] = [
    ("dsv4_indexed_mixed_attention_h8", ROW_MLIR_FLOAT_SPELLING, "float4 q0 = q4[simd_lane +  0];"),
    ("dsv4_indexed_mixed_attention_h8_half", ROW_HALF_MLIR, "half4 q0 = (half4)q4[simd_lane +  0];"),
    ("dsv4_indexed_mixed_attention_h8_rb4", BATCHED_MLIR, "half4 q0 = (half4)q4[simd_lane +  0];"),
    ("dsv4_indexed_mixed_attention_h8_rb4_float", BATCHED_FLOAT_MLIR, "float4 q0 = q4[simd_lane +  0];"),
];

/// The committed float one-row kernel predates the stage operand, so its MLIR
/// is the original call with the float spelling appended.
const ROW_MLIR_FLOAT_SPELLING: &str = "row-float";

fn mlir_for(entry: &str) -> String {
    if entry == ROW_MLIR_FLOAT_SPELLING {
        ROW_HALF_MLIR
            .replace("ds4_dsv4_indexed_mixed_attention_h8_half(", "ds4_dsv4_indexed_mixed_attention_h8(")
            .replace("llvm.mlir.constant(16 : i32) : i32\n    %ret", "llvm.mlir.constant(32 : i32) : i32\n    %ret")
    } else {
        entry.to_string()
    }
}

#[test]
fn every_shipped_precision_is_reproduced() {
    for (stem, entry, staging) in SHIPPED {
        let emitted = try_emit(&mlir_for(entry)).unwrap();
        assert!(emitted.contains(staging), "{stem} must stage with `{staging}`");
        assert!(emitted.contains(&format!("kernel void ds4_{stem}(")), "{stem}: kernel name");
        assert_eq!(emitted, golden(stem), "{stem} differs from its committed file");
    }
}

#[test]
fn default_operands_keep_the_emitter_defaults() {
    // Without shape operands the one-row kernel stages half and the batched
    // kernel half, as the emitter did before the operands existed.
    let row = try_emit(ROW_MLIR).unwrap();
    assert_eq!(row.replace("ds4_dsv4_indexed_mixed_attention_h8(", "ds4_dsv4_indexed_mixed_attention_h8_half("), golden("dsv4_indexed_mixed_attention_h8_half"));
    assert_eq!(try_emit(BATCHED_MLIR).unwrap(), golden("dsv4_indexed_mixed_attention_h8_rb4"));
    assert_eq!(
        try_emit(&shaped(512, 8, 32, None)).unwrap(),
        golden("dsv4_indexed_mixed_attention_h8").replace("ds4_dsv4_indexed_mixed_attention_h8(", "attn_gate(")
    );
}

#[test]
fn shape_reaches_the_kernels_and_bad_shapes_are_refused() {
    let src = try_emit(&shaped(256, 4, 32, Some(3))).unwrap();
    for needle in [
        "threadgroup float4 kv_shared[3*64];",
        "tgpig.y * 4u + simd_id",
        "off < n_rows * 64u; off += 128u",
        "uint r = off >> 6;",
        "uint c = off & 63u;",
        "float4 q1 = q4[simd_lane + 32];",
        "float score = dot((float4)q0,(float4)k0) + dot((float4)q1,(float4)k1);",
        "uint rows[3];",
    ] {
        assert!(src.contains(needle), "missing {needle:?}:\n{src}");
    }
    assert!(!src.contains("q2") && !src.contains("8u + simd_id"), "{src}");
    let wide = try_emit(&shaped(1024, 8, 16, None)).unwrap();
    assert!(wide.contains("dst4[simd_lane + 224] = o7 * inv_s;"), "{wide}");
    for (mlir, why) in [
        (shaped(300, 8, 16, None), "head_dim 300 must be 128, 256, 512 or 1024"),
        (shaped(512, 2, 16, None), "2 heads per group give 64 threads, fewer than the 128 float4 rows"),
        (shaped(512, 33, 16, Some(4)), "heads_per_group 33"),
        (shaped(512, 8, 8, None), "stage_bits 8 must be 16 (half) or 32 (float)"),
        (shaped(512, 8, 16, Some(17)), "rows_per_batch 17"),
    ] {
        let e = try_emit(&mlir).unwrap_err();
        assert!(e.contains(why), "{why}: {e}");
    }
    // The batched kernel stages in chunks, so few threads are fine there.
    assert!(try_emit(&shaped(512, 2, 16, Some(4))).is_ok());
    let runtime = shaped(512, 8, 16, None).replace("%c1, %hd, %hg, %sb)", "%c1, %arg6, %hg, %sb)");
    assert!(runtime.contains("%arg6, %hg, %sb)"));
    assert!(try_emit(&runtime).unwrap_err().contains("operand 24 (head_dim) must be a positive integer constant"));
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    const N_TOKENS: usize = 3;
    const POS0: usize = 20;
    const N_RAW: usize = 9;
    const RAW_CAP: usize = 12;
    const RAW_START: usize = 5;
    const WINDOW: usize = 6;
    const RATIO: usize = 4;
    const N_COMP: usize = 6;
    const TOP_K: usize = 7;

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// f32 -> binary16 -> f32, rounding to nearest even.
    fn half(x: f32) -> f32 {
        if x == 0.0 || !x.is_finite() {
            return x;
        }
        let a = x.abs();
        let e = (a.log2().floor() as i32).max(-14);
        let step = 2f32.powi(e - 10);
        let n = a / step;
        let mut r = n.round();
        if (n - n.floor() - 0.5).abs() < f32::EPSILON && r % 2.0 != 0.0 {
            r -= 1.0;
        }
        (r * step).copysign(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference(hd: usize, n_head: usize, stage_half: bool, q: &[f32], raw: &[f32], comp: &[f32], topk: &[i32], sinks: &[f32], scale: f32) -> Vec<f32> {
        let stage = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| if stage_half { half(x) } else { x } as f64).collect() };
        let mut out = vec![0f32; N_TOKENS * n_head * hd];
        for t in 0..N_TOKENS {
            let qpos = POS0 + t;
            let last_pos = POS0 + N_TOKENS - 1;
            let first_raw_pos = last_pos + 1 - N_RAW;
            let raw_last_pos = first_raw_pos + N_RAW - 1;
            let window_first = if qpos + 1 > WINDOW { qpos + 1 - WINDOW } else { 0 };
            let (first, last) = (first_raw_pos.max(window_first), qpos.min(raw_last_pos));
            let mut rows: Vec<&[f32]> = Vec::new();
            if first <= last {
                for pos in first..=last {
                    let row = (RAW_START + pos - first_raw_pos) % RAW_CAP;
                    rows.push(&raw[row * hd..(row + 1) * hd]);
                }
            }
            let visible = ((qpos + 1) / RATIO).min(N_COMP);
            for &idx in &topk[t * TOP_K..(t + 1) * TOP_K] {
                if idx < 0 {
                    continue;
                }
                if idx as usize >= visible {
                    break;
                }
                rows.push(&comp[idx as usize * hd..(idx as usize + 1) * hd]);
            }
            for h in 0..n_head {
                let qh = stage(&q[(t * n_head + h) * hd..(t * n_head + h + 1) * hd]);
                let keys: Vec<Vec<f64>> = rows.iter().map(|r| stage(r)).collect();
                let scores: Vec<f64> =
                    keys.iter().map(|k| k.iter().zip(&qh).map(|(a, b)| a * b).sum::<f64>() * scale as f64).collect();
                let sink = sinks[h] as f64;
                let m = scores.iter().copied().fold(sink, f64::max);
                let denom: f64 = scores.iter().map(|s| (s - m).exp()).sum::<f64>() + (sink - m).exp();
                for d in 0..hd {
                    let num: f64 = keys.iter().zip(&scores).map(|(k, s)| (s - m).exp() * k[d]).sum();
                    out[(t * n_head + h) * hd + d] = (num / denom) as f32;
                }
            }
        }
        out
    }

    fn run(dev: &Device, hd: usize, heads: usize, stage_bits: u32, rows: Option<u32>) {
        // Two full threadgroups of heads. The kernels require n_head to be a
        // multiple of heads_per_group: threads for heads past n_head return
        // before staging their share of the key row, and the heads that share
        // their threadgroup then read unstaged memory.
        let n_head = 2 * heads;
        let src = try_emit(&shaped(hd as u32, heads as u32, stage_bits, rows)).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("attn_gate", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
        assert_eq!(pipe.thread_execution_width(), 32, "kernel assumes a 32-lane simdgroup");
        // Metal lowers a pipeline's thread ceiling as its threadgroup memory
        // grows (704 threads with 16 KB staged on an M1 Ultra). Past it the
        // dispatch is rejected and the output buffer is left untouched.
        let threads = 32 * heads as u64;
        assert!(
            pipe.max_total_threads_per_threadgroup() >= threads,
            "head_dim {hd} rows {rows:?}: {threads} threads exceed this pipeline's limit of {}",
            pipe.max_total_threads_per_threadgroup()
        );

        let q = values(N_TOKENS * n_head * hd, 1 + hd as u64);
        let raw = values(RAW_CAP * hd, 2 + hd as u64);
        let comp = values(N_COMP * hd, 3 + hd as u64);
        let sinks: Vec<f32> = values(n_head, 4).iter().map(|v| v * 2.0).collect();
        // Per token: a skipped slot, an out-of-window id that stops the scan, and ids after it.
        let topk: Vec<i32> = (0..N_TOKENS).flat_map(|t| [3 - t as i32, -1, 0, 4, 5, 1, 2]).collect();
        let scale = 1.0 / (hd as f32).sqrt();

        let o = MTLResourceOptions::StorageModeShared;
        let fbuf = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
        let bufs = [
            fbuf(&q),
            fbuf(&raw),
            fbuf(&comp),
            dev.new_buffer_with_data(topk.as_ptr() as *const _, (topk.len() * 4) as u64, o),
            fbuf(&sinks),
            dev.new_buffer((N_TOKENS * n_head * hd * 4) as u64, o),
        ];
        let hb = (hd * 4) as u32;
        let scalars: [u32; 17] = [
            N_TOKENS as u32, n_head as u32, N_RAW as u32, N_COMP as u32, TOP_K as u32, RATIO as u32, WINDOW as u32,
            POS0 as u32, RAW_START as u32, RAW_CAP as u32,
            n_head as u32 * hb, hb, hb, hb, (TOP_K * 4) as u32, n_head as u32 * hb, hb,
        ];
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, v) in scalars.iter().enumerate() {
            enc.set_bytes(6 + i as u64, 4, v as *const u32 as *const _);
        }
        enc.set_bytes(23, 4, &scale as *const f32 as *const _);
        let groups = MTLSize::new(N_TOKENS as u64, n_head.div_ceil(heads) as u64, 1);
        enc.dispatch_thread_groups(groups, MTLSize::new(32, heads as u64, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let got = unsafe { std::slice::from_raw_parts(bufs[5].contents() as *const f32, N_TOKENS * n_head * hd) };
        let want = reference(hd, n_head, stage_bits == 16, &q, &raw, &comp, &topk, &sinks, scale);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1e-2),
                "head_dim {hd} heads/group {heads} stage {stage_bits} rows {rows:?}: token {} head {} d {}: got {g}, want {w}",
                i / (n_head * hd),
                (i / hd) % n_head,
                i % hd
            );
        }
    }

    #[test]
    fn emitted_kernels_match_the_cpu_reference_at_several_shapes() {
        let Some(dev) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        for (hd, heads, stage) in [(512, 8, 16), (512, 8, 32), (128, 4, 32), (256, 8, 16), (1024, 8, 32), (128, 1, 16), (128, 32, 32)] {
            run(&dev, hd, heads, stage, None);
        }
        for (hd, heads, stage, rows) in [(512, 8, 16, 4), (512, 8, 32, 4), (128, 2, 32, 1), (256, 4, 16, 3), (512, 16, 32, 2), (1024, 8, 32, 8)] {
            run(&dev, hd, heads, stage, Some(rows));
        }
    }
}
