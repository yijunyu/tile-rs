//! Laguna's GQA decode attention, emitted from its intrinsic.
//!
//! The kernel splits the key range over simdgroups and merges their separately
//! normalised partials, so the split count sets the thread count and the size of
//! the partial staging. At the shipped 8 the emitted kernel must match the
//! committed laguna.metal byte for byte; at other splits, on a Metal GPU, the
//! output must match one CPU reference, since the merge is mathematically
//! invariant to how the keys are divided.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const MLIR: &str = r#"
module {
  llvm.func @laguna_attention_decode_gqa_f16(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @__tile_laguna_attention_decode_gqa_f16(%a, %r, %c) : (i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg4, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

/// The kernel body only, so the file prelude does not enter the comparison.
fn kernel_body(src: &str) -> String {
    let i = src.find("kernel void laguna_attention_decode_gqa_f16(").expect("kernel");
    let rest = &src[i..];
    let end = rest[10..].find("\nkernel void").map(|j| j + 10).unwrap_or(rest.len());
    rest[..end].trim_end().to_string()
}

fn with_split(split: &str) -> String {
    // The fixture also calls load/store intrinsics; splice into the attention call.
    let call = MLIR.find("llvm.call @__tile_laguna_attention_decode_gqa_f16").unwrap();
    let open = MLIR[call..].find('(').unwrap() + call;
    let close = MLIR[open..].find(')').unwrap() + open;
    let types_open = MLIR[close..].find('(').unwrap() + close;
    let types_close = MLIR[types_open..].find(')').unwrap() + types_open;
    let spliced = format!("{}, %sp{}, i32{}", &MLIR[..close], &MLIR[close..types_close], &MLIR[types_close..]);
    let line = spliced[..spliced.find("llvm.call @__tile_laguna_attention_decode_gqa_f16").unwrap()]
        .rfind('\n')
        .unwrap()
        + 1;
    format!("{}    %sp = llvm.mlir.constant({split} : i32) : i32\n{}", &spliced[..line], &spliced[line..])
}

fn shipped() -> String {
    let p = format!("{}/../../benchmarks/ds4_msl/emitted/laguna.metal", env!("CARGO_MANIFEST_DIR"));
    kernel_body(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}")))
}

#[test]
fn shipped_split_matches_the_committed_kernel() {
    assert_eq!(kernel_body(&try_emit(MLIR).unwrap()), shipped());
    assert_eq!(kernel_body(&try_emit(&with_split("8")).unwrap()), shipped());
}

#[test]
fn split_reaches_the_kernel_and_bad_splits_are_refused() {
    let src = try_emit(&with_split("4")).unwrap();
    assert!(src.contains("constexpr uint split_simd_groups = 4u;"), "{src}");
    assert!(src.contains("threadgroup float scratch[4u + 4u + 4u * 128u];"), "{src}");
    assert!(!src.contains("8u + 8u"), "{src}");
    assert!(try_emit(&with_split("33")).unwrap_err().contains("split_simd_groups 33 must be from 1 to 32"));
    let runtime = with_split("8").replace(", %sp)", ", %a)");
    assert!(runtime.contains(", %a)"));
    assert!(try_emit(&runtime).unwrap_err().contains("operand 3 (split_simd_groups) must be a positive integer constant"));
}

#[cfg(target_os = "macos")]
#[test]
fn every_split_matches_one_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};
    const HEAD_DIM: usize = 128;
    const N_HEAD: usize = 4;
    const N_HEAD_KV: usize = 2;
    const CACHE_CAP: usize = 16;
    const KEY_START: usize = 5;
    const KEY_COUNT: usize = 11;
    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut seed = 0xDEAD_BEEF_1234_5678u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    /// f32 -> binary16 -> f32, as the half key and value caches store them.
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

    let q: Vec<f32> = (0..N_HEAD * HEAD_DIM).map(|_| next()).collect();
    let gate: Vec<f32> = (0..N_HEAD).map(|_| next() * 3.0).collect();
    let cache_width = N_HEAD_KV * HEAD_DIM;
    let keys: Vec<f32> = (0..CACHE_CAP * cache_width).map(|_| half(next())).collect();
    let values: Vec<f32> = (0..CACHE_CAP * cache_width).map(|_| half(next())).collect();
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    // One reference for every split: softmax over the key window, then a softplus gate.
    let mut want = vec![0f32; N_HEAD * HEAD_DIM];
    for head in 0..N_HEAD {
        let kv_head = head / (N_HEAD / N_HEAD_KV);
        let scores: Vec<f64> = (0..KEY_COUNT)
            .map(|i| {
                let row = (KEY_START + i) % CACHE_CAP;
                let base = row * cache_width + kv_head * HEAD_DIM;
                let dot: f64 = (0..HEAD_DIM).map(|d| (q[head * HEAD_DIM + d] * keys[base + d]) as f64).sum();
                dot * scale as f64
            })
            .collect();
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        let g = gate[head];
        let gate_scale = if g > 20.0 { g } else { (1.0 + g.exp()).ln() };
        for d in 0..HEAD_DIM {
            let num: f64 = (0..KEY_COUNT)
                .map(|i| {
                    let row = (KEY_START + i) % CACHE_CAP;
                    let base = row * cache_width + kv_head * HEAD_DIM;
                    (scores[i] - m).exp() * values[base + d] as f64
                })
                .sum();
            want[head * HEAD_DIM + d] = (num / denom) as f32 * gate_scale;
        }
    }

    let o = MTLResourceOptions::StorageModeShared;
    let halves = |v: &[f32]| -> Vec<u16> {
        v.iter()
            .map(|&x| {
                // Exact: these values already round-trip through binary16.
                let bits = x.to_bits();
                let sign = ((bits >> 16) & 0x8000) as u16;
                if x == 0.0 {
                    return sign;
                }
                let e = ((bits >> 23) & 0xff) as i32 - 127;
                let mant = (bits & 0x7f_ffff) >> 13;
                sign | (((e + 15) as u16) << 10) | mant as u16
            })
            .collect()
    };
    for split in [8u32, 1, 2, 4, 16] {
        let src = try_emit(&with_split(&split.to_string())).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("laguna_attention_decode_gqa_f16", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
        let threads = 32 * split as u64;
        assert!(pipe.max_total_threads_per_threadgroup() >= threads, "split {split}");

        let kh = halves(&keys);
        let vh = halves(&values);
        let fbuf = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
        let hbuf = |v: &[u16]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 2) as u64, o);
        let (bq, bg, bk, bv) = (fbuf(&q), fbuf(&gate), hbuf(&kh), hbuf(&vh));
        let bo = dev.new_buffer((N_HEAD * HEAD_DIM * 4) as u64, o);
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [&bq, &bg, &bk, &bv, &bo].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, v) in [N_HEAD as u32, N_HEAD_KV as u32, HEAD_DIM as u32, CACHE_CAP as u32, KEY_START as u32, KEY_COUNT as u32]
            .iter()
            .enumerate()
        {
            enc.set_bytes(5 + i as u64, 4, v as *const u32 as *const _);
        }
        enc.set_bytes(11, 4, &scale as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(N_HEAD as u64, 1, 1), MTLSize::new(threads, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(bo.contents() as *const f32, N_HEAD * HEAD_DIM) };
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1e-2),
                "split {split}: head {} d {}: got {g}, want {w}",
                i / HEAD_DIM,
                i % HEAD_DIM
            );
        }
    }
}
