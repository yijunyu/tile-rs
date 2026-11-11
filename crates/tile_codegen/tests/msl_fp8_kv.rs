//! The DeepSeek-V4 FP8 KV-cache kernels, emitted from their intrinsics.
//!
//! At the shipped block of 64 both emitted files must stay byte-identical to
//! the committed kernels. At other block sizes, on a Metal GPU, they must
//! match a CPU port of the engine's block-wise E4M3FN round trip.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const STORE_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_kv_fp8_store(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %hd = llvm.mlir.constant(576 : i32) : i32
    %nr = llvm.mlir.constant(64 : i32) : i32
    %rr = llvm.mlir.constant(3 : i32) : i32
    %res = llvm.call @__tile_kv_fp8_store_f32(%hd, %nr, %hd, %nr, %rr) : (i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const QUANTIZE_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_fp8_kv_quantize(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00 = llvm.mlir.constant(576 : i32) : i32
    %ne01 = llvm.mlir.constant(4 : i32) : i32
    %ne02 = llvm.mlir.constant(2 : i32) : i32
    %ne03 = llvm.mlir.constant(1 : i32) : i32
    %nb01 = llvm.mlir.constant(576 : i32) : i32
    %nb02 = llvm.mlir.constant(2304 : i32) : i32
    %nb03 = llvm.mlir.constant(4608 : i32) : i32
    %nrot = llvm.mlir.constant(64 : i32) : i32
    %res = llvm.call @__tile_fp8_kv_quantize_f32(%ne00, %ne01, %ne00, %ne01, %ne02, %ne03, %nb01, %nb02, %nb03, %nb01, %nb02, %nb03, %nrot) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
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

fn store_mlir(block: &str) -> String {
    format!(
        r#"
module {{
  llvm.func @store_gate(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i32) attributes {{hacc.entry}} {{
    ^bb0:
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %b = llvm.mlir.constant({block} : i32) : i32
    %r = llvm.call @__tile_kv_fp8_store_f32(%arg0, %arg1, %c1, %c1, %c1, %b) : (!llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32) -> i32
    llvm.return
  }}
}}
"#
    )
}

fn quantize_mlir(block: &str) -> String {
    format!(
        r#"
module {{
  llvm.func @quantize_gate(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {{hacc.entry}} {{
    ^bb0:
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %b = llvm.mlir.constant({block} : i32) : i32
    %r = llvm.call @__tile_fp8_kv_quantize_f32(%arg0, %arg1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1, %b) : (!llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }}
}}
"#
    )
}

#[test]
fn shipped_block_is_byte_identical_to_the_shipped_kernels() {
    assert_eq!(try_emit(STORE_MLIR).unwrap(), golden("dsv4_kv_fp8_store"));
    assert_eq!(try_emit(QUANTIZE_MLIR).unwrap(), golden("dsv4_fp8_kv_quantize"));
    assert_eq!(try_emit(&store_mlir("64")).unwrap(), golden("dsv4_kv_fp8_store").replace("ds4_dsv4_kv_fp8_store(", "store_gate("));
    assert_eq!(
        try_emit(&quantize_mlir("64")).unwrap(),
        golden("dsv4_fp8_kv_quantize").replace("ds4_dsv4_fp8_kv_quantize(", "quantize_gate(")
    );
}

#[test]
fn block_reaches_the_kernels_and_bad_blocks_are_refused() {
    let src = try_emit(&store_mlir("16")).unwrap();
    assert!(src.contains("scratch[16]") && src.contains("off += 16)") && src.contains("stride = 8u;"), "{src}");
    assert!(!src.contains("scratch[64]") && !src.contains("stride = 32u"), "{src}");
    let q = try_emit(&quantize_mlir("128")).unwrap();
    assert!(q.contains("scratch[128]") && q.contains("i += 128u)") && q.contains("stride = 64u;"), "{q}");
    assert!(try_emit(&store_mlir("48")).unwrap_err().contains("kv_fp8_store: block 48 must be a power of two"));
    assert!(try_emit(&quantize_mlir("2048")).unwrap_err().contains("fp8_kv_quantize: block 2048"));
    let runtime = store_mlir("16").replace("%c1, %b)", "%c1, %arg2)");
    assert!(runtime.contains("%c1, %arg2)"));
    assert!(try_emit(&runtime).unwrap_err().contains("operand 5 (block) must be a positive integer constant"));
}

/// Port of the engine's CPU reference (antirez ds4.c), with the block a parameter.
fn e4m3fn_value(i: i32) -> f32 {
    const EXP_SCALE: [f32; 16] =
        [0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
    let (exp, mant) = ((i >> 3) & 0x0f, i & 0x07);
    if exp == 0 { mant as f32 * 0.001_953_125 } else { (1.0 + mant as f32 * 0.125) * EXP_SCALE[exp as usize] }
}

fn e4m3fn_round_trip(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);
    let (mut lo, mut hi) = (0i32, 126i32);
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value(mid) <= ax { lo = mid } else { hi = mid - 1 }
    }
    let mut best = lo;
    if best < 126 {
        let (bd, nd) = ((ax - e4m3fn_value(best)).abs(), (ax - e4m3fn_value(best + 1)).abs());
        if nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0) {
            best += 1;
        }
    }
    sign * e4m3fn_value(best)
}

fn quantize_row(x: &mut [f32], n_nope: usize, block: usize) {
    let mut off = 0;
    while off < n_nope {
        let end = (off + block).min(n_nope);
        let amax = x[off..end].iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0e-4);
        let scale = (amax / 448.0).log2().ceil().exp2();
        for v in &mut x[off..end] {
            *v = e4m3fn_round_trip((*v / scale).clamp(-448.0, 448.0)) * scale;
        }
        off += block;
    }
}

/// f32 -> binary16 -> f32, rounding to nearest even.
fn half_round_trip(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    let e = (a.log2().floor() as i32).max(-14);
    let step = 2f32.powi(e - 10);
    let n = a / step;
    let r = n.round();
    let r = if (n - n.floor() - 0.5).abs() < f32::EPSILON && r % 2.0 != 0.0 { r - 1.0 } else { r };
    (r * step).copysign(x)
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Vary the magnitude by block so each block picks its own scale.
                let mag = [0.003f32, 0.4, 7.0, 90.0][(i / 5) % 4];
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * mag
            })
            .collect()
    }

    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-5 * want.abs().max(1e-3)
    }

    fn pipeline(dev: &Device, src: &str, name: &str) -> metal::ComputePipelineState {
        let lib = dev.new_library_with_source(src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function(name, None).unwrap()));
        dev.new_compute_pipeline_state(&desc).unwrap()
    }

    #[test]
    fn emitted_kernels_match_the_cpu_round_trip_at_several_blocks() {
        let Some(dev) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let o = MTLResourceOptions::StorageModeShared;
        for block in [64usize, 16, 128, 8] {
            for (head_dim, n_rot) in [(576usize, 64usize), (213, 13)] {
                let n_nope = head_dim - n_rot;

                // kv_fp8_store: quantize one row in place, write its half round trip to raw row 2.
                let pipe = pipeline(&dev, &try_emit(&store_mlir(&block.to_string())).unwrap(), "store_gate");
                let kv = values(head_dim, 7 + block as u64);
                let bkv = dev.new_buffer_with_data(kv.as_ptr() as *const _, (head_dim * 4) as u64, o);
                let braw = dev.new_buffer((4 * head_dim * 4) as u64, o);
                let queue = dev.new_command_queue();
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&bkv), 0);
                enc.set_buffer(1, Some(&braw), 0);
                for (i, v) in [head_dim as u32, n_rot as u32, 2u32].iter().enumerate() {
                    enc.set_bytes(2 + i as u64, 4, v as *const u32 as *const _);
                }
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(block as u64, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let mut want = kv.clone();
                quantize_row(&mut want, n_nope, block);
                let got = unsafe { std::slice::from_raw_parts(bkv.contents() as *const f32, head_dim) };
                let raw = unsafe { std::slice::from_raw_parts((braw.contents() as *const f32).add(2 * head_dim), head_dim) };
                for i in 0..head_dim {
                    assert!(close(got[i], want[i]), "store block {block} head_dim {head_dim} kv[{i}]: got {} want {}", got[i], want[i]);
                    let want_raw = half_round_trip(want[i]);
                    assert!(close(raw[i], want_raw), "store block {block} head_dim {head_dim} raw[{i}]: got {} want {want_raw}", raw[i]);
                }

                // fp8_kv_quantize: 3 x 2 x 1 rows, src -> dst.
                let (ne01, ne02, ne03) = (3usize, 2usize, 1usize);
                let n_rows = ne01 * ne02 * ne03;
                let pipe = pipeline(&dev, &try_emit(&quantize_mlir(&block.to_string())).unwrap(), "quantize_gate");
                let src = values(n_rows * head_dim, 99 + block as u64);
                let bsrc = dev.new_buffer_with_data(src.as_ptr() as *const _, (src.len() * 4) as u64, o);
                let bdst = dev.new_buffer((src.len() * 4) as u64, o);
                let (nb1, nb2, nb3) = (head_dim, head_dim * ne01, head_dim * ne01 * ne02);
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&bsrc), 0);
                enc.set_buffer(1, Some(&bdst), 0);
                let scalars = [head_dim, ne01, ne02, ne03, nb1, nb2, nb3, nb1, nb2, nb3, n_rot].map(|v| v as u32);
                for (i, v) in scalars.iter().enumerate() {
                    enc.set_bytes(2 + i as u64, 4, v as *const u32 as *const _);
                }
                enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(block as u64, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let got = unsafe { std::slice::from_raw_parts(bdst.contents() as *const f32, src.len()) };
                for r in 0..n_rows {
                    let mut want = src[r * head_dim..(r + 1) * head_dim].to_vec();
                    quantize_row(&mut want, n_nope, block);
                    for i in 0..head_dim {
                        let g = got[r * head_dim + i];
                        assert!(close(g, want[i]), "quantize block {block} head_dim {head_dim} row {r} [{i}]: got {g} want {}", want[i]);
                    }
                }
            }
        }
    }
}
