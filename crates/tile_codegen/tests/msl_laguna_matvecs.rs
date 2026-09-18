//! The two Laguna half-precision matvecs: the fused q/k/v/gate projection and
//! the attention output projection with its residual add.
//!
//! Both walked their input in 32-element blocks and wrote three numbers into
//! the kernel text: the simdgroups a threadgroup runs, the output rows each
//! owns, and how many elements of a block one lane reads. The row count was
//! also a literal accumulator list with room for exactly two. All three are now
//! trailing operands. The committed kernels pin the shipped split byte for
//! byte; at other splits, on a Metal GPU, both must match a CPU reference.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const QKVG: &str = r#"
module {
  llvm.func @laguna_qkvg_f16_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg4, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @__tile_laguna_qkvg_f16_f32(%a, %r, %c) : (i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg5, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const RESIDUAL: &str = r#"
module {
  llvm.func @laguna_attn_output_residual_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @__tile_laguna_attn_output_residual_f32(%a, %r, %c) : (i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const QKVG_CALLEE: &str = "__tile_laguna_qkvg_f16_f32";
const RESIDUAL_CALLEE: &str = "__tile_laguna_attn_output_residual_f32";

/// nsg, nr0, nf as the committed kernels were built.
const SHIPPED: [u32; 3] = [4, 2, 16];

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
    for (mlir, callee) in [(QKVG, QKVG_CALLEE), (RESIDUAL, RESIDUAL_CALLEE)] {
        let bare = try_emit(mlir).unwrap();
        assert_eq!(bare, try_emit(&with_split(mlir, callee, &SHIPPED)).unwrap());
        for want in [
            "constexpr short NR0 = 2;",
            "constexpr short NW  = 32;   // N_SIMDWIDTH",
            "constexpr short NB  = 32;",
            "constexpr short NF  = 16;",
            "constexpr short NSG = 4;",
            "float sums[NR0] = {0.0f, 0.0f};",
        ] {
            assert!(bare.contains(want), "missing {want} in\n{bare}");
        }
    }
}

#[test]
fn the_split_reaches_the_kernels_and_bad_ones_are_refused() {
    let src = try_emit(&with_split(QKVG, QKVG_CALLEE, &[2, 4, 8])).unwrap();
    for want in [
        "constexpr short NSG = 2;",
        "constexpr short NR0 = 4;",
        "constexpr short NF  = 8;",
        "float sums[NR0] = {0.0f, 0.0f, 0.0f, 0.0f};",
    ] {
        assert!(src.contains(want), "missing {want} in\n{src}");
    }
    // The block is the simdgroup width, so it stays where the split does not.
    assert!(src.contains("constexpr short NB  = 32;"), "{src}");
    let r = try_emit(&with_split(RESIDUAL, RESIDUAL_CALLEE, &[1, 1, 32])).unwrap();
    assert!(r.contains("float sums[NR0] = {0.0f};") && r.contains("constexpr short NF  = 32;"), "{r}");

    let err = |m: &str, c: &str, v: &[u32]| try_emit(&with_split(m, c, v)).unwrap_err();
    assert!(err(QKVG, QKVG_CALLEE, &[64, 2, 16]).contains("nsg 64"));
    assert!(err(QKVG, QKVG_CALLEE, &[4, 32, 16]).contains("nr0 32"));
    // 12 neither divides the block nor keeps the four-wide read whole.
    assert!(err(QKVG, QKVG_CALLEE, &[4, 2, 12]).contains("nf 12"));
    assert!(err(RESIDUAL, RESIDUAL_CALLEE, &[4, 2, 2]).contains("nf 2"));
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

    const IN_DIM: usize = 2048; // 64 blocks of 32
    // Each projection's height is a multiple of every row count tested, since a
    // threadgroup picks its projection from its first row.
    const Q_DIM: usize = 24;
    const KV_DIM: usize = 12;
    const GATE_DIM: usize = 24;
    const OUT_DIM: usize = 36; // the attention output projection

    let mut state = 0xFACE_B00C_0042_1111u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Small multiples of 1/256 are exact in binary16 and in f32.
        ((state >> 40) as i64 % 129 - 64) as f32 / 256.0
    };
    let to_half = |v: &[f32]| -> Vec<u16> {
        v.iter()
            .map(|&x| {
                if x == 0.0 {
                    return 0;
                }
                let bits = x.to_bits();
                ((bits >> 16) & 0x8000) as u16
                    | ((((bits >> 23) & 0xff) as i32 - 127 + 15) as u16) << 10
                    | ((bits & 0x7f_ffff) >> 13) as u16
            })
            .collect()
    };

    let mut weights = |rows: usize| -> (Vec<f32>, Vec<u16>) {
        let w: Vec<f32> = (0..rows * IN_DIM).map(|_| next()).collect();
        let h = to_half(&w);
        (w, h)
    };
    let (wq, hq) = weights(Q_DIM);
    let (wk, hk) = weights(KV_DIM);
    let (wv, hv) = weights(KV_DIM);
    let (wg, hg) = weights(GATE_DIM);
    let (wo, ho) = weights(OUT_DIM);
    let x: Vec<f32> = (0..IN_DIM).map(|_| next()).collect();
    let residual: Vec<f32> = (0..OUT_DIM).map(|_| next()).collect();

    let o = MTLResourceOptions::StorageModeShared;
    let halfbuf = |h: &[u16]| dev.new_buffer_with_data(h.as_ptr() as *const _, (h.len() * 2) as u64, o);
    let floatbuf = |f: &[f32]| dev.new_buffer_with_data(f.as_ptr() as *const _, (f.len() * 4) as u64, o);
    let (bq, bk, bv, bg, bo_w) = (halfbuf(&hq), halfbuf(&hk), halfbuf(&hv), halfbuf(&hg), halfbuf(&ho));
    let bx = floatbuf(&x);
    let bres = floatbuf(&residual);

    let dot = |w: &[f32], row: usize| -> f32 { (0..IN_DIM).map(|k| w[row * IN_DIM + k] * x[k]).sum() };

    for split @ [nsg, nr0, nf] in [SHIPPED, [1, 1, 32], [2, 4, 8], [8, 3, 4], [2, 2, 16]] {
        let build = |mlir: &str, callee: &str, entry: &str| {
            let src = try_emit(&with_split(mlir, callee, &[nsg, nr0, nf])).unwrap();
            let lib =
                dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
            dev.new_compute_pipeline_state(&desc).unwrap()
        };
        let near = |got: f32, want: f32, what: &str, row: usize| {
            assert!(
                (got - want).abs() <= 2e-3 * want.abs().max(1e-2),
                "{what} at split {split:?}, row {row}: got {got}, want {want}"
            );
        };

        // The fused q/k/v/gate projection: four weight blocks, four outputs.
        {
            let pipe = build(QKVG, QKVG_CALLEE, "laguna_qkvg_f16_f32");
            let outs: Vec<_> =
                [Q_DIM, KV_DIM, KV_DIM, GATE_DIM].map(|n| dev.new_buffer((n * 4) as u64, o)).to_vec();
            let u: [u32; 4] = [IN_DIM as u32, Q_DIM as u32, KV_DIM as u32, GATE_DIM as u32];
            let total = (Q_DIM + 2 * KV_DIM + GATE_DIM) as u64;
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bq, &bk, &bv, &bg, &bx, &outs[0], &outs[1], &outs[2], &outs[3]].into_iter().enumerate()
            {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in u.iter().enumerate() {
                enc.set_bytes(9 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(total.div_ceil(nr0 as u64), 1, 1),
                MTLSize::new((nsg * 32) as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            for (buf, w, n, what) in [
                (&outs[0], &wq, Q_DIM, "q"),
                (&outs[1], &wk, KV_DIM, "k"),
                (&outs[2], &wv, KV_DIM, "v"),
                (&outs[3], &wg, GATE_DIM, "gate"),
            ] {
                let got = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) };
                for row in 0..n {
                    near(got[row], dot(w, row), what, row);
                }
            }
        }

        // The attention output projection, with its residual added in.
        {
            let pipe = build(RESIDUAL, RESIDUAL_CALLEE, "laguna_attn_output_residual_f32");
            let bout = dev.new_buffer((OUT_DIM * 4) as u64, o);
            let u: [u32; 2] = [IN_DIM as u32, OUT_DIM as u32];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            for (i, b) in [&bo_w, &bx, &bres, &bout].into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            for (i, v) in u.iter().enumerate() {
                enc.set_bytes(4 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((OUT_DIM as u64).div_ceil(nr0 as u64), 1, 1),
                MTLSize::new((nsg * 32) as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = unsafe { std::slice::from_raw_parts(bout.contents() as *const f32, OUT_DIM) };
            for row in 0..OUT_DIM {
                near(got[row], residual[row] + dot(&wo, row), "attention output", row);
            }
        }
    }
}
