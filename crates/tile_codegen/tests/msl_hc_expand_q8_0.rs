//! The two q8_0 HC expand kernels: a quantized matvec whose result is fed
//! straight into a four-way hidden-channel expansion.
//!
//! Both wrote into their text how many simdgroups a threadgroup runs, and that
//! is now a trailing operand. Nothing else in them moves: the two output rows
//! and the eight-element slice of a quantization block a lane takes are both
//! written out by hand rather than looped, so a number in the text is not the
//! only thing a rewrite would have to change.
//!
//! At the shipped split the emitted text is unchanged; at other splits, on a
//! Metal GPU, both must match a CPU reference of the same arithmetic.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const EXPAND: &str = r#"
module {
  llvm.func @ds4_dsv4_q8_hc_expand4_q8_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00      = llvm.mlir.constant(64 : i32) : i32
    %ne01      = llvm.mlir.constant(4  : i32) : i32
    %nb01      = llvm.mlir.constant(68 : i32) : i32
    %n_hc      = llvm.mlir.constant(4  : i32) : i32
    %n_tokens  = llvm.mlir.constant(1  : i32) : i32
    %nb_block0 = llvm.mlir.constant(4  : i32) : i32
    %nb_res0   = llvm.mlir.constant(16 : i32) : i32
    %nb_res1   = llvm.mlir.constant(4  : i32) : i32
    %nb_post0  = llvm.mlir.constant(4  : i32) : i32
    %nb_comb0  = llvm.mlir.constant(16 : i32) : i32
    %nb_comb1  = llvm.mlir.constant(4  : i32) : i32
    %nb0       = llvm.mlir.constant(16 : i32) : i32
    %nb1       = llvm.mlir.constant(4  : i32) : i32
    %r = llvm.call @__tile_dsv4_q8_hc_expand4_q8_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %ne00, %ne01, %nb01, %n_hc, %n_tokens, %nb_block0, %nb_res0, %nb_res1, %nb_post0, %nb_comb0, %nb_comb1, %nb0, %nb1) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const SHARED_DOWN: &str = r#"
module {
  llvm.func @ds4_dsv4_shared_down_hc_expand4_q8_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00      = llvm.mlir.constant(64 : i32) : i32
    %ne01      = llvm.mlir.constant(4  : i32) : i32
    %nb01      = llvm.mlir.constant(68 : i32) : i32
    %n_hc      = llvm.mlir.constant(4  : i32) : i32
    %n_tokens  = llvm.mlir.constant(1  : i32) : i32
    %nb_block0 = llvm.mlir.constant(4  : i32) : i32
    %nb_res0   = llvm.mlir.constant(16 : i32) : i32
    %nb_res1   = llvm.mlir.constant(4  : i32) : i32
    %nb_post0  = llvm.mlir.constant(4  : i32) : i32
    %nb_comb0  = llvm.mlir.constant(16 : i32) : i32
    %nb_comb1  = llvm.mlir.constant(4  : i32) : i32
    %nb0       = llvm.mlir.constant(16 : i32) : i32
    %nb1       = llvm.mlir.constant(4  : i32) : i32
    %r = llvm.call @__tile_dsv4_shared_down_hc_expand4_q8_0(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %ne00, %ne01, %nb01, %n_hc, %n_tokens, %nb_block0, %nb_res0, %nb_res1, %nb_post0, %nb_comb0, %nb_comb1, %nb0, %nb1) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const EXPAND_CALLEE: &str = "__tile_dsv4_q8_hc_expand4_q8_0";
const SHARED_DOWN_CALLEE: &str = "__tile_dsv4_shared_down_hc_expand4_q8_0";

/// The simdgroup count the kernels were written at.
const SHIPPED: [u32; 1] = [2];

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
    for (mlir, callee) in [(EXPAND, EXPAND_CALLEE), (SHARED_DOWN, SHARED_DOWN_CALLEE)] {
        let bare = try_emit(mlir).unwrap();
        assert_eq!(bare, try_emit(&with_split(mlir, callee, &SHIPPED)).unwrap());
        for want in [
            "constexpr short NSG = 2;",
            "constexpr short NW  = 32;",
            "constexpr short NQ  = 8;",
            "constexpr short NR0 = 2;",
        ] {
            assert!(bare.contains(want), "missing {want} in\n{bare}");
        }
    }
}

#[test]
fn the_split_reaches_the_kernels_and_bad_ones_are_refused() {
    for (mlir, callee) in [(EXPAND, EXPAND_CALLEE), (SHARED_DOWN, SHARED_DOWN_CALLEE)] {
        let src = try_emit(&with_split(mlir, callee, &[8])).unwrap();
        assert!(src.contains("constexpr short NSG = 8;"), "{src}");
        // The hand-written rows, lane slice and block size do not move with it.
        for fixed in ["constexpr short NR0 = 2;", "constexpr short NQ  = 8;", "constexpr int   QK8_0 = 32;"] {
            assert!(src.contains(fixed), "missing {fixed} in\n{src}");
        }
        assert!(try_emit(&with_split(mlir, callee, &[64])).unwrap_err().contains("nsg 64"));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};

    /// Mirrors `struct HcMvUniforms` in the emitted prelude.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MvUniforms {
        ne00: i32,
        ne01: i32,
        nb01: u32,
    }

    /// Mirrors `struct HcExpandUniforms`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct HcUniforms {
        n_hc: i32,
        n_tokens: i32,
        nb_block0: u32,
        nb_res0: u32,
        nb_res1: u32,
        nb_post0: u32,
        nb_comb0: u32,
        nb_comb1: u32,
        nb0: u32,
        nb1: u32,
    }

    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let queue = dev.new_command_queue();

    // Enough blocks per row that a threadgroup has to stride through them: with
    // fewer, every simdgroup count covers the row in one pass and a wrong stride
    // cannot show.
    const NE00: usize = 4096;
    const ROWS: usize = 6; // odd against the two rows a threadgroup owns
    const HC: usize = 4;
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    // q8_0 weights, with a scale that survives binary16 exactly.
    let nb = NE00 / BLOCK;
    let mut w_bytes = vec![0u8; ROWS * nb * BLOCK_BYTES];
    let mut w_deq = vec![0f32; ROWS * NE00];
    for r in 0..ROWS {
        for b in 0..nb {
            let scale = 1.0f32 / (1 << (4 + (next() % 3))) as f32;
            let bits = scale.to_bits();
            let half = ((bits >> 16) & 0x8000) as u16
                | ((((bits >> 23) & 0xff) as i32 - 127 + 15) as u16) << 10
                | ((bits & 0x7f_ffff) >> 13) as u16;
            let off = (r * nb + b) * BLOCK_BYTES;
            w_bytes[off..off + 2].copy_from_slice(&half.to_le_bytes());
            for i in 0..BLOCK {
                let q = (next() % 255) as i64 - 127;
                w_bytes[off + 2 + i] = (q as i8) as u8;
                w_deq[r * NE00 + b * BLOCK + i] = q as f32 * scale;
            }
        }
    }
    let small = |n: usize, next: &mut dyn FnMut() -> u64| -> Vec<f32> {
        (0..n).map(|_| ((next() % 512) as f32 - 256.0) / 256.0).collect()
    };
    let x = small(NE00, &mut next);
    let block_in = small(ROWS, &mut next);
    let residual = small(ROWS * HC, &mut next);
    let post = small(HC, &mut next);
    let comb = small(HC * HC, &mut next);

    let o = MTLResourceOptions::StorageModeShared;
    let f = |v: &[f32]| dev.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * 4) as u64, o);
    let b_w = dev.new_buffer_with_data(w_bytes.as_ptr() as *const _, w_bytes.len() as u64, o);
    let (b_x, b_block, b_res, b_post, b_comb) =
        (f(&x), f(&block_in), f(&residual), f(&post), f(&comb));

    let mv = MvUniforms { ne00: NE00 as i32, ne01: ROWS as i32, nb01: (nb * BLOCK_BYTES) as u32 };
    let hc = HcUniforms {
        n_hc: HC as i32,
        n_tokens: 1,
        nb_block0: 4,
        nb_res0: (HC * 4) as u32,
        nb_res1: 4,
        nb_post0: 4,
        nb_comb0: (HC * 4) as u32,
        nb_comb1: 4,
        nb0: (HC * 4) as u32,
        nb1: 4,
    };

    // What the matvec should produce, and then the expansion around it.
    let matvec: Vec<f32> =
        (0..ROWS).map(|r| (0..NE00).map(|k| w_deq[r * NE00 + k] * x[k]).sum()).collect();

    for split @ [nsg] in [SHIPPED, [1], [4], [8], [16]] {
        let run = |mlir: &str, callee: &str, entry: &str, shared_down: bool| -> (Vec<f32>, Vec<f32>) {
            let src = try_emit(&with_split(mlir, callee, &[nsg])).unwrap();
            let lib =
                dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
            let desc = ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&lib.get_function(entry, None).unwrap()));
            let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
            let b_mv_out = dev.new_buffer((ROWS * 4) as u64, o);
            let b_out = dev.new_buffer((ROWS * HC * 4) as u64, o);
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            let bufs: Vec<&metal::Buffer> = if shared_down {
                vec![&b_w, &b_x, &b_mv_out, &b_block, &b_res, &b_post, &b_comb, &b_out]
            } else {
                vec![&b_w, &b_x, &b_mv_out, &b_res, &b_post, &b_comb, &b_out]
            };
            let n = bufs.len() as u64;
            for (i, b) in bufs.into_iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_bytes(n, std::mem::size_of::<MvUniforms>() as u64, &mv as *const MvUniforms as *const _);
            enc.set_bytes(n + 1, std::mem::size_of::<HcUniforms>() as u64, &hc as *const HcUniforms as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((ROWS as u64).div_ceil(2), 1, 1),
                MTLSize::new((nsg * 32) as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            (
                unsafe { std::slice::from_raw_parts(b_mv_out.contents() as *const f32, ROWS) }.to_vec(),
                unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, ROWS * HC) }.to_vec(),
            )
        };

        for (mlir, callee, entry, shared_down) in [
            (EXPAND, EXPAND_CALLEE, "ds4_dsv4_q8_hc_expand4_q8_0", false),
            (SHARED_DOWN, SHARED_DOWN_CALLEE, "ds4_dsv4_shared_down_hc_expand4_q8_0", true),
        ] {
            let (got_mv, got_out) = run(mlir, callee, entry, shared_down);
            for row in 0..ROWS {
                let want = matvec[row];
                assert!(
                    (got_mv[row] - want).abs() <= 2e-3 * want.abs().max(1e-2),
                    "{entry} matvec at split {split:?}, row {row}: got {}, want {want}",
                    got_mv[row]
                );
                // The shared-down form adds its own block input to the matvec first.
                let v = if shared_down { matvec[row] + block_in[row] } else { matvec[row] };
                for dst in 0..HC {
                    let mut want = v * post[dst];
                    for src in 0..HC {
                        want += comb[dst * HC + src] * residual[row * HC + src];
                    }
                    let g = got_out[row * HC + dst];
                    assert!(
                        (g - want).abs() <= 2e-3 * want.abs().max(1e-2),
                        "{entry} expansion at split {split:?}, row {row} channel {dst}: got {g}, want {want}"
                    );
                }
            }
        }
    }
}
