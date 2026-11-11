//! The DeepSeek-V4 per-expert MoE ID map, emitted from one intrinsic.
//!
//! `__tile_mul_mm_id_map0_full` takes the expert fan-out `ne20` from its operand;
//! the nine `__tile_mul_mm_id_map0_ne20_<N>_full` intrinsics are aliases with
//! `ne20` in the name. Each alias must emit exactly what the generic intrinsic
//! emits for its N, and on a Metal GPU the kernel must build the same expert
//! lists as a CPU reference at several fan-outs and expert capacities.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const ALIAS_1: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_1(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(1  : i32) : i32
    %nb21 = llvm.mlir.constant(4  : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_1_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_2: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_2(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(2  : i32) : i32
    %nb21 = llvm.mlir.constant(8  : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_2_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_4: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_4(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(4  : i32) : i32
    %nb21 = llvm.mlir.constant(16 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_4_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_5: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_5(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(5  : i32) : i32
    %nb21 = llvm.mlir.constant(20 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_5_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_6: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_6(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(6  : i32) : i32
    %nb21 = llvm.mlir.constant(24 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_6_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_8: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_8(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(8  : i32) : i32
    %nb21 = llvm.mlir.constant(32 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_8_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_10: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_10(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(10 : i32) : i32
    %nb21 = llvm.mlir.constant(40 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_10_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_16: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_16(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(16 : i32) : i32
    %nb21 = llvm.mlir.constant(64 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_16_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;
const ALIAS_22: &str = r#"
module {
  llvm.func @ds4_kernel_mul_mm_id_map0_ne20_22(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne02 = llvm.mlir.constant(0  : i32) : i32
    %ne10 = llvm.mlir.constant(0  : i32) : i32
    %ne11 = llvm.mlir.constant(0  : i32) : i32
    %nb11 = llvm.mlir.constant(0  : i32) : i32
    %nb12 = llvm.mlir.constant(0  : i32) : i32
    %ne21 = llvm.mlir.constant(4  : i32) : i32
    %ne20 = llvm.mlir.constant(22 : i32) : i32
    %nb21 = llvm.mlir.constant(88 : i32) : i32
    %r = llvm.call @__tile_mul_mm_id_map0_ne20_22_full(%arg0, %arg1, %arg2, %ne02, %ne10, %ne11, %nb11, %nb12, %ne21, %ne20, %nb21) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const ALIASES: [(u32, &str); 9] = [
    (1, ALIAS_1),
    (2, ALIAS_2),
    (4, ALIAS_4),
    (5, ALIAS_5),
    (6, ALIAS_6),
    (8, ALIAS_8),
    (10, ALIAS_10),
    (16, ALIAS_16),
    (22, ALIAS_22),
];

fn try_emit(mlir: &str) -> Result<String, String> {
    TargetRegistry::with_builtin().select("msl").expect("msl").emit(mlir, &EmitOpts::default()).map(|o| o.source)
}

/// The generic intrinsic, with `ne20` and optionally `max_experts` as constants.
fn generic(func: &str, ne20: &str, max_experts: Option<&str>) -> String {
    let (extra_op, extra_ty, extra_const) = match max_experts {
        Some(m) => (", %mx", ", i32", format!("\n    %mx   = llvm.mlir.constant({m} : i32) : i32")),
        None => ("", "", String::new()),
    };
    format!(
        r#"
module {{
  llvm.func @{func}(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i32) attributes {{hacc.entry}} {{
    ^bb0:
    %c0   = llvm.mlir.constant(0 : i32) : i32
    %ne21 = llvm.mlir.constant(4 : i32) : i32
    %ne20 = llvm.mlir.constant({ne20} : i32) : i32
    %nb21 = llvm.mlir.constant(32 : i32) : i32{extra_const}
    %r = llvm.call @__tile_mul_mm_id_map0_full(%arg0, %arg1, %arg2, %c0, %c0, %c0, %c0, %c0, %ne21, %ne20, %nb21{extra_op}) : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32{extra_ty}) -> i32
    llvm.return
  }}
}}
"#
    )
}

fn func_name(alias_mlir: &str) -> &str {
    let at = alias_mlir.find("llvm.func @").unwrap() + "llvm.func @".len();
    let end = alias_mlir[at..].find('(').unwrap();
    &alias_mlir[at..at + end]
}

#[test]
fn every_named_alias_emits_the_generic_kernel() {
    for (n, alias) in ALIASES {
        let via_alias = try_emit(alias).unwrap();
        let via_generic = try_emit(&generic(func_name(alias), &n.to_string(), None)).unwrap();
        assert_eq!(via_alias, via_generic, "ne20 {n}");
        assert!(via_alias.contains(&format!("constexpr short NE20 = {n};")), "ne20 {n}");
        assert!(via_alias.contains(&format!("shmem_ids[256 * {n}]")), "ne20 {n}");
    }
}

#[test]
fn capacity_reaches_the_kernel_and_bad_shapes_are_refused() {
    let src = try_emit(&generic("g", "6", Some("64"))).unwrap();
    assert!(src.contains("shmem_ids[64 * 6]") && src.contains("constexpr short NE20 = 6;"), "{src}");
    // A name that disagrees with its operand.
    let mismatch = ALIASES[5].1.replace("llvm.mlir.constant(8  : i32) : i32\n    %nb21", "llvm.mlir.constant(7  : i32) : i32\n    %nb21");
    assert!(mismatch.contains("constant(7  : i32)"), "fixture layout changed");
    let e = try_emit(&mismatch).unwrap_err();
    assert!(e.contains("operand 9 (ne20) is 7, but the intrinsic name says 8"), "{e}");
    // The generic intrinsic needs a constant ne20.
    let runtime = generic("g", "8", None).replace("%ne21, %ne20, %nb21)", "%ne21, %arg3, %nb21)");
    assert!(runtime.contains("%arg3, %nb21)"));
    assert!(try_emit(&runtime).unwrap_err().contains("operand 9 (ne20) must be a positive integer constant"));
    assert!(try_emit(&generic("g", "22", Some("1024"))).unwrap_err().contains("past the 32768 of threadgroup memory"));
    assert!(try_emit(&generic("g", "8", Some("2000"))).unwrap_err().contains("max_experts 2000 must be from 1 to 1024"));
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_kernels_match_the_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};
    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |bound: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as u32
    };
    for (ne20, experts, ne21) in [(8u32, 256u32, 37u32), (6, 64, 10), (22, 128, 5), (1, 16, 33), (16, 32, 7), (4, 512, 9)] {
        let src = try_emit(&generic("map0_gate", &ne20.to_string(), Some(&experts.to_string()))).unwrap();
        let lib = dev.new_library_with_source(&src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function("map0_gate", None).unwrap()));
        let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
        assert!(pipe.max_total_threads_per_threadgroup() >= experts as u64);

        // Each token row selects ne20 distinct experts. The 512-expert shape checks
        // correctness past the shipped capacity of 256. It cannot catch a kernel
        // that under-sizes its staging: those writes land past the declared array
        // but inside the threadgroup allocation and read back intact, so the
        // capacity is pinned by the emitted-text test above instead.
        let mut src2 = Vec::with_capacity((ne21 * ne20) as usize);
        for _ in 0..ne21 {
            let mut row: Vec<i32> = Vec::new();
            while row.len() < ne20 as usize {
                let e = next(experts) as i32;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            src2.extend(row);
        }
        let o = MTLResourceOptions::StorageModeShared;
        let b_src = dev.new_buffer_with_data(src2.as_ptr() as *const _, (src2.len() * 4) as u64, o);
        let b_tpe = dev.new_buffer((experts * 4) as u64, o);
        let b_ids = dev.new_buffer((experts * ne21 * 4) as u64, o);
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&b_src), 0);
        enc.set_buffer(1, Some(&b_tpe), 0);
        enc.set_buffer(2, Some(&b_ids), 0);
        let (zero_i, zero_u) = (0i32, 0u64);
        for i in [3u64, 4, 5] {
            enc.set_bytes(i, 4, &zero_i as *const i32 as *const _);
        }
        for i in [6u64, 7] {
            enc.set_bytes(i, 8, &zero_u as *const u64 as *const _);
        }
        let (ne21_i, ne20_i, nb21_u) = (ne21 as i32, ne20 as i32, (ne20 * 4) as u64);
        enc.set_bytes(8, 4, &ne21_i as *const i32 as *const _);
        enc.set_bytes(9, 4, &ne20_i as *const i32 as *const _);
        enc.set_bytes(10, 8, &nb21_u as *const u64 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(experts as u64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let tpe = unsafe { std::slice::from_raw_parts(b_tpe.contents() as *const u32, experts as usize) };
        let ids = unsafe { std::slice::from_raw_parts(b_ids.contents() as *const i32, (experts * ne21) as usize) };
        for e in 0..experts {
            let want: Vec<i32> = (0..ne21)
                .flat_map(|row| (0..ne20).map(move |i| (row, i)))
                .filter(|&(row, i)| src2[(row * ne20 + i) as usize] == e as i32)
                .map(|(row, i)| (row * ne20 + i) as i32)
                .collect();
            let base = (e * ne21) as usize;
            assert_eq!(tpe[e as usize] as usize, want.len(), "ne20 {ne20} experts {experts}: count for expert {e}");
            assert_eq!(&ids[base..base + want.len()], &want[..], "ne20 {ne20} experts {experts}: ids for expert {e}");
        }
    }
}
