//! The DeepSeek-V4 bitonic sorts, emitted from their intrinsics.
//!
//! Both kernels staged a fixed number of elements: 256 per row for the top-k
//! sort and 1024 for argsort. Each capacity is now a trailing operand, and the
//! shipped values must emit the committed kernels byte for byte. At other
//! capacities, on a Metal GPU, the kernels must sort like a CPU reference.
#![cfg(feature = "emitters")]

use tile_codegen::{EmitOpts, TargetRegistry};

const SORT_MLIR: &str = r#"
module {
  llvm.func @ds4_dsv4_sort_i32_rows_asc(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %k = llvm.mlir.constant(64 : i32) : i32
    %r = llvm.mlir.constant(4 : i32) : i32
    %res = llvm.call @__tile_sort_i32_rows_asc_i32(%k, %r, %k, %r) : (i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const ARGSORT_MLIR: &str = r#"
module {
  llvm.func @ds4_argsort_f32_i32_desc(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00  = llvm.mlir.constant(8 : i32) : i32
    %ne01  = llvm.mlir.constant(2 : i32) : i32
    %top_k = llvm.mlir.constant(4 : i32) : i32
    %ne0   = llvm.mlir.constant(4 : i32) : i32
    %nb01  = llvm.mlir.constant(32 : i32) : i32
    %r = llvm.call @__tile_argsort_f32_i32_desc(%arg0, %arg1, %ne00, %ne01, %top_k, %ne0, %nb01) : (!llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
}
"#;

const ARGSORT_FULL_MLIR: &str = r#"
module {
  llvm.func @ds4_kernel_argsort_f32_i32_desc_full(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %ne00  = llvm.mlir.constant(16 : i32) : i32
    %ne01  = llvm.mlir.constant(2 : i32) : i32
    %ne02  = llvm.mlir.constant(1 : i32) : i32
    %ne03  = llvm.mlir.constant(1 : i32) : i32
    %nb00  = llvm.mlir.constant(4 : i32) : i32
    %nb01  = llvm.mlir.constant(64 : i32) : i32
    %nb02  = llvm.mlir.constant(128 : i32) : i32
    %nb03  = llvm.mlir.constant(128 : i32) : i32
    %ne0   = llvm.mlir.constant(8 : i32) : i32
    %ne1   = llvm.mlir.constant(2 : i32) : i32
    %ne2   = llvm.mlir.constant(1 : i32) : i32
    %ne3   = llvm.mlir.constant(1 : i32) : i32
    %top_k = llvm.mlir.constant(4 : i32) : i32
    %r = llvm.call @__tile_argsort_f32_i32_desc_full(%arg0, %arg1, %ne00, %ne01, %ne02, %ne03, %nb00, %nb01, %nb02, %nb03, %ne0, %ne1, %ne2, %ne3, %top_k) : (!llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
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

/// Append a capacity operand to a fixture's call.
fn with_capacity(mlir: &str, capacity: &str) -> String {
    let call = mlir.find("llvm.call").expect("call");
    let open = mlir[call..].find('(').unwrap() + call;
    let close = mlir[open..].find(')').unwrap() + open;
    let types_open = mlir[close..].find('(').unwrap() + close;
    let types_close = mlir[types_open..].find(')').unwrap() + types_open;
    let spliced = format!(
        "{}, %cap{}, i32{}",
        &mlir[..close],
        &mlir[close..types_close],
        &mlir[types_close..]
    );
    let line = spliced[..spliced.find("llvm.call").unwrap()].rfind('\n').unwrap() + 1;
    format!(
        "{}    %cap = llvm.mlir.constant({capacity} : i32) : i32\n{}",
        &spliced[..line],
        &spliced[line..]
    )
}

#[test]
fn shipped_capacities_are_byte_identical() {
    assert_eq!(try_emit(SORT_MLIR).unwrap(), golden("dsv4_sort_i32_rows_asc"));
    assert_eq!(try_emit(ARGSORT_MLIR).unwrap(), golden("argsort_f32_i32_desc"));
    assert_eq!(try_emit(ARGSORT_FULL_MLIR).unwrap(), golden("argsort_f32_i32_desc_full"));
    assert_eq!(try_emit(&with_capacity(SORT_MLIR, "256")).unwrap(), golden("dsv4_sort_i32_rows_asc"));
    assert_eq!(try_emit(&with_capacity(ARGSORT_MLIR, "1024")).unwrap(), golden("argsort_f32_i32_desc"));
    assert_eq!(try_emit(&with_capacity(ARGSORT_FULL_MLIR, "1024")).unwrap(), golden("argsort_f32_i32_desc_full"));
}

#[test]
fn capacity_reaches_the_kernels_and_bad_ones_are_refused() {
    let sorted = try_emit(&with_capacity(SORT_MLIR, "64")).unwrap();
    assert!(sorted.contains("constexpr uint MAX_TOPK = 64;"), "{sorted}");
    assert!(!sorted.contains("256"), "{sorted}");
    let arg = try_emit(&with_capacity(ARGSORT_MLIR, "16")).unwrap();
    assert!(arg.contains("threadgroup int shmem_i32[16];"), "{arg}");
    assert!(!arg.contains("1024"), "{arg}");
    for (mlir, why) in [
        (with_capacity(SORT_MLIR, "300"), "max_top_k 300 must be a power of two"),
        (with_capacity(SORT_MLIR, "2048"), "max_top_k 2048"),
        (with_capacity(ARGSORT_MLIR, "48"), "max_row 48 must be a power of two"),
    ] {
        let e = try_emit(&mlir).unwrap_err();
        assert!(e.contains(why), "{why}: {e}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn emitted_sorts_match_a_cpu_reference() {
    use metal::{CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize};
    let Some(dev) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let pipeline = |src: &str, name: &str| {
        let lib = dev.new_library_with_source(src, &CompileOptions::new()).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&lib.get_function(name, None).unwrap()));
        dev.new_compute_pipeline_state(&desc).unwrap()
    };
    let o = MTLResourceOptions::StorageModeShared;

    // Per-row ascending int sort: one threadgroup per row, top_k threads.
    for (top_k, rows, capacity) in [(64usize, 4usize, 256u32), (8, 3, 8), (256, 2, 256), (16, 5, 1024)] {
        let pipe = pipeline(&try_emit(&with_capacity(SORT_MLIR, &capacity.to_string())).unwrap(), "ds4_dsv4_sort_i32_rows_asc");
        let src: Vec<i32> = (0..rows * top_k).map(|_| (next() >> 40) as i32).collect();
        let b_src = dev.new_buffer_with_data(src.as_ptr() as *const _, (src.len() * 4) as u64, o);
        let b_dst = dev.new_buffer((src.len() * 4) as u64, o);
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&b_src), 0);
        enc.set_buffer(1, Some(&b_dst), 0);
        for (i, v) in [top_k as u32, rows as u32].iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, v as *const u32 as *const _);
        }
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(top_k as u64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(b_dst.contents() as *const i32, src.len()) };
        for r in 0..rows {
            let mut want = src[r * top_k..(r + 1) * top_k].to_vec();
            want.sort_unstable();
            assert_eq!(&got[r * top_k..(r + 1) * top_k], &want[..], "sort top_k {top_k} rows {rows} cap {capacity}: row {r}");
        }
    }

    // Descending argsort: one threadgroup per row, a power-of-two thread count.
    for (ne00, ne01, top_k, capacity) in [(8usize, 2usize, 4usize, 1024u32), (8, 2, 8, 8), (100, 3, 10, 128), (512, 2, 512, 512)] {
        let threads = ne00.next_power_of_two();
        assert!(threads as u32 <= capacity);
        let pipe = pipeline(&try_emit(&with_capacity(ARGSORT_MLIR, &capacity.to_string())).unwrap(), "ds4_argsort_f32_i32_desc");
        // Distinct values, so the descending order is unique.
        let mut vals: Vec<f32> = (0..ne00 * ne01).map(|i| i as f32 * 0.5).collect();
        for i in (1..vals.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            vals.swap(i, j);
        }
        let b_src = dev.new_buffer_with_data(vals.as_ptr() as *const _, (vals.len() * 4) as u64, o);
        let b_dst = dev.new_buffer((top_k * ne01 * 4) as u64, o);
        let queue = dev.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&b_src), 0);
        enc.set_buffer(1, Some(&b_dst), 0);
        for (i, v) in [ne00 as u32, ne01 as u32, top_k as u32, top_k as u32, (ne00 * 4) as u32].iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, v as *const u32 as *const _);
        }
        enc.dispatch_thread_groups(MTLSize::new(ne01 as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(b_dst.contents() as *const i32, top_k * ne01) };
        for r in 0..ne01 {
            let mut order: Vec<usize> = (0..ne00).collect();
            order.sort_by(|&a, &b| vals[r * ne00 + b].partial_cmp(&vals[r * ne00 + a]).unwrap());
            let want: Vec<i32> = order[..top_k].iter().map(|&i| i as i32).collect();
            assert_eq!(&got[r * top_k..(r + 1) * top_k], &want[..], "argsort ne00 {ne00} top_k {top_k} cap {capacity}: row {r}");
        }
    }
}
