//! MLIR-to-MUSA (Moore Threads GPU) translator.
//!
//! MUSA is source-level compatible with CUDA — same kernel syntax, same math
//! functions, same `<<<>>>` launch.  The only differences are:
//!   - Headers: `cuda_runtime.h` → `musa_runtime.h`
//!   - Runtime API prefix: `cuda*` → `musa*`
//!   - Compiler: `nvcc` → `mcc`
//!   - Architecture: `sm_XX` → `mp_XX`
//!   - Source extension: `.cu` → `.mu`
//!
//! This backend delegates to `mlir_to_gpu` and applies mechanical substitutions.

use crate::mlir_to_gpu::convert_mlir_to_gpu;

pub fn convert_mlir_to_musa(mlir_text: &str) -> Result<String, String> {
    let cuda_src = convert_mlir_to_gpu(mlir_text)?;
    Ok(cuda_to_musa(&cuda_src))
}

fn cuda_to_musa(cuda: &str) -> String {
    cuda
        // Banner
        .replace("mlir_to_gpu", "mlir_to_musa")
        .replace("CUDA", "MUSA")
        // Headers
        .replace("cuda_runtime.h", "musa_runtime.h")
        .replace("cuda_fp16.h", "musa_fp16.h")
        // Compile comments
        .replace("nvcc", "mcc")
        .replace("sm_80", "mp_22")
        .replace(".gpu.cu", ".musa.mu")
        .replace("--cuda-gpu-arch=", "--musa-gpu-arch=")
        // Runtime API (if present in comments/host code)
        .replace("cudaMalloc", "musaMalloc")
        .replace("cudaFree", "musaFree")
        .replace("cudaMemcpy", "musaMemcpy")
        .replace("cudaDeviceSynchronize", "musaDeviceSynchronize")
        .replace("cudaError_t", "musaError_t")
        .replace("cudaSuccess", "musaSuccess")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax_mlir() -> &'static str {
        r#"
module {
  llvm.func @tile_softmax(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1024 : i32) : i32
    %3 = llvm.call @ascend_tile_load_f32(%arg0, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %4 = llvm.call @ascend_tile_softmax_f32(%3, %3, %1, %2) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %4, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#
    }

    fn add_mlir() -> &'static str {
        r#"
module {
  llvm.func @tile_add(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(512 : i32) : i32
    %3 = llvm.call @ascend_tile_load_f32(%arg0, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %4 = llvm.call @ascend_tile_load_f32(%arg1, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %5 = llvm.call @ascend_tile_add_f32(%3, %3, %4, %1, %2) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %5, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#
    }

    #[test]
    fn test_musa_header() {
        let mu = convert_mlir_to_musa(softmax_mlir()).unwrap();
        assert!(mu.contains("musa_runtime.h"), "missing musa_runtime.h");
        assert!(mu.contains("mcc"), "missing mcc compiler reference");
        assert!(mu.contains("mp_22"), "missing MUSA arch");
        assert!(mu.contains("mlir_to_musa"), "missing mlir_to_musa banner");
        assert!(!mu.contains("cuda_runtime.h"), "should not contain cuda_runtime.h");
        assert!(!mu.contains("nvcc"), "should not contain nvcc");
    }

    #[test]
    fn test_musa_kernel_syntax_unchanged() {
        let mu = convert_mlir_to_musa(softmax_mlir()).unwrap();
        // Kernel syntax is identical to CUDA
        assert!(mu.contains("__global__"), "missing __global__");
        assert!(mu.contains("__shared__"), "missing __shared__");
        assert!(mu.contains("__syncthreads"), "missing __syncthreads");
        assert!(mu.contains("__shfl_down_sync"), "missing __shfl_down_sync");
        assert!(mu.contains("expf("), "missing expf");
        assert!(mu.contains("warp_reduce_max"), "missing warp helpers");
    }

    #[test]
    fn test_musa_add() {
        let mu = convert_mlir_to_musa(add_mlir()).unwrap();
        assert!(mu.contains("__global__ void tile_add"), "missing kernel");
        assert!(mu.contains("+"), "missing addition");
        assert!(mu.contains("musa_runtime.h"), "missing musa header");
    }

    fn silu_mul_mlir() -> &'static str {
        r#"
module {
  llvm.func @gated_mlp(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(8960 : i32) : i32
    %3 = llvm.call @ascend_tile_load_f32(%arg0, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %4 = llvm.call @ascend_tile_load_f32(%arg1, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %5 = llvm.call @ascend_tile_silu_f32(%3, %3, %1, %2) : (i32, i32, i32, i32) -> i32
    %6 = llvm.call @ascend_tile_mul_f32(%5, %4, %1, %2) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %6, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#
    }

    #[test]
    fn test_musa_silu_mul_fusion() {
        let mu = convert_mlir_to_musa(silu_mul_mlir()).unwrap();
        assert!(mu.contains("__global__ void gated_mlp"), "missing kernel:\n{}", mu);
        // Must contain the fused SiLUMul pattern (converted from CUDA to MUSA)
        assert!(mu.contains("SiLUMul fusion"), "must have SiLUMul fusion comment:\n{}", mu);
        assert!(mu.contains("_silu_v"), "must have silu temp var:\n{}", mu);
        assert!(mu.contains("expf(-_silu_v)"), "must compute silu:\n{}", mu);
        // Must be MUSA-ified
        assert!(mu.contains("musa_runtime.h"), "must have MUSA header:\n{}", mu);
        assert!(!mu.contains("cuda_runtime.h"), "must NOT have CUDA header:\n{}", mu);
    }
}
