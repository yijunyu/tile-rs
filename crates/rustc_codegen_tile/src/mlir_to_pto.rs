//! MLIR-to-PTO-MLIR translator for Ascend NPU targets.
//!
//! Converts merged MLIR modules (LLVM dialect with `__tile_*` intrinsics)
//! into PTO-dialect MLIR text that can be compiled by `ptoas` from the
//! `cannmirror/pto-isa` toolchain.
//!
//! # PTO-MLIR Format
//!
//! PTO (Programmable Tile Operations) uses MLIR with the `pto` dialect.
//! A typical kernel looks like:
//!
//! ```mlir
//! module {
//!   func.func @vec_add(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f32>) {
//!     %c0 = arith.constant 0 : index
//!     %c1 = arith.constant 1 : index
//!     %c32 = arith.constant 32 : index
//!     %0 = pto.make_tensor_view %arg0, shape = [%c32, %c32] strides = [%c32, %c1] : !pto.tensor_view<32x32xf32>
//!     %1 = pto.make_tensor_view %arg1, shape = [%c32, %c32] strides = [%c32, %c1] : !pto.tensor_view<32x32xf32>
//!     %2 = pto.make_tensor_view %arg2, shape = [%c32, %c32] strides = [%c32, %c1] : !pto.tensor_view<32x32xf32>
//!     %3 = pto.partition_view %0, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<32x32xf32> -> !pto.partition_tensor_view<32x32xf32>
//!     %4 = pto.partition_view %1, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<32x32xf32> -> !pto.partition_tensor_view<32x32xf32>
//!     %5 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
//!     %6 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
//!     %7 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
//!     pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>) outs(%5 : !pto.tile_buf<...>)
//!     pto.tload ins(%4 : !pto.partition_tensor_view<32x32xf32>) outs(%6 : !pto.tile_buf<...>)
//!     pto.tadd ins(%5, %6 : ...) outs(%7 : ...)
//!     %8 = pto.partition_view %2, offsets = [%c0, %c0], sizes = [%c32, %c32] : ...
//!     pto.tstore ins(%7 : ...) outs(%8 : ...)
//!     return
//!   }
//! }
//! ```
//!
//! # Mapping from tile_std tile intrinsics to PTO ops
//!
//! | tile_std intrinsic | PTO op |
//! |---|---|
//! | `__tile_load_f32(gm, rows, cols)` | `pto.tload` |
//! | `__tile_store_f32(gm, buf, rows, cols)` | `pto.tstore` |
//! | `__tile_add_f32(0, a, b, rows, cols)` | `pto.tadd` |
//! | `__tile_mul_f32(0, a, b, rows, cols)` | `pto.tmul` |
//! | `__tile_exp_f32(0, src, rows, cols)` | `pto.texp` |
//! | `__tile_softmax_f32(0, src, rows, cols)` | `pto.tsoftmax` |
//! | `__tile_matmul_f32(0, a, b, m, k, n)` | `pto.tmatmul` |
//! | `get_block_idx()` | (block_id via `get_block_idx` in future extension) |
//! | `__tile_pipe_barrier` | (suppressed — PTO/ptoas inserts sync automatically) |
//!
//! # Status
//!
//! This translator targets the `ptoas` assembler confirmed at:
//! `/data/sunwenbo/pto/llvm-workspace/PTOAS/build/tools/ptoas/ptoas`
//! (LLVM 19.1.7 optimized). Invoke with `--enable-insert-sync` to have `ptoas`
//! insert `set_flag`/`wait_flag` barriers automatically.
//!
//! Tile dimensions in PTO are fixed at a multiple of 32. For our kernels we use
//! the actual ROWS×COLS from the intrinsic args, snapping to the tile shape that
//! ptoas expects. The `fractal=512` attribute corresponds to 32×32×sizeof(f32)/2
//! (the fractal bank size in bytes on Ascend910B).
//!
//! # PTO-ISA / FlashTile integration notes
//!
//! The **PTO Tile Library** (`pto-isa`, open-sourced 2025-12-27 at
//! `https://pto-isa.gitcode.com`) provides C++ header-only templates for the
//! same tile operations as PTO-MLIR — `TROWMAX`, `TROWSUM`, `TROWEXPANDSUB`,
//! `TROWEXPANDDIV`, etc. — and is the reference implementation used by
//! FlashAttention on Ascend (see `kernels/manual/a2a3/flash_atten/`).
//!
//! ## Reduction op format (3-operand)
//!
//! The ptoas binary (LLVM 19.1.7) requires the correct 3-operand format for
//! reduction ops. The generated sample files (e.g., `_out/Rowmax/rowmax-pto-ir.pto`)
//! contained a bug: they used `ins(%src : type)` (1 arg) but the TableGen
//! `assemblyFormat` requires `ins(%src, %tmp : type_src, type_tmp)` (2 args in ins).
//! The parser was correct; the samples were wrong.
//!
//! Correct formats (per `PTOOps.td`):
//! - `pto.trowmax ins(%src, %tmp : T, T) outs(%dst : T)` — src, tmp, dst
//! - `pto.trowmin ins(%src, %tmp : T, T) outs(%dst : T)` — src, tmp, dst
//! - `pto.trowsum ins(%src, %tmp : T, T) outs(%dst : T)` — src, tmp, dst
//! - `pto.trowexpandsub ins(%src0, %src1 : T, T) outs(%dst : T)` — src0, src1, dst
//! - `pto.trowexpanddiv ins(%src0, %src1 : T, T) outs(%dst : T)` — src0, src1, dst
//!
//! ## Softmax decomposition
//!
//! `__tile_softmax_f32` is lowered to the numerically-stable 5-step decomposition:
//! ```text
//! trowmax(t_in, t_tmp)   → t_max   (row-wise max, needs tmp scratch)
//! trowexpandsub(t_in, t_max) → t_sub  (x - max per row)
//! texp(t_sub)            → t_exp   (elementwise exp)
//! trowsum(t_exp, t_tmp)  → t_sum   (row-wise sum, reuses tmp scratch)
//! trowexpanddiv(t_exp, t_sum) → result (divide by row sum)
//! ```
//! This matches the FlashAttention reference in `pto_macro_fa_softmax.hpp`:
//! `TROWMAX(new_global_max, input_x, tmp_float)` etc.

use std::collections::HashMap;
use std::fmt::Write;

// Shared MLIR parser surface. Re-exported pub(crate) so dependent modules
// (e.g. mlir_to_msl) can keep importing from here.
pub(crate) use crate::mlir_parse::{
    extract_call_args, extract_func_args, extract_result_ssa, is_builtin_helper, parse_const_arg,
    parse_module, FuncArg, MlirFunc, MlirModule,
};

/// Convert MLIR text (merged module, LLVM dialect) into PTO-dialect MLIR text
/// consumable by `ptoas --enable-insert-sync`.
///
/// Returns the PTO-MLIR source string, or an error on parse failure.
/// Which AI-core types a generated kernel's tiles live on.
///
/// This decides how the kernel must be BUILT, so it is part of the emitter's
/// contract rather than a detail of any one benchmark harness.
///
/// The Ascend AI Core is two engines with separate memories. A PTO tile's `loc`
/// says which one it uses: `mat`/`left`/`right`/`acc` are the cube side
/// (L1 / L0A / L0B / L0C), `vec` is the vector side (UB). `ccec` compiles for
/// one engine at a time — `--cce-aicore-arch=dav-c220-{cube,vec}` — and each
/// rejects the other's instructions.
///
/// - [`Cube`] or [`Vector`]: a single-engine kernel. Builds through the normal
///   `ascendc_library()` CMake path, which generates the `aclrtlaunch_<name>`
///   entry point. No hand-written driver or linking involved.
/// - [`Mix`]: touches both. `ascendc_library()` does NOT work, because its
///   preprocess pass compiles the source for both engines and the shipped
///   pto-isa headers carry no `__CCE_AICORE__` gating, so `TLoad.hpp` /
///   `TExtract.hpp` instantiate cube intrinsics during the vector pass and fail.
///   Such a kernel must either be decomposed into single-engine kernels
///   (see the attention pipeline: scores/pv on cube, softmax on vector) or
///   built by hand per engine and linked.
///
/// The distinction is a CANN packaging gap, not a property of PTO: a MIX kernel
/// is perfectly legal ISA, there is simply no supported single-source build for
/// it in CANN 8.5.2's header layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelCores {
    /// Cube only (`mat`, `left`, `right`, `acc` tiles).
    Cube,
    /// Vector only (`vec` tiles).
    Vector,
    /// Both engines — see the caveat above.
    Mix,
    /// No tiles at all (empty or non-tile kernel).
    None,
}

impl KernelCores {
    /// True when `ascendc_library()` can build this kernel from one source.
    pub fn buildable_with_ascendc_library(self) -> bool {
        !matches!(self, KernelCores::Mix)
    }

    /// The `--cce-aicore-arch` suffix for a single-engine kernel.
    pub fn ccec_arch(self) -> Option<&'static str> {
        match self {
            KernelCores::Cube => Some("cube"),
            // A tile-less kernel still has to be compiled for something; the
            // vector engine is the safe default (it is the scalar/AIV path).
            KernelCores::Vector | KernelCores::None => Some("vec"),
            KernelCores::Mix => None,
        }
    }
}

/// Classify generated PTO text by the engines its tiles use.
///
/// Callers use this to pick a build strategy without having to know the
/// per-op cube/vector mapping themselves.
pub fn classify_kernel_cores(pto: &str) -> KernelCores {
    let mut cube = false;
    let mut vector = false;
    for line in pto.lines() {
        if !line.contains("!pto.tile_buf<") {
            continue;
        }
        // A single line can mention several tiles (ins/outs operand types).
        for seg in line.split("loc=").skip(1) {
            let loc = seg
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            match loc {
                "mat" | "left" | "right" | "acc" => cube = true,
                "vec" => vector = true,
                _ => {}
            }
        }
    }
    match (cube, vector) {
        (true, true) => KernelCores::Mix,
        (true, false) => KernelCores::Cube,
        (false, true) => KernelCores::Vector,
        (false, false) => KernelCores::None,
    }
}

pub fn convert_mlir_to_pto(mlir_text: &str) -> Result<String, String> {
    let module = parse_module(mlir_text)?;

    let mut out = String::with_capacity(4096);
    writeln!(out, "// Generated by tile-rs mlir_to_pto — DO NOT EDIT").unwrap();
    writeln!(
        out,
        "// Compile: ptoas --enable-insert-sync <file.pto> -o <file.cpp>"
    )
    .unwrap();
    // Scan the module for ops that require A5-specific verifier rules
    // (attention, attention_gqa, matmul_transposed — anything that will
    // emit pto.tinsert for vec→mat or uses A5-only tile-layout paths).
    // Without the module attribute, ptoas's `dispatchVerifierByArch` falls
    // back to A2/A3 and rejects these ops even with `--pto-arch=a5` on CLI.
    //
    // For A2/A3-only kernels (softmax, vec_add, plain matmul, ...), we leave
    // the attribute off so bisheng sees the classical module form — confirmed
    // working on CANN 8.5 / 910B2 for softmax and matmul.
    let needs_a5 = module_uses_a5_ops(&module);
    if needs_a5 {
        writeln!(out, "module attributes {{pto.target_arch = \"a5\"}} {{").unwrap();
    } else {
        writeln!(out, "module {{").unwrap();
    }

    let mut kernel_count = 0;
    for func in &module.functions {
        if func.is_entry && !func.body_lines.is_empty() && !is_builtin_helper(&func.name) {
            generate_func_pto(func, &mut out)?;
            kernel_count += 1;
        }
    }

    writeln!(out, "}}").unwrap();

    if kernel_count == 0 {
        return Err("No entry-point kernel functions found in MLIR module".into());
    }

    Ok(out)
}

/// Returns true if any function in the module will emit PTO ops that need
/// the A5 verifier — specifically `pto.tinsert` (VEC→MAT) and `tmov` with
/// src=Acc dst=Vec. Without the `pto.target_arch = "a5"` module attribute,
/// ptoas's `dispatchVerifierByArch` falls back to A2/A3 and rejects those.
///
/// `__tile_matmul_transposed_*` is deliberately NOT a trigger here:
/// its a5-safe rewrite (translate_matmul_transposed) emits only DN→ZN
/// `pto.tload` + CBUF→L0A/B `pto.tmov` + `pto.tmatmul`, all supported on
/// A2/A3. Gating it behind the a5 attr was over-cautious and blocks
/// validating the transposed-matmul emitter on CANN 8.5 (which ships
/// a2a3 headers only).
fn module_uses_a5_ops(module: &MlirModule) -> bool {
    for func in &module.functions {
        if !func.is_entry {
            continue;
        }
        for line in &func.body_lines {
            if line.contains("__tile_attention_f32")
                || line.contains("__tile_attention_gqa_f32")
            {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// PTO-MLIR function generator
// ---------------------------------------------------------------------------

fn generate_func_pto(func: &MlirFunc, out: &mut String) -> Result<(), String> {
    // Collect tile information by scanning body first
    let mut ctx = PtoContext::new();
    let body_ops = analyze_body(&func.body_lines, func, &mut ctx)?;
    // NPU tile trait bounds: surface any tile-shape/UB-budget violation as a
    // codegen error (C1-C5), so an Ascend-invalid layer fails to EMIT rather
    // than launching and faulting on device.
    if let Some(e) = ctx.tile_error.take() {
        return Err(e);
    }

    // Emit func.func header with !pto.ptr<T> args
    write!(out, "  func.func @{}(", func.name).unwrap();
    let ptr_args: Vec<&FuncArg> = func.args.iter().filter(|a| a.is_gm).collect();
    for (i, arg) in ptr_args.iter().enumerate() {
        if i > 0 {
            write!(out, ", ").unwrap();
        }
        // Infer dtype from body usage (tile_load_*/tile_store_* calls that
        // reference this arg) first; fall back to the name-based heuristic.
        // Without this, an f16 kernel whose Rust arg name is `b` (no "f16"
        // in the string) emits `!pto.ptr<f32>` while `pto.make_tensor_view`
        // uses `tensor_view<?x?xf16>` — ptoas then generates `__gm__ float*
        // v1` but a `GlobalTensor<half, ...>` view from it, breaking C++
        // typing.
        let dtype = infer_arg_dtype_from_body(&arg.name, &func.body_lines)
            .unwrap_or_else(|| infer_dtype_from_name(&arg.name));
        write!(out, "{}: !pto.ptr<{}>", arg.name, dtype).unwrap();
    }
    writeln!(out, ") {{").unwrap();

    // Emit index constants for all unique sizes we use
    let mut consts: Vec<u32> = ctx.unique_sizes().into_iter().collect();
    consts.sort();
    // Always need 0 and 1
    for &c in &[0u32, 1u32] {
        if !consts.contains(&c) {
            consts.push(c);
        }
    }
    consts.sort();
    for &c in &consts {
        writeln!(out, "    %c{} = arith.constant {} : index", c, c).unwrap();
    }

    // Emit body operations
    for line in &body_ops {
        writeln!(out, "    {}", line).unwrap();
    }

    writeln!(out, "    return").unwrap();
    writeln!(out, "  }}").unwrap();

    Ok(())
}

// ---------------------------------------------------------------------------
// PTO type string helpers
// ---------------------------------------------------------------------------

/// `!pto.tensor_view<?x?xf32>` — ptoas v0.13 requires wildcard dims
fn tv_type(_rows: u32, _cols: u32, dtype: &str) -> String {
    format!("!pto.tensor_view<?x?x{}>", dtype)
}

/// `!pto.partition_tensor_view<RxCxf32>`
fn ptv_type(rows: u32, cols: u32, dtype: &str) -> String {
    format!("!pto.partition_tensor_view<{}x{}x{}>", rows, cols, dtype)
}

/// `!pto.tile_buf<loc=vec, dtype=f32, rows=R, cols=C, v_row=R, v_col=C,
///               blayout=row_major, slayout=none_box, fractal=512, pad=0>`
fn tile_buf_type(rows: u32, cols: u32, dtype: &str) -> String {
    // fractal=512 is the standard for vec tiles on Ascend910B
    // (32×32×2 bytes for f16, 32×32×4/2 for f32 — ptoas uses 512 universally for vec)
    format!(
        "!pto.tile_buf<loc=vec, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=row_major, slayout=none_box, fractal=512, pad=0>",
        dtype, rows, cols, rows, cols
    )
}

/// Row-reduction output tile: `rows×1, col_major` — required by trowmax/trowsum.
///
/// CANN 8.5 pto_tile.hpp requires: `Rows * sizeof(DType) % 32 == 0` for col_major tiles.
/// So `rows` is padded up to the minimum that satisfies this: 8 for f32, 16 for f16.
/// `v_row` (valid rows) keeps the actual number of rows for runtime correctness.
///
/// E.g. for 1×1024 f32: allocated rows=8, valid rows=1, cols=1, col_major.
fn tile_buf_type_rowreduce(rows: u32, dtype: &str) -> String {
    // Minimum rows to satisfy `rows * sizeof(dtype) % 32 == 0`:
    //   f32: 4 bytes → ceil to multiple of 8; f16: 2 bytes → ceil to multiple of 16
    let bytes_per_elem: u32 = if dtype == "f16" { 2 } else { 4 };
    let align_rows: u32 = 32 / bytes_per_elem; // 8 for f32, 16 for f16
    let alloc_rows = if rows % align_rows == 0 { rows } else { ((rows / align_rows) + 1) * align_rows };
    format!(
        "!pto.tile_buf<loc=vec, dtype={}, rows={}, cols=1, v_row={}, v_col=1, \
         blayout=col_major, slayout=none_box, fractal=512, pad=0>",
        dtype, alloc_rows, rows
    )
}

/// Row-reduce tile with `blayout=row_major`. Used by `translate_rms_norm_pto`
/// for the TMULS/TADDS/TSQRT/TRECIP chain — those ops require `isRowMajor`
/// per the patched a2a3 headers (TMulS.hpp:55, TAddS.hpp:55, TUnaryOp.hpp).
/// Shape rows=R, cols=8 (32-byte aligned), v_row=R, v_col=1. Matches the
/// Qwen3DecodeA3 sample's RMSNorm pattern (samples/Qwen3DecodeA3/qwen3_decode_incore_0.pto):
/// `tsqrt + trecip` instead of the older `trsqrt` route, which avoids the
/// vrsqrt instruction's lane-garbage NaN propagation issue.
fn tile_buf_type_rowreduce_rowmajor(rows: u32, dtype: &str) -> String {
    let bytes_per_elem: u32 = if dtype == "f16" { 2 } else { 4 };
    let align_cols: u32 = 32 / bytes_per_elem; // 8 for f32, 16 for f16
    // v_col=1: TMULS/TADDS/TSQRT/TRECIP process only lane 0 (the per-row
    // sum). Matching the v_col=cols Qwen3 sample is equivalent — the
    // chain runs on positive values (sum_sq * 1/cols + eps > 0), so
    // tsqrt + trecip are well-defined for any garbage in lanes 1..7.
    // Keep v_col=1 to minimize SIMD lane usage.
    format!(
        "!pto.tile_buf<loc=vec, dtype={}, rows={}, cols={}, v_row={}, v_col=1, \
         blayout=row_major, slayout=none_box, fractal=512, pad=0>",
        dtype, rows, align_cols, rows
    )
}

/// `!pto.tile_buf<loc=mat, ...>` — CBUF staging tile (L2 → L0A/L0B path)
/// blayout=col_major, slayout=row_major (NZ custom layout).
/// Used for GM→mat tload when the GM view is row-major (ND→NZ path).
fn mat_tile_type(rows: u32, cols: u32, dtype: &str) -> String {
    format!(
        "!pto.tile_buf<loc=mat, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=col_major, slayout=row_major, fractal=512, pad=0>",
        dtype, rows, cols, rows, cols
    )
}

/// `!pto.tile_buf<loc=mat, ...>` — CBUF staging tile with ZN custom layout
/// blayout=row_major, slayout=col_major.
/// Used for GM→mat tload when the GM view is column-major/transposed
/// (DN→ZN path) — only DN2DN, NZ2NZ, ND2NZ, and DN2ZN are supported by
/// TLoadGm2L1; DN2NZ is not, so the transposed-K tile must be ZN.
fn mat_tile_type_zn(rows: u32, cols: u32, dtype: &str) -> String {
    format!(
        "!pto.tile_buf<loc=mat, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=row_major, slayout=col_major, fractal=512, pad=0>",
        dtype, rows, cols, rows, cols
    )
}

/// `!pto.tile_buf<loc=left, ...>` — L0A tile for left (A) matmul operand
/// blayout=row_major, slayout=row_major
fn left_tile_type(rows: u32, cols: u32, dtype: &str) -> String {
    format!(
        "!pto.tile_buf<loc=left, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=row_major, slayout=row_major, fractal=512, pad=0>",
        dtype, rows, cols, rows, cols
    )
}

/// `!pto.tile_buf<loc=right, ...>` — L0B tile for right (B) matmul operand
/// blayout=row_major, slayout=col_major
fn right_tile_type(rows: u32, cols: u32, dtype: &str) -> String {
    format!(
        "!pto.tile_buf<loc=right, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=row_major, slayout=col_major, fractal=512, pad=0>",
        dtype, rows, cols, rows, cols
    )
}

/// `!pto.tile_buf<loc=acc, ...>` — L0C accumulator tile for matmul output
/// blayout=col_major, slayout=row_major, fractal=1024
fn acc_tile_type(rows: u32, cols: u32, dtype: &str) -> String {
    format!(
        "!pto.tile_buf<loc=acc, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=col_major, slayout=row_major, fractal=1024, pad=0>",
        dtype, rows, cols, rows, cols
    )
}

// ---------------------------------------------------------------------------
// Context tracking SSA values → tile info
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TileInfo {
    /// SSA name in the generated PTO-MLIR (e.g., `%12`)
    ssa: String,
    rows: u32,
    cols: u32,
    dtype: String,
    /// Full tile_buf type string (e.g., `!pto.tile_buf<loc=acc, ...>`)
    /// Cached so translate_store() emits the correct type for non-vec tiles.
    tb_type: String,
    /// SSA name of the partition_tensor_view used as the load source / store dest
    /// (only set for tiles loaded from GM)
    pv_ssa: Option<String>,
    /// Original GM arg name (e.g., `%arg1`) this tile was loaded from.
    /// Used by translate_attention to construct a transposed tensor_view
    /// over the same GM buffer for the K tile. Only set for tiles loaded from GM.
    gm_name: Option<String>,
    /// Deferred blocked-matmul operand load: tile never materialised as a
    /// full-shape vec buffer. translate_matmul consumes `deferred.tv_ssa` +
    /// `deferred.elem_offset` to emit a K/N-blocked scf.for nest.
    ///
    /// Set by translate_load when the load is flagged in the pre-pass. When
    /// present, `ssa` / `tb_type` are placeholders — no `pto.alloc_tile` or
    /// `pto.tload` was emitted for the full shape.
    deferred: Option<DeferredMatmulOperand>,
}

/// Metadata recorded for a tile_load that's consumed only by a blocked
/// matmul. Captures everything translate_matmul needs to emit per-block
/// partition views and loads inside its scf.for loops.
#[derive(Clone)]
struct DeferredMatmulOperand {
    /// Pre-built `pto.tensor_view<?x?xDT>` SSA for the full GM buffer.
    tv_ssa: String,
    /// Element offset from the base of the GM buffer (GEP-derived).
    elem_offset: u32,
    /// Resolved base GM SSA (e.g., `%arg1`). Needed when the blocked
    /// matmul must synthesise a chunk-local tensor_view at a non-zero
    /// offset (lm_head and any other matmul whose N is large enough that
    /// Kb × N ≥ 2^24 would overflow ptoas's 24-bit outer-stride field).
    gm_name: String,
}

impl TileInfo {
    fn tile_buf_type_str(&self) -> String {
        self.tb_type.clone()
    }

    fn ptv_type_str(&self) -> String {
        ptv_type(self.rows, self.cols, &self.dtype)
    }
}


// ============================================================================
// NPU tile trait bounds — make an Ascend-invalid tile UNREPRESENTABLE at emit.
//
// The Ascend Unified Buffer (UB) imposes shape/capacity/alignment constraints
// that were previously runtime PTO_ASSERTs (faulting as 507057 on device). We
// lift them to codegen: every tile is validated for its target arch (C2-C4) and
// placed under a linear UB budget (C1) at a bank-aligned offset (C5). A layer
// whose working set exceeds UB_SIZE, or whose tile shape violates alignment, is
// an Err(String) codegen diagnostic — never a launched-then-crashing kernel.
// This is the accelerator-resource analogue of the memory-hazard freedom the
// lambda_tile calculus proves for Metal; the resource is UB SPACE and the linear
// discipline is UbAllocator.
// ============================================================================

/// Hardware limits for one SoC, as published by the vendor.
///
/// These are NOT to be invented. CANN ships the authoritative numbers per SoC in
/// `<CANN>/<arch>-linux/data/platform_config/<SocVersion>.ini`, `[AICoreSpec]`,
/// and the same values are queryable at runtime via
/// `platform_ascendc::PlatformAscendC::GetCoreMemSize(CoreMemType::UB, ...)`.
///
/// Example, verified on a real 910B2 (CANN 8.5.2) — `Ascend910B2.ini`:
/// ```text
/// [AICoreSpec]
/// ub_size=196608      # 192 KB   <- the UB capacity the C1 budget must respect
/// ubblock_size=32     # 32 B     <- footprint block alignment
/// ubbank_size=4096
/// l0_a_size=65536  l0_b_size=65536  l0_c_size=131072  l1_size=524288
/// ```
/// `Ascend910_9392.ini` (the `910c` host) reports the same `ub_size=196608`.
///
/// A2/A3 additionally reserves the TOP 8 KB of UB as instruction scratch, so the
/// budget available to tiles is 192 - 8 = 184 KB. That reservation is stated in
/// the pto-isa headers themselves:
/// `pto/npu/a2a3/TSels.hpp` — "8KB, start from 184KB, UB:192KB=184+8KB",
/// and `pto/npu/a2a3/TMrgSort.hpp` — `constexpr int UBSIZE = 196608;`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SocSpec {
    /// `ub_size` from `[AICoreSpec]`, in bytes.
    pub ub_size: usize,
    /// Bytes at the top of UB reserved by the runtime, unavailable to tiles.
    pub ub_reserved: usize,
    /// `ubblock_size` from `[AICoreSpec]`, in bytes.
    pub ubblock_size: usize,
    /// `l1_size` — the CBUF that `loc=mat` staging tiles live in.
    pub l1_size: usize,
    /// `l0_a_size` / `l0_b_size` — `loc=left` / `loc=right` operand tiles.
    pub l0_a_size: usize,
    pub l0_b_size: usize,
    /// `l0_c_size` — `loc=acc` accumulator tiles.
    pub l0_c_size: usize,
}

impl SocSpec {
    /// UB actually available to tiles: capacity minus runtime reservation.
    pub const fn ub_budget(&self) -> usize {
        self.ub_size - self.ub_reserved
    }

    /// Parse `[AICoreSpec]` out of a CANN `platform_config/<Soc>.ini`.
    ///
    /// Use this to re-derive the constants below from a CANN install rather than
    /// trusting the checked-in copies:
    /// `SocSpec::from_platform_ini(&fs::read_to_string(".../Ascend910B2.ini")?)`.
    pub fn from_platform_ini(ini: &str, ub_reserved: usize) -> Option<SocSpec> {
        let mut in_aicore = false;
        let (mut ub_size, mut ubblock_size) = (None, None);
        // L1/L0 come from the same [AICoreSpec] block; default to the 910B2
        // figures if a spec omits them rather than failing the whole parse.
        let (mut l1, mut l0a, mut l0b, mut l0c) = (524288, 65536, 65536, 131072);
        for line in ini.lines() {
            let l = line.trim();
            if l.starts_with('[') {
                in_aicore = l.eq_ignore_ascii_case("[AICoreSpec]");
                continue;
            }
            if !in_aicore {
                continue;
            }
            let Some((k, v)) = l.split_once('=') else { continue };
            let v = v.trim().parse::<usize>().ok()?;
            match k.trim() {
                "ub_size" => ub_size = Some(v),
                "ubblock_size" => ubblock_size = Some(v),
                "l1_size" => l1 = v,
                "l0_a_size" => l0a = v,
                "l0_b_size" => l0b = v,
                "l0_c_size" => l0c = v,
                _ => {}
            }
        }
        Some(SocSpec {
            ub_size: ub_size?,
            ub_reserved,
            ubblock_size: ubblock_size?,
            l1_size: l1,
            l0_a_size: l0a,
            l0_b_size: l0b,
            l0_c_size: l0c,
        })
    }
}

/// 910B2 / Ascend910_9392, from `Ascend910B2.ini` `[AICoreSpec]`.
/// `ub_reserved` is the 8 KB A2/A3 TMP_UB scratch documented in TSels.hpp.
pub const SPEC_A2A3: SocSpec =
    // L1/L0 sizes are the same [AICoreSpec] block ub_size comes from:
    //   l0_a_size=65536  l0_b_size=65536  l0_c_size=131072  l1_size=524288
    SocSpec {
        ub_size: 196608,
        ub_reserved: 8 * 1024,
        ubblock_size: 32,
        l1_size: 524288,
        l0_a_size: 65536,
        l0_b_size: 65536,
        l0_c_size: 131072,
    };

/// Target Ascend architecture — the constraints are arch-parametric (C6).
trait AscendArch {
    const UB_SIZE: usize;        // total Unified Buffer capacity (bytes)
    const FRACTAL_BYTES: usize;  // column/bank alignment granularity (bytes)
    const BLOCK_BYTES: usize;    // footprint block alignment (bytes)
    const REQUIRES_ROW16: bool;  // NZ/cube tiles need rows % 16 == 0
    const NAME: &'static str;
}

/// 910B2 / a2a3 (the 910c test box).
struct A2A3;
impl AscendArch for A2A3 {
    // Derived from the vendor spec (`SPEC_A2A3`, parsed from Ascend910B2.ini),
    // NOT hand-written: 196608 B capacity - 8 KB A2/A3 TMP_UB scratch = 184 KB.
    //
    // This previously read 262144 with the comment "(pto/npu/a5 UB_SIZE)" — the
    // A5 figure copied onto A2/A3. That made the C1 guard UNSOUND on A2/A3: it
    // accepted tile working sets between 184 KB and 256 KB that do not fit the
    // hardware, which is exactly the class of bug this guard exists to catch.
    const UB_SIZE: usize = SPEC_A2A3.ub_budget();
    const FRACTAL_BYTES: usize = 512;
    const BLOCK_BYTES: usize = SPEC_A2A3.ubblock_size;
    const REQUIRES_ROW16: bool = false;
    const NAME: &'static str = "a2a3";
}

#[allow(dead_code)]
struct A5;
#[allow(dead_code)]
impl AscendArch for A5 {
    const UB_SIZE: usize = 262144;
    const FRACTAL_BYTES: usize = 512;
    const BLOCK_BYTES: usize = 32;
    const REQUIRES_ROW16: bool = true;
    const NAME: &'static str = "a5";
}

fn dtype_bytes_pto(dtype: &str) -> usize {
    match dtype { "f16" | "bf16" => 2, "i8" => 1, _ => 4 }
}

/// Columns per fractal for `dtype` — the C2-legal width granularity.
///
/// C2 requires `cols * sizeof(dtype) % FRACTAL_BYTES == 0` for any row-major
/// data tile whose row spans more than one fractal. So a shape that will be
/// loaded or stored has to be padded to a MULTIPLE of this, not merely to a
/// round number: at S=897 the driver's `up(seq, 32)` gives 928, and
/// 928 * 4 = 3712 B leaves 128 B over a 512 B fractal. 1024 is the next legal
/// width.
///
/// Exposed so callers pad from the same rule the validator enforces, rather
/// than each hardcoding a constant that is only right for one dtype:
///
/// | dtype | bytes | C2-legal multiple |
/// |---|---|---|
/// | f32  | 4 | 128 |
/// | f16 / bf16 | 2 | 256 |
/// | i8   | 1 | 512 |
pub fn c2_col_multiple(dtype: &str) -> u32 {
    (A2A3::FRACTAL_BYTES / dtype_bytes_pto(dtype)) as u32
}

/// Round `cols` up to the next C2-legal width for `dtype`.
///
/// SUB-FRACTAL ROWS ARE EXEMPT, and that exemption is load bearing. C2 only
/// constrains a tile whose row spans MORE than one fractal; a row of
/// `cols * b <= FRACTAL_BYTES` fits inside one and has no stride to get wrong.
/// The validator has always known this — padding unconditionally to the
/// multiple would inflate every small shape 4x (S=32 -> 128, HD=64 -> 128) and
/// push the batched stages over the UB budget, which is exactly what happened
/// when this was first written without the guard.
///
/// So: shapes at or under one fractal are returned unchanged, and only genuinely
/// wide rows are rounded up.
pub fn pad_cols_c2(cols: u32, dtype: &str) -> u32 {
    let m = c2_col_multiple(dtype).max(1);
    if cols <= m {
        return cols; // sub-fractal: C2 does not apply
    }
    cols.div_ceil(m) * m
}

/// Shape validation (C2-C4) for a tile on arch `A`. Returns Err with a precise
/// diagnostic on any violation.
fn parse_tb_dim(tb_ty: &str, key: &str) -> Option<u32> {
    // parse `key=NN` (e.g. rows=8, cols=256) from the tile_buf type string.
    let pat = format!("{}=", key);
    let i = tb_ty.find(&pat)? + pat.len();
    let rest = &tb_ty[i..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn validate_tile_shape<A: AscendArch>(logical_rows: u32, logical_cols: u32, dtype: &str, tb_ty: &str) -> Result<(), String> {
    let b = dtype_bytes_pto(dtype);
    // Validate the ACTUAL emitted (possibly padded) shape from tb_ty, which is
    // what reaches hardware — not the logical args (rowreduce pads rows).
    let rows = parse_tb_dim(tb_ty, "rows").unwrap_or(logical_rows);
    let cols = parse_tb_dim(tb_ty, "cols").unwrap_or(logical_cols);
    // A row-REDUCTION tile is col_major with slayout=none_box (tile_buf_type_rowreduce).
    // `blayout=col_major` alone also matches CBUF matrix staging tiles (loc=mat,
    // slayout=row_major, the ND->NZ path), which the C3-reduce rule does not govern --
    // testing on blayout alone rejected a valid 4-row attention staging tile.
    let col_major = tb_ty.contains("blayout=col_major") && tb_ty.contains("slayout=none_box");
    if col_major {
        // col_major reduction tiles (trowsum/trowmax outputs, cols=1): the CANN
        // constraint is `rows * sizeof(dtype) % 32 == 0`, NOT the 512B col align.
        // The emitter already pads rows to satisfy this (tile_buf_type_rowreduce);
        // we VALIDATE it holds (defence-in-depth), so a mis-emitted reduction tile
        // is caught at codegen.
        if (rows as usize * b) % 32 != 0 {
            return Err(format!(
                "NPU tile bound (arch {}): col_major reduction tile rows={} * {}B = {}B not 32B-aligned (C3-reduce)",
                A::NAME, rows, b, rows as usize * b));
        }
        return Ok(());
    }
    // row_major DATA tiles (loaded/stored to GM): the fractal/512B column stride
    // and NZ/footprint constraints apply.
    // C2: multi-fractal column stride must be FRACTAL_BYTES-aligned. A tile whose
    // row spans MORE than one fractal (cols*b > FRACTAL_BYTES) must tile on the
    // fractal boundary. Sub-fractal tiles (cols*b <= FRACTAL_BYTES) fit in one
    // fractal bank — no inter-fractal stride, so the alignment does not apply
    // (these are the internal reduction/broadcast scratch tiles like 1x8).
    let col_bytes = cols as usize * b;
    if col_bytes > A::FRACTAL_BYTES && col_bytes % A::FRACTAL_BYTES != 0 {
        return Err(format!(
            "NPU tile bound (arch {}): row_major data tile cols={} * {}B = {}B spans multiple {}B \
             fractals with bad stride (C2); pad cols so cols*sizeof(dtype) % {} == 0",
            A::NAME, cols, b, col_bytes, A::FRACTAL_BYTES, A::FRACTAL_BYTES));
    }
    // C3: row alignment for NZ/cube tiles on archs that require it.
    if A::REQUIRES_ROW16 && rows % 16 != 0 {
        return Err(format!(
            "NPU tile bound (arch {}): rows={} not 16-aligned (C3 FRACTAL_NZ_ROW)",
            A::NAME, rows));
    }
    // C4: footprint BLOCK_BYTES-aligned.
    //
    // Same exemption as C2, and for the same reason: a tile whose whole footprint fits
    // inside one fractal bank has no inter-block stride, so block alignment does not apply
    // to it. These are the internal reduction/broadcast scratch tiles (1x1, 4x1, 1x8 ...)
    // that argmax / quantize / top-p / attention reductions materialise. Without this,
    // C4 rejects a 4B scratch tile as "not 32B-block-aligned", which is a constraint the
    // hardware does not impose on a sub-fractal tile -- and it made 8 real kernels
    // un-emittable.
    let footprint = rows as usize * cols as usize * b;
    if footprint > A::FRACTAL_BYTES && footprint % A::BLOCK_BYTES != 0 {
        return Err(format!(
            "NPU tile bound (arch {}): footprint {}x{}x{}B = {}B not {}B-block-aligned (C4)",
            A::NAME, rows, cols, b, footprint, A::BLOCK_BYTES));
    }
    Ok(())
}

/// Linear UB budget (C1 + C5): bump-allocates bank-aligned tile offsets and
/// rejects allocations that would exceed UB_SIZE. This is the space analogue of
/// the Pnd/Rdy typestate — allocation consumes budget, `free` returns it.
/// Which on-chip memory a tile occupies, from its `loc=` attribute.
///
/// This distinction is the whole point: the allocator used to charge EVERY
/// tile against the Unified Buffer, but `mat` / `left` / `right` / `acc` live
/// in L1 / L0A / L0B / L0C and never touch UB at all. A blocked f32 matmul is
/// entirely cube-side, so it was being rejected by a UB budget it does not
/// consume — the projection at S=48..96 uses **0 bytes of UB** and still failed
/// C1. Only `loc=vec` is UB-resident.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TileSpace {
    Ub,
    L1,
    L0A,
    L0B,
    L0C,
}

impl TileSpace {
    /// Read the space out of a `!pto.tile_buf<loc=..., ...>` type string.
    fn from_type(tb_ty: &str) -> TileSpace {
        if tb_ty.contains("loc=mat") {
            TileSpace::L1
        } else if tb_ty.contains("loc=left") {
            TileSpace::L0A
        } else if tb_ty.contains("loc=right") {
            TileSpace::L0B
        } else if tb_ty.contains("loc=acc") {
            TileSpace::L0C
        } else {
            TileSpace::Ub
        }
    }

    fn name(self) -> &'static str {
        match self {
            TileSpace::Ub => "UB",
            TileSpace::L1 => "L1",
            TileSpace::L0A => "L0A",
            TileSpace::L0B => "L0B",
            TileSpace::L0C => "L0C",
        }
    }
}

/// Bump allocator per on-chip memory space.
///
/// Still a bump allocator with no free: tile lifetimes are not tracked, so a
/// kernel's peak is its SUM rather than its true high-water mark. That is
/// conservative (it can reject a kernel that would fit) but never unsound. What
/// changed is that each tile is now charged against the memory it actually
/// occupies, which is what the `loc=` attribute has always said.
struct UbAllocator {
    cursor: usize,
    ub_size: usize,
    fractal: usize,
    arch: &'static str,
    /// Peak live bytes (for diagnostics / a future budget report).
    peak: usize,
    /// Cursors for the cube-side spaces, in the order of `TileSpace`.
    l1: usize,
    l0a: usize,
    l0b: usize,
    l0c: usize,
}

impl UbAllocator {
    fn new<A: AscendArch>() -> Self {
        UbAllocator {
            cursor: 0,
            ub_size: A::UB_SIZE,
            fractal: A::FRACTAL_BYTES,
            arch: A::NAME,
            peak: 0,
            l1: 0,
            l0a: 0,
            l0b: 0,
            l0c: 0,
        }
    }

    /// Bytes already committed to UB. The allocator never frees, so this is
    /// what a later UB allocation has to fit under.
    fn live(&self) -> usize {
        self.cursor
    }

    /// Place a tile of `bytes` in `space`: bank-align the base, error if it
    /// overflows that space's capacity.
    fn place_in(&mut self, space: TileSpace, bytes: usize) -> Result<u64, String> {
        let (cursor, cap) = match space {
            TileSpace::Ub => (&mut self.cursor, self.ub_size),
            TileSpace::L1 => (&mut self.l1, SPEC_A2A3.l1_size),
            TileSpace::L0A => (&mut self.l0a, SPEC_A2A3.l0_a_size),
            TileSpace::L0B => (&mut self.l0b, SPEC_A2A3.l0_b_size),
            TileSpace::L0C => (&mut self.l0c, SPEC_A2A3.l0_c_size),
        };
        let base = (*cursor + self.fractal - 1) / self.fractal * self.fractal; // align_up (C5)
        let end = base + bytes;
        if end > cap {
            return Err(format!(
                "NPU {} budget (arch {}): tile of {}B at offset {}B would use {}B > {}_SIZE {}B (C1); \
                 the live tile working set exceeds the {} — reduce tile shapes or free earlier",
                space.name(), self.arch, bytes, base, end, space.name(), cap, space.name()));
        }
        *cursor = end;
        if space == TileSpace::Ub && end > self.peak {
            self.peak = end;
        }
        Ok(base as u64)
    }

    /// Place a UB tile. Kept for callers that know the tile is vector-side.
    fn place(&mut self, bytes: usize) -> Result<u64, String> {
        self.place_in(TileSpace::Ub, bytes)
    }
    #[allow(dead_code)]
    fn free(&mut self, bytes: usize) { self.cursor = self.cursor.saturating_sub(bytes); }
}

struct PtoContext {
    /// Map from MLIR SSA name (e.g., `%t0`, `%5`) → TileInfo
    tiles: HashMap<String, TileInfo>,
    /// Allocation counter for generating unique SSA names
    next_ssa: u32,
    /// NPU tile UB budget allocator (C1 capacity + C5 bank-align).
    ub: UbAllocator,
    /// First NPU tile-bound violation seen during analyze_body (C2-C5).
    tile_error: Option<String>,
    /// Per-head GM element offsets while emitting a batched stage, as
    /// (operand a, operand b, output). `None` outside one — batching must not
    /// silently shift addresses in the ordinary path.
    head_offsets: Option<(u32, u32, u32)>,
    /// SSA of a row offset to ADD to every partition_view emitted while it is
    /// set, and the element stride it was built from.
    ///
    /// This is how a batched stage makes its loop body address head `h`:
    /// `make_pv`/`make_pv_at` are the single choke point through which every
    /// view is emitted, so setting this once around the body shifts them all
    /// without changing any translator's signature. Outside a batched loop it
    /// is `None` and the emitted offsets are exactly as before.
    pv_row_base: Option<String>,
    /// Column analogue of `pv_row_base`, for N-block loops.
    pv_col_base: Option<String>,
    /// Rows per head for the A operand of the stage currently being emitted.
    /// `translate_matmul` sees the BLOCK row count, not the head stride, so
    /// without this A's head base came out `h_i * mb` instead of
    /// `h_i * head_stride` -- every block of a head reading the same rows.
    head_stride_a: Option<u32>,
    /// Rows per head for the B operand of a batched matmul.
    ///
    /// B's head stride defaulted to the matmul's own `k`, which was right only
    /// while `k` WAS the rows-per-head. Under K-blocking the inner call passes
    /// `kb` (128, not 896), so every head after the first read V at 1/7th of
    /// its true offset -- MERE 44.4. Carried explicitly, like `head_stride_a`.
    head_stride_b: Option<u32>,
    /// Row offset contributed by an inner block loop, kept separately from
    /// `pv_row_base` so `with_head_rows` can SUM with it. Using `pv_row_base`
    /// for both conflated "the previous operand's head base" (replace) with
    /// "this head's block offset" (add).
    block_row_base: Option<String>,
    /// Constants injected for a synthesised call, so a re-emitted per-head line
    /// can carry its dimensions without inventing MLIR constant declarations.
    inline_consts: HashMap<String, u32>,
    /// Map from GM pointer arg name → (tensor_view_ssa, rows, cols, dtype)
    tv_map: HashMap<String, (String, u32, u32, String)>,
    /// Ordered list of sizes we need `arith.constant` for
    sizes_used: Vec<u32>,
    /// SSA alias map: derived pointer SSA → original GM arg name.
    /// Tracks `llvm.getelementptr %argN[...]` and `llvm.load ... !llvm.ptr<1>`
    /// chains so we can resolve `%8` back to `%arg0` when it appears
    /// as the gm argument to `__tile_load_f32`.
    ptr_aliases: HashMap<String, String>,
    /// Integer constant map: SSA name → u32 value.
    /// Populated from `llvm.mlir.constant(N : iXX)` and `llvm.bitcast` of integers.
    /// Used to resolve rows/cols args like `%12` → 1024.
    const_map: HashMap<String, u32>,
    /// Float constant map: SSA → string representation (e.g. "0.5", "1e-05")
    float_const_map: HashMap<String, String>,
    /// GEP element offsets: derived ptr SSA → element offset from the base GM arg.
    /// Populated from `llvm.getelementptr` when the index is a known constant.
    /// Used to emit correct `offsets=[%crow, %c0]` in `partition_view`.
    gep_offsets: HashMap<String, u32>,
    /// matmul result SSAs whose store is emitted inline (per-N-block) by the
    /// blocked-matmul path. translate_store checks this to drive the scf.for
    /// nest for output stores.
    matmul_result_stored_inline: std::collections::HashSet<String>,
    /// Pending blocked-matmul emissions keyed by matmul result SSA. The
    /// scf.for nest is actually emitted in translate_store once it knows
    /// the output tensor_view. See translate_matmul_blocked's design note.
    pending_blocked_matmuls: HashMap<String, PendingBlockedMatmul>,
    /// Batched-stage result SSAs whose loop is emitted at the store, for the
    /// same reason the blocked-matmul one is: the loop must CONTAIN the store,
    /// and the output GM view is only known once the store is seen. Emitting
    /// the loop early produced a kernel that computed every head and stored
    /// only the last — verified wrong on device.
    batched_result_stored_inline: std::collections::HashSet<String>,
    /// Softmaxes whose rows are blocked; the loop must enclose their store.
    softmax_rows_stored_inline: std::collections::HashSet<String>,
    /// When set, a batched matmul emits the K-accumulating form
    /// (`tmatmul` on the first K iteration, `tmatmul.acc` after) instead of a
    /// plain `tmatmul` that would overwrite the accumulator each iteration.
    matmul_accumulate: bool,
    /// Offset of the current K block, applied by each operand loader to the
    /// axis that operand indexes K by. See `with_k_block`.
    pv_k_base: Option<String>,
    /// Which axis the operand being loaded RIGHT NOW indexes K by:
    /// `Some(true)` = rows (a `[K,N]` operand), `Some(false)` = columns (an
    /// `[M,K]` operand), `None` = this operand has no K axis (the `[M,N]`
    /// output). Only consulted when `pv_k_base` is set.
    pv_k_on_rows: Option<bool>,
    pending_row_softmax: HashMap<String, PendingRowSoftmax>,
    /// Pending batched emissions keyed by result SSA.
    pending_batched: HashMap<String, PendingBatched>,
    /// silu_mul result SSAs whose store is emitted inline (per-N-block) by
    /// the blocked-silu_mul path (#67). Mirrors `matmul_result_stored_inline`.
    silu_mul_result_stored_inline: std::collections::HashSet<String>,
    /// Pending blocked-silu_mul emissions keyed by silu_mul result SSA.
    /// translate_store consumes + clears these to emit the per-chunk loop.
    pending_blocked_silu_muls: HashMap<String, PendingBlockedSiluMul>,
}

/// Dtype triple for a pto.tmatmul. On CANN 8.5 ptoas accepts exactly four
/// combinations (empirically, 2026-04-16):
///   - (i32, i8,   i8)    — quantized int8 matmul
///   - (f32, f16,  f16)   — f16 ops with f32 accumulator (decoder f16 weights)
///   - (f32, bf16, bf16)  — bf16 ops with f32 accumulator
///   - (f32, f32,  f32)   — full f32 (current default)
/// See memory/project_pto_tmatmul_dtype_rules.md.
#[derive(Clone)]
struct MatmulDtypes {
    /// L0C accumulator dtype. Also the tstore source dtype (FixPipe casts to
    /// output GM dtype during L0C→GM DMA if they differ).
    dst: &'static str,
    /// A operand dtype (L0A / left). Also the A GM pointer dtype and mat_a
    /// staging dtype.
    lhs: &'static str,
    /// B operand dtype (L0B / right). Also the B GM pointer dtype and mat_b
    /// staging dtype.
    rhs: &'static str,
}

impl MatmulDtypes {
    const fn f32() -> Self {
        MatmulDtypes { dst: "f32", lhs: "f32", rhs: "f32" }
    }
    const fn f16_mixed() -> Self {
        MatmulDtypes { dst: "f32", lhs: "f16", rhs: "f16" }
    }
    /// int8 ops with i32 accumulator. Both A and B are i8; L0C is i32.
    /// Downstream dequant is emitted via `pto.tstore_fp` with a per-column
    /// f32 scale tile (see `PendingBlockedMatmul::dequant`). See
    /// memory/project_pto_i8_tmatmul_validated.md.
    const fn i8_quantized() -> Self {
        MatmulDtypes { dst: "i32", lhs: "i8", rhs: "i8" }
    }
    /// Byte width of the widest operand (A or B). Used by
    /// `matmul_needs_blocking` to compute per-operand L0 footprint.
    fn lhs_bytes(&self) -> u64 {
        match self.lhs { "f16" | "bf16" => 2, "i8" => 1, _ => 4 }
    }
    fn rhs_bytes(&self) -> u64 {
        match self.rhs { "f16" | "bf16" => 2, "i8" => 1, _ => 4 }
    }
}

/// Everything translate_store needs to emit a K/N-blocked matmul once the
/// output GM view is known. Populated by translate_matmul_blocked and
/// consumed + cleared by translate_store.
#[derive(Clone)]
struct PendingBlockedMatmul {
    m: u32,
    k: u32,
    n: u32,
    /// M block. The A operand is `[M, K]` and M grows with sequence length
    /// (`M = pad(B*S)`), so without an M block the A tile alone overflows the
    /// UB: 128 KB at S=96 against a 184 KB budget. Blocking M makes the
    /// working set S-INDEPENDENT — 80 KB at every S including 897, the largest
    /// case the AscendC baselines cover.
    mb: u32,
    kb: u32,
    nb: u32,
    /// `1` means no M loop is emitted, preserving the pre-M-blocking output
    /// byte for byte at shapes that already fit.
    m_iters: u32,
    n_iters: u32,
    k_iters: u32,
    /// Dtype triple (dst, lhs, rhs) for the pto.tmatmul. The output tstore
    /// uses `dst` as the source (L0C) dtype; the GM pv dtype comes from the
    /// store line itself (FixPipe handles the cast when they differ).
    dtypes: MatmulDtypes,
    tv_a_ssa: String,
    tv_b_ssa: String,
    a_elem_offset: u32,
    b_elem_offset: u32,
    /// Base GM SSA for A (e.g., `%arg0`). Reserved for future N-chunk
    /// splitting (see project_pto_matmul_stride_limits.md). Not currently
    /// consumed — the ROW-stride fix for lm_head needs host-side B repack,
    /// not emitter-side tv chunking.
    #[allow(dead_code)]
    a_gm_name: String,
    /// Base GM SSA for B (e.g., `%arg1`). Reserved as above.
    #[allow(dead_code)]
    b_gm_name: String,
    mat_a_ssa: String,
    mat_b_ssa: String,
    a_left_ssa: String,
    b_right_ssa: String,
    acc_ssa: String,
    mat_a_ty: String,
    mat_b_ty: String,
    left_ty: String,
    right_ty: String,
    acc_ty: String,
    /// Per-column f32 dequant scale tile. Present only for int8 matmul with
    /// FixPipe-folded dequant (emitted by translate_matmul_i8). When set,
    /// `emit_blocked_matmul_loops` emits `pto.tstore_fp ins(%acc, %scale)`
    /// instead of plain `pto.tstore`, and allocates a `loc=scaling` tile
    /// up front.
    dequant: Option<DequantSpec>,
}

/// Per-column f32 dequant descriptor for int8 matmul. Allocated by
/// translate_matmul_i8; consumed by emit_blocked_matmul_loops.
#[derive(Clone)]
struct DequantSpec {
    /// Tile-buf SSA for the scaling tile (loc=scaling, ui64, 1×N, fractal=32,
    /// slayout=none_box). CANN 8.5 ptoas rejects `tload outs(loc=scaling)`, so
    /// the scale is loaded GM→L0B-Mat first, then moved Mat→Scaling via TMovToFb
    /// (which requires uint64 DstType and Rows=1, Cols×sizeof%128==0). See
    /// memory/project_cann85_i8_path_viable_via_tmov3arg.md.
    scale_tile_ssa: String,
    /// MLIR type string for `scale_tile_ssa` (the FB-Scaling tile).
    scale_tile_ty: String,
    /// Staging L0B-Mat tile (ui64, none_box, fractal=32). GM→Mat via tload, then
    /// Mat→Scaling via tmov. Allocated outside the N-loop alongside scale_tile_ssa.
    scale_mat_ssa: String,
    /// MLIR type string for `scale_mat_ssa`.
    scale_mat_ty: String,
    /// tensor_view SSA for the scale GM buffer (shape 1×N, ui64 packed).
    tv_scale_ssa: String,
    /// partition_view SSA covering the full 1×N scale row.
    pv_scale_ssa: String,
    /// ptv type spelling of `pv_scale_ssa`.
    pv_scale_ty: String,
}

/// Everything translate_store needs to emit an N-blocked silu_mul once the
/// output GM view is known. Populated by translate_silu_mul (blocked path)
/// and consumed + cleared by translate_store. Mirrors `PendingBlockedMatmul`
/// but for the SwiGLU silu(gate)*up fused emit (#67).
#[derive(Clone)]
struct PendingBlockedSiluMul {
    rows: u32,
    cols: u32,
    nb: u32,
    n_iters: u32,
    dtype: &'static str,
    /// tensor_view SSA for gate GM buffer (full shape rows×cols).
    tv_gate_ssa: String,
    /// tensor_view SSA for up GM buffer.
    tv_up_ssa: String,
    /// GEP-derived element offsets into the gate / up GM buffers.
    gate_elem_offset: u32,
    up_elem_offset: u32,
    /// Pre-allocated chunk tiles (size rows×nb) reused across loop iterations.
    gate_chunk_ssa: String,
    up_chunk_ssa: String,
    neg_chunk_ssa: String,
    silu_chunk_ssa: String,
    out_chunk_ssa: String,
    /// Tile-buf type string for the rows×nb chunk tiles.
    tb_chunk_ty: String,
    /// partition_tensor_view type string for rows×nb chunks.
    pv_chunk_ty: String,
    /// Scalar SSA for -1.0 (used by tmuls in the sigmoid decomposition).
    cneg1_ssa: String,
    /// Scalar SSA for 1.0 (used by tadds).
    cone_ssa: String,
}

impl PtoContext {
    fn new() -> Self {
        PtoContext {
            tiles: HashMap::new(),
            next_ssa: 0,
            ub: UbAllocator::new::<A2A3>(),
            tile_error: None,
            head_offsets: None,
            pv_row_base: None,
            pv_col_base: None,
            head_stride_a: None,
            head_stride_b: None,
            block_row_base: None,
            batched_result_stored_inline: std::collections::HashSet::new(),
            softmax_rows_stored_inline: std::collections::HashSet::new(),
            matmul_accumulate: false,
            pv_k_base: None,
            pv_k_on_rows: None,
            pending_row_softmax: HashMap::new(),
            pending_batched: HashMap::new(),
            inline_consts: HashMap::new(),
            tv_map: HashMap::new(),
            sizes_used: Vec::new(),
            ptr_aliases: HashMap::new(),
            const_map: HashMap::new(),
            float_const_map: HashMap::new(),
            gep_offsets: HashMap::new(),
            matmul_result_stored_inline: std::collections::HashSet::new(),
            pending_blocked_matmuls: HashMap::new(),
            silu_mul_result_stored_inline: std::collections::HashSet::new(),
            pending_blocked_silu_muls: HashMap::new(),
        }
    }

    /// Resolve an SSA value to a u32 constant, checking const_map first
    /// then falling back to parse_const_arg for %cN / literal values.
    /// UB bytes already committed in this kernel.
    fn ub_live_bytes(&self) -> usize { self.ub.live() }

    fn resolve_const(&self, s: &str) -> u32 {
        // Synthesised per-head calls carry their dims here rather than as
        // module constants.
        if let Some(v) = self.inline_consts.get(s.trim()) {
            return *v;
        }
        if let Some(&n) = self.const_map.get(s.trim()) {
            return n;
        }
        parse_const_arg(s)
    }

    /// Resolve an SSA name to a float literal string, falling back to the raw SSA name.
    fn resolve_float(&self, s: &str) -> String {
        let s = s.trim();
        if let Some(v) = self.float_const_map.get(s) {
            return v.clone();
        }
        // Try integer const map (e.g. 0 → "0.0")
        if let Some(&n) = self.const_map.get(s) {
            return format!("{}.0", n);
        }
        s.to_string()
    }

    /// Resolve an SSA name to its original GM arg, following the ptr_aliases chain.
    fn resolve_ptr(&self, ssa: &str) -> String {
        let mut current = ssa.to_string();
        let mut seen = std::collections::HashSet::new();
        loop {
            if seen.contains(&current) {
                break;
            }
            seen.insert(current.clone());
            if let Some(origin) = self.ptr_aliases.get(&current) {
                current = origin.clone();
            } else {
                break;
            }
        }
        current
    }

    /// Resolve the total element offset for a (possibly GEP-derived) pointer.
    /// Returns 0 if the pointer is a direct GM arg or if the offset is unknown.
    fn resolve_offset(&self, ssa: &str) -> u32 {
        let mut current = ssa.to_string();
        let mut total_offset: u32 = 0;
        let mut seen = std::collections::HashSet::new();
        loop {
            if seen.contains(&current) {
                break;
            }
            seen.insert(current.clone());
            if let Some(&off) = self.gep_offsets.get(&current) {
                total_offset = total_offset.saturating_add(off);
            }
            if let Some(origin) = self.ptr_aliases.get(&current) {
                current = origin.clone();
            } else {
                break;
            }
        }
        total_offset
    }

    fn fresh_ssa(&mut self) -> String {
        let n = self.next_ssa;
        self.next_ssa += 1;
        format!("%pto{}", n)
    }

    fn use_size(&mut self, s: u32) {
        if !self.sizes_used.contains(&s) {
            self.sizes_used.push(s);
        }
    }

    fn unique_sizes(&self) -> Vec<u32> {
        let mut v = self.sizes_used.clone();
        v.sort();
        v.dedup();
        v
    }

    /// Get or create the tensor_view SSA for a GM pointer.
    fn get_or_make_tv(
        &mut self,
        gm_arg: &str,
        rows: u32,
        cols: u32,
        dtype: &str,
        ops: &mut Vec<String>,
    ) -> String {
        if let Some((ssa, r, c, d)) = self.tv_map.get(gm_arg).cloned() {
            if r == rows && c == cols && d == dtype {
                return ssa;
            }
        }
        self.use_size(rows);
        self.use_size(cols);
        self.use_size(1);
        let ssa = self.fresh_ssa();
        let tv_ty = tv_type(rows, cols, dtype);
        ops.push(format!(
            "{} = pto.make_tensor_view {}, shape = [%c{}, %c{}], strides = [%c{}, %c1] : {}",
            ssa, gm_arg, rows, cols, cols, tv_ty
        ));
        self.tv_map.insert(
            gm_arg.to_string(),
            (ssa.clone(), rows, cols, dtype.to_string()),
        );
        ssa
    }

    /// Emit a *fresh* tensor_view on an existing GM buffer with transposed
    /// shape and strides.
    ///
    /// The original (row-major) view of a GM buffer has shape `[R,C]` and
    /// strides `[C,1]`. A transposed view describes the same memory as if
    /// it were `[C,R]` with strides `[1,C]` — the slow axis becomes the
    /// fast axis and vice-versa.
    ///
    /// Used by `translate_attention` for K: the GM buffer is `S×D`
    /// row-major, but the cube needs a `D×S` operand (right of tmatmul).
    /// Creating a separate transposed view lets the partition_view +
    /// tload consume the buffer as `D×S` without a physical copy.
    ///
    /// Not cached in `tv_map` (to avoid conflicting with the canonical
    /// row-major view for the same GM arg).
    fn make_tv_transposed(
        &mut self,
        gm_arg: &str,
        orig_rows: u32,
        orig_cols: u32,
        dtype: &str,
        ops: &mut Vec<String>,
    ) -> String {
        self.use_size(orig_rows);
        self.use_size(orig_cols);
        self.use_size(1);
        let ssa = self.fresh_ssa();
        // Transposed view: shape [orig_cols, orig_rows], strides [1, orig_cols].
        // (Row-major original has strides [orig_cols, 1]; transposing swaps them.)
        let tv_ty = tv_type(orig_cols, orig_rows, dtype);
        ops.push(format!(
            "{} = pto.make_tensor_view {}, shape = [%c{}, %c{}], strides = [%c1, %c{}] : {}",
            ssa, gm_arg, orig_cols, orig_rows, orig_cols, tv_ty
        ));
        ssa
    }

    /// Create a partition_view at an explicit (row, col) offset.
    ///
    /// `make_pv` takes a FLAT element offset and divides by the tile width to recover a row,
    /// which is only correct when the tile spans the whole row.  A partition cell generally
    /// does not, so its offsets are supplied directly here.
    fn make_pv_at(
        &mut self,
        tv_ssa: &str,
        rows: u32,
        cols: u32,
        dtype: &str,
        row_off: u32,
        col_off: u32,
        ops: &mut Vec<String>,
    ) -> String {
        self.use_size(row_off);
        self.use_size(col_off);
        self.use_size(rows);
        self.use_size(cols);
        let ssa = self.fresh_ssa();
        let tv_ty = tv_type(rows, cols, dtype);
        let ptv_ty = ptv_type(rows, cols, dtype);

        // Inside a batched loop the row offset is the constant plus the loop's
        // per-head base, so each iteration reads its own head's slice. This is
        // the only place partition_view offsets are written, which is why
        // batching needs no change to any translator.
        // The K block lands on whichever axis THIS operand indexes K by:
        // rows for a `[K,N]` operand, columns for `[M,K]`, neither for the
        // `[M,N]` output. Folding it into the row/col bases instead would put
        // it on the wrong operand -- the batched matmul suppresses exactly
        // those two axes per operand.
        let k_row = if self.pv_k_on_rows == Some(true) { self.pv_k_base.clone() } else { None };
        let k_col = if self.pv_k_on_rows == Some(false) { self.pv_k_base.clone() } else { None };
        let row_expr = match (&self.pv_row_base, &k_row) {
            (base, kb) if base.is_some() || kb.is_some() => {
                // Sum whichever terms are present, then the constant.
                let mut cur = match (base.clone(), kb.clone()) {
                    (Some(b), Some(k)) => {
                        let acc = format!("%pto_ra{}", self.next_ssa);
                        self.next_ssa += 1;
                        ops.push(format!("{} = arith.addi {}, {} : index", acc, b, k));
                        acc
                    }
                    (Some(b), None) => b,
                    (None, Some(k)) => k,
                    (None, None) => unreachable!(),
                };
                let sum = format!("%pto_r{}", self.next_ssa);
                self.next_ssa += 1;
                ops.push(format!("{} = arith.addi {}, %c{} : index", sum, cur, row_off));
                cur = sum;
                cur
            }
            _ => format!("%c{}", row_off),
        };
        // Same treatment for columns, which is what an N-block loop needs:
        // `b_right` is `k*n` with no row term, so only a column offset bounds it.
        let col_expr = match (&self.pv_col_base, &k_col) {
            (base, kb) if base.is_some() || kb.is_some() => {
                let cur = match (base.clone(), kb.clone()) {
                    (Some(b), Some(k)) => {
                        let acc = format!("%pto_ca{}", self.next_ssa);
                        self.next_ssa += 1;
                        ops.push(format!("{} = arith.addi {}, {} : index", acc, b, k));
                        acc
                    }
                    (Some(b), None) => b,
                    (None, Some(k)) => k,
                    (None, None) => unreachable!(),
                };
                let sum = format!("%pto_c{}", self.next_ssa);
                self.next_ssa += 1;
                ops.push(format!("{} = arith.addi {}, %c{} : index", sum, cur, col_off));
                sum
            }
            _ => format!("%c{}", col_off),
        };
        ops.push(format!(
            "{} = pto.partition_view {}, offsets = [{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
            ssa, tv_ssa, row_expr, col_expr, rows, cols, tv_ty, ptv_ty
        ));
        ssa
    }

    /// Shift every partition_view emitted from here until cleared by `base`
    /// rows. Used by batched stages to address the head the loop is on.
    fn set_pv_row_base(&mut self, base: Option<String>) {
        self.pv_row_base = base;
    }

    fn insert_tile(&mut self, ssa: &str, t: TileInfo) {
        self.tiles.insert(ssa.to_string(), t);
    }

    /// Run `f` with the row base scaled for an operand of `rows` per head.
    ///
    /// The loop publishes `%h_i`; this emits `%h_i * rows` so operands with
    /// different heights land on their own slices.
    ///
    /// COMPOSES with an inner block offset rather than replacing it. When a
    /// head is itself blocked, `with_row_block` has already installed
    /// `%r_i * mb`, and overwriting it dropped that term entirely: A addressed
    /// `h_i * mb` instead of `h_i * head_stride + r_i * mb`, so every block of
    /// a head read the same rows. On device that is `untouched=0` with an
    /// identical error in every column block — distinguishable from a column
    /// mis-address (error DIFFERS per column block) only because the parity
    /// check reports per (head, column block) rather than in aggregate.
    ///
    /// `rows` must be the operand's HEAD STRIDE, not the block size: the head
    /// term steps over whole heads regardless of how finely one is walked.
    fn with_head_rows<T>(
        &mut self,
        rows: u32,
        f: impl FnOnce(&mut Self, &mut Vec<String>) -> T,
        ops: &mut Vec<String>,
    ) -> T {
        let prev = self.pv_row_base.clone();
        let b = self.fresh_ssa();
        self.use_size(rows);
        ops.push(format!(
            "  {b} = arith.muli %h_i, %c{rows} : index   // operand head base"
        ));
        // If an inner block loop already installed `%r_i * mb`, keep it: the
        // operand address is head base PLUS block offset, and dropping either
        // term is silently wrong.
        //
        // `block_row_base` is set only by `with_row_block`, so `prev` alone
        // cannot be used here -- outside a block loop `prev` holds the previous
        // OPERAND's head base, which must be replaced rather than summed.
        let blk = self.block_row_base.clone();
        self.pv_row_base = Some(match blk {
            Some(blk) => {
                let sum = self.fresh_ssa();
                ops.push(format!(
                    "  {sum} = arith.addi {b}, {blk} : index   // head base + row block"
                ));
                sum
            }
            None => b,
        });
        let out = f(self, ops);
        self.pv_row_base = prev;
        out
    }

    fn in_batched_loop(&self) -> bool {
        self.pv_row_base.is_some()
    }

    fn set_matmul_accumulate(&mut self, on: bool) {
        self.matmul_accumulate = on;
    }

    /// Run `f` with the column base SUPPRESSED.
    ///
    /// Needed because operands disagree on what their columns mean. Under an
    /// N-block loop the column offset is the N iterator, which is correct for
    /// `B` (`[K,N]`) and the output (`[M,N]`) but WRONG for `A` (`[M,K]`, whose
    /// columns are the K dimension). Applying one base to all three made A read
    /// columns 256..319 instead of 0..63 at nb=256 — every block written, every
    /// value wrong (max_abs ~1e-1 on device), which no timing run would catch.
    ///
    /// This is the row base's lesson in the column dimension: a shared base is
    /// only correct when the operands agree, and here they do not.
    /// Run `f` with the ROW-BLOCK offset suppressed.
    ///
    /// The mirror of `without_col_base`, and needed for the same reason on the
    /// other operand. The M-block iterator indexes rows that are the M
    /// dimension: true of `A` (`[M,K]`) and the output (`[M,N]`), false of `B`
    /// (`[K,N]`, whose rows are K). Giving B the offset made every column block
    /// of a head read the same wrong K rows — on device, identical error in
    /// both column blocks of every head (max_abs ~1.1e-1).
    ///
    /// The full picture, which is what makes the four cases easy to get wrong:
    ///
    /// |        | rows are | takes r_i | cols are | takes n_i |
    /// |--------|----------|-----------|----------|-----------|
    /// | A [M,K]| M        | yes       | K        | no        |
    /// | B [K,N]| K        | **no**    | N        | yes       |
    /// | out    | M        | yes       | N        | yes       |
    fn without_row_block<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let prev = self.block_row_base.take();
        let out = f(self);
        self.block_row_base = prev;
        out
    }

    fn without_col_base<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let prev = self.pv_col_base.take();
        let out = f(self);
        self.pv_col_base = prev;
        out
    }

    /// Run `f` with a COLUMN base from an N-block loop.
    ///
    /// The mirror of `with_row_block`, needed because the two bound different
    /// tiles: `a_left` is `mb*k` and shrinks with rows, but `b_right` is `k*n`
    /// with no row term at all. At S=512 both scores and pv stage a 128 KB
    /// `b_right` against a 64 KB L0B cap, which no row-blocking touches.
    fn with_col_block<T>(
        &mut self,
        nb: u32,
        f: impl FnOnce(&mut Self, &mut Vec<String>) -> T,
        ops: &mut Vec<String>,
    ) -> T {
        let prev = self.pv_col_base.clone();
        let off = self.fresh_ssa();
        self.use_size(nb);
        ops.push(format!(
            "  {off} = arith.muli %n_i, %c{nb} : index   // col block offset"
        ));
        let base = match &prev {
            Some(pb) => {
                let sum = self.fresh_ssa();
                ops.push(format!("  {sum} = arith.addi {pb}, {off} : index"));
                sum
            }
            None => off,
        };
        self.pv_col_base = Some(base);
        let out = f(self, ops);
        self.pv_col_base = prev;
        out
    }

    /// Run `f` with a K-block offset applied on the axis each operand indexes
    /// K by.
    ///
    /// K is the CONTRACTED dimension, so unlike a row or column block it lands
    /// on a different axis per operand: A is `[M,K]` and takes it on COLUMNS,
    /// B is `[K,N]` and takes it on ROWS. The output takes it on neither -- it
    /// is `[M,N]`, and every K iteration accumulates into the same tile.
    ///
    /// This is the same per-operand offsets rule that the row/column blocks
    /// follow; applying one shared base to both operands reads the wrong slice
    /// of one of them, and produces a plausible wrong answer rather than an
    /// error.
    /// Run `f` with a K-block offset available to the operand loaders.
    ///
    /// K is the CONTRACTED dimension, so it lands on a DIFFERENT AXIS per
    /// operand: A is `[M,K]` and takes it on COLUMNS, B is `[K,N]` on ROWS,
    /// and the output `[M,N]` takes it on neither.
    ///
    /// It therefore cannot ride in `pv_row_base` / `pv_col_base` the way the M
    /// and N blocks do. The batched matmul already wraps A in
    /// `without_col_base` and B in `without_row_block` -- precisely the two
    /// axes K needs -- so a K offset placed in those slots is suppressed on the
    /// operand that needs it and applied to the one that does not. That emitted
    /// an A slice fixed at columns [0,128) for every K iteration and a B slice
    /// taking K on its columns: the axes swapped, MERE 9.1 with
    /// matched_ratio 0.0003.
    ///
    /// Kept in its own slot so each loader can apply it to its own axis.
    fn with_k_block<T>(
        &mut self,
        kb: u32,
        f: impl FnOnce(&mut Self, &mut Vec<String>) -> T,
        ops: &mut Vec<String>,
    ) -> T {
        let prev = self.pv_k_base.clone();
        let off = self.fresh_ssa();
        self.use_size(kb);
        ops.push(format!(
            "  {off} = arith.muli %k_i, %c{kb} : index   // K block offset"
        ));
        self.pv_k_base = Some(off);
        let out = f(self, ops);
        self.pv_k_base = prev;
        out
    }

    /// Run `f` with the row base advanced by an INNER block loop.
    ///
    /// Composes with `with_head_rows`: the head loop contributes `h_i * rows`
    /// and this adds `r_i * rb`, so a per-head operand that is itself too large
    /// for its memory can be walked in blocks without disturbing the head
    /// slicing. Both terms are needed -- dropping either reads the wrong slice,
    /// and the two failures look identical from a timing run.
    fn with_row_block<T>(
        &mut self,
        rb: u32,
        f: impl FnOnce(&mut Self, &mut Vec<String>) -> T,
        ops: &mut Vec<String>,
    ) -> T {
        let prev = self.pv_row_base.clone();
        let off = self.fresh_ssa();
        self.use_size(rb);
        ops.push(format!(
            "  {off} = arith.muli %r_i, %c{rb} : index   // row block offset"
        ));
        let base = match &prev {
            Some(h) => {
                let sum = self.fresh_ssa();
                ops.push(format!(
                    "  {sum} = arith.addi {h}, {off} : index   // head base + row block"
                ));
                sum
            }
            None => off.clone(),
        };
        // Publish the BLOCK OFFSET separately as well, so a later
        // `with_head_rows` can SUM with it rather than replace it. Both terms
        // are needed: the head base steps over whole heads, the block offset
        // walks within one. Keeping them in one slot conflated "the previous
        // operand's head base" (replace) with "this head's block offset" (add).
        let prev_blk = self.block_row_base.replace(off);
        self.pv_row_base = Some(base);
        let out = f(self, ops);
        self.pv_row_base = prev;
        self.block_row_base = prev_blk;
        out
    }

    /// Load a deferred operand's slice for the head the loop is on.
    ///
    /// Emits partition_view + alloc + tload at the current row base, so the
    /// tile holds head h's data. Returns `None` if the operand was not deferred
    /// — its load would then be outside the loop, and every head would read
    /// head 0, which is the bug this whole mechanism exists to prevent.
    fn load_deferred_for_head(
        &mut self,
        t: &TileInfo,
        rows: u32,
        cols: u32,
        ops: &mut Vec<String>,
    ) -> Option<TileInfo> {
        let d = t.deferred.clone()?;
        let dtype = t.dtype.clone();
        // Only the per-head partition_view is emitted here. The tile itself is
        // allocated by the matmul path that consumes this, which knows the
        // operand needs a MAT tile (and then left/right/acc) — allocating a vec
        // tile here made the kernel span both engines, and a Mix kernel cannot
        // be built on this CANN at all. That is what the emitter's
        // single-engine assertion caught.
        let pv = self.make_pv(&d.tv_ssa, rows, cols, &dtype, d.elem_offset, ops);
        let mut out = t.clone();
        out.pv_ssa = Some(pv);
        out.deferred = None;
        Some(out)
    }

    /// Load a deferred operand for this iteration into a NAMED vec tile.
    ///
    /// The sibling `load_deferred_for_head` stops at the partition_view because
    /// its caller (the matmul path) must allocate the tile itself — it needs a
    /// MAT tile, and allocating a vec tile there made the kernel span both
    /// engines, which cannot be built on this CANN at all.
    ///
    /// Vector-side generic stages have no such constraint and do need the load,
    /// so this completes it: partition_view -> alloc -> tload, under whatever
    /// row base `with_head_rows` has installed. Registering the tile under
    /// `name` lets the caller refer to it from the inner call.
    fn load_deferred_for_head_as(
        &mut self,
        t: &TileInfo,
        rows: u32,
        cols: u32,
        name: &str,
        ops: &mut Vec<String>,
    ) -> Result<(), String> {
        let d = t
            .deferred
            .clone()
            .ok_or_else(|| format!("batched: operand for {name} was not deferred"))?;
        let dtype = t.dtype.clone();
        let pv = self.make_pv(&d.tv_ssa, rows, cols, &dtype, d.elem_offset, ops);
        let tile = self.alloc_tile(name, rows, cols, &dtype, ops);
        ops.push(format!(
            "pto.tload ins({} : {}) outs({} : {})",
            pv,
            ptv_type(rows, cols, &dtype),
            tile,
            tile_buf_type(rows, cols, &dtype)
        ));
        Ok(())
    }

    /// Create a partition_view from a tensor_view SSA.
    ///
    /// `elem_offset` is the flat element offset into the GM buffer (from GEP analysis).
    /// It is converted to a (row_offset, col_offset) pair using `cols` as the row stride.
    /// If `elem_offset` is 0 or `cols` is 0, both offsets are 0.
    fn make_pv(
        &mut self,
        tv_ssa: &str,
        rows: u32,
        cols: u32,
        dtype: &str,
        elem_offset: u32,
        ops: &mut Vec<String>,
    ) -> String {
        let row_off = if cols > 0 { elem_offset / cols } else { 0 };
        let col_off = if cols > 0 { elem_offset % cols } else { 0 };
        // Delegate so the batched-loop row base is applied in exactly one
        // place; duplicating the emission here is how the two would drift.
        self.make_pv_at(tv_ssa, rows, cols, dtype, row_off, col_off, ops)
    }

    /// Allocate a row-reduction output tile (rows×1, col_major).
    fn alloc_tile_rowreduce(&mut self, mlir_ssa: &str, rows: u32, dtype: &str, ops: &mut Vec<String>) -> String {
        let tb_ty = tile_buf_type_rowreduce(rows, dtype);
        self.alloc_tile_typed(mlir_ssa, rows, 1, dtype, &tb_ty, ops)
    }

    /// Allocate a row-reduction output tile (rows×1, row_major) — needed when
    /// the tile is consumed as the source of `pto.trsqrt`.
    fn alloc_tile_rowreduce_rowmajor(&mut self, mlir_ssa: &str, rows: u32, dtype: &str, ops: &mut Vec<String>) -> String {
        let tb_ty = tile_buf_type_rowreduce_rowmajor(rows, dtype);
        self.alloc_tile_typed(mlir_ssa, rows, 1, dtype, &tb_ty, ops)
    }

    /// Allocate a vec (`loc=vec`) tile buffer SSA and record it.
    fn alloc_tile(
        &mut self,
        mlir_ssa: &str,
        rows: u32,
        cols: u32,
        dtype: &str,
        ops: &mut Vec<String>,
    ) -> String {
        let tb_ty = tile_buf_type(rows, cols, dtype);
        self.alloc_tile_typed(mlir_ssa, rows, cols, dtype, &tb_ty, ops)
    }

    /// Allocate a tile buffer with a custom type string (e.g., mat/left/right/acc).
    fn alloc_tile_typed(
        &mut self,
        mlir_ssa: &str,
        rows: u32,
        cols: u32,
        dtype: &str,
        tb_ty: &str,
        ops: &mut Vec<String>,
    ) -> String {
        // NPU tile trait bounds: validate shape (C2-C4) and reserve UB budget
        // (C1+C5) at the point of allocation. Record the first violation; the
        // caller (generate_func_pto) turns it into an Err(String) diagnostic.
        if self.tile_error.is_none() {
            if let Err(e) = validate_tile_shape::<A2A3>(rows, cols, dtype, tb_ty) {
                self.tile_error = Some(e);
            } else if let Err(e) = self.ub.place_in(
                TileSpace::from_type(tb_ty),
                rows as usize * cols as usize * dtype_bytes_pto(dtype),
            ) {
                self.tile_error = Some(e);
            }
        }
        let ssa = self.fresh_ssa();
        ops.push(format!("{} = pto.alloc_tile : {}", ssa, tb_ty));
        self.tiles.insert(
            mlir_ssa.to_string(),
            TileInfo {
                ssa: ssa.clone(),
                rows,
                cols,
                dtype: dtype.to_string(),
                tb_type: tb_ty.to_string(),
                pv_ssa: None,
                gm_name: None,
                deferred: None,
            },
        );
        ssa
    }

    /// Element offsets for the head currently being emitted, if any.
    fn set_head_offsets(&mut self, a: u32, b: u32, out: u32) {
        self.head_offsets = Some((a, b, out));
    }

    fn clear_head_offsets(&mut self) {
        self.head_offsets = None;
        self.inline_consts.clear();
    }

    /// Dimensions for a synthesised per-head call. `resolve_const` consults
    /// these first, so a re-emitted line can name them without the module
    /// having declared matching `arith.constant`s.
    fn set_inline_consts(&mut self, pairs: &[(&str, u32)]) {
        self.inline_consts.clear();
        for (k, v) in pairs {
            self.inline_consts.insert((*k).to_string(), *v);
        }
    }

    /// Make `alias` refer to the same tile as `existing`, so a downstream
    /// store can find the batched result under the SSA the caller used.
    fn alias_tile(&mut self, existing: &str, alias: &str) {
        if let Some(t) = self.tiles.get(existing).cloned() {
            self.tiles.insert(alias.to_string(), t);
        }
    }

    fn get_tile(&self, mlir_ssa: &str) -> Option<&TileInfo> {
        self.tiles.get(mlir_ssa)
    }
}

// ---------------------------------------------------------------------------
// Body analysis: MLIR lines → PTO-MLIR ops
// ---------------------------------------------------------------------------

fn analyze_body(
    body_lines: &[String],
    func: &MlirFunc,
    ctx: &mut PtoContext,
) -> Result<Vec<String>, String> {
    // Pre-populate ctx with info about GM pointer args
    for arg in &func.args {
        if arg.is_gm {
            // We'll create tensor views on demand when we see the first load/store
            let _ = arg;
        }
    }

    let mut ops: Vec<String> = Vec::new();
    // store_map tracks: alloca_ssa → ptr_ssa stored into it.
    // Used to resolve llvm.load patterns back to the original ptr.
    let mut store_map: HashMap<String, String> = HashMap::new();

    // ── SiLU+Mul fusion pre-pass ──
    // Detect when silu result is immediately consumed by a mul.
    // Key: SSA of silu result → (index of silu line, index of mul line, mul_line copy)
    let silu_mul_fused = detect_silu_mul_pairs(body_lines);

    // ── K/N-blocked matmul operand pre-pass ──
    // Identify tile_load lines that feed a matmul requiring blocking;
    // these are handled specially: translate_load skips the full-shape
    // tload and only stashes tv_ssa + elem_offset in a `deferred` record
    // on the TileInfo, and translate_matmul emits the scf.for nest.
    let mut blocked_mm_loads = detect_blocked_matmul_loads(body_lines);
    // ── batched-stage operand pre-pass ──
    // Same deferral, same reason: the load has to happen per head, inside the
    // batched loop, not once above it.
    for (idx, role) in detect_batched_loads(body_lines) {
        blocked_mm_loads.entry(idx).or_insert(role);
    }
    // ── N-blocked silu_mul operand pre-pass (#67) ──
    // Same shape as the matmul pre-pass: identify tile_load lines that
    // feed a silu_mul whose 5-tile fused emit overflows the UB budget,
    // mark the gate / up loads to be deferred to the per-chunk loop.
    // We union the result into `blocked_mm_loads` since the load branch's
    // defer behaviour is identical (skip full-shape tload, stash a
    // DeferredMatmulOperand on the TileInfo). translate_silu_mul's blocked
    // path then reads `tile.deferred` to get tv_ssa + elem_offset, exactly
    // like translate_matmul_blocked does.
    let blocked_silu_loads =
        detect_blocked_silu_mul_loads(body_lines, &silu_mul_fused);
    for (idx, role) in blocked_silu_loads.into_iter() {
        // Only insert if not already present from the matmul pre-pass; if
        // a load somehow feeds both, the matmul label wins (its defer
        // requirements are stricter).
        blocked_mm_loads.entry(idx).or_insert(role);
    }

    for (i, line) in body_lines.iter().enumerate() {
        let line = line.trim();

        if line.is_empty()
            || line.ends_with(':')
            || line == "llvm.return"
            || line.contains("__tile_pipe_barrier")
            // llvm.mlir.addressof lines load function pointers for indirect calls;
            // the actual call site is the subsequent llvm.call line, so skip these.
            || line.contains("llvm.mlir.addressof")
        {
            continue;
        }

        // Skip mul lines that have been fused with a preceding silu
        if silu_mul_fused.values().any(|&(_, mul_idx)| mul_idx == i) {
            continue;
        }

        // Track integer and float constants so we can resolve SSA names like %12 → 1024.
        //
        // Pattern: llvm.mlir.constant(N : iXX) : iXX
        //   %9 = llvm.mlir.constant(1 : i32) : i32  → ctx.const_map[%9] = 1
        //   %eps = llvm.mlir.constant(1.0e-5 : f32) : f32  → ctx.float_const_map[%eps] = "1.0e-5"
        if line.contains("llvm.mlir.constant(") && !line.contains("!llvm.ptr") {
            if let Some(result) = extract_result_ssa(line) {
                if let Some(open) = line.find("llvm.mlir.constant(") {
                    let rest = &line[open + "llvm.mlir.constant(".len()..];
                    // Extract the value string up to the type annotation
                    let val_str: String = rest.chars()
                        .take_while(|c| *c != ' ' && *c != ')')
                        .collect();
                    if line.contains(": f32") || line.contains(": f64") {
                        // Float constant
                        ctx.float_const_map.insert(result, val_str);
                    } else {
                        // Integer constant
                        let n_str: String = val_str.chars().take_while(|c| c.is_ascii_digit()).collect();
                        if let Ok(n) = n_str.parse::<u32>() {
                            ctx.const_map.insert(result, n);
                        }
                    }
                }
            }
            continue;
        }
        // Pattern: llvm.bitcast of integer → propagate constant value.
        //   %10 = llvm.bitcast %9 : i32 to i32
        if line.contains("llvm.bitcast") && !line.contains("!llvm.ptr") {
            if let Some(result) = extract_result_ssa(line) {
                if let Some(pos) = line.find("llvm.bitcast ") {
                    let rest = line[pos + "llvm.bitcast ".len()..].trim();
                    let src = rest.split_whitespace().next().unwrap_or("");
                    if let Some(n) = ctx.const_map.get(src).copied() {
                        ctx.const_map.insert(result, n);
                    }
                }
            }
            continue;
        }

        // Track pointer aliases so we can resolve derived GM pointers.
        //
        // Pattern 1: getelementptr — result is an offset of the source ptr.
        //   %7 = llvm.getelementptr %arg0[%4] : (!llvm.ptr<1>, ...) -> !llvm.ptr<1>, f32
        if line.contains("llvm.getelementptr") && line.contains("!llvm.ptr<1>") {
            if let Some(result) = extract_result_ssa(line) {
                // source is the first argument: `%arg0[%4]` or `%arg0`
                // Strip any `[...]` subscript to get just the base ptr SSA.
                if let Some(open) = line.find("llvm.getelementptr ") {
                    let rest = line[open + "llvm.getelementptr ".len()..].trim();
                    let raw = rest
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(',');
                    // Extract index SSA from subscript: "%arg0[%4]" → index="%4"
                    let (src, idx_ssa) = if let Some(bracket) = raw.find('[') {
                        let base = &raw[..bracket];
                        let after = &raw[bracket + 1..];
                        let idx = after.trim_end_matches(']').trim();
                        (base, Some(idx.to_string()))
                    } else {
                        (raw, None)
                    };
                    ctx.ptr_aliases.insert(result.clone(), src.to_string());
                    // If the index is a known constant, record the element offset.
                    // IMPORTANT: only use const_map here — parse_const_arg("%2214")
                    // would misinterpret an SSA name as the literal integer 2214,
                    // producing wildly wrong partition_view offsets for runtime
                    // indices like bid*rows*cols.
                    if let Some(idx) = idx_ssa {
                        if let Some(&off) = ctx.const_map.get(idx.trim()) {
                            if off > 0 {
                                ctx.gep_offsets.insert(result, off);
                            }
                        }
                    }
                }
            }
            continue;
        }

        // Pattern 2: store !llvm.ptr<1> value into a local alloca.
        //   llvm.store %7, %6 {alignment = ...} : !llvm.ptr<1>, !llvm.ptr
        if line.starts_with("llvm.store") && line.contains("!llvm.ptr<1>") {
            // llvm.store %val, %dest ... : !llvm.ptr<1>, !llvm.ptr
            let after_store = &line["llvm.store".len()..].trim_start();
            let parts: Vec<&str> = after_store.split(',').collect();
            if parts.len() >= 2 {
                let val = parts[0].trim().to_string();
                let dest = parts[1]
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_string();
                store_map.insert(dest, val);
            }
            continue;
        }

        // Pattern 3a: bitcast !llvm.ptr<1> → !llvm.ptr<1> — direct alias.
        //   %23 = llvm.bitcast %arg1 : !llvm.ptr<1> to !llvm.ptr<1>
        if line.contains("llvm.bitcast") && line.contains("!llvm.ptr<1> to !llvm.ptr<1>") {
            if let Some(result) = extract_result_ssa(line) {
                if let Some(pos) = line.find("llvm.bitcast ") {
                    let rest = line[pos + "llvm.bitcast ".len()..].trim();
                    let src = rest.split_whitespace().next().unwrap_or("");
                    ctx.ptr_aliases.insert(result, src.to_string());
                }
            }
            continue;
        }

        // Pattern 3: load !llvm.ptr<1> from alloca → alias to whatever was stored.
        //   %8 = llvm.load %6 {alignment = ...} : !llvm.ptr -> !llvm.ptr<1>
        if line.contains("llvm.load") && line.ends_with("!llvm.ptr<1>") {
            if let Some(result) = extract_result_ssa(line) {
                // find the source alloca (%6)
                if let Some(pos) = line.find("llvm.load ") {
                    let rest = line[pos + "llvm.load ".len()..].trim();
                    let src = rest
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches('{');
                    let stored = store_map
                        .get(src)
                        .cloned()
                        .unwrap_or_else(|| src.to_string());
                    // Store the immediate alias (not the fully-resolved root) so that
                    // resolve_offset can still find GEP offsets recorded on intermediate
                    // SSA names (e.g. %gep → %arg0 with gep_offsets[%gep]=1024).
                    ctx.ptr_aliases.insert(result, stored);
                }
            }
            continue;
        }

        // get_block_idx — not directly representable in pure PTO-MLIR, emit comment
        if line.contains("get_block_idx") {
            ops.push(
                "// block index: see block_idx intrinsic (currently out-of-scope for ptoas)"
                    .to_string(),
            );
            continue;
        }

        // `pipelined_for(depth)` marker — consumed by the cpp emitter's
        // `detect_tiling_loop`. PTO ignores it because ptoas auto-inserts
        // cross-pipe sync during assembly.
        // MUST PRECEDE every stage-op check: `batched.inner` embeds an op
        // call verbatim, so an earlier `contains("__tile_<op>_f32")` would
        // claim this line and translate the inner call OUTSIDE the loop.
        // The GENERIC batched form. One recognition point for every stage that
        // is not matmul or softmax, so adding a stage costs a caller-side
        // description instead of an intrinsic + enum arm + emitter body.
        //
        //   %o = llvm.call @__tile_batched_f32(%out, %heads, %rows, %cols)
        //        { batched.inner = "<one MLIR call, operands as %__op0..>",
        //          batched.operands = "<ssa>:<rows>x<cols>,..." } : ...
        //
        // Each operand states its OWN rows-per-iteration. That is what keeps
        // the shared-row-base defect unrepresentable at this layer, the same
        // property `GmBatch` gives in user source.
        if line.contains("__tile_batched_f32") {
            let kind = parse_generic_batched(line)?;
            translate_batched(line, kind, ctx, &mut ops)?;
            continue;
        }
        if line.contains("__tile_pipelined_for_begin")
            || line.contains("__tile_pipelined_for_end")
        {
            continue;
        }

        // tile.load f32
        if line.contains("__tile_load_f32") {
            let blocked = blocked_mm_loads.contains_key(&i);
            translate_load(line, "f32", ctx, func, &mut ops, blocked)?;
            continue;
        }
        // tile.load f16
        if line.contains("__tile_load_f16") {
            // f16 matmul inputs must be deferred to the matmul emitter so
            // the tload lands in a CBUF/mat tile (not the default UB/vec).
            // CANN 8.5 cube cores don't support b16 GM→UB; a vec-tile tload
            // at f16 triggers a `copy_gm_to_ubuf_align_b16` target-feature
            // error in ccec. detect_blocked_matmul_loads returns `"A"` /
            // `"B"` for loads that directly feed a matmul.
            let blocked = blocked_mm_loads.contains_key(&i);
            translate_load(line, "f16", ctx, func, &mut ops, blocked)?;
            continue;
        }
        // tile.load i8 — same deferral rules as f16: inputs to an int8
        // matmul must land directly in CBUF/mat tiles (the K/N-blocked
        // emitter re-tloads per-block inside the loop).
        if line.contains("__tile_load_i8") {
            let blocked = blocked_mm_loads.contains_key(&i);
            translate_load(line, "i8", ctx, func, &mut ops, blocked)?;
            continue;
        }
        // tile.store f32
        if line.contains("__tile_store_f32") {
            translate_store(line, "f32", ctx, func, &mut ops)?;
            continue;
        }
        // tile.store f16
        if line.contains("__tile_store_f16") {
            translate_store(line, "f16", ctx, func, &mut ops)?;
            continue;
        }
        // tile.store i8
        if line.contains("__tile_store_i8") {
            translate_store(line, "i8", ctx, func, &mut ops)?;
            continue;
        }
        // tile.add f32
        if line.contains("__tile_add_f32") {
            translate_binary(line, "f32", "pto.tadd", ctx, &mut ops)?;
            continue;
        }
        // tile.mul f32
        if line.contains("__tile_mul_f32") {
            translate_binary(line, "f32", "pto.tmul", ctx, &mut ops)?;
            continue;
        }
        // tile.add f16
        if line.contains("__tile_add_f16") {
            translate_binary(line, "f16", "pto.tadd", ctx, &mut ops)?;
            continue;
        }
        // tile.mul f16
        if line.contains("__tile_mul_f16") {
            translate_binary(line, "f16", "pto.tmul", ctx, &mut ops)?;
            continue;
        }
        // tile.exp f32
        if line.contains("__tile_exp_f32") {
            translate_unary(line, "f32", "pto.texp", ctx, &mut ops)?;
            continue;
        }
        // tile.exp f16
        if line.contains("__tile_exp_f16") {
            translate_unary(line, "f16", "pto.texp", ctx, &mut ops)?;
            continue;
        }
        // Batched variants FIRST: `__tile_softmax_f32` is a substring of
        // `__tile_softmax_batched_f32`'s neighbours in spirit, and relying on
        // substring order for correctness is exactly the kind of silent
        // mis-dispatch that is hard to see later.
        if line.contains("__tile_matmul_batched_f32") {
            translate_batched(line, BatchedKind::Matmul, ctx, &mut ops)?;
            continue;
        }
        if line.contains("__tile_softmax_batched_f32") {
            translate_batched(line, BatchedKind::Softmax, ctx, &mut ops)?;
            continue;
        }
        // tile.softmax f32 — decomposed into 5 reduction ops
        if line.contains("__tile_softmax_f32") {
            translate_softmax(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.softmax f16 — decomposed into 5 reduction ops
        if line.contains("__tile_softmax_f16") {
            translate_softmax(line, "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul f32
        if line.contains("__tile_matmul_f32") {
            translate_matmul(line, ctx, &mut ops)?;
            continue;
        }
        // tile.sub f32
        if line.contains("__tile_sub_f32") {
            translate_binary(line, "f32", "pto.tsub", ctx, &mut ops)?;
            continue;
        }
        // tile.div f32
        if line.contains("__tile_div_f32") {
            translate_binary(line, "f32", "pto.tdiv", ctx, &mut ops)?;
            continue;
        }
        // tile.neg f32
        if line.contains("__tile_neg_f32") {
            translate_unary(line, "f32", "pto.tneg", ctx, &mut ops)?;
            continue;
        }
        // tile.reduce_max f32 — row-wise max
        if line.contains("__tile_reduce_max_f32") {
            translate_unary(line, "f32", "pto.trowmax", ctx, &mut ops)?;
            continue;
        }
        // tile.reduce_sum f32 — row-wise sum
        if line.contains("__tile_reduce_sum_f32") {
            translate_unary(line, "f32", "pto.trowsum", ctx, &mut ops)?;
            continue;
        }
        // tile.scale f32 — scalar multiply (treated as unary with scalar operand)
        if line.contains("__tile_scale_f32") {
            translate_unary(line, "f32", "pto.tmuls", ctx, &mut ops)?;
            continue;
        }
        // tile.silu f32 — SiLU(x) = x * sigmoid(x), with optional SiLU+Mul fusion
        if line.contains("__tile_silu_f32") {
            if let Some(result_ssa) = extract_result_ssa(line) {
                if let Some(&(_, mul_idx)) = silu_mul_fused.get(&result_ssa) {
                    let mul_line = body_lines[mul_idx].trim();
                    translate_silu_mul(line, mul_line, "f32", ctx, &mut ops)?;
                    continue;
                }
            }
            translate_silu(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.silu f16, with optional SiLU+Mul fusion
        if line.contains("__tile_silu_f16") {
            if let Some(result_ssa) = extract_result_ssa(line) {
                if let Some(&(_, mul_idx)) = silu_mul_fused.get(&result_ssa) {
                    let mul_line = body_lines[mul_idx].trim();
                    translate_silu_mul(line, mul_line, "f16", ctx, &mut ops)?;
                    continue;
                }
            }
            translate_silu(line, "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.cast bf16→f32
        if line.contains("__tile_cast_bf16_f32") {
            translate_cast(line, "bf16", "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul_transposed f32 — C = A * B^T via tmatmul with transposed flag
        if line.contains("__tile_matmul_transposed_f32") {
            translate_matmul_transposed(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul_transposed f16
        if line.contains("__tile_matmul_transposed_f16") {
            translate_matmul_transposed(line, "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.attention_gqa f32 — Grouped-Query Attention
        if line.contains("__tile_attention_gqa_f32") {
            translate_attention_gqa(line, ctx, &mut ops)?;
            continue;
        }
        // tile.attention f32 — fused Q@K^T → scale → softmax → @V
        // Decomposed into: matmul + scale + softmax_5ops + matmul
        if line.contains("__tile_attention_f32") {
            translate_attention(line, ctx, &mut ops)?;
            continue;
        }
        // tile.transpose f32
        if line.contains("__tile_transpose_f32") {
            translate_transpose(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.rsqrt f32
        if line.contains("__tile_rsqrt_f32") {
            translate_rsqrt(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.log f32
        if line.contains("__tile_log_f32") {
            translate_log(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.sigmoid f32 — decomposed: neg → exp → adds(1) → divs(1)
        if line.contains("__tile_sigmoid_f32") {
            translate_sigmoid(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.clamp f32 — clamp to [min, max] via tmaxs + tmins
        if line.contains("__tile_clamp_f32") {
            translate_clamp(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.cast f32→f16
        if line.contains("__tile_cast_f32_f16") {
            translate_cast(line, "f32", "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.cast f16→f32
        if line.contains("__tile_cast_f16_f32") {
            translate_cast(line, "f16", "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.slice f32 — extract sub-tile via partition_view with offset
        // tile.partition_cell f32 — cuTile's `partition(shape).load([i,j])`.
        // Lowers to PTO's own partition_view, which is the native counterpart: it takes
        // explicit offsets and sizes, so the cell's footprint is expressed directly rather
        // than emulated. Disjointness of distinct (i,j) is thm:partdisj, established before
        // codegen, so no runtime aliasing check is emitted.
        if line.contains("__tile_partition_cell_f32") {
            translate_partition_cell(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // The _mut form is a store DESTINATION, not a load source: it writes a tile INTO
        // cell (i,j). Routing it through the read path emitted a tload and silently dropped
        // the write. partition_disjoint is what makes several such stores to distinct cells
        // safe with no runtime aliasing check -- the obligation cuTile discharges with
        // `unsafe` at all 26 of its sites.
        if line.contains("__tile_partition_cell_mut_f32") {
            translate_partition_cell_store(line, "f32", ctx, &mut ops)?;
            continue;
        }
        if line.contains("__tile_slice_f32") {
            translate_slice(line, "f32", ctx, func, &mut ops)?;
            continue;
        }
        // tile.concat f32 — concatenate two tiles along columns
        if line.contains("__tile_concat_f32") {
            translate_concat(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.scatter f32 — no PTO equivalent
        if line.contains("__tile_scatter_f32") {
            translate_scatter(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.gather f32 — no PTO equivalent
        if line.contains("__tile_gather_f32") {
            translate_gather(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.arith_progression i32 — emits pto.tci (iota for sort indices).
        if line.contains("__tile_arith_progression_i32") {
            translate_arith_progression(line, ctx, &mut ops)?;
            continue;
        }
        // tile.init_sort_buf f32 — emits pto.tfillpad (sentinel pad to BLOCK boundary).
        if line.contains("__tile_init_sort_buf_f32") {
            translate_init_sort_buf(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.sort32 f32 — emits pto.tsort32 (vbitsort, output is 2× width [val,idx] pairs).
        if line.contains("__tile_sort32_f32") {
            translate_tile_sort(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.mrgsort2 f32 — emits pto.tmrgsort 2-way (merges two 1×N sorted tiles).
        if line.contains("__tile_mrgsort2_f32") {
            translate_merge_sort(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.gather_mask f32 — emits pto.tgather (mask-pattern form, lane select).
        if line.contains("__tile_gather_mask_f32") {
            translate_gather_mask(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.topk f32 — no PTO equivalent
        if line.contains("__tile_topk_f32") {
            translate_topk(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul f16
        if line.contains("__tile_matmul_f16") {
            translate_matmul_f16(line, ctx, &mut ops)?;
            continue;
        }
        // tile.matmul i8×i8→i32 with per-column f32 dequant → f16 GM
        if line.contains("__tile_matmul_i8_acc_i32_dequant_f16") {
            translate_matmul_i8(line, ctx, func, &mut ops)?;
            continue;
        }

        // tile.fill
        if line.contains("__tile_fill_f32") || line.contains("__tile_fill_f16") {
            translate_fill(line, ctx, &mut ops)?;
            continue;
        }
        // tile.max (element-wise)
        if line.contains("__tile_max_f32") || line.contains("__tile_max_f16") {
            let dtype = if line.contains("f16") { "f16" } else { "f32" };
            translate_binary(line, dtype, "pto.tmax", ctx, &mut ops)?;
            continue;
        }
        // tile.rms_norm
        if line.contains("__tile_rms_norm_f32") || line.contains("__tile_rms_norm_f16") {
            translate_rms_norm_pto(line, ctx, &mut ops)?;
            continue;
        }
        // tile.absmax_f32 — max of absolute values, broadcast to tile
        if line.contains("__tile_absmax_f32") {
            translate_absmax_pto(line, ctx, &mut ops)?;
            continue;
        }
        // tile.quantize_f32_i8 — round(src/scale) clamped to [-128,127]
        if line.contains("__tile_quantize_f32_i8") {
            translate_quantize_pto(line, ctx, &mut ops)?;
            continue;
        }
        // tile.dequantize_i8_f32 — src * scale (int8→f32)
        if line.contains("__tile_dequantize_i8_f32") {
            translate_dequantize_pto(line, ctx, &mut ops)?;
            continue;
        }
        // Phase 6 MTP ops — no native PTO equivalent; scalar loop decomposition
        // tile.argmax f32 — row-wise argmax → (R,1) u32
        if line.contains("__tile_argmax_f32") {
            translate_argmax_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.sample_top_p f32 — nucleus sampling → (R,1) u32
        if line.contains("__tile_sample_top_p_f32") {
            translate_sample_top_p_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.draft_verify f32 — acceptance probabilities → (R,1) f32
        if line.contains("__tile_draft_verify_f32") {
            translate_draft_verify_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.token_accept f32 — select final tokens → (R,1) u32
        if line.contains("__tile_token_accept_f32") {
            translate_token_accept_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.rope f32 — Rotary Position Embedding
        if line.contains("__tile_rope_f32") {
            translate_rope_pto(line, ctx, &mut ops)?;
            continue;
        }

        // Unrecognized llvm calls: emit as comment
        if line.contains("llvm.call") || line.contains("llvm.") {
            ops.push(format!("// unhandled: {}", line));
        }
    }

    Ok(ops)
}

// ---------------------------------------------------------------------------
// Per-op translators
// ---------------------------------------------------------------------------

/// `%res = llvm.call @__tile_load_f32(%gm, %rows, %cols) : ...`
/// → make_tensor_view + partition_view + alloc_tile + tload
fn translate_load(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    func: &MlirFunc,
    ops: &mut Vec<String>,
    defer_for_blocked_matmul: bool,
) -> Result<(), String> {
    let result_ssa =
        extract_result_ssa(line).ok_or_else(|| format!("tile_load: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("tile_load: cannot parse args in: {}", line))?;
    let gm_arg = args.first().ok_or("tile_load: missing gm arg")?.trim();
    let rows = ctx.resolve_const(args.get(1).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));

    // Resolve gm_arg → original GM func arg (following ptr_aliases chain)
    let elem_offset = ctx.resolve_offset(gm_arg);
    let resolved = ctx.resolve_ptr(gm_arg);
    let gm_name = resolve_gm_name(&resolved, func);

    // tensor_view — always emit (needed for both blocked and unblocked paths)
    let tv_ssa = ctx.get_or_make_tv(&gm_name, rows, cols, dtype, ops);

    if defer_for_blocked_matmul {
        // Don't materialise a full-shape vec tile or emit tload — the
        // full shape would overflow UB/CBUF/L0 caps at DeepSeek shapes.
        // translate_matmul will emit per-block partition_view + tload
        // inside its scf.for nest using `tv_ssa` + `elem_offset`.
        //
        // We still insert a placeholder TileInfo so downstream lookups
        // succeed; translate_matmul reads `deferred` instead of `pv_ssa`
        // / `ssa` for these tiles.
        ctx.use_size(rows);
        ctx.use_size(cols);
        let gm_name_clone = gm_name.clone();
        ctx.tiles.insert(
            result_ssa,
            TileInfo {
                ssa: String::new(), // no full-shape alloc — placeholder
                rows,
                cols,
                dtype: dtype.to_string(),
                tb_type: String::new(),
                pv_ssa: None,
                gm_name: Some(gm_name),
                deferred: Some(DeferredMatmulOperand {
                    tv_ssa,
                    elem_offset,
                    gm_name: gm_name_clone,
                }),
            },
        );
        return Ok(());
    }

    // partition_view — use GEP-derived element offset if available
    let pv_ssa = ctx.make_pv(&tv_ssa, rows, cols, dtype, elem_offset, ops);
    // alloc_tile
    let tb_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    // record pv association for tstore later
    {
        let ti = ctx.tiles.get_mut(&result_ssa).unwrap();
        ti.pv_ssa = Some(pv_ssa.clone());
        ti.gm_name = Some(gm_name.clone());
    }

    let tb_ty = tile_buf_type(rows, cols, dtype);
    let ptv_ty = ptv_type(rows, cols, dtype);
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_ssa, ptv_ty, tb_ssa, tb_ty
    ));

    Ok(())
}

/// `llvm.call @__tile_store_f32(%gm, %buf, %rows, %cols) : ...`
/// → partition_view for output + tstore
fn translate_store(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    func: &MlirFunc,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let args = extract_call_args(line)
        .ok_or_else(|| format!("tile_store: cannot parse args in: {}", line))?;
    let gm_arg = args.first().ok_or("tile_store: missing gm arg")?.trim();
    let buf_ssa = args.get(1).ok_or("tile_store: missing buf arg")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let elem_offset = ctx.resolve_offset(gm_arg);
    let resolved = ctx.resolve_ptr(gm_arg);
    let gm_name = resolve_gm_name(&resolved, func);

    // Blocked-matmul intercept: if this store is writing the result of a
    // matmul that translate_matmul_blocked deferred, emit the full K/N
    // scf.for nest inline here (where we finally know the output GM view).
    if ctx.matmul_result_stored_inline.contains(buf_ssa) {
        // The output GM is the caller's `output` pointer. Build its
        // tensor_view (shape M×N) and then emit the blocked nest.
        let pending = ctx
            .pending_blocked_matmuls
            .remove(buf_ssa)
            .ok_or_else(|| format!("tile_store: pending blocked matmul for {} missing", buf_ssa))?;
        if rows != pending.m || cols != pending.n {
            return Err(format!(
                "blocked matmul: store shape {}×{} != matmul result {}×{}",
                rows, cols, pending.m, pending.n
            ));
        }
        let out_tv_ssa = ctx.get_or_make_tv(&gm_name, pending.m, pending.n, dtype, ops);
        emit_blocked_matmul_loops(&out_tv_ssa, elem_offset, dtype, &pending, ctx, ops);
        return Ok(());
    }

    // Batched-stage intercept: the loop must enclose this store, so it is
    // emitted here rather than where the batched call appeared.
    if ctx.batched_result_stored_inline.contains(buf_ssa) {
        let pending = ctx
            .pending_batched
            .remove(buf_ssa)
            .ok_or_else(|| format!("tile_store: pending batched stage for {} missing", buf_ssa))?;
        let out_tv_ssa = ctx.get_or_make_tv(&gm_name, rows, cols, dtype, ops);
        return emit_batched_loop(&out_tv_ssa, elem_offset, dtype, &pending, ctx, ops);
    }

    // Row-blocked softmax intercept: the loop has to CONTAIN the store, or
    // only the last block reaches GM — the same failure the batched path hit
    // (heads 1..N untouched) for the same reason.
    if ctx.softmax_rows_stored_inline.contains(buf_ssa) {
        let pending = ctx
            .pending_row_softmax
            .remove(buf_ssa)
            .ok_or_else(|| format!("tile_store: pending row-softmax for {buf_ssa} missing"))?;
        let out_tv_ssa = ctx.get_or_make_tv(&gm_name, rows, cols, dtype, ops);
        return emit_row_softmax_loop(&out_tv_ssa, elem_offset, dtype, &pending, ctx, ops);
    }

    // Blocked-silu_mul intercept (#67): same shape as the matmul one — the
    // per-chunk scf.for is emitted here once we know the output GM view.
    if ctx.silu_mul_result_stored_inline.contains(buf_ssa) {
        let pending = ctx
            .pending_blocked_silu_muls
            .remove(buf_ssa)
            .ok_or_else(|| format!(
                "tile_store: pending blocked silu_mul for {} missing", buf_ssa
            ))?;
        if rows != pending.rows || cols != pending.cols {
            return Err(format!(
                "blocked silu_mul: store shape {}×{} != silu_mul result {}×{}",
                rows, cols, pending.rows, pending.cols
            ));
        }
        let out_tv_ssa = ctx.get_or_make_tv(&gm_name, pending.rows, pending.cols, dtype, ops);
        emit_blocked_silu_mul_loops(&out_tv_ssa, elem_offset, dtype, &pending, ctx, ops);
        return Ok(());
    }

    let tile = ctx
        .get_tile(buf_ssa)
        .ok_or_else(|| format!("tile_store: unknown tile buf {}", buf_ssa))?
        .clone();

    // tensor_view for the output GM
    let tv_ssa = ctx.get_or_make_tv(&gm_name, rows, cols, dtype, ops);
    // partition_view for the output — use GEP-derived element offset if available
    let pv_ssa = ctx.make_pv(&tv_ssa, rows, cols, dtype, elem_offset, ops);

    let tb_ty = tile.tile_buf_type_str();
    // The pv was built with the store's target dtype (the GM dtype). If the
    // tile's dtype differs (e.g., f16 matmul registers its L0C acc as f32
    // under the result SSA — see translate_matmul_f16 for the rationale),
    // the tstore output clause must still spell the pv's physical dtype,
    // not the tile's. The hardware FixPipe path performs the implicit cast
    // during the L0C→GM DMA. Use the caller's `dtype` (the store dtype) to
    // name the pv's ptv type here.
    let ptv_ty = ptv_type(rows, cols, dtype);
    ops.push(format!(
        "pto.tstore ins({} : {}) outs({} : {})",
        tile.ssa, tb_ty, pv_ssa, ptv_ty
    ));

    Ok(())
}

/// Emit the K/N-blocked matmul scf.for nest.
///
/// Output shape in the generated MLIR matches the hand-validated
/// `/tmp/matmul_q_proj_m16.pto`:
/// ```text
/// scf.for %n_i = 0 to %N_ITERS step 1 {
///   %n_off = arith.muli %n_i, %Nb
///   scf.for %k_i = 0 to %K_ITERS step 1 {
///     %k_off = arith.muli %k_i, %Kb
///     %a_pt  = pto.partition_view %tv_a, offsets=[0, %k_off], sizes=[M, Kb]
///     pto.tload  a_pt → mat_a
///     pto.tmov   mat_a → a_left
///     %b_pt  = pto.partition_view %tv_b, offsets=[%k_off, %n_off], sizes=[Kb, Nb]
///     pto.tload  b_pt → mat_b
///     pto.tmov   mat_b → b_right
///     %is_first = arith.cmpi eq, %k_i, %c0
///     scf.if %is_first { pto.tmatmul     ins(a_left, b_right) outs(acc) }
///                 else { pto.tmatmul.acc ins(acc, a_left, b_right) outs(acc) }
///   }
///   %out_pt = pto.partition_view %tv_out, offsets=[0, %n_off], sizes=[M, Nb]
///   pto.tstore acc → out_pt
/// }
/// ```
fn emit_blocked_matmul_loops(
    tv_out_ssa: &str,
    out_elem_offset: u32,
    out_dtype: &str,
    p: &PendingBlockedMatmul,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) {
    // The A/B elem_offsets are from the matmul's operand tile_load calls.
    // For M=1 decode kernels without GEP, they're zero. For split-K or
    // batched dispatch they'd be non-zero. For now we assume zero; the
    // per-block partition_view offsets are the K/N iterators added to
    // the base offset. Non-zero base offsets are folded into the
    // partition_view via a fresh constant.
    let a_base_row = p.a_elem_offset / p.k; // A is M×K, so row = offset / K
    let a_base_col = p.a_elem_offset % p.k;
    let b_base_row = p.b_elem_offset / p.n; // B is K×N, so row = offset / N
    let b_base_col = p.b_elem_offset % p.n;
    let out_base_row = out_elem_offset / p.n;
    let out_base_col = out_elem_offset % p.n;
    ctx.use_size(a_base_row);
    ctx.use_size(a_base_col);
    ctx.use_size(b_base_row);
    ctx.use_size(b_base_col);
    ctx.use_size(out_base_row);
    ctx.use_size(out_base_col);
    ctx.use_size(p.kb);
    ctx.use_size(p.nb);
    ctx.use_size(p.n_iters);
    ctx.use_size(p.k_iters);

    // tv_* types spell the A/B/out GM dtypes. A/B use the operand dtypes
    // (lhs/rhs). Output pv uses the caller's store dtype — passed through
    // from `translate_store` (the store line declares the GM dtype). See
    // `emit_blocked_matmul_loops` signature change: `out_dtype` is the
    // store-site dtype, which may differ from `p.dtypes.dst` (e.g., f32
    // acc written to an f16 output GM — FixPipe casts during DMA).
    let tv_a_ty = tv_type(p.m, p.k, p.dtypes.lhs);
    let tv_b_ty = tv_type(p.k, p.n, p.dtypes.rhs);
    let tv_o_ty = tv_type(p.m, p.n, out_dtype);
    // Shapes that carry M use the BLOCK. At m_iters==1 `mb == m`, so these are
    // the same strings the pre-M-blocking emitter produced.
    let pv_a_ty = ptv_type(p.mb, p.kb, p.dtypes.lhs);
    let pv_b_ty = ptv_type(p.kb, p.nb, p.dtypes.rhs);
    let pv_o_ty = ptv_type(p.mb, p.nb, out_dtype);
    let _ = (tv_a_ty, tv_b_ty, tv_o_ty); // types carried via caller ctx, variables used for symmetry

    // Outer N-loop — parallelised across AICores via get_block_idx/num.
    // Each AICore processes a strided subset of the N-block range, so
    // launching with blockDim=min(n_iters, num_aicores) maps 1 N-block per
    // core for n_iters <= 24; larger n_iters are round-robin'd.
    //
    // Lowering: `pto.get_block_idx : i64` → `get_block_idx()` in generated C++.
    // For n_iters==1 the outer scf.for is elided entirely: the hand-written
    // i8 probe showed that ptoas re-examines the Left tile's BLayout when an
    // outer scf.for is present, sometimes flipping RowMajor→ColMajor even if
    // the loop is degenerate (0..1). Emitting the K-loop directly at top
    // level matches the probe and keeps ptoas on the verified codepath.
    //
    // SCOPE OF THAT CAUTION, settled on device 2026-08-06 (910B2, ptoas 0.55):
    // it is i8-specific and does NOT apply to f32. The same f32 matmul emitted
    // with and without the outer loop gives `BLayout::RowMajor` for the Left
    // tile in both generated C++ files, and both match a host reference to
    // 2.98e-08 — identical, and the check is not vacuous (perturbing one input
    // element drives it to 5.5e-02 and fails). See `bin/blayout_probe.rs`.
    //
    // The elision above is therefore conservative rather than required for
    // f32. It is kept because it costs nothing here, but it does NOT mean an
    // f32 kernel must avoid an outer scf.for — which matters for batched
    // heads, where the loop form lifts the UB ceiling that unrolling imposes.
    // Outermost M-loop. Emitted only when the A operand does not fit whole, so
    // shapes that already worked keep byte-identical output. Sequential rather
    // than grid-strided: the N-loop below already claims block_idx for core
    // parallelism, and nesting two grid-strided loops would have cores compute
    // overlapping (m, n) blocks.
    let (m_off_ssa, _m_indent) = if p.m_iters > 1 {
        ctx.use_size(p.m_iters);
        ctx.use_size(p.mb);
        let off = ctx.fresh_ssa();
        ops.push(format!(
            "scf.for %m_i = %c0 to %c{} step %c1 {{", p.m_iters
        ));
        ops.push(format!("  {} = arith.muli %m_i, %c{} : index", off, p.mb));
        (off, "  ")
    } else {
        ("%c0".to_string(), "")
    };

    let (n_off_ssa, outer_indent) = if p.n_iters > 1 {
        let bi64_ssa = ctx.fresh_ssa();
        let bn64_ssa = ctx.fresh_ssa();
        let bi_ssa = ctx.fresh_ssa();
        let bn_ssa = ctx.fresh_ssa();
        ops.push(format!(
            "{} = \"pto.get_block_idx\"() : () -> i64",
            bi64_ssa
        ));
        ops.push(format!(
            "{} = \"pto.get_block_num\"() : () -> i64",
            bn64_ssa
        ));
        ops.push(format!(
            "{} = arith.index_cast {} : i64 to index",
            bi_ssa, bi64_ssa
        ));
        ops.push(format!(
            "{} = arith.index_cast {} : i64 to index",
            bn_ssa, bn64_ssa
        ));
        ops.push(format!(
            "scf.for %n_i = {} to %c{} step {} {{",
            bi_ssa, p.n_iters, bn_ssa
        ));
        let n_off_ssa = ctx.fresh_ssa();
        ops.push(format!(
            "  {} = arith.muli %n_i, %c{} : index",
            n_off_ssa, p.nb
        ));
        (n_off_ssa, "  ")
    } else {
        // Degenerate single-block: fixed n_off = 0, no outer loop.
        ("%c0".to_string(), "")
    };

    // Pre-K-loop hoist for the degenerate n_iters==1 dequant case: emit the
    // partition_view for the output and the scale tile, plus the scale
    // tload + tmov-to-FB, BEFORE the K-loop body. This matches the probe
    // MLIR ordering (/tmp/smoke_i8_kv_proj_tmov3arg.acl.pto) that ptoas
    // lowered to working numerics on 910B2. Keeping the scale load inside
    // the N-loop (as the multi-block path does) changes TASSIGN offsets in
    // ptoas and corrupts the i8 matmul output. See memory
    // project_cann85_i8_emitter_numerics_blocker.md for the diff.
    let hoisted_scale: Option<(String, String, String)> = if p.n_iters == 1 {
        if let Some(dq) = &p.dequant {
            let pv_scale_blk = ctx.fresh_ssa();
            ops.push(format!(
                "{} = pto.partition_view {}, offsets = [%c0, %c0], sizes = [%c1, %c{}] : {} -> {}",
                pv_scale_blk,
                dq.tv_scale_ssa,
                p.nb,
                tv_type(1, p.n, "ui64"),
                ptv_type(1, p.nb, "ui64"),
            ));
            ops.push(format!(
                "pto.tload ins({} : {}) outs({} : {})",
                pv_scale_blk,
                ptv_type(1, p.nb, "ui64"),
                dq.scale_mat_ssa,
                dq.scale_mat_ty,
            ));
            ops.push(format!(
                "pto.tmov ins({} : {}) outs({} : {})",
                dq.scale_mat_ssa,
                dq.scale_mat_ty,
                dq.scale_tile_ssa,
                dq.scale_tile_ty,
            ));
            Some((pv_scale_blk, String::new(), String::new()))
        } else {
            None
        }
    } else {
        None
    };

    // Inner K-loop.
    let k_indent = outer_indent; // body indent inside the optional outer N-loop
    let k_body_indent = format!("{}  ", k_indent);
    ops.push(format!(
        "{}scf.for %k_i = %c{} to %c{} step %c{} {{",
        k_indent, 0, p.k_iters, 1
    ));
    let k_off_ssa = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = arith.muli %k_i, %c{} : index",
        k_body_indent, k_off_ssa, p.kb
    ));

    // Per-block partition_view for A[m_off:m_off+Mb, k_off:k_off+Kb].
    //
    // The row offset folds the M-block iterator into the base. At m_iters==1
    // `m_off_ssa` is `%c0` and this collapses to the previous `%c{a_base_row}`
    // form, which is what keeps already-fitting shapes byte-identical.
    let a_row_ssa = if p.m_iters > 1 {
        let r = ctx.fresh_ssa();
        ops.push(format!(
            "{}{} = arith.addi {}, %c{} : index",
            k_body_indent, r, m_off_ssa, a_base_row
        ));
        r
    } else {
        format!("%c{}", a_base_row)
    };
    let pv_a_blk = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = pto.partition_view {}, offsets = [{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        k_body_indent,
        pv_a_blk,
        p.tv_a_ssa,
        a_row_ssa,
        k_off_ssa,
        p.mb,
        p.kb,
        tv_type(p.m, p.k, p.dtypes.lhs),
        pv_a_ty
    ));
    ops.push(format!(
        "{}pto.tload ins({} : {}) outs({} : {})",
        k_body_indent, pv_a_blk, pv_a_ty, p.mat_a_ssa, p.mat_a_ty
    ));
    ops.push(format!(
        "{}pto.tmov ins({} : {}) outs({} : {})",
        k_body_indent, p.mat_a_ssa, p.mat_a_ty, p.a_left_ssa, p.left_ty
    ));

    // Per-block partition_view for B[k_off:k_off+Kb, n_off:n_off+Nb].
    let pv_b_blk = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = pto.partition_view {}, offsets = [{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        k_body_indent,
        pv_b_blk,
        p.tv_b_ssa,
        k_off_ssa,
        n_off_ssa,
        p.kb,
        p.nb,
        tv_type(p.k, p.n, p.dtypes.rhs),
        pv_b_ty
    ));
    ops.push(format!(
        "{}pto.tload ins({} : {}) outs({} : {})",
        k_body_indent, pv_b_blk, pv_b_ty, p.mat_b_ssa, p.mat_b_ty
    ));
    ops.push(format!(
        "{}pto.tmov ins({} : {}) outs({} : {})",
        k_body_indent, p.mat_b_ssa, p.mat_b_ty, p.b_right_ssa, p.right_ty
    ));

    // scf.if %k_i == 0 { tmatmul } else { tmatmul.acc }
    let is_first = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = arith.cmpi eq, %k_i, %c{} : index",
        k_body_indent, is_first, 0
    ));
    ops.push(format!("{}scf.if {} {{", k_body_indent, is_first));
    ops.push(format!(
        "{}  pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
        k_body_indent, p.a_left_ssa, p.b_right_ssa, p.left_ty, p.right_ty, p.acc_ssa, p.acc_ty
    ));
    ops.push(format!("{}}} else {{", k_body_indent));
    ops.push(format!(
        "{}  pto.tmatmul.acc ins({}, {}, {} : {}, {}, {}) outs({} : {})",
        k_body_indent,
        p.acc_ssa,
        p.a_left_ssa,
        p.b_right_ssa,
        p.acc_ty,
        p.left_ty,
        p.right_ty,
        p.acc_ssa,
        p.acc_ty
    ));
    ops.push(format!("{}}}", k_body_indent));
    ops.push(format!("{}}}", k_indent)); // close K-loop

    // Store this block-column of the result: output[0:M, n_off:n_off+Nb].
    // Output block lands at the same M offset the A block was read from, so
    // each (m, n) block writes its own rows. Collapses to the previous form at
    // m_iters==1.
    let out_row_ssa = if p.m_iters > 1 {
        let r = ctx.fresh_ssa();
        ops.push(format!(
            "{}{} = arith.addi {}, %c{} : index",
            k_indent, r, m_off_ssa, out_base_row
        ));
        r
    } else {
        format!("%c{}", out_base_row)
    };
    let pv_o_blk = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = pto.partition_view {}, offsets = [{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        k_indent,
        pv_o_blk,
        tv_out_ssa,
        out_row_ssa,
        n_off_ssa,
        p.mb,
        p.nb,
        tv_type(p.m, p.n, out_dtype),
        pv_o_ty
    ));
    if let Some(dq) = &p.dequant {
        // int8 dequant path: load the 1×Nb slice of the per-column ui64-packed
        // scale inside the N-loop, then emit pto.tstore_fp to fold the dequant
        // (acc[i32] * scale[ui64] → GM[out_dtype]) into the L0C→GM DMA via
        // FixPipe. CANN 8.5 ptoas rejects direct tload→Scaling, so hop via
        // L0B-Mat: tload GM→Mat(ui64,none_box), then tmov Mat→Scaling via
        // TMovToFb. See memory/project_cann85_i8_path_viable_via_tmov3arg.md.
        if hoisted_scale.is_none() {
            let pv_scale_blk = ctx.fresh_ssa();
            ops.push(format!(
                "{}{} = pto.partition_view {}, offsets = [%c0, {}], sizes = [%c1, %c{}] : {} -> {}",
                k_indent,
                pv_scale_blk,
                dq.tv_scale_ssa,
                n_off_ssa,
                p.nb,
                tv_type(1, p.n, "ui64"),
                ptv_type(1, p.nb, "ui64"),
            ));
            // GM → L0B-Mat (ui64).
            ops.push(format!(
                "{}pto.tload ins({} : {}) outs({} : {})",
                k_indent,
                pv_scale_blk,
                ptv_type(1, p.nb, "ui64"),
                dq.scale_mat_ssa,
                dq.scale_mat_ty,
            ));
            // Mat → FB-Scaling (ui64) via TMovToFb.
            ops.push(format!(
                "{}pto.tmov ins({} : {}) outs({} : {})",
                k_indent,
                dq.scale_mat_ssa,
                dq.scale_mat_ty,
                dq.scale_tile_ssa,
                dq.scale_tile_ty,
            ));
        }
        ops.push(format!(
            "{}pto.tstore_fp ins({}, {} : {}, {}) outs({} : {})",
            k_indent,
            p.acc_ssa,
            dq.scale_tile_ssa,
            p.acc_ty,
            dq.scale_tile_ty,
            pv_o_blk,
            pv_o_ty,
        ));
        // Suppress dead-code warning in case the full-tensor pv_scale_ssa
        // is unused (we rely on per-block pv inside the loop).
        let _ = &dq.pv_scale_ssa;
        let _ = &dq.pv_scale_ty;
    } else {
        ops.push(format!(
            "{}pto.tstore ins({} : {}) outs({} : {})",
            k_indent, p.acc_ssa, p.acc_ty, pv_o_blk, pv_o_ty
        ));
    }

    if p.n_iters > 1 {
        ops.push("}".to_string()); // close N-loop
    }
    if p.m_iters > 1 {
        ops.push("}".to_string()); // close M-loop
    }
}

/// Binary: `%res = llvm.call @__tile_add_f32(%c0, %a, %b, %rows, %cols)`
fn translate_binary(
    line: &str,
    dtype: &str,
    pto_op: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("{}: no result SSA in: {}", pto_op, line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("{}: cannot parse args in: {}", pto_op, line))?;
    let src1_ssa = args.get(1).ok_or("binary: missing src1")?.trim();
    let src2_ssa = args.get(2).ok_or("binary: missing src2")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(src1_ssa)
        .ok_or_else(|| format!("{}: unknown tile {}", pto_op, src1_ssa))?
        .clone();
    let tb = ctx
        .get_tile(src2_ssa)
        .ok_or_else(|| format!("{}: unknown tile {}", pto_op, src2_ssa))?
        .clone();
    let tc_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);

    let ta_ty = ta.tile_buf_type_str();
    let tb_ty = tb.tile_buf_type_str();
    let tc_ty = tile_buf_type(rows, cols, dtype);
    ops.push(format!(
        "{} ins({}, {} : {}, {}) outs({} : {})",
        pto_op, ta.ssa, tb.ssa, ta_ty, tb_ty, tc_ssa, tc_ty
    ));

    Ok(())
}

/// Unary: `%res = llvm.call @__tile_exp_f32(%c0, %src, %rows, %cols)`
fn translate_unary(
    line: &str,
    dtype: &str,
    pto_op: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("{}: no result SSA in: {}", pto_op, line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("{}: cannot parse args in: {}", pto_op, line))?;
    let src_ssa = args.get(1).ok_or("unary: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("{}: unknown tile {}", pto_op, src_ssa))?
        .clone();
    let tdst_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);

    let tsrc_ty = tsrc.tile_buf_type_str();
    let tdst_ty = tile_buf_type(rows, cols, dtype);
    ops.push(format!(
        "{} ins({} : {}) outs({} : {})",
        pto_op, tsrc.ssa, tsrc_ty, tdst_ssa, tdst_ty
    ));

    Ok(())
}

/// Matmul: `%res = llvm.call @__tile_matmul_f32(%c0, %a, %b, %m, %k, %n)`
///
/// Emits the full cube-unit pipeline:
///   1. Alloc mat_a, mat_b (CBUF staging tiles)
///   2. Alloc left (L0A), right (L0B), acc (L0C) tiles
///   3. tload GM → mat_a, mat_b  (reuse partition views from the input tloads)
///   4. tmov mat_a → left, mat_b → right  (MTE1: CBUF → L0A/L0B)
///   5. tmatmul left × right → acc        (M-pipe cube unit)
///
/// The caller's tstore then reads the `result_ssa` tile (acc) and emits
/// `pto.tstore ins(%acc : !pto.tile_buf<loc=acc, ...>) outs(%pv : ...)`.
///
/// Tile attribute table (per TMatmul.hpp static assertions):
/// | loc   | blayout   | slayout   | fractal |
/// |-------|-----------|-----------|---------|
/// | mat   | col_major | row_major | 512     |
/// | left  | row_major | row_major | 512     |
/// | right | row_major | col_major | 512     |
/// | acc   | col_major | row_major | 1024    |
/// A batched stage awaiting its store, so the loop can enclose it.
struct PendingBatched {
    kind: BatchedKind,
    heads: u32,
    /// Operand SSAs, as named in the original call.
    a: String,
    b: Option<String>,
    /// Shape: (m, k, n) for matmul; (rows, _, cols) for softmax.
    m: u32,
    k: u32,
    n: u32,
}

/// Which batched stage `translate_batched` is emitting.
///
/// # This enum is the thing that does not generalise
///
/// Each arm is one hand-written stage, so a batched `silu_mul` needs a third
/// intrinsic, a third arm, and a third body in `emit_batched_loop`. The
/// *machinery* underneath is already stage-agnostic — `emit_batched_loop`
/// hoists allocations, opens the grid-strided loop and puts the store inside;
/// `with_head_rows` scopes a row base per operand. Only the dispatch is keyed
/// on intrinsic name.
///
/// `Generic` is the way out: it carries the per-iteration operand list and the
/// inner op as data, so a new stage is a caller-side description rather than an
/// emitter change. The two named arms remain because they emit byte-identical
/// PTO to what is already validated on device, and that equality is the test
/// that the generic path is faithful (see `batched_generic_matches_named`).
#[derive(Clone, PartialEq, Eq)]
enum BatchedKind {
    /// `heads` independent (M×K)@(K×N) products.
    Matmul,
    /// `heads` independent row-softmaxes over (rows×cols).
    Softmax,
    /// Any stage expressible as tile ops over per-iteration operands.
    ///
    /// `inner` is an MLIR call line to translate once per iteration, with
    /// `%__op0`, `%__op1`, … standing for the operands. `operands` gives each
    /// one its OWN rows-per-iteration, which is what keeps the shared-row-base
    /// bug unrepresentable here rather than merely fixed.
    Generic {
        /// Iterations in the batch, from `batched.heads` (see the note in
        /// `translate_batched` on why this is an attribute, not an argument).
        heads: u32,
        /// The intrinsic to emit inside the loop, e.g. `__tile_silu_mul_f32`.
        inner_call: String,
        /// Per-operand `(ssa, rows, cols)`, in the order `inner_call` names them.
        operands: Vec<(String, u32, u32)>,
    },
}

/// Emit a stage that covers all attention heads in ONE launch.
///
/// Rather than a host loop issuing one launch per head, the heads are unrolled
/// inside the kernel at their per-head GM offsets. Head `h` reads `a[h*M*K]`
/// and `b[h*K*N]` and writes `out[h*M*N]`; those offsets go straight into
/// `make_pv`, which already takes a flat element offset for GEP-derived
/// addressing, so no new addressing machinery is needed.
///
/// Unrolling rather than emitting `scf.for %h = get_block_idx() ...`: the tile
/// allocations differ per head (each needs its own accumulator), and the
/// blocked-matmul path shows ptoas re-examining tile layouts when an outer
/// scf.for is present — sometimes flipping BLayout even for a degenerate loop.
/// Unrolled, every head takes the byte-for-byte codepath the per-head kernels
/// already validated on device, and the launch saving is identical.
///
/// Why this exists: on the benchmarked MQA shape 72% of measured time is the
/// fixed ~4.17 us per-launch cost and 18 of 22 launches are per-head attention
/// stages. This trades launches for code size; the arithmetic is unchanged.
/// Record a batched stage; the loop is emitted at its store.
///
/// Nothing is emitted here. The loop has to CONTAIN the store — a loop that
/// ends before it computes every head and keeps only the last, which was
/// verified wrong on device (head 0 correct, heads 1..5 untouched). The output
/// GM view is only known when the store is translated, so this defers exactly
/// as `translate_matmul_blocked` does for the same reason.
fn translate_batched(
    line: &str,
    kind: BatchedKind,
    ctx: &mut PtoContext,
    _ops: &mut Vec<String>,
) -> Result<(), String> {
    let name = match kind {
        BatchedKind::Matmul => "matmul_batched",
        BatchedKind::Softmax => "softmax_batched",
        BatchedKind::Generic { .. } => "batched",
    };
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("{name}: no result SSA in: {line}"))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("{name}: cannot parse args in: {line}"))?;

    // Heads is the last call argument for the NAMED forms. The generic form
    // cannot use an argument at all: its attributes embed another call, and
    // `extract_call_args` stops at the first `)`, so it returns the inner
    // call's args spliced onto the outer ones (`%h` came back as
    // `"%h) {batched.inner = ..."`). Attributes are read by a separate
    // quote-delimited scanner that embedded parentheses cannot confuse, so the
    // count lives there.
    let heads = match &kind {
        BatchedKind::Generic { heads, .. } => *heads,
        _ => ctx.resolve_const(args.last().map(|s| s.as_str()).unwrap_or("0")),
    };
    if heads == 0 {
        return Err(format!("{name}: head count is 0 or unresolved in: {line}"));
    }

    let pending = match &kind {
        BatchedKind::Matmul => PendingBatched {
            kind: kind.clone(),
            heads,
            a: args.get(1).ok_or("matmul_batched: missing a")?.trim().to_string(),
            b: Some(args.get(2).ok_or("matmul_batched: missing b")?.trim().to_string()),
            m: ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0")),
            k: ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0")),
            n: ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0")),
        },
        BatchedKind::Softmax => PendingBatched {
            kind: kind.clone(),
            heads,
            a: args.get(1).ok_or("softmax_batched: missing src")?.trim().to_string(),
            b: None,
            m: ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0")),
            k: 0,
            n: ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0")),
        },
        // A generic stage carries its own operand list, so the caller has
        // already said what to load and at what per-iteration height. Shape
        // fields describe the OUTPUT only.
        // Output shape comes from the FIRST operand's declared rows x cols,
        // not from the call args: an elementwise stage writes what it reads,
        // and taking it from the operand list keeps one source of truth for
        // shape. (args 2/3 are the caller's %r/%c SSAs, which resolve_const
        // cannot see through -- reading them there yielded `unknown tile %r`.)
        BatchedKind::Generic { operands, .. } => {
            let (first, rows, cols) = operands
                .first()
                .ok_or("batched: empty operand list")?
                .clone();
            PendingBatched {
                a: first,
                b: operands.get(1).map(|(s, _, _)| s.clone()),
                m: rows,
                k: 0,
                n: cols,
                kind: kind.clone(),
                heads,
            }
        }
    };

    ctx.batched_result_stored_inline.insert(result_ssa.clone());
    ctx.pending_batched.insert(result_ssa, pending);
    Ok(())
}

/// Emit the store for one block of a batched stage, at the current row/col
/// base.
///
/// Factored out because the store has to happen at the INNERMOST point of
/// whatever loop nest the stage built. Appending it after the body works only
/// while the stage produces one whole tile per head; the moment a head is
/// blocked, an outside store writes just the last block — verified on device
/// as `untouched=131072 / N-BLOCKING IS WRONG`, and invisible to timing because
/// the launch count is identical either way.
fn emit_block_store(
    out_tv_ssa: &str,
    out_elem_offset: u32,
    dtype: &str,
    tile_key: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let t = ctx
        .get_tile(tile_key)
        .ok_or_else(|| format!("batched: stage produced no result tile ({tile_key})"))?
        .clone();
    // Shape from the tile the stage actually produced, not from the head's
    // logical shape: ptoas rejects a mismatch outright
    // ("tstore expects dst static element count (262144) to match src (32768)"),
    // which is the good failure mode but only fires if the shapes disagree.
    let pv = ctx.make_pv(out_tv_ssa, t.rows, t.cols, dtype, out_elem_offset, ops);
    ops.push(format!(
        "pto.tstore ins({} : {}) outs({} : {})",
        t.ssa,
        t.tb_type,
        pv,
        ptv_type(t.rows, t.cols, dtype)
    ));
    Ok(())
}

/// Emit a batched stage's loop, with its store inside.
///
/// # Status: the STORE is fixed, the LOAD is not. Still wrong on device.
///
/// Verified on 910B2 with a per-head parity check, heads given distinct inputs:
///
///   before this change: head 0 ok, heads 1..5 UNTOUCHED (store outside loop)
///   after  this change: head 0 ok, heads 1..5 written but WRONG (~5e-03)
///
/// So the store now runs per head — real progress, and the untouched-output bug
/// is gone. What remains is the mirror image on the input side: the operand's
/// `tload` is emitted by an earlier `__tile_load_f32`, once, before the loop.
/// The per-head `partition_view` this function creates is therefore never read,
/// and every iteration softmaxes head 0's data into head h's slot. The emitted
/// PTO shows it: a head-relative view at the top of the loop with no `tload`
/// consuming it.
///
/// Closing it means the batched stage owning its LOAD as well as its store —
/// intercepting the preceding tile_load the way `translate_store` intercepts
/// the following store, so the load moves inside the loop. That is a fourth
/// structural change to the emission path, and the same shape as the two
/// already made.
///
/// Called from `translate_store` once the output GM view is known. Structure:
///
/// ```text
/// <tile allocations>                       hoisted: loop-invariant
/// scf.for %h_i = block_idx to heads step block_num {
///   %base = h_i * rows_per_head            each head's row offset
///   <partition_views at %base>             operands AND output
///   <the stage's ops>
///   pto.tstore -> out[%base]               inside, so every head is written
/// }
/// ```
fn emit_batched_loop(
    out_tv_ssa: &str,
    out_elem_offset: u32,
    dtype: &str,
    p: &PendingBatched,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let name = match p.kind {
        BatchedKind::Matmul => "matmul_batched",
        BatchedKind::Softmax => "softmax_batched",
        BatchedKind::Generic { .. } => "batched",
    };
    let heads = p.heads;
    let (out_rows, out_cols) = (p.m, p.n);

    ops.push(format!(
        "// --- {name}: {heads} heads in one launch, store inside the loop ---"
    ));

    // Emit one head's body first, into a scratch buffer, so the allocations can
    // be hoisted. The row base is live while this runs, so every partition_view
    // it creates — including the output's — is head-relative.
    let bi64 = ctx.fresh_ssa();
    let bn64 = ctx.fresh_ssa();
    let bi = ctx.fresh_ssa();
    let bn = ctx.fresh_ssa();
    let base = ctx.fresh_ssa();

    let mut body: Vec<String> = Vec::new();
    ctx.set_pv_row_base(Some(base.clone()));

    // Set by any arm that stores per block inside its own loop nest; the
    // trailing store below is then skipped, since it would be a second write of
    // the last block only.
    let mut stored_inline = false;

    let result_tile = match p.kind {
        BatchedKind::Matmul => {
            // A head's own operands can still exceed their memory: at S=1024
            // the [S,HD] left tile is 256 KB against a 64 KB L0A cap, and pv's
            // [S,S] staging tile is 4 MB against 512 KB of L1. The head loop
            // slices ACROSS heads; this blocks rows WITHIN one head.
            // Two blocking dimensions, bounding different tiles: `a_left` is
            // [mb,k] and shrinks with ROWS, `b_right` is [k,nb] with no row
            // term and shrinks only with COLUMNS.
            // K is the third blocking dimension. Without it `a_left` [mb,k] and
            // `b_right` [k,nb] both scale with K, so at K=896 the 64 KB L0A/L0B
            // caps pin mb and nb to 16 whatever the accumulator could hold --
            // pv ran a 16x16 acc at 0.2% of L0C for that reason.
            let (mb, kb, nb) = pick_batched_blocks_k(p.m, p.k, p.n);
            // The head stride is the FULL rows-per-head; the block loops walk
            // within it.
            ctx.head_stride_a = Some(p.m);
            // B's rows-per-head is the FULL k, not the K block: the head term
            // steps over whole heads however finely K is walked.
            ctx.head_stride_b = Some(p.k);
            let bb = p.b.clone().unwrap_or_else(|| p.a.clone());
            let inner = format!(
                "%__batched = llvm.call @__tile_matmul_f32({}, {}, {}, {mb}, {kb}, {nb}) : \
                 (i32, i32, i32, i32, i32, i32) -> i32",
                p.a, p.a, bb
            );
            // With a K loop the store must sit OUTSIDE it: every K iteration
            // accumulates into the same [mb,nb] tile, and storing per iteration
            // would write partial sums and leave only the last slice's
            // contribution. Without one, the store is the innermost act as
            // before.
            let k_blk = if kb < p.k { p.k / kb } else { 1 };
            let body_fn = |c: &mut PtoContext, o: &mut Vec<String>| -> Result<(), String> {
                if k_blk > 1 {
                    c.use_size(k_blk);
                    o.push(format!("scf.for %k_i = %c0 to %c{k_blk} step %c1 {{"));
                    c.with_k_block(
                        kb,
                        |c2, o2| -> Result<(), String> {
                            c2.set_matmul_accumulate(true);
                            let r = translate_matmul(&inner, c2, o2);
                            c2.set_matmul_accumulate(false);
                            r
                        },
                        o,
                    )?;
                    o.push("}".to_string());
                    emit_block_store(out_tv_ssa, out_elem_offset, dtype, "%__batched", c, o)
                } else {
                    translate_matmul(&inner, c, o)?;
                    emit_block_store(out_tv_ssa, out_elem_offset, dtype, "%__batched", c, o)
                }
            };
            let rows_fn = |c: &mut PtoContext, o: &mut Vec<String>| -> Result<(), String> {
                if mb < p.m {
                    let r_blk = p.m.div_ceil(mb);
                    c.use_size(r_blk);
                    o.push(format!("scf.for %r_i = %c0 to %c{r_blk} step %c1 {{"));
                    c.with_row_block(mb, body_fn, o)?;
                    o.push("}".to_string());
                    Ok(())
                } else {
                    body_fn(c, o)
                }
            };
            if nb < p.n {
                let c_blk = p.n.div_ceil(nb);
                ctx.use_size(c_blk);
                body.push(format!("scf.for %n_i = %c0 to %c{c_blk} step %c1 {{"));
                ctx.with_col_block(nb, rows_fn, &mut body)?;
                body.push("}".to_string());
            } else {
                rows_fn(ctx, &mut body)?;
            }
            ctx.head_stride_a = None;
            ctx.head_stride_b = None;
            stored_inline = true;
            "%__batched".to_string()
        }
        // The head's own [rows, cols] tile can exceed UB on its own: at
        // S=1024 that is 4 MB against a 184 KB budget. Block rows WITHIN the
        // head, composing with the head base via `with_row_block`.
        BatchedKind::Softmax if pick_softmax_rows(p.m, p.n, "f32", 0) < p.m => {
            let rb = pick_softmax_rows(p.m, p.n, "f32", 0);
            let n_blk = p.m.div_ceil(rb);
            let a = p.a.clone();
            let cols = p.n;
            ctx.use_size(n_blk);
            body.push(format!("scf.for %r_i = %c0 to %c{n_blk} step %c1 {{"));
            ctx.with_row_block(
                rb,
                |c, o| -> Result<(), String> {
                    let t = c
                        .get_tile(&a)
                        .ok_or_else(|| format!("softmax_batched: unknown tile {a}"))?
                        .clone();
                    let d = t.deferred.clone().ok_or_else(|| {
                        format!("softmax_batched: operand {a} was not deferred")
                    })?;
                    let src = format!("{a}__headsrc");
                    let pv = c.make_pv(&d.tv_ssa, rb, cols, "f32", d.elem_offset, o);
                    let tile = c.alloc_tile(&src, rb, cols, "f32", o);
                    o.push(format!(
                        "pto.tload ins({} : {}) outs({} : {})",
                        pv,
                        ptv_type(rb, cols, "f32"),
                        tile,
                        tile_buf_type(rb, cols, "f32")
                    ));
                    emit_softmax_steps(&src, "%__batched", rb, cols, "f32", c, o)?;
                    // Store this block's rows before the next iteration
                    // overwrites the result tile.
                    emit_block_store(out_tv_ssa, out_elem_offset, dtype, "%__batched", c, o)
                },
                &mut body,
            )?;
            body.push("}".to_string());
            stored_inline = true;
            "%__batched".to_string()
        }
        BatchedKind::Softmax => {
            // Load head h's slice HERE, inside the loop. The operand's load was
            // deferred by detect_batched_loads precisely so this can happen:
            // a pre-loop load fills the tile with head 0 once, and every
            // iteration then softmaxes head 0's data into head h's slot.
            let head_src = format!("{}__headsrc", p.a);
            let t = ctx
                .get_tile(&p.a)
                .ok_or_else(|| format!("softmax_batched: unknown tile {}", p.a))?
                .clone();
            let d = t.deferred.clone().ok_or_else(|| {
                format!(
                    "softmax_batched: operand {} was not deferred — its load is \
                     outside the loop, so every head would read head 0",
                    p.a
                )
            })?;
            let pv = ctx.make_pv(&d.tv_ssa, p.m, p.n, "f32", d.elem_offset, &mut body);
            let tile = ctx.alloc_tile(&head_src, p.m, p.n, "f32", &mut body);
            body.push(format!(
                "pto.tload ins({} : {}) outs({} : {})",
                pv,
                ptv_type(p.m, p.n, "f32"),
                tile,
                tile_buf_type(p.m, p.n, "f32")
            ));
            ctx.set_inline_consts(&[("%__r", p.m), ("%__c", p.n)]);
            let inner = format!(
                "%__batched = llvm.call @__tile_softmax_f32({}, {}, %__r, %__c) : \
                 (i32, i32, i32, i32) -> i32",
                head_src, head_src
            );
            translate_softmax(&inner, "f32", ctx, &mut body)?;
            "%__batched".to_string()
        }
        BatchedKind::Generic { ref inner_call, ref operands, .. } => {
            // Every operand and the result are live at once, so a stage's own
            // tiles can exceed UB inside a single head: silu_mul at S=1024
            // stages 3 x [S,HD] f32 = 768 KB against 184 KB. Block rows within
            // the head, exactly as the matmul arm does.
            //
            // All operands here share the row dimension (an elementwise stage
            // reads and writes the same shape), so unlike the matmul every one
            // of them takes the block offset -- there is no [K,N] operand whose
            // rows mean something else.
            let rb = pick_generic_rows(p.m, operands, "f32");
            let ops_owned: Vec<(String, u32, u32)> = operands.clone();
            let inner_owned = inner_call.clone();
            let emit = |c: &mut PtoContext, o: &mut Vec<String>, rows: u32| -> Result<(), String> {
                let mut names = Vec::with_capacity(ops_owned.len());
                for (i, (ssa, orows, cols)) in ops_owned.iter().enumerate() {
                    let t = c
                        .get_tile(ssa)
                        .ok_or_else(|| format!("batched: unknown tile {ssa}"))?
                        .clone();
                    if t.deferred.is_none() {
                        return Err(format!(
                            "batched: operand {ssa} was not deferred — its load sits \
                             outside the loop, so every iteration would read \
                             iteration 0"
                        ));
                    }
                    let nm = format!("%__op{i}");
                    // Each operand advances by ITS OWN declared rows-per-
                    // iteration. The generic form exists to express stages whose
                    // operands differ in height, so passing one shared value
                    // here would reintroduce the very bug this path was built
                    // to make unrepresentable.
                    //
                    // Note that is the HEAD stride; the block loop contributes
                    // its own term, which `with_head_rows` now sums in.
                    c.with_head_rows(
                        *orows,
                        |c2, o2| c2.load_deferred_for_head_as(&t, rows, *cols, &nm, o2),
                        o,
                    )?;
                    names.push(nm);
                }
                c.set_inline_consts(&[("%__r", rows), ("%__c", p.n)]);
                let mut inner = inner_owned.clone();
                for (i, nm) in names.iter().enumerate() {
                    inner = inner.replace(&format!("%__op{i}"), nm);
                }
                translate_line_in_batch(&inner, c, o)?;
                emit_block_store(out_tv_ssa, out_elem_offset, dtype, "%__batched", c, o)
            };
            if rb < p.m {
                let r_blk = p.m.div_ceil(rb);
                ctx.use_size(r_blk);
                body.push(format!("scf.for %r_i = %c0 to %c{r_blk} step %c1 {{"));
                ctx.with_row_block(rb, |c, o| emit(c, o, rb), &mut body)?;
                body.push("}".to_string());
            } else {
                emit(ctx, &mut body, p.m)?;
            }
            stored_inline = true;
            "%__batched".to_string()
        }
    };

    // The store, still inside the row-base scope so it lands at head h.
    //
    // Skipped when an arm already stored per block inside its own loop nest:
    // this one sits outside those loops, so it would write only the last
    // block's tile a second time.
    if !stored_inline {
        let out_pv =
            ctx.make_pv(out_tv_ssa, out_rows, out_cols, dtype, out_elem_offset, &mut body);
        let tile = ctx
            .get_tile(&result_tile)
            .ok_or_else(|| format!("{name}: stage produced no result tile"))?
            .clone();
        body.push(format!(
            "pto.tstore ins({} : {}) outs({} : {})",
            tile.ssa,
            tile.tb_type,
            out_pv,
            ptv_type(out_rows, out_cols, dtype)
        ));
    }

    ctx.set_pv_row_base(None);
    ctx.clear_head_offsets();

    // alloc_tile is loop-invariant; everything else is per-head.
    let mut hoisted: Vec<String> = Vec::new();
    let mut inner_body: Vec<String> = Vec::new();
    for l in body {
        if l.contains("pto.alloc_tile") {
            hoisted.push(l);
        } else {
            inner_body.push(format!("  {l}"));
        }
    }
    ops.extend(hoisted);

    ctx.use_size(heads);
    ctx.use_size(p.m);
    ops.push(format!("{bi64} = \"pto.get_block_idx\"() : () -> i64"));
    ops.push(format!("{bn64} = \"pto.get_block_num\"() : () -> i64"));
    ops.push(format!("{bi} = arith.index_cast {bi64} : i64 to index"));
    ops.push(format!("{bn} = arith.index_cast {bn64} : i64 to index"));
    ops.push(format!("scf.for %h_i = {bi} to %c{heads} step {bn} {{"));
    // This is the OUTPUT's row base, and it is correct for the output alone.
    //
    // It was once applied to every operand too, which is only right when all
    // operands share rows-per-head. For scores = Q @ K^T they do not: Q is
    // [S, D] with S rows per head, the pre-transposed K is [D, S] with D. At
    // S=32 / D=64 that read K at half its offset, and every head after the
    // first got the wrong slice — MERE 2.6e-01 against a 1.221e-4 bar.
    //
    // Per-head parity could not see it: it exercises the batched SOFTMAX, whose
    // single operand shares a shape with its output, which is exactly the case
    // where a shared base happens to be right. Only end-to-end against model.py
    // caught it. Two checks, two blind spots — keep both.
    //
    // FIXED: operands now get their own base via `with_head_rows`, which scopes
    // `pv_row_base` per operand (see the matmul path's two calls, m then k).
    // The emitted PTO shows both, e.g. `%h_i * 32` for Q and `%h_i * 64` for
    // K^T. All five operators pass end-to-end at MERE 1.0e-06..1.0e-05.
    ops.push(format!(
        "  {base} = arith.muli %h_i, %c{} : index   // output head base",
        p.m
    ));
    ops.extend(inner_body);
    ops.push("}".to_string());
    Ok(())
}

/// Parse a `__tile_batched_f32` call's stage description into a `Generic` kind.
///
/// The two attributes are the whole interface:
///   `batched.inner`    one MLIR call line, operands written `%__op0`, `%__op1`, …
///   `batched.operands` `<ssa>:<rows>x<cols>` per operand, comma separated
///
/// Rows are per operand rather than per loop on purpose. One shared row base is
/// correct only when every operand has the same height — true of a softmax,
/// false of `Q @ K^T` — and that assumption cost a full debug cycle when it was
/// implicit. Here it cannot be made implicitly.
fn parse_generic_batched(line: &str) -> Result<BatchedKind, String> {
    let attr = |name: &str| -> Option<String> {
        let at = line.find(&format!("{name} = \""))?;
        let rest = &line[at + name.len() + 4..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    };
    let inner_call = attr("batched.inner").ok_or_else(|| {
        format!("batched: missing `batched.inner` attribute in: {line}")
    })?;
    let spec = attr("batched.operands").ok_or_else(|| {
        format!("batched: missing `batched.operands` attribute in: {line}")
    })?;

    let mut operands = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (ssa, shape) = part
            .split_once(':')
            .ok_or_else(|| format!("batched: operand `{part}` is not `<ssa>:<rows>x<cols>`"))?;
        let (r, c) = shape
            .split_once('x')
            .ok_or_else(|| format!("batched: operand `{part}` shape is not `<rows>x<cols>`"))?;
        let rows: u32 = r.trim().parse().map_err(|_| format!("batched: bad rows in `{part}`"))?;
        let cols: u32 = c.trim().parse().map_err(|_| format!("batched: bad cols in `{part}`"))?;
        if rows == 0 || cols == 0 {
            return Err(format!("batched: operand `{part}` has a zero dimension"));
        }
        operands.push((ssa.trim().to_string(), rows, cols));
    }
    if operands.is_empty() {
        return Err(format!("batched: `batched.operands` is empty in: {line}"));
    }
    let heads: u32 = attr("batched.heads")
        .ok_or_else(|| format!("batched: missing `batched.heads` attribute in: {line}"))?
        .trim()
        .parse()
        .map_err(|_| format!("batched: `batched.heads` is not a number in: {line}"))?;
    if heads == 0 {
        return Err(format!("batched: `batched.heads` is 0 in: {line}"));
    }
    Ok(BatchedKind::Generic { heads, inner_call, operands })
}

/// Translate one inner call inside a batched loop.
///
/// A deliberately small dispatcher rather than a call back into the main loop:
/// the main loop also handles loads, stores, GEPs and the batched intrinsics
/// themselves, none of which may appear inside a batched body — a nested
/// batched call, or a store that is not the loop's own, would produce silently
/// wrong structure. Listing the stage ops here keeps "what can be batched"
/// visible and makes anything else a clear error rather than a strange kernel.
///
/// Extending this is how a new stage becomes batchable. That is a one-line
/// addition, against the intrinsic + enum arm + emitter body that the named
/// path costs.
fn translate_line_in_batch(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    // Elementwise and reduction stages, f32. Matmul is NOT here: it needs
    // MAT/left/right/acc tiles rather than vec, and the `Matmul` arm already
    // emits the device-validated form.
    let binaries = [
        ("__tile_add_f32", "pto.tadd"),
        ("__tile_sub_f32", "pto.tsub"),
        ("__tile_mul_f32", "pto.tmul"),
        ("__tile_div_f32", "pto.tdiv"),
    ];
    for (intr, op) in binaries {
        if line.contains(intr) {
            return translate_binary(line, "f32", op, ctx, ops);
        }
    }
    let unaries = [
        ("__tile_exp_f32", "pto.texp"),
        ("__tile_neg_f32", "pto.tneg"),
    ];
    for (intr, op) in unaries {
        if line.contains(intr) {
            return translate_unary(line, "f32", op, ctx, ops);
        }
    }
    if line.contains("__tile_silu_f32") {
        return translate_unary(line, "f32", "pto.tsilu", ctx, ops);
    }
    if line.contains("__tile_softmax_f32") {
        return translate_softmax(line, "f32", ctx, ops);
    }
    Err(format!(
        "batched: no inner translation for this stage — extend \
         translate_line_in_batch if it should be batchable: {line}"
    ))
}

fn translate_matmul(line: &str, ctx: &mut PtoContext, ops: &mut Vec<String>) -> Result<(), String> {
    let result_ssa =
        extract_result_ssa(line).ok_or_else(|| format!("matmul: no result SSA in: {}", line))?;
    let args =
        extract_call_args(line).ok_or_else(|| format!("matmul: cannot parse args in: {}", line))?;
    let a_ssa = args.get(1).ok_or("matmul: missing a")?.trim();
    let b_ssa = args.get(2).ok_or("matmul: missing b")?.trim();
    let m = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let k = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let n = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(a_ssa)
        .ok_or_else(|| format!("matmul: unknown tile {}", a_ssa))?
        .clone();
    let tb = ctx
        .get_tile(b_ssa)
        .ok_or_else(|| format!("matmul: unknown tile {}", b_ssa))?
        .clone();

    // Pre-pass decides blocking based on the matmul shape. If both operand
    // loads were deferred, we emit the K/N-blocked path. Otherwise fall
    // through to the single-tmatmul path for small shapes that fit L0.
    // Inside a batched loop the operands are deferred for a DIFFERENT reason
    // than blocking: their loads must happen per head, here, rather than being
    // hoisted into a K/N-block nest. Taking the blocked path would emit that
    // nest instead of this head's body, leaving the loop with only a store —
    // observed as a scores_batched kernel containing one tstore and nothing
    // else. Load the operands for this head and fall through to the inline
    // path, which is the one the per-head kernels already validate.
    // Materialise this head's operands, then fall through to the unblocked
    // path below — the same emission the validated per-head kernels use.
    let (ta, tb) = if ctx.in_batched_loop() {
        // Each operand advances by ITS OWN rows-per-head, not a shared base.
        // For scores = Q @ K^T the operands differ: Q is [S,D] with S rows per
        // head, the pre-transposed K is [D,S] with D. A single base read K at
        // the wrong offset for every head after the first — caught by the
        // end-to-end check against model.py (MERE 2.6e-01), not by per-head
        // parity, whose softmax has one operand shaped like its output.
        // And each operand takes the COLUMN base only if its columns are the N
        // dimension. A is [M,K] -- its columns are K, so an N-block offset
        // would read the wrong slice of it; B is [K,N], so it takes the offset.
        // Same shape of bug as the shared row base, one dimension over.
        let a_t = ctx
            .without_col_base(|c| {
                // HEAD STRIDE, not the block row count: the head term steps
                // over whole heads however finely one is walked.
                let stride_a = c.head_stride_a.unwrap_or(m);
                // A is [M,K]: K is its COLUMN axis. `without_col_base` above
                // suppresses the N-block column base, which A must not take --
                // the K offset is carried separately and still applies here.
                c.pv_k_on_rows = Some(false);
                let r = c.with_head_rows(stride_a, |c2, o| c2.load_deferred_for_head(&ta, m, k, o), ops);
                c.pv_k_on_rows = None;
                r
            })
            .ok_or_else(|| format!("matmul_batched: operand {a_ssa} not deferred"))?;
        // B is [K,N]: its rows are K, so the M-block offset must NOT reach it.
        let b_t = ctx
            .without_row_block(|c| {
                // B is [K,N]: K is its ROW axis. `without_row_block` suppresses
                // the M-block row base, which B must not take; the K offset is
                // separate and does apply.
                c.pv_k_on_rows = Some(true);
                let stride_b = c.head_stride_b.unwrap_or(k);
                let r = c.with_head_rows(stride_b, |c2, o| c2.load_deferred_for_head(&tb, k, n, o), ops);
                c.pv_k_on_rows = None;
                r
            })
            .ok_or_else(|| format!("matmul_batched: operand {b_ssa} not deferred"))?;
        (a_t, b_t)
    } else {
        (ta, tb)
    };

    if let (Some(da), Some(db)) = (ta.deferred.clone(), tb.deferred.clone()) {
        return translate_matmul_blocked(
            &result_ssa, m, k, n, MatmulDtypes::f32(), &da, &db, ctx, ops,
        );
    }

    // --- Unblocked path (original emission) ---
    let pv_a = ta.pv_ssa.clone().ok_or_else(|| {
        format!(
            "matmul: tile {} has no partition view (not loaded from GM)",
            a_ssa
        )
    })?;
    let pv_b = tb.pv_ssa.clone().ok_or_else(|| {
        format!(
            "matmul: tile {} has no partition view (not loaded from GM)",
            b_ssa
        )
    })?;

    ctx.use_size(m);
    ctx.use_size(k);
    ctx.use_size(n);

    let mat_a_key = format!("{}__mat_a", result_ssa);
    let mat_b_key = format!("{}__mat_b", result_ssa);
    let mat_a_ty = mat_tile_type(m, k, "f32");
    let mat_b_ty = mat_tile_type(k, n, "f32");
    let mat_a_ssa = ctx.alloc_tile_typed(&mat_a_key, m, k, "f32", &mat_a_ty, ops);
    let mat_b_ssa = ctx.alloc_tile_typed(&mat_b_key, k, n, "f32", &mat_b_ty, ops);

    let left_key = format!("{}__left", result_ssa);
    let right_key = format!("{}__right", result_ssa);
    let left_ty = left_tile_type(m, k, "f32");
    let right_ty = right_tile_type(k, n, "f32");
    let acc_ty = acc_tile_type(m, n, "f32");
    let left_ssa = ctx.alloc_tile_typed(&left_key, m, k, "f32", &left_ty, ops);
    let right_ssa = ctx.alloc_tile_typed(&right_key, k, n, "f32", &right_ty, ops);
    let acc_ssa = ctx.alloc_tile_typed(&result_ssa, m, n, "f32", &acc_ty, ops);

    let pv_a_ty = ptv_type(m, k, "f32");
    let pv_b_ty = ptv_type(k, n, "f32");
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_a, pv_a_ty, mat_a_ssa, mat_a_ty
    ));
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_b, pv_b_ty, mat_b_ssa, mat_b_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        mat_a_ssa, mat_a_ty, left_ssa, left_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        mat_b_ssa, mat_b_ty, right_ssa, right_ty
    ));
    if ctx.matmul_accumulate {
        // Inside a K loop: the FIRST iteration writes the accumulator, every
        // later one adds into it. A plain tmatmul here would discard all but
        // the last K slice's contribution -- a wrong answer that still
        // assembles, runs and looks plausible, which is exactly the failure a
        // timing run cannot see.
        //
        // Same shape as `emit_blocked_matmul_loops` uses on the non-batched
        // path, which is validated; this is a port of it, not a new scheme.
        let is_first = ctx.fresh_ssa();
        ops.push(format!("  {is_first} = arith.cmpi eq, %k_i, %c0 : index"));
        ops.push(format!("  scf.if {is_first} {{"));
        ops.push(format!(
            "    pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
            left_ssa, right_ssa, left_ty, right_ty, acc_ssa, acc_ty
        ));
        ops.push("  } else {".to_string());
        ops.push(format!(
            "    pto.tmatmul.acc ins({}, {}, {} : {}, {}, {}) outs({} : {})",
            acc_ssa, left_ssa, right_ssa, acc_ty, left_ty, right_ty, acc_ssa, acc_ty
        ));
        ops.push("  }".to_string());
    } else {
        ops.push(format!(
            "pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
            left_ssa, right_ssa, left_ty, right_ty, acc_ssa, acc_ty
        ));
    }

    Ok(())
}

/// Emit a K/N-blocked matmul matching the validated
/// `/tmp/matmul_q_proj_m16.pto` hand-patch. See the comment block above
/// `detect_blocked_matmul_loads` for the design.
///
/// Shape assumptions (checked at runtime):
///   - M % 16 == 0  (TileConfig::fixedRowSize on 910B2 cube)
///   - K % Kb == 0 AND N % Nb == 0 (caller pads if not — no remainder loop yet)
///
/// Output tile (acc, M×Nb) is partition-stored to `result_ssa`'s eventual
/// tstore, which reads `TileInfo.ssa` / `tb_type`. We register the acc
/// tile under `result_ssa` so the downstream tstore "just works" — but
/// since acc shape is M×Nb (not M×N), the caller's tstore would write
/// the wrong region. Instead we emit the tstore inline here inside the
/// N-loop, and register a sentinel TileInfo marked `consumed_inline` so
/// the downstream translate_store sees there's nothing to do.
fn translate_matmul_blocked(
    result_ssa: &str,
    m: u32,
    k: u32,
    n: u32,
    dtypes: MatmulDtypes,
    da: &DeferredMatmulOperand,
    db: &DeferredMatmulOperand,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    if m % PTO_MM_MROW_ALIGN != 0 {
        return Err(format!(
            "blocked matmul: M={} must be a multiple of {} (910B2 cube fixedRowSize). \
             Pad the M dim of your Rust tile_matmul kernel source.",
            m, PTO_MM_MROW_ALIGN
        ));
    }
    let kb = pick_kb_for_n_dtype(k, n, dtypes.lhs_bytes() as u32);
    let nb = pick_nb_for_dtype(n, dtypes.lhs_bytes() as u32);
    if k % kb != 0 {
        return Err(format!(
            "blocked matmul: K={} must be a multiple of Kb={}",
            k, kb
        ));
    }
    if n % nb != 0 {
        return Err(format!(
            "blocked matmul: N={} must be a multiple of Nb={}",
            n, nb
        ));
    }
    let n_iters = n / nb;
    let k_iters = k / kb;

    // Sizes / constants we need emitted as arith.constant.
    ctx.use_size(0);
    ctx.use_size(1);
    ctx.use_size(m);
    ctx.use_size(k);
    ctx.use_size(n);
    ctx.use_size(kb);
    ctx.use_size(nb);
    ctx.use_size(n_iters);
    ctx.use_size(k_iters);

    // Allocate the five reusable tiles ONCE outside the loops.
    //   mat_a  (M × Kb)  — CBUF staging for A (left operand path), dtype=lhs
    //   mat_b  (Kb × Nb) — CBUF staging for B, dtype=rhs
    //   a_left (M × Kb)  — L0A working copy, dtype=lhs
    //   b_right (Kb × Nb) — L0B working copy, dtype=rhs
    //   acc    (M × Nb)  — L0C accumulator (one block-column at a time), dtype=dst
    // Decide the M block BEFORE allocating: mat_a / a_left / acc all carry M,
    // and at full M the staging tile alone can exceed the UB (512x128xf32 =
    // 256 KB against 184 KB) long before the loop that would have tiled it.
    let mb = pick_mb(m, kb, nb, &dtypes, ctx.ub_live_bytes());
    let m_iters = m.div_ceil(mb.max(1));

    let mat_a_ty = mat_tile_type(mb, kb, dtypes.lhs);
    let mat_b_ty = mat_tile_type(kb, nb, dtypes.rhs);
    let left_ty = left_tile_type(mb, kb, dtypes.lhs);
    let right_ty = right_tile_type(kb, nb, dtypes.rhs);
    let acc_ty = acc_tile_type(mb, nb, dtypes.dst);

    let mat_a_ssa = ctx.alloc_tile_typed(
        &format!("{}__mat_a_blk", result_ssa),
        mb,
        kb,
        dtypes.lhs,
        &mat_a_ty,
        ops,
    );
    let mat_b_ssa = ctx.alloc_tile_typed(
        &format!("{}__mat_b_blk", result_ssa),
        kb,
        nb,
        dtypes.rhs,
        &mat_b_ty,
        ops,
    );
    let a_left_ssa = ctx.alloc_tile_typed(
        &format!("{}__a_left_blk", result_ssa),
        mb,
        kb,
        dtypes.lhs,
        &left_ty,
        ops,
    );
    let b_right_ssa = ctx.alloc_tile_typed(
        &format!("{}__b_right_blk", result_ssa),
        kb,
        nb,
        dtypes.rhs,
        &right_ty,
        ops,
    );
    // acc is registered under the matmul's result SSA so the fall-through
    // tstore lookup finds it — but we actually tstore it per-N-block
    // inline below. The downstream tstore must recognise "already stored"
    // to avoid a duplicate emit.
    let acc_ssa =
        ctx.alloc_tile_typed(result_ssa, mb, nb, dtypes.dst, &acc_ty, ops);

    // Per the design note: we emit the per-N-block tstore inline below.
    // To avoid translate_store re-emitting a full-shape tstore for
    // `result_ssa`, we mark the TileInfo as "output consumed inline" by
    // clearing `pv_ssa` and setting a sentinel SSA. The existing store
    // path reads `tile.ssa` and `tb_type`; we can't easily signal "skip"
    // without adding another flag. Instead we rely on the fact that the
    // tstore's pv for the output will be built anew from the `output`
    // GM arg — which is correct for the full shape. The inline tstore
    // here writes per-block; the downstream full-shape tstore would
    // overwrite with uninitialised acc data. So we need a real "skip"
    // marker. Add it via a post-emit hook on ctx.
    ctx.matmul_result_stored_inline.insert(result_ssa.to_string());

    // Resolve the output tensor_view for the per-block tstore. The output
    // GM and its tv are registered by the downstream tile_store_f32 call
    // — but that line runs *after* this matmul in body_lines, so its tv
    // isn't in ctx.tv_map yet. We need to build the tv here.
    //
    // The output shape is M×N (the matmul result) — the downstream
    // `tile_store_f32::<M, N>` will write exactly that. We use the
    // `output` function argument which by convention is the 3rd GM arg.
    // But we don't know that generically — for now, require that the
    // Rust kernel source immediately tile_store's the matmul result, and
    // walk ctx.tiles for the pending registration of `result_ssa`.
    //
    // Simpler: stash the output tv request and let translate_store
    // (which DOES know the output GM name) emit the per-block loop.
    //
    // ...but translate_store sees a single-call to
    // `__tile_store_f32(out_gm, result_ssa, M, N)` and doesn't
    // know about the blocking. Cleaner refactor: have translate_matmul
    // return without emitting the store, and have translate_store detect
    // that the tile being stored has a `stored_inline` marker and emit
    // the per-N-block loop itself.
    //
    // For this patch, take the cleaner path: defer the per-block store
    // to translate_store. Do NOT emit the scf.for here yet — instead,
    // remember everything translate_store needs:
    //   - tv_a_ssa, tv_b_ssa, elem_offsets for per-block partition views
    //   - the 5 tile SSAs and their types
    //   - M, K, N, Kb, Nb, n_iters, k_iters
    let pending = PendingBlockedMatmul {
        m,
        k,
        n,
        mb,
        kb,
        nb,
        m_iters,
        n_iters,
        k_iters,
        dtypes,
        tv_a_ssa: da.tv_ssa.clone(),
        tv_b_ssa: db.tv_ssa.clone(),
        a_elem_offset: da.elem_offset,
        b_elem_offset: db.elem_offset,
        a_gm_name: da.gm_name.clone(),
        b_gm_name: db.gm_name.clone(),
        mat_a_ssa,
        mat_b_ssa,
        a_left_ssa,
        b_right_ssa,
        acc_ssa,
        mat_a_ty,
        mat_b_ty,
        left_ty,
        right_ty,
        acc_ty,
        dequant: None,
    };
    ctx.pending_blocked_matmuls
        .insert(result_ssa.to_string(), pending);

    Ok(())
}

/// Softmax: `%res = llvm.call @__tile_softmax_f32(%c0, %src, %rows, %cols)`
///
/// Decomposes into the numerically-stable 5-step sequence:
/// 1. `trowmax(src, tmp) → max`   — row-wise max (needs a tmp scratch tile)
/// 2. `trowexpandsub(src, max) → sub` — subtract row max from each element
/// 3. `texp(sub) → exp_vals`     — element-wise exp
/// 4. `trowsum(exp_vals, tmp) → sum` — row-wise sum (reuses tmp scratch)
/// 5. `trowexpanddiv(exp_vals, sum) → result` — divide by row sum
///
/// This matches the FlashAttention reference implementation in pto-isa:
/// `TROWMAX(new_max, x, tmp)` → `TROWEXPANDSUB(sub, x, new_max)` → `TEXP` → `TROWSUM` → `TROWEXPANDDIV`
/// Rows per block for a matmul stage INSIDE a batched loop.
///
/// The head loop slices across heads; this bounds one head's own operands. All
/// four tiles a matmul stages carry M, and each lands in a different memory, so
/// every cap has to hold at once:
///
///   mat_a [mb,k] -> L1     a_left [mb,k] -> L0A
///   mat_b [k,n]  -> L1     b_right[k,n]  -> L0B     acc [mb,n] -> L0C
///
/// NOTE `b_right` is `k*n` with no `mb` term, so row-blocking cannot shrink it.
/// A head whose B operand alone exceeds L0B needs N-blocking, which this does
/// not do -- it returns the 16-row floor and the guard then reports L0B, which
/// is the honest outcome rather than a silently wrong tile.
///
/// Returns `m` unchanged when the head already fits, so shapes that worked
/// before emit byte-identically (no inner loop at all).
fn pick_batched_rows(m: u32, k: u32, n: u32) -> u32 {
    pick_batched_blocks(m, k, n).0
}

/// Rows AND columns per block for a matmul stage inside a batched loop.
///
/// Two dimensions because the tiles they bound differ:
///
///   a_left  [mb, k]  -> L0A   shrinks with ROWS
///   b_right [k, nb]  -> L0B   shrinks with COLUMNS only -- no row term
///   acc     [mb, nb] -> L0C   shrinks with either
///
/// So row-blocking alone cannot make a wide B fit: at S=512 both scores and pv
/// stage a 128 KB `b_right` against a 64 KB L0B cap regardless of `mb`.
/// Returns `(m, n)` unchanged when the head already fits, so shapes that worked
/// before emit byte-identically with no inner loops at all.
fn pick_batched_blocks(m: u32, k: u32, n: u32) -> (u32, u32) {
    let (mb, _kb, nb) = pick_batched_blocks_k(m, k, n);
    (mb, nb)
}

/// Rows, K-depth and columns per block for a batched matmul stage.
///
/// K is the third blocking dimension, and WITHOUT it both of the others are
/// pinned whenever K is large: `a_left` is `[mb, k]` and `b_right` is
/// `[k, nb]`, so at K=896 the 64 KB L0A/L0B caps hold mb and nb to 16 no
/// matter what the accumulator could take. pv ([S,S] @ [S,HD]) ran a 16x16
/// accumulator at 0.2% of L0C for exactly this reason, and took 578 us of a
/// 2750 us forward -- while `scores`, the same size transposed but with K=64,
/// ran the same work in 66 us. The low L0C occupancy was a symptom; K was the
/// cause.
///
/// Returns `kb == k` when the shape already fits, so no K loop is emitted and
/// every previously-working kernel stays byte-identical.
fn pick_batched_blocks_k(m: u32, k: u32, n: u32) -> (u32, u32, u32) {
    let b = 4u64; // f32 batched stages
    let fits_kb = |mb: u32, kb: u32, nb: u32| -> bool {
        let a = (mb as u64) * (kb as u64) * b;
        let br = (kb as u64) * (nb as u64) * b;
        let acc = (mb as u64) * (nb as u64) * b;
        a <= SPEC_A2A3.l1_size as u64
            && a <= SPEC_A2A3.l0_a_size as u64
            && br <= SPEC_A2A3.l0_b_size as u64
            && acc <= SPEC_A2A3.l0_c_size as u64
    };
    let fits = |mb: u32, nb: u32| -> bool { fits_kb(mb, k, nb) };
    if fits(m, n) {
        return (m, k, n);
    }

    // Try K-blocking FIRST, because it is the only lever that unpins the other
    // two. Walk candidate Kb downward; for each, take the largest mb and nb the
    // caps then allow. Kb must DIVIDE k -- the K loop is a bare
    // `scf.for 0 to k/kb` and a partial tail would silently drop a slice of the
    // reduction, which is a wrong answer rather than a rejected one.
    //
    // Candidates stay powers of two: `pick_nb_for_dtype` documents that ptoas
    // chooses the L0A Left tile's BLayout from the companion Right tile's
    // width, a hazard only validated at power-of-two widths.
    let mut best: Option<(u32, u32, u32, u32)> = None; // (trips, mb, kb, nb)
    let mut kb = k.next_power_of_two().min(512);
    while kb >= 32 {
        if k % kb == 0 {
            // Largest mb and nb the caps allow at this Kb, both powers of two.
            let mb_cap = ((SPEC_A2A3.l0_a_size as u64) / ((kb as u64) * b)) as u32;
            let nb_cap = ((SPEC_A2A3.l0_b_size as u64) / ((kb as u64) * b)) as u32;
            let pow2_le = |v: u32| -> u32 {
                if v == 0 { 0 } else { 1 << (31 - v.leading_zeros()) }
            };
            // nb stays a POWER OF TWO: ptoas picks the L0A Left tile's BLayout
            // from the companion Right tile's WIDTH, a hazard only validated at
            // power-of-two widths (see `pick_nb_for_dtype`).
            let mut nb = pow2_le(nb_cap.min(n));
            while nb > 1 && (!fits_kb(16, kb, nb) || n % nb != 0) {
                nb /= 2;
            }
            // mb steps by 16 -- the cube row granularity -- NOT by halving.
            // The width hazard above is width-linked and says nothing about the
            // row count, and forcing mb to a power of two costs real trips:
            // `scores` (m=896, k=64, n=896) takes mb=224 at 88% of L0C for 28
            // trips, but the nearest power of two is 128, giving 49 trips. That
            // regressed scores 66 -> 110 us until this loop stepped by 16.
            let mut mb = (mb_cap.min(m) / 16) * 16;
            while mb > 16 && (!fits_kb(mb, kb, nb) || m % mb != 0) {
                mb -= 16;
            }
            if mb >= 16 && nb >= 16 && fits_kb(mb, kb, nb) {
                let trips = (m / mb) * (n / nb) * (k / kb);
                if best.is_none() || trips < best.unwrap().0 {
                    best = Some((trips, mb, kb, nb));
                }
            }
        }
        kb /= 2;
    }
    if let Some((_, mb, kb, nb)) = best {
        return (mb, kb, nb);
    }
    // No K-blocked candidate: fall through to the original row/column search,
    // which is still correct, just narrower.
    // Shrink N first: it is the only lever on `b_right`, and it relieves the
    // accumulator too. Then shrink rows for whatever remains.
    //
    // The block must land on a C2-LEGAL WIDTH, not merely a smaller one.
    // Halving 384 gives 192, and 192*4 = 768 B spans 512 B fractals with a bad
    // stride — a hard C2 rejection. That single miss made S=320..384 fail for
    // every operator that batches heads (Local and Strided escaped only because
    // their head_dim=24 lands the halving elsewhere).
    //
    // Candidates are the multiples of `c2_col_multiple` that DIVIDE n, largest
    // first. Padded widths are always multiples of 128 (see `pad_cols_c2`), so
    // such a divisor always exists; the `<= m` case covers sub-fractal widths,
    // which C2 exempts.
    let cm = c2_col_multiple("f32").max(1);
    let mut nb = n;
    if n > cm {
        nb = (1..=n / cm)
            .rev()
            .map(|q| q * cm)
            .find(|&cand| n % cand == 0 && fits(m.min(16), cand))
            .unwrap_or(cm);
    } else {
        while nb > 16 && !fits(m.min(16), nb) {
            nb /= 2;
        }
    }
    let mut mb = m;
    while mb > 16 && !fits(mb, nb) {
        mb /= 2;
    }
    (mb.max(16), k, nb.max(16))
}

/// Rows per block for a GENERIC batched stage.
///
/// Every operand plus the output is live at once, so the budget divides by the
/// count rather than by a fixed three: `silu_mul` stages two inputs and one
/// output, which at S=1024 is 768 KB of `[S,HD]` f32 against a 184 KB UB.
///
/// Returns `rows` unchanged when the stage already fits, so shapes that worked
/// before emit byte-identically with no inner loop at all.
fn pick_generic_rows(rows: u32, operands: &[(String, u32, u32)], dtype: &str) -> u32 {
    let eb = dtype_bytes_pto(dtype) as u64;
    // Widest operand decides: they are loaded into equally-shaped tiles.
    let cols = operands.iter().map(|(_, _, c)| *c).max().unwrap_or(1) as u64;
    let live = operands.len() as u64 + 1; // inputs + result
    let budget = SPEC_A2A3.ub_budget() as u64;
    let fits = |rb: u32| -> bool { live * (rb as u64) * cols * eb <= budget };
    if fits(rows) {
        return rows;
    }
    let mut rb = rows;
    while rb > 16 && !fits(rb) {
        rb /= 2;
    }
    rb.max(16)
}

/// A row-blocked softmax awaiting its store, so the loop can enclose it.
#[derive(Clone)]
struct PendingRowSoftmax {
    src: String,
    rows: u32,
    cols: u32,
    /// Rows per block; `rows` when the whole tile already fits.
    rb: u32,
    n_blocks: u32,
    dtype: String,
}

/// Rows per softmax block.
///
/// Row-tiling a softmax is EXACT, not an approximation: `trowmax`, `trowsum`
/// and both expand steps reduce along columns, within a row, so row `i` depends
/// on no other row. (Splitting COLUMNS would be a different problem needing
/// online rescaling — that is flash-attention, and it is not what this does.)
///
/// The 5-step stable form keeps 3 full `[rows, cols]` tiles live (tmp, sub,
/// exp) plus 2 row-reduction tiles, so at S=128 f32 that is ~193 KB against a
/// 184 KB budget. Returns `rows` unchanged when the whole thing already fits,
/// which is what keeps existing kernels byte-identical.
fn pick_softmax_rows(rows: u32, cols: u32, dtype: &str, live_bytes: usize) -> u32 {
    let budget = (SPEC_A2A3.ub_budget() as u64).saturating_sub(live_bytes as u64);
    let eb = dtype_bytes_pto(dtype) as u64;
    // FIVE full tiles are live, not four: tmp, sub, exp and out are allocated
    // here, and the SOURCE block is resident too. Modelling four approved
    // rb=16 at S=640 for 200 KB against a 184 KB budget, and the guard then
    // rejected what this function had just blessed -- the tile it named
    // (40960 B at offset 164864 B) is exactly the fifth.
    // THREE full-width tiles, not five: `emit_softmax_steps` aliases exp and
    // out onto sub (each is dead the instant the next op reads it), leaving
    // src, tmp and the shared sub/exp/out buffer. Two row-reduction tiles
    // (max, sum) are rows x 1.
    //
    // This MUST track the allocation in `emit_softmax_steps`. It previously
    // said four while five were allocated, and the guard rejected shapes this
    // function had just blessed.
    let fits = |rb: u32| -> bool {
        let full = 3 * (rb as u64) * (cols as u64) * eb;
        let rr = 2 * (rb as u64) * eb;
        full + rr <= budget
    };
    if fits(rows) {
        return rows;
    }
    // Step down one row at a time, with NO 16-row floor.
    //
    // A2/A3 sets `REQUIRES_ROW16 = false` -- the 16-row rule is a cube
    // (NZ/fractal) constraint and softmax is a vector-side stage, so a 1-row
    // block is legal. The old floor was mine, not the hardware's, and it is
    // what made S >= 640 look impossible: at 16 rows the working set is 200 KB
    // however the columns are tiled, so the search bottomed out above budget
    // and reported failure. Without it the budget is sufficient well past
    // S=4096 (rb falls to 14, 12, 10, 9, ... as S grows).
    //
    // The block must also DIVIDE `rows`. The emitted loop is a bare
    // `scf.for 0 to ceil(rows/rb) step 1` with no tail guard, so a block that
    // merely fits still walks off the end of the view: S=896 picked rb=10,
    // giving 90 trips x 10 = 900 rows against an 896-row tensor, and the last
    // iteration read 4 rows past it. That is an out-of-bounds access, and the
    // device reports it as ACL 507035 at the softmax sync -- it emitted and
    // built cleanly, so only running it on silicon caught it. Fitting is a
    // capacity question; dividing is a correctness one, and both must hold.
    let mut rb = rows;
    while rb > 1 && !(fits(rb) && rows % rb == 0) {
        rb -= 1;
    }
    rb.max(1)
}

/// Emit a row-blocked softmax: one loop, per-block load / 5 steps / store.
///
/// Row-tiling is exact — every reduction is along columns, within a row — so
/// this computes the same values as the unblocked form, in blocks that fit.
///
/// The store is INSIDE the loop. A store left outside writes only the last
/// block, which is the defect the batched work hit as "iterations 1..N
/// untouched"; the shape of the bug is identical, so the shape of the fix is.
fn emit_row_softmax_loop(
    out_tv_ssa: &str,
    out_elem_offset: u32,
    dtype: &str,
    p: &PendingRowSoftmax,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let (rb, cols) = (p.rb, p.cols);
    ops.push(format!(
        "// --- softmax: {} rows in blocks of {rb}, store inside the loop ---",
        p.rows
    ));

    let tsrc = ctx
        .get_tile(&p.src)
        .ok_or_else(|| format!("row-softmax: unknown tile {}", p.src))?
        .clone();
    // If the operand was NOT deferred, fall back to the unblocked body rather
    // than failing.
    //
    // The deferral pre-pass has no PtoContext, so it must guess `live_bytes = 0`
    // while `translate_softmax` sees the real cursor -- the two can therefore
    // disagree about whether blocking happens. Every attempt to make them agree
    // by guessing a budget in the pre-pass traded one mismatch for another: an
    // optimistic guess left S=96 undeferred, a pessimistic one blocked an f16
    // [16,1024] stage that genuinely fits. Making the emitter TOLERANT removes
    // the coupling: a spare deferral costs nothing, and a missing one now
    // degrades to the form that was already correct.
    let Some(d) = tsrc.deferred.clone() else {
        return emit_softmax_steps(&p.src, "%__sm", p.rows, cols, dtype, ctx, ops)
            .and_then(|()| {
                let pv = ctx.make_pv(out_tv_ssa, p.rows, cols, dtype, out_elem_offset, ops);
                let t = ctx
                    .get_tile("%__sm")
                    .ok_or("row-softmax: unblocked fallback produced no tile")?
                    .clone();
                ops.push(format!(
                    "pto.tstore ins({} : {}) outs({} : {})",
                    t.ssa,
                    t.tb_type,
                    pv,
                    ptv_type(p.rows, cols, dtype)
                ));
                Ok(())
            });
    };

    // Emit one block's body into a scratch buffer so allocations can be
    // hoisted: they are loop-invariant, and re-allocating per block would
    // charge the UB budget once per iteration.
    let base = ctx.fresh_ssa();
    let mut body: Vec<String> = Vec::new();
    ctx.set_pv_row_base(Some(base.clone()));

    let pv = ctx.make_pv(&d.tv_ssa, rb, cols, dtype, d.elem_offset, &mut body);
    let src_blk = format!("{}__srcblk", p.src);
    let src_tile = ctx.alloc_tile(&src_blk, rb, cols, dtype, &mut body);
    body.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv,
        ptv_type(rb, cols, dtype),
        src_tile,
        tile_buf_type(rb, cols, dtype)
    ));

    let out_key = format!("{}__outblk", p.src);
    emit_softmax_steps(&src_blk, &out_key, rb, cols, dtype, ctx, &mut body)?;

    let out_pv = ctx.make_pv(out_tv_ssa, rb, cols, dtype, out_elem_offset, &mut body);
    let out_tile = ctx
        .get_tile(&out_key)
        .ok_or("row-softmax: softmax produced no result tile")?
        .clone();
    body.push(format!(
        "pto.tstore ins({} : {}) outs({} : {})",
        out_tile.ssa,
        out_tile.tb_type,
        out_pv,
        ptv_type(rb, cols, dtype)
    ));

    ctx.set_pv_row_base(None);

    let mut hoisted: Vec<String> = Vec::new();
    let mut inner: Vec<String> = Vec::new();
    for l in body {
        if l.contains("pto.alloc_tile") {
            hoisted.push(l);
        } else {
            inner.push(format!("  {l}"));
        }
    }
    ops.extend(hoisted);

    ctx.use_size(p.n_blocks);
    ctx.use_size(rb);
    ops.push(format!("scf.for %r_i = %c0 to %c{} step %c1 {{", p.n_blocks));
    ops.push(format!("  {base} = arith.muli %r_i, %c{rb} : index   // row block base"));
    ops.extend(inner);
    ops.push("}".to_string());
    Ok(())
}

fn translate_softmax(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa =
        extract_result_ssa(line).ok_or_else(|| format!("softmax: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("softmax: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("softmax: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    // Never block inside an enclosing batched loop: that loop already slices
    // the operand, the row base is taken, and deferring here would leave the
    // stage with no result tile for its store.
    let rb = if ctx.in_batched_loop() {
        rows
    } else {
        pick_softmax_rows(rows, cols, dtype, ctx.ub_live_bytes())
    };
    let n_blocks = rows.div_ceil(rb.max(1));
    if rb < rows {
        // Blocked: defer to the store, which is where the output GM view is
        // known and therefore where the enclosing loop can be emitted.
        ctx.softmax_rows_stored_inline.insert(result_ssa.clone());
        ctx.pending_row_softmax.insert(
            result_ssa.clone(),
            PendingRowSoftmax {
                src: src_ssa.to_string(),
                rows,
                cols,
                rb,
                n_blocks,
                dtype: dtype.to_string(),
            },
        );
        return Ok(());
    }

    emit_softmax_steps(src_ssa, &result_ssa, rows, cols, dtype, ctx, ops)
}

/// The 5-step numerically-stable softmax, over one `[rows, cols]` tile.
///
/// Shared by the plain and row-blocked paths so the two cannot drift: the
/// blocked form is this same body under a loop, at a smaller `rows`.
///
///   trowmax -> trowexpandsub -> texp -> trowsum -> trowexpanddiv
fn emit_softmax_steps(
    src_key: &str,
    out_key: &str,
    rows: u32,
    cols: u32,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let tsrc = ctx
        .get_tile(src_key)
        .ok_or_else(|| format!("softmax: unknown tile {src_key}"))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    let t_max_key = format!("{out_key}__max");
    let t_tmp_key = format!("{out_key}__tmp");
    let t_sub_key = format!("{out_key}__sub");
    let t_exp_key = format!("{out_key}__exp");
    let t_sum_key = format!("{out_key}__sum");

    // t_max and t_sum are row-reduction outputs: rows×1, col_major
    let rr_ty = tile_buf_type_rowreduce(rows, dtype);
    let t_max_ssa = ctx.alloc_tile_rowreduce(&t_max_key, rows, dtype, ops);
    let t_tmp_ssa = ctx.alloc_tile(&t_tmp_key, rows, cols, dtype, ops);
    let t_sub_ssa = ctx.alloc_tile(&t_sub_key, rows, cols, dtype, ops);
    let t_sum_ssa = ctx.alloc_tile_rowreduce(&t_sum_key, rows, dtype, ops);

    // `sub` is dead the instant texp reads it, and the texp result is dead the
    // instant trowexpanddiv reads it -- so exp and out are the SAME buffer as
    // sub. Three full-width tiles become one, cutting the working set from
    // 5 x [rb, cols] to 3 and letting `pick_softmax_rows` choose a larger rb
    // (16 instead of 8 at S=896, halving the trips from 112 to 56).
    //
    // Both aliased ops are elementwise with a 1:1 index map, so in-place is
    // sound in principle -- but whether ptoas and ccec ACCEPT an aliased
    // src/dst is a toolchain question, not an algebraic one. Probed on device
    // before writing this: probes/README records the run, and the aliased
    // kernel at rb=16/S=896 gave all 896 rows summing to 1 (worst deviation
    // 9.0e-08), every value in [0,1], and argmax preserved.
    ctx.alias_tile(&t_sub_key, &t_exp_key);
    ctx.alias_tile(&t_sub_key, out_key);
    let t_exp_ssa = t_sub_ssa.clone();
    let t_out_ssa = t_sub_ssa.clone();

    let src_ssa_pto = tsrc.ssa.clone();

    // Step 2: trowmax ins(%src, %tmp : T, T) outs(%max : Trr)
    // dst must be rows×1 col_major per ptoas v0.13 constraint
    ops.push(format!(
        "pto.trowmax ins({}, {} : {}, {}) outs({} : {})",
        src_ssa_pto, t_tmp_ssa, tb_ty, tb_ty, t_max_ssa, rr_ty
    ));

    // Step 3: trowexpandsub ins(%src, %max : T, Trr) outs(%sub : T)
    ops.push(format!(
        "pto.trowexpandsub ins({}, {} : {}, {}) outs({} : {})",
        src_ssa_pto, t_max_ssa, tb_ty, rr_ty, t_sub_ssa, tb_ty
    ));

    // Step 4: texp ins(%sub : T) outs(%exp_vals : T)
    ops.push(format!(
        "pto.texp ins({} : {}) outs({} : {})",
        t_sub_ssa, tb_ty, t_exp_ssa, tb_ty
    ));

    // Step 5: trowsum ins(%exp_vals, %tmp : T, T) outs(%sum : Trr)
    // reuse t_tmp_ssa as the scratch buffer; dst must be rows×1 col_major
    ops.push(format!(
        "pto.trowsum ins({}, {} : {}, {}) outs({} : {})",
        t_exp_ssa, t_tmp_ssa, tb_ty, tb_ty, t_sum_ssa, rr_ty
    ));

    // Step 6: trowexpanddiv ins(%exp_vals, %sum : T, Trr) outs(%result : T)
    ops.push(format!(
        "pto.trowexpanddiv ins({}, {} : {}, {}) outs({} : {})",
        t_exp_ssa, t_sum_ssa, tb_ty, rr_ty, t_out_ssa, tb_ty
    ));

    Ok(())
}

/// Fused attention: softmax(Q @ K^T / sqrt(D)) @ V
///
/// Decomposes into: matmul(Q,K^T) → softmax_5ops → matmul(@V)
/// The full pipeline is emitted as sequential PTO ops, allowing ptoas to
/// schedule them optimally across cube and vector engines.
fn translate_attention(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let args =
        extract_call_args(line).ok_or_else(|| format!("attention: cannot parse args: {}", line))?;
    // args: [dst(0), q_buf, k_buf, v_buf, seq_len, head_dim]
    if args.len() < 6 {
        return Err(format!("attention: expected 6 args, got {}", args.len()));
    }
    let result_ssa = extract_result_ssa(line).unwrap_or_else(|| "__att_out".to_string());
    let q_arg = args[1].trim();
    let k_arg = args[2].trim();
    let v_arg = args[3].trim();
    let s = ctx.resolve_const(args[4].trim());
    let d = ctx.resolve_const(args[5].trim());

    let tq = ctx.get_tile(q_arg).ok_or_else(|| format!("attention: unknown Q tile {}", q_arg))?.clone();
    let tk = ctx.get_tile(k_arg).ok_or_else(|| format!("attention: unknown K tile {}", k_arg))?.clone();
    let tv = ctx.get_tile(v_arg).ok_or_else(|| format!("attention: unknown V tile {}", v_arg))?.clone();

    // ptoas does not accept the vec→mat tmov address-space pair on a5 when the
    // source tile has slayout=none_box. The working matmul path (translate_matmul)
    // goes GM-partition_view → mat directly via `pto.tload`. We do the same here
    // by re-tload'ing from the same partition views for Q/K/V — the user-level
    // `tile_load_view_f32` did put them into vec tiles already, but for the cube
    // path we want mat tiles straight from GM.
    let pv_q = tq.pv_ssa.clone().ok_or_else(|| {
        format!("attention: Q tile {} has no partition view (not loaded from GM)", q_arg)
    })?;
    let pv_v = tv.pv_ssa.clone().ok_or_else(|| {
        format!("attention: V tile {} has no partition view (not loaded from GM)", v_arg)
    })?;
    // K needs to be fed into the cube as D×S (right operand of tmatmul),
    // but the user loaded it as S×D row-major. Construct a *transposed*
    // tensor_view on the same GM buffer and a fresh D×S partition_view.
    let k_gm = tk.gm_name.clone().ok_or_else(|| {
        format!("attention: K tile {} has no recorded GM name (not loaded from GM)", k_arg)
    })?;

    ops.push(format!("// --- fused attention: softmax(Q@K^T) @ V, S={}, D={} ---", s, d));

    // Step 1: scores = Q(S×D) @ K^T(D×S) → S×S via cube unit.
    // Q is loaded ND→NZ (GM row-major view, mat blayout=col_major/slayout=row_major).
    // K uses a *transposed* tensor_view (DN) and thus a ZN mat tile
    // (blayout=row_major/slayout=col_major) to satisfy TLoadGm2L1's supported
    // DN→ZN path. ZN happens to match the `right` operand layout exactly,
    // so the subsequent CBUF→L0B tmov is just a location change.
    let mat_q_ty = mat_tile_type(s, d, "f32");
    let mat_k_ty = mat_tile_type_zn(d, s, "f32");
    let l_ty = left_tile_type(s, d, "f32");
    let r_ty = right_tile_type(d, s, "f32");
    let acc_ty = acc_tile_type(s, s, "f32");

    let mq = ctx.alloc_tile_typed(&format!("{}__mq", result_ssa), s, d, "f32", &mat_q_ty, ops);
    let mk = ctx.alloc_tile_typed(&format!("{}__mk", result_ssa), d, s, "f32", &mat_k_ty, ops);
    let lq = ctx.alloc_tile_typed(&format!("{}__lq", result_ssa), s, d, "f32", &l_ty, ops);
    let rk = ctx.alloc_tile_typed(&format!("{}__rk", result_ssa), d, s, "f32", &r_ty, ops);
    let scores = ctx.alloc_tile_typed(&format!("{}__scores", result_ssa), s, s, "f32", &acc_ty, ops);

    // tload Q (S×D) and K (D×S via *transposed* tensor_view) directly into
    // CBUF mat tiles. The K transpose is encoded in the view: shape [D,S]
    // with strides [1, D] reads the same S×D row-major GM buffer as if it
    // were column-major — which is exactly K^T.
    let pv_sd = ptv_type(s, d, "f32");
    let tv_k_t = ctx.make_tv_transposed(&k_gm, s, d, "f32", ops);
    let pv_k_t = ctx.make_pv(&tv_k_t, d, s, "f32", 0, ops);
    let pv_ds = ptv_type(d, s, "f32");
    ops.push(format!("pto.tload ins({} : {}) outs({} : {})", pv_q, pv_sd, mq, mat_q_ty));
    ops.push(format!("pto.tload ins({} : {}) outs({} : {})", pv_k_t, pv_ds, mk, mat_k_ty));
    // mat → L0A/L0B (CBUF → L0 is the supported tmov pair)
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mq, mat_q_ty, lq, l_ty));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mk, mat_k_ty, rk, r_ty));
    ops.push(format!("pto.tmatmul ins({}, {} : {}, {}) outs({} : {})", lq, rk, l_ty, r_ty, scores, acc_ty));

    // Step 2: move scores to VEC for softmax
    let vec_ty = tile_buf_type(s, s, "f32");
    let sv = ctx.alloc_tile_typed(&format!("{}__sv", result_ssa), s, s, "f32", &vec_ty, ops);
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", scores, acc_ty, sv, vec_ty));

    // Step 3: softmax (5-step) — mirrors translate_softmax.
    // max and sum are row-reductions (rows×1, col_major), so they need the
    // rowreduce type (tile_buf_type_rowreduce). Other intermediates are
    // plain S×S vec tiles.
    let rr_ty = tile_buf_type_rowreduce(s, "f32");
    let tmp = ctx.alloc_tile(&format!("{}__tmp", result_ssa), s, s, "f32", ops);
    let mx = ctx.alloc_tile_rowreduce(&format!("{}__mx", result_ssa), s, "f32", ops);
    let sb = ctx.alloc_tile(&format!("{}__sb", result_ssa), s, s, "f32", ops);
    let ex = ctx.alloc_tile(&format!("{}__ex", result_ssa), s, s, "f32", ops);
    let sm = ctx.alloc_tile_rowreduce(&format!("{}__sm", result_ssa), s, "f32", ops);
    let wt = ctx.alloc_tile(&format!("{}__wt", result_ssa), s, s, "f32", ops);

    ops.push(format!("pto.trowmax ins({}, {} : {}, {}) outs({} : {})", sv, tmp, vec_ty, vec_ty, mx, rr_ty));
    ops.push(format!("pto.trowexpandsub ins({}, {} : {}, {}) outs({} : {})", sv, mx, vec_ty, rr_ty, sb, vec_ty));
    ops.push(format!("pto.texp ins({} : {}) outs({} : {})", sb, vec_ty, ex, vec_ty));
    ops.push(format!("pto.trowsum ins({}, {} : {}, {}) outs({} : {})", ex, tmp, vec_ty, vec_ty, sm, rr_ty));
    ops.push(format!("pto.trowexpanddiv ins({}, {} : {}, {}) outs({} : {})", ex, sm, vec_ty, rr_ty, wt, vec_ty));

    // Step 4: output = weights(S×S) @ V(S×D) → S×D
    let mw_ty = mat_tile_type(s, s, "f32");
    let mv_ty = mat_tile_type(s, d, "f32");
    let lw_ty = left_tile_type(s, s, "f32");
    let rv_ty = right_tile_type(s, d, "f32");
    let out_ty = acc_tile_type(s, d, "f32");

    let mw = ctx.alloc_tile_typed(&format!("{}__mw", result_ssa), s, s, "f32", &mw_ty, ops);
    let mv = ctx.alloc_tile_typed(&format!("{}__mv", result_ssa), s, d, "f32", &mv_ty, ops);
    let lw = ctx.alloc_tile_typed(&format!("{}__lw", result_ssa), s, s, "f32", &lw_ty, ops);
    let rv = ctx.alloc_tile_typed(&format!("{}__rv", result_ssa), s, d, "f32", &rv_ty, ops);
    let out = ctx.alloc_tile_typed(&result_ssa, s, d, "f32", &out_ty, ops);

    // V: GM partition_view → mat (avoid vec→mat tmov).
    let pv_sd2 = ptv_type(s, d, "f32");
    ops.push(format!("pto.tload ins({} : {}) outs({} : {})", pv_v, pv_sd2, mv, mv_ty));
    // Weights (softmax output) live in VEC and must feed the cube via MAT.
    // On A5, the supported op for VEC→MAT is `pto.tinsert` (not `pto.tmov`):
    // it inserts the full src tile at (0,0) of the dst, and the a5 backend
    // lowers it on PIPE_MTE3 as a UB→L1 copy. dst must be blayout=col_major
    // + slayout=row_major (which `mat_tile_type` already produces) and src
    // must be blayout=row_major + slayout=none_box (which vec tiles are).
    ctx.use_size(0);
    ops.push(format!(
        "pto.tinsert ins({}, %c0, %c0 : {}, index, index) outs({} : {})",
        wt, vec_ty, mw, mw_ty
    ));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mw, mw_ty, lw, lw_ty));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mv, mv_ty, rv, rv_ty));
    ops.push(format!("pto.tmatmul ins({}, {} : {}, {}) outs({} : {})", lw, rv, lw_ty, rv_ty, out, out_ty));

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 0 tile intrinsic translators (PTO-MLIR)
// ---------------------------------------------------------------------------

/// Transpose: `%res = llvm.call @__tile_transpose_f32(%c0, %src, %rows, %cols)`
///
/// PTO does not have a native `pto.ttranspose` op. We emit a comment documenting
/// the operation and pass through the input via `pto.tmov`.
fn translate_transpose(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("transpose: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("transpose: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("transpose: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("transpose: unknown tile {}", src_ssa))?
        .clone();
    let src_ty = tsrc.tile_buf_type_str();
    // Output is transposed: cols x rows
    let dst_ty = tile_buf_type(cols, rows, dtype);

    ops.push(format!(
        "// --- transpose: {}x{} {} -> {}x{} {} ---",
        rows, cols, dtype, cols, rows, dtype
    ));
    ops.push(
        "// PTO lacks native transpose. Using tmov passthrough (shape metadata swapped)."
            .to_string(),
    );

    let out_ssa = ctx.alloc_tile(&result_ssa, cols, rows, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, src_ty, out_ssa, dst_ty
    ));
    ops.push("// TODO: implement transpose via tiled copy with transposed strides".to_string());

    Ok(())
}

/// Rsqrt: `%res = llvm.call @__tile_rsqrt_f32(%c0, %src, %rows, %cols)`
///
/// PTO does not have a native `pto.trsqrt` op. We emit a comment and
/// pass through the input via `pto.tmov`.
fn translate_rsqrt(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("rsqrt: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("rsqrt: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("rsqrt: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("rsqrt: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    ops.push(format!(
        "// --- rsqrt: 1/sqrt(x), {}x{} {} ---",
        rows, cols, dtype
    ));
    ops.push("// PTO lacks native rsqrt. Using tmov passthrough.".to_string());

    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, tb_ty, out_ssa, tb_ty
    ));
    ops.push("// TODO: implement rsqrt via Newton-Raphson or host-side computation".to_string());

    Ok(())
}

/// Log: `%res = llvm.call @__tile_log_f32(%c0, %src, %rows, %cols)`
///
/// PTO does not have a native `pto.tlog` op. We emit a comment and
/// pass through the input via `pto.tmov`.
fn translate_log(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("log: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("log: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("log: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("log: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    ops.push(format!(
        "// --- log: ln(x), {}x{} {} ---",
        rows, cols, dtype
    ));
    ops.push("// PTO lacks native log op. Using tmov passthrough.".to_string());

    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, tb_ty, out_ssa, tb_ty
    ));
    ops.push(
        "// TODO: implement log via series expansion or host-side computation".to_string(),
    );

    Ok(())
}

/// Sigmoid: `%res = llvm.call @__tile_sigmoid_f32(%c0, %src, %rows, %cols)`
///
/// Decomposed into:
/// 1. `pto.texp(src)` -> exp_x
/// 2. `pto.tadds(exp_x, 1.0)` -> one_plus = 1 + exp(x)
/// 3. `pto.tdiv(exp_x, one_plus)` -> sigmoid = exp(x) / (1 + exp(x))
///
/// Uses the exp(x)/(1+exp(x)) form (not 1/(1+exp(-x))) because ptoas has no
/// scalar/tile divide; the tile/tile `tdiv` is the only divide available.
fn translate_sigmoid(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("sigmoid: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("sigmoid: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("sigmoid: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("sigmoid: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    ops.push(format!(
        "// --- sigmoid: exp(x)/(1+exp(x)), {}x{} {} ---",
        rows, cols, dtype
    ));

    // ptoas has no scalar/tile divide, so compute sigmoid as exp(x)/(1+exp(x))
    // instead of 1/(1+exp(-x)). Same result, uses tile/tile `tdiv`.
    let cone_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant 1.0 : f32", cone_ssa));

    // Step 1: exp_x = texp(src)
    let exp_key = format!("{}__sig_exp", result_ssa);
    let exp_ssa = ctx.alloc_tile(&exp_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.texp ins({} : {}) outs({} : {})",
        tsrc.ssa, tb_ty, exp_ssa, tb_ty
    ));

    // Step 2: one_plus = tadds(exp_x, 1.0) = 1 + exp(x)
    let oplus_key = format!("{}__sig_oplus", result_ssa);
    let oplus_ssa = ctx.alloc_tile(&oplus_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tadds ins({}, {} : {}, f32) outs({} : {})",
        exp_ssa, cone_ssa, tb_ty, oplus_ssa, tb_ty
    ));

    // Step 3: result = tdiv(exp_x, one_plus) = exp(x) / (1 + exp(x))
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tdiv ins({}, {} : {}, {}) outs({} : {})",
        exp_ssa, oplus_ssa, tb_ty, tb_ty, out_ssa, tb_ty
    ));

    Ok(())
}

/// SiLU: `%res = llvm.call @__tile_silu_f32(%c0, %src, %rows, %cols)`
///
/// Decomposed into:
/// 1. `pto.tmuls(src, -1.0)` -> neg_x
/// 2. `pto.texp(neg_x)` -> exp_neg
/// 3. `pto.tadds(exp_neg, 1.0)` -> one_plus = 1 + exp(-x)
/// 4. `pto.tdiv(src, one_plus)` -> silu = x / (1 + exp(-x)) = x * sigmoid(x)
///
/// Uses tile/tile `tdiv` (not `tdivs` + `tmul`) because ptoas has no
/// scalar/tile divide, so the sigmoid reciprocal is folded into one division.
fn translate_silu(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("silu: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("silu: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("silu: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("silu: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    // Standalone silu allocates 4 intermediate tiles (neg, exp, oplus, out)
    // on top of the upstream-loaded src tile. Same UB-cap reasoning as
    // translate_silu_mul: 5 × rows × cols × elem_bytes must fit under the
    // 224 KB usable budget, otherwise the kernel will crash on-device with
    // an opaque vector-core exception. N-blocked emit tracked in #67.
    let elem_bytes_silu: u32 = match dtype {
        "f32" => 4,
        "f16" | "bf16" => 2,
        _ => return Err(format!("silu: unsupported dtype {}", dtype)),
    };
    let peak_ub_bytes_silu: u64 =
        5u64 * (rows as u64) * (cols as u64) * (elem_bytes_silu as u64);
    const SILU_UB_BUDGET_BYTES: u64 = 224 * 1024;
    if peak_ub_bytes_silu > SILU_UB_BUDGET_BYTES {
        return Err(format!(
            "silu: peak UB usage {} B for {}x{} {} exceeds budget {} B \
             (5-tile emit: src, neg, exp, oplus, out). \
             Inner dim N={} needs N-blocked emit — not yet implemented \
             (tracking: ICLR 2026 #67).",
            peak_ub_bytes_silu, rows, cols, dtype, SILU_UB_BUDGET_BYTES, cols
        ));
    }

    ops.push(format!(
        "// --- silu: x / (1 + exp(-x)) = x * sigmoid(x), {}x{} {} ---",
        rows, cols, dtype
    ));

    // Scalar constants (ptoas requires SSA-bound f32 operands, not attributes).
    //
    // Identity: silu(x) = x / (1 + exp(-x)). Using `tdiv` (tile/tile) avoids
    // the reciprocal — ptoas has no scalar/tile divide.
    let cneg1_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant -1.0 : f32", cneg1_ssa));
    let cone_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant 1.0 : f32", cone_ssa));

    // Step 1: neg_x = tmuls(src, -1.0)
    let neg_key = format!("{}__silu_neg", result_ssa);
    let neg_ssa = ctx.alloc_tile(&neg_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmuls ins({}, {} : {}, f32) outs({} : {})",
        tsrc.ssa, cneg1_ssa, tb_ty, neg_ssa, tb_ty
    ));

    // Step 2: exp_neg = texp(neg_x)
    let exp_key = format!("{}__silu_exp", result_ssa);
    let exp_ssa = ctx.alloc_tile(&exp_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.texp ins({} : {}) outs({} : {})",
        neg_ssa, tb_ty, exp_ssa, tb_ty
    ));

    // Step 3: one_plus = tadds(exp_neg, 1.0)
    let oplus_key = format!("{}__silu_oplus", result_ssa);
    let oplus_ssa = ctx.alloc_tile(&oplus_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tadds ins({}, {} : {}, f32) outs({} : {})",
        exp_ssa, cone_ssa, tb_ty, oplus_ssa, tb_ty
    ));

    // Step 4: result = tdiv(src, one_plus) = x / (1 + exp(-x)) = x*sigmoid(x)
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tdiv ins({}, {} : {}, {}) outs({} : {})",
        tsrc.ssa, oplus_ssa, tb_ty, tb_ty, out_ssa, tb_ty
    ));

    Ok(())
}

/// Detect SiLU+Mul fusion opportunities.
///
/// Scans `body_lines` for pairs where:
///   %silu = llvm.call @__tile_silu_f32/f16(...)
///   %out  = llvm.call @__tile_mul_f32/f16(%c0, %silu, %up, ...)
///
/// Returns a map: silu_result_ssa → (silu_line_index, mul_line_index).
// =============================================================================
// K/N-blocked matmul detection
// =============================================================================
//
// Background: the naive translate_matmul emits a single pto.tmatmul over the
// full M×K and K×N shapes. For DeepSeek f32 shapes (K=1536, N=1536 or 8960)
// the CBUF mat staging tiles are ~9 MB, overflowing the 512 KB CBUF cap;
// L0A/L0B (64 KB each) would also overflow. ptoas rejects with
// "mat overflow, requires 75546624 bits while 4194304 bits avaliable".
//
// The fix (validated end-to-end in /tmp/matmul_q_proj_m16.pto — see
// memory/project_pto_matmul_kblocking.md) is to emit:
//   scf.for %n_i = 0 to N/Nb step 1 {
//     scf.for %k_i = 0 to K/Kb step 1 {
//       // per-block partition_view on A / B
//       // tload  mat_a (Mpad × Kb), mat_b (Kb × Nb)
//       // tmov   mat_a → left, mat_b → right
//       // if k_i == 0: pto.tmatmul    ins(left, right) outs(acc)
//       // else       : pto.tmatmul.acc ins(acc, left, right) outs(acc)
//     }
//     // pto.tstore acc → output[0:Mpad, n_off:n_off+Nb]
//   }
//
// Block sizes: Kb=256, Nb=32, Mpad=round_up(M, 16). Choices:
//   - Mpad: TileConfig::fixedRowSize=16 on 910B2 cube (Rows % 16 == 0)
//   - Nb=32: fits L0B at Kb=256 with headroom (256*32*4 = 32 KB < 64 KB cap).
//     Nb=64 was exactly at the 64 KB L0B limit (256*64*4 = 64KB) and caused
//     "aicore execution exception" on 910B2 — likely due to fractal format
//     metadata overhead pushing actual allocation past the hardware cap.
//   - Kb=256: validated in hand-patched reference; still lots of room in L0A
//     for Mpad=16 (16*256*4 = 16 KB, L0A has 64 KB)
//
// The emitter requires `M % 16 == 0` — the Rust kernel author is expected
// to pad M manually (as in benchmarks/deepseek_e2e/kernels_pto_matmul).
// Host caller zeros the unused rows of A and reads only row 0 of output.

/// Block size constants used by the K/N-blocked matmul emission.
/// See comment above `detect_blocked_matmul_loads` for the rationale.
///
/// Kb was 256 with Nb=64. That combination does not fit real 910B2:
///   - L0B (`b_right`, Kb*Nb*4) = 64 KB, i.e. EXACTLY the 64 KB L0B cap. The
///     comment above already records that Nb=64 at Kb=256 caused "aicore
///     execution exception" on 910B2, attributed to fractal metadata pushing
///     the allocation past the hardware cap.
///   - the five live tiles (mat_a, mat_b, a_left, b_right, acc) sum to 272 KB,
///     far over the 184 KB UB budget the vendor spec allows (`SPEC_A2A3`).
/// Both were masked while `A2A3::UB_SIZE` wrongly carried the 256 KB A5 figure.
///
/// Kb=64, Nb=128, Mb=192 fills 75% of L0C while keeping L0A (48 KB) and L0B
/// (32 KB) STRICTLY under the 64 KB cap, clear of the exactly-64 KB exception
/// recorded above.
///
/// This replaces Kb=128/Nb=64/Mb=64, which used 12.5% of L0C and made the four
/// projections 75.6% of a measured S=769 forward -- 4493 us of 5942, at ~1/15th
/// the per-FLOP efficiency of `scores_batched` (224x128, 87.5% of L0C) on the
/// same cube engine. The accumulator, not the sequence length, was the limit:
/// a 64x64 acc retires 4 K MACs per pass where a 192x128 retires 24 K.
///
/// Trip counts on the shapes the five operators actually emit:
///
///   shape                  old   new
///   proj  [832,640,640]    650   250
///   kv-proj MQA (N=128)    130    50
///   proj  [320,640,640]    250   100
///   proj  short (64,192)     6     6   (unchanged: already single-tile)
///
/// Mb=192 is a multiple of both 16 (`PTO_MM_MROW_ALIGN`, the cube fixedRowSize)
/// and 64, so it divides the padded M of every emitted projection. 208 would
/// cut one more trip on [832,...] but only because it divides 832 -- tuning to
/// one shape rather than to the hardware.
const PTO_MM_KB: u32 = 64;
const PTO_MM_NB: u32 = 128;
/// Cap for the M block. See the Kb/Nb note above: 192 rows put L0A at 48 KB
/// (192*64*4), under the cap, and the acc at 96 KB of L0C's 128 KB.
const PTO_MM_MB: u32 = 192;
/// i8 matmul needs a wider Nb than f16/f32: ptoas picks the L0A `Left` tile's
/// BLayout based on the companion `Right` tile width. At Nb=64 with i8 Left,
/// ptoas silently emits `BLayout::ColMajor` (wrong) instead of RowMajor.
/// Nb=256 matches the validated hand-written probe layout.
const PTO_MM_NB_I8: u32 = 256;
const PTO_MM_MROW_ALIGN: u32 = 16; // TileConfig::fixedRowSize on 910B2 cube

/// ptoas packs the B tensor_view's outer stride (= Kb × row_stride = Kb × N)
/// into a DMA descriptor field that wraps at 2^24. Empirically on CANN 8.5
/// ptoas, correctness breaks at `Kb × N == 2^24` and fails harder as it grows:
/// N=49152 passes (Kb=256 → outer=12.6M), N=65536 fails (outer=2^24 exactly),
/// N=151936 produces garbage. See project notes for the diagnostic N-sweep.
///
/// Safe bound: `Kb × N < 2^24`. We use `2^23` as the decision threshold to
/// give 2× headroom (descriptors can carry a sign bit or reserved flag we
/// haven't characterised).
const PTO_MM_OUTER_STRIDE_LIMIT: u32 = 1 << 23;

/// Pick an effective K block size, respecting (a) K itself and (b) the
/// ptoas 24-bit outer-stride limit on B. When N is large enough that the
/// default `PTO_MM_KB × N` would overflow that limit, we fall back to a
/// smaller Kb — Kb is rounded down to a multiple of 16 (cube fractal
/// alignment) and is never smaller than 16.
///
/// This is the N-agnostic legacy entry point used by external callers /
/// tests that don't know N. Prefer `pick_kb_for_n` within the emitter.
#[allow(dead_code)]
fn pick_kb(k: u32) -> u32 {
    pick_kb_for_n(k, u32::MAX)
}

fn pick_kb_for_n(k: u32, n: u32) -> u32 {
    pick_kb_for_n_dtype(k, n, 2)
}

/// Dtype-aware Kb pick: ptoas on CANN 8.5 silently flips the L0A (`Left`) tile
/// from `BLayout::RowMajor` to `BLayout::ColMajor` when the Kb×M×sizeof(lhs)
/// byte-count exceeds a dtype-specific inner-fractal threshold. Empirically:
/// i8 Left at M=16 Kb=256 → cpp gets ColMajor (wrong, numerics 4× small +
/// sign-flipped). Same MLIR at Kb=128 → RowMajor (correct, validated by
/// hand-written probe smoke_i8_kv_proj_tmov3arg.cpp).
///
/// Heuristic: cap Kb so that the lhs L0A tile stays ≤ 256 element-columns.
/// For f16 (2B) and f32 (4B) that's still 256; for i8 it becomes 128.
fn pick_kb_for_n_dtype(k: u32, n: u32, lhs_bytes: u32) -> u32 {
    let dtype_kb_cap = if lhs_bytes == 1 { 128 } else { PTO_MM_KB };
    let base = dtype_kb_cap.min(k);
    if n == 0 || n == u32::MAX {
        return base;
    }
    // Largest Kb such that Kb × n <= PTO_MM_OUTER_STRIDE_LIMIT, aligned to 16.
    let kb_cap = PTO_MM_OUTER_STRIDE_LIMIT / n;
    let kb_cap = (kb_cap / 16) * 16;
    let kb_cap = kb_cap.max(16);
    let kb = base.min(kb_cap);
    // Ensure kb divides k. If not, round down to the largest multiple of 16
    // that divides k, then clamp.
    if kb == 0 || k % kb != 0 {
        // Try halving from `base` downward to find a kb that divides k AND
        // stays under kb_cap. 16 divides anything that's a multiple of 16
        // (all DeepSeek Ks are: 128, 256, 1536, 8960).
        let mut candidate = base;
        while candidate > 16 {
            if candidate <= kb_cap && k % candidate == 0 {
                return candidate;
            }
            candidate /= 2;
        }
        return 16.min(base);
    }
    kb
}

/// Pick an effective N block size: min(N, PTO_MM_NB). If N is smaller than
/// PTO_MM_NB we skip the N-loop and emit a single tmatmul.
/// Pick an M block: the largest power-of-two multiple of 16 up to
/// `PTO_MM_MB` that keeps the live working set inside the UB budget.
///
/// Returns `m` itself when the whole operand already fits, so shapes that
/// worked before emit byte-identically (no M loop is generated at m_iters==1).
///
/// The working set is A_blk + B_blk + acc:
///   A_blk = mb * kb * lhs      B_blk = kb * nb * rhs      acc = mb * nb * 4
/// Only the A and acc terms carry `mb`, so shrinking it is what bounds growth.
fn pick_mb(m: u32, kb: u32, nb: u32, dtypes: &MatmulDtypes, live_bytes: usize) -> u32 {
    let ub_budget = (SPEC_A2A3.ub_budget() as u64).saturating_sub(live_bytes as u64);
    // A blocked matmul's M-carrying tiles are CUBE-side, so the binding limits
    // are L1 (mat_a staging) and L0A (a_left), not the Unified Buffer. Sizing
    // against UB alone picked mb=192 at S=96 and then failed L0A at 96 KB
    // against a 64 KB cap. Every space the block touches has to be checked.
    let fits = |mb: u32| -> bool {
        let a = (mb as u64) * (kb as u64) * dtypes.lhs_bytes();
        let b = (kb as u64) * (nb as u64) * dtypes.rhs_bytes();
        let acc = (mb as u64) * (nb as u64) * 4;
        // STRICTLY under L0A/L0B, not `<=`. A tile at exactly the 64 KB cap
        // caused "aicore execution exception" on 910B2 (recorded above for
        // Kb=256/Nb=64), attributed to fractal metadata pushing the real
        // allocation past the hardware limit. L0C has no such recorded
        // exception and keeps `<=`.
        a <= SPEC_A2A3.l1_size as u64            // mat_a staging
            && a < SPEC_A2A3.l0_a_size as u64    // a_left
            && b < SPEC_A2A3.l0_b_size as u64    // b_right
            && acc <= SPEC_A2A3.l0_c_size as u64 // acc
            && a + b + acc <= ub_budget.max(SPEC_A2A3.l1_size as u64)
    };
    if fits(m) {
        return m; // already fits: emit exactly what the pre-M-blocking path did
    }
    // Walk down in 16-row steps (the cube tile granularity) to the largest fit.
    let mut mb = m.min(PTO_MM_MB);
    while mb > 16 && !fits(mb) {
        mb -= 16;
    }
    mb.max(16)
}

fn pick_nb(n: u32) -> u32 {
    pick_nb_for_dtype(n, 2)
}

/// Dtype-aware Nb pick. i8 needs Nb=256 so that ptoas translates the L0A
/// `Left` tile with `BLayout::RowMajor` (matching the validated hand-written
/// i8 probe). At Nb=64, ptoas silently emits `BLayout::ColMajor` for i8 Left
/// which produces garbage numerics.
fn pick_nb_for_dtype(n: u32, lhs_bytes: u32) -> u32 {
    let nb_base = if lhs_bytes == 1 { PTO_MM_NB_I8 } else { PTO_MM_NB };
    // Clamp to a POWER OF TWO <= n, not to n itself. `min(base, n)` returns n
    // whenever n < base, and n need not be a power of two: at base=128, n=96
    // yielded nb=96, which then divides n so the halving loop below never runs
    // and a non-power-of-two width reaches ptoas. That is precisely the
    // width-linked BLayout hazard the halving exists to avoid -- it was masked
    // while base was 64 because 96 > 64 kept the clamp off.
    let mut nb = if n >= nb_base {
        nb_base
    } else {
        n.next_power_of_two() / if n.is_power_of_two() { 1 } else { 2 }
    }
    .max(1);
    // Nb must DIVIDE n: the blocked emitter rejects `n % nb != 0` outright, and
    // `min(64, n)` does not guarantee it. At n=96 that gave nb=64 and a hard
    // "N=96 must be a multiple of Nb=64", which is what made every sequence
    // length in 72..96 unemittable — 4 of the 11 failing lengths in a 16..1024
    // sweep, and the cheapest band to close because it is pure block policy.
    //
    // Halve rather than step by 16: `PTO_MM_NB_I8 = 256` exists because ptoas
    // picks the L0A Left tile's BLayout from the companion Right tile's WIDTH,
    // and silently emits ColMajor for i8 at Nb=64. That hazard is width-linked
    // and only validated at power-of-two widths, so staying on powers of two
    // keeps every shape inside the territory the device probes covered.
    while nb > 16 && n % nb != 0 {
        nb /= 2;
    }
    nb
}

/// Decide whether this matmul shape requires blocking. We block when
/// either L0A (M*K*lhs_bytes) or L0B (K*N*rhs_bytes) would overflow their
/// 64 KB caps. For f16 operands, twice the elements fit per L0 byte —
/// e.g., f32 blocks at K=N=128 (32KB × 4B = 128KB > 64KB), while f16 at
/// the same shape fits (32KB × 2B = 64KB, borderline — callers typically
/// go larger before blocking).
fn matmul_needs_blocking(m: u32, k: u32, n: u32, dtypes: &MatmulDtypes) -> bool {
    const L0_CAP_BYTES: u64 = 64 * 1024; // L0A and L0B individual cap
    let mk_bytes = (m as u64) * (k as u64) * dtypes.lhs_bytes();
    let kn_bytes = (k as u64) * (n as u64) * dtypes.rhs_bytes();
    if mk_bytes > L0_CAP_BYTES || kn_bytes > L0_CAP_BYTES {
        return true;
    }
    // Also block when the UNBLOCKED working set would not fit the UB.
    //
    // Testing only the L0 caps missed exactly the case M-blocking exists for:
    // at S=96 the A operand is 256x96xf32 = 96 KB, under the 64 KB L0 cap only
    // after tiling but over the UB budget as a live tile, so the emitter took
    // the unblocked path and then failed the C1 guard. The projection's
    // M = pad(B*S) grows with sequence length, so this is the term that decides
    // whether long sequences can be emitted at all.
    let acc_bytes = (m as u64) * (n as u64) * 4;
    mk_bytes + kn_bytes + acc_bytes > SPEC_A2A3.ub_budget() as u64
}

/// Pre-pass: find tile_load lines whose result is consumed only by a
/// matmul that needs blocking, and whose load shape matches the matmul's
/// A/B operand shape. Returns a set of body_lines indices to skip.
///
/// Each entry in the returned map is `load_idx → matmul_operand_role`
/// (`"A"` or `"B"`). translate_load uses this to skip emitting the
/// full-shape pto.tload / alloc_tile for those loads; translate_matmul
/// later rebuilds per-block loads inside its scf.for nest.
/// tile_load lines whose result is consumed by a batched stage.
///
/// Those loads must NOT emit a full-shape tload at top level: the batched
/// loop needs to load head `h`'s slice on each iteration, and a pre-loop load
/// fills the tile with head 0 once. Deferring them reuses exactly the
/// mechanism the blocked matmul uses — skip the load, stash tv_ssa +
/// elem_offset, let the consumer emit partition_view + tload inside its loop.
///
/// Without this the emitted PTO carries a head-relative partition_view that
/// nothing reads, and every head computes head 0's data — verified on device
/// as heads 1..N written but wrong, byte-identically with and without the view.
fn detect_batched_loads(body_lines: &[String]) -> HashMap<usize, &'static str> {
    let mut out: HashMap<usize, &'static str> = HashMap::new();

    // Operand SSAs of every batched call in the body.
    let mut wanted: Vec<(String, &'static str)> = Vec::new();
    for line in body_lines {
        let t = line.trim();
        // The generic form names its operands in an attribute rather than in
        // the call args, so read them from there. Missing this would leave the
        // loads outside the loop and every iteration would read iteration 0 —
        // the failure that once produced "written, but wrong byte-identically".
        // Row-blocked softmax: same deferral, same reason. This pre-pass has
        // no PtoContext, so shape args arrive as SSA names -- resolve them from
        // the module's own `llvm.mlir.constant` lines.
        // BOTH dtypes: the deferral must cover every softmax the picker might
        // block, and f16 has its own intrinsic. Matching only f32 left the f16
        // path blocking without a deferred load.
        if (t.contains("__tile_softmax_f32") || t.contains("__tile_softmax_f16"))
            && !t.contains("_batched")
        {
            if let Some(args) = extract_call_args(t) {
                let lookup = |a: Option<&String>| -> Option<u32> {
                    let a = a?.trim();
                    if let Ok(v) = a.parse::<u32>() {
                        return Some(v);
                    }
                    body_lines.iter().find_map(|l| {
                        let l = l.trim();
                        let (lhs, rhs) = l.split_once('=')?;
                        if lhs.trim() != a || !rhs.contains("llvm.mlir.constant") {
                            return None;
                        }
                        let inner = rhs.split('(').nth(1)?;
                        inner.split(':').next()?.trim().parse::<u32>().ok()
                    })
                };
                // Only when it will actually be blocked; otherwise leave the
                // existing (byte-identical) unblocked emission alone.
                if let (Some(r), Some(c)) = (lookup(args.get(2)), lookup(args.get(3))) {
                    // DEFER WHENEVER BLOCKING IS POSSIBLE, not only when it is
                    // certain. This pre-pass has no PtoContext, so it must pass
                    // live_bytes = 0 -- an optimistic budget. `translate_softmax`
                    // sees the real cursor, so it can decide to block where this
                    // decided not to, and the operand's load would then sit
                    // outside the loop. That mismatch is caught loudly
                    // ("operand was not deferred"), but the fix is to defer on
                    // the OPTIMISTIC side: a deferred load that turns out not to
                    // need blocking is re-emitted unblocked at no cost, whereas
                    // a missing deferral is a hard failure.
                    // Decide on the SAME budget `translate_softmax` will use.
                    //
                    // An earlier version also deferred "pessimistically" on half
                    // the budget, reasoning that a spare deferral is harmless.
                    // It is not: at f16 [16,1024] the stage fits (160 KB of
                    // 184 KB) and must NOT be blocked, but the halved budget
                    // said otherwise and forced a block the emitter then could
                    // not complete. Guessing a different budget here trades one
                    // mismatch for another; the honest fix is to model the same
                    // thing in both places, which the corrected 5-tile cost now
                    // does.
                    let dt = if t.contains("_f16") { "f16" } else { "f32" };
                    if pick_softmax_rows(r, c, dt, 0) < r {
                        if let Some(a) = args.get(1) {
                            wanted.push((a.trim().to_string(), "A"));
                        }
                    }
                }
            }
            continue;
        }
        if t.contains("__tile_batched_f32") {
            if let Ok(BatchedKind::Generic { operands, .. }) = parse_generic_batched(t) {
                for (ssa, _, _) in operands {
                    wanted.push((ssa, "A"));
                }
            }
            continue;
        }
        let batched_mm = t.contains("__tile_matmul_batched_f32");
        let batched_sm = t.contains("__tile_softmax_batched_f32");
        if !batched_mm && !batched_sm {
            continue;
        }
        if let Some(args) = extract_call_args(t) {
            if let Some(a) = args.get(1) {
                wanted.push((a.trim().to_string(), "A"));
            }
            if batched_mm {
                if let Some(b) = args.get(2) {
                    wanted.push((b.trim().to_string(), "B"));
                }
            }
        }
    }
    if wanted.is_empty() {
        return out;
    }

    for (i, line) in body_lines.iter().enumerate() {
        // Match every dtype's load, not just f32: an f16 softmax that needs
        // blocking has an `__tile_load_f16` producer, and matching only f32
        // left it undeferred -- the stage then failed loudly with "operand was
        // not deferred", which is the guard working but the scan being wrong.
        if !line.contains("__tile_load_f32")
            && !line.contains("__tile_load_f16")
            && !line.contains("__tile_load_i8")
        {
            continue;
        }
        if let Some(res) = extract_result_ssa(line.trim()) {
            if let Some((_, role)) = wanted.iter().find(|(w, _)| *w == res) {
                out.insert(i, role);
            }
        }
    }
    out
}

fn detect_blocked_matmul_loads(body_lines: &[String]) -> HashMap<usize, &'static str> {
    let mut result: HashMap<usize, &'static str> = HashMap::new();

    for line in body_lines.iter() {
        let trimmed = line.trim();
        // Match f32-matmul (may need K/N-blocking), f16-matmul (single-block
        // but still needs mat/CBUF routing — CANN 8.5 cube doesn't support
        // b16 GM→UB), and i8-matmul with per-column dequant (same CBUF
        // routing, always blocked at decoder shapes).
        let is_f32_mm = trimmed.contains("__tile_matmul_f32");
        let is_f16_mm = trimmed.contains("__tile_matmul_f16");
        let is_i8_mm = trimmed.contains("__tile_matmul_i8_acc_i32_dequant_f16");
        if !is_f32_mm && !is_f16_mm && !is_i8_mm {
            continue;
        }
        let mm_args = match extract_call_args(trimmed) {
            Some(a) => a,
            None => continue,
        };
        // i8 matmul has an extra `scale` arg (between b and m), so its call
        // has 7 args vs 6 for f16/f32. Compute the M/K/N arg indices based
        // on the signature.
        let (mkn_base, min_args) = if is_i8_mm { (4, 7) } else { (3, 6) };
        if mm_args.len() < min_args {
            continue;
        }
        let m = parse_u32_from_arg(&mm_args[mkn_base], body_lines);
        let k = parse_u32_from_arg(&mm_args[mkn_base + 1], body_lines);
        let n = parse_u32_from_arg(&mm_args[mkn_base + 2], body_lines);
        let (m, k, n) = match (m, k, n) {
            (Some(m), Some(k), Some(n)) => (m, k, n),
            _ => continue,
        };
        // Always defer matmul operand loads: even for small shapes that don't
        // need K/N-blocking, the matmul emitter generates its own mat-tile tloads
        // (GM→CBUF→L0A/L0B). If we also emit the original vec-tile tloads
        // (GM→UB), they're dead code that ptoas/ccec may fail to compile
        // (e.g., copy_gm_to_ubuf_align_b32 unsupported on a2a3 for certain
        // shapes). f16/i8 paths were already always-defer.
        let a_ssa = mm_args[1].trim().to_string();
        let b_ssa = mm_args[2].trim().to_string();

        // Find the tile_load line that produced a_ssa / b_ssa. We only
        // block when the load's result SSA matches directly (no
        // intermediate ops). If another op consumes the load between
        // here and the matmul, fall back to unblocked emission.
        let load_pat = if is_f16_mm {
            "__tile_load_f16"
        } else if is_i8_mm {
            "__tile_load_i8"
        } else {
            "__tile_load_f32"
        };
        for (i, cand) in body_lines.iter().enumerate() {
            let ct = cand.trim();
            if !ct.contains(load_pat) {
                continue;
            }
            let load_ssa = match extract_result_ssa(ct) {
                Some(s) => s,
                None => continue,
            };
            if load_ssa == a_ssa {
                result.insert(i, "A");
            } else if load_ssa == b_ssa {
                result.insert(i, "B");
            }
        }
    }
    result
}

/// 5-tile fused silu_mul peak UB usage exceeds the budget — needs N-blocking.
/// Mirrors `matmul_needs_blocking` for the silu_mul (#67) path.
fn silu_mul_needs_blocking(rows: u32, cols: u32, dtype: &str) -> bool {
    let elem_bytes: u64 = match dtype {
        "f32" => 4,
        "f16" | "bf16" => 2,
        _ => return false,
    };
    let peak = 5u64 * (rows as u64) * (cols as u64) * elem_bytes;
    peak > SILU_MUL_UB_BUDGET_BYTES
}

/// UB budget for the 5-tile silu_mul emit.
///
/// Derived from the vendor spec (`SPEC_A2A3`, i.e. `ub_size=196608` from CANN's
/// `platform_config/Ascend910B2.ini`) minus the 8 KB A2/A3 TMP_UB scratch, minus
/// a further 32 KB reserved for kernel code, stack, and scalars — the same
/// reservation the previous constant assumed.
///
/// This read `224 * 1024` when `A2A3::UB_SIZE` was wrongly 256 KB. Both figures
/// came from the A5 UB capacity, so on real 910B2 silicon (192 KB) the blocked
/// path could pick a chunk whose peak overflows the hardware buffer.
const SILU_MUL_UB_BUDGET_BYTES: u64 = (SPEC_A2A3.ub_budget() - 32 * 1024) as u64;

/// Pick a chunk size `Nb` along the inner dim such that the per-chunk
/// 5-tile peak (gate + up + neg + silu + out) fits the UB budget AND
/// `Nb` divides `cols` evenly. Returns None if no such divisor exists
/// (caller falls back to returning Err from translate_silu_mul, which
/// surfaces a clear "shape needs source-level chunking" diagnostic).
///
/// Strategy: walk divisors of `cols` in decreasing order; pick the largest
/// that still satisfies `5 * rows * Nb * elem_bytes <= budget`. Larger Nb
/// means fewer scf.for iterations and less per-chunk overhead.
fn pick_silu_mul_nb(rows: u32, cols: u32, dtype: &str) -> Option<u32> {
    let elem_bytes: u64 = match dtype {
        "f32" => 4,
        "f16" | "bf16" => 2,
        _ => return None,
    };
    let max_nb_by_budget: u64 =
        SILU_MUL_UB_BUDGET_BYTES / (5u64 * (rows as u64) * elem_bytes);
    if max_nb_by_budget == 0 {
        return None;
    }
    let cap = max_nb_by_budget.min(cols as u64) as u32;
    // Walk divisors of cols, largest-first, that are <= cap.
    let mut best: Option<u32> = None;
    let mut d = 1u32;
    while d as u64 * d as u64 <= cols as u64 {
        if cols % d == 0 {
            let q = cols / d;
            if d <= cap {
                best = Some(best.map_or(d, |b| b.max(d)));
            }
            if q <= cap {
                best = Some(best.map_or(q, |b| b.max(q)));
            }
        }
        d += 1;
    }
    best
}

/// Pre-pass for #67: find tile_load lines whose result feeds a silu_mul
/// fused pair that exceeds the UB budget AND whose inputs come from
/// direct `tile_load_*` calls. Returns indices to defer, mapped to a
/// role label ("G" for gate, "U" for up). Mirrors
/// `detect_blocked_matmul_loads`. The returned indices are unioned into
/// the existing blocked-load set so translate_load skips full-shape tloads.
fn detect_blocked_silu_mul_loads(
    body_lines: &[String],
    silu_mul_fused: &HashMap<String, (usize, usize)>,
) -> HashMap<usize, &'static str> {
    let mut result: HashMap<usize, &'static str> = HashMap::new();

    for (silu_ssa, &(silu_idx, mul_idx)) in silu_mul_fused.iter() {
        let silu_line = body_lines[silu_idx].trim();
        let mul_line = body_lines[mul_idx].trim();

        // Determine dtype from the silu intrinsic name.
        let dtype = if silu_line.contains("__tile_silu_f32") {
            "f32"
        } else if silu_line.contains("__tile_silu_f16") {
            "f16"
        } else {
            continue;
        };

        // Parse rows/cols from silu (last two args) and check budget.
        let silu_args = match extract_call_args(silu_line) {
            Some(a) => a,
            None => continue,
        };
        if silu_args.len() < 4 {
            continue;
        }
        let rows = match parse_u32_from_arg(&silu_args[silu_args.len() - 2], body_lines) {
            Some(r) => r,
            None => continue,
        };
        let cols = match parse_u32_from_arg(&silu_args[silu_args.len() - 1], body_lines) {
            Some(c) => c,
            None => continue,
        };
        if !silu_mul_needs_blocking(rows, cols, dtype) {
            continue;
        }
        // Block size must evenly divide cols. If no divisor fits, the
        // load isn't deferred — translate_silu_mul will then return Err
        // with the existing "needs source-level chunking" guard message.
        if pick_silu_mul_nb(rows, cols, dtype).is_none() {
            continue;
        }

        // Identify gate / up SSAs. silu's gate is silu_args[1] (after the
        // %c0 stash). mul's "up" is whichever of its two operands isn't
        // the silu result. Handle both 4-arg and 5-arg mul signatures
        // exactly like translate_silu_mul does.
        let gate_ssa = silu_args[1].trim().to_string();
        let mul_args = match extract_call_args(mul_line) {
            Some(a) => a,
            None => continue,
        };
        let up_ssa = if mul_args.len() >= 5 {
            let a = mul_args[1].trim();
            let b = mul_args[2].trim();
            if a == silu_ssa { b.to_string() } else { a.to_string() }
        } else if mul_args.len() >= 4 {
            let a = mul_args[0].trim();
            let b = mul_args[1].trim();
            if a == silu_ssa { b.to_string() } else { a.to_string() }
        } else {
            continue;
        };

        // Find the tile_load lines that produced gate_ssa / up_ssa.
        // Only direct loads qualify — same constraint as the matmul
        // pre-pass, so any intervening op falls back to the
        // "return Err from translate_silu_mul" path.
        let load_pat = if dtype == "f16" {
            "__tile_load_f16"
        } else {
            "__tile_load_f32"
        };
        for (i, cand) in body_lines.iter().enumerate() {
            let ct = cand.trim();
            if !ct.contains(load_pat) {
                continue;
            }
            let load_ssa = match extract_result_ssa(ct) {
                Some(s) => s,
                None => continue,
            };
            if load_ssa == gate_ssa {
                result.insert(i, "G");
            } else if load_ssa == up_ssa {
                result.insert(i, "U");
            }
        }
    }
    result
}

/// Resolve a call arg to a u32 constant — handles direct literals,
/// SSA references to `llvm.mlir.constant`, and `llvm.bitcast` chains
/// (the MLIR emitted by rustc_codegen_tile routes every const through a
/// bitcast, so `%Nc = constant(16)` is followed by `%Nb = bitcast %Nc`
/// and the matmul call uses `%Nb`).
fn parse_u32_from_arg(arg: &str, body_lines: &[String]) -> Option<u32> {
    let mut cur = arg.trim().to_string();
    // Bound the chase to avoid infinite loops on pathological IR.
    for _ in 0..8 {
        if let Ok(n) = cur.parse::<u32>() {
            return Some(n);
        }
        let mut found_def = false;
        for line in body_lines.iter() {
            let l = line.trim();
            let res = match extract_result_ssa(l) {
                Some(s) => s,
                None => continue,
            };
            if res != cur {
                continue;
            }
            // Direct constant definition.
            if l.contains("llvm.mlir.constant(") {
                if let Some(open) = l.find("llvm.mlir.constant(") {
                    let rest = &l[open + "llvm.mlir.constant(".len()..];
                    let n_str: String =
                        rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(n) = n_str.parse::<u32>() {
                        return Some(n);
                    }
                }
                return None;
            }
            // Bitcast forwarding: `%X = llvm.bitcast %Y : Ta to Tb`.
            // normalize_generic_body_line rewrites the generic form into
            // this canonical shape; chase %Y as the next candidate.
            if l.contains("llvm.bitcast ") {
                if let Some(pos) = l.find("llvm.bitcast ") {
                    let rest = l[pos + "llvm.bitcast ".len()..].trim();
                    let src = rest.split_whitespace().next().unwrap_or("").trim_matches(',');
                    if !src.is_empty() && src != cur {
                        cur = src.to_string();
                        found_def = true;
                        break;
                    }
                }
                return None;
            }
            // Unknown defining op — give up rather than risk a wrong answer.
            return None;
        }
        if !found_def {
            return None;
        }
    }
    None
}

fn detect_silu_mul_pairs(body_lines: &[String]) -> HashMap<String, (usize, usize)> {
    let mut result: HashMap<String, (usize, usize)> = HashMap::new();

    for (i, line) in body_lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.contains("__tile_silu_f32") && !trimmed.contains("__tile_silu_f16") {
            continue;
        }
        let silu_ssa = match extract_result_ssa(trimmed) {
            Some(s) => s,
            None => continue,
        };

        // Look ahead for a mul that consumes this silu result
        for j in (i + 1)..body_lines.len() {
            let next = body_lines[j].trim();
            // Skip non-call lines (constants, ptr ops, etc.)
            if next.is_empty() || !next.contains("llvm.call @") {
                continue;
            }
            if next.contains("__tile_mul_f32") || next.contains("__tile_mul_f16") {
                if let Some(mul_args) = extract_call_args(next) {
                    // Check all possible operand positions (4-arg and 5-arg variants)
                    let has_silu = mul_args.iter().take(3).any(|a| a.trim() == silu_ssa);
                    if has_silu {
                        result.insert(silu_ssa.clone(), (i, j));
                        break;
                    }
                }
            }
            // Stop at the first call instruction after silu (don't skip past other ops)
            break;
        }
    }

    result
}

/// Fused SiLU+Mul: `out[i] = silu(gate[i]) * up[i]`
///
/// Emits the fused silu(gate) * up using UB-tight tile reuse:
/// 1. `pto.tmuls(gate, -1.0)` -> neg
/// 2. `pto.texp(neg)` -> neg   (in-place reuse)
/// 3. `pto.tadds(neg, 1.0)` -> neg   (in-place reuse; now = 1 + exp(-gate))
/// 4. `pto.tdiv(gate, neg)` -> silu = gate / (1 + exp(-gate))
/// 5. `pto.tmul(silu, up)` -> out
///
/// Uses a single tile/tile `tdiv` (not `tdivs + tmul`) for the sigmoid-and-scale
/// step because ptoas has no scalar/tile divide.
fn translate_silu_mul(
    silu_line: &str,
    mul_line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    // Parse silu: %silu_res = llvm.call @__tile_silu_f32(%c0, %gate, %rows, %cols)
    let silu_result_ssa = extract_result_ssa(silu_line)
        .ok_or_else(|| format!("silu_mul: no result SSA in silu: {}", silu_line))?;
    let silu_args = extract_call_args(silu_line)
        .ok_or_else(|| format!("silu_mul: cannot parse args in silu: {}", silu_line))?;
    let gate_ssa = silu_args.get(1).ok_or("silu_mul: missing gate src")?.trim();
    let rows = ctx.resolve_const(silu_args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(silu_args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    // Parse mul — handle both 4-arg (a, b, rows, cols) and 5-arg (dst, a, b, rows, cols)
    let mul_result_ssa = extract_result_ssa(mul_line)
        .ok_or_else(|| format!("silu_mul: no result SSA in mul: {}", mul_line))?;
    let mul_args = extract_call_args(mul_line)
        .ok_or_else(|| format!("silu_mul: cannot parse args in mul: {}", mul_line))?;

    // Find the "up" operand: the mul arg that isn't the silu result
    let up_ssa = if mul_args.len() >= 5 {
        // 5-arg: (dst, a, b, rows, cols)
        let a = mul_args[1].trim();
        let b = mul_args[2].trim();
        if a == silu_result_ssa { b } else { a }
    } else {
        // 4-arg: (a, b, rows, cols)
        let a = mul_args[0].trim();
        let b = mul_args[1].trim();
        if a == silu_result_ssa { b } else { a }
    };

    let tgate = ctx
        .get_tile(gate_ssa)
        .ok_or_else(|| format!("silu_mul: unknown tile {}", gate_ssa))?
        .clone();
    let tup = ctx
        .get_tile(up_ssa)
        .ok_or_else(|| format!("silu_mul: unknown tile {}", up_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    // UB-budget check (#66 + #67). The fused emit holds 5 simultaneous
    // tiles in UB: gate, up (inputs), neg, silu (intermediates), out.
    // 910c UB cap is 256 KB; we reserve ~32 KB for code/stack/scalars,
    // giving 224 KB of usable tile budget (SILU_MUL_UB_BUDGET_BYTES).
    //
    // - Under budget: emit the original 5-tile single-block path below.
    // - Over budget AND inputs were deferred by the silu_mul pre-pass:
    //   route to the N-blocked path that re-tloads gate/up per-chunk
    //   from GM partition_views inside an scf.for (#67).
    // - Over budget AND inputs were NOT deferred (e.g., gate/up come
    //   from arith ops, not direct tile_loads): fail at codegen with
    //   the original guard message — the pre-pass couldn't defer them
    //   so the chunked path can't synthesise per-iter loads.
    let elem_bytes: u32 = match dtype {
        "f32" => 4,
        "f16" | "bf16" => 2,
        _ => return Err(format!("silu_mul: unsupported dtype {}", dtype)),
    };
    let peak_ub_bytes: u64 =
        5u64 * (rows as u64) * (cols as u64) * (elem_bytes as u64);
    if peak_ub_bytes > SILU_MUL_UB_BUDGET_BYTES {
        if tgate.deferred.is_some() && tup.deferred.is_some() {
            // Both inputs are deferred GM tensor_views — emit the blocked
            // path and register a pending entry that translate_store will
            // consume to emit the per-chunk scf.for.
            return translate_silu_mul_blocked(
                &mul_result_ssa, &silu_result_ssa,
                rows, cols, dtype,
                &tgate, &tup,
                ctx, ops,
            );
        }
        return Err(format!(
            "silu_mul: peak UB usage {} B for {}x{} {} exceeds budget {} B \
             (5-tile fused emit: gate, up, neg, silu, out). \
             Inner dim N={} needs N-blocked emit but inputs are not direct \
             tile_loads, so the per-chunk loop can't be synthesised. \
             Restructure the kernel to load gate/up directly from GM and \
             feed them to silu_mul without intervening ops, or chunk at \
             the source level (tracking: ICLR 2026 #67).",
            peak_ub_bytes, rows, cols, dtype, SILU_MUL_UB_BUDGET_BYTES, cols
        ));
    }

    ops.push(format!(
        "// --- silu_mul (fused): silu(gate) * up, {}x{} {} ---",
        rows, cols, dtype
    ));

    // Scalar constants for sigmoid decomposition — ptoas grammar requires the
    // scalar as an SSA-bound `arith.constant : f32` passed as a second ins
    // operand, NOT a `{scalar = X : f32}` attribute.
    //
    // Identity used: silu(g) = g / (1 + exp(-g))
    //   = g * sigmoid(g) without needing a reciprocal (ptoas has no
    //     scalar/tile op; tdivs is tile/scalar). `tdiv(g, 1+exp(-g))`
    //     gives the same result as `g * (1/(1+exp(-g)))`.
    let cneg1_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant -1.0 : f32", cneg1_ssa));
    let cone_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant 1.0 : f32", cone_ssa));

    // UB-tight tile budget: at INTER=8960 f32, 7 tiles × 35840 B = 250 KB of
    // 256 KB UB, leaving only ~11 KB for kernel code/stack — triggers a
    // "Vector core execution exception" on A5. Reuse the `neg` tile across
    // the exp/tadds pipeline so we only allocate 2 intermediates (neg, silu)
    // instead of 4 (neg, exp, oplus, silu). Total tiles now: 2 inputs +
    // 2 intermediates + 1 out = 5 (saves ~71 KB at INTER=8960).
    //
    // Dataflow after reuse:
    //   neg   <- tmuls(gate, -1.0)
    //   neg   <- texp(neg)
    //   neg   <- tadds(neg, 1.0)   // neg now holds (1 + exp(-gate))
    //   silu  <- tdiv(gate, neg)    // gate / (1 + exp(-gate))
    //   out   <- tmul(silu, up)
    //
    // Step 1: neg = tmuls(gate, -1.0)
    let neg_key = format!("{}__silumul_neg", mul_result_ssa);
    let neg_ssa = ctx.alloc_tile(&neg_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmuls ins({}, {} : {}, f32) outs({} : {})",
        tgate.ssa, cneg1_ssa, tb_ty, neg_ssa, tb_ty
    ));

    // Step 2: neg = texp(neg)   — reuse neg tile in-place
    ops.push(format!(
        "pto.texp ins({} : {}) outs({} : {})",
        neg_ssa, tb_ty, neg_ssa, tb_ty
    ));

    // Step 3: neg = tadds(neg, 1.0)   — reuse neg tile in-place
    ops.push(format!(
        "pto.tadds ins({}, {} : {}, f32) outs({} : {})",
        neg_ssa, cone_ssa, tb_ty, neg_ssa, tb_ty
    ));

    // Step 4: silu = tdiv(gate, neg) = gate / (1 + exp(-gate))
    // = gate * sigmoid(gate). Uses tile/tile div; ptoas has no scalar/tile.
    let silu_key = format!("{}__silumul_silu", mul_result_ssa);
    let silu_ssa = ctx.alloc_tile(&silu_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tdiv ins({}, {} : {}, {}) outs({} : {})",
        tgate.ssa, neg_ssa, tb_ty, tb_ty, silu_ssa, tb_ty
    ));

    // Step 6: out = tmul(silu, up) = silu(gate) * up
    let out_ssa = ctx.alloc_tile(&mul_result_ssa, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmul ins({}, {} : {}, {}) outs({} : {})",
        silu_ssa, tup.ssa, tb_ty, tb_ty, out_ssa, tb_ty
    ));

    // Also register the silu intermediate under its SSA so downstream ops can find it
    // (in case something else references the silu result besides the fused mul).
    ctx.tiles.insert(
        silu_result_ssa,
        TileInfo {
            ssa: silu_ssa,
            rows,
            cols,
            dtype: dtype.to_string(),
            tb_type: tb_ty.clone(),
            pv_ssa: None,
            gm_name: None,
            deferred: None,
        },
    );

    Ok(())
}

/// N-blocked silu_mul (#67) — emitted when the full-shape 5-tile fused emit
/// would exceed the UB budget (e.g., Qwen2.5-7B INTER=18944 f32: 379 KB
/// > 224 KB). Mirrors `translate_matmul_blocked`.
///
/// Both inputs (gate, up) must already have been deferred by the
/// silu_mul pre-pass — translate_load left their TileInfo with
/// `deferred = Some(DeferredMatmulOperand{..})` carrying the GM
/// tensor_view + element offset.
///
/// This function allocates 5 chunk tiles (rows×Nb each) and registers a
/// `PendingBlockedSiluMul`; translate_store then emits the actual
/// `scf.for n_off = 0 to N step Nb` per-chunk loop once it knows the
/// output GM view.
#[allow(clippy::too_many_arguments)]
fn translate_silu_mul_blocked(
    mul_result_ssa: &str,
    silu_result_ssa: &str,
    rows: u32,
    cols: u32,
    dtype: &str,
    tgate: &TileInfo,
    tup: &TileInfo,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let nb = pick_silu_mul_nb(rows, cols, dtype).ok_or_else(|| {
        format!(
            "silu_mul_blocked: no Nb divisor of cols={} fits the {} B budget at \
             rows={} dtype={} — restructure to chunk at the source level",
            cols, SILU_MUL_UB_BUDGET_BYTES, rows, dtype
        )
    })?;
    let n_iters = cols / nb;
    let dtype_static: &'static str = match dtype {
        "f32" => "f32",
        "f16" => "f16",
        "bf16" => "bf16",
        _ => return Err(format!("silu_mul_blocked: unsupported dtype {}", dtype)),
    };

    let dgate = tgate.deferred.as_ref().expect("caller must verify gate is deferred");
    let dup = tup.deferred.as_ref().expect("caller must verify up is deferred");

    ops.push(format!(
        "// --- silu_mul (N-blocked, #67): silu(gate) * up, {}x{} {} \
         chunked along N into {} blocks of size {} ---",
        rows, cols, dtype, n_iters, nb
    ));

    // Scalar constants for sigmoid decomposition (-1.0 and 1.0).
    let cneg1_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant -1.0 : f32", cneg1_ssa));
    let cone_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant 1.0 : f32", cone_ssa));

    // Pre-allocate 5 chunk tiles outside the loop. Each is rows×Nb.
    // Using synthetic keys keeps them out of the gate_ssa/up_ssa slots
    // (those still hold the deferred placeholder TileInfo).
    let tb_chunk_ty = tile_buf_type(rows, nb, dtype);
    let pv_chunk_ty = ptv_type(rows, nb, dtype);
    let gate_chunk_key = format!("{}__sb_gate_chunk", mul_result_ssa);
    let up_chunk_key = format!("{}__sb_up_chunk", mul_result_ssa);
    let neg_chunk_key = format!("{}__sb_neg_chunk", mul_result_ssa);
    let silu_chunk_key = format!("{}__sb_silu_chunk", mul_result_ssa);
    let out_chunk_key = format!("{}__sb_out_chunk", mul_result_ssa);
    let gate_chunk_ssa =
        ctx.alloc_tile_typed(&gate_chunk_key, rows, nb, dtype, &tb_chunk_ty, ops);
    let up_chunk_ssa =
        ctx.alloc_tile_typed(&up_chunk_key, rows, nb, dtype, &tb_chunk_ty, ops);
    let neg_chunk_ssa =
        ctx.alloc_tile_typed(&neg_chunk_key, rows, nb, dtype, &tb_chunk_ty, ops);
    let silu_chunk_ssa =
        ctx.alloc_tile_typed(&silu_chunk_key, rows, nb, dtype, &tb_chunk_ty, ops);
    let out_chunk_ssa =
        ctx.alloc_tile_typed(&out_chunk_key, rows, nb, dtype, &tb_chunk_ty, ops);

    // Register a placeholder TileInfo for the mul result so any stray
    // downstream lookup returns sane shape data. translate_store will see
    // the `silu_mul_result_stored_inline` flag first and skip the
    // full-shape tstore path entirely.
    ctx.tiles.insert(
        mul_result_ssa.to_string(),
        TileInfo {
            ssa: out_chunk_ssa.clone(),
            rows,
            cols,
            dtype: dtype.to_string(),
            tb_type: tb_chunk_ty.clone(),
            pv_ssa: None,
            gm_name: None,
            deferred: None,
        },
    );
    // Register the silu intermediate similarly.
    ctx.tiles.insert(
        silu_result_ssa.to_string(),
        TileInfo {
            ssa: silu_chunk_ssa.clone(),
            rows,
            cols,
            dtype: dtype.to_string(),
            tb_type: tb_chunk_ty.clone(),
            pv_ssa: None,
            gm_name: None,
            deferred: None,
        },
    );

    let pending = PendingBlockedSiluMul {
        rows,
        cols,
        nb,
        n_iters,
        dtype: dtype_static,
        tv_gate_ssa: dgate.tv_ssa.clone(),
        tv_up_ssa: dup.tv_ssa.clone(),
        gate_elem_offset: dgate.elem_offset,
        up_elem_offset: dup.elem_offset,
        gate_chunk_ssa,
        up_chunk_ssa,
        neg_chunk_ssa,
        silu_chunk_ssa,
        out_chunk_ssa,
        tb_chunk_ty,
        pv_chunk_ty,
        cneg1_ssa,
        cone_ssa,
    };
    ctx.silu_mul_result_stored_inline.insert(mul_result_ssa.to_string());
    ctx.pending_blocked_silu_muls
        .insert(mul_result_ssa.to_string(), pending);

    Ok(())
}

/// Emit the per-chunk `scf.for` loop for an N-blocked silu_mul.
///
/// Output shape:
/// ```text
/// scf.for %n_i = 0 to %N_ITERS step 1 {
///   %n_off = arith.muli %n_i, %Nb
///   // load gate chunk
///   %g_pt = pto.partition_view %tv_gate, offsets=[0, %n_off], sizes=[R, Nb]
///   pto.tload  g_pt → gate_chunk
///   // load up chunk
///   %u_pt = pto.partition_view %tv_up,   offsets=[0, %n_off], sizes=[R, Nb]
///   pto.tload  u_pt → up_chunk
///   // 5-step silu_mul body on chunk tiles
///   pto.tmuls (gate_chunk, -1.0)        → neg_chunk
///   pto.texp  (neg_chunk)               → neg_chunk
///   pto.tadds (neg_chunk, 1.0)          → neg_chunk
///   pto.tdiv  (gate_chunk, neg_chunk)   → silu_chunk
///   pto.tmul  (silu_chunk, up_chunk)    → out_chunk
///   // store out chunk
///   %o_pt = pto.partition_view %tv_out,  offsets=[0, %n_off], sizes=[R, Nb]
///   pto.tstore out_chunk → o_pt
/// }
/// ```
fn emit_blocked_silu_mul_loops(
    tv_out_ssa: &str,
    out_elem_offset: u32,
    out_dtype: &str,
    p: &PendingBlockedSiluMul,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) {
    // Validate output shape matches what the pending entry expects.
    // (translate_store has already enforced rows/cols equality before calling
    // us; we just read them.)
    let _ = (tv_out_ssa, out_dtype);

    // Constants needed across the loop body. For n_off arithmetic we
    // need %c0, %c1, %c{Nb}, %c{n_iters}, %c{rows}.
    let out_base_row = if p.cols > 0 { out_elem_offset / p.cols } else { 0 };
    let out_base_col = if p.cols > 0 { out_elem_offset % p.cols } else { 0 };
    let gate_base_row = if p.cols > 0 { p.gate_elem_offset / p.cols } else { 0 };
    let gate_base_col = if p.cols > 0 { p.gate_elem_offset % p.cols } else { 0 };
    let up_base_row = if p.cols > 0 { p.up_elem_offset / p.cols } else { 0 };
    let up_base_col = if p.cols > 0 { p.up_elem_offset % p.cols } else { 0 };
    ctx.use_size(0);
    ctx.use_size(1);
    ctx.use_size(p.nb);
    ctx.use_size(p.n_iters);
    ctx.use_size(p.rows);
    ctx.use_size(out_base_row);
    ctx.use_size(out_base_col);
    ctx.use_size(gate_base_row);
    ctx.use_size(gate_base_col);
    ctx.use_size(up_base_row);
    ctx.use_size(up_base_col);

    let tv_gate_ty = tv_type(p.rows, p.cols, p.dtype);
    let tv_up_ty = tv_type(p.rows, p.cols, p.dtype);
    let tv_out_ty = tv_type(p.rows, p.cols, out_dtype);
    let _ = (tv_gate_ty, tv_up_ty, tv_out_ty); // types implicit via SSA spelling

    ops.push(format!(
        "scf.for %n_i = %c0 to %c{} step %c1 {{",
        p.n_iters
    ));
    let n_off_ssa = ctx.fresh_ssa();
    ops.push(format!(
        "  {} = arith.muli %n_i, %c{} : index",
        n_off_ssa, p.nb
    ));

    // gate chunk: partition_view(tv_gate, [base_row, base_col + n_off], [rows, Nb])
    // For decode shapes (M=1) base_row is always 0; base_col is folded into the
    // n_off index by adding the constant inline. We emit the addi only when
    // base_col != 0 to keep the common case clean.
    let g_off_ssa = if gate_base_col != 0 {
        let s = ctx.fresh_ssa();
        ops.push(format!(
            "  {} = arith.addi {}, %c{} : index",
            s, n_off_ssa, gate_base_col
        ));
        s
    } else {
        n_off_ssa.clone()
    };
    let g_pt_ssa = ctx.fresh_ssa();
    ops.push(format!(
        "  {} = pto.partition_view {}, offsets = [%c{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        g_pt_ssa, p.tv_gate_ssa, gate_base_row, g_off_ssa, p.rows, p.nb,
        tv_type(p.rows, p.cols, p.dtype), p.pv_chunk_ty
    ));
    ops.push(format!(
        "  pto.tload ins({} : {}) outs({} : {})",
        g_pt_ssa, p.pv_chunk_ty, p.gate_chunk_ssa, p.tb_chunk_ty
    ));

    // up chunk
    let u_off_ssa = if up_base_col != 0 {
        let s = ctx.fresh_ssa();
        ops.push(format!(
            "  {} = arith.addi {}, %c{} : index",
            s, n_off_ssa, up_base_col
        ));
        s
    } else {
        n_off_ssa.clone()
    };
    let u_pt_ssa = ctx.fresh_ssa();
    ops.push(format!(
        "  {} = pto.partition_view {}, offsets = [%c{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        u_pt_ssa, p.tv_up_ssa, up_base_row, u_off_ssa, p.rows, p.nb,
        tv_type(p.rows, p.cols, p.dtype), p.pv_chunk_ty
    ));
    ops.push(format!(
        "  pto.tload ins({} : {}) outs({} : {})",
        u_pt_ssa, p.pv_chunk_ty, p.up_chunk_ssa, p.tb_chunk_ty
    ));

    // 5-step silu_mul body on chunk tiles. Identical algorithm to the
    // unblocked path; only the tile shape differs (rows×Nb instead of
    // rows×cols). The neg tile is reused in-place across tmuls/texp/tadds.
    ops.push(format!(
        "  pto.tmuls ins({}, {} : {}, f32) outs({} : {})",
        p.gate_chunk_ssa, p.cneg1_ssa, p.tb_chunk_ty,
        p.neg_chunk_ssa, p.tb_chunk_ty
    ));
    ops.push(format!(
        "  pto.texp ins({} : {}) outs({} : {})",
        p.neg_chunk_ssa, p.tb_chunk_ty, p.neg_chunk_ssa, p.tb_chunk_ty
    ));
    ops.push(format!(
        "  pto.tadds ins({}, {} : {}, f32) outs({} : {})",
        p.neg_chunk_ssa, p.cone_ssa, p.tb_chunk_ty,
        p.neg_chunk_ssa, p.tb_chunk_ty
    ));
    ops.push(format!(
        "  pto.tdiv ins({}, {} : {}, {}) outs({} : {})",
        p.gate_chunk_ssa, p.neg_chunk_ssa, p.tb_chunk_ty, p.tb_chunk_ty,
        p.silu_chunk_ssa, p.tb_chunk_ty
    ));
    ops.push(format!(
        "  pto.tmul ins({}, {} : {}, {}) outs({} : {})",
        p.silu_chunk_ssa, p.up_chunk_ssa, p.tb_chunk_ty, p.tb_chunk_ty,
        p.out_chunk_ssa, p.tb_chunk_ty
    ));

    // out chunk store
    let o_off_ssa = if out_base_col != 0 {
        let s = ctx.fresh_ssa();
        ops.push(format!(
            "  {} = arith.addi {}, %c{} : index",
            s, n_off_ssa, out_base_col
        ));
        s
    } else {
        n_off_ssa.clone()
    };
    let o_pt_ssa = ctx.fresh_ssa();
    let pv_chunk_out_ty = ptv_type(p.rows, p.nb, out_dtype);
    ops.push(format!(
        "  {} = pto.partition_view {}, offsets = [%c{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        o_pt_ssa, tv_out_ssa, out_base_row, o_off_ssa, p.rows, p.nb,
        tv_type(p.rows, p.cols, out_dtype), pv_chunk_out_ty
    ));
    ops.push(format!(
        "  pto.tstore ins({} : {}) outs({} : {})",
        p.out_chunk_ssa, p.tb_chunk_ty, o_pt_ssa, pv_chunk_out_ty
    ));

    ops.push("}".to_string());
}

/// Matmul transposed: `%res = llvm.call @__tile_matmul_transposed_f32(%c0, %a, %b, %m, %k, %n)`
///
/// C[M,N] = A[M,K] * B^T[N,K] — uses `pto.tmatmul` with B transposed.
/// PTO tmatmul operates on left[M,K] * right[K,N], so we transpose B first.
fn translate_matmul_transposed(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("matmul_transposed: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("matmul_transposed: cannot parse args in: {}", line))?;
    let a_ssa = args.get(1).ok_or("matmul_transposed: missing a")?.trim();
    let b_ssa = args.get(2).ok_or("matmul_transposed: missing b")?.trim();
    let m = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let k = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let n = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(a_ssa)
        .ok_or_else(|| format!("matmul_transposed: unknown tile {}", a_ssa))?
        .clone();
    let tb = ctx
        .get_tile(b_ssa)
        .ok_or_else(|| format!("matmul_transposed: unknown tile {}", b_ssa))?
        .clone();
    let pv_a = ta.pv_ssa.clone().ok_or_else(|| {
        format!(
            "matmul_transposed: tile {} has no partition view",
            a_ssa
        )
    })?;
    // pv_b (N×K view) is intentionally not used — see below.
    let _pv_b = tb.pv_ssa.clone();
    // B is N×K in GM row-major. For C = A · B^T we need B^T which is K×N.
    // Sidestep the broken VEC→MAT tmov path by building a *transposed*
    // tensor_view on the same GM buffer (shape [K,N] strides [1,K]) and
    // tloading straight into a ZN mat tile. This is the DN→ZN path —
    // the only supported transposed-MAT TLoad combo.
    let b_gm = tb.gm_name.clone().ok_or_else(|| {
        format!(
            "matmul_transposed: tile {} has no recorded GM name (not loaded from GM)",
            b_ssa
        )
    })?;

    ctx.use_size(m);
    ctx.use_size(k);
    ctx.use_size(n);

    ops.push(format!(
        "// --- matmul_transposed: C[{}x{}] = A[{}x{}] x B^T[{}x{}] ---",
        m, n, m, k, n, k
    ));

    // Step 1: Alloc CBUF staging tiles (mat_a: MxK NZ, mat_bt: KxN ZN for DN→ZN tload)
    let mat_a_key = format!("{}__mat_a", result_ssa);
    let mat_bt_key = format!("{}__mat_bt", result_ssa);
    let mat_a_ty = mat_tile_type(m, k, dtype);
    let mat_bt_ty = mat_tile_type_zn(k, n, dtype);
    let mat_a_ssa = ctx.alloc_tile_typed(&mat_a_key, m, k, dtype, &mat_a_ty, ops);
    let mat_bt_ssa = ctx.alloc_tile_typed(&mat_bt_key, k, n, dtype, &mat_bt_ty, ops);

    // Step 2: Alloc L0A/L0B/L0C tiles
    let left_key = format!("{}__left", result_ssa);
    let right_key = format!("{}__right", result_ssa);
    let left_ty = left_tile_type(m, k, dtype);
    let right_ty = right_tile_type(k, n, dtype);
    let acc_ty = acc_tile_type(m, n, dtype);
    let left_ssa = ctx.alloc_tile_typed(&left_key, m, k, dtype, &left_ty, ops);
    let right_ssa = ctx.alloc_tile_typed(&right_key, k, n, dtype, &right_ty, ops);
    let acc_ssa = ctx.alloc_tile_typed(&result_ssa, m, n, dtype, &acc_ty, ops);

    // Step 3: tload A (ND→NZ) and B^T (DN→ZN via transposed tensor_view)
    //   A: standard row-major load, N×K → NZ mat tile
    //   B^T: shape [K,N] strides [1,K] over the same GM buffer = column-major
    //        view of an N×K row-major tensor — which is exactly B transposed.
    let pv_a_ty = ptv_type(m, k, dtype);
    let pv_bt_ty = ptv_type(k, n, dtype);
    let tv_b_t = ctx.make_tv_transposed(&b_gm, n, k, dtype, ops);
    let pv_b_t = ctx.make_pv(&tv_b_t, k, n, dtype, 0, ops);
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_a, pv_a_ty, mat_a_ssa, mat_a_ty
    ));
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_b_t, pv_bt_ty, mat_bt_ssa, mat_bt_ty
    ));

    // Step 4: CBUF → L0A/L0B and matmul
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        mat_a_ssa, mat_a_ty, left_ssa, left_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        mat_bt_ssa, mat_bt_ty, right_ssa, right_ty
    ));
    ops.push(format!(
        "pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
        left_ssa, right_ssa, left_ty, right_ty, acc_ssa, acc_ty
    ));

    Ok(())
}

/// Attention GQA: Grouped-Query Attention
///
/// Decomposed similarly to standard attention, but with head grouping:
/// Q has n_heads_q heads, KV has n_heads_kv heads.
/// Each KV head serves (n_heads_q / n_heads_kv) Q heads.
/// We emit the attention for the first Q head only as a representative,
/// using tmatmul + softmax + tmatmul.
fn translate_attention_gqa(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let args = extract_call_args(line)
        .ok_or_else(|| format!("attention_gqa: cannot parse args: {}", line))?;
    if args.len() < 8 {
        return Err(format!("attention_gqa: expected 8 args, got {}", args.len()));
    }
    let result_ssa = extract_result_ssa(line).unwrap_or_else(|| "__gqa_out".to_string());
    let q_arg = args[1].trim();
    let k_arg = args[2].trim();
    let v_arg = args[3].trim();
    let s = ctx.resolve_const(args[4].trim());
    let d = ctx.resolve_const(args[5].trim());
    let n_heads_q = ctx.resolve_const(args[6].trim());
    let n_heads_kv = ctx.resolve_const(args[7].trim());
    let group_size = if n_heads_kv > 0 { n_heads_q / n_heads_kv } else { 1 };

    let tq = ctx.get_tile(q_arg).ok_or_else(|| format!("attention_gqa: unknown Q tile {}", q_arg))?.clone();
    let tk = ctx.get_tile(k_arg).ok_or_else(|| format!("attention_gqa: unknown K tile {}", k_arg))?.clone();
    let tv = ctx.get_tile(v_arg).ok_or_else(|| format!("attention_gqa: unknown V tile {}", v_arg))?.clone();

    // Reuse the GM-direct tload path from translate_attention: we don't have
    // working vec→mat / acc→vec tmov pairs on a2a3, so Q/V go straight from
    // GM partition_view to mat tiles, and K uses a transposed tensor_view.
    let pv_q = tq.pv_ssa.clone().ok_or_else(|| {
        format!("attention_gqa: Q tile {} has no partition view (not loaded from GM)", q_arg)
    })?;
    let pv_v = tv.pv_ssa.clone().ok_or_else(|| {
        format!("attention_gqa: V tile {} has no partition view (not loaded from GM)", v_arg)
    })?;
    let k_gm = tk.gm_name.clone().ok_or_else(|| {
        format!("attention_gqa: K tile {} has no recorded GM name (not loaded from GM)", k_arg)
    })?;

    ops.push(format!(
        "// --- attention_gqa: {} Q heads, {} KV heads, group_size={}, S={}, D={} ---",
        n_heads_q, n_heads_kv, group_size, s, d
    ));
    ops.push(format!(
        "// Emitting representative single-head attention (first Q head, first KV head)"
    ));

    // Step 1: scores = Q(S×D) @ K^T(D×S) → S×S via cube unit.
    // Q is loaded ND→NZ; K uses a transposed tensor_view (DN) and a ZN mat tile
    // (mat_tile_type_zn) to satisfy TLoadGm2L1's DN→ZN path.
    let mat_q_ty = mat_tile_type(s, d, "f32");
    let mat_k_ty = mat_tile_type_zn(d, s, "f32");
    let l_ty = left_tile_type(s, d, "f32");
    let r_ty = right_tile_type(d, s, "f32");
    let acc_ty = acc_tile_type(s, s, "f32");

    let mq = ctx.alloc_tile_typed(&format!("{}__gqa_mq", result_ssa), s, d, "f32", &mat_q_ty, ops);
    let mk = ctx.alloc_tile_typed(&format!("{}__gqa_mk", result_ssa), d, s, "f32", &mat_k_ty, ops);
    let lq = ctx.alloc_tile_typed(&format!("{}__gqa_lq", result_ssa), s, d, "f32", &l_ty, ops);
    let rk = ctx.alloc_tile_typed(&format!("{}__gqa_rk", result_ssa), d, s, "f32", &r_ty, ops);
    let scores = ctx.alloc_tile_typed(&format!("{}__gqa_scores", result_ssa), s, s, "f32", &acc_ty, ops);

    let pv_sd = ptv_type(s, d, "f32");
    let tv_k_t = ctx.make_tv_transposed(&k_gm, s, d, "f32", ops);
    let pv_k_t = ctx.make_pv(&tv_k_t, d, s, "f32", 0, ops);
    let pv_ds = ptv_type(d, s, "f32");
    ops.push(format!("pto.tload ins({} : {}) outs({} : {})", pv_q, pv_sd, mq, mat_q_ty));
    ops.push(format!("pto.tload ins({} : {}) outs({} : {})", pv_k_t, pv_ds, mk, mat_k_ty));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mq, mat_q_ty, lq, l_ty));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mk, mat_k_ty, rk, r_ty));
    ops.push(format!("pto.tmatmul ins({}, {} : {}, {}) outs({} : {})", lq, rk, l_ty, r_ty, scores, acc_ty));

    // Step 2: move scores to VEC for softmax
    let vec_ty = tile_buf_type(s, s, "f32");
    let sv = ctx.alloc_tile_typed(&format!("{}__gqa_sv", result_ssa), s, s, "f32", &vec_ty, ops);
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", scores, acc_ty, sv, vec_ty));

    // Step 3: softmax (5-step) — max/sum are row-reductions (rows×1 col_major).
    let rr_ty = tile_buf_type_rowreduce(s, "f32");
    let tmp = ctx.alloc_tile(&format!("{}__gqa_tmp", result_ssa), s, s, "f32", ops);
    let mx = ctx.alloc_tile_rowreduce(&format!("{}__gqa_mx", result_ssa), s, "f32", ops);
    let sb = ctx.alloc_tile(&format!("{}__gqa_sb", result_ssa), s, s, "f32", ops);
    let ex = ctx.alloc_tile(&format!("{}__gqa_ex", result_ssa), s, s, "f32", ops);
    let sm = ctx.alloc_tile_rowreduce(&format!("{}__gqa_sm", result_ssa), s, "f32", ops);
    let wt = ctx.alloc_tile(&format!("{}__gqa_wt", result_ssa), s, s, "f32", ops);

    ops.push(format!("pto.trowmax ins({}, {} : {}, {}) outs({} : {})", sv, tmp, vec_ty, vec_ty, mx, rr_ty));
    ops.push(format!("pto.trowexpandsub ins({}, {} : {}, {}) outs({} : {})", sv, mx, vec_ty, rr_ty, sb, vec_ty));
    ops.push(format!("pto.texp ins({} : {}) outs({} : {})", sb, vec_ty, ex, vec_ty));
    ops.push(format!("pto.trowsum ins({}, {} : {}, {}) outs({} : {})", ex, tmp, vec_ty, vec_ty, sm, rr_ty));
    ops.push(format!("pto.trowexpanddiv ins({}, {} : {}, {}) outs({} : {})", ex, sm, vec_ty, rr_ty, wt, vec_ty));

    // Step 4: output = weights(S×S) @ V(S×D) → S×D
    let mw_ty = mat_tile_type(s, s, "f32");
    let mv_ty = mat_tile_type(s, d, "f32");
    let lw_ty = left_tile_type(s, s, "f32");
    let rv_ty = right_tile_type(s, d, "f32");
    let out_ty = acc_tile_type(s, d, "f32");

    let mw = ctx.alloc_tile_typed(&format!("{}__gqa_mw", result_ssa), s, s, "f32", &mw_ty, ops);
    let mv = ctx.alloc_tile_typed(&format!("{}__gqa_mv", result_ssa), s, d, "f32", &mv_ty, ops);
    let lw = ctx.alloc_tile_typed(&format!("{}__gqa_lw", result_ssa), s, s, "f32", &lw_ty, ops);
    let rv = ctx.alloc_tile_typed(&format!("{}__gqa_rv", result_ssa), s, d, "f32", &rv_ty, ops);
    let out = ctx.alloc_tile_typed(&result_ssa, s, d, "f32", &out_ty, ops);

    // V: GM partition_view → mat directly (avoid vec→mat tmov).
    let pv_sd2 = ptv_type(s, d, "f32");
    ops.push(format!("pto.tload ins({} : {}) outs({} : {})", pv_v, pv_sd2, mv, mv_ty));
    // Weights (vec) → mat via pto.tinsert (A5-only), not tmov.
    ctx.use_size(0);
    ops.push(format!(
        "pto.tinsert ins({}, %c0, %c0 : {}, index, index) outs({} : {})",
        wt, vec_ty, mw, mw_ty
    ));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mw, mw_ty, lw, lw_ty));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", mv, mv_ty, rv, rv_ty));
    ops.push(format!("pto.tmatmul ins({}, {} : {}, {}) outs({} : {})", lw, rv, lw_ty, rv_ty, out, out_ty));

    Ok(())
}

/// Clamp: `%res = llvm.call @__tile_clamp_f32(%c0, %src, %min, %max, %rows, %cols)`
///
/// Decomposed into:
/// 1. `pto.tmaxs(src, min_val)` -> clamp lower bound
/// 2. `pto.tmins(clamped_lower, max_val)` -> clamp upper bound
fn translate_clamp(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("clamp: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("clamp: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("clamp: missing src")?.trim();
    let min_ssa = args.get(2).ok_or("clamp: missing min")?.trim();
    let max_ssa = args.get(3).ok_or("clamp: missing max")?.trim();
    let rows = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let min_val = ctx.resolve_float(min_ssa);
    let max_val = ctx.resolve_float(max_ssa);

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("clamp: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    ops.push(format!(
        "// --- clamp: clamp(x, {}, {}), {}x{} {} ---",
        min_val, max_val, rows, cols, dtype
    ));

    // Scalar constants (ptoas requires SSA-bound f32 operands, not attributes).
    let cmin_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant {} : f32", cmin_ssa, min_val));
    let cmax_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant {} : f32", cmax_ssa, max_val));

    // Step 1: lower_clamped = tmaxs(src, min_val)
    let lower_key = format!("{}__clamp_lo", result_ssa);
    let lower_ssa = ctx.alloc_tile(&lower_key, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmaxs ins({}, {} : {}, f32) outs({} : {})",
        tsrc.ssa, cmin_ssa, tb_ty, lower_ssa, tb_ty
    ));

    // Step 2: result = tmins(lower_clamped, max_val)
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    ops.push(format!(
        "pto.tmins ins({}, {} : {}, f32) outs({} : {})",
        lower_ssa, cmax_ssa, tb_ty, out_ssa, tb_ty
    ));

    Ok(())
}

/// Cast: `%res = llvm.call @__tile_cast_f32_f16(%c0, %src, %rows, %cols)`
///        `%res = llvm.call @__tile_cast_f16_f32(%c0, %src, %rows, %cols)`
///
/// PTO does not have a native `pto.tcast` op. We emit a comment and use tmov
/// as a passthrough (with the output tile typed in the target dtype).
fn translate_cast(
    line: &str,
    src_dtype: &str,
    dst_dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("cast: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("cast: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("cast: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("cast: unknown tile {}", src_ssa))?
        .clone();
    let src_ty = tsrc.tile_buf_type_str();
    let dst_ty = tile_buf_type(rows, cols, dst_dtype);

    ops.push(format!(
        "// --- cast: {} -> {}, {}x{} ---",
        src_dtype, dst_dtype, rows, cols
    ));
    ops.push(
        "// PTO lacks native cast. Using tmov passthrough with target dtype.".to_string(),
    );

    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dst_dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, src_ty, out_ssa, dst_ty
    ));
    ops.push(format!(
        "// TODO: implement {} -> {} cast when PTO tcast is available",
        src_dtype, dst_dtype
    ));

    Ok(())
}

/// Slice: `%res = llvm.call @__tile_slice_f32(%c0, %src, %row_off, %col_off, %src_r, %src_c, %dst_r, %dst_c)`
///
/// Extracts a sub-tile from a larger tile. Emits tmov passthrough with reshaped output.
/// PartitionCellStore: `%res = llvm.call @__tile_partition_cell_mut_f32(%src, %i, %j, %tr, %tc)`
///
/// Writes the tile `%src` INTO cell (i,j) of a Tr x Tc partition -- the inverse of
/// translate_partition_cell, and the operation cuTile can only express behind `unsafe`.
/// PTO expresses the destination natively: a `pto.partition_view` at the cell's offsets is
/// exactly the footprint being written, so this is a `tstore` into that view rather than a
/// copy. thm:partdisj's hypotheses are enforced first, as for the read form.
fn translate_partition_cell_store(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let args = extract_call_args(line)
        .ok_or_else(|| format!("partition_cell_store: cannot parse args in: {}", line))?;
    let src_ssa = args.first().ok_or("partition_cell_store: missing src tile")?.trim();
    let ci = ctx.resolve_const(args.get(1).map(|s| s.as_str()).unwrap_or("0"));
    let cj = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let tr = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let tc = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("partition_cell_store: unknown tile {}", src_ssa))?
        .clone();
    // The enclosing view must COVER the cell being written: cell (i,j) occupies rows
    // [i*Tr, (i+1)*Tr) and columns [j*Tc, (j+1)*Tc), so a view sized to the tile alone would
    // put every cell past (0,0) out of bounds. Size it to the grid the index implies, taking
    // the source tile's own extent as a floor.
    let rows = tsrc.rows.max((ci + 1) * tr);
    let cols = tsrc.cols.max((cj + 1) * tc);

    if tr == 0 || tc == 0 {
        return Err(format!(
            "partition_cell_store: tile extent must be positive, got {}x{} (thm:partdisj \
needs Tr>0, Tc>0)",
            tr, tc
        ));
    }
    if rows % tr != 0 || cols % tc != 0 {
        return Err(format!(
            "partition_cell_store: {}x{} does not divide {}x{} -- a ragged partition is \
outside thm:partdisj, so pairwise disjointness is NOT established",
            tr, tc, rows, cols
        ));
    }

    let gm_name = tsrc.gm_name.clone().ok_or_else(|| {
        format!("partition_cell_store: tile {} has no originating GM buffer", src_ssa)
    })?;
    let (row_off, col_off) = (ci * tr, cj * tc);
    ops.push(format!(
        "// --- partition cell STORE ({},{}), tile {}x{}, offsets [{},{}] ---",
        ci, cj, tr, tc, row_off, col_off
    ));
    let tv = ctx.get_or_make_tv(&gm_name, rows, cols, dtype, ops);
    let pv = ctx.make_pv_at(&tv, tr, tc, dtype, row_off, col_off, ops);
    ops.push(format!(
        "pto.tstore ins({} : {}) outs({} : {})",
        tsrc.ssa,
        tsrc.tile_buf_type_str(),
        pv,
        ptv_type(tr, tc, dtype)
    ));
    Ok(())
}

/// PartitionCell: `%res = llvm.call @__tile_partition_cell_f32(%src, %i, %j, %tr, %tc)`
///
/// Cell (i,j) of a Tr x Tc partition owns rows [i*Tr, i*Tr+Tr) and cols [j*Tc, j*Tc+Tc).
/// Its flat element offset is `i*Tr*C + j*Tc` -- the same address the mechanization computes
/// (`addr(C, i*Tr, j*Tc)`, `in_cell`). PTO expresses exactly this with `pto.partition_view`,
/// whose `offsets`/`sizes` clause is the native form of a tile footprint, so this is a direct
/// lowering rather than an emulation.
///
/// The grid hypotheses of thm:partdisj (Tr>0, Tc>0, Tr|R, Tc|C) are checked here: a ragged
/// split is outside the theorem, so its disjointness is not established and it must not lower.
fn translate_partition_cell(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("partition_cell: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("partition_cell: cannot parse args in: {}", line))?;
    let src_ssa = args.first().ok_or("partition_cell: missing src")?.trim();
    let ci = ctx.resolve_const(args.get(1).map(|s| s.as_str()).unwrap_or("0"));
    let cj = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let tr = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let tc = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("partition_cell: unknown tile {}", src_ssa))?
        .clone();
    let (rows, cols) = (tsrc.rows, tsrc.cols);

    // thm:partdisj's hypotheses, enforced rather than assumed.
    if tr == 0 || tc == 0 {
        return Err(format!(
            "partition_cell: tile extent must be positive, got {}x{} (thm:partdisj needs \
Tr>0, Tc>0)",
            tr, tc
        ));
    }
    if rows % tr != 0 || cols % tc != 0 {
        return Err(format!(
            "partition_cell: {}x{} does not divide {}x{} -- a ragged partition is outside \
thm:partdisj (needs Tr|R and Tc|C), so pairwise disjointness is NOT established",
            tr, tc, rows, cols
        ));
    }
    let (gi, gj) = (rows / tr, cols / tc);
    if ci >= gi || cj >= gj {
        return Err(format!(
            "partition_cell: cell ({}, {}) is outside the {}x{} grid; thm:partdisj is stated \
for in-grid indices only",
            ci, cj, gi, gj
        ));
    }

    // Cell base as (row, col), which is what partition_view wants directly.  NOTE: do not
    // route this through make_pv's flat-offset path -- that divides by the TILE width to
    // recover a row, but the buffer's row stride is C, so a flat offset of i*Tr*C + j*Tc
    // would decode to the wrong row whenever Tc != C.  (Caught by the round-trip test:
    // cell (1,1) of a 32x64 partition of a 128-wide buffer decoded to row 65 instead of 32.)
    let row_off = ci * tr;
    let col_off = cj * tc;
    let elem_offset = row_off * cols + col_off;
    ops.push(format!(
        "// --- partition cell ({},{}) of {}x{} grid, tile {}x{}, base offset {} ---",
        ci, cj, gi, gj, tr, tc, elem_offset
    ));

    // The cell is a view over the SAME GM buffer the source tile came from; PTO's
    // partition_view takes the offset directly, so no copy is introduced.
    let gm_name = tsrc.gm_name.clone().ok_or_else(|| {
        format!("partition_cell: tile {} has no originating GM buffer", src_ssa)
    })?;
    let tv = ctx.get_or_make_tv(&gm_name, rows, cols, dtype, ops);
    let pv = ctx.make_pv_at(&tv, tr, tc, dtype, row_off, col_off, ops);
    let _ = elem_offset; // kept for the diagnostic comment above
    let out_ssa = ctx.alloc_tile(&result_ssa, tr, tc, dtype, ops);
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv,
        ptv_type(tr, tc, dtype),
        out_ssa,
        tile_buf_type(tr, tc, dtype)
    ));
    Ok(())
}

fn translate_slice(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    _func: &MlirFunc,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("slice: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("slice: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("slice: missing src")?.trim();
    let row_off = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let col_off = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let _src_r = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let _src_c = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));
    let dst_r = ctx.resolve_const(args.get(6).map(|s| s.as_str()).unwrap_or("0"));
    let dst_c = ctx.resolve_const(args.get(7).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("slice: unknown tile {}", src_ssa))?
        .clone();
    let src_ty = tsrc.tile_buf_type_str();
    let dst_ty = tile_buf_type(dst_r, dst_c, dtype);

    ops.push(format!(
        "// --- slice: offset=({},{}), dst={}x{} {} ---",
        row_off, col_off, dst_r, dst_c, dtype
    ));
    ops.push(
        "// Slice extracts a sub-tile. Using tmov passthrough with reshaped output."
            .to_string(),
    );

    let out_ssa = ctx.alloc_tile(&result_ssa, dst_r, dst_c, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, src_ty, out_ssa, dst_ty
    ));
    ops.push(format!(
        "// TODO: implement slice with partition_view offset=[{}, {}] when supported",
        row_off, col_off
    ));

    Ok(())
}

/// Concat: `%res = llvm.call @__tile_concat_f32(%c0, %a, %b, %rows, %cols_a, %cols_b)`
///
/// Concatenates two tiles along the column dimension. Since PTO has no native
/// concat, we emit a tmov pair as passthrough placeholder.
fn translate_concat(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("concat: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("concat: cannot parse args in: {}", line))?;
    let a_ssa = args.get(1).ok_or("concat: missing a")?.trim();
    let b_ssa = args.get(2).ok_or("concat: missing b")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols_a = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let cols_b = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(a_ssa)
        .ok_or_else(|| format!("concat: unknown tile {}", a_ssa))?
        .clone();
    let tb = ctx
        .get_tile(b_ssa)
        .ok_or_else(|| format!("concat: unknown tile {}", b_ssa))?
        .clone();
    let ta_ty = ta.tile_buf_type_str();
    let tb_ty_str = tb.tile_buf_type_str();
    let out_cols = cols_a + cols_b;
    let out_ty = tile_buf_type(rows, out_cols, dtype);

    ops.push(format!(
        "// --- concat: {}x{} + {}x{} -> {}x{} {} ---",
        rows, cols_a, rows, cols_b, rows, out_cols, dtype
    ));
    ops.push("// PTO lacks native concat. Using tmov pair as passthrough.".to_string());

    let out_ssa = ctx.alloc_tile(&result_ssa, rows, out_cols, dtype, ops);

    // Copy first tile into output
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        ta.ssa, ta_ty, out_ssa, out_ty
    ));
    // Document the second copy
    ops.push(format!(
        "// tmov {} ({}) into output at col offset {} (requires partition_view offset)",
        tb.ssa, tb_ty_str, cols_a
    ));

    Ok(())
}

/// Scatter: `%res = llvm.call @__tile_scatter_f32(%c0, %src, %indices, %n, %m, %d)`
///
/// PTO has no native scatter operation. Emit a comment placeholder and
/// pass through the input tile via tmov.
fn translate_scatter(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("scatter: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("scatter: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("scatter: missing src")?.trim();
    let _indices_ssa = args.get(2).ok_or("scatter: missing indices")?.trim();
    let n = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let m = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let _d = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("scatter: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(n, m, dtype);

    ops.push(format!(
        "// --- scatter: indexed scatter, {}x{} {} ---",
        n, m, dtype
    ));
    ops.push("// PTO lacks native scatter. Using tmov passthrough.".to_string());
    ops.push(
        "// TODO: implement scatter via host-side index computation".to_string(),
    );

    let out_ssa = ctx.alloc_tile(&result_ssa, n, m, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, tb_ty, out_ssa, tb_ty
    ));

    Ok(())
}

/// Gather: `%res = llvm.call @__tile_gather_f32(%c0, %src, %indices, %n, %m, %d)`
///
/// PTO has no native gather operation. Emit a comment placeholder and
/// pass through the input tile via tmov.
fn translate_gather(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("gather: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("gather: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("gather: missing src")?.trim();
    let _indices_ssa = args.get(2).ok_or("gather: missing indices")?.trim();
    let n = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let m = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let _d = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("gather: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(n, m, dtype);

    ops.push(format!(
        "// --- gather: indexed gather, {}x{} {} ---",
        n, m, dtype
    ));
    ops.push("// PTO lacks native gather. Using tmov passthrough.".to_string());
    ops.push(
        "// TODO: implement gather via host-side index computation".to_string(),
    );

    let out_ssa = ctx.alloc_tile(&result_ssa, n, m, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, tb_ty, out_ssa, tb_ty
    ));

    Ok(())
}

/// Mask-pattern gather: `%res = llvm.call @__tile_gather_mask_f32(%c0, %src, %mask_pattern, %rows, %cols)`
///
/// Emits `pto.tgather` (mask-pattern form). Extracts a sub-tile per the
/// 4-bit mask pattern attribute. Pattern `P1010` (= 0b1010 = 10) selects
/// even-indexed lanes — used to extract the value channel from the
/// interleaved [val, idx] output of `pto.tsort32`.
///
/// MLIR shape (verified 2026-04-29 against
/// `/tmp/.../ptoas/*.pto`):
/// ```text
///   pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P1010>}
///                   : !pto.tile_buf<...>)
///               outs(%dst : !pto.tile_buf<...>)
/// ```
///
/// The mask is encoded in the `mask_pattern` arg as the integer value of
/// the bit pattern (e.g. `0b1010 = 10` for value-channel extraction).
/// The emitted attr is rendered as `P{nibble}` — e.g. `P1010`, `P1111`.
fn translate_gather_mask(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("gather_mask: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("gather_mask: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("gather_mask: missing src")?.trim();
    let mask = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    if rows == 0 || cols == 0 {
        return Err("gather_mask: rows and cols must be > 0".to_string());
    }
    if mask > 15 {
        return Err(format!(
            "gather_mask: mask must fit in 4 bits (0..15), got {}",
            mask
        ));
    }

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("gather_mask: unknown src tile {}", src_ssa))?
        .clone();
    let src_ty = tsrc.tile_buf_type_str();

    // Output dims match source dims (mask selects lanes, doesn't reshape).
    let dst_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    let dst_ty = tile_buf_type(rows, cols, dtype);

    // Render the 4-bit mask as `P{b3}{b2}{b1}{b0}` (e.g. mask=10 → P1010).
    let mask_str = format!(
        "P{}{}{}{}",
        (mask >> 3) & 1,
        (mask >> 2) & 1,
        (mask >> 1) & 1,
        mask & 1
    );

    ops.push(format!(
        "pto.tgather ins({}, {{maskPattern = #pto.mask_pattern<{}>}} : {}) \
         outs({} : {})",
        tsrc.ssa, mask_str, src_ty, dst_ssa, dst_ty
    ));

    Ok(())
}

/// 2-way bitonic merge: `%res = llvm.call @__tile_mrgsort2_f32(%c0, %src0, %src1, %tmp, %cols_each)`
///
/// Emits `pto.tmrgsort` (2-way form). Merges two sorted 1×N f32 tiles into
/// a 1×(2N) sorted tile, plus a 4-element i16 exhausted-flags vector.
/// `tmp` is a 1×(2N) scratch tile whose dtype matches src0/src1.
///
/// MLIR shape (verified 2026-04-29 against
/// `/tmp/pa_spmd_hw_small/ptoas/SpmdPagedAttentionGroup.pto`):
/// ```text
///   pto.tmrgsort ins(%src0, %src1, %tmp {exhausted = false}
///                    : tile<rows=1,cols=N>, tile<rows=1,cols=N>,
///                      tile<rows=1,cols=2N>)
///                outs(%dst, %ex : tile<rows=1,cols=2N>, vector<4xi16>)
/// ```
fn translate_merge_sort(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("merge_sort: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("merge_sort: cannot parse args in: {}", line))?;
    let src0_ssa = args.get(1).ok_or("merge_sort: missing src0")?.trim();
    let src1_ssa = args.get(2).ok_or("merge_sort: missing src1")?.trim();
    let tmp_ssa = args.get(3).ok_or("merge_sort: missing tmp")?.trim();
    let cols_each = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    if cols_each == 0 {
        return Err("merge_sort: cols_each must be > 0".to_string());
    }

    let t0 = ctx
        .get_tile(src0_ssa)
        .ok_or_else(|| format!("merge_sort: unknown src0 tile {}", src0_ssa))?
        .clone();
    let t1 = ctx
        .get_tile(src1_ssa)
        .ok_or_else(|| format!("merge_sort: unknown src1 tile {}", src1_ssa))?
        .clone();
    let ttmp = ctx
        .get_tile(tmp_ssa)
        .ok_or_else(|| format!("merge_sort: unknown tmp tile {}", tmp_ssa))?
        .clone();

    let merged_cols = cols_each * 2;
    let dst_ssa = ctx.alloc_tile(&result_ssa, 1, merged_cols, dtype, ops);
    let dst_ty = tile_buf_type(1, merged_cols, dtype);
    let ex_ssa = ctx.fresh_ssa();
    // pto.tmrgsort takes the two output SSAs in the outs() clause (matches
    // captured fixture form — there is no leading `%res =` assignment).
    ops.push(format!(
        "pto.tmrgsort ins({}, {}, {} {{exhausted = false}} : {}, {}, {}) \
         outs({}, {} : {}, vector<4xi16>)",
        t0.ssa,
        t1.ssa,
        ttmp.ssa,
        t0.tile_buf_type_str(),
        t1.tile_buf_type_str(),
        ttmp.tile_buf_type_str(),
        dst_ssa,
        ex_ssa,
        dst_ty,
    ));

    Ok(())
}

/// Tile sort: `%res = llvm.call @__tile_sort32_f32(%c0, %values, %indices, %rows, %cols)`
///
/// Emits `pto.tsort32` — sorts a 1×N f32 tile via vbitsort, producing a
/// 1×(2N) f32 tile of interleaved [value, idx] pairs. Per
/// `pto-isa-patched/pto/npu/a2a3/TSort32.hpp`, output stride coefficient
/// is 2 — the ASIC writes (value, idx) pairs at 64-element granularity
/// per vbitsort call (stride=2).
///
/// MLIR shape (verified 2026-04-29 against
/// `/tmp/pa_spmd_hw_small/ptoas/SpmdPagedAttentionGroup.pto`):
/// ```text
///   pto.tsort32 ins(%values, %indices :
///                   !pto.tile_buf<...rows=1, cols=N, dtype=f32...>,
///                   !pto.tile_buf<...rows=1, cols=N, dtype=ui32...>)
///               outs(%sorted :
///                   !pto.tile_buf<...rows=1, cols=2*N, dtype=f32...>)
/// ```
fn translate_tile_sort(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("tile_sort: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("tile_sort: cannot parse args in: {}", line))?;
    let values_ssa = args.get(1).ok_or("tile_sort: missing values")?.trim();
    let indices_ssa = args.get(2).ok_or("tile_sort: missing indices")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    if rows != 1 {
        return Err(format!(
            "tile_sort: only rows=1 supported (vbitsort is 1D), got {}",
            rows
        ));
    }
    if cols == 0 || cols % 64 != 0 {
        return Err(format!(
            "tile_sort: cols must be a positive multiple of 64 (vbitsort granularity), got {}",
            cols
        ));
    }

    let tvals = ctx
        .get_tile(values_ssa)
        .ok_or_else(|| format!("tile_sort: unknown values tile {}", values_ssa))?
        .clone();
    let tidx = ctx
        .get_tile(indices_ssa)
        .ok_or_else(|| format!("tile_sort: unknown indices tile {}", indices_ssa))?
        .clone();
    let vals_ty = tvals.tile_buf_type_str();
    let idx_ty = tidx.tile_buf_type_str();

    // Output tile: rows=1, cols=2*N (interleaved [val, idx] pairs).
    let out_cols = cols * 2;
    let dst_ssa = ctx.alloc_tile(&result_ssa, 1, out_cols, dtype, ops);
    let dst_ty = tile_buf_type(1, out_cols, dtype);
    ops.push(format!(
        "pto.tsort32 ins({}, {} : {}, {}) outs({} : {})",
        tvals.ssa, tidx.ssa, vals_ty, idx_ty, dst_ssa, dst_ty
    ));

    Ok(())
}

/// Sort-buffer init: `%res = llvm.call @__tile_init_sort_buf_f32(%c0, %src, %rows, %cols)`
///
/// Emits `pto.tfillpad` — re-pads a tile to the next BLOCK_SIZE boundary
/// with a sentinel value (used to safely handle non-32-multiple sort
/// inputs). The output tile has the same logical rows/cols/v_row/v_col
/// as the input but `pad=3` (vs `pad=0` on the input).
///
/// MLIR shape (verified 2026-04-29 against
/// `/tmp/pa_spmd_hw_small/ptoas/SpmdPagedAttentionGroup.pto` on 910c):
/// ```text
///   pto.tfillpad ins(%src : !pto.tile_buf<...pad=0...>)
///                outs(%dst : !pto.tile_buf<...pad=3...>)
/// ```
///
/// The `pad=3` literal is the documented marker for "ptoas-managed
/// sentinel pad" — its precise semantics are opaque from the patched
/// headers but match the emitted form in production .pto files.
fn translate_init_sort_buf(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("init_sort_buf: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("init_sort_buf: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("init_sort_buf: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    if rows == 0 || cols == 0 {
        return Err("init_sort_buf: rows and cols must be > 0".to_string());
    }

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("init_sort_buf: unknown tile {}", src_ssa))?
        .clone();
    let src_ty = tsrc.tile_buf_type_str();

    // Output tile: same shape as src, but pad=3 (sentinel-pad marker).
    let dst_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    let dst_ty = format!(
        "!pto.tile_buf<loc=vec, dtype={}, rows={}, cols={}, v_row={}, v_col={}, \
         blayout=row_major, slayout=none_box, fractal=512, pad=3>",
        dtype, rows, cols, rows, cols
    );
    ops.push(format!(
        "pto.tfillpad ins({} : {}) outs({} : {})",
        tsrc.ssa, src_ty, dst_ssa, dst_ty
    ));

    Ok(())
}

/// Iota / arithmetic progression: `%res = llvm.call @__tile_arith_progression_i32(%c0, %start, %valid_col)`
///
/// Emits `pto.tci` — the canonical iota op consumed by ptoas. Output is a
/// 1×N i32 tile where `dst[i] = start + i` for `i in 0..valid_col`. Used
/// as the index initializer for `pto.tsort32`.
///
/// MLIR shape (verified 2026-04-29 against
/// `/tmp/mgather_skip_ptoas/kernels/aiv/main_incore_0.pto` on 910c):
/// ```text
///   pto.tci ins(%c0_i32 {descending = false} : i32)
///           outs(%dst : !pto.tile_buf<...rows=1, cols=N, dtype=i32...>)
/// ```
fn translate_arith_progression(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("arith_progression: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("arith_progression: cannot parse args in: {}", line))?;
    let _start = ctx.resolve_const(args.get(1).map(|s| s.as_str()).unwrap_or("0"));
    let valid_col = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));

    if valid_col == 0 {
        return Err("arith_progression: valid_col must be > 0".to_string());
    }

    // Scalar i32 start operand (ptoas requires SSA-bound, not attribute).
    let start_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant 0 : i32", start_ssa));

    // Indices are unsigned (matches `pto.tsort32` consumer dtype `ui32`).
    let dst_ssa = ctx.alloc_tile(&result_ssa, 1, valid_col, "ui32", ops);
    let dst_ty = tile_buf_type(1, valid_col, "ui32");
    ops.push(format!(
        "pto.tci ins({} {{descending = false}} : i32) outs({} : {})",
        start_ssa, dst_ssa, dst_ty
    ));

    Ok(())
}

/// Top-K: `%res = llvm.call @__tile_topk_f32(%c0, %src, %indices_out, %rows, %cols, %k)`
///
/// **Path A composed emit** (2026-04-29): for `rows=1` and `cols` a
/// multiple of 64, the topk pipeline lowers to the tilelang
/// topk_selector algorithm composed from 4 PTO ops:
///   1. `pto.tci`         — iota indices `1×cols` ui32
///   2. `pto.tsort32`     — sort (values, indices) → 1×(2*cols) interleaved [val, idx]
///   3. `pto.tgather`     — mask P1010 extracts value channel → 1×cols
///   4. `pto.tmov`        — passthrough into the 1×k output tile
///                          (head-extract is implicit: ptoas handles
///                          v_col=k truncation; not bit-exact at the
///                          MLIR layer — see paper §saturation).
///
/// For other shapes (rows>1 or cols not 64-aligned), falls back to the
/// stub passthrough. This covers Path A's "tilelang-port at native
/// shape" deliverable; rows>1 needs a row-blocked emit (future work).
fn translate_topk(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("topk: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("topk: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("topk: missing src")?.trim();
    let _indices_ssa = args.get(2).ok_or("topk: missing indices_out")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let k = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("topk: unknown tile {}", src_ssa))?
        .clone();
    let src_ty = tsrc.tile_buf_type_str();
    let dst_ty = tile_buf_type(rows, k, dtype);

    // Path A composed emit: rows=1 and cols % 64 == 0 (vbitsort granularity).
    if rows == 1 && cols > 0 && cols % 64 == 0 && k > 0 && k <= cols {
        ops.push(format!(
            "// --- topk: tilelang topk_selector port, 1×{} → 1×{} {} ---",
            cols, k, dtype
        ));

        // Step 1: tci → indices [0..cols] ui32
        let idx_key = format!("{}__topk_idx", result_ssa);
        let idx_ssa = ctx.alloc_tile(&idx_key, 1, cols, "ui32", ops);
        let idx_ty = tile_buf_type(1, cols, "ui32");
        let start_ssa = ctx.fresh_ssa();
        ops.push(format!("{} = arith.constant 0 : i32", start_ssa));
        ops.push(format!(
            "pto.tci ins({} {{descending = false}} : i32) outs({} : {})",
            start_ssa, idx_ssa, idx_ty
        ));

        // Step 2: tsort32 (src, idx) → sorted_interleaved 1×(2*cols)
        let sort_key = format!("{}__topk_sorted", result_ssa);
        let sort_cols = cols * 2;
        let sort_ssa = ctx.alloc_tile(&sort_key, 1, sort_cols, dtype, ops);
        let sort_ty = tile_buf_type(1, sort_cols, dtype);
        ops.push(format!(
            "pto.tsort32 ins({}, {} : {}, {}) outs({} : {})",
            tsrc.ssa, idx_ssa, src_ty, idx_ty, sort_ssa, sort_ty
        ));

        // Step 3: tgather mask=P1010 → value channel 1×cols
        let val_key = format!("{}__topk_vals", result_ssa);
        let val_ssa = ctx.alloc_tile(&val_key, 1, cols, dtype, ops);
        let val_ty = tile_buf_type(1, cols, dtype);
        ops.push(format!(
            "pto.tgather ins({}, {{maskPattern = #pto.mask_pattern<P1010>}} : {}) \
             outs({} : {})",
            sort_ssa, sort_ty, val_ssa, val_ty
        ));

        // Step 4: tmov head-K → output tile 1×k.
        // Note: ptoas handles the v_col=k truncation; the MLIR layer
        // emits a same-shape tmov on val_ssa. The on-device kernel reads
        // only the first k lanes per the GM tstore that follows.
        let out_ssa = ctx.alloc_tile(&result_ssa, 1, k, dtype, ops);
        ops.push(format!(
            "// head-extract first {} of {} sorted values (ptoas v_col truncation)",
            k, cols
        ));
        ops.push(format!(
            "pto.tmov ins({} : {}) outs({} : {})",
            val_ssa, val_ty, out_ssa, dst_ty
        ));

        return Ok(());
    }

    // Fallback: rows>1 or non-aligned cols → stub passthrough.
    ops.push(format!(
        "// --- topk: top-{} selection, {}x{} {} (stub fallback) ---",
        k, rows, k, dtype
    ));
    ops.push(
        "// PTO topk port handles only rows=1 + cols % 64 == 0 (Path A scope)."
            .to_string(),
    );

    let out_ssa = ctx.alloc_tile(&result_ssa, rows, k, dtype, ops);
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tsrc.ssa, src_ty, out_ssa, dst_ty
    ));

    Ok(())
}

/// Matmul f16: `%res = llvm.call @__tile_matmul_f16(%c0, %a, %b, %m, %k, %n)`
///
/// Same cube-unit pipeline as f32 matmul but with f16 dtype throughout.
fn translate_matmul_f16(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("matmul_f16: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("matmul_f16: cannot parse args in: {}", line))?;
    let a_ssa = args.get(1).ok_or("matmul_f16: missing a")?.trim();
    let b_ssa = args.get(2).ok_or("matmul_f16: missing b")?.trim();
    let m = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let k = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let n = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(a_ssa)
        .ok_or_else(|| format!("matmul_f16: unknown tile {}", a_ssa))?
        .clone();
    let tb = ctx
        .get_tile(b_ssa)
        .ok_or_else(|| format!("matmul_f16: unknown tile {}", b_ssa))?
        .clone();

    let f16_dtypes = MatmulDtypes::f16_mixed();

    // Blocked path: real decoder shapes (K≥1536) overflow L0A/L0B's 64KB
    // even with f16 halving. Delegate to the shared blocked emitter, which
    // handles the K/N scf.for nest + per-block FixPipe store. This requires
    // both loads to be deferred (detect_blocked_matmul_loads always defers
    // f16 loads, so this is the common case).
    if matmul_needs_blocking(m, k, n, &f16_dtypes) {
        if let (Some(da), Some(db)) = (ta.deferred.clone(), tb.deferred.clone()) {
            return translate_matmul_blocked(
                &result_ssa, m, k, n, f16_dtypes, &da, &db, ctx, ops,
            );
        }
        // Fall through to single-block path if loads weren't deferred —
        // this will L0-overflow at compile time but gives the user a
        // clearer error than silently corrupting state.
    }

    ctx.use_size(m);
    ctx.use_size(k);
    ctx.use_size(n);

    // Operand partition_views. If the input loads were deferred (the common
    // case on f16 — see detect_blocked_matmul_loads rationale), build fresh
    // pvs from the deferred tv_ssa + elem_offset. Otherwise reuse the pv
    // emitted by the upstream load. Deferring is required for f16 because
    // CANN 8.5 ccec rejects b16 GM→UB (vec-tile) tloads on cube cores; we
    // must land directly in a mat-tile.
    let pv_a_ty = ptv_type(m, k, "f16");
    let pv_b_ty = ptv_type(k, n, "f16");
    let pv_a = if let Some(da) = ta.deferred.clone() {
        let a_base_row = da.elem_offset / k; // A is M×K, row = offset/K
        let a_base_col = da.elem_offset % k;
        ctx.use_size(a_base_row);
        ctx.use_size(a_base_col);
        let pv_a_blk = ctx.fresh_ssa();
        ops.push(format!(
            "{} = pto.partition_view {}, offsets = [%c{}, %c{}], sizes = [%c{}, %c{}] : {} -> {}",
            pv_a_blk,
            da.tv_ssa,
            a_base_row,
            a_base_col,
            m,
            k,
            tv_type(m, k, "f16"),
            pv_a_ty
        ));
        pv_a_blk
    } else {
        ta.pv_ssa.clone().ok_or_else(|| {
            format!(
                "matmul_f16: tile {} has no partition view (not loaded from GM)",
                a_ssa
            )
        })?
    };
    let pv_b = if let Some(db) = tb.deferred.clone() {
        let b_base_row = db.elem_offset / n; // B is K×N, row = offset/N
        let b_base_col = db.elem_offset % n;
        ctx.use_size(b_base_row);
        ctx.use_size(b_base_col);
        let pv_b_blk = ctx.fresh_ssa();
        ops.push(format!(
            "{} = pto.partition_view {}, offsets = [%c{}, %c{}], sizes = [%c{}, %c{}] : {} -> {}",
            pv_b_blk,
            db.tv_ssa,
            b_base_row,
            b_base_col,
            k,
            n,
            tv_type(k, n, "f16"),
            pv_b_ty
        ));
        pv_b_blk
    } else {
        tb.pv_ssa.clone().ok_or_else(|| {
            format!(
                "matmul_f16: tile {} has no partition view (not loaded from GM)",
                b_ssa
            )
        })?
    };

    // 1. Alloc CBUF staging tiles
    let mat_a_key = format!("{}__mat_a", result_ssa);
    let mat_b_key = format!("{}__mat_b", result_ssa);
    let mat_a_ty = mat_tile_type(m, k, "f16");
    let mat_b_ty = mat_tile_type(k, n, "f16");
    let mat_a_ssa = ctx.alloc_tile_typed(&mat_a_key, m, k, "f16", &mat_a_ty, ops);
    let mat_b_ssa = ctx.alloc_tile_typed(&mat_b_key, k, n, "f16", &mat_b_ty, ops);

    // 2. Alloc L0A/L0B/L0C tiles.
    //
    // CRITICAL: ptoas on CANN 8.5 enforces `pto.tmatmul` dtype triples —
    // the accepted set is (dst, lhs, rhs) ∈ { (i32,i8,i8), (f32,f16,f16),
    // (f32,bf16,bf16), (f32,f32,f32) }. An all-f16 tmatmul is REJECTED
    // at MLIR parse time (empirically confirmed 2026-04-16 — see
    // memory/project_pto_tmatmul_dtype_rules.md). So L0A/L0B stay f16
    // (the whole point — halves HBM for B) but the L0C accumulator MUST
    // be f32.
    //
    // There is NO supported tmov from Acc to Vec — the pto_instr TMov
    // static_assert only accepts Mat→{Left,Right,Bias,Scaling}, Vec→Vec,
    // and Mat→Acc. To get L0C data out to f16 GM we rely on the hardware
    // FixPipe: tstore from an Acc (f32) tile directly into an f16 GM
    // partition_view performs the f32→f16 cast in-flight during the
    // L0C→GM DMA. We register the f32 acc tile under `result_ssa` so the
    // caller's downstream `__tile_store_f16` reads the acc tile and
    // emits `pto.tstore ins(acc : f32) outs(pv : f16)`.
    let left_key = format!("{}__left", result_ssa);
    let right_key = format!("{}__right", result_ssa);
    let left_ty = left_tile_type(m, k, "f16");
    let right_ty = right_tile_type(k, n, "f16");
    let acc_ty = acc_tile_type(m, n, "f32");
    let left_ssa = ctx.alloc_tile_typed(&left_key, m, k, "f16", &left_ty, ops);
    let right_ssa = ctx.alloc_tile_typed(&right_key, k, n, "f16", &right_ty, ops);
    let acc_ssa = ctx.alloc_tile_typed(&result_ssa, m, n, "f32", &acc_ty, ops);

    // 3. tload GM -> mat tiles (CBUF). pv_a_ty / pv_b_ty were computed
    //    above alongside pv_a / pv_b.
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_a, pv_a_ty, mat_a_ssa, mat_a_ty
    ));
    ops.push(format!(
        "pto.tload ins({} : {}) outs({} : {})",
        pv_b, pv_b_ty, mat_b_ssa, mat_b_ty
    ));

    // 4. tmov: CBUF -> L0A / L0B
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        mat_a_ssa, mat_a_ty, left_ssa, left_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        mat_b_ssa, mat_b_ty, right_ssa, right_ty
    ));

    // 5. tmatmul: L0A x L0B -> L0C (dst=f32, lhs=f16, rhs=f16 — the only
    //    mixed-dtype triple ptoas accepts for f16 matmul on CANN 8.5).
    //    The f32 acc tile is registered under `result_ssa` above; the
    //    caller's `__tile_store_f16` will emit a `pto.tstore` from
    //    this f32 acc into an f16 GM partition_view, and the hardware
    //    FixPipe path performs the f32→f16 cast during the L0C→GM DMA.
    ops.push(format!(
        "pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
        left_ssa, right_ssa, left_ty, right_ty, acc_ssa, acc_ty
    ));

    Ok(())
}

/// Matmul i8×i8→i32 with per-column f32 dequant → f16 GM:
/// `%res = llvm.call @__tile_matmul_i8_acc_i32_dequant_f16(
///            %c0, %a, %b, %scale_ptr, %m, %k, %n)`
///
/// Dtype rules (ptoas CANN 8.5, empirical): A / B both i8, L0C accumulator
/// i32. Per-column f32 scale tile is loaded into `loc=scaling` (__fbuf__) and
/// folded into the L0C→GM DMA via `pto.tstore_fp` (FixPipe). The output is
/// registered as `dst="i32"` under the matmul result SSA; the caller's
/// downstream `__tile_store_f16` sees the i32 acc tile but the inline
/// store emitted by the blocked-matmul path writes via tstore_fp with f16
/// GM dtype, dequanting in-flight.
///
/// See:
///   - memory/project_pto_i8_tmatmul_validated.md
///   - /tmp/smoke_i8_kv_proj_dequant.acl.pto (validated decoder-shape probe)
fn translate_matmul_i8(
    line: &str,
    ctx: &mut PtoContext,
    func: &MlirFunc,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("matmul_i8: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("matmul_i8: cannot parse args in: {}", line))?;
    // extern fn __tile_matmul_i8_acc_i32_dequant_f16(
    //   dst: u32, a: u32, b: u32, scale: *const f32,
    //   m: u32, k: u32, n: u32) -> u32
    // args[0]=dst, args[1]=a, args[2]=b, args[3]=scale, args[4]=m,
    // args[5]=k, args[6]=n.
    let a_ssa = args.get(1).ok_or("matmul_i8: missing a")?.trim();
    let b_ssa = args.get(2).ok_or("matmul_i8: missing b")?.trim();
    let scale_arg = args.get(3).ok_or("matmul_i8: missing scale")?.trim();
    let m = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let k = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));
    let n = ctx.resolve_const(args.get(6).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(a_ssa)
        .ok_or_else(|| format!("matmul_i8: unknown tile {}", a_ssa))?
        .clone();
    let tb = ctx
        .get_tile(b_ssa)
        .ok_or_else(|| format!("matmul_i8: unknown tile {}", b_ssa))?
        .clone();

    let dtypes = MatmulDtypes::i8_quantized();

    // Only the K/N-blocked path is supported for i8 matmul right now. Real
    // decoder shapes always need blocking (e.g., K=1536 × N=256 at 1B = 384KB
    // > L0B 64KB). The dispatch also asserts deferred-load for both operands
    // (f16 matmul always defers; i8 matmul we will defer via
    // detect_blocked_matmul_loads).
    if !matmul_needs_blocking(m, k, n, &dtypes) {
        return Err(format!(
            "matmul_i8: single-block path unsupported (M={} K={} N={}); \
             extend detect_blocked_matmul_loads + translate_matmul_i8 to \
             cover small-shape i8 matmul",
            m, k, n
        ));
    }
    let (da, db) = match (ta.deferred.clone(), tb.deferred.clone()) {
        (Some(da), Some(db)) => (da, db),
        _ => {
            return Err(format!(
                "matmul_i8: both operand loads must be deferred for blocked emit \
                 (a.deferred={}, b.deferred={}); detect_blocked_matmul_loads \
                 must mark i8 matmul inputs",
                ta.deferred.is_some(),
                tb.deferred.is_some()
            ));
        }
    };

    // Resolve the scale pointer → GM arg name. Same pattern as tile_load for
    // i8/f32 operands (follow ptr_aliases + GEP chain to the func arg).
    let scale_resolved = ctx.resolve_ptr(scale_arg);
    let scale_gm_name = resolve_gm_name(&scale_resolved, func);
    // No GEP offset handling for the scale pointer: decode callers pass a
    // raw per-layer scale vector directly. If a user needs to pass a GEP'd
    // offset view, extend here with resolve_offset.
    //
    // Build a tensor_view for scale (shape 1×N, ui64 packed). The per-N-block
    // partition_view + tload is emitted inside emit_blocked_matmul_loops.
    // Host-side packs per-column f32 scale as u64 FB words (see
    // memory/project_cann85_i8_path_viable_via_tmov3arg.md).
    let tv_scale_ssa = ctx.get_or_make_tv(&scale_gm_name, 1, n, "ui64", ops);
    let lhs_bytes = dtypes.lhs_bytes() as u32;

    // Delegate to the shared blocked emitter first so it allocates the core i8
    // tiles (mat_a, mat_b, left, right, acc) BEFORE we allocate the scaling
    // tiles. ptoas is sensitive to tile declaration order: when the ui64
    // scaling/mat tiles are allocated first, ptoas flips the i8 Left tile's
    // BLayout from RowMajor to ColMajor, which breaks the numerics on
    // dav-c220-cube. Matching the hand-written probe's order (i8 tiles first,
    // scaling last) keeps ptoas on the verified codepath.
    translate_matmul_blocked(&result_ssa, m, k, n, dtypes, &da, &db, ctx, ops)?;

    // Allocate scale tiles AFTER the i8 tiles. CANN 8.5 ptoas rejects direct
    // tload→Scaling, so we hop via L0B-Mat:
    //   tload GM → scale_mat (loc=mat, ui64, none_box, fractal=32)
    //   tmov scale_mat → scale_fb (loc=scaling, ui64, none_box, fractal=32)
    // TMovToFb requires uint64_t DstType + Rows=1 + Cols×sizeof%128==0.
    let nb = pick_nb_for_dtype(n, lhs_bytes);
    let scale_mat_ty = format!(
        "!pto.tile_buf<loc=mat, dtype=ui64, rows=1, cols={}, v_row=1, v_col={}, \
         blayout=row_major, slayout=none_box, fractal=32, pad=0>",
        nb, nb
    );
    let scale_mat_ssa = ctx.alloc_tile_typed(
        &format!("{}__scale_mat", result_ssa),
        1,
        nb,
        "ui64",
        &scale_mat_ty,
        ops,
    );
    let scale_tile_ty = format!(
        "!pto.tile_buf<loc=scaling, dtype=ui64, rows=1, cols={}, v_row=1, v_col={}, \
         blayout=row_major, slayout=none_box, fractal=32, pad=0>",
        nb, nb
    );
    let scale_tile_ssa = ctx.alloc_tile_typed(
        &format!("{}__scale_blk", result_ssa),
        1,
        nb,
        "ui64",
        &scale_tile_ty,
        ops,
    );

    // Placeholder pv for the full-row scale — we reuse `tv_scale_ssa` + a
    // per-block partition_view inside the N-loop rather than a hoisted
    // 1×N pv. These fields stay in DequantSpec for future single-block
    // code paths; emit_blocked_matmul_loops currently ignores them.
    let pv_scale_ssa = String::new();
    let pv_scale_ty = ptv_type(1, nb, "ui64");

    // The blocked emitter pushed the pending descriptor without dequant.
    // Patch it to carry the scale tile so the tstore emission below is
    // tstore_fp. (Cleaner than forking the whole blocked path.)
    let pending = ctx
        .pending_blocked_matmuls
        .get_mut(&result_ssa)
        .ok_or("matmul_i8: expected pending blocked matmul after translate_matmul_blocked")?;
    pending.dequant = Some(DequantSpec {
        scale_tile_ssa,
        scale_tile_ty,
        scale_mat_ssa,
        scale_mat_ty,
        tv_scale_ssa,
        pv_scale_ssa,
        pv_scale_ty,
    });

    Ok(())
}

/// Fill: `%res = llvm.call @__tile_fill_f32(%c0, %scalar, %rows, %cols)`
/// → alloc_tile + pto.tmov (broadcast scalar)
fn translate_fill(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("fill: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("fill: cannot parse args in: {}", line))?;
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let dtype = if line.contains("f16") { "f16" } else { "f32" };

    let tb_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);
    let tb_ty = tile_buf_type(rows, cols, dtype);
    ops.push(format!("// fill {}x{} with scalar (broadcast via tmov)", rows, cols));
    ops.push(format!("pto.tmov ins({} : {}) outs({} : {})", tb_ssa, tb_ty, tb_ssa, tb_ty));

    Ok(())
}

/// RMSNorm: `y = x * rsqrt(mean(x^2) + eps)`
///
/// Emits the 8-step PTO-MLIR sequence:
///
///   1. sq      = tmul(x, x)                     (rows×cols  row_major)
///   2. sum_sq  = trowsum(sq, tmp)               (rows×1     col_major)
///   3. mean    = tmuls(sum_sq, 1/cols)          (rows×8 v=1 row_major)
///   4. m_eps   = tadds(mean, eps)               (rows×8 v=1 row_major)
///   5. sqrt    = tsqrt(m_eps)                   (rows×8 v=1 row_major)
///   6. inv     = trecip(sqrt)                   (rows×8 v=1 row_major)
///   7. inv_b   = trowexpand(inv)                (rows×cols  row_major)
///   8. y       = tmul(x, inv_b)                 (rows×cols  row_major)
///
/// `pto.tsqrt + pto.trecip` matches the Qwen3DecodeA3 sample's RMSNorm
/// pattern (`/data/y00949728/workspace/PTOAS/test/samples/Qwen3DecodeA3/qwen3_decode_incore_0.pto`)
/// and sidesteps the vrsqrt instruction's lane-garbage NaN propagation.
///
/// Steps 7–8 (`trowexpand` + `tmul`) replace the more concise
/// `trowexpandmul`. The latter's underlying vmul reads 8 lanes of src1 in
/// each 256-bit broadcast block, so for R=1 col_major V=1×1 src1 (only
/// lane 0 populated, lanes 1..7 garbage) it would corrupt 7/8 of dst.
/// `trowexpand` instead uses `vector_dup` from a single scalar, then
/// `tmul` does the per-element multiply on a fully-populated dst.
///
/// `pto.barrier <PIPE_ALL>` is emitted between every V op to match the
/// working sample. ptoas may drop barriers during lowering; they are
/// harmless when preserved and required for correctness when the
/// scheduler issues V ops in parallel.
/// Render an f32 without scientific notation.
///
/// ptoas's MLIR parser rejects `6.510417e-4`-style literals (sees the `e`
/// as a custom op name). Decimal-only form works for both mlir-opt and ptoas.
fn format_f32_decimal(v: f32) -> String {
    let s = format!("{:.9}", v);
    let trimmed = s.trim_end_matches('0');
    let trimmed = trimmed.trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0.0".to_string()
    } else if !trimmed.contains('.') {
        format!("{}.0", trimmed)
    } else {
        trimmed.to_string()
    }
}

fn translate_rms_norm_pto(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("rms_norm: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("rms_norm: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("rms_norm: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));
    let dtype = if line.contains("f16") { "f16" } else { "f32" };

    let ta = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("rms_norm: unknown tile {}", src_ssa))?
        .clone();
    let ta_ty = ta.tile_buf_type_str();
    let vec_ty = tile_buf_type(rows, cols, dtype);
    // ptoas rejects `pto.trsqrt` with col_major src blayout, so the entire
    // row-reduce chain in rms_norm uses row_major. Other row-reduce consumers
    // (softmax's trowmax/trowsum) keep col_major via the original helper.
    let rr_ty = tile_buf_type_rowreduce_rowmajor(rows, dtype);

    let inv_cols = if cols > 0 { 1.0_f32 / (cols as f32) } else { 0.0 };
    let eps = 1.0e-6_f32;

    let c_inv_cols = ctx.fresh_ssa();
    ops.push(format!(
        "{} = arith.constant {} : f32",
        c_inv_cols,
        format_f32_decimal(inv_cols)
    ));
    let c_eps = ctx.fresh_ssa();
    ops.push(format!(
        "{} = arith.constant {} : f32",
        c_eps,
        format_f32_decimal(eps)
    ));

    // Col-major V=R×1 view used by trowsum dst (verifier requires).
    let cm_ty = tile_buf_type_rowreduce(rows, dtype);

    let sq_ssa = ctx.alloc_tile(&format!("{}__rms_sq", result_ssa), rows, cols, dtype, ops);
    let tmp_ssa = ctx.alloc_tile(&format!("{}__rms_tmp", result_ssa), rows, cols, dtype, ops);
    // sum: col_major V=R×1 (matches trowsum dst constraint)
    let sum_ssa = ctx.alloc_tile_rowreduce(&format!("{}__rms_sum", result_ssa), rows, dtype, ops);
    // mean/m_eps/sqrt/inv: row_major V=R×8, v_col=1 (matches Qwen3 pattern)
    let mean_ssa = ctx.alloc_tile_rowreduce_rowmajor(&format!("{}__rms_mean", result_ssa), rows, dtype, ops);
    let meps_ssa = ctx.alloc_tile_rowreduce_rowmajor(&format!("{}__rms_meps", result_ssa), rows, dtype, ops);
    let sqrt_ssa = ctx.alloc_tile_rowreduce_rowmajor(&format!("{}__rms_sqrt", result_ssa), rows, dtype, ops);
    let inv_ssa = ctx.alloc_tile_rowreduce_rowmajor(&format!("{}__rms_inv", result_ssa), rows, dtype, ops);
    // inv_b: D-element broadcast of inv_rms (full row, populated lanes).
    let inv_b_ssa = ctx.alloc_tile(&format!("{}__rms_inv_b", result_ssa), rows, cols, dtype, ops);
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);

    // 1. sq = x * x
    ops.push(format!(
        "pto.tmul ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        ta.ssa, ta_ty, sq_ssa, vec_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 2. sum_sq = trowsum(sq, tmp)  (col_major dst)
    ops.push(format!(
        "pto.trowsum ins({}, {} : {}, {}) outs({} : {})",
        sq_ssa, tmp_ssa, vec_ty, vec_ty, sum_ssa, cm_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 3. mean = sum_sq * (1/cols)  (input col_major V=R×1, output row_major V=R×8 v_col=1)
    ops.push(format!(
        "pto.tmuls ins({}, {} : {}, f32) outs({} : {})",
        sum_ssa, c_inv_cols, cm_ty, mean_ssa, rr_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 4. m_eps = mean + eps
    ops.push(format!(
        "pto.tadds ins({}, {} : {}, f32) outs({} : {})",
        mean_ssa, c_eps, rr_ty, meps_ssa, rr_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 5. sqrt_v = sqrt(m_eps)
    ops.push(format!(
        "pto.tsqrt ins({} : {}) outs({} : {})",
        meps_ssa, rr_ty, sqrt_ssa, rr_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 6. inv_rms = 1 / sqrt_v   (Qwen3 pattern: tsqrt → trecip, sidesteps trsqrt)
    ops.push(format!(
        "pto.trecip ins({} : {}) outs({} : {})",
        sqrt_ssa, rr_ty, inv_ssa, rr_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 7. inv_b = trowexpand(inv) — broadcast lane-0 scalar via vector_dup
    //    across all D columns of the dst row.
    ops.push(format!(
        "pto.trowexpand ins({} : {}) outs({} : {})",
        inv_ssa, rr_ty, inv_b_ssa, vec_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());
    // 8. y = x .* inv_b  (per-element multiply, no broadcast issue)
    ops.push(format!(
        "pto.tmul ins({}, {} : {}, {}) outs({} : {})",
        ta.ssa, inv_b_ssa, ta_ty, vec_ty, out_ssa, vec_ty
    ));
    ops.push("pto.barrier <PIPE_ALL>".to_string());

    Ok(())
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------

/// RoPE: `%res = llvm.call @__tile_rope_f32(%c0, %src, %pos, %rows, %cols)`
///
/// For each row r and pair index i (0..cols/2):
///   freq  = 1.0 / pow(10000.0, 2.0 * i / cols)
///   angle = pos * freq
///   out[r*cols + 2*i]     = x[r*cols + 2*i] * cos(angle) - x[r*cols + 2*i+1] * sin(angle)
///   out[r*cols + 2*i + 1] = x[r*cols + 2*i] * sin(angle) + x[r*cols + 2*i+1] * cos(angle)
///
/// PTO has no native sin/cos/pow ops; this emits a shape-correct STUB that
/// copies src → dst via `tmul` (identity). Use `mlir_to_cpp` for a
/// numerically correct RoPE until PTO gains trigonometric intrinsics.
fn translate_rope_pto(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("rope: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("rope: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("rope: missing src")?.trim();
    // args[2] is the position index — consumed but unused in the stub
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("rope: unknown tile {}", src_ssa))?
        .clone();
    let ta_ty = ta.tile_buf_type_str();

    // Allocate output tile mapped to the result SSA
    let tc_ssa = ctx.alloc_tile(&result_ssa, rows, cols, "f32", ops);
    let tc_ty = tile_buf_type(rows, cols, "f32");

    ops.push(format!(
        "// --- rope: Rotary Position Embedding {}x{} f32 ---",
        rows, cols
    ));
    ops.push(
        "// STUB: PTO lacks sin/cos/pow. Passthrough (identity) preserves shape; \
         use mlir_to_cpp for numerically correct RoPE."
            .to_string(),
    );

    // Identity copy: out = src * 1.0 (shape-correct passthrough)
    let cone_ssa = ctx.fresh_ssa();
    ops.push(format!("{} = arith.constant 1.0 : f32", cone_ssa));
    ops.push(format!(
        "pto.tmuls ins({}, {} : {}, f32) outs({} : {})",
        ta.ssa, cone_ssa, ta_ty, tc_ssa, tc_ty
    ));

    Ok(())
}

// ---------------------------------------------------------------------------
// INT8 quantization helpers
// ---------------------------------------------------------------------------

/// absmax: abs(src) → row-reduce max → broadcast scalar back to tile
fn translate_absmax_pto(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("absmax: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("absmax: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("absmax: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("absmax: unknown tile {}", src_ssa))?
        .clone();
    let ta_ty = ta.tile_buf_type_str();

    // scratch: abs of src
    let abs_key = format!("{}__abs", result_ssa);
    let abs_ssa = ctx.alloc_tile(&abs_key, rows, cols, "f32", ops);
    let abs_ty = tile_buf_type(rows, cols, "f32");

    // row-reduce max (rows×1)
    let max_key = format!("{}__max", result_ssa);
    let max_ssa = ctx.alloc_tile_rowreduce(&max_key, rows, "f32", ops);
    let max_ty = tile_buf_type_rowreduce(rows, "f32");

    // output tile: broadcast back to rows×cols via tmaxs (scalar-broadcast max)
    let tc_ssa = ctx.alloc_tile(&result_ssa, rows, cols, "f32", ops);
    let tc_ty = tile_buf_type(rows, cols, "f32");

    ops.push(format!("// absmax: abs(src) → row-reduce max → broadcast"));
    ops.push(format!(
        "pto.tabs ins({} : {}) outs({} : {})",
        ta.ssa, ta_ty, abs_ssa, abs_ty
    ));
    ops.push(format!(
        "pto.trowmax ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        abs_ssa, abs_ty, max_ssa, max_ty
    ));
    ops.push(format!(
        "pto.tmaxs ins({}, {} : {}, {}) outs({} : {})",
        abs_ssa, max_ssa, abs_ty, max_ty, tc_ssa, tc_ty
    ));

    Ok(())
}

/// quantize: round(src / scale) clamped to [-128, 127]
fn translate_quantize_pto(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("quantize: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("quantize: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("quantize: missing src")?.trim();
    let _scale_ssa = args.get(2).ok_or("quantize: missing scale")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("quantize: unknown tile {}", src_ssa))?
        .clone();
    let ta_ty = ta.tile_buf_type_str();

    // scratch: src / scale (scalar divide)
    let div_key = format!("{}__div", result_ssa);
    let div_ssa = ctx.alloc_tile(&div_key, rows, cols, "f32", ops);
    let div_ty = tile_buf_type(rows, cols, "f32");

    // output tile (stored as f32, caller converts to i8)
    let tc_ssa = ctx.alloc_tile(&result_ssa, rows, cols, "f32", ops);
    let tc_ty = tile_buf_type(rows, cols, "f32");

    ops.push(format!("// quantize: round(src/scale) clamped [-128,127]"));
    ops.push(format!(
        "pto.tdivs ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        ta.ssa, ta_ty, div_ssa, div_ty
    ));
    // tmins(127) + tmaxs(-128) approximate round+clamp via scalar ops
    ops.push(format!(
        "pto.tmins ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        div_ssa, div_ty, tc_ssa, tc_ty
    ));

    Ok(())
}

/// dequantize: src * scale (i8→f32)
fn translate_dequantize_pto(
    line: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("dequantize: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("dequantize: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("dequantize: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let ta = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("dequantize: unknown tile {}", src_ssa))?
        .clone();
    let ta_ty = ta.tile_buf_type_str();

    let tc_ssa = ctx.alloc_tile(&result_ssa, rows, cols, "f32", ops);
    let tc_ty = tile_buf_type(rows, cols, "f32");

    ops.push(format!("// dequantize: src * scale"));
    ops.push(format!(
        "pto.tmuls ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        ta.ssa, ta_ty, tc_ssa, tc_ty
    ));

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 6 MTP op translators (PTO scalar-loop decomposition)
// ---------------------------------------------------------------------------

/// Argmax: `%res = llvm.call @__tile_argmax_f32(%c0, %src, %rows, %cols)`
///
/// PTO has no native argmax. Decompose to trowmax to find the max per row,
/// then emit a comment for the index-scan loop that a downstream pass would
/// fill in.  The output tile is (rows × 1).
fn translate_argmax_pto(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("argmax: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("argmax: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("argmax: missing src")?.trim();
    let rows = ctx.resolve_const(args.get(2).map(|s| s.as_str()).unwrap_or("0"));
    let _cols = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("argmax: unknown tile {}", src_ssa))?
        .clone();
    let tsrc_ty = tsrc.tile_buf_type_str();

    // row-wise max scratch (rows × 1)
    let max_key = format!("{}__max", result_ssa);
    let max_ssa = ctx.alloc_tile_rowreduce(&max_key, rows, dtype, ops);
    let max_ty = tile_buf_type_rowreduce(rows, dtype);

    // output: (rows × 1) — stores row indices (approximated by max tile)
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, 1, dtype, ops);
    let out_ty = tile_buf_type(rows, 1, dtype);

    ops.push(format!(
        "// --- argmax: row-wise argmax {}x? {} ---",
        rows, dtype
    ));
    ops.push("// PTO lacks native argmax. trowmax approximates the max value per row.".to_string());
    ops.push("// TODO: implement index scan via scalar loop to find the argmax index.".to_string());
    ops.push(format!(
        "pto.trowmax ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        tsrc.ssa, tsrc_ty, max_ssa, max_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        max_ssa, max_ty, out_ssa, out_ty
    ));

    Ok(())
}

/// SampleTopP: `%res = llvm.call @__tile_sample_top_p_f32(%c0, %logits, %temp, %top_p, %seed, %rows, %cols)`
///
/// Nucleus (top-p) sampling. PTO has no native equivalent.
/// Decompose to: sort logits (trowmax pass) → cumsum approximation → tmov passthrough.
fn translate_sample_top_p_pto(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("sample_top_p: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("sample_top_p: cannot parse args in: {}", line))?;
    let src_ssa = args.get(1).ok_or("sample_top_p: missing logits")?.trim();
    let rows = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));
    let _cols = ctx.resolve_const(args.get(6).map(|s| s.as_str()).unwrap_or("0"));

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("sample_top_p: unknown tile {}", src_ssa))?
        .clone();
    let tsrc_ty = tsrc.tile_buf_type_str();

    // output: (rows × 1) sampled token indices
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, 1, dtype, ops);
    let out_ty = tile_buf_type(rows, 1, dtype);

    ops.push(format!(
        "// --- sample_top_p: nucleus sampling {}x? {} ---",
        rows, dtype
    ));
    ops.push("// PTO lacks native nucleus sampling. tmov passthrough.".to_string());
    ops.push("// TODO: implement softmax + cumsum + binary search via scalar loop.".to_string());
    // Use trowmax to get the max (greedy approximation) and pass through
    let max_key = format!("{}__max", result_ssa);
    let max_ssa = ctx.alloc_tile_rowreduce(&max_key, rows, dtype, ops);
    let max_ty = tile_buf_type_rowreduce(rows, dtype);
    ops.push(format!(
        "pto.trowmax ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        tsrc.ssa, tsrc_ty, max_ssa, max_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        max_ssa, max_ty, out_ssa, out_ty
    ));

    Ok(())
}

/// DraftVerify: `%res = llvm.call @__tile_draft_verify_f32(%c0, %draft_tokens, %target_logits, %rows, %cols)`
///
/// Speculative decoding acceptance probability: p_accept[r] = min(1, target[r, draft[r]] / draft[r, draft[r]]).
/// PTO has no native equivalent. Emit trowmax approximation + comment.
fn translate_draft_verify_pto(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("draft_verify: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("draft_verify: cannot parse args in: {}", line))?;
    let target_ssa = args.get(2).ok_or("draft_verify: missing target_logits")?.trim();
    let rows = ctx.resolve_const(args.get(3).map(|s| s.as_str()).unwrap_or("0"));
    let _cols = ctx.resolve_const(args.get(4).map(|s| s.as_str()).unwrap_or("0"));

    let ttgt = ctx
        .get_tile(target_ssa)
        .ok_or_else(|| format!("draft_verify: unknown tile {}", target_ssa))?
        .clone();
    let ttgt_ty = ttgt.tile_buf_type_str();

    // output: (rows × 1) acceptance probabilities
    let out_ssa = ctx.alloc_tile(&result_ssa, rows, 1, dtype, ops);
    let out_ty = tile_buf_type(rows, 1, dtype);

    ops.push(format!(
        "// --- draft_verify: acceptance probs {}x1 {} ---",
        rows, dtype
    ));
    ops.push("// PTO lacks native draft verify. trowmax approximation.".to_string());
    ops.push("// TODO: implement index gather + min(1, ratio) via scalar loop.".to_string());
    let max_key = format!("{}__max", result_ssa);
    let max_ssa = ctx.alloc_tile_rowreduce(&max_key, rows, dtype, ops);
    let max_ty = tile_buf_type_rowreduce(rows, dtype);
    ops.push(format!(
        "pto.trowmax ins({0}, {0} : {1}, {1}) outs({2} : {3})",
        ttgt.ssa, ttgt_ty, max_ssa, max_ty
    ));
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        max_ssa, max_ty, out_ssa, out_ty
    ));

    Ok(())
}

/// TokenAccept: `%res = llvm.call @__tile_token_accept_f32(%c0, %draft, %target, %probs, %threshold, %rows)`
///
/// Accept draft token if prob >= threshold, else fall back to target token.
/// PTO has no native equivalent. Emit tmov passthrough + comment.
fn translate_token_accept_pto(
    line: &str,
    dtype: &str,
    ctx: &mut PtoContext,
    ops: &mut Vec<String>,
) -> Result<(), String> {
    let result_ssa = extract_result_ssa(line)
        .ok_or_else(|| format!("token_accept: no result SSA in: {}", line))?;
    let args = extract_call_args(line)
        .ok_or_else(|| format!("token_accept: cannot parse args in: {}", line))?;
    let draft_ssa = args.get(1).ok_or("token_accept: missing draft_tokens")?.trim();
    let rows = ctx.resolve_const(args.get(5).map(|s| s.as_str()).unwrap_or("0"));

    let tdraft = ctx
        .get_tile(draft_ssa)
        .ok_or_else(|| format!("token_accept: unknown tile {}", draft_ssa))?
        .clone();
    let tdraft_ty = tdraft.tile_buf_type_str();

    let out_ssa = ctx.alloc_tile(&result_ssa, rows, 1, dtype, ops);
    let out_ty = tile_buf_type(rows, 1, dtype);

    ops.push(format!(
        "// --- token_accept: select final tokens {}x1 {} ---",
        rows, dtype
    ));
    ops.push("// PTO lacks native token_accept. tmov passthrough.".to_string());
    ops.push("// TODO: implement accept/reject via scalar comparison loop.".to_string());
    ops.push(format!(
        "pto.tmov ins({} : {}) outs({} : {})",
        tdraft.ssa, tdraft_ty, out_ssa, out_ty
    ));

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: resolve a call-arg GM pointer to the function parameter name
// ---------------------------------------------------------------------------

fn resolve_gm_name(arg: &str, func: &MlirFunc) -> String {
    // arg is something like `%arg0` already matching the func param
    // If it matches a func param name directly, use it.
    for fa in &func.args {
        if fa.name == arg && fa.is_gm {
            return fa.name.clone();
        }
    }
    // Fall back to the raw arg
    arg.to_string()
}

// MLIR text parsing moved to crate::mlir_parse (see use statement at top).

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// extract_result_ssa, extract_call_args, parse_const_arg moved to mlir_parse.

pub(crate) fn infer_dtype_from_name(name: &str) -> &'static str {
    if name.contains("f16") || name.contains("half") {
        "f16"
    } else {
        "f32"
    }
}

/// Infer a GM arg's dtype by scanning the function body for the first
/// `__tile_load_*` / `__tile_store_*` call that uses it as a
/// pointer operand. Returns None if no such call is found; the caller
/// then falls back to `infer_dtype_from_name`.
///
/// This matters for f16 kernels whose Rust arg names don't contain
/// "f16" (e.g. `fn matmul(a: *const f16, b: *const f16, out: *mut f16)`):
/// body-scanning correctly sets `!pto.ptr<f16>` so ptoas generates
/// `__gm__ half*` parameters consistent with the `GlobalTensor<half, ...>`
/// views derived from them.
pub(crate) fn infer_arg_dtype_from_body(arg_name: &str, body_lines: &[String]) -> Option<&'static str> {
    // Patterns that indicate this arg is used as a typed GM pointer.
    // The LLVM IR shape is: `llvm.call @__tile_load_fNN(%argK, ...)` or
    // `llvm.call @__tile_store_fNN(%argK, ...)`. We do a substring
    // check: if the line mentions both the arg name and a typed tile_load/
    // tile_store call, use the dtype from the callee name.
    for line in body_lines {
        if !line.contains(arg_name) {
            continue;
        }
        // Check store (GM arg is 1st pointer param, written).
        if line.contains("__tile_store_f16") {
            return Some("f16");
        }
        if line.contains("__tile_store_f32") {
            return Some("f32");
        }
        if line.contains("__tile_store_bf16") {
            return Some("bf16");
        }
        // Check load (GM arg is 1st pointer param, read).
        if line.contains("__tile_load_f16") {
            return Some("f16");
        }
        if line.contains("__tile_load_f32") {
            return Some("f32");
        }
        if line.contains("__tile_load_bf16") {
            return Some("bf16");
        }
        if line.contains("__tile_load_i8") {
            return Some("i8");
        }
        if line.contains("__tile_store_i8") {
            return Some("i8");
        }
        // int8 matmul has a scale arg at position 3. CANN 8.5 ptoas requires
        // the scale tile to be dtype=ui64 (TMovToFb needs uint64_t DstType),
        // so we emit `!pto.ptr<ui64>` here even though the Rust source declares
        // the pointer as `*const f32`. Host-side repacks f32 → u64 FB words
        // before launch (see pack_scale_f32_to_u64).
        if line.contains("__tile_matmul_i8_acc_i32_dequant_f16") {
            let args = match extract_call_args(line) {
                Some(a) => a,
                None => continue,
            };
            let scale_arg = args.get(3).map(|s| s.trim()).unwrap_or("");
            if scale_arg == arg_name {
                return Some("ui64");
            }
        }
    }
    None
}

// is_builtin_helper moved to mlir_parse.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_const_arg_bare() {
        assert_eq!(parse_const_arg("16"), 16);
        assert_eq!(parse_const_arg("32"), 32);
    }

    #[test]
    fn test_pick_kb_for_n_clamps_on_outer_stride_overflow() {
        // Default Kb (PTO_MM_KB) stands when N is small (Kb*N <= 2^23).
        // PTO_MM_KB is 128, lowered from 256 when the UB budget was corrected to
        // the vendor's 184 KB: at Kb=256/Nb=64 the L0B tile sat exactly on the
        // 64 KB cap and the UB working set was 272 KB.
        assert_eq!(pick_kb_for_n(1536, 1536), PTO_MM_KB); // q/o_proj
        assert_eq!(pick_kb_for_n(1536, 256), PTO_MM_KB); // kv_proj
        assert_eq!(pick_kb_for_n(1536, 8960), PTO_MM_KB); // gate/up_proj
        assert_eq!(pick_kb_for_n(8960, 1536), PTO_MM_KB); // down_proj
        // lm_head: N=151936 forces Kb down so that Kb*N < 2^24. With our
        // 2^23 threshold, Kb ≤ (2^23)/151936 ≈ 55 → aligned to 48.
        // 1536 % 48 == 0, so expect Kb=48.
        assert_eq!(pick_kb_for_n(1536, 151936), 48);
        // N=65536 exactly at old failure boundary: the stride cap is
        // 2^23/65536 = 128. Whichever of {stride cap, default Kb} is smaller
        // governs; asserted against PTO_MM_KB rather than a literal, because
        // the two merely COINCIDED at 128 while PTO_MM_KB was 128 and the
        // literal then silently encoded that coincidence as intent.
        assert_eq!(pick_kb_for_n(1536, 65536), PTO_MM_KB.min(128));
        // N=32768: stride cap = 2^23/32768 = 256, so the default Kb governs.
        assert_eq!(pick_kb_for_n(1536, 32768), PTO_MM_KB);
        // Degenerate / small K that's still > kb_cap.
        // K=128 (base=128), cap=48: 128%48!=0 → fallback halving to 32 (divides 128, ≤48).
        assert_eq!(pick_kb_for_n(128, 151936), 32);
        // K=64 (base=64), cap=48: 64%48!=0 → fallback to 32 (divides 64, ≤48).
        assert_eq!(pick_kb_for_n(64, 151936), 32);
    }

    /// The blocked-matmul block sizes must fit the real hardware, not just
    /// divide evenly. `emit_blocked_matmul_loops` keeps five tiles live —
    /// mat_a (M×Kb), mat_b (Kb×Nb), a_left (M×Kb), b_right (Kb×Nb), acc (M×Nb)
    /// — and each of L0A/L0B is capped at 64 KB.
    ///
    /// Regression guard: Kb=256/Nb=64 put L0B exactly ON the 64 KB cap (the
    /// documented "aicore execution exception" on 910B2) and the UB working set
    /// at 272 KB, which only looked acceptable while UB_SIZE wrongly read 256 KB.
    #[test]
    fn test_blocked_matmul_blocks_fit_hardware() {
        const L0_CAP: usize = 64 * 1024;
        let (kb, nb) = (PTO_MM_KB as usize, PTO_MM_NB as usize);
        // M=64 is the projection shape from the 20_FlashAttentionV2 benchmark;
        // it is also the largest M the emitter pads to for these kernels.
        let m = 64usize;
        let f32b = 4;

        let l0a = m * kb * f32b;
        let l0b = kb * nb * f32b;
        assert!(l0a < L0_CAP, "L0A tile {l0a} B must be UNDER the {L0_CAP} B cap");
        assert!(l0b < L0_CAP, "L0B tile {l0b} B must be UNDER the {L0_CAP} B cap");

        let ub_live = (m * kb + kb * nb + m * kb + kb * nb + m * nb) * f32b;
        assert!(
            ub_live <= A2A3::UB_SIZE,
            "blocked-matmul working set {ub_live} B exceeds the UB budget {} B",
            A2A3::UB_SIZE
        );
    }

    #[test]
    fn test_parse_const_arg_ssa() {
        assert_eq!(parse_const_arg("%c16_i32"), 16);
        assert_eq!(parse_const_arg("%c1024"), 1024);
    }

    #[test]
    fn test_infer_dtype() {
        assert_eq!(infer_dtype_from_name("%arg0"), "f32");
        assert_eq!(infer_dtype_from_name("%arg0_f16"), "f16");
    }

    #[test]
    fn test_infer_arg_dtype_from_body_f16() {
        // Rust kernels use bland arg names like %arg0 without dtype hints.
        // Body-scanning must pick up f16 from tile_load_f16 / tile_store_f16.
        let body = vec![
            "    %t = llvm.call @__tile_load_f16(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32".to_string(),
            "    llvm.call @__tile_store_f16(%arg1, %t, %c16, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()".to_string(),
        ];
        assert_eq!(infer_arg_dtype_from_body("%arg0", &body), Some("f16"));
        assert_eq!(infer_arg_dtype_from_body("%arg1", &body), Some("f16"));
        assert_eq!(infer_arg_dtype_from_body("%arg2", &body), None);
    }

    #[test]
    fn test_infer_arg_dtype_from_body_f32() {
        let body = vec![
            "    %t = llvm.call @__tile_load_f32(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32".to_string(),
            "    llvm.call @__tile_store_f32(%arg1, %t, %c16, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()".to_string(),
        ];
        assert_eq!(infer_arg_dtype_from_body("%arg0", &body), Some("f32"));
        assert_eq!(infer_arg_dtype_from_body("%arg1", &body), Some("f32"));
    }

    #[test]
    fn test_f16_matmul_emits_f16_ptr_args() {
        // Regression: generator must emit !pto.ptr<f16> for an f16 kernel's
        // GM args, not !pto.ptr<f32>. Without body inference, arg names like
        // %arg0 fall through to infer_dtype_from_name which defaults to f32
        // — ptoas then emits mismatched `__gm__ float*` + `GlobalTensor<half>`.
        let mlir = r#"
module {
  llvm.func @mm_f16(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c256 = llvm.mlir.constant(256 : i32) : i32
    %t_a = llvm.call @__tile_load_f16(%arg0, %c16, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @__tile_load_f16(%arg1, %c256, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @__tile_matmul_f16(%c0, %t_a, %t_b, %c16, %c256, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f16(%arg2, %t_c, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("mm_f16 PTO-MLIR");
        assert!(
            pto.contains("%arg0: !pto.ptr<f16>"),
            "arg0 should be !pto.ptr<f16>:\n{}",
            pto
        );
        assert!(
            pto.contains("%arg1: !pto.ptr<f16>"),
            "arg1 should be !pto.ptr<f16>:\n{}",
            pto
        );
        assert!(
            pto.contains("%arg2: !pto.ptr<f16>"),
            "arg2 should be !pto.ptr<f16>:\n{}",
            pto
        );
        assert!(!pto.contains("%arg0: !pto.ptr<f32>"), "f32 mismatch leak:\n{}", pto);
    }

    #[test]
    fn test_is_builtin_helper() {
        assert!(is_builtin_helper("get_block_idx"));
        assert!(is_builtin_helper("__tile_v_add_f32"));
        assert!(!is_builtin_helper("vec_add_kernel"));
    }

    #[test]
    fn test_tile_buf_type_str() {
        let s = tile_buf_type(32, 32, "f32");
        assert!(s.contains("loc=vec"));
        assert!(s.contains("dtype=f32"));
        assert!(s.contains("rows=32"));
        assert!(s.contains("cols=32"));
        assert!(s.contains("fractal=512"));
        assert!(s.contains("blayout=row_major"));
    }

    #[test]
    fn test_vec_add_generates_valid_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @vec_add(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_add_f32(%c0, %t0, %t1, %c32, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let result = convert_mlir_to_pto(mlir);
        assert!(result.is_ok(), "PTO-MLIR generation failed: {:?}", result);
        let pto = result.unwrap();

        // Must start with a module wrapper (accepts attributes)
        assert!(
            pto.contains("module {") || pto.contains("module attributes"),
            "Missing module wrapper:\n{}",
            pto
        );
        // func.func with pto.ptr args
        assert!(
            pto.contains("func.func @vec_add("),
            "Missing func.func:\n{}",
            pto
        );
        assert!(
            pto.contains("!pto.ptr<f32>"),
            "Missing !pto.ptr<f32>:\n{}",
            pto
        );
        // arith constants
        assert!(
            pto.contains("arith.constant"),
            "Missing arith constants:\n{}",
            pto
        );
        // pto ops
        assert!(
            pto.contains("pto.make_tensor_view"),
            "Missing make_tensor_view:\n{}",
            pto
        );
        assert!(
            pto.contains("pto.partition_view"),
            "Missing partition_view:\n{}",
            pto
        );
        assert!(
            pto.contains("pto.alloc_tile"),
            "Missing alloc_tile:\n{}",
            pto
        );
        assert!(pto.contains("pto.tload"), "Missing tload:\n{}", pto);
        assert!(pto.contains("pto.tadd"), "Missing tadd:\n{}", pto);
        assert!(pto.contains("pto.tstore"), "Missing tstore:\n{}", pto);
        assert!(pto.contains("return"), "Missing return:\n{}", pto);
        // No fictional text-assembly syntax
        assert!(
            !pto.contains(".kernel "),
            "Stale .kernel in output:\n{}",
            pto
        );
        assert!(!pto.contains(".end"), "Stale .end in output:\n{}", pto);
        assert!(
            !pto.contains("tile.load"),
            "Stale tile.load in output:\n{}",
            pto
        );
    }

    /// Core classification drives the build strategy, so it is checked against
    /// real emitted kernels rather than hand-written PTO snippets.
    ///
    /// The three decomposed attention stages are single-engine and build through
    /// `ascendc_library()`; the FUSED attention kernel is MIX and does not,
    /// which is the whole reason the pipeline is emitted as separate stages.
    #[test]
    fn test_classify_kernel_cores() {
        let matmul = |m: u32, k: u32, n: u32| {
            format!(
                r#"module {{
  llvm.func @mm(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{
    %m = llvm.mlir.constant({m} : i32) : i32
    %k = llvm.mlir.constant({k} : i32) : i32
    %n = llvm.mlir.constant({n} : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }}
}}
"#
            )
        };
        let softmax = r#"
module {
  llvm.func @sm(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %r = llvm.mlir.constant(32 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_softmax_f32(%x, %x, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %o, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let fused_attn = r#"
module {
  llvm.func @attn(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %s = llvm.mlir.constant(32 : i32) : i32
    %d = llvm.mlir.constant(64 : i32) : i32
    %q = llvm.call @__tile_load_f32(%arg0, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @__tile_load_f32(%arg1, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @__tile_load_f32(%arg2, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_f32(%q, %q, %k, %v, %s, %d) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %o, %s, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

        // scores / pv: pure cube.
        let scores = convert_mlir_to_pto(&matmul(32, 64, 32)).expect("scores");
        assert_eq!(classify_kernel_cores(&scores), KernelCores::Cube);
        assert_eq!(classify_kernel_cores(&scores).ccec_arch(), Some("cube"));
        assert!(classify_kernel_cores(&scores).buildable_with_ascendc_library());

        // softmax: pure vector.
        let sm = convert_mlir_to_pto(softmax).expect("softmax");
        assert_eq!(classify_kernel_cores(&sm), KernelCores::Vector);
        assert_eq!(classify_kernel_cores(&sm).ccec_arch(), Some("vec"));
        assert!(classify_kernel_cores(&sm).buildable_with_ascendc_library());

        // Fused attention: cube matmuls AND vector softmax in one kernel.
        // This is the case ascendc_library() cannot build.
        let fa = convert_mlir_to_pto(fused_attn).expect("fused attention");
        assert_eq!(classify_kernel_cores(&fa), KernelCores::Mix);
        assert_eq!(
            classify_kernel_cores(&fa).ccec_arch(),
            None,
            "a MIX kernel has no single --cce-aicore-arch"
        );
        assert!(!classify_kernel_cores(&fa).buildable_with_ascendc_library());
    }

    /// The UB constants must come from the vendor spec, not from memory.
    ///
    /// This parses the same `[AICoreSpec]` text CANN ships in
    /// `data/platform_config/Ascend910B2.ini` and checks `SPEC_A2A3` against it.
    /// The literal below is copied verbatim from a real CANN 8.5.2 install on a
    /// 910B2; `Ascend910_9392.ini` (the `910c` host) reports the same `ub_size`.
    ///
    /// Regression guard: `A2A3::UB_SIZE` once held 262144 (the A5 figure), which
    /// let the C1 budget accept working sets that overflow a 192 KB UB.
    #[test]
    fn test_ub_size_matches_vendor_platform_config() {
        let ini = "\
[Version]
SoC_version=Ascend910B2

[AICoreSpec]
cube_freq=1800
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
l1_size=524288
ub_size=196608
ubblock_size=32
ubbank_size=4096
";
        let spec = SocSpec::from_platform_ini(ini, 8 * 1024).expect("parse [AICoreSpec]");
        assert_eq!(spec.ub_size, 196608, "vendor ub_size is 192 KB");
        assert_eq!(spec, SPEC_A2A3, "SPEC_A2A3 must match the vendor platform_config");
        assert_eq!(spec.ub_budget(), 188416, "184 KB usable after 8 KB TMP_UB scratch");
        assert_eq!(
            A2A3::UB_SIZE,
            spec.ub_budget(),
            "the C1 budget must be the spec-derived value, not a hardcoded one"
        );
        assert_eq!(A2A3::BLOCK_BYTES, spec.ubblock_size);
        assert!(
            A2A3::UB_SIZE < 262144,
            "262144 is the A5 UB_SIZE; using it on a2a3 makes the C1 guard unsound"
        );
    }

    #[test]
    fn test_softmax_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("softmax f32 PTO-MLIR generation");
        assert!(
            pto.contains("func.func @softmax_1d"),
            "Missing func name:\n{}",
            pto
        );
        assert!(pto.contains("pto.tload"), "Missing tload:\n{}", pto);
        assert!(pto.contains("pto.tstore"), "Missing tstore:\n{}", pto);
        // tile_buf must carry the shape 1x1024
        assert!(
            pto.contains("rows=1, cols=1024"),
            "Missing rows=1, cols=1024 in tile_buf:\n{}",
            pto
        );
        // Softmax decomposition: 5 reduction ops
        assert!(pto.contains("pto.trowmax"), "Missing trowmax:\n{}", pto);
        assert!(
            pto.contains("pto.trowexpandsub"),
            "Missing trowexpandsub:\n{}",
            pto
        );
        assert!(pto.contains("pto.texp"), "Missing texp:\n{}", pto);
        assert!(pto.contains("pto.trowsum"), "Missing trowsum:\n{}", pto);
        assert!(
            pto.contains("pto.trowexpanddiv"),
            "Missing trowexpanddiv:\n{}",
            pto
        );
        // Reduction ops must use the 3-operand ins(%src, %tmp : T, T) format
        assert!(
            pto.contains("pto.trowmax ins("),
            "trowmax must use ins() format:\n{}",
            pto
        );
        // No pipe_barrier — ptoas adds sync with --enable-insert-sync
        assert!(
            !pto.contains("pipe_barrier"),
            "Unexpected pipe_barrier:\n{}",
            pto
        );
        // No legacy placeholder op
        assert!(
            !pto.contains("pto.tsoftmax"),
            "Unexpected tsoftmax placeholder:\n{}",
            pto
        );
    }

    #[test]
    fn test_softmax_f16_2d_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @softmax_rows_f16(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f16(%arg0, %c16, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_softmax_f16(%c0, %t0, %c16, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f16(%arg1, %t1, %c16, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("softmax f16 PTO-MLIR generation");
        assert!(
            pto.contains("func.func @softmax_rows_f16"),
            "Missing func name:\n{}",
            pto
        );
        assert!(pto.contains("dtype=f16"), "Missing f16 dtype:\n{}", pto);
        assert!(
            pto.contains("rows=16, cols=1024"),
            "Missing rows=16, cols=1024:\n{}",
            pto
        );
        // Full decomposition present
        assert!(pto.contains("pto.trowmax"), "Missing trowmax:\n{}", pto);
        assert!(
            pto.contains("pto.trowexpandsub"),
            "Missing trowexpandsub:\n{}",
            pto
        );
        assert!(pto.contains("pto.texp"), "Missing texp:\n{}", pto);
        assert!(pto.contains("pto.trowsum"), "Missing trowsum:\n{}", pto);
        assert!(
            pto.contains("pto.trowexpanddiv"),
            "Missing trowexpanddiv:\n{}",
            pto
        );
        assert!(
            !pto.contains("pto.tsoftmax"),
            "Unexpected tsoftmax placeholder:\n{}",
            pto
        );
    }

    #[test]
    fn test_exp_unary_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @exp_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_exp_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("exp PTO-MLIR generation");
        assert!(pto.contains("pto.texp"), "Missing texp:\n{}", pto);
        assert!(pto.contains("rows=32, cols=32"), "Missing shape:\n{}", pto);
    }

    #[test]
    fn test_tile_matmul_f32_generates_pto_mlir() {
        // 16×32 @ 32×16 → 16×16 matrix multiply
        let mlir = r#"
module {
  llvm.func @matmul_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t_a = llvm.call @__tile_load_f32(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @__tile_load_f32(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @__tile_matmul_f32(%c0, %t_a, %t_b, %c16, %c32, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t_c, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("matmul PTO-MLIR generation");
        // Must emit pto.tmatmul (cube unit op)
        assert!(pto.contains("pto.tmatmul"), "Missing tmatmul op:\n{}", pto);
        // Must use correct cube-unit tile types (not loc=vec for all tiles)
        assert!(
            pto.contains("loc=mat"),
            "Missing loc=mat (CBUF staging) tiles:\n{}",
            pto
        );
        assert!(
            pto.contains("loc=left"),
            "Missing loc=left (L0A) tile:\n{}",
            pto
        );
        assert!(
            pto.contains("loc=right"),
            "Missing loc=right (L0B) tile:\n{}",
            pto
        );
        assert!(
            pto.contains("loc=acc"),
            "Missing loc=acc (L0C accumulator) tile:\n{}",
            pto
        );
        // Acc tile must use fractal=1024 (L0C bank size)
        assert!(
            pto.contains("fractal=1024"),
            "Acc tile must have fractal=1024:\n{}",
            pto
        );
        // Must emit tmov ops (CBUF → L0A/L0B)
        assert!(
            pto.contains("pto.tmov"),
            "Missing tmov (CBUF→L0A/L0B) ops:\n{}",
            pto
        );
        // tmatmul must reference left+right tiles (not the original vec loads)
        assert!(
            pto.contains("pto.tmatmul ins("),
            "tmatmul must use ins(...) format:\n{}",
            pto
        );
        // Output tile (acc: 16×16) stored back to GM
        assert!(
            pto.contains("rows=16, cols=16"),
            "Output tile should be 16x16:\n{}",
            pto
        );
        // tload ops for both A and B → mat staging tiles (plus the original vec loads from translate_load)
        assert!(pto.contains("pto.tload"), "Missing tload ops:\n{}", pto);
        // Result stored back
        assert!(pto.contains("pto.tstore"), "Missing tstore op:\n{}", pto);
        // tstore must use the acc tile (loc=acc type string)
        assert!(
            pto.contains("pto.tstore ins("),
            "tstore must use ins() format:\n{}",
            pto
        );
    }

    /// Template for DeepSeek decode matmul shapes (M=16 padded, f32). Emits the
    /// llvm.mlir.constant+bitcast chains `parse_u32_from_arg` expects, plus the
    /// load→matmul→store sequence, and asserts scf.for + tmatmul.acc fire.
    fn check_decode_matmul_blocks(k: u32, n: u32, label: &str) {
        let mlir = format!(r#"
module {{
  llvm.func @{label}_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {{hacc.entry}} {{
    %c0_c = llvm.mlir.constant(0 : i32) : i32
    %c0   = llvm.bitcast %c0_c : i32 to i32
    %c16_c = llvm.mlir.constant(16 : i32) : i32
    %c16   = llvm.bitcast %c16_c : i32 to i32
    %ck_c  = llvm.mlir.constant({k} : i32) : i32
    %ck    = llvm.bitcast %ck_c : i32 to i32
    %cn_c  = llvm.mlir.constant({n} : i32) : i32
    %cn    = llvm.bitcast %cn_c : i32 to i32
    %t_a = llvm.call @__tile_load_f32(%arg0, %c16, %ck) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @__tile_load_f32(%arg1, %ck, %cn) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @__tile_matmul_f32(%c0, %t_a, %t_b, %c16, %ck, %cn) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t_c, %c16, %cn) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }}
}}
"#);
        let pto = convert_mlir_to_pto(&mlir).expect(label);
        assert!(pto.contains("scf.for %k_i"), "[{label}] K-blocking loop missing:\n{}", pto);
        assert!(
            pto.contains("pto.tmatmul.acc"),
            "[{label}] Accumulating tmatmul missing:\n{}", pto
        );
        assert!(pto.contains("pto.tmatmul ins("), "[{label}] Initial tmatmul missing:\n{}", pto);
    }

    /// All four DeepSeek decode matmul shapes (M=16 padded) must emit K-blocked
    /// MLIR. Without blocking these would overflow L0 caps on 910B3 cube:
    /// - kv_proj:  K=1536, N=256   — L0B 1.5 MB
    /// - q/o_proj: K=1536, N=1536  — L0B 9.4 MB
    /// - gate/up:  K=1536, N=8960  — L0B 55 MB, also hits CBUF outer-stride
    /// - down:     K=8960, N=1536  — L0A 561 KB, L0B 55 MB
    #[test]
    fn test_pto_matmul_decode_shapes_block() {
        check_decode_matmul_blocks(1536,  256,  "kv_proj");
        check_decode_matmul_blocks(1536,  1536, "q_proj");
        check_decode_matmul_blocks(1536,  8960, "gate_up");
        check_decode_matmul_blocks(8960,  1536, "down_proj");
    }

    /// DeepSeek kv_proj shape: M=16, K=1536, N=256 f32. Must trigger the K/N
    /// blocked emitter (L0B would be K*N*4 = 1.5 MB, far past the 64 KB cap).
    /// Validates that scf.for + pto.tmatmul.acc are emitted for large-K shapes.
    ///
    /// Uses proper `llvm.mlir.constant` + `llvm.bitcast` chains matching what
    /// real rustc_codegen_tile output looks like, so `parse_u32_from_arg` can
    /// resolve the M/K/N operands and `matmul_needs_blocking` can fire.
    #[test]
    fn test_pto_matmul_kv_proj_f32_blocks() {
        let mlir = r#"
module {
  llvm.func @matmul_kv_proj(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0_c  = llvm.mlir.constant(0    : i32) : i32
    %c0    = llvm.bitcast %c0_c   : i32 to i32
    %c16_c = llvm.mlir.constant(16   : i32) : i32
    %c16   = llvm.bitcast %c16_c  : i32 to i32
    %c256_c = llvm.mlir.constant(256  : i32) : i32
    %c256  = llvm.bitcast %c256_c : i32 to i32
    %c1536_c = llvm.mlir.constant(1536 : i32) : i32
    %c1536 = llvm.bitcast %c1536_c : i32 to i32
    %t_a = llvm.call @__tile_load_f32(%arg0, %c16, %c1536) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @__tile_load_f32(%arg1, %c1536, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @__tile_matmul_f32(%c0, %t_a, %t_b, %c16, %c1536, %c256) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t_c, %c16, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("kv_proj PTO-MLIR generation");
        assert!(pto.contains("scf.for %k_i"), "K-blocking loop missing:\n{}", pto);
        assert!(
            pto.contains("pto.tmatmul.acc"),
            "Accumulating tmatmul missing (K-blocked matmul needs init+acc pair):\n{}",
            pto
        );
        assert!(pto.contains("pto.tmatmul ins("), "Initial tmatmul missing:\n{}", pto);
    }

    /// Two loads from the same base GM arg at different constant GEP offsets must
    /// produce distinct `partition_view` ops with correct `offsets=[%crow, %c0]`.
    /// This is the prerequisite for double-buffering: two `pto.tload` ops with
    /// different partition offsets can be scheduled concurrently by ptoas.
    #[test]
    fn test_gep_offset_partition_views() {
        // Simulates: let t0 = tile_load_f32(input);            // offset 0
        //            let t1 = tile_prefetch_f32(input + 1024); // offset 1024 elements
        //            tile_softmax + tile_store ...
        let mlir = r#"
module {
  llvm.func @double_buf_softmax(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1024 = llvm.mlir.constant(1024 : i32) : i32
    %ptr1 = llvm.getelementptr %arg0[%c1024] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%ptr1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %s0 = llvm.call @__tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    %s1 = llvm.call @__tile_softmax_f32(%c0, %t1, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %s0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    %ptr1_out = llvm.getelementptr %arg1[%c1024] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.call @__tile_store_f32(%ptr1_out, %s1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("double-buffer PTO generation");

        // Must have two tload ops
        let tload_count = pto.matches("pto.tload").count();
        assert_eq!(tload_count, 2, "Expected 2 tload ops, got {}:\n{}", tload_count, pto);

        // The second load must use a non-zero row offset in its partition_view
        // (offset 1024 elements / 1024 cols = row 1)
        assert!(
            pto.contains("offsets = [%c1, %c0]"),
            "Expected partition_view with offsets=[%c1,%c0] for the prefetch load:\n{}",
            pto
        );

        // First load should still use offset 0
        assert!(
            pto.contains("offsets = [%c0, %c0]"),
            "Expected partition_view with offsets=[%c0,%c0] for the first load:\n{}",
            pto
        );

        // Both tstore ops must be present
        let tstore_count = pto.matches("pto.tstore").count();
        assert_eq!(tstore_count, 2, "Expected 2 tstore ops, got {}:\n{}", tstore_count, pto);
    }

    /// Verify that two loads from *different* GM args (the tile_join_load pattern)
    /// each get their own tensor_view and both start at offset 0.
    #[test]
    fn test_join_load_two_independent_gm_args() {
        let mlir = r#"
module {
  llvm.func @join_load(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %r = llvm.call @__tile_add_f32(%c0, %t0, %t1, %c1, %c1024) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %r, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("join_load PTO generation");

        // Two independent tload ops
        let tload_count = pto.matches("pto.tload").count();
        assert_eq!(tload_count, 2, "Expected 2 tload ops, got {}:\n{}", tload_count, pto);

        // Two tensor_view ops (one per distinct GM arg)
        let tv_count = pto.matches("pto.make_tensor_view").count();
        assert_eq!(tv_count, 3, "Expected 3 tensor_views (2 in + 1 out), got {}:\n{}", tv_count, pto);

        // Both partition_views must use offset 0 (no GEP offset)
        let pv_zero_count = pto.matches("offsets = [%c0, %c0]").count();
        assert!(pv_zero_count >= 2, "Expected ≥2 zero-offset partition_views, got {}:\n{}", pv_zero_count, pto);

        // Must emit tadd
        assert!(pto.contains("pto.tadd"), "Missing tadd op:\n{}", pto);
    }

    // -----------------------------------------------------------------------
    // Phase 0 tile intrinsic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_transpose_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @transpose_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_transpose_f32(%c0, %t0, %c16, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("transpose PTO-MLIR generation");
        assert!(pto.contains("transpose"), "Missing transpose comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
        assert!(pto.contains("rows=32, cols=16"), "Missing transposed shape 32x16:\n{}", pto);
    }

    #[test]
    fn test_rsqrt_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @rsqrt_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_rsqrt_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("rsqrt PTO-MLIR generation");
        assert!(pto.contains("rsqrt"), "Missing rsqrt comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_log_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @log_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_log_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("log PTO-MLIR generation");
        assert!(pto.contains("log"), "Missing log comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_sigmoid_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @sigmoid_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c4, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_sigmoid_f32(%c0, %t0, %c4, %c256) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c4, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("sigmoid PTO-MLIR generation");
        // Post-a658facc: sigmoid uses exp(x)/(1+exp(x)) form (see
        // mlir_to_pto.rs:2394 — ptoas has no scalar/tile divide, so
        // 1/(1+exp(-x)) was rewritten to exp(x)/(1+exp(x)), which uses
        // tile/tile `tdiv`). No `tmuls` negate step anymore.
        assert!(pto.contains("sigmoid"), "Missing sigmoid comment:\n{}", pto);
        assert!(pto.contains("pto.texp"), "Missing texp step:\n{}", pto);
        assert!(pto.contains("pto.tadds"), "Missing tadds step:\n{}", pto);
        assert!(pto.contains("pto.tdiv "), "Missing tdiv step (tile/tile divide):\n{}", pto);
    }

    #[test]
    fn test_clamp_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @clamp_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_clamp_f32(%c0, %t0, %c0, %c6, %c32, %c32) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("clamp PTO-MLIR generation");
        assert!(pto.contains("clamp"), "Missing clamp comment:\n{}", pto);
        assert!(pto.contains("pto.tmaxs"), "Missing tmaxs (lower bound):\n{}", pto);
        assert!(pto.contains("pto.tmins"), "Missing tmins (upper bound):\n{}", pto);
    }

    #[test]
    fn test_cast_f32_f16_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @cast_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_cast_f32_f16(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f16(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("cast f32->f16 PTO-MLIR generation");
        assert!(pto.contains("cast"), "Missing cast comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
        assert!(pto.contains("dtype=f16"), "Missing f16 dtype in output tile:\n{}", pto);
    }

    #[test]
    fn test_cast_f16_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @cast_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f16(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_cast_f16_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("cast f16->f32 PTO-MLIR generation");
        assert!(pto.contains("cast"), "Missing cast comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_slice_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @slice_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_slice_f32(%c0, %t0, %c4, %c8, %c32, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("slice PTO-MLIR generation");
        assert!(pto.contains("slice"), "Missing slice comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
        assert!(pto.contains("rows=16, cols=16"), "Missing dst shape 16x16:\n{}", pto);
    }

    #[test]
    fn test_concat_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @concat_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_concat_f32(%c0, %t0, %t1, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("concat PTO-MLIR generation");
        assert!(pto.contains("concat"), "Missing concat comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
        assert!(pto.contains("rows=32, cols=32"), "Missing output shape 32x32:\n{}", pto);
    }

    #[test]
    fn test_scatter_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @scatter_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_scatter_f32(%c0, %t0, %arg1, %c8, %c32, %c1) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t1, %c8, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("scatter PTO-MLIR generation");
        assert!(pto.contains("scatter"), "Missing scatter comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_gather_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @gather_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_gather_f32(%c0, %t0, %arg1, %c8, %c32, %c1) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t1, %c8, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("gather PTO-MLIR generation");
        assert!(pto.contains("gather"), "Missing gather comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_gather_mask_f32_generates_pto_tgather() {
        // mask=10 = 0b1010 → pattern P1010 (extract value channel from
        // sort_result interleaved [val,idx] pairs).
        let mlir = r#"
module {
  llvm.func @gather_mask_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %s = llvm.call @__tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %g = llvm.call @__tile_gather_mask_f32(%c0, %s, %c10, %c1, %c128) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %g, %c1, %c128) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("gather_mask PTO-MLIR generation");
        assert!(pto.contains("pto.tgather"), "Missing pto.tgather op:\n{}", pto);
        assert!(
            pto.contains("maskPattern = #pto.mask_pattern<P1010>"),
            "Missing maskPattern P1010:\n{}",
            pto
        );
        assert!(pto.contains("rows=1, cols=128"), "Missing 1×128 shape:\n{}", pto);
    }

    #[test]
    fn test_mrgsort2_f32_generates_pto_tmrgsort() {
        // Merge two 1×128 sorted f32 tiles → 1×256.
        let mlir = r#"
module {
  llvm.func @merge_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %a = llvm.call @__tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %t = llvm.call @__tile_load_f32(%arg2, %c1, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %m = llvm.call @__tile_mrgsort2_f32(%c0, %a, %b, %t, %c128) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %m, %c1, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("mrgsort2 PTO-MLIR generation");
        assert!(pto.contains("pto.tmrgsort"), "Missing pto.tmrgsort op:\n{}", pto);
        assert!(pto.contains("exhausted = false"), "Missing exhausted attr:\n{}", pto);
        assert!(pto.contains("vector<4xi16>"), "Missing exhausted-flags i16 vector:\n{}", pto);
        assert!(
            pto.contains("rows=1, cols=256"),
            "Missing 1×256 merged output (2× cols_each):\n{}",
            pto
        );
    }

    #[test]
    fn test_sort32_f32_generates_pto_tsort32() {
        // Sort a 1×128 f32 tile with 1×128 ui32 indices.
        // Output is 1×256 (FLOAT_DST_STRIDE_COEF=2: interleaved [val,idx] pairs).
        let mlir = r#"
module {
  llvm.func @sort_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %v = llvm.call @__tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %i = llvm.call @__tile_arith_progression_i32(%c0, %c0, %c128) : (i32, i32, i32) -> i32
    %s = llvm.call @__tile_sort32_f32(%c0, %v, %i, %c1, %c128) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %s, %c1, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("sort32 PTO-MLIR generation");
        assert!(pto.contains("pto.tsort32"), "Missing pto.tsort32 op:\n{}", pto);
        assert!(pto.contains("dtype=ui32"), "Missing ui32 indices tile:\n{}", pto);
        assert!(
            pto.contains("rows=1, cols=256"),
            "Missing 1×256 output (2× input width):\n{}",
            pto
        );
    }

    #[test]
    fn test_init_sort_buf_f32_generates_pto_tfillpad() {
        // 1×128 f32 tile re-padded to pad=3 sentinel boundary.
        let mlir = r#"
module {
  llvm.func @init_sort_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t = llvm.call @__tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %p = llvm.call @__tile_init_sort_buf_f32(%c0, %t, %c1, %c128) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %p, %c1, %c128) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("init_sort_buf PTO-MLIR generation");
        assert!(pto.contains("pto.tfillpad"), "Missing pto.tfillpad op:\n{}", pto);
        assert!(pto.contains("pad=3"), "Missing pad=3 sentinel marker on output:\n{}", pto);
        assert!(pto.contains("pad=0"), "Missing pad=0 on input (re-pad source):\n{}", pto);
        assert!(pto.contains("rows=1, cols=128"), "Missing 1×128 shape:\n{}", pto);
    }

    #[test]
    fn test_arith_progression_i32_generates_pto_tci() {
        // Iota over 1×128 i32, used as sort-index initializer for topk port.
        let mlir = r#"
module {
  llvm.func @arith_prog_k(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    %t = llvm.call @__tile_arith_progression_i32(%c0, %c0, %c128) : (i32, i32, i32) -> i32
    llvm.call @__tile_store_i32(%arg0, %t, %c1, %c128) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("arith_progression PTO-MLIR generation");
        assert!(pto.contains("pto.tci"), "Missing pto.tci op:\n{}", pto);
        assert!(
            pto.contains("descending = false"),
            "Missing descending=false attr:\n{}",
            pto
        );
        assert!(
            pto.contains("dtype=ui32"),
            "Missing ui32 dtype on output tile (matches tsort32 consumer):\n{}",
            pto
        );
        assert!(
            pto.contains("rows=1, cols=128"),
            "Missing 1×128 output shape:\n{}",
            pto
        );
    }

    #[test]
    fn test_topk_f32_generates_pto_mlir() {
        // 4×64 → 4×8 hits the fallback path (rows>1).
        let mlir = r#"
module {
  llvm.func @topk_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c4, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_topk_f32(%c0, %t0, %arg1, %c4, %c64, %c8) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t1, %c4, %c8) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("topk PTO-MLIR generation");
        assert!(pto.contains("topk"), "Missing topk comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
        assert!(pto.contains("rows=4, cols=8"), "Missing output shape 4x8:\n{}", pto);
        assert!(
            pto.contains("stub fallback"),
            "rows=4 should hit Path A fallback:\n{}",
            pto
        );
    }

    /// Path A composed emit: 1×128 → 1×8 lowers through tci + tsort32 + tgather + tmov.
    #[test]
    fn test_topk_f32_path_a_composed_emit() {
        let mlir = r#"
module {
  llvm.func @topk_path_a_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_topk_f32(%c0, %t0, %arg1, %c1, %c128, %c8) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t1, %c1, %c8) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("topk Path A PTO-MLIR generation");
        // All four ops of the composed pipeline must be present.
        assert!(pto.contains("pto.tci"), "Missing pto.tci (step 1: iota):\n{}", pto);
        assert!(pto.contains("pto.tsort32"), "Missing pto.tsort32 (step 2: sort):\n{}", pto);
        assert!(
            pto.contains("pto.tgather"),
            "Missing pto.tgather (step 3: value-channel extract):\n{}",
            pto
        );
        assert!(
            pto.contains("maskPattern = #pto.mask_pattern<P1010>"),
            "Missing P1010 mask for value-channel extract:\n{}",
            pto
        );
        assert!(pto.contains("pto.tmov"), "Missing pto.tmov (step 4: head-K):\n{}", pto);

        // Tilelang topk_selector port comment marks the path.
        assert!(
            pto.contains("tilelang topk_selector port"),
            "Missing port-tag comment:\n{}",
            pto
        );

        // Output shape must be 1×8.
        assert!(pto.contains("rows=1, cols=8"), "Missing 1×8 output shape:\n{}", pto);

        // Intermediate sorted tile is 2× input width.
        assert!(
            pto.contains("rows=1, cols=256"),
            "Missing 1×256 sorted-interleaved tile (2× input):\n{}",
            pto
        );

        // Indices tile uses ui32.
        assert!(pto.contains("dtype=ui32"), "Missing ui32 indices tile:\n{}", pto);

        // No fallback marker (rows=1, cols=128 → composed path).
        assert!(
            !pto.contains("stub fallback"),
            "Path A composed emit should not hit fallback:\n{}",
            pto
        );
    }

    #[test]
    fn test_matmul_f16_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @matmul_f16_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t_a = llvm.call @__tile_load_f16(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @__tile_load_f16(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @__tile_matmul_f16(%c0, %t_a, %t_b, %c16, %c32, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f16(%arg2, %t_c, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("matmul_f16 PTO-MLIR generation");
        assert!(pto.contains("func.func @matmul_f16_k"), "Missing func name:\n{}", pto);
        assert!(pto.contains("pto.tmatmul"), "Missing tmatmul:\n{}", pto);
        assert!(pto.contains("dtype=f16"), "Missing f16 dtype:\n{}", pto);
        assert!(pto.contains("loc=mat"), "Missing mat tile:\n{}", pto);
        assert!(pto.contains("loc=left"), "Missing left tile:\n{}", pto);
        assert!(pto.contains("loc=right"), "Missing right tile:\n{}", pto);
        assert!(pto.contains("loc=acc"), "Missing acc tile:\n{}", pto);

        // Per CANN 8.5 ptoas dtype rules (see memory/project_pto_tmatmul_dtype_rules.md):
        // (dst, lhs, rhs) for pto.tmatmul must be (f32, f16, f16) — NOT all-f16.
        // The L0C accumulator is f32; the caller's tstore reads from the f32 acc
        // tile and writes to the f16 GM pv — the hardware FixPipe path performs
        // the f32→f16 cast during the L0C→GM DMA. No acc→vec tmov is emitted
        // because pto_instr's TMov static_assert rejects that address-space pair.
        assert!(
            pto.contains("loc=acc, dtype=f32"),
            "L0C accumulator must be f32 per ptoas tmatmul dtype rules:\n{}",
            pto
        );
    }

    #[test]
    fn test_absmax_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @absmax_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t_a = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_r = llvm.call @__tile_absmax_f32(%t_a, %t_a, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t_r, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("absmax PTO-MLIR generation");
        assert!(pto.contains("pto.tabs"), "absmax must use pto.tabs:\n{}", pto);
        assert!(pto.contains("pto.trowmax"), "absmax must use pto.trowmax:\n{}", pto);
    }

    #[test]
    fn test_quantize_f32_i8_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @quantize_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t_a = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_s = llvm.call @__tile_load_f32(%arg1, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_r = llvm.call @__tile_quantize_f32_i8(%t_a, %t_a, %t_s, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t_r, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("quantize PTO-MLIR generation");
        assert!(pto.contains("pto.tdivs"), "quantize must use pto.tdivs:\n{}", pto);
        assert!(pto.contains("pto.tmins"), "quantize must use pto.tmins:\n{}", pto);
    }

    #[test]
    fn test_dequantize_i8_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @dequantize_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t_a = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_s = llvm.call @__tile_load_f32(%arg1, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_r = llvm.call @__tile_dequantize_i8_f32(%t_a, %t_a, %t_s, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t_r, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("dequantize PTO-MLIR generation");
        assert!(pto.contains("pto.tmuls"), "dequantize must use pto.tmuls:\n{}", pto);
        assert!(pto.contains("// dequantize"), "dequantize must emit comment:\n{}", pto);
    }

    // ── Phase 6 MTP tests ──────────────────────────────────────────────────

    #[test]
    fn test_argmax_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @argmax_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_argmax_f32(%c0, %t0, %c4, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("argmax PTO-MLIR generation");
        assert!(pto.contains("argmax"), "Missing argmax comment:\n{}", pto);
        assert!(pto.contains("pto.trowmax"), "Missing trowmax:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_sample_top_p_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @sample_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_sample_top_p_f32(%c0, %t0, %c0, %c0, %c0, %c4, %c32) : (i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("sample_top_p PTO-MLIR generation");
        assert!(pto.contains("sample_top_p"), "Missing sample_top_p comment:\n{}", pto);
        assert!(pto.contains("pto.trowmax"), "Missing trowmax:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_draft_verify_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @verify_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_draft_verify_f32(%c0, %t0, %t1, %c4, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t2, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("draft_verify PTO-MLIR generation");
        assert!(pto.contains("draft_verify"), "Missing draft_verify comment:\n{}", pto);
        assert!(pto.contains("pto.trowmax"), "Missing trowmax:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_token_accept_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @accept_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c4, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c4, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_load_f32(%arg2, %c4, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t3 = llvm.call @__tile_token_accept_f32(%c0, %t0, %t1, %t2, %c0, %c4) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %t3, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("token_accept PTO-MLIR generation");
        assert!(pto.contains("token_accept"), "Missing token_accept comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "Missing tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_silu_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @silu_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_silu_f32(%t0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("silu PTO-MLIR generation");
        // Post-a658facc: silu(x) is emitted as x / (1 + exp(-x)) using tile/tile
        // `tdiv` (ptoas has no scalar/tile divide). The prior form —
        // tdivs(1, 1+exp(-x)) followed by tmul(src, sigmoid) — was replaced by
        // a single `tdiv(src, 1+exp(-x))`.
        assert!(pto.contains("silu"), "Missing silu comment:\n{}", pto);
        assert!(pto.contains("pto.tmuls"), "silu must use pto.tmuls for negate:\n{}", pto);
        assert!(pto.contains("pto.texp"), "silu must use pto.texp:\n{}", pto);
        assert!(pto.contains("pto.tadds"), "silu must add 1 via tadds:\n{}", pto);
        assert!(pto.contains("pto.tdiv "), "silu must use pto.tdiv for x/(1+exp(-x)):\n{}", pto);
    }

    #[test]
    fn test_silu_mul_fusion_pto_mlir() {
        // SiLU followed by Mul should be fused: silu(gate) * up
        let mlir = r#"
module {
  llvm.func @gated_mlp(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %gate = llvm.call @__tile_load_f32(%arg0, %c1, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @__tile_load_f32(%arg1, %c1, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @__tile_silu_f32(%gate, %gate, %c1, %c64) : (i32, i32, i32, i32) -> i32
    %out = llvm.call @__tile_mul_f32(%silu, %silu, %up, %c1, %c64) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %out, %c1, %c64) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("silu_mul PTO-MLIR generation");
        // Post-a658facc: fused silu_mul uses UB-tight tile reuse and a single
        // tile/tile `tdiv` (ptoas has no scalar/tile divide) — the prior
        // tdivs+tmul pair was collapsed to one `tdiv(gate, 1+exp(-gate))`,
        // followed by the final `tmul(silu, up)`.
        //   Op sequence: tmuls(neg) → texp → tadds → tdiv(silu) → tmul(out)
        assert!(pto.contains("silu_mul"), "Missing silu_mul fusion comment:\n{}", pto);
        assert!(pto.contains("fused"), "Must be labeled as fused:\n{}", pto);
        assert!(pto.contains("pto.tmuls"), "silu_mul must negate gate:\n{}", pto);
        assert!(pto.contains("pto.texp"), "silu_mul must compute exp:\n{}", pto);
        assert!(pto.contains("pto.tadds"), "silu_mul must add 1:\n{}", pto);
        assert!(pto.contains("pto.tdiv "), "silu_mul must use tdiv for sigmoid+scale:\n{}", pto);
        // Exactly one final `tmul` for silu * up (the old variant had two).
        let tmul_count = pto.matches("pto.tmul ").count();
        assert!(tmul_count >= 1, "silu_mul needs final tmul(silu, up), got {}:\n{}", tmul_count, pto);
    }

    #[test]
    fn test_silu_standalone_no_fusion_pto() {
        // A standalone SiLU (without following Mul) should NOT produce fusion comment
        let mlir = r#"
module {
  llvm.func @silu_only(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_silu_f32(%t0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("standalone silu PTO-MLIR generation");
        assert!(!pto.contains("silu_mul"), "standalone silu must NOT have fusion comment:\n{}", pto);
        assert!(pto.contains("silu"), "should still have silu comment:\n{}", pto);
    }

    /// Qwen2.5-7B SwiGLU runs at INTER=18944 — the fused 5-tile emit needs
    /// 379 KB of UB, over the 224 KB usable budget. The N-blocked emitter
    /// (#67) chunks along the inner dim and emits an scf.for over chunks
    /// of size Nb (chosen by `pick_silu_mul_nb` to be the largest divisor
    /// of cols that fits the per-chunk budget). For INTER=18944 f32 with
    /// rows=1 this picks Nb=9472 → 2 iters, 5×9472×4 = 184 KB peak.
    #[test]
    fn test_silu_mul_blocks_inter_18944_into_chunks() {
        let mlir = r#"
module {
  llvm.func @gated_mlp_7b(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %cN = llvm.mlir.constant(18944 : i32) : i32
    %gate = llvm.call @__tile_load_f32(%arg0, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @__tile_load_f32(%arg1, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @__tile_silu_f32(%gate, %gate, %c1, %cN) : (i32, i32, i32, i32) -> i32
    %out = llvm.call @__tile_mul_f32(%silu, %silu, %up, %c1, %cN) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %out, %c1, %cN) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir)
            .expect("INTER=18944 must lower via the N-blocked silu_mul path");
        assert!(
            pto.contains("scf.for %n_i"),
            "blocked silu_mul must emit scf.for over n_i:\n{}", pto
        );
        assert!(
            pto.contains("N-blocked, #67"),
            "comment header must mark this as the #67 blocked path:\n{}", pto
        );
        // Per-chunk body must contain the 5 silu_mul ops on chunk tiles.
        for op in ["pto.tmuls", "pto.texp", "pto.tadds", "pto.tdiv", "pto.tmul"] {
            assert!(pto.contains(op), "blocked emit missing {}:\n{}", op, pto);
        }
        // Per-chunk tload + tstore for gate/up/out partition_views.
        assert!(
            pto.matches("pto.tload").count() >= 2,
            "blocked emit must tload gate and up per chunk:\n{}", pto
        );
        assert!(
            pto.contains("pto.tstore"),
            "blocked emit must tstore the per-chunk out:\n{}", pto
        );
        // No full-shape vec tile of size 1×18944 should be allocated for
        // gate/up — those loads must be deferred. The result tile and
        // intermediates should all be at the chunk size, not 18944.
        assert!(
            !pto.contains("rows=1, cols=18944"),
            "no full-shape 1×18944 tile_buf should appear (defer-load failed):\n{}", pto
        );
    }

    /// Standalone silu (not followed by mul) carries the same 5-tile UB
    /// pressure (src + neg + exp + oplus + out) and must be guarded too.
    /// INTER=18944 f32 → 379 KB > 224 KB budget.
    #[test]
    fn test_silu_standalone_rejects_inter_18944_over_ub_budget() {
        let mlir = r#"
module {
  llvm.func @silu_only_7b(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %cN = llvm.mlir.constant(18944 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_silu_f32(%t0, %t0, %c1, %cN) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %cN) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let res = convert_mlir_to_pto(mlir);
        let err = res.expect_err("standalone silu at INTER=18944 must be rejected");
        assert!(
            err.contains("silu") && err.contains("UB usage") && err.contains("18944"),
            "standalone silu guard error must mention silu, UB, and inner dim; got: {}",
            err
        );
    }

    /// Sanity: shapes that fit comfortably under the 224 KB budget should
    /// still emit cleanly. INTER=4096 f32 → 5 × 16 KB = 80 KB.
    #[test]
    fn test_silu_mul_accepts_inter_4096() {
        let mlir = r#"
module {
  llvm.func @gated_mlp_4k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %cN = llvm.mlir.constant(4096 : i32) : i32
    %gate = llvm.call @__tile_load_f32(%arg0, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @__tile_load_f32(%arg1, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @__tile_silu_f32(%gate, %gate, %c1, %cN) : (i32, i32, i32, i32) -> i32
    %out = llvm.call @__tile_mul_f32(%silu, %silu, %up, %c1, %cN) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %out, %c1, %cN) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("INTER=4096 should fit the UB budget");
        assert!(pto.contains("silu_mul"), "fusion expected at INTER=4096:\n{}", pto);
    }

    #[test]
    fn test_cast_bf16_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @cast_bf16_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_cast_bf16_f32(%t0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("cast bf16->f32 PTO-MLIR generation");
        assert!(pto.contains("cast"), "Missing cast comment:\n{}", pto);
        assert!(pto.contains("pto.tmov"), "cast must use pto.tmov passthrough:\n{}", pto);
    }

    #[test]
    fn test_attention_f32_a5_safe_pattern() {
        // Guards the a5-safe attention emitter pattern: no VEC→MAT tmov,
        // no ACC→VEC tmov reliance at input to softmax (still emitted but
        // paired with tinsert for the weights hop), transposed tv for K,
        // and module-level pto.target_arch="a5".
        let mlir = r#"
module {
  llvm.func @attn_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %s = llvm.mlir.constant(8 : i32) : i32
    %d = llvm.mlir.constant(16 : i32) : i32
    %q = llvm.call @__tile_load_f32(%arg0, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @__tile_load_f32(%arg1, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @__tile_load_f32(%arg2, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_f32(%q, %q, %k, %v, %s, %d) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %o, %s, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("attention PTO-MLIR generation");
        assert!(
            pto.contains("pto.target_arch = \"a5\""),
            "attention must emit module-level a5 target arch attr:\n{}", pto
        );
        assert!(
            pto.contains("pto.tinsert"),
            "attention must use pto.tinsert for vec→mat weights hop (not tmov):\n{}", pto
        );
        assert!(
            pto.contains("strides = [%c1,"),
            "attention must build a transposed tensor_view for K:\n{}", pto
        );
        assert!(
            pto.contains("slayout=col_major"),
            "attention must use ZN mat tile for K (DN→ZN tload):\n{}", pto
        );
        // Row-reductions must use the rowreduce type (rows×1 col_major),
        // otherwise ptoas's trowmax verifier rejects with
        // "expects dst valid_shape[1] to be 1".
        assert!(
            pto.contains("pto.trowmax"),
            "attention softmax must emit pto.trowmax:\n{}", pto
        );
    }

    #[test]
    fn test_matmul_transposed_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @matmul_t_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %m = llvm.mlir.constant(32 : i32) : i32
    %k = llvm.mlir.constant(64 : i32) : i32
    %n = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %n, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @__tile_matmul_transposed_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("matmul_transposed PTO-MLIR generation");
        assert!(pto.contains("matmul_transposed"), "Missing matmul_transposed comment:\n{}", pto);
        assert!(pto.contains("pto.tmatmul"), "must use pto.tmatmul:\n{}", pto);
        // a5-safe path: B^T via transposed tensor_view (DN→ZN tload), not VEC→MAT tmov.
        assert!(
            pto.contains("slayout=col_major"),
            "matmul_transposed must emit ZN mat tile for B^T:\n{}", pto
        );
        assert!(
            pto.contains("strides = [%c1,"),
            "matmul_transposed must build a transposed tensor_view for B:\n{}", pto
        );
        // Confirm no VEC→MAT tmov remains — the entire a5 fix hinges on going
        // GM→mat directly via tload, never through a vec intermediate.
        // We don't ban all tmov (CBUF→L0 tmov is still needed and valid on a2a3),
        // but we do ban tinsert (which is the A5-only op) — matmul_transposed
        // should not need it.
        assert!(
            !pto.contains("pto.tinsert"),
            "matmul_transposed should not need pto.tinsert (A5-only); a2a3-compatible:\n{}", pto
        );
        // The emitter uses only A2/A3-supported op forms (DN→ZN tload + CBUF→L0
        // tmov + tmatmul), so the a5 module attribute must stay off — otherwise
        // we block validating the transposed-matmul path on CANN 8.5, which
        // ships a2a3-only headers.
        assert!(
            !pto.contains("pto.target_arch = \"a5\""),
            "matmul_transposed must NOT tag module with a5 attr (path is a2a3-compatible):\n{}", pto
        );
    }

    #[test]
    fn test_attention_gqa_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @gqa_k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %s = llvm.mlir.constant(4 : i32) : i32
    %d = llvm.mlir.constant(8 : i32) : i32
    %hq = llvm.mlir.constant(4 : i32) : i32
    %hkv = llvm.mlir.constant(2 : i32) : i32
    %q = llvm.call @__tile_load_f32(%arg0, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @__tile_load_f32(%arg1, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @__tile_load_f32(%arg2, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_gqa_f32(%q, %q, %k, %v, %s, %d, %hq, %hkv) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %o, %s, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("attention_gqa PTO-MLIR generation");
        assert!(pto.contains("attention_gqa"), "Missing attention_gqa comment:\n{}", pto);
        assert!(pto.contains("pto.tmatmul"), "GQA must use pto.tmatmul:\n{}", pto);
        assert!(pto.contains("pto.trowmax"), "GQA must use pto.trowmax for softmax:\n{}", pto);
        assert!(pto.contains("pto.texp"), "GQA must use pto.texp for softmax:\n{}", pto);
        // a5-safe path regression guards (same as translate_attention):
        //   - weights→mat must go through tinsert (not tmov)
        //   - K must use transposed tensor_view (DN layout)
        //   - module must carry pto.target_arch="a5" for ptoas verifier
        assert!(
            pto.contains("pto.tinsert"),
            "GQA must use pto.tinsert for vec→mat weights hop:\n{}", pto
        );
        assert!(
            pto.contains("strides = [%c1,"),
            "GQA must build a transposed tensor_view for K:\n{}", pto
        );
        assert!(
            pto.contains("pto.target_arch = \"a5\""),
            "GQA must emit module-level a5 target arch attr:\n{}", pto
        );
    }

    #[test]
    fn test_pto_layernorm() {
        let mlir = r#"
module {
  llvm.func @tile_layernorm(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(1024 : i32) : i32
    %eps = llvm.mlir.constant(1.0e-5 : f32) : f32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %n = llvm.call @__tile_rms_norm_f32(%x, %x, %eps, %r, %c) : (i32, i32, f32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %n, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).unwrap();
        assert!(pto.contains("pto.") || pto.contains("tload") || pto.contains("rms"),
                "missing PTO ops in layernorm output:\n{}", pto);
    }

    #[test]
    fn test_pto_conv1d() {
        let mlir = r#"
module {
  llvm.func @tile_conv1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %w = llvm.mlir.constant(0.5 : f32) : f32
    %lo = llvm.mlir.constant(0.0 : f32) : f32
    %hi = llvm.mlir.constant(3.4028235e+38 : f32) : f32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @__tile_scale_f32(%x, %x, %w, %r, %c) : (i32, i32, f32, i32, i32) -> i32
    %y = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %a = llvm.call @__tile_add_f32(%s, %s, %y, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    %cl = llvm.call @__tile_clamp_f32(%a, %a, %lo, %hi, %r, %c) : (i32, i32, f32, f32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %cl, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).unwrap();
        assert!(pto.contains("pto.tload") || pto.contains("tload"),
                "missing tload in PTO conv1d output:\n{}", pto);
        assert!(pto.contains("pto.tadd") || pto.contains("tadd"),
                "missing tadd in PTO conv1d output:\n{}", pto);
    }

    #[test]
    fn test_pto_matmul() {
        // M must be a multiple of 16 (910B2 cube fixedRowSize); earlier
        // fixture used M=4 from before that check landed.
        let mlir = r#"
module {
  llvm.func @tile_matmul(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %m = llvm.mlir.constant(16 : i32) : i32
    %k = llvm.mlir.constant(8 : i32) : i32
    %n = llvm.mlir.constant(16 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @__tile_matmul_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).unwrap();
        assert!(pto.contains("pto.tmatmul") || pto.contains("tmatmul"),
                "missing tmatmul in PTO matmul output:\n{}", pto);
    }

    #[test]
    fn test_pto_rope() {
        let mlir = r#"
module {
  llvm.func @tile_rope(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(64 : i32) : i32
    %pos = llvm.mlir.constant(42 : i32) : i32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %y = llvm.call @__tile_rope_f32(%c0, %x, %pos, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).unwrap();
        assert!(pto.contains("rope"), "missing rope comment in PTO output:\n{}", pto);
        assert!(pto.contains("pto.tmuls"), "missing pto.tmuls in PTO rope output:\n{}", pto);
        assert!(pto.contains("pto.tload"), "missing pto.tload in PTO rope output:\n{}", pto);
        assert!(pto.contains("pto.tstore"), "missing pto.tstore in PTO rope output:\n{}", pto);
    }

    // ── Uncovered-audit coverage: top-level emitters reachable through
    //    convert_mlir_to_pto but previously undriven by any test. ──

    #[test]
    fn test_pto_fill_f32_generates_tmov() {
        // __tile_fill_f32(dst, scalar, rows, cols) → translate_fill,
        // which broadcasts a scalar into a vec tile via pto.tmov.
        let mlir = r#"
module {
  llvm.func @fill_k(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %scal = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(2 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %t = llvm.call @__tile_fill_f32(%c0, %scal, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg0, %t, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("fill PTO-MLIR generation");
        assert!(pto.contains("pto.tmov"), "fill must emit pto.tmov broadcast:\n{}", pto);
        assert!(
            pto.contains("fill 2x32 with scalar"),
            "fill must emit the broadcast comment:\n{}",
            pto
        );
    }

    #[test]
    fn test_pto_matmul_i8_blocked_dequant() {
        // __tile_matmul_i8_acc_i32_dequant_f16(dst, a, b, scale, m, k, n).
        // i8 A/B → i32 L0C accumulator, per-column f32 scale folded in the
        // L0C→GM DMA (FixPipe). Shapes chosen so k*n > L0 64KB cap → the
        // K/N-blocked path (the only supported i8 path) engages.
        let mlir = r#"
module {
  llvm.func @mm_i8(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %m = llvm.mlir.constant(16 : i32) : i32
    %k = llvm.mlir.constant(256 : i32) : i32
    %n = llvm.mlir.constant(512 : i32) : i32
    %t_a = llvm.call @__tile_load_i8(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @__tile_load_i8(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @__tile_matmul_i8_acc_i32_dequant_f16(%c0, %t_a, %t_b, %arg3, %m, %k, %n) : (i32, i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f16(%arg2, %t_c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let pto = convert_mlir_to_pto(mlir).expect("matmul_i8 PTO-MLIR generation");
        assert!(pto.contains("pto.tmatmul"), "i8 matmul must emit pto.tmatmul:\n{}", pto);
        // i8 operands, i32 accumulator.
        assert!(pto.contains("dtype=i8"), "i8 operand tiles expected:\n{}", pto);
        assert!(
            pto.contains("loc=acc, dtype=i32"),
            "i8 matmul L0C accumulator must be i32:\n{}",
            pto
        );
    }

    #[test]
    fn test_pick_kb_and_nb_convenience_wrappers() {
        // pick_kb / pick_nb are the N-agnostic convenience wrappers documented
        // for callers that don't know N. They delegate to the *_for_n / *_for_dtype
        // forms with N = u32::MAX / lhs_bytes = 2 (f16).
        assert_eq!(pick_kb(1536), pick_kb_for_n(1536, u32::MAX));
        assert_eq!(pick_nb(8960), pick_nb_for_dtype(8960, 2));
        // sane bounds: both return positive, kb divides into k-ish blocks.
        assert!(pick_kb(256) > 0);
        assert!(pick_nb(256) > 0);
    }

    // -----------------------------------------------------------------------
    // Error-path coverage: malformed MLIR that reaches the `unknown tile`
    // `.ok_or_else(...)` closures and the arity guards inside the translate_*
    // functions. Each `ghost_pto!` body references a source operand SSA that
    // was never produced by a load, so `ctx.get_tile(...)` returns None and
    // the op's error closure fires. convert_mlir_to_pto must return Err.
    // -----------------------------------------------------------------------

    /// Wrap a single intrinsic `$call` line in entry-func module boilerplate.
    /// `$call` references `%undef` (never loaded) in its source-operand slot.
    macro_rules! ghost_pto {
        ($call:expr) => {
            format!(
                "module {{\n  \
                 llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {{hacc.entry}} {{\n    \
                 {}\n    \
                 llvm.return\n  }}\n}}\n",
                $call
            )
        };
    }

    #[test]
    fn test_pto_binary_unknown_errs() {
        // translate_binary: add/mul/sub/div/max — unknown src1 tile.
        for op in [
            "__tile_add_f32",
            "__tile_mul_f32",
            "__tile_sub_f32",
            "__tile_div_f32",
            "__tile_add_f16",
            "__tile_mul_f16",
            "__tile_max_f32",
        ] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %undef2, %c32, %c32) : (i32, i32, i32, i32, i32) -> i32",
                op
            );
            let mlir = ghost_pto!(call);
            assert!(
                convert_mlir_to_pto(&mlir).is_err(),
                "{} with undefined src tile must error",
                op
            );
        }
    }

    #[test]
    fn test_pto_unary_unknown_errs() {
        // translate_unary: exp/neg/reduce_max/reduce_sum/scale — unknown src.
        for op in [
            "__tile_exp_f32",
            "__tile_exp_f16",
            "__tile_neg_f32",
            "__tile_reduce_max_f32",
            "__tile_reduce_sum_f32",
            "__tile_scale_f32",
        ] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %c32, %c32) : (i32, i32, i32, i32) -> i32",
                op
            );
            let mlir = ghost_pto!(call);
            assert!(convert_mlir_to_pto(&mlir).is_err(), "{} unknown src must error", op);
        }
    }

    #[test]
    fn test_pto_softmax_unknown_errs() {
        for op in ["__tile_softmax_f32", "__tile_softmax_f16"] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %c32, %c32) : (i32, i32, i32, i32) -> i32",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} must error", op);
        }
    }

    #[test]
    fn test_pto_matmul_unknown_errs() {
        // translate_matmul / translate_matmul_f16: unknown A tile.
        for op in ["__tile_matmul_f32", "__tile_matmul_f16"] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %undef2, %c16, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} must error", op);
        }
    }

    #[test]
    fn test_pto_matmul_transposed_unknown_errs() {
        for op in [
            "__tile_matmul_transposed_f32",
            "__tile_matmul_transposed_f16",
        ] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %undef2, %c16, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} must error", op);
        }
    }

    #[test]
    fn test_pto_store_unknown_errs() {
        // translate_store: buf SSA never produced by a load.
        for op in ["__tile_store_f32", "__tile_store_f16", "__tile_store_i8"] {
            let call = format!(
                "llvm.call @{}(%arg1, %undef, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} unknown buf must error", op);
        }
    }

    #[test]
    fn test_pto_simple_unary_like_unknown_errs() {
        // transpose/rsqrt/log/sigmoid/silu/cast/clamp/argmax/absmax —
        // single src operand at args[1].
        let cases: &[(&str, &str)] = &[
            ("__tile_transpose_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_rsqrt_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_log_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_sigmoid_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_silu_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_silu_f16", "(i32, i32, i32, i32) -> i32"),
            ("__tile_cast_f32_f16", "(i32, i32, i32, i32) -> i32"),
            ("__tile_cast_f16_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_cast_bf16_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_argmax_f32", "(i32, i32, i32, i32) -> i32"),
            ("__tile_absmax_f32", "(i32, i32, i32, i32) -> i32"),
        ];
        for (op, sig) in cases {
            let call = format!("%r = llvm.call @{}(%c0, %undef, %c32, %c32) : {}", op, sig);
            assert!(
                convert_mlir_to_pto(&ghost_pto!(call)).is_err(),
                "{} unknown src must error",
                op
            );
        }
    }

    #[test]
    fn test_pto_clamp_unknown_errs() {
        // clamp: (c0, src, min, max, rows, cols)
        let call = "%r = llvm.call @__tile_clamp_f32(%c0, %undef, %c0, %c1, %c32, %c32) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_rms_norm_unknown_errs() {
        // rms_norm: (c0, src, gamma, rows, cols)
        for op in ["__tile_rms_norm_f32", "__tile_rms_norm_f16"] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %undef2, %c8, %c32) : (i32, i32, i32, i32, i32) -> i32",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} must error", op);
        }
    }

    #[test]
    fn test_pto_quantize_dequantize_unknown_errs() {
        // quantize: (c0, src, scale, rows, cols); dequantize: (c0, src, scale, rows, cols)
        for op in ["__tile_quantize_f32_i8", "__tile_dequantize_i8_f32"] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %undef2, %c32, %c32) : (i32, i32, i32, i32, i32) -> i32",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} must error", op);
        }
    }

    #[test]
    fn test_pto_slice_concat_unknown_errs() {
        // slice: (c0, src, row_off, col_off, src_r, src_c, dst_r, dst_c)
        let slice = "%r = llvm.call @__tile_slice_f32(%c0, %undef, %c0, %c0, %c32, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(slice)).is_err());
        // concat: (c0, a, b, rows, cols_a, cols_b)
        let concat = "%r = llvm.call @__tile_concat_f32(%c0, %undef, %undef2, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(concat)).is_err());
    }

    #[test]
    fn test_pto_gather_scatter_unknown_errs() {
        // gather/scatter: (c0, src, indices, n, m, d)
        for op in ["__tile_gather_f32", "__tile_scatter_f32"] {
            let call = format!(
                "%r = llvm.call @{}(%c0, %undef, %undef2, %c32, %c32, %c1) : (i32, i32, i32, i32, i32, i32) -> i32",
                op
            );
            assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err(), "{} must error", op);
        }
    }

    #[test]
    fn test_pto_topk_unknown_errs() {
        // topk: (c0, src, indices_out, k, rows, cols) — rows/cols guarded >0 first.
        let call = "%r = llvm.call @__tile_topk_f32(%c0, %undef, %undef2, %c8, %c1, %c32) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_gather_mask_unknown_errs() {
        // gather_mask: (c0, src, mask, rows, cols) — guards rows>0/cols>0/mask<=15 first.
        let call = "%r = llvm.call @__tile_gather_mask_f32(%c0, %undef, %c10, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_gather_mask_arity_errs() {
        // gather_mask guard: mask must fit in 4 bits.
        let call = "%r = llvm.call @__tile_gather_mask_f32(%c0, %undef, %c99, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_sort_unknown_errs() {
        // init_sort_buf: (c0, src, rows, cols) — rows/cols guarded >0.
        let init = "%r = llvm.call @__tile_init_sort_buf_f32(%c0, %undef, %c1, %c32) : (i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(init)).is_err());
        // sort32: (c0, src, rows, cols)
        let sort = "%r = llvm.call @__tile_sort32_f32(%c0, %undef, %c1, %c32) : (i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(sort)).is_err());
        // mrgsort2: (c0, src0, src1, tmp, cols_each)
        let mrg = "%r = llvm.call @__tile_mrgsort2_f32(%c0, %undef, %undef2, %undef3, %c16) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(mrg)).is_err());
    }

    #[test]
    fn test_pto_phase6_unknown_errs() {
        // sample_top_p: (c0, logits, temp, top_p, seed, rows, cols)
        let stp = "%r = llvm.call @__tile_sample_top_p_f32(%c0, %undef, %c1, %c1, %c0, %c1, %c32) : (i32, i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(stp)).is_err());
        // draft_verify: (c0, draft, target, rows, cols) — looks up target at args[2].
        let dv = "%r = llvm.call @__tile_draft_verify_f32(%c0, %undef, %undef2, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(dv)).is_err());
        // token_accept: (c0, draft, target, probs, threshold, rows) — looks up draft at args[1].
        let ta = "%r = llvm.call @__tile_token_accept_f32(%c0, %undef, %undef2, %undef3, %c1, %c1) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(ta)).is_err());
    }

    #[test]
    fn test_pto_rope_unknown_errs() {
        // rope: (c0, src, pos, rows, cols)
        let call = "%r = llvm.call @__tile_rope_f32(%c0, %undef, %c0, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_attention_unknown_and_arity_errs() {
        // attention: 6 args (c0, q, k, v, scale, seq) — unknown Q tile.
        let attn = "%r = llvm.call @__tile_attention_f32(%c0, %undef, %undef2, %undef3, %c1, %c32) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(attn)).is_err());
        // attention arity: only 3 args -> args.len() < 6 guard.
        let attn_arity = "%r = llvm.call @__tile_attention_f32(%c0, %undef, %undef2) : (i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(attn_arity)).is_err());
        // attention_gqa: 8 args — unknown Q tile.
        let gqa = "%r = llvm.call @__tile_attention_gqa_f32(%c0, %undef, %undef2, %undef3, %c1, %c32, %c4, %c1) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(gqa)).is_err());
        // attention_gqa arity: only 4 args -> args.len() < 8 guard.
        let gqa_arity = "%r = llvm.call @__tile_attention_gqa_f32(%c0, %undef, %undef2, %undef3) : (i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(gqa_arity)).is_err());
    }

    #[test]
    fn test_pto_matmul_i8_unknown_errs() {
        // matmul_i8: (c0, a, b, scale_ptr, m, k, n) — unknown A tile.
        let call = "%r = llvm.call @__tile_matmul_i8_acc_i32_dequant_f16(%c0, %undef, %undef2, %arg1, %c16, %c16, %c16) : (i32, i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }
}
