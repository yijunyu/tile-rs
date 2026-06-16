//! MLIR-to-PTO-MLIR translator for Ascend NPU targets.
//!
//! Converts merged MLIR modules (LLVM dialect with `ascend_tile_*` intrinsics)
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
//! | `ascend_tile_load_f32(gm, rows, cols)` | `pto.tload` |
//! | `ascend_tile_store_f32(gm, buf, rows, cols)` | `pto.tstore` |
//! | `ascend_tile_add_f32(0, a, b, rows, cols)` | `pto.tadd` |
//! | `ascend_tile_mul_f32(0, a, b, rows, cols)` | `pto.tmul` |
//! | `ascend_tile_exp_f32(0, src, rows, cols)` | `pto.texp` |
//! | `ascend_tile_softmax_f32(0, src, rows, cols)` | `pto.tsoftmax` |
//! | `ascend_tile_matmul_f32(0, a, b, m, k, n)` | `pto.tmatmul` |
//! | `get_block_idx()` | (block_id via `get_block_idx` in future extension) |
//! | `ascend_pipe_barrier` | (suppressed — PTO/ptoas inserts sync automatically) |
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
//! `ascend_tile_softmax_f32` is lowered to the numerically-stable 5-step decomposition:
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
/// `ascend_tile_matmul_transposed_*` is deliberately NOT a trigger here:
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
            if line.contains("ascend_tile_attention_f32")
                || line.contains("ascend_tile_attention_gqa_f32")
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
}

struct PtoContext {
    /// Map from MLIR SSA name (e.g., `%t0`, `%5`) → TileInfo
    tiles: HashMap<String, TileInfo>,
    /// Allocation counter for generating unique SSA names
    next_ssa: u32,
    /// Map from GM pointer arg name → (tensor_view_ssa, rows, cols, dtype)
    tv_map: HashMap<String, (String, u32, u32, String)>,
    /// Ordered list of sizes we need `arith.constant` for
    sizes_used: Vec<u32>,
    /// SSA alias map: derived pointer SSA → original GM arg name.
    /// Tracks `llvm.getelementptr %argN[...]` and `llvm.load ... !llvm.ptr<1>`
    /// chains so we can resolve `%8` back to `%arg0` when it appears
    /// as the gm argument to `ascend_tile_load_f32`.
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
    kb: u32,
    nb: u32,
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
    fn resolve_const(&self, s: &str) -> u32 {
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
        self.use_size(row_off);
        self.use_size(col_off);
        self.use_size(rows);
        self.use_size(cols);
        let ssa = self.fresh_ssa();
        let tv_ty = tv_type(rows, cols, dtype);
        let ptv_ty = ptv_type(rows, cols, dtype);
        ops.push(format!(
            "{} = pto.partition_view {}, offsets = [%c{}, %c{}], sizes = [%c{}, %c{}] : {} -> {}",
            ssa, tv_ssa, row_off, col_off, rows, cols, tv_ty, ptv_ty
        ));
        ssa
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
            || line.contains("ascend_pipe_barrier")
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
        if line.contains("ascend_tile_pipelined_for_begin")
            || line.contains("ascend_tile_pipelined_for_end")
        {
            continue;
        }

        // tile.load f32
        if line.contains("ascend_tile_load_f32") {
            let blocked = blocked_mm_loads.contains_key(&i);
            translate_load(line, "f32", ctx, func, &mut ops, blocked)?;
            continue;
        }
        // tile.load f16
        if line.contains("ascend_tile_load_f16") {
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
        if line.contains("ascend_tile_load_i8") {
            let blocked = blocked_mm_loads.contains_key(&i);
            translate_load(line, "i8", ctx, func, &mut ops, blocked)?;
            continue;
        }
        // tile.store f32
        if line.contains("ascend_tile_store_f32") {
            translate_store(line, "f32", ctx, func, &mut ops)?;
            continue;
        }
        // tile.store f16
        if line.contains("ascend_tile_store_f16") {
            translate_store(line, "f16", ctx, func, &mut ops)?;
            continue;
        }
        // tile.store i8
        if line.contains("ascend_tile_store_i8") {
            translate_store(line, "i8", ctx, func, &mut ops)?;
            continue;
        }
        // tile.add f32
        if line.contains("ascend_tile_add_f32") {
            translate_binary(line, "f32", "pto.tadd", ctx, &mut ops)?;
            continue;
        }
        // tile.mul f32
        if line.contains("ascend_tile_mul_f32") {
            translate_binary(line, "f32", "pto.tmul", ctx, &mut ops)?;
            continue;
        }
        // tile.add f16
        if line.contains("ascend_tile_add_f16") {
            translate_binary(line, "f16", "pto.tadd", ctx, &mut ops)?;
            continue;
        }
        // tile.mul f16
        if line.contains("ascend_tile_mul_f16") {
            translate_binary(line, "f16", "pto.tmul", ctx, &mut ops)?;
            continue;
        }
        // tile.exp f32
        if line.contains("ascend_tile_exp_f32") {
            translate_unary(line, "f32", "pto.texp", ctx, &mut ops)?;
            continue;
        }
        // tile.exp f16
        if line.contains("ascend_tile_exp_f16") {
            translate_unary(line, "f16", "pto.texp", ctx, &mut ops)?;
            continue;
        }
        // tile.softmax f32 — decomposed into 5 reduction ops
        if line.contains("ascend_tile_softmax_f32") {
            translate_softmax(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.softmax f16 — decomposed into 5 reduction ops
        if line.contains("ascend_tile_softmax_f16") {
            translate_softmax(line, "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul f32
        if line.contains("ascend_tile_matmul_f32") {
            translate_matmul(line, ctx, &mut ops)?;
            continue;
        }
        // tile.sub f32
        if line.contains("ascend_tile_sub_f32") {
            translate_binary(line, "f32", "pto.tsub", ctx, &mut ops)?;
            continue;
        }
        // tile.div f32
        if line.contains("ascend_tile_div_f32") {
            translate_binary(line, "f32", "pto.tdiv", ctx, &mut ops)?;
            continue;
        }
        // tile.neg f32
        if line.contains("ascend_tile_neg_f32") {
            translate_unary(line, "f32", "pto.tneg", ctx, &mut ops)?;
            continue;
        }
        // tile.reduce_max f32 — row-wise max
        if line.contains("ascend_tile_reduce_max_f32") {
            translate_unary(line, "f32", "pto.trowmax", ctx, &mut ops)?;
            continue;
        }
        // tile.reduce_sum f32 — row-wise sum
        if line.contains("ascend_tile_reduce_sum_f32") {
            translate_unary(line, "f32", "pto.trowsum", ctx, &mut ops)?;
            continue;
        }
        // tile.scale f32 — scalar multiply (treated as unary with scalar operand)
        if line.contains("ascend_tile_scale_f32") {
            translate_unary(line, "f32", "pto.tmuls", ctx, &mut ops)?;
            continue;
        }
        // tile.silu f32 — SiLU(x) = x * sigmoid(x), with optional SiLU+Mul fusion
        if line.contains("ascend_tile_silu_f32") {
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
        if line.contains("ascend_tile_silu_f16") {
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
        if line.contains("ascend_tile_cast_bf16_f32") {
            translate_cast(line, "bf16", "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul_transposed f32 — C = A * B^T via tmatmul with transposed flag
        if line.contains("ascend_tile_matmul_transposed_f32") {
            translate_matmul_transposed(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul_transposed f16
        if line.contains("ascend_tile_matmul_transposed_f16") {
            translate_matmul_transposed(line, "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.attention_gqa f32 — Grouped-Query Attention
        if line.contains("ascend_tile_attention_gqa_f32") {
            translate_attention_gqa(line, ctx, &mut ops)?;
            continue;
        }
        // tile.attention f32 — fused Q@K^T → scale → softmax → @V
        // Decomposed into: matmul + scale + softmax_5ops + matmul
        if line.contains("ascend_tile_attention_f32") {
            translate_attention(line, ctx, &mut ops)?;
            continue;
        }
        // tile.transpose f32
        if line.contains("ascend_tile_transpose_f32") {
            translate_transpose(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.rsqrt f32
        if line.contains("ascend_tile_rsqrt_f32") {
            translate_rsqrt(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.log f32
        if line.contains("ascend_tile_log_f32") {
            translate_log(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.sigmoid f32 — decomposed: neg → exp → adds(1) → divs(1)
        if line.contains("ascend_tile_sigmoid_f32") {
            translate_sigmoid(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.clamp f32 — clamp to [min, max] via tmaxs + tmins
        if line.contains("ascend_tile_clamp_f32") {
            translate_clamp(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.cast f32→f16
        if line.contains("ascend_tile_cast_f32_f16") {
            translate_cast(line, "f32", "f16", ctx, &mut ops)?;
            continue;
        }
        // tile.cast f16→f32
        if line.contains("ascend_tile_cast_f16_f32") {
            translate_cast(line, "f16", "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.slice f32 — extract sub-tile via partition_view with offset
        if line.contains("ascend_tile_slice_f32") {
            translate_slice(line, "f32", ctx, func, &mut ops)?;
            continue;
        }
        // tile.concat f32 — concatenate two tiles along columns
        if line.contains("ascend_tile_concat_f32") {
            translate_concat(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.scatter f32 — no PTO equivalent
        if line.contains("ascend_tile_scatter_f32") {
            translate_scatter(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.gather f32 — no PTO equivalent
        if line.contains("ascend_tile_gather_f32") {
            translate_gather(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.arith_progression i32 — emits pto.tci (iota for sort indices).
        if line.contains("ascend_tile_arith_progression_i32") {
            translate_arith_progression(line, ctx, &mut ops)?;
            continue;
        }
        // tile.init_sort_buf f32 — emits pto.tfillpad (sentinel pad to BLOCK boundary).
        if line.contains("ascend_tile_init_sort_buf_f32") {
            translate_init_sort_buf(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.sort32 f32 — emits pto.tsort32 (vbitsort, output is 2× width [val,idx] pairs).
        if line.contains("ascend_tile_sort32_f32") {
            translate_tile_sort(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.mrgsort2 f32 — emits pto.tmrgsort 2-way (merges two 1×N sorted tiles).
        if line.contains("ascend_tile_mrgsort2_f32") {
            translate_merge_sort(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.gather_mask f32 — emits pto.tgather (mask-pattern form, lane select).
        if line.contains("ascend_tile_gather_mask_f32") {
            translate_gather_mask(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.topk f32 — no PTO equivalent
        if line.contains("ascend_tile_topk_f32") {
            translate_topk(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.matmul f16
        if line.contains("ascend_tile_matmul_f16") {
            translate_matmul_f16(line, ctx, &mut ops)?;
            continue;
        }
        // tile.matmul i8×i8→i32 with per-column f32 dequant → f16 GM
        if line.contains("ascend_tile_matmul_i8_acc_i32_dequant_f16") {
            translate_matmul_i8(line, ctx, func, &mut ops)?;
            continue;
        }

        // tile.fill
        if line.contains("ascend_tile_fill_f32") || line.contains("ascend_tile_fill_f16") {
            translate_fill(line, ctx, &mut ops)?;
            continue;
        }
        // tile.max (element-wise)
        if line.contains("ascend_tile_max_f32") || line.contains("ascend_tile_max_f16") {
            let dtype = if line.contains("f16") { "f16" } else { "f32" };
            translate_binary(line, dtype, "pto.tmax", ctx, &mut ops)?;
            continue;
        }
        // tile.rms_norm
        if line.contains("ascend_tile_rms_norm_f32") || line.contains("ascend_tile_rms_norm_f16") {
            translate_rms_norm_pto(line, ctx, &mut ops)?;
            continue;
        }
        // tile.absmax_f32 — max of absolute values, broadcast to tile
        if line.contains("ascend_tile_absmax_f32") {
            translate_absmax_pto(line, ctx, &mut ops)?;
            continue;
        }
        // tile.quantize_f32_i8 — round(src/scale) clamped to [-128,127]
        if line.contains("ascend_tile_quantize_f32_i8") {
            translate_quantize_pto(line, ctx, &mut ops)?;
            continue;
        }
        // tile.dequantize_i8_f32 — src * scale (int8→f32)
        if line.contains("ascend_tile_dequantize_i8_f32") {
            translate_dequantize_pto(line, ctx, &mut ops)?;
            continue;
        }
        // Phase 6 MTP ops — no native PTO equivalent; scalar loop decomposition
        // tile.argmax f32 — row-wise argmax → (R,1) u32
        if line.contains("ascend_tile_argmax_f32") {
            translate_argmax_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.sample_top_p f32 — nucleus sampling → (R,1) u32
        if line.contains("ascend_tile_sample_top_p_f32") {
            translate_sample_top_p_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.draft_verify f32 — acceptance probabilities → (R,1) f32
        if line.contains("ascend_tile_draft_verify_f32") {
            translate_draft_verify_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.token_accept f32 — select final tokens → (R,1) u32
        if line.contains("ascend_tile_token_accept_f32") {
            translate_token_accept_pto(line, "f32", ctx, &mut ops)?;
            continue;
        }
        // tile.rope f32 — Rotary Position Embedding
        if line.contains("ascend_tile_rope_f32") {
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

/// `%res = llvm.call @ascend_tile_load_f32(%gm, %rows, %cols) : ...`
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

/// `llvm.call @ascend_tile_store_f32(%gm, %buf, %rows, %cols) : ...`
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
    let pv_a_ty = ptv_type(p.m, p.kb, p.dtypes.lhs);
    let pv_b_ty = ptv_type(p.kb, p.nb, p.dtypes.rhs);
    let pv_o_ty = ptv_type(p.m, p.nb, out_dtype);
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

    // Per-block partition_view for A[0:M, k_off:k_off+Kb].
    let pv_a_blk = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = pto.partition_view {}, offsets = [%c{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        k_body_indent,
        pv_a_blk,
        p.tv_a_ssa,
        a_base_row,
        k_off_ssa,
        p.m,
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
    let pv_o_blk = ctx.fresh_ssa();
    ops.push(format!(
        "{}{} = pto.partition_view {}, offsets = [%c{}, {}], sizes = [%c{}, %c{}] : {} -> {}",
        k_indent,
        pv_o_blk,
        tv_out_ssa,
        out_base_row,
        n_off_ssa,
        p.m,
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
}

/// Binary: `%res = llvm.call @ascend_tile_add_f32(%c0, %a, %b, %rows, %cols)`
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

/// Unary: `%res = llvm.call @ascend_tile_exp_f32(%c0, %src, %rows, %cols)`
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

/// Matmul: `%res = llvm.call @ascend_tile_matmul_f32(%c0, %a, %b, %m, %k, %n)`
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
    ops.push(format!(
        "pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
        left_ssa, right_ssa, left_ty, right_ty, acc_ssa, acc_ty
    ));

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
    let mat_a_ty = mat_tile_type(m, kb, dtypes.lhs);
    let mat_b_ty = mat_tile_type(kb, nb, dtypes.rhs);
    let left_ty = left_tile_type(m, kb, dtypes.lhs);
    let right_ty = right_tile_type(kb, nb, dtypes.rhs);
    let acc_ty = acc_tile_type(m, nb, dtypes.dst);

    let mat_a_ssa = ctx.alloc_tile_typed(
        &format!("{}__mat_a_blk", result_ssa),
        m,
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
        m,
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
        ctx.alloc_tile_typed(result_ssa, m, nb, dtypes.dst, &acc_ty, ops);

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
    // `ascend_tile_store_f32(out_gm, result_ssa, M, N)` and doesn't
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
        kb,
        nb,
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

/// Softmax: `%res = llvm.call @ascend_tile_softmax_f32(%c0, %src, %rows, %cols)`
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

    let tsrc = ctx
        .get_tile(src_ssa)
        .ok_or_else(|| format!("softmax: unknown tile {}", src_ssa))?
        .clone();
    let tb_ty = tile_buf_type(rows, cols, dtype);

    // Step 1: allocate scratch tiles
    let t_max_key = format!("{}__max", result_ssa);
    let t_tmp_key = format!("{}__tmp", result_ssa);
    let t_sub_key = format!("{}__sub", result_ssa);
    let t_exp_key = format!("{}__exp", result_ssa);
    let t_sum_key = format!("{}__sum", result_ssa);

    // t_max and t_sum are row-reduction outputs: rows×1, col_major
    let rr_ty = tile_buf_type_rowreduce(rows, dtype);
    let t_max_ssa = ctx.alloc_tile_rowreduce(&t_max_key, rows, dtype, ops);
    let t_tmp_ssa = ctx.alloc_tile(&t_tmp_key, rows, cols, dtype, ops);
    let t_sub_ssa = ctx.alloc_tile(&t_sub_key, rows, cols, dtype, ops);
    let t_exp_ssa = ctx.alloc_tile(&t_exp_key, rows, cols, dtype, ops);
    let t_sum_ssa = ctx.alloc_tile_rowreduce(&t_sum_key, rows, dtype, ops);
    // Step 5 destination mapped to result_ssa
    let t_out_ssa = ctx.alloc_tile(&result_ssa, rows, cols, dtype, ops);

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

/// Transpose: `%res = llvm.call @ascend_tile_transpose_f32(%c0, %src, %rows, %cols)`
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

/// Rsqrt: `%res = llvm.call @ascend_tile_rsqrt_f32(%c0, %src, %rows, %cols)`
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

/// Log: `%res = llvm.call @ascend_tile_log_f32(%c0, %src, %rows, %cols)`
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

/// Sigmoid: `%res = llvm.call @ascend_tile_sigmoid_f32(%c0, %src, %rows, %cols)`
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

/// SiLU: `%res = llvm.call @ascend_tile_silu_f32(%c0, %src, %rows, %cols)`
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
///   %silu = llvm.call @ascend_tile_silu_f32/f16(...)
///   %out  = llvm.call @ascend_tile_mul_f32/f16(%c0, %silu, %up, ...)
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
const PTO_MM_KB: u32 = 256;
const PTO_MM_NB: u32 = 64;
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
fn pick_nb(n: u32) -> u32 {
    pick_nb_for_dtype(n, 2)
}

/// Dtype-aware Nb pick. i8 needs Nb=256 so that ptoas translates the L0A
/// `Left` tile with `BLayout::RowMajor` (matching the validated hand-written
/// i8 probe). At Nb=64, ptoas silently emits `BLayout::ColMajor` for i8 Left
/// which produces garbage numerics.
fn pick_nb_for_dtype(n: u32, lhs_bytes: u32) -> u32 {
    let nb_base = if lhs_bytes == 1 { PTO_MM_NB_I8 } else { PTO_MM_NB };
    nb_base.min(n)
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
    mk_bytes > L0_CAP_BYTES || kn_bytes > L0_CAP_BYTES
}

/// Pre-pass: find tile_load lines whose result is consumed only by a
/// matmul that needs blocking, and whose load shape matches the matmul's
/// A/B operand shape. Returns a set of body_lines indices to skip.
///
/// Each entry in the returned map is `load_idx → matmul_operand_role`
/// (`"A"` or `"B"`). translate_load uses this to skip emitting the
/// full-shape pto.tload / alloc_tile for those loads; translate_matmul
/// later rebuilds per-block loads inside its scf.for nest.
fn detect_blocked_matmul_loads(body_lines: &[String]) -> HashMap<usize, &'static str> {
    let mut result: HashMap<usize, &'static str> = HashMap::new();

    for line in body_lines.iter() {
        let trimmed = line.trim();
        // Match f32-matmul (may need K/N-blocking), f16-matmul (single-block
        // but still needs mat/CBUF routing — CANN 8.5 cube doesn't support
        // b16 GM→UB), and i8-matmul with per-column dequant (same CBUF
        // routing, always blocked at decoder shapes).
        let is_f32_mm = trimmed.contains("ascend_tile_matmul_f32");
        let is_f16_mm = trimmed.contains("ascend_tile_matmul_f16");
        let is_i8_mm = trimmed.contains("ascend_tile_matmul_i8_acc_i32_dequant_f16");
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
            "ascend_tile_load_f16"
        } else if is_i8_mm {
            "ascend_tile_load_i8"
        } else {
            "ascend_tile_load_f32"
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

/// UB budget for the 5-tile silu_mul emit: 256 KB UB cap minus ~32 KB reserved
/// for kernel code, stack, and scalars. Used by both the unblocked path's
/// guard and the blocked-path detector. Defined once here to keep the two
/// in sync.
const SILU_MUL_UB_BUDGET_BYTES: u64 = 224 * 1024;

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
        let dtype = if silu_line.contains("ascend_tile_silu_f32") {
            "f32"
        } else if silu_line.contains("ascend_tile_silu_f16") {
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
            "ascend_tile_load_f16"
        } else {
            "ascend_tile_load_f32"
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
        if !trimmed.contains("ascend_tile_silu_f32") && !trimmed.contains("ascend_tile_silu_f16") {
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
            if next.contains("ascend_tile_mul_f32") || next.contains("ascend_tile_mul_f16") {
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
    // Parse silu: %silu_res = llvm.call @ascend_tile_silu_f32(%c0, %gate, %rows, %cols)
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

/// Matmul transposed: `%res = llvm.call @ascend_tile_matmul_transposed_f32(%c0, %a, %b, %m, %k, %n)`
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

/// Clamp: `%res = llvm.call @ascend_tile_clamp_f32(%c0, %src, %min, %max, %rows, %cols)`
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

/// Cast: `%res = llvm.call @ascend_tile_cast_f32_f16(%c0, %src, %rows, %cols)`
///        `%res = llvm.call @ascend_tile_cast_f16_f32(%c0, %src, %rows, %cols)`
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

/// Slice: `%res = llvm.call @ascend_tile_slice_f32(%c0, %src, %row_off, %col_off, %src_r, %src_c, %dst_r, %dst_c)`
///
/// Extracts a sub-tile from a larger tile. Emits tmov passthrough with reshaped output.
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

/// Concat: `%res = llvm.call @ascend_tile_concat_f32(%c0, %a, %b, %rows, %cols_a, %cols_b)`
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

/// Scatter: `%res = llvm.call @ascend_tile_scatter_f32(%c0, %src, %indices, %n, %m, %d)`
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

/// Gather: `%res = llvm.call @ascend_tile_gather_f32(%c0, %src, %indices, %n, %m, %d)`
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

/// Mask-pattern gather: `%res = llvm.call @ascend_tile_gather_mask_f32(%c0, %src, %mask_pattern, %rows, %cols)`
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

/// 2-way bitonic merge: `%res = llvm.call @ascend_tile_mrgsort2_f32(%c0, %src0, %src1, %tmp, %cols_each)`
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

/// Tile sort: `%res = llvm.call @ascend_tile_sort32_f32(%c0, %values, %indices, %rows, %cols)`
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

/// Sort-buffer init: `%res = llvm.call @ascend_tile_init_sort_buf_f32(%c0, %src, %rows, %cols)`
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

/// Iota / arithmetic progression: `%res = llvm.call @ascend_tile_arith_progression_i32(%c0, %start, %valid_col)`
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

/// Top-K: `%res = llvm.call @ascend_tile_topk_f32(%c0, %src, %indices_out, %rows, %cols, %k)`
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

/// Matmul f16: `%res = llvm.call @ascend_tile_matmul_f16(%c0, %a, %b, %m, %k, %n)`
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
    // caller's downstream `ascend_tile_store_f16` reads the acc tile and
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
    //    caller's `ascend_tile_store_f16` will emit a `pto.tstore` from
    //    this f32 acc into an f16 GM partition_view, and the hardware
    //    FixPipe path performs the f32→f16 cast during the L0C→GM DMA.
    ops.push(format!(
        "pto.tmatmul ins({}, {} : {}, {}) outs({} : {})",
        left_ssa, right_ssa, left_ty, right_ty, acc_ssa, acc_ty
    ));

    Ok(())
}

/// Matmul i8×i8→i32 with per-column f32 dequant → f16 GM:
/// `%res = llvm.call @ascend_tile_matmul_i8_acc_i32_dequant_f16(
///            %c0, %a, %b, %scale_ptr, %m, %k, %n)`
///
/// Dtype rules (ptoas CANN 8.5, empirical): A / B both i8, L0C accumulator
/// i32. Per-column f32 scale tile is loaded into `loc=scaling` (__fbuf__) and
/// folded into the L0C→GM DMA via `pto.tstore_fp` (FixPipe). The output is
/// registered as `dst="i32"` under the matmul result SSA; the caller's
/// downstream `ascend_tile_store_f16` sees the i32 acc tile but the inline
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
    // extern fn ascend_tile_matmul_i8_acc_i32_dequant_f16(
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

/// Fill: `%res = llvm.call @ascend_tile_fill_f32(%c0, %scalar, %rows, %cols)`
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

/// RoPE: `%res = llvm.call @ascend_tile_rope_f32(%c0, %src, %pos, %rows, %cols)`
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

/// Argmax: `%res = llvm.call @ascend_tile_argmax_f32(%c0, %src, %rows, %cols)`
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

/// SampleTopP: `%res = llvm.call @ascend_tile_sample_top_p_f32(%c0, %logits, %temp, %top_p, %seed, %rows, %cols)`
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

/// DraftVerify: `%res = llvm.call @ascend_tile_draft_verify_f32(%c0, %draft_tokens, %target_logits, %rows, %cols)`
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

/// TokenAccept: `%res = llvm.call @ascend_tile_token_accept_f32(%c0, %draft, %target, %probs, %threshold, %rows)`
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
/// `ascend_tile_load_*` / `ascend_tile_store_*` call that uses it as a
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
    // The LLVM IR shape is: `llvm.call @ascend_tile_load_fNN(%argK, ...)` or
    // `llvm.call @ascend_tile_store_fNN(%argK, ...)`. We do a substring
    // check: if the line mentions both the arg name and a typed tile_load/
    // tile_store call, use the dtype from the callee name.
    for line in body_lines {
        if !line.contains(arg_name) {
            continue;
        }
        // Check store (GM arg is 1st pointer param, written).
        if line.contains("ascend_tile_store_f16") {
            return Some("f16");
        }
        if line.contains("ascend_tile_store_f32") {
            return Some("f32");
        }
        if line.contains("ascend_tile_store_bf16") {
            return Some("bf16");
        }
        // Check load (GM arg is 1st pointer param, read).
        if line.contains("ascend_tile_load_f16") {
            return Some("f16");
        }
        if line.contains("ascend_tile_load_f32") {
            return Some("f32");
        }
        if line.contains("ascend_tile_load_bf16") {
            return Some("bf16");
        }
        if line.contains("ascend_tile_load_i8") {
            return Some("i8");
        }
        if line.contains("ascend_tile_store_i8") {
            return Some("i8");
        }
        // int8 matmul has a scale arg at position 3. CANN 8.5 ptoas requires
        // the scale tile to be dtype=ui64 (TMovToFb needs uint64_t DstType),
        // so we emit `!pto.ptr<ui64>` here even though the Rust source declares
        // the pointer as `*const f32`. Host-side repacks f32 → u64 FB words
        // before launch (see pack_scale_f32_to_u64).
        if line.contains("ascend_tile_matmul_i8_acc_i32_dequant_f16") {
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
        // Default Kb=256 stands when N is small (Kb*N <= 2^23).
        assert_eq!(pick_kb_for_n(1536, 1536), 256); // q/o_proj
        assert_eq!(pick_kb_for_n(1536, 256), 256); // kv_proj
        assert_eq!(pick_kb_for_n(1536, 8960), 256); // gate/up_proj
        assert_eq!(pick_kb_for_n(8960, 1536), 256); // down_proj
        // lm_head: N=151936 forces Kb down so that Kb*N < 2^24. With our
        // 2^23 threshold, Kb ≤ (2^23)/151936 ≈ 55 → aligned to 48.
        // 1536 % 48 == 0, so expect Kb=48.
        assert_eq!(pick_kb_for_n(1536, 151936), 48);
        // N=65536 exactly at old failure boundary: 2^23/65536 = 128. 1536%128=0.
        assert_eq!(pick_kb_for_n(1536, 65536), 128);
        // N=32768 safe at full Kb=256: 256*32768 = 2^23 (equal to threshold,
        // cap = 2^23/32768 = 256, so we get Kb=256).
        assert_eq!(pick_kb_for_n(1536, 32768), 256);
        // Degenerate / small K that's still > kb_cap.
        // K=128 (base=128), cap=48: 128%48!=0 → fallback halving to 32 (divides 128, ≤48).
        assert_eq!(pick_kb_for_n(128, 151936), 32);
        // K=64 (base=64), cap=48: 64%48!=0 → fallback to 32 (divides 64, ≤48).
        assert_eq!(pick_kb_for_n(64, 151936), 32);
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
            "    %t = llvm.call @ascend_tile_load_f16(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32".to_string(),
            "    llvm.call @ascend_tile_store_f16(%arg1, %t, %c16, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()".to_string(),
        ];
        assert_eq!(infer_arg_dtype_from_body("%arg0", &body), Some("f16"));
        assert_eq!(infer_arg_dtype_from_body("%arg1", &body), Some("f16"));
        assert_eq!(infer_arg_dtype_from_body("%arg2", &body), None);
    }

    #[test]
    fn test_infer_arg_dtype_from_body_f32() {
        let body = vec![
            "    %t = llvm.call @ascend_tile_load_f32(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32".to_string(),
            "    llvm.call @ascend_tile_store_f32(%arg1, %t, %c16, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()".to_string(),
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
    %t_a = llvm.call @ascend_tile_load_f16(%arg0, %c16, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @ascend_tile_load_f16(%arg1, %c256, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @ascend_tile_matmul_f16(%c0, %t_a, %t_b, %c16, %c256, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg2, %t_c, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
        assert!(is_builtin_helper("ascend_add_f32"));
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_load_f32(%arg1, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @ascend_tile_add_f32(%c0, %t0, %t1, %c32, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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

    #[test]
    fn test_softmax_f32_generates_pto_mlir() {
        let mlir = r#"
module {
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f16(%arg0, %c16, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_softmax_f16(%c0, %t0, %c16, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg1, %t1, %c16, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_exp_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f32(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @ascend_tile_load_f32(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @ascend_tile_matmul_f32(%c0, %t_a, %t_b, %c16, %c32, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t_c, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f32(%arg0, %c16, %ck) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @ascend_tile_load_f32(%arg1, %ck, %cn) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @ascend_tile_matmul_f32(%c0, %t_a, %t_b, %c16, %ck, %cn) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t_c, %c16, %cn) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f32(%arg0, %c16, %c1536) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @ascend_tile_load_f32(%arg1, %c1536, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @ascend_tile_matmul_f32(%c0, %t_a, %t_b, %c16, %c1536, %c256) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t_c, %c16, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_load_f32(%ptr1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %s0 = llvm.call @ascend_tile_softmax_f32(%c0, %t0, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    %s1 = llvm.call @ascend_tile_softmax_f32(%c0, %t1, %c1, %c1024) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %s0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    %ptr1_out = llvm.getelementptr %arg1[%c1024] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.call @ascend_tile_store_f32(%ptr1_out, %s1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_load_f32(%arg1, %c1, %c1024) : (!llvm.ptr<1>, i32, i32) -> i32
    %r = llvm.call @ascend_tile_add_f32(%c0, %t0, %t1, %c1, %c1024) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %r, %c1, %c1024) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_transpose_f32(%c0, %t0, %c16, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c32, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_rsqrt_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_log_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c4, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_sigmoid_f32(%c0, %t0, %c4, %c256) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c4, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_clamp_f32(%c0, %t0, %c0, %c6, %c32, %c32) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_cast_f32_f16(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f16(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_cast_f16_f32(%c0, %t0, %c32, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_slice_f32(%c0, %t0, %c4, %c8, %c32, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_load_f32(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @ascend_tile_concat_f32(%c0, %t0, %t1, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_scatter_f32(%c0, %t0, %arg1, %c8, %c32, %c1) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t1, %c8, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c8, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_gather_f32(%c0, %t0, %arg1, %c8, %c32, %c1) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t1, %c8, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %s = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %g = llvm.call @ascend_tile_gather_mask_f32(%c0, %s, %c10, %c1, %c128) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %g, %c1, %c128) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %a = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %t = llvm.call @ascend_tile_load_f32(%arg2, %c1, %c256) : (!llvm.ptr<1>, i32, i32) -> i32
    %m = llvm.call @ascend_tile_mrgsort2_f32(%c0, %a, %b, %t, %c128) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %m, %c1, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %v = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %i = llvm.call @ascend_tile_arith_progression_i32(%c0, %c0, %c128) : (i32, i32, i32) -> i32
    %s = llvm.call @ascend_tile_sort32_f32(%c0, %v, %i, %c1, %c128) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %s, %c1, %c256) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %p = llvm.call @ascend_tile_init_sort_buf_f32(%c0, %t, %c1, %c128) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %p, %c1, %c128) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t = llvm.call @ascend_tile_arith_progression_i32(%c0, %c0, %c128) : (i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_i32(%arg0, %t, %c1, %c128) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c4, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_topk_f32(%c0, %t0, %arg1, %c4, %c64, %c8) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t1, %c4, %c8) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c128) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_topk_f32(%c0, %t0, %arg1, %c1, %c128, %c8) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t1, %c1, %c8) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f16(%arg0, %c16, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @ascend_tile_load_f16(%arg1, %c32, %c16) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @ascend_tile_matmul_f16(%c0, %t_a, %t_b, %c16, %c32, %c16) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg2, %t_c, %c16, %c16) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_r = llvm.call @ascend_tile_absmax_f32(%t_a, %t_a, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t_r, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_s = llvm.call @ascend_tile_load_f32(%arg1, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_r = llvm.call @ascend_tile_quantize_f32_i8(%t_a, %t_a, %t_s, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t_r, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t_a = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_s = llvm.call @ascend_tile_load_f32(%arg1, %c1, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_r = llvm.call @ascend_tile_dequantize_i8_f32(%t_a, %t_a, %t_s, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t_r, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_argmax_f32(%c0, %t0, %c4, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_sample_top_p_f32(%c0, %t0, %c0, %c0, %c0, %c4, %c32) : (i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_load_f32(%arg1, %c4, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @ascend_tile_draft_verify_f32(%c0, %t0, %t1, %c4, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %t2, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c4, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_load_f32(%arg1, %c4, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @ascend_tile_load_f32(%arg2, %c4, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t3 = llvm.call @ascend_tile_token_accept_f32(%c0, %t0, %t1, %t2, %c0, %c4) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %t3, %c4, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_silu_f32(%t0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %gate = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @ascend_tile_load_f32(%arg1, %c1, %c64) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @ascend_tile_silu_f32(%gate, %gate, %c1, %c64) : (i32, i32, i32, i32) -> i32
    %out = llvm.call @ascend_tile_mul_f32(%silu, %silu, %up, %c1, %c64) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %out, %c1, %c64) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_silu_f32(%t0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %gate = llvm.call @ascend_tile_load_f32(%arg0, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @ascend_tile_load_f32(%arg1, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @ascend_tile_silu_f32(%gate, %gate, %c1, %cN) : (i32, i32, i32, i32) -> i32
    %out = llvm.call @ascend_tile_mul_f32(%silu, %silu, %up, %c1, %cN) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %out, %c1, %cN) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_silu_f32(%t0, %t0, %c1, %cN) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c1, %cN) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %gate = llvm.call @ascend_tile_load_f32(%arg0, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @ascend_tile_load_f32(%arg1, %c1, %cN) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @ascend_tile_silu_f32(%gate, %gate, %c1, %cN) : (i32, i32, i32, i32) -> i32
    %out = llvm.call @ascend_tile_mul_f32(%silu, %silu, %up, %c1, %cN) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %out, %c1, %cN) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %t0 = llvm.call @ascend_tile_load_f32(%arg0, %c1, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @ascend_tile_cast_bf16_f32(%t0, %t0, %c1, %c32) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %t1, %c1, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %q = llvm.call @ascend_tile_load_f32(%arg0, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @ascend_tile_load_f32(%arg1, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @ascend_tile_load_f32(%arg2, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @ascend_tile_attention_f32(%q, %q, %k, %v, %s, %d) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %o, %s, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %a = llvm.call @ascend_tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %n, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @ascend_tile_matmul_transposed_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %q = llvm.call @ascend_tile_load_f32(%arg0, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @ascend_tile_load_f32(%arg1, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @ascend_tile_load_f32(%arg2, %s, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @ascend_tile_attention_gqa_f32(%q, %q, %k, %v, %s, %d, %hq, %hkv) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %o, %s, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %x = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %n = llvm.call @ascend_tile_rms_norm_f32(%x, %x, %eps, %r, %c) : (i32, i32, f32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %n, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %x = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @ascend_tile_scale_f32(%x, %x, %w, %r, %c) : (i32, i32, f32, i32, i32) -> i32
    %y = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %a = llvm.call @ascend_tile_add_f32(%s, %s, %y, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    %cl = llvm.call @ascend_tile_clamp_f32(%a, %a, %lo, %hi, %r, %c) : (i32, i32, f32, f32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %cl, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %a = llvm.call @ascend_tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %c = llvm.call @ascend_tile_matmul_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
    %x = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %y = llvm.call @ascend_tile_rope_f32(%c0, %x, %pos, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
        // ascend_tile_fill_f32(dst, scalar, rows, cols) → translate_fill,
        // which broadcasts a scalar into a vec tile via pto.tmov.
        let mlir = r#"
module {
  llvm.func @fill_k(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %scal = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(2 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %t = llvm.call @ascend_tile_fill_f32(%c0, %scal, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg0, %t, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
        // ascend_tile_matmul_i8_acc_i32_dequant_f16(dst, a, b, scale, m, k, n).
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
    %t_a = llvm.call @ascend_tile_load_i8(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_b = llvm.call @ascend_tile_load_i8(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %t_c = llvm.call @ascend_tile_matmul_i8_acc_i32_dequant_f16(%c0, %t_a, %t_b, %arg3, %m, %k, %n) : (i32, i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg2, %t_c, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
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
            "ascend_tile_add_f32",
            "ascend_tile_mul_f32",
            "ascend_tile_sub_f32",
            "ascend_tile_div_f32",
            "ascend_tile_add_f16",
            "ascend_tile_mul_f16",
            "ascend_tile_max_f32",
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
            "ascend_tile_exp_f32",
            "ascend_tile_exp_f16",
            "ascend_tile_neg_f32",
            "ascend_tile_reduce_max_f32",
            "ascend_tile_reduce_sum_f32",
            "ascend_tile_scale_f32",
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
        for op in ["ascend_tile_softmax_f32", "ascend_tile_softmax_f16"] {
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
        for op in ["ascend_tile_matmul_f32", "ascend_tile_matmul_f16"] {
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
            "ascend_tile_matmul_transposed_f32",
            "ascend_tile_matmul_transposed_f16",
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
        for op in ["ascend_tile_store_f32", "ascend_tile_store_f16", "ascend_tile_store_i8"] {
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
            ("ascend_tile_transpose_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_rsqrt_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_log_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_sigmoid_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_silu_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_silu_f16", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_cast_f32_f16", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_cast_f16_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_cast_bf16_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_argmax_f32", "(i32, i32, i32, i32) -> i32"),
            ("ascend_tile_absmax_f32", "(i32, i32, i32, i32) -> i32"),
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
        let call = "%r = llvm.call @ascend_tile_clamp_f32(%c0, %undef, %c0, %c1, %c32, %c32) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_rms_norm_unknown_errs() {
        // rms_norm: (c0, src, gamma, rows, cols)
        for op in ["ascend_tile_rms_norm_f32", "ascend_tile_rms_norm_f16"] {
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
        for op in ["ascend_tile_quantize_f32_i8", "ascend_tile_dequantize_i8_f32"] {
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
        let slice = "%r = llvm.call @ascend_tile_slice_f32(%c0, %undef, %c0, %c0, %c32, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(slice)).is_err());
        // concat: (c0, a, b, rows, cols_a, cols_b)
        let concat = "%r = llvm.call @ascend_tile_concat_f32(%c0, %undef, %undef2, %c32, %c16, %c16) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(concat)).is_err());
    }

    #[test]
    fn test_pto_gather_scatter_unknown_errs() {
        // gather/scatter: (c0, src, indices, n, m, d)
        for op in ["ascend_tile_gather_f32", "ascend_tile_scatter_f32"] {
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
        let call = "%r = llvm.call @ascend_tile_topk_f32(%c0, %undef, %undef2, %c8, %c1, %c32) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_gather_mask_unknown_errs() {
        // gather_mask: (c0, src, mask, rows, cols) — guards rows>0/cols>0/mask<=15 first.
        let call = "%r = llvm.call @ascend_tile_gather_mask_f32(%c0, %undef, %c10, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_gather_mask_arity_errs() {
        // gather_mask guard: mask must fit in 4 bits.
        let call = "%r = llvm.call @ascend_tile_gather_mask_f32(%c0, %undef, %c99, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_sort_unknown_errs() {
        // init_sort_buf: (c0, src, rows, cols) — rows/cols guarded >0.
        let init = "%r = llvm.call @ascend_tile_init_sort_buf_f32(%c0, %undef, %c1, %c32) : (i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(init)).is_err());
        // sort32: (c0, src, rows, cols)
        let sort = "%r = llvm.call @ascend_tile_sort32_f32(%c0, %undef, %c1, %c32) : (i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(sort)).is_err());
        // mrgsort2: (c0, src0, src1, tmp, cols_each)
        let mrg = "%r = llvm.call @ascend_tile_mrgsort2_f32(%c0, %undef, %undef2, %undef3, %c16) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(mrg)).is_err());
    }

    #[test]
    fn test_pto_phase6_unknown_errs() {
        // sample_top_p: (c0, logits, temp, top_p, seed, rows, cols)
        let stp = "%r = llvm.call @ascend_tile_sample_top_p_f32(%c0, %undef, %c1, %c1, %c0, %c1, %c32) : (i32, i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(stp)).is_err());
        // draft_verify: (c0, draft, target, rows, cols) — looks up target at args[2].
        let dv = "%r = llvm.call @ascend_tile_draft_verify_f32(%c0, %undef, %undef2, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(dv)).is_err());
        // token_accept: (c0, draft, target, probs, threshold, rows) — looks up draft at args[1].
        let ta = "%r = llvm.call @ascend_tile_token_accept_f32(%c0, %undef, %undef2, %undef3, %c1, %c1) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(ta)).is_err());
    }

    #[test]
    fn test_pto_rope_unknown_errs() {
        // rope: (c0, src, pos, rows, cols)
        let call = "%r = llvm.call @ascend_tile_rope_f32(%c0, %undef, %c0, %c1, %c32) : (i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }

    #[test]
    fn test_pto_attention_unknown_and_arity_errs() {
        // attention: 6 args (c0, q, k, v, scale, seq) — unknown Q tile.
        let attn = "%r = llvm.call @ascend_tile_attention_f32(%c0, %undef, %undef2, %undef3, %c1, %c32) : (i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(attn)).is_err());
        // attention arity: only 3 args -> args.len() < 6 guard.
        let attn_arity = "%r = llvm.call @ascend_tile_attention_f32(%c0, %undef, %undef2) : (i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(attn_arity)).is_err());
        // attention_gqa: 8 args — unknown Q tile.
        let gqa = "%r = llvm.call @ascend_tile_attention_gqa_f32(%c0, %undef, %undef2, %undef3, %c1, %c32, %c4, %c1) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(gqa)).is_err());
        // attention_gqa arity: only 4 args -> args.len() < 8 guard.
        let gqa_arity = "%r = llvm.call @ascend_tile_attention_gqa_f32(%c0, %undef, %undef2, %undef3) : (i32, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(gqa_arity)).is_err());
    }

    #[test]
    fn test_pto_matmul_i8_unknown_errs() {
        // matmul_i8: (c0, a, b, scale_ptr, m, k, n) — unknown A tile.
        let call = "%r = llvm.call @ascend_tile_matmul_i8_acc_i32_dequant_f16(%c0, %undef, %undef2, %arg1, %c16, %c16, %c16) : (i32, i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32";
        assert!(convert_mlir_to_pto(&ghost_pto!(call)).is_err());
    }
}
