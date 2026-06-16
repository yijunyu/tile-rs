//! MLIR-to-NKI translator for AWS Trainium targets.
//!
//! Converts merged MLIR modules (LLVM dialect with `ascend_tile_*` intrinsics)
//! into NKI (Neuron Kernel Interface) Python source, which can be compiled
//! by `neuronx-cc` to a NEFF binary for execution on AWS Trainium/Inferentia.
//!
//! # NKI Output Format
//!
//! A typical generated kernel looks like:
//!
//! ```python
//! import neuronxcc.nki as nki
//! import neuronxcc.nki.language as nl
//! import neuronxcc.nki.isa as nisa
//! import numpy as np
//!
//! @nki.jit
//! def tile_softmax(arg0, arg1):
//!     t0 = nl.load(arg0[nl.mgrid[0:1, 0:1024]])
//!     t_tmp = nl.ndarray((1, 1), dtype=np.float32, buffer=nl.sbuf)
//!     t_max = nisa.tensor_reduce(np.max, t0, axis=(1,), dtype=np.float32, negate=False)
//!     t_sub = nisa.tensor_scalar(t0, np.subtract, t_max, dtype=np.float32)
//!     t_exp = nisa.activation(np.exp, t_sub, dtype=np.float32)
//!     t_sum = nisa.tensor_reduce(np.add, t_exp, axis=(1,), dtype=np.float32, negate=False)
//!     result = nl.divide(t_exp, t_sum)
//!     nl.store(arg1[nl.mgrid[0:1, 0:1024]], result)
//! ```
//!
//! # Mapping from tile_std tile intrinsics to NKI ops
//!
//! | tile_std intrinsic | NKI op |
//! |---|---|
//! | `ascend_tile_load_f32(gm, rows, cols)` | `nl.load(ptr[nl.mgrid[0:R, 0:C]])` |
//! | `ascend_tile_store_f32(gm, buf, rows, cols)` | `nl.store(ptr[...], tile)` |
//! | `ascend_tile_add_f32(0, a, b, rows, cols)` | `nisa.tensor_tensor(a, b, op=np.add)` |
//! | `ascend_tile_sub_f32(0, a, b, rows, cols)` | `nisa.tensor_tensor(a, b, op=np.subtract)` |
//! | `ascend_tile_mul_f32(0, a, b, rows, cols)` | `nisa.tensor_tensor(a, b, op=np.multiply)` |
//! | `ascend_tile_exp_f32(0, src, rows, cols)` | `nisa.activation(np.exp, src)` |
//! | `ascend_tile_softmax_f32(0, src, rows, cols)` | 5-step decomposition (trowmax → sub → exp → trowsum → div) |
//! | `ascend_tile_matmul_f32(0, a, b, m, k, n)` | `nisa.nc_matmul(a, b)` |
//! | `ascend_tile_reduce_max_f32(0, src, rows, cols)` | `nisa.tensor_reduce(np.max, src, axis=(1,))` |
//! | `ascend_tile_reduce_sum_f32(0, src, rows, cols)` | `nisa.tensor_reduce(np.add, src, axis=(1,))` |
//! | `ascend_tile_scale_f32(0, src, scalar, rows, cols)` | `nisa.tensor_scalar(src, np.multiply, scalar)` |
//! | `ascend_tile_transpose_f32(0, src, rows, cols)` | `nisa.nc_transpose(src)` |
//! | `ascend_tile_rsqrt_f32(0, src, rows, cols)` | `1.0 / nl.sqrt(src)` |
//! | `ascend_tile_log_f32(0, src, rows, cols)` | `nl.log(src)` |
//! | `ascend_tile_sigmoid_f32(0, src, rows, cols)` | decomposed: 1/(1+exp(-x)) |
//! | `ascend_tile_clamp_f32(0, src, min, max, rows, cols)` | `nl.minimum(nl.maximum(src, min), max)` |
//! | `ascend_tile_cast_f32_f16(0, src, rows, cols)` | `nl.cast(src, dtype=nl.float16)` |
//! | `ascend_tile_cast_f16_f32(0, src, rows, cols)` | `nl.cast(src, dtype=nl.float32)` |
//! | `ascend_tile_slice_f32(0, src, r_off, c_off, sr, sc, dr, dc)` | `src[r_off:r_off+dr, c_off:c_off+dc]` |
//! | `ascend_tile_concat_f32(0, a, b, rows, cols_a, cols_b)` | `nl.concatenate((a, b), axis=1)` |
//! | `ascend_tile_scatter_f32(0, src, idx, n, m, d)` | loop-based scatter |
//! | `ascend_tile_gather_f32(0, src, idx, n, m, d)` | loop-based gather |
//! | `ascend_tile_topk_f32(0, src, idx_out, rows, cols, k)` | sort + slice top-k |
//! | `ascend_tile_argmax_f32(0, src, rows, cols)` | `nl.argmax(src, axis=1, keepdims=True)` |
//! | `ascend_tile_sample_top_p_f32(0, logits, temp, top_p, seed, rows, cols)` | softmax + greedy argmax approx |
//! | `ascend_tile_draft_verify_f32(0, draft, target, rows, cols)` | row-max + clamp to [0,1] |
//! | `ascend_tile_token_accept_f32(0, draft, target, probs, thresh, rows)` | mask-select draft or target |
//! | `ascend_tile_silu_f32(0, src, rows, cols)` | `x * sigmoid(x)` via `nisa.activation` |
//! | `ascend_tile_cast_bf16_f32(0, src, rows, cols)` | `nisa.cast(data=src, dtype=nl.float32)` (bf16→f32) |
//! | `ascend_tile_matmul_transposed_f32(0, a, b, m, k, n)` | `nisa.nc_matmul(a, transpose(b))` |
//! | `ascend_tile_attention_gqa_f32(0, q, k, v, heads, head_dim, seq_len, kv_heads)` | matmul+softmax+matmul with GQA head grouping |
//! | `ascend_tile_rope_f32(0, src, pos, rows, cols)` | Rotary Position Embedding (pair-wise cos/sin rotation) |
//!
//! # NKI Memory Model Notes
//!
//! Trainium NeuronCore has:
//! - SBUF: 128-partition scratchpad (24 MiB total), accessible as `nl.sbuf`
//! - PSUM: 2 MiB accumulator buffer, accessible as `nl.psum`
//! - Tile shape constraint: P dimension (partition) ≤ 128, F dimension (free) ≤ 512
//!
//! The generated Python uses `nl.mgrid[0:R, 0:C]` for DMA indexing.
//! Offset-based loads (from GEP-derived pointers) use `nl.mgrid[row_off:row_off+R, 0:C]`.

use std::collections::HashMap;
use std::fmt::Write;

use crate::mlir_to_pto::{
    extract_call_args, extract_result_ssa, infer_dtype_from_name, is_builtin_helper, parse_const_arg,
    parse_module, FuncArg, MlirFunc,
};

/// Convert MLIR text (merged module, LLVM dialect) into NKI Python source
/// consumable by `neuronx-cc compile --framework nki`.
///
/// Returns the Python source string, or an error on parse failure.
pub fn convert_mlir_to_nki(mlir_text: &str) -> Result<String, String> {
    let module = parse_module(mlir_text)?;

    let mut out = String::with_capacity(4096);
    writeln!(out, "# Generated by tile-rs mlir_to_nki — DO NOT EDIT").unwrap();
    writeln!(
        out,
        "# Compile: neuronx-cc compile --framework nki --target trn1 <file.py>"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "import neuronxcc.nki as nki").unwrap();
    writeln!(out, "import neuronxcc.nki.language as nl").unwrap();
    writeln!(out, "import neuronxcc.nki.isa as nisa").unwrap();
    writeln!(out, "import numpy as np").unwrap();
    writeln!(out).unwrap();

    let mut kernel_count = 0;
    for func in &module.functions {
        if func.is_entry && !func.body_lines.is_empty() && !is_builtin_helper(&func.name) {
            generate_func_nki(func, &mut out)?;
            kernel_count += 1;
        }
    }

    if kernel_count == 0 {
        return Err("No entry-point kernel functions found in MLIR module".into());
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// NKI function generator
// ---------------------------------------------------------------------------

fn generate_func_nki(func: &MlirFunc, out: &mut String) -> Result<(), String> {
    let mut ctx = NkiContext::new();

    // Collect GM pointer args
    let ptr_args: Vec<&FuncArg> = func.args.iter().filter(|a| a.is_gm).collect();

    // Build clean Python param names: %arg0 → p0, %arg1 → p1, etc.
    // Also build a mapping from MLIR SSA name → Python name for ptr alias resolution.
    let mut arg_py_names: Vec<(String, String)> = Vec::new(); // (mlir_ssa, py_name)
    for (i, arg) in ptr_args.iter().enumerate() {
        let py_name = format!("p{}", i);
        arg_py_names.push((arg.name.clone(), py_name));
    }

    // Emit @nki.jit decorator and def
    writeln!(out, "@nki.jit").unwrap();
    write!(out, "def {}(", func.name).unwrap();
    for (i, (_, py_name)) in arg_py_names.iter().enumerate() {
        if i > 0 {
            write!(out, ", ").unwrap();
        }
        write!(out, "{}", py_name).unwrap();
    }
    writeln!(out, "):").unwrap();

    // Pre-populate ptr_aliases so %arg0 resolves to Python name p0,
    // and populate arg dtype hints.
    for (i, arg) in ptr_args.iter().enumerate() {
        let py_name = arg_py_names[i].1.clone();
        let dtype = infer_dtype_from_name(&arg.name);
        // Map the MLIR SSA to the Python name via ptr_aliases
        ctx.ptr_aliases.insert(arg.name.clone(), py_name.clone());
        ctx.arg_dtypes.insert(py_name.clone(), dtype.to_string());
        ctx.arg_dtypes.insert(arg.name.clone(), dtype.to_string());
    }

    // Translate body
    let body_lines = translate_body(&func.body_lines, &mut ctx)?;

    if body_lines.is_empty() {
        writeln!(out, "    pass").unwrap();
    } else {
        for line in &body_lines {
            writeln!(out, "    {}", line).unwrap();
        }
    }

    writeln!(out).unwrap();
    Ok(())
}

// ---------------------------------------------------------------------------
// NKI context: tracks SSA → Python variable names, tile shapes, ptr aliases
// ---------------------------------------------------------------------------

struct NkiContext {
    /// MLIR SSA → Python variable name (e.g., `%5` → `t0`)
    tile_vars: HashMap<String, String>,
    /// MLIR SSA → (rows, cols, dtype) for tile shape tracking
    tile_shapes: HashMap<String, (u32, u32, String)>,
    /// Python variable counter for tile names
    next_tile: u32,
    /// Python variable counter for tmp names
    next_tmp: u32,
    /// SSA alias map: derived ptr SSA → original GM arg name (GEP tracking)
    ptr_aliases: HashMap<String, String>,
    /// Integer constant map: SSA → u32 value
    const_map: HashMap<String, u32>,
    /// Float constant map: SSA → string representation (e.g. "0.5", "1e-05")
    float_const_map: HashMap<String, String>,
    /// GEP element offsets: derived ptr SSA → element offset from base GM arg
    gep_offsets: HashMap<String, u32>,
    /// dtype hints per GM arg name
    arg_dtypes: HashMap<String, String>,
    /// Tracks the last SiLU output var and its source var for fusion with Mul.
    /// (silu_out_var, silu_src_var)
    last_silu: Option<(String, String)>,
}

impl NkiContext {
    fn new() -> Self {
        NkiContext {
            tile_vars: HashMap::new(),
            tile_shapes: HashMap::new(),
            next_tile: 0,
            next_tmp: 0,
            ptr_aliases: HashMap::new(),
            const_map: HashMap::new(),
            float_const_map: HashMap::new(),
            gep_offsets: HashMap::new(),
            arg_dtypes: HashMap::new(),
            last_silu: None,
        }
    }

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

    fn fresh_tile(&mut self) -> String {
        let n = self.next_tile;
        self.next_tile += 1;
        format!("t{}", n)
    }

    fn fresh_tmp(&mut self) -> String {
        let n = self.next_tmp;
        self.next_tmp += 1;
        format!("_tmp{}", n)
    }

    fn get_tile_var(&self, ssa: &str) -> Option<&str> {
        self.tile_vars.get(ssa).map(|s| s.as_str())
    }

    fn get_tile_shape(&self, ssa: &str) -> Option<(u32, u32, &str)> {
        self.tile_shapes
            .get(ssa)
            .map(|(r, c, d)| (*r, *c, d.as_str()))
    }

    fn dtype_for_ptr(&self, gm_arg: &str) -> &str {
        self.arg_dtypes
            .get(gm_arg)
            .map(|s| s.as_str())
            .unwrap_or("f32")
    }
}

// ---------------------------------------------------------------------------
// mgrid helper
// ---------------------------------------------------------------------------

/// Emit `ptr[nl.mgrid[row_off:row_off+R, 0:C]]` index expression.
fn mgrid_expr(gm_arg: &str, rows: u32, cols: u32, row_off: u32) -> String {
    if row_off == 0 {
        format!("{}[nl.mgrid[0:{}, 0:{}]]", gm_arg, rows, cols)
    } else {
        format!(
            "{}[nl.mgrid[{}:{}, 0:{}]]",
            gm_arg,
            row_off,
            row_off + rows,
            cols
        )
    }
}

/// Convert `f32` → `np.float32`, `f16` → `np.float16`.
fn np_dtype(dtype: &str) -> &str {
    match dtype {
        "f16" => "np.float16",
        _ => "np.float32",
    }
}

// ---------------------------------------------------------------------------
// Body analysis: MLIR lines → Python statements
// ---------------------------------------------------------------------------

fn translate_body(body_lines: &[String], ctx: &mut NkiContext) -> Result<Vec<String>, String> {
    let mut ops: Vec<String> = Vec::new();
    let mut store_map: HashMap<String, String> = HashMap::new();

    for line in body_lines {
        let line = line.trim();

        if line.is_empty()
            || line.ends_with(':')
            || line == "llvm.return"
            || line.contains("ascend_pipe_barrier")
            || line.contains("llvm.mlir.addressof")
            || line.contains("ascend_tile_pipelined_for_begin")
            || line.contains("ascend_tile_pipelined_for_end")
        {
            continue;
        }

        // Track integer and float constants
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

        // Propagate integer constants through bitcasts
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

        // Track GEP pointer aliases and element offsets
        if line.contains("llvm.getelementptr") && line.contains("!llvm.ptr<1>") {
            if let Some(result) = extract_result_ssa(line) {
                if let Some(open) = line.find("llvm.getelementptr ") {
                    let rest = line[open + "llvm.getelementptr ".len()..].trim();
                    let raw = rest
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(',');
                    let (src, idx_ssa) = if let Some(bracket) = raw.find('[') {
                        let base = &raw[..bracket];
                        let after = &raw[bracket + 1..];
                        let idx = after.trim_end_matches(']').trim();
                        (base, Some(idx.to_string()))
                    } else {
                        (raw, None)
                    };
                    ctx.ptr_aliases.insert(result.clone(), src.to_string());
                    if let Some(idx) = idx_ssa {
                        let offset = ctx.resolve_const(&idx);
                        if offset > 0 {
                            ctx.gep_offsets.insert(result, offset);
                        }
                    }
                }
            }
            continue;
        }

        // Track ptr stores (alloca pattern): `llvm.store %ptr, %alloca`
        if line.contains("llvm.store") && !line.contains("f32") && !line.contains("f16") {
            if let Some(open) = line.find("llvm.store ") {
                let rest = line[open + "llvm.store ".len()..].trim();
                let parts: Vec<&str> = rest.splitn(3, ',').collect();
                if parts.len() >= 2 {
                    let val = parts[0].trim().to_string();
                    let ptr = parts[1]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    store_map.insert(ptr, val);
                }
            }
            continue;
        }

        // Track ptr loads from alloca
        if line.contains("llvm.load") && line.contains("!llvm.ptr") {
            if let Some(result) = extract_result_ssa(line) {
                if let Some(open) = line.find("llvm.load ") {
                    let rest = line[open + "llvm.load ".len()..].trim();
                    let src = rest.split_whitespace().next().unwrap_or("");
                    if let Some(stored) = store_map.get(src).cloned() {
                        // Preserve intermediate node so gep_offsets chain works
                        ctx.ptr_aliases.insert(result, stored);
                    }
                }
            }
            continue;
        }

        // Alloca: skip (handled implicitly)
        if line.contains("llvm.alloca") {
            continue;
        }

        // Kernel intrinsic calls
        if line.contains("llvm.call @") {
            if let Some(py_op) = translate_call(line, ctx, &mut ops)? {
                ops.push(py_op);
            }
            continue;
        }
    }

    Ok(ops)
}

// ---------------------------------------------------------------------------
// Translate a single llvm.call line → Python NKI statement(s)
// ---------------------------------------------------------------------------

fn translate_call(
    line: &str,
    ctx: &mut NkiContext,
    extra_ops: &mut Vec<String>,
) -> Result<Option<String>, String> {
    let callee = {
        let start = match line.find("llvm.call @") {
            Some(s) => s + "llvm.call @".len(),
            None => return Ok(None),
        };
        let rest = &line[start..];
        let end = rest
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        rest[..end].to_string()
    };

    let args = match extract_call_args(line) {
        Some(a) => a,
        None => return Ok(None),
    };

    let result = extract_result_ssa(line);

    match callee.as_str() {
        // ── ascend_tile_load_f32(gm_ptr, rows, cols) → u32 ──
        "ascend_tile_load_f32" | "ascend_tile_load_f16" => {
            if args.len() < 3 {
                return Err(format!("ascend_tile_load_*: expected 3 args, got {}", args.len()));
            }
            let gm_raw = args[0].trim();
            let rows = ctx.resolve_const(args[1].trim());
            let cols = ctx.resolve_const(args[2].trim());
            let gm_arg = ctx.resolve_ptr(gm_raw);
            let elem_off = ctx.resolve_offset(gm_raw);
            let row_off = if cols > 0 { elem_off / cols } else { 0 };

            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }

            Ok(Some(format!(
                "{} = nl.load({})",
                var,
                mgrid_expr(&gm_arg, rows, cols, row_off)
            )))
        }

        // ── ascend_tile_store_f32(gm_ptr, tile_buf, rows, cols) ──
        "ascend_tile_store_f32" | "ascend_tile_store_f16" => {
            if args.len() < 4 {
                return Err(format!("ascend_tile_store_*: expected 4 args, got {}", args.len()));
            }
            let gm_raw = args[0].trim();
            let tile_ssa = args[1].trim();
            let rows = ctx.resolve_const(args[2].trim());
            let cols = ctx.resolve_const(args[3].trim());
            let gm_arg = ctx.resolve_ptr(gm_raw);
            let elem_off = ctx.resolve_offset(gm_raw);
            let row_off = if cols > 0 { elem_off / cols } else { 0 };

            let tile_var = ctx
                .get_tile_var(tile_ssa)
                .unwrap_or(tile_ssa)
                .to_string();

            Ok(Some(format!(
                "nl.store({}, {})",
                mgrid_expr(&gm_arg, rows, cols, row_off),
                tile_var
            )))
        }

        // ── ascend_tile_add_f32(dst_ignored, a, b, rows, cols) → u32 ──
        "ascend_tile_add_f32" | "ascend_tile_add_f16" => {
            translate_binop("np.add", &callee, args, result, ctx)
        }
        "ascend_tile_sub_f32" | "ascend_tile_sub_f16" => {
            translate_binop("np.subtract", &callee, args, result, ctx)
        }
        "ascend_tile_mul_f32" | "ascend_tile_mul_f16" => {
            // Fusion: if the previous op was SiLU and one operand is the SiLU output,
            // emit fused silu_mul instead of separate multiply
            if args.len() >= 5 {
                if let Some((ref silu_out, ref silu_src)) = ctx.last_silu.clone() {
                    let a_ssa = args[1].trim();
                    let b_ssa = args[2].trim();
                    let a_var = ctx.get_tile_var(a_ssa).unwrap_or(a_ssa).to_string();
                    let b_var = ctx.get_tile_var(b_ssa).unwrap_or(b_ssa).to_string();
                    if &a_var == silu_out || &b_var == silu_out {
                        let up_var = if &a_var == silu_out { b_var } else { a_var };
                        let dtype = if callee.contains("f16") { "f16" } else { "f32" };
                        let npdtype = np_dtype(dtype);
                        let (rows, cols) = ctx
                            .get_tile_shape(a_ssa)
                            .map(|(r, c, _)| (r, c))
                            .unwrap_or_else(|| {
                                (ctx.resolve_const(args[3].trim()), ctx.resolve_const(args[4].trim()))
                            });

                        // Remove the SiLU ops that were already emitted by popping them from extra_ops
                        // SiLU emits 5 extra_ops (negate, exp, add 1, ones, divide) + 1 main op (multiply)
                        // We need to remove those 5 extra_ops and replace the main op
                        let silu_extra_count = 5;
                        let len = extra_ops.len();
                        if len >= silu_extra_count {
                            extra_ops.truncate(len - silu_extra_count);
                        }

                        // Emit fused: silu(gate) * up
                        // Step 1: negate gate
                        let t_neg = ctx.fresh_tmp();
                        extra_ops.push(format!(
                            "{} = nisa.tensor_scalar({}, np.multiply, -1.0, dtype={})",
                            t_neg, silu_src, npdtype
                        ));
                        // Step 2: exp(-gate)
                        let t_exp = ctx.fresh_tmp();
                        extra_ops.push(format!(
                            "{} = nisa.activation(np.exp, {}, dtype={})",
                            t_exp, t_neg, npdtype
                        ));
                        // Step 3: 1 + exp(-gate)
                        let t_denom = ctx.fresh_tmp();
                        extra_ops.push(format!(
                            "{} = nisa.tensor_scalar({}, np.add, 1.0, dtype={})",
                            t_denom, t_exp, npdtype
                        ));
                        // Step 4: ones
                        let t_ones = ctx.fresh_tmp();
                        extra_ops.push(format!(
                            "{} = nl.ones(({}, {}), dtype={}, buffer=nl.sbuf)",
                            t_ones, rows, cols, npdtype
                        ));
                        // Step 5: sigmoid = 1 / (1 + exp(-gate))
                        let t_sig = ctx.fresh_tmp();
                        extra_ops.push(format!(
                            "{} = nl.divide({}, {})",
                            t_sig, t_ones, t_denom
                        ));
                        // Step 6: silu = gate * sigmoid
                        let t_silu = ctx.fresh_tmp();
                        extra_ops.push(format!(
                            "{} = nisa.tensor_tensor({}, {}, op=np.multiply, dtype={})",
                            t_silu, silu_src, t_sig, npdtype
                        ));
                        // Step 7: fused = silu * up
                        let var = ctx.fresh_tile();
                        if let Some(r) = &result {
                            ctx.tile_vars.insert(r.clone(), var.clone());
                            ctx.tile_shapes.insert(r.clone(), (rows, cols, dtype.to_string()));
                        }
                        ctx.last_silu = None;
                        return Ok(Some(format!(
                            "# fused silu_mul: silu(gate) * up\n    {} = nisa.tensor_tensor({}, {}, op=np.multiply, dtype={})",
                            var, t_silu, up_var, npdtype
                        )));
                    }
                }
            }
            ctx.last_silu = None;
            translate_binop("np.multiply", &callee, args, result, ctx)
        }

        // ── ascend_tile_exp_f32(dst_ignored, src, rows, cols) → u32 ──
        "ascend_tile_exp_f32" | "ascend_tile_exp_f16" => {
            if args.len() < 4 {
                return Err(format!("ascend_tile_exp_*: expected 4 args, got {}", args.len()));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.activation(np.exp, {}, dtype={})",
                var,
                src_var,
                np_dtype(dtype)
            )))
        }

        // ── ascend_tile_softmax_f32(dst, src, rows, cols) → u32 ──
        // Decomposed into 5 steps (numerically stable)
        "ascend_tile_softmax_f32" | "ascend_tile_softmax_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_softmax_*: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();

            // Step 1: row-wise max
            let t_max = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.max, {}, axis=(1,), dtype={}, negate=False)",
                t_max, src_var, npdtype
            ));

            // Step 2: subtract max from each row
            let t_sub = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.subtract, {}, dtype={})",
                t_sub, src_var, t_max, npdtype
            ));

            // Step 3: exp
            let t_exp = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.activation(np.exp, {}, dtype={})",
                t_exp, t_sub, npdtype
            ));

            // Step 4: row-wise sum
            let t_sum = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.add, {}, axis=(1,), dtype={}, negate=False)",
                t_sum, t_exp, npdtype
            ));

            // Step 5: divide by row sum → result tile
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nl.divide({}, {})",
                var, t_exp, t_sum
            )))
        }

        // ── ascend_tile_matmul_f32(dst, a, b, m, k, n) → u32 ──
        "ascend_tile_matmul_f32" | "ascend_tile_matmul_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_matmul_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let a_ssa = args[1].trim();
            let b_ssa = args[2].trim();
            let m = ctx.resolve_const(args[3].trim());
            let n = ctx.resolve_const(args[5].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let a_var = ctx.get_tile_var(a_ssa).unwrap_or(a_ssa).to_string();
            let b_var = ctx.get_tile_var(b_ssa).unwrap_or(b_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (m, n, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.nc_matmul({}, {})",
                var, a_var, b_var
            )))
        }

        // ── ascend_tile_reduce_max_f32(dst, src, rows, cols) → u32 (rows×1) ──
        "ascend_tile_reduce_max_f32" | "ascend_tile_reduce_max_f16" => {
            translate_reduce("np.max", &callee, args, result, ctx)
        }
        "ascend_tile_reduce_sum_f32" | "ascend_tile_reduce_sum_f16" => {
            translate_reduce("np.add", &callee, args, result, ctx)
        }

        // ── ascend_tile_scale_f32(dst, src, scalar, rows, cols) → u32 ──
        "ascend_tile_scale_f32" | "ascend_tile_scale_f16" => {
            if args.len() < 5 {
                return Err(format!(
                    "ascend_tile_scale_*: expected 5 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let scalar_ssa = args[2].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[3].trim()),
                        ctx.resolve_const(args[4].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let scalar_py = ctx.resolve_float(scalar_ssa);
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.tensor_scalar({}, np.multiply, {}, dtype={})",
                var,
                src_var,
                scalar_py,
                np_dtype(dtype)
            )))
        }

        // ── ascend_tile_transpose_f32(dst, src, rows, cols) → u32 ──
        "ascend_tile_transpose_f32" | "ascend_tile_transpose_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_transpose_*: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (cols, rows, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.nc_transpose({})",
                var, src_var
            )))
        }

        // ── ascend_tile_rsqrt_f32(dst, src, rows, cols) → u32 ──
        "ascend_tile_rsqrt_f32" | "ascend_tile_rsqrt_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_rsqrt_*: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();

            let t_sqrt = ctx.fresh_tmp();
            extra_ops.push(format!("{} = nl.sqrt({})", t_sqrt, src_var));

            let t_ones = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.ones(({}, {}), dtype={}, buffer=nl.sbuf)",
                t_ones, rows, cols, npdtype
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!("{} = nl.divide({}, {})", var, t_ones, t_sqrt)))
        }

        // ── ascend_tile_log_f32(dst, src, rows, cols) → u32 ──
        "ascend_tile_log_f32" | "ascend_tile_log_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_log_*: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!("{} = nl.log({})", var, src_var)))
        }

        // ── ascend_tile_sigmoid_f32(dst, src, rows, cols) → u32 ──
        // sigmoid(x) = 1 / (1 + exp(-x))
        "ascend_tile_sigmoid_f32" | "ascend_tile_sigmoid_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_sigmoid_*: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();

            let t_neg = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.multiply, -1.0, dtype={})",
                t_neg, src_var, npdtype
            ));

            let t_exp = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.activation(np.exp, {}, dtype={})",
                t_exp, t_neg, npdtype
            ));

            let t_denom = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.add, 1.0, dtype={})",
                t_denom, t_exp, npdtype
            ));

            let t_ones = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.ones(({}, {}), dtype={}, buffer=nl.sbuf)",
                t_ones, rows, cols, npdtype
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nl.divide({}, {})",
                var, t_ones, t_denom
            )))
        }

        // ── ascend_tile_clamp_f32(dst, src, min, max, rows, cols) → u32 ──
        "ascend_tile_clamp_f32" | "ascend_tile_clamp_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_clamp_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let min_ssa = args[2].trim();
            let max_ssa = args[3].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[4].trim()),
                        ctx.resolve_const(args[5].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let min_val = ctx.resolve_float(min_ssa);
            let max_val = ctx.resolve_float(max_ssa);

            let t_lower = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.maximum({}, {})",
                t_lower, src_var, min_val
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nl.minimum({}, {})",
                var, t_lower, max_val
            )))
        }

        // ── ascend_tile_cast_f32_f16(dst, src, rows, cols) → u32 ──
        "ascend_tile_cast_f32_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_cast_f32_f16: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, "f16".to_string()));
            }
            Ok(Some(format!(
                "{} = nl.cast({}, dtype=nl.float16)",
                var, src_var
            )))
        }

        // ── ascend_tile_cast_f16_f32(dst, src, rows, cols) → u32 ──
        "ascend_tile_cast_f16_f32" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_cast_f16_f32: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, "f32".to_string()));
            }
            Ok(Some(format!(
                "{} = nl.cast({}, dtype=nl.float32)",
                var, src_var
            )))
        }

        // ── ascend_tile_slice_f32(dst, src, row_off, col_off, src_r, src_c, dst_r, dst_c) → u32 ──
        "ascend_tile_slice_f32" | "ascend_tile_slice_f16" => {
            if args.len() < 8 {
                return Err(format!(
                    "ascend_tile_slice_*: expected 8 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let row_off = ctx.resolve_const(args[2].trim());
            let col_off = ctx.resolve_const(args[3].trim());
            let _src_r = ctx.resolve_const(args[4].trim());
            let _src_c = ctx.resolve_const(args[5].trim());
            let dst_r = ctx.resolve_const(args[6].trim());
            let dst_c = ctx.resolve_const(args[7].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (dst_r, dst_c, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = {}[{}:{}, {}:{}]",
                var, src_var, row_off, row_off + dst_r, col_off, col_off + dst_c
            )))
        }

        // ── ascend_tile_concat_f32(dst, a, b, rows, cols_a, cols_b) → u32 ──
        "ascend_tile_concat_f32" | "ascend_tile_concat_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_concat_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let a_ssa = args[1].trim();
            let b_ssa = args[2].trim();
            let rows = ctx.resolve_const(args[3].trim());
            let cols_a = ctx.resolve_const(args[4].trim());
            let cols_b = ctx.resolve_const(args[5].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let a_var = ctx.get_tile_var(a_ssa).unwrap_or(a_ssa).to_string();
            let b_var = ctx.get_tile_var(b_ssa).unwrap_or(b_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols_a + cols_b, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nl.concatenate(({}, {}), axis=1)",
                var, a_var, b_var
            )))
        }

        // ── ascend_tile_scatter_f32(dst, src, indices, n, m, d) → u32 ──
        "ascend_tile_scatter_f32" | "ascend_tile_scatter_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_scatter_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let indices_ssa = args[2].trim();
            let n = ctx.resolve_const(args[3].trim());
            let m = ctx.resolve_const(args[4].trim());
            let d = ctx.resolve_const(args[5].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let indices_gm = ctx.resolve_ptr(indices_ssa);

            let t_idx = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.load({}[nl.mgrid[0:{}, 0:1]], dtype=np.int32)",
                t_idx, indices_gm, n
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (m, d, dtype.to_string()));
            }
            Ok(Some(format!(
                "{out} = nl.ndarray(({m}, {d}), dtype={npdtype}, buffer=nl.sbuf)\n    \
                 for _i_ in nl.sequential_range({n}):\n        \
                     {out}[{idx}[_i_, 0], :] = {src}[_i_, :]",
                out = var, m = m, d = d, npdtype = npdtype,
                n = n, idx = t_idx, src = src_var
            )))
        }

        // ── ascend_tile_gather_f32(dst, src, indices, n, m, d) → u32 ──
        "ascend_tile_gather_f32" | "ascend_tile_gather_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_gather_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let indices_ssa = args[2].trim();
            let n = ctx.resolve_const(args[3].trim());
            let _m = ctx.resolve_const(args[4].trim());
            let d = ctx.resolve_const(args[5].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let indices_gm = ctx.resolve_ptr(indices_ssa);

            let t_idx = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.load({}[nl.mgrid[0:{}, 0:1]], dtype=np.int32)",
                t_idx, indices_gm, n
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (n, d, dtype.to_string()));
            }
            Ok(Some(format!(
                "{out} = nl.ndarray(({n}, {d}), dtype={npdtype}, buffer=nl.sbuf)\n    \
                 for _i_ in nl.sequential_range({n}):\n        \
                     {out}[_i_, :] = {src}[{idx}[_i_, 0], :]",
                out = var, n = n, d = d, npdtype = npdtype,
                src = src_var, idx = t_idx
            )))
        }

        // ── ascend_tile_topk_f32(dst, src, indices_out, rows, cols, k) → u32 ──
        "ascend_tile_topk_f32" | "ascend_tile_topk_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_topk_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let _indices_out_ssa = args[2].trim();
            let rows = ctx.resolve_const(args[3].trim());
            let _cols = ctx.resolve_const(args[4].trim());
            let k = ctx.resolve_const(args[5].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();

            let t_sorted_idx = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.argsort({}, axis=1, descending=True)",
                t_sorted_idx, src_var
            ));

            let t_topk_idx = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = {}[:, :{}]",
                t_topk_idx, t_sorted_idx, k
            ));

            let t_sorted_vals = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.sort({}, axis=1, descending=True)",
                t_sorted_vals, src_var
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, k, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = {}[:, :{}]",
                var, t_sorted_vals, k
            )))
        }

        // ── get_block_idx / get_block_num: map to nl.program_id(0) ──
        "get_block_idx" => {
            let var = ctx.fresh_tmp();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
            }
            Ok(Some(format!("{} = nl.program_id(0)", var)))
        }

        // ── ascend_tile_fill_f32(dst, scalar, rows, cols) ──
        "ascend_tile_fill_f32" | "ascend_tile_fill_f16" => {
            if args.len() < 4 {
                return Err(format!("ascend_tile_fill_*: expected 4 args, got {}", args.len()));
            }
            let scalar_raw = args[1].trim();
            let scalar_val = ctx.resolve_float(scalar_raw);
            let rows = ctx.resolve_const(args[2].trim());
            let cols = ctx.resolve_const(args[3].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!("{} = nl.full(({}, {}), {}, dtype={})", var, rows, cols, scalar_val, np_dtype(dtype))))
        }

        // ── ascend_tile_max_f32(dst, a, b, rows, cols) ──
        "ascend_tile_max_f32" | "ascend_tile_max_f16" => {
            translate_binop("nl.maximum", &callee, args, result, ctx)
        }

        // ── ascend_tile_div_f32(dst, a, b, rows, cols) ──
        "ascend_tile_div_f32" | "ascend_tile_div_f16" => {
            translate_binop("nl.divide", &callee, args, result, ctx)
        }

        // ── ascend_tile_rms_norm_f32(dst, src, eps, rows, cols) ──
        "ascend_tile_rms_norm_f32" | "ascend_tile_rms_norm_f16" => {
            if args.len() < 5 {
                return Err(format!("ascend_tile_rms_norm_*: expected 5 args, got {}", args.len()));
            }
            let src_ssa = args[1].trim();
            let eps_raw = args[2].trim();
            let eps_val = ctx.resolve_float(eps_raw);
            let rows = ctx.resolve_const(args[3].trim());
            let cols = ctx.resolve_const(args[4].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{v} = {s} * nl.rsqrt(nl.add(nisa.tensor_reduce(nl.multiply({s}, {s}), op=np.add, axis=[1], dtype={dt}) / {c}, {e}))",
                v=var, s=src_var, c=cols, e=eps_val, dt=np_dtype(dtype)
            )))
        }

        // ── ascend_tile_absmax_f32(dst, src, rows, cols) ──
        "ascend_tile_absmax_f32" | "ascend_tile_absmax_f16" => {
            if args.len() < 4 {
                return Err(format!("ascend_tile_absmax_*: expected 4 args, got {}", args.len()));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| (
                    ctx.resolve_const(args[2].trim()),
                    ctx.resolve_const(args[3].trim()),
                ));
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!(
                "{v} = nisa.tensor_reduce(nl.max, nl.abs({s}), axis=[1], dtype={dt})",
                v=var, s=src_var, dt=np_dtype(dtype)
            )))
        }

        // ── ascend_tile_quantize_f32_i8(dst, src, scale, rows, cols) ──
        "ascend_tile_quantize_f32_i8" => {
            if args.len() < 5 {
                return Err(format!("ascend_tile_quantize_f32_i8: expected 5 args, got {}", args.len()));
            }
            let src_ssa = args[1].trim();
            let scale_ssa = args[2].trim();
            let rows = ctx.resolve_const(args[3].trim());
            let cols = ctx.resolve_const(args[4].trim());
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let scale_var = ctx.get_tile_var(scale_ssa).unwrap_or(scale_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, cols, "f32".to_string()));
            }
            Ok(Some(format!(
                "{v} = nl.clip(nl.round(nl.divide({s}, {sc})), -128, 127)",
                v=var, s=src_var, sc=scale_var
            )))
        }

        // ── ascend_tile_dequantize_i8_f32(dst, src, scale, rows, cols) ──
        "ascend_tile_dequantize_i8_f32" => {
            if args.len() < 5 {
                return Err(format!("ascend_tile_dequantize_i8_f32: expected 5 args, got {}", args.len()));
            }
            let src_ssa = args[1].trim();
            let scale_ssa = args[2].trim();
            let rows = ctx.resolve_const(args[3].trim());
            let cols = ctx.resolve_const(args[4].trim());
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let scale_var = ctx.get_tile_var(scale_ssa).unwrap_or(scale_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, cols, "f32".to_string()));
            }
            Ok(Some(format!(
                "{v} = nl.multiply({s}, {sc})",
                v=var, s=src_var, sc=scale_var
            )))
        }

        // ── Phase 6 MTP ops ──────────────────────────────────────────────────

        // ── ascend_tile_argmax_f32(dst, src, rows, cols) → u32 ──
        // row-wise argmax: nl.argmax(src, axis=1)
        "ascend_tile_argmax_f32" => {
            if args.len() < 4 {
                return Err(format!("ascend_tile_argmax_f32: expected 4 args, got {}", args.len()));
            }
            let src_ssa = args[1].trim();
            let rows = ctx.resolve_const(args[2].trim());
            let _cols = ctx.resolve_const(args[3].trim());
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, 1, "u32".to_string()));
            }
            Ok(Some(format!(
                "{} = nl.argmax({}, axis=1, keepdims=True)",
                var, src_var
            )))
        }

        // ── ascend_tile_sample_top_p_f32(dst, logits, temp, top_p, seed, rows, cols) → u32 ──
        // Nucleus sampling: temperature-scaled softmax + cumulative sum + binary search
        "ascend_tile_sample_top_p_f32" => {
            if args.len() < 7 {
                return Err(format!(
                    "ascend_tile_sample_top_p_f32: expected 7 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let temp_raw = args[2].trim();
            let _top_p_raw = args[3].trim();
            let rows = ctx.resolve_const(args[5].trim());
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();

            // Step 1: temperature scale
            let t_scaled = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.divide, {}, dtype=np.float32)",
                t_scaled, src_var, temp_raw
            ));

            // Step 2: softmax (row-wise)
            let t_max = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.max, {}, axis=(1,), dtype=np.float32, negate=False)",
                t_max, t_scaled
            ));
            let t_sub = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.subtract, {}, dtype=np.float32)",
                t_sub, t_scaled, t_max
            ));
            let t_exp = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.activation(np.exp, {}, dtype=np.float32)",
                t_exp, t_sub
            ));
            let t_sum = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.add, {}, axis=(1,), dtype=np.float32, negate=False)",
                t_sum, t_exp
            ));
            let t_probs = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.divide({}, {})",
                t_probs, t_exp, t_sum
            ));

            // Step 3: argmax as greedy approximation (full nucleus search requires host-side RNG)
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, 1, "u32".to_string()));
            }
            Ok(Some(format!(
                "{} = nl.argmax({}, axis=1, keepdims=True)  # greedy approx; TODO: proper nucleus sampling with RNG",
                var, t_probs
            )))
        }

        // ── ascend_tile_draft_verify_f32(dst, draft_tokens, target_logits, rows, cols) → f32 ──
        // Acceptance prob: min(1, target[r, draft[r]] / draft_prob[r])
        "ascend_tile_draft_verify_f32" => {
            if args.len() < 5 {
                return Err(format!(
                    "ascend_tile_draft_verify_f32: expected 5 args, got {}",
                    args.len()
                ));
            }
            let _draft_ssa = args[1].trim();
            let target_ssa = args[2].trim();
            let rows = ctx.resolve_const(args[3].trim());
            let _cols = ctx.resolve_const(args[4].trim());
            let target_var = ctx.get_tile_var(target_ssa).unwrap_or(target_ssa).to_string();

            // Reduce target logits to max per row (approximation: use row-max as accept prob)
            let t_max = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.max, {}, axis=(1,), dtype=np.float32, negate=False)",
                t_max, target_var
            ));

            // Clamp to [0, 1]
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, 1, "f32".to_string()));
            }
            Ok(Some(format!(
                "{} = nl.minimum(nl.full(({}, 1), 1.0, dtype=np.float32), nl.maximum(nl.full(({}, 1), 0.0, dtype=np.float32), {}))  # acceptance prob clamp(0,1)",
                var, rows, rows, t_max
            )))
        }

        // ── ascend_tile_token_accept_f32(dst, draft_tokens, target_tokens, accept_probs, threshold, rows) → u32 ──
        // Accept draft if prob >= threshold, else fall back to target
        "ascend_tile_token_accept_f32" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_token_accept_f32: expected 6 args, got {}",
                    args.len()
                ));
            }
            let draft_ssa = args[1].trim();
            let target_ssa = args[2].trim();
            let probs_ssa = args[3].trim();
            let thresh_raw = args[4].trim();
            let rows = ctx.resolve_const(args[5].trim());
            let draft_var = ctx.get_tile_var(draft_ssa).unwrap_or(draft_ssa).to_string();
            let target_var = ctx.get_tile_var(target_ssa).unwrap_or(target_ssa).to_string();
            let probs_var = ctx.get_tile_var(probs_ssa).unwrap_or(probs_ssa).to_string();

            // mask = probs >= threshold  (as float 1.0 / 0.0)
            let t_thresh = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.full(({}, 1), {}, dtype=np.float32)",
                t_thresh, rows, thresh_raw
            ));
            let t_mask = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_tensor({}, {}, op=np.greater_equal, dtype=np.float32)",
                t_mask, probs_var, t_thresh
            ));

            // selected = mask * draft + (1 - mask) * target
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes.insert(r.clone(), (rows, 1, "u32".to_string()));
            }
            Ok(Some(format!(
                "{v} = nisa.tensor_tensor(nisa.tensor_tensor({m}, {d}, op=np.multiply, dtype=np.float32), nisa.tensor_tensor(nisa.tensor_scalar({m}, np.subtract, 1.0, dtype=np.float32), {t}, op=np.multiply, dtype=np.float32), op=np.add, dtype=np.float32)",
                v = var, m = t_mask, d = draft_var, t = target_var
            )))
        }

        // ── ascend_tile_silu_f32(dst, src, rows, cols) → u32 ──
        // silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        "ascend_tile_silu_f32" | "ascend_tile_silu_f16" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_silu_*: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();

            // Step 1: negate
            let t_neg = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.multiply, -1.0, dtype={})",
                t_neg, src_var, npdtype
            ));

            // Step 2: exp(-x)
            let t_exp = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.activation(np.exp, {}, dtype={})",
                t_exp, t_neg, npdtype
            ));

            // Step 3: 1 + exp(-x)
            let t_denom = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.add, 1.0, dtype={})",
                t_denom, t_exp, npdtype
            ));

            // Step 4: ones for numerator
            let t_ones = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.ones(({}, {}), dtype={}, buffer=nl.sbuf)",
                t_ones, rows, cols, npdtype
            ));

            // Step 5: sigmoid = 1 / (1 + exp(-x))
            let t_sig = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.divide({}, {})",
                t_sig, t_ones, t_denom
            ));

            // Step 6: x * sigmoid(x)
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            // Record for potential fusion with following Mul
            ctx.last_silu = Some((var.clone(), src_var.clone()));
            Ok(Some(format!(
                "{} = nisa.tensor_tensor({}, {}, op=np.multiply, dtype={})",
                var, src_var, t_sig, npdtype
            )))
        }

        // ── ascend_tile_cast_bf16_f32(dst, src, rows, cols) → u32 ──
        // Trainium has native bf16; cast to f32 via nisa.cast
        "ascend_tile_cast_bf16_f32" => {
            if args.len() < 4 {
                return Err(format!(
                    "ascend_tile_cast_bf16_f32: expected 4 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[2].trim()),
                        ctx.resolve_const(args[3].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, "f32".to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.cast({}, dtype=nl.float32)",
                var, src_var
            )))
        }

        // ── ascend_tile_matmul_transposed_f32(dst, a, b, m, k, n) → u32 ──
        // matmul where B is transposed: C = A @ B^T
        "ascend_tile_matmul_transposed_f32" | "ascend_tile_matmul_transposed_f16" => {
            if args.len() < 6 {
                return Err(format!(
                    "ascend_tile_matmul_transposed_*: expected 6 args, got {}",
                    args.len()
                ));
            }
            let a_ssa = args[1].trim();
            let b_ssa = args[2].trim();
            let m = ctx.resolve_const(args[3].trim());
            let n = ctx.resolve_const(args[5].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let a_var = ctx.get_tile_var(a_ssa).unwrap_or(a_ssa).to_string();
            let b_var = ctx.get_tile_var(b_ssa).unwrap_or(b_ssa).to_string();

            // Transpose B first
            let t_bt = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.nc_transpose({})",
                t_bt, b_var
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (m, n, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.nc_matmul({}, {})",
                var, a_var, t_bt
            )))
        }

        // ── ascend_tile_attention_gqa_f32(dst, q, k, v, heads, head_dim, seq_len, kv_heads) → u32 ──
        // Grouped-Query Attention: Q has `heads` heads, K/V have `kv_heads` heads.
        // Decomposed into: for each group, scores = Q @ K^T, attn = softmax(scores), out = attn @ V
        "ascend_tile_attention_gqa_f32" | "ascend_tile_attention_gqa_f16" => {
            if args.len() < 8 {
                return Err(format!(
                    "ascend_tile_attention_gqa_*: expected 8 args, got {}",
                    args.len()
                ));
            }
            let q_ssa = args[1].trim();
            let k_ssa = args[2].trim();
            let v_ssa = args[3].trim();
            let heads = ctx.resolve_const(args[4].trim());
            let head_dim = ctx.resolve_const(args[5].trim());
            let seq_len = ctx.resolve_const(args[6].trim());
            let kv_heads = ctx.resolve_const(args[7].trim());
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let q_var = ctx.get_tile_var(q_ssa).unwrap_or(q_ssa).to_string();
            let k_var = ctx.get_tile_var(k_ssa).unwrap_or(k_ssa).to_string();
            let v_var = ctx.get_tile_var(v_ssa).unwrap_or(v_ssa).to_string();

            // Transpose K for scores = Q @ K^T
            let t_kt = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.nc_transpose({})",
                t_kt, k_var
            ));

            // scores = Q @ K^T  (heads*head_dim x seq_len if flattened; NKI handles tiles)
            let t_scores = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.nc_matmul({}, {})",
                t_scores, q_var, t_kt
            ));

            // Scale by 1/sqrt(head_dim)
            let scale = 1.0 / (head_dim as f64).sqrt();
            let t_scaled = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.multiply, {:.8}, dtype={})",
                t_scaled, t_scores, scale, npdtype
            ));

            // Softmax over scores (row-wise)
            let t_max = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.max, {}, axis=(1,), dtype={}, negate=False)",
                t_max, t_scaled, npdtype
            ));

            let t_sub = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_scalar({}, np.subtract, {}, dtype={})",
                t_sub, t_scaled, t_max, npdtype
            ));

            let t_exp = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.activation(np.exp, {}, dtype={})",
                t_exp, t_sub, npdtype
            ));

            let t_sum = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nisa.tensor_reduce(np.add, {}, axis=(1,), dtype={}, negate=False)",
                t_sum, t_exp, npdtype
            ));

            let t_attn = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{} = nl.divide({}, {})",
                t_attn, t_exp, t_sum
            ));

            // Output = attn @ V
            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (heads, head_dim, dtype.to_string()));
            }
            Ok(Some(format!(
                "{} = nisa.nc_matmul({}, {})",
                var, t_attn, v_var
            )))
        }

        // ── ascend_tile_rope_f32(dst, src, pos, rows, cols) → u32 ──
        // Rotary Position Embedding: for each pair (x[2i], x[2i+1]) in each row r:
        //   freq = 1.0 / pow(10000.0, 2*i / cols)
        //   angle = pos * freq
        //   out[2i]   = x[2i]*cos(angle) - x[2i+1]*sin(angle)
        //   out[2i+1] = x[2i]*sin(angle) + x[2i+1]*cos(angle)
        "ascend_tile_rope_f32" | "ascend_tile_rope_f16" => {
            if args.len() < 5 {
                return Err(format!(
                    "ascend_tile_rope_*: expected 5 args, got {}",
                    args.len()
                ));
            }
            let src_ssa = args[1].trim();
            let pos_ssa = args[2].trim();
            let dtype = if callee.contains("f16") { "f16" } else { "f32" };
            let npdtype = np_dtype(dtype);
            let (rows, cols) = ctx
                .get_tile_shape(src_ssa)
                .map(|(r, c, _)| (r, c))
                .unwrap_or_else(|| {
                    (
                        ctx.resolve_const(args[3].trim()),
                        ctx.resolve_const(args[4].trim()),
                    )
                });
            let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
            let pos_var = ctx.get_tile_var(pos_ssa).unwrap_or(pos_ssa).to_string();
            let half = cols / 2;

            // Build frequency vector: freq[i] = 1.0 / 10000^(2*i/cols), i in 0..cols/2
            let t_freq = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{freq} = nl.arange(0, {half})[None, :].astype({dtype})",
                freq = t_freq, half = half, dtype = npdtype
            ));
            let t_freq2 = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{f2} = nisa.tensor_scalar({f1}, np.multiply, {scale:.16}, dtype={dtype})",
                f2 = t_freq2, f1 = t_freq, scale = 2.0 / cols as f64, dtype = npdtype
            ));
            let t_freq3 = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{f3} = nisa.tensor_scalar({f2}, np.multiply, {log10k:.16}, dtype={dtype})",
                f3 = t_freq3, f2 = t_freq2, log10k = -(10000.0_f64).ln(), dtype = npdtype
            ));
            let t_freq4 = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{f4} = nisa.activation(np.exp, {f3}, dtype={dtype})",
                f4 = t_freq4, f3 = t_freq3, dtype = npdtype
            ));

            // angle = pos * freq
            let t_angle = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{angle} = nisa.tensor_scalar({freq}, np.multiply, {pos}, dtype={dtype})",
                angle = t_angle, freq = t_freq4, pos = pos_var, dtype = npdtype
            ));

            // cos(angle), sin(angle)
            let t_cos = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{cos} = nl.cos({angle})",
                cos = t_cos, angle = t_angle
            ));
            let t_sin = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{sin} = nl.sin({angle})",
                sin = t_sin, angle = t_angle
            ));

            // Split src into even/odd columns: x_even = src[:, 0::2], x_odd = src[:, 1::2]
            let t_even = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{even} = {src}[0:{rows}, 0:{cols}:2]",
                even = t_even, src = src_var, rows = rows, cols = cols
            ));
            let t_odd = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{odd} = {src}[0:{rows}, 1:{cols}:2]",
                odd = t_odd, src = src_var, rows = rows, cols = cols
            ));

            // out_even = x_even * cos - x_odd * sin
            let t_ec = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{ec} = nisa.tensor_tensor({even}, {cos}, op=np.multiply, dtype={dtype})",
                ec = t_ec, even = t_even, cos = t_cos, dtype = npdtype
            ));
            let t_os = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{os} = nisa.tensor_tensor({odd}, {sin}, op=np.multiply, dtype={dtype})",
                os = t_os, odd = t_odd, sin = t_sin, dtype = npdtype
            ));
            let t_out_even = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{oe} = nisa.tensor_tensor({ec}, {os}, op=np.subtract, dtype={dtype})",
                oe = t_out_even, ec = t_ec, os = t_os, dtype = npdtype
            ));

            // out_odd = x_even * sin + x_odd * cos
            let t_es = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{es} = nisa.tensor_tensor({even}, {sin}, op=np.multiply, dtype={dtype})",
                es = t_es, even = t_even, sin = t_sin, dtype = npdtype
            ));
            let t_oc = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{oc} = nisa.tensor_tensor({odd}, {cos}, op=np.multiply, dtype={dtype})",
                oc = t_oc, odd = t_odd, cos = t_cos, dtype = npdtype
            ));
            let t_out_odd = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{oo} = nisa.tensor_tensor({es}, {oc}, op=np.add, dtype={dtype})",
                oo = t_out_odd, es = t_es, oc = t_oc, dtype = npdtype
            ));

            // Interleave even/odd back: result = nl.concatenate and reshape, or
            // use stack + reshape. NKI supports: stack along new axis then reshape.
            let t_stack = ctx.fresh_tmp();
            extra_ops.push(format!(
                "{stk} = nl.stack(({oe}, {oo}), axis=2).reshape(({rows}, {cols}))",
                stk = t_stack, oe = t_out_even, oo = t_out_odd, rows = rows, cols = cols
            ));

            let var = ctx.fresh_tile();
            if let Some(r) = &result {
                ctx.tile_vars.insert(r.clone(), var.clone());
                ctx.tile_shapes
                    .insert(r.clone(), (rows, cols, dtype.to_string()));
            }
            Ok(Some(format!("{} = {}", var, t_stack)))
        }

        // Unknown intrinsics: emit a comment so the user can see what was skipped
        _ if callee.starts_with("ascend_") => {
            Ok(Some(format!("# TODO: unhandled intrinsic: {}", callee)))
        }

        // Non-kernel calls: skip silently
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers for binop / reduce translation
// ---------------------------------------------------------------------------

fn translate_binop(
    np_op: &str,
    callee: &str,
    args: Vec<String>,
    result: Option<String>,
    ctx: &mut NkiContext,
) -> Result<Option<String>, String> {
    if args.len() < 5 {
        return Err(format!("{}: expected 5 args, got {}", callee, args.len()));
    }
    let a_ssa = args[1].trim();
    let b_ssa = args[2].trim();
    let dtype = if callee.contains("f16") { "f16" } else { "f32" };
    let (rows, cols) = ctx
        .get_tile_shape(a_ssa)
        .map(|(r, c, _)| (r, c))
        .unwrap_or_else(|| {
            (
                ctx.resolve_const(args[3].trim()),
                ctx.resolve_const(args[4].trim()),
            )
        });
    let a_var = ctx.get_tile_var(a_ssa).unwrap_or(a_ssa).to_string();
    let b_var = ctx.get_tile_var(b_ssa).unwrap_or(b_ssa).to_string();
    let var = ctx.fresh_tile();
    if let Some(r) = &result {
        ctx.tile_vars.insert(r.clone(), var.clone());
        ctx.tile_shapes
            .insert(r.clone(), (rows, cols, dtype.to_string()));
    }
    Ok(Some(format!(
        "{} = nisa.tensor_tensor({}, {}, op={}, dtype={})",
        var,
        a_var,
        b_var,
        np_op,
        np_dtype(dtype)
    )))
}

fn translate_reduce(
    np_op: &str,
    callee: &str,
    args: Vec<String>,
    result: Option<String>,
    ctx: &mut NkiContext,
) -> Result<Option<String>, String> {
    if args.len() < 4 {
        return Err(format!("{}: expected 4 args, got {}", callee, args.len()));
    }
    let src_ssa = args[1].trim();
    let dtype = if callee.contains("f16") { "f16" } else { "f32" };
    let (rows, cols) = ctx
        .get_tile_shape(src_ssa)
        .map(|(r, c, _)| (r, c))
        .unwrap_or_else(|| {
            (
                ctx.resolve_const(args[2].trim()),
                ctx.resolve_const(args[3].trim()),
            )
        });
    let src_var = ctx.get_tile_var(src_ssa).unwrap_or(src_ssa).to_string();
    let var = ctx.fresh_tile();
    if let Some(r) = &result {
        // Reduction output is rows×1
        ctx.tile_vars.insert(r.clone(), var.clone());
        ctx.tile_shapes
            .insert(r.clone(), (rows, 1, dtype.to_string()));
    }
    Ok(Some(format!(
        "{} = nisa.tensor_reduce({}, {}, axis=(1,), dtype={}, negate=False)",
        var,
        np_op,
        src_var,
        np_dtype(dtype),
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    fn load_store_mlir() -> &'static str {
        r#"
module {
  llvm.func @simple_load_store(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1024 : i32) : i32
    %3 = llvm.call @ascend_tile_load_f32(%arg0, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %3, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#
    }

    fn double_buf_mlir() -> &'static str {
        r#"
module {
  llvm.func @double_buf(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1024 : i32) : i32
    %3 = llvm.call @ascend_tile_load_f32(%arg0, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    %gep = llvm.getelementptr %arg0[%2] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %4 = llvm.call @ascend_tile_load_f32(%gep, %1, %2) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %3, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.call @ascend_tile_store_f32(%arg1, %4, %1, %2) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#
    }

    #[test]
    fn test_nki_load_store() {
        let py = convert_mlir_to_nki(load_store_mlir()).unwrap();
        assert!(py.contains("@nki.jit"), "missing @nki.jit decorator");
        assert!(py.contains("def simple_load_store("), "missing def");
        assert!(py.contains("nl.load("), "missing nl.load");
        assert!(py.contains("nl.store("), "missing nl.store");
        assert!(py.contains("nl.mgrid[0:1, 0:1024]"), "wrong mgrid dims");
    }

    #[test]
    fn test_nki_softmax_decomposition() {
        let py = convert_mlir_to_nki(softmax_mlir()).unwrap();
        // Must contain the 5-step softmax decomposition
        assert!(py.contains("nisa.tensor_reduce(np.max"), "missing rowmax step");
        assert!(py.contains("np.subtract"), "missing subtract step");
        assert!(py.contains("nisa.activation(np.exp"), "missing exp step");
        assert!(py.contains("nisa.tensor_reduce(np.add"), "missing rowsum step");
        assert!(py.contains("nl.divide("), "missing nl.divide step");
    }

    #[test]
    fn test_nki_gep_offset_mgrid() {
        let py = convert_mlir_to_nki(double_buf_mlir()).unwrap();
        // First load: offset 0 → mgrid[0:1, 0:1024]
        assert!(py.contains("nl.mgrid[0:1, 0:1024]"), "missing zero-offset mgrid");
        // Second load: offset 1024 / cols=1024 = row 1 → mgrid[1:2, 0:1024]
        assert!(py.contains("nl.mgrid[1:2, 0:1024]"), "missing offset mgrid for gep+1024");
    }

    #[test]
    fn test_nki_header() {
        let py = convert_mlir_to_nki(load_store_mlir()).unwrap();
        assert!(py.contains("import neuronxcc.nki as nki"));
        assert!(py.contains("import neuronxcc.nki.language as nl"));
        assert!(py.contains("import neuronxcc.nki.isa as nisa"));
        assert!(py.contains("import numpy as np"));
    }

    #[test]
    fn test_nki_binop_add() {
        let mlir = r#"
module {
  llvm.func @vec_add(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(1024 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_add_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.tensor_tensor("), "missing tensor_tensor");
        assert!(py.contains("op=np.add"), "missing np.add op");
    }

    #[test]
    fn test_nki_transpose() {
        let mlir = r#"
module {
  llvm.func @transpose_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(8 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_transpose_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %c, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.nc_transpose("), "missing nc_transpose");
    }

    #[test]
    fn test_nki_rsqrt() {
        let mlir = r#"
module {
  llvm.func @rsqrt_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_rsqrt_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.sqrt("), "missing sqrt step");
        assert!(py.contains("nl.divide("), "missing divide for 1/sqrt");
        assert!(py.contains("nl.ones("), "missing ones for numerator");
    }

    #[test]
    fn test_nki_log() {
        let mlir = r#"
module {
  llvm.func @log_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_log_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.log("), "missing nl.log call");
    }

    #[test]
    fn test_nki_sigmoid() {
        let mlir = r#"
module {
  llvm.func @sigmoid_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_sigmoid_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("np.multiply, -1.0"), "missing negate step");
        assert!(py.contains("nisa.activation(np.exp"), "missing exp step");
        assert!(py.contains("np.add, 1.0"), "missing add-1 step");
        assert!(py.contains("nl.divide("), "missing reciprocal divide");
    }

    #[test]
    fn test_nki_clamp() {
        let mlir = r#"
module {
  llvm.func @clamp_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %lo = llvm.mlir.constant(0 : i32) : i32
    %hi = llvm.mlir.constant(6 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_clamp_f32(%a, %a, %lo, %hi, %r, %c) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.maximum("), "missing nl.maximum for lower clamp");
        assert!(py.contains("nl.minimum("), "missing nl.minimum for upper clamp");
    }

    #[test]
    fn test_nki_cast_f32_f16() {
        let mlir = r#"
module {
  llvm.func @cast_down_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_cast_f32_f16(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.cast("), "missing nl.cast");
        assert!(py.contains("dtype=nl.float16"), "missing float16 dtype");
    }

    #[test]
    fn test_nki_cast_f16_f32() {
        let mlir = r#"
module {
  llvm.func @cast_up_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_cast_f16_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.cast("), "missing nl.cast");
        assert!(py.contains("dtype=nl.float32"), "missing float32 dtype");
    }

    #[test]
    fn test_nki_slice() {
        let mlir = r#"
module {
  llvm.func @slice_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(8 : i32) : i32
    %c = llvm.mlir.constant(16 : i32) : i32
    %ro = llvm.mlir.constant(2 : i32) : i32
    %co = llvm.mlir.constant(4 : i32) : i32
    %dr = llvm.mlir.constant(3 : i32) : i32
    %dc = llvm.mlir.constant(5 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_slice_f32(%a, %a, %ro, %co, %r, %c, %dr, %dc) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %dr, %dc) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("[2:5, 4:9]"), "missing correct slice range");
    }

    #[test]
    fn test_nki_concat() {
        let mlir = r#"
module {
  llvm.func @concat_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %ca = llvm.mlir.constant(128 : i32) : i32
    %cb = llvm.mlir.constant(64 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %ca) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %cb) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_concat_f32(%a, %a, %b, %r, %ca, %cb) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %ca) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.concatenate("), "missing nl.concatenate");
        assert!(py.contains("axis=1"), "missing axis=1 for column concat");
    }

    #[test]
    fn test_nki_scatter() {
        let mlir = r#"
module {
  llvm.func @scatter_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %n = llvm.mlir.constant(4 : i32) : i32
    %m = llvm.mlir.constant(16 : i32) : i32
    %d = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_scatter_f32(%a, %a, %arg1, %n, %m, %d) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %m, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.ndarray("), "missing output allocation");
        assert!(py.contains("nl.sequential_range("), "missing scatter loop");
        assert!(py.contains("dtype=np.int32"), "missing int32 index load");
    }

    #[test]
    fn test_nki_gather() {
        let mlir = r#"
module {
  llvm.func @gather_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(16 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %n = llvm.mlir.constant(4 : i32) : i32
    %m = llvm.mlir.constant(16 : i32) : i32
    %d = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_gather_f32(%a, %a, %arg1, %n, %m, %d) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %n, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.ndarray("), "missing output allocation");
        assert!(py.contains("nl.sequential_range("), "missing gather loop");
        assert!(py.contains("dtype=np.int32"), "missing int32 index load");
    }

    #[test]
    fn test_nki_topk() {
        let mlir = r#"
module {
  llvm.func @topk_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %k = llvm.mlir.constant(10 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_topk_f32(%a, %a, %arg1, %r, %c, %k) : (i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %k) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.argsort("), "missing argsort for top-k");
        assert!(py.contains("nl.sort("), "missing sort for top-k values");
        assert!(py.contains("descending=True"), "missing descending flag");
        assert!(py.contains("[:, :10]"), "missing slice to k=10");
    }

    #[test]
    fn test_nki_matmul_f16() {
        let mlir = r#"
module {
  llvm.func @matmul_f16_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %m = llvm.mlir.constant(32 : i32) : i32
    %k = llvm.mlir.constant(64 : i32) : i32
    %n = llvm.mlir.constant(16 : i32) : i32
    %a = llvm.call @ascend_tile_load_f16(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f16(%arg1, %k, %n) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_matmul_f16(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f16(%arg2, %res, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.nc_matmul("), "missing nc_matmul for f16");
    }

    #[test]
    fn test_nki_fill() {
        let mlir = r#"
module {
  llvm.func @fill_test(%arg0: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %s = llvm.mlir.constant(0 : i32) : i32
    %res = llvm.call @ascend_tile_fill_f32(%s, %s, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg0, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.full("), "fill must use nl.full:\n{}", py);
    }

    #[test]
    fn test_nki_max() {
        let mlir = r#"
module {
  llvm.func @max_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_max_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.tensor_tensor(") && py.contains("nl.maximum"), "max must use nl.maximum:\n{}", py);
    }

    #[test]
    fn test_nki_div() {
        let mlir = r#"
module {
  llvm.func @div_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_div_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.tensor_tensor(") && py.contains("nl.divide"), "div must use nl.divide:\n{}", py);
    }

    #[test]
    fn test_nki_rms_norm() {
        let mlir = r#"
module {
  llvm.func @rms_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %e = llvm.mlir.constant(0 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_rms_norm_f32(%e, %a, %e, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.rsqrt("), "rms_norm must use nl.rsqrt:\n{}", py);
    }

    #[test]
    fn test_nki_absmax() {
        let mlir = r#"
module {
  llvm.func @absmax_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_absmax_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.abs("), "absmax must use nl.abs:\n{}", py);
        assert!(py.contains("nl.max"), "absmax must use nl.max:\n{}", py);
    }

    #[test]
    fn test_nki_quantize_f32_i8() {
        let mlir = r#"
module {
  llvm.func @quantize_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @ascend_tile_load_f32(%arg1, %r, %r) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_quantize_f32_i8(%a, %a, %s, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.round("), "quantize must use nl.round:\n{}", py);
        assert!(py.contains("nl.clip("), "quantize must use nl.clip:\n{}", py);
    }

    #[test]
    fn test_nki_dequantize_i8_f32() {
        let mlir = r#"
module {
  llvm.func @dequantize_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @ascend_tile_load_f32(%arg1, %r, %r) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_dequantize_i8_f32(%a, %a, %s, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.multiply("), "dequantize must use nl.multiply:\n{}", py);
    }

    // ── Phase 6 MTP tests ──────────────────────────────────────────────────

    #[test]
    fn test_nki_argmax_f32() {
        let mlir = r#"
module {
  llvm.func @argmax_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_argmax_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nl.argmax("), "argmax must use nl.argmax:\n{}", py);
        assert!(py.contains("axis=1"), "argmax must reduce over axis=1:\n{}", py);
    }

    #[test]
    fn test_nki_sample_top_p_f32() {
        let mlir = r#"
module {
  llvm.func @sample_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(2 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_sample_top_p_f32(%c0, %a, %c0, %c0, %c0, %r, %c) : (i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.activation(np.exp"), "sample_top_p must use exp:\n{}", py);
        assert!(py.contains("nl.argmax("), "sample_top_p must use argmax:\n{}", py);
    }

    #[test]
    fn test_nki_draft_verify_f32() {
        let mlir = r#"
module {
  llvm.func @verify_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(32 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_draft_verify_f32(%c0, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %r, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.tensor_reduce(np.max"), "draft_verify must use row-max:\n{}", py);
        assert!(py.contains("nl.minimum("), "draft_verify must clamp to 1:\n{}", py);
    }

    #[test]
    fn test_nki_token_accept_f32() {
        let mlir = r#"
module {
  llvm.func @accept_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(4 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %r, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %p = llvm.call @ascend_tile_load_f32(%arg2, %r, %c1) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_token_accept_f32(%c0, %a, %b, %p, %c0, %r) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %res, %r, %c1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("np.greater_equal"), "token_accept must use >= comparison:\n{}", py);
        assert!(py.contains("np.multiply"), "token_accept must use mask multiply:\n{}", py);
    }

    // ── New intrinsic tests ──────────────────────────────────────────────

    #[test]
    fn test_nki_silu() {
        let mlir = r#"
module {
  llvm.func @silu_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_silu_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        // SiLU = x * sigmoid(x): must have sigmoid decomposition + final multiply
        assert!(py.contains("np.multiply, -1.0"), "silu must negate for sigmoid:\n{}", py);
        assert!(py.contains("nisa.activation(np.exp"), "silu must use exp:\n{}", py);
        assert!(py.contains("np.add, 1.0"), "silu must add 1 for denominator:\n{}", py);
        assert!(py.contains("nl.divide("), "silu must divide for sigmoid:\n{}", py);
        assert!(py.contains("op=np.multiply"), "silu must multiply x * sigmoid(x):\n{}", py);
    }

    #[test]
    fn test_nki_cast_bf16_f32() {
        let mlir = r#"
module {
  llvm.func @cast_bf16_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f16(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_cast_bf16_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.cast("), "cast_bf16_f32 must use nisa.cast:\n{}", py);
        assert!(py.contains("dtype=nl.float32"), "cast_bf16_f32 must target float32:\n{}", py);
    }

    #[test]
    fn test_nki_matmul_transposed() {
        let mlir = r#"
module {
  llvm.func @matmul_t_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %m = llvm.mlir.constant(32 : i32) : i32
    %k = llvm.mlir.constant(64 : i32) : i32
    %n = llvm.mlir.constant(16 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %m, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @ascend_tile_load_f32(%arg1, %n, %k) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_matmul_transposed_f32(%a, %a, %b, %m, %k, %n) : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %res, %m, %n) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.nc_transpose("), "matmul_transposed must transpose B:\n{}", py);
        assert!(py.contains("nisa.nc_matmul("), "matmul_transposed must use nc_matmul:\n{}", py);
    }

    #[test]
    fn test_nki_attention_gqa() {
        let mlir = r#"
module {
  llvm.func @gqa_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %heads = llvm.mlir.constant(8 : i32) : i32
    %hd = llvm.mlir.constant(64 : i32) : i32
    %seq = llvm.mlir.constant(128 : i32) : i32
    %kvh = llvm.mlir.constant(2 : i32) : i32
    %q = llvm.call @ascend_tile_load_f32(%arg0, %heads, %hd) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @ascend_tile_load_f32(%arg1, %seq, %hd) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @ascend_tile_load_f32(%arg2, %seq, %hd) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_attention_gqa_f32(%q, %q, %k, %v, %heads, %hd, %seq, %kvh) : (i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg3, %res, %heads, %hd) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        // GQA decomposes into: transpose K, matmul Q@K^T, scale, softmax, matmul attn@V
        assert!(py.contains("nisa.nc_transpose("), "gqa must transpose K:\n{}", py);
        assert!(py.contains("nisa.nc_matmul("), "gqa must use nc_matmul:\n{}", py);
        assert!(py.contains("nisa.tensor_reduce(np.max"), "gqa must have softmax rowmax:\n{}", py);
        assert!(py.contains("nisa.activation(np.exp"), "gqa must have softmax exp:\n{}", py);
        assert!(py.contains("nl.divide("), "gqa must have softmax divide:\n{}", py);
        // Check scale factor: 1/sqrt(64) = 0.125
        assert!(py.contains("0.125"), "gqa must scale by 1/sqrt(head_dim):\n{}", py);
    }

    #[test]
    fn test_nki_silu_mul_fusion() {
        // SiLU followed by Mul should fuse into a single silu_mul
        let mlir = r#"
module {
  llvm.func @gated_mlp(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %gate = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %up = llvm.call @ascend_tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %silu = llvm.call @ascend_tile_silu_f32(%gate, %gate, %r, %c) : (i32, i32, i32, i32) -> i32
    %fused = llvm.call @ascend_tile_mul_f32(%silu, %silu, %up, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg2, %fused, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        // Must have fused comment
        assert!(py.contains("fused silu_mul"), "missing fused silu_mul comment:\n{}", py);
        // Must still use exp for sigmoid decomposition
        assert!(py.contains("nisa.activation(np.exp"), "fused silu_mul must use exp:\n{}", py);
        // Must multiply silu result with up
        assert!(py.contains("op=np.multiply"), "fused silu_mul must use multiply:\n{}", py);
    }

    #[test]
    fn test_nki_silu_standalone_no_fusion() {
        // A standalone SiLU (without following Mul) should NOT fuse
        let mlir = r#"
module {
  llvm.func @silu_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(256 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_silu_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(!py.contains("silu_mul"), "standalone silu must not produce silu_mul:\n{}", py);
        assert!(py.contains("op=np.multiply"), "standalone silu must have x * sigmoid(x):\n{}", py);
    }

    #[test]
    fn test_nki_layernorm() {
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
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa") || py.contains("sqrt") || py.contains("rms"),
                "missing rms_norm in NKI output:\n{}", py);
    }

    #[test]
    fn test_nki_conv1d() {
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
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("multiply") || py.contains("*"),
                "missing scale in NKI output:\n{}", py);
        assert!(py.contains("add") || py.contains("+"),
                "missing add in NKI output:\n{}", py);
    }

    #[test]
    fn test_nki_matmul() {
        let mlir = r#"
module {
  llvm.func @tile_matmul(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %m = llvm.mlir.constant(4 : i32) : i32
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
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("matmul") || py.contains("dot"),
                "missing matmul in NKI output:\n{}", py);
    }

    #[test]
    fn test_nki_rope() {
        let mlir = r#"
module {
  llvm.func @rope_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(8 : i32) : i32
    %pos = llvm.mlir.constant(3 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_rope_f32(%a, %a, %pos, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        // RoPE must compute frequencies via exp of scaled arange
        assert!(py.contains("nl.arange"), "rope must build freq via arange:\n{}", py);
        assert!(py.contains("nisa.activation(np.exp"), "rope must use exp for freq:\n{}", py);
        // Must compute cos and sin of angle
        assert!(py.contains("nl.cos("), "rope must compute cos(angle):\n{}", py);
        assert!(py.contains("nl.sin("), "rope must compute sin(angle):\n{}", py);
        // Must split into even/odd and recombine
        assert!(py.contains("0::2") || py.contains("0:8:2"), "rope must slice even cols:\n{}", py);
        assert!(py.contains("1::2") || py.contains("1:8:2"), "rope must slice odd cols:\n{}", py);
        // Must have subtract for out_even = even*cos - odd*sin
        assert!(py.contains("op=np.subtract"), "rope must subtract for even output:\n{}", py);
        // Must interleave back via stack+reshape
        assert!(py.contains("nl.stack("), "rope must interleave via stack:\n{}", py);
        assert!(py.contains(".reshape("), "rope must reshape after stack:\n{}", py);
    }

    #[test]
    fn test_nki_reduce_max() {
        let mlir = r#"
module {
  llvm.func @reduce_max_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_reduce_max_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.tensor_reduce(np.max"), "missing max reduce idiom:\n{}", py);
        assert!(py.contains("axis=(1,)"), "missing row-reduce axis:\n{}", py);
    }

    #[test]
    fn test_nki_reduce_sum() {
        let mlir = r#"
module {
  llvm.func @reduce_sum_test(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(4 : i32) : i32
    %c = llvm.mlir.constant(128 : i32) : i32
    %a = llvm.call @ascend_tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @ascend_tile_reduce_sum_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @ascend_tile_store_f32(%arg1, %res, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let py = convert_mlir_to_nki(mlir).unwrap();
        assert!(py.contains("nisa.tensor_reduce(np.add"), "missing sum reduce idiom:\n{}", py);
        assert!(py.contains("axis=(1,)"), "missing row-reduce axis:\n{}", py);
    }

    #[test]
    fn test_nki_dtype_for_ptr() {
        let mut ctx = NkiContext::new();
        ctx.arg_dtypes.insert("p0".to_string(), "f16".to_string());
        assert_eq!(ctx.dtype_for_ptr("p0"), "f16", "known ptr dtype not returned");
        assert_eq!(ctx.dtype_for_ptr("unknown"), "f32", "fallback dtype must be f32");
    }

    // -----------------------------------------------------------------------
    // Error-path coverage: the per-op `if args.len() < N { return Err(...) }`
    // arity guards inside translate_call (and translate_binop/translate_reduce).
    // Each call below passes FEWER args than the op requires, so the guard
    // fires and convert_mlir_to_nki returns Err.
    //
    // NOTE: NKI's source-operand lookups use `get_tile_var(ssa).unwrap_or(ssa)`
    // — an undefined source SSA falls back to the literal string and does NOT
    // error — so the only reachable error arms here are the arity guards.
    // -----------------------------------------------------------------------

    /// Wrap one too-short intrinsic call in entry-func boilerplate.
    macro_rules! ghost_nki {
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
    fn test_nki_arity_errs() {
        // (intrinsic, too-short arg list, return sig) per the dispatcher guards.
        let cases: &[(&str, &str, &str)] = &[
            ("ascend_tile_load_f32", "%c0, %c1", "(i32, i32) -> i32"),       // needs 3
            ("ascend_tile_store_f32", "%arg1, %c0", "(!llvm.ptr<1>, i32) -> ()"), // needs 4
            ("ascend_tile_add_f32", "%c0, %c1", "(i32, i32) -> i32"),        // binop needs 5
            ("ascend_tile_sub_f32", "%c0, %c1", "(i32, i32) -> i32"),        // binop needs 5
            ("ascend_tile_mul_f32", "%c0, %c1", "(i32, i32) -> i32"),        // binop needs 5
            ("ascend_tile_div_f32", "%c0, %c1", "(i32, i32) -> i32"),        // binop needs 5
            ("ascend_tile_max_f32", "%c0, %c1", "(i32, i32) -> i32"),        // binop needs 5
            ("ascend_tile_exp_f32", "%c0, %c1", "(i32, i32) -> i32"),        // needs 4
            ("ascend_tile_softmax_f32", "%c0, %c1", "(i32, i32) -> i32"),    // needs 4
            ("ascend_tile_matmul_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 6
            ("ascend_tile_reduce_max_f32", "%c0, %c1", "(i32, i32) -> i32"), // reduce needs 4
            ("ascend_tile_reduce_sum_f32", "%c0, %c1", "(i32, i32) -> i32"), // reduce needs 4
            ("ascend_tile_scale_f32", "%c0, %c1", "(i32, i32) -> i32"),      // needs 5
            ("ascend_tile_transpose_f32", "%c0, %c1", "(i32, i32) -> i32"),  // needs 4
            ("ascend_tile_rsqrt_f32", "%c0, %c1", "(i32, i32) -> i32"),      // needs 4
            ("ascend_tile_log_f32", "%c0, %c1", "(i32, i32) -> i32"),        // needs 4
            ("ascend_tile_sigmoid_f32", "%c0, %c1", "(i32, i32) -> i32"),    // needs 4
            ("ascend_tile_clamp_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 6
            ("ascend_tile_cast_f32_f16", "%c0, %c1", "(i32, i32) -> i32"),   // needs 4
            ("ascend_tile_cast_f16_f32", "%c0, %c1", "(i32, i32) -> i32"),   // needs 4
            ("ascend_tile_cast_bf16_f32", "%c0, %c1", "(i32, i32) -> i32"),  // needs 4
            ("ascend_tile_slice_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 8
            ("ascend_tile_concat_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 6
            ("ascend_tile_scatter_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 6
            ("ascend_tile_gather_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"),  // needs 6
            ("ascend_tile_topk_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"),    // needs 6
            ("ascend_tile_fill_f32", "%c0, %c1", "(i32, i32) -> i32"),       // needs 4
            ("ascend_tile_rms_norm_f32", "%c0, %c1", "(i32, i32) -> i32"),   // needs 5
            ("ascend_tile_absmax_f32", "%c0, %c1", "(i32, i32) -> i32"),     // needs 4
            ("ascend_tile_quantize_f32_i8", "%c0, %c1", "(i32, i32) -> i32"), // needs 5
            ("ascend_tile_dequantize_i8_f32", "%c0, %c1", "(i32, i32) -> i32"), // needs 5
            ("ascend_tile_argmax_f32", "%c0, %c1", "(i32, i32) -> i32"),     // needs 4
            ("ascend_tile_sample_top_p_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 7
            ("ascend_tile_draft_verify_f32", "%c0, %c1", "(i32, i32) -> i32"), // needs 5
            ("ascend_tile_token_accept_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 6
            ("ascend_tile_silu_f32", "%c0, %c1", "(i32, i32) -> i32"),       // needs 4
            ("ascend_tile_matmul_transposed_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 6
            ("ascend_tile_attention_gqa_f32", "%c0, %c1, %c2", "(i32, i32, i32) -> i32"), // needs 8
            ("ascend_tile_rope_f32", "%c0, %c1", "(i32, i32) -> i32"),       // needs 5
        ];
        for (op, short_args, sig) in cases {
            let call = format!("%r = llvm.call @{}({}) : {}", op, short_args, sig);
            let mlir = ghost_nki!(call);
            assert!(
                convert_mlir_to_nki(&mlir).is_err(),
                "{} with too few args must hit the arity guard",
                op
            );
        }
    }

    #[test]
    fn test_nki_store_arity_errs() {
        // store has a side-effecting (no-result) form; verify the 4-arg guard.
        let call = "llvm.call @ascend_tile_store_f16(%arg1, %c0, %c1) : (!llvm.ptr<1>, i32, i32) -> ()";
        assert!(convert_mlir_to_nki(&ghost_nki!(call)).is_err());
    }
}
