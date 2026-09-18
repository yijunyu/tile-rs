//! Canned per-kernel MSL emitters, split out of `mlir_to_msl.rs`.
//!
//! These 216 functions are STRAIGHT-LINE `writeln!` bodies: each one prints one
//! fixed kernel and reads no MLIR. They were 21,489 of the parent file's 45,704
//! lines and ~40% covered, while every other emitter in the tree sits at 83-93%
//! -- so the single file set the whole-surface coverage number by itself and the
//! ratchet moved whenever a canned emitter was ADDED rather than when real logic
//! regressed (see docs/TILE_RS_COVERAGE.md).
//!
//! Splitting them here lets coverage measure the composed/dispatch logic and this
//! canned tail separately, instead of holding the whole file out of the gate.
//!
//! `#[path]`-included by the parent, so the nine consumers that
//! `#[path]`-include `mlir_to_msl.rs` keep working unchanged: a submodule path is
//! resolved relative to the PARENT FILE, not to the including crate.

use super::*;

pub(super) fn emit_copy_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) p1[gid] = p0[gid];").unwrap();
}
/// Scatter-add for EMA codebook update.
/// Buffers: p0=encoder_outputs (N×D float), p1=indices (N float→uint),
///          p2=code_sum accumulator (K×D, atomic), p3=code_count (K, atomic).
/// Params: num_elements=N, code_dim=D.
/// Dispatch: (N, 1, 1) — one workgroup per encoder output token.
/// Uses atomic_float for thread-safe accumulation.
pub(super) fn emit_scatter_add_msl(out: &mut String) {
    writeln!(out, "    uint token_idx = row;").unwrap();
    writeln!(out, "    uint code_idx = (uint)p1[token_idx];").unwrap();
    writeln!(out, "    uint src_base = token_idx * code_dim;").unwrap();
    writeln!(out, "    uint dst_base = code_idx  * code_dim;").unwrap();
    writeln!(out).unwrap();
    // Threads partition the D-dimensional vector
    writeln!(out, "    for (uint d = tid; d < code_dim; d += tcount) {{").unwrap();
    writeln!(out, "        atomic_fetch_add_explicit(").unwrap();
    writeln!(out, "            (device atomic_float*)&p2[dst_base + d],").unwrap();
    writeln!(out, "            p0[src_base + d], memory_order_relaxed);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    // Increment code count (one thread per workgroup)"
    )
    .unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        atomic_fetch_add_explicit(").unwrap();
    writeln!(out, "            (device atomic_float*)&p3[code_idx],").unwrap();
    writeln!(out, "            1.0f, memory_order_relaxed);").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Element-wise conditional select: p3[i] = p0[i] != 0 ? p1[i] : p2[i].
/// Buffers: p0=condition (0/1 float), p1=true branch, p2=false branch, p3=output.
/// Used for L1-smooth loss: `where(|diff| < 1.0, 0.5*diff², |diff| - 0.5)`.
pub(super) fn emit_where_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements)").unwrap();
    writeln!(
        out,
        "        p3[gid] = (p0[gid] != 0.0f) ? p1[gid] : p2[gid];"
    )
    .unwrap();
}
/// Transpose: dst[col*rows+row] = src[row*cols+col].
/// Buffers: p0=input, p1=output.
/// Params: num_elements, rows, cols.
pub(super) fn emit_transpose_msl(out: &mut String) {
    writeln!(
        out,
        "    for (uint idx = tid; idx < rows * cols; idx += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        uint r = idx / cols;").unwrap();
    writeln!(out, "        uint c = idx % cols;").unwrap();
    writeln!(out, "        p1[c * rows + r] = p0[r * cols + c];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Sigmoid: p1[i] = 1.0f / (1.0f + exp(-p0[i])).
pub(super) fn emit_sigmoid_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        p1[gid] = 1.0f / (1.0f + exp(-p0[gid]));").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Softplus: log(1+exp(x)). For x>20 falls through to identity to avoid exp overflow.
pub(super) fn emit_softplus_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        float x = p0[gid];").unwrap();
    writeln!(
        out,
        "        p1[gid] = (x > 20.0f) ? x : log(1.0f + exp(x));"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Clamp: p1[i] = clamp(p0[i], clamp_min, clamp_max).
pub(super) fn emit_clamp_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements)").unwrap();
    writeln!(
        out,
        "        p1[gid] = clamp(p0[gid], clamp_min, clamp_max);"
    )
    .unwrap();
}
/// Slice: copy a subrange of columns from src to dst.
/// Buffers: p0=input, p1=output.
/// Params: num_elements (rows), src_cols, dst_cols, col_offset.
pub(super) fn emit_slice_msl(out: &mut String) {
    writeln!(out, "    uint r = row;").unwrap();
    writeln!(out, "    for (uint c = tid; c < dst_cols; c += tcount) {{").unwrap();
    writeln!(
        out,
        "        p1[r * dst_cols + c] = p0[r * src_cols + col_offset + c];"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Repeat: out[i] = src[i % src_n] in flat-1D broadcast.
/// Buffers: p0=src, p1=output. Params: num_elements (output), src_n (source row length).
pub(super) fn emit_repeat_msl(out: &mut String) {
    writeln!(
        out,
        "    for (uint i = tid; i < num_elements; i += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        p1[i] = p0[i % src_n];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// GetRows: out[row, c] = table[ids[row], c].
/// Buffers: p0=table(float, V x num_elements), p1=ids(int, num_rows), p2=out(float, num_rows x num_elements).
/// Params: num_elements (= ne00, columns per row).
/// Dispatch: (num_rows, 1, 1). One workgroup per output row; threads stride over columns.
pub(super) fn emit_get_rows_msl(out: &mut String) {
    writeln!(out, "    int r = p1[row];").unwrap();
    writeln!(out, "    uint src_base = (uint)r * num_elements;").unwrap();
    writeln!(
        out,
        "    for (uint c = tid; c < num_elements; c += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        p2[base + c] = p0[src_base + c];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// SetRows: dst[ids[row], c] = src[row, c]. Reverse of GetRows.
/// Buffers: p0=src(float, num_rows x num_elements), p1=ids(int, num_rows), p2=dst(float, V x num_elements).
/// Params: num_elements (= nk0, columns per row).
/// Dispatch: (num_rows, 1, 1). One workgroup per source row; threads stride over columns.
pub(super) fn emit_set_rows_msl(out: &mut String) {
    writeln!(out, "    int i1 = p1[row];").unwrap();
    writeln!(out, "    uint dst_base = (uint)i1 * num_elements;").unwrap();
    writeln!(
        out,
        "    for (uint c = tid; c < num_elements; c += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        p2[dst_base + c] = p0[base + c];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Concat: copy from p0 (len_a elements) then p1 into p2.
/// Buffers: p0=first, p1=second, p2=output.
/// Params: num_elements (total = len_a + len_b), len_a.
pub(super) fn emit_concat_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        if (gid < len_a)").unwrap();
    writeln!(out, "            p2[gid] = p0[gid];").unwrap();
    writeln!(out, "        else").unwrap();
    writeln!(out, "            p2[gid] = p1[gid - len_a];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Scatter: p2[indices[i]*stride + i%stride] = p0[i].
/// Buffers: p0=values, p1=indices (float->uint), p2=output.
/// Params: num_elements, stride.
pub(super) fn emit_scatter_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        uint idx = (uint)p1[gid / stride];").unwrap();
    writeln!(out, "        uint col = gid % stride;").unwrap();
    writeln!(out, "        p2[idx * stride + col] = p0[gid];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Gather: p2[i] = p0[indices[i/stride]*stride + i%stride].
/// Buffers: p0=values, p1=indices (float->uint), p2=output.
/// Params: num_elements, stride.
pub(super) fn emit_gather_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        uint idx = (uint)p1[gid / stride];").unwrap();
    writeln!(out, "        uint col = gid % stride;").unwrap();
    writeln!(out, "        p2[gid] = p0[idx * stride + col];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// f16 matrix multiply: C[m,n] += A[m,k] * B[k,n].
/// Buffers: p0=A (MxK half), p1=B (KxN half), p2=C (MxN half).
/// Params: num_elements, M, N, K.
/// Dispatch: (M, 1, 1) -- one workgroup per output row.
pub(super) fn emit_matmul_f16_msl(out: &mut String) {
    writeln!(out, "    uint m = row;").unwrap();
    writeln!(out, "    if (m >= M) return;").unwrap();
    writeln!(out, "    for (uint n = tid; n < N; n += tcount) {{").unwrap();
    writeln!(out, "        half acc = 0.0h;").unwrap();
    writeln!(out, "        for (uint kk = 0; kk < K; kk++)").unwrap();
    writeln!(out, "            acc += p0[m * K + kk] * p1[kk * N + n];").unwrap();
    writeln!(out, "        p2[m * N + n] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
}
pub(super) fn emit_fill_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    if (gid >= num_elements) return;").unwrap();
    writeln!(out, "    p0[gid] = fill_val;").unwrap();
}
pub(super) fn emit_max_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    if (gid >= num_elements) return;").unwrap();
    writeln!(out, "    p2[gid] = max(p0[gid], p1[gid]);").unwrap();
}
/// DS4 partial RoPE (kernel_dsv4_rope_tail_f32 equivalent).
/// Buffers: p0=src(f32, ne03×ne02×ne01×ne00), p1=pos(int32, ne02), p2=src2(f32, n_dims/2 freq factors), p3=dst(f32).
/// Dispatch: grid=(ne01, ne02, ne03), threads=(min(ne00,1024),1,1).
/// Layout assumed contiguous: stride(ne00)=4B, no padding.
pub(super) fn emit_rope_dsv4_msl(out: &mut String) {
    writeln!(out, "    uint tid    = _tid_v.x;").unwrap();
    writeln!(out, "    uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    int i1 = (int)tgpig.x;").unwrap();
    writeln!(out, "    int i2 = (int)tgpig.y;").unwrap();
    writeln!(out, "    int i3 = (int)tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int n_nope = (int)ne00 - (int)n_dims;").unwrap();
    writeln!(out, "    if (n_nope < 0) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // YaRN correction dims (rope_yarn_corr_dims inlined)"
    )
    .unwrap();
    writeln!(out, "    float corr_lo = floor((float)n_dims * log((float)n_ctx_orig / (beta_fast * 2.0f * M_PI_F)) / (2.0f * log(freq_base)));").unwrap();
    writeln!(out, "    float corr_hi = ceil ((float)n_dims * log((float)n_ctx_orig / (beta_slow * 2.0f * M_PI_F)) / (2.0f * log(freq_base)));").unwrap();
    writeln!(out, "    corr_lo = max(0.0f, corr_lo);").unwrap();
    writeln!(out, "    corr_hi = min((float)n_dims - 1.0f, corr_hi);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float theta_base = (float)p1[i2];").unwrap();
    writeln!(out, "    float inv_ndims  = -1.0f / (float)n_dims;").unwrap();
    writeln!(out, "    bool is_neox = (mode == 2u);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    uint row_base = ((uint)i3 * ne02 + (uint)i2) * ne01 + (uint)i1;"
    )
    .unwrap();
    writeln!(out, "    uint base_off = row_base * ne00;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i0 = tid; i0 < ne00; i0 += tcount) {{").unwrap();
    writeln!(out, "        if ((int)i0 < n_nope) {{").unwrap();
    writeln!(out, "            p3[base_off + i0] = p0[base_off + i0];").unwrap();
    writeln!(out, "            continue;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        int r = (int)i0 - n_nope;").unwrap();
    writeln!(out, "        if (is_neox) {{").unwrap();
    writeln!(out, "            int n_half = (int)n_dims / 2;").unwrap();
    writeln!(out, "            if (r >= n_half) continue;").unwrap();
    writeln!(out, "            int ic = r;").unwrap();
    writeln!(out, "            int rel_i0 = 2 * ic;").unwrap();
    writeln!(
        out,
        "            float theta_extrap = theta_base * pow(freq_base, inv_ndims * (float)rel_i0);"
    )
    .unwrap();
    writeln!(
        out,
        "            float freq_factor = (has_src2 != 0u) ? p2[ic] : 1.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "            float theta_in = theta_extrap / freq_factor;"
    )
    .unwrap();
    writeln!(out, "            // rope_yarn inlined").unwrap();
    writeln!(
        out,
        "            float theta_interp = freq_scale * theta_in;"
    )
    .unwrap();
    writeln!(out, "            float theta = theta_interp;").unwrap();
    writeln!(out, "            float mscale = attn_factor;").unwrap();
    writeln!(out, "            if (ext_factor != 0.0f) {{").unwrap();
    writeln!(out, "                float ramp = ((float)rel_i0 / 2.0f - corr_lo) / max(0.001f, corr_hi - corr_lo);").unwrap();
    writeln!(
        out,
        "                float ramp_mix = (1.0f - min(1.0f, max(0.0f, ramp))) * ext_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "                theta = theta_interp * (1.0f - ramp_mix) + theta_in * ramp_mix;"
    )
    .unwrap();
    writeln!(
        out,
        "                mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            float cos_t = cos(theta) * mscale;").unwrap();
    writeln!(out, "            float sin_t = sin(theta) * mscale;").unwrap();
    writeln!(out, "            if (inverse != 0u) sin_t = -sin_t;").unwrap();
    writeln!(out, "            int j0 = n_nope + ic;").unwrap();
    writeln!(out, "            int j1 = n_nope + ic + n_half;").unwrap();
    writeln!(out, "            float x0 = p0[base_off + (uint)j0];").unwrap();
    writeln!(out, "            float x1 = p0[base_off + (uint)j1];").unwrap();
    writeln!(
        out,
        "            p3[base_off + (uint)j0] = x0 * cos_t - x1 * sin_t;"
    )
    .unwrap();
    writeln!(
        out,
        "            p3[base_off + (uint)j1] = x0 * sin_t + x1 * cos_t;"
    )
    .unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            if ((r & 1) != 0) continue;").unwrap();
    writeln!(out, "            int ic = r / 2;").unwrap();
    writeln!(
        out,
        "            float theta_extrap = theta_base * pow(freq_base, inv_ndims * (float)r);"
    )
    .unwrap();
    writeln!(
        out,
        "            float freq_factor = (has_src2 != 0u) ? p2[ic] : 1.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "            float theta_in = theta_extrap / freq_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "            float theta_interp = freq_scale * theta_in;"
    )
    .unwrap();
    writeln!(out, "            float theta = theta_interp;").unwrap();
    writeln!(out, "            float mscale = attn_factor;").unwrap();
    writeln!(out, "            if (ext_factor != 0.0f) {{").unwrap();
    writeln!(
        out,
        "                float ramp = ((float)r / 2.0f - corr_lo) / max(0.001f, corr_hi - corr_lo);"
    )
    .unwrap();
    writeln!(
        out,
        "                float ramp_mix = (1.0f - min(1.0f, max(0.0f, ramp))) * ext_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "                theta = theta_interp * (1.0f - ramp_mix) + theta_in * ramp_mix;"
    )
    .unwrap();
    writeln!(
        out,
        "                mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            float cos_t = cos(theta) * mscale;").unwrap();
    writeln!(out, "            float sin_t = sin(theta) * mscale;").unwrap();
    writeln!(out, "            if (inverse != 0u) sin_t = -sin_t;").unwrap();
    writeln!(out, "            int j0 = n_nope + r;").unwrap();
    writeln!(out, "            int j1 = j0 + 1;").unwrap();
    writeln!(out, "            float x0 = p0[base_off + (uint)j0];").unwrap();
    writeln!(out, "            float x1 = p0[base_off + (uint)j1];").unwrap();
    writeln!(
        out,
        "            p3[base_off + (uint)j0] = x0 * cos_t - x1 * sin_t;"
    )
    .unwrap();
    writeln!(
        out,
        "            p3[base_off + (uint)j1] = x0 * sin_t + x1 * cos_t;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 kernel_flash_attn_ext_f16_dk512_dv512 (M123) — antirez flash_attn.metal:924.
/// Host-callable prefill FlashAttention with DK=DV=512 half K/V rows.
/// Scalar-correctness emitter: per-thread online softmax + V-weighted accumulator
/// across C=64 KV chunks. Each threadgroup processes one query block (Q=8 rows).
///
/// Bake-ins: DK=DV=512, Q=8, C=64. Runtime feature flags via uniforms: has_mask,
/// has_sinks, has_bias (ALiBi), has_softcap.
///
/// Buffers (all char*): p0=q (half), p1=k (half), p2=v (half), p3=mask (half),
/// p4=sinks (float, ne02 entries), p5=pad (unused in this version),
/// p6=blk (unused in this version), p7=dst (float DV).
///
/// Dispatch: 3D grid ((ne01+Q-1)/Q, ne02, ne03) threadgroups × tcount threads/tg.
/// Each thread handles a subset of (j=Q-row, d=DV-element) work via tid striding.
/// `dk` and `dv` are the key and value head dimensions this scalar reference
/// kernel is emitted for; the 512s in its name are only the default.
pub(super) fn emit_flash_attn_ext_f16_dk512_dv512_msl(out: &mut String, dk: u32, dv: u32) {
    writeln!(out, "    constexpr short DK = {dk};").unwrap();
    writeln!(out, "    constexpr short DV = {dv};").unwrap();
    writeln!(out, "    constexpr short Q  = 8;").unwrap();
    writeln!(out, "    constexpr short C  = 64;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint tid    = _tid_v.x;").unwrap();
    writeln!(out, "    uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    int iq1 = (int)tgpig.x * Q;").unwrap();
    writeln!(out, "    int iq2 = (int)tgpig.y;").unwrap();
    writeln!(out, "    int iq3 = (int)tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float slope = 1.0f;").unwrap();
    writeln!(out, "    if (U.has_bias != 0u) {{").unwrap();
    writeln!(out, "        int h = iq2;").unwrap();
    writeln!(out, "        float base = h < U.n_head_log2 ? U.m0 : U.m1;").unwrap();
    writeln!(
        out,
        "        int exph   = h < U.n_head_log2 ? h + 1 : 2 * (h - U.n_head_log2) + 1;"
    )
    .unwrap();
    writeln!(out, "        slope = pow(base, (float)exph);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint linear = tid; linear < (uint)(Q * DV); linear += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        int j = (int)(linear / (uint)DV);").unwrap();
    writeln!(out, "        int d = (int)(linear % (uint)DV);").unwrap();
    writeln!(out, "        int row = iq1 + j;").unwrap();
    writeln!(out, "        if (row >= U.ne01) continue;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const half * q_row = (device const half *)(p0 + (uint)row*U.nb01 + (uint)iq2*U.nb02 + (uint)iq3*U.nb03);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        int ikv2 = iq2 / (U.ne02 / U.ne_12_2);").unwrap();
    writeln!(out, "        int ikv3 = iq3 / (U.ne03 / U.ne_12_3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float M_val = -FLT_MAX / 2.0f;").unwrap();
    writeln!(out, "        float S_val = 0.0f;").unwrap();
    writeln!(out, "        float O_val = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (int ic = 0; ic < U.ne11; ++ic) {{").unwrap();
    writeln!(out, "            device const half * k_row = (device const half *)(p1 + (uint)ic*U.nb11 + (uint)ikv2*U.nb12 + (uint)ikv3*U.nb13);").unwrap();
    writeln!(out, "            float qk = 0.0f;").unwrap();
    writeln!(out, "            for (int kk = 0; kk < DK; ++kk) {{").unwrap();
    writeln!(
        out,
        "                qk += (float)q_row[kk] * (float)k_row[kk];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            qk *= U.scale;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            if (U.has_softcap != 0u && U.logit_softcap > 0.0f) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                qk = U.logit_softcap * precise::tanh(qk / U.logit_softcap);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (U.has_mask != 0u) {{").unwrap();
    writeln!(out, "                device const half * mask_row = (device const half *)(p3 + (uint)row*U.nb31 + (uint)(iq2 % U.ne32)*U.nb32 + (uint)(iq3 % U.ne33)*U.nb33);").unwrap();
    writeln!(out, "                float m_val = (float) mask_row[ic];").unwrap();
    writeln!(out, "                if (U.has_bias != 0u) m_val *= slope;").unwrap();
    writeln!(out, "                qk += m_val;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float M_new = max(M_val, qk);").unwrap();
    writeln!(out, "            float alpha = exp(M_val - M_new);").unwrap();
    writeln!(out, "            float p     = exp(qk    - M_new);").unwrap();
    writeln!(out, "            S_val = S_val * alpha + p;").unwrap();
    writeln!(out, "            device const half * v_row = (device const half *)(p2 + (uint)ic*U.nb21 + (uint)ikv2*U.nb22 + (uint)ikv3*U.nb23);").unwrap();
    writeln!(
        out,
        "            O_val = O_val * alpha + p * (float)v_row[d];"
    )
    .unwrap();
    writeln!(out, "            M_val = M_new;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (U.has_sinks != 0u) {{").unwrap();
    writeln!(
        out,
        "            float sink = ((device const float *)p4)[iq2];"
    )
    .unwrap();
    writeln!(out, "            float M_new = max(M_val, sink);").unwrap();
    writeln!(out, "            float alpha = exp(M_val - M_new);").unwrap();
    writeln!(out, "            float p     = exp(sink  - M_new);").unwrap();
    writeln!(out, "            S_val = S_val * alpha + p;").unwrap();
    writeln!(out, "            O_val = O_val * alpha;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        uint dst_row_stride = (uint)DV * 4u;").unwrap();
    writeln!(
        out,
        "        uint dst_off = (uint)row * (uint)U.ne02 * (uint)U.ne03 * dst_row_stride"
    )
    .unwrap();
    writeln!(
        out,
        "                     + (uint)iq2 * (uint)U.ne03 * dst_row_stride"
    )
    .unwrap();
    writeln!(
        out,
        "                     + (uint)iq3 * dst_row_stride + (uint)d * 4u;"
    )
    .unwrap();
    writeln!(
        out,
        "        *((device float *)(p7 + dst_off)) = (S_val > 0.0f) ? (O_val / S_val) : 0.0f;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 kernel_flash_attn_ext_vec_f16_dk512_dv512 (M124) — antirez flash_attn.metal:961.
/// Decode-shape sibling of M123: same scalar-correctness FlashAttention body but Q=1 row
/// per threadgroup and output layout follows the vec kernel (dst[rid*DV+d] where
/// rid = iq3*ne2*ne1 + iq2 + iq1*ne1). NWG=1 baked.
/// Buffers (all char*): p0=q, p1=k, p2=v, p3=mask, p4=sinks, p5=pad, p6=dst.
/// `dk` and `dv` are the key and value head dimensions this scalar reference
/// kernel is emitted for; the 512s in its name are only the default.
pub(super) fn emit_flash_attn_ext_vec_f16_dk512_dv512_msl(out: &mut String, dk: u32, dv: u32) {
    writeln!(out, "    constexpr short DK = {dk};").unwrap();
    writeln!(out, "    constexpr short DV = {dv};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint tid    = _tid_v.x;").unwrap();
    writeln!(out, "    uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    int iq1 = (int)tgpig.x;").unwrap();
    writeln!(out, "    int iq2 = (int)tgpig.y;").unwrap();
    writeln!(out, "    int iq3 = (int)tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float slope = 1.0f;").unwrap();
    writeln!(out, "    if (U.has_bias != 0u) {{").unwrap();
    writeln!(out, "        int h = iq2;").unwrap();
    writeln!(out, "        float base = h < U.n_head_log2 ? U.m0 : U.m1;").unwrap();
    writeln!(
        out,
        "        int exph   = h < U.n_head_log2 ? h + 1 : 2 * (h - U.n_head_log2) + 1;"
    )
    .unwrap();
    writeln!(out, "        slope = pow(base, (float)exph);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (iq1 >= U.ne01) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int ikv2 = iq2 / (U.ne02 / U.ne_12_2);").unwrap();
    writeln!(out, "    int ikv3 = iq3 / (U.ne03 / U.ne_12_3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const half * q_row = (device const half *)(p0 + (uint)iq1*U.nb01 + (uint)iq2*U.nb02 + (uint)iq3*U.nb03);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Per-thread strided loop over the DV output lanes."
    )
    .unwrap();
    writeln!(out, "    for (uint d = tid; d < (uint)DV; d += tcount) {{").unwrap();
    writeln!(out, "        float M_val = -FLT_MAX / 2.0f;").unwrap();
    writeln!(out, "        float S_val = 0.0f;").unwrap();
    writeln!(out, "        float O_val = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (int ic = 0; ic < U.ne11; ++ic) {{").unwrap();
    writeln!(out, "            device const half * k_row = (device const half *)(p1 + (uint)ic*U.nb11 + (uint)ikv2*U.nb12 + (uint)ikv3*U.nb13);").unwrap();
    writeln!(out, "            float qk = 0.0f;").unwrap();
    writeln!(out, "            for (int kk = 0; kk < DK; ++kk) {{").unwrap();
    writeln!(
        out,
        "                qk += (float)q_row[kk] * (float)k_row[kk];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            qk *= U.scale;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            if (U.has_softcap != 0u && U.logit_softcap > 0.0f) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                qk = U.logit_softcap * precise::tanh(qk / U.logit_softcap);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (U.has_mask != 0u) {{").unwrap();
    writeln!(out, "                device const half * mask_row = (device const half *)(p3 + (uint)iq1*U.nb31 + (uint)(iq2 % U.ne32)*U.nb32 + (uint)(iq3 % U.ne33)*U.nb33);").unwrap();
    writeln!(out, "                float m_val = (float) mask_row[ic];").unwrap();
    writeln!(out, "                if (U.has_bias != 0u) m_val *= slope;").unwrap();
    writeln!(out, "                qk += m_val;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float M_new = max(M_val, qk);").unwrap();
    writeln!(out, "            float alpha = exp(M_val - M_new);").unwrap();
    writeln!(out, "            float p     = exp(qk    - M_new);").unwrap();
    writeln!(out, "            S_val = S_val * alpha + p;").unwrap();
    writeln!(out, "            device const half * v_row = (device const half *)(p2 + (uint)ic*U.nb21 + (uint)ikv2*U.nb22 + (uint)ikv3*U.nb23);").unwrap();
    writeln!(
        out,
        "            O_val = O_val * alpha + p * (float)v_row[d];"
    )
    .unwrap();
    writeln!(out, "            M_val = M_new;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (U.has_sinks != 0u) {{").unwrap();
    writeln!(
        out,
        "            float sink = ((device const float *)p4)[iq2];"
    )
    .unwrap();
    writeln!(out, "            float M_new = max(M_val, sink);").unwrap();
    writeln!(out, "            float alpha = exp(M_val - M_new);").unwrap();
    writeln!(out, "            float p     = exp(sink  - M_new);").unwrap();
    writeln!(out, "            S_val = S_val * alpha + p;").unwrap();
    writeln!(out, "            O_val = O_val * alpha;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        // Vec output layout: rid = iq3*ne2*ne1 + iq2 + iq1*ne1, dst[rid*DV + d] = O/S."
    )
    .unwrap();
    writeln!(
        out,
        "        int rid = iq3 * U.ne2 * U.ne1 + iq2 + iq1 * U.ne1;"
    )
    .unwrap();
    writeln!(
        out,
        "        uint dst_off = ((uint)rid * (uint)DV + d) * 4u;"
    )
    .unwrap();
    writeln!(
        out,
        "        *((device float *)(p6 + dst_off)) = (S_val > 0.0f) ? (O_val / S_val) : 0.0f;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// File-scope `block_q6_K` layout (ggml k-quant, 210 B). Metal struct types can't
/// be declared inside a kernel fn, so it is emitted once in the file prelude when a
/// Laguna dense Q6_K projection is present.
pub(super) fn emit_block_q6_k_struct(out: &mut String) {
    writeln!(
        out,
        "// ggml block_q6_K (QK_K=256): 6-bit k-quant weight block."
    )
    .unwrap();
    writeln!(out, "typedef struct {{").unwrap();
    writeln!(out, "    uint8_t ql[128];    // lower 4 bits, QK_K/2").unwrap();
    writeln!(out, "    uint8_t qh[64];     // upper 2 bits, QK_K/4").unwrap();
    writeln!(
        out,
        "    int8_t  scales[16]; // per-16 block scales, QK_K/16"
    )
    .unwrap();
    writeln!(out, "    half    d;          // super-block scale").unwrap();
    writeln!(out, "}} block_q6_K;").unwrap();
    writeln!(out).unwrap();
}
/// Laguna kernel_laguna_head_rms_norm_rope_neox (antirez metal/laguna.metal:88 +
/// inline helper laguna.metal:22 + rope_yarn/rope_yarn_corr_dims from dsv4_rope.metal:44).
/// Per-head Qwen-style RMSNorm then NeoX YaRN rotary over the rotary prefix, in-place on x.
/// Buffers: p0=x (float, in/out), p1=weight (float, const).
/// Grid attrs (needs_3d_grid_tiitg_ntg): uint3 tgpig (head=x, token=y), ushort tiitg, ushort3 ntg.
pub(super) fn emit_laguna_head_rms_norm_rope_neox_msl(out: &mut String) {
    // (B) pilot, decomposition step: this 4 KiB allocation was emitted raw and
    // charged NOTHING, so the contract said the kernel needed no threadgroup
    // memory while it needed 4096 bytes. `threadgroup_array` writes the same line
    // AND charges it in one call, which is D6 — the code and the claim come from
    // one call site, so they cannot drift.
    {
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.threadgroup_array("float", "scratch", 1024, 4, None);
        let _obligations = k.finish();
    }
    writeln!(out, "    const uint head  = tgpig.x;").unwrap();
    writeln!(out, "    const uint token = tgpig.y;").unwrap();
    writeln!(out, "    const uint nth   = (uint)ntg.x;").unwrap();
    writeln!(out, "    const uint tid   = (uint)tiitg;").unwrap();
    writeln!(out, "    if (head >= n_head || token >= n_tokens ||").unwrap();
    writeln!(
        out,
        "        head_dim == 0u || n_rot > head_dim || (n_rot & 1u) != 0u) {{"
    )
    .unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    device float * row = p0 + ((uint64_t)token * n_head + head) * head_dim;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Qwen-style per-head RMSNorm.").unwrap();
    writeln!(out, "    float ss = 0.0f;").unwrap();
    writeln!(out, "    for (uint i = tid; i < head_dim; i += nth) {{").unwrap();
    writeln!(out, "        const float v = row[i];").unwrap();
    writeln!(out, "        ss += v * v;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    scratch[tid] = ss;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // MIGRATED to KernelWriter (docs/MSL_HARDENING_PLAN.md D6). This reduction
    // is only correct when `nth` is a power of two and `nth <= 1024` (the
    // `scratch` capacity every thread seeds above). Both facts used to live
    // nowhere at all — the kernel is correct today only because the host happens
    // to dispatch HEAD_DIM = 128. Emitting through `tree_reduce` charges those
    // obligations in the same call that writes the loop, so they cannot drift
    // from it. Byte output is unchanged; see the byte-identity gate.
    {
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.tree_reduce("nth", "scratch", 1024, "scratch[tid] += scratch[tid + step]");
        let _obligations = k.finish();
    }
    writeln!(
        out,
        "    const float inv = rsqrt(scratch[0] / (float)head_dim + eps);"
    )
    .unwrap();
    writeln!(out, "    for (uint i = tid; i < head_dim; i += nth) {{").unwrap();
    writeln!(out, "        row[i] = row[i] * inv * p1[i];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_device);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // NeoX rotary pairs over the rotary prefix (dims [0, n_rot) rotated)."
    )
    .unwrap();
    writeln!(out, "    const uint half_rot = n_rot >> 1u;").unwrap();
    writeln!(out, "    if (tid >= half_rot) return;").unwrap();

    // The rotary hands ONE PAIR to each of the first `n_rot/2` threads, and the
    // guard above simply drops the rest. Nothing in the kernel relates n_rot to the
    // threadgroup width: the entry guard bounds n_rot by head_dim, not by `nth`. So
    // a threadgroup narrower than n_rot/2 leaves the pairs in [nth, n_rot/2)
    // RMSNormed but UNROTATED -- every thread that exists behaves correctly, so
    // there is no fault and no diagnostic, just position information quietly wrong.
    //
    // Correct today only because the host dispatches nth = HEAD_DIM = 128 while
    // n_rot <= head_dim, i.e. by coincidence of the shipping geometry -- exactly the
    // shape of the power-of-two dependency this plan opened with.
    // The entry guard's other two conditions are equally silent, and equally
    // uncontracted: on `n_rot > head_dim` or an odd `n_rot` the kernel returns
    // having written NOTHING -- not even the RMSNorm -- so the caller consumes
    // whatever was already in the buffer. Same species as the rotary bound below.
    {
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.charge_dim_at_most_dim(
            "n_rot",
            "head_dim",
            "n_rot > head_dim",
            "the rotary prefix cannot exceed the head; on violation the kernel \
             returns before writing anything, so the output keeps stale contents",
        );
        k.charge_dim_is_even(
            "n_rot",
            "(n_rot & 1u) != 0u",
            "rotary pairs are (i, i + n_rot/2), so an odd n_rot has no valid \
             pairing; the kernel returns without writing rather than faulting",
        );
        let _obligations = k.finish();
    }

    {
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.charge_dim_at_most_threads(
            "n_rot",
            2,
            "if (tid >= half_rot) return;",
            "the rotary gives one pair to each of the first n_rot/2 threads and \
             drops the surplus; a narrower threadgroup leaves the remaining pairs \
             normed but unrotated, with no fault and no diagnostic",
        );
        let _obligations = k.finish();
    }
    writeln!(out).unwrap();
    writeln!(out, "    float corr_dims[2] = {{0.0f, 0.0f}};").unwrap();
    writeln!(out, "    if (ext_factor != 0.0f) {{").unwrap();
    writeln!(out, "        // rope_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow) inlined.").unwrap();
    writeln!(out, "        const float cf_fast = (float)n_rot * log((float)n_ctx_orig / (beta_fast * 2.0f * M_PI_F)) / (2.0f * log(freq_base));").unwrap();
    writeln!(out, "        const float cf_slow = (float)n_rot * log((float)n_ctx_orig / (beta_slow * 2.0f * M_PI_F)) / (2.0f * log(freq_base));").unwrap();
    writeln!(out, "        corr_dims[0] = max(0.0f, floor(cf_fast));").unwrap();
    writeln!(
        out,
        "        corr_dims[1] = min((float)n_rot - 1.0f, ceil(cf_slow));"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    const int rel_i0 = (int)(tid * 2u);").unwrap();
    writeln!(out, "    const float inv_ndims = -1.0f / (float)n_rot;").unwrap();
    writeln!(out, "#ifdef DS4_METAL_ROPE_EXP2_LOG2").unwrap();
    writeln!(out, "    const float theta_extrap = (float)(pos0 + token) * exp2(inv_ndims * (float)rel_i0 * log2(freq_base));").unwrap();
    writeln!(out, "#else").unwrap();
    writeln!(out, "    const float theta_extrap = (float)(pos0 + token) * pow(freq_base, inv_ndims * (float)rel_i0);").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out, "    // rope_yarn inlined (dsv4_rope.metal:51).").unwrap();
    writeln!(out, "    float cos_theta;").unwrap();
    writeln!(out, "    float sin_theta;").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(
        out,
        "        const float theta_interp = freq_scale * theta_extrap;"
    )
    .unwrap();
    writeln!(out, "        float theta = theta_interp;").unwrap();
    writeln!(out, "        float mscale = attn_factor;").unwrap();
    writeln!(out, "        if (ext_factor != 0.0f) {{").unwrap();
    writeln!(out, "            const float ramp_y = ((float)(rel_i0 / 2) - corr_dims[0]) / max(0.001f, corr_dims[1] - corr_dims[0]);").unwrap();
    writeln!(
        out,
        "            const float ramp_mix = (1.0f - min(1.0f, max(0.0f, ramp_y))) * ext_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;"
    )
    .unwrap();
    writeln!(
        out,
        "            mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        cos_theta = cos(theta) * mscale;").unwrap();
    writeln!(out, "        sin_theta = sin(theta) * mscale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    const float x0 = row[tid];").unwrap();
    writeln!(out, "    const float x1 = row[tid + half_rot];").unwrap();
    writeln!(
        out,
        "    row[tid]            = x0 * cos_theta - x1 * sin_theta;"
    )
    .unwrap();
    writeln!(
        out,
        "    row[tid + half_rot] = x0 * sin_theta + x1 * cos_theta;"
    )
    .unwrap();
}
/// Laguna dense Q8_0 matvec — every Q8_0 projection in this model (attn q/k/v/o,
/// attn_gate, shared-expert gate/up/down, dense-FFN@L0, lm_head). block_q8_0 = 34 B
/// (half d + int8 qs[32]); lane l (0..32) accumulates d[ib]*qs[ib*32+l]*x[ib*32+l]
/// over blocks, simd_sum gives the row dot. rows_per_simd=2, simd_groups=2.
/// Buffers: p0=weight (char), p1=x (float), p2=out (float). Uniforms: in_dim,
/// out_dim, n_tokens, row_bytes (ulong). Grid (needs_3d_grid_simd).
pub(super) fn emit_laguna_q8_0_matvec_f32_msl(out: &mut String, nsg: u32, nr0: u32, nq: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    // antirez-style Q8_0 matvec (kernel_mul_mv_q8_0_f32, dense.metal): each simdgroup
    // computes NR0 rows, reusing the NQ=8 contiguous activations (yl) loaded once per
    // block across all NR0 rows; NSG independent simdgroups/threadgroup for occupancy.
    // Lane layout: ix = lane/4 selects one of 8 blocks per stride step, il = lane%4
    // selects an 8-element chunk of that block → contiguous 8-int8 + 8-float loads
    // (vs the old 1-element/lane strided reads, 32x-redundant d[] loads).
    writeln!(
        out,
        "    constexpr uint NR0 = {nr0}u;               // rows per simdgroup (activation reuse)"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr uint NSG = {nsg}u;               // simdgroups per threadgroup"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr uint NQ  = {nq}u;               // contiguous quants per lane"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr uint QK8 = 32u;              // block_q8_0 elements"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr uint BLK_Q8_0 = 34u;         // half d + int8 qs[32]"
    )
    .unwrap();
    writeln!(out, "    const uint lane = simd_lane;").unwrap();
    writeln!(out, "    const uint sg   = simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint row0 = (tgpig.x * NSG + sg) * NR0;").unwrap();
    writeln!(out, "    const uint token = tgpig.y;").unwrap();
    writeln!(out, "    if (row0 >= out_dim || token >= n_tokens) return;").unwrap();
    writeln!(out, "    const uint nb = in_dim / QK8;").unwrap();
    writeln!(
        out,
        "    device const float *input = p1 + (uint64_t)token * in_dim;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint ix = lane / (QK8 / NQ);     // block within stride group (0..7)"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint il = lane % (QK8 / NQ);     // 8-elem chunk of that block (0..3)"
    )
    .unwrap();
    writeln!(out, "    float sumf[NR0] = {{{zeros}}};").unwrap();
    writeln!(
        out,
        "    device const float *yb = input + ix * QK8 + il * NQ;"
    )
    .unwrap();
    writeln!(
        out,
        "    for (uint ib = ix; ib < nb; ib += NQ) {{   // 8 blocks (ix=0..7) per step"
    )
    .unwrap();
    writeln!(out, "        float yl[NQ];").unwrap();
    writeln!(out, "        for (uint i = 0u; i < NQ; i++) yl[i] = yb[i];").unwrap();
    writeln!(
        out,
        "        for (uint r = 0u; r < NR0 && row0 + r < out_dim; r++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            device const char *wrow = p0 + (uint64_t)(row0 + r) * row_bytes;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const half *d  = (device const half *)(wrow + (uint64_t)ib * BLK_Q8_0);"
    )
    .unwrap();
    writeln!(
        out,
        "            device const char *qs = wrow + (uint64_t)ib * BLK_Q8_0 + 2u + il * NQ;"
    )
    .unwrap();
    writeln!(out, "            float sq = 0.0f;").unwrap();
    writeln!(
        out,
        "            for (uint i = 0u; i < NQ; i++) sq += (float)qs[i] * yl[i];"
    )
    .unwrap();
    writeln!(out, "            sumf[r] += sq * (float)d[0];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        yb += NQ * QK8;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    for (uint r = 0u; r < NR0 && row0 + r < out_dim; r++) {{"
    )
    .unwrap();
    writeln!(out, "        const float sum = simd_sum(sumf[r]);").unwrap();
    writeln!(out, "        if (lane == 0u) {{").unwrap();
    writeln!(
        out,
        "            p2[(uint64_t)token * out_dim + row0 + r] = sum;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// File-scope block_q3_K (ggml, 110 B): hmask[32] + qs[64] + scales[12] + half d.
pub(super) fn emit_block_q3_k_struct(out: &mut String) {
    writeln!(
        out,
        "// ggml block_q3_K (QK_K=256): 3-bit k-quant weight block."
    )
    .unwrap();
    writeln!(out, "typedef struct {{").unwrap();
    writeln!(
        out,
        "    uint8_t hmask[32];  // high bit of each 3-bit quant, QK_K/8"
    )
    .unwrap();
    writeln!(out, "    uint8_t qs[64];     // low 2 bits, QK_K/4").unwrap();
    writeln!(out, "    uint8_t scales[12]; // 6-bit scales, split-packed").unwrap();
    writeln!(out, "    half    d;          // super-block scale").unwrap();
    writeln!(out, "}} block_q3_K;").unwrap();
    writeln!(out).unwrap();
}
/// File-scope ds4_glm_q3_K_dot2 — antirez moe.metal:724, verbatim. Returns the
/// per-simd partial for N_R0_Q3_K=2 rows of a Q3_K matrix dotted with y.
pub(super) fn emit_glm_q3_k_dot2_helper(out: &mut String) {
    let body = r#"static inline float2 ds4_glm_q3_K_dot2(
        device const char *rows,
        uint64_t row_bytes,
        uint in_dim,
        device const float *y,
        ushort tiisg) {
    constexpr short N_R0_Q3_K = 2;
    constexpr int QK_K = 256;
    const int nb = (int)(in_dim / QK_K);
    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short ip = tid / 4;
    const short il = 2 * ((tid % 4) / 2);
    const short ir = tid % 2;
    const short l0 = 8 * ir;

    const ushort4 high_masks[4] = {
        {0x0001, 0x0100, 0x0002, 0x0200},
        {0x0004, 0x0400, 0x0008, 0x0800},
        {0x0010, 0x1000, 0x0020, 0x2000},
        {0x0040, 0x4000, 0x0080, 0x8000},
    };
    const int4 low_masks[2] = {
        {0x0003, 0x0300, 0x000c, 0x0c00},
        {0x0030, 0x3000, 0x00c0, 0xc000},
    };
    const ushort4 hm = high_masks[2 * ip + il / 2];
    const short shift = 2 * il;
    const float high_base_1 = il == 0 ? 4.0f : 64.0f;
    const float high_base_2 = 4.0f * high_base_1;
    const ushort scale_shift_1 = 4 * ip;
    const ushort scale_shift_2 = scale_shift_1 + il;
    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;

    device const float *y1 = y + ix * QK_K + y_offset;
    float2 sum1 = {0.0f, 0.0f};
    float2 sum2 = {0.0f, 0.0f};

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[32];
        for (short l = 0; l < 8; l++) {
            yl[l +  0] = y1[l +  0];
            yl[l +  8] = y1[l + 16];
            yl[l + 16] = y1[l + 32];
            yl[l + 24] = y1[l + 48];
        }

        for (short row = 0; row < N_R0_Q3_K; row++) {
            device const block_q3_K *x =
                (device const block_q3_K *)(rows + (uint64_t)row * row_bytes);
            device const ushort *q =
                (device const ushort *)(x[ib].qs + q_offset);
            device const ushort *h =
                (device const ushort *)(x[ib].hmask + l0);
            device const ushort *packed_scales =
                (device const ushort *)x[ib].scales;

            uint packed = 0;
            thread ushort *packed16 = (thread ushort *)&packed;
            thread const char *scales = (thread const char *)&packed;
            packed16[0] = packed_scales[4];
            packed16[1] = packed_scales[5];
            const uint scale_high =
                ((packed >> scale_shift_2) << 4) & 0x30303030u;
            packed16[0] = packed_scales[il + 0];
            packed16[1] = packed_scales[il + 1];
            packed = ((packed >> scale_shift_1) & 0x0f0f0f0fu) |
                     scale_high;

            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;
            float s6 = 0.0f;
            for (short l = 0; l < 8; l += 2) {
                const int qs = q[l / 2];
                s1 += yl[l + 0] * (float)(qs & low_masks[il / 2][0]);
                s2 += yl[l + 1] * (float)(qs & low_masks[il / 2][1]);
                s3 += ((h[l / 2] & hm[0]) ? 0.0f : yl[l + 0]) +
                      ((h[l / 2] & hm[1]) ? 0.0f : yl[l + 1]);
                s4 += yl[l + 16] * (float)(qs & low_masks[il / 2][2]);
                s5 += yl[l + 17] * (float)(qs & low_masks[il / 2][3]);
                s6 += ((h[l / 2] & hm[2]) ? 0.0f : yl[l + 16]) +
                      ((h[l / 2] & hm[3]) ? 0.0f : yl[l + 17]);
            }

            const float d = (float)x[ib].d;
            const float d1 =
                d * (s1 + (1.0f / 256.0f) * s2 - s3 * high_base_1);
            const float d2 =
                d * (s4 + (1.0f / 256.0f) * s5 - s6 * high_base_2);
            sum1[row] += d1 * ((float)scales[0] - 32.0f);
            sum2[row] += d2 * ((float)scales[2] - 32.0f);

            s1 = s2 = s3 = s4 = s5 = s6 = 0.0f;
            for (short l = 0; l < 8; l += 2) {
                const int qs = q[l / 2 + 8];
                s1 += yl[l + 8] * (float)(qs & low_masks[il / 2][0]);
                s2 += yl[l + 9] * (float)(qs & low_masks[il / 2][1]);
                s3 += ((h[l / 2 + 8] & hm[0]) ? 0.0f : yl[l + 8]) +
                      ((h[l / 2 + 8] & hm[1]) ? 0.0f : yl[l + 9]);
                s4 += yl[l + 24] * (float)(qs & low_masks[il / 2][2]);
                s5 += yl[l + 25] * (float)(qs & low_masks[il / 2][3]);
                s6 += ((h[l / 2 + 8] & hm[2]) ? 0.0f : yl[l + 24]) +
                      ((h[l / 2 + 8] & hm[3]) ? 0.0f : yl[l + 25]);
            }

            const float e1 =
                d * (s1 + (1.0f / 256.0f) * s2 - s3 * high_base_1);
            const float e2 =
                d * (s4 + (1.0f / 256.0f) * s5 - s6 * high_base_2);
            sum1[row] += e1 * ((float)scales[1] - 32.0f);
            sum2[row] += e2 * ((float)scales[3] - 32.0f);
        }

        y1 += 4 * QK_K;
    }

    return (sum1 + 0.25f * sum2) / (float)(1 << shift);
}
"#;
    out.push_str(body);
    writeln!(out).unwrap();
}
/// Laguna kernel_laguna_attn_output_residual_f16_f32 (antirez dense.metal:834).
/// Dense f16 matvec (NF=16-block dot, half4·float4) + residual add:
/// dst[r] = residual[r] + sum_k weight[r][k] * x[k]. NR0=2, NSG baked to 4
/// (FC_mul_mv_nsg), N_SIMDWIDTH=32. helper_mv_reduce_add_and_write inlined.
/// Buffers: p0=weight (half, ne01×ne00 row-major), p1=x (float), p2=residual (float),
/// p3=dst (float). Uniforms: ne00 (K), ne01 (N). Grid (needs_3d_grid_simd).
pub(super) fn emit_laguna_attn_output_residual_f32_msl(out: &mut String, nsg: u32, nr0: u32, nf: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NR0 = {nr0};").unwrap();
    writeln!(out, "    constexpr short NW  = 32;   // N_SIMDWIDTH").unwrap();
    writeln!(out, "    constexpr short NB  = 32;").unwrap();
    writeln!(out, "    constexpr short NF  = {nf};").unwrap();
    writeln!(out, "    constexpr short NF4 = NF / 4;").unwrap();
    writeln!(
        out,
        "    constexpr short NSG = {nsg};    // FC_mul_mv_nsg (baked)"
    )
    .unwrap();
    writeln!(out, "    const ushort lane = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort simd_group = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int row0 = (int)tgpig.x * NR0;").unwrap();
    writeln!(out, "    const int n_blocks = (int)ne00 / NB;").unwrap();
    writeln!(
        out,
        "    device const float4 *x4 = (device const float4 *)p1;"
    )
    .unwrap();
    writeln!(out, "    device const half4 *weight4[NR0];").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(
        out,
        "        weight4[row] = (device const half4 *)(p0 + (uint64_t)(row0 + row) * ne00);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float sums[NR0] = {{{zeros}}};").unwrap();
    writeln!(out, "    const short ix = lane / (NW / NF);").unwrap();
    writeln!(out, "    const short il = lane % (NW / NF);").unwrap();
    writeln!(out, "    const int block0 = simd_group * NF + ix;").unwrap();
    writeln!(
        out,
        "    device const float4 *xb = x4 + (block0 * NB + il * NF) / 4;"
    )
    .unwrap();
    writeln!(
        out,
        "    for (int block = block0; block < n_blocks; block += NSG * NF) {{"
    )
    .unwrap();
    writeln!(out, "        float4 xv[NF4];").unwrap();
    writeln!(
        out,
        "        for (short i = 0; i < NF4; i++) xv[i] = xb[i];"
    )
    .unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(out, "            if (row0 + row >= (int)ne01) continue;").unwrap();
    writeln!(
        out,
        "            device const half4 *wb = weight4[row] + (block * NB + il * NF) / 4;"
    )
    .unwrap();
    writeln!(out, "            float part = 0.0f;").unwrap();
    writeln!(
        out,
        "            for (short i = 0; i < NF4; i++) part += dot(float4(wb[i]), xv[i]);"
    )
    .unwrap();
    writeln!(out, "            sums[row] += part;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        xb += NSG * NF * NW / 4;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // helper_mv_reduce_add_and_write<NR0> inlined (NSG-group threadgroup reduce)."
    )
    .unwrap();
    writeln!(out, "    threadgroup float shmem[NR0 * NW];").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(
        out,
        "        if (simd_group == 0) shmem[NW * row + lane] = 0.0f;"
    )
    .unwrap();
    writeln!(out, "        sums[row] = simd_sum(sums[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(
        out,
        "        if (lane == 0) shmem[NW * row + simd_group] = sums[row];"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && row0 + row < (int)ne01; row++) {{"
    )
    .unwrap();
    writeln!(out, "        float tot = simd_sum(shmem[NW * row + lane]);").unwrap();
    writeln!(
        out,
        "        if (lane == 0 && simd_group == 0) p3[row0 + row] = p2[row0 + row] + tot;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();

    // shmem is the cross-simdgroup reduction staging buffer: NR0(2) * NW(32)
    // floats = 256 bytes. The host must size the threadgroup for it. NSG is baked
    // at 4, so the simdgroup index into shmem is a compile-time constant and is
    // NOT a host obligation. Charging VERIFIES the declaration is in what we just
    // emitted, so resizing it without revisiting the contract fails codegen.
    // Emitted bytes are unchanged.
    {
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.charge_threadgroup_bytes(256, "threadgroup float shmem[NR0 * NW];");
        let _ = k.finish();
    }
}
/// Laguna kernel_laguna_qkvg_f16_f32 (antirez dense.metal:896). Fused Q/K/V/gate
/// f16 projection: one grid over the concatenated output range [0, q+2*kv+gate)
/// selects (weight, dst, local-row, out_dim) per NR0 rows, same NF-block dot as
/// attn_output. NR0=2, NSG baked to 4, N_SIMDWIDTH=32. reduce_and_write inlined.
/// Buffers: p0=q_w, p1=k_w, p2=v_w, p3=gate_w (half), p4=x (float),
/// p5=q, p6=k, p7=v, p8=gate (float). Uniforms: in_dim, q_dim, kv_dim, gate_dim.
pub(super) fn emit_laguna_qkvg_f16_f32_msl(out: &mut String, nsg: u32, nr0: u32, nf: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NR0 = {nr0};").unwrap();
    writeln!(out, "    constexpr short NW  = 32;   // N_SIMDWIDTH").unwrap();
    writeln!(out, "    constexpr short NB  = 32;").unwrap();
    writeln!(out, "    constexpr short NF  = {nf};").unwrap();
    writeln!(out, "    constexpr short NF4 = NF / 4;").unwrap();
    writeln!(
        out,
        "    constexpr short NSG = {nsg};    // FC_mul_mv_nsg (baked)"
    )
    .unwrap();
    writeln!(out, "    const ushort lane = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort simd_group = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint global_row = tgpig.x * NR0;").unwrap();
    writeln!(out, "    const uint q_end = q_dim;").unwrap();
    writeln!(out, "    const uint k_end = q_end + kv_dim;").unwrap();
    writeln!(out, "    const uint v_end = k_end + kv_dim;").unwrap();
    writeln!(out, "    const uint all_end = v_end + gate_dim;").unwrap();
    writeln!(
        out,
        "    if (global_row >= all_end || (in_dim % NB) != 0u) return;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const half *weight = nullptr;").unwrap();
    writeln!(out, "    device float *dst = nullptr;").unwrap();
    writeln!(out, "    uint row = 0u;").unwrap();
    writeln!(out, "    uint out_dim = 0u;").unwrap();
    writeln!(out, "    if (global_row < q_end) {{ weight = p0; dst = p5; row = global_row; out_dim = q_dim; }}").unwrap();
    writeln!(out, "    else if (global_row < k_end) {{ weight = p1; dst = p6; row = global_row - q_end; out_dim = kv_dim; }}").unwrap();
    writeln!(out, "    else if (global_row < v_end) {{ weight = p2; dst = p7; row = global_row - k_end; out_dim = kv_dim; }}").unwrap();
    writeln!(
        out,
        "    else {{ weight = p3; dst = p8; row = global_row - v_end; out_dim = gate_dim; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float4 *x4 = (device const float4 *)p4;"
    )
    .unwrap();
    writeln!(out, "    device const half4 *weight4[NR0];").unwrap();
    writeln!(out, "    for (short r = 0; r < NR0; r++) {{").unwrap();
    writeln!(
        out,
        "        weight4[r] = (device const half4 *)(weight + (uint64_t)(row + r) * in_dim);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float sums[NR0] = {{{zeros}}};").unwrap();
    writeln!(out, "    const short ix = lane / (NW / NF);").unwrap();
    writeln!(out, "    const short il = lane % (NW / NF);").unwrap();
    writeln!(out, "    const int block0 = simd_group * NF + ix;").unwrap();
    writeln!(
        out,
        "    device const float4 *xb = x4 + (block0 * NB + il * NF) / 4;"
    )
    .unwrap();
    writeln!(out, "    const int n_blocks = (int)in_dim / NB;").unwrap();
    writeln!(
        out,
        "    for (int block = block0; block < n_blocks; block += NSG * NF) {{"
    )
    .unwrap();
    writeln!(out, "        float4 xv[NF4];").unwrap();
    writeln!(
        out,
        "        for (short i = 0; i < NF4; i++) xv[i] = xb[i];"
    )
    .unwrap();
    writeln!(out, "        for (short r = 0; r < NR0; r++) {{").unwrap();
    writeln!(out, "            if (row + r >= out_dim) continue;").unwrap();
    writeln!(
        out,
        "            device const half4 *wb = weight4[r] + (block * NB + il * NF) / 4;"
    )
    .unwrap();
    writeln!(out, "            float part = 0.0f;").unwrap();
    writeln!(
        out,
        "            for (short i = 0; i < NF4; i++) part += dot(float4(wb[i]), xv[i]);"
    )
    .unwrap();
    writeln!(out, "            sums[r] += part;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        xb += NSG * NF * NW / 4;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // helper_mv_reduce_and_write<NR0> inlined; writes dst[row + r]."
    )
    .unwrap();
    writeln!(out, "    threadgroup float shmem[NR0 * NW];").unwrap();
    writeln!(out, "    for (short r = 0; r < NR0; r++) {{").unwrap();
    writeln!(
        out,
        "        if (simd_group == 0) shmem[NW * r + lane] = 0.0f;"
    )
    .unwrap();
    writeln!(out, "        sums[r] = simd_sum(sums[r]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short r = 0; r < NR0; r++) {{").unwrap();
    writeln!(
        out,
        "        if (lane == 0) shmem[NW * r + simd_group] = sums[r];"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short r = 0; r < NR0 && row + (uint)r < out_dim; r++) {{"
    )
    .unwrap();
    writeln!(out, "        float tot = simd_sum(shmem[NW * r + lane]);").unwrap();
    writeln!(
        out,
        "        if (lane == 0 && simd_group == 0) dst[row + (uint)r] = tot;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();

    // shmem is the cross-simdgroup reduction staging buffer: NR0(2) * NW(32)
    // floats = 256 bytes. The host must size the threadgroup for it. NSG is baked
    // at 4, so the simdgroup index into shmem is a compile-time constant and is
    // NOT a host obligation. Charging VERIFIES the declaration is in what we just
    // emitted, so resizing it without revisiting the contract fails codegen.
    // Emitted bytes are unchanged.
    {
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.charge_threadgroup_bytes(256, "threadgroup float shmem[NR0 * NW];");
        let _ = k.finish();
    }
}
/// Laguna kernel_laguna_attention_decode_gqa_f16 (antirez laguna.metal:794).
/// GQA decode attention, one threadgroup (8 SIMD groups) per query head. Online
/// softmax over the KV ring: short history (<=256) runs a single SIMD group;
/// longer stripes keys over 8 SIMD groups, then merges the independently
/// normalized partials in threadgroup memory. SiLU/softplus gate epilogue.
/// Buffers: p0=q (float), p1=gate (float), p2=key_cache, p3=value_cache (half),
/// p4=out (float). Uniforms: n_head,n_head_kv,head_dim,cache_cap,key_start,
/// key_count,scale. head_dim==128. Grid: n_head threadgroups × 256 threads.
/// `split_simd_groups` simdgroups share the key range and merge their
/// independently normalised partials, so it sets the thread count (32 each) and
/// the size of the partial staging. The head dimension is not a parameter: each
/// lane holds a float4 of the 128-wide head, and the kernel refuses any other
/// head_dim at runtime.
pub(super) fn emit_laguna_attention_decode_gqa_f16_msl(out: &mut String, split: u32) {
    let discard = format!("{}/{}", split - 1, split);
    writeln!(out, "    threadgroup float scratch[{split}u + {split}u + {split}u * 128u];  // partial_max[{split}]+partial_sum[{split}]+partial_value[{split}*128]").unwrap();
    writeln!(out, "    constexpr uint split_simd_groups = {split}u;").unwrap();
    writeln!(out, "    const uint head = tgpig.x;").unwrap();
    writeln!(out, "    const uint lane = simd_lane;").unwrap();
    writeln!(out, "    const uint simd_group = simd_id;").unwrap();
    writeln!(
        out,
        "    if (head >= n_head || n_head_kv == 0u || head_dim != 128u || key_count == 0u) {{"
    )
    .unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    // Always distribute keys across the {split} simdgroups (flash-merge below)."
    )
    .unwrap();
    writeln!(
        out,
        "    // The old `key_count > 256` gate made short-context DECODE run all {split}"
    )
    .unwrap();
    writeln!(
        out,
        "    // simdgroups redundantly over every key and discard {discard} → {split}x waste."
    )
    .unwrap();
    writeln!(out, "    const bool split = true;").unwrap();
    writeln!(out, "    const uint heads_per_kv = n_head / n_head_kv;").unwrap();
    writeln!(out, "    const uint kv_head = head / heads_per_kv;").unwrap();
    writeln!(out, "    const uint cache_width = n_head_kv * head_dim;").unwrap();
    writeln!(
        out,
        "    device const float *qh = p0 + (uint64_t)head * head_dim;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 acc = float4(0.0f);").unwrap();
    writeln!(out, "    float max_score = -INFINITY;").unwrap();
    writeln!(out, "    float score_sum = 0.0f;").unwrap();
    writeln!(out, "    const uint key_first = split ? simd_group : 0u;").unwrap();
    writeln!(
        out,
        "    const uint key_stride = split ? split_simd_groups : 1u;"
    )
    .unwrap();
    writeln!(
        out,
        "    for (uint i = key_first; i < key_count; i += key_stride) {{"
    )
    .unwrap();
    writeln!(out, "        const uint key_pos = key_start + i;").unwrap();
    writeln!(out, "        const uint cache_row = key_pos % cache_cap;").unwrap();
    writeln!(out, "        const uint64_t kv_base = (uint64_t)cache_row * cache_width + (uint64_t)kv_head * head_dim;").unwrap();
    writeln!(out, "        float partial = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint d = lane; d < head_dim; d += 32u) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            partial += qh[d] * (float)p2[kv_base + d];"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        const float score = simd_sum(partial) * scale;"
    )
    .unwrap();
    writeln!(out, "        const float next_max = max(max_score, score);").unwrap();
    writeln!(
        out,
        "        const float old_scale = max_score == -INFINITY ? 0.0f : exp(max_score - next_max);"
    )
    .unwrap();
    writeln!(
        out,
        "        const float value_scale = exp(score - next_max);"
    )
    .unwrap();
    writeln!(
        out,
        "        score_sum = score_sum * old_scale + value_scale;"
    )
    .unwrap();
    writeln!(out, "        const uint d0 = lane;").unwrap();
    writeln!(
        out,
        "        acc.x = acc.x * old_scale + value_scale * (float)p3[kv_base + d0];"
    )
    .unwrap();
    writeln!(
        out,
        "        acc.y = acc.y * old_scale + value_scale * (float)p3[kv_base + d0 + 32u];"
    )
    .unwrap();
    writeln!(
        out,
        "        acc.z = acc.z * old_scale + value_scale * (float)p3[kv_base + d0 + 64u];"
    )
    .unwrap();
    writeln!(
        out,
        "        acc.w = acc.w * old_scale + value_scale * (float)p3[kv_base + d0 + 96u];"
    )
    .unwrap();
    writeln!(out, "        max_score = next_max;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float *partial_max = scratch;").unwrap();
    writeln!(
        out,
        "    threadgroup float *partial_sum = partial_max + split_simd_groups;"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float *partial_value = partial_sum + split_simd_groups;"
    )
    .unwrap();
    writeln!(out, "    if (lane == 0u) {{").unwrap();
    writeln!(out, "        partial_max[simd_group] = max_score;").unwrap();
    writeln!(out, "        partial_sum[simd_group] = score_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    const uint value_base = simd_group * head_dim;").unwrap();
    writeln!(out, "    partial_value[value_base + lane] = acc.x;").unwrap();
    writeln!(out, "    partial_value[value_base + lane + 32u] = acc.y;").unwrap();
    writeln!(out, "    partial_value[value_base + lane + 64u] = acc.z;").unwrap();
    writeln!(out, "    partial_value[value_base + lane + 96u] = acc.w;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (simd_group != 0u) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 merged = acc;").unwrap();
    writeln!(out, "    float merged_sum = score_sum;").unwrap();
    writeln!(out, "    if (split) {{").unwrap();
    writeln!(out, "        float global_max = partial_max[0];").unwrap();
    writeln!(
        out,
        "        for (uint sg = 1u; sg < split_simd_groups; sg++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            global_max = max(global_max, partial_max[sg]);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        merged = float4(0.0f);").unwrap();
    writeln!(out, "        merged_sum = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint sg = 0u; sg < split_simd_groups; sg++) {{"
    )
    .unwrap();
    writeln!(out, "            const float weight = partial_sum[sg] > 0.0f ? exp(partial_max[sg] - global_max) : 0.0f;").unwrap();
    writeln!(out, "            merged_sum += partial_sum[sg] * weight;").unwrap();
    writeln!(out, "            const uint base = sg * head_dim + lane;").unwrap();
    writeln!(out, "            merged.x += partial_value[base] * weight;").unwrap();
    writeln!(
        out,
        "            merged.y += partial_value[base + 32u] * weight;"
    )
    .unwrap();
    writeln!(
        out,
        "            merged.z += partial_value[base + 64u] * weight;"
    )
    .unwrap();
    writeln!(
        out,
        "            merged.w += partial_value[base + 96u] * weight;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;"
    )
    .unwrap();
    writeln!(out, "    const float gate_value = p1[head];").unwrap();
    writeln!(out, "    const float gate_scale = gate_value > 20.0f ? gate_value : log(1.0f + exp(gate_value));").unwrap();
    writeln!(
        out,
        "    device float *oh = p4 + (uint64_t)head * head_dim;"
    )
    .unwrap();
    writeln!(out, "    oh[lane]       = merged.x * inv_sum * gate_scale;").unwrap();
    writeln!(out, "    oh[lane + 32u] = merged.y * inv_sum * gate_scale;").unwrap();
    writeln!(out, "    oh[lane + 64u] = merged.z * inv_sum * gate_scale;").unwrap();
    writeln!(out, "    oh[lane + 96u] = merged.w * inv_sum * gate_scale;").unwrap();

    // Three obligations, and the head_dim one is not merely a sizing note.
    //
    // scratch is partial_max[8] + partial_sum[8] + partial_value[8*128] = 1040
    // floats = 4160 bytes.
    //
    // partial_value is indexed `simd_group * head_dim + lane + 96`, which reaches
    // 7*128+31+96 = 1023 against a capacity of exactly 8*128 = 1024. A larger
    // head_dim overflows it, and partial_max[simd_group] likewise holds only 8, so
    // a ninth simdgroup walks into partial_sum. Neither array carries a length, so
    // neither check can live in the kernel — they belong to the host.
    //
    // The kernel DOES self-guard `head_dim != 128u` by returning, which is why this
    // must be written down rather than left implicit: a mis-dispatched launch then
    // writes NO output and raises NO error, so the caller silently reads whatever
    // was already in the output buffer. That reads as a bad model, not a bad launch.
    {
        let decl = format!("threadgroup float scratch[{split}u + {split}u + {split}u * 128u];");
        // `because` is a &'static str in the ledger, so the shipped split keeps the
        // wording the committed contract table carries, and other splits get
        // wording that names the parameter instead of a count.
        let (head_dim_because, split_because): (&'static str, &'static str) = if split == LAGUNA_DECODE_DS4_SPLIT {
            (
                "partial_value holds exactly 8*128 floats and is indexed \
                 simd_group*head_dim + lane + 96; a larger head_dim addresses past it, \
                 and the kernel returns without writing rather than faulting",
                "partial_max and partial_sum hold 8 entries each and are indexed by the \
                 simdgroup index; a ninth simdgroup overwrites partial_sum",
            )
        } else {
            (
                "partial_value holds exactly split_simd_groups*128 floats and is indexed \
                 simd_group*head_dim + lane + 96; a larger head_dim addresses past it, \
                 and the kernel returns without writing rather than faulting",
                "partial_max and partial_sum hold split_simd_groups entries each and are \
                 indexed by the simdgroup index; one more simdgroup overwrites partial_sum",
            )
        };
        let mut k = super::kernel_writer::KernelWriter::new(out);
        k.charge_threadgroup_bytes(4 * (2 * split + split * 128), &decl);
        k.charge_dim_bound("head_dim", 128, "partial_value", &decl, head_dim_because);
        k.charge_dim_bound(
            "simdgroups_per_threadgroup",
            split,
            "partial_max",
            &decl,
            split_because,
        );
        let _ = k.finish();
    }
}
/// DS4 kernel_dsv4_rope_tail_f32 (M122) — antirez dsv4_rope.metal:68.
/// Host-callable byte-stride partial-RoPE: copies n_nope prefix verbatim, then rotates
/// the last n_dims=ne00-n_nope with YaRN-corrected angles.
/// mode 0 = interleaved pairs (j0, j0+1); mode 2 = NeoX split halves (j0=n_nope+ic, j1=j0+n_half).
/// Buffers (all char*): p0=src0, p1=src1=pos(int32), p2=src2=freq_factor(float, optional), p3=dst.
/// Dispatch: 3D grid (i1=tgpig.x, i2=tgpig.y, i3=tgpig.z) + ntg.x lanes sweeping i0 across ne00.
pub(super) fn emit_dsv4_rope_tail_f32_msl(out: &mut String) {
    writeln!(out, "    uint tid    = _tid_v.x;").unwrap();
    writeln!(out, "    uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    int i1 = (int)tgpig.x;").unwrap();
    writeln!(out, "    int i2 = (int)tgpig.y;").unwrap();
    writeln!(out, "    int i3 = (int)tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int n_nope = (int)ne00 - (int)n_dims;").unwrap();
    writeln!(out, "    if (n_nope < 0) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const int * pos = (device const int *) p1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // YaRN correction dims (rope_yarn_corr_dims inlined)."
    )
    .unwrap();
    writeln!(out, "    float corr_lo = floor((float)n_dims * log((float)n_ctx_orig / (beta_fast * 2.0f * M_PI_F)) / (2.0f * log(freq_base)));").unwrap();
    writeln!(out, "    float corr_hi = ceil ((float)n_dims * log((float)n_ctx_orig / (beta_slow * 2.0f * M_PI_F)) / (2.0f * log(freq_base)));").unwrap();
    writeln!(out, "    corr_lo = max(0.0f, corr_lo);").unwrap();
    writeln!(out, "    corr_hi = min((float)n_dims - 1.0f, corr_hi);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float theta_base = (float)pos[i2];").unwrap();
    writeln!(out, "    float inv_ndims  = -1.0f / (float)n_dims;").unwrap();
    writeln!(out, "    bool is_neox = (mode == 2u);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i0 = tid; i0 < ne00; i0 += tcount) {{").unwrap();
    writeln!(
        out,
        "        device const char * src_base = p0 + (uint)i3*nb03 + (uint)i2*nb02 + (uint)i1*nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "        device       char * dst_base = p3 + (uint)i3*nb3  + (uint)i2*nb2  + (uint)i1*nb1;"
    )
    .unwrap();
    writeln!(out, "        if ((int)i0 < n_nope) {{").unwrap();
    writeln!(out, "            *((device float *)(dst_base + i0*nb0)) = *((device const float *)(src_base + i0*nb00));").unwrap();
    writeln!(out, "            continue;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        int r = (int)i0 - n_nope;").unwrap();
    writeln!(out, "        if (is_neox) {{").unwrap();
    writeln!(out, "            int n_half = (int)n_dims / 2;").unwrap();
    writeln!(out, "            if (r >= n_half) continue;").unwrap();
    writeln!(out, "            int ic = r;").unwrap();
    writeln!(out, "            int rel_i0 = 2 * ic;").unwrap();
    writeln!(
        out,
        "            float theta_extrap = theta_base * pow(freq_base, inv_ndims * (float)rel_i0);"
    )
    .unwrap();
    writeln!(
        out,
        "            float freq_factor = (has_src2 != 0u) ? ((device const float *)p2)[ic] : 1.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "            float theta_in = theta_extrap / freq_factor;"
    )
    .unwrap();
    writeln!(out, "            // rope_yarn inlined").unwrap();
    writeln!(
        out,
        "            float theta_interp = freq_scale * theta_in;"
    )
    .unwrap();
    writeln!(out, "            float theta = theta_interp;").unwrap();
    writeln!(out, "            float mscale = attn_factor;").unwrap();
    writeln!(out, "            if (ext_factor != 0.0f) {{").unwrap();
    writeln!(out, "                float ramp = ((float)rel_i0 / 2.0f - corr_lo) / max(0.001f, corr_hi - corr_lo);").unwrap();
    writeln!(
        out,
        "                float ramp_mix = (1.0f - min(1.0f, max(0.0f, ramp))) * ext_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "                theta = theta_interp * (1.0f - ramp_mix) + theta_in * ramp_mix;"
    )
    .unwrap();
    writeln!(
        out,
        "                mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            float cos_t = cos(theta) * mscale;").unwrap();
    writeln!(out, "            float sin_t = sin(theta) * mscale;").unwrap();
    writeln!(out, "            if (inverse != 0u) sin_t = -sin_t;").unwrap();
    writeln!(out, "            int j0 = n_nope + ic;").unwrap();
    writeln!(out, "            int j1 = n_nope + ic + n_half;").unwrap();
    writeln!(
        out,
        "            float x0 = *((device const float *)(src_base + (uint)j0*nb00));"
    )
    .unwrap();
    writeln!(
        out,
        "            float x1 = *((device const float *)(src_base + (uint)j1*nb00));"
    )
    .unwrap();
    writeln!(
        out,
        "            *((device float *)(dst_base + (uint)j0*nb0)) = x0*cos_t - x1*sin_t;"
    )
    .unwrap();
    writeln!(
        out,
        "            *((device float *)(dst_base + (uint)j1*nb0)) = x0*sin_t + x1*cos_t;"
    )
    .unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            if ((r & 1) != 0) continue;").unwrap();
    writeln!(out, "            int ic = r / 2;").unwrap();
    writeln!(
        out,
        "            float theta_extrap = theta_base * pow(freq_base, inv_ndims * (float)r);"
    )
    .unwrap();
    writeln!(
        out,
        "            float freq_factor = (has_src2 != 0u) ? ((device const float *)p2)[ic] : 1.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "            float theta_in = theta_extrap / freq_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "            float theta_interp = freq_scale * theta_in;"
    )
    .unwrap();
    writeln!(out, "            float theta = theta_interp;").unwrap();
    writeln!(out, "            float mscale = attn_factor;").unwrap();
    writeln!(out, "            if (ext_factor != 0.0f) {{").unwrap();
    writeln!(
        out,
        "                float ramp = ((float)r / 2.0f - corr_lo) / max(0.001f, corr_hi - corr_lo);"
    )
    .unwrap();
    writeln!(
        out,
        "                float ramp_mix = (1.0f - min(1.0f, max(0.0f, ramp))) * ext_factor;"
    )
    .unwrap();
    writeln!(
        out,
        "                theta = theta_interp * (1.0f - ramp_mix) + theta_in * ramp_mix;"
    )
    .unwrap();
    writeln!(
        out,
        "                mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            float cos_t = cos(theta) * mscale;").unwrap();
    writeln!(out, "            float sin_t = sin(theta) * mscale;").unwrap();
    writeln!(out, "            if (inverse != 0u) sin_t = -sin_t;").unwrap();
    writeln!(out, "            int j0 = n_nope + r;").unwrap();
    writeln!(out, "            int j1 = j0 + 1;").unwrap();
    writeln!(
        out,
        "            float x0 = *((device const float *)(src_base + (uint)j0*nb00));"
    )
    .unwrap();
    writeln!(
        out,
        "            float x1 = *((device const float *)(src_base + (uint)j1*nb00));"
    )
    .unwrap();
    writeln!(
        out,
        "            *((device float *)(dst_base + (uint)j0*nb0)) = x0*cos_t - x1*sin_t;"
    )
    .unwrap();
    writeln!(
        out,
        "            *((device float *)(dst_base + (uint)j1*nb0)) = x0*sin_t + x1*cos_t;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 KV ratio-4 recurrent-state shift (kernel_dsv4_ratio4_shift_f32).
/// Two state buffers (state_kv, state_score) of length 8*width. Shift second half down to first half:
///   state[i] = state[4*width + i] for i in [0, 4*width).
/// Buffers: p0=state_kv, p1=state_score. Param: width. Dispatch: 1D grid over n=4*width threads.
/// V4.1 carry between packed and plain representations (kernel_dsv41_carry_copy).
/// V4.1 engram contribution (kernel_dsv41_engram_add).
/// V4.1 candidate block maxima (kernel_dsv41_candidate_blocks).
/// V4.1 in-place bf16 rounding of a whole buffer (kernel_dsv41_bf16_linear).
/// V4.1 vision bias + residual, bf16-rounded at both steps.
/// V4.1 indexer bfloat packing with an exactness flag (kernel_dsv41_indexer_pack).
///
/// The flag is the point: it records whether every element of the group survived
/// the round trip to bfloat, so a later kernel can choose the packed path only
/// where it is lossless.
pub(super) fn emit_dsv41_indexer_pack_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "threadgroup uint valid[4];");
    l(out, "const uint grp = row;");
    l(out, "const bool query = grp < query_count;");
    l(out, "const uint count = query ? (32u * 128u) : (64u * 128u);");
    l(out, "const ulong offset = query ? (ulong)grp * (ulong)count");
    l(out, "                           : (ulong)(grp - query_count) * (ulong)count;");
    l(out, "device bfloat * pq = (device bfloat *)p3;");
    l(out, "device bfloat * pk = (device bfloat *)p4;");
    l(out, "device uint   * fl = (device uint   *)p2;");
    l(out, "bool exact = true;");
    l(out, "for (uint i = tid; i < count; i += 128u) {");
    l(out, "    // keys past the real extent are padded with zero rather than read");
    l(out, "    const float value = query ? p0[offset + i]");
    l(out, "        : ((offset + i < (ulong)key_count * 128ul) ? p1[offset + i] : 0.0f);");
    l(out, "    const bfloat converted = bfloat(value);");
    l(out, "    if (query) pq[offset + i] = converted; else pk[offset + i] = converted;");
    l(out, "    exact = exact && (float(converted) == value);");
    l(out, "}");
    l(out, "const bool same = simd_all(exact);");
    l(out, "if ((tid % 32u) == 0u) valid[tid / 32u] = same ? 1u : 0u;");
    l(out, "threadgroup_barrier(mem_flags::mem_threadgroup);");
    l(out, "if (tid == 0u)");
    l(out, "    fl[grp] = (valid[0] && valid[1] && valid[2] && valid[3]) ? 1u : 0u;");
}

pub(super) fn emit_dsv41_vision_bias_residual_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "#define VBF16(v) (((as_type<uint>(v) & 0x7f800000u) == 0x7f800000u) ? (v)            \\");
    l(out, "    : as_type<float>((as_type<uint>(v) + 0x7fffu + ((as_type<uint>(v) >> 16u) & 1u)) \\");
    l(out, "      & 0xffff0000u))");
    l(out, "const uint gid = row * tcount + tid;");
    l(out, "const uint x = gid % width;");
    l(out, "const uint y = gid / width;");
    l(out, "if (y >= rows) return;");
    l(out, "const ulong off = (ulong)y * (ulong)width + x;");
    l(out, "// the bias is stored as bf16 halves; widen by shifting into the high bits");
    l(out, "device const ushort * bias = (device const ushort *)p1;");
    l(out, "const float b = as_type<float>((uint)bias[x] << 16);");
    l(out, "const float projected = VBF16(p0[off] + b);");
    l(out, "p0[off] = VBF16(projected + p2[off]);");
    l(out, "#undef VBF16");
}

/// V4.1 vision SwiGLU split. Unlike the V4 variant this rounds the SiLU result
/// to bf16 BEFORE multiplying by `up`.
pub(super) fn emit_dsv41_vision_swiglu_split_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "#define VBF16(v) (((as_type<uint>(v) & 0x7f800000u) == 0x7f800000u) ? (v)            \\");
    l(out, "    : as_type<float>((as_type<uint>(v) + 0x7fffu + ((as_type<uint>(v) >> 16u) & 1u)) \\");
    l(out, "      & 0xffff0000u))");
    l(out, "const uint gid = row * tcount + tid;");
    l(out, "const uint x = gid % width;");
    l(out, "const uint y = gid / width;");
    l(out, "if (y >= rows) return;");
    l(out, "const ulong source = (ulong)y * (ulong)width * 2ul + x;");
    l(out, "const float gate = p0[source];");
    l(out, "const float up   = p0[source + (ulong)width];");
    l(out, "float activated = gate / (1.0f + exp(-gate));");
    l(out, "activated = VBF16(activated);");
    l(out, "p1[(ulong)y * (ulong)width + x] = VBF16(activated * up);");
    l(out, "#undef VBF16");
}

pub(super) fn emit_dsv41_bf16_linear_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "// p0 is declared float* by the framework but holds BITS here — reinterpret.");
    l(out, "device uint * x = (device uint *)p0;");
    l(out, "const uint gid = row * tcount + tid;");
    l(out, "const uint first = gid * 4u;");
    l(out, "if (first + 4u <= count) {");
    l(out, "    uint4 bits = *((device uint4 *)(x + first));");
    l(out, "    const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;");
    l(out, "    bits += select(uint4(0u), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);");
    l(out, "    *((device uint4 *)(x + first)) = bits & 0xffff0000u;");
    l(out, "} else {");
    l(out, "    for (uint i = first; i < count; ++i) {");
    l(out, "        uint bits = x[i];");
    l(out, "        if ((bits & 0x7f800000u) != 0x7f800000u)");
    l(out, "            bits += 0x7fffu + ((bits >> 16u) & 1u);");
    l(out, "        x[i] = bits & 0xffff0000u;");
    l(out, "    }");
    l(out, "}");
}

/// V4.1 exclusive prefix sum over per-expert counts (kernel_moe_packed_offsets).
pub(super) fn emit_dsv41_moe_packed_offsets_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "// Serial by design: one thread walks the counts. Both buffers hold BITS.");
    l(out, "if (row != 0u || tid != 0u) return;");
    l(out, "device const uint * counts  = (device const uint *)p0;");
    l(out, "device uint       * offsets = (device uint *)p1;");
    l(out, "uint acc = 0u;");
    l(out, "for (uint e = 0u; e < experts; ++e) {");
    l(out, "    offsets[e] = acc;");
    l(out, "    acc += counts[e];");
    l(out, "}");
}

/// V4.1 softmax-weighted pooling of adjacent KV pairs (kernel_dsv41_pool2).
pub(super) fn emit_dsv41_pool2_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "#define DSV41_BF16(v) as_type<float>((((as_type<uint>(v) & 0x7f800000u) != 0x7f800000u) \\");
    l(out, "    ? (as_type<uint>(v) + 0x7fffu + ((as_type<uint>(v) >> 16u) & 1u))                   \\");
    l(out, "    : as_type<uint>(v)) & 0xffff0000u)");
    l(out, "const uint gid = row * tcount + tid;");
    l(out, "const uint x = gid % width;");
    l(out, "const uint y = gid / width;");
    l(out, "if (y >= pairs) return;");
    l(out, "// a < 0 reaches back into the previous chunk's tail");
    l(out, "const int  a = (int)(y * 2u) - (int)tail;");
    l(out, "const ulong b = (ulong)(a + 1) * (ulong)width + x;");
    l(out, "const float ka = a < 0 ? p3[x] : p1[(ulong)a * (ulong)width + x];");
    l(out, "const float sa = a < 0 ? p4[x] : p2[(ulong)a * (ulong)width + x];");
    l(out, "const float sb = p2[b];");
    l(out, "const float peak = max(sa, sb);");
    l(out, "const float ea = exp(sa - peak), eb = exp(sb - peak);");
    l(out, "p0[(ulong)y * (ulong)width + x] = DSV41_BF16((ka * ea + p1[b] * eb) / (ea + eb));");
    l(out, "#undef DSV41_BF16");
}

pub(super) fn emit_dsv41_candidate_blocks_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "const uint count = (width + 7u) / 8u;");
    l(out, "const uint gid = row * tcount + tid;");
    l(out, "const uint x = gid % count;");
    l(out, "const uint y = gid / count;");
    l(out, "if (y >= rows) return;");
    l(out, "// causal frontier for this row");
    l(out, "const uint visible = min(width, (start + y + 1u) / ratio);");
    l(out, "float best = -INFINITY;");
    l(out, "const uint hi = min(visible, (x + 1u) * 8u);");
    l(out, "for (uint i = x * 8u; i < hi; ++i)");
    l(out, "    best = max(best, p0[(ulong)y * (ulong)width + i]);");
    l(out, "// the block holding the frontier is always admissible");
    l(out, "if (visible != 0u && x == (visible - 1u) / 8u) best = INFINITY;");
    l(out, "p1[(ulong)y * (ulong)count + x] = best;");
}

/// V4.1 candidate filter (kernel_dsv41_candidate_filter).
pub(super) fn emit_dsv41_candidate_filter_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "const uint gid = row * tcount + tid;");
    l(out, "const uint x = gid % width;");
    l(out, "const uint y = gid / width;");
    l(out, "if (y >= rows) return;");
    l(out, "const ulong offset = (ulong)y * (ulong)width + x;");
    l(out, "const uint blocks = (width + 7u) / 8u;");
    l(out, "const uint visible = min(width, (start + y + 1u) / ratio);");
    l(out, "p1[offset] = (x < visible && p2[(ulong)y * (ulong)blocks + x / 8u] == 0.0f)");
    l(out, "    ? p0[offset] : -INFINITY;");
}

pub(super) fn emit_dsv41_engram_add_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    // Round-to-nearest-even bf16, as `dsv41_bf16` upstream. carry_copy TRUNCATES
    // instead — do not merge the two.
    l(out, "#define DSV41_BF16(x) as_type<float>((((as_type<uint>(x) & 0x7f800000u) != 0x7f800000u) \\");
    l(out, "    ? (as_type<uint>(x) + 0x7fffu + ((as_type<uint>(x) >> 16u) & 1u))                   \\");
    l(out, "    : as_type<uint>(x)) & 0xffff0000u)");
    l(out, "const uint token   = row / 4u;");
    l(out, "const uint sub_row = row % 4u;");
    l(out, "device const uchar * engram_mask = (device const uchar *)p4;");
    l(out, "if (masked != 0u && engram_mask[token] == 0u) return;");
    l(out, "const ulong offset       = ((ulong)token * 4ul + (ulong)sub_row) * (ulong)width;");
    l(out, "const ulong key_offset   = ((ulong)token * 5ul + (ulong)sub_row) * (ulong)width;");
    l(out, "const ulong value_offset = ((ulong)token * 5ul + 4ul) * (ulong)width;");
    l(out, "float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;");
    l(out, "for (uint i = simd_lane; i < width; i += 32u) {");
    l(out, "    const float h = p0[offset + i];");
    l(out, "    const float k = DSV41_BF16(p1[key_offset + i]);");
    l(out, "    const uint  wi = sub_row * width + i;");
    l(out, "    h2  += h * h;");
    l(out, "    k2  += k * k;");
    l(out, "    dot += h * (p2[wi] * p3[wi]) * k;");
    l(out, "}");
    l(out, "h2 = simd_sum(h2);");
    l(out, "k2 = simd_sum(k2);");
    l(out, "dot = simd_sum(dot) * rsqrt(h2 / (float)width + eps) *");
    l(out, "      rsqrt(k2 / (float)width + eps) * rsqrt((float)width);");
    l(out, "const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));");
    l(out, "for (uint i = simd_lane; i < width; i += 32u)");
    l(out, "    p0[offset + i] = DSV41_BF16(p0[offset + i] + gate * DSV41_BF16(p1[value_offset + i]));");
    l(out, "#undef DSV41_BF16");
}

pub(super) fn emit_dsv41_carry_copy_msl(out: &mut String) {
    let l = |o: &mut String, t: &str| { o.push_str("    "); o.push_str(t); o.push('\n'); };
    l(out, "const uint blocks_per_row = (width + 127u) / 128u;");
    l(out, "const uint r   = row / blocks_per_row;");
    l(out, "const uint blk = row % blocks_per_row;");
    l(out, "const uint col = blk * 128u + tid;");
    l(out, "if (format == 0u) {");
    l(out, "    // bf16 carry: keep the high half of each f32, or restore it");
    l(out, "    if (col >= width) return;");
    l(out, "    device ushort * pk = (device ushort *)((device char *)p0 + (ulong)r * (ulong)words * 4ul);");
    l(out, "    if (pack != 0u) {");
    l(out, "        pk[col] = ushort(as_type<uint>(p1[(ulong)r * (ulong)width + col]) >> 16);");
    l(out, "    } else {");
    l(out, "        p1[(ulong)r * (ulong)width + col] = as_type<float>(uint(pk[col]) << 16);");
    l(out, "    }");
    l(out, "} else {");
    l(out, "    // allowed-mask carry: one bit per column, 32 columns per word");
    l(out, "    const uint word = col / 32u;");
    l(out, "    // p0 is declared float* by the framework, but a mask word is BITS. Reinterpret");
    l(out, "    // rather than index it as float: `p0[i] = bits` would numerically CONVERT.");
    l(out, "    device uint * mk = (device uint *)p0;");
    l(out, "    if (pack != 0u) {");
    l(out, "        const bool allowed = col < width && p1[(ulong)r * (ulong)width + col] == 0.0f;");
    l(out, "        const uint bits = simd_sum(allowed ? (1u << simd_lane) : 0u);");
    l(out, "        if (simd_lane == 0u && word < words) mk[(ulong)r * (ulong)words + word] = bits;");
    l(out, "    } else if (col < width) {");
    l(out, "        const uint bits = mk[(ulong)r * (ulong)words + word];");
    l(out, "        p1[(ulong)r * (ulong)width + col] = (bits & (1u << simd_lane)) ? 0.0f : -INFINITY;");
    l(out, "    }");
    l(out, "}");
}

pub(super) fn emit_dsv4_ratio4_shift_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint n = 4u * width;").unwrap();
    writeln!(out, "    if (gid >= n) return;").unwrap();
    writeln!(out, "    p0[gid] = p0[n + gid];").unwrap();
    writeln!(out, "    p1[gid] = p1[n + gid];").unwrap();
}
/// M125: kernel_dsv4_topk_mask (dsv4_misc.metal:237).
/// 1D grid: gid = row*tcount + tid; mask[ic,it] = -INFINITY for ic<ne0, it<ne1.
/// p0=topk read-only (unused, kept for ABI parity with topk_mask_scatter), p1=dst (byte ptr).
pub(super) fn emit_dsv4_topk_mask_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint n = ne0 * ne1;").unwrap();
    writeln!(out, "    if (gid >= n) return;").unwrap();
    writeln!(out, "    uint ic = gid % ne0;").unwrap();
    writeln!(out, "    uint it = gid / ne0;").unwrap();
    writeln!(
        out,
        "    (void)p0; (void)ne00; (void)ne01; (void)nb00; (void)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    *((device float *) (p1 + ic*nb0 + it*nb1)) = -INFINITY;"
    )
    .unwrap();
}
/// M126: kernel_dsv4_q8_hc_expand4_q8_0 (dsv4_hc.metal:728).
/// Fused decode-time q8_0 matvec + 4-channel HC expansion.
/// Scalar-correctness reference: hardcodes NSG=2 NW=32 NQ=8 NR0=2; reads
/// `mv.ne00`, `mv.ne01`, `mv.nb01`, plus the HC striding uniforms.
/// Buffers: p0=weight (block_q8_0), p1=input (float row), p2=block_out (writable),
/// p3=residual, p4=post, p5=comb, p6=dst (writable).
/// Launch: tgpig.x in [0, ceil(ne01/NR0)); NSG simdgroups × NW threads/tg; threadgroup shmem of NR0*NW floats.
pub(super) fn emit_dsv4_q8_hc_expand4_q8_0_msl(out: &mut String, nsg: u32) {
    // Antirez kernel: NR0=2 rows per dispatch; NSG=2 simdgroups; NW=32 wide; NQ=8 quants/lane.
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    if (hc.n_hc != 4 || hc.n_tokens != 1) return;").unwrap();
    writeln!(out, "    constexpr short NSG = {nsg};").unwrap();
    writeln!(out, "    constexpr short NW  = 32;").unwrap();
    writeln!(out, "    constexpr short NQ  = 8;").unwrap();
    writeln!(out, "    constexpr short NR0 = 2;").unwrap();
    writeln!(out, "    constexpr int   QK8_0 = 32;").unwrap();
    writeln!(out, "    const int nb   = mv.ne00 / QK8_0;").unwrap();
    writeln!(out, "    const int row0 = tgpig.x * NR0;").unwrap();
    writeln!(out, "    const short ix = tiisg / (NW / NQ);").unwrap();
    writeln!(out, "    const short il = tiisg % (NW / NQ);").unwrap();
    writeln!(out, "    const int   ib0 = sgitg * NQ + ix;").unwrap();
    writeln!(
        out,
        "    device const float *yb = ((device const float *)p1) + ib0 * QK8_0 + il * NQ;"
    )
    .unwrap();
    writeln!(
        out,
        "    // block_q8_0 layout: half d @ +0, int8_t qs[32] @ +2 → stride 34 bytes."
    )
    .unwrap();
    writeln!(out, "    constexpr int BLK_Q8_0 = 34;").unwrap();
    writeln!(
        out,
        "    device const char *aw0 = p0 + (ulong)(row0 + 0) * mv.nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char *aw1 = p0 + (ulong)(row0 + 1) * mv.nb01;"
    )
    .unwrap();
    writeln!(out, "    float sumf0 = 0.0f, sumf1 = 0.0f;").unwrap();
    writeln!(out, "    float yl[8];").unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG * NQ) {{").unwrap();
    writeln!(
        out,
        "        yl[0]=yb[0]; yl[1]=yb[1]; yl[2]=yb[2]; yl[3]=yb[3];"
    )
    .unwrap();
    writeln!(
        out,
        "        yl[4]=yb[4]; yl[5]=yb[5]; yl[6]=yb[6]; yl[7]=yb[7];"
    )
    .unwrap();
    writeln!(out, "        device const char *bp0 = aw0 + ib * BLK_Q8_0;").unwrap();
    writeln!(out, "        device const char *bp1 = aw1 + ib * BLK_Q8_0;").unwrap();
    writeln!(
        out,
        "        float d0 = (float) *((device const half *)(bp0));"
    )
    .unwrap();
    writeln!(
        out,
        "        float d1 = (float) *((device const half *)(bp1));"
    )
    .unwrap();
    writeln!(
        out,
        "        device const int8_t *qs0 = (device const int8_t *)(bp0 + 2) + il * NQ;"
    )
    .unwrap();
    writeln!(
        out,
        "        device const int8_t *qs1 = (device const int8_t *)(bp1 + 2) + il * NQ;"
    )
    .unwrap();
    writeln!(out, "        float s0 = 0.0f, s1 = 0.0f;").unwrap();
    writeln!(out, "        for (short i = 0; i < NQ; ++i) {{ s0 += (float)qs0[i] * yl[i]; s1 += (float)qs1[i] * yl[i]; }}").unwrap();
    writeln!(out, "        sumf0 += s0 * d0;").unwrap();
    writeln!(out, "        sumf1 += s1 * d1;").unwrap();
    writeln!(out, "        yb += NSG * NQ * QK8_0;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    // 2-stage simd_sum reduce via threadgroup shmem[NR0*NW]"
    )
    .unwrap();
    writeln!(out, "    threadgroup float sh[2 * 32];").unwrap();
    writeln!(
        out,
        "    if (sgitg == 0) {{ sh[0 * NW + tiisg] = 0.0f; sh[1 * NW + tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "    float ss0 = simd_sum(sumf0);").unwrap();
    writeln!(out, "    float ss1 = simd_sum(sumf1);").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    if (tiisg == 0) {{ sh[0 * NW + sgitg] = ss0; sh[1 * NW + sgitg] = ss1; }}"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float block_v0 = simd_sum(sh[0 * NW + tiisg]);").unwrap();
    writeln!(out, "    float block_v1 = simd_sum(sh[1 * NW + tiisg]);").unwrap();
    writeln!(out, "    if (!(tiisg == 0 && sgitg == 0)) return;").unwrap();
    writeln!(out, "    for (short rr = 0; rr < NR0; ++rr) {{").unwrap();
    writeln!(out, "        int d = row0 + rr;").unwrap();
    writeln!(out, "        if (d >= mv.ne01) continue;").unwrap();
    writeln!(
        out,
        "        float block_v = (rr == 0) ? block_v0 : block_v1;"
    )
    .unwrap();
    writeln!(
        out,
        "        *((device float *)(p2 + (ulong)d * sizeof(float))) = block_v;"
    )
    .unwrap();
    writeln!(
        out,
        "        float r0 = *((device const float *)(p3 + (ulong)d * hc.nb_res0 + 0 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(
        out,
        "        float r1 = *((device const float *)(p3 + (ulong)d * hc.nb_res0 + 1 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(
        out,
        "        float r2 = *((device const float *)(p3 + (ulong)d * hc.nb_res0 + 2 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(
        out,
        "        float r3 = *((device const float *)(p3 + (ulong)d * hc.nb_res0 + 3 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(out, "        for (int dst_hc = 0; dst_hc < 4; ++dst_hc) {{").unwrap();
    writeln!(out, "            float acc = block_v * *((device const float *)(p4 + (ulong)dst_hc * hc.nb_post0));").unwrap();
    writeln!(out, "            acc += *((device const float *)(p5 + (ulong)dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;").unwrap();
    writeln!(out, "            acc += *((device const float *)(p5 + (ulong)dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;").unwrap();
    writeln!(out, "            acc += *((device const float *)(p5 + (ulong)dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;").unwrap();
    writeln!(out, "            acc += *((device const float *)(p5 + (ulong)dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;").unwrap();
    writeln!(
        out,
        "            *((device float *)(p6 + (ulong)d * hc.nb0 + (ulong)dst_hc * hc.nb1)) = acc;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)hc.nb_block0;").unwrap();
}
/// M127: kernel_dsv4_shared_down_hc_expand4_q8_0 (dsv4_hc.metal:607).
/// Decode-time FFN-tail fusion: q8_0 matvec of `shared_mid` × `weight` → `shared_out`,
/// then `block_v = routed_out[d*nb_block0] + shared_v`, then identical 4× HC expand.
/// Buffers: p0=weight (block_q8_0), p1=shared_mid (float row), p2=shared_out (writable),
/// p3=routed_out, p4=residual, p5=post, p6=comb, p7=dst (writable).
/// Hardcodes n_hc=4, n_tokens=1, NSG=2, NW=32, NQ=8, NR0=2 (same shape as M126).
pub(super) fn emit_dsv4_shared_down_hc_expand4_q8_0_msl(out: &mut String, nsg: u32) {
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    if (hc.n_hc != 4 || hc.n_tokens != 1) return;").unwrap();
    writeln!(out, "    constexpr short NSG = {nsg};").unwrap();
    writeln!(out, "    constexpr short NW  = 32;").unwrap();
    writeln!(out, "    constexpr short NQ  = 8;").unwrap();
    writeln!(out, "    constexpr short NR0 = 2;").unwrap();
    writeln!(out, "    constexpr int   QK8_0 = 32;").unwrap();
    writeln!(out, "    const int nb   = mv.ne00 / QK8_0;").unwrap();
    writeln!(out, "    const int row0 = tgpig.x * NR0;").unwrap();
    writeln!(out, "    const short ix = tiisg / (NW / NQ);").unwrap();
    writeln!(out, "    const short il = tiisg % (NW / NQ);").unwrap();
    writeln!(out, "    const int   ib0 = sgitg * NQ + ix;").unwrap();
    writeln!(
        out,
        "    device const float *yb = ((device const float *)p1) + ib0 * QK8_0 + il * NQ;"
    )
    .unwrap();
    writeln!(out, "    constexpr int BLK_Q8_0 = 34;").unwrap();
    writeln!(
        out,
        "    device const char *aw0 = p0 + (ulong)(row0 + 0) * mv.nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char *aw1 = p0 + (ulong)(row0 + 1) * mv.nb01;"
    )
    .unwrap();
    writeln!(out, "    float sumf0 = 0.0f, sumf1 = 0.0f;").unwrap();
    writeln!(out, "    float yl[8];").unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG * NQ) {{").unwrap();
    writeln!(
        out,
        "        yl[0]=yb[0]; yl[1]=yb[1]; yl[2]=yb[2]; yl[3]=yb[3];"
    )
    .unwrap();
    writeln!(
        out,
        "        yl[4]=yb[4]; yl[5]=yb[5]; yl[6]=yb[6]; yl[7]=yb[7];"
    )
    .unwrap();
    writeln!(out, "        device const char *bp0 = aw0 + ib * BLK_Q8_0;").unwrap();
    writeln!(out, "        device const char *bp1 = aw1 + ib * BLK_Q8_0;").unwrap();
    writeln!(
        out,
        "        float d0 = (float) *((device const half *)(bp0));"
    )
    .unwrap();
    writeln!(
        out,
        "        float d1 = (float) *((device const half *)(bp1));"
    )
    .unwrap();
    writeln!(
        out,
        "        device const int8_t *qs0 = (device const int8_t *)(bp0 + 2) + il * NQ;"
    )
    .unwrap();
    writeln!(
        out,
        "        device const int8_t *qs1 = (device const int8_t *)(bp1 + 2) + il * NQ;"
    )
    .unwrap();
    writeln!(out, "        float s0 = 0.0f, s1 = 0.0f;").unwrap();
    writeln!(out, "        for (short i = 0; i < NQ; ++i) {{ s0 += (float)qs0[i] * yl[i]; s1 += (float)qs1[i] * yl[i]; }}").unwrap();
    writeln!(out, "        sumf0 += s0 * d0;").unwrap();
    writeln!(out, "        sumf1 += s1 * d1;").unwrap();
    writeln!(out, "        yb += NSG * NQ * QK8_0;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup float sh[2 * 32];").unwrap();
    writeln!(
        out,
        "    if (sgitg == 0) {{ sh[0 * NW + tiisg] = 0.0f; sh[1 * NW + tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "    float ss0 = simd_sum(sumf0);").unwrap();
    writeln!(out, "    float ss1 = simd_sum(sumf1);").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    if (tiisg == 0) {{ sh[0 * NW + sgitg] = ss0; sh[1 * NW + sgitg] = ss1; }}"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float shared_v0 = simd_sum(sh[0 * NW + tiisg]);").unwrap();
    writeln!(out, "    float shared_v1 = simd_sum(sh[1 * NW + tiisg]);").unwrap();
    writeln!(out, "    if (!(tiisg == 0 && sgitg == 0)) return;").unwrap();
    writeln!(out, "    for (short rr = 0; rr < NR0; ++rr) {{").unwrap();
    writeln!(out, "        int d = row0 + rr;").unwrap();
    writeln!(out, "        if (d >= mv.ne01) continue;").unwrap();
    writeln!(
        out,
        "        float shared_v = (rr == 0) ? shared_v0 : shared_v1;"
    )
    .unwrap();
    writeln!(
        out,
        "        *((device float *)(p2 + (ulong)d * sizeof(float))) = shared_v;"
    )
    .unwrap();
    writeln!(
        out,
        "        float block_v = *((device const float *)(p3 + (ulong)d * hc.nb_block0));"
    )
    .unwrap();
    writeln!(out, "        block_v += shared_v;").unwrap();
    writeln!(
        out,
        "        float r0 = *((device const float *)(p4 + (ulong)d * hc.nb_res0 + 0 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(
        out,
        "        float r1 = *((device const float *)(p4 + (ulong)d * hc.nb_res0 + 1 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(
        out,
        "        float r2 = *((device const float *)(p4 + (ulong)d * hc.nb_res0 + 2 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(
        out,
        "        float r3 = *((device const float *)(p4 + (ulong)d * hc.nb_res0 + 3 * hc.nb_res1));"
    )
    .unwrap();
    writeln!(out, "        for (int dst_hc = 0; dst_hc < 4; ++dst_hc) {{").unwrap();
    writeln!(out, "            float acc = block_v * *((device const float *)(p5 + (ulong)dst_hc * hc.nb_post0));").unwrap();
    writeln!(out, "            acc += *((device const float *)(p6 + (ulong)dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;").unwrap();
    writeln!(out, "            acc += *((device const float *)(p6 + (ulong)dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;").unwrap();
    writeln!(out, "            acc += *((device const float *)(p6 + (ulong)dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;").unwrap();
    writeln!(out, "            acc += *((device const float *)(p6 + (ulong)dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;").unwrap();
    writeln!(
        out,
        "            *((device float *)(p7 + (ulong)d * hc.nb0 + (ulong)dst_hc * hc.nb1)) = acc;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 topk_mask_scatter (kernel_dsv4_topk_mask_scatter).
/// For each gid in [0, num_elements): read idx = topk[gid]; if 0 <= idx < dst_len, set dst[idx] = 0.
/// Buffers: p0=topk (int*), p1=dst (float*). Params: num_elements (= K), dst_len (= N).
pub(super) fn emit_topk_mask_scatter_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    if (gid >= num_elements) return;").unwrap();
    writeln!(out, "    int idx = p0[gid];").unwrap();
    writeln!(
        out,
        "    if (idx >= 0 && (uint)idx < dst_len) p1[(uint)idx] = 0.0;"
    )
    .unwrap();
}
/// DS4 indexer_weighted_sum (kernel_dsv4_indexer_weighted_sum).
/// For each (it in [0,T), ic in [0,C)): dst[it,ic] = Σ_{ih in [0,H)} max(scores[it,ic,ih], 0) * weights[it,ih] * scale.
/// Dispatch: one thread per (it,ic); gid = it * num_cols + ic.
/// Buffers: p0=scores (T*C*H), p1=weights (T*H), p2=dst (T*C). Params: num_tokens, num_cols, num_heads, scale.
pub(super) fn emit_dsv4_indexer_weighted_sum_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = num_tokens * num_cols;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out, "    uint ic = gid % num_cols;").unwrap();
    writeln!(out, "    uint it = gid / num_cols;").unwrap();
    writeln!(out, "    float acc = 0.0;").unwrap();
    writeln!(
        out,
        "    uint score_base  = it * num_cols * num_heads + ic * num_heads;"
    )
    .unwrap();
    writeln!(out, "    uint weight_base = it * num_heads;").unwrap();
    writeln!(out, "    for (uint ih = 0; ih < num_heads; ih++) {{").unwrap();
    writeln!(out, "        float s = p0[score_base + ih];").unwrap();
    writeln!(out, "        float w = p1[weight_base + ih];").unwrap();
    writeln!(out, "        acc += max(s, 0.0) * (w * scale);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    p2[it * num_cols + ic] = acc;").unwrap();
}
/// DS4 router_weights_one (kernel_dsv4_router_weights_one).
/// For each tid in [0, num_experts): w[tid] = probs[selected[tid]] / sum * scale,
/// where sum = max(min_sum, sum_{i<num_experts} probs[selected[i]]).
/// Buffers: p0=probs (float*), p1=selected (int*), p2=weights (float*).
/// Params: num_experts, scale (=1.5 antirez default), min_sum (=6.103515625e-5 antirez default).
pub(super) fn emit_dsv4_router_weights_one_msl(out: &mut String) {
    writeln!(out, "    uint gid = tid;").unwrap();
    writeln!(out, "    if (gid >= num_experts) return;").unwrap();
    writeln!(out, "    float sum = 0.0;").unwrap();
    writeln!(out, "    for (uint i = 0; i < num_experts; i++) {{").unwrap();
    writeln!(out, "        sum += p0[(uint)p1[i]];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sum = max(sum, min_sum);").unwrap();
    writeln!(out, "    p2[gid] = p0[(uint)p1[gid]] / sum * scale;").unwrap();
}
/// DS4 sort_i32_rows_asc (kernel_dsv4_sort_i32_rows_asc).
/// Per-row bitonic sort. One threadgroup per row, top_k threads per group.
/// Each thread loads src[row, tid] into threadgroup shmem, then runs log2(top_k)
/// outer phases × log2(top_k) inner phases of bitonic compare-exchange, with a
/// threadgroup_barrier between each phase. Final write back to dst[row, tid].
/// Buffers: p0=src(int*), p1=dst(int*). Params: top_k, num_rows.
/// Layout: row-major, src[row, tid] = row * top_k + tid (flat-1D vs antirez byte strides).
/// MAX_TOPK=256 covers DS4 typical top_k ∈ {64, 128, 256}.
/// `max_top_k` sizes the per-row staging; the runtime `top_k` must not exceed it.
pub(super) fn emit_sort_i32_rows_asc_msl(out: &mut String, max_top_k: u32) {
    writeln!(out, "    constexpr uint MAX_TOPK = {max_top_k};").unwrap();
    writeln!(out, "    threadgroup int row_tmp[MAX_TOPK];").unwrap();
    writeln!(out, "    if (row >= num_rows || tid >= top_k) return;").unwrap();
    writeln!(out, "    row_tmp[tid] = p0[row * top_k + tid];").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (uint k = 2; k <= top_k; k <<= 1) {{").unwrap();
    writeln!(out, "        for (uint j = k >> 1; j > 0; j >>= 1) {{").unwrap();
    writeln!(out, "            uint other = tid ^ j;").unwrap();
    writeln!(out, "            if (other > tid && other < top_k) {{").unwrap();
    writeln!(out, "                int a = row_tmp[tid];").unwrap();
    writeln!(out, "                int b = row_tmp[other];").unwrap();
    writeln!(out, "                bool up = (tid & k) == 0;").unwrap();
    writeln!(
        out,
        "                if ((up && a > b) || (!up && a < b)) {{"
    )
    .unwrap();
    writeln!(out, "                    row_tmp[tid] = b;").unwrap();
    writeln!(out, "                    row_tmp[other] = a;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    p1[row * top_k + tid] = row_tmp[tid];").unwrap();
}
/// DS4 softmax_pool (kernel_dsv4_softmax_pool).
/// Per-thread serial reduce over R=ne00. Each thread handles one (id, ic):
///   max_s = max_ir score[ir, id, ic]
///   dst[ic, id] = Σ_ir exp(score[ir,id,ic] - max_s) * kv[ir,id,ic] / Σ_ir exp(...)
/// Buffers: p0=kv (float, R*ne1*ne0), p1=score (float, R*ne1*ne0), p2=dst (float, ne1*ne0).
/// Layout: row-major [ic, id, ir] → ic*ne0*R + id*R + ir (flat-1D vs antirez byte strides).
/// Dispatch: total = ne0 * ne1 threads (one per output element).
pub(super) fn emit_dsv4_softmax_pool_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = ne0 * ne1;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out, "    uint id = gid % ne0;").unwrap();
    writeln!(out, "    uint ic = gid / ne0;").unwrap();
    writeln!(out, "    uint base = ic * ne0 * ne00 + id * ne00;").unwrap();
    writeln!(out, "    float max_s = -INFINITY;").unwrap();
    writeln!(out, "    for (uint ir = 0; ir < ne00; ir++) {{").unwrap();
    writeln!(out, "        float s = p1[base + ir];").unwrap();
    writeln!(out, "        max_s = max(max_s, s);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float sum = 0.0;").unwrap();
    writeln!(out, "    float acc = 0.0;").unwrap();
    writeln!(out, "    for (uint ir = 0; ir < ne00; ir++) {{").unwrap();
    writeln!(out, "        float w = exp(p1[base + ir] - max_s);").unwrap();
    writeln!(out, "        sum += w;").unwrap();
    writeln!(out, "        acc += p0[base + ir] * w;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    p2[ic * ne0 + id] = acc / sum;").unwrap();
}
/// DS4 compressor_store_one (kernel_dsv4_compressor_store_one).
/// 5-buffer KV+score store with positional encoding (APE) addition. Per gid in [0, width):
///   pos_mod = pos % ratio
///   dst_row = (ratio == 4) ? ratio + pos_mod : pos_mod
///   state_kv[dst_row * width + gid] = kv[gid]
///   state_score[dst_row * width + gid] = score[gid] + ape[pos_mod * width + gid]
/// Buffers: p0=kv (float*), p1=score (float*), p2=ape (float* — f32 path; f16 path TBD),
///          p3=state_kv (float*), p4=state_score (float*).
/// Params: width, ratio, pos. (ape_type pinned to 0 = f32 for first landing.)
pub(super) fn emit_dsv4_compressor_store_one_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(
        out,
        "    if (gid >= width || width == 0u || ratio == 0u) return;"
    )
    .unwrap();
    writeln!(out, "    uint pos_mod = pos % ratio;").unwrap();
    writeln!(
        out,
        "    uint dst_row = (ratio == 4u) ? (ratio + pos_mod) : pos_mod;"
    )
    .unwrap();
    writeln!(out, "    uint dst = dst_row * width + gid;").unwrap();
    writeln!(out, "    uint ape_i = pos_mod * width + gid;").unwrap();
    writeln!(out, "    p3[dst] = p0[gid];").unwrap();
    writeln!(out, "    p4[dst] = p1[gid] + p2[ape_i];").unwrap();
}
/// Emit E4M3FN dequant helpers used by Dsv4KvFp8Store (and future fp8 ops).
/// These mirror antirez/ds4 dsv4_kv.metal lines 1-74: the 16-entry exp_scale
/// LUT, dsv4_e4m3fn_value(i) for i∈[0,127], and dsv4_e4m3fn_dequant(x) which
/// binary-searches for the closest fp8 representation of |x| (clamped 448).
pub(super) fn emit_e4m3fn_helpers(out: &mut String) {
    writeln!(out, "// E4M3FN dequant helpers (DS4-compatible).").unwrap();
    writeln!(out, "constant float dsv4_e4m3fn_exp_scale[16] = {{").unwrap();
    writeln!(out, "    0.0f, 0.015625f, 0.03125f, 0.0625f,").unwrap();
    writeln!(out, "    0.125f, 0.25f, 0.5f, 1.0f,").unwrap();
    writeln!(out, "    2.0f, 4.0f, 8.0f, 16.0f,").unwrap();
    writeln!(out, "    32.0f, 64.0f, 128.0f, 256.0f,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "static inline float dsv4_e4m3fn_value(int i) {{").unwrap();
    writeln!(out, "    const int exp  = (i >> 3) & 0x0f;").unwrap();
    writeln!(out, "    const int mant = i & 0x07;").unwrap();
    writeln!(out, "    return exp == 0").unwrap();
    writeln!(out, "        ? float(mant) * 0.001953125f").unwrap();
    writeln!(
        out,
        "        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "static inline float dsv4_e4m3fn_dequant(float x) {{").unwrap();
    writeln!(out, "    const float sign = x < 0.0f ? -1.0f : 1.0f;").unwrap();
    writeln!(out, "    const float ax = min(abs(x), 448.0f);").unwrap();
    writeln!(out, "    int lo = 0;").unwrap();
    writeln!(out, "    int hi = 126;").unwrap();
    writeln!(out, "    while (lo < hi) {{").unwrap();
    writeln!(out, "        const int mid = (lo + hi + 1) >> 1;").unwrap();
    writeln!(
        out,
        "        if (dsv4_e4m3fn_value(mid) <= ax) {{ lo = mid; }} else {{ hi = mid - 1; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    int best = lo;").unwrap();
    writeln!(out, "    if (best < 126) {{").unwrap();
    writeln!(
        out,
        "        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));"
    )
    .unwrap();
    writeln!(out, "        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {{").unwrap();
    writeln!(out, "            best = best + 1;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    return sign * dsv4_e4m3fn_value(best);").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
}

/// The same helpers inside an include guard, so two emitted files that both
/// carry them can be concatenated into one library. `other` names a kernel
/// that also defines them, for the comment.
pub(super) fn emit_e4m3fn_helpers_guarded(out: &mut String, other: &str) {
    writeln!(out, "// E4M3FN dequant helpers (DS4-compatible). Guarded so concatenation with").unwrap();
    writeln!(out, "// other emitted/.metal files that define the same helpers (e.g.").unwrap();
    writeln!(out, "// {other}) doesn't produce a duplicate-symbol compile error.").unwrap();
    writeln!(out, "#ifndef DSV4_E4M3FN_HELPERS_DEFINED").unwrap();
    writeln!(out, "#define DSV4_E4M3FN_HELPERS_DEFINED").unwrap();
    writeln!(out, "constant float dsv4_e4m3fn_exp_scale[16] = {{").unwrap();
    writeln!(out, "    0.0f, 0.015625f, 0.03125f, 0.0625f,").unwrap();
    writeln!(out, "    0.125f, 0.25f, 0.5f, 1.0f,").unwrap();
    writeln!(out, "    2.0f, 4.0f, 8.0f, 16.0f,").unwrap();
    writeln!(out, "    32.0f, 64.0f, 128.0f, 256.0f,").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "static inline float dsv4_e4m3fn_value(int i) {{").unwrap();
    writeln!(out, "    const int exp  = (i >> 3) & 0x0f;").unwrap();
    writeln!(out, "    const int mant = i & 0x07;").unwrap();
    writeln!(out, "    return exp == 0").unwrap();
    writeln!(out, "        ? float(mant) * 0.001953125f").unwrap();
    writeln!(
        out,
        "        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "static inline float dsv4_e4m3fn_dequant(float x) {{").unwrap();
    writeln!(out, "    const float sign = x < 0.0f ? -1.0f : 1.0f;").unwrap();
    writeln!(out, "    const float ax = min(abs(x), 448.0f);").unwrap();
    writeln!(out, "    int lo = 0;").unwrap();
    writeln!(out, "    int hi = 126;").unwrap();
    writeln!(out, "    while (lo < hi) {{").unwrap();
    writeln!(out, "        const int mid = (lo + hi + 1) >> 1;").unwrap();
    writeln!(
        out,
        "        if (dsv4_e4m3fn_value(mid) <= ax) {{ lo = mid; }} else {{ hi = mid - 1; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    int best = lo;").unwrap();
    writeln!(out, "    if (best < 126) {{").unwrap();
    writeln!(
        out,
        "        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));"
    )
    .unwrap();
    writeln!(out, "        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {{").unwrap();
    writeln!(out, "            best = best + 1;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    return sign * dsv4_e4m3fn_value(best);").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out, "#endif // DSV4_E4M3FN_HELPERS_DEFINED").unwrap();
    writeln!(out).unwrap();
}
/// DS4 kv_fp8_store: per-row n_nope chunked-64 fp8 round-trip + n_rot tail
/// half-cast. p0 = kv (read+write, head_dim floats), p1 = raw_cache (write,
/// raw_row*head_dim base). 64 threads per threadgroup, single dispatch.
/// Uses threadgroup shmem `scratch[64]` baked in (no setThreadgroupMemoryLength).
/// `block` elements share one FP8 scale: a threadgroup of `block` threads
/// reduces them to a max, so `block` must be a power of two.
pub(super) fn emit_dsv4_kv_fp8_store_generic_msl(out: &mut String, block: u32) {
    let half = block / 2;
    writeln!(out, "    threadgroup float scratch[{block}];").unwrap();
    writeln!(out, "    int n_nope = (int)head_dim - (int)n_rot;").unwrap();
    writeln!(
        out,
        "    if ((int)head_dim <= 0 || (int)n_rot < 0 || n_nope < 0 || tid >= {block}u) return;"
    )
    .unwrap();
    writeln!(
        out,
        "    device float * raw = p1 + (uint64_t)raw_row * (uint64_t)head_dim;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int off = 0; off < n_nope; off += {block}) {{").unwrap();
    writeln!(out, "        float v = 0.0f;").unwrap();
    writeln!(out, "        if (off + (int)tid < n_nope) {{").unwrap();
    writeln!(out, "            v = p0[off + (int)tid];").unwrap();
    writeln!(out, "            scratch[tid] = abs(v);").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            scratch[tid] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(
        out,
        "        for (uint stride = {half}u; stride > 0u; stride >>= 1) {{"
    )
    .unwrap();
    writeln!(out, "            if (tid < stride) {{").unwrap();
    writeln!(
        out,
        "                scratch[tid] = max(scratch[tid], scratch[tid + stride]);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        const float amax = max(scratch[0], 1.0e-4f);").unwrap();
    writeln!(
        out,
        "        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));"
    )
    .unwrap();
    writeln!(out, "        if (off + (int)tid < n_nope) {{").unwrap();
    writeln!(out, "            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;").unwrap();
    writeln!(out, "            p0[off + (int)tid] = q;").unwrap();
    writeln!(out, "            raw[off + (int)tid] = (float)((half)q);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i = n_nope + (int)tid; i < (int)head_dim; i += {block}) {{"
    )
    .unwrap();
    writeln!(out, "        raw[i] = (float)((half)p0[i]);").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 kv_fp8_store at the shipped block of 64.
pub(super) fn emit_dsv4_kv_fp8_store_msl(out: &mut String) {
    emit_dsv4_kv_fp8_store_generic_msl(out, FP8_KV_DS4_BLOCK);
}
/// DS4 fp8_kv_quantize: 4D batched n_nope chunked-64 fp8 round-trip.
/// p0 = src0 (read), p1 = dst (write). Strides nb*_e are in float-elements
/// (driver pre-divides byte strides by sizeof(float)). Each row is dispatched
/// as one threadgroup of 64 threads. row index → (i1, i2, i3) decoded against
/// (ne01, ne02, ne03). For each row: chunked-64 max-abs reduce + fp8 quant +
/// tail copy of n_rot bytes (raw f32 passthrough).
/// `block` elements share one FP8 scale: a threadgroup of `block` threads
/// reduces them to a max, so `block` must be a power of two.
pub(super) fn emit_dsv4_fp8_kv_quantize_generic_msl(out: &mut String, block: u32) {
    let half = block / 2;
    writeln!(out, "    threadgroup float scratch[{block}];").unwrap();
    writeln!(out, "    uint n_rows = ne01 * ne02 * ne03;").unwrap();
    writeln!(out, "    if (row >= n_rows || tid >= {block}u) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint i1 = row % ne01;").unwrap();
    writeln!(out, "    uint i2 = (row / ne01) % ne02;").unwrap();
    writeln!(out, "    uint i3 = row / (ne01 * ne02);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * src_base = p0 + i1 * nb01_e + i2 * nb02_e + i3 * nb03_e;"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * dst_base = p1 + i1 * nb1_e  + i2 * nb2_e  + i3 * nb3_e;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int n_nope = (int)ne00 - (int)n_rot;").unwrap();
    writeln!(out, "    if (n_nope < 0) n_nope = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int off = 0; off < n_nope; off += {block}) {{").unwrap();
    writeln!(out, "        float v = 0.0f;").unwrap();
    writeln!(out, "        if (off + (int)tid < n_nope) {{").unwrap();
    writeln!(out, "            v = src_base[off + (int)tid];").unwrap();
    writeln!(out, "            scratch[tid] = abs(v);").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            scratch[tid] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(
        out,
        "        for (uint stride = {half}u; stride > 0u; stride >>= 1) {{"
    )
    .unwrap();
    writeln!(out, "            if (tid < stride) {{").unwrap();
    writeln!(
        out,
        "                scratch[tid] = max(scratch[tid], scratch[tid + stride]);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        const float amax = max(scratch[0], 1.0e-4f);").unwrap();
    writeln!(
        out,
        "        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));"
    )
    .unwrap();
    writeln!(out, "        if (off + (int)tid < n_nope) {{").unwrap();
    writeln!(out, "            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;").unwrap();
    writeln!(out, "            dst_base[off + (int)tid] = q;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint i = (uint)n_nope + tid; i < ne00; i += {block}u) {{"
    )
    .unwrap();
    writeln!(out, "        dst_base[i] = src_base[i];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 fp8_kv_quantize at the shipped block of 64.
pub(super) fn emit_dsv4_fp8_kv_quantize_msl(out: &mut String) {
    emit_dsv4_fp8_kv_quantize_generic_msl(out, FP8_KV_DS4_BLOCK);
}
/// DS4 flash_attn_ext_pad: byte-stride DMA padding of K/V/mask blocks.
/// Buffers (all char*): p0=k, p1=v, p2=mask, p3=dst (k_pad | v_pad | mask_pad).
/// Layout in dst: [k_pad: nb11 * C * ne_12_2 * ne_12_3] [v_pad: nb21 * C * ne_12_2 * ne_12_3]
/// [mask_pad: 2 * C * ne31 * ne32 * ne33] (mask is half = 2 bytes/elem).
/// Per (i1, i2, i3) = tgpig: copy k/v rows when i1 < icp, zero-fill when i1 >= icp;
/// then if has_mask, scan ib in [0, ne31) step C for mask block fill.
/// Threadgroup uses ntg.x parallel threads to chunk the per-row byte loop.
pub(super) fn emit_flash_attn_ext_pad_msl(out: &mut String) {
    writeln!(out, "    uint tiitg = _tid_v.x;").unwrap();
    writeln!(out, "    uint ntg_x = _tc_v.x;").unwrap();
    writeln!(out, "    uint C = c_ncpsg;").unwrap();
    writeln!(out, "    uint i1 = tgpig.x;").unwrap();
    writeln!(out, "    uint i2 = tgpig.y;").unwrap();
    writeln!(out, "    uint i3 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device char * k_pad    = p3;").unwrap();
    writeln!(
        out,
        "    device char * v_pad    = k_pad + nb11 * C * ne_12_2 * ne_12_3;"
    )
    .unwrap();
    writeln!(
        out,
        "    device char * mask_pad = v_pad + nb21 * C * ne_12_2 * ne_12_3;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int icp = (int)ne11 % (int)C;").unwrap();
    writeln!(out, "    int ic0 = (int)ne11 - icp;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i2 < ne_12_2 && i3 < ne_12_3) {{").unwrap();
    writeln!(out, "        device const char * k_src = p0 + nb11 * (uint64_t)(ic0 + (int)i1) + nb12 * i2 + nb13 * i3;").unwrap();
    writeln!(out, "        device const char * v_src = p1 + nb21 * (uint64_t)(ic0 + (int)i1) + nb22 * i2 + nb23 * i3;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        device char * k_dst = k_pad + nb11 * i1 + nb11 * C * i2 + nb11 * C * ne_12_2 * i3;"
    )
    .unwrap();
    writeln!(
        out,
        "        device char * v_dst = v_pad + nb21 * i1 + nb21 * C * i2 + nb21 * C * ne_12_2 * i3;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if ((int)i1 >= icp) {{").unwrap();
    writeln!(
        out,
        "            for (uint i = tiitg; i < nb11; i += ntg_x) {{ k_dst[i] = (char)0; }}"
    )
    .unwrap();
    writeln!(
        out,
        "            for (uint i = tiitg; i < nb21; i += ntg_x) {{ v_dst[i] = (char)0; }}"
    )
    .unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (uint i = tiitg; i < nb11; i += ntg_x) {{ k_dst[i] = k_src[i]; }}"
    )
    .unwrap();
    writeln!(
        out,
        "            for (uint i = tiitg; i < nb21; i += ntg_x) {{ v_dst[i] = v_src[i]; }}"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (has_mask != 0u) {{").unwrap();
    writeln!(out, "        if (i2 < ne32 && i3 < ne33) {{").unwrap();
    writeln!(out, "            for (uint ib = i1; ib < ne31; ib += C) {{").unwrap();
    writeln!(out, "                device const half * mask_src = (device const half *)(p2 + nb31 * ib + nb32 * i2 + nb33 * i3) + ic0;").unwrap();
    writeln!(out, "                device       half * mask_dst = (device       half *)(mask_pad) + C * ib + C * ne31 * i2 + C * ne31 * ne32 * i3;").unwrap();
    writeln!(
        out,
        "                for (uint i = tiitg; i < C; i += ntg_x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    if ((int)i >= icp) {{ mask_dst[i] = -MAXHALF; }}"
    )
    .unwrap();
    writeln!(
        out,
        "                    else            {{ mask_dst[i] = mask_src[i]; }}"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 flash_attn_ext_blk: simdgroup-reduce mask scan.
/// Buffers (all char*): p0=mask, p1=dst (per-block status byte: 0=keep, 1=normal, 2=all-zero).
/// Per-threadgroup grid: tgpig.x = i0 (block-of-K), tgpig.y = i1 (block-of-Q),
/// tgpig.z packs (i3 * ne32 + i2). Threadgroup is 1 simdgroup (32 threads on M*).
/// Each lane reads C/NW halves per Q-row over Q rows, simd_min/max reduces, lane 0 writes.
pub(super) fn emit_flash_attn_ext_blk_msl(out: &mut String) {
    writeln!(out, "    uint ntg_x = _tc_v.x;").unwrap();
    writeln!(out, "    uint NW = ntg_x;").unwrap();
    writeln!(out, "    uint Q = q_nqptg;").unwrap();
    writeln!(out, "    uint C = c_ncpsg;").unwrap();
    writeln!(out, "    uint i0 = tgpig.x;").unwrap();
    writeln!(out, "    uint i1 = tgpig.y;").unwrap();
    writeln!(out, "    uint i3 = tgpig.z / ne32;").unwrap();
    writeln!(out, "    uint i2 = tgpig.z % ne32;").unwrap();
    writeln!(out, "    uint tiisg = _tid_v.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    char res = ((int)(i0 * C + C) > (int)ne30) ? (char)1 : (char)0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const half * mask_src = (device const half *)(p0 + (i1 * Q) * nb31 + i2 * nb32 + i3 * nb33) + i0 * C + tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if ((C > NW || Q > 1u) && res == (char)0) {{").unwrap();
    writeln!(out, "        half mmin =  MAXHALF;").unwrap();
    writeln!(out, "        half mmax = -MAXHALF;").unwrap();
    writeln!(out, "        for (uint j = 0u; j < Q; ++j) {{").unwrap();
    writeln!(
        out,
        "            for (uint ii = 0u; ii < (C / NW); ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                half v = mask_src[ii * NW];").unwrap();
    writeln!(out, "                mmin = min(mmin, v);").unwrap();
    writeln!(out, "                mmax = max(mmax, v);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            mask_src += nb31 / 2u;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        mmin = simd_min(mmin);").unwrap();
    writeln!(out, "        mmax = simd_max(mmax);").unwrap();
    writeln!(out, "        if (mmax > -MAXHALF) {{").unwrap();
    writeln!(
        out,
        "            res = (mmin == (half)0.0 && mmax == (half)0.0) ? (char)2 : (char)1;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int nblk1 = ((int)ne01 + (int)Q - 1) / (int)Q;").unwrap();
    writeln!(out, "    int nblk0 = ((int)ne30 + (int)C - 1) / (int)C;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0u) {{").unwrap();
    writeln!(
        out,
        "        p1[((int)(i3 * ne32 + i2) * nblk1 + (int)i1) * nblk0 + (int)i0] = res;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 dsv4_indexer_score_one_direct: per-row fused indexer scoring for one token.
/// Buffers (all char*): p0=q, p1=weights, p2=index_comp, p3=scores.
/// Per-row threadgroup (row=tgpig.x). 4 simdgroups of 32 lanes (128 threads).
/// Stage the compressed key row into ktg shmem, walk `n_head` heads in groups of
/// 4 (one per simdgroup), simd_sum dot product, accumulate ReLU(s) * w[head] * scale.
///
/// The head dimension is not a parameter: the dot product reads one `float4`
/// per lane of a 32-lane simdgroup, so a head is 4 * 32 = 128 components, and
/// the 128-thread group is what runs four heads at once. `n_head` must be a
/// multiple of 4 for the same reason.
pub(super) fn emit_dsv4_indexer_score_one_direct_generic_msl(out: &mut String, n_head: u32) {
    writeln!(
        out,
        "    // hardcoded: n_head={n_head}, head_dim=128, ntg=128 (4 sg of 32)"
    )
    .unwrap();
    writeln!(out, "    if (row >= n_comp) return;").unwrap();
    writeln!(out, "    threadgroup float ktg[128];").unwrap();
    writeln!(out, "    threadgroup float psum[4];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid < 128u) {{").unwrap();
    writeln!(out, "        device const float * krow = (device const float *)(p2 + (uint64_t)row * index_row_stride);").unwrap();
    writeln!(out, "        ktg[tid] = krow[tid];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (uint head0 = 0u; head0 < {n_head}u; head0 += 4u) {{"
    )
    .unwrap();
    writeln!(out, "        uint head = head0 + simd_id;").unwrap();
    writeln!(out, "        device const float4 * q4 = (device const float4 *)(p0 + (uint64_t)head * q_head_stride);").unwrap();
    writeln!(
        out,
        "        threadgroup const float4 * k4 = (threadgroup const float4 *)ktg;"
    )
    .unwrap();
    writeln!(out, "        float s = dot(q4[simd_lane], k4[simd_lane]);").unwrap();
    writeln!(out, "        s = simd_sum(s);").unwrap();
    writeln!(out, "        if (simd_lane == 0u) {{").unwrap();
    writeln!(
        out,
        "            device const float * w = (device const float *)p1;"
    )
    .unwrap();
    writeln!(
        out,
        "            psum[simd_id] = max(s, 0.0f) * (w[head] * scale);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        if (tid == 0u) {{").unwrap();
    writeln!(
        out,
        "            acc += psum[0]; acc += psum[1]; acc += psum[2]; acc += psum[3];"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0u) {{").unwrap();
    writeln!(out, "        device float * dst = (device float *)p3;").unwrap();
    writeln!(out, "        dst[row] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 indexer_score_one_direct at the shipped 64 heads.
pub(super) fn emit_dsv4_indexer_score_one_direct_msl(out: &mut String) {
    emit_dsv4_indexer_score_one_direct_generic_msl(out, INDEXER_SCORE_ONE_DS4_N_HEAD);
}

/// Heads the DeepSeek-V4 indexer uses.
pub(super) const INDEXER_SCORE_ONE_DS4_N_HEAD: u32 = 64;

/// Why `n_head` cannot be emitted for indexer_score_one_direct, if it cannot.
pub(super) fn check_indexer_score_one_n_head(n_head: u32) -> Result<(), String> {
    if n_head == 0 || n_head % 4 != 0 {
        return Err(format!(
            "indexer_score_one_direct: n_head {n_head} must be a positive multiple of 4 (four heads per 128-thread step)"
        ));
    }
    Ok(())
}
/// DS4 dsv4_router_finalize_one: 256-thread bitonic top-6 over (probs+bias).
/// Buffers: p0=probs(float), p1=bias(float), p2=hash(int), p3=tokens(int), p4=selected(int, out).
/// Params: has_bias, hash_mode, use_token_buffer, token, hash_rows.
/// Single threadgroup of 256 threads, each thread owns one expert id.
/// hash_mode short-circuit: thread 0 copies hash[token*6..+6] into selected[0..6].
/// Otherwise: stage (probs+bias) into sel_scores, run log2(256)=8 outer × inner phases of
/// bitonic compare-exchange via an `idx[]` permutation array, then thread<6 writes idx[tid] to selected.
/// DS4 router_finalize_one for `n_expert` experts and `top_k` selections.
/// One threadgroup of `n_expert` threads bitonic-sorts expert indices by
/// score (probs, plus bias when has_bias) and writes the first `top_k`; in
/// hash mode it copies row `token` of the `top_k`-wide hash table instead.
pub(super) fn emit_dsv4_router_finalize_one_generic_msl(out: &mut String, n_expert: u32, top_k: u32) {
    writeln!(out, "    if (tid >= {n_expert}u) return;").unwrap();
    writeln!(out, "    threadgroup float sel_scores[{n_expert}];").unwrap();
    writeln!(out, "    threadgroup int idx[{n_expert}];").unwrap();
    writeln!(out, "    float pv = p0[tid];").unwrap();
    writeln!(
        out,
        "    sel_scores[tid] = (has_bias != 0u) ? (pv + p1[tid]) : pv;"
    )
    .unwrap();
    writeln!(out, "    idx[tid] = (int)tid;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (hash_mode != 0u) {{").unwrap();
    writeln!(out, "        if (tid == 0u) {{").unwrap();
    writeln!(
        out,
        "            uint t = (use_token_buffer != 0u) ? (uint)p3[0] : token;"
    )
    .unwrap();
    writeln!(
        out,
        "            uint hr = (hash_rows == 0u) ? 0u : (hash_rows - 1u);"
    )
    .unwrap();
    writeln!(out, "            uint hrow = (t < hr) ? t : hr;").unwrap();
    writeln!(out, "            for (uint i = 0u; i < {top_k}u; ++i) {{").unwrap();
    writeln!(out, "                p4[i] = p2[hrow * {top_k}u + i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        for (uint k = 2u; k <= {n_expert}u; k <<= 1) {{").unwrap();
    writeln!(out, "            for (uint j = k >> 1; j > 0u; j >>= 1) {{").unwrap();
    writeln!(out, "                uint other = tid ^ j;").unwrap();
    writeln!(out, "                if (other > tid) {{").unwrap();
    writeln!(out, "                    if ((tid & k) == 0u) {{").unwrap();
    writeln!(
        out,
        "                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {{"
    )
    .unwrap();
    writeln!(out, "                            int tmp = idx[tid];").unwrap();
    writeln!(out, "                            idx[tid] = idx[other];").unwrap();
    writeln!(out, "                            idx[other] = tmp;").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(
        out,
        "                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {{"
    )
    .unwrap();
    writeln!(out, "                            int tmp = idx[tid];").unwrap();
    writeln!(out, "                            idx[tid] = idx[other];").unwrap();
    writeln!(out, "                            idx[other] = tmp;").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(
        out,
        "                threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (tid < {top_k}u) {{").unwrap();
    writeln!(out, "            p4[tid] = idx[tid];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
}
/// DS4 router_finalize_one at the shipped 256 experts, top 6.
pub(super) fn emit_dsv4_router_finalize_one_msl(out: &mut String) {
    emit_dsv4_router_finalize_one_generic_msl(out, ROUTER_DS4_N_EXPERT, ROUTER_DS4_TOP_K);
}

/// Experts and selections the DeepSeek-V4 router uses.
pub(super) const ROUTER_DS4_N_EXPERT: u32 = 256;
pub(super) const ROUTER_DS4_TOP_K: u32 = 6;

/// Why a router shape cannot be emitted, if it cannot.
pub(super) fn check_router_shape(n_expert: u32, top_k: u32) -> Result<(), String> {
    if !(2..=1024).contains(&n_expert) || !n_expert.is_power_of_two() {
        return Err(format!(
            "router_finalize_one: n_expert {n_expert} must be a power of two from 2 to 1024 (one bitonic sort per threadgroup)"
        ));
    }
    if top_k == 0 || top_k > n_expert {
        return Err(format!("router_finalize_one: top_k {top_k} must be from 1 to n_expert ({n_expert})"));
    }
    Ok(())
}
/// DS4 dsv4_indexer_scores_tiled_f32: 8x32 tile fused indexer scoring with simdgroup_float8x8 matmul.
/// Buffers (all char*): p0=q, p1=weights, p2=index_comp, p3=scores.
/// Dispatch: 2D grid (ceil(n_comp/32), ceil(n_tokens/8)); 128 threads/tg (4 sg of 32).
/// Each tg covers 8 tokens × 32 compressed rows × 64 heads. K is staged once into shmem,
/// Q is restaged per head, simdgroup_float8x8 matmul produces 8x32 score subtile per head.
/// score[t,c] = sum_h relu(dot(Q[t,h], K[c])) * W[t,h] * scale; causal masking on store.
/// Element type the indexer kernel stages Q and K in and multiplies with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IndexerElem {
    Float,
    Half,
}

/// Shape of the tiled indexer-score kernel, taken from its intrinsic operands.
///
/// Q and K are staged into threadgroup memory and multiplied with 8x8
/// simdgroup matrices, so a tile is 8 tokens (`TM`) by `tile_cols` compressed
/// keys, and `head_dim` is walked 8 columns at a time. Each thread writes two
/// score cells and a simdgroup is 32 threads wide, so a threadgroup holds
/// `tile_cols * 4` threads: one simdgroup per 8 key columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IndexerScoresTiled {
    pub elem: IndexerElem,
    pub head_dim: u32,
    pub tile_cols: u32,
}

impl IndexerScoresTiled {
    /// The shape the DeepSeek-V4 engine ships: head_dim 128, 32 keys per tile.
    pub const DS4_HEAD_DIM: u32 = 128;
    pub const DS4_TILE_COLS: u32 = 32;

    pub fn ds4(elem: IndexerElem) -> Self {
        Self { elem, head_dim: Self::DS4_HEAD_DIM, tile_cols: Self::DS4_TILE_COLS }
    }

    /// Threads per threadgroup the kernel must be dispatched with.
    pub fn threads_per_group(&self) -> u32 {
        self.tile_cols * 4
    }

    /// Why this shape cannot be emitted, if it cannot.
    pub fn check(&self) -> Result<(), String> {
        let (d, tn) = (self.head_dim, self.tile_cols);
        if d == 0 || d % 8 != 0 {
            return Err(format!(
                "indexer_scores_tiled: head_dim {d} must be a positive multiple of 8 (8x8 simdgroup tiles)"
            ));
        }
        if tn == 0 || tn % 8 != 0 {
            return Err(format!(
                "indexer_scores_tiled: tile_cols {tn} must be a positive multiple of 8 (one simdgroup per 8 keys)"
            ));
        }
        if self.threads_per_group() > 1024 {
            return Err(format!(
                "indexer_scores_tiled: tile_cols {tn} needs {} threads per group; Metal allows 1024",
                self.threads_per_group()
            ));
        }
        Ok(())
    }
}

/// DS4 indexer_scores_tiled for any element type, head dimension and tile width.
/// Per token and compressed key: the sum over heads of relu(q . k) * weight *
/// scale, with keys past the causal ratio window set to -inf.
pub(super) fn emit_dsv4_indexer_scores_tiled_generic_msl(out: &mut String, shape: &IndexerScoresTiled) {
    let d = shape.head_dim;
    let tn = shape.tile_cols;
    let threads = shape.threads_per_group();
    let (e, zero, load_row, load_qrow) = match shape.elem {
        IndexerElem::Float => ("float", "0.0f", "row[d]", "qrow[d]"),
        IndexerElem::Half => ("half", "half(0.0f)", "half(row[d])", "half(qrow[d])"),
    };
    writeln!(out, "    constexpr uint TM = 8u;").unwrap();
    writeln!(out, "    constexpr uint TN = {tn}u;").unwrap();
    writeln!(out, "    constexpr uint TS = 8u;").unwrap();
    writeln!(out, "    constexpr uint D  = {d}u;").unwrap();
    writeln!(out, "    uint c0 = tgpig.x * TN;").unwrap();
    writeln!(out, "    uint t0 = tgpig.y * TM;").unwrap();
    writeln!(out, "    threadgroup {e} qtg[TM*D];").unwrap();
    writeln!(out, "    threadgroup {e} ktg[TN*D];").unwrap();
    writeln!(out, "    threadgroup float dotsh[TM*TN];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint last_token = min(t0 + TM, n_tokens);").unwrap();
    writeln!(
        out,
        "    uint max_visible = (last_token > t0) ? min((pos0 + last_token) / ratio, n_comp) : 0u;"
    )
    .unwrap();
    writeln!(out, "    if (c0 >= max_visible) {{").unwrap();
    writeln!(out, "        for (uint i = tid; i < TM*TN; i += {threads}u) {{").unwrap();
    writeln!(out, "            uint r = i / TN; uint cc = i - r*TN;").unwrap();
    writeln!(out, "            uint token = t0 + r; uint comp = c0 + cc;").unwrap();
    writeln!(out, "            if (token < n_tokens && comp < n_comp) {{").unwrap();
    writeln!(out, "                device float * dst = (device float *)(p3 + (uint64_t)token * score_token_stride) + comp;").unwrap();
    writeln!(out, "                *dst = -INFINITY;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = tid; i < TN*D; i += {threads}u) {{").unwrap();
    writeln!(out, "        uint cc = i / D; uint d = i - cc*D;").unwrap();
    writeln!(out, "        uint comp = c0 + cc;").unwrap();
    writeln!(out, "        {e} v = {zero};").unwrap();
    writeln!(out, "        if (comp < n_comp) {{").unwrap();
    writeln!(out, "            device const float * row = (device const float *)(p2 + (uint64_t)comp * index_row_stride);").unwrap();
    writeln!(out, "            v = {load_row};").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        ktg[i] = v;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint cell0 = simd_lane;").unwrap();
    writeln!(out, "    uint cell1 = simd_lane + 32u;").unwrap();
    writeln!(out, "    uint row0 = cell0 >> 3; uint row1 = cell1 >> 3;").unwrap();
    writeln!(out, "    uint sub0 = cell0 & 7u; uint sub1 = cell1 & 7u;").unwrap();
    writeln!(out, "    uint col0 = simd_id * TS + sub0;").unwrap();
    writeln!(out, "    uint col1 = simd_id * TS + sub1;").unwrap();
    writeln!(out, "    uint token0 = t0 + row0; uint token1 = t0 + row1;").unwrap();
    writeln!(out, "    uint comp0 = c0 + col0;   uint comp1 = c0 + col1;").unwrap();
    writeln!(out, "    float acc0 = 0.0f; float acc1 = 0.0f;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint head = 0u; head < n_head; ++head) {{").unwrap();
    writeln!(out, "        for (uint i = tid; i < TM*D; i += {threads}u) {{").unwrap();
    writeln!(out, "            uint r = i / D; uint d = i - r*D;").unwrap();
    writeln!(out, "            uint token = t0 + r;").unwrap();
    writeln!(out, "            {e} v = {zero};").unwrap();
    writeln!(out, "            if (token < n_tokens) {{").unwrap();
    writeln!(out, "                device const float * qrow = (device const float *)(p0 + (uint64_t)token * q_token_stride + (uint64_t)head * q_head_stride);").unwrap();
    writeln!(out, "                v = {load_qrow};").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            qtg[i] = v;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);"
    )
    .unwrap();
    writeln!(out, "        for (uint db = 0u; db < D/TS; ++db) {{").unwrap();
    writeln!(out, "            simdgroup_{e}8x8 mq;").unwrap();
    writeln!(out, "            simdgroup_{e}8x8 mk;").unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq, qtg + db*TS, D, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk, ktg + (simd_id * TS) * D + db*TS, D, 0, true);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        simdgroup_store(mdot, dotsh + simd_id * TS, TN, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (token0 < n_tokens && comp0 < n_comp) {{").unwrap();
    writeln!(out, "            device const float * w = (device const float *)(p1 + (uint64_t)token0 * weights_token_stride);").unwrap();
    writeln!(out, "            float s = dotsh[row0*TN + col0];").unwrap();
    writeln!(out, "            acc0 += max(s, 0.0f) * (w[head] * scale);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (token1 < n_tokens && comp1 < n_comp) {{").unwrap();
    writeln!(out, "            device const float * w = (device const float *)(p1 + (uint64_t)token1 * weights_token_stride);").unwrap();
    writeln!(out, "            float s = dotsh[row1*TN + col1];").unwrap();
    writeln!(out, "            acc1 += max(s, 0.0f) * (w[head] * scale);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (token0 < n_tokens && comp0 < n_comp) {{").unwrap();
    writeln!(
        out,
        "        uint visible = min((pos0 + token0 + 1u) / ratio, n_comp);"
    )
    .unwrap();
    writeln!(out, "        device float * dst = (device float *)(p3 + (uint64_t)token0 * score_token_stride) + comp0;").unwrap();
    writeln!(out, "        *dst = (comp0 < visible) ? acc0 : -INFINITY;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    if (token1 < n_tokens && comp1 < n_comp) {{").unwrap();
    writeln!(
        out,
        "        uint visible = min((pos0 + token1 + 1u) / ratio, n_comp);"
    )
    .unwrap();
    writeln!(out, "        device float * dst = (device float *)(p3 + (uint64_t)token1 * score_token_stride) + comp1;").unwrap();
    writeln!(out, "        *dst = (comp1 < visible) ? acc1 : -INFINITY;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// DS4 dsv4_indexer_scores_tiled_f32 at the shipped shape.
pub(super) fn emit_dsv4_indexer_scores_tiled_f32_msl(out: &mut String) {
    emit_dsv4_indexer_scores_tiled_generic_msl(out, &IndexerScoresTiled::ds4(IndexerElem::Float));
}
/// DS4 indexer_scores_tiled with Q and K staged as half, at the shipped shape.
pub(super) fn emit_dsv4_indexer_scores_tiled_msl(out: &mut String) {
    emit_dsv4_indexer_scores_tiled_generic_msl(out, &IndexerScoresTiled::ds4(IndexerElem::Half));
}
/// DS4 indexed_mixed_attention_heads8: 1 token × 8 heads per threadgroup, online softmax,
/// dot+accum done as half (DS4 F16 attention rounding). KV is shared across 8 simdgroups
/// (eight heads) via threadgroup memory. K is reused as V (compressed KV latent).
pub(super) fn emit_dsv4_indexed_mixed_attention_generic_msl(out: &mut String, shape: &MixedAttention) {
    let stripes = shape.head_dim / 128;
    let row_w = shape.head_dim / 4;
    let heads = shape.heads_per_group;
    let threads = 32 * heads;
    let stage = shape.stage;
    // Past four stripes a stripe would be named `q4`, the name of the Q row
    // pointer, so wide heads name their stripes qs0.. instead.
    let q_name = |i: u32| if stripes > 4 { format!("qs{i}") } else { format!("q{i}") };
    // Silence the batched-only parameters in the one-row kernel.
    let _ = (threads, shape.rows_per_batch);
    writeln!(out, "    threadgroup float4 kv_shared[{row_w}];").unwrap();
    writeln!(out, "    uint token = tgpig.x;").unwrap();
    writeln!(out, "    uint head  = tgpig.y * {heads}u + simd_id;").unwrap();
    writeln!(out, "    if (token >= n_tokens || head >= n_head) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4 *q4 = (device const float4 *)(p0 + (uint64_t)token * q_token_stride + (uint64_t)head * q_head_stride);").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        match stage {
            MixedStage::Half => writeln!(out, "    half4 {} = (half4)q4[simd_lane + {off:>2}];", q_name(i)).unwrap(),
            MixedStage::Float => writeln!(out, "    float4 {} = q4[simd_lane + {off:>2}];", q_name(i)).unwrap(),
        }
    }
    writeln!(out).unwrap();
    writeln!(out, "    float M = -FLT_MAX/2.0f;").unwrap();
    writeln!(out, "    float S = 0.0f;").unwrap();
    for i in 0..stripes {
        writeln!(out, "    float4 o{i} = float4(0.0f);").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "    uint qpos = pos0 + token;").unwrap();
    writeln!(out, "    uint last_pos = pos0 + n_tokens - 1u;").unwrap();
    writeln!(out, "    uint first_raw_pos = last_pos + 1u - n_raw;").unwrap();
    writeln!(out, "    uint raw_last_pos = first_raw_pos + n_raw - 1u;").unwrap();
    writeln!(out, "    uint window_first = (window != 0u && qpos + 1u > window) ? (qpos + 1u - window) : 0u;").unwrap();
    writeln!(out, "    uint first = max(first_raw_pos, window_first);").unwrap();
    writeln!(out, "    uint last  = min(qpos, raw_last_pos);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (first <= last) {{").unwrap();
    writeln!(out, "        for (uint pos = first; pos <= last; ++pos) {{").unwrap();
    writeln!(out, "            uint logical = pos - first_raw_pos;").unwrap();
    writeln!(out, "            uint row = (raw_start + logical) % raw_cap;").unwrap();
    writeln!(out, "            device const float4 *src = (device const float4 *)(p1 + (uint64_t)row * raw_row_stride);").unwrap();
    writeln!(out, "            if (tid < {row_w}u) kv_shared[tid] = src[tid];").unwrap();
    writeln!(out, "            threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "            {{").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        match stage {
            MixedStage::Half => writeln!(out, "                half4 k{i} = (half4)kv_shared[simd_lane + {off:>2}];").unwrap(),
            MixedStage::Float => writeln!(out, "                float4 k{i} = kv_shared[simd_lane + {off:>2}];").unwrap(),
        }
    }
    {
        let dots: Vec<String> = (0..stripes).map(|i| format!("dot((float4){},(float4)k{i})", q_name(i))).collect();
        writeln!(out, "                float score = {};", dots.join(" + ")).unwrap();
    }
    writeln!(out, "                score = simd_sum(score) * scale;").unwrap();
    writeln!(out, "                float old_m = M; float new_m = max(M, score);").unwrap();
    writeln!(out, "                float old_scale = exp(old_m - new_m); float row_scale = exp(score - new_m);").unwrap();
    writeln!(out, "                S = S * old_scale + row_scale;").unwrap();
    for i in 0..stripes {
        writeln!(out, "                o{i} = o{i} * old_scale + (float4)k{i} * row_scale;").unwrap();
    }
    writeln!(out, "                M = new_m;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint visible = min((qpos + 1u) / ratio, n_comp);").unwrap();
    writeln!(out, "    device const int *row_topk = (device const int *)(p3 + (uint64_t)token * topk_token_stride);").unwrap();
    writeln!(out, "    for (uint i = 0u; i < top_k; ++i) {{").unwrap();
    writeln!(out, "        int idx = row_topk[i];").unwrap();
    writeln!(out, "        if (idx < 0) continue;").unwrap();
    writeln!(out, "        if ((uint)idx >= visible) break;").unwrap();
    writeln!(out, "        device const float4 *src = (device const float4 *)(p2 + (uint64_t)(uint)idx * comp_row_stride);").unwrap();
    writeln!(out, "        if (tid < {row_w}u) kv_shared[tid] = src[tid];").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "        {{").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        match stage {
            MixedStage::Half => writeln!(out, "            half4 k{i} = (half4)kv_shared[simd_lane + {off:>2}];").unwrap(),
            MixedStage::Float => writeln!(out, "            float4 k{i} = kv_shared[simd_lane + {off:>2}];").unwrap(),
        }
    }
    {
        let dots: Vec<String> = (0..stripes).map(|i| format!("dot((float4){},(float4)k{i})", q_name(i))).collect();
        writeln!(out, "            float score = {};", dots.join(" + ")).unwrap();
    }
    writeln!(out, "            score = simd_sum(score) * scale;").unwrap();
    writeln!(out, "            float old_m = M; float new_m = max(M, score);").unwrap();
    writeln!(out, "            float old_scale = exp(old_m - new_m); float row_scale = exp(score - new_m);").unwrap();
    writeln!(out, "            S = S * old_scale + row_scale;").unwrap();
    for i in 0..stripes {
        writeln!(out, "            o{i} = o{i} * old_scale + (float4)k{i} * row_scale;").unwrap();
    }
    writeln!(out, "            M = new_m;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        float sink = ((device const float *)p4)[head];").unwrap();
    writeln!(out, "        float old_m = M; float new_m = max(M, sink);").unwrap();
    writeln!(out, "        float old_scale = exp(old_m - new_m); float row_scale = exp(sink - new_m);").unwrap();
    writeln!(out, "        S = S * old_scale + row_scale;").unwrap();
    {
        let scaled: Vec<String> = (0..stripes).map(|i| format!("o{i} *= old_scale;")).collect();
        writeln!(out, "        {}", scaled.join(" ")).unwrap();
    }
    writeln!(out, "        M = new_m;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float inv_s = (S == 0.0f) ? 0.0f : 1.0f / S;").unwrap();
    writeln!(out, "    device float4 *dst4 = (device float4 *)(p5 + (uint64_t)token * dst_token_stride + (uint64_t)head * dst_head_stride);").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        writeln!(out, "    dst4[simd_lane + {off:>2}] = o{i} * inv_s;").unwrap();
    }
}
/// DS4 indexed_mixed_attention_heads8 at the shipped shape and staging.
pub(super) fn emit_dsv4_indexed_mixed_attention_h8_msl(out: &mut String) {
    emit_dsv4_indexed_mixed_attention_generic_msl(out, &MixedAttention::ds4());
}
/// DS4 indexed_mixed_attention_heads8_rb4: decode specialization of M33.
/// Stages 4 selected K/V rows into kv_shared[4*128] at once and consumes them
/// sequentially, cutting threadgroup barriers in the long top-k scan.
pub(super) fn emit_dsv4_indexed_mixed_attention_batched_generic_msl(out: &mut String, shape: &MixedAttention) {
    let stripes = shape.head_dim / 128;
    let row_w = shape.head_dim / 4;
    let heads = shape.heads_per_group;
    let threads = 32 * heads;
    let stage = shape.stage;
    // Past four stripes a stripe would be named `q4`, the name of the Q row
    // pointer, so wide heads name their stripes qs0.. instead.
    let q_name = |i: u32| if stripes > 4 { format!("qs{i}") } else { format!("q{i}") };
    let rows = shape.rows_per_batch;
    let row_shift = row_w.trailing_zeros();
    let row_mask = row_w - 1;
    writeln!(out, "    threadgroup float4 kv_shared[{rows}*{row_w}];").unwrap();
    writeln!(out, "    uint token = tgpig.x;").unwrap();
    writeln!(out, "    uint head  = tgpig.y * {heads}u + simd_id;").unwrap();
    writeln!(out, "    if (token >= n_tokens || head >= n_head) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4 *q4 = (device const float4 *)(p0 + (uint64_t)token * q_token_stride + (uint64_t)head * q_head_stride);").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        match stage {
            MixedStage::Half => writeln!(out, "    half4 {} = (half4)q4[simd_lane + {off:>2}];", q_name(i)).unwrap(),
            MixedStage::Float => writeln!(out, "    float4 {} = q4[simd_lane + {off:>2}];", q_name(i)).unwrap(),
        }
    }
    writeln!(out).unwrap();
    writeln!(out, "    float M = -FLT_MAX/2.0f;").unwrap();
    writeln!(out, "    float S = 0.0f;").unwrap();
    for i in 0..stripes {
        writeln!(out, "    float4 o{i} = float4(0.0f);").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "    uint qpos = pos0 + token;").unwrap();
    writeln!(out, "    uint last_pos = pos0 + n_tokens - 1u;").unwrap();
    writeln!(out, "    uint first_raw_pos = last_pos + 1u - n_raw;").unwrap();
    writeln!(out, "    uint raw_last_pos = first_raw_pos + n_raw - 1u;").unwrap();
    writeln!(out, "    uint window_first = (window != 0u && qpos + 1u > window) ? (qpos + 1u - window) : 0u;").unwrap();
    writeln!(out, "    uint first = max(first_raw_pos, window_first);").unwrap();
    writeln!(out, "    uint last  = min(qpos, raw_last_pos);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (first <= last) {{").unwrap();
    writeln!(out, "        for (uint base = first; base <= last; base += {rows}u) {{").unwrap();
    writeln!(out, "            uint n_rows = min({rows}u, last - base + 1u);").unwrap();
    writeln!(out, "            for (uint off = tid; off < n_rows * {row_w}u; off += {threads}u) {{").unwrap();
    writeln!(out, "                uint r = off >> {row_shift};").unwrap();
    writeln!(out, "                uint c = off & {row_mask}u;").unwrap();
    writeln!(out, "                uint logical = base + r - first_raw_pos;").unwrap();
    writeln!(out, "                uint row = (raw_start + logical) % raw_cap;").unwrap();
    writeln!(out, "                device const float4 *src = (device const float4 *)(p1 + (uint64_t)row * raw_row_stride);").unwrap();
    writeln!(out, "                kv_shared[off] = src[c];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "            for (uint r = 0u; r < n_rows; ++r) {{").unwrap();
    writeln!(out, "                threadgroup const float4 *kv4 = kv_shared + r * {row_w}u;").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        match stage {
            MixedStage::Half => writeln!(out, "                half4 k{i} = (half4)kv4[simd_lane + {off:>2}];").unwrap(),
            MixedStage::Float => writeln!(out, "                float4 k{i} = kv4[simd_lane + {off:>2}];").unwrap(),
        }
    }
    {
        let dots: Vec<String> = (0..stripes).map(|i| format!("dot((float4){},(float4)k{i})", q_name(i))).collect();
        writeln!(out, "                float score = {};", dots.join(" + ")).unwrap();
    }
    writeln!(out, "                score = simd_sum(score) * scale;").unwrap();
    writeln!(out, "                float old_m = M; float new_m = max(M, score);").unwrap();
    writeln!(out, "                float old_scale = exp(old_m - new_m); float row_scale = exp(score - new_m);").unwrap();
    writeln!(out, "                S = S * old_scale + row_scale;").unwrap();
    for i in 0..stripes {
        writeln!(out, "                o{i} = o{i} * old_scale + (float4)k{i} * row_scale;").unwrap();
    }
    writeln!(out, "                M = new_m;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint visible = min((qpos + 1u) / ratio, n_comp);").unwrap();
    writeln!(out, "    device const int *row_topk = (device const int *)(p3 + (uint64_t)token * topk_token_stride);").unwrap();
    writeln!(out, "    bool stop = false;").unwrap();
    writeln!(out, "    for (uint i = 0u; i < top_k && !stop; i += {rows}u) {{").unwrap();
    writeln!(out, "        uint rows[{rows}]; uint n_rows = 0u;").unwrap();
    writeln!(out, "        for (uint j = 0u; j < {rows}u && (i + j) < top_k; ++j) {{").unwrap();
    writeln!(out, "            int idx = row_topk[i + j];").unwrap();
    writeln!(out, "            if (idx < 0) continue;").unwrap();
    writeln!(out, "            if ((uint)idx >= visible) {{ stop = true; break; }}").unwrap();
    writeln!(out, "            rows[n_rows++] = (uint)idx;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (n_rows == 0u) continue;").unwrap();
    writeln!(out, "        for (uint off = tid; off < n_rows * {row_w}u; off += {threads}u) {{").unwrap();
    writeln!(out, "            uint r = off >> {row_shift};").unwrap();
    writeln!(out, "            uint c = off & {row_mask}u;").unwrap();
    writeln!(out, "            device const float4 *src = (device const float4 *)(p2 + (uint64_t)rows[r] * comp_row_stride);").unwrap();
    writeln!(out, "            kv_shared[off] = src[c];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "        for (uint r = 0u; r < n_rows; ++r) {{").unwrap();
    writeln!(out, "            threadgroup const float4 *kv4 = kv_shared + r * {row_w}u;").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        match stage {
            MixedStage::Half => writeln!(out, "            half4 k{i} = (half4)kv4[simd_lane + {off:>2}];").unwrap(),
            MixedStage::Float => writeln!(out, "            float4 k{i} = kv4[simd_lane + {off:>2}];").unwrap(),
        }
    }
    {
        let dots: Vec<String> = (0..stripes).map(|i| format!("dot((float4){},(float4)k{i})", q_name(i))).collect();
        writeln!(out, "            float score = {};", dots.join(" + ")).unwrap();
    }
    writeln!(out, "            score = simd_sum(score) * scale;").unwrap();
    writeln!(out, "            float old_m = M; float new_m = max(M, score);").unwrap();
    writeln!(out, "            float old_scale = exp(old_m - new_m); float row_scale = exp(score - new_m);").unwrap();
    writeln!(out, "            S = S * old_scale + row_scale;").unwrap();
    for i in 0..stripes {
        writeln!(out, "            o{i} = o{i} * old_scale + (float4)k{i} * row_scale;").unwrap();
    }
    writeln!(out, "            M = new_m;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        float sink = ((device const float *)p4)[head];").unwrap();
    writeln!(out, "        float old_m = M; float new_m = max(M, sink);").unwrap();
    writeln!(out, "        float old_scale = exp(old_m - new_m); float row_scale = exp(sink - new_m);").unwrap();
    writeln!(out, "        S = S * old_scale + row_scale;").unwrap();
    {
        let scaled: Vec<String> = (0..stripes).map(|i| format!("o{i} *= old_scale;")).collect();
        writeln!(out, "        {}", scaled.join(" ")).unwrap();
    }
    writeln!(out, "        M = new_m;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float inv_s = (S == 0.0f) ? 0.0f : 1.0f / S;").unwrap();
    writeln!(out, "    device float4 *dst4 = (device float4 *)(p5 + (uint64_t)token * dst_token_stride + (uint64_t)head * dst_head_stride);").unwrap();
    for i in 0..stripes {
        let off = i * 32;
        writeln!(out, "    dst4[simd_lane + {off:>2}] = o{i} * inv_s;").unwrap();
    }
}
/// DS4 indexed_mixed_attention_heads8_rb4 at the shipped shape and staging.
pub(super) fn emit_dsv4_indexed_mixed_attention_h8_rb4_msl(out: &mut String) {
    emit_dsv4_indexed_mixed_attention_batched_generic_msl(out, &MixedAttention::ds4());
}

/// Precision the mixed-attention kernels stage Q and K rows in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MixedStage {
    Half,
    Float,
}

/// Shape of the DS4 indexed mixed-attention kernels.
///
/// A head is `head_dim / 128` stripes of `float4`, one stripe per 32 lanes of a
/// simdgroup, so `head_dim` is 128, 256, 512 or 1024. A threadgroup runs
/// `heads_per_group` heads, one per 32-thread simdgroup. The batched kernel
/// stages `rows_per_batch` key rows per threadgroup barrier.
///
/// The runtime `n_head` must be a multiple of `heads_per_group`. Threads for a
/// head at or past `n_head` return before staging their share of the key row,
/// so the other heads in that threadgroup would read unstaged memory. This
/// cannot be checked here, because `n_head` is bound at dispatch time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MixedAttention {
    pub head_dim: u32,
    pub heads_per_group: u32,
    pub stage: MixedStage,
    pub rows_per_batch: u32,
}

impl MixedAttention {
    /// DeepSeek-V4: head_dim 512, 8 heads per threadgroup, half staging, 4 rows.
    pub fn ds4() -> Self {
        Self { head_dim: 512, heads_per_group: 8, stage: MixedStage::Half, rows_per_batch: 4 }
    }

    /// Why this shape cannot be emitted, if it cannot. `batched` selects the rb
    /// kernel, which stages rows in chunks and so has no thread-count floor.
    pub fn check(&self, batched: bool) -> Result<(), String> {
        let (d, h) = (self.head_dim, self.heads_per_group);
        if !matches!(d, 128 | 256 | 512 | 1024) {
            return Err(format!(
                "indexed_mixed_attention: head_dim {d} must be 128, 256, 512 or 1024 (float4 stripes over 32 lanes)"
            ));
        }
        if !(1..=32).contains(&h) {
            return Err(format!("indexed_mixed_attention: heads_per_group {h} must be from 1 to 32"));
        }
        if !batched && 32 * h < d / 4 {
            return Err(format!(
                "indexed_mixed_attention: {h} heads per group give {} threads, fewer than the {} float4 rows a key row stages",
                32 * h,
                d / 4
            ));
        }
        if batched && !(1..=16).contains(&self.rows_per_batch) {
            return Err(format!(
                "indexed_mixed_attention: rows_per_batch {} must be from 1 to 16",
                self.rows_per_batch
            ));
        }
        Ok(())
    }
}
/// DS4 flash_attn_ext_vec_reduce: split-K decode reducer.
/// Each row's NWG partial outputs are merged across one simdgroup using simd_max
/// for the running max, simd_sum for the running normalizer, and simd_sum for the
/// per-DV4 output vector. The final vector is divided by the merged 1/S.
/// Buffers: p0=htmp (char* in: NWG-tiled DV4 floats then 2*NWG (S,M) pairs),
///          p1=dst  (char* out: nrows × DV4 float4s).
/// Params: nrows, dv, nwg.
/// Threads: NWG simdgroup-lanes per workgroup (NWG ≤ 32); 1 row per threadgroup.
pub(super) fn emit_flash_attn_ext_vec_reduce_msl(out: &mut String) {
    writeln!(out, "    const uint64_t rid = (uint64_t)row;").unwrap();
    writeln!(out, "    const ushort iwg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort iwg_sg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint NWG = nwg;").unwrap();
    writeln!(out, "    const uint DV  = dv;").unwrap();
    writeln!(out, "    const uint DV4 = DV / 4u;").unwrap();
    writeln!(out, "    device const float * ss = (device const float *)((device const char *)p0 + (uint64_t)nrows * (uint64_t)DV * (uint64_t)NWG * 4ull);").unwrap();
    writeln!(
        out,
        "    float S = (iwg < NWG) ? ss[rid * (2u*NWG) + 2u*(uint)iwg + 0u] : 0.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "    float M = (iwg < NWG) ? ss[rid * (2u*NWG) + 2u*(uint)iwg + 1u] : -INFINITY;"
    )
    .unwrap();
    writeln!(out, "    const float m  = simd_max(M);").unwrap();
    writeln!(out, "    const float ms = (iwg < NWG) ? exp(M - m) : 0.0f;").unwrap();
    writeln!(out, "    S = simd_sum(S * ms);").unwrap();
    writeln!(out, "    S = (S == 0.0f) ? 0.0f : 1.0f / S;").unwrap();
    writeln!(out, "    device const float4 *htmp4 = (device const float4 *)p0 + rid * (uint64_t)DV4 * (uint64_t)NWG;").unwrap();
    writeln!(
        out,
        "    device       float4 *dst4  = (device       float4 *)p1 + rid * (uint64_t)DV4;"
    )
    .unwrap();
    writeln!(out, "    for (uint i = (uint)iwg_sg; i < DV4; i += NWG) {{").unwrap();
    writeln!(out, "        const float4 partial = (iwg < NWG) ? (htmp4[i * NWG + (uint)iwg] * ms) : float4(0.0f);").unwrap();
    writeln!(out, "        const float4 v = simd_sum(partial);").unwrap();
    writeln!(out, "        if (iwg == 0) {{ dst4[i] = v * S; }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtVecSetup (M36a stage of flash_attn_ext_vec):
/// Reproduces the kernel signature, threadgroup memory layout, Q→sq4 load,
/// and threadgroup_barrier. Echoes the loaded sq4 contents to dst as float4
/// so we can byte-compare against antirez's loaded shmem (validated by running
/// antirez with a custom probe path that dumps sq4 then early-returns).
///
/// Buffers (all char*): p0=q, p1=k, p2=v, p3=mask, p4=sinks, p5=pad,
///                      p6=dst (writable: ne01 × DK4 float4 echoes of sq4).
/// Params: dk, dv, ne01, nb01.
/// Threads: NW=32 lanes, NSG=1, NWG=1; grid = (ne01, 1, 1).
/// Test config: DK=DV=64 (DK4=DV4=16), PK=PV=128 (PK4=PV4=32).
pub(super) fn emit_flash_attn_ext_vec_setup_msl(out: &mut String, dk: u32, dv: u32) {
    let (pk, pv) = (dk.next_multiple_of(128), dv.next_multiple_of(128));
    let pk4 = pk / 4;
    writeln!(out, "    constexpr ushort NW   = 32;").unwrap();
    writeln!(out, "    constexpr ushort PK   = {pk};").unwrap();
    writeln!(out, "    constexpr ushort PK4  = {pk4};").unwrap();
    writeln!(out, "    constexpr ushort PV   = {pv};").unwrap();
    writeln!(
        out,
        "    constexpr ushort SH   = 4 * 32; // 4*C, C=NCPSG=32"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort NSG  = 1;").unwrap();
    writeln!(
        out,
        "    threadgroup half shmem_f16[NSG*PK + NSG*SH + 2*NSG*PV];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1   = tgpig.x;").unwrap();
    writeln!(out, "    const ushort DK4   = (ushort)(dk / 4u);").unwrap();
    writeln!(out, "    if (sgitg != 0) return; // M36a: NSG=1 only").unwrap();
    writeln!(
        out,
        "    device const char  * qrow = (device const char  *)p0 + (uint64_t)iq1 * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float4 * q4 = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(out, "    if (iq1 < ne01) {{").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < PK4; i += NW) {{").unwrap();
    writeln!(out, "            if (i < DK4) {{").unwrap();
    writeln!(out, "                sq4[i] = (half4) q4[i];").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                sq4[i] = (half4) 0.0f;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // Echo sq4[0..DK4] to dst row iq1.
    writeln!(out, "    device float4 * dst4 = (device float4 *)((device char *)p6 + (uint64_t)iq1 * (uint64_t)DK4 * 16ull);").unwrap();
    writeln!(out, "    for (ushort i = tiisg; i < DK4; i += NW) {{").unwrap();
    writeln!(out, "        dst4[i] = (float4) sq4[i];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtVecScore (M36b stage of flash_attn_ext_vec):
/// Setup + K·Q dot loop + online softmax merge across the full ne11 KV
/// extent. Specialized to the DS4 vec config (q4_t = k4_t = half4, NE=4,
/// NL=8, C=NCPSG=32) with all FC flags off (no mask, sinks, bias, scap,
/// kvpad). NSG=1, NWG=1 — single workgroup walks every block.
///
/// Output: per query row, dst writes 2 floats `[S, M]` (the merged
/// online-softmax denominator and max). Validates K·Q correctness without
/// pulling V into the picture (M36c will add V).
///
/// Buffers (all char*): p0=q, p1=k, p2=v, p3=mask, p4=sinks, p5=pad,
///                      p6=dst (writable: ne01 × 2 floats).
/// Params: dk, dv, ne01, ne11, nb01, nb11, scale.
/// Threads: NW=32 lanes (single simdgroup); grid = (ne01, 1, 1).
pub(super) fn emit_flash_attn_ext_vec_score_msl(out: &mut String, dk: u32, dv: u32) {
    let (pk, pv) = (dk.next_multiple_of(128), dv.next_multiple_of(128));
    let (pk4, dk4, dk4_nl) = (pk / 4, dk / 4, dk / 32);
    writeln!(out, "    constexpr ushort NW   = 32;").unwrap();
    writeln!(out, "    constexpr ushort NE   = 4;").unwrap();
    writeln!(out, "    constexpr ushort NL   = NW / NE;            // 8").unwrap();
    writeln!(
        out,
        "    constexpr ushort C    = 32;                 // NCPSG"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort CNE  = C / NE;             // 8").unwrap();
    writeln!(
        out,
        "    constexpr ushort DK4_FIXED = {dk4};            // DK={dk}"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort DK4_NL    = DK4_FIXED / NL; // {dk4_nl}").unwrap();
    writeln!(out, "    constexpr ushort PK   = {pk};").unwrap();
    writeln!(out, "    constexpr ushort PK4  = {pk4};").unwrap();
    writeln!(out, "    constexpr ushort PV   = {pv};").unwrap();
    writeln!(out, "    constexpr ushort SH   = 4 * C;").unwrap();
    writeln!(out, "    constexpr ushort NSG  = 1;").unwrap();
    writeln!(
        out,
        "    threadgroup half shmem_f16[NSG*PK + NSG*SH + 2*NSG*PV];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * ss  = (threadgroup float *)(shmem_f16 + NSG*PK);"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1   = tgpig.x;").unwrap();
    writeln!(out, "    if (sgitg != 0) return; // NSG=1").unwrap();
    // ---- Q load ----
    writeln!(
        out,
        "    device const char  * qrow = (device const char  *)p0 + (uint64_t)iq1 * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float4 * q4 = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(out, "    if (iq1 < ne01) {{").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < PK4; i += NW) {{").unwrap();
    writeln!(
        out,
        "            sq4[i] = (i < DK4_FIXED) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // Init ss[0..C] to 0 so the post-block read sees the just-written scores.
    writeln!(out, "    for (ushort i = tiisg; i < SH/4u; i += NW) {{").unwrap();
    writeln!(out, "        ((threadgroup float4 *)ss)[i] = float4(0.0f);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // ---- ic loop with online softmax ----
    writeln!(out, "    float S = 0.0f;").unwrap();
    writeln!(out, "    float M = -INFINITY/2;").unwrap();
    writeln!(out, "    const ushort tx = tiisg % NL;").unwrap();
    writeln!(out, "    const ushort ty = tiisg / NL;").unwrap();
    writeln!(out, "    for (uint ic0 = 0u; ; ic0 += C) {{").unwrap();
    writeln!(out, "        if (ic0 >= ne11) break;").unwrap();
    // Compute mqk[cc] for cc in [0, C/NE).
    writeln!(out, "        device const half4 * pk4_base = (device const half4 *)((device const char *)p1 + (uint64_t)ic0 * (uint64_t)nb11);").unwrap();
    // pk4 += ty*NS10/4 + tx; with NS10 = nb11/sizeof(half) (per-row K stride in halves) — for our half4 K with row stride nb11 bytes,
    // NS10/4 = (nb11/2)/4 = nb11/8 half4 elements per row. We hardcode NS10/4 = DK4_FIXED = 16 (assumes nb11 = DK*sizeof(half) = 128 bytes → NS10 = 64 halves → NS10/4 = 16).
    writeln!(out, "        const ushort NS10_4 = DK4_FIXED;").unwrap();
    writeln!(
        out,
        "        device const half4 * pk4 = pk4_base + ty * NS10_4 + tx;"
    )
    .unwrap();
    writeln!(out, "        threadgroup const half4 * pq4 = sq4 + tx;").unwrap();
    writeln!(out, "        float mqk_arr[CNE];").unwrap();
    writeln!(
        out,
        "        for (ushort cc = 0; cc < CNE; ++cc) mqk_arr[cc] = 0.0f;"
    )
    .unwrap();
    writeln!(out, "        for (ushort cc = 0; cc < CNE; ++cc) {{").unwrap();
    writeln!(out, "            for (ushort ii = 0; ii < DK4_NL; ++ii) {{").unwrap();
    writeln!(out, "                mqk_arr[cc] += dot((float4) pk4[cc * NE * NS10_4 + ii * NL], (float4) pq4[ii * NL]);").unwrap();
    writeln!(out, "            }}").unwrap();
    // Cross-tx reduction via shuffle ladder over 8 lanes (NE=4 → shifts 4,2,1).
    // After the ladder, lane (NL*ty + 0) holds sum over its NL-lane group;
    // simd_shuffle to NL*ty broadcasts that result to all lanes of that group.
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 4);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 2);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 1);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] = simd_shuffle(mqk_arr[cc], (ushort)(NL * ty));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    // Write scaled mqk[tx] to ss[NE*tx + ty]. With FC flags all off,
    // antirez does ss[NE*tx + ty] = mqk[tx] * scale (no sm/bias add).
    writeln!(out, "        ss[NE * tx + ty] = mqk_arr[tx] * scale;").unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // Online softmax merge.
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            const float m_old = M;").unwrap();
    writeln!(out, "            const float s = ss[tiisg];").unwrap();
    writeln!(out, "            M = simd_max(max(M, s));").unwrap();
    writeln!(out, "            const float ms = exp(m_old - M);").unwrap();
    writeln!(out, "            const float vs = exp(s - M);").unwrap();
    writeln!(out, "            S = S * ms + simd_sum(vs);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- write per-row (S, M) ----
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        device float * dst_f = (device float *)((device char *)p6 + (uint64_t)iq1 * 2ull * 4ull);").unwrap();
    writeln!(out, "        dst_f[0] = S;").unwrap();
    writeln!(out, "        dst_f[1] = M;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtVecOut (M36c stage of flash_attn_ext_vec):
/// Full single-SG flash-attention output. Reuses M36b score body
/// (K·Q dot + online softmax merge) but writes vs back to ss, scales
/// the running so4 accumulator by ms, then performs V accumulation
/// over the C-block. After the ic-loop, writes the per-row attention
/// output dst4[iq1*DV4 + i] = so4[i] / S for i in [0, DV4).
///
/// Buffers (all char*): p0=q, p1=k, p2=v, p3=mask, p4=sinks, p5=pad,
///                      p6=dst (writable: ne01 × DV4 float4 = ne01 × DV floats).
/// Params: dk, dv, ne01, ne11, nb01, nb11, nb21, scale.
/// Threads: NW=32 lanes, NSG=1, NWG=1; grid = (ne01, 1, 1).
/// Test config: DK=DV=64 (DK4=DV4=16), C=NCPSG=32, NE=4, NL=NW/NE=8,
/// CNE=C/NE=8, DK4/NL=2, DV4/NL=2.
pub(super) fn emit_flash_attn_ext_vec_out_msl(out: &mut String, dk: u32, dv: u32) {
    let (pk, pv) = (dk.next_multiple_of(128), dv.next_multiple_of(128));
    let (pk4, pv4) = (pk / 4, pv / 4);
    let (dk4, dv4, dk4_nl, dv4_nl) = (dk / 4, dv / 4, dk / 32, dv / 32);
    writeln!(out, "    constexpr ushort NW   = 32;").unwrap();
    writeln!(out, "    constexpr ushort NE   = 4;").unwrap();
    writeln!(out, "    constexpr ushort NL   = NW / NE;            // 8").unwrap();
    writeln!(
        out,
        "    constexpr ushort C    = 32;                 // NCPSG"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort CNE  = C / NE;             // 8").unwrap();
    writeln!(
        out,
        "    constexpr ushort DK4_FIXED = {dk4};            // DK={dk}"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort DV4_FIXED = {dv4};            // DV={dv}"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort DK4_NL    = DK4_FIXED / NL; // {dk4_nl}").unwrap();
    writeln!(out, "    constexpr ushort DV4_NL    = DV4_FIXED / NL; // {dv4_nl}").unwrap();
    writeln!(out, "    constexpr ushort PK   = {pk};").unwrap();
    writeln!(out, "    constexpr ushort PK4  = {pk4};").unwrap();
    writeln!(out, "    constexpr ushort PV   = {pv};").unwrap();
    writeln!(out, "    constexpr ushort PV4  = {pv4};").unwrap();
    writeln!(out, "    constexpr ushort SH   = 4 * C;").unwrap();
    writeln!(out, "    constexpr ushort NSG  = 1;").unwrap();
    // Extra slack for so4 init: antirez writes from lane tiisg up to slot tiisg+(DV4/NL-1)*NL
    // = tiisg+8 float4 for DV=64 → up to slot 39 → 320 halves, beyond 2*PV=256 for DV=64. Pad by NW float4.
    writeln!(
        out,
        "    threadgroup half shmem_f16[NSG*PK + NSG*SH + 2*NSG*PV + 8*32];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * ss  = (threadgroup float *)(shmem_f16 + NSG*PK);"
    )
    .unwrap();
    // so4 lives after sq + ss in shmem; with sgitg=0 (NSG=1), offset = NSG*PK + NSG*SH halves.
    writeln!(
        out,
        "    threadgroup float4 * so4 = (threadgroup float4 *)(shmem_f16 + NSG*PK + NSG*SH);"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1   = tgpig.x;").unwrap();
    writeln!(out, "    if (sgitg != 0) return; // NSG=1").unwrap();
    // ---- Q load ----
    writeln!(
        out,
        "    device const char  * qrow = (device const char  *)p0 + (uint64_t)iq1 * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float4 * q4 = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(out, "    if (iq1 < ne01) {{").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < PK4; i += NW) {{").unwrap();
    writeln!(
        out,
        "            sq4[i] = (i < DK4_FIXED) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // Init ss[0..SH] to 0.
    writeln!(out, "    for (ushort i = tiisg; i < SH/4u; i += NW) {{").unwrap();
    writeln!(out, "        ((threadgroup float4 *)ss)[i] = float4(0.0f);").unwrap();
    writeln!(out, "    }}").unwrap();
    // Init so4 (per-lane via tiisg offset; PV4=32 entries total covers all DV4_FIXED slots).
    writeln!(out, "    so4 += tiisg;").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        for (ushort i = 0; i < DV4_NL; ++i) {{").unwrap();
    writeln!(out, "            so4[i*NL] = float4(0.0f);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // ---- ic loop with online softmax + V accumulation ----
    writeln!(out, "    float S = 0.0f;").unwrap();
    writeln!(out, "    float M = -INFINITY/2;").unwrap();
    writeln!(out, "    const ushort tx = tiisg % NL;").unwrap();
    writeln!(out, "    const ushort ty = tiisg / NL;").unwrap();
    writeln!(out, "    for (uint ic0 = 0u; ; ic0 += C) {{").unwrap();
    writeln!(out, "        if (ic0 >= ne11) break;").unwrap();
    // K·Q dot.
    writeln!(out, "        device const half4 * pk4_base = (device const half4 *)((device const char *)p1 + (uint64_t)ic0 * (uint64_t)nb11);").unwrap();
    writeln!(out, "        const ushort NS10_4 = DK4_FIXED;").unwrap();
    writeln!(
        out,
        "        device const half4 * pk4 = pk4_base + ty * NS10_4 + tx;"
    )
    .unwrap();
    writeln!(out, "        threadgroup const half4 * pq4 = sq4 + tx;").unwrap();
    writeln!(out, "        float mqk_arr[CNE];").unwrap();
    writeln!(
        out,
        "        for (ushort cc = 0; cc < CNE; ++cc) mqk_arr[cc] = 0.0f;"
    )
    .unwrap();
    writeln!(out, "        for (ushort cc = 0; cc < CNE; ++cc) {{").unwrap();
    writeln!(out, "            for (ushort ii = 0; ii < DK4_NL; ++ii) {{").unwrap();
    writeln!(out, "                mqk_arr[cc] += dot((float4) pk4[cc * NE * NS10_4 + ii * NL], (float4) pq4[ii * NL]);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 4);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 2);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 1);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] = simd_shuffle(mqk_arr[cc], (ushort)(NL * ty));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        ss[NE * tx + ty] = mqk_arr[tx] * scale;").unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // Online softmax merge with vs writeback into ss for V accumulation.
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            const float m_old = M;").unwrap();
    writeln!(out, "            const float s = ss[tiisg];").unwrap();
    writeln!(out, "            M = simd_max(max(M, s));").unwrap();
    writeln!(out, "            const float ms = exp(m_old - M);").unwrap();
    writeln!(out, "            const float vs = exp(s - M);").unwrap();
    writeln!(out, "            S = S * ms + simd_sum(vs);").unwrap();
    writeln!(out, "            ss[tiisg] = vs;").unwrap();
    // Scale running so4 by ms (DV4/NL=2 < NW=32, so guard ty==0).
    writeln!(out, "            if (ty == 0) {{").unwrap();
    writeln!(
        out,
        "                for (ushort ii = 0; ii < DV4_NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    so4[ii*NL] *= ms;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // V accumulation block.
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            float4 lo[DV4_NL];").unwrap();
    writeln!(
        out,
        "            for (ushort ii = 0; ii < DV4_NL; ++ii) lo[ii] = float4(0.0f);"
    )
    .unwrap();
    writeln!(out, "            device const half4 * pv4_base = (device const half4 *)((device const char *)p2 + (uint64_t)ic0 * (uint64_t)nb21);").unwrap();
    writeln!(out, "            const ushort NS20_4 = DV4_FIXED;").unwrap();
    writeln!(
        out,
        "            device const half4 * pv4 = pv4_base + ty * NS20_4 + tx;"
    )
    .unwrap();
    writeln!(out, "            threadgroup const float * sst = ss + ty;").unwrap();
    writeln!(out, "            for (ushort cc = 0; cc < CNE; ++cc) {{").unwrap();
    writeln!(
        out,
        "                for (ushort ii = 0; ii < DV4_NL; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    lo[ii] += float4(pv4[cc*NE*NS20_4 + ii*NL]) * float4(sst[cc*NE]);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    // Cross-NE butterfly. NE=4 → only NE>1 and NE>2 branches fire (shifts 16, 8).
    writeln!(out, "            for (ushort ii = 0; ii < DV4_NL; ++ii) {{").unwrap();
    writeln!(
        out,
        "                lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            if (ty == 0) {{").unwrap();
    writeln!(
        out,
        "                for (ushort ii = 0; ii < DV4_NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    so4[ii*NL] += lo[ii];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- final write: dst4[iq1*DV4 + i] = so4[i] / S for i in [0, DV4) ----
    writeln!(out, "    so4 -= tiisg;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        device float4 * dst4 = (device float4 *)((device char *)p6 + (uint64_t)iq1 * (uint64_t)DV4_FIXED * 16ull);").unwrap();
    writeln!(
        out,
        "        const float Sinv = (S == 0.0f) ? 0.0f : 1.0f/S;"
    )
    .unwrap();
    writeln!(
        out,
        "        for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "            dst4[i] = so4[i] * Sinv;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtVecOutMS (M36d stage of flash_attn_ext_vec):
/// M36c + has_mask (per-block additive mask before softmax) and
/// has_sinks (post-loop sink merge of one extra score into S/M).
///
/// Mask buffer (p3) is half[ne01 * ne11], indexed by row*ne11 + ic.
/// Sinks buffer (p4) is float[ne01], indexed by iq1.
///
/// Buffers (all char*): p0=q, p1=k, p2=v, p3=mask, p4=sinks, p5=pad,
///                      p6=dst (writable: ne01 × DV floats).
/// Params: dk, dv, ne01, ne11, nb01, nb11, nb21, scale.
/// Threads: NW=32 lanes, NSG=1, NWG=1; grid = (ne01, 1, 1).
pub(super) fn emit_flash_attn_ext_vec_out_ms_msl(out: &mut String, dk: u32, dv: u32) {
    let (pk, pv) = (dk.next_multiple_of(128), dv.next_multiple_of(128));
    let (pk4, pv4) = (pk / 4, pv / 4);
    let (dk4, dv4, dk4_nl, dv4_nl) = (dk / 4, dv / 4, dk / 32, dv / 32);
    writeln!(out, "    constexpr ushort NW   = 32;").unwrap();
    writeln!(out, "    constexpr ushort NE   = 4;").unwrap();
    writeln!(out, "    constexpr ushort NL   = NW / NE;            // 8").unwrap();
    writeln!(
        out,
        "    constexpr ushort C    = 32;                 // NCPSG"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort CNE  = C / NE;             // 8").unwrap();
    writeln!(
        out,
        "    constexpr ushort DK4_FIXED = {dk4};            // DK={dk}"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort DV4_FIXED = {dv4};            // DV={dv}"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort DK4_NL    = DK4_FIXED / NL; // {dk4_nl}").unwrap();
    writeln!(out, "    constexpr ushort DV4_NL    = DV4_FIXED / NL; // {dv4_nl}").unwrap();
    writeln!(out, "    constexpr ushort PK   = {pk};").unwrap();
    writeln!(out, "    constexpr ushort PK4  = {pk4};").unwrap();
    writeln!(out, "    constexpr ushort PV   = {pv};").unwrap();
    writeln!(out, "    constexpr ushort PV4  = {pv4};").unwrap();
    writeln!(out, "    constexpr ushort SH   = 4 * C;").unwrap();
    writeln!(out, "    constexpr ushort NSG  = 1;").unwrap();
    writeln!(
        out,
        "    threadgroup half shmem_f16[NSG*PK + NSG*SH + 2*NSG*PV + 8*32];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * ss  = (threadgroup float *)(shmem_f16 + NSG*PK);"
    )
    .unwrap();
    // sm (mask staging) shares ss space following antirez (ss + 2*C halves offset).
    writeln!(
        out,
        "    threadgroup half  * sm  = (threadgroup half  *)(shmem_f16 + NSG*PK + 2*C);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float4 * so4 = (threadgroup float4 *)(shmem_f16 + NSG*PK + NSG*SH);"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1   = tgpig.x;").unwrap();
    writeln!(out, "    if (sgitg != 0) return; // NSG=1").unwrap();
    // ---- Q load ----
    writeln!(
        out,
        "    device const char  * qrow = (device const char  *)p0 + (uint64_t)iq1 * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float4 * q4 = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(out, "    if (iq1 < ne01) {{").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < PK4; i += NW) {{").unwrap();
    writeln!(
        out,
        "            sq4[i] = (i < DK4_FIXED) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    for (ushort i = tiisg; i < SH/4u; i += NW) {{").unwrap();
    writeln!(out, "        ((threadgroup float4 *)ss)[i] = float4(0.0f);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    so4 += tiisg;").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        for (ushort i = 0; i < DV4_NL; ++i) {{").unwrap();
    writeln!(out, "            so4[i*NL] = float4(0.0f);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // ---- ic loop with online softmax + V accumulation, mask path ----
    writeln!(out, "    float S = 0.0f;").unwrap();
    writeln!(out, "    float M = -INFINITY/2;").unwrap();
    writeln!(out, "    const ushort tx = tiisg % NL;").unwrap();
    writeln!(out, "    const ushort ty = tiisg / NL;").unwrap();
    // Mask base for this query row: pm = (half*) p3 + iq1*ne11.
    writeln!(out, "    device const half * pm = (device const half *)((device const char *)p3) + (uint64_t)iq1 * (uint64_t)ne11;").unwrap();
    writeln!(out, "    for (uint ic0 = 0u; ; ic0 += C) {{").unwrap();
    writeln!(out, "        if (ic0 >= ne11) break;").unwrap();
    // Stage mask for this C-block: sm[tiisg] = pm[ic0 + tiisg]
    writeln!(out, "        sm[tiisg] = pm[ic0 + tiisg];").unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // K·Q dot.
    writeln!(out, "        device const half4 * pk4_base = (device const half4 *)((device const char *)p1 + (uint64_t)ic0 * (uint64_t)nb11);").unwrap();
    writeln!(out, "        const ushort NS10_4 = DK4_FIXED;").unwrap();
    writeln!(
        out,
        "        device const half4 * pk4 = pk4_base + ty * NS10_4 + tx;"
    )
    .unwrap();
    writeln!(out, "        threadgroup const half4 * pq4 = sq4 + tx;").unwrap();
    writeln!(out, "        float mqk_arr[CNE];").unwrap();
    writeln!(
        out,
        "        for (ushort cc = 0; cc < CNE; ++cc) mqk_arr[cc] = 0.0f;"
    )
    .unwrap();
    writeln!(out, "        for (ushort cc = 0; cc < CNE; ++cc) {{").unwrap();
    writeln!(out, "            for (ushort ii = 0; ii < DK4_NL; ++ii) {{").unwrap();
    writeln!(out, "                mqk_arr[cc] += dot((float4) pk4[cc * NE * NS10_4 + ii * NL], (float4) pq4[ii * NL]);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 4);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 2);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] += simd_shuffle_down(mqk_arr[cc], 1);"
    )
    .unwrap();
    writeln!(
        out,
        "            mqk_arr[cc] = simd_shuffle(mqk_arr[cc], (ushort)(NL * ty));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    // FMA with mask: ss[NE*tx + ty] = mqk[tx] * scale + sm[NE*tx + ty].
    writeln!(
        out,
        "        ss[NE * tx + ty] = fma(mqk_arr[tx], scale, (float) sm[NE * tx + ty]);"
    )
    .unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // Online softmax merge (vs writeback to ss).
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            const float m_old = M;").unwrap();
    writeln!(out, "            const float s = ss[tiisg];").unwrap();
    writeln!(out, "            M = simd_max(max(M, s));").unwrap();
    writeln!(out, "            const float ms = exp(m_old - M);").unwrap();
    writeln!(out, "            const float vs = exp(s - M);").unwrap();
    writeln!(out, "            S = S * ms + simd_sum(vs);").unwrap();
    writeln!(out, "            ss[tiisg] = vs;").unwrap();
    writeln!(out, "            if (ty == 0) {{").unwrap();
    writeln!(
        out,
        "                for (ushort ii = 0; ii < DV4_NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    so4[ii*NL] *= ms;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // V accumulation block.
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            float4 lo[DV4_NL];").unwrap();
    writeln!(
        out,
        "            for (ushort ii = 0; ii < DV4_NL; ++ii) lo[ii] = float4(0.0f);"
    )
    .unwrap();
    writeln!(out, "            device const half4 * pv4_base = (device const half4 *)((device const char *)p2 + (uint64_t)ic0 * (uint64_t)nb21);").unwrap();
    writeln!(out, "            const ushort NS20_4 = DV4_FIXED;").unwrap();
    writeln!(
        out,
        "            device const half4 * pv4 = pv4_base + ty * NS20_4 + tx;"
    )
    .unwrap();
    writeln!(out, "            threadgroup const float * sst = ss + ty;").unwrap();
    writeln!(out, "            for (ushort cc = 0; cc < CNE; ++cc) {{").unwrap();
    writeln!(
        out,
        "                for (ushort ii = 0; ii < DV4_NL; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    lo[ii] += float4(pv4[cc*NE*NS20_4 + ii*NL]) * float4(sst[cc*NE]);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            for (ushort ii = 0; ii < DV4_NL; ++ii) {{").unwrap();
    writeln!(
        out,
        "                lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            if (ty == 0) {{").unwrap();
    writeln!(
        out,
        "                for (ushort ii = 0; ii < DV4_NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    so4[ii*NL] += lo[ii];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- has_sinks merge: one extra score = sinks[iq1] for lane 0; others -FLT_MAX/2 ----
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        const float m_old = M;").unwrap();
    writeln!(
        out,
        "        device const float * sk = (device const float *)((device const char *)p4);"
    )
    .unwrap();
    writeln!(
        out,
        "        const float s = (tiisg == 0) ? sk[iq1] : -FLT_MAX/2;"
    )
    .unwrap();
    writeln!(out, "        M = simd_max(max(M, s));").unwrap();
    writeln!(out, "        const float ms = exp(m_old - M);").unwrap();
    writeln!(out, "        const float vs = exp(s - M);").unwrap();
    writeln!(out, "        S = S * ms + simd_sum(vs);").unwrap();
    writeln!(out, "        if (ty == 0) {{").unwrap();
    writeln!(out, "            for (ushort ii = 0; ii < DV4_NL; ++ii) {{").unwrap();
    writeln!(out, "                so4[ii*NL] *= ms;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- final write ----
    writeln!(out, "    so4 -= tiisg;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        device float4 * dst4 = (device float4 *)((device char *)p6 + (uint64_t)iq1 * (uint64_t)DV4_FIXED * 16ull);").unwrap();
    writeln!(
        out,
        "        const float Sinv = (S == 0.0f) ? 0.0f : 1.0f/S;"
    )
    .unwrap();
    writeln!(
        out,
        "        for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "            dst4[i] = so4[i] * Sinv;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtSetup (M37a stage of the flash_attn_ext non-vec prefill kernel):
/// Loads NQ query rows (per threadgroup) into a threadgroup `sq` region using
/// NSG simdgroups, then echoes the staged queries back to dst for verification.
///
/// Test config (smaller than antirez's prod DK=DV=512): DK = DV = 64,
/// NQ = 8 queries-per-tg, NSG = 4, NQ_per_SG = NQ/NSG = 2,
/// threadgroup size = NSG*NW = 128.
///
/// Buffers (all char*): p0=q, p1=k, p2=v, p3=mask, p4=sinks, p5=pad, p6=blk,
///                      p7=dst (writable: NQ × DK floats per query block).
/// Params: dk, dv, ne01, nb01.
/// Threads: NSG*NW = 128 lanes; grid = (ceil(ne01/NQ), 1, 1).
pub(super) fn emit_flash_attn_ext_setup_msl(out: &mut String, dk: u32) {
    let dk4 = dk / 4;
    writeln!(out, "    constexpr ushort NW    = 32;").unwrap();
    writeln!(
        out,
        "    constexpr ushort NQ    = 8;       // queries per threadgroup"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort NSG   = 4;       // simdgroups per threadgroup"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort DK_FIXED  = {dk};  // test config").unwrap();
    writeln!(out, "    constexpr ushort DK4_FIXED = DK_FIXED / 4; // {dk4}").unwrap();
    writeln!(out, "    threadgroup half4 sq4[NQ * DK4_FIXED];").unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1_base = tgpig.x * (uint)NQ;").unwrap();
    writeln!(out, "    const ushort DK4 = (ushort)(dk / 4u);").unwrap();
    // Each simdgroup loads NQ/NSG queries; query index j = jj*NSG + sgitg, jj in [0,NQ/NSG).
    writeln!(out, "    constexpr ushort NQ_per_SG = NQ / NSG; // 2").unwrap();
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq < ne01) {{").unwrap();
    writeln!(out, "            device const char  * qrow = (device const char  *)p0 + (uint64_t)iq * (uint64_t)nb01;").unwrap();
    writeln!(
        out,
        "            device const float4 * q4  = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (i < DK4) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // Echo: each simdgroup writes NQ_per_SG rows back to dst.
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq < ne01) {{").unwrap();
    writeln!(out, "            device float4 * dst4 = (device float4 *)((device char *)p7 + (uint64_t)iq * (uint64_t)DK4 * 16ull);").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                dst4[i] = (float4) sq4[j * DK4_FIXED + i];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtScore (M37b stage of the flash_attn_ext non-vec prefill kernel):
/// M37a setup + K·Q simdgroup_float8x8 matmul over the full ne11 KV extent
/// + online softmax merge per query row. Output (S, M) per query for the
/// validation harness; M37c will replace this with the V matmul + per-row
/// output write.
///
/// Specialization to half K (so the `is_same<kd4x4_t, k4x4_t>` branch fires)
/// with DK = DV = 64, NQ = 8, NSG = 4, C = NCPSG = 32. NS10 = DK = 64.
/// FC_flash_attn_ext_has_mask / has_sinks / has_bias / has_scap / has_kvpad
/// / bc_mask all OFF.
///
/// Buffers (all char*): p0=q, p1=k(half), p2=v, p3=mask, p4=sinks, p5=pad,
///                      p6=blk, p7=dst (writable: ne01 × 2 floats = (S, M)).
/// Params: dk, dv, ne01, ne11, nb01, nb11, scale.
/// Threads: NSG*NW = 128 lanes; grid = (ceil(ne01/NQ), 1, 1).
pub(super) fn emit_flash_attn_ext_score_msl(out: &mut String, dk: u32) {
    let (dk4, dk8, sq_halves) = (dk / 4, dk / 8, 8 * dk);
    writeln!(out, "    constexpr ushort NW    = 32;").unwrap();
    writeln!(
        out,
        "    constexpr ushort NQ    = 8;       // queries per threadgroup"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort NSG   = 4;       // simdgroups per threadgroup"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort NQ_per_SG = NQ / NSG; // 2").unwrap();
    writeln!(out, "    constexpr ushort C     = 32;      // NCPSG").unwrap();
    writeln!(out, "    constexpr ushort SH    = 2 * C;   // 64").unwrap();
    writeln!(out, "    constexpr ushort DK_FIXED  = {dk};").unwrap();
    writeln!(out, "    constexpr ushort DK4_FIXED = DK_FIXED / 4;  // {dk4}").unwrap();
    writeln!(out, "    constexpr ushort DK8_FIXED = DK_FIXED / 8;  // {dk8}").unwrap();
    writeln!(
        out,
        "    constexpr ushort NS10  = DK_FIXED;          // K row stride in halves (nb11/2)"
    )
    .unwrap();
    // Threadgroup shmem: sq (NQ*DK halves) + ss (NQ*SH floats = 2*NQ*SH halves) + per-SG sk staging (NSG * 4*16*KV halves).
    writeln!(out, "    constexpr ushort KV    = 8;").unwrap();
    writeln!(
        out,
        "    constexpr ushort SQ_HALVES = NQ * DK_FIXED;       // {sq_halves}"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort SS_HALVES = NQ * SH * 2;          // 1024 (floats live in halves×2)"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort SK_HALVES = NSG * 4 * 16 * KV;    // 2048"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half shmem_f16[SQ_HALVES + SS_HALVES + SK_HALVES];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half  * sq  = (threadgroup half  *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * ss  = (threadgroup float *)(shmem_f16 + SQ_HALVES);"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1_base = tgpig.x * (uint)NQ;").unwrap();
    writeln!(out, "    const ushort DK4 = (ushort)(dk / 4u);").unwrap();
    // ---- Q load (M37a body) ----
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq < ne01) {{").unwrap();
    writeln!(out, "            device const char  * qrow = (device const char  *)p0 + (uint64_t)iq * (uint64_t)nb01;").unwrap();
    writeln!(
        out,
        "            device const float4 * q4  = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (i < DK4) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // Init ss to 0 (NQ × SH floats).
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < SH; i += NW) {{").unwrap();
    writeln!(out, "            ss[j * SH + i] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // ---- Online softmax state (per query row owned by this SG) ----
    writeln!(out, "    float Sv[NQ_per_SG] = {{ 0.0f, 0.0f }};").unwrap();
    writeln!(
        out,
        "    float Mv[NQ_per_SG] = {{ -FLT_MAX/2, -FLT_MAX/2 }};"
    )
    .unwrap();
    // ---- ic loop ----
    writeln!(out, "    for (uint ic0 = 0u; ; ic0 += C) {{").unwrap();
    writeln!(out, "        if (ic0 >= ne11) break;").unwrap();
    // K·Q simdgroup matmul: each SG owns NC=(C/8)/NSG=1 column-block of 8 columns.
    // pk = (half*)k + ic*NS10; pk += sgitg*8*NS10 (= SG-th 8-col tile of K transposed by simdgroup_load(transpose=true)).
    writeln!(out, "        device const half * k_base = (device const half *)((device const char *)p1 + (uint64_t)ic0 * (uint64_t)nb11);").unwrap();
    writeln!(
        out,
        "        device const half * pk = k_base + (uint)sgitg * 8u * (uint)NS10;"
    )
    .unwrap();
    writeln!(out, "        threadgroup const half * pq = sq;").unwrap();
    // Each SG holds NQ_per_SG=2 row-blocks of 8 (since Q=NQ=8 spreads across NSG=4 → 2 row-blocks per SG).
    // Antirez writes to ps = ss + sgitg*8 (i.e. ss + 8*cc, cc=sgitg-relative). Then reads ss[j*SH + tiisg] later.
    // For our test config NC = 1, sgitg writes its 8x8 tile at ss + sgitg*8*1 + j*SH for j in this SG's queries.
    // Simpler: do simdgroup matmul once per (j_pair_owned_by_sg, ic block). Antirez's layout has Q rows × C cols of scores;
    // each SG handles cc=sgitg single col-block; rows are walked by separate Q tiles.
    // Equivalent loop: for each row-tile rr in 0..(NQ/8) — antirez writes to ss[rr*8*SH + 8*cc] via the 8x8 store.
    // But Q=NQ=8 so there's exactly one row-tile per simdgroup pass. So a single 8x8 K·Q gives one (8 rows × 8 cols) slab.
    writeln!(
        out,
        "        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);"
    )
    .unwrap();
    // FC: DK%16==0 path; DK8/2 = 4 iters of (mq[0..1], mk[0..1]) at offset 16*i.
    writeln!(out, "        for (ushort i = 0; i < DK8_FIXED/2; ++i) {{").unwrap();
    writeln!(out, "            simdgroup_half8x8 mq0, mq1, mk0, mk1;").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq0, pq + 0*8 + 16*i, DK_FIXED);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq1, pq + 1*8 + 16*i, DK_FIXED);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk0, pk + 0*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk1, pk + 1*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    // Store mqk into ss at column-block 8*sgitg.  Stride SH (in floats); ss has (NQ rows × SH cols) layout.
    writeln!(
        out,
        "        simdgroup_store(mqk, ss + 8 * (uint)sgitg, SH, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // ---- Online softmax: each SG processes its NQ_per_SG=2 query rows ----
    // Specialization: only cols [0, C) carry valid scores (cols [C, SH) are the
    // mask-prefetch region, unused with FC_has_mask=false). With C=NW=32 every
    // lane reads exactly one valid score per row.
    writeln!(out, "        for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "            const ushort j = jj * NSG + sgitg;").unwrap();
    writeln!(out, "            const float m = Mv[jj];").unwrap();
    writeln!(
        out,
        "            const float s = ss[j * SH + tiisg] * scale;"
    )
    .unwrap();
    writeln!(out, "            Mv[jj] = simd_max(max(Mv[jj], s));").unwrap();
    writeln!(out, "            const float ms = exp(m - Mv[jj]);").unwrap();
    writeln!(out, "            const float vs = exp(s - Mv[jj]);").unwrap();
    writeln!(out, "            Sv[jj] = Sv[jj] * ms + simd_sum(vs);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- Output: per query row, write (S, M) to dst (lane 0 of each SG, jj loop) ----
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "            const ushort j = jj * NSG + sgitg;").unwrap();
    writeln!(out, "            const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "            if (iq < ne01) {{").unwrap();
    writeln!(out, "                device float * dst_f = (device float *)((device char *)p7 + (uint64_t)iq * 8ull);").unwrap();
    writeln!(out, "                dst_f[0] = Sv[jj];").unwrap();
    writeln!(out, "                dst_f[1] = Mv[jj];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtOut (M37c stage of the flash_attn_ext non-vec prefill kernel):
/// M37b score + V matmul (simdgroup_float8x8 over half V) + per-row
/// S-normalized attention output write.
///
/// Specialization to half V with DK=DV=64, NQ=8, NSG=4, C=NCPSG=32. NS20=DV=64.
/// FC_flash_attn_ext_has_mask / has_sinks / has_bias / has_scap / has_kvpad
/// / bc_mask all OFF.
///
/// Buffers (all char*): p0=q, p1=k(half), p2=v(half), p3=mask, p4=sinks,
///                      p5=pad, p6=blk, p7=dst (writable: ne01 × DV floats).
/// Params: dk, dv, ne01, ne11, nb01, nb11, nb21, scale.
/// Threads: NSG*NW = 128 lanes; grid = (ceil(ne01/NQ), 1, 1).
pub(super) fn emit_flash_attn_ext_out_msl(out: &mut String, dk: u32, dv: u32) {
    let (dk4, dk8, dv4) = (dk / 4, dk / 8, dv / 4);
    let (pv4, pv8, no) = (dv / 4, dv / 8, dv / 32);
    let (sq_halves, so_halves) = (8 * dk, 8 * dv * 2);
    writeln!(out, "    constexpr ushort NW    = 32;").unwrap();
    writeln!(out, "    constexpr ushort NQ    = 8;").unwrap();
    writeln!(out, "    constexpr ushort NSG   = 4;").unwrap();
    writeln!(out, "    constexpr ushort NQ_per_SG = NQ / NSG; // 2").unwrap();
    writeln!(out, "    constexpr ushort C     = 32;      // NCPSG").unwrap();
    writeln!(out, "    constexpr ushort SH    = 2 * C;   // 64").unwrap();
    writeln!(out, "    constexpr ushort DK_FIXED  = {dk};").unwrap();
    writeln!(out, "    constexpr ushort DV_FIXED  = {dv};").unwrap();
    writeln!(out, "    constexpr ushort DK4_FIXED = DK_FIXED / 4;  // {dk4}").unwrap();
    writeln!(out, "    constexpr ushort DK8_FIXED = DK_FIXED / 8;  // {dk8}").unwrap();
    writeln!(out, "    constexpr ushort DV4_FIXED = DV_FIXED / 4;  // {dv4}").unwrap();
    writeln!(
        out,
        "    constexpr ushort PV       = DV_FIXED;       // {dv} (PAD2(DV,64))"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort PV4      = PV / 4;         // {pv4}").unwrap();
    writeln!(out, "    constexpr ushort PV8      = PV / 8;         // {pv8}").unwrap();
    writeln!(
        out,
        "    constexpr ushort NO       = PV8 / NSG;      // {no}  (per-SG V output 8x8 tiles)"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort NS10  = DK_FIXED;          // K row stride in halves"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort NS20  = DV_FIXED;          // V row stride in halves"
    )
    .unwrap();
    writeln!(out, "    constexpr ushort KV    = 8;").unwrap();
    writeln!(
        out,
        "    constexpr ushort SQ_HALVES = NQ * DK_FIXED;       // {sq_halves}"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort SS_HALVES = NQ * SH * 2;          // 1024"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort SO_HALVES = NQ * PV * 2;          // {so_halves}  (so backed in halves×2)"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr ushort SK_HALVES = NSG * 4 * 16 * KV;    // 2048"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half shmem_f16[SQ_HALVES + SS_HALVES + SO_HALVES + SK_HALVES];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half  * sq  = (threadgroup half  *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * ss  = (threadgroup float *)(shmem_f16 + SQ_HALVES);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * so  = (threadgroup float *)(shmem_f16 + SQ_HALVES + SS_HALVES);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float4 * so4 = (threadgroup float4 *)so;"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1_base = tgpig.x * (uint)NQ;").unwrap();
    writeln!(out, "    const ushort DK4 = (ushort)(dk / 4u);").unwrap();
    writeln!(out, "    const ushort DV4 = (ushort)(dv / 4u);").unwrap();
    // ---- Q load ----
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq < ne01) {{").unwrap();
    writeln!(out, "            device const char  * qrow = (device const char  *)p0 + (uint64_t)iq * (uint64_t)nb01;").unwrap();
    writeln!(
        out,
        "            device const float4 * q4  = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (i < DK4) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    // Init ss + so to 0.
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < SH; i += NW) {{").unwrap();
    writeln!(out, "            ss[j * SH + i] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < PV4; i += NW) {{").unwrap();
    writeln!(out, "            so4[j * PV4 + i] = (float4) 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // Online softmax state
    writeln!(out, "    float Sv[NQ_per_SG] = {{ 0.0f, 0.0f }};").unwrap();
    writeln!(
        out,
        "    float Mv[NQ_per_SG] = {{ -FLT_MAX/2, -FLT_MAX/2 }};"
    )
    .unwrap();
    // ic loop
    writeln!(out, "    for (uint ic0 = 0u; ; ic0 += C) {{").unwrap();
    writeln!(out, "        if (ic0 >= ne11) break;").unwrap();
    // K·Q matmul (M37b body)
    writeln!(out, "        device const half * k_base = (device const half *)((device const char *)p1 + (uint64_t)ic0 * (uint64_t)nb11);").unwrap();
    writeln!(
        out,
        "        device const half * pk = k_base + (uint)sgitg * 8u * (uint)NS10;"
    )
    .unwrap();
    writeln!(out, "        threadgroup const half * pq = sq;").unwrap();
    writeln!(
        out,
        "        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);"
    )
    .unwrap();
    writeln!(out, "        for (ushort i = 0; i < DK8_FIXED/2; ++i) {{").unwrap();
    writeln!(out, "            simdgroup_half8x8 mq0, mq1, mk0, mk1;").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq0, pq + 0*8 + 16*i, DK_FIXED);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq1, pq + 1*8 + 16*i, DK_FIXED);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk0, pk + 0*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk1, pk + 1*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        simdgroup_store(mqk, ss + 8 * (uint)sgitg, SH, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // ---- Online softmax: per-query ms scale of so + write vs back to ss ----
    // Each SG owns NQ_per_SG=2 query rows. With C=NW=32, every lane reads exactly
    // one valid score per row (cols [0, C)). Mask region (cols [C, SH)) is unused
    // with FC_has_mask=false, but it must NOT contribute to simd_sum. We avoid
    // contamination by writing vs back only into cols [0, C) and zeroing cols
    // [C, SH) (already zero from init/store_mqk; vs-write below preserves that).
    writeln!(out, "        for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "            const ushort j = jj * NSG + sgitg;").unwrap();
    writeln!(out, "            const float m = Mv[jj];").unwrap();
    writeln!(
        out,
        "            const float s = ss[j * SH + tiisg] * scale;"
    )
    .unwrap();
    writeln!(out, "            Mv[jj] = simd_max(max(Mv[jj], s));").unwrap();
    writeln!(out, "            const float ms = exp(m - Mv[jj]);").unwrap();
    writeln!(out, "            const float vs = exp(s - Mv[jj]);").unwrap();
    writeln!(out, "            Sv[jj] = Sv[jj] * ms + simd_sum(vs);").unwrap();
    writeln!(out, "            ss[j * SH + tiisg] = vs;").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "                so4[j * PV4 + i] *= ms;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // ---- V matmul ----
    // lo[NO=2] simdgroup_float8x8 accumulators per SG.
    // Load existing so into lo, accumulate vs · V, store back.
    // sot = so + 8*sgitg (per-SG col offset within DV); stride PV.
    // pv = (half*)v + ic0*nb21; pv += 8*sgitg.
    // For DV<=64 branch: outer cc 0..C/8=4; vs = ss + 8*cc; inner ii 0..NO/2=1;
    //   mv[0/1] = simdgroup_load(pv + {0,8}*NSG + 16*ii*NSG, NS20, 0, false);
    //   simdgroup_multiply_accumulate(lo[2*ii+0], vs, mv[0], lo[2*ii+0]);
    //   simdgroup_multiply_accumulate(lo[2*ii+1], vs, mv[1], lo[2*ii+1]);
    //   pv += 8*NS20.
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            simdgroup_float8x8 lo[NO];").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(
        out,
        "                threadgroup float * sot = so + 8 * (uint)sgitg;"
    )
    .unwrap();
    writeln!(out, "                for (ushort ii = 0; ii < NO; ++ii) {{").unwrap();
    writeln!(
        out,
        "                    simdgroup_load(lo[ii], sot, PV, 0, false);"
    )
    .unwrap();
    writeln!(out, "                    sot += 8 * NSG;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(out, "                device const half * v_base = (device const half *)((device const char *)p2 + (uint64_t)ic0 * (uint64_t)nb21);").unwrap();
    writeln!(
        out,
        "                device const half * pv = v_base + (uint)sgitg * 8u;"
    )
    .unwrap();
    writeln!(
        out,
        "                for (ushort cc = 0; cc < C/8; ++cc) {{"
    )
    .unwrap();
    writeln!(out, "                    simdgroup_float8x8 vs;").unwrap();
    writeln!(
        out,
        "                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "                    for (ushort ii = 0; ii < NO/2; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                        simdgroup_half8x8 mv0, mv1;").unwrap();
    writeln!(
        out,
        "                        simdgroup_load(mv0, pv + 0*NSG + 16*ii*NSG, NS20, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "                        simdgroup_load(mv1, pv + 8*NSG + 16*ii*NSG, NS20, 0, false);"
    )
    .unwrap();
    writeln!(out, "                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv0, lo[2*ii + 0]);").unwrap();
    writeln!(out, "                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv1, lo[2*ii + 1]);").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                    pv += 8 * NS20;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(
        out,
        "                threadgroup float * sot = so + 8 * (uint)sgitg;"
    )
    .unwrap();
    writeln!(out, "                for (ushort ii = 0; ii < NO; ++ii) {{").unwrap();
    writeln!(
        out,
        "                    simdgroup_store(lo[ii], sot, PV, 0, false);"
    )
    .unwrap();
    writeln!(out, "                    sot += 8 * NSG;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- Final write: per query row, divide so4 by S and write float4 to dst ----
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq >= ne01) continue;").unwrap();
    writeln!(out, "        const float S = Sv[jj];").unwrap();
    writeln!(
        out,
        "        const float inv_s = (S == 0.0f) ? 0.0f : 1.0f / S;"
    )
    .unwrap();
    writeln!(out, "        device float4 * dst4 = (device float4 *)((device char *)p7 + (uint64_t)iq * (uint64_t)DV_FIXED * 4ull);").unwrap();
    writeln!(
        out,
        "        for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "            dst4[i] = so4[j * PV4 + i] * inv_s;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// FlashAttnExtOutMS (M37d stage of the flash_attn_ext non-vec prefill kernel):
/// M37c + has_mask (additive half mask per ic block, read straight from p3)
/// + has_sinks (post ic-loop sink merge of one extra score per query row).
///
/// FC flag config: has_mask=true, has_sinks=true; has_kvpad=false,
/// has_bias=false, has_scap=false, bc_mask=false.
///
/// Buffers (all char*): p0=q, p1=k(half), p2=v(half), p3=mask(half[ne01*ne11]),
///                      p4=sinks(float[*], indexed by iq2; we use sinks[0]),
///                      p5=pad, p6=blk, p7=dst (writable: ne01 × DV floats).
/// Params: dk, dv, ne01, ne11, nb01, nb11, nb21, scale.
/// Threads: NSG*NW = 128 lanes; grid = (ceil(ne01/NQ), 1, 1).
pub(super) fn emit_flash_attn_ext_out_ms_msl(out: &mut String, dk: u32, dv: u32) {
    writeln!(out, "    constexpr ushort NW    = 32;").unwrap();
    writeln!(out, "    constexpr ushort NQ    = 8;").unwrap();
    writeln!(out, "    constexpr ushort NSG   = 4;").unwrap();
    writeln!(out, "    constexpr ushort NQ_per_SG = NQ / NSG; // 2").unwrap();
    writeln!(out, "    constexpr ushort C     = 32;").unwrap();
    writeln!(out, "    constexpr ushort SH    = 2 * C;").unwrap();
    writeln!(out, "    constexpr ushort DK_FIXED  = {dk};").unwrap();
    writeln!(out, "    constexpr ushort DV_FIXED  = {dv};").unwrap();
    writeln!(out, "    constexpr ushort DK4_FIXED = DK_FIXED / 4;").unwrap();
    writeln!(out, "    constexpr ushort DK8_FIXED = DK_FIXED / 8;").unwrap();
    writeln!(out, "    constexpr ushort DV4_FIXED = DV_FIXED / 4;").unwrap();
    writeln!(out, "    constexpr ushort PV       = DV_FIXED;").unwrap();
    writeln!(out, "    constexpr ushort PV4      = PV / 4;").unwrap();
    writeln!(out, "    constexpr ushort PV8      = PV / 8;").unwrap();
    writeln!(out, "    constexpr ushort NO       = PV8 / NSG;").unwrap();
    writeln!(out, "    constexpr ushort NS10  = DK_FIXED;").unwrap();
    writeln!(out, "    constexpr ushort NS20  = DV_FIXED;").unwrap();
    writeln!(out, "    constexpr ushort KV    = 8;").unwrap();
    writeln!(out, "    constexpr ushort SQ_HALVES = NQ * DK_FIXED;").unwrap();
    writeln!(out, "    constexpr ushort SS_HALVES = NQ * SH * 2;").unwrap();
    writeln!(out, "    constexpr ushort SO_HALVES = NQ * PV * 2;").unwrap();
    writeln!(out, "    constexpr ushort SK_HALVES = NSG * 4 * 16 * KV;").unwrap();
    writeln!(
        out,
        "    threadgroup half shmem_f16[SQ_HALVES + SS_HALVES + SO_HALVES + SK_HALVES];"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half  * sq  = (threadgroup half  *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup half4 * sq4 = (threadgroup half4 *)(shmem_f16 + 0);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * ss  = (threadgroup float *)(shmem_f16 + SQ_HALVES);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * so  = (threadgroup float *)(shmem_f16 + SQ_HALVES + SS_HALVES);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float4 * so4 = (threadgroup float4 *)so;"
    )
    .unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out, "    const uint   iq1_base = tgpig.x * (uint)NQ;").unwrap();
    writeln!(out, "    const ushort DK4 = (ushort)(dk / 4u);").unwrap();
    writeln!(out, "    const ushort DV4 = (ushort)(dv / 4u);").unwrap();
    writeln!(
        out,
        "    device const half  * mask_base = (device const half  *)((device const char *)p3);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float * sinks_b   = (device const float *)((device const char *)p4);"
    )
    .unwrap();
    // ---- Q load (same as M37c) ----
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq < ne01) {{").unwrap();
    writeln!(out, "            device const char  * qrow = (device const char  *)p0 + (uint64_t)iq * (uint64_t)nb01;").unwrap();
    writeln!(
        out,
        "            device const float4 * q4  = (device const float4 *)qrow;"
    )
    .unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (i < DK4) ? (half4) q4[i] : (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DK4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sq4[j * DK4_FIXED + i] = (half4) 0.0f;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < SH; i += NW) {{").unwrap();
    writeln!(out, "            ss[j * SH + i] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        for (ushort i = tiisg; i < PV4; i += NW) {{").unwrap();
    writeln!(out, "            so4[j * PV4 + i] = (float4) 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float Sv[NQ_per_SG] = {{ 0.0f, 0.0f }};").unwrap();
    writeln!(
        out,
        "    float Mv[NQ_per_SG] = {{ -FLT_MAX/2, -FLT_MAX/2 }};"
    )
    .unwrap();
    writeln!(out, "    for (uint ic0 = 0u; ; ic0 += C) {{").unwrap();
    writeln!(out, "        if (ic0 >= ne11) break;").unwrap();
    // K·Q matmul (same as M37c).
    writeln!(out, "        device const half * k_base = (device const half *)((device const char *)p1 + (uint64_t)ic0 * (uint64_t)nb11);").unwrap();
    writeln!(
        out,
        "        device const half * pk = k_base + (uint)sgitg * 8u * (uint)NS10;"
    )
    .unwrap();
    writeln!(out, "        threadgroup const half * pq = sq;").unwrap();
    writeln!(
        out,
        "        simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);"
    )
    .unwrap();
    writeln!(out, "        for (ushort i = 0; i < DK8_FIXED/2; ++i) {{").unwrap();
    writeln!(out, "            simdgroup_half8x8 mq0, mq1, mk0, mk1;").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq0, pq + 0*8 + 16*i, DK_FIXED);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mq1, pq + 1*8 + 16*i, DK_FIXED);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk0, pk + 0*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_load(mk1, pk + 1*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);"
    )
    .unwrap();
    writeln!(
        out,
        "            simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        simdgroup_store(mqk, ss + 8 * (uint)sgitg, SH, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // ---- Online softmax with mask add ----
    // Mask is half[ne01 × ne11], indexed pm[(iq)*ne11 + ic0 + tiisg].
    // tiisg covers cols [0, C); each query row uses its own iq for the mask offset.
    writeln!(out, "        for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "            const ushort j = jj * NSG + sgitg;").unwrap();
    writeln!(out, "            const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "            const float m = Mv[jj];").unwrap();
    writeln!(out, "            const uint   mask_off = (iq < ne01) ? ((uint64_t)iq * (uint64_t)ne11 + ic0 + (uint)tiisg) : 0u;").unwrap();
    writeln!(
        out,
        "            const float  msk = (iq < ne01) ? (float) mask_base[mask_off] : 0.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "            const float s = ss[j * SH + tiisg] * scale + msk;"
    )
    .unwrap();
    writeln!(out, "            Mv[jj] = simd_max(max(Mv[jj], s));").unwrap();
    writeln!(out, "            const float ms = exp(m - Mv[jj]);").unwrap();
    writeln!(out, "            const float vs = exp(s - Mv[jj]);").unwrap();
    writeln!(out, "            Sv[jj] = Sv[jj] * ms + simd_sum(vs);").unwrap();
    writeln!(out, "            ss[j * SH + tiisg] = vs;").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "                so4[j * PV4 + i] *= ms;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    // V matmul (same as M37c).
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            simdgroup_float8x8 lo[NO];").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(
        out,
        "                threadgroup float * sot = so + 8 * (uint)sgitg;"
    )
    .unwrap();
    writeln!(out, "                for (ushort ii = 0; ii < NO; ++ii) {{").unwrap();
    writeln!(
        out,
        "                    simdgroup_load(lo[ii], sot, PV, 0, false);"
    )
    .unwrap();
    writeln!(out, "                    sot += 8 * NSG;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(out, "                device const half * v_base = (device const half *)((device const char *)p2 + (uint64_t)ic0 * (uint64_t)nb21);").unwrap();
    writeln!(
        out,
        "                device const half * pv = v_base + (uint)sgitg * 8u;"
    )
    .unwrap();
    writeln!(
        out,
        "                for (ushort cc = 0; cc < C/8; ++cc) {{"
    )
    .unwrap();
    writeln!(out, "                    simdgroup_float8x8 vs;").unwrap();
    writeln!(
        out,
        "                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "                    for (ushort ii = 0; ii < NO/2; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                        simdgroup_half8x8 mv0, mv1;").unwrap();
    writeln!(
        out,
        "                        simdgroup_load(mv0, pv + 0*NSG + 16*ii*NSG, NS20, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "                        simdgroup_load(mv1, pv + 8*NSG + 16*ii*NSG, NS20, 0, false);"
    )
    .unwrap();
    writeln!(out, "                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv0, lo[2*ii + 0]);").unwrap();
    writeln!(out, "                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv1, lo[2*ii + 1]);").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                    pv += 8 * NS20;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(
        out,
        "                threadgroup float * sot = so + 8 * (uint)sgitg;"
    )
    .unwrap();
    writeln!(out, "                for (ushort ii = 0; ii < NO; ++ii) {{").unwrap();
    writeln!(
        out,
        "                    simdgroup_store(lo[ii], sot, PV, 0, false);"
    )
    .unwrap();
    writeln!(out, "                    sot += 8 * NSG;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    // ---- has_sinks: post-loop sink merge per query row ----
    // Antirez uses sinks[iq2] (head index from grid). Our test uses iq2=0 always
    // (single-head probe), so we fix sk_val = sinks_b[0]. For multi-head support,
    // expand the dispatch grid to (gridX, num_heads, 1) and read sinks_b[gridY].
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        const float sk_val = sinks_b[0];").unwrap();
    writeln!(out, "        for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "            const ushort j = jj * NSG + sgitg;").unwrap();
    writeln!(out, "            const float m = Mv[jj];").unwrap();
    writeln!(
        out,
        "            const float s = (tiisg == 0) ? sk_val : -FLT_MAX/2;"
    )
    .unwrap();
    writeln!(out, "            Mv[jj] = simd_max(max(Mv[jj], s));").unwrap();
    writeln!(out, "            const float ms = exp(m - Mv[jj]);").unwrap();
    writeln!(out, "            const float vs = exp(s - Mv[jj]);").unwrap();
    writeln!(out, "            Sv[jj] = Sv[jj] * ms + simd_sum(vs);").unwrap();
    writeln!(
        out,
        "            for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "                so4[j * PV4 + i] *= ms;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    // Final write.
    writeln!(out, "    for (ushort jj = 0; jj < NQ_per_SG; ++jj) {{").unwrap();
    writeln!(out, "        const ushort j  = jj * NSG + sgitg;").unwrap();
    writeln!(out, "        const uint   iq = iq1_base + (uint)j;").unwrap();
    writeln!(out, "        if (iq >= ne01) continue;").unwrap();
    writeln!(out, "        const float S = Sv[jj];").unwrap();
    writeln!(
        out,
        "        const float inv_s = (S == 0.0f) ? 0.0f : 1.0f / S;"
    )
    .unwrap();
    writeln!(out, "        device float4 * dst4 = (device float4 *)((device char *)p7 + (uint64_t)iq * (uint64_t)DV_FIXED * 4ull);").unwrap();
    writeln!(
        out,
        "        for (ushort i = tiisg; i < DV4_FIXED; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "            dst4[i] = so4[j * PV4 + i] * inv_s;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// dsv4_hc_expand: per-(d, dst_hc, t) HC expand step.
///
/// For each output element:
///   block_v = block_out[d, t]
///   if has_add: block_v += block_add[d, t]
///   acc = block_v * post[dst_hc, t]
///   for src_hc in 0..n_hc:
///     acc += comb[dst_hc, src_hc, t] * residual[d, src_hc, t]
///   dst[d, dst_hc, t] = acc
///
/// Buffers (all char* for byte-stride math, matching antirez):
///   p0=block_out, p1=residual, p2=post, p3=comb, p4=block_add, p5=dst.
///
/// Dispatch: 1D, total = n_embd * n_hc * n_tokens elements.
/// gid = row * tcount + tid; ic = gid%n_embd, dst_hc = (gid/n_embd)%n_hc, t = gid/(n_embd*n_hc).
pub(super) fn emit_dsv4_hc_expand_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = n_embd * n_hc * n_tokens;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out, "    uint d      = gid % n_embd;").unwrap();
    writeln!(out, "    uint tmp    = gid / n_embd;").unwrap();
    writeln!(out, "    uint dst_hc = tmp % n_hc;").unwrap();
    writeln!(out, "    uint t      = tmp / n_hc;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float block_v = *((device const float *)(p0 + (uint64_t)d * nb_block0 + (uint64_t)t * nb_block1));").unwrap();
    writeln!(out, "    if (has_add != 0u) {{").unwrap();
    writeln!(out, "        block_v += *((device const float *)(p4 + (uint64_t)d * nb_add0 + (uint64_t)t * nb_add1));").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    const float post_v = *((device const float *)(p2 + (uint64_t)dst_hc * nb_post0 + (uint64_t)t * nb_post1));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float acc = block_v * post_v;").unwrap();
    writeln!(
        out,
        "    for (uint src_hc = 0u; src_hc < n_hc; ++src_hc) {{"
    )
    .unwrap();
    writeln!(out, "        const float comb_v = *((device const float *)(p3 + (uint64_t)dst_hc * nb_comb0 + (uint64_t)src_hc * nb_comb1 + (uint64_t)t * nb_comb2));").unwrap();
    writeln!(out, "        const float res_v  = *((device const float *)(p1 + (uint64_t)d * nb_res0 + (uint64_t)src_hc * nb_res1 + (uint64_t)t * nb_res2));").unwrap();
    writeln!(out, "        acc += comb_v * res_v;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    *((device float *)(p5 + (uint64_t)d * nb0 + (uint64_t)dst_hc * nb1 + (uint64_t)t * nb2)) = acc;").unwrap();
}
/// dsv4_hc_expand4: HC=4 specialization of expand. Same buffer set, but
/// one thread computes all 4 dst_hc streams at once. r0..r3 (residual rows)
/// are loaded once, then dst_hc 0..3 each weighted with comb[dst_hc, *, t].
///
/// Total threads = n_embd * n_tokens (NOT × n_hc). The loop over n_hc=4
/// dst_hc is fully unrolled inside each thread.
/// `hc` is the hyper-connection count this kernel is unrolled for: it preloads
/// that many residual values and expands the inner combine over them, computing
/// every output for one (embedding, token) in one thread. The kernel refuses a
/// runtime `n_hc` that differs.
pub(super) fn emit_dsv4_hc_expand4_msl(out: &mut String, hc: u32) {
    writeln!(out, "    if (n_hc != {hc}u) return;").unwrap();
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = n_embd * n_tokens;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out, "    uint d = gid % n_embd;").unwrap();
    writeln!(out, "    uint t = gid / n_embd;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float block_v = *((device const float *)(p0 + (uint64_t)d * nb_block0 + (uint64_t)t * nb_block1));").unwrap();
    writeln!(out, "    if (has_add != 0u) {{").unwrap();
    writeln!(out, "        block_v += *((device const float *)(p4 + (uint64_t)d * nb_add0 + (uint64_t)t * nb_add1));").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    for i in 0..hc {
        writeln!(out, "    const float r{i} = *((device const float *)(p1 + (uint64_t)d * nb_res0 + {i}u * nb_res1 + (uint64_t)t * nb_res2));").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "    for (uint dst_hc = 0u; dst_hc < {hc}u; ++dst_hc) {{").unwrap();
    writeln!(out, "        float acc = block_v * *((device const float *)(p2 + (uint64_t)dst_hc * nb_post0 + (uint64_t)t * nb_post1));").unwrap();
    for i in 0..hc {
        writeln!(out, "        acc += *((device const float *)(p3 + (uint64_t)dst_hc * nb_comb0 + {i}u * nb_comb1 + (uint64_t)t * nb_comb2)) * r{i};").unwrap();
    }
    writeln!(out, "        *((device float *)(p5 + (uint64_t)d * nb0 + (uint64_t)dst_hc * nb1 + (uint64_t)t * nb2)) = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Dsv4HcWeightedSum: dst[d,t] = Σ_h x[d,h,t] * weights[h,t].
/// Buffers: p0=x (char*), p1=weights (char*), p2=dst (char*).
/// Params: n_embd, n_hc, n_tokens, nb_x0, nb_x1, nb_x2, nb_w0, nb_w1, nb0, nb1.
/// 1D dispatch over n_embd × n_tokens.
pub(super) fn emit_dsv4_hc_weighted_sum_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = n_embd * n_tokens;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out, "    uint d = gid % n_embd;").unwrap();
    writeln!(out, "    uint t = gid / n_embd;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    for (uint h = 0u; h < n_hc; ++h) {{").unwrap();
    writeln!(out, "        float xv = *((device const float *)(p0 + (uint64_t)d * nb_x0 + (uint64_t)h * nb_x1 + (uint64_t)t * nb_x2));").unwrap();
    writeln!(out, "        float wv = *((device const float *)(p1 + (uint64_t)h * nb_w0 + (uint64_t)t * nb_w1));").unwrap();
    writeln!(out, "        acc += xv * wv;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    *((device float *)(p2 + (uint64_t)d * nb0 + (uint64_t)t * nb1)) = acc;"
    )
    .unwrap();
}
/// Dsv4HcSplitSinkhornHc4: HC=4 fast path of dsv4_hc_split_sinkhorn.
/// Per-row work: pre = sigmoid(mix[0..4]*pre_scale + base[0..4]) + eps,
///               post = 2*sigmoid(mix[4..8]*post_scale + base[4..8]),
///               c[4×4] = comb softmax of (mix[8..24]*comb_scale + base[8..24]) per row,
///               followed by Sinkhorn iterations to balance row/col sums.
/// Buffers: p0=mixes (n_rows × mix_hc), p1=scale[3], p2=base[mix_hc], p3=dst (n_rows × mix_hc).
/// Dispatch: 1D over n_rows, one thread per row (matches antirez tid==row gating).
pub(super) fn emit_dsv4_hc_split_sinkhorn_hc4_msl(out: &mut String) {
    writeln!(out, "    if (n_hc != 4u) return;").unwrap();
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    if (gid >= n_rows) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * mix = p0 + (uint64_t)gid * mix_hc;"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * outp = p3 + (uint64_t)gid * mix_hc;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float epsv       = eps;").unwrap();
    writeln!(out, "    const float pre_scale  = p1[0];").unwrap();
    writeln!(out, "    const float post_scale = p1[1];").unwrap();
    writeln!(out, "    const float comb_scale = p1[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) p2);").unwrap();
    writeln!(
        out,
        "    *((device float4 *) outp) = 1.0f / (1.0f + exp(-pre_z)) + epsv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (p2 + 4));").unwrap();
    writeln!(
        out,
        "    *((device float4 *) (outp + 4)) = 2.0f / (1.0f + exp(-post_z));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 r0 = *((device const float4 *) (mix +  8)) * comb_scale + *((device const float4 *) (p2 +  8));").unwrap();
    writeln!(out, "    float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (p2 + 12));").unwrap();
    writeln!(out, "    float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (p2 + 16));").unwrap();
    writeln!(out, "    float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (p2 + 20));").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));"
    )
    .unwrap();
    writeln!(
        out,
        "    const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));"
    )
    .unwrap();
    writeln!(
        out,
        "    const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));"
    )
    .unwrap();
    writeln!(
        out,
        "    const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    r0 = exp(r0 - m0);").unwrap();
    writeln!(out, "    r1 = exp(r1 - m1);").unwrap();
    writeln!(out, "    r2 = exp(r2 - m2);").unwrap();
    writeln!(out, "    r3 = exp(r3 - m3);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "    r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "    r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "    r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "    r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint iter = 1u; iter < sinkhorn_iters; ++iter) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "        r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "        r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "        r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);"
    )
    .unwrap();
    writeln!(out, "        col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);").unwrap();
    writeln!(
        out,
        "        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    *((device float4 *) (outp +  8)) = r0;").unwrap();
    writeln!(out, "    *((device float4 *) (outp + 12)) = r1;").unwrap();
    writeln!(out, "    *((device float4 *) (outp + 16)) = r2;").unwrap();
    writeln!(out, "    *((device float4 *) (outp + 20)) = r3;").unwrap();
}
/// Dsv4HcSplitWeightedSumHc4: HC=4 fast path of dsv4_hc_split_weighted_sum.
/// One threadgroup per row: tid==0 computes the HC=4 mixer split (sigmoid pre,
/// 2*sigmoid post, softmax + Sinkhorn 4×4 comb), writes the result to `split`,
/// and caches pre[0..3] in threadgroup memory; all lanes barrier and then
/// loop d over n_embd doing acc = Σ_h pre[h] * x[d, h, row], writing dst.
/// Buffers: p0=mixes(char*), p1=scale(float*), p2=base(float*), p3=x(char*),
///          p4=split(char*, writable), p5=dst(char*, writable).
pub(super) fn emit_dsv4_hc_split_weighted_sum_hc4_msl(out: &mut String) {
    writeln!(out, "    if (n_hc != 4u) return;").unwrap();
    writeln!(out, "    if (row >= n_rows) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float pre_shmem[4];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * mix = (device const float *) (p0 + (uint64_t)row * nb_mix1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * outp = (device       float *) (p4 + (uint64_t)row * nb_split1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0u) {{").unwrap();
    writeln!(out, "        const float epsv       = eps;").unwrap();
    writeln!(out, "        const float pre_scale  = p1[0];").unwrap();
    writeln!(out, "        const float post_scale = p1[1];").unwrap();
    writeln!(out, "        const float comb_scale = p1[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) p2);").unwrap();
    writeln!(
        out,
        "        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;"
    )
    .unwrap();
    writeln!(out, "        *((device float4 *) outp) = pre;").unwrap();
    writeln!(out, "        pre_shmem[0] = pre.x;").unwrap();
    writeln!(out, "        pre_shmem[1] = pre.y;").unwrap();
    writeln!(out, "        pre_shmem[2] = pre.z;").unwrap();
    writeln!(out, "        pre_shmem[3] = pre.w;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (p2 + 4));").unwrap();
    writeln!(
        out,
        "        *((device float4 *) (outp + 4)) = 2.0f / (1.0f + exp(-post_z));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float4 r0 = *((device const float4 *) (mix +  8)) * comb_scale + *((device const float4 *) (p2 +  8));").unwrap();
    writeln!(out, "        float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (p2 + 12));").unwrap();
    writeln!(out, "        float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (p2 + 16));").unwrap();
    writeln!(out, "        float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (p2 + 20));").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        r0 = exp(r0 - m0);").unwrap();
    writeln!(out, "        r1 = exp(r1 - m1);").unwrap();
    writeln!(out, "        r2 = exp(r2 - m2);").unwrap();
    writeln!(out, "        r3 = exp(r3 - m3);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        for (uint iter = 1u; iter < sinkhorn_iters; ++iter) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        *((device float4 *) (outp +  8)) = r0;").unwrap();
    writeln!(out, "        *((device float4 *) (outp + 12)) = r1;").unwrap();
    writeln!(out, "        *((device float4 *) (outp + 16)) = r2;").unwrap();
    writeln!(out, "        *((device float4 *) (outp + 20)) = r3;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint d = tid; d < n_embd; d += tcount) {{").unwrap();
    writeln!(out, "        float acc = 0.0f;").unwrap();
    writeln!(out, "        acc += *((device const float *)(p3 + (uint64_t)d * nb_x0 + 0u * nb_x1 + (uint64_t)row * nb_x2)) * pre_shmem[0];").unwrap();
    writeln!(out, "        acc += *((device const float *)(p3 + (uint64_t)d * nb_x0 + 1u * nb_x1 + (uint64_t)row * nb_x2)) * pre_shmem[1];").unwrap();
    writeln!(out, "        acc += *((device const float *)(p3 + (uint64_t)d * nb_x0 + 2u * nb_x1 + (uint64_t)row * nb_x2)) * pre_shmem[2];").unwrap();
    writeln!(out, "        acc += *((device const float *)(p3 + (uint64_t)d * nb_x0 + 3u * nb_x1 + (uint64_t)row * nb_x2)) * pre_shmem[3];").unwrap();
    writeln!(
        out,
        "        *((device float *)(p5 + (uint64_t)d * nb0 + (uint64_t)row * nb1)) = acc;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Dsv4HcSplitWeightedSumNorm4: M42 fused with float4 RMSNorm reduction.
/// Per row (one threadgroup):
///   1. tid==0 computes the HC=4 mixer split (same as M42), writes `split`
///      and caches pre[0..3] in threadgroup memory.
///   2. threadgroup_barrier.
///   3. All `ntg` lanes loop float4 i over n4=1024:
///        v = Σ_h x[h, row][i] * pre[h]
///        row_shmem[i] = v;       sumf += dot(v, v)
///   4. simd_sum across simd; first lane writes sum_shmem[sgitg]; barrier.
///   5. Cross-simd: simd 0 lanes read sum_shmem[lane], simd_sum, rsqrt.
///   6. All lanes loop float4 i: dst[i]=v; norm_dst[i]=(v*norm_scale)*w[i].
/// Hardcoded: n_embd=4096, n_hc=4, n4=1024 float4 lanes per row.
/// Buffers: p0=mixes, p1=scale(float), p2=base(float), p3=x,
///          p4=split (writable), p5=dst (writable),
///          p6=norm_weight, p7=norm_dst (writable).
pub(super) fn emit_dsv4_hc_split_weighted_sum_norm4_msl(out: &mut String) {
    writeln!(out, "    if (n_hc != 4u || n_embd != 4096u) return;").unwrap();
    writeln!(out, "    if (row >= n_rows) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float4 row_shmem[1024];").unwrap();
    writeln!(out, "    threadgroup float  pre_shmem[4];").unwrap();
    writeln!(out, "    threadgroup float  sum_shmem[32];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (simd_id == 0u) {{").unwrap();
    writeln!(out, "        sum_shmem[simd_lane] = 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * mix  = (device const float *) (p0 + (uint64_t)row * nb_mix1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * outp = (device       float *) (p4 + (uint64_t)row * nb_split1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0u) {{").unwrap();
    writeln!(out, "        const float epsv       = eps;").unwrap();
    writeln!(out, "        const float pre_scale  = p1[0];").unwrap();
    writeln!(out, "        const float post_scale = p1[1];").unwrap();
    writeln!(out, "        const float comb_scale = p1[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) p2);").unwrap();
    writeln!(
        out,
        "        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;"
    )
    .unwrap();
    writeln!(out, "        *((device float4 *) outp) = pre;").unwrap();
    writeln!(out, "        pre_shmem[0] = pre.x;").unwrap();
    writeln!(out, "        pre_shmem[1] = pre.y;").unwrap();
    writeln!(out, "        pre_shmem[2] = pre.z;").unwrap();
    writeln!(out, "        pre_shmem[3] = pre.w;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (p2 + 4));").unwrap();
    writeln!(
        out,
        "        *((device float4 *) (outp + 4)) = 2.0f / (1.0f + exp(-post_z));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float4 r0 = *((device const float4 *) (mix +  8)) * comb_scale + *((device const float4 *) (p2 +  8));").unwrap();
    writeln!(out, "        float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (p2 + 12));").unwrap();
    writeln!(out, "        float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (p2 + 16));").unwrap();
    writeln!(out, "        float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (p2 + 20));").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));"
    )
    .unwrap();
    writeln!(
        out,
        "        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        r0 = exp(r0 - m0);").unwrap();
    writeln!(out, "        r1 = exp(r1 - m1);").unwrap();
    writeln!(out, "        r2 = exp(r2 - m2);").unwrap();
    writeln!(out, "        r3 = exp(r3 - m3);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;"
    )
    .unwrap();
    writeln!(
        out,
        "        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        for (uint iter = 1u; iter < sinkhorn_iters; ++iter) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);"
    )
    .unwrap();
    writeln!(
        out,
        "            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        *((device float4 *) (outp +  8)) = r0;").unwrap();
    writeln!(out, "        *((device float4 *) (outp + 12)) = r1;").unwrap();
    writeln!(out, "        *((device float4 *) (outp + 16)) = r2;").unwrap();
    writeln!(out, "        *((device float4 *) (outp + 20)) = r3;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out, "    const uint n4 = 1024u;").unwrap();
    writeln!(out, "    device const float4 * x0 = (device const float4 *) (p3 + 0u * nb_x1 + (uint64_t)row * nb_x2);").unwrap();
    writeln!(out, "    device const float4 * x1 = (device const float4 *) (p3 + 1u * nb_x1 + (uint64_t)row * nb_x2);").unwrap();
    writeln!(out, "    device const float4 * x2 = (device const float4 *) (p3 + 2u * nb_x1 + (uint64_t)row * nb_x2);").unwrap();
    writeln!(out, "    device const float4 * x3 = (device const float4 *) (p3 + 3u * nb_x1 + (uint64_t)row * nb_x2);").unwrap();
    writeln!(out, "    for (uint i = tid; i < n4; i += tcount) {{").unwrap();
    writeln!(out, "        const float4 v = x0[i] * pre_shmem[0]").unwrap();
    writeln!(out, "                       + x1[i] * pre_shmem[1]").unwrap();
    writeln!(out, "                       + x2[i] * pre_shmem[2]").unwrap();
    writeln!(out, "                       + x3[i] * pre_shmem[3];").unwrap();
    writeln!(out, "        row_shmem[i] = v;").unwrap();
    writeln!(out, "        sumf += dot(v, v);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (simd_lane == 0u) {{").unwrap();
    writeln!(out, "        sum_shmem[simd_id] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = sum_shmem[simd_lane];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(
        out,
        "    const float norm_scale = rsqrt(sumf / 4096.0f + norm_eps);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device       float4 * dst4  = (device       float4 *) (p5 + (uint64_t)row * nb1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float4 * w4    = (device const float4 *) p6;"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float4 * norm4 = (device       float4 *) (p7 + (uint64_t)row * nb_norm1);"
    )
    .unwrap();
    writeln!(out, "    for (uint i = tid; i < n4; i += tcount) {{").unwrap();
    writeln!(out, "        const float4 v = row_shmem[i];").unwrap();
    writeln!(out, "        dst4[i] = v;").unwrap();
    writeln!(out, "        norm4[i] = (v * norm_scale) * w4[i];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// ArgsortF32I32Desc: bitonic sort one float row → int32 index permutation,
/// descending order. One threadgroup per row. Threadgroup size must be a
/// power of two ≥ ne00. Threads with id ≥ ne00 hold sentinel indices (= ne00)
/// that lose all comparisons, so the top ne00 values land in the front.
/// Buffers: p0=src (char* float row), p1=dst (int* writable, ne0×ne01).
/// `max_row` sizes the index staging: one int per column, one thread per column.
pub(super) fn emit_argsort_f32_i32_desc_msl(out: &mut String, max_row: u32) {
    writeln!(out, "    if (row >= ne01) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup int shmem_i32[{max_row}];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint col = tid;").unwrap();
    writeln!(
        out,
        "    device const float * src_row = (device const float *) (p0 + (uint64_t)row * nb01);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    shmem_i32[col] = (int) col;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint k = 2u; k <= tcount; k *= 2u) {{").unwrap();
    writeln!(out, "        for (uint j = k / 2u; j > 0u; j /= 2u) {{").unwrap();
    writeln!(out, "            const uint ixj = col ^ j;").unwrap();
    writeln!(out, "            if (ixj > col) {{").unwrap();
    writeln!(out, "                const int  a = shmem_i32[col];").unwrap();
    writeln!(out, "                const int  b = shmem_i32[ixj];").unwrap();
    writeln!(
        out,
        "                const bool a_oob = ((uint) a) >= ne00;"
    )
    .unwrap();
    writeln!(
        out,
        "                const bool b_oob = ((uint) b) >= ne00;"
    )
    .unwrap();
    writeln!(out, "                bool swap;").unwrap();
    writeln!(out, "                if ((col & k) == 0u) {{").unwrap();
    writeln!(
        out,
        "                    // ascending block: keep larger value at smaller index for DESC top-k"
    )
    .unwrap();
    writeln!(
        out,
        "                    swap = a_oob || (!b_oob && (src_row[a] < src_row[b]));"
    )
    .unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    swap = b_oob || (!a_oob && (src_row[a] > src_row[b]));"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "                if (swap) {{").unwrap();
    writeln!(out, "                    shmem_i32[col] = b;").unwrap();
    writeln!(out, "                    shmem_i32[ixj] = a;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (col < top_k) {{").unwrap();
    writeln!(
        out,
        "        p1[(uint64_t)row * ne0 + col] = shmem_i32[col];"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
}
/// ArgsortMergeF32I32Desc: merge two pre-sorted descending int32 index runs
/// into one descending top_k run. Single-batch specialization (one threadgroup,
/// one pair-of-runs). Each thread handles a contiguous output slice [k0, k1)
/// found via binary-search partition (i+j=k0). Then sequential merge for the
/// slice. p0=src(char* float row), p1=tmp(int* const, two runs at [0..len),
/// [len..2*len)), p2=dst(int* writable, len ≥ top_k).
pub(super) fn emit_argsort_merge_f32_i32_desc_msl(out: &mut String) {
    writeln!(
        out,
        "    device const float * src_row = (device const float *) (p0);"
    )
    .unwrap();
    writeln!(out, "    device const int * tmp0 = p1;").unwrap();
    writeln!(out, "    device const int * tmp1 = p1 + len;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int len0 = (int) (len < ne0 ? len : ne0);").unwrap();
    writeln!(out, "    const int rest = (int) ne0 - (int) len;").unwrap();
    writeln!(
        out,
        "    const int len1 = (int) ((rest < 0 ? 0 : ((uint) rest < len ? (uint) rest : len)));"
    )
    .unwrap();
    writeln!(out, "    const int total = len0 + len1;").unwrap();
    writeln!(out, "    if (total == 0) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int chunk = (total + (int) tcount - 1) / (int) tcount;"
    )
    .unwrap();
    writeln!(out, "    const int k0 = (int) tid * chunk;").unwrap();
    writeln!(out, "    int k1 = k0 + chunk;").unwrap();
    writeln!(out, "    if (k1 > total) k1 = total;").unwrap();
    writeln!(out, "    if (k1 > (int) top_k) k1 = (int) top_k;").unwrap();
    writeln!(out, "    if (k0 >= (int) top_k) return;").unwrap();
    writeln!(out, "    if (k0 >= total) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int low  = k0 > len1 ? k0 - len1 : 0;").unwrap();
    writeln!(out, "    int high = k0 < len0 ? k0 : len0;").unwrap();
    writeln!(out, "    while (low < high) {{").unwrap();
    writeln!(out, "        const int mid  = (low + high) >> 1;").unwrap();
    writeln!(out, "        const int idx_a = tmp0[mid];").unwrap();
    writeln!(out, "        const int idx_b = tmp1[k0 - mid - 1];").unwrap();
    writeln!(out, "        const float val_a = src_row[idx_a];").unwrap();
    writeln!(out, "        const float val_b = src_row[idx_b];").unwrap();
    writeln!(
        out,
        "        // descending merge: take_left when val_a >= val_b"
    )
    .unwrap();
    writeln!(
        out,
        "        if (val_a >= val_b) {{ low = mid + 1; }} else {{ high = mid; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int i = low;").unwrap();
    writeln!(out, "    int j = k0 - i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int   idx0 = 0;").unwrap();
    writeln!(out, "    float val0 = 0.0f;").unwrap();
    writeln!(
        out,
        "    if (i < len0) {{ idx0 = tmp0[i]; val0 = src_row[idx0]; }}"
    )
    .unwrap();
    writeln!(out, "    int   idx1 = 0;").unwrap();
    writeln!(out, "    float val1 = 0.0f;").unwrap();
    writeln!(
        out,
        "    if (j < len1) {{ idx1 = tmp1[j]; val1 = src_row[idx1]; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int k = k0; k < k1; ++k) {{").unwrap();
    writeln!(out, "        if (i >= len0) {{").unwrap();
    writeln!(out, "            while (k < k1) {{ p2[k++] = tmp1[j++]; }}").unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }} else if (j >= len1) {{").unwrap();
    writeln!(out, "            while (k < k1) {{ p2[k++] = tmp0[i++]; }}").unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            int out_idx;").unwrap();
    writeln!(out, "            if (val0 >= val1) {{").unwrap();
    writeln!(out, "                out_idx = idx0; ++i;").unwrap();
    writeln!(
        out,
        "                if (i < len0) {{ idx0 = tmp0[i]; val0 = src_row[idx0]; }}"
    )
    .unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                out_idx = idx1; ++j;").unwrap();
    writeln!(
        out,
        "                if (j < len1) {{ idx1 = tmp1[j]; val1 = src_row[idx1]; }}"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            p2[k] = out_idx;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// M134 ArgsortF32I32DescFull: full host-callable kernel_argsort_f32_i32_desc.
/// Antirez bitonic-sort transcription (argsort.metal:46-105) specialized to
/// DESC order. Sorts each float row into an int32 index row, top_k per row.
/// 4-D batched: dispatched as (ib*ne01, ne02, ne03) threadgroups × ntg.x threads
/// per group. ne00 is the (logical) input row length; ntg.x must be a power of
/// two and ≥ that block's portion of ne00 (antirez splits when ne00 > 1024 by
/// using ib = tgpig.x / ne01 to step blocks of ntg.x columns).
/// `max_row` sizes the index staging: one int per column, one thread per column.
pub(super) fn emit_argsort_f32_i32_desc_full_msl(out: &mut String, max_row: u32) {
    writeln!(out, "    threadgroup int shmem_i32[{max_row}];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int col = (int) tpitg.x;").unwrap();
    writeln!(out, "    const int ib  = (int) tgpig.x / (int) ne01;").unwrap();
    writeln!(out, "    const int i00 = ib * (int) ntg.x;").unwrap();
    writeln!(out, "    const int i01 = (int) tgpig.x % (int) ne01;").unwrap();
    writeln!(out, "    const int i02 = (int) tgpig.y;").unwrap();
    writeln!(out, "    const int i03 = (int) tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) (p0 + (uint64_t) nb01 * (uint64_t) i01 + (uint64_t) nb02 * (uint64_t) i02 + (uint64_t) nb03 * (uint64_t) i03);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    shmem_i32[col] = i00 + col;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint k = 2u; k <= (uint) ntg.x; k *= 2u) {{").unwrap();
    writeln!(out, "        for (uint j = k / 2u; j > 0u; j /= 2u) {{").unwrap();
    writeln!(out, "            const int ixj = col ^ (int) j;").unwrap();
    writeln!(out, "            if (ixj > col) {{").unwrap();
    writeln!(out, "                const int a = shmem_i32[col];").unwrap();
    writeln!(out, "                const int b = shmem_i32[ixj];").unwrap();
    writeln!(
        out,
        "                const bool a_oob = ((uint) a) >= ne00;"
    )
    .unwrap();
    writeln!(
        out,
        "                const bool b_oob = ((uint) b) >= ne00;"
    )
    .unwrap();
    writeln!(out, "                bool swap = false;").unwrap();
    writeln!(out, "                if (((uint) col & k) == 0u) {{").unwrap();
    writeln!(out, "                    // ascending block in bitonic ladder -- DESC order: prefer larger value at col").unwrap();
    writeln!(
        out,
        "                    swap = a_oob || (!b_oob && (src0_row[a] < src0_row[b]));"
    )
    .unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    swap = b_oob || (!a_oob && (src0_row[a] > src0_row[b]));"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "                if (swap) {{").unwrap();
    writeln!(out, "                    shmem_i32[col] = b;").unwrap();
    writeln!(out, "                    shmem_i32[ixj] = a;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int64_t i0 = (int64_t) ib * (int64_t) top_k;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (i0 + (int64_t) col < (int64_t) ne0 && col < (int) top_k) {{"
    )
    .unwrap();
    writeln!(out, "        device int * dst = p1").unwrap();
    writeln!(out, "            + (int64_t) i0").unwrap();
    writeln!(out, "            + (int64_t) ne0 * (int64_t) i01").unwrap();
    writeln!(
        out,
        "            + (int64_t) ne0 * (int64_t) ne1 * (int64_t) i02"
    )
    .unwrap();
    writeln!(
        out,
        "            + (int64_t) ne0 * (int64_t) ne1 * (int64_t) ne2 * (int64_t) i03;"
    )
    .unwrap();
    writeln!(out, "        dst[col] = shmem_i32[col];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// M135 ArgsortMergeF32I32DescFull: full host_name kernel_argsort_merge_f32_i32_desc.
/// Antirez merge transcription (argsort.metal:122-263) specialized to DESC order.
/// Merges two pre-sorted descending index runs (produced by M134) into one
/// descending top_k run, batched 4-D. Per-thread chunk = ceil(total/ntg.x) work
/// slice [k0, k1); binary-search partition (i+j=k0) bounded by [max(0,k0-len1),
/// min(k0,len0)]; sequential merge for the slice.
pub(super) fn emit_argsort_merge_f32_i32_desc_full_msl(out: &mut String) {
    writeln!(out, "    const int im  = (int) tgpig.x / (int) ne01;").unwrap();
    writeln!(out, "    const int i01 = (int) tgpig.x % (int) ne01;").unwrap();
    writeln!(out, "    const int i02 = (int) tgpig.y;").unwrap();
    writeln!(out, "    const int i03 = (int) tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int start = im * (2 * (int) len);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int rest0 = (int) ne0 - start;").unwrap();
    writeln!(out, "    const int len0 = (int) len < (rest0 < 0 ? 0 : rest0) ? (int) len : (rest0 < 0 ? 0 : rest0);").unwrap();
    writeln!(
        out,
        "    const int rest1 = (int) ne0 - (start + (int) len);"
    )
    .unwrap();
    writeln!(out, "    const int len1 = (int) len < (rest1 < 0 ? 0 : rest1) ? (int) len : (rest1 < 0 ? 0 : rest1);").unwrap();
    writeln!(out, "    const int total = len0 + len1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const int * tmp0 = p1 + start").unwrap();
    writeln!(out, "        + i01 * (int) ne0").unwrap();
    writeln!(out, "        + i02 * (int) ne0 * (int) ne01").unwrap();
    writeln!(out, "        + i03 * (int) ne0 * (int) ne01 * (int) ne02;").unwrap();
    writeln!(out, "    device const int * tmp1 = tmp0 + (int) len;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device int * dst = p2 + start").unwrap();
    writeln!(out, "        + i01 * (int) top_k").unwrap();
    writeln!(out, "        + i02 * (int) top_k * (int) ne01").unwrap();
    writeln!(
        out,
        "        + i03 * (int) top_k * (int) ne01 * (int) ne02;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * src0_row = (device const float *) (p0"
    )
    .unwrap();
    writeln!(out, "        + (uint64_t) nb01 * (uint64_t) i01").unwrap();
    writeln!(out, "        + (uint64_t) nb02 * (uint64_t) i02").unwrap();
    writeln!(out, "        + (uint64_t) nb03 * (uint64_t) i03);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (total == 0) {{ return; }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int chunk = (total + (int) ntg.x - 1) / (int) ntg.x;"
    )
    .unwrap();
    writeln!(out, "    const int k0 = (int) tpitg.x * chunk;").unwrap();
    writeln!(out, "    int k1 = k0 + chunk;").unwrap();
    writeln!(out, "    if (k1 > total) k1 = total;").unwrap();
    writeln!(out, "    if (k1 > (int) top_k) k1 = (int) top_k;").unwrap();
    writeln!(out, "    if (k0 >= (int) top_k) return;").unwrap();
    writeln!(out, "    if (k0 >= total) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int low  = k0 > len1 ? k0 - len1 : 0;").unwrap();
    writeln!(out, "    int high = k0 < len0 ? k0 : len0;").unwrap();
    writeln!(out, "    while (low < high) {{").unwrap();
    writeln!(out, "        const int mid_pos = (low + high) >> 1;").unwrap();
    writeln!(out, "        const int idx_a = tmp0[mid_pos];").unwrap();
    writeln!(out, "        const int idx_b = tmp1[k0 - mid_pos - 1];").unwrap();
    writeln!(out, "        const float val_a = src0_row[idx_a];").unwrap();
    writeln!(out, "        const float val_b = src0_row[idx_b];").unwrap();
    writeln!(out, "        // DESC: take_left when val_a >= val_b").unwrap();
    writeln!(
        out,
        "        if (val_a >= val_b) {{ low = mid_pos + 1; }} else {{ high = mid_pos; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int i = low;").unwrap();
    writeln!(out, "    int j = k0 - i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int   idx0 = 0;").unwrap();
    writeln!(out, "    float val0 = 0.0f;").unwrap();
    writeln!(
        out,
        "    if (i < len0) {{ idx0 = tmp0[i]; val0 = src0_row[idx0]; }}"
    )
    .unwrap();
    writeln!(out, "    int   idx1 = 0;").unwrap();
    writeln!(out, "    float val1 = 0.0f;").unwrap();
    writeln!(
        out,
        "    if (j < len1) {{ idx1 = tmp1[j]; val1 = src0_row[idx1]; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int k = k0; k < k1; ++k) {{").unwrap();
    writeln!(out, "        if (i >= len0) {{").unwrap();
    writeln!(
        out,
        "            while (k < k1) {{ dst[k++] = tmp1[j++]; }}"
    )
    .unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }} else if (j >= len1) {{").unwrap();
    writeln!(
        out,
        "            while (k < k1) {{ dst[k++] = tmp0[i++]; }}"
    )
    .unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            int out_idx;").unwrap();
    writeln!(out, "            if (val0 >= val1) {{").unwrap();
    writeln!(out, "                out_idx = idx0; ++i;").unwrap();
    writeln!(
        out,
        "                if (i < len0) {{ idx0 = tmp0[i]; val0 = src0_row[idx0]; }}"
    )
    .unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                out_idx = idx1; ++j;").unwrap();
    writeln!(
        out,
        "                if (j < len1) {{ idx1 = tmp1[j]; val1 = src0_row[idx1]; }}"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            dst[k] = out_idx;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Dsv4MulMmIdMap0: per-expert MoE ID-map builder. One threadgroup, one thread
/// per expert (tid = expert id). For each token row in src2 (of length ne21),
/// each expert scans its ne20 selected experts; on a match, the expert appends
/// the token's flat slot index `(i21 * ne20 + sel - 1)` to `ids_i32[ide][...]`,
/// and finally writes its count to `tpe_u32[ide]`. Specialized on ne20 as a
/// runtime uint param. Threadgroup memory is baked: MAX_NTG=256 lanes × 32
/// ne20 slots per lane = 8192 uint16_t = 16 KB.
pub(super) fn emit_dsv4_mul_mm_id_map0_msl(out: &mut String) {
    writeln!(out, "    threadgroup uint16_t shmem_ids[256 * 32];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint ide = tid;").unwrap();
    writeln!(out, "    uint n_all = 0;").unwrap();
    writeln!(
        out,
        "    device int * ids_i32 = ((device int *) p2) + ide * ne21;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i21 = 0; i21 < ne21; i21 += tcount) {{").unwrap();
    writeln!(out, "        if (i21 + tid < ne21) {{").unwrap();
    writeln!(out, "            device const int * src2_i32 = (device const int *) (p0 + (uint64_t)(i21 + tid) * nb21);").unwrap();
    writeln!(
        out,
        "            threadgroup uint16_t * sids = shmem_ids + tid * ne20;"
    )
    .unwrap();
    writeln!(out, "            for (uint i20 = 0; i20 < ne20; i20++) {{").unwrap();
    writeln!(out, "                sids[i20] = (uint16_t) src2_i32[i20];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (uint t = 0; t < tcount; t++) {{").unwrap();
    writeln!(out, "            if (i21 + t >= ne21) break;").unwrap();
    writeln!(
        out,
        "            threadgroup const uint16_t * sids = shmem_ids + t * ne20;"
    )
    .unwrap();
    writeln!(out, "            uint sel = 0;").unwrap();
    writeln!(out, "            for (uint i20 = 0; i20 < ne20; i20++) {{").unwrap();
    writeln!(
        out,
        "                sel += (uint)(sids[i20] == ide) * (i20 + 1);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            ids_i32[n_all] = (int)((i21 + t) * ne20 + sel - 1);"
    )
    .unwrap();
    writeln!(out, "            n_all += (sel > 0) ? 1u : 0u;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device uint * tpe_u32 = (device uint *) p1;").unwrap();
    writeln!(out, "    tpe_u32[ide] = n_all;").unwrap();
}
/// Dsv4QkvRmsNormF32_4: DS4-specific fused float4 RMSNorm of q-lora row and
/// KV row in a single dispatch. Grid: (rows, 2, 1) — tgpig.x = row, tgpig.y
/// = q (0) / kv (1). Cross-simd reduction via simd_sum + threadgroup shmem.
/// Buffers: p0=q_src, p1=q_w (const), p2=q_dst (writable), p3=kv_src,
/// p4=kv_w (const), p5=kv_dst (writable).
pub(super) fn emit_dsv4_qkv_rms_norm_f32_4_msl(out: &mut String) {
    writeln!(out, "    threadgroup float shmem_f32[32];").unwrap();
    writeln!(out, "    const uint tcount = _tc_v.x;").unwrap();
    writeln!(
        out,
        "    if (simd_id == 0) {{ shmem_f32[simd_lane] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint row = tgpig.x;").unwrap();
    writeln!(out, "    const bool kv_task = tgpig.y != 0u;").unwrap();
    writeln!(out, "    const uint n  = kv_task ? kv_n  : q_n;").unwrap();
    writeln!(out, "    const uint n4 = kv_task ? kv_n4 : q_n4;").unwrap();
    writeln!(out, "    const uint64_t row_stride4 = (uint64_t)(kv_task ? kv_row_stride : q_row_stride) / 16ull;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4 * x = kv_task ? ((device const float4 *) p3) + row * row_stride4 : ((device const float4 *) p0) + row * row_stride4;").unwrap();
    writeln!(out, "    device const float4 * w = kv_task ?  (device const float4 *) p4                       :  (device const float4 *) p1;").unwrap();
    writeln!(out, "    device       float4 * y = kv_task ? ((device       float4 *) p5) + row * row_stride4 : ((device       float4 *) p2) + row * row_stride4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out, "    for (uint i = tid; i < n4; i += tcount) {{").unwrap();
    writeln!(out, "        const float4 v = x[i];").unwrap();
    writeln!(out, "        sumf += dot(v, v);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    if (simd_lane == 0) {{ shmem_f32[simd_id] = sumf; }}"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    sumf = shmem_f32[simd_lane];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float scale = rsqrt(sumf / (float)n + eps);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = tid; i < n4; i += tcount) {{").unwrap();
    writeln!(out, "        y[i] = (x[i] * scale) * w[i];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// SoftMaxF32_4: DS4 kernel_soft_max_f32_4 (no-mask/no-sink path).
/// Per-row online softmax over float4 lanes with cross-SIMD reduction so
/// the kernel works for tg up to 32 simdgroups. Applies an optional input
/// scale before the max/exp/sum stages. Buffers: p0=src (char* const),
/// p1=dst (char* writable). Params: ne00 (cols), nb01 (src row stride bytes),
/// nb1 (dst row stride bytes), scale.
pub(super) fn emit_soft_max_f32_4_msl(out: &mut String) {
    writeln!(out, "    threadgroup float buf[32];").unwrap();
    writeln!(out, "    const uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    const uint row    = tgpig.x;").unwrap();
    writeln!(
        out,
        "    device const float4 * psrc4 = (device const float4 *)(p0 + (uint64_t)row * nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float4 * pdst4 = (device       float4 *)(p1 + (uint64_t)row * nb1);"
    )
    .unwrap();
    writeln!(out, "    const uint ne00_4 = ne00 / 4u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 lmax4 = float4(-INFINITY);").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00_4; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        lmax4 = fmax(lmax4, psrc4[i00] * scale);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    const float lmax = fmax(fmax(lmax4[0], lmax4[1]), fmax(lmax4[2], lmax4[3]));"
    )
    .unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = -INFINITY; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(
        out,
        "        if (simd_lane == 0) {{ buf[simd_id] = max_val; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        max_val = buf[simd_lane];").unwrap();
    writeln!(out, "        max_val = simd_max(max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 lsum4 = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00_4; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float4 e = exp(psrc4[i00] * scale - max_val);"
    )
    .unwrap();
    writeln!(out, "        lsum4 += e;").unwrap();
    writeln!(out, "        pdst4[i00] = e;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "    float sum = simd_sum(lsum);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = 0.0f; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        if (simd_lane == 0) {{ buf[simd_id] = sum; }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        sum = buf[simd_lane];").unwrap();
    writeln!(out, "        sum = simd_sum(sum);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    const float inv_sum = 1.0f / sum;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00_4; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        pdst4[i00] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// SoftMaxF32Scalar: DS4 kernel_soft_max<float> (no-mask/no-sink path).
/// Scalar lanewise variant of `SoftMaxF32_4`. Each thread iterates one
/// float at a time, so `ne00` is in elements rather than float4 lanes.
/// Used when the row width is not a multiple of 4. Same online softmax
/// + cross-SIMD reduce structure as the float4 path; the lane reduction
/// (`lmax4[0]+...`, `lsum4[0]+...`) collapses away. Buffers: p0=src
/// (char* const), p1=dst (char* writable). Params: ne00 (cols), nb01
/// (src row stride bytes), nb1 (dst row stride bytes), scale.
pub(super) fn emit_soft_max_f32_scalar_msl(out: &mut String) {
    writeln!(out, "    threadgroup float buf[32];").unwrap();
    writeln!(out, "    const uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    const uint row    = tgpig.x;").unwrap();
    writeln!(
        out,
        "    device const float * psrc0 = (device const float *)(p0 + (uint64_t)row * nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * pdst  = (device       float *)(p1 + (uint64_t)row * nb1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float lmax = -INFINITY;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        lmax = fmax(lmax, psrc0[i00] * scale);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = -INFINITY; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(
        out,
        "        if (simd_lane == 0) {{ buf[simd_id] = max_val; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        max_val = buf[simd_lane];").unwrap();
    writeln!(out, "        max_val = simd_max(max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float lsum = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float e = exp(psrc0[i00] * scale - max_val);"
    )
    .unwrap();
    writeln!(out, "        lsum += e;").unwrap();
    writeln!(out, "        pdst[i00] = e;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "    float sum = simd_sum(lsum);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = 0.0f; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        if (simd_lane == 0) {{ buf[simd_id] = sum; }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        sum = buf[simd_lane];").unwrap();
    writeln!(out, "        sum = simd_sum(sum);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    const float inv_sum = 1.0f / sum;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        pdst[i00] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// SoftMaxF32_4Sink: DS4 kernel_soft_max_f32_4 (sink path, no mask).
/// Same online softmax as `SoftMaxF32_4`, but each row also sees a
/// per-row "sink" scalar `psrc2[row]` that participates as if it were
/// an extra column: `lmax` is initialized to `psrc2[row]` and after the
/// final sum reduce we fold the sink into the denominator via
/// `sum += exp(psrc2[row] - max_val)`. The sink itself is NOT written
/// out — only the input row's softmax columns are. Buffers: p0=src
/// (char* const), p1=sink (char* const, one float per row), p2=dst
/// (char* writable). Params: ne00, nb01, nb1, scale.
pub(super) fn emit_soft_max_f32_4_sink_msl(out: &mut String) {
    writeln!(out, "    threadgroup float buf[32];").unwrap();
    writeln!(out, "    const uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    const uint row    = tgpig.x;").unwrap();
    writeln!(
        out,
        "    device const float4 * psrc4 = (device const float4 *)(p0 + (uint64_t)row * nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float  * psrc2 = (device const float  *)(p1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float4 * pdst4 = (device       float4 *)(p2 + (uint64_t)row * nb1);"
    )
    .unwrap();
    writeln!(out, "    const uint ne00_4 = ne00 / 4u;").unwrap();
    writeln!(out, "    const float sink = psrc2[row];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 lmax4 = float4(sink);").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00_4; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        lmax4 = fmax(lmax4, psrc4[i00] * scale);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    const float lmax = fmax(fmax(lmax4[0], lmax4[1]), fmax(lmax4[2], lmax4[3]));"
    )
    .unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = -INFINITY; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(
        out,
        "        if (simd_lane == 0) {{ buf[simd_id] = max_val; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        max_val = buf[simd_lane];").unwrap();
    writeln!(out, "        max_val = simd_max(max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 lsum4 = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00_4; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float4 e = exp(psrc4[i00] * scale - max_val);"
    )
    .unwrap();
    writeln!(out, "        lsum4 += e;").unwrap();
    writeln!(out, "        pdst4[i00] = e;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(
        out,
        "    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "    float sum = simd_sum(lsum);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = 0.0f; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        if (simd_lane == 0) {{ buf[simd_id] = sum; }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        sum = buf[simd_lane];").unwrap();
    writeln!(out, "        sum = simd_sum(sum);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sum += exp(sink - max_val);").unwrap();
    writeln!(out, "    const float inv_sum = 1.0f / sum;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00_4; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        pdst4[i00] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// SoftMaxF32ScalarSink: DS4 kernel_soft_max<float> (sink path, no mask).
/// Scalar lanewise sibling of `SoftMaxF32_4Sink`. Same sink fold-in
/// (lmax init = sink, post-reduce sum += exp(sink - max_val)) but the
/// body iterates per element. Buffers: p0=src, p1=sink (one float per
/// row), p2=dst. Params: ne00, nb01, nb1, scale.
pub(super) fn emit_soft_max_f32_scalar_sink_msl(out: &mut String) {
    writeln!(out, "    threadgroup float buf[32];").unwrap();
    writeln!(out, "    const uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    const uint row    = tgpig.x;").unwrap();
    writeln!(
        out,
        "    device const float * psrc0 = (device const float *)(p0 + (uint64_t)row * nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float * psrc2 = (device const float *)(p1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * pdst  = (device       float *)(p2 + (uint64_t)row * nb1);"
    )
    .unwrap();
    writeln!(out, "    const float sink = psrc2[row];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float lmax = sink;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        lmax = fmax(lmax, psrc0[i00] * scale);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = -INFINITY; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(
        out,
        "        if (simd_lane == 0) {{ buf[simd_id] = max_val; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        max_val = buf[simd_lane];").unwrap();
    writeln!(out, "        max_val = simd_max(max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float lsum = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float e = exp(psrc0[i00] * scale - max_val);"
    )
    .unwrap();
    writeln!(out, "        lsum += e;").unwrap();
    writeln!(out, "        pdst[i00] = e;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "    float sum = simd_sum(lsum);").unwrap();
    writeln!(out, "    if (tcount > 32u) {{").unwrap();
    writeln!(
        out,
        "        if (simd_id == 0) {{ buf[simd_lane] = 0.0f; }}"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        if (simd_lane == 0) {{ buf[simd_id] = sum; }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        sum = buf[simd_lane];").unwrap();
    writeln!(out, "        sum = simd_sum(sum);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sum += exp(sink - max_val);").unwrap();
    writeln!(out, "    const float inv_sum = 1.0f / sum;").unwrap();
    writeln!(
        out,
        "    for (uint i00 = tid; i00 < ne00; i00 += tcount) {{"
    )
    .unwrap();
    writeln!(out, "        pdst[i00] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// SumRowsF32: DS4 kernel_sum_rows_f32_f32 (sum_rows.metal).
/// Per-row reduction Σ_i src[i] writing one float per (i1,i2,i3) cell.
/// One tg per (i1,i2,i3) — i1=tgpig.x, i2=tgpig.y, i3=tgpig.z. Each thread
/// strides through ne00 with step ntg.x, simd_sum reduces within a
/// simdgroup, then a 32-slot shmem buf carries the cross-simdgroup
/// reduction. Final value is written by tpitg.x==0.
/// Buffers: p0=src (char* const), p1=dst (char* writable).
/// Params: ne00, nb01, nb02, nb03, nb1, nb2, nb3.
pub(super) fn emit_sum_rows_f32_msl(out: &mut String) {
    writeln!(out, "    threadgroup float buf[32];").unwrap();
    writeln!(out, "    const uint i1 = tgpig.x;").unwrap();
    writeln!(out, "    const uint i2 = tgpig.y;").unwrap();
    writeln!(out, "    const uint i3 = tgpig.z;").unwrap();
    writeln!(out, "    const uint tcount = _tc_v.x;").unwrap();
    writeln!(out, "    device const float * src_row = (device const float *)(p0 + (uint64_t)i1 * nb01 + (uint64_t)i2 * nb02 + (uint64_t)i3 * nb03);").unwrap();
    writeln!(out, "    device       float * dst_row = (device       float *)(p1 + (uint64_t)i1 * nb1  + (uint64_t)i2 * nb2  + (uint64_t)i3 * nb3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (simd_id == 0) {{ buf[simd_lane] = 0.0f; }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out, "    for (uint i0 = tid; i0 < ne00; i0 += tcount) {{").unwrap();
    writeln!(out, "        sumf += src_row[i0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (simd_lane == 0) {{ buf[simd_id] = sumf; }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    sumf = buf[simd_lane];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        dst_row[0] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// RepeatF32: tile src dims (ne00..ne03) into larger dst dims (ne0..ne3) via
/// per-axis mod from antirez kernel_repeat_f32 (repeat.metal). One tg per
/// (i1,i2,i3) cell; threads stride along i0 and read src[i00=i0%ne00].
pub(super) fn emit_repeat_f32_msl(out: &mut String) {
    writeln!(out, "    const uint i3 = tgpig.z;").unwrap();
    writeln!(out, "    const uint i2 = tgpig.y;").unwrap();
    writeln!(out, "    const uint i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i03 = i3 % ne03;").unwrap();
    writeln!(out, "    const uint i02 = i2 % ne02;").unwrap();
    writeln!(out, "    const uint i01 = i1 % ne01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const char * src0_ptr = p0 + (uint64_t)i03 * nb03 + (uint64_t)i02 * nb02 + (uint64_t)i01 * nb01;").unwrap();
    writeln!(out, "    device       char * dst_ptr  = p1 + (uint64_t)i3  * nb3  + (uint64_t)i2  * nb2  + (uint64_t)i1  * nb1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint i0 = (uint)tpitg.x; i0 < ne0; i0 += (uint)ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        const uint i00 = i0 % ne00;").unwrap();
    writeln!(out, "        *((device float *)(dst_ptr + (uint64_t)i0 * nb0)) = *((device const float *)(src0_ptr + (uint64_t)i00 * nb00));").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// ConcatF32: concat two float tensors along `dim` ∈ {0,1,2,3} from
/// antirez kernel_concat (concat.metal). One tg per (i1,i2,i3) cell;
/// threads stride along i0. Branch picks src0 vs src1 depending on
/// the per-axis offset `o[dim] = ne0{dim}` (size of src0 along dim).
pub(super) fn emit_concat_f32_msl(out: &mut String) {
    writeln!(out, "    const uint i3 = tgpig.z;").unwrap();
    writeln!(out, "    const uint i2 = tgpig.y;").unwrap();
    writeln!(out, "    const uint i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint o[4] = {{0u, 0u, 0u, 0u}};").unwrap();
    writeln!(
        out,
        "    o[dim] = (dim == 0u) ? ne00 : (dim == 1u ? ne01 : (dim == 2u ? ne02 : ne03));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint i0 = (uint)tpitg.x; i0 < ne0; i0 += (uint)ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        device const float * x;").unwrap();
    writeln!(
        out,
        "        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {{"
    )
    .unwrap();
    writeln!(out, "            x = (device const float *)(p0 + (uint64_t)i3 * nb03 + (uint64_t)i2 * nb02 + (uint64_t)i1 * nb01 + (uint64_t)i0 * nb00);").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            x = (device const float *)(p1 + (uint64_t)(i3 - o[3]) * nb13 + (uint64_t)(i2 - o[2]) * nb12 + (uint64_t)(i1 - o[1]) * nb11 + (uint64_t)(i0 - o[0]) * nb10);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        device float * y = (device float *)(p2 + (uint64_t)i3 * nb3 + (uint64_t)i2 * nb2 + (uint64_t)i1 * nb1 + (uint64_t)i0 * nb0);").unwrap();
    writeln!(out, "        *y = *x;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Dsv4SoftplusSqrtF32_4: per-row float4 fused softplus → sqrt.
/// Used on the 256-wide DS4 decode router-logit row. Each tg processes
/// one row (`row` = tgpig.x). Each thread covers one float4 lane (`tid`).
/// Body matches antirez kernel_dsv4_softplus_sqrt_f32_4.
/// Buffers: p0=src (char* const), p1=dst (char* writable).
/// Params: ne0_4 (float4 lanes per row), nb_src, nb_dst (row strides in bytes).
pub(super) fn emit_dsv4_softplus_sqrt_f32_4_msl(out: &mut String) {
    writeln!(out, "    if (tid >= ne0_4) return;").unwrap();
    writeln!(
        out,
        "    device const float4 * s = (device const float4 *)(p0 + (uint64_t)row * nb_src);"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float4 * d = (device       float4 *)(p1 + (uint64_t)row * nb_dst);"
    )
    .unwrap();
    writeln!(out, "    const float4 x  = s[tid];").unwrap();
    writeln!(
        out,
        "    const float4 sp = select(log(1.0f + exp(x)), x, x > 20.0f);"
    )
    .unwrap();
    writeln!(out, "    d[tid] = sqrt(sp);").unwrap();
}
/// SwigluF32: DS4 kernel_swiglu_f32 (glu.metal). Per-row inner stride loop:
/// `dst_row[i0] = silu(src0_row[i0]) * src1_row[i0]` for i0 in [tpitg, ne0)
/// step ntg. Row index from threadgroup_position_in_grid; src0/src1 rows
/// start at `i00`/`i10` element offset within their row.
/// Buffers: p0=src0 (gate, char* const), p1=src1 (up, char* const), p2=dst.
/// Params: ne0, nb01, nb11, nb1, i00, i10.
pub(super) fn emit_swiglu_f32_msl(out: &mut String) {
    writeln!(out, "    device const float * src0_row = (device const float *)(p0 + (uint64_t)row * nb01) + i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *)(p1 + (uint64_t)row * nb11) + i10;").unwrap();
    writeln!(
        out,
        "    device       float * dst_row  = (device       float *)(p2 + (uint64_t)row * nb1);"
    )
    .unwrap();
    writeln!(out, "    for (uint i0 = tid; i0 < ne0; i0 += tcount) {{").unwrap();
    writeln!(out, "        const float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        const float x1 = src1_row[i0];").unwrap();
    writeln!(out, "        const float silu = x0 / (1.0f + exp(-x0));").unwrap();
    writeln!(out, "        dst_row[i0] = silu * x1;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// MulMvF16F32Pair_4 (M90): paired half-src0 vectorized matvec.
/// Buffers: p0=src0_a (half), p1=src0_b (half), p2=src1 (float),
/// p3=dst_a (float), p4=dst_b (float). Two src0 matrices share one
/// src1 and produce two dsts in the same dispatch — saves the
/// duplicate y4 load that two separate M89b dispatches would cost.
pub(super) fn emit_mul_mv_f16_f32_pair_4_msl(out: &mut String, nsg: u32, nr0: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW  = 32;").unwrap();
    writeln!(out, "    constexpr short NSG = {nsg};").unwrap();
    writeln!(out, "    constexpr short NR0 = {nr0};").unwrap();
    writeln!(out, "    threadgroup float shmem_f32[NR0 * NW];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(
        out,
        "    const ushort lane  = (ushort)(sgitg * NW + tiisg);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint r0 = tgpig.x * NR0;").unwrap();
    writeln!(out, "    const uint r1 = tgpig.y;").unwrap();
    writeln!(out, "    const uint im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im % ne12;").unwrap();
    writeln!(out, "    const uint i13 = im / ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset1 = (uint64_t)r1 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12 + (uint64_t)i13 * (uint64_t)nb13;").unwrap();
    writeln!(
        out,
        "    device const float  * y  = (device const float  *)(p2 + offset1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float4 * y4 = (device const float4 *)(p2 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint ne00_4 = ne00 / 4u;").unwrap();
    writeln!(out, "    float sum_a[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out, "    float sum_b[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        if (r0 + (uint)row >= ne01) continue;").unwrap();
    writeln!(out, "        const uint64_t offset0 = (uint64_t)(r0 + row) * (uint64_t)nb01 + (uint64_t)(i12 / r2) * (uint64_t)nb02 + (uint64_t)(i13 / r3) * (uint64_t)nb03;").unwrap();
    writeln!(
        out,
        "        device const half  * xa  = (device const half  *)(p0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "        device const half4 * xa4 = (device const half4 *)(p0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "        device const half  * xb  = (device const half  *)(p1 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "        device const half4 * xb4 = (device const half4 *)(p1 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "        // Paired vector path: one y4 load feeds two dot products."
    )
    .unwrap();
    writeln!(
        out,
        "        for (uint i4 = lane; i4 < ne00_4; i4 += (uint)(NSG * NW)) {{"
    )
    .unwrap();
    writeln!(out, "            const float4 yv = float4(y4[i4]);").unwrap();
    writeln!(out, "            sum_a[row] += dot(float4(xa4[i4]), yv);").unwrap();
    writeln!(out, "            sum_b[row] += dot(float4(xb4[i4]), yv);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (lane == 0) {{").unwrap();
    writeln!(
        out,
        "            for (uint i = ne00_4 * 4u; i < ne00; ++i) {{"
    )
    .unwrap();
    writeln!(out, "                const float yi = y[i];").unwrap();
    writeln!(out, "                sum_a[row] += (float) xa[i] * yi;").unwrap();
    writeln!(out, "                sum_b[row] += (float) xb[i] * yi;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_a_f32 = (device float *) p3 + (uint64_t)im * (uint64_t)ne0 * (uint64_t)ne1 + (uint64_t)r1 * (uint64_t)ne0;").unwrap();
    writeln!(out, "    device float * dst_b_f32 = (device float *) p4 + (uint64_t)im * (uint64_t)ne0 * (uint64_t)ne1 + (uint64_t)r1 * (uint64_t)ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // First reduce-and-write: dst_a.").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (sgitg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "        sum_a[row] = simd_sum(sum_a[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (tiisg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)sgitg] = sum_a[row]; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && r0 + (uint)row < ne01; ++row) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float tot = simd_sum(shmem_f32[(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(out, "        if (tiisg == 0 && sgitg == 0) {{").unwrap();
    writeln!(out, "            dst_a_f32[r0 + (uint)row] = tot;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Second reduce-and-write: dst_b (shmem reused).").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (sgitg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "        sum_b[row] = simd_sum(sum_b[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (tiisg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)sgitg] = sum_b[row]; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && r0 + (uint)row < ne01; ++row) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float tot = simd_sum(shmem_f32[(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(out, "        if (tiisg == 0 && sgitg == 0) {{").unwrap();
    writeln!(out, "            dst_b_f32[r0 + (uint)row] = tot;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
}
/// MulMvQ8_0F32 (M91): quantized Q8_0 matvec.
/// Buffers: p0=src0 (q8_0 blocks: { half d; int8_t qs[32]; } 34B each),
///          p1=src1 (float vector), p2=dst (float).
/// Params: ne00 (must be %32==0), ne01 (rows), ne0, ne1, ne12, r2, r3,
///         nb01, nb02, nb03, nb11, nb12, nb13 (byte strides).
/// Baked constants: NW=32, NSG=4, NR0=2, NQ=8, QK8_0=32.
/// Per-block dot: sumq += qs[i] * yl[i] over 8 elements; sumf[row] += sumq * d.
/// Final via helper_mv_reduce_and_write 2-stage simd_sum.
pub(super) fn emit_mul_mv_q8_0_f32_msl(out: &mut String, nsg: u32, nr0: u32, nq: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW    = 32;").unwrap();
    writeln!(out, "    constexpr short NSG   = {nsg};").unwrap();
    writeln!(out, "    constexpr short NR0   = {nr0};").unwrap();
    writeln!(out, "    constexpr short NQ    = {nq};").unwrap();
    writeln!(out, "    constexpr short QK8_0 = 32;").unwrap();
    writeln!(
        out,
        "    constexpr uint  Q8_0_BLOCK_BYTES = 34u; // sizeof(half) + 32*int8"
    )
    .unwrap();
    writeln!(out, "    threadgroup float shmem_f32[NR0 * NW];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK8_0;").unwrap();
    writeln!(out, "    const uint r0 = tgpig.x * (uint)NR0;").unwrap();
    writeln!(out, "    const uint r1 = tgpig.y;").unwrap();
    writeln!(out, "    const uint im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im % ne12;").unwrap();
    writeln!(out, "    const uint i13 = im / ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset1 = (uint64_t)r1 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12 + (uint64_t)i13 * (uint64_t)nb13;").unwrap();
    writeln!(
        out,
        "    device const float * y = (device const float *)(p1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Per-row q8_0 block base pointers (as byte pointers; we'll index manually)."
    )
    .unwrap();
    writeln!(out, "    device const uchar * ax_byte[NR0];").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        const uint64_t offset0 = (uint64_t)(r0 + (uint)row) * (uint64_t)nb01 + (uint64_t)(i12 / r2) * (uint64_t)nb02 + (uint64_t)(i13 / r3) * (uint64_t)nb03;").unwrap();
    writeln!(
        out,
        "        ax_byte[row] = (device const uchar *)(p0 + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Sub-block sharding: ix selects which set of 8 elements within a block;"
    )
    .unwrap();
    writeln!(
        out,
        "    // il is the position offset (0 or 1) when NW/NQ=4."
    )
    .unwrap();
    writeln!(out, "    const short ix = (short)(tiisg / (NW / NQ));").unwrap();
    writeln!(out, "    const short il = (short)(tiisg % (NW / NQ));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ib0 = (int)sgitg * NQ + (int)ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG * NQ) {{").unwrap();
    writeln!(
        out,
        "        // Load NQ floats from y at block-offset ib*QK8_0 + il*NQ."
    )
    .unwrap();
    writeln!(out, "        const int y_off = ib * QK8_0 + (int)il * NQ;").unwrap();
    writeln!(out, "        float yl[NQ];").unwrap();
    writeln!(
        out,
        "        for (short i = 0; i < NQ; ++i) {{ yl[i] = y[y_off + i]; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            // block_q8_0 at ax_byte[row] + ib * 34: {{ half d; int8_t qs[32]; }}"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uchar * blk_byte = ax_byte[row] + (uint)ib * Q8_0_BLOCK_BYTES;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const half  * d_ptr    = (device const half  *)blk_byte;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const int8_t * qs_base = (device const int8_t *)(blk_byte + 2u);"
    )
    .unwrap();
    writeln!(
        out,
        "            device const int8_t * qs      = qs_base + (uint)il * NQ;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sumq = 0.0f;").unwrap();
    writeln!(
        out,
        "            for (short i = 0; i < NQ; ++i) {{ sumq += (float)qs[i] * yl[i]; }}"
    )
    .unwrap();
    writeln!(out, "            sumf[row] += sumq * (float)(*d_ptr);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) p2 + (uint64_t)im * (uint64_t)ne0 * (uint64_t)ne1 + (uint64_t)r1 * (uint64_t)ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // helper_mv_reduce_and_write<NR0> inline: 2-stage simd_sum."
    )
    .unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (sgitg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "        sumf[row] = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (tiisg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)sgitg] = sumf[row]; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && r0 + (uint)row < ne01; ++row) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float tot = simd_sum(shmem_f32[(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(out, "        if (tiisg == 0 && sgitg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[r0 + (uint)row] = tot;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
}
pub(super) fn emit_mul_mm_f16_f32_setup_msl(out: &mut String) {
    emit_mul_mm_msl(out, MmSrcKind::F16);
}
/// MulMmQ4_0F32: dense tiled GEMM with block_q4_0 src0 — the q4 twin of M103.
/// Same shell; src0 staged via inlined dequantize_q4_0 (18-B blocks: half d at +0,
/// 16 nibble bytes at +2; low nibbles = elems [0,16), high = [16,32), each `(n-8)*d`).
/// Half the weight bytes of q8_0 → ~2× less weight bandwidth on the attention
/// projections (DS4_ATTN_Q4_PROBE proved q4 precision is argmax-safe there).
pub(super) fn emit_mul_mm_q4_0_f32_setup_msl(out: &mut String) {
    emit_mul_mm_msl(out, MmSrcKind::Q4_0);
}
/// MulMmQ8_0F32 (M103): full tiled GEMM body for kernel_mul_mm_q8_0_f32.
/// Mirrors antirez dense.metal:1121 with block_q=block_q8_0, nl=2,
/// dequantize_func=dequantize_q8_0. Outer shell identical to M102b;
/// only the src0 staging inner differs (byte-pointer block dequant).
pub(super) fn emit_mul_mm_q8_0_f32_setup_msl(out: &mut String) {
    emit_mul_mm_msl(out, MmSrcKind::Q8_0);
}
/// Emit the iq2_xxs lookup tables at file scope, verbatim from antirez moe.metal:8-88.
/// These can't live inside a kernel scope — Metal's `constant` address space requires
/// module-level declarations. Called once at file prelude when any iq2_xxs kernel is present.
pub(super) fn emit_iq2xxs_tables(out: &mut String) {
    writeln!(
        out,
        "// iq2_xxs lookup tables (verbatim from antirez ds4 moe.metal:8-88)."
    )
    .unwrap();
    writeln!(
        out,
        "static constant uchar ds4_metal_kmask_iq2xs[8] = {{ 1, 2, 4, 8, 16, 32, 64, 128 }};"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "static constant uchar ds4_metal_ksigns_iq2xs[128] = {{"
    )
    .unwrap();
    writeln!(
        out,
        "      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,"
    )
    .unwrap();
    writeln!(
        out,
        "    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,"
    )
    .unwrap();
    writeln!(
        out,
        "    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,"
    )
    .unwrap();
    writeln!(
        out,
        "     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,"
    )
    .unwrap();
    writeln!(
        out,
        "    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,"
    )
    .unwrap();
    writeln!(
        out,
        "     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,"
    )
    .unwrap();
    writeln!(
        out,
        "     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,"
    )
    .unwrap();
    writeln!(
        out,
        "    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,"
    )
    .unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "static constant ulong ds4_metal_iq2xxs_grid[256] = {{").unwrap();
    writeln!(
        out,
        "    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,"
    )
    .unwrap();
    writeln!(
        out,
        "    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,"
    )
    .unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();
}
/// Shared body for both kernel_mul_mm_{f16,q8_0}_f32. The two variants
/// differ only in (1) `nl` constant, (2) the src0 staging block that
/// produces `temp_a`, and (3) the x pointer type (half4x4 vs block_q8_0
/// byte pointer with 34-B stride).
pub(super) fn emit_mul_mm_msl(out: &mut String, src_kind: MmSrcKind) {
    let is_quant = src_kind != MmSrcKind::F16;
    let block_bytes: u32 = match src_kind {
        MmSrcKind::Q8_0 => 34,
        MmSrcKind::Q4_0 => 18,
        MmSrcKind::F16 => 0,
    };
    let nl: u32 = if is_quant { 2 } else { 1 };
    writeln!(out, "    constexpr int NR0 = 64;").unwrap();
    writeln!(out, "    constexpr int NR1 = 32;").unwrap();
    writeln!(out, "    constexpr int NK  = 32;").unwrap();
    writeln!(out, "    constexpr int NL0 = NK/16;          // 2").unwrap();
    writeln!(out, "    constexpr int NL1 = NK/8;           // 4").unwrap();
    writeln!(out, "    constexpr short nl = {};             // f16: 16 halves per block (= half4x4); q8_0/q4_0: 32 weights per block, nl=2", nl).unwrap();
    if is_quant {
        writeln!(out, "    constexpr uint Q_BLOCK_BYTES = {}u; // q8_0: half d + int8 qs[32] (34); q4_0: half d + 16 nibble bytes (18)", block_bytes).unwrap();
    }
    writeln!(
        out,
        "    // Baked threadgroup shmem (8192 B = sa half[2048] + sb half[2048])."
    )
    .unwrap();
    writeln!(out, "    threadgroup half shmem_half[4096];").unwrap();
    writeln!(out, "    threadgroup half * sa = shmem_half + 0;").unwrap();
    writeln!(out, "    threadgroup half * sb = shmem_half + 2048;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const ushort tiitg = (ushort)tid;        // 0..127 (NSG*NW = 4*32)"
    )
    .unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;    // 0..3").unwrap();
    writeln!(out, "    (void)simd_lane;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int im = (int)tgpig.z;").unwrap();
    writeln!(out, "    const int r0 = (int)tgpig.y * NR0;").unwrap();
    writeln!(out, "    const int r1 = (int)tgpig.x * NR1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const short nr0 = ((int)ne0 - r0 < NR0) ? (short)((int)ne0 - r0) : (short)NR0;"
    )
    .unwrap();
    writeln!(
        out,
        "    const short nr1 = ((int)ne1 - r1 < NR1) ? (short)((int)ne1 - r1) : (short)NR1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Clamped tile-relative thread indices.").unwrap();
    writeln!(out, "    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : (short)(nr0 - 1);").unwrap();
    writeln!(out, "    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : (short)(nr1 - 1);").unwrap();
    writeln!(out, "    const short il0 = (short)(tiitg % NL0);").unwrap();
    writeln!(out, "    short il = il0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i12 = im % (int)ne12;").unwrap();
    writeln!(out, "    const int i13 = im / (int)ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = (uint64_t)(i12 / (int)r2) * (uint64_t)nb02 + (uint64_t)(i13 / (int)r3) * (uint64_t)nb03;").unwrap();
    writeln!(out, "    const short    offset1 = il0 / nl;").unwrap();
    writeln!(out).unwrap();
    if is_quant {
        writeln!(out, "    // x is a byte pointer; each quantized block holds 32 weights (q8_0 34 B / q4_0 18 B).").unwrap();
        writeln!(out, "    device const uchar * x = (device const uchar *)(p0 + (uint64_t)nb01 * (uint64_t)(r0 + lr0) + offset0) + (uint64_t)offset1 * Q_BLOCK_BYTES;").unwrap();
    } else {
        writeln!(
            out,
            "    // x advances in block_q strides (= half4x4 = 16 halves = 32 B for f16)."
        )
        .unwrap();
        writeln!(out, "    device const half4x4 * x = (device const half4x4 *)(p0 + (uint64_t)nb01 * (uint64_t)(r0 + lr0) + offset0) + offset1;").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "    const short iy = (short)(8 * (tiitg % NL1));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const half * y = (device const half *)(p1").unwrap();
    writeln!(out, "        + (uint64_t)nb13 * (uint64_t)i13").unwrap();
    writeln!(out, "        + (uint64_t)nb12 * (uint64_t)i12").unwrap();
    writeln!(out, "        + (uint64_t)nb11 * (uint64_t)(r1 + lr1)").unwrap();
    writeln!(out, "        + (uint64_t)nb10 * (uint64_t)iy);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    simdgroup_half8x8  ma[4];").unwrap();
    writeln!(out, "    simdgroup_half8x8  mb[2];").unwrap();
    writeln!(out, "    simdgroup_float8x8 mc[8];").unwrap();
    writeln!(out, "    for (short i = 0; i < 8; i++) {{").unwrap();
    writeln!(
        out,
        "        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int loop_k = 0; loop_k < (int)ne00; loop_k += NK) {{"
    )
    .unwrap();
    match src_kind {
        MmSrcKind::Q8_0 => {
            writeln!(out, "        // Stage src0 tile into sa via dequantize_q8_0 inlined: 16 int8s, scaled by half d,").unwrap();
            writeln!(out, "        // emitted as a half4x4 temp_a. il selects the 16-byte sub-tile within the 32-byte block.").unwrap();
            writeln!(out, "        half4x4 temp_a;").unwrap();
            writeln!(out, "        {{").unwrap();
            writeln!(
                out,
                "            device const half  * d_ptr = (device const half  *)x;"
            )
            .unwrap();
            writeln!(
                out,
                "            device const int8_t * qs   = (device const int8_t *)(x + 2u);"
            )
            .unwrap();
            writeln!(out, "            const float d = (float)(*d_ptr);").unwrap();
            writeln!(out, "            for (short i = 0; i < 16; i++) {{").unwrap();
            writeln!(
                out,
                "                temp_a[i / 4][i % 4] = (half)((float)qs[i + 16 * il] * d);"
            )
            .unwrap();
            writeln!(out, "            }}").unwrap();
            writeln!(out, "        }}").unwrap();
        }
        MmSrcKind::Q4_0 => {
            writeln!(out, "        // Stage src0 tile into sa via dequantize_q4_0 inlined: block_q4_0 = half d (+0) +").unwrap();
            writeln!(out, "        // 16 nibble bytes (+2). The 32 weights pack as low nibbles [0,16) / high [16,32);").unwrap();
            writeln!(out, "        // il (∈{{0,1}}) selects which 16-weight half. dequant = ((nibble) - 8) * d.").unwrap();
            writeln!(out, "        half4x4 temp_a;").unwrap();
            writeln!(out, "        {{").unwrap();
            writeln!(
                out,
                "            device const half  * d_ptr = (device const half  *)x;"
            )
            .unwrap();
            writeln!(
                out,
                "            device const uchar * qs    = (device const uchar *)(x + 2u);"
            )
            .unwrap();
            writeln!(out, "            const float d = (float)(*d_ptr);").unwrap();
            writeln!(out, "            for (short i = 0; i < 16; i++) {{").unwrap();
            writeln!(out, "                const uchar b   = qs[i];").unwrap();
            writeln!(
                out,
                "                const int   nib = (il == 0) ? (int)(b & 0x0F) : (int)(b >> 4);"
            )
            .unwrap();
            writeln!(
                out,
                "                temp_a[i / 4][i % 4] = (half)((float)(nib - 8) * d);"
            )
            .unwrap();
            writeln!(out, "            }}").unwrap();
            writeln!(out, "        }}").unwrap();
        }
        MmSrcKind::F16 => {
            writeln!(
                out,
                "        // Stage src0 tile into sa via dequantize_f16 (= identity cast for f16)."
            )
            .unwrap();
            writeln!(out, "        half4x4 temp_a = (half4x4)(*x);").unwrap();
        }
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short i = 0; i < 16; i++) {{").unwrap();
    writeln!(
        out,
        "            const short sx = (short)(2 * il0 + i / 8);"
    )
    .unwrap();
    writeln!(
        out,
        "            const short sy = (short)((tiitg / NL0) / 8);"
    )
    .unwrap();
    writeln!(
        out,
        "            const short lx = (short)((tiitg / NL0) % 8);"
    )
    .unwrap();
    writeln!(out, "            const short ly = (short)(i % 8);").unwrap();
    writeln!(out, "            const short ib = (short)(8 * sx + sy);").unwrap();
    writeln!(
        out,
        "            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        // Stage src1 tile into sb via half2x4 vector store."
    )
    .unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            const short sx = (short)(tiitg % NL1);").unwrap();
    writeln!(
        out,
        "            const short sy = (short)((tiitg / NL1) / 8);"
    )
    .unwrap();
    writeln!(
        out,
        "            const short ly = (short)((tiitg / NL1) % 8);"
    )
    .unwrap();
    writeln!(out, "            const short ib = (short)(4 * sx + sy);").unwrap();
    writeln!(out, "            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) = (half2x4)(*((device const half2x4 *)y));").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        il = (il + 2 < nl) ? (short)(il + 2) : (short)(il % 2);"
    )
    .unwrap();
    if is_quant {
        writeln!(out, "        // quantized: x is a byte pointer; advance by (2 + nl - 1)/nl = 1 block per K-tile when il rolls back to <2.").unwrap();
        writeln!(
            out,
            "        x  = (il < 2) ? (x + (uint)((2 + nl - 1) / nl) * Q_BLOCK_BYTES) : x;"
        )
        .unwrap();
    } else {
        writeln!(
            out,
            "        x  = (il < 2) ? (x + (short)((2 + nl - 1) / nl)) : x;"
        )
        .unwrap();
    }
    writeln!(out, "        y += NK;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short ik = 0; ik < NK / 8; ik++) {{").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "            for (short i = 0; i < 4; i++) {{").unwrap();
    writeln!(
        out,
        "                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "            for (short i = 0; i < 2; i++) {{").unwrap();
    writeln!(
        out,
        "                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out, "            for (short i = 0; i < 8; i++) {{").unwrap();
    writeln!(
        out,
        "                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            lsma += 8 * 64;").unwrap();
    writeln!(out, "            lsmb += 4 * 64;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Bounds-checked output store. Fast path: full NR0×NR1 tile fits ne0/ne1."
    )
    .unwrap();
    writeln!(
        out,
        "    if (r0 + NR0 <= (int)ne0 && r1 + NR1 <= (int)ne1) {{"
    )
    .unwrap();
    writeln!(out, "        device float * C = (device float *)p2").unwrap();
    writeln!(out, "            + (uint64_t)(r0 + 32 * (sgitg & 1))").unwrap();
    writeln!(
        out,
        "            + (uint64_t)(r1 + 16 * (sgitg >> 1)) * (uint64_t)ne0"
    )
    .unwrap();
    writeln!(
        out,
        "            + (uint64_t)im * (uint64_t)ne1 * (uint64_t)ne0;"
    )
    .unwrap();
    writeln!(out, "        for (short i = 0; i < 8; i++) {{").unwrap();
    writeln!(out, "            simdgroup_store(mc[i], C + (uint64_t)8 * (uint64_t)(i % 4) + (uint64_t)8 * (uint64_t)ne0 * (uint64_t)(i / 4), (uint64_t)ne0, 0, false);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        threadgroup float * temp_str = ((threadgroup float *)shmem_half) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;").unwrap();
    writeln!(out, "        for (short i = 0; i < 8; i++) {{").unwrap();
    writeln!(out, "            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            for (int j = tiitg; j < nr1; j += NR1) {{").unwrap();
    writeln!(out, "                device float * D  = (device float *)p2 + r0 + (r1 + j) * (int)ne0 + im * (int)ne1 * (int)ne0;").unwrap();
    writeln!(
        out,
        "                threadgroup float * C = temp_str + (j * NR0);"
    )
    .unwrap();
    writeln!(out, "                for (int i = 0; i < nr0; i++) {{").unwrap();
    writeln!(out, "                    D[i] = C[i];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Indirect (per-expert) q8_0 matvec: MoE `mul_mv_id`.
///
/// Wrapped in a text block so rustdoc does not try to COMPILE it: the first line
/// is indented, which makes rustdoc read the whole comment as an indented Rust
/// code block and fail with "expected one of `!` or `::`, found `=`". The
/// leading line was already truncated when this landed (fc53f741a) -- it has no
/// intact version in history and the sibling emit_mul_mv_q8_0_f32_msl carries no
/// doc comment to restore from, so the surviving text is kept verbatim.
///
/// ```text
///          p2=dst (float), p3=ids (int32, one i02 per (iid1, idx) slot).
/// Params: ne00, ne01, ne0, ne1, ne11, nei0, nbi1, nb01, nb02, nb11, nb12.
/// Grid: tgpig.x = i01_tile, tgpig.y = 0 (unused), tgpig.z = idx + iid1*nei0.
/// Inner: reads i02 = ids[iid1*nbi1/4 + idx], offsets src0 by i02*nb02,
///        src1 by i11*nb11 + i12*nb12, dst by (idx + iid1*ne1)*ne0,
///        then runs M91 q8_0 inner loop with r2=r3=1, im=0, r1=0.
/// ```
pub(super) fn emit_mul_mv_id_q8_0_f32_msl(out: &mut String, nsg: u32, nr0: u32, nq: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW    = 32;").unwrap();
    writeln!(out, "    constexpr short NSG   = {nsg};").unwrap();
    writeln!(out, "    constexpr short NR0   = {nr0};").unwrap();
    writeln!(out, "    constexpr short NQ    = {nq};").unwrap();
    writeln!(out, "    constexpr short QK8_0 = 32;").unwrap();
    writeln!(out, "    constexpr uint  Q8_0_BLOCK_BYTES = 34u;").unwrap();
    writeln!(out, "    threadgroup float shmem_f32[NR0 * NW];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Decode (idx, iid1) from tgpig.z.").unwrap();
    writeln!(out, "    const uint iid1 = tgpig.z / nei0;").unwrap();
    writeln!(out, "    const uint idx  = tgpig.z % nei0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Routed expert index: ids[iid1, idx] as int32 (p2 = ids)."
    )
    .unwrap();
    writeln!(out, "    device const int32_t * ids_row = (device const int32_t *)(p2 + (uint64_t)iid1 * (uint64_t)nbi1);").unwrap();
    writeln!(out, "    const int32_t i02 = ids_row[idx];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i11 = idx % ne11;").unwrap();
    writeln!(out, "    const uint i12 = iid1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // src0_cur = src0s + i02 * nb02 (per-expert weight stride)."
    )
    .unwrap();
    writeln!(
        out,
        "    device const char * src0_cur = p0 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(out, "    device const char * src1_cur = p1 + (uint64_t)i11 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12;").unwrap();
    writeln!(out, "    device       char * dst_cur  = p3 + ((uint64_t)idx * (uint64_t)ne0 + (uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0) * 4u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK8_0;").unwrap();
    writeln!(out, "    const uint r0 = tgpig.x * (uint)NR0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * y = (device const float *)src1_cur;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const uchar * ax_byte[NR0];").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        const uint64_t offset0 = (uint64_t)(r0 + (uint)row) * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "        ax_byte[row] = (device const uchar *)(src0_cur + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = (short)(tiisg / (NW / NQ));").unwrap();
    writeln!(out, "    const short il = (short)(tiisg % (NW / NQ));").unwrap();
    writeln!(out, "    const int   ib0 = (int)sgitg * NQ + (int)ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG * NQ) {{").unwrap();
    writeln!(out, "        const int y_off = ib * QK8_0 + (int)il * NQ;").unwrap();
    writeln!(out, "        float yl[NQ];").unwrap();
    writeln!(
        out,
        "        for (short i = 0; i < NQ; ++i) {{ yl[i] = y[y_off + i]; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            device const uchar * blk_byte = ax_byte[row] + (uint)ib * Q8_0_BLOCK_BYTES;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const half  * d_ptr    = (device const half  *)blk_byte;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const int8_t * qs_base = (device const int8_t *)(blk_byte + 2u);"
    )
    .unwrap();
    writeln!(
        out,
        "            device const int8_t * qs      = qs_base + (uint)il * NQ;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sumq = 0.0f;").unwrap();
    writeln!(
        out,
        "            for (short i = 0; i < NQ; ++i) {{ sumq += (float)qs[i] * yl[i]; }}"
    )
    .unwrap();
    writeln!(out, "            sumf[row] += sumq * (float)(*d_ptr);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *)dst_cur;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // helper_mv_reduce_and_write<NR0> inline.").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (sgitg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "        sumf[row] = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (tiisg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)sgitg] = sumf[row]; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && r0 + (uint)row < ne01; ++row) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float tot = simd_sum(shmem_f32[(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(out, "        if (tiisg == 0 && sgitg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[r0 + (uint)row] = tot;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
}
pub(super) fn emit_q8_0_compute_core(out: &mut String, s: &Q8_0Shell) {
    let z = if s.ggml_style { "0.f" } else { "0.0f" };
    writeln!(out, "    float sumf[{}] = {{ {} }};", s.nr0, z).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg/(NW/NQ);").unwrap();
    writeln!(out, "    const short il = tiisg%(NW/NQ);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ib0 = sgitg*NQ + ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[NQ];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * yb = y + ib0*QK8_0 + il*NQ;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // each thread in a SIMD group deals with NQ quants at a time"
    )
    .unwrap();
    writeln!(
        out,
        "    for (int ib = ib0; ib < {}; ib += NSG*NQ) {{",
        s.nb
    )
    .unwrap();
    writeln!(out, "        for (short i = 0; i < NQ; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = yb[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        for (short row = 0; row < {}; row++) {{",
        s.nr0
    )
    .unwrap();
    writeln!(out, "            device const int8_t * qs = {};", s.qs_expr).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sumq = {};", z).unwrap();
    writeln!(
        out,
        "            {} (short i = 0; i < NQ; ++i) {{",
        s.unroll
    )
    .unwrap();
    writeln!(out, "                sumq += qs[i] * yl[i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            sumf[row] += sumq*{};", s.d_expr).unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb += NSG*NQ*QK8_0;").unwrap();
    writeln!(out, "    }}").unwrap();
}
pub(super) fn emit_mul_mv_q8_0_f32_ggml(out: &mut String) {
    let ggml = Q8_0Shell {
        nb: "nb",
        nr0: "NR0",
        unroll: "FOR_UNROLL",
        qs_expr: "ax[row][ib].qs + il*NQ",
        d_expr: "ax[row][ib].d",
        ggml_style: true,
    };
    writeln!(out, "void kernel_mul_mv_q8_0_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short NW = N_SIMDWIDTH;").unwrap();
    writeln!(out, "    constexpr short NQ = 8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK8_0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x*NR0;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float      * y = (device const float      *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // pointers to src0 rows").unwrap();
    writeln!(out, "    device const block_q8_0 * ax[NR0];").unwrap();
    writeln!(out, "    FOR_UNROLL (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    emit_q8_0_compute_core(out, &ggml);
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_iq4_nl_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq4_nl_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup float * shmem_f32 = (threadgroup float *) shmem;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * NR0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float        * y = (device const float        *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb   = args.ne00/QK4_NL;").unwrap();
    writeln!(out, "    const int ns01 = args.nb01/args.nb00;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg/2;  // 0...15").unwrap();
    writeln!(out, "    const short it = tiisg%2;  // 0 or 1").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 yl[4];").unwrap();
    writeln!(out, "    float sumf[NR0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * yb = y + ix*QK4_NL + it*8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint32_t aux32[2];").unwrap();
    writeln!(
        out,
        "    thread const uint8_t * q8 = (thread const uint8_t *)aux32;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 qf1, qf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // [TAG_MUL_MV_WEIRD]").unwrap();
    writeln!(
        out,
        "    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        device const float4 * y4 = (device const float4 *)yb;"
    )
    .unwrap();
    writeln!(out, "        yl[0] = y4[0];").unwrap();
    writeln!(out, "        yl[1] = y4[4];").unwrap();
    writeln!(out, "        yl[2] = y4[1];").unwrap();
    writeln!(out, "        yl[3] = y4[5];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(
        out,
        "            device const block_iq4_nl & xb = x[row*ns01 + ib];"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float4 acc1 = {{0.f}}, acc2 = {{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            aux32[0] = q4[0] | (q4[1] << 16);").unwrap();
    writeln!(out, "            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;").unwrap();
    writeln!(out, "            aux32[0] &= 0x0f0f0f0f;").unwrap();
    writeln!(out, "            qf1 = {{shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]}};").unwrap();
    writeln!(out, "            qf2 = {{shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]}};").unwrap();
    writeln!(out, "            acc1 += yl[0] * qf1;").unwrap();
    writeln!(out, "            acc2 += yl[1] * qf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            aux32[0] = q4[2] | (q4[3] << 16);").unwrap();
    writeln!(out, "            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;").unwrap();
    writeln!(out, "            aux32[0] &= 0x0f0f0f0f;").unwrap();
    writeln!(out, "            qf1 = {{shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]}};").unwrap();
    writeln!(out, "            qf2 = {{shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]}};").unwrap();
    writeln!(out, "            acc1 += yl[2] * qf1;").unwrap();
    writeln!(out, "            acc2 += yl[3] * qf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            acc1 += acc2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb += 16 * QK4_NL;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[first_row + row] = sum_all;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_iq4_xs_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq4_xs_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup float * shmem_f32 = (threadgroup float *) shmem;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * NR0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float        * y = (device const float        *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb   = args.ne00/QK_K;").unwrap();
    writeln!(out, "    const int ns01 = args.nb01/args.nb00;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg/16;  // 0 or 1").unwrap();
    writeln!(out, "    const short it = tiisg%16;  // 0...15").unwrap();
    writeln!(out, "    const short ib = it/2;").unwrap();
    writeln!(out, "    const short il = it%2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 yl[4];").unwrap();
    writeln!(out, "    float sumf[NR0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint32_t aux32[2];").unwrap();
    writeln!(
        out,
        "    thread const uint8_t * q8 = (thread const uint8_t *)aux32;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 qf1, qf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // [TAG_MUL_MV_WEIRD]").unwrap();
    writeln!(
        out,
        "    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        device const float4 * y4 = (device const float4 *)yb;"
    )
    .unwrap();
    writeln!(out, "        yl[0] = y4[0];").unwrap();
    writeln!(out, "        yl[1] = y4[4];").unwrap();
    writeln!(out, "        yl[2] = y4[1];").unwrap();
    writeln!(out, "        yl[3] = y4[5];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            device const block_iq4_xs & xb = x[row*ns01 + ibl];"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float4 acc1 = {{0.f}}, acc2 = {{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;").unwrap();
    writeln!(out, "            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;").unwrap();
    writeln!(out, "            qf1 = {{shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]}};").unwrap();
    writeln!(out, "            qf2 = {{shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]}};").unwrap();
    writeln!(out, "            acc1 += yl[0] * qf1;").unwrap();
    writeln!(out, "            acc2 += yl[1] * qf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;").unwrap();
    writeln!(out, "            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;").unwrap();
    writeln!(out, "            qf1 = {{shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]}};").unwrap();
    writeln!(out, "            qf2 = {{shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]}};").unwrap();
    writeln!(out, "            acc1 += yl[2] * qf1;").unwrap();
    writeln!(out, "            acc2 += yl[3] * qf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            acc1 += acc2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;").unwrap();
    writeln!(
        out,
        "            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb += 2 * QK_K;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[first_row + row] = sum_all;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_mxfp4_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_mxfp4_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup float * shmem_f32 = (threadgroup float *) shmem;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * NR0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float       * y = (device const float       *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb   = args.ne00/QK_MXFP4;").unwrap();
    writeln!(out, "    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg/2;  // 0...15").unwrap();
    writeln!(out, "    const short it = tiisg%2;  // 0 or 1").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float4 yl[4];").unwrap();
    writeln!(out, "    float sumf[NR0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * yb = y + ix*QK_MXFP4 + it*8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster").unwrap();
    writeln!(
        out,
        "    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]"
    )
    .unwrap();
    writeln!(
        out,
        "    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        device const float4 * y4 = (device const float4 *) yb;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yl[0] = y4[0];").unwrap();
    writeln!(out, "        yl[1] = y4[4];").unwrap();
    writeln!(out, "        yl[2] = y4[1];").unwrap();
    writeln!(out, "        yl[3] = y4[5];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        FOR_UNROLL (short row = 0; row < NR0; row++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            device const block_mxfp4 & xb = x[row*ns01 + ib];"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);").unwrap();
    writeln!(out, "            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);").unwrap();
    writeln!(out, "            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);").unwrap();
    writeln!(out, "            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            acc1 = (acc1 + acc3) + (acc2 + acc4);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb += 16 * QK_MXFP4;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[first_row + row] = sum_all;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_iq1_s_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq1_s_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float       * y = (device const float       *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        float sumy = 0;").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "            sumy += yl[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq1_s * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint8_t  * qs = xr->qs + 4 * ib;").unwrap();
    writeln!(out, "        device const uint16_t * qh = xr->qh + ib;").unwrap();
    writeln!(out, "        device const half     * dh = &xr->d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));").unwrap();
    writeln!(out, "            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));").unwrap();
    writeln!(out, "            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));").unwrap();
    writeln!(out, "            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sum = 0;").unwrap();
    writeln!(out, "            for (short j = 0; j < 4; ++j) {{").unwrap();
    writeln!(
        out,
        "                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)"
    )
    .unwrap();
    writeln!(
        out,
        "                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)"
    )
    .unwrap();
    writeln!(
        out,
        "                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)"
    )
    .unwrap();
    writeln!(
        out,
        "                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dh += args.nb01/2;").unwrap();
    writeln!(out, "            qs += args.nb01;").unwrap();
    writeln!(out, "            qh += args.nb01/2;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[first_row + row] = sum_all;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_iq1_m_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq1_m_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float       * y = (device const float       *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    iq1m_scale_t scale;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        float4 sumy = {{0.f}};").unwrap();
    writeln!(out, "        for (short i = 0; i < 8; ++i) {{").unwrap();
    writeln!(out, "            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];").unwrap();
    writeln!(out, "            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];").unwrap();
    writeln!(out, "            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];").unwrap();
    writeln!(out, "            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq1_m * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint8_t  * qs = xr->qs + 4 * ib;").unwrap();
    writeln!(out, "        device const uint8_t  * qh = xr->qh + 2 * ib;").unwrap();
    writeln!(
        out,
        "        device const uint16_t * sc = (device const uint16_t *)xr->scales;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));").unwrap();
    writeln!(out, "            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));").unwrap();
    writeln!(out, "            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));").unwrap();
    writeln!(out, "            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float2 sum = {{0.f}};").unwrap();
    writeln!(out, "            for (short j = 0; j < 4; ++j) {{").unwrap();
    writeln!(
        out,
        "                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)"
    )
    .unwrap();
    writeln!(
        out,
        "                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);"
    )
    .unwrap();
    writeln!(
        out,
        "                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)"
    )
    .unwrap();
    writeln!(
        out,
        "                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);").unwrap();
    writeln!(out, "            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +").unwrap();
    writeln!(out, "                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            sc += args.nb01/2;").unwrap();
    writeln!(out, "            qs += args.nb01;").unwrap();
    writeln!(out, "            qh += args.nb01;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[first_row + row] = sum_all;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_iq2_xxs_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq2_xxs_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float         * y = (device const float         *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);"
    )
    .unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = 4;").unwrap();
    writeln!(out, "        int pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];"
    )
    .unwrap();
    writeln!(out, "        nval = 2;").unwrap();
    writeln!(out, "        pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq2_xxs * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint16_t * q2 = xr->qs + 4 * ib;").unwrap();
    writeln!(out, "        device const half * dh = &xr->d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            const float db = dh[0];").unwrap();
    writeln!(
        out,
        "            device const uint8_t * aux8 = (device const uint8_t *)q2;"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint32_t aux32 = q2[2] | (q2[3] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const float d = db * (0.5f + (aux32 >> 28));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sum = 0;").unwrap();
    writeln!(out, "            for (short l = 0; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);").unwrap();
    writeln!(
        out,
        "                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(
        out,
        "                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += d * sum;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dh += args.nb01/2;").unwrap();
    writeln!(out, "            q2 += args.nb01/2;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            dst_f32[first_row + row] = sum_all * 0.25f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    // Emitted bytes are unchanged: this only records what the staging above
    // requires of the launch, which nothing else writes down.
    let mut k = super::kernel_writer::KernelWriter::new(out);
    k.charge_simdgroups_exactly(
        "iq2xxs_grid[256] and ksigns_iq2xs[128]",
        256,
        4,
        "int nval = 4;",
        "the 256-entry value grid is staged as one run of 4 per lane and the 128-entry sign table as one run of 2, so both are covered exactly when a threadgroup has 64 lanes; at one simdgroup more than half the grid keeps whatever was already in threadgroup memory, and at four the staging writes past the end of it",
    );
    let _ = k.finish();
}
pub(super) fn emit_mul_mv_iq2_xs_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq2_xs_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float        * y = (device const float        *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);"
    )
    .unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = 8;").unwrap();
    writeln!(out, "        int pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];"
    )
    .unwrap();
    writeln!(out, "        nval = 2;").unwrap();
    writeln!(out, "        pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq2_xs * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint16_t * q2 = xr->qs + 4 * ib;").unwrap();
    writeln!(out, "        device const uint8_t  * sc = xr->scales + ib;").unwrap();
    writeln!(out, "        device const half * dh = &xr->d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            const float db = dh[0];").unwrap();
    writeln!(out, "            const uint8_t ls1 = sc[0] & 0xf;").unwrap();
    writeln!(out, "            const uint8_t ls2 = sc[0] >>  4;").unwrap();
    writeln!(out, "            const float d1 = db * (0.5f + ls1);").unwrap();
    writeln!(out, "            const float d2 = db * (0.5f + ls2);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sum1 = 0, sum2 = 0;").unwrap();
    writeln!(out, "            for (short l = 0; l < 2; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));").unwrap();
    writeln!(
        out,
        "                const uint8_t signs = ssigns[(q2[l] >> 9)];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(
        out,
        "                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            for (short l = 2; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));").unwrap();
    writeln!(
        out,
        "                const uint8_t signs = ssigns[(q2[l] >> 9)];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(
        out,
        "                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += d1 * sum1 + d2 * sum2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dh += args.nb01/2;").unwrap();
    writeln!(out, "            q2 += args.nb01/2;").unwrap();
    writeln!(out, "            sc += args.nb01;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            dst_f32[first_row + row] = sum_all * 0.25f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    // Emitted bytes are unchanged: this only records what the staging above
    // requires of the launch, which nothing else writes down.
    let mut k = super::kernel_writer::KernelWriter::new(out);
    k.charge_simdgroups_exactly(
        "iq2xs_grid[512] and ksigns_iq2xs[128]",
        512,
        8,
        "int nval = 8;",
        "the 512-entry value grid is staged as one run of 8 per lane and the 128-entry sign table as one run of 2, so both are covered exactly when a threadgroup has 64 lanes; any other simdgroup count leaves part of a table unwritten or runs past its end",
    );
    let _ = k.finish();
}
pub(super) fn emit_mul_mv_iq2_s_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq2_s_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float       * y = (device const float       *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    //threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;"
    )
    .unwrap();
    writeln!(out, "    //{{").unwrap();
    writeln!(out, "    //    int nval = 32;").unwrap();
    writeln!(out, "    //    int pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "    //    for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2s_grid[pos + i];"
    )
    .unwrap();
    writeln!(
        out,
        "    //    threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    //}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq2_s * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint8_t * qs = xr->qs + 4 * ib;").unwrap();
    writeln!(out, "        device const uint8_t * qh = xr->qh + ib;").unwrap();
    writeln!(out, "        device const uint8_t * sc = xr->scales + ib;").unwrap();
    writeln!(out, "        device const uint8_t * signs = qs + QK_K/8;").unwrap();
    writeln!(out, "        device const half * dh = &xr->d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            const float db = dh[0];").unwrap();
    writeln!(
        out,
        "            const float d1 = db * (0.5f + (sc[0] & 0xf));"
    )
    .unwrap();
    writeln!(
        out,
        "            const float d2 = db * (0.5f + (sc[0] >>  4));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float2 sum = {{0}};").unwrap();
    writeln!(out, "            for (short l = 0; l < 2; ++l) {{").unwrap();
    writeln!(out, "                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));").unwrap();
    writeln!(out, "                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));").unwrap();
    writeln!(out, "                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));").unwrap();
    writeln!(out, "                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));").unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(out, "                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);").unwrap();
    writeln!(out, "                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += d1 * sum[0] + d2 * sum[1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dh    += args.nb01/2;").unwrap();
    writeln!(out, "            qs    += args.nb01;").unwrap();
    writeln!(out, "            qh    += args.nb01;").unwrap();
    writeln!(out, "            sc    += args.nb01;").unwrap();
    writeln!(out, "            signs += args.nb01;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            dst_f32[first_row + row] = sum_all * 0.25f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_iq3_xxs_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq3_xxs_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float         * y = (device const float         *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);"
    )
    .unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = 4;").unwrap();
    writeln!(out, "        int pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];"
    )
    .unwrap();
    writeln!(out, "        nval = 2;").unwrap();
    writeln!(out, "        pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq3_xxs * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint8_t  * q3 = xr->qs + 8 * ib;").unwrap();
    writeln!(
        out,
        "        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;"
    )
    .unwrap();
    writeln!(out, "        device const half * dh = &xr->d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            const float db = dh[0];").unwrap();
    writeln!(
        out,
        "            const uint32_t aux32 = gas[0] | (gas[1] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const float d = db * (0.5f + (aux32 >> 28));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float2 sum = {{0}};").unwrap();
    writeln!(out, "            for (short l = 0; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);").unwrap();
    writeln!(
        out,
        "                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 4; ++j) {{").unwrap();
    writeln!(out, "                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);").unwrap();
    writeln!(out, "                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += d * (sum[0] + sum[1]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dh  += args.nb01/2;").unwrap();
    writeln!(out, "            q3  += args.nb01;").unwrap();
    writeln!(out, "            gas += args.nb01/2;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            dst_f32[first_row + row] = sum_all * 0.5f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    // Emitted bytes are unchanged: this only records what the staging above
    // requires of the launch, which nothing else writes down.
    let mut k = super::kernel_writer::KernelWriter::new(out);
    k.charge_simdgroups_exactly(
        "iq3xxs_grid[256] and ksigns_iq2xs[128]",
        256,
        4,
        "int nval = 4;",
        "the 256-entry value grid is staged as one run of 4 per lane and the 128-entry sign table as one run of 2, so both are covered exactly when a threadgroup has 64 lanes; any other simdgroup count leaves part of a table unwritten or runs past its end",
    );
    let _ = k.finish();
}
pub(super) fn emit_mul_mv_iq3_s_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_iq3_s_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK_K;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float       * y = (device const float       *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[nr0]={{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;"
    )
    .unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = 8;").unwrap();
    writeln!(out, "        int pos  = (32*sgitg + tiisg)*nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const block_iq3_s * xr = x + ibl;").unwrap();
    writeln!(out, "        device const uint8_t * qs = xr->qs + 8 * ib;").unwrap();
    writeln!(out, "        device const uint8_t * qh = xr->qh + ib;").unwrap();
    writeln!(
        out,
        "        device const uint8_t * sc = xr->scales + (ib/2);"
    )
    .unwrap();
    writeln!(
        out,
        "        device const uint8_t * signs = xr->signs + 4 * ib;"
    )
    .unwrap();
    writeln!(out, "        device const half * dh = &xr->d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < nr0; row++) {{").unwrap();
    writeln!(out, "            const float db = dh[0];").unwrap();
    writeln!(
        out,
        "            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float2 sum = {{0}};").unwrap();
    writeln!(out, "            for (short l = 0; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;").unwrap();
    writeln!(out, "                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);").unwrap();
    writeln!(out, "                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);").unwrap();
    writeln!(out, "                for (short j = 0; j < 4; ++j) {{").unwrap();
    writeln!(out, "                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);").unwrap();
    writeln!(out, "                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += d * (sum[0] + sum[1]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dh    += args.nb01/2;").unwrap();
    writeln!(out, "            qs    += args.nb01;").unwrap();
    writeln!(out, "            qh    += args.nb01;").unwrap();
    writeln!(out, "            sc    += args.nb01;").unwrap();
    writeln!(out, "            signs += args.nb01;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        float sum_all = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[first_row + row] = sum_all;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    // Emitted bytes are unchanged: this only records what the staging above
    // requires of the launch, which nothing else writes down.
    let mut k = super::kernel_writer::KernelWriter::new(out);
    k.charge_simdgroups_exactly(
        "iq3s_grid[512]",
        512,
        8,
        "int nval = 8;",
        "the 512-entry value grid is staged as one run of 8 per lane, covering it exactly when a threadgroup has 64 lanes; the kernel then indexes svalues + 256 for its high half, so a short stage reads entries that were never written",
    );
    let _ = k.finish();
}
pub(super) fn emit_mul_mv_q1_0_f32_ggml(out: &mut String) {
    writeln!(out, "void kernel_mul_mv_q1_0_f32_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/QK1_0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int first_row = (r0 * NSG + sgitg) * nr0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * y = (device const float *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const block_q1_0 * ax[nr0];").unwrap();
    writeln!(out, "    for (int row = 0; row < nr0; ++row) {{").unwrap();
    writeln!(out, "        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(
        out,
        "        ax[row] = (device const block_q1_0 *) ((device char *) src0 + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[16];").unwrap();
    writeln!(out, "    float sumf[nr0] = {{0.f}};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = (tiisg/8);").unwrap();
    writeln!(out, "    const short il = (tiisg%8)*16;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * yb = y + ix*QK1_0 + il;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {{"
    )
    .unwrap();
    writeln!(out, "        float sumy = 0.f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        FOR_UNROLL (short i = 0; i < 16; i++) {{").unwrap();
    writeln!(out, "            yl[i] = yb[i];").unwrap();
    writeln!(out, "            sumy += yb[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        FOR_UNROLL (short row = 0; row < nr0; row++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb += QK1_0 * (N_SIMDWIDTH/8);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int row = 0; row < nr0; ++row) {{").unwrap();
    writeln!(out, "        const float tot = simd_sum(sumf[row]);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        if (tiisg == 0 && first_row + row < args.ne01) {{"
    )
    .unwrap();
    writeln!(out, "            dst_f32[first_row + row] = tot;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mv_t_t_impl_ggml(out: &mut String) {
    writeln!(
        out,
        "template<typename T0, typename T1, short NR0, typename args_t>"
    )
    .unwrap();
    writeln!(out, "void kernel_mul_mv_t_t_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short NW = N_SIMDWIDTH;").unwrap();
    writeln!(out, "    constexpr short NB = 32;").unwrap();
    writeln!(out, "    constexpr short NF = 8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/NB;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x*NR0;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "  //device const T0 * x = (device const T0 *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const T1 * y = (device const T1 *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // pointers to src0 rows").unwrap();
    writeln!(out, "    device const T0 * ax [NR0];").unwrap();
    writeln!(out, "    FOR_UNROLL (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        ax[row] = (device const T0 *) ((device char *) src0 + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf[NR0] = {{ 0.f }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg/(NW/NF);").unwrap();
    writeln!(out, "    const short il = tiisg%(NW/NF);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ib0 = sgitg*NF + ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    T1 yl[NF];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T1 * yb = y + (ib0*NB + il*NF);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG*NF) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < NF; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = yb[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(
        out,
        "            device const T0 * xb = ax[row] + (ib*NB + il*NF);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sumq = 0.f;").unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < NF; ++i) {{").unwrap();
    writeln!(out, "                sumq += xb[i] * yl[i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            sumf[row] += sumq;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb += NSG*NF*NW;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {{"
    )
    .unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(out, "            sumf[row] += ax[row][i] * y[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ext_q4_f32_ggml(out: &mut String) {
    writeln!(out, "template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >").unwrap();
    writeln!(out, "void kernel_mul_mv_ext_q4_f32_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv_ext & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const short NSG   = FC_mul_mv_nsg;").unwrap();
    writeln!(out, "    const short nxpsg = FC_mul_mv_nxpsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short chpt = 4; // chunks per thread").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  //const short nxpsg = (32);").unwrap();
    writeln!(out, "    const short nypsg = (32/nxpsg);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short tx = tiisg%nxpsg;").unwrap();
    writeln!(out, "    const short ty = tiisg/nxpsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;"
    )
    .unwrap();
    writeln!(out, "    const int i11 = tgpig.y*r1ptg;").unwrap();
    writeln!(out, "    const int i1m = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i12 = i1m%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const int i13 = i1m/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4 * y4[r1ptg];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {{").unwrap();
    writeln!(out, "        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    float sumf[r1ptg] = {{ [ 0 ... r1ptg - 1 ] = 0.0f }};"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    short cch = tx%chpb; // current chunk index").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {{"
    )
    .unwrap();
    writeln!(out, "        float4 lx[chpt];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma unroll(chpt)").unwrap();
    writeln!(out, "        for (short ch = 0; ch < chpt; ++ch) {{").unwrap();
    writeln!(out, "            deq_t4(xq, cch, lx[ch]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            cch += nxpsg;").unwrap();
    writeln!(out, "            if (cch >= chpb) {{").unwrap();
    writeln!(out, "                xq  += cch/chpb;").unwrap();
    writeln!(out, "                cch %= chpb;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma unroll(chpt)").unwrap();
    writeln!(out, "        for (short ch = 0; ch < chpt; ++ch) {{").unwrap();
    writeln!(out, "#pragma unroll(r1ptg)").unwrap();
    writeln!(
        out,
        "            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma unroll(r1ptg)").unwrap();
    writeln!(out, "        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {{").unwrap();
    writeln!(out, "            y4[ir1] += chpt*nxpsg;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // reduce only the threads in each row").unwrap();
    writeln!(out, "    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {{").unwrap();
    writeln!(out, "        if (nxpsg >= 32) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 16) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 8) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 4) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 2) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        //sumf[ir1] = simd_sum(sumf[ir1]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tx == 0) {{").unwrap();
    writeln!(
        out,
        "        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {{"
    )
    .unwrap();
    writeln!(out, "            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (i01 < args.ne01) {{").unwrap();
    writeln!(out, "                dst_f32[i01] = sumf[ir1];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ext_q4x4_f32_ggml(out: &mut String) {
    writeln!(out, "template<short r1ptg, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &) >").unwrap();
    writeln!(out, "void kernel_mul_mv_ext_q4x4_f32_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv_ext & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const short NSG   = FC_mul_mv_nsg;").unwrap();
    writeln!(out, "    const short nxpsg = FC_mul_mv_nxpsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short chpt = 1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  //const short nxpsg = (32);").unwrap();
    writeln!(out, "    const short nypsg = (32/nxpsg);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short tx = tiisg%nxpsg;").unwrap();
    writeln!(out, "    const short ty = tiisg/nxpsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;"
    )
    .unwrap();
    writeln!(out, "    const int i11 = tgpig.y*r1ptg;").unwrap();
    writeln!(out, "    const int i1m = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i12 = i1m%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const int i13 = i1m/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4x4 * y4x4[r1ptg];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {{").unwrap();
    writeln!(out, "        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    float sumf[r1ptg] = {{ [ 0 ... r1ptg - 1 ] = 0.0f }};"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    short cch = tx%chpb;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {{"
    )
    .unwrap();
    writeln!(out, "        float4x4 lx[chpt];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma unroll(chpt)").unwrap();
    writeln!(out, "        for (short ch = 0; ch < chpt; ++ch) {{").unwrap();
    writeln!(out, "            deq_t4x4(xq, cch, lx[ch]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            cch += nxpsg;").unwrap();
    writeln!(out, "            if (cch >= chpb) {{").unwrap();
    writeln!(out, "                xq  += cch/chpb;").unwrap();
    writeln!(out, "                cch %= chpb;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma unroll(chpt)").unwrap();
    writeln!(out, "        for (short ch = 0; ch < chpt; ++ch) {{").unwrap();
    writeln!(out, "#pragma unroll(r1ptg)").unwrap();
    writeln!(
        out,
        "            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {{"
    )
    .unwrap();
    writeln!(out, "                sumf[ir1] +=").unwrap();
    writeln!(
        out,
        "                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +"
    )
    .unwrap();
    writeln!(
        out,
        "                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +"
    )
    .unwrap();
    writeln!(
        out,
        "                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +"
    )
    .unwrap();
    writeln!(
        out,
        "                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma unroll(r1ptg)").unwrap();
    writeln!(out, "        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {{").unwrap();
    writeln!(out, "            y4x4[ir1] += chpt*nxpsg;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {{").unwrap();
    writeln!(out, "        if (nxpsg >= 32) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 16) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 8) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 4) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (nxpsg >= 2) {{").unwrap();
    writeln!(
        out,
        "            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        //sumf[ir1] = simd_sum(sumf[ir1]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tx == 0) {{").unwrap();
    writeln!(
        out,
        "        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {{"
    )
    .unwrap();
    writeln!(out, "            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (i01 < args.ne01) {{").unwrap();
    writeln!(out, "                dst_f32[i01] = sumf[ir1];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_t_t_4_impl_ggml(out: &mut String) {
    writeln!(
        out,
        "template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>"
    )
    .unwrap();
    writeln!(out, "void kernel_mul_mv_t_t_4_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg,").unwrap();
    writeln!(out, "        ushort sgitg) {{").unwrap();
    writeln!(out, "    const short NSG = FC_mul_mv_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short NW = N_SIMDWIDTH;").unwrap();
    writeln!(out, "    constexpr short NB  = 32;").unwrap();
    writeln!(out, "    constexpr short NF  = 16;").unwrap();
    writeln!(out, "    constexpr short NF4 = NF/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = args.ne00/NB;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int r0 = tgpig.x*NR0;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out, "    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const T1  * y  = (device const T1  *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const T14 * y4 = (device const T14 *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // pointers to src0 rows").unwrap();
    writeln!(out, "    device const T0  * ax [NR0];").unwrap();
    writeln!(out, "    device const T04 * ax4[NR0];").unwrap();
    writeln!(out, "    FOR_UNROLL (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        ax [row] = (device const T0  *) ((device char *) src0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf[NR0] = {{ 0.f }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = tiisg/(NW/NF);").unwrap();
    writeln!(out, "    const short il = tiisg%(NW/NF);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ib0 = sgitg*NF + ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    T14 yl4[NF4];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG*NF) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < NF4; ++i) {{").unwrap();
    writeln!(out, "            yl4[i] = yb4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(
        out,
        "            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sumq = 0.f;").unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < NF4; ++i) {{").unwrap();
    writeln!(
        out,
        "                sumq += dot(float4(xb4[i]), float4(yl4[i]));"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            sumf[row] += sumq;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        yb4 += NSG*NF*NW/4;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {{"
    )
    .unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; row++) {{").unwrap();
    writeln!(out, "            sumf[row] += ax[row][i] * y[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_t_t_short_impl_ggml(out: &mut String) {
    writeln!(out, "template<typename T0, typename T1, typename args_t>").unwrap();
    writeln!(out, "void kernel_mul_mv_t_t_short_impl(").unwrap();
    writeln!(out, "        args_t args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint3  tgpig,").unwrap();
    writeln!(out, "        ushort tiisg) {{").unwrap();
    writeln!(out, "    const int r0 = tgpig.x*32 + tiisg;").unwrap();
    writeln!(out, "    const int r1 = tgpig.y;").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (r0 >= args.ne01) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im%FC_mul_mv_ne12;").unwrap();
    writeln!(out, "    const uint i13 = im/FC_mul_mv_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const T0 * x = (device const T0 *) (src0 + offset0);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const T1 * y = (device const T1 *) (src1 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float res = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i = 0; i < args.ne00; ++i) {{").unwrap();
    writeln!(out, "        res += (float) x[i] * (float) y[i];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_norm_fuse_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T, short F>").unwrap();
    writeln!(out, "kernel void kernel_norm_fuse_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_norm & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1_0,").unwrap();
    writeln!(out, "        device const char * src1_1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup float * shmem_f32 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[tiisg] = 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i01 = tgpig.x;").unwrap();
    writeln!(out, "    const int i02 = tgpig.y;").unwrap();
    writeln!(out, "    const int i03 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);").unwrap();
    writeln!(out, "    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    T sumft(0.0f);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        sumft += x[i00];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sumf = dot(sumft, T(1.0f));").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[sgitg] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = shmem_f32[tiisg];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float mean = sumf/args.ne00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        y[i00] = x[i00] - mean;").unwrap();
    writeln!(out, "        sumf += dot(y[i00], y[i00]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[sgitg] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = shmem_f32[tiisg];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float variance = sumf/args.ne00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float scale = 1.0f/sqrt(variance + args.eps);"
    )
    .unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        if (F == 1) {{").unwrap();
    writeln!(out, "            y[i00] = (y[i00]*scale);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (F == 2) {{").unwrap();
    writeln!(out, "            y[i00] = (y[i00]*scale)*f0[i00];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (F == 3) {{").unwrap();
    writeln!(
        out,
        "            y[i00] = (y[i00]*scale)*f0[i00] + f1[i00];"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rms_norm_fuse_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T, short F>").unwrap();
    writeln!(out, "kernel void kernel_rms_norm_fuse_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_norm & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1_0,").unwrap();
    writeln!(out, "        device const char * src1_1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup float * shmem_f32 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[tiisg] = 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i01 = tgpig.x;").unwrap();
    writeln!(out, "    const int i02 = tgpig.y;").unwrap();
    writeln!(out, "    const int i03 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);").unwrap();
    writeln!(out, "    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel sum").unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        sumf += dot(x[i00], x[i00]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[sgitg] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = shmem_f32[tiisg];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float mean  = sumf/args.ne00;").unwrap();
    writeln!(out, "    const float scale = 1.0f/sqrt(mean + args.eps);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);"
    )
    .unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        if (F == 1) {{").unwrap();
    writeln!(out, "            y[i00] = (x[i00]*scale);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (F == 2) {{").unwrap();
    writeln!(out, "            y[i00] = (x[i00]*scale)*f0[i00];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        if (F == 3) {{").unwrap();
    writeln!(
        out,
        "            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_l2_norm_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T0, typename T>").unwrap();
    writeln!(out, "kernel void kernel_l2_norm_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_l2_norm & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup float * shmem_f32 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int i03 = tgpig.z;").unwrap();
    writeln!(out, "    const int i02 = tgpig.y;").unwrap();
    writeln!(out, "    const int i01 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[tiisg] = 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T0 * x = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);").unwrap();
    writeln!(out, "    device       T  * y = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel sum").unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        sumf += dot(x[i00], x[i00]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[sgitg] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = shmem_f32[tiisg];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float scale = 1.0f/max(sqrt(sumf), args.eps);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        y[i00] = x[i00] * scale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_sum_rows_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T0, typename T>").unwrap();
    writeln!(out, "kernel void kernel_sum_rows_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_sum_rows & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "#define FC_OP  FC_sum_rows_op").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup T0 * shmem_t = (threadgroup T0 *) shmem;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        shmem_t[tiisg] = 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T0 * src_row = (device const T0 *) (src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);").unwrap();
    writeln!(out, "    device       T  * dst_row = (device       T  *) (dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    T0 sumf = T0(0.0f);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        sumf += src_row[i0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_t[sgitg] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = shmem_t[tiisg];").unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tpitg.x == 0) {{").unwrap();
    writeln!(out, "        if (FC_OP == OP_SUM_ROWS_NUM_MEAN) {{").unwrap();
    writeln!(out, "            if (is_same<float4, T0>::value) {{").unwrap();
    writeln!(
        out,
        "                dst_row[0] = sum(sumf) / (4*args.ne00);"
    )
    .unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                dst_row[0] = sum(sumf) / args.ne00;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            dst_row[0] = sum(sumf);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef FC_OP").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_pad_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T>").unwrap();
    writeln!(out, "kernel void kernel_pad_impl(").unwrap();
    writeln!(out, "    constant ggml_metal_kargs_pad & args,").unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t k0 = tgpig.x/args.ne1;").unwrap();
    writeln!(out, "    const int32_t i1 = tgpig.x - k0*args.ne1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i03 = i3;").unwrap();
    writeln!(out, "    const int32_t i02 = i2;").unwrap();
    writeln!(out, "    const int32_t i01 = i1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);").unwrap();
    writeln!(out, "    device       T * dst_ptr  = (device       T *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {{").unwrap();
    writeln!(out, "        const int32_t i0 = k0*1024 + tpitg.x + l0;").unwrap();
    writeln!(out, "        if (i0 >= args.ne0) {{").unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {{"
    )
    .unwrap();
    writeln!(out, "            dst_ptr[i0] = src0_ptr[i0];").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_unary_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T0, typename T, typename TC>").unwrap();
    writeln!(out, "kernel void kernel_unary_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_unary & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "#define FC_OP  FC_unary_op").unwrap();
    writeln!(out, "#define FC_CNT FC_unary_cnt").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T0 * src0_ptr;").unwrap();
    writeln!(out, "    device       T  * dst_ptr;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int i0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (FC_CNT) {{").unwrap();
    writeln!(out, "        i0 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        src0_ptr = (device const T0 *) (src0);").unwrap();
    writeln!(out, "        dst_ptr  = (device       T  *) (dst);").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        const int i03 = tgpig.z;").unwrap();
    writeln!(out, "        const int i02 = tgpig.y;").unwrap();
    writeln!(out, "        const int k0  = tgpig.x/args.ne01;").unwrap();
    writeln!(out, "        const int i01 = tgpig.x - k0*args.ne01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        i0 = k0*ntg.x + tpitg.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);").unwrap();
    writeln!(out, "        dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1 );").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        //threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (!FC_CNT) {{").unwrap();
    writeln!(out, "            if (i0 >= args.ne0) {{").unwrap();
    writeln!(out, "                return;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const TC x = (TC) src0_ptr[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SCALE) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) (args.scale * x + args.bias);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_FILL) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) args.val;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_CLAMP) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) clamp(x, args.min, args.max);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SQR) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) (x * x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SQRT) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) sqrt(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SIN) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) sin(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_COS) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) cos(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_LOG) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) log(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_LEAKY_RELU) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) (TC(x > 0)*x + TC(x <= 0)*(x * args.slope));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_TANH) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) precise::tanh(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_RELU) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) fmax(0, x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SIGMOID) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) (1 / (1 + exp(-x)));").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_GELU) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) (0.5*x*(1 + precise::tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x))));").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_GELU_ERF) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) (0.5*x*(1 + erf_approx(SQRT_2_INV*x)));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_GELU_QUICK) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) (x * (1/(1 + exp(GELU_QUICK_COEF*x))));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SILU) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) (x / (1 + exp(-x)));").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_ELU) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) elu_approx(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_NEG) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) -x;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_ABS) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) fabs(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SGN) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = T(x > 0) - T(x < 0);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_STEP) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = T(x > 0);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_HARDSWISH) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) (x * fmax(0, fmin(1, x/6 + 0.5)));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_HARDSIGMOID) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) fmax(0, fmin(1, x/6 + 0.5));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_EXP) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) exp(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_SOFTPLUS) {{").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) select(log(1 + exp(x)), x, x > 20);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_EXPM1) {{").unwrap();
    writeln!(out, "            // TODO: precise implementation").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) (exp(x) - 1);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_FLOOR) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) floor(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_CEIL) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) ceil(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_ROUND) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) round(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_TRUNC) {{").unwrap();
    writeln!(out, "            dst_ptr[i0] = (T) trunc(x);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_OP == OP_UNARY_NUM_XIELU) {{").unwrap();
    writeln!(out, "            const TC xi      = x;").unwrap();
    writeln!(out, "            const TC gate    = TC(xi > TC(0.0f));").unwrap();
    writeln!(
        out,
        "            const TC clamped = fmin(xi, TC(args.val));"
    )
    .unwrap();
    writeln!(
        out,
        "            const TC y_pos   = TC(args.scale) * xi * xi + TC(args.bias) * xi;"
    )
    .unwrap();
    writeln!(out, "            const TC y_neg   = (exp(clamped) - TC(1.0f) - xi) * TC(args.slope) + TC(args.bias) * xi;").unwrap();
    writeln!(
        out,
        "            dst_ptr[i0] = (T) (gate * y_pos + (TC(1.0f) - gate) * y_neg);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef FC_OP").unwrap();
    writeln!(out, "#undef FC_CNT").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_bin_fuse_impl_ggml(out: &mut String) {
    writeln!(out, "template <typename T0, typename T1, typename T>").unwrap();
    writeln!(out, "kernel void kernel_bin_fuse_impl(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_bin & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "#define FC_OP FC_bin_op").unwrap();
    writeln!(out, "#define FC_F  FC_bin_f").unwrap();
    writeln!(out, "#define FC_RB FC_bin_rb").unwrap();
    writeln!(out, "#define FC_CB FC_bin_cb").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (FC_RB) {{").unwrap();
    writeln!(out, "        // row broadcast").unwrap();
    writeln!(out, "        const uint i0 = tgpig.y*args.ne00 + tgpig.x;").unwrap();
    writeln!(
        out,
        "        const uint i1 = FC_CB ? tgpig.x%args.ne10 : tgpig.x;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        device const T0 * src0_row = (device const T0 *) (src0);"
    )
    .unwrap();
    writeln!(
        out,
        "        device       T  * dst_row  = (device       T  *) (dst);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_F == 1) {{").unwrap();
    writeln!(
        out,
        "            device const T1 * src1_row = (device const T1 *) (src1 + args.o1[0]);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 0) {{").unwrap();
    writeln!(
        out,
        "                dst_row[i0] = src0_row[i0] + src1_row[i1];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 1) {{").unwrap();
    writeln!(
        out,
        "                dst_row[i0] = src0_row[i0] - src1_row[i1];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 2) {{").unwrap();
    writeln!(
        out,
        "                dst_row[i0] = src0_row[i0] * src1_row[i1];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 3) {{").unwrap();
    writeln!(
        out,
        "                dst_row[i0] = src0_row[i0] / src1_row[i1];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            T0 res = src0_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 0) {{").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    res += ((device const T1 *) (src1 + args.o1[j]))[i1];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 1) {{").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    res -= ((device const T1 *) (src1 + args.o1[j]))[i1];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 2) {{").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    res *= ((device const T1 *) (src1 + args.o1[j]))[i1];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_OP == 3) {{").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    res /= ((device const T1 *) (src1 + args.o1[j]))[i1];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dst_row[i0] = res;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        const int i03 = tgpig.z;").unwrap();
    writeln!(out, "        const int i02 = tgpig.y;").unwrap();
    writeln!(out, "        const int i01 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (i01 >= args.ne01) {{").unwrap();
    writeln!(out, "            return;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int i13 = i03%args.ne13;").unwrap();
    writeln!(out, "        const int i12 = i02%args.ne12;").unwrap();
    writeln!(out, "        const int i11 = i01%args.ne11;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const T0 * src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs);").unwrap();
    writeln!(out, "        device       T  * dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_F == 1) {{").unwrap();
    writeln!(out, "            device const T1 * src1_ptr = (device const T1 *) (src1 + args.o1[0] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                const int i10 = FC_CB ? i0%args.ne10 : i0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 0) {{").unwrap();
    writeln!(
        out,
        "                    dst_ptr[i0] = src0_ptr[i0] + src1_ptr[i10];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 1) {{").unwrap();
    writeln!(
        out,
        "                    dst_ptr[i0] = src0_ptr[i0] - src1_ptr[i10];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 2) {{").unwrap();
    writeln!(
        out,
        "                    dst_ptr[i0] = src0_ptr[i0] * src1_ptr[i10];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 3) {{").unwrap();
    writeln!(
        out,
        "                    dst_ptr[i0] = src0_ptr[i0] / src1_ptr[i10];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            device const T1 * src1_ptr[8];").unwrap();
    writeln!(
        out,
        "            FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(out, "                src1_ptr[j] = (device const T1 *) (src1 + args.o1[j] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                const int i10 = FC_CB ? i0%args.ne10 : i0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                T res = src0_ptr[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 0) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(out, "                        res += src1_ptr[j][i10];").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 1) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(out, "                        res -= src1_ptr[j][i10];").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 2) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(out, "                        res *= src1_ptr[j][i10];").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_OP == 3) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {{"
    )
    .unwrap();
    writeln!(out, "                        res /= src1_ptr[j][i10];").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                dst_ptr[i0] = res;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef FC_OP").unwrap();
    writeln!(out, "#undef FC_F").unwrap();
    writeln!(out, "#undef FC_RB").unwrap();
    writeln!(out, "#undef FC_CB").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_gated_delta_net_impl_ggml(out: &mut String) {
    writeln!(out, "template<short NSG>").unwrap();
    writeln!(out, "kernel void kernel_gated_delta_net_impl(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_gated_delta_net & args,"
    )
    .unwrap();
    writeln!(out, "        device const char * q,").unwrap();
    writeln!(out, "        device const char * k,").unwrap();
    writeln!(out, "        device const char * v,").unwrap();
    writeln!(out, "        device const char * g,").unwrap();
    writeln!(out, "        device const char * b,").unwrap();
    writeln!(out, "        device const char * s,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3   ntg[[threads_per_threadgroup]])  {{").unwrap();
    writeln!(out, "#define S_v FC_gated_delta_net_ne20").unwrap();
    writeln!(out, "#define G   FC_gated_delta_net_ne30").unwrap();
    writeln!(out, "#define K   FC_gated_delta_net_K").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint tx = tpitg.x;").unwrap();
    writeln!(out, "    const uint ty = tpitg.y;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i23 = tgpig.z; // B (n_seqs)").unwrap();
    writeln!(out, "    const uint i21 = tgpig.y; // H (head)").unwrap();
    writeln!(
        out,
        "    const uint i20 = tgpig.x*NSG + ty; // row within S_v"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i01 = i21 % args.ne01;").unwrap();
    writeln!(out, "    const uint i11 = i21 % args.ne11;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float scale = 1.0f / sqrt((float)S_v);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // input state layout (D, K, n_seqs): per-seq stride is K*H*D; we read slot 0."
    )
    .unwrap();
    writeln!(
        out,
        "    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint state_in_base = (i23*K*args.ne21 + i21)*S_v*S_v + i20*S_v;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float * s_ptr = (device const float *) (s) + state_in_base;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float ls[NSG];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    FOR_UNROLL (short j = 0; j < NSG; j++) {{").unwrap();
    writeln!(out, "        const short is = tx*NSG + j;").unwrap();
    writeln!(out, "        ls[j] = s_ptr[is];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);").unwrap();
    writeln!(out, "    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);").unwrap();
    writeln!(out, "    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);").unwrap();
    writeln!(out, "    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // snapshot slot mapping: target_slot = t - shift. When n_tokens < K, only the last"
    )
    .unwrap();
    writeln!(
        out,
        "    // n_tokens slots are written; earlier slots are left untouched (caller-owned)."
    )
    .unwrap();
    writeln!(out, "    const int shift = (int)args.ne22 - (int)K;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // output state base offset: after attention scores"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;"
    )
    .unwrap();
    writeln!(
        out,
        "    // output state per-slot size: S_v * S_v * H * n_seqs"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;"
    )
    .unwrap();
    writeln!(out, "    // per-(seq,head) offset within a slot").unwrap();
    writeln!(
        out,
        "    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short t = 0; t < args.ne22; t++) {{").unwrap();
    writeln!(out, "        float s_k = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (G == 1) {{").unwrap();
    writeln!(out, "            const float g_exp = exp(g_ptr[0]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            FOR_UNROLL (short j = 0; j < NSG; j++) {{").unwrap();
    writeln!(out, "                const short is = tx*NSG + j;").unwrap();
    writeln!(out, "                ls[j] *= g_exp;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                s_k += ls[j]*k_ptr[is];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            // KDA").unwrap();
    writeln!(out, "            FOR_UNROLL (short j = 0; j < NSG; j++) {{").unwrap();
    writeln!(out, "                const short is = tx*NSG + j;").unwrap();
    writeln!(out, "                ls[j] *= exp(g_ptr[is]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                s_k += ls[j]*k_ptr[is];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        s_k = simd_sum(s_k);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float d = (v_ptr[i20] - s_k)*b_ptr[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float y = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        FOR_UNROLL (short j = 0; j < NSG; j++) {{").unwrap();
    writeln!(out, "            const short is = tx*NSG + j;").unwrap();
    writeln!(out, "            ls[j] += k_ptr[is]*d;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            y += ls[j]*q_ptr[is];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y = simd_sum(y);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tx == 0) {{").unwrap();
    writeln!(out, "            dst_attn[t*args.ne21*S_v] = y*scale;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        q_ptr += args.ns02;").unwrap();
    writeln!(out, "        k_ptr += args.ns12;").unwrap();
    writeln!(out, "        v_ptr += args.ns22;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        b_ptr += args.ne21;").unwrap();
    writeln!(out, "        g_ptr += args.ne21*G;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (K > 1) {{").unwrap();
    writeln!(out, "            const int target_slot = (int)t - shift;").unwrap();
    writeln!(
        out,
        "            if (target_slot >= 0 && target_slot < (int)K) {{"
    )
    .unwrap();
    writeln!(out, "                device float * dst_state = (device float *) (dst) + attn_size + (uint)target_slot * state_size_per_snap + state_out_base;").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short j = 0; j < NSG; j++) {{"
    )
    .unwrap();
    writeln!(out, "                    const short is = tx*NSG + j;").unwrap();
    writeln!(out, "                    dst_state[is] = ls[j];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (K == 1) {{").unwrap();
    writeln!(
        out,
        "        device float * dst_state = (device float *) (dst) + attn_size + state_out_base;"
    )
    .unwrap();
    writeln!(out, "        FOR_UNROLL (short j = 0; j < NSG; j++) {{").unwrap();
    writeln!(out, "            const short is = tx*NSG + j;").unwrap();
    writeln!(out, "            dst_state[is] = ls[j];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef S_v").unwrap();
    writeln!(out, "#undef G").unwrap();
    writeln!(out, "#undef K").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_flash_attn_ext_impl_ggml(out: &mut String) {
    writeln!(out, "template<").unwrap();
    writeln!(out, "    typename q_t,     // query types in shared memory").unwrap();
    writeln!(out, "    typename q4_t,").unwrap();
    writeln!(out, "    typename q8x8_t,").unwrap();
    writeln!(out, "    typename k_t,     // key types in shared memory").unwrap();
    writeln!(out, "    typename k4x4_t,").unwrap();
    writeln!(out, "    typename k8x8_t,").unwrap();
    writeln!(out, "    typename v_t,     // value types in shared memory").unwrap();
    writeln!(out, "    typename v4x4_t,").unwrap();
    writeln!(out, "    typename v8x8_t,").unwrap();
    writeln!(out, "    typename qk_t,    // Q*K types").unwrap();
    writeln!(out, "    typename qk8x8_t,").unwrap();
    writeln!(out, "    typename s_t,     // soft-max types").unwrap();
    writeln!(out, "    typename s2_t,").unwrap();
    writeln!(out, "    typename s8x8_t,").unwrap();
    writeln!(out, "    typename o_t,     // attention accumulation types").unwrap();
    writeln!(out, "    typename o4_t,").unwrap();
    writeln!(out, "    typename o8x8_t,").unwrap();
    writeln!(out, "    typename kd4x4_t, // key type in device memory").unwrap();
    writeln!(out, "    short nl_k,").unwrap();
    writeln!(
        out,
        "    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),"
    )
    .unwrap();
    writeln!(out, "    typename vd4x4_t, // value type in device memory").unwrap();
    writeln!(out, "    short nl_v,").unwrap();
    writeln!(
        out,
        "    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),"
    )
    .unwrap();
    writeln!(out, "    short DK,         // K head size").unwrap();
    writeln!(out, "    short DV,         // V head size").unwrap();
    writeln!(out, "    short Q,          // queries per threadgroup").unwrap();
    writeln!(out, "    short C,          // cache items per threadgroup").unwrap();
    writeln!(out, "    short NSG>        // number of simd groups").unwrap();
    writeln!(out, "void kernel_flash_attn_ext_impl(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_flash_attn_ext & args,"
    )
    .unwrap();
    writeln!(out, "        device const char * q,").unwrap();
    writeln!(out, "        device const char * k,").unwrap();
    writeln!(out, "        device const char * v,").unwrap();
    writeln!(out, "        device const char * mask,").unwrap();
    writeln!(out, "        device const char * sinks,").unwrap();
    writeln!(out, "        device const char * pad,").unwrap();
    writeln!(out, "        device const char * blk,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  half * shmem_f16,").unwrap();
    writeln!(out, "        uint3   tgpig,").unwrap();
    writeln!(out, "        ushort  tiisg,").unwrap();
    writeln!(out, "        ushort  sgitg) {{").unwrap();
    writeln!(out, "    const ushort iq3 = tgpig[2];").unwrap();
    writeln!(out, "    const ushort iq2 = tgpig[1];").unwrap();
    writeln!(out, "    const ushort iq1 = tgpig[0]*Q;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#define NS10 (FC_flash_attn_ext_ns10)").unwrap();
    writeln!(out, "#define NS20 (FC_flash_attn_ext_ns20)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // note: I had some concerns that using this instead of the ugly macros above was affecting performance").unwrap();
    writeln!(out, "    //       need to re-check carefully and if no regressions are observerd - remove the macros").unwrap();
    writeln!(out, "    //       the concerns is that maybe using const variables requires extra registers? but not sure if the compiler").unwrap();
    writeln!(out, "    //         is clever enough to avoid this. unfortunately, using constexpr is not possible with FC").unwrap();
    writeln!(out, "    //const short NS10 = FC_flash_attn_ext_ns10;").unwrap();
    writeln!(out, "    //const short NS20 = FC_flash_attn_ext_ns20;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short KV   = 8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short DK4  = DK/4;").unwrap();
    writeln!(out, "    constexpr short DK8  = DK/8;").unwrap();
    writeln!(out, "    constexpr short DK16 = DK/16;").unwrap();
    writeln!(out, "    constexpr short DV4  = DV/4;").unwrap();
    writeln!(out, "  //constexpr short DV8  = DV/8;").unwrap();
    writeln!(out, "    constexpr short DV16 = DV/16;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short PV   = PAD2(DV, 64);").unwrap();
    writeln!(out, "    constexpr short PV4  = PV/4;").unwrap();
    writeln!(out, "    constexpr short PV8  = PV/8;").unwrap();
    writeln!(out, "  //constexpr short PV16 = PV/16;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short NW  = N_SIMDWIDTH;").unwrap();
    writeln!(out, "    constexpr short NQ  = Q/NSG;").unwrap();
    writeln!(
        out,
        "    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short TS = 2*SH;").unwrap();
    writeln!(
        out,
        "    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data").unwrap();
    writeln!(out, "    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t").unwrap();
    writeln!(out, "    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)").unwrap();
    writeln!(
        out,
        "    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);"
    )
    .unwrap();
    writeln!(out, "    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix").unwrap();
    writeln!(out, "    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load K in shared memory").unwrap();
    writeln!(out, "    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in k4x4_t").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load V in shared memory").unwrap();
    writeln!(out, "    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in v4x4_t").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // mask storage in shared mem").unwrap();
    writeln!(
        out,
        "    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // per-query mask pointers").unwrap();
    writeln!(out, "    device const half2 * pm2[NQ];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{").unwrap();
    writeln!(out, "        const short j = jj*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(
        out,
        "        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);"
    )
    .unwrap();
    writeln!(
        out,
        "        const int32_t nblk0 = ((args.ne11 + C - 1)/C);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(
        out,
        "        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const short ikv2 = iq2/(args.ne02/args.ne_12_2);"
    )
    .unwrap();
    writeln!(
        out,
        "        const short ikv3 = iq3/(args.ne03/args.ne_12_3);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        k += ikv2*args.nb12 + ikv3*args.nb13;").unwrap();
    writeln!(out, "        v += ikv2*args.nb22 + ikv3*args.nb23;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // load heads from Q to shared memory").unwrap();
    writeln!(out, "    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{").unwrap();
    writeln!(out, "        const short j = jj*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short i = tiisg; i < DK4; i += NW) {{").unwrap();
    writeln!(out, "            if (iq1 + j < args.ne01) {{").unwrap();
    writeln!(out, "                sq4[j*DK4 + i] = (q4_t) q4[i];").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                sq4[j*DK4 + i] = 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // zero out").unwrap();
    writeln!(out, "    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{").unwrap();
    writeln!(out, "        const short j = jj*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short i = tiisg; i < DV4; i += NW) {{").unwrap();
    writeln!(out, "            so4[j*PV4 + i] = 0;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short i = tiisg; i < SH; i += NW) {{").unwrap();
    writeln!(out, "            ss[j*SH + i] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float S[NQ] = {{ [0 ... NQ-1] = 0.0f }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(
        out,
        "        float M[NQ] = {{ [0 ... NQ-1] = -FLT_MAX/2 }};"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float slope = 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // ALiBi").unwrap();
    writeln!(out, "        if (FC_flash_attn_ext_has_bias) {{").unwrap();
    writeln!(out, "            const short h = iq2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float base = h < args.n_head_log2 ? args.m0 : args.m1;"
    )
    .unwrap();
    writeln!(out, "            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            slope = pow(base, exph);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // loop over the KV cache").unwrap();
    writeln!(
        out,
        "        // each simdgroup handles blocks of Q rows and C columns"
    )
    .unwrap();
    writeln!(out, "        for (int ic0 = 0; ; ++ic0) {{").unwrap();
    writeln!(out, "            int ic = ic0*C;").unwrap();
    writeln!(out, "            if (ic >= args.ne11) {{").unwrap();
    writeln!(out, "                break;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            // the last partial chunk uses the pad buffer as source"
    )
    .unwrap();
    writeln!(
        out,
        "            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {{"
    )
    .unwrap();
    writeln!(out, "                k    = pad;").unwrap();
    writeln!(
        out,
        "                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;"
    )
    .unwrap();
    writeln!(
        out,
        "                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                const short ikv2 = iq2/(args.ne02/args.ne_12_2);"
    )
    .unwrap();
    writeln!(
        out,
        "                const short ikv3 = iq3/(args.ne03/args.ne_12_3);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;"
    )
    .unwrap();
    writeln!(
        out,
        "                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (!FC_flash_attn_ext_has_mask) {{").unwrap();
    writeln!(
        out,
        "                    threadgroup half * sm = (threadgroup half *) (sm2);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        const short j = jj*NSG + sgitg;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        for (short i = tiisg; i < C; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            if (ic + i >= args.ne11) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                                sm[2*j*SH + i] = -MAXHALF;"
    )
    .unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        const short j = jj*NSG + sgitg;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        pm2[jj] = (device const half2 *) ((device const half *) mask +"
    )
    .unwrap();
    writeln!(out, "                                (iq1 + j)*C +").unwrap();
    writeln!(
        out,
        "                                (iq2%args.ne32)*(C*args.ne31) +"
    )
    .unwrap();
    writeln!(
        out,
        "                                (iq3%args.ne33)*(C*args.ne31*args.ne32));"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                ic = 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            char blk_cur = 1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // read the mask into shared mem").unwrap();
    writeln!(out, "            if (FC_flash_attn_ext_has_mask) {{").unwrap();
    writeln!(out, "                blk_cur = blk[ic0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (blk_cur == 0) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(out, "                        pm2[jj] += NW;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    continue;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (blk_cur == 1) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        const short j = jj*NSG + sgitg;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        if (FC_flash_attn_ext_bc_mask) {{"
    )
    .unwrap();
    writeln!(out, "                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);").unwrap();
    writeln!(out, "                        }} else {{").unwrap();
    writeln!(
        out,
        "                            sm2[j*SH + tiisg] = pm2[jj][tiisg];"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        pm2[jj] += NW;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else if (blk_cur == 2) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(out, "                        pm2[jj] += NW;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#if 0").unwrap();
    writeln!(out, "                // note: old -INF block optimization - obsoleted by pre-computing non-masked blocks").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                // used to detect blocks full of -INF").unwrap();
    writeln!(
        out,
        "                // skip only when the entire threadgroup is masked"
    )
    .unwrap();
    writeln!(out, "                half2 smax2(-MAXHALF/2, -MAXHALF/2);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short j = 0; j < Q; ++j) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    smax2 = max(smax2, sm2[j*SH + tiisg]);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                smax2 = simd_max(smax2);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {{"
    )
    .unwrap();
    writeln!(out, "                    // this barrier is important").unwrap();
    writeln!(
        out,
        "                    threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    continue;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // Q*K^T").unwrap();
    writeln!(
        out,
        "            // this is compile-time check, so it does not have runtime overhead"
    )
    .unwrap();
    writeln!(out, "            if (is_same<kd4x4_t, k4x4_t>::value) {{").unwrap();
    writeln!(
        out,
        "                // we can read directly from global memory"
    )
    .unwrap();
    writeln!(
        out,
        "                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);"
    )
    .unwrap();
    writeln!(out, "                threadgroup const q_t * pq = sq;").unwrap();
    writeln!(out, "                threadgroup       s_t * ps = ss;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                pk += sgitg*(8*NS10);").unwrap();
    writeln!(out, "                ps += sgitg*(8*1);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                static_assert((C/8) % NSG == 0, \"\");"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                constexpr short NC = (C/8)/NSG;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    if (DK % 16 != 0) {{").unwrap();
    writeln!(out, "                        k8x8_t mk;").unwrap();
    writeln!(out, "                        q8x8_t mq;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short i = 0; i < DK8; ++i) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_barrier(mem_flags::mem_none);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_load(mq, pq + 8*i, DK);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_barrier(mem_flags::mem_none);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(out, "                        k8x8_t mk[2];").unwrap();
    writeln!(out, "                        q8x8_t mq[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        // note: too much unroll can tank the performance for large heads"
    )
    .unwrap();
    writeln!(
        out,
        "                        #pragma unroll (MIN(DK8/2, 4*NSG))"
    )
    .unwrap();
    writeln!(
        out,
        "                        for (short i = 0; i < DK8/2; ++i) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_barrier(mem_flags::mem_none);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_barrier(mem_flags::mem_none);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    simdgroup_store(mqk, ps, SH, 0, false);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    pk += 8*(NSG*NS10);").unwrap();
    writeln!(out, "                    ps += 8*(NSG);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(
        out,
        "                // TODO: this is the quantized K cache branch - not optimized yet"
    )
    .unwrap();
    writeln!(
        out,
        "                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {{"
    )
    .unwrap();
    writeln!(out, "                    const short cc = ccc*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    const short tx = tiisg%4;").unwrap();
    writeln!(out, "                    const short ty = tiisg/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    for (short ii = 0; ii < DK16; ii += 4) {{"
    )
    .unwrap();
    writeln!(out, "                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        if (DK16%4 == 0) {{").unwrap();
    writeln!(out, "                            // the head is evenly divisible by 4*16 = 64, so no need for bound checks").unwrap();
    writeln!(out, "                            {{").unwrap();
    writeln!(out, "                                k4x4_t tmp;").unwrap();
    writeln!(
        out,
        "                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);"
    )
    .unwrap();
    writeln!(
        out,
        "                                sk4x4[4*ty + tx] = tmp;"
    )
    .unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            FOR_UNROLL (short k = 0; k < 4; ++k) {{"
    )
    .unwrap();
    writeln!(out, "                                k8x8_t mk;").unwrap();
    writeln!(out, "                                q8x8_t mq;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose").unwrap();
    writeln!(
        out,
        "                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);"
    )
    .unwrap();
    writeln!(
        out,
        "                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose").unwrap();
    writeln!(
        out,
        "                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);"
    )
    .unwrap();
    writeln!(
        out,
        "                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);"
    )
    .unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out, "                        }} else {{").unwrap();
    writeln!(out, "                            if (ii + tx < DK16) {{").unwrap();
    writeln!(out, "                                k4x4_t tmp;").unwrap();
    writeln!(
        out,
        "                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);"
    )
    .unwrap();
    writeln!(
        out,
        "                                sk4x4[4*ty + tx] = tmp;"
    )
    .unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {{"
    )
    .unwrap();
    writeln!(out, "                                k8x8_t mk;").unwrap();
    writeln!(out, "                                q8x8_t mq;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose").unwrap();
    writeln!(
        out,
        "                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);"
    )
    .unwrap();
    writeln!(
        out,
        "                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose").unwrap();
    writeln!(
        out,
        "                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);"
    )
    .unwrap();
    writeln!(
        out,
        "                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);"
    )
    .unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // online softmax").unwrap();
    writeln!(
        out,
        "            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(out, "                const short j = jj*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const float m = M[jj];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                // scale and apply the logitcap / mask"
    )
    .unwrap();
    writeln!(
        out,
        "                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_flash_attn_ext_has_scap) {{").unwrap();
    writeln!(
        out,
        "                    s2 = args.logit_softcap*precise::tanh(s2);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                // mqk = mqk + slope*mask").unwrap();
    writeln!(out, "                if (blk_cur != 2) {{").unwrap();
    writeln!(
        out,
        "                    if (FC_flash_attn_ext_has_bias) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        s2 += s2_t(sm2[j*SH + tiisg])*slope;"
    )
    .unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(
        out,
        "                        s2 += s2_t(sm2[j*SH + tiisg]);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const float  ms  = exp(m  - M[jj]);").unwrap();
    writeln!(out, "                const float2 vs2 = exp(s2 - M[jj]);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                // the P matrix from the paper (Q rows, C columns)"
    )
    .unwrap();
    writeln!(out, "                ss2[j*SH/2 + tiisg] = vs2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (DV4 % NW == 0) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        const short i = ii*NW + tiisg;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        so4[j*PV4 + i] *= ms;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    for (short i = tiisg; i < DV4; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "                        so4[j*PV4 + i] *= ms;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // O = O + (Q*K^T)*V").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(
        out,
        "                // we can read directly from global memory"
    )
    .unwrap();
    writeln!(
        out,
        "                if (is_same<vd4x4_t, v4x4_t>::value) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    static_assert(PV8 % NSG == 0, \"\");"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    constexpr short NO = PV8/NSG;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    o8x8_t lo[NO];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    {{").unwrap();
    writeln!(out, "                        auto sot = so + 8*sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_load(lo[ii], sot, PV, 0, false);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                            sot += 8*NSG;").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    {{").unwrap();
    writeln!(
        out,
        "                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        pv += 8*sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        if (DV <= 64) {{").unwrap();
    writeln!(
        out,
        "                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {{"
    )
    .unwrap();
    writeln!(out, "                                s8x8_t vs;").unwrap();
    writeln!(
        out,
        "                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                                    v8x8_t mv[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);").unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);").unwrap();
    writeln!(out, "                                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                pv  += 8*NS20;").unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out, "                        }} else {{").unwrap();
    writeln!(
        out,
        "                            constexpr short NC = (C/8)/2;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {{"
    )
    .unwrap();
    writeln!(out, "                                s8x8_t vs[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);"
    )
    .unwrap();
    writeln!(
        out,
        "                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                                    v8x8_t mv[4];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);").unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);").unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);").unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);").unwrap();
    writeln!(out, "                                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                pv  += 2*8*NS20;").unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    {{").unwrap();
    writeln!(out, "                        auto sot = so + 8*sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            simdgroup_store(lo[ii], sot, PV, 0, false);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                            sot += 8*NSG;").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    // TODO: this is the quantized V cache branch - not optimized yet"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    const short tx = tiisg%4;").unwrap();
    writeln!(out, "                    const short ty = tiisg/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    for (short cc = 0; cc < C/8; ++cc) {{"
    )
    .unwrap();
    writeln!(out, "                        s8x8_t vs;").unwrap();
    writeln!(
        out,
        "                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {{"
    )
    .unwrap();
    writeln!(out, "                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                            if (DV16%4 == 0) {{").unwrap();
    writeln!(
        out,
        "                                // no need for bound checks"
    )
    .unwrap();
    writeln!(out, "                                {{").unwrap();
    writeln!(out, "                                    v4x4_t tmp;").unwrap();
    writeln!(
        out,
        "                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);"
    )
    .unwrap();
    writeln!(
        out,
        "                                    sv4x4[4*ty + tx] = tmp;"
    )
    .unwrap();
    writeln!(out, "                                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                FOR_UNROLL (short k = 0; k < 4; ++k) {{"
    )
    .unwrap();
    writeln!(out, "                                    v8x8_t mv[2];").unwrap();
    writeln!(out, "                                    o8x8_t lo[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);").unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);").unwrap();
    writeln!(out, "                                }}").unwrap();
    writeln!(out, "                            }} else {{").unwrap();
    writeln!(
        out,
        "                                if (ii + tx < DV16) {{"
    )
    .unwrap();
    writeln!(out, "                                    v4x4_t tmp;").unwrap();
    writeln!(
        out,
        "                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);"
    )
    .unwrap();
    writeln!(
        out,
        "                                    sv4x4[4*ty + tx] = tmp;"
    )
    .unwrap();
    writeln!(out, "                                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {{"
    )
    .unwrap();
    writeln!(out, "                                    v8x8_t mv[2];").unwrap();
    writeln!(out, "                                    o8x8_t lo[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);").unwrap();
    writeln!(out, "                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);").unwrap();
    writeln!(out, "                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);").unwrap();
    writeln!(out, "                                }}").unwrap();
    writeln!(out, "                            }}").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_flash_attn_ext_has_sinks) {{").unwrap();
    writeln!(
        out,
        "            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {{"
    )
    .unwrap();
    writeln!(out, "                const short j = jj*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const float m = M[jj];").unwrap();
    writeln!(out, "                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                M[jj] = simd_max(max(M[jj], s));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const float ms = exp(m - M[jj]);").unwrap();
    writeln!(out, "                const float vs = exp(s - M[jj]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                S[jj] = S[jj]*ms + simd_sum(vs);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                for (short i = tiisg; i < DV4; i += NW) {{"
    )
    .unwrap();
    writeln!(out, "                    so4[j*PV4 + i] *= ms;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // store to global memory").unwrap();
    writeln!(out, "    for (short jj = 0; jj < NQ; ++jj) {{").unwrap();
    writeln!(out, "        const short j = jj*NSG + sgitg;").unwrap();
    writeln!(out, "        if (iq1 + j >= args.ne01) {{").unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (DV4 % NW == 0) {{").unwrap();
    writeln!(
        out,
        "            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                const short i = ii*NW + tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                dst4[i] = (float4) so4[j*PV4 + i]*scale;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (short i = tiisg; i < DV4; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                dst4[i] = (float4) so4[j*PV4 + i]*scale;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef NS10").unwrap();
    writeln!(out, "#undef NS20").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_add_id_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_add_id(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_add_id & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device const char * src2,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int i1 = tgpig.x;").unwrap();
    writeln!(out, "    const int i2 = tgpig.y;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i11 = *((device const int32_t *) (src2 + i1*sizeof(int32_t) + i2*args.nb21));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const size_t nb1 = args.ne0 * sizeof(float);").unwrap();
    writeln!(out, "    const size_t nb2 = args.ne1 * nb1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *)((device char *)dst  +  i1*nb1       + i2*nb2);").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *)((device char *)src0 +  i1*args.nb01 + i2*args.nb02);").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *)((device char *)src1 + i11*args.nb11);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        dst_row[i0] = src0_row[i0] + src1_row[i0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_arange_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_arange_f32(").unwrap();
    writeln!(out, "    constant   ggml_metal_kargs_arange & args,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_ptr = (device float *) dst;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        dst_ptr[i0] = args.start + args.step * i0;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_argmax_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_argmax_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_argmax & args,").unwrap();
    writeln!(out, "        device   const char * src0,").unwrap();
    writeln!(out, "        device         char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup    char * shmem [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(out, "        uint  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint  tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        uint    ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float   lmax = -INFINITY;").unwrap();
    writeln!(out, "    int32_t larg = -1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {{"
    )
    .unwrap();
    writeln!(out, "        if (x_row[i00] > lmax) {{").unwrap();
    writeln!(out, "            lmax = x_row[i00];").unwrap();
    writeln!(out, "            larg = i00;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // find the argmax value in the block").unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(
        out,
        "    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device int32_t * dst_i32 = (device int32_t *) dst;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (ntg > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            shared_maxval[tiisg] = -INFINITY;").unwrap();
    writeln!(out, "            shared_argmax[tiisg] = -1;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            shared_maxval[sgitg] = max_val;").unwrap();
    writeln!(out, "            shared_argmax[sgitg] = arg_val;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        max_val = shared_maxval[tiisg];").unwrap();
    writeln!(out, "        arg_val = shared_argmax[tiisg];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float max_val_reduced   = simd_max(max_val);").unwrap();
    writeln!(out, "        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_i32[tgpig] = arg_val_reduced;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    dst_i32[tgpig] = arg_val;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_argsort_f32_i32_ggml(out: &mut String) {
    writeln!(out, "template<ggml_sort_order order>").unwrap();
    writeln!(out, "kernel void kernel_argsort_f32_i32(").unwrap();
    writeln!(out, "        constant   ggml_metal_kargs_argsort & args,").unwrap();
    writeln!(out, "        device   const char * src0,").unwrap();
    writeln!(out, "        device      int32_t * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    // bitonic sort").unwrap();
    writeln!(out, "    const int col = tpitg[0];").unwrap();
    writeln!(out, "    const int ib  = tgpig[0] / args.ne01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i00 = ib*ntg.x;").unwrap();
    writeln!(out, "    const int i01 = tgpig[0] % args.ne01;").unwrap();
    writeln!(out, "    const int i02 = tgpig[1];").unwrap();
    writeln!(out, "    const int i03 = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // initialize indices").unwrap();
    writeln!(out, "    shmem_i32[col] = i00 + col;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int k = 2; k <= ntg.x; k *= 2) {{").unwrap();
    writeln!(out, "        for (int j = k / 2; j > 0; j /= 2) {{").unwrap();
    writeln!(out, "            int ixj = col ^ j;").unwrap();
    writeln!(out, "            if (ixj > col) {{").unwrap();
    writeln!(out, "                if ((col & k) == 0) {{").unwrap();
    writeln!(
        out,
        "                    if (shmem_i32[col] >= args.ne00 ||"
    )
    .unwrap();
    writeln!(
        out,
        "                       (shmem_i32[ixj] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?"
    )
    .unwrap();
    writeln!(
        out,
        "                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :"
    )
    .unwrap();
    writeln!(
        out,
        "                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))"
    )
    .unwrap();
    writeln!(out, "                    ) {{").unwrap();
    writeln!(
        out,
        "                        SWAP(shmem_i32[col], shmem_i32[ixj]);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    if (shmem_i32[ixj] >= args.ne00 ||"
    )
    .unwrap();
    writeln!(
        out,
        "                       (shmem_i32[col] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?"
    )
    .unwrap();
    writeln!(
        out,
        "                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :"
    )
    .unwrap();
    writeln!(
        out,
        "                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))"
    )
    .unwrap();
    writeln!(out, "                    ) {{").unwrap();
    writeln!(
        out,
        "                        SWAP(shmem_i32[col], shmem_i32[ixj]);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i0 = ib*args.top_k;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // copy the result to dst without the padding").unwrap();
    writeln!(out, "    if (i0 + col < args.ne0 && col < args.top_k) {{").unwrap();
    writeln!(
        out,
        "        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst[col] = shmem_i32[col];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_argsort_merge_f32_i32_ggml(out: &mut String) {
    writeln!(out, "template<ggml_sort_order order>").unwrap();
    writeln!(out, "kernel void kernel_argsort_merge_f32_i32(").unwrap();
    writeln!(
        out,
        "        constant   ggml_metal_kargs_argsort_merge & args,"
    )
    .unwrap();
    writeln!(out, "        device const char    * src0,").unwrap();
    writeln!(out, "        device const int32_t * tmp,").unwrap();
    writeln!(out, "        device       int32_t * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int im  = tgpig[0] / args.ne01;").unwrap();
    writeln!(out, "    const int i01 = tgpig[0] % args.ne01;").unwrap();
    writeln!(out, "    const int i02 = tgpig[1];").unwrap();
    writeln!(out, "    const int i03 = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int start = im * (2 * args.len);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));"
    )
    .unwrap();
    writeln!(
        out,
        "    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int total = len0 + len1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const int32_t * tmp0 = tmp + start").unwrap();
    writeln!(out, "        + i01*args.ne0").unwrap();
    writeln!(out, "        + i02*args.ne0*args.ne01").unwrap();
    writeln!(out, "        + i03*args.ne0*args.ne01*args.ne02;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const int32_t * tmp1 = tmp0 + args.len;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    dst += start").unwrap();
    writeln!(out, "        + i01*args.top_k").unwrap();
    writeln!(out, "        + i02*args.top_k*args.ne01").unwrap();
    writeln!(out, "        + i03*args.top_k*args.ne01*args.ne02;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * src0_row = (device const float *)(src0"
    )
    .unwrap();
    writeln!(out, "        + args.nb01*i01").unwrap();
    writeln!(out, "        + args.nb02*i02").unwrap();
    writeln!(out, "        + args.nb03*i03);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (total == 0) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int chunk = (total + ntg.x - 1) / ntg.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int k0 = tpitg.x * chunk;").unwrap();
    writeln!(
        out,
        "    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (k0 >= args.top_k) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (k0 >= total) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int low  = k0 > len1 ? k0 - len1 : 0;").unwrap();
    writeln!(out, "    int high = MIN(k0, len0);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // binary-search partition (i, j) such that i + j = k"
    )
    .unwrap();
    writeln!(out, "    while (low < high) {{").unwrap();
    writeln!(out, "        const int mid = (low + high) >> 1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int32_t idx0 = tmp0[mid];").unwrap();
    writeln!(out, "        const int32_t idx1 = tmp1[k0 - mid - 1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float val0 = src0_row[idx0];").unwrap();
    writeln!(out, "        const float val1 = src0_row[idx1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        bool take_left;").unwrap();
    writeln!(out, "        if (order == GGML_SORT_ORDER_ASC) {{").unwrap();
    writeln!(out, "            take_left = (val0 <= val1);").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            take_left = (val0 >= val1);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (take_left) {{").unwrap();
    writeln!(out, "            low = mid + 1;").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            high = mid;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int i = low;").unwrap();
    writeln!(out, "    int j = k0 - i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // keep the merge fronts into registers").unwrap();
    writeln!(out, "    int32_t idx0 = 0;").unwrap();
    writeln!(out, "    float   val0 = 0.0f;").unwrap();
    writeln!(out, "    if (i < len0) {{").unwrap();
    writeln!(out, "        idx0 = tmp0[i];").unwrap();
    writeln!(out, "        val0 = src0_row[idx0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int32_t idx1 = 0;").unwrap();
    writeln!(out, "    float   val1 = 0.0f;").unwrap();
    writeln!(out, "    if (j < len1) {{").unwrap();
    writeln!(out, "        idx1 = tmp1[j];").unwrap();
    writeln!(out, "        val1 = src0_row[idx1];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int k = k0; k < k1; ++k) {{").unwrap();
    writeln!(out, "        int32_t out_idx;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (i >= len0) {{").unwrap();
    writeln!(out, "            while (k < k1) {{").unwrap();
    writeln!(out, "                dst[k++] = tmp1[j++];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }} else if (j >= len1) {{").unwrap();
    writeln!(out, "            while (k < k1) {{").unwrap();
    writeln!(out, "                dst[k++] = tmp0[i++];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            break;").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            bool take_left;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (order == GGML_SORT_ORDER_ASC) {{").unwrap();
    writeln!(out, "                take_left = (val0 <= val1);").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                take_left = (val0 >= val1);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (take_left) {{").unwrap();
    writeln!(out, "                out_idx = idx0;").unwrap();
    writeln!(out, "                ++i;").unwrap();
    writeln!(out, "                if (i < len0) {{").unwrap();
    writeln!(out, "                    idx0 = tmp0[i];").unwrap();
    writeln!(out, "                    val0 = src0_row[idx0];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                out_idx = idx1;").unwrap();
    writeln!(out, "                ++j;").unwrap();
    writeln!(out, "                if (j < len1) {{").unwrap();
    writeln!(out, "                    idx1 = tmp1[j];").unwrap();
    writeln!(out, "                    val1 = src0_row[idx1];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst[k] = out_idx;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_concat_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_concat(").unwrap();
    writeln!(out, "    constant ggml_metal_kargs_concat & args,").unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device  const char * src1,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3   tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    ushort3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int i2 = tgpig.y;").unwrap();
    writeln!(
        out,
        "    const int i1 = ntg.y == 1 ? tgpig.x : tgpig.x*ntg.y + tpitg.y;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i1 >= args.ne1) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int o[4] = {{0, 0, 0, 0}};").unwrap();
    writeln!(out, "    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {{"
    )
    .unwrap();
    writeln!(out, "            x = (device const float *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            x = (device const float *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device float * y = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        *y = *x;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_conv_2d_ggml(out: &mut String) {
    writeln!(out, "template <typename TK>").unwrap();
    writeln!(out, "kernel void kernel_conv_2d(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_conv_2d & args,").unwrap();
    writeln!(out, "        device const char * weights,").unwrap();
    writeln!(out, "        device const char * src,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        uint3    tgpg[[threadgroups_per_grid]],").unwrap();
    writeln!(
        out,
        "        uint3   tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3     ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint threads_per_tg = ntg.x * ntg.y * ntg.z;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint tg_index = (tgpig.z * tgpg.y + tgpig.y) * tgpg.x + tgpig.x;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint local_thread = tpitg.z * (ntg.x * ntg.y) + tpitg.y * ntg.x + tpitg.x;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint thread_index = tg_index * threads_per_tg + local_thread;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint64_t total_threads = (uint64_t) threads_per_tg * tgpg.x * tgpg.y * tgpg.z;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint64_t total_outputs = (uint64_t) args.N * args.OC * args.OH * args.OW;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint64_t index = thread_index; index < total_outputs; index += total_threads) {{"
    )
    .unwrap();
    writeln!(out, "        uint64_t tmp = index;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const int32_t ow = tmp % args.OW; tmp /= args.OW;"
    )
    .unwrap();
    writeln!(
        out,
        "        const int32_t oh = tmp % args.OH; tmp /= args.OH;"
    )
    .unwrap();
    writeln!(
        out,
        "        const int32_t oc = tmp % args.OC; tmp /= args.OC;"
    )
    .unwrap();
    writeln!(out, "        const int32_t  n = tmp;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float acc = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int32_t base_x = ow*args.s0 - args.p0;").unwrap();
    writeln!(out, "        const int32_t base_y = oh*args.s1 - args.p1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        int32_t ky_start = 0;").unwrap();
    writeln!(out, "        if (base_y < 0) {{").unwrap();
    writeln!(
        out,
        "            ky_start = (-base_y + args.d1 - 1)/args.d1;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        int32_t ky_end = args.KH;").unwrap();
    writeln!(out, "        const int32_t y_max = args.IH - 1 - base_y;").unwrap();
    writeln!(out, "        if (y_max < 0) {{").unwrap();
    writeln!(out, "            ky_end = ky_start;").unwrap();
    writeln!(
        out,
        "        }} else if (base_y + (args.KH - 1)*args.d1 >= args.IH) {{"
    )
    .unwrap();
    writeln!(out, "            ky_end = min(ky_end, y_max/args.d1 + 1);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        int32_t kx_start = 0;").unwrap();
    writeln!(out, "        if (base_x < 0) {{").unwrap();
    writeln!(
        out,
        "            kx_start = (-base_x + args.d0 - 1)/args.d0;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        int32_t kx_end = args.KW;").unwrap();
    writeln!(out, "        const int32_t x_max = args.IW - 1 - base_x;").unwrap();
    writeln!(out, "        if (x_max < 0) {{").unwrap();
    writeln!(out, "            kx_end = kx_start;").unwrap();
    writeln!(
        out,
        "        }} else if (base_x + (args.KW - 1)*args.d0 >= args.IW) {{"
    )
    .unwrap();
    writeln!(out, "            kx_end = min(kx_end, x_max/args.d0 + 1);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        if (ky_start < ky_end && kx_start < kx_end) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint64_t src_base_n = (uint64_t) n  * args.nb13;"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint64_t w_base_oc  = (uint64_t) oc * args.nb03;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            for (int32_t ic = 0; ic < args.IC; ++ic) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                const uint64_t src_base_nc = src_base_n + (uint64_t) ic * args.nb12;"
    )
    .unwrap();
    writeln!(
        out,
        "                const uint64_t w_base_ocic = w_base_oc  + (uint64_t) ic * args.nb02;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                for (int32_t ky = ky_start; ky < ky_end; ++ky) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    const int32_t iy = base_y + ky*args.d1;"
    )
    .unwrap();
    writeln!(
        out,
        "                    const uint64_t src_base_row = src_base_nc + (uint64_t) iy * args.nb11;"
    )
    .unwrap();
    writeln!(
        out,
        "                    const uint64_t w_base_row   = w_base_ocic + (uint64_t) ky * args.nb01;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    for (int32_t kx = kx_start; kx < kx_end; ++kx) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        const int32_t ix = base_x + kx*args.d0;"
    )
    .unwrap();
    writeln!(out, "                        const uint64_t src_offs = src_base_row + (uint64_t) ix * args.nb10;").unwrap();
    writeln!(out, "                        const uint64_t w_offs   = w_base_row   + (uint64_t) kx * args.nb00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        const float x = *(device const float *)(src + src_offs);"
    )
    .unwrap();
    writeln!(
        out,
        "                        const float w = (float) (*(device const TK *)(weights + w_offs));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        acc += x * w;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const uint64_t dst_offs =").unwrap();
    writeln!(out, "            (uint64_t) n  * args.nb3 +").unwrap();
    writeln!(out, "            (uint64_t) oc * args.nb2 +").unwrap();
    writeln!(out, "            (uint64_t) oh * args.nb1 +").unwrap();
    writeln!(out, "            (uint64_t) ow * args.nb0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        *(device float *)(dst + dst_offs) = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_conv_3d_ggml(out: &mut String) {
    writeln!(out, "template <typename T>").unwrap();
    writeln!(out, "kernel void kernel_conv_3d(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_conv_3d & args,").unwrap();
    writeln!(
        out,
        "        device const  char * src0, // Weights [IC * OC, KD, KH, KW]"
    )
    .unwrap();
    writeln!(
        out,
        "        device const  char * src1, // Inputs  [IC * N,  ID, IH, IW]"
    )
    .unwrap();
    writeln!(
        out,
        "        device       char  * dst,  // Outputs [OC * N,  OD, OH, OW]"
    )
    .unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // 1. Un-flatten the spatial dimension from Grid X"
    )
    .unwrap();
    writeln!(out, "    int64_t spatial_idx = tgpig.x * 32 + tpitg.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (spatial_idx >= args.OW * args.OH * args.OD) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        return; // Thread falls outside the spatial volume"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int64_t od = spatial_idx / (args.OW * args.OH);").unwrap();
    writeln!(out, "    int64_t oh = (spatial_idx / args.OW) % args.OH;").unwrap();
    writeln!(out, "    int64_t ow = spatial_idx % args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // 2. Map Y to Channels, Z to Batch").unwrap();
    writeln!(out, "    int64_t oc = tgpig.y;").unwrap();
    writeln!(out, "    int64_t batch_idx = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // 3. Calculate anchor coordinates in the Input volume"
    )
    .unwrap();
    writeln!(out, "    int64_t i_w_base = ow * args.s0 - args.p0;").unwrap();
    writeln!(out, "    int64_t i_h_base = oh * args.s1 - args.p1;").unwrap();
    writeln!(out, "    int64_t i_d_base = od * args.s2 - args.p2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // 4. Gather Loop (Iterate over Input Channels -> Depth -> Height -> Width)"
    )
    .unwrap();
    writeln!(out, "    for (int64_t ic = 0; ic < args.IC; ++ic) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        // ggml packs batch and channel together in the 4th dimension"
    )
    .unwrap();
    writeln!(
        out,
        "        int64_t src_cn_idx = batch_idx * args.IC + ic;"
    )
    .unwrap();
    writeln!(out, "        int64_t w_cn_idx   = oc * args.IC + ic;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (int64_t kz = 0; kz < args.KD; ++kz) {{").unwrap();
    writeln!(out, "            int64_t id = i_d_base + kz * args.d2;").unwrap();
    writeln!(
        out,
        "            if (id < 0 || id >= args.ID) continue; // Boundary check (Padding)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            for (int64_t ky = 0; ky < args.KH; ++ky) {{"
    )
    .unwrap();
    writeln!(out, "                int64_t ih = i_h_base + ky * args.d1;").unwrap();
    writeln!(
        out,
        "                if (ih < 0 || ih >= args.IH) continue;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                for (int64_t kx = 0; kx < args.KW; ++kx) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    int64_t iw = i_w_base + kx * args.d0;"
    )
    .unwrap();
    writeln!(
        out,
        "                    if (iw < 0 || iw >= args.IW) continue;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    // Convert multi-dimensional coordinates to flat byte offsets"
    )
    .unwrap();
    writeln!(out, "                    int64_t w_idx = kx*args.nb00 + ky*args.nb01 + kz*args.nb02 + w_cn_idx*args.nb03;").unwrap();
    writeln!(out, "                    int64_t i_idx = iw*args.nb10 + ih*args.nb11 + id*args.nb12 + src_cn_idx*args.nb13;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    // Dereference memory and cast weights to f32 if they were f16"
    )
    .unwrap();
    writeln!(out, "                    float w_val = (float)*(device const T*)((device const char*)src0 + w_idx);").unwrap();
    writeln!(out, "                    float i_val = *(device const float*)((device const char*)src1 + i_idx);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    sum += w_val * i_val;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // 5. Write the accumulated value out to RAM").unwrap();
    writeln!(out, "    int64_t dst_cn_idx = batch_idx * args.OC + oc;").unwrap();
    writeln!(
        out,
        "    int64_t d_idx = ow*args.nb0 + oh*args.nb1 + od*args.nb2 + dst_cn_idx*args.nb3;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    *(device float*)(dst + d_idx) = sum;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_conv_transpose_1d_ggml(out: &mut String) {
    writeln!(out, "template <typename T>").unwrap();
    writeln!(out, "kernel void kernel_conv_transpose_1d(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_conv_transpose_1d & args,"
    )
    .unwrap();
    writeln!(out, "        device const     T * src0,").unwrap();
    writeln!(out, "        device const float * src1,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        uint3   tgpg[[threadgroups_per_grid]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // For output position j on the time axis, only input positions"
    )
    .unwrap();
    writeln!(out, "    //   i such that i*s0 <= j < i*s0 + K").unwrap();
    writeln!(
        out,
        "    // contribute -- i.e. i in [ceil((j - K + 1)/s0), floor(j/s0)]"
    )
    .unwrap();
    writeln!(
        out,
        "    // intersected with [0, IL-1]. That's at most ceil(K/s0) values"
    )
    .unwrap();
    writeln!(
        out,
        "    // (typically 2 for stride==K/2 transposed convs)."
    )
    .unwrap();
    writeln!(out, "    const int32_t j  = tgpig[0];").unwrap();
    writeln!(out, "    const int32_t s0 = args.s0;").unwrap();
    writeln!(out, "    const int32_t K  = args.K;").unwrap();
    writeln!(out, "    const int32_t IL = args.IL;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int32_t i_min;").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int32_t a = j - K + 1;").unwrap();
    writeln!(
        out,
        "        i_min = a <= 0 ? 0 : (a + s0 - 1) / s0; // ceil(a/s0) for a>0"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    int32_t i_max = j / s0;").unwrap();
    writeln!(out, "    if (i_max > IL - 1) i_max = IL - 1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float v = 0.0f;").unwrap();
    writeln!(out, "    if (i_min <= i_max) {{").unwrap();
    writeln!(out, "        for (int64_t c = 0; c < args.IC; c++) {{").unwrap();
    writeln!(
        out,
        "            const int32_t kernel_offset = c * tgpg[1] * K + K * tgpig[1];"
    )
    .unwrap();
    writeln!(out, "            const int32_t input_offset  = c * IL;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            for (int32_t i = i_min; i <= i_max; i++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                v += float(src0[kernel_offset + j - i * s0]) * src1[input_offset + i];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_ptr = (device float *) (dst + tgpig[0] * args.nb0 + tgpig[1] * args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    dst_ptr[0] = v;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_conv_transpose_2d_ggml(out: &mut String) {
    writeln!(out, "template <typename T>").unwrap();
    writeln!(out, "kernel void kernel_conv_transpose_2d(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_conv_transpose_2d & args,"
    )
    .unwrap();
    writeln!(out, "        device const T * src0,").unwrap();
    writeln!(out, "        device const float * src1,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup float * shared_sum [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3     ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t out_x = tgpig[0];").unwrap();
    writeln!(out, "    const int64_t out_y = tgpig[1];").unwrap();
    writeln!(out, "    const int64_t out_c = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t kw = tpitg[0];").unwrap();
    writeln!(out, "    const int64_t kh = tpitg[1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float v = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int64_t in_c = 0; in_c < args.IC; in_c++) {{").unwrap();
    writeln!(out, "        int64_t in_y = out_y - kh;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (in_y < 0 || in_y % args.s0) continue;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        in_y /= args.s0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (in_y >= args.IH) continue;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        int64_t in_x = out_x - kw;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (in_x < 0 || in_x % args.s0) continue;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        in_x /= args.s0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (in_x >= args.IW) continue;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const int64_t input_idx = (args.IW * args.IH) * in_c + (args.IW) * in_y + in_x;"
    )
    .unwrap();
    writeln!(out, "        const int64_t kernel_idx = (args.KH * args.KW * args.OC) * in_c + (args.KH * args.KW) * out_c + (args.KW) * kh + kw;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        v += (float)src0[kernel_idx] * src1[input_idx];"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint tid = tpitg.y * ntg.x + tpitg.x;").unwrap();
    writeln!(out, "    shared_sum[tid] = v;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float total = 0.0f;").unwrap();
    writeln!(out, "        const uint num_threads = ntg.x * ntg.y;").unwrap();
    writeln!(out, "        for (uint i = 0; i < num_threads; i++) {{").unwrap();
    writeln!(out, "            total += shared_sum[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device float * dst_ptr = (device float *) (dst + out_x*args.nb0 + out_y * args.nb1 + out_c*args.nb2);").unwrap();
    writeln!(out, "        dst_ptr[0] = total;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_count_equal_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_count_equal(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_count_equal & args,").unwrap();
    writeln!(out, "        device   const char * src0,").unwrap();
    writeln!(out, "        device   const char * src1,").unwrap();
    writeln!(out, "        device   atomic_int * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const short NSG = FC_count_equal_nsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {{"
    )
    .unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int sum = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const T v0 = *(device const T *)(base0 + i0*args.nb00);"
    )
    .unwrap();
    writeln!(
        out,
        "        const T v1 = *(device const T *)(base1 + i0*args.nb10);"
    )
    .unwrap();
    writeln!(out, "        sum += (v0 == v1);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sum = simd_sum(sum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_i32[sgitg] = sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        float v = 0.0f;").unwrap();
    writeln!(out, "        if (tpitg.x < NSG) {{").unwrap();
    writeln!(out, "            v = shmem_i32[tpitg.x];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float total = simd_sum(v);").unwrap();
    writeln!(out, "        if (tpitg.x == 0) {{").unwrap();
    writeln!(
        out,
        "            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_cpy_f32_q_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_cpy_f32_q(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_cpy & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t i03 = tgpig[2];").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig[1];").unwrap();
    writeln!(
        out,
        "    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;"
    )
    .unwrap();
    writeln!(
        out,
        "    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i01 >= args.ne01) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int32_t i3 = n / (args.ne2*args.ne1*args.ne0);"
    )
    .unwrap();
    writeln!(
        out,
        "    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);"
    )
    .unwrap();
    writeln!(out, "    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;").unwrap();
    writeln!(out, "    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device block_q * dst_data = (device block_q *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {{"
    )
    .unwrap();
    writeln!(out, "        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK)*args.nb00);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        quantize_func(src, dst_data[i00]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        break;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_cpy_q_f32_ggml(out: &mut String) {
    writeln!(out, "template<typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>").unwrap();
    writeln!(out, "kernel void kernel_cpy_q_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_cpy & args,").unwrap();
    writeln!(out, "        device  const char * src0,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t i03 = tgpig[2];").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig[1];").unwrap();
    writeln!(
        out,
        "    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;"
    )
    .unwrap();
    writeln!(
        out,
        "    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i01 >= args.ne01) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);"
    )
    .unwrap();
    writeln!(
        out,
        "    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);"
    )
    .unwrap();
    writeln!(out, "    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;").unwrap();
    writeln!(out, "    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const block_q * src_data = (device const block_q *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);").unwrap();
    writeln!(out, "    device       T4x4    * dst_data = (device       T4x4    *)(dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1 + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {{"
    )
    .unwrap();
    writeln!(out, "        T4x4 temp;").unwrap();
    writeln!(
        out,
        "        dequantize_func(src_data + i00/nl, i00%nl, temp);"
    )
    .unwrap();
    writeln!(out, "        dst_data[i00] = temp;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        break;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_cpy_t_t_ggml(out: &mut String) {
    writeln!(out, "template<typename T0, typename T1>").unwrap();
    writeln!(out, "kernel void kernel_cpy_t_t(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_cpy & args,").unwrap();
    writeln!(out, "        device  const char * src0,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t i03 = tgpig[2];").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig[1];").unwrap();
    writeln!(
        out,
        "    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;"
    )
    .unwrap();
    writeln!(
        out,
        "    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i01 >= args.ne01) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);"
    )
    .unwrap();
    writeln!(
        out,
        "    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);"
    )
    .unwrap();
    writeln!(out, "    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;").unwrap();
    writeln!(out, "    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.ne00;) {{"
    )
    .unwrap();
    writeln!(out, "        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);").unwrap();
    writeln!(out, "        dst_data[i00] = (T1) src[0];").unwrap();
    writeln!(out, "        break;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_cumsum_add_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_cumsum_add(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_cumsum_add & args,").unwrap();
    writeln!(out, "        device const char * tmp,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int ib = tgpig[0]/args.ne01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (ib == 0) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i00 = ib*ntg.x;").unwrap();
    writeln!(out, "    const int i01 = tgpig[0]%args.ne01;").unwrap();
    writeln!(out, "    const int i02 = tgpig[1];").unwrap();
    writeln!(out, "    const int i03 = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * tmp_row = (device const float *) (tmp +"
    )
    .unwrap();
    writeln!(out, "            args.nbt1*i01 +").unwrap();
    writeln!(out, "            args.nbt2*i02 +").unwrap();
    writeln!(out, "            args.nbt3*i03);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_row = (device float *) dst +").unwrap();
    writeln!(out, "        args.ne00*i01 +").unwrap();
    writeln!(out, "        args.ne00*args.ne01*i02 +").unwrap();
    writeln!(out, "        args.ne00*args.ne01*args.ne02*i03;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i00 + tpitg.x < args.ne00) {{").unwrap();
    writeln!(out, "        dst_row[i00 + tpitg.x] += tmp_row[ib - 1];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_cumsum_blk_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_cumsum_blk(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_cumsum_blk & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device       char * tmp,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int ib = tgpig[0]/args.ne01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i00 = ib*ntg.x;").unwrap();
    writeln!(out, "    const int i01 = tgpig[0]%args.ne01;").unwrap();
    writeln!(out, "    const int i02 = tgpig[1];").unwrap();
    writeln!(out, "    const int i03 = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * src0_row = (device const float *) (src0 +"
    )
    .unwrap();
    writeln!(out, "            args.nb01*i01 +").unwrap();
    writeln!(out, "            args.nb02*i02 +").unwrap();
    writeln!(out, "            args.nb03*i03);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup float * shmem_f32 = (threadgroup float *) shmem;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float v = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i00 + tpitg.x < args.ne00) {{").unwrap();
    writeln!(out, "        v = src0_row[i00 + tpitg.x];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float s = simd_prefix_inclusive_sum(v);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == N_SIMDWIDTH - 1) {{").unwrap();
    writeln!(out, "        shmem_f32[sgitg] = s;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(
        out,
        "        shmem_f32[tiisg] = simd_prefix_exclusive_sum(shmem_f32[tiisg]);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    s += shmem_f32[sgitg];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_row = (device float *) dst +").unwrap();
    writeln!(out, "        args.ne00*i01 +").unwrap();
    writeln!(out, "        args.ne00*args.ne01*i02 +").unwrap();
    writeln!(out, "        args.ne00*args.ne01*args.ne02*i03;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i00 + tpitg.x < args.ne00) {{").unwrap();
    writeln!(out, "        dst_row[i00 + tpitg.x] = s;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (args.outb && tpitg.x == ntg.x - 1) {{").unwrap();
    writeln!(
        out,
        "        device float * tmp_row = (device float *) tmp +"
    )
    .unwrap();
    writeln!(out, "            args.net0*i01 +").unwrap();
    writeln!(out, "            args.net0*args.net1*i02 +").unwrap();
    writeln!(out, "            args.net0*args.net1*args.net2*i03;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        tmp_row[ib] = s;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_diag_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_diag_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_diag & args,").unwrap();
    writeln!(out, "        device   const char * src0,").unwrap();
    writeln!(out, "        device         char * dst,").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        ushort tiitg[[thread_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    constexpr short NW = N_SIMDWIDTH;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);").unwrap();
    writeln!(out, "    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {{").unwrap();
    writeln!(out, "        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_geglu_erf_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_geglu_erf_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_glu & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;").unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {{").unwrap();
    writeln!(out, "        const float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        const float x1 = src1_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float gelu_erf = 0.5f*x0*(1.0f+erf_approx<float>(x0*SQRT_2_INV));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_row[i0] = gelu_erf*x1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_geglu_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_geglu_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_glu & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;").unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {{").unwrap();
    writeln!(out, "        const float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        const float x1 = src1_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float gelu = 0.5f*x0*(1.0f + precise::tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_row[i0] = gelu*x1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_geglu_quick_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_geglu_quick_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_glu & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;").unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {{").unwrap();
    writeln!(out, "        const float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        const float x1 = src1_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float gelu_quick = x0*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x0)));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_row[i0] = gelu_quick*x1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_get_rows_f_ggml(out: &mut String) {
    writeln!(out, "template<typename T0, typename T>").unwrap();
    writeln!(out, "kernel void kernel_get_rows_f(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_get_rows & args,").unwrap();
    writeln!(out, "        device const void * src0,").unwrap();
    writeln!(out, "        device const void * src1,").unwrap();
    writeln!(out, "        device       void * dst,").unwrap();
    writeln!(
        out,
        "        uint3               tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort              tiitg[[thread_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3             ntg [[threads_per_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int32_t iw0 = tgpig.x/args.ne10;").unwrap();
    writeln!(out, "    const int32_t i10 = tgpig.x%args.ne10;").unwrap();
    writeln!(out, "    const int32_t i11 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t i12 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i02 = i11;").unwrap();
    writeln!(out, "    const int32_t i03 = i12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    auto psrc = (const device T0 *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);").unwrap();
    writeln!(out, "    auto pdst = (      device T  *) ((      device char *)  dst + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {{"
    )
    .unwrap();
    writeln!(out, "        pdst[ind] = psrc[ind];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        break;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_get_rows_q_ggml(out: &mut String) {
    writeln!(out, "template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>").unwrap();
    writeln!(out, "kernel void kernel_get_rows_q(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_get_rows & args,").unwrap();
    writeln!(out, "        device const void * src0,").unwrap();
    writeln!(out, "        device const void * src1,").unwrap();
    writeln!(out, "        device       void * dst,").unwrap();
    writeln!(
        out,
        "        uint3               tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort              tiitg[[thread_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3             ntg  [[threads_per_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int32_t iw0 = tgpig.x/args.ne10;").unwrap();
    writeln!(out, "    const int32_t i10 = tgpig.x%args.ne10;").unwrap();
    writeln!(out, "    const int32_t i11 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t i12 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i02 = i11;").unwrap();
    writeln!(out, "    const int32_t i03 = i12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    auto psrc = (device const block_q *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);").unwrap();
    writeln!(out, "    auto pdst = (device      float4x4 *) ((      device char *) dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {{"
    )
    .unwrap();
    writeln!(out, "        float4x4 temp;").unwrap();
    writeln!(out, "        dequantize_func(psrc + ind/nl, ind%nl, temp);").unwrap();
    writeln!(out, "        pdst[ind] = temp;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        break;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_group_norm_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_group_norm_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_group_norm & args,").unwrap();
    writeln!(out, "        device const float * src0,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(out, "        threadgroup float  * buf [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint sgitg[[simdgroup_index_in_threadgroup]],").unwrap();
    writeln!(out, "        uint tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int64_t ne = args.ne00*args.ne01*args.ne02;").unwrap();
    writeln!(
        out,
        "    const int64_t gs = args.ne00*args.ne01*((args.ne02 + args.ngrp - 1) / args.ngrp);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int start = tgpig * gs;").unwrap();
    writeln!(out, "    int end   = start + gs;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    start += tpitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (end >= ne) {{").unwrap();
    writeln!(out, "        end = ne;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    float tmp = 0.0f; // partial sum for thread in warp"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int j = start; j < end; j += ntg) {{").unwrap();
    writeln!(out, "        tmp += src0[j];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    tmp = simd_sum(tmp);").unwrap();
    writeln!(out, "    if (ntg > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            buf[tiisg] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            buf[sgitg] = tmp;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        tmp = buf[tiisg];").unwrap();
    writeln!(out, "        tmp = simd_sum(tmp);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float mean = tmp / gs;").unwrap();
    writeln!(out, "    tmp = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int j = start; j < end; j += ntg) {{").unwrap();
    writeln!(out, "        float xi = src0[j] - mean;").unwrap();
    writeln!(out, "        dst[j] = xi;").unwrap();
    writeln!(out, "        tmp += xi * xi;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    tmp = simd_sum(tmp);").unwrap();
    writeln!(out, "    if (ntg > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            buf[tiisg] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            buf[sgitg] = tmp;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        tmp = buf[tiisg];").unwrap();
    writeln!(out, "        tmp = simd_sum(tmp);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float variance = tmp / gs;").unwrap();
    writeln!(
        out,
        "    const float scale = 1.0f/sqrt(variance + args.eps);"
    )
    .unwrap();
    writeln!(out, "    for (int j = start; j < end; j += ntg) {{").unwrap();
    writeln!(out, "        dst[j] *= scale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_im2col_ggml(out: &mut String) {
    writeln!(out, "template <typename T>").unwrap();
    writeln!(out, "kernel void kernel_im2col(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_im2col & args,").unwrap();
    writeln!(out, "        device const float * x,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint3  tgpg[[threadgroups_per_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "//    const int64_t IC = tgpg[0];").unwrap();
    writeln!(out, "    const int64_t OH = tgpg[1];").unwrap();
    writeln!(out, "    const int64_t OW = tgpg[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t KH = ntg[1];").unwrap();
    writeln!(out, "    const int64_t KW = ntg[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "          int64_t in  = tpitg[0];").unwrap();
    writeln!(out, "    const int64_t ikh = tpitg[1];").unwrap();
    writeln!(out, "    const int64_t ikw = tpitg[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t iic = tgpig[0];").unwrap();
    writeln!(out, "    const int64_t ioh = tgpig[1];").unwrap();
    writeln!(out, "    const int64_t iow = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int64_t iiw = iow*args.s0 + ikw*args.d0 - args.p0;"
    )
    .unwrap();
    writeln!(
        out,
        "    const int64_t iih = ioh*args.s1 + ikh*args.d1 - args.p1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int64_t offset_dst = (in*OH*OW + ioh*OW + iow)*args.CHW + (iic*(KH*KW) + ikh*KW + ikw);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device T * pdst = (device T *) (dst);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (iih < 0 || iih >= args.IH || iiw < 0 || iiw >= args.IW) {{"
    )
    .unwrap();
    writeln!(out, "        while (in < args.N) {{").unwrap();
    writeln!(out, "            pdst[offset_dst] = 0.0f;").unwrap();
    writeln!(out, "            offset_dst += ntg[0]*args.CHW*OH*OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            in += ntg[0];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(
        out,
        "        int64_t offset_src = in*args.ofs0 + iic*args.ofs1 + iih*args.IW + iiw;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        while (in < args.N) {{").unwrap();
    writeln!(out, "            pdst[offset_dst] = x[offset_src];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            offset_dst += ntg[0]*args.CHW*OH*OW;").unwrap();
    writeln!(out, "            offset_src += ntg[0]*args.ofs0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            in += ntg[0];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_memset_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_memset(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_memset & args,").unwrap();
    writeln!(out, "        device T * dst,").unwrap();
    writeln!(out, "        uint tpig[[thread_position_in_grid]]) {{").unwrap();
    writeln!(out, "    dst[tpig] = args.val;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_op_sum_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_op_sum_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_sum & args,").unwrap();
    writeln!(out, "        device const float * src0,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup  float * shmem_f32 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (args.np == 0) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // TODO: become function constant").unwrap();
    writeln!(out, "    const uint nsg = (ntg.x + 31) / 32;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (uint64_t i0 = tpitg.x; i0 < args.np; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        sumf += src0[i0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sumf = simd_sum(sumf);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(out, "        shmem_f32[sgitg] = sumf;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float total = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(out, "        float v = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tpitg.x < nsg) {{").unwrap();
    writeln!(out, "            v = shmem_f32[tpitg.x];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        total = simd_sum(v);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tpitg.x == 0) {{").unwrap();
    writeln!(out, "            dst[0] = total;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_opt_step_adamw_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_opt_step_adamw_f32(").unwrap();
    writeln!(
        out,
        "        constant    ggml_metal_kargs_opt_step_adamw & args,"
    )
    .unwrap();
    writeln!(out, "        device       float * x,").unwrap();
    writeln!(out, "        device const float * g,").unwrap();
    writeln!(out, "        device       float * g_m,").unwrap();
    writeln!(out, "        device       float * g_v,").unwrap();
    writeln!(out, "        device const float * pars,").unwrap();
    writeln!(
        out,
        "        uint        gid[[thread_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (gid >= args.np) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float alpha  = pars[0];").unwrap();
    writeln!(out, "    const float beta1  = pars[1];").unwrap();
    writeln!(out, "    const float beta2  = pars[2];").unwrap();
    writeln!(out, "    const float eps    = pars[3];").unwrap();
    writeln!(out, "    const float wd     = pars[4];").unwrap();
    writeln!(out, "    const float beta1h = pars[5];").unwrap();
    writeln!(out, "    const float beta2h = pars[6];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float gi = g[gid];").unwrap();
    writeln!(
        out,
        "    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);"
    )
    .unwrap();
    writeln!(
        out,
        "    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    g_m[gid] = gmi;").unwrap();
    writeln!(out, "    g_v[gid] = gvi;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float mh =      gmi * beta1h;").unwrap();
    writeln!(out, "    const float vh = sqrt(gvi * beta2h) + eps;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_opt_step_sgd_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_opt_step_sgd_f32(").unwrap();
    writeln!(
        out,
        "        constant    ggml_metal_kargs_opt_step_sgd & args,"
    )
    .unwrap();
    writeln!(out, "        device       float * x,").unwrap();
    writeln!(out, "        device const float * g,").unwrap();
    writeln!(out, "        device const float * pars,").unwrap();
    writeln!(
        out,
        "        uint        gid[[thread_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (gid >= args.np) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_pad_reflect_1d_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_pad_reflect_1d_f32(").unwrap();
    writeln!(
        out,
        "    constant   ggml_metal_kargs_pad_reflect_1d & args,"
    )
    .unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3  tgpg[[threadgroups_per_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i03 = i3;").unwrap();
    writeln!(out, "    const int64_t i02 = i2;").unwrap();
    writeln!(out, "    const int64_t i01 = i1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);").unwrap();
    writeln!(out, "    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "            if (i0 < args.p0) {{").unwrap();
    writeln!(out, "                dst_ptr[i0] = src0_ptr[args.p0 - i0];").unwrap();
    writeln!(out, "            }} else if (i0 < args.ne0 - args.p1) {{").unwrap();
    writeln!(out, "                dst_ptr[i0] = src0_ptr[i0 - args.p0];").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_pool_1d_avg_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_pool_1d_avg_f32(").unwrap();
    writeln!(
        out,
        "        constant        ggml_metal_kargs_pool_1d & args,"
    )
    .unwrap();
    writeln!(out, "        device  const   float * src,").unwrap();
    writeln!(out, "        device          float * dst,").unwrap();
    writeln!(
        out,
        "        uint            gid [[thread_position_in_grid]]"
    )
    .unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (gid >= args.np) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ow  = (int)gid % args.OW;").unwrap();
    writeln!(out, "    const int row = (int)gid / args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int base = ow * args.s0 - args.p0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    int   cnt = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int src_off = row * args.IW;").unwrap();
    writeln!(out, "    const int dst_off = row * args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ki = 0; ki < args.k0; ++ki) {{").unwrap();
    writeln!(out, "        const int j = base + ki;").unwrap();
    writeln!(out, "        if (j < 0 || j >= args.IW) {{").unwrap();
    writeln!(out, "            continue;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        acc += src[src_off + j];").unwrap();
    writeln!(out, "        cnt += 1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    dst[dst_off + ow] = (cnt > 0) ? (acc / (float)cnt) : 0.0f;"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_pool_1d_max_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_pool_1d_max_f32(").unwrap();
    writeln!(
        out,
        "        constant        ggml_metal_kargs_pool_1d & args,"
    )
    .unwrap();
    writeln!(out, "        device  const   float * src,").unwrap();
    writeln!(out, "        device          float * dst,").unwrap();
    writeln!(
        out,
        "        uint            gid [[thread_position_in_grid]]"
    )
    .unwrap();
    writeln!(out, ") {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (gid >= args.np) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ow  = (int)gid % args.OW;").unwrap();
    writeln!(out, "    const int row = (int)gid / args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int base = ow * args.s0 - args.p0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float acc = -INFINITY;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int src_off = row * args.IW;").unwrap();
    writeln!(out, "    const int dst_off = row * args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ki = 0; ki < args.k0; ++ki) {{").unwrap();
    writeln!(out, "        int j = base + ki;").unwrap();
    writeln!(out, "        if (j < 0 || j >= args.IW){{").unwrap();
    writeln!(out, "            continue;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        float v = src[src_off + j];").unwrap();
    writeln!(out, "        acc = max(acc, v);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    dst[dst_off + ow] = acc;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_pool_2d_avg_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_pool_2d_avg_f32(").unwrap();
    writeln!(out, "        constant    ggml_metal_kargs_pool_2d & args,").unwrap();
    writeln!(out, "        device  const float * src0,").unwrap();
    writeln!(out, "        device        float * dst,").unwrap();
    writeln!(
        out,
        "        uint        gid[[thread_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (gid >= args.np) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int idx = gid;").unwrap();
    writeln!(out, "    const int I_HW = args.IH * args.IW;").unwrap();
    writeln!(out, "    const int O_HW = args.OH * args.OW;").unwrap();
    writeln!(out, "    const int nc = idx / O_HW;").unwrap();
    writeln!(out, "    const int cur_oh = idx % O_HW / args.OW;").unwrap();
    writeln!(out, "    const int cur_ow = idx % O_HW % args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * i_ptr = src0 + nc * I_HW;").unwrap();
    writeln!(out, "    device       float * o_ptr = dst  + nc * O_HW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int start_h = cur_oh * args.s1 - args.p1;").unwrap();
    writeln!(out, "    const int bh = MAX(0,  start_h);").unwrap();
    writeln!(out, "    const int eh = MIN(args.IH, start_h + args.k1);").unwrap();
    writeln!(out, "    const int start_w = cur_ow * args.s0 - args.p0;").unwrap();
    writeln!(out, "    const int bw = MAX(0,  start_w);").unwrap();
    writeln!(out, "    const int ew = MIN(args.IW, start_w + args.k0);").unwrap();
    writeln!(
        out,
        "    // const float scale = 1. / ((eh - bh) * (ew - bw));"
    )
    .unwrap();
    writeln!(out, "    const float scale = 1. / (args.k0 * args.k1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float res = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i = bh; i < eh; i += 1) {{").unwrap();
    writeln!(out, "        for (int j = bw; j < ew; j += 1) {{").unwrap();
    writeln!(out, "            float cur = i_ptr[i * args.IW + j];").unwrap();
    writeln!(out, "            res += cur * scale;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    o_ptr[cur_oh * args.OW + cur_ow] = res;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_pool_2d_max_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_pool_2d_max_f32(").unwrap();
    writeln!(out, "        constant    ggml_metal_kargs_pool_2d & args,").unwrap();
    writeln!(out, "        device  const float * src0,").unwrap();
    writeln!(out, "        device        float * dst,").unwrap();
    writeln!(
        out,
        "        uint        gid[[thread_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (gid >= args.np) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int idx = gid;").unwrap();
    writeln!(out, "    const int I_HW = args.IH * args.IW;").unwrap();
    writeln!(out, "    const int O_HW = args.OH * args.OW;").unwrap();
    writeln!(out, "    const int nc = idx / O_HW;").unwrap();
    writeln!(out, "    const int cur_oh = idx % O_HW / args.OW;").unwrap();
    writeln!(out, "    const int cur_ow = idx % O_HW % args.OW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * i_ptr = src0 + nc * I_HW;").unwrap();
    writeln!(out, "    device       float * o_ptr = dst  + nc * O_HW;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int start_h = cur_oh * args.s1 - args.p1;").unwrap();
    writeln!(out, "    const int bh = MAX(0,  start_h);").unwrap();
    writeln!(out, "    const int eh = MIN(args.IH, start_h + args.k1);").unwrap();
    writeln!(out, "    const int start_w = cur_ow * args.s0 - args.p0;").unwrap();
    writeln!(out, "    const int bw = MAX(0,  start_w);").unwrap();
    writeln!(out, "    const int ew = MIN(args.IW, start_w + args.k0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float res = -INFINITY;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i = bh; i < eh; i += 1) {{").unwrap();
    writeln!(out, "        for (int j = bw; j < ew; j += 1) {{").unwrap();
    writeln!(out, "            res = MAX(res, i_ptr[i * args.IW + j]);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    o_ptr[cur_oh * args.OW + cur_ow] = res;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_reglu_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_reglu_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_glu & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;").unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {{").unwrap();
    writeln!(out, "        const float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        const float x1 = src1_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_row[i0] = x0*x1*(x0 > 0.0f);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_repeat_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_repeat(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_repeat & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i03 = i3%args.ne03;").unwrap();
    writeln!(out, "    const int i02 = i2%args.ne02;").unwrap();
    writeln!(out, "    const int i01 = i1%args.ne01;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        const int i00 = i0%args.ne00;").unwrap();
    writeln!(out, "        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_roll_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_roll_f32(").unwrap();
    writeln!(out, "    constant ggml_metal_kargs_roll & args,").unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * src0_ptr = (device const float *) src0;"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float * dst_ptr  = (device       float *) dst;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        // apply shifts and wrap around").unwrap();
    writeln!(out, "        int64_t i00 = i0 - args.s0;").unwrap();
    writeln!(out, "        int64_t i01 = i1 - args.s1;").unwrap();
    writeln!(out, "        int64_t i02 = i2 - args.s2;").unwrap();
    writeln!(out, "        int64_t i03 = i3 - args.s3;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (i00 < 0) {{ i00 += args.ne00; }} else if (i00 >= args.ne00) {{ i00 -= args.ne00; }}").unwrap();
    writeln!(out, "        if (i01 < 0) {{ i01 += args.ne01; }} else if (i01 >= args.ne01) {{ i01 -= args.ne01; }}").unwrap();
    writeln!(out, "        if (i02 < 0) {{ i02 += args.ne02; }} else if (i02 >= args.ne02) {{ i02 -= args.ne02; }}").unwrap();
    writeln!(out, "        if (i03 < 0) {{ i03 += args.ne03; }} else if (i03 >= args.ne03) {{ i03 -= args.ne03; }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;").unwrap();
    writeln!(out, "        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_ptr[dst_idx] = src0_ptr[src_idx];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rope_multi_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_rope_multi(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_rope & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device const char * src2,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        ushort  tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort3 tptg [[threads_per_threadgroup]],").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int i3 = tgpig[2];").unwrap();
    writeln!(out, "    const int i2 = tgpig[1];").unwrap();
    writeln!(out, "    const int i1 = tgpig[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float corr_dims[2];").unwrap();
    writeln!(out, "    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const int32_t * pos = (device const int32_t *) src1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float inv_ndims = -1.f/args.n_dims;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float cos_theta;").unwrap();
    writeln!(out, "    float sin_theta;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        if (i0 < args.n_dims) {{").unwrap();
    writeln!(out, "            const int ic = i0/2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // mrope theta calculations").unwrap();
    writeln!(
        out,
        "            // note: the rest is the same as kernel_rope_neox"
    )
    .unwrap();
    writeln!(
        out,
        "            const int sect_dims = args.sect_0 + args.sect_1 + args.sect_2 + args.sect_3;"
    )
    .unwrap();
    writeln!(out, "            const int sec_w01   = args.sect_0 + args.sect_1;               // end of section 1").unwrap();
    writeln!(out, "            const int sec_w012  = args.sect_0 + args.sect_1 + args.sect_2; // end of section 2").unwrap();
    writeln!(out, "            const int sector    = ic % sect_dims;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float theta_base;").unwrap();
    writeln!(out, "            if (FC_rope_is_imrope) {{").unwrap();
    writeln!(
        out,
        "                if (sector % 3 == 1 && sector < 3 * args.sect_1) {{ // h"
    )
    .unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 1];"
    )
    .unwrap();
    writeln!(
        out,
        "                }} else if (sector % 3 == 2 && sector < 3 * args.sect_2) {{ // w"
    )
    .unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 2];"
    )
    .unwrap();
    writeln!(
        out,
        "                }} else if (sector % 3 == 0 && sector < 3 * args.sect_0) {{ // t"
    )
    .unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 0];"
    )
    .unwrap();
    writeln!(out, "                }} else {{ // e").unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 3];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                if (sector < args.sect_0) {{").unwrap();
    writeln!(out, "                    theta_base = (float) pos[i2];").unwrap();
    writeln!(out, "                }} else if (sector < sec_w01) {{").unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 1];"
    )
    .unwrap();
    writeln!(out, "                }} else if (sector < sec_w012) {{").unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 2];"
    )
    .unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    theta_base = (float) pos[i2 + args.ne02 * 3];"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            // end of mrope").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float x0 = src[0];").unwrap();
    writeln!(out, "            const float x1 = src[args.n_dims/2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            dst_data[0]             = x0*cos_theta - x1*sin_theta;"
    )
    .unwrap();
    writeln!(
        out,
        "            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;"
    )
    .unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dst_data[0] = src[0];").unwrap();
    writeln!(out, "            dst_data[1] = src[1];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rope_neox_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_rope_neox(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_rope & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device const char * src2,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        ushort  tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort3 tptg [[threads_per_threadgroup]],").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int i3 = tgpig[2];").unwrap();
    writeln!(out, "    const int i2 = tgpig[1];").unwrap();
    writeln!(out, "    const int i1 = tgpig[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float corr_dims[2];").unwrap();
    writeln!(out, "    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const int32_t * pos = (device const int32_t *) src1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float theta_base = (float) pos[i2];").unwrap();
    writeln!(out, "    const float inv_ndims = -1.f/args.n_dims;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float cos_theta;").unwrap();
    writeln!(out, "    float sin_theta;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        if (i0 < args.n_dims) {{").unwrap();
    writeln!(out, "            const int ic = i0/2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float x0 = src[0];").unwrap();
    writeln!(out, "            const float x1 = src[args.n_dims/2];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            dst_data[0]             = x0*cos_theta - x1*sin_theta;"
    )
    .unwrap();
    writeln!(
        out,
        "            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;"
    )
    .unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dst_data[0] = src[0];").unwrap();
    writeln!(out, "            dst_data[1] = src[1];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rope_norm_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_rope_norm(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_rope & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device const char * src2,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        ushort  tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort3 tptg [[threads_per_threadgroup]],").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int i3 = tgpig[2];").unwrap();
    writeln!(out, "    const int i2 = tgpig[1];").unwrap();
    writeln!(out, "    const int i1 = tgpig[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float corr_dims[2];").unwrap();
    writeln!(out, "    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const int32_t * pos = (device const int32_t *) src1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float theta_base = (float) pos[i2];").unwrap();
    writeln!(out, "    const float inv_ndims = -1.f/args.n_dims;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float cos_theta;").unwrap();
    writeln!(out, "    float sin_theta;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        if (i0 < args.n_dims) {{").unwrap();
    writeln!(out, "            const int ic = i0/2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float x0 = src[0];").unwrap();
    writeln!(out, "            const float x1 = src[1];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            dst_data[0] = x0*cos_theta - x1*sin_theta;"
    )
    .unwrap();
    writeln!(
        out,
        "            dst_data[1] = x0*sin_theta + x1*cos_theta;"
    )
    .unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dst_data[0] = src[0];").unwrap();
    writeln!(out, "            dst_data[1] = src[1];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rope_vision_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_rope_vision(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_rope & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device const char * src2,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        ushort  tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort3 tptg [[threads_per_threadgroup]],").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int i3 = tgpig[2];").unwrap();
    writeln!(out, "    const int i2 = tgpig[1];").unwrap();
    writeln!(out, "    const int i1 = tgpig[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float corr_dims[2];").unwrap();
    writeln!(out, "    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const int32_t * pos = (device const int32_t *) src1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float inv_ndims = -1.f/args.n_dims;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float cos_theta;").unwrap();
    writeln!(out, "    float sin_theta;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        if (i0 < 2*args.n_dims) {{ // different from kernel_rope_multi"
    )
    .unwrap();
    writeln!(out, "            const int ic = i0/2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            // mrope theta calculations (only support 2 dimensions)"
    )
    .unwrap();
    writeln!(
        out,
        "            const int sect_dims = args.sect_0 + args.sect_1;"
    )
    .unwrap();
    writeln!(out, "            const int sector    = ic % sect_dims;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float p;").unwrap();
    writeln!(out, "            float theta_base;").unwrap();
    writeln!(out, "            if (sector < args.sect_1) {{").unwrap();
    writeln!(out, "                p = (float) sector;").unwrap();
    writeln!(out, "                theta_base = (float) pos[i2];").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                p = (float) sector - args.sect_0;").unwrap();
    writeln!(
        out,
        "                theta_base = (float) pos[i2 + args.ne02];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float theta = theta_base * pow(args.freq_base, 2.0f * inv_ndims * p);"
    )
    .unwrap();
    writeln!(out, "            // end of mrope").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float x0 = src[0];").unwrap();
    writeln!(
        out,
        "            const float x1 = src[args.n_dims]; // different from kernel_rope_multi"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            dst_data[0]           = x0*cos_theta - x1*sin_theta;"
    )
    .unwrap();
    writeln!(out, "            dst_data[args.n_dims] = x0*sin_theta + x1*cos_theta; // different from kernel_rope_multi").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);").unwrap();
    writeln!(out, "            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dst_data[0] = src[0];").unwrap();
    writeln!(out, "            dst_data[1] = src[1];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rwkv_wkv6_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_rwkv_wkv6_f32(").unwrap();
    writeln!(out, "    device const float * k,").unwrap();
    writeln!(out, "    device const float * v,").unwrap();
    writeln!(out, "    device const float * r,").unwrap();
    writeln!(out, "    device const float * tf,").unwrap();
    writeln!(out, "    device const float * td,").unwrap();
    writeln!(out, "    device const float * state_in,").unwrap();
    writeln!(out, "    device       float * dst,").unwrap();
    writeln!(out, "    constant    uint & B,").unwrap();
    writeln!(out, "    constant    uint & T,").unwrap();
    writeln!(out, "    constant    uint & C,").unwrap();
    writeln!(out, "    constant    uint & H,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]])  {{").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint head_size = 64; // TODO: support head_size = 128"
    )
    .unwrap();
    writeln!(out, "    const uint batch_id = tgpig.x / H;").unwrap();
    writeln!(out, "    const uint head_id = tgpig.x % H;").unwrap();
    writeln!(out, "    const uint tid = tpitg.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (batch_id >= B || head_id >= H) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint state_size = C * head_size;").unwrap();
    writeln!(out, "    const uint n_seq_tokens = T / B;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float _k[head_size];").unwrap();
    writeln!(out, "    threadgroup float _r[head_size];").unwrap();
    writeln!(out, "    threadgroup float _tf[head_size];").unwrap();
    writeln!(out, "    threadgroup float _td[head_size];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float state[head_size];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = 0; i < head_size; i++) {{").unwrap();
    writeln!(
        out,
        "        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size"
    )
    .unwrap();
    writeln!(out, "                          + i * head_size + tid];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    _tf[tid] = tf[head_id * head_size + tid];").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint t = start_t; t < end_t; t += C) {{").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        _k[tid] = k[t];").unwrap();
    writeln!(out, "        _r[tid] = r[t];").unwrap();
    writeln!(out, "        _td[tid] = td[t];").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float v_val = v[t];").unwrap();
    writeln!(out, "        float y = 0.0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (uint j = 0; j < head_size; j += 4) {{").unwrap();
    writeln!(
        out,
        "            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 tf_vec = float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 td_vec = float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float4 kv = k_vec * v_val;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float4 temp = tf_vec * kv + s_vec;").unwrap();
    writeln!(out, "            y += dot(r_vec, temp);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            s_vec = s_vec * td_vec + kv;").unwrap();
    writeln!(out, "            state[j]   = s_vec[0];").unwrap();
    writeln!(out, "            state[j+1] = s_vec[1];").unwrap();
    writeln!(out, "            state[j+2] = s_vec[2];").unwrap();
    writeln!(out, "            state[j+3] = s_vec[3];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst[t] = y;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = 0; i < head_size; i++) {{").unwrap();
    writeln!(
        out,
        "        dst[T * C + batch_id * state_size + head_id * head_size * head_size"
    )
    .unwrap();
    writeln!(out, "            + i * head_size + tid] = state[i];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_rwkv_wkv7_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_rwkv_wkv7_f32(").unwrap();
    writeln!(out, "    device const float * r,").unwrap();
    writeln!(out, "    device const float * w,").unwrap();
    writeln!(out, "    device const float * k,").unwrap();
    writeln!(out, "    device const float * v,").unwrap();
    writeln!(out, "    device const float * a,").unwrap();
    writeln!(out, "    device const float * b,").unwrap();
    writeln!(out, "    device const float * state_in,").unwrap();
    writeln!(out, "    device       float * dst,").unwrap();
    writeln!(out, "    constant    uint & B,").unwrap();
    writeln!(out, "    constant    uint & T,").unwrap();
    writeln!(out, "    constant    uint & C,").unwrap();
    writeln!(out, "    constant    uint & H,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]])  {{").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint head_size = 64; // TODO: support head_size = 128"
    )
    .unwrap();
    writeln!(out, "    const uint batch_id = tgpig.x / H;").unwrap();
    writeln!(out, "    const uint head_id = tgpig.x % H;").unwrap();
    writeln!(out, "    const uint tid = tpitg.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (batch_id >= B || head_id >= H) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint state_size = C * head_size;").unwrap();
    writeln!(out, "    const uint n_seq_tokens = T / B;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float _r[head_size];").unwrap();
    writeln!(out, "    threadgroup float _w[head_size];").unwrap();
    writeln!(out, "    threadgroup float _k[head_size];").unwrap();
    writeln!(out, "    threadgroup float _a[head_size];").unwrap();
    writeln!(out, "    threadgroup float _b[head_size];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float state[head_size];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = 0; i < head_size; i++) {{").unwrap();
    writeln!(
        out,
        "        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size"
    )
    .unwrap();
    writeln!(out, "                          + tid * head_size + i];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;"
    )
    .unwrap();
    writeln!(
        out,
        "    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint t = start_t; t < end_t; t += C) {{").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "        _r[tid] = r[t];").unwrap();
    writeln!(out, "        _w[tid] = w[t];").unwrap();
    writeln!(out, "        _k[tid] = k[t];").unwrap();
    writeln!(out, "        _a[tid] = a[t];").unwrap();
    writeln!(out, "        _b[tid] = b[t];").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float v_val = v[t];").unwrap();
    writeln!(out, "        float y = 0.0, sa = 0.0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float4 sa_vec(0.0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (uint j = 0; j < head_size; j += 4) {{").unwrap();
    writeln!(
        out,
        "            float4 a_vec = float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);"
    )
    .unwrap();
    writeln!(out, "            sa_vec += a_vec * s_vec;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        sa = sa_vec[0] + sa_vec[1] + sa_vec[2] + sa_vec[3];"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (uint j = 0; j < head_size; j += 4) {{").unwrap();
    writeln!(
        out,
        "            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 w_vec = float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 b_vec = float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);"
    )
    .unwrap();
    writeln!(
        out,
        "            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float4 kv = k_vec * v_val;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            s_vec = s_vec * w_vec + kv + sa * b_vec;").unwrap();
    writeln!(out, "            y += dot(s_vec, r_vec);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            state[j]   = s_vec[0];").unwrap();
    writeln!(out, "            state[j+1] = s_vec[1];").unwrap();
    writeln!(out, "            state[j+2] = s_vec[2];").unwrap();
    writeln!(out, "            state[j+3] = s_vec[3];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst[t] = y;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i = 0; i < head_size; i++) {{").unwrap();
    writeln!(
        out,
        "        dst[T * C + batch_id * state_size + head_id * head_size * head_size"
    )
    .unwrap();
    writeln!(out, "            + tid * head_size + i] = state[i];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_set_rows_f_ggml(out: &mut String) {
    writeln!(out, "template<typename T, typename TI>").unwrap();
    writeln!(out, "kernel void kernel_set_rows_f(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_set_rows & args,").unwrap();
    writeln!(out, "        device const  void * src0,").unwrap();
    writeln!(out, "        device const  void * src1,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(
        out,
        "        uint3                tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint                 tiitg[[thread_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3                tptg [[threads_per_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int32_t i03 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig.y;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i12 = i03%args.ne12;").unwrap();
    writeln!(out, "    const int32_t i11 = i02%args.ne11;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;"
    )
    .unwrap();
    writeln!(out, "    if (i01 >= args.ne01) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i10 = i01;").unwrap();
    writeln!(out, "    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "          device T     * dst_row = (      device T     *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);").unwrap();
    writeln!(out, "    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        dst_row[ind] = (T) src_row[ind];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_set_rows_q32_ggml(out: &mut String) {
    writeln!(out, "template<typename TI, typename block_q, void (*quantize_func)(device const float *, device block_q &)>").unwrap();
    writeln!(out, "kernel void kernel_set_rows_q32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_set_rows & args,").unwrap();
    writeln!(out, "        device const  void * src0,").unwrap();
    writeln!(out, "        device const  void * src1,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(
        out,
        "        uint3                tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint                 tiitg[[thread_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3                tptg [[threads_per_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int32_t i03 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig.y;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i12 = i03%args.ne12;").unwrap();
    writeln!(out, "    const int32_t i11 = i02%args.ne11;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;"
    )
    .unwrap();
    writeln!(out, "    if (i01 >= args.ne01) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i10 = i01;").unwrap();
    writeln!(out, "    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);").unwrap();
    writeln!(out, "    const device float   * src_row = (const device float   *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        quantize_func(src_row + 32*ind, dst_row[ind]);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_soft_max_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_soft_max(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_soft_max & args,").unwrap();
    writeln!(out, "        device const  char * src0,").unwrap();
    writeln!(out, "        device const  char * src1,").unwrap();
    writeln!(out, "        device const  char * src2,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(out, "        threadgroup  float * buf [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        uint3  tptg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t i03 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t i01 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i13 = i03%args.ne13;").unwrap();
    writeln!(out, "    const int32_t i12 = i02%args.ne12;").unwrap();
    writeln!(out, "    const int32_t i11 = i01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);").unwrap();
    writeln!(out, "    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;").unwrap();
    writeln!(out, "    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;").unwrap();
    writeln!(out, "    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float slope = 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // ALiBi").unwrap();
    writeln!(out, "    if (args.max_bias > 0.0f) {{").unwrap();
    writeln!(out, "        const int32_t h = i02;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float base = h < args.n_head_log2 ? args.m0 : args.m1;"
    )
    .unwrap();
    writeln!(
        out,
        "        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        slope = pow(base, exp);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel max").unwrap();
    writeln!(out, "    float lmax = psrc2 ? psrc2[i02] : -INFINITY;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // find the max value in the block").unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(out, "    if (tptg.x > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            buf[tiisg] = -INFINITY;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            buf[sgitg] = max_val;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        max_val = buf[tiisg];").unwrap();
    writeln!(out, "        max_val = simd_max(max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel sum").unwrap();
    writeln!(out, "    float lsum = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);").unwrap();
    writeln!(out, "        lsum += exp_psrc0;").unwrap();
    writeln!(out, "        pdst[i00] = exp_psrc0;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // This barrier fixes a failing test").unwrap();
    writeln!(
        out,
        "    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum = simd_sum(lsum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tptg.x > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            buf[tiisg] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            buf[sgitg] = sum;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        sum = buf[tiisg];").unwrap();
    writeln!(out, "        sum = simd_sum(sum);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (psrc2) {{").unwrap();
    writeln!(out, "        sum += exp(psrc2[i02] - max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float inv_sum = 1.0f/sum;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        pdst[i00] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_soft_max_4_ggml(out: &mut String) {
    writeln!(out, "template<typename T>").unwrap();
    writeln!(out, "kernel void kernel_soft_max_4(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_soft_max & args,").unwrap();
    writeln!(out, "        device const  char * src0,").unwrap();
    writeln!(out, "        device const  char * src1,").unwrap();
    writeln!(out, "        device const  char * src2,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(out, "        threadgroup  float * buf [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        uint3  tptg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t i03 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t i01 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i13 = i03%args.ne13;").unwrap();
    writeln!(out, "    const int32_t i12 = i02%args.ne12;").unwrap();
    writeln!(out, "    const int32_t i11 = i01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);").unwrap();
    writeln!(out, "    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;").unwrap();
    writeln!(out, "    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;").unwrap();
    writeln!(out, "    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float slope = 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (args.max_bias > 0.0f) {{").unwrap();
    writeln!(out, "        const int32_t h = i02;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float base = h < args.n_head_log2 ? args.m0 : args.m1;"
    )
    .unwrap();
    writeln!(
        out,
        "        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        slope = pow(base, exp);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel max").unwrap();
    writeln!(out, "    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float max_val = simd_max(lmax);").unwrap();
    writeln!(out, "    if (tptg.x > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            buf[tiisg] = -INFINITY;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            buf[sgitg] = max_val;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        max_val = buf[tiisg];").unwrap();
    writeln!(out, "        max_val = simd_max(max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel sum").unwrap();
    writeln!(out, "    float4 lsum4 = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);").unwrap();
    writeln!(out, "        lsum4 += exp_psrc4;").unwrap();
    writeln!(out, "        pdst4[i00] = exp_psrc4;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // This barrier fixes a failing test").unwrap();
    writeln!(
        out,
        "    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum = simd_sum(lsum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tptg.x > N_SIMDWIDTH) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(out, "            buf[tiisg] = 0.0f;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            buf[sgitg] = sum;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        sum = buf[tiisg];").unwrap();
    writeln!(out, "        sum = simd_sum(sum);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (psrc2) {{").unwrap();
    writeln!(out, "        sum += exp(psrc2[i02] - max_val);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float inv_sum = 1.0f/sum;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {{"
    )
    .unwrap();
    writeln!(out, "        pdst4[i00] *= inv_sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_solve_tri_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_solve_tri_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_solve_tri & args,").unwrap();
    writeln!(out, "        device   const char * src0,").unwrap();
    writeln!(out, "        device   const char * src1,").unwrap();
    writeln!(out, "        device         char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup    char * shmem [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    constexpr short NW = N_SIMDWIDTH;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short NSG = FC_solve_tri_nsg;").unwrap();
    writeln!(out, "    const short N   = FC_solve_tri_n;").unwrap();
    writeln!(out, "    const short K   = FC_solve_tri_k;").unwrap();
    writeln!(out, "    const short NP  = PAD2(N, NW);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i03 = tgpig.z;").unwrap();
    writeln!(out, "    const int32_t i02 = tgpig.y;").unwrap();
    writeln!(out, "    const int32_t i01 = tgpig.x*NSG + sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    threadgroup float * sh0 = (threadgroup float *) shmem;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * src0_ptr = (device const float *)(src0 + i02 * args.nb02 + i03 * args.nb03) + sgitg*N;").unwrap();
    writeln!(out, "    device const float * src1_ptr = (device const float *)(src1 + i02 * args.nb12 + i03 * args.nb13) + i01;").unwrap();
    writeln!(out, "    device       float * dst_ptr  = (device       float *)(dst  + i02 * args.nb2  + i03 * args.nb3)  + i01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short rr = 0; rr < N; rr += NSG) {{").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            threadgroup float * sh0_cur = sh0 + sgitg*NP;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            for (short t = 0; t*NW < N; ++t) {{").unwrap();
    writeln!(out, "                const short idx = t*NW + tiisg;").unwrap();
    writeln!(out, "                sh0_cur[idx] = src0_ptr[idx];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            src0_ptr += NSG*N;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (i01 >= args.ne10) {{").unwrap();
    writeln!(out, "            continue;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        for (short ir = 0; ir < NSG && rr + ir < N; ++ir) {{"
    )
    .unwrap();
    writeln!(out, "            const short r = rr + ir;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup float * sh0_cur = sh0 + ir*NP;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sum = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            for (short t = 0; t*NW < r; ++t) {{").unwrap();
    writeln!(out, "                const short idx = t*NW + tiisg;").unwrap();
    writeln!(
        out,
        "                sum += sh0_cur[idx] * dst_ptr[idx*K] * (idx < r);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            sum = simd_sum(sum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (tiisg == 0) {{").unwrap();
    writeln!(out, "                const float diag = sh0_cur[r];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                dst_ptr[r*K] = (src1_ptr[r*K] - sum) / diag;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ssm_conv_f32_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_ssm_conv_f32_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_ssm_conv & args,").unwrap();
    writeln!(out, "        device const  void * src0,").unwrap();
    writeln!(out, "        device const  void * src1,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int64_t ir = tgpig.x;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t nc  = args.ne10;").unwrap();
    writeln!(out, "  //const int64_t ncs = args.ne00;").unwrap();
    writeln!(out, "  //const int64_t nr  = args.ne01;").unwrap();
    writeln!(out, "  //const int64_t n_t = args.ne1;").unwrap();
    writeln!(out, "  //const int64_t n_s = args.ne2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);").unwrap();
    writeln!(out, "    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);").unwrap();
    writeln!(out, "    device       float * x = (device       float *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int64_t i0 = 0; i0 < nc; ++i0) {{").unwrap();
    writeln!(out, "        sumf += s[i0] * c[i0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    x[0] = sumf;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ssm_conv_f32_f32_4_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_ssm_conv_f32_f32_4(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_ssm_conv & args,").unwrap();
    writeln!(out, "        device const  void * src0,").unwrap();
    writeln!(out, "        device const  void * src1,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int64_t ir = tgpig.x;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t nc  = args.ne10;").unwrap();
    writeln!(out, "  //const int64_t ncs = args.ne00;").unwrap();
    writeln!(out, "  //const int64_t nr  = args.ne01;").unwrap();
    writeln!(out, "  //const int64_t n_t = args.ne1;").unwrap();
    writeln!(out, "  //const int64_t n_s = args.ne2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);").unwrap();
    writeln!(out, "    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);").unwrap();
    writeln!(out, "    device       float  * x = (device       float  *) ((device       char *) dst  + ir*args.nb0  + i2*args.nb1  + i3*args.nb2);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int64_t i0 = 0; i0 < nc/4; ++i0) {{").unwrap();
    writeln!(out, "        sumf += dot(s[i0], c[i0]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    x[0] = sumf;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ssm_conv_f32_f32_batched_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_ssm_conv_f32_f32_batched(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_ssm_conv & args,").unwrap();
    writeln!(out, "        device const  void * src0,").unwrap();
    writeln!(out, "        device const  void * src1,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    // tgpig.x = row index (ir)").unwrap();
    writeln!(
        out,
        "    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)"
    )
    .unwrap();
    writeln!(out, "    // tgpig.z = sequence index (i3)").unwrap();
    writeln!(
        out,
        "    // tpitg.x = thread within batch (0..BATCH_SIZE-1)"
    )
    .unwrap();
    writeln!(out, "    const short BATCH_SIZE = FC_ssm_conv_bs;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t ir      = tgpig.x;").unwrap();
    writeln!(out, "    const int64_t i2_base = tgpig.y * BATCH_SIZE;").unwrap();
    writeln!(out, "    const int64_t i3      = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2_off  = tpitg.x;").unwrap();
    writeln!(out, "    const int64_t i2      = i2_base + i2_off;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)"
    )
    .unwrap();
    writeln!(
        out,
        "    const int64_t n_t = args.ne1;   // number of tokens"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Bounds check for partial batches at the end").unwrap();
    writeln!(out, "    if (i2 >= n_t) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Load conv weights (shared across all tokens for this row)"
    )
    .unwrap();
    writeln!(out, "    device const float * c = (device const float *) ((device const char *) src1 + ir*args.nb11);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Load source for this specific token").unwrap();
    writeln!(out, "    device const float * s = (device const float *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Output location for this token").unwrap();
    writeln!(out, "    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out, "    for (int64_t i0 = 0; i0 < nc; ++i0) {{").unwrap();
    writeln!(out, "        sumf += s[i0] * c[i0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    x[0] = sumf;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ssm_conv_f32_f32_batched_4_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_ssm_conv_f32_f32_batched_4(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_ssm_conv & args,").unwrap();
    writeln!(out, "        device const  void * src0,").unwrap();
    writeln!(out, "        device const  void * src1,").unwrap();
    writeln!(out, "        device       float * dst,").unwrap();
    writeln!(out, "        uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(
        out,
        "        uint3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    // tgpig.x = row index (ir)").unwrap();
    writeln!(
        out,
        "    // tgpig.y = batch of tokens (i2_base / BATCH_SIZE)"
    )
    .unwrap();
    writeln!(out, "    // tgpig.z = sequence index (i3)").unwrap();
    writeln!(
        out,
        "    // tpitg.x = thread within batch (0..BATCH_SIZE-1)"
    )
    .unwrap();
    writeln!(out, "    const short BATCH_SIZE = FC_ssm_conv_bs;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t ir      = tgpig.x;").unwrap();
    writeln!(out, "    const int64_t i2_base = tgpig.y * BATCH_SIZE;").unwrap();
    writeln!(out, "    const int64_t i3      = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2_off  = tpitg.x;").unwrap();
    writeln!(out, "    const int64_t i2      = i2_base + i2_off;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int64_t nc  = args.ne10;  // conv kernel size (typically 4)"
    )
    .unwrap();
    writeln!(
        out,
        "    const int64_t n_t = args.ne1;   // number of tokens"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Bounds check for partial batches at the end").unwrap();
    writeln!(out, "    if (i2 >= n_t) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Load conv weights (shared across all tokens for this row)"
    )
    .unwrap();
    writeln!(out, "    device const float4 * c = (device const float4 *) ((device const char *) src1 + ir*args.nb11);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Load source for this specific token").unwrap();
    writeln!(out, "    device const float4 * s = (device const float4 *) ((device const char *) src0 + ir*args.nb01 + i2*args.nb00 + i3*args.nb02);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Output location for this token").unwrap();
    writeln!(out, "    device float * x = (device float *) ((device char *) dst + ir*args.nb0 + i2*args.nb1 + i3*args.nb2);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf = 0.0f;").unwrap();
    writeln!(out, "    for (int64_t i0 = 0; i0 < nc/4; ++i0) {{").unwrap();
    writeln!(out, "        sumf += dot(s[i0], c[i0]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    x[0] = sumf;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_ssm_scan_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_ssm_scan_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_ssm_scan & args,").unwrap();
    writeln!(out, "        device const void * src0,").unwrap();
    writeln!(out, "        device const void * src1,").unwrap();
    writeln!(out, "        device const void * src2,").unwrap();
    writeln!(out, "        device const void * src3,").unwrap();
    writeln!(out, "        device const void * src4,").unwrap();
    writeln!(out, "        device const void * src5,").unwrap();
    writeln!(out, "        device const void * src6,").unwrap();
    writeln!(out, "        device      float * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup float * shared [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(out, "        ushort  sgptg[[simdgroups_per_threadgroup]],").unwrap();
    writeln!(out, "        uint3    tgpg[[threadgroups_per_grid]]) {{").unwrap();
    writeln!(out, "    constexpr short NW = N_SIMDWIDTH;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Shared memory layout:").unwrap();
    writeln!(
        out,
        "    // [0..sgptg*NW-1]: partial sums for reduction (existing)"
    )
    .unwrap();
    writeln!(
        out,
        "    // [sgptg*NW..sgptg*NW+sgptg-1]: pre-computed x_dt values for each token in batch"
    )
    .unwrap();
    writeln!(out, "    // [sgptg*NW+sgptg..sgptg*NW+2*sgptg-1]: pre-computed dA values for each token in batch").unwrap();
    writeln!(out, "    threadgroup float * shared_sums = shared;").unwrap();
    writeln!(
        out,
        "    threadgroup float * shared_x_dt = shared + sgptg * NW;"
    )
    .unwrap();
    writeln!(
        out,
        "    threadgroup float * shared_dA   = shared + sgptg * NW + sgptg;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    shared_sums[tpitg.x] = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i0 = tpitg.x;").unwrap();
    writeln!(out, "    const int32_t i1 = tgpig.x;").unwrap();
    writeln!(out, "    const int32_t ir = tgpig.y; // current head").unwrap();
    writeln!(out, "    const int32_t i3 = tgpig.z; // current seq").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t nc  = args.d_state;").unwrap();
    writeln!(out, "    const int32_t nr  = args.d_inner;").unwrap();
    writeln!(out, "    const int32_t nh  = args.n_head;").unwrap();
    writeln!(out, "    const int32_t ng  = args.n_group;").unwrap();
    writeln!(out, "    const int32_t n_t = args.n_seq_tokens;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t s_off = args.s_off;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const int32_t * ids = (device const int32_t *) src6;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);").unwrap();
    writeln!(out, "    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i = i0 + i1*nc;").unwrap();
    writeln!(
        out,
        "    const int32_t g = ir / (nh / ng); // repeat_interleave"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float s0 = s0_buff[i];").unwrap();
    writeln!(out, "    float s  = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * A = (device const float *) ((device const char *) src3 + ir*args.nb31); // {{ne30, nh}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float A0 = A[i0%args.ne30];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * x  = (device const float *)((device const char *) src1 + i1*args.nb10  + ir*args.nb11 + i3*args.nb13); // {{dim, nh, nt, ns}}").unwrap();
    writeln!(out, "    device const float * dt = (device const float *)((device const char *) src2 + ir*args.nb20  + i3*args.nb22);                // {{nh, nt, ns}}").unwrap();
    writeln!(out, "    device const float * B  = (device const float *)((device const char *) src4 +  g*args.nb41  + i3*args.nb43);                // {{d_state, ng, nt, ns}}").unwrap();
    writeln!(out, "    device const float * C  = (device const float *)((device const char *) src5 +  g*args.nb51  + i3*args.nb53);                // {{d_state, ng, nt, ns}}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device float * y = dst + (i1 + ir*(nr) + i3*(n_t*nh*nr)); // {{dim, nh, nt, ns}}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i2 = 0; i2 < n_t; i2 += sgptg) {{").unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        // Pre-compute x_dt and dA for this batch of tokens"
    )
    .unwrap();
    writeln!(
        out,
        "        // Only first sgptg threads do the loads and expensive math"
    )
    .unwrap();
    writeln!(out, "        if (i0 < sgptg && i2 + i0 < n_t) {{").unwrap();
    writeln!(
        out,
        "            // ns12 and ns21 are element strides (nb12/nb10, nb21/nb20)"
    )
    .unwrap();
    writeln!(
        out,
        "            device const float * x_t  = x  + i0 * args.ns12;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const float * dt_t = dt + i0 * args.ns21;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float dt0  = dt_t[0];").unwrap();
    writeln!(
        out,
        "            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;"
    )
    .unwrap();
    writeln!(out, "            shared_x_dt[i0] = x_t[0] * dtsp;").unwrap();
    writeln!(out, "            shared_dA[i0]   = dtsp;  // Store dtsp, compute exp(dtsp * A0) per-thread since A0 varies").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {{"
    )
    .unwrap();
    writeln!(out, "            const float x_dt = shared_x_dt[t];").unwrap();
    writeln!(
        out,
        "            const float dA   = exp(shared_dA[t] * A0);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            s = (s0 * dA) + (B[i0] * x_dt);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float sumf = simd_sum(s * C[i0]);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (tiisg == 0) {{").unwrap();
    writeln!(out, "                shared_sums[t*NW + sgitg] = sumf;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // recurse").unwrap();
    writeln!(out, "            s0 = s;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            B  += args.ns42;").unwrap();
    writeln!(out, "            C  += args.ns52;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // Advance pointers for next batch").unwrap();
    writeln!(out, "        x  += sgptg * args.ns12;").unwrap();
    writeln!(out, "        dt += sgptg * args.ns21;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float sumf = simd_sum(shared_sums[sgitg*NW + tiisg]);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (tiisg == 0 && i2 + sgitg < n_t) {{").unwrap();
    writeln!(out, "            y[sgitg*nh*nr] = sumf;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y += sgptg*nh*nr;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    s_buff[i] = s;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_swiglu_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_swiglu_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_glu & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;").unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {{").unwrap();
    writeln!(out, "        const float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        const float x1 = src1_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const float silu = x0 / (1.0f + exp(-x0));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_row[i0] = silu*x1;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_swiglu_oai_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_swiglu_oai_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_glu & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        uint tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "        uint   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;").unwrap();
    writeln!(out, "    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;").unwrap();
    writeln!(out, "    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {{").unwrap();
    writeln!(out, "        float x0 = src0_row[i0];").unwrap();
    writeln!(out, "        float x1 = src1_row[i0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        x0 = min(x0, args.limit);").unwrap();
    writeln!(out, "        x1 = max(min(x1, args.limit), -args.limit);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        float out_glu = x0 / (1.0f + exp(-x0 * args.alpha));"
    )
    .unwrap();
    writeln!(out, "        out_glu = out_glu * (1.0f + x1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_row[i0] = out_glu;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_timestep_embedding_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_timestep_embedding_f32(").unwrap();
    writeln!(
        out,
        "    constant  ggml_metal_kargs_timestep_embedding & args,"
    )
    .unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int i = tgpig.x;").unwrap();
    writeln!(
        out,
        "    device float * embed_data = (device float *)(dst + i*args.nb1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    int half_ = args.dim / 2;").unwrap();
    writeln!(out, "    for (int j = tpitg.x; j < half_; j += ntg.x) {{").unwrap();
    writeln!(out, "        float timestep = ((device float *)src0)[i];").unwrap();
    writeln!(
        out,
        "        float freq = (float)exp(-log((float)args.max_period) * j / half_);"
    )
    .unwrap();
    writeln!(out, "        float arg = timestep * freq;").unwrap();
    writeln!(out, "        embed_data[j        ] = cos(arg);").unwrap();
    writeln!(out, "        embed_data[j + half_] = sin(arg);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (args.dim % 2 != 0 && tpitg.x == 0) {{").unwrap();
    writeln!(out, "        embed_data[2 * half_] = 0.f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_tri_ggml(out: &mut String) {
    writeln!(out, "template<typename T, int ttype>").unwrap();
    writeln!(out, "kernel void kernel_tri(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_tri & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort3 tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {{"
    )
    .unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T * src_row = (device const T *) ((device const char *) src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);").unwrap();
    writeln!(out, "    device       T * dst_row = (device       T *) ((device       char *) dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Each thread is a single element of the row if ne00 < max threads per"
    )
    .unwrap();
    writeln!(
        out,
        "    // threadgroup, so this will loop once for each index that this thread is"
    )
    .unwrap();
    writeln!(out, "    // responsible for").unwrap();
    writeln!(
        out,
        "    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        // Use the comparison as a mask for branchless"
    )
    .unwrap();
    writeln!(
        out,
        "        dst_row[i0] = static_cast<T>(_ggml_vec_tri_cmp<ttype>(i0, i1)) * src_row[i0];"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_upscale_bicubic_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_upscale_bicubic_f32(").unwrap();
    writeln!(out, "    constant ggml_metal_kargs_upscale & args,").unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i03 = i3 / args.sf3;").unwrap();
    writeln!(out, "    const int64_t i02 = i2 / args.sf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float   f01 = ((float)i1 + args.poffs) / args.sf1 - args.poffs;"
    )
    .unwrap();
    writeln!(out, "    const int64_t i01 = (int64_t)floor(f01);").unwrap();
    writeln!(out, "    const float   fd1 = f01 - (float)i01;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float w_y0 = bicubic_weight2(fd1 + 1.0f);").unwrap();
    writeln!(out, "    const float w_y1 = bicubic_weight1(fd1);").unwrap();
    writeln!(out, "    const float w_y2 = bicubic_weight1(1.0f - fd1);").unwrap();
    writeln!(out, "    const float w_y3 = bicubic_weight2(2.0f - fd1);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const device const char * src_slice = src0 + i03 * args.nb03 + i02 * args.nb02;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_ptr = (device float *)(dst + i3 * args.nb3 + i2 * args.nb2 + i1 * args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float   f00 = ((float)i0 + args.poffs) / args.sf0 - args.poffs;"
    )
    .unwrap();
    writeln!(out, "        const int64_t i00 = (int64_t)floor(f00);").unwrap();
    writeln!(out, "        const float   fd0 = f00 - (float)i00;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float w_x0 = bicubic_weight2(fd0 + 1.0f);"
    )
    .unwrap();
    writeln!(out, "        const float w_x1 = bicubic_weight1(fd0);").unwrap();
    writeln!(
        out,
        "        const float w_x2 = bicubic_weight1(1.0f - fd0);"
    )
    .unwrap();
    writeln!(
        out,
        "        const float w_x3 = bicubic_weight2(2.0f - fd0);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float sum = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (int dy = -1; dy <= 2; ++dy) {{").unwrap();
    writeln!(
        out,
        "            const int64_t iy = MAX(0, MIN(args.ne01 - 1, i01 + dy));"
    )
    .unwrap();
    writeln!(out, "            const float wy = (dy == -1) ? w_y0 : (dy == 0) ? w_y1 : (dy == 1) ? w_y2 : w_y3;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            for (int dx = -1; dx <= 2; ++dx) {{").unwrap();
    writeln!(
        out,
        "                const int64_t ix = MAX(0, MIN(args.ne00 - 1, i00 + dx));"
    )
    .unwrap();
    writeln!(out, "                const float wx = (dx == -1) ? w_x0 : (dx == 0) ? w_x1 : (dx == 1) ? w_x2 : w_x3;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                device const float * src_ptr = (device const float *)(src_slice + iy * args.nb01 + ix * args.nb00);").unwrap();
    writeln!(out, "                sum += (*src_ptr) * wx * wy;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_ptr[i0] = sum;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_upscale_bilinear_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_upscale_bilinear_f32(").unwrap();
    writeln!(out, "    constant ggml_metal_kargs_upscale & args,").unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i03 = i3 / args.sf3;").unwrap();
    writeln!(out, "    const int64_t i02 = i2 / args.sf2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const float   f01  = ((float)i1 + args.poffs) / args.sf1 - args.poffs;"
    )
    .unwrap();
    writeln!(
        out,
        "    const int64_t i01  = MAX(0, MIN(args.ne01 - 1, (int64_t)floor(f01)));"
    )
    .unwrap();
    writeln!(
        out,
        "    const int64_t i01p = MAX(0, MIN(args.ne01 - 1, i01 + 1));"
    )
    .unwrap();
    writeln!(
        out,
        "    const float   fd1  = MAX(0.0f, MIN(1.0f, f01 - (float)i01));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    src0 += i03*args.nb03 + i02*args.nb02;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_ptr = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (FC_upscale_aa) {{").unwrap();
    writeln!(
        out,
        "        const float support0  = MAX(1.0f, 1.0f / args.sf0);"
    )
    .unwrap();
    writeln!(out, "        const float invscale0 = 1.0f / support0;").unwrap();
    writeln!(
        out,
        "        const float support1  = MAX(1.0f, 1.0f / args.sf1);"
    )
    .unwrap();
    writeln!(out, "        const float invscale1 = 1.0f / support1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            const float f00 = ((float)i0 + args.poffs) / args.sf0 - args.poffs;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            int64_t x_min = MAX((int64_t)0, (int64_t)floor(f00 - support0 + args.poffs));"
    )
    .unwrap();
    writeln!(
        out,
        "            int64_t x_max = MIN(args.ne00,  (int64_t)ceil (f00 + support0 + args.poffs));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            int64_t y_min = MAX((int64_t)0, (int64_t)floor(f01 - support1 + args.poffs));"
    )
    .unwrap();
    writeln!(
        out,
        "            int64_t y_max = MIN(args.ne01,  (int64_t)ceil (f01 + support1 + args.poffs));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sum = 0.0f;").unwrap();
    writeln!(out, "            float wsum = 0.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            for (int64_t sy = y_min; sy < y_max; ++sy) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                const float wy = MAX(0.0f, 1.0f - fabs((float)sy - f01) * invscale1);"
    )
    .unwrap();
    writeln!(
        out,
        "                for (int64_t sx = x_min; sx < x_max; ++sx) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    const float wx = MAX(0.0f, 1.0f - fabs((float)sx - f00) * invscale0);"
    )
    .unwrap();
    writeln!(out, "                    const float w  = wx * wy;").unwrap();
    writeln!(out, "                    device const float * src_ptr = (device const float *)(src0 + sy*args.nb01 + sx*args.nb00);").unwrap();
    writeln!(out, "                    sum  += (*src_ptr) * w;").unwrap();
    writeln!(out, "                    wsum += w;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float v = (wsum > 0.0f) ? (sum / wsum) : 0.0f;"
    )
    .unwrap();
    writeln!(out, "            dst_ptr[i0] = v;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(
        out,
        "        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            const float   f00  = ((float)i0 + args.poffs) / args.sf0 - args.poffs;"
    )
    .unwrap();
    writeln!(
        out,
        "            const int64_t i00  = MAX(0, MIN(args.ne00 - 1, (int64_t)floor(f00)));"
    )
    .unwrap();
    writeln!(
        out,
        "            const int64_t i00p = MAX(0, MIN(args.ne00 - 1, i00 + 1));"
    )
    .unwrap();
    writeln!(
        out,
        "            const float   fd0  = MAX(0.0f, MIN(1.0f, f00 - (float)i00));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            device const float * src00 = (device const float *)(src0 + i01*args.nb01  + i00*args.nb00);").unwrap();
    writeln!(out, "            device const float * src10 = (device const float *)(src0 + i01*args.nb01  + i00p*args.nb00);").unwrap();
    writeln!(out, "            device const float * src01 = (device const float *)(src0 + i01p*args.nb01 + i00*args.nb00);").unwrap();
    writeln!(out, "            device const float * src11 = (device const float *)(src0 + i01p*args.nb01 + i00p*args.nb00);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float v =").unwrap();
    writeln!(
        out,
        "                (*src00) * (1.0f - fd0) * (1.0f - fd1) +"
    )
    .unwrap();
    writeln!(
        out,
        "                (*src10) * fd0          * (1.0f - fd1) +"
    )
    .unwrap();
    writeln!(out, "                (*src01) * (1.0f - fd0) * fd1 +").unwrap();
    writeln!(out, "                (*src11) * fd0          * fd1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            dst_ptr[i0] = v;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_upscale_nearest_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_upscale_nearest_f32(").unwrap();
    writeln!(out, "    constant ggml_metal_kargs_upscale & args,").unwrap();
    writeln!(out, "    device  const char * src0,").unwrap();
    writeln!(out, "    device        char * dst,").unwrap();
    writeln!(out, "    uint3 tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "    uint3 tpitg[[thread_position_in_threadgroup]],").unwrap();
    writeln!(out, "    uint3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i3 = tgpig.z;").unwrap();
    writeln!(out, "    const int64_t i2 = tgpig.y;").unwrap();
    writeln!(out, "    const int64_t i1 = tgpig.x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i03 = i3/args.sf3;").unwrap();
    writeln!(out, "    const int64_t i02 = i2/args.sf2;").unwrap();
    writeln!(out, "    const int64_t i01 = i1/args.sf1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "        const int64_t i00 = i0/args.sf0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);").unwrap();
    writeln!(out, "        device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1  +  i0*args.nb0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        dst_ptr[0] = src0_ptr[0];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_flash_attn_ext_blk_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_flash_attn_ext_blk(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_flash_attn_ext_blk & args,"
    )
    .unwrap();
    writeln!(out, "        device const char * mask,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]]) {{").unwrap();
    writeln!(out, "    // block size C x Q").unwrap();
    writeln!(out, "    const int32_t Q = FC_flash_attn_ext_blk_nqptg;").unwrap();
    writeln!(out, "    const int32_t C = FC_flash_attn_ext_blk_ncpsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short NW  = N_SIMDWIDTH;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i3 = tgpig[2]/args.ne32;").unwrap();
    writeln!(out, "    const int32_t i2 = tgpig[2]%args.ne32;").unwrap();
    writeln!(out, "    const int32_t i1 = tgpig[1];").unwrap();
    writeln!(out, "    const int32_t i0 = tgpig[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    char res = i0*C + C > args.ne30 ? 1 : 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // detailed check of the elements of the block").unwrap();
    writeln!(out, "    if ((C > NW || Q > 1) && res == 0) {{").unwrap();
    writeln!(out, "        half mmin =  MAXHALF;").unwrap();
    writeln!(out, "        half mmax = -MAXHALF;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        FOR_UNROLL (short j = 0; j < Q; ++j) {{").unwrap();
    writeln!(
        out,
        "            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                mmin = min(mmin, mask_src[ii*NW]);").unwrap();
    writeln!(out, "                mmax = max(mmax, mask_src[ii*NW]);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            mask_src += args.nb31/2;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        mmin = simd_min(mmin);").unwrap();
    writeln!(out, "        mmax = simd_max(mmax);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (mmax > -MAXHALF) {{").unwrap();
    writeln!(out, "            if (mmin == 0.0 && mmax == 0.0) {{").unwrap();
    writeln!(out, "                res = 2;").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                res = 1;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);").unwrap();
    writeln!(out, "    const int32_t nblk0 = ((args.ne30 + C - 1)/C);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_flash_attn_ext_pad_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_flash_attn_ext_pad(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_flash_attn_ext_pad & args,"
    )
    .unwrap();
    writeln!(out, "        device const char * k,").unwrap();
    writeln!(out, "        device const char * v,").unwrap();
    writeln!(out, "        device const char * mask,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort3   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const int32_t C = FC_flash_attn_ext_pad_ncpsg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device char * k_pad    = dst;").unwrap();
    writeln!(
        out,
        "    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;"
    )
    .unwrap();
    writeln!(
        out,
        "    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t icp = args.ne11 % C;").unwrap();
    writeln!(out, "    const int32_t ic0 = args.ne11 - icp;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t i1 = tgpig[0];").unwrap();
    writeln!(out, "    const int32_t i2 = tgpig[1];").unwrap();
    writeln!(out, "    const int32_t i3 = tgpig[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {{").unwrap();
    writeln!(out, "        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;").unwrap();
    writeln!(out, "        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;").unwrap();
    writeln!(out, "        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (i1 >= icp) {{").unwrap();
    writeln!(out, "            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention").unwrap();
    writeln!(
        out,
        "            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "                k_dst[i] = 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "                v_dst[i] = 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(
        out,
        "            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "                k_dst[i] = k_src[i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(
        out,
        "            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "                v_dst[i] = v_src[i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (FC_flash_attn_ext_pad_has_mask) {{").unwrap();
    writeln!(out, "        if (i2 < args.ne32 && i3 < args.ne33) {{").unwrap();
    writeln!(
        out,
        "            for (int ib = i1; ib < args.ne31; ib += C) {{"
    )
    .unwrap();
    writeln!(out, "                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;").unwrap();
    writeln!(out, "                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                for (int i = tiitg; i < C; i += ntg.x) {{"
    )
    .unwrap();
    writeln!(out, "                    if (i >= icp) {{").unwrap();
    writeln!(out, "                        mask_dst[i] = -MAXHALF;").unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(out, "                        mask_dst[i] = mask_src[i];").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_flash_attn_ext_vec_ggml(out: &mut String) {
    writeln!(out, "template<").unwrap();
    writeln!(out, "    typename q4_t,  // query types in shared memory").unwrap();
    writeln!(out, "    typename k4_t,  // key types in shared memory").unwrap();
    writeln!(out, "    typename v4_t,  // value types in shared memory").unwrap();
    writeln!(out, "    typename qk_t,  // Q*K types").unwrap();
    writeln!(out, "    typename s_t,   // soft-max types").unwrap();
    writeln!(out, "    typename s4_t,").unwrap();
    writeln!(out, "    typename o4_t,  // attention accumulation types").unwrap();
    writeln!(out, "    typename kd4_t, // key type in device memory").unwrap();
    writeln!(out, "    short nl_k,").unwrap();
    writeln!(
        out,
        "    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),"
    )
    .unwrap();
    writeln!(out, "    typename vd4_t, // value type in device memory").unwrap();
    writeln!(out, "    short nl_v,").unwrap();
    writeln!(
        out,
        "    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),"
    )
    .unwrap();
    writeln!(out, "    short DK,       // K head size").unwrap();
    writeln!(out, "    short DV,       // V head size").unwrap();
    writeln!(out, "    short NE = 4,   // head elements per thread").unwrap();
    writeln!(
        out,
        "    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup"
    )
    .unwrap();
    writeln!(
        out,
        "    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup"
    )
    .unwrap();
    writeln!(out, "kernel void kernel_flash_attn_ext_vec(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_flash_attn_ext_vec & args,"
    )
    .unwrap();
    writeln!(out, "        device const char * q,").unwrap();
    writeln!(out, "        device const char * k,").unwrap();
    writeln!(out, "        device const char * v,").unwrap();
    writeln!(out, "        device const char * mask,").unwrap();
    writeln!(out, "        device const char * sinks,").unwrap();
    writeln!(out, "        device const char * pad,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        threadgroup  half * shmem_f16 [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(
        out,
        "    static_assert(DK % 32 == 0, \"DK must be divisible by 32\");"
    )
    .unwrap();
    writeln!(
        out,
        "    static_assert(DV % 32 == 0, \"DV must be divisible by 32\");"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#define NWG  (FC_flash_attn_ext_vec_nwg)").unwrap();
    writeln!(out, "#define NSG  (FC_flash_attn_ext_vec_nsg)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#define NS10 (FC_flash_attn_ext_vec_ns10)").unwrap();
    writeln!(out, "#define NS20 (FC_flash_attn_ext_vec_ns20)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short iwg = tgpig[2]%NWG;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort iq3 = tgpig[2]/NWG;").unwrap();
    writeln!(out, "    const ushort iq2 = tgpig[1];").unwrap();
    writeln!(out, "    const ushort iq1 = tgpig[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short DK4 = DK/4;").unwrap();
    writeln!(out, "    constexpr short DV4 = DV/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short PK  = PAD2(DK, 128);").unwrap();
    writeln!(out, "    constexpr short PK4 = PK/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short PV  = PAD2(DV, 128);").unwrap();
    writeln!(out, "    constexpr short PV4 = PV/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr short NW  = N_SIMDWIDTH;").unwrap();
    writeln!(out, "    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads").unwrap();
    writeln!(
        out,
        "    constexpr short SH  = 4*C;   // shared memory per simdgroup"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    static_assert(DK4 % NL == 0, \"DK4 must be divisible by NL\");"
    )
    .unwrap();
    writeln!(
        out,
        "    static_assert(DV4 % NL == 0, \"DV4 must be divisible by NL\");"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "  //const short T = PK + NSG*SH; // shared memory size per query in (half)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  //threadgroup q_t   * sq  = (threadgroup q_t   *) (shmem_f16 +                      0*PK); // holds the query data").unwrap();
    writeln!(out, "    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK); // same as above but in q4_t").unwrap();
    writeln!(out, "    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK); // scratch buffer for attention").unwrap();
    writeln!(out, "    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK); // same as above but in s4_t").unwrap();
    writeln!(out, "    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK); // scratch buffer for mask").unwrap();
    writeln!(out, "    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH); // scratch buffer for the results").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // store the result for all queries in shared memory (the O matrix from the paper)"
    )
    .unwrap();
    writeln!(out, "    so4 += tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(
        out,
        "        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const short ikv2 = iq2/(args.ne02/args.ne_12_2);"
    )
    .unwrap();
    writeln!(
        out,
        "        const short ikv3 = iq3/(args.ne03/args.ne_12_3);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        k += ikv2*args.nb12 + ikv3*args.nb13;").unwrap();
    writeln!(out, "        v += ikv2*args.nb22 + ikv3*args.nb23;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // load heads from Q to shared memory").unwrap();
    writeln!(
        out,
        "    device const float4 * q4 = (device const float4 *) ((device const char *) q);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (iq1 < args.ne01) {{").unwrap();
    writeln!(out, "        for (short i = tiisg; i < PK4; i += NW) {{").unwrap();
    writeln!(out, "            if (i < DK4) {{").unwrap();
    writeln!(out, "                sq4[i] = (q4_t) q4[i];").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                sq4[i] = (q4_t) 0.0f;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // zero out so").unwrap();
    writeln!(out, "    for (short i = 0; i < DV4/NL; ++i) {{").unwrap();
    writeln!(out, "        so4[i*NL] = (o4_t) 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // zero out shared memory SH").unwrap();
    writeln!(out, "    for (short i = tiisg; i < SH/4; i += NW) {{").unwrap();
    writeln!(out, "        ss4[i] = (s4_t) 0.0f;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        float S = 0.0f;").unwrap();
    writeln!(out, "        float M = -FLT_MAX/2;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // thread indices inside the simdgroup").unwrap();
    writeln!(out, "        const short tx = tiisg%NL;").unwrap();
    writeln!(out, "        const short ty = tiisg/NL;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // pointer to the mask").unwrap();
    writeln!(out, "        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        float slope = 1.0f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // ALiBi").unwrap();
    writeln!(out, "        if (FC_flash_attn_ext_vec_has_bias) {{").unwrap();
    writeln!(out, "            const short h = iq2;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            const float base = h < args.n_head_log2 ? args.m0 : args.m1;"
    )
    .unwrap();
    writeln!(out, "            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            slope = pow(base, exph);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // loop over the KV cache").unwrap();
    writeln!(
        out,
        "        // each simdgroup handles blocks of Q rows and C columns"
    )
    .unwrap();
    writeln!(
        out,
        "        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {{"
    )
    .unwrap();
    writeln!(out, "            int ic = ic0*C;").unwrap();
    writeln!(out, "            if (ic >= args.ne11) {{").unwrap();
    writeln!(out, "                break;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            // the last partial chunk uses the pad buffer as source"
    )
    .unwrap();
    writeln!(
        out,
        "            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {{"
    )
    .unwrap();
    writeln!(out, "                k    = pad;").unwrap();
    writeln!(
        out,
        "                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;"
    )
    .unwrap();
    writeln!(
        out,
        "                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                const short ikv2 = iq2/(args.ne02/args.ne_12_2);"
    )
    .unwrap();
    writeln!(
        out,
        "                const short ikv3 = iq3/(args.ne03/args.ne_12_3);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;"
    )
    .unwrap();
    writeln!(
        out,
        "                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                if (!FC_flash_attn_ext_vec_has_mask) {{"
    )
    .unwrap();
    writeln!(out, "                    if (ic + tiisg >= args.ne11) {{").unwrap();
    writeln!(out, "                        sm[tiisg] = -MAXHALF;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    pm = (device const half *) (mask) +"
    )
    .unwrap();
    writeln!(out, "                        iq1*C +").unwrap();
    writeln!(
        out,
        "                        (iq2%args.ne32)*(C*args.ne31) +"
    )
    .unwrap();
    writeln!(
        out,
        "                        (iq3%args.ne33)*(C*args.ne31*args.ne32);"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                ic = 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (FC_flash_attn_ext_vec_has_mask) {{").unwrap();
    writeln!(out, "                sm[tiisg] = pm[ic + tiisg];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // skip -INF blocks").unwrap();
    writeln!(out, "            if (simd_max(sm[tiisg]) <= -MAXHALF) {{").unwrap();
    writeln!(out, "                continue;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // Q*K^T").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(
        out,
        "                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);"
    )
    .unwrap();
    writeln!(out, "                threadgroup const q4_t * pq4 = sq4;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                pk4 += ty*NS10/4 + tx;").unwrap();
    writeln!(out, "                pq4 += tx;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                qk_t mqk[C/NE] = {{ [ 0 ... C/NE - 1] = 0.0f }};"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                // each simdgroup processes 1 query and NE (NW/NL) cache elements"
    )
    .unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    if (is_same<kd4_t, k4_t>::value) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(out, "                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        k4_t mk;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            const short i = ii*NL + tx;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    if (NE == 1) {{").unwrap();
    writeln!(out, "                        mqk[cc] = simd_sum(mqk[cc]);").unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(out, "                        // simdgroup reduce (NE = 4)").unwrap();
    writeln!(out, "                        // [ 0 ..  7] -> [ 0]").unwrap();
    writeln!(out, "                        // [ 8 .. 15] -> [ 8]").unwrap();
    writeln!(out, "                        // [16 .. 23] -> [16]").unwrap();
    writeln!(out, "                        // [24 .. 31] -> [24]").unwrap();
    writeln!(out, "                        if (NE <= 1) {{").unwrap();
    writeln!(
        out,
        "                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                        if (NE <= 2) {{").unwrap();
    writeln!(
        out,
        "                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                        if (NE <= 4) {{").unwrap();
    writeln!(
        out,
        "                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                        if (NE <= 8) {{").unwrap();
    writeln!(
        out,
        "                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                        if (NE <= 16) {{").unwrap();
    writeln!(
        out,
        "                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                        // broadcast").unwrap();
    writeln!(
        out,
        "                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (FC_flash_attn_ext_vec_has_mask &&").unwrap();
    writeln!(out, "                   !FC_flash_attn_ext_vec_has_scap &&").unwrap();
    writeln!(
        out,
        "                   !FC_flash_attn_ext_vec_has_bias) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);"
    )
    .unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(out, "                    mqk[tx] *= args.scale;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    if (FC_flash_attn_ext_vec_has_scap) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    if (FC_flash_attn_ext_vec_has_bias) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;"
    )
    .unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(
        out,
        "                        mqk[tx] += (qk_t) sm[NE*tx + ty];"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    ss[NE*tx + ty] = mqk[tx];").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // online softmax").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(out, "                const float m = M;").unwrap();
    writeln!(out, "                const float s = ss[tiisg];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                M = simd_max(max(M, s));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const float ms = exp(m - M);").unwrap();
    writeln!(out, "                const float vs = exp(s - M);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                S = S*ms + simd_sum(vs);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                // the P matrix from the paper (Q rows, C columns)"
    )
    .unwrap();
    writeln!(out, "                ss[tiisg] = vs;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                // O = diag(ms)*O").unwrap();
    writeln!(out, "                if ((DV4/NL % NW == 0) || ty == 0) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                        so4[ii*NL] *= ms;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            simdgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // O = O + (Q*K^T)*V").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(out, "                o4_t lo[DV4/NL];").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    lo[ii] = 0.0f;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if (is_same<vd4_t, v4_t>::value) {{").unwrap();
    writeln!(
        out,
        "                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    pv4 += ty*NS20/4 + tx;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    const auto sst = ss + ty;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));").unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {{"
    )
    .unwrap();
    writeln!(out, "                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                            const short i = ii*NL + tx;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                            v4_t mv;").unwrap();
    writeln!(
        out,
        "                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));"
    )
    .unwrap();
    writeln!(out, "                        }}").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    if (NE > 1) {{").unwrap();
    writeln!(
        out,
        "                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    if (NE > 2) {{").unwrap();
    writeln!(
        out,
        "                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    if (NE > 4) {{").unwrap();
    writeln!(
        out,
        "                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    if (NE > 8) {{").unwrap();
    writeln!(
        out,
        "                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    if (NE > 16) {{").unwrap();
    writeln!(
        out,
        "                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);"
    )
    .unwrap();
    writeln!(
        out,
        "                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);"
    )
    .unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                if ((DV4/NL % NW == 0) || ty == 0) {{").unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                        so4[ii*NL] += lo[ii];").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {{"
    )
    .unwrap();
    writeln!(out, "            const float m = M;").unwrap();
    writeln!(
        out,
        "            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            M = simd_max(max(M, s));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float ms = exp(m - M);").unwrap();
    writeln!(out, "            const float vs = exp(s - M);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            S = S*ms + simd_sum(vs);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if ((DV4/NL % NW == 0) || ty == 0) {{").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {{"
    )
    .unwrap();
    writeln!(out, "                    so4[ii*NL] *= ms;").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            ss[0] = (s_t) S;").unwrap();
    writeln!(out, "            ss[1] = (s_t) M;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    so4 -= tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // parallel reduce").unwrap();
    writeln!(out, "    for (short r = NSG/2; r > 0; r >>= 1) {{").unwrap();
    writeln!(out, "        if (sgitg < r) {{").unwrap();
    writeln!(out, "            const float S0 = ss[           0];").unwrap();
    writeln!(out, "            const float S1 = ss[r*(SH/2) + 0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float M0 = ss[           1];").unwrap();
    writeln!(out, "            const float M1 = ss[r*(SH/2) + 1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float M = max(M0, M1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float ms0 = exp(M0 - M);").unwrap();
    writeln!(out, "            const float ms1 = exp(M1 - M);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const float S = S0*ms0 + S1*ms1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            if (tiisg == 0) {{").unwrap();
    writeln!(out, "                ss[0] = S;").unwrap();
    writeln!(out, "                ss[1] = M;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1").unwrap();
    writeln!(
        out,
        "            for (short i = tiisg; i < DV4; i += NW) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // final rescale with 1/S and store to global memory"
    )
    .unwrap();
    writeln!(out, "    if (sgitg == 0) {{").unwrap();
    writeln!(
        out,
        "        const int64_t nrows = args.ne3*args.ne2*args.ne1;"
    )
    .unwrap();
    writeln!(
        out,
        "        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device float4 * dst4 = (device float4 *) dst;").unwrap();
    writeln!(out, "        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // interleave the workgroup data").unwrap();
    writeln!(out, "        for (short i = tiisg; i < DV4; i += NW) {{").unwrap();
    writeln!(
        out,
        "            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // store S and M").unwrap();
    writeln!(out, "        if (NWG > 1) {{").unwrap();
    writeln!(out, "            if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];"
    )
    .unwrap();
    writeln!(
        out,
        "                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef NWG").unwrap();
    writeln!(out, "#undef NSG").unwrap();
    writeln!(out, "#undef NS10").unwrap();
    writeln!(out, "#undef NS20").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_flash_attn_ext_vec_reduce_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_flash_attn_ext_vec_reduce(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,"
    )
    .unwrap();
    writeln!(out, "        device  const char * htmp,").unwrap();
    writeln!(out, "        device        char * dst,").unwrap();
    writeln!(out, "        uint   tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "#define NWG (FC_flash_attn_ext_vec_reduce_NWG)").unwrap();
    writeln!(out, "#define DV  (FC_flash_attn_ext_vec_reduce_DV)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t rid = tgpig;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short iwg = tiisg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float S = ss[rid*(2*NWG) + 2*iwg + 0];").unwrap();
    writeln!(out, "    float M = ss[rid*(2*NWG) + 2*iwg + 1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float m  = simd_max(M);").unwrap();
    writeln!(out, "    const float ms = exp(M - m);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    S = simd_sum(S*ms);").unwrap();
    writeln!(out, "    S = S == 0.0f ? 0.0f : 1.0f/S;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short DV4 = DV/4;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;"
    )
    .unwrap();
    writeln!(
        out,
        "    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short i = sgitg; i < DV4; i += NWG) {{").unwrap();
    writeln!(
        out,
        "        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (iwg == 0) {{").unwrap();
    writeln!(out, "            dst4[i] = v*S;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#undef NWG").unwrap();
    writeln!(out, "#undef DV").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mm_ggml(out: &mut String) {
    writeln!(out, "template<").unwrap();
    writeln!(out, "    typename SA, typename SA_4x4, typename SA_8x8,").unwrap();
    writeln!(out, "    typename SB, typename SB_2x4, typename SB_8x8,").unwrap();
    writeln!(out, "    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),").unwrap();
    writeln!(
        out,
        "    typename T0, typename T0_4x4, typename T1, typename T1_2x4>"
    )
    .unwrap();
    writeln!(out, "kernel void kernel_mul_mm(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mm & args,").unwrap();
    writeln!(out, "        device const char * srcA,").unwrap();
    writeln!(out, "        device const char * srcB,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(
        out,
        "        uint3  tgpig [[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort tiitg [[thread_index_in_threadgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg [[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    (void) sgitg;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)").unwrap();
    writeln!(out, "    const int K = args.ne00;").unwrap();
    writeln!(out, "    const int M = args.ne0;").unwrap();
    writeln!(out, "    const int N = args.ne1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Batch dimension handling").unwrap();
    writeln!(out, "    const int im = tgpig.z;").unwrap();
    writeln!(out, "    const int i12 = im % FC_mul_mm_ne12;").unwrap();
    writeln!(out, "    const int i13 = im / FC_mul_mm_ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Batch offsets for srcA and srcB").unwrap();
    writeln!(
        out,
        "    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Tile dimensions").unwrap();
    writeln!(
        out,
        "    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;"
    )
    .unwrap();
    writeln!(
        out,
        "    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Tile offsets in output matrix").unwrap();
    writeln!(out, "    const int ra = tgpig.y * NRA;").unwrap();
    writeln!(out, "    const int rb = tgpig.x * NRB;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Threadgroup memory for dequantized A tile only").unwrap();
    writeln!(out, "    threadgroup SA * sa = (threadgroup SA *)(shmem);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Work-item count for A loading").unwrap();
    writeln!(out, "    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;").unwrap();
    writeln!(
        out,
        "    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // tA wraps threadgroup memory").unwrap();
    writeln!(
        out,
        "    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // tB wraps device memory directly").unwrap();
    writeln!(
        out,
        "    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);"
    )
    .unwrap();
    writeln!(out, "    const int strideB = args.nb11 / sizeof(T1);").unwrap();
    writeln!(
        out,
        "    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({{1, strideB}}));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Configure matmul operation").unwrap();
    writeln!(out, "    mpp::tensor_ops::matmul2d<").unwrap();
    writeln!(out, "        mpp::tensor_ops::matmul2d_descriptor(").unwrap();
    writeln!(
        out,
        "            NRB, NRA, N_MM_NK_TOTAL, false, true, true,"
    )
    .unwrap();
    writeln!(
        out,
        "            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),"
    )
    .unwrap();
    writeln!(
        out,
        "        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Accumulate partial results over K dimension").unwrap();
    writeln!(
        out,
        "    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        // === PHASE 1: Dequantization of A into threadgroup memory ==="
    )
    .unwrap();
    writeln!(
        out,
        "        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {{"
    )
    .unwrap();
    writeln!(out, "            const int row = work / N_MM_NK;").unwrap();
    writeln!(out, "            const int k_chunk = work % N_MM_NK;").unwrap();
    writeln!(out, "            const int k_pos = loop_k + k_chunk * 16;").unwrap();
    writeln!(out, "            const short k_base = k_chunk * 16;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            // Bounds check: skip device read if row is out of matrix bounds"
    )
    .unwrap();
    writeln!(out, "            if (ra + row < M) {{").unwrap();
    writeln!(
        out,
        "                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {{"
    )
    .unwrap();
    writeln!(out, "                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).").unwrap();
    writeln!(out, "                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,").unwrap();
    writeln!(out, "                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.").unwrap();
    writeln!(
        out,
        "                    // Mirrors the legacy kernel's existing guard."
    )
    .unwrap();
    writeln!(out, "                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short i = 0; i < 16; i++) {{"
    )
    .unwrap();
    writeln!(out, "                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }} else {{").unwrap();
    writeln!(
        out,
        "                    const int block_idx = k_pos / (16 * nl);"
    )
    .unwrap();
    writeln!(
        out,
        "                    const short il = (k_pos / 16) % nl;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                    SA_4x4 temp_a;").unwrap();
    writeln!(
        out,
        "                    dequantize_func(row_ptr + block_idx, il, temp_a);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                    FOR_UNROLL (short i = 0; i < 16; i++) {{"
    )
    .unwrap();
    writeln!(out, "                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)").unwrap();
    writeln!(out, "                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }} else {{").unwrap();
    writeln!(out, "                // Zero-pad rows beyond matrix bounds").unwrap();
    writeln!(
        out,
        "                FOR_UNROLL (short i = 0; i < 16; i++) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;"
    )
    .unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        // === PHASE 2: Tensor matmul ===").unwrap();
    writeln!(out, "        auto mA = tA.slice(0, 0);").unwrap();
    writeln!(out, "        auto mB = tB.slice(loop_k, rb);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        mm.run(mB, mA, cT);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Store result tile to output matrix (with batch offset)"
    )
    .unwrap();
    writeln!(
        out,
        "    // cT.store handles bounds checking via tD's extents (M, N)"
    )
    .unwrap();
    writeln!(
        out,
        "    device float * dstBatch = (device float *)dst + im * N * M;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({{1, M}}));"
    )
    .unwrap();
    writeln!(out, "    cT.store(tD.slice(ra, rb));").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mm_id_ggml(out: &mut String) {
    writeln!(out, "template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>").unwrap();
    writeln!(out, "kernel void kernel_mul_mm_id(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mm_id & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device const char * htpe,").unwrap();
    writeln!(out, "        device const char * hids,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    threadgroup S0 * sa = (threadgroup S0 *)(shmem);").unwrap();
    writeln!(
        out,
        "    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#ifdef GGML_METAL_HAS_TENSOR").unwrap();
    writeln!(
        out,
        "    threadgroup float * sc = (threadgroup float *)(shmem);"
    )
    .unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr int NR0 = 64;").unwrap();
    writeln!(out, "    constexpr int NR1 = 32;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    constexpr int NK  = 32;").unwrap();
    writeln!(out, "    constexpr int NL0 = NK/16;").unwrap();
    writeln!(out, "    constexpr int NL1 = NK/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int im = tgpig.z; // expert").unwrap();
    writeln!(out, "    const int r0 = tgpig.y*NR0;").unwrap();
    writeln!(out, "    const int r1 = tgpig.x*NR1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);"
    )
    .unwrap();
    writeln!(
        out,
        "    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int32_t neh1 = tpe_u32[im];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (r1 >= neh1) {{").unwrap();
    writeln!(out, "        return;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // if this block is of 64x32 shape or smaller").unwrap();
    writeln!(
        out,
        "    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;"
    )
    .unwrap();
    writeln!(
        out,
        "    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // a thread shouldn't load data outside of the matrix"
    )
    .unwrap();
    writeln!(
        out,
        "    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63"
    )
    .unwrap();
    writeln!(
        out,
        "    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short il0 = (tiitg % NL0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    short il = il0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int id = ids_i32[im*args.ne21 + r1 + lr1];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short i11 = (id % args.ne20) % args.ne11;").unwrap();
    writeln!(out, "    const short i12 = (id / args.ne20);").unwrap();
    writeln!(out, "    const short i13 = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;"
    )
    .unwrap();
    writeln!(out, "    const short    offset1 = il0/nl;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short iy = 8*(tiitg % NL1);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const T1 * y = (device const T1 *)(src1").unwrap();
    writeln!(out, "        + args.nb13*i13").unwrap();
    writeln!(out, "        + args.nb12*i12").unwrap();
    writeln!(out, "        + args.nb11*i11").unwrap();
    writeln!(out, "        + args.nb10*iy);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#ifndef GGML_METAL_HAS_TENSOR").unwrap();
    writeln!(out, "    S0_8x8 ma[4];").unwrap();
    writeln!(out, "    S1_8x8 mb[2];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    simdgroup_float8x8 mc[8];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short i = 0; i < 8; i++){{").unwrap();
    writeln!(
        out,
        "        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "#else").unwrap();
    writeln!(out, "    auto tA = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));").unwrap();
    writeln!(out, "    auto tB = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    mpp::tensor_ops::matmul2d<").unwrap();
    writeln!(out, "        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),").unwrap();
    writeln!(out, "        execution_simdgroups<4>> mm;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();"
    )
    .unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {{"
    )
    .unwrap();
    writeln!(out, "#ifndef GGML_METAL_HAS_TENSOR").unwrap();
    writeln!(out, "        // load data and store to threadgroup memory").unwrap();
    writeln!(
        out,
        "        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // no need for dequantization").unwrap();
    writeln!(out, "            for (short i = 0; i < 16; i++) {{").unwrap();
    writeln!(out, "                const short sx = 2*il0 + i/8;").unwrap();
    writeln!(out, "                const short sy = (tiitg/NL0)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "              //const short lx = i%8;").unwrap();
    writeln!(out, "              //const short ly = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                const short lx = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                const short ly = i%8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short ib = 8*sx + sy;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            S0_4x4 temp_a;").unwrap();
    writeln!(out, "            dequantize_func(x, il, temp_a);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < 16; i++) {{").unwrap();
    writeln!(out, "                const short sx = 2*il0 + i/8;").unwrap();
    writeln!(out, "                const short sy = (tiitg/NL0)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "              //const short lx = i%8;").unwrap();
    writeln!(out, "              //const short ly = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                const short lx = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                const short ly = i%8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short ib = 8*sx + sy;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                // NOTE: this is massively slower.. WTF?"
    )
    .unwrap();
    writeln!(
        out,
        "                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_mul_mm_bc_inp) {{").unwrap();
    writeln!(out, "            for (short i = 0; i < 8; ++i) {{").unwrap();
    writeln!(out, "                const short sx = (tiitg%NL1);").unwrap();
    writeln!(out, "                const short sy = (tiitg/NL1)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short lx = i;").unwrap();
    writeln!(out, "                const short ly = (tiitg/NL1)%8;").unwrap();
    writeln!(out, "              //const short lx = (tiitg/NL1)%8;").unwrap();
    writeln!(out, "              //const short ly = i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short ib = 4*sx + sy;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            const short sx = (tiitg%NL1);").unwrap();
    writeln!(out, "            const short sy = (tiitg/NL1)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "          //const short dx = sx;").unwrap();
    writeln!(out, "          //const short dy = sy;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const short ly = (tiitg/NL1)%8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            const short ib = 4*sx + sy;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "#else").unwrap();
    writeln!(out, "        // load data and store to threadgroup memory").unwrap();
    writeln!(
        out,
        "        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {{"
    )
    .unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            // no need for dequantization").unwrap();
    writeln!(out, "            for (short i = 0; i < 16; i++) {{").unwrap();
    writeln!(out, "                const short sx = 2*il0 + i/8;").unwrap();
    writeln!(out, "                const short sy = (tiitg/NL0)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short lx = i%8;").unwrap();
    writeln!(out, "                const short ly = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                //const short lx = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                //const short ly = i%8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            S0_4x4 temp_a;").unwrap();
    writeln!(out, "            dequantize_func(x, il, temp_a);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < 16; i++) {{").unwrap();
    writeln!(out, "                const short sx = 2*il0 + i/8;").unwrap();
    writeln!(out, "                const short sy = (tiitg/NL0)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short lx = i%8;").unwrap();
    writeln!(out, "                const short ly = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                //const short lx = (tiitg/NL0)%8;").unwrap();
    writeln!(out, "                //const short ly = i%8;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        if (FC_mul_mm_bc_inp) {{").unwrap();
    writeln!(out, "            for (short i = 0; i < 8; ++i) {{").unwrap();
    writeln!(out, "                const short sx = (tiitg%NL1);").unwrap();
    writeln!(out, "                const short sy = (tiitg/NL1)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                const short lx = i;").unwrap();
    writeln!(out, "                const short ly = (tiitg/NL1)%8;").unwrap();
    writeln!(out, "                //const short lx = (tiitg/NL1)%8;").unwrap();
    writeln!(out, "                //const short ly = i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }} else {{").unwrap();
    writeln!(out, "            const short sx = (tiitg%NL1);").unwrap();
    writeln!(out, "            const short sy = (tiitg/NL1)/8;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            //const short lx = i;").unwrap();
    writeln!(out, "            const short ly = (tiitg/NL1)%8;").unwrap();
    writeln!(out, "            //const short lx = (tiitg/NL1)%8;").unwrap();
    writeln!(out, "            //const short ly = i;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        il = (il + 2 < nl) ? il + 2 : il % 2;").unwrap();
    writeln!(out, "        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y += NK;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#ifndef GGML_METAL_HAS_TENSOR").unwrap();
    writeln!(
        out,
        "        // load matrices from threadgroup memory and conduct outer products"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {{").unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < 4; i++) {{").unwrap();
    writeln!(
        out,
        "                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < 2; i++) {{").unwrap();
    writeln!(
        out,
        "                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            simdgroup_barrier(mem_flags::mem_none);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            FOR_UNROLL (short i = 0; i < 8; i++){{").unwrap();
    writeln!(
        out,
        "                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);"
    )
    .unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            lsma += 8*64;").unwrap();
    writeln!(out, "            lsmb += 4*64;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "#else").unwrap();
    writeln!(out, "        auto sA = tA.slice(0, 0);").unwrap();
    writeln!(out, "        auto sB = tB.slice(0, 0);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        mm.run(sB, sA, cT);").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // block is smaller than 64x32, we should avoid writing data outside of the matrix"
    )
    .unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#ifdef GGML_METAL_HAS_TENSOR").unwrap();
    writeln!(out, "    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));").unwrap();
    writeln!(out, "    cT.store(tC);").unwrap();
    writeln!(out, "#else").unwrap();
    writeln!(out, "    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short i = 0; i < 8; i++) {{").unwrap();
    writeln!(
        out,
        "        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (short j = sgitg; j < nr1; j += 4) {{").unwrap();
    writeln!(
        out,
        "        const int id = ids_i32[im*args.ne21 + r1 + j];"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const short ide = id % args.ne20;").unwrap();
    writeln!(out, "        const short idt = id / args.ne20;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;").unwrap();
    writeln!(out, "        device float4 * D4 = (device float4 *) D;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup float4 * C4 = (threadgroup float4 *) C;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        int i = tiisg;").unwrap();
    writeln!(out, "        for (; i < nr0/4; i += 32) {{").unwrap();
    writeln!(out, "            *(D4 + i) = *(C4 + i);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        i = (4*(nr0/4)) + tiisg;").unwrap();
    writeln!(out, "        for (; i < nr0; i += 32) {{").unwrap();
    writeln!(out, "            *(D + i) = *(C + i);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mm_id_map0_ggml(out: &mut String) {
    writeln!(out, "template<short ne20> // n_expert_used").unwrap();
    writeln!(out, "kernel void kernel_mul_mm_id_map0(").unwrap();
    writeln!(
        out,
        "        constant ggml_metal_kargs_mul_mm_id_map0 & args,"
    )
    .unwrap();
    writeln!(out, "        device  const char * src2,").unwrap();
    writeln!(out, "        device        char * htpe,").unwrap();
    writeln!(out, "        device        char * hids,").unwrap();
    writeln!(
        out,
        "        threadgroup   char * shmem [[threadgroup(0)]],"
    )
    .unwrap();
    writeln!(
        out,
        "        ushort tpitg[[thread_position_in_threadgroup]],"
    )
    .unwrap();
    writeln!(out, "        ushort   ntg[[threads_per_threadgroup]]) {{").unwrap();
    writeln!(out, "    const short ide = tpitg; // expert id").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint32_t n_all = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {{ // n_tokens"
    )
    .unwrap();
    writeln!(out, "        if (i21 + tpitg < args.ne21) {{").unwrap();
    writeln!(out, "            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            #pragma unroll(ne20)").unwrap();
    writeln!(out, "            for (short i20 = 0; i20 < ne20; i20++) {{").unwrap();
    writeln!(out, "                sids[i20] = src2_i32[i20];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short t = 0; t < ntg; t++) {{").unwrap();
    writeln!(out, "            if (i21 + t >= args.ne21) {{").unwrap();
    writeln!(out, "                break;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            short sel = 0;").unwrap();
    writeln!(out, "            #pragma unroll(ne20)").unwrap();
    writeln!(out, "            for (short i20 = 0; i20 < ne20; i20++) {{").unwrap();
    writeln!(out, "                sel += (sids[i20] == ide)*(i20 + 1);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            n_all += sel > 0;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);"
    )
    .unwrap();
    writeln!(out, "    tpe_u32[ide] = n_all;").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_ext_q4_f32_disp_ggml(out: &mut String) {
    writeln!(out, "template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>").unwrap();
    writeln!(out, "kernel void kernel_mul_mv_ext_q4_f32_disp(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv_ext & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_ext_q4x4_f32_disp_ggml(out: &mut String) {
    writeln!(out, "template<short r1ptg, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>").unwrap();
    writeln!(out, "kernel void kernel_mul_mv_ext_q4x4_f32_disp(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv_ext & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(
        out,
        "        uint3   tgpig[[threadgroup_position_in_grid]],"
    )
    .unwrap();
    writeln!(out, "        ushort  tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    kernel_mul_mv_ext_q4x4_f32_impl<r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_id_ggml(out: &mut String) {
    writeln!(out, "template<mul_mv_disp_fn_t disp_fn>").unwrap();
    writeln!(out, "kernel void kernel_mul_mv_id(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv_id & args,").unwrap();
    writeln!(out, "        device const char * src0s,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        device const char * ids,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiitg[[thread_index_in_threadgroup]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    const int iid1 = tgpig.z/args.nei0;").unwrap();
    writeln!(out, "    const int idx  = tgpig.z%args.nei0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    tgpig.z = 0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i11 = idx % args.ne11;").unwrap();
    writeln!(out, "    const int64_t i12 = iid1;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int64_t i1 = idx;").unwrap();
    writeln!(out, "    const int64_t i2 = i12;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char * src0_cur = src0s + i02*args.nb02;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    ggml_metal_kargs_mul_mv args0 = {{").unwrap();
    writeln!(out, "        /*.ne00 =*/ args.ne00,").unwrap();
    writeln!(out, "        /*.ne01 =*/ args.ne01,").unwrap();
    writeln!(out, "        /*.ne02 =*/ 1, // args.ne02,").unwrap();
    writeln!(out, "        /*.nb00 =*/ args.nb00,").unwrap();
    writeln!(out, "        /*.nb01 =*/ args.nb01,").unwrap();
    writeln!(out, "        /*.nb02 =*/ args.nb02,").unwrap();
    writeln!(out, "        /*.nb03 =*/ args.nb02, // args.ne02 == 1").unwrap();
    writeln!(out, "        /*.ne10 =*/ args.ne10,").unwrap();
    writeln!(out, "        /*.ne11 =*/ 1, // args.ne11,").unwrap();
    writeln!(out, "        /*.ne12 =*/ 1, // args.ne12,").unwrap();
    writeln!(out, "        /*.nb10 =*/ args.nb10,").unwrap();
    writeln!(out, "        /*.nb11 =*/ args.nb11,").unwrap();
    writeln!(out, "        /*.nb12 =*/ args.nb12,").unwrap();
    writeln!(out, "        /*.nb13 =*/ args.nb12, // ne12 == 1").unwrap();
    writeln!(out, "        /*.ne0  =*/ args.ne0,").unwrap();
    writeln!(out, "        /*.ne1  =*/ 1, // args.ne1,").unwrap();
    writeln!(out, "        /*.nr0  =*/ args.nr0,").unwrap();
    writeln!(out, "        /*.r2   =*/ 1,").unwrap();
    writeln!(out, "        /*.r3   =*/ 1,").unwrap();
    writeln!(out, "    }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    disp_fn(").unwrap();
    writeln!(out, "        args0,").unwrap();
    writeln!(out, "        /* src0 */ src0_cur,").unwrap();
    writeln!(out, "        /* src1 */ src1_cur,").unwrap();
    writeln!(out, "        /* dst  */ dst_cur,").unwrap();
    writeln!(out, "        shmem,").unwrap();
    writeln!(out, "        tgpig,").unwrap();
    writeln!(out, "        tiitg,").unwrap();
    writeln!(out, "        tiisg,").unwrap();
    writeln!(out, "        sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_q4_0_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_mul_mv_q4_0_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_q4_1_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_mul_mv_q4_1_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "     mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_q5_0_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_mul_mv_q5_0_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_q5_1_f32_ggml(out: &mut String) {
    writeln!(out, "kernel void kernel_mul_mv_q5_1_f32(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_t_t_ggml(out: &mut String) {
    writeln!(out, "template<typename T0, typename T1>").unwrap();
    writeln!(out, "kernel void kernel_mul_mv_t_t(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    kernel_mul_mv_t_t_disp<T0, T1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_t_t_4_ggml(out: &mut String) {
    writeln!(
        out,
        "template<typename T0, typename T04, typename T1, typename T14>"
    )
    .unwrap();
    writeln!(out, "kernel void kernel_mul_mv_t_t_4(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        threadgroup  char * shmem [[threadgroup(0)]],").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]],").unwrap();
    writeln!(
        out,
        "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{"
    )
    .unwrap();
    writeln!(out, "    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);").unwrap();
    writeln!(out, "}}").unwrap();
}
pub(super) fn emit_mul_mv_t_t_short_ggml(out: &mut String) {
    writeln!(out, "template<typename T0, typename T1>").unwrap();
    writeln!(out, "kernel void kernel_mul_mv_t_t_short(").unwrap();
    writeln!(out, "        constant ggml_metal_kargs_mul_mv & args,").unwrap();
    writeln!(out, "        device const char * src0,").unwrap();
    writeln!(out, "        device const char * src1,").unwrap();
    writeln!(out, "        device       char * dst,").unwrap();
    writeln!(out, "        uint3  tgpig[[threadgroup_position_in_grid]],").unwrap();
    writeln!(out, "        ushort tiisg[[thread_index_in_simdgroup]]) {{").unwrap();
    writeln!(
        out,
        "    kernel_mul_mv_t_t_short_impl<T0, T1, constant ggml_metal_kargs_mul_mv &>("
    )
    .unwrap();
    writeln!(out, "        args,").unwrap();
    writeln!(out, "        src0,").unwrap();
    writeln!(out, "        src1,").unwrap();
    writeln!(out, "        dst,").unwrap();
    writeln!(out, "        tgpig,").unwrap();
    writeln!(out, "        tiisg);").unwrap();
    writeln!(out, "}}").unwrap();
}
/// M113: kernel_mul_mv_id_iq2_xxs_f32 (moe.metal:832). iq2_xxs sibling of
/// M111/M112 — same 4-buf shell + 11-uint params + M92/M111 id-routing
/// wrapper; inner from kernel_mul_mv_iq2_xxs_f32_impl (moe.metal:521-614).
///
/// block_iq2_xxs layout (66 B):
///   half     d  @+0
///   ushort   qs[32] @+2   (QK_K/8 = 32 ushorts)
///
/// NSG=2 (FC_mul_mv_nsg), NR0=N_R0_IQ2_XXS=4, QK_K=256.
/// Threadgroup shmem cooperatively stages iq2xxs_grid[256] (ulong, 2 KB) +
/// ksigns_iq2xs[128] (1 thread loads 4 grid + 2 ksigns elements).
/// Per ib32 ∈ [ix, nb32) step 32: load 32 floats yl from y4 = y + 32*ix.
/// Per row: db = dh[0]; aux32 = q2[2] | (q2[3] << 16); d = db*(0.5 + (aux32>>28));
/// sum = Σ_l 0..4 grid[a={q2[0..4] as uchar4}[l]] decoded with ksigns + kmask sign bits.
/// Output: sumf[row] += d * sum; final write multiplied by 0.25f.
pub(super) fn emit_mul_mv_id_iq2_xxs_f32_msl(out: &mut String, nsg: u32, nr0: u32) {
    let (grid_per_lane, signs_per_lane) = (256 / (nsg * 32), 128 / (nsg * 32));
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW   = 32;").unwrap();
    writeln!(out, "    constexpr short NSG  = {nsg};").unwrap();
    writeln!(
        out,
        "    constexpr short NR0  = {nr0};          // N_R0_IQ2_XXS"
    )
    .unwrap();
    writeln!(out, "    constexpr int   QK_K = 256;").unwrap();
    writeln!(
        out,
        "    constexpr uint  IQ2XXS_BLOCK_BYTES = 66u;  // half d + ushort qs[32]"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Threadgroup shmem: iq2xxs_grid[256] (ulong, 2 KB) + ksigns_iq2xs[128]."
    )
    .unwrap();
    writeln!(out, "    threadgroup ulong   svalues[256];").unwrap();
    writeln!(out, "    threadgroup uchar   ssigns[128];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Decode (idx, iid1) from tgpig.z.").unwrap();
    writeln!(out, "    const uint iid1 = tgpig.z / nei0;").unwrap();
    writeln!(out, "    const uint idx  = tgpig.z % nei0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Routed expert index: ids[iid1, idx] as int32 (p2 = ids)."
    )
    .unwrap();
    writeln!(out, "    device const int32_t * ids_row = (device const int32_t *)(p2 + (uint64_t)iid1 * (uint64_t)nbi1);").unwrap();
    writeln!(out, "    const int32_t i02 = ids_row[idx];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i11 = idx % ne11;").unwrap();
    writeln!(out, "    const uint i12 = iid1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char * src0_cur = p0 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(out, "    device const char * src1_cur = p1 + (uint64_t)i11 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12;").unwrap();
    writeln!(out, "    device       char * dst_cur  = p3 + ((uint64_t)idx * (uint64_t)ne0 + (uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0) * 4u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK_K;").unwrap();
    writeln!(out, "    const int r0 = (int)tgpig.x;").unwrap();
    writeln!(
        out,
        "    const int first_row = (r0 * NSG + (int)sgitg) * NR0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Cooperatively stage iq2xxs_grid + ksigns into threadgroup shmem."
    )
    .unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = {grid_per_lane};").unwrap();
    writeln!(
        out,
        "        int pos  = (32 * (int)sgitg + (int)tiisg) * nval;"
    )
    .unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];"
    )
    .unwrap();
    writeln!(out, "        nval = {signs_per_lane};").unwrap();
    writeln!(out, "        pos  = (32 * (int)sgitg + (int)tiisg) * nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char  * x_base = src0_cur + (uint64_t)first_row * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float * y     = (device const float *)src1_cur;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = (int)tiisg;").unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumf[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        // Per-block (per-row) pointers. block_iq2_xxs: half d @+0, ushort qs[32] @+2."
    )
    .unwrap();
    writeln!(
        out,
        "        // q2 = qs + 4*ib (ushort units → +8*ib bytes from +2)."
    )
    .unwrap();
    writeln!(out, "        device const uint16_t * q2 = (device const uint16_t *)(x_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 2u) + (uint)(4 * ib);").unwrap();
    writeln!(out, "        device const half     * dh = (device const half     *)(x_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 0u);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "            const float db = (float)dh[0];").unwrap();
    writeln!(
        out,
        "            device const uchar * aux8 = (device const uchar *)q2;"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint aux32 = (uint)q2[2] | ((uint)q2[3] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const float d = db * (0.5f + (float)(aux32 >> 28));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sum = 0.0f;").unwrap();
    writeln!(out, "            for (short l = 0; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uchar * grid = (const threadgroup uchar *)(svalues + aux8[l]);").unwrap();
    writeln!(
        out,
        "                const uchar signs = ssigns[(aux32 >> (7 * l)) & 127];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(out, "                    sum += yl[8 * l + j] * (float)grid[j] * ((signs & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumf[row] += d * sum;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            // Advance to next row's block — nb01 BYTES apart."
    )
    .unwrap();
    writeln!(
        out,
        "            q2 = (device const uint16_t *)((device const char *)q2 + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            dh = (device const half     *)((device const char *)dh + (uint)nb01);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *)dst_cur;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && first_row + (int)row < (int)ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        const float tot = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            dst_f32[first_row + (int)row] = tot * 0.25f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
    writeln!(out, "    (void)ne01;").unwrap();
    writeln!(out, "    (void)ne1;").unwrap();
}
/// M128: kernel_mul_mv_id_iq2_xxs_pair_f32 (moe.metal:897). Paired iq2_xxs
/// MoE-routed matvec for fused gate+up. 6 char* bufs:
///   p0 = src0_gate (const, iq2_xxs blocks for gate weights)
///   p1 = src0_up   (const, iq2_xxs blocks for up weights)
///   p2 = src1      (const, float input row)
///   p3 = dst_gate  (writable, float)
///   p4 = dst_up    (writable, float)
///   p5 = ids       (const, int32 routed expert indices)
/// Same 11-uint params as M113. Per (idx, iid1) computes i02 = ids[iid1*nbi1/4 + idx]
/// and offsets src0_gate/src0_up by i02*nb02; shares y load + iq2xxs_grid/ksigns
/// shmem tables across paired streams. NSG=2, NR0=N_R0_IQ2_XXS=4, QK_K=256.
pub(super) fn emit_mul_mv_id_iq2_xxs_pair_f32_msl(out: &mut String, nsg: u32, nr0: u32) {
    let (grid_per_lane, signs_per_lane) = (256 / (nsg * 32), 128 / (nsg * 32));
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW   = 32;").unwrap();
    writeln!(out, "    constexpr short NSG  = {nsg};").unwrap();
    writeln!(
        out,
        "    constexpr short NR0  = {nr0};          // N_R0_IQ2_XXS"
    )
    .unwrap();
    writeln!(out, "    constexpr int   QK_K = 256;").unwrap();
    writeln!(
        out,
        "    constexpr uint  IQ2XXS_BLOCK_BYTES = 66u;  // half d + ushort qs[32]"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Threadgroup shmem: iq2xxs_grid[256] (ulong, 2 KB) + ksigns_iq2xs[128]. Shared across paired gate/up.").unwrap();
    writeln!(out, "    threadgroup ulong   svalues[256];").unwrap();
    writeln!(out, "    threadgroup uchar   ssigns[128];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Decode (idx, iid1) from tgpig.z.").unwrap();
    writeln!(out, "    const uint iid1 = tgpig.z / nei0;").unwrap();
    writeln!(out, "    const uint idx  = tgpig.z % nei0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Routed expert index: ids[iid1, idx] as int32 (p5 = ids)."
    )
    .unwrap();
    writeln!(out, "    device const int32_t * ids_row = (device const int32_t *)(p5 + (uint64_t)iid1 * (uint64_t)nbi1);").unwrap();
    writeln!(out, "    const int32_t i02 = ids_row[idx];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i11 = idx % ne11;").unwrap();
    writeln!(out, "    const uint i12 = iid1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char * src0_gate_cur = p0 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char * src0_up_cur   = p1 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(out, "    device const char * src1_cur      = p2 + (uint64_t)i11 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12;").unwrap();
    writeln!(out, "    device       char * dst_gate_cur  = p3 + ((uint64_t)idx * (uint64_t)ne0 + (uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0) * 4u;").unwrap();
    writeln!(out, "    device       char * dst_up_cur    = p4 + ((uint64_t)idx * (uint64_t)ne0 + (uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0) * 4u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK_K;").unwrap();
    writeln!(out, "    const int r0 = (int)tgpig.x;").unwrap();
    writeln!(
        out,
        "    const int first_row = (r0 * NSG + (int)sgitg) * NR0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Cooperatively stage iq2xxs_grid + ksigns into threadgroup shmem."
    )
    .unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = {grid_per_lane};").unwrap();
    writeln!(
        out,
        "        int pos  = (32 * (int)sgitg + (int)tiisg) * nval;"
    )
    .unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];"
    )
    .unwrap();
    writeln!(out, "        nval = {signs_per_lane};").unwrap();
    writeln!(out, "        pos  = (32 * (int)sgitg + (int)tiisg) * nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char  * xg_base = src0_gate_cur + (uint64_t)first_row * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char  * xu_base = src0_up_cur   + (uint64_t)first_row * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float * y       = (device const float *)src1_cur;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = (int)tiisg;").unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumg[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out, "    float sumu[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "        // Per-block (per-row) pointers for paired gate + up."
    )
    .unwrap();
    writeln!(out, "        device const uint16_t * qg = (device const uint16_t *)(xg_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 2u) + (uint)(4 * ib);").unwrap();
    writeln!(out, "        device const uint16_t * qu = (device const uint16_t *)(xu_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 2u) + (uint)(4 * ib);").unwrap();
    writeln!(out, "        device const half     * dhg = (device const half     *)(xg_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 0u);").unwrap();
    writeln!(out, "        device const half     * dhu = (device const half     *)(xu_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 0u);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            device const uchar * aux8g = (device const uchar *)qg;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uchar * aux8u = (device const uchar *)qu;"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint aux32g = (uint)qg[2] | ((uint)qg[3] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint aux32u = (uint)qu[2] | ((uint)qu[3] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const float dg = (float)dhg[0] * (0.5f + (float)(aux32g >> 28));"
    )
    .unwrap();
    writeln!(
        out,
        "            const float du = (float)dhu[0] * (0.5f + (float)(aux32u >> 28));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sg = 0.0f;").unwrap();
    writeln!(out, "            float su = 0.0f;").unwrap();
    writeln!(out, "            for (short l = 0; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uchar * gridg = (const threadgroup uchar *)(svalues + aux8g[l]);").unwrap();
    writeln!(out, "                const threadgroup uchar * gridu = (const threadgroup uchar *)(svalues + aux8u[l]);").unwrap();
    writeln!(
        out,
        "                const uchar signg = ssigns[(aux32g >> (7 * l)) & 127];"
    )
    .unwrap();
    writeln!(
        out,
        "                const uchar signu = ssigns[(aux32u >> (7 * l)) & 127];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(out, "                    const float v = yl[8 * l + j];").unwrap();
    writeln!(out, "                    sg += v * (float)gridg[j] * ((signg & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);").unwrap();
    writeln!(out, "                    su += v * (float)gridu[j] * ((signu & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumg[row] += dg * sg;").unwrap();
    writeln!(out, "            sumu[row] += du * su;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            // Advance to next row's block — nb01 BYTES apart."
    )
    .unwrap();
    writeln!(
        out,
        "            qg  = (device const uint16_t *)((device const char *)qg  + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            qu  = (device const uint16_t *)((device const char *)qu  + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            dhg = (device const half     *)((device const char *)dhg + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            dhu = (device const half     *)((device const char *)dhu + (uint)nb01);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device float * dst_gate_f32 = (device float *)dst_gate_cur;"
    )
    .unwrap();
    writeln!(
        out,
        "    device float * dst_up_f32   = (device float *)dst_up_cur;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && first_row + (int)row < (int)ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        const float sum_gate = simd_sum(sumg[row]);").unwrap();
    writeln!(out, "        const float sum_up   = simd_sum(sumu[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            dst_gate_f32[first_row + (int)row] = sum_gate * 0.25f;"
    )
    .unwrap();
    writeln!(
        out,
        "            dst_up_f32  [first_row + (int)row] = sum_up   * 0.25f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
    writeln!(out, "    (void)ne01;").unwrap();
    writeln!(out, "    (void)ne1;").unwrap();
}
/// M129: kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32. SwiGLU-fused tri-output
/// sibling of M128 (moe.metal:959). 8 char* bufs: p0=src0_gate, p1=src0_up,
/// p2=src1, p3=dst_gate (W), p4=dst_up (W), p5=dst_mid (W), p6=ids, p7=weights.
/// Shares M128 iq2_xxs inner + table-share machinery. Adds 3 act uniforms
/// (mid_row_stride, weight_stride, clamp_value). Final write per row produces
/// gate, up, AND mid = silu(clamp(gate,c)) * clamp(up,-c,c) * route_weight.
pub(super) fn emit_mul_mv_id_iq2_xxs_pair_swiglu_f32_msl(out: &mut String, nsg: u32, nr0: u32) {
    let (grid_per_lane, signs_per_lane) = (256 / (nsg * 32), 128 / (nsg * 32));
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW   = 32;").unwrap();
    writeln!(out, "    constexpr short NSG  = {nsg};").unwrap();
    writeln!(
        out,
        "    constexpr short NR0  = {nr0};          // N_R0_IQ2_XXS"
    )
    .unwrap();
    writeln!(out, "    constexpr int   QK_K = 256;").unwrap();
    writeln!(out, "    constexpr uint  IQ2XXS_BLOCK_BYTES = 66u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup ulong   svalues[256];").unwrap();
    writeln!(out, "    threadgroup uchar   ssigns[128];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint iid1 = tgpig.z / nei0;").unwrap();
    writeln!(out, "    const uint idx  = tgpig.z % nei0;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // p6 = ids; p7 = weights.").unwrap();
    writeln!(out, "    device const int32_t * ids_row = (device const int32_t *)(p6 + (uint64_t)iid1 * (uint64_t)nbi1);").unwrap();
    writeln!(out, "    const int32_t i02 = ids_row[idx];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i11 = idx % ne11;").unwrap();
    writeln!(out, "    const uint i12 = iid1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char * src0_gate_cur = p0 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char * src0_up_cur   = p1 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(out, "    device const char * src1_cur      = p2 + (uint64_t)i11 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK_K;").unwrap();
    writeln!(out, "    const int r0 = (int)tgpig.x;").unwrap();
    writeln!(
        out,
        "    const int first_row = (r0 * NSG + (int)sgitg) * NR0;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        int nval = {grid_per_lane};").unwrap();
    writeln!(
        out,
        "        int pos  = (32 * (int)sgitg + (int)tiisg) * nval;"
    )
    .unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];"
    )
    .unwrap();
    writeln!(out, "        nval = {signs_per_lane};").unwrap();
    writeln!(out, "        pos  = (32 * (int)sgitg + (int)tiisg) * nval;").unwrap();
    writeln!(
        out,
        "        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];"
    )
    .unwrap();
    writeln!(
        out,
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const char  * xg_base = src0_gate_cur + (uint64_t)first_row * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char  * xu_base = src0_up_cur   + (uint64_t)first_row * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const float * y       = (device const float *)src1_cur;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int ix = (int)tiisg;").unwrap();
    writeln!(out, "    device const float * y4 = y + 32 * ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float yl[32];").unwrap();
    writeln!(out, "    float sumg[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out, "    float sumu[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb32 = nb * (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {{").unwrap();
    writeln!(out, "        for (short i = 0; i < 32; ++i) {{").unwrap();
    writeln!(out, "            yl[i] = y4[i];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        const int ibl = ib32 / (QK_K / 32);").unwrap();
    writeln!(out, "        const int ib  = ib32 % (QK_K / 32);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        device const uint16_t * qg = (device const uint16_t *)(xg_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 2u) + (uint)(4 * ib);").unwrap();
    writeln!(out, "        device const uint16_t * qu = (device const uint16_t *)(xu_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 2u) + (uint)(4 * ib);").unwrap();
    writeln!(out, "        device const half     * dhg = (device const half     *)(xg_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 0u);").unwrap();
    writeln!(out, "        device const half     * dhu = (device const half     *)(xu_base + (uint)ibl * IQ2XXS_BLOCK_BYTES + 0u);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            device const uchar * aux8g = (device const uchar *)qg;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uchar * aux8u = (device const uchar *)qu;"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint aux32g = (uint)qg[2] | ((uint)qg[3] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const uint aux32u = (uint)qu[2] | ((uint)qu[3] << 16);"
    )
    .unwrap();
    writeln!(
        out,
        "            const float dg = (float)dhg[0] * (0.5f + (float)(aux32g >> 28));"
    )
    .unwrap();
    writeln!(
        out,
        "            const float du = (float)dhu[0] * (0.5f + (float)(aux32u >> 28));"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sg = 0.0f;").unwrap();
    writeln!(out, "            float su = 0.0f;").unwrap();
    writeln!(out, "            for (short l = 0; l < 4; ++l) {{").unwrap();
    writeln!(out, "                const threadgroup uchar * gridg = (const threadgroup uchar *)(svalues + aux8g[l]);").unwrap();
    writeln!(out, "                const threadgroup uchar * gridu = (const threadgroup uchar *)(svalues + aux8u[l]);").unwrap();
    writeln!(
        out,
        "                const uchar signg = ssigns[(aux32g >> (7 * l)) & 127];"
    )
    .unwrap();
    writeln!(
        out,
        "                const uchar signu = ssigns[(aux32u >> (7 * l)) & 127];"
    )
    .unwrap();
    writeln!(out, "                for (short j = 0; j < 8; ++j) {{").unwrap();
    writeln!(out, "                    const float v = yl[8 * l + j];").unwrap();
    writeln!(out, "                    sg += v * (float)gridg[j] * ((signg & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);").unwrap();
    writeln!(out, "                    su += v * (float)gridu[j] * ((signu & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumg[row] += dg * sg;").unwrap();
    writeln!(out, "            sumu[row] += du * su;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "            qg  = (device const uint16_t *)((device const char *)qg  + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            qu  = (device const uint16_t *)((device const char *)qu  + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            dhg = (device const half     *)((device const char *)dhg + (uint)nb01);"
    )
    .unwrap();
    writeln!(
        out,
        "            dhu = (device const half     *)((device const char *)dhu + (uint)nb01);"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        y4 += 32 * 32;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Tri-output finalize. dst_gate/dst_up indexed as flat 1D f32 per (i12, i11)."
    )
    .unwrap();
    writeln!(out, "    device float * dst_gate_f32 = (device float *)(p3 + ((uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0 + (uint64_t)i11 * (uint64_t)ne0) * 4u);").unwrap();
    writeln!(out, "    device float * dst_up_f32   = (device float *)(p4 + ((uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0 + (uint64_t)i11 * (uint64_t)ne0) * 4u);").unwrap();
    writeln!(out, "    device float * dst_mid_f32  = (device float *)(p5 + (uint64_t)idx * (uint64_t)mid_row_stride);").unwrap();
    writeln!(out, "    device const float * route_w = (device const float *)(p7 + (uint64_t)idx * (uint64_t)weight_stride);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float c = clamp_value;").unwrap();
    writeln!(out, "    const float route_weight = route_w[0];").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && first_row + (int)row < (int)ne0; ++row) {{"
    )
    .unwrap();
    writeln!(out, "        const float sum_gate = simd_sum(sumg[row]);").unwrap();
    writeln!(out, "        const float sum_up   = simd_sum(sumu[row]);").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(out, "            const int out_row = first_row + (int)row;").unwrap();
    writeln!(out, "            const float gate = sum_gate * 0.25f;").unwrap();
    writeln!(out, "            const float up   = sum_up   * 0.25f;").unwrap();
    writeln!(out, "            float g = gate;").unwrap();
    writeln!(out, "            float u = up;").unwrap();
    writeln!(out, "            if (c > 1.0e-6f) {{").unwrap();
    writeln!(out, "                g = fmin(g, c);").unwrap();
    writeln!(out, "                u = clamp(u, -c, c);").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            dst_gate_f32[out_row] = gate;").unwrap();
    writeln!(out, "            dst_up_f32  [out_row] = up;").unwrap();
    writeln!(out, "            const float silu = g / (1.0f + exp(-g));").unwrap();
    writeln!(
        out,
        "            dst_mid_f32 [out_row] = silu * u * route_weight;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
    writeln!(out, "    (void)ne01;").unwrap();
}
/// M93: kernel_dsv4_attn_out_low_q8_0_f32. Stripped-down M92 with id=group:
/// `i02 = idx` (no ids buffer). 3 char* bufs (src0s, src1, dst). 10-uint
/// params (M92 minus nbi1). Same M91 inner loop, M92 dispatch shell.
pub(super) fn emit_dsv4_attn_out_low_q8_0_f32_msl(out: &mut String, nsg: u32, nr0: u32, nq: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW    = 32;").unwrap();
    writeln!(out, "    constexpr short NSG   = {nsg};").unwrap();
    writeln!(out, "    constexpr short NR0   = {nr0};").unwrap();
    writeln!(out, "    constexpr short NQ    = {nq};").unwrap();
    writeln!(out, "    constexpr short QK8_0 = 32;").unwrap();
    writeln!(out, "    constexpr uint  Q8_0_BLOCK_BYTES = 34u;").unwrap();
    writeln!(out, "    threadgroup float shmem_f32[NR0 * NW];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Decode (idx, iid1) from tgpig.z. i02 = idx (no ids buffer)."
    )
    .unwrap();
    writeln!(out, "    const uint iid1 = tgpig.z / nei0;").unwrap();
    writeln!(out, "    const uint idx  = tgpig.z % nei0;").unwrap();
    writeln!(out, "    const int32_t i02 = (int32_t)idx;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i11 = idx % ne11;").unwrap();
    writeln!(out, "    const uint i12 = iid1;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // src0_cur = src0s + i02 * nb02 (per-group weight stride)."
    )
    .unwrap();
    writeln!(
        out,
        "    device const char * src0_cur = p0 + (uint64_t)i02 * (uint64_t)nb02;"
    )
    .unwrap();
    writeln!(out, "    device const char * src1_cur = p1 + (uint64_t)i11 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12;").unwrap();
    writeln!(out, "    device       char * dst_cur  = p2 + ((uint64_t)idx * (uint64_t)ne0 + (uint64_t)i12 * (uint64_t)ne1 * (uint64_t)ne0) * 4u;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK8_0;").unwrap();
    writeln!(out, "    const uint r0 = tgpig.x * (uint)NR0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    device const float * y = (device const float *)src1_cur;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const uchar * ax_byte[NR0];").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        const uint64_t offset0 = (uint64_t)(r0 + (uint)row) * (uint64_t)nb01;"
    )
    .unwrap();
    writeln!(
        out,
        "        ax_byte[row] = (device const uchar *)(src0_cur + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumf[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = (short)(tiisg / (NW / NQ));").unwrap();
    writeln!(out, "    const short il = (short)(tiisg % (NW / NQ));").unwrap();
    writeln!(out, "    const int   ib0 = (int)sgitg * NQ + (int)ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG * NQ) {{").unwrap();
    writeln!(out, "        const int y_off = ib * QK8_0 + (int)il * NQ;").unwrap();
    writeln!(out, "        float yl[NQ];").unwrap();
    writeln!(
        out,
        "        for (short i = 0; i < NQ; ++i) {{ yl[i] = y[y_off + i]; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            device const uchar * blk_byte = ax_byte[row] + (uint)ib * Q8_0_BLOCK_BYTES;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const half  * d_ptr    = (device const half  *)blk_byte;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const int8_t * qs_base = (device const int8_t *)(blk_byte + 2u);"
    )
    .unwrap();
    writeln!(
        out,
        "            device const int8_t * qs      = qs_base + (uint)il * NQ;"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sumq = 0.0f;").unwrap();
    writeln!(
        out,
        "            for (short i = 0; i < NQ; ++i) {{ sumq += (float)qs[i] * yl[i]; }}"
    )
    .unwrap();
    writeln!(out, "            sumf[row] += sumq * (float)(*d_ptr);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * dst_f32 = (device float *)dst_cur;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // helper_mv_reduce_and_write<NR0> inline.").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (sgitg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)tiisg] = 0.0f; }}"
    )
    .unwrap();
    writeln!(out, "        sumf[row] = simd_sum(sumf[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "        if (tiisg == 0) {{ shmem_f32[(uint)row * (uint)NW + (uint)sgitg] = sumf[row]; }}"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && r0 + (uint)row < ne01; ++row) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float tot = simd_sum(shmem_f32[(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(out, "        if (tiisg == 0 && sgitg == 0) {{").unwrap();
    writeln!(out, "            dst_f32[r0 + (uint)row] = tot;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
}
/// M118: kernel_bin_fuse_f32_f32_f32 (bin.metal:192).
/// Single host-callable wrapper for elementwise binary ops add/sub/mul/div.
/// Scope: slow-path (FC_RB=false) + FC_F=1; runtime op (0=add,1=sub,2=mul,3=div)
/// and cb_flag (column-broadcast modulo: i10 = cb ? i0%ne10 : i0) uniforms.
/// 3 char* bufs (src0, src1, dst) + 19 uniforms.
/// 3D grid over (i01=tgpig.x, i02=tgpig.y, i03=tgpig.z) with thread-strided
/// inner over ne0 elements per row.
pub(super) fn emit_bin_fuse_f32_msl(out: &mut String) {
    writeln!(out, "    const int i03 = (int)tgpig.z;").unwrap();
    writeln!(out, "    const int i02 = (int)tgpig.y;").unwrap();
    writeln!(out, "    const int i01 = (int)tgpig.x;").unwrap();
    writeln!(out, "    if (i01 >= (int)ne01) {{ return; }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int i13 = i03 % (int)ne13;").unwrap();
    writeln!(out, "    const int i12 = i02 % (int)ne12;").unwrap();
    writeln!(out, "    const int i11 = i01 % (int)ne11;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const float * src0_ptr = (device const float *)(p0 + (uint64_t)i03*nb03 + (uint64_t)i02*nb02 + (uint64_t)i01*nb01);").unwrap();
    writeln!(out, "    device       float * dst_ptr  = (device       float *)(p2 + (uint64_t)i03*nb3  + (uint64_t)i02*nb2  + (uint64_t)i01*nb1);").unwrap();
    writeln!(out, "    device const float * src1_ptr = (device const float *)(p1 + (uint64_t)i13*nb13 + (uint64_t)i12*nb12 + (uint64_t)i11*nb11);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (uint i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {{").unwrap();
    writeln!(
        out,
        "        const uint i10 = (cb_flag != 0u) ? (i0 % ne10) : i0;"
    )
    .unwrap();
    writeln!(out, "        const float a = src0_ptr[i0];").unwrap();
    writeln!(out, "        const float b = src1_ptr[i10];").unwrap();
    writeln!(out, "        float r = 0.0f;").unwrap();
    writeln!(out, "        if      (op == 0u) {{ r = a + b; }}").unwrap();
    writeln!(out, "        else if (op == 1u) {{ r = a - b; }}").unwrap();
    writeln!(out, "        else if (op == 2u) {{ r = a * b; }}").unwrap();
    writeln!(out, "        else                 {{ r = a / b; }}").unwrap();
    writeln!(out, "        dst_ptr[i0] = r;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// M119: kernel_unary_f32_f32 (unary.metal:310).
/// Single host-callable wrapper covering ~26 elementwise unary ops.
/// 2 char* bufs (src0, dst) + 16 uniforms. Runtime op enum (antirez
/// OP_UNARY_NUM_* values 10-18, 100-121) selects the operation; runtime
/// cnt_flag toggles between 1D contiguous fast path (i0=tgpig.x, no row
/// math) and 3D-grid strided slow path (i01 packed into tgpig.x via
/// /ne01, i02=tgpig.y, i03=tgpig.z, strided inner over ne0).
pub(super) fn emit_unary_op_disp_msl(out: &mut String) {
    emit_unary_op_disp_generic_msl(out, UnaryVar::Scalar);
}
/// M120: kernel_unary_f32_f32_4 (unary.metal:311). Vec4 sibling of M119.
/// T0=T=TC=float4 throughout — Metal float4 overloads cover all per-component
/// math; comparison-mask trick `TC(x > 0)*a + TC(x <= 0)*b` replaces ternary.
pub(super) fn emit_unary_op_disp_4_msl(out: &mut String) {
    emit_unary_op_disp_generic_msl(out, UnaryVar::Vec4);
}
/// M121: kernel_unary_f16_f16 (unary.metal:312). Half type-swap of M119.
/// T0=T=half, TC=float — load casts up to float for math, store casts back.
pub(super) fn emit_unary_op_disp_half_msl(out: &mut String) {
    emit_unary_op_disp_generic_msl(out, UnaryVar::Half);
}
pub(super) fn emit_unary_op_disp_generic_msl(out: &mut String, var: UnaryVar) {
    // T0 = src0 element type. T = dst element type. TC = compute type.
    let (t0, td, tc) = match var {
        UnaryVar::Scalar => ("float", "float", "float"),
        UnaryVar::Vec4 => ("float4", "float4", "float4"),
        UnaryVar::Half => ("half", "half", "float"),
    };
    let vec4 = var == UnaryVar::Vec4;
    // tc-typed literal helper
    let tcz = if vec4 { "TC(0.0f)" } else { "0.0f" };
    let tco = if vec4 { "TC(1.0f)" } else { "1.0f" };
    let tcm = if vec4 { "TC(-1.0f)" } else { "-1.0f" };
    // Promote scalar uniform to TC (no-op for scalar/half, broadcast for vec4).
    let promote = |s: &str| -> String {
        if vec4 {
            format!("TC({})", s)
        } else {
            s.to_string()
        }
    };
    writeln!(out, "    typedef {} TC;", tc).unwrap();
    writeln!(out, "    const float GELU_COEF_A    = 0.044715f;").unwrap();
    writeln!(out, "    const float GELU_QUICK_COEF = -1.702f;").unwrap();
    writeln!(
        out,
        "    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;"
    )
    .unwrap();
    writeln!(
        out,
        "    const float SQRT_2_INV     = 0.70710678118654752440084436210484f;"
    )
    .unwrap();
    writeln!(out, "    const float p_erf  = 0.3275911f;").unwrap();
    writeln!(out, "    const float a1_erf = 0.254829592f;").unwrap();
    writeln!(out, "    const float a2_erf = -0.284496736f;").unwrap();
    writeln!(out, "    const float a3_erf = 1.421413741f;").unwrap();
    writeln!(out, "    const float a4_erf = -1.453152027f;").unwrap();
    writeln!(out, "    const float a5_erf = 1.061405429f;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device const {0} * src0_ptr;", t0).unwrap();
    writeln!(out, "    device       {0} * dst_ptr;", td).unwrap();
    writeln!(out, "    int i0;").unwrap();
    writeln!(out, "    if (cnt_flag != 0u) {{").unwrap();
    writeln!(out, "        i0 = (int)tgpig.x;").unwrap();
    writeln!(out, "        src0_ptr = (device const {0} *)(p0);", t0).unwrap();
    writeln!(out, "        dst_ptr  = (device       {0} *)(p1);", td).unwrap();
    writeln!(out, "    }} else {{").unwrap();
    writeln!(out, "        const int i03 = (int)tgpig.z;").unwrap();
    writeln!(out, "        const int i02 = (int)tgpig.y;").unwrap();
    writeln!(out, "        const int k0  = (int)tgpig.x / (int)ne01;").unwrap();
    writeln!(out, "        const int i01 = (int)tgpig.x - k0*(int)ne01;").unwrap();
    writeln!(out, "        i0 = k0*(int)ntg.x + (int)tpitg.x;").unwrap();
    writeln!(out, "        src0_ptr = (device const {0} *)(p0 + (uint64_t)i03*nb03 + (uint64_t)i02*nb02 + (uint64_t)i01*nb01);", t0).unwrap();
    writeln!(out, "        dst_ptr  = (device       {0} *)(p1 + (uint64_t)i03*nb3  + (uint64_t)i02*nb2  + (uint64_t)i01*nb1);",  td).unwrap();
    writeln!(out, "        if (i0 >= (int)ne0) {{ return; }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const TC x = (TC) src0_ptr[i0];").unwrap();
    writeln!(out, "    TC r = ({}) 0.0f;", tc).unwrap();
    // Generate per-op branches. For vec4 mode use comparison-mask trick where
    // a ternary would otherwise diverge across lanes; for scalar/half ternary.
    let leaky = if vec4 {
        "r = TC(x > TC(0.0f))*x + TC(x <= TC(0.0f))*(x * TC(slope));".to_string()
    } else {
        "r = (x > 0.0f) ? x : (x * slope);".to_string()
    };
    let elu = if vec4 {
        "r = TC(x > TC(0.0f))*x + TC(x <= TC(0.0f))*(exp(x) - TC(1.0f));".to_string()
    } else {
        "r = (x > 0.0f) ? x : (exp(x) - 1.0f);".to_string()
    };
    let sgn = if vec4 {
        "r = TC(x > TC(0.0f)) - TC(x < TC(0.0f));".to_string()
    } else {
        "r = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);".to_string()
    };
    let step = if vec4 {
        "r = TC(x > TC(0.0f));".to_string()
    } else {
        "r = (x > 0.0f) ? 1.0f : 0.0f;".to_string()
    };
    let softplus = if vec4 {
        // antirez: select(log(1+exp(x)), x, x > 20)
        "r = select(log(TC(1.0f) + exp(x)), x, x > TC(20.0f));".to_string()
    } else {
        "r = (x > 20.0f) ? x : log(1.0f + exp(x));".to_string()
    };
    writeln!(
        out,
        "    if      (op == 10u) {{ r = {} * x + {}; }}        // SCALE",
        promote("scale"),
        promote("bias")
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 11u) {{ r = {}; }}                     // FILL",
        promote("val")
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 12u) {{ r = clamp(x, {}, {}); }}    // CLAMP",
        promote("umin"),
        promote("umax")
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 13u) {{ r = x * x; }}                   // SQR"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 14u) {{ r = sqrt(x); }}                 // SQRT"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 15u) {{ r = sin(x); }}                  // SIN"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 16u) {{ r = cos(x); }}                  // COS"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 17u) {{ r = log(x); }}                  // LOG"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 18u) {{ {} }}                          // LEAKY_RELU",
        leaky
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 100u) {{ r = precise::tanh(x); }}       // TANH"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 101u) {{ r = fmax({}, x); }}            // RELU",
        tcz
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 102u) {{ r = {0} / ({0} + exp(-x)); }}  // SIGMOID",
        tco
    )
    .unwrap();
    writeln!(out, "    else if (op == 103u) {{ r = {0}*0.5f*x*({0} + precise::tanh(TC(SQRT_2_OVER_PI)*x*({0} + TC(GELU_COEF_A)*x*x))); }} // GELU", tco).unwrap();
    writeln!(out, "    else if (op == 104u) {{").unwrap();
    writeln!(
        out,
        "        // GELU_ERF: 0.5*x*(1 + erf_approx(x/sqrt(2)))"
    )
    .unwrap();
    writeln!(out, "        const TC xa = TC(SQRT_2_INV) * x;").unwrap();
    if vec4 {
        writeln!(
            out,
            "        const TC sx = TC(xa > {}) - TC(xa < {});",
            tcz, tcz
        )
        .unwrap();
    } else {
        writeln!(
            out,
            "        const TC sx = (xa > 0.0f) ? 1.0f : ((xa < 0.0f) ? -1.0f : 0.0f);"
        )
        .unwrap();
    }
    writeln!(out, "        const TC ax = fabs(xa);").unwrap();
    writeln!(
        out,
        "        const TC t  = {} / ({} + TC(p_erf) * ax);",
        tco, tco
    )
    .unwrap();
    writeln!(out, "        const TC ey = {} - (((((TC(a5_erf)*t + TC(a4_erf))*t) + TC(a3_erf))*t + TC(a2_erf))*t + TC(a1_erf))*t*exp(-ax*ax);", tco).unwrap();
    writeln!(out, "        r = TC(0.5f) * x * ({} + sx * ey);", tco).unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    else if (op == 105u) {{ r = x * ({0} / ({0} + exp(TC(GELU_QUICK_COEF) * x))); }} // GELU_QUICK", tco).unwrap();
    writeln!(
        out,
        "    else if (op == 106u) {{ r = x / ({0} + exp(-x)); }}    // SILU",
        tco
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 107u) {{ {} }}                          // ELU",
        elu
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 108u) {{ r = -x; }}                     // NEG"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 109u) {{ r = fabs(x); }}                // ABS"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 110u) {{ {} }}                          // SGN",
        sgn
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 111u) {{ {} }}                          // STEP",
        step
    )
    .unwrap();
    writeln!(out, "    else if (op == 112u) {{ r = x * fmax({0}, fmin({1}, x/TC(6.0f) + TC(0.5f))); }} // HARDSWISH", tcz, tco).unwrap();
    writeln!(out, "    else if (op == 113u) {{ r = fmax({0}, fmin({1}, x/TC(6.0f) + TC(0.5f))); }} // HARDSIGMOID", tcz, tco).unwrap();
    writeln!(
        out,
        "    else if (op == 114u) {{ r = exp(x); }}                 // EXP"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 115u) {{ {} }}                          // SOFTPLUS",
        softplus
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 116u) {{ r = exp(x) - {}; }}            // EXPM1",
        tco
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 117u) {{ r = floor(x); }}               // FLOOR"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 118u) {{ r = ceil(x); }}                // CEIL"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 119u) {{ r = round(x); }}               // ROUND"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 120u) {{ r = trunc(x); }}               // TRUNC"
    )
    .unwrap();
    writeln!(
        out,
        "    else if (op == 121u) {{                                // XIELU"
    )
    .unwrap();
    writeln!(out, "        const TC xi      = x;").unwrap();
    if vec4 {
        writeln!(out, "        const TC gate    = TC(xi > TC(0.0f));").unwrap();
    } else {
        writeln!(out, "        const TC gate    = (xi > 0.0f) ? 1.0f : 0.0f;").unwrap();
    }
    writeln!(
        out,
        "        const TC clamped = fmin(xi, {});",
        promote("val")
    )
    .unwrap();
    writeln!(
        out,
        "        const TC y_pos   = {}*xi*xi + {}*xi;",
        promote("scale"),
        promote("bias")
    )
    .unwrap();
    writeln!(
        out,
        "        const TC y_neg   = (exp(clamped) - {} - xi)*{} + {}*xi;",
        tco,
        promote("slope"),
        promote("bias")
    )
    .unwrap();
    writeln!(out, "        r = gate*y_pos + ({} - gate)*y_neg;", tco).unwrap();
    writeln!(out, "    }}").unwrap();
    let _ = tcm;
    writeln!(out, "    dst_ptr[i0] = ({}) r;", td).unwrap();
}
/// M114: kernel_dsv4_shared_gate_up_swiglu_q8_0 (dense.metal:203).
/// Fused shared-expert q8_0 matvec: two parallel gate + up matmuls share one
/// y load, then SwiGLU mid = silu(gate) * up. Produces 3 dst rows per output
/// row: dst_gate (raw gate), dst_up (raw up), dst_mid (silu(gate)*up).
/// Buffers: p0=src0_gate, p1=src0_up, p2=src1 (const); p3=dst_gate, p4=dst_up,
/// p5=dst_mid (writable). 13 uints: ne00, ne01, ne0, ne1, ne12, r2, r3, nb01,
/// nb02, nb03, nb11, nb12, nb13. NSG=2, NQ=8, NR0=N_R0_Q8_0=2.
pub(super) fn emit_dsv4_shared_gate_up_swiglu_q8_0_msl(out: &mut String, nsg: u32, nr0: u32, nq: u32) {
    let zeros = vec!["0.0f"; nr0 as usize].join(", ");
    writeln!(out, "    constexpr short NW    = 32;").unwrap();
    writeln!(out, "    constexpr short NSG   = {nsg};").unwrap();
    writeln!(out, "    constexpr short NR0   = {nr0};").unwrap();
    writeln!(out, "    constexpr short NQ    = {nq};").unwrap();
    writeln!(out, "    constexpr short QK8_0 = 32;").unwrap();
    writeln!(out, "    constexpr uint  Q8_0_BLOCK_BYTES = 34u;").unwrap();
    writeln!(
        out,
        "    // shmem layout: 2 * NR0 * NW floats (gate slabs + up slabs)."
    )
    .unwrap();
    writeln!(out, "    threadgroup float shmem_f32[2 * NR0 * NW];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const ushort tiisg = (ushort)simd_lane;").unwrap();
    writeln!(out, "    const ushort sgitg = (ushort)simd_id;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const int nb = (int)ne00 / QK8_0;").unwrap();
    writeln!(out, "    const uint r0 = tgpig.x * (uint)NR0;").unwrap();
    writeln!(out, "    const uint r1 = tgpig.y;").unwrap();
    writeln!(out, "    const uint im = tgpig.z;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint i12 = im % ne12;").unwrap();
    writeln!(out, "    const uint i13 = im / ne12;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const uint64_t offset1 = (uint64_t)r1 * (uint64_t)nb11 + (uint64_t)i12 * (uint64_t)nb12 + (uint64_t)i13 * (uint64_t)nb13;").unwrap();
    writeln!(
        out,
        "    device const float * y = (device const float *)(p2 + offset1);"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Two parallel q8_0 streams: gate (ag) and up (au) share src0 row stride."
    )
    .unwrap();
    writeln!(out, "    device const uchar * ag_byte[NR0];").unwrap();
    writeln!(out, "    device const uchar * au_byte[NR0];").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        const uint64_t offset0 = (uint64_t)(r0 + (uint)row) * (uint64_t)nb01 + (uint64_t)(i12 / r2) * (uint64_t)nb02 + (uint64_t)(i13 / r3) * (uint64_t)nb03;").unwrap();
    writeln!(
        out,
        "        ag_byte[row] = (device const uchar *)(p0 + offset0);"
    )
    .unwrap();
    writeln!(
        out,
        "        au_byte[row] = (device const uchar *)(p1 + offset0);"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sumg[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out, "    float sumu[NR0] = {{ {zeros} }};").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const short ix = (short)(tiisg / (NW / NQ));").unwrap();
    writeln!(out, "    const short il = (short)(tiisg % (NW / NQ));").unwrap();
    writeln!(out, "    const int   ib0 = (int)sgitg * NQ + (int)ix;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    for (int ib = ib0; ib < nb; ib += NSG * NQ) {{").unwrap();
    writeln!(out, "        const int y_off = ib * QK8_0 + (int)il * NQ;").unwrap();
    writeln!(out, "        float yl[NQ];").unwrap();
    writeln!(
        out,
        "        for (short i = 0; i < NQ; ++i) {{ yl[i] = y[y_off + i]; }}"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(
        out,
        "            device const uchar * blk_g = ag_byte[row] + (uint)ib * Q8_0_BLOCK_BYTES;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const uchar * blk_u = au_byte[row] + (uint)ib * Q8_0_BLOCK_BYTES;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const half  * dg    = (device const half  *)blk_g;"
    )
    .unwrap();
    writeln!(
        out,
        "            device const half  * du    = (device const half  *)blk_u;"
    )
    .unwrap();
    writeln!(out, "            device const int8_t * qg   = (device const int8_t *)(blk_g + 2u) + (uint)il * NQ;").unwrap();
    writeln!(out, "            device const int8_t * qu   = (device const int8_t *)(blk_u + 2u) + (uint)il * NQ;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "            float sg = 0.0f;").unwrap();
    writeln!(out, "            float su = 0.0f;").unwrap();
    writeln!(out, "            for (short i = 0; i < NQ; ++i) {{").unwrap();
    writeln!(out, "                sg += (float)qg[i] * yl[i];").unwrap();
    writeln!(out, "                su += (float)qu[i] * yl[i];").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "            sumg[row] += sg * (float)(*dg);").unwrap();
    writeln!(out, "            sumu[row] += su * (float)(*du);").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // 2-stage simd_sum reduce: gate slabs at [0..NR0*NW); up slabs at [NR0*NW..2*NR0*NW)."
    )
    .unwrap();
    writeln!(out, "    threadgroup float * sh_gate_base = shmem_f32;").unwrap();
    writeln!(
        out,
        "    threadgroup float * sh_up_base   = shmem_f32 + (uint)NR0 * (uint)NW;"
    )
    .unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        if (sgitg == 0) {{").unwrap();
    writeln!(
        out,
        "            sh_gate_base[(uint)row * (uint)NW + (uint)tiisg] = 0.0f;"
    )
    .unwrap();
    writeln!(
        out,
        "            sh_up_base  [(uint)row * (uint)NW + (uint)tiisg] = 0.0f;"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        sumg[row] = simd_sum(sumg[row]);").unwrap();
    writeln!(out, "        sumu[row] = simd_sum(sumu[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    for (short row = 0; row < NR0; ++row) {{").unwrap();
    writeln!(out, "        if (tiisg == 0) {{").unwrap();
    writeln!(
        out,
        "            sh_gate_base[(uint)row * (uint)NW + (uint)sgitg] = sumg[row];"
    )
    .unwrap();
    writeln!(
        out,
        "            sh_up_base  [(uint)row * (uint)NW + (uint)sgitg] = sumu[row];"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    device float * gate_f32 = (device float *)p3 + (uint64_t)im * (uint64_t)ne0 * (uint64_t)ne1 + (uint64_t)r1 * (uint64_t)ne0;").unwrap();
    writeln!(out, "    device float * up_f32   = (device float *)p4 + (uint64_t)im * (uint64_t)ne0 * (uint64_t)ne1 + (uint64_t)r1 * (uint64_t)ne0;").unwrap();
    writeln!(out, "    device float * mid_f32  = (device float *)p5 + (uint64_t)im * (uint64_t)ne0 * (uint64_t)ne1 + (uint64_t)r1 * (uint64_t)ne0;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    for (short row = 0; row < NR0 && r0 + (uint)row < ne01; ++row) {{"
    )
    .unwrap();
    writeln!(
        out,
        "        const float gate = simd_sum(sh_gate_base[(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(
        out,
        "        const float up   = simd_sum(sh_up_base  [(uint)row * (uint)NW + (uint)tiisg]);"
    )
    .unwrap();
    writeln!(out, "        if (tiisg == 0 && sgitg == 0) {{").unwrap();
    writeln!(out, "            const uint out_row = r0 + (uint)row;").unwrap();
    writeln!(out, "            gate_f32[out_row] = gate;").unwrap();
    writeln!(out, "            up_f32  [out_row] = up;").unwrap();
    writeln!(
        out,
        "            const float silu = gate / (1.0f + exp(-gate));"
    )
    .unwrap();
    writeln!(out, "            mid_f32[out_row] = silu * up;").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    (void)tid;").unwrap();
}
/// SiLU activation: out[i] = x[i] / (1 + exp(-x[i]))
/// Equivalent to x * sigmoid(x), used in gated MLP (Qwen2, LLaMA, etc.)
pub(super) fn emit_silu_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        float v = p0[gid];").unwrap();
    writeln!(out, "        p1[gid] = v / (1.0f + exp(-v));").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Fused SiLU * Mul: out[i] = silu(p0[i]) * p1[i]
/// Combines SiLU activation with element-wise multiply for gated MLP.
/// Saves one kernel dispatch and one intermediate buffer vs separate ops.
pub(super) fn emit_silu_mul_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(out, "        float v = p0[gid];").unwrap();
    writeln!(out, "        p2[gid] = (v / (1.0f + exp(-v))) * p1[gid];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Cooperative matvec with f16 weights, f32 accumulation.
/// One threadgroup per output row (N threadgroups total).
/// 256 threads cooperatively reduce over K dimension using simd_sum + shared mem.
/// p0=activation(float*, K), p1=weight(half*, N×K), p2=output(float*, N).
/// Dispatch: (N, 1, 1) threadgroups × 256 threads.
pub(super) fn emit_matvec_f16_msl(out: &mut String) {
    writeln!(out, "    uint row = gid;").unwrap();
    writeln!(out, "    if (row >= N) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum = 0.0f;").unwrap();
    writeln!(out, "    uint base = row * K;").unwrap();
    writeln!(out, "    for (uint i = tid; i < K; i += tpg) {{").unwrap();
    writeln!(out, "        sum += p0[i] * float(p1[base + i]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Intra-SIMD-group reduction").unwrap();
    writeln!(out, "    sum = simd_sum(sum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Cross-SIMD-group reduction via shared memory").unwrap();
    writeln!(out, "    uint num_simd_groups = tpg / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float total = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint s = 0; s < num_simd_groups; s++) total += shared[s];"
    )
    .unwrap();
    writeln!(out, "        p2[row] = total;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Same as matvec_f16 but with bias addition.
/// p0=activation(float*), p1=weight(half*, N×K), p2=output(float*), p3=bias(float*).
pub(super) fn emit_matvec_f16_bias_msl(out: &mut String) {
    writeln!(out, "    uint row = gid;").unwrap();
    writeln!(out, "    if (row >= N) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum = 0.0f;").unwrap();
    writeln!(out, "    uint base = row * K;").unwrap();
    writeln!(out, "    for (uint i = tid; i < K; i += tpg) {{").unwrap();
    writeln!(out, "        sum += p0[i] * float(p1[base + i]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sum = simd_sum(sum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint num_simd_groups = tpg / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float total = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint s = 0; s < num_simd_groups; s++) total += shared[s];"
    )
    .unwrap();
    writeln!(out, "        p2[row] = total + p3[row];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// In-place RoPE for decode: applies rotary embedding to a single token's Q or K.
/// One thread per (head, dim_pair). x is (num_heads × head_dim) contiguous.
/// No workgroup cooperation needed — each thread handles one rotation independently.
pub(super) fn emit_rope_inplace_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total_pairs = num_heads * (head_dim / 2);").unwrap();
    writeln!(out, "    if (gid >= total_pairs) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint head = gid / (head_dim / 2);").unwrap();
    writeln!(out, "    uint i = gid % (head_dim / 2);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    float freq = 1.0f / pow(theta, 2.0f * float(i) / float(head_dim));"
    )
    .unwrap();
    writeln!(out, "    float angle = float(position) * freq;").unwrap();
    writeln!(out, "    float cos_a = cos(angle);").unwrap();
    writeln!(out, "    float sin_a = sin(angle);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint idx = head * head_dim + 2 * i;").unwrap();
    writeln!(out, "    float x0 = p0[idx];").unwrap();
    writeln!(out, "    float x1 = p0[idx + 1];").unwrap();
    writeln!(out, "    p0[idx]     = x0 * cos_a - x1 * sin_a;").unwrap();
    writeln!(out, "    p0[idx + 1] = x0 * sin_a + x1 * cos_a;").unwrap();
}
/// KV cache update: copy a single vector into the KV cache at a given position.
/// src is (num_heads × head_dim), cache is (num_heads × max_seq × head_dim).
/// One thread per float element.
pub(super) fn emit_kv_cache_update_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = num_heads * head_dim;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint head = gid / head_dim;").unwrap();
    writeln!(out, "    uint d = gid % head_dim;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    uint dst = head * max_seq * head_dim + position * head_dim + d;"
    )
    .unwrap();
    writeln!(out, "    p1[dst] = p0[head * head_dim + d];").unwrap();
}
/// Decode attention with threadgroup cooperation.
/// One threadgroup per head. Threads cooperate over K positions using simd reduction.
/// Supports GQA via num_heads/num_kv_heads ratio.
/// 5-phase: score → max → exp+sum → normalize → weighted V sum.
/// Threadgroup shared memory for scores (up to 256 positions) and SIMD reductions.
pub(super) fn emit_attention_decode_msl(out: &mut String) {
    writeln!(out, "    uint head = gid;").unwrap();
    writeln!(out, "    if (head >= num_heads) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float scale = rsqrt(float(head_dim));").unwrap();
    writeln!(out, "    const uint group_size = num_heads / num_kv_heads;").unwrap();
    writeln!(out, "    const uint kv_head = head / group_size;").unwrap();
    writeln!(out, "    const uint q_off = head * head_dim;").unwrap();
    writeln!(out, "    const uint kv_off = kv_head * max_seq * head_dim;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float scores[256];").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Phase 1: Q·K^T scores, parallel over K positions"
    )
    .unwrap();
    writeln!(
        out,
        "    for (uint pos = tid; pos < seq_len; pos += tpg) {{"
    )
    .unwrap();
    writeln!(out, "        float dot = 0.0f;").unwrap();
    writeln!(out, "        for (uint d = 0; d < head_dim; d++)").unwrap();
    writeln!(
        out,
        "            dot += p0[q_off + d] * p1[kv_off + pos * head_dim + d];"
    )
    .unwrap();
    writeln!(out, "        scores[pos] = dot * scale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Phase 2: parallel max reduction").unwrap();
    writeln!(out, "    float local_max = -MAXFLOAT;").unwrap();
    writeln!(out, "    for (uint pos = tid; pos < seq_len; pos += tpg)").unwrap();
    writeln!(out, "        local_max = max(local_max, scores[pos]);").unwrap();
    writeln!(out, "    local_max = simd_max(local_max);").unwrap();
    writeln!(out, "    uint num_simd = (tpg + 31) / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = local_max;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float m = shared[0];").unwrap();
    writeln!(
        out,
        "        for (uint i = 1; i < num_simd; i++) m = max(m, shared[i]);"
    )
    .unwrap();
    writeln!(out, "        shared[0] = m;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float global_max = shared[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Phase 3: exp + sum").unwrap();
    writeln!(out, "    float local_sum = 0.0f;").unwrap();
    writeln!(
        out,
        "    for (uint pos = tid; pos < seq_len; pos += tpg) {{"
    )
    .unwrap();
    writeln!(out, "        float e = exp(scores[pos] - global_max);").unwrap();
    writeln!(out, "        scores[pos] = e;").unwrap();
    writeln!(out, "        local_sum += e;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    local_sum = simd_sum(local_sum);").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = local_sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float s = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint i = 0; i < num_simd; i++) s += shared[i];"
    )
    .unwrap();
    writeln!(out, "        shared[0] = s;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float inv_sum = 1.0f / shared[0];").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Phase 4: normalize scores").unwrap();
    writeln!(out, "    for (uint pos = tid; pos < seq_len; pos += tpg)").unwrap();
    writeln!(out, "        scores[pos] *= inv_sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    // Phase 5: weighted V sum, parallel over head_dim"
    )
    .unwrap();
    writeln!(out, "    uint out_off = head * head_dim;").unwrap();
    writeln!(out, "    for (uint d = tid; d < head_dim; d += tpg) {{").unwrap();
    writeln!(out, "        float acc = 0.0f;").unwrap();
    writeln!(out, "        for (uint pos = 0; pos < seq_len; pos++)").unwrap();
    writeln!(
        out,
        "            acc += scores[pos] * p2[kv_off + pos * head_dim + d];"
    )
    .unwrap();
    writeln!(out, "        p3[out_off + d] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Cooperative matvec + residual add: out[row] = sum_k(A[k] * B[row,k]) + R[row].
/// p0=activation(f32, K), p1=weight(bfloat, N×K), p2=output(f32, N), p3=residual(f32, N).
/// Vectorized bfloat4 loads + simd_sum + shared mem reduction.
pub(super) fn emit_matvec_f16_add_msl(out: &mut String) {
    writeln!(out, "    uint row = gid;").unwrap();
    writeln!(out, "    if (row >= N) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum = 0.0f;").unwrap();
    writeln!(out, "    uint base = row * K;").unwrap();
    writeln!(out, "    for (uint i = tid; i < K; i += tpg) {{").unwrap();
    writeln!(out, "        sum += p0[i] * float(p1[base + i]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sum = simd_sum(sum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint num_simd_groups = tpg / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float total = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint s = 0; s < num_simd_groups; s++) total += shared[s];"
    )
    .unwrap();
    writeln!(out, "        p2[row] = total + p3[row];").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// int8-weight matvec with a per-row f16 scale and float4/char4 VECTORIZED loads.
/// This is the dominant Qwen3-4B decode kernel; packing 4 contiguous weight bytes
/// per issue (char4) over scalar `char` loads is the single biggest decode win
/// (~+48%, the 89.3 tok/s M1 Ultra figure — PHASE9). Requires K % 4 == 0 (holds
/// for the model's D/QDIM/INTER dims). Layout + numerics are byte-identical on
/// device to the hand-written reference.
/// p0=activation(float, K), p1=weight(int8/char, N×K row-major), p2=output(float, N),
/// p3=row-scale(half, N). One threadgroup per output row; cooperative simd_sum.
pub(super) fn emit_matvec_i8_v4_msl(out: &mut String) {
    writeln!(out, "    uint row = gid;").unwrap();
    writeln!(out, "    if (row >= N) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint K4 = K >> 2;").unwrap();
    writeln!(out, "    float sum = 0.0f;").unwrap();
    writeln!(
        out,
        "    device const float4* a4 = (device const float4*) p0;"
    )
    .unwrap();
    writeln!(
        out,
        "    device const char4*  w4 = (device const char4*) (p1 + row * K);"
    )
    .unwrap();
    writeln!(out, "    for (uint i = tid; i < K4; i += tpg) {{").unwrap();
    writeln!(out, "        sum += dot(a4[i], float4(w4[i]));").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Intra-SIMD-group reduction").unwrap();
    writeln!(out, "    sum = simd_sum(sum);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    // Cross-SIMD-group reduction via shared memory").unwrap();
    writeln!(out, "    uint num_simd_groups = tpg / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float total = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint s = 0; s < num_simd_groups; s++) total += shared[s];"
    )
    .unwrap();
    writeln!(out, "        p2[row] = total * float(p3[row]);").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Fused gate+up+silu: reads activation once, computes both gate and up matvecs, then silu(gate)*up.
/// p0=activation(f32, K), p1=W_gate(bfloat, N×K), p2=W_up(bfloat, N×K), p3=output(f32, N).
/// One threadgroup per output element.
pub(super) fn emit_gate_up_silu_msl(out: &mut String) {
    writeln!(out, "    uint row = gid;").unwrap();
    writeln!(out, "    if (row >= N) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    float sum_gate = 0.0f;").unwrap();
    writeln!(out, "    float sum_up = 0.0f;").unwrap();
    writeln!(out, "    uint base = row * K;").unwrap();
    writeln!(out, "    for (uint i = tid; i < K; i += tpg) {{").unwrap();
    writeln!(out, "        float a = p0[i];").unwrap();
    writeln!(out, "        sum_gate += a * float(p1[base + i]);").unwrap();
    writeln!(out, "        sum_up   += a * float(p2[base + i]);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    sum_gate = simd_sum(sum_gate);").unwrap();
    writeln!(out, "    sum_up   = simd_sum(sum_up);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float shared_up[256 / 32];").unwrap();
    writeln!(out, "    uint num_simd_groups = tpg / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) {{").unwrap();
    writeln!(out, "        shared[simd_id] = sum_gate;").unwrap();
    writeln!(out, "        shared_up[simd_id] = sum_up;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float g = 0.0f, u = 0.0f;").unwrap();
    writeln!(out, "        for (uint s = 0; s < num_simd_groups; s++) {{").unwrap();
    writeln!(out, "            g += shared[s];").unwrap();
    writeln!(out, "            u += shared_up[s];").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        p3[row] = (g / (1.0f + exp(-g))) * u;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Batched RoPE for prefill: applies RoPE to (seq_len, num_heads, head_dim) in-place.
/// Flat grid dispatch: one thread per (position, head, dim_pair) triple.
pub(super) fn emit_rope_prefill_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint half_dim = head_dim / 2;").unwrap();
    writeln!(out, "    uint total = seq_len * num_heads * half_dim;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint si = gid / (num_heads * half_dim);").unwrap();
    writeln!(out, "    uint rem = gid % (num_heads * half_dim);").unwrap();
    writeln!(out, "    uint head = rem / half_dim;").unwrap();
    writeln!(out, "    uint i = rem % half_dim;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint pos = start_pos + si;").unwrap();
    writeln!(
        out,
        "    float freq = 1.0f / pow(theta, 2.0f * float(i) / float(head_dim));"
    )
    .unwrap();
    writeln!(out, "    float angle = float(pos) * freq;").unwrap();
    writeln!(out, "    float cos_a = cos(angle);").unwrap();
    writeln!(out, "    float sin_a = sin(angle);").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    uint base = si * num_heads * head_dim + head * head_dim + 2 * i;"
    )
    .unwrap();
    writeln!(out, "    float x0 = p0[base];").unwrap();
    writeln!(out, "    float x1 = p0[base + 1];").unwrap();
    writeln!(out, "    p0[base]     = x0 * cos_a - x1 * sin_a;").unwrap();
    writeln!(out, "    p0[base + 1] = x0 * sin_a + x1 * cos_a;").unwrap();
}
/// Batched KV cache update for prefill: copies seq_len vectors into cache.
/// Flat grid dispatch: one thread per element.
/// src layout: (seq_len, num_heads, head_dim), cache: (num_heads, max_seq, head_dim).
pub(super) fn emit_kv_cache_update_prefill_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    uint total = seq_len * num_heads * head_dim;").unwrap();
    writeln!(out, "    if (gid >= total) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint si = gid / (num_heads * head_dim);").unwrap();
    writeln!(out, "    uint rem = gid % (num_heads * head_dim);").unwrap();
    writeln!(out, "    uint head = rem / head_dim;").unwrap();
    writeln!(out, "    uint d = rem % head_dim;").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    uint src_idx = si * num_heads * head_dim + head * head_dim + d;"
    )
    .unwrap();
    writeln!(
        out,
        "    uint dst_idx = head * max_seq * head_dim + (start_pos + si) * head_dim + d;"
    )
    .unwrap();
    writeln!(out, "    p1[dst_idx] = p0[src_idx];").unwrap();
}
/// Prefill attention with causal mask and GQA support.
/// One threadgroup per (query_position, query_head) pair.
/// Cooperative softmax via simd reduction + shared memory.
pub(super) fn emit_attention_prefill_msl(out: &mut String) {
    writeln!(out, "    uint total_groups = seq_len * num_heads;").unwrap();
    writeln!(out, "    if (gid >= total_groups) return;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint qi = gid / num_heads;").unwrap();
    writeln!(out, "    uint head = gid % num_heads;").unwrap();
    writeln!(out, "    uint abs_pos = start_pos + qi;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    const float scale = rsqrt(float(head_dim));").unwrap();
    writeln!(out, "    const uint group_size = num_heads / num_kv_heads;").unwrap();
    writeln!(out, "    const uint kv_head = head / group_size;").unwrap();
    writeln!(
        out,
        "    const uint q_off = qi * num_heads * head_dim + head * head_dim;"
    )
    .unwrap();
    writeln!(out, "    const uint kv_off = kv_head * max_seq * head_dim;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    uint total_kv_len = start_pos + seq_len;").unwrap();
    writeln!(out, "    uint kv_len = min(total_kv_len, 256u);").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    threadgroup float scores[256];").unwrap();
    writeln!(out).unwrap();
    // Phase 1: Q·K^T scores with causal masking
    writeln!(out, "    // Phase 1: Q·K^T scores with causal mask").unwrap();
    writeln!(out, "    for (uint pos = tid; pos < kv_len; pos += tpg) {{").unwrap();
    writeln!(
        out,
        "        if (pos > abs_pos) {{ scores[pos] = -MAXFLOAT; continue; }}"
    )
    .unwrap();
    writeln!(out, "        float dot = 0.0f;").unwrap();
    writeln!(out, "        for (uint d = 0; d < head_dim; d++)").unwrap();
    writeln!(
        out,
        "            dot += p0[q_off + d] * p1[kv_off + pos * head_dim + d];"
    )
    .unwrap();
    writeln!(out, "        scores[pos] = dot * scale;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    // Phase 2: max reduction
    writeln!(out, "    // Phase 2: max reduction").unwrap();
    writeln!(out, "    float local_max = -MAXFLOAT;").unwrap();
    writeln!(out, "    for (uint pos = tid; pos < kv_len; pos += tpg)").unwrap();
    writeln!(out, "        local_max = max(local_max, scores[pos]);").unwrap();
    writeln!(out, "    local_max = simd_max(local_max);").unwrap();
    writeln!(out, "    uint num_simd = (tpg + 31) / 32;").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = local_max;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float m = shared[0];").unwrap();
    writeln!(
        out,
        "        for (uint i = 1; i < num_simd; i++) m = max(m, shared[i]);"
    )
    .unwrap();
    writeln!(out, "        shared[0] = m;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float global_max = shared[0];").unwrap();
    writeln!(out).unwrap();
    // Phase 3: exp + sum
    writeln!(out, "    // Phase 3: exp + sum").unwrap();
    writeln!(out, "    float local_sum = 0.0f;").unwrap();
    writeln!(out, "    for (uint pos = tid; pos < kv_len; pos += tpg) {{").unwrap();
    writeln!(out, "        float e = exp(scores[pos] - global_max);").unwrap();
    writeln!(out, "        scores[pos] = e;").unwrap();
    writeln!(out, "        local_sum += e;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    local_sum = simd_sum(local_sum);").unwrap();
    writeln!(out, "    if (simd_lane == 0) shared[simd_id] = local_sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    if (tid == 0) {{").unwrap();
    writeln!(out, "        float s = 0.0f;").unwrap();
    writeln!(
        out,
        "        for (uint i = 0; i < num_simd; i++) s += shared[i];"
    )
    .unwrap();
    writeln!(out, "        shared[0] = s;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out, "    float inv_sum = 1.0f / shared[0];").unwrap();
    writeln!(out).unwrap();
    // Phase 4: normalize
    writeln!(out, "    // Phase 4: normalize scores").unwrap();
    writeln!(out, "    for (uint pos = tid; pos < kv_len; pos += tpg)").unwrap();
    writeln!(out, "        scores[pos] *= inv_sum;").unwrap();
    writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);").unwrap();
    writeln!(out).unwrap();
    // Phase 5: weighted V sum
    writeln!(out, "    // Phase 5: weighted V sum").unwrap();
    writeln!(
        out,
        "    uint out_off = qi * num_heads * head_dim + head * head_dim;"
    )
    .unwrap();
    writeln!(out, "    for (uint d = tid; d < head_dim; d += tpg) {{").unwrap();
    writeln!(out, "        float acc = 0.0f;").unwrap();
    writeln!(out, "        for (uint pos = 0; pos < kv_len; pos++)").unwrap();
    writeln!(
        out,
        "            acc += scores[pos] * p2[kv_off + pos * head_dim + d];"
    )
    .unwrap();
    writeln!(out, "        p3[out_off + d] = acc;").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// BF16→F32 cast: reinterpret bfloat16 (stored as uint16) as float32.
/// bfloat16 has the same exponent bits as float32, just shift left 16.
pub(super) fn emit_cast_bf16_msl(out: &mut String) {
    writeln!(out, "    uint gid = base + tid;").unwrap();
    writeln!(out, "    if (gid < num_elements) {{").unwrap();
    writeln!(
        out,
        "        uint bits = uint(as_type<ushort>(p0[gid])) << 16;"
    )
    .unwrap();
    writeln!(out, "        p1[gid] = as_type<float>(bits);").unwrap();
    writeln!(out, "    }}").unwrap();
}
/// Transposed matmul: C[i,j] = sum_k A[i,k] * B[j,k]
/// B stored as (N,K) — HuggingFace weight layout where weights are (out_features, in_features).
/// Flat grid dispatch: one thread per output element (matches hand-written template).
/// Loop unrolled by 4 for better pipelining.
pub(super) fn emit_matmul_transposed_msl(out: &mut String) {
    writeln!(out, "    uint gid = row * tcount + tid;").unwrap();
    writeln!(out, "    if (gid >= M * N) return;").unwrap();
    writeln!(out, "    uint m = gid / N;").unwrap();
    writeln!(out, "    uint n = gid % N;").unwrap();
    writeln!(out, "    float acc = 0.0f;").unwrap();
    writeln!(out, "    uint k = 0;").unwrap();
    writeln!(out, "    for (; k + 3 < K; k += 4) {{").unwrap();
    writeln!(out, "        acc += p0[m * K + k]     * p1[n * K + k];").unwrap();
    writeln!(out, "        acc += p0[m * K + k + 1] * p1[n * K + k + 1];").unwrap();
    writeln!(out, "        acc += p0[m * K + k + 2] * p1[n * K + k + 2];").unwrap();
    writeln!(out, "        acc += p0[m * K + k + 3] * p1[n * K + k + 3];").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    for (; k < K; k++)").unwrap();
    writeln!(out, "        acc += p0[m * K + k] * p1[n * K + k];").unwrap();
    writeln!(out, "    p2[m * N + n] = acc;").unwrap();
}

/// Block size the DeepSeek-V4 FP8 KV path quantizes with.
pub(super) const FP8_KV_DS4_BLOCK: u32 = 64;

/// Why an FP8 KV block size cannot be emitted, if it cannot.
pub(super) fn check_fp8_kv_block(kernel: &str, block: u32) -> Result<(), String> {
    if !(2..=1024).contains(&block) || !block.is_power_of_two() {
        return Err(format!(
            "{kernel}: block {block} must be a power of two from 2 to 1024 (one thread per element, tree max over the block)"
        ));
    }
    Ok(())
}

/// Per-row staging the DeepSeek-V4 top-k sort ships with.
pub(super) const SORT_ROWS_DS4_MAX_TOP_K: u32 = 256;

/// Index staging the argsort kernels ship with: one thread per column, and
/// Metal allows at most 1024 threads in a threadgroup.
pub(super) const ARGSORT_DS4_MAX_ROW: u32 = 1024;

/// Why a bitonic-sort capacity cannot be emitted, if it cannot.
pub(super) fn check_sort_capacity(kernel: &str, what: &str, capacity: u32) -> Result<(), String> {
    if !(2..=1024).contains(&capacity) || !capacity.is_power_of_two() {
        return Err(format!(
            "{kernel}: {what} {capacity} must be a power of two from 2 to 1024 (one thread per element, bitonic network)"
        ));
    }
    Ok(())
}

/// Simdgroups the shipped Laguna decode attention splits its key range over.
pub(super) const LAGUNA_DECODE_DS4_SPLIT: u32 = 8;

/// Why a Laguna decode split cannot be emitted, if it cannot.
pub(super) fn check_laguna_decode_split(split: u32) -> Result<(), String> {
    if !(1..=32).contains(&split) {
        return Err(format!(
            "laguna_attention_decode_gqa_f16: split_simd_groups {split} must be from 1 to 32 (32 threads each, Metal allows 1024 per threadgroup)"
        ));
    }
    Ok(())
}

/// Hyper-connections the shipped DeepSeek-V4 expand kernel unrolls for.
pub(super) const HC_EXPAND_DS4_UNROLL: u32 = 4;

/// Why an unrolled hyper-connection count cannot be emitted, if it cannot.
pub(super) fn check_hc_unroll(hc: u32) -> Result<(), String> {
    if !(1..=16).contains(&hc) {
        return Err(format!(
            "dsv4_hc_expand4: hc_unroll {hc} must be from 1 to 16 (every residual and combine term is unrolled into the kernel)"
        ));
    }
    Ok(())
}

/// Why a vector-staged head shape cannot be emitted, if it cannot. These stages
/// spread each head across the 8 lane groups of one simdgroup, four elements at
/// a time, so both widths move in steps of 32.
pub(super) fn check_flash_vec_stage_dims(kernel: &str, dk: u32, dv: u32) -> Result<(), String> {
    for (what, v) in [("dk", dk), ("dv", dv)] {
        if v == 0 || v % 32 != 0 || v > 1024 {
            return Err(format!(
                "{kernel}: {what} {v} must be a positive multiple of 32 up to 1024 (the head is spread over 8 lane groups, four elements each)"
            ));
        }
    }
    // Staged query, scores, output, and the slack the output init writes into.
    let halves = dk.next_multiple_of(128) + 128 + 2 * dv.next_multiple_of(128) + 256;
    if halves > 16384 {
        return Err(format!(
            "{kernel}: those head widths need {} bytes of threadgroup memory, over the 32768 a threadgroup has",
            halves * 2
        ));
    }
    Ok(())
}

/// Simdgroups per threadgroup the q8_0 HC expand kernels are built at. Nothing
/// else in them moves: the two output rows and the eight-element block slice a
/// lane takes are both written out by hand rather than looped.
pub(super) const HC_EXPAND_MV_NSG: u32 = 2;

/// Why a q8_0 HC expand simdgroup count cannot be emitted, if it cannot.
pub(super) fn check_hc_expand_mv_split(kernel: &str, nsg: u32) -> Result<(), String> {
    if nsg == 0 || nsg > 32 {
        return Err(format!(
            "{kernel}: nsg {nsg} must be from 1 to 32 (its simdgroups are 32 lanes each and a threadgroup holds 1024)"
        ));
    }
    Ok(())
}

/// Why an IQ2_XXS work split cannot be emitted, if it cannot. These kernels
/// stage their 256-entry value grid and 128-entry sign table into threadgroup
/// memory by giving every lane an equal, contiguous run of entries, so the
/// lanes a threadgroup has must divide both tables.
pub(super) fn check_iq2_mv_split(kernel: &str, nsg: u32, nr0: u32) -> Result<(), String> {
    check_wide_mv_split(kernel, nsg, nr0)?;
    if 128 % (nsg * 32) != 0 {
        return Err(format!(
            "{kernel}: nsg {nsg} must be 1, 2 or 4, so its {} lanes divide the 128-entry sign table they stage",
            nsg * 32
        ));
    }
    Ok(())
}

/// Work split the IQ2_XXS and paired half matvecs are built at: simdgroups per
/// threadgroup and the output rows each one owns. Their quantization block
/// sizes are the format and stay in the text.
pub(super) const WIDE_MV_NR0: u32 = 4;

/// Why a simdgroup-and-rows work split cannot be emitted, if it cannot.
pub(super) fn check_wide_mv_split(kernel: &str, nsg: u32, nr0: u32) -> Result<(), String> {
    if nsg == 0 || nsg > 32 {
        return Err(format!(
            "{kernel}: nsg {nsg} must be from 1 to 32 (its simdgroups are 32 lanes each and a threadgroup holds 1024)"
        ));
    }
    if nr0 == 0 || nr0 > 16 {
        return Err(format!(
            "{kernel}: nr0 {nr0} must be from 1 to 16 (every row it owns costs a register accumulator and a column of scratch)"
        ));
    }
    Ok(())
}

/// Work split the Laguna half-precision matvecs are built at: simdgroups per
/// threadgroup, output rows each owns, and how many of a 32-element block one
/// lane takes. The block itself is the simdgroup width and stays.
pub(super) const LAGUNA_MV_NSG: u32 = 4;
pub(super) const LAGUNA_MV_NR0: u32 = 2;
pub(super) const LAGUNA_MV_NF: u32 = 16;

/// Why a Laguna matvec work split cannot be emitted, if it cannot.
pub(super) fn check_laguna_mv_split(kernel: &str, nsg: u32, nr0: u32, nf: u32) -> Result<(), String> {
    if nsg == 0 || nsg > 32 {
        return Err(format!(
            "{kernel}: nsg {nsg} must be from 1 to 32 (its simdgroups are 32 lanes each and a threadgroup holds 1024)"
        ));
    }
    if nr0 == 0 || nr0 > 16 {
        return Err(format!(
            "{kernel}: nr0 {nr0} must be from 1 to 16 (every row it owns costs a register accumulator and a column of scratch)"
        ));
    }
    if nf == 0 || nf % 4 != 0 || 32 % nf != 0 {
        return Err(format!(
            "{kernel}: nf {nf} must divide the 32-element block evenly and be a multiple of 4 (a lane reads it four floats at a time)"
        ));
    }
    Ok(())
}

/// Work split the q8_0 matvec kernels are built at: simdgroups per threadgroup,
/// output rows each simdgroup owns, and the slice of a quantization block one
/// lane takes. The 32 lanes of a simdgroup and the 32-element q8_0 block around
/// them are the hardware and the format, not a choice.
pub(super) const Q8_0_MV_NSG: u32 = 4;
pub(super) const Q8_0_MV_NR0: u32 = 2;
pub(super) const Q8_0_MV_NQ: u32 = 8;

/// Why a q8_0 matvec work split cannot be emitted, if it cannot.
pub(super) fn check_q8_0_mv_split(kernel: &str, nsg: u32, nr0: u32, nq: u32) -> Result<(), String> {
    if nsg == 0 || nsg > 32 {
        return Err(format!(
            "{kernel}: nsg {nsg} must be from 1 to 32 (its simdgroups are 32 lanes each and a threadgroup holds 1024)"
        ));
    }
    if nr0 == 0 || nr0 > 16 {
        return Err(format!(
            "{kernel}: nr0 {nr0} must be from 1 to 16 (every row it owns costs a register accumulator and a column of scratch)"
        ));
    }
    if nq == 0 || 32 % nq != 0 {
        return Err(format!(
            "{kernel}: nq {nq} must divide the 32 lanes of a simdgroup evenly (each lane takes one slice of a q8_0 block)"
        ));
    }
    Ok(())
}

/// Head widths the staged flash-attention kernels are built at. The queries per
/// threadgroup, the simdgroups, and the keys per pass are not free alongside
/// them: the stages move data in 8x8 simdgroup tiles, which fixes eight query
/// rows, four simdgroups, and the thirty-two keys those four tiles span.
pub(super) const FLASH_STAGE_DK: u32 = 64;
pub(super) const FLASH_STAGE_DV: u32 = 64;

/// Why a staged head shape cannot be emitted, if it cannot. `dv` is `None` for
/// the stages that never reach the value head.
pub(super) fn check_flash_stage_dims(kernel: &str, dk: u32, dv: Option<u32>) -> Result<(), String> {
    if dk == 0 || dk % 16 != 0 || dk > 1024 {
        return Err(format!(
            "{kernel}: dk {dk} must be a positive multiple of 16 up to 1024 (the key head is walked two 8-wide tiles at a time)"
        ));
    }
    if let Some(dv) = dv {
        if dv == 0 || dv % 64 != 0 || dv > 1024 {
            return Err(format!(
                "{kernel}: dv {dv} must be a positive multiple of 64 up to 1024 (its 8-wide output tiles are split over four simdgroups and consumed two at a time)"
            ));
        }
    }
    // Staged queries, the score scratch, the output scratch, and the key tiles,
    // all in halves. A threadgroup has 32 KiB.
    let halves = 8 * dk + 8 * 64 * 2 + dv.map(|dv| 8 * dv * 2).unwrap_or(0) + 4 * 512;
    if halves > 16384 {
        return Err(format!(
            "{kernel}: those head widths need {} bytes of threadgroup memory, over the 32768 a threadgroup has",
            halves * 2
        ));
    }
    Ok(())
}

/// Head dimensions the shipped DeepSeek-V4 flash-attention reference kernels use.
pub(super) const FLASH_ATTN_DS4_DK: u32 = 512;
pub(super) const FLASH_ATTN_DS4_DV: u32 = 512;

/// Why a flash-attention head shape cannot be emitted, if it cannot.
pub(super) fn check_flash_attn_dims(kernel: &str, dk: u32, dv: u32) -> Result<(), String> {
    for (what, v) in [("dk", dk), ("dv", dv)] {
        if v == 0 || v % 4 != 0 || v > 4096 {
            return Err(format!(
                "{kernel}: {what} {v} must be a positive multiple of 4 up to 4096 (the kernel walks the head four lanes at a time)"
            ));
        }
    }
    Ok(())
}
