//! Tile-level abstractions for Ascend NPU kernels.
//!
//! This module provides a type-safe tile buffer handle and tile operations
//! that mirror the PTO (Programmable Tile Operations) ISA for Ascend.
//!
//! # Design
//!
//! Each tile operation is backed by an `extern "C"` intrinsic that carries
//! shape information as const generic parameters. The `mlir_to_cpp` codegen
//! translates these to AscendC `TBuf`/`DataCopy` API calls (current path),
//! and `mlir_to_pto` will translate them to `.pto` assembly (future path).
//!
//! # Ownership and safety
//!
//! `Tile<ROWS, COLS, T>` is a *move-only* handle (no `Copy`). This ensures:
//! - Each tile buffer is consumed by exactly one `tile_store` (no double-DMA)
//! - Tile values cannot alias once loaded (borrow checker enforces single owner)
//!
//! # Example
//!
//! ```rust,ignore
//! use tile_std::tile::{tile_load_f32, tile_store_f32, tile_add_f32};
//!
//! #[kernel]
//! pub fn vec_add(a: *const f32, b: *const f32, c: *mut f32, n: u32) {
//!     let idx = unsafe { get_block_idx() };
//!     let off = idx * 16;
//!     let ta: Tile<16, 1, f32> = unsafe { tile_load_f32::<16, 1>(a.add(off)) };
//!     let tb: Tile<16, 1, f32> = unsafe { tile_load_f32::<16, 1>(b.add(off)) };
//!     let tc = unsafe { tile_add_f32(ta, tb) };
//!     unsafe { tile_store_f32(c.add(off), tc) };
//! }
//! ```

use crate::core::marker::PhantomData;

/// A typed tile buffer handle.
///
/// Encodes the tile dimensions (`ROWS` × `COLS`) and element type `T`
/// as const generics so that shape mismatches are compile errors.
///
/// `Tile` is intentionally not `Copy`: each tile must be consumed by
/// exactly one store operation, preventing double-DMA bugs.
pub struct Tile<const ROWS: usize, const COLS: usize, T> {
    /// Internal buffer ID used by the kernel runtime.
    /// This maps to a `TBuf` handle in the AscendC C++ codegen path.
    pub(crate) buf_id: u32,
    _phantom: PhantomData<T>,
}

// --- Device-memory views --------------------------------------------------
//
// `GmView` and `GmViewMut` are the typed replacement for bare `*const T` /
// `*mut T` kernel parameters. They carry:
//
//   * the element type `T` (so `tile_load` knows the dtype without a suffix),
//   * the tile shape `ROWS × COLS` (so the host must commit to a concrete
//     shape at the call site — shape mismatches between host and kernel
//     become compile errors instead of memory corruption),
//   * a lifetime `'a` (so the view can't be stashed in a global or returned
//     past the end of the kernel).
//
// Both are `#[repr(transparent)]` around a raw pointer, so `extern "C"`
// kernels still accept them with the same ABI as `*const T` / `*mut T`.
// The `mlir_to_pto` codegen reads pointer-kind from MIR's `ptr<1>` addrspace
// and is blind to Rust newtypes; lowering is unchanged.
//
// The ONLY unsafe primitives are `GmDeviceCtx::new()` and
// `ctx.view{,_mut}::<R,C,T>(ptr)`: the one place where the caller is
// asserting "this pointer really does back `R * C` elements of `T`".
// In practice that's the host-side launcher, once per kernel invocation.
// Everything downstream — every `tile_load_view_*` / `tile_store_view_*`
// call, and every kernel body — is safe.
//
// Views borrow `&'a self` from the ctx so they cannot outlive it; returning
// a view from a function is a borrow-checker error.

/// Typed, shape-annotated read-only view over `ROWS × COLS` elements of
/// `T` in device global memory.
#[repr(transparent)]
pub struct GmView<'a, const ROWS: usize, const COLS: usize, T> {
    ptr: *const T,
    _brand: PhantomData<&'a T>,
}

/// Typed, shape-annotated mutable view over `ROWS × COLS` elements of `T`
/// in device global memory.
#[repr(transparent)]
pub struct GmViewMut<'a, const ROWS: usize, const COLS: usize, T> {
    ptr: *mut T,
    _brand: PhantomData<&'a mut T>,
}

impl<'a, const R: usize, const C: usize, T> GmView<'a, R, C, T> {
    /// Drop back to a raw pointer. The inner intrinsics need this; end
    /// users typically don't.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<'a, const R: usize, const C: usize, T> GmViewMut<'a, R, C, T> {
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
}

/// Device-memory context: a zero-sized token whose lifetime `'a` brands
/// every view it mints. The launcher owns one, and views borrow from it,
/// so no view can outlive the launch.
///
/// The two `unsafe` hops (`new` + `view{,_mut}`) are the only unsafe the
/// caller ever writes, and they sit at the host-side boundary where
/// "this pointer really backs R*C elements" is a claim the caller can
/// make in good faith.
pub struct GmDeviceCtx<'a> {
    _brand: PhantomData<&'a mut ()>,
}

impl<'a> GmDeviceCtx<'a> {
    /// Create a fresh context.
    ///
    /// # Safety
    /// The caller is asserting that any pointers they will feed into
    /// `view` / `view_mut` on this ctx are valid for the entire scope
    /// of `'a` (which is bounded by the ctx's stack slot).
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Self { _brand: PhantomData }
    }

    /// Mint a typed read-only view. The returned view is borrowed against
    /// `self`, so it can't outlive the ctx.
    ///
    /// # Safety
    /// `ptr` must back at least `R*C` contiguous readable elements of `T`.
    #[inline(always)]
    pub unsafe fn view<const R: usize, const C: usize, T>(
        &'a self, ptr: *const T,
    ) -> GmView<'a, R, C, T> {
        GmView { ptr, _brand: PhantomData }
    }

    /// Mint a typed mutable view.
    ///
    /// # Safety
    /// `ptr` must back at least `R*C` contiguous writable elements of `T`
    /// and must not alias any other live view from this ctx.
    #[inline(always)]
    pub unsafe fn view_mut<const R: usize, const C: usize, T>(
        &'a self, ptr: *mut T,
    ) -> GmViewMut<'a, R, C, T> {
        GmViewMut { ptr, _brand: PhantomData }
    }
}

// --- f32 tile operations --------------------------------------------------

extern "C" {
    /// Load a `ROWS × COLS` f32 tile from global memory into a local tile buffer.
    ///
    /// In the AscendC path: `DataCopy(local_buf, gm_ptr, ROWS * COLS)`.
    /// In the PTO path: `tile.load %dst, %src, [ROWS, COLS]`.
    ///
    /// # Safety
    /// `gm` must point to at least `ROWS * COLS` valid f32 values in global memory.
    pub fn __tile_load_f32(gm: *const f32, rows: u32, cols: u32) -> u32;

    /// Store a local tile buffer to global memory.
    ///
    /// Consumes the tile buffer (by ID). In the AscendC path: `DataCopy(gm_ptr, local_buf, N)`.
    /// In the PTO path: `tile.store %dst, %src, [ROWS, COLS]`.
    ///
    /// # Safety
    /// `gm` must point to at least `ROWS * COLS` writable f32 values in global memory.
    pub fn __tile_store_f32(gm: *mut f32, buf: u32, rows: u32, cols: u32);

    /// Element-wise add two f32 tiles of the same shape.
    pub fn __tile_add_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise multiply two f32 tiles of the same shape.
    pub fn __tile_mul_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise exp of an f32 tile.
    pub fn __tile_exp_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise softmax of an f32 tile (in-place reduction over columns).
    pub fn __tile_softmax_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Matrix multiply: (M × K) @ (K × N) → (M × N), all f32.
    pub fn __tile_matmul_f32(dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32) -> u32;

    /// Element-wise subtract two f32 tiles: a - b.
    pub fn __tile_sub_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise divide two f32 tiles: a / b.
    pub fn __tile_div_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise negate an f32 tile: -a.
    pub fn __tile_neg_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise max reduction of an f32 tile.
    /// Returns a (ROWS × 1) tile where each row is the max of that row.
    pub fn __tile_reduce_max_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise sum reduction of an f32 tile.
    pub fn __tile_reduce_sum_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Scalar multiply: scale all elements of a tile by a scalar.
    pub fn __tile_scale_f32(dst: u32, src: u32, scalar: f32, rows: u32, cols: u32) -> u32;

    /// Element-wise clamp into `[lo, hi]`. Used to fence activations under
    /// MoE / DS4-style gating (cf. DS4 SwiGLU clamp_value).
    pub fn __tile_clamp_f32(dst: u32, src: u32, lo: f32, hi: f32, rows: u32, cols: u32) -> u32;

    /// Fused attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// Q: (S × D), K: (S × D), V: (S × D) → out: (S × D)
    /// Fuses matmul + scale + softmax + matmul into a single PTOAS pipeline.
    pub fn __tile_attention_f32(
        dst: u32, q: u32, k: u32, v: u32,
        seq_len: u32, head_dim: u32,
    ) -> u32;

    // ── Transformer building-block intrinsics ───────────────────────

    /// SiLU/Swish activation: silu(x) = x * sigmoid(x).
    pub fn __tile_silu_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// RoPE: applies sin/cos rotation to (R, D) tile at position `pos`.
    pub fn __tile_rope_f32(dst: u32, src: u32, pos: u32, rows: u32, cols: u32) -> u32;

    /// DS4 partial-RoPE: rotate trailing n_dims of head_dim with YaRN scaling. Copy n_nope prefix unchanged.
    /// p0=src, p1=pos(int32), p2=src2_freq(float, optional), p3=dst.
    pub fn __tile_rope_dsv4_f32(
        dst: u32, src: u32, pos: u32, src2: u32,
        ne01: u32, ne02: u32, ne00: u32,
    ) -> u32;

    /// DS4 kernel_dsv4_rope_tail_f32 (M122, antirez dsv4_rope.metal:68): byte-stride partial-RoPE with YaRN.
    /// p0=src0 (char* float source), p1=src1 (char* int32 pos), p2=src2 (char* float freq_factor, ignored when has_src2=0),
    /// p3=dst (char* float). 4D dims ne00..ne03 + byte strides nb00..nb03 + dst byte strides nb0..nb3.
    /// 3D grid (i1, i2, i3) = tgpig.{x,y,z}; tcount lanes sweep i0 across ne00.
    pub fn __tile_dsv4_rope_tail_f32(
        src0: u32, src1: u32, src2: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        nb0: u32, nb1: u32, nb2: u32, nb3: u32,
        n_dims: u32, mode: u32, n_ctx_orig: u32, inverse: u32, has_src2: u32,
        freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32,
        beta_fast: f32, beta_slow: f32,
    ) -> u32;

    /// DS4 kernel_flash_attn_ext_f16_dk512_dv512 (M123). Host-callable prefill
    /// FlashAttention with DK=DV=512 half K/V rows. Specialized: Q=8, C=64.
    /// p0=q (half), p1=k (half), p2=v (half), p3=mask (half), p4=sinks (float),
    /// p5=pad (unused), p6=blk (unused), p7=dst (float DV).
    pub fn __tile_flash_attn_ext_f16_dk512_dv512(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, blk: u32, dst: u32,
        ne01: u32, ne02: u32, ne03: u32, nb01: u32, nb02: u32, nb03: u32,
        ne11: u32, ne_12_2: u32, ne_12_3: u32, nb11: u32, nb12: u32, nb13: u32,
        nb21: u32, nb22: u32, nb23: u32,
        ne31: u32, ne32: u32, ne33: u32, nb31: u32, nb32: u32, nb33: u32,
        scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: u32,
        logit_softcap: f32, has_mask: u32, has_sinks: u32, has_bias: u32, has_softcap: u32,
    ) -> u32;

    /// DS4 kernel_flash_attn_ext_vec_f16_dk512_dv512 (M124) — decode-shape sibling of M123.
    /// Same FlashAttention algorithm with Q=1 row per dispatch and output layout
    /// dst[rid*DV+d] where rid = iq3*ne2*ne1 + iq2 + iq1*ne1.
    pub fn __tile_flash_attn_ext_vec_f16_dk512_dv512(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, dst: u32,
        ne01: u32, ne02: u32, ne03: u32, nb01: u32, nb02: u32, nb03: u32,
        ne11: u32, ne_12_2: u32, ne_12_3: u32, nb11: u32, nb12: u32, nb13: u32,
        nb21: u32, nb22: u32, nb23: u32,
        ne31: u32, ne32: u32, ne33: u32, nb31: u32, nb32: u32, nb33: u32,
        ne1: u32, ne2: u32, ne3: u32,
        scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: u32,
        logit_softcap: f32, has_mask: u32, has_sinks: u32, has_bias: u32, has_softcap: u32,
    ) -> u32;

    /// DS4 KV ratio-4 recurrent-state shift: state[i] = state[4*width + i] for two state buffers.
    /// p0=state_kv, p1=state_score (both writable). 1D dispatch over n=4*width elements.
    pub fn __tile_dsv4_ratio4_shift_f32(state_kv: u32, state_score: u32, width: u32) -> u32;

    /// DS4 kernel_dsv4_topk_mask (dsv4_misc.metal:237): 1D grid -INFINITY mask fill.
    /// p0=topk (byte ptr, read-only — unused; kept for ABI parity), p1=dst (byte ptr, writable float).
    /// For gid<ne0*ne1: dst[ic*nb0 + it*nb1] = -INFINITY where ic=gid%ne0, it=gid/ne0.
    pub fn __tile_dsv4_topk_mask_f32(
        topk: u32, dst: u32,
        ne00: u32, ne01: u32, nb00: u32, nb01: u32,
        ne0: u32, ne1: u32, nb0: u32, nb1: u32,
    ) -> u32;

    /// DS4 topk_mask_scatter: for gid in [0, num_elements), idx = topk[gid]; if 0<=idx<dst_len, dst[idx] = 0.0.
    /// p0=topk (int*), p1=dst (float*, modified in place).
    pub fn __tile_topk_mask_scatter_f32(topk: u32, dst: u32, num_elements: u32, dst_len: u32) -> u32;

    /// DS4 kernel_dsv4_q8_hc_expand4_q8_0 (dsv4_hc.metal:728) M126: fused decode-time q8_0 matvec + 4-channel HC expansion.
    /// 7 char* bufs (weight, input, block_out (write), residual, post, comb, dst (write)).
    /// Two struct uniforms emitted by the codegen — passed here as (ne00, ne01, nb01) and the 9 HC stride uniforms.
    /// Scalar-correctness reference: hardcodes NSG=2 NW=32 NQ=8 NR0=2; n_hc must be 4 and n_tokens must be 1.
    pub fn __tile_dsv4_q8_hc_expand4_q8_0(
        weight: u32, input: u32, block_out: u32, residual: u32, post: u32, comb: u32, dst: u32,
        ne00: u32, ne01: u32, nb01: u32,
        n_hc: u32, n_tokens: u32, nb_block0: u32,
        nb_res0: u32, nb_res1: u32, nb_post0: u32,
        nb_comb0: u32, nb_comb1: u32, nb0: u32, nb1: u32,
    ) -> u32;

    /// DS4 kernel_dsv4_shared_down_hc_expand4_q8_0 (dsv4_hc.metal:607) M127: M126 add-sibling.
    /// 8 char* bufs (weight, shared_mid, shared_out (write), routed_out, residual, post, comb, dst (write)).
    /// Same struct uniforms as M126. n_hc=4, n_tokens=1.
    /// Body: q8_0 matvec → shared_v; block_v = routed_out[d*nb_block0] + shared_v; then identical HC expand.
    pub fn __tile_dsv4_shared_down_hc_expand4_q8_0(
        weight: u32, shared_mid: u32, shared_out: u32, routed_out: u32,
        residual: u32, post: u32, comb: u32, dst: u32,
        ne00: u32, ne01: u32, nb01: u32,
        n_hc: u32, n_tokens: u32, nb_block0: u32,
        nb_res0: u32, nb_res1: u32, nb_post0: u32,
        nb_comb0: u32, nb_comb1: u32, nb0: u32, nb1: u32,
    ) -> u32;

    /// DS4 router_weights_one: w[i] = probs[selected[i]] / max(min_sum, Σ probs[selected[*]]) * scale.
    /// p0=probs (float*), p1=selected (int*), p2=weights (float*). Antirez defaults: scale=1.5, min_sum=6.103515625e-5.
    pub fn __tile_router_weights_one_f32(probs: u32, selected: u32, weights: u32, num_experts: u32) -> u32;

    /// DS4 indexer_weighted_sum: dst[t,c] = Σ_{h<H} max(scores[t,c,h],0) * weights[t,h] * scale.
    /// p0=scores (T*C*H), p1=weights (T*H), p2=dst (T*C). Dispatch one thread per (t,c).
    pub fn __tile_indexer_weighted_sum_f32(scores: u32, weights: u32, dst: u32, num_tokens: u32, num_cols: u32, num_heads: u32) -> u32;

    /// DS4 sort_i32_rows_asc: bitonic sort each row of an int32 (num_rows × top_k) buffer ascending.
    /// p0=src (int*), p1=dst (int*). One threadgroup per row, top_k threads per group.
    /// top_k must be a power of two ≤ 256.
    pub fn __tile_sort_i32_rows_asc_i32(src: u32, dst: u32, top_k: u32, num_rows: u32) -> u32;

    /// DS4 softmax_pool: per (id, ic) reduce dst[ic,id] = Σ_ir softmax(score[ir,id,ic]) * kv[ir,id,ic].
    /// p0=kv (R*ne1*ne0), p1=score (R*ne1*ne0), p2=dst (ne1*ne0). Dispatch one thread per (id, ic).
    pub fn __tile_softmax_pool_f32(kv: u32, score: u32, dst: u32, ne00: u32, ne0: u32, ne1: u32) -> u32;

    /// DS4 compressor_store_one: 5-buffer KV+score store with positional encoding (APE) addition.
    /// p0=kv (width), p1=score (width), p2=ape (ratio*width — f32 path; f16 TBD),
    /// p3=state_kv (8*width), p4=state_score (8*width). One thread per `gid` in [0, width).
    pub fn __tile_compressor_store_one_f32(kv: u32, score: u32, ape: u32, state_kv: u32, state_score: u32, width: u32, ratio: u32, pos: u32) -> u32;

    /// DS4 kv_fp8_store: per-row n_nope chunked-64 fp8 round-trip + n_rot tail half-cast.
    /// p0=kv (head_dim, in/out), p1=raw_cache (raw_row*head_dim base offset, write).
    /// Single threadgroup of 64 threads, no batching: one row per dispatch.
    pub fn __tile_kv_fp8_store_f32(kv: u32, raw_cache: u32, head_dim: u32, n_rot: u32, raw_row: u32) -> u32;

    /// DS4 fp8_kv_quantize: 4D batched n_nope chunked-64 fp8 round-trip.
    /// p0=src0 (read), p1=dst (write). Element-stride params nb01_e/nb02_e/nb03_e
    /// for src, nb1_e/nb2_e/nb3_e for dst (driver pre-divides byte strides by 4).
    /// Dispatches (n_rows = ne01*ne02*ne03) threadgroups of 64 threads each.
    pub fn __tile_fp8_kv_quantize_f32(
        src: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb01_e: u32, nb02_e: u32, nb03_e: u32,
        nb1_e: u32, nb2_e: u32, nb3_e: u32,
        n_rot: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_pad: byte-stride DMA padding of K/V/mask into a single dst buffer.
    /// dst layout: k_pad | v_pad | mask_pad (where mask is half-precision, 2 bytes/elem).
    /// Dispatches (C, ne_12_2_max, ne_12_3_max) threadgroups, ntg_x threads per tg.
    pub fn __tile_flash_attn_ext_pad_f32(
        k: u32, v: u32, mask: u32, dst: u32,
        ne11: u32, ne_12_2: u32, ne_12_3: u32,
        nb11: u32, nb12: u32, nb13: u32,
        nb21: u32, nb22: u32, nb23: u32,
        ne31: u32, ne32: u32, ne33: u32,
        nb31: u32, nb32: u32, nb33: u32,
        has_mask: u32, c_ncpsg: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_blk: simdgroup-reduce mask scan, writes per-block status byte.
    /// Buffers: mask (half), dst (u8 per-block status: 0 keep, 1 normal, 2 all-zero).
    pub fn __tile_flash_attn_ext_blk_f32(
        mask: u32, dst: u32,
        ne01: u32, ne30: u32, ne32: u32, ne33: u32,
        nb31: u32, nb32: u32, nb33: u32,
        q_nqptg: u32, c_ncpsg: u32,
    ) -> u32;

    /// DS4 dsv4_indexer_score_one_direct: per-row fused 64-head scoring.
    /// Buffers: q, weights, index_comp, scores.
    pub fn __tile_indexer_score_one_direct_f32(
        q: u32, weights: u32, index_comp: u32, scores: u32,
        n_comp: u32, q_head_stride: u32, index_row_stride: u32, scale: u32,
    ) -> u32;

    /// DS4 dsv4_router_finalize_one: 256-thread bitonic top-6 over (probs+bias).
    /// Buffers: probs(float, 256), bias(float, 256), hash(int*), tokens(int*), selected(int, 6 out).
    /// hash_mode short-circuits to copying hash[token*6..+6] into selected.
    pub fn __tile_router_finalize_one_f32(
        probs: u32, bias: u32, hash: u32, tokens: u32, selected: u32,
        has_bias: u32, hash_mode: u32, use_token_buffer: u32, token: u32, hash_rows: u32,
    ) -> u32;

    /// DS4 dsv4_indexer_scores_tiled_f32: 8x32 tile fused indexer scoring with simdgroup matmul.
    /// Buffers: q, weights, index_comp, scores (all char*).
    pub fn __tile_indexer_scores_tiled_f32(
        q: u32, weights: u32, index_comp: u32, scores: u32,
        n_comp: u32, n_tokens: u32, n_head: u32, pos0: u32, ratio: u32,
        q_token_stride: u32, q_head_stride: u32, weights_token_stride: u32,
        index_row_stride: u32, score_token_stride: u32, scale: u32,
    ) -> u32;

    /// DS4 dsv4_indexed_mixed_attention_heads8: ratio-4 mixed attention,
    /// 1 token × 8 heads per threadgroup, online softmax with half K dot/accum.
    /// Buffers: q, raw_kv, comp_kv, topk, sinks, dst (all char*).
    pub fn __tile_indexed_mixed_attention_h8_f32(
        q: u32, raw_kv: u32, comp_kv: u32, topk: u32, sinks: u32, dst: u32,
        n_tokens: u32, n_head: u32, n_raw: u32, n_comp: u32, top_k: u32, ratio: u32,
        window: u32, pos0: u32, raw_start: u32, raw_cap: u32,
        q_token_stride: u32, q_head_stride: u32, raw_row_stride: u32,
        comp_row_stride: u32, topk_token_stride: u32,
        dst_token_stride: u32, dst_head_stride: u32,
        scale: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_vec_reduce: split-K decode reducer that merges NWG
    /// partial output vectors and (S,M) softmax states into the final attention.
    /// Buffers: htmp (char* in), dst (char* out).
    pub fn __tile_flash_attn_ext_vec_reduce_f32(
        htmp: u32, dst: u32, nrows: u32, dv: u32, nwg: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_vec stage M36a: kernel signature + Q→sq4 load echo.
    /// Reproduces antirez's threadgroup memory layout and Q load. Buffers:
    /// q, k, v, mask, sinks, pad (char* in); dst (char* out, ne01 × DK4 float4).
    /// Params: dk, dv, ne01, nb01.
    pub fn __tile_flash_attn_ext_vec_setup_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, nb01: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_vec stage M36b: setup + K·Q dot + online softmax merge.
    /// Specialized to DK=DV=64, NE=4, NL=8, C=32, NSG=NWG=1. All FC flags off.
    /// Buffers: q, k, v, mask, sinks, pad (char* in); dst (char* out, ne01 × 2 floats [S,M]).
    /// Params: dk, dv, ne01, ne11, nb01, nb11, scale.
    pub fn __tile_flash_attn_ext_vec_score_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, ne11: u32, nb01: u32, nb11: u32, scale: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_vec stage M36c: full single-SG flash-attention output.
    /// Score body (M36b) + V accumulation + per-row attention output write.
    /// Specialized to DK=DV=64, NE=4, NL=8, C=32, NSG=NWG=1. All FC flags off.
    /// Buffers: q, k, v, mask, sinks, pad (char* in); dst (char* out, ne01 × DV floats).
    /// Params: dk, dv, ne01, ne11, nb01, nb11, nb21, scale.
    pub fn __tile_flash_attn_ext_vec_out_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, ne11: u32, nb01: u32, nb11: u32, nb21: u32, scale: u32,
    ) -> u32;

    /// DS4 flash_attn_ext_vec stage M36d: M36c + has_mask + has_sinks paths baked in.
    /// Mask buffer is half[ne01 * ne11], sinks buffer is float[ne01].
    /// Same buffers/params as M36c.
    pub fn __tile_flash_attn_ext_vec_out_ms_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, ne11: u32, nb01: u32, nb11: u32, nb21: u32, scale: u32,
    ) -> u32;

    /// DS4 flash_attn_ext (non-vec, prefill) stage M37a: setup + Q load echo.
    /// Stages NQ queries (per threadgroup) across NSG simdgroups into shared
    /// memory and echoes the staged data back to dst for verification.
    /// 8 buffers: q, k, v, mask, sinks, pad, blk (in), dst (out).
    pub fn __tile_flash_attn_ext_setup_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, blk: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, nb01: u32,
    ) -> u32;

    /// DS4 flash_attn_ext (non-vec, prefill) stage M37b: M37a + K·Q simdgroup
    /// matmul + online softmax merge across the full ne11 KV extent. Output
    /// (S, M) per query row to dst. K is half[ne11 × DK], no FC flags on.
    pub fn __tile_flash_attn_ext_score_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, blk: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, ne11: u32, nb01: u32, nb11: u32, scale: u32,
    ) -> u32;

    /// DS4 flash_attn_ext (non-vec, prefill) stage M37c: M37b + V matmul +
    /// per-row S-normalized attention output. Output is float[ne01 × DV] in
    /// dst. V is half[ne11 × DV] with row stride nb21 bytes, no FC flags on.
    pub fn __tile_flash_attn_ext_out_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, blk: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, ne11: u32, nb01: u32, nb11: u32, nb21: u32, scale: u32,
    ) -> u32;

    /// DS4 flash_attn_ext (non-vec, prefill) stage M37d: M37c + has_mask FMA
    /// in softmax + has_sinks post-loop merge. mask is half[ne01 × ne11] (row
    /// major), sinks is float[1] (single-head test). Output float[ne01 × DV].
    pub fn __tile_flash_attn_ext_out_ms_f32(
        q: u32, k: u32, v: u32, mask: u32, sinks: u32, pad: u32, blk: u32, dst: u32,
        dk: u32, dv: u32, ne01: u32, ne11: u32, nb01: u32, nb11: u32, nb21: u32, scale: u32,
    ) -> u32;

    /// DS4 dsv4_hc_expand: per-(d, dst_hc, t) HC expand step. Computes
    ///   acc = (block_out[d,t] + has_add ? block_add[d,t] : 0) * post[dst_hc,t]
    ///       + Σ_{src_hc} comb[dst_hc,src_hc,t] * residual[d,src_hc,t]
    /// Buffers: block_out, residual, post, comb, block_add (in), dst (out).
    pub fn __tile_dsv4_hc_expand_f32(
        block_out: u32, residual: u32, post: u32, comb: u32, block_add: u32, dst: u32,
        n_embd: u32, n_hc: u32, n_tokens: u32,
        nb_block0: u32, nb_block1: u32, nb_add0: u32, nb_add1: u32,
        nb_res0: u32, nb_res1: u32, nb_res2: u32,
        nb_post0: u32, nb_post1: u32,
        nb_comb0: u32, nb_comb1: u32, nb_comb2: u32,
        nb0: u32, nb1: u32, nb2: u32, has_add: u32,
    ) -> u32;

    /// DS4 dsv4_hc_expand4: HC=4 specialization. Same args layout as expand;
    /// one thread writes all 4 dst_hc streams. Total = n_embd × n_tokens.
    pub fn __tile_dsv4_hc_expand4_f32(
        block_out: u32, residual: u32, post: u32, comb: u32, block_add: u32, dst: u32,
        n_embd: u32, n_hc: u32, n_tokens: u32,
        nb_block0: u32, nb_block1: u32, nb_add0: u32, nb_add1: u32,
        nb_res0: u32, nb_res1: u32, nb_res2: u32,
        nb_post0: u32, nb_post1: u32,
        nb_comb0: u32, nb_comb1: u32, nb_comb2: u32,
        nb0: u32, nb1: u32, nb2: u32, has_add: u32,
    ) -> u32;

    /// DS4 dsv4_hc_weighted_sum: per-(d, t) reduce
    ///   dst[d,t] = Σ_h x[d,h,t] * weights[h,t].
    /// Buffers: x (in), weights (in), dst (out). 1D dispatch over n_embd × n_tokens.
    pub fn __tile_dsv4_hc_weighted_sum_f32(
        x: u32, weights: u32, dst: u32,
        n_embd: u32, n_hc: u32, n_tokens: u32,
        nb_x0: u32, nb_x1: u32, nb_x2: u32,
        nb_w0: u32, nb_w1: u32,
        nb0: u32, nb1: u32,
    ) -> u32;

    /// DS4 dsv4_hc_split_sinkhorn HC=4 fast path. Per-row HC mixer split
    /// (sigmoid pre / 2*sigmoid post / softmax + Sinkhorn-balanced 4×4 comb).
    /// Buffers: mixes (n_rows × mix_hc), scale[3], base[mix_hc], dst (n_rows × mix_hc).
    pub fn __tile_dsv4_hc_split_sinkhorn_hc4_f32(
        mixes: u32, scale: u32, base: u32, dst: u32,
        n_rows: u32, n_hc: u32, mix_hc: u32, sinkhorn_iters: u32, eps: u32,
    ) -> u32;

    /// DS4 dsv4_hc_split_weighted_sum HC=4 fast path. One threadgroup per row:
    /// tid=0 does HC mixer split (caches pre[0..3] in shmem, writes split[]),
    /// then all lanes loop d to produce dst[d,row] = Σ_h pre[h] * x[d,h,row].
    /// Buffers: mixes, scale, base, x (in), split, dst (out).
    pub fn __tile_dsv4_hc_split_weighted_sum_hc4_f32(
        mixes: u32, scale: u32, base: u32, x: u32, split: u32, dst: u32,
        n_embd: u32, n_hc: u32, n_rows: u32, sinkhorn_iters: u32,
        nb_mix1: u32, nb_split1: u32, nb_x0: u32, nb_x1: u32, nb_x2: u32,
        nb0: u32, nb1: u32, eps: u32,
    ) -> u32;

    /// DS4 dsv4_hc_split_weighted_sum_norm4: M42 fused with float4 RMSNorm.
    /// Hardcoded n_embd=4096, n_hc=4, n4=1024 float4 lanes per row.
    /// Per row: tid=0 mixer split, all lanes float4 reduce x*pre + sum(v·v),
    /// cross-simd reduce → rsqrt → write dst (raw) and norm_dst (scaled × weight).
    /// Buffers: mixes, scale, base, x (in), split (out), dst (out),
    /// norm_weight (in), norm_dst (out).
    pub fn __tile_dsv4_hc_split_weighted_sum_norm4_f32(
        mixes: u32, scale: u32, base: u32, x: u32, split: u32, dst: u32,
        norm_weight: u32, norm_dst: u32,
        n_embd: u32, n_hc: u32, n_rows: u32, sinkhorn_iters: u32,
        nb_mix1: u32, nb_split1: u32, nb_x0: u32, nb_x1: u32, nb_x2: u32,
        nb0: u32, nb1: u32, nb_norm1: u32,
        eps: u32, norm_eps: u32,
    ) -> u32;

    /// DS4 argsort_f32_i32_desc: bitonic sort one float row → int32 index
    /// permutation, descending. One threadgroup per row. Threadgroup size
    /// must be a power of two ≥ ne00. Buffers: src (float row), dst (int).
    pub fn __tile_argsort_f32_i32_desc(
        src: u32, dst: u32,
        ne00: u32, ne01: u32, top_k: u32, ne0: u32, nb01: u32,
    ) -> u32;

    /// DS4 argsort_merge_f32_i32_desc: merge two pre-sorted descending int32
    /// index runs into one descending top_k run. Single-batch, one threadgroup.
    /// Buffers: src (float row, char* in emitter), tmp (int* const, two runs at
    /// [0..len) and [len..2*len)), dst (int* writable). Params: ne0 (output
    /// stride / total source width), top_k, len, nb01 (src row byte stride).
    pub fn __tile_argsort_merge_f32_i32_desc(
        src: u32, tmp: u32, dst: u32,
        ne0: u32, top_k: u32, len: u32, nb01: u32,
    ) -> u32;

    /// DS4 kernel_argsort_f32_i32_desc M134 full host_name (argsort.metal:108):
    /// Bitonic sort one float row into an int32 index row, descending. Full 4-D
    /// batched surface from antirez. Dispatched as (ib*ne01, ne02, ne03)
    /// threadgroups; ntg.x threads per group must be a power of two and large
    /// enough to cover ne00 (or step ne00 in ntg.x-sized blocks when ib > 0).
    pub fn __tile_argsort_f32_i32_desc_full(
        src0: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne0: u32, ne1: u32, ne2: u32, ne3: u32,
        top_k: u32,
    ) -> u32;

    /// DS4 kernel_argsort_merge_f32_i32_desc M135 full host_name (argsort.metal:266):
    /// Merge two pre-sorted descending int32 index runs (produced by M134) into one
    /// descending top_k run. Full 4-D batched surface. Dispatched as (im*ne01, ne02,
    /// ne03) threadgroups; ntg.x threads per group cooperate via per-thread chunks of
    /// total = len0+len1.
    pub fn __tile_argsort_merge_f32_i32_desc_full(
        src0: u32, tmp: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne0: u32, ne1: u32, ne2: u32, ne3: u32,
        top_k: u32, len: u32,
    ) -> u32;

    /// DS4 dsv4_moe_swiglu_weight: per-row fused SwiGLU + route-weight scale.
    /// mid[i] = silu(clamp(gate[i], +c)) * clamp(up[i], +-c) * weights[0].
    /// Optional clamp writeback when write_clamped != 0. One threadgroup per
    /// row. Buffers all char*: gate (writable), up (writable), mid (writable),
    /// weights (const).
    pub fn __tile_moe_swiglu_weight_f32(
        gate: u32, up: u32, mid: u32, weights: u32,
        width: u32, rows: u32,
        gate_row_stride: u32, up_row_stride: u32, mid_row_stride: u32,
        weight_stride: u32, write_clamped: u32, clamp_value: u32,
    ) -> u32;

    /// DS4 dsv4_moe_swiglu_weight_f16: same as the f32 variant but mid is
    /// half-precision (mid_row[i] = (half)(silu(g) * u * weights[0])). Cuts
    /// the large mid write/read traffic; grouped MM converts F32 → half
    /// before MMA anyway, so this does not change effective MM input precision.
    pub fn __tile_moe_swiglu_weight_f16(
        gate: u32, up: u32, mid: u32, weights: u32,
        width: u32, rows: u32,
        gate_row_stride: u32, up_row_stride: u32, mid_row_stride: u32,
        weight_stride: u32, write_clamped: u32, clamp_value: u32,
    ) -> u32;

    /// DS4 mul_mm_id_map0: per-expert MoE ID-map builder. One threadgroup,
    /// one thread per expert. For each token row in src2 (length ne21), each
    /// expert id scans its ne20 selected experts; on a match, appends the
    /// flat slot index to hids[ide][...] and increments tpe[ide].
    /// Buffers all char*: src2 (const), htpe (writable), hids (writable).
    pub fn __tile_mul_mm_id_map0_f32(
        src2: u32, htpe: u32, hids: u32,
        ne20: u32, ne21: u32, nb21: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_map0_ne20_8 M136 (moe.metal:1510): full host_name
    /// surface for ne20=8 template instantiation of kernel_mul_mm_id_map0.
    /// 3 char* bufs (src2 const, htpe writable, hids writable). 8 uniforms
    /// matching antirez ds4_metal_args_mul_mm_id_map0 layout (ne02, ne10, ne11,
    /// nb11, nb12, ne21, ne20, nb21); body bakes NE20=8 and only reads ne21/nb21.
    pub fn __tile_mul_mm_id_map0_ne20_8_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M137 sibling of M136 with NE20=4 baked.
    pub fn __tile_mul_mm_id_map0_ne20_4_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M138 sibling of M136 with NE20=1 baked.
    pub fn __tile_mul_mm_id_map0_ne20_1_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M139 sibling of M136 with NE20=2 baked.
    pub fn __tile_mul_mm_id_map0_ne20_2_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M140 sibling of M136 with NE20=5 baked.
    pub fn __tile_mul_mm_id_map0_ne20_5_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M141 sibling of M136 with NE20=6 baked.
    pub fn __tile_mul_mm_id_map0_ne20_6_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M142 sibling of M136 with NE20=10 baked.
    pub fn __tile_mul_mm_id_map0_ne20_10_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M143 sibling of M136 with NE20=16 baked.
    pub fn __tile_mul_mm_id_map0_ne20_16_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// M144 sibling of M136 with NE20=22 baked.
    pub fn __tile_mul_mm_id_map0_ne20_22_full(
        src2: u32, htpe: u32, hids: u32,
        ne02: u32, ne10: u32, ne11: u32, nb11: u32, nb12: u32,
        ne21: u32, ne20: u32, nb21: u32,
    ) -> u32;

    /// DS4 dsv4_qkv_rms_norm_f32_4: fused float4 RMSNorm of q-lora row and
    /// KV row in one dispatch. Grid (rows, 2, 1): y=0 → q row, y=1 → KV row.
    /// Each row reduces ‖x‖² via simd_sum + threadgroup shmem and writes
    /// y[i] = (x[i] * rsqrt(mean+eps)) * w[i]. Buffers all char*: q_src,
    /// q_weight (const), q_dst (writable), kv_src, kv_weight (const),
    /// kv_dst (writable).
    pub fn __tile_qkv_rms_norm_f32_4(
        q_src: u32, q_weight: u32, q_dst: u32,
        kv_src: u32, kv_weight: u32, kv_dst: u32,
        q_n: u32, q_n4: u32, kv_n: u32, kv_n4: u32,
        q_row_stride: u32, kv_row_stride: u32, eps: u32,
    ) -> u32;

    /// DS4 kernel_rms_norm_mul_f32_4 (norm.metal F=2 variant): float4-vectorized
    /// RMSNorm of one row × per-element learned weight. Grid is one tg per row
    /// (tgpig.x), and reduction uses simd_sum + threadgroup shmem broadcast.
    /// Buffers all char*: src (const), weight (const), dst (writable).
    pub fn __tile_rms_norm_mul_f32_4(
        src: u32, weight: u32, dst: u32,
        n: u32, n4: u32, row_stride: u32, eps: u32,
    ) -> u32;

    /// DS4 kernel_rms_norm_f32_4 (norm.metal F=1 variant): plain float4-vectorized
    /// RMSNorm of one row, no weight multiply. Same dispatch as
    /// `__tile_rms_norm_mul_f32_4` minus the weight buffer.
    pub fn __tile_rms_norm_f32_4(
        src: u32, dst: u32,
        n: u32, n4: u32, row_stride: u32, eps: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 (softmax.metal, no-mask/no-sink path):
    /// per-row online softmax over float4 lanes with cross-SIMD reduction.
    /// Applies an input scale before max/exp/sum. Buffers: src (const), dst.
    /// Params: ne00 (cols), nb01 (src row stride bytes), nb1 (dst row
    /// stride bytes), scale.
    pub fn __tile_softmax_f32_4_strided(
        src: u32, dst: u32,
        ne00: u32, nb01: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 mask path (f16 mask, no sink, slope=1):
    /// adds `pmask[i00]` (half→float) into the score before max/exp.
    /// Buffers: src (const, float4 row), mask (const, half4 row), dst.
    /// Params: ne00 (cols), nb01 (src row stride bytes), nb_mask (mask
    /// row stride bytes), nb1 (dst row stride bytes), scale.
    pub fn __tile_softmax_f32_4_mask_f16(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 mask path (f32 mask). Same dispatch and
    /// param list as the f16 variant; mask buffer is float4 instead of half4.
    pub fn __tile_softmax_f32_4_mask_f32(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> (softmax.metal scalar lanewise template,
    /// no-mask/no-sink path): per-row online softmax over scalar floats
    /// with cross-SIMD reduction. Used when the row width is not a
    /// multiple of 4. Same buffer + param shape as the float4 variant —
    /// `ne00` is in elements, not float4 lanes.
    pub fn __tile_softmax_f32_scalar_strided(
        src: u32, dst: u32,
        ne00: u32, nb01: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> mask path (f16 mask, no sink, slope=1).
    /// Scalar lanewise sibling of `_4_mask_f16` — buffers same shape but
    /// rows are scalar `float`/`half` rather than float4/half4.
    pub fn __tile_softmax_f32_scalar_mask_f16(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> mask path (f32 mask, no sink, slope=1).
    /// Scalar lanewise sibling of `_4_mask_f32`.
    pub fn __tile_softmax_f32_scalar_mask_f32(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 sink path (no mask). Adds a per-row sink
    /// scalar from `sink[row]`: `lmax` is initialized to it and the final
    /// denominator gains `exp(sink[row] - max_val)`. The sink itself is not
    /// written. Buffers: src (const, float4 row), sink (const, one float
    /// per row), dst (writable). Params: ne00, nb01, nb1, scale.
    pub fn __tile_softmax_f32_4_sink(
        src: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> sink path (no mask, scalar lanewise).
    /// Scalar lanewise sibling of `_4_sink`. Same sink fold-in semantics.
    pub fn __tile_softmax_f32_scalar_sink(
        src: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 mask + sink (f16 mask). Combines the M68
    /// mask add with the M72 sink fold-in. Buffers: src, mask (half row),
    /// sink (one float per row), dst.
    pub fn __tile_softmax_f32_4_mask_f16_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 mask + sink (f32 mask). Same shape as
    /// the f16 variant but mask is float4 instead of half4.
    pub fn __tile_softmax_f32_4_mask_f32_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> mask + sink (f16 mask, scalar lanewise).
    /// Scalar lanewise sibling of `_4_mask_f16_sink`. Combines the M71 mask
    /// add with the M73 sink fold-in. Buffers: src, mask (half row), sink
    /// (one float per row), dst.
    pub fn __tile_softmax_f32_scalar_mask_f16_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> mask + sink (f32 mask, scalar lanewise).
    /// Same shape as the f16 scalar variant but mask is float instead of half.
    pub fn __tile_softmax_f32_scalar_mask_f32_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 ALiBi mask path (f16 mask, no sink).
    /// `slope` is a host-supplied per-launch scalar — caller pre-computes
    /// `pow(base, exp)` from the head index. Score = src*scale + slope*mask.
    pub fn __tile_softmax_f32_4_alibi_f16(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 ALiBi mask path (f32 mask, no sink).
    pub fn __tile_softmax_f32_4_alibi_f32(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 ALiBi + sink (f16 mask). Combines ALiBi
    /// mask scaling with M72 sink fold-in.
    pub fn __tile_softmax_f32_4_alibi_f16_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 ALiBi + sink (f32 mask).
    pub fn __tile_softmax_f32_4_alibi_f32_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> ALiBi mask path (f16 mask, no sink, scalar lanewise).
    /// Scalar sibling of `__tile_softmax_f32_4_alibi_f16`.
    pub fn __tile_softmax_f32_scalar_alibi_f16(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> ALiBi mask path (f32 mask, no sink, scalar lanewise).
    pub fn __tile_softmax_f32_scalar_alibi_f32(
        src: u32, mask: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> ALiBi + sink (f16 mask, scalar lanewise).
    pub fn __tile_softmax_f32_scalar_alibi_f16_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_soft_max<float> ALiBi + sink (f32 mask, scalar lanewise).
    pub fn __tile_softmax_f32_scalar_alibi_f32_sink(
        src: u32, mask: u32, sink: u32, dst: u32,
        ne00: u32, nb01: u32, nb_mask: u32, nb1: u32, scale: u32, slope: u32,
    ) -> u32;

    /// DS4 kernel_sum_rows_f32_f32 (sum_rows.metal): per-row reduction
    /// Σ_i src[i] writing one float per (i1,i2,i3) cell. One tg per
    /// (i1,i2,i3) — i1=tgpig.x, i2=tgpig.y, i3=tgpig.z. `_strided`
    /// suffix avoids shadowing the legacy `__tile_sum_rows_f32`
    /// (cpp-tile / no-batch variant) which is still mapped to the
    /// classic `KernelType::SumRows` emitter.
    pub fn __tile_sum_rows_f32_strided(
        src: u32, dst: u32,
        ne00: u32, nb01: u32, nb02: u32, nb03: u32,
        nb1: u32, nb2: u32, nb3: u32,
    ) -> u32;

    /// DS4 kernel_cpy_f32_f32 (cpy.metal): strided/typed copy across graph
    /// boundaries. Source dims (ne00..ne03) + strides (nb00..nb03) and dest
    /// dims (ne0..ne3) + strides (nb0..nb3) may differ for layout
    /// materialization. The `_strided` suffix mirrors the M70 pattern and
    /// avoids shadowing the legacy `__tile_cpy_f32` mapping.
    pub fn __tile_cpy_f32_f32_strided(
        src: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne0:  u32, ne1:  u32, ne2:  u32, ne3:  u32,
        nb0:  u32, nb1:  u32, nb2:  u32, nb3:  u32,
    ) -> u32;

    /// DS4 kernel_cpy_f32_f16 (cpy.metal): f32 src → f16 dst sibling of
    /// `__tile_cpy_f32_f32_strided`. Same dims/strides.
    pub fn __tile_cpy_f32_f16_strided(
        src: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne0:  u32, ne1:  u32, ne2:  u32, ne3:  u32,
        nb0:  u32, nb1:  u32, nb2:  u32, nb3:  u32,
    ) -> u32;

    /// DS4 kernel_cpy_f16_f32 (cpy.metal): f16 src → f32 dst sibling of
    /// `__tile_cpy_f32_f32_strided`. Same dims/strides.
    pub fn __tile_cpy_f16_f32_strided(
        src: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne0:  u32, ne1:  u32, ne2:  u32, ne3:  u32,
        nb0:  u32, nb1:  u32, nb2:  u32, nb3:  u32,
    ) -> u32;

    /// DS4 kernel_repeat_f32 (repeat.metal): broadcast/tile a smaller src
    /// tensor (ne00..ne03 dims) into a larger dst (ne0..ne3) by mod-indexing
    /// each axis. Same 16-uint stride/dim block as `__tile_cpy_*_strided`.
    pub fn __tile_repeat_f32_strided(
        src: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne0:  u32, ne1:  u32, ne2:  u32, ne3:  u32,
        nb0:  u32, nb1:  u32, nb2:  u32, nb3:  u32,
    ) -> u32;

    /// DS4 kernel_concat (concat.metal): float concat of two tensors along
    /// `dim` ∈ {0,1,2,3}. src0 dims/strides ne00..ne03+nb00..nb03, src1
    /// dims/strides ne10..ne13+nb10..nb13, dst dims/strides ne0..ne3+nb0..nb3.
    /// The `_strided` suffix preserves the per-axis stride convention.
    pub fn __tile_concat_f32_strided(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32, ne03: u32,
        nb00: u32, nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32, ne13: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0:  u32, ne1:  u32, ne2:  u32, ne3:  u32,
        nb0:  u32, nb1:  u32, nb2:  u32, nb3:  u32,
        dim:  u32,
    ) -> u32;

    /// DS4 kernel_get_rows_f32 (get_rows.metal): gather table rows by int32 ids.
    /// src0 = float table (ne00, ne01, ne02, ne03 with byte strides nb01..nb03),
    /// src1 = int32 ids (ne10 ids in dim0; byte strides nb10..nb12),
    /// dst = float (byte strides nb1..nb3). ne00t is the per-row thread-loop bound.
    pub fn __tile_get_rows_f32_strided(
        src0: u32, src1: u32, dst: u32,
        ne00t: u32, ne00: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32,
        nb10: u32, nb11: u32, nb12: u32,
        nb1: u32, nb2: u32, nb3: u32,
    ) -> u32;

    /// DS4 kernel_get_rows_f16 (get_rows.metal): half-table src → float dst
    /// sibling of `__tile_get_rows_f32_strided`. Same param block.
    pub fn __tile_get_rows_f16_strided(
        src0: u32, src1: u32, dst: u32,
        ne00t: u32, ne00: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32,
        nb10: u32, nb11: u32, nb12: u32,
        nb1: u32, nb2: u32, nb3: u32,
    ) -> u32;

    /// DS4 kernel_get_rows_i32 (get_rows.metal): int32 table → int32 dst
    /// sibling of `__tile_get_rows_f32_strided`. Same param block.
    pub fn __tile_get_rows_i32_strided(
        src0: u32, src1: u32, dst: u32,
        ne00t: u32, ne00: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32,
        nb10: u32, nb11: u32, nb12: u32,
        nb1: u32, nb2: u32, nb3: u32,
    ) -> u32;

    /// DS4 kernel_set_rows_f32_i32 (set_rows.metal): scatter inverse of
    /// get_rows. T=float, TI=int32_t. 13 uint params: nk0 (row width),
    /// ne01 (row-count bound), nb01..nb03 (src strides), ne11/ne12 (id mod
    /// bounds), nb10..nb12 (ids strides), nb1..nb3 (dst strides).
    pub fn __tile_set_rows_f32_i32_strided(
        src0: u32, src1: u32, dst: u32,
        nk0: u32, ne01: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne11: u32, ne12: u32,
        nb10: u32, nb11: u32, nb12: u32,
        nb1: u32, nb2: u32, nb3: u32,
    ) -> u32;

    /// DS4 kernel_dsv4_softplus_sqrt_f32_4: per-row float4 fused softplus → sqrt
    /// for decode router-logit transform. One tg per row, each thread covers one
    /// float4 lane. Buffers: src (const), dst (writable).
    pub fn __tile_dsv4_softplus_sqrt_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (SIGMOID op): per-row float4 lanewise sigmoid.
    /// Same dispatch as `__tile_dsv4_softplus_sqrt_f32_4`.
    pub fn __tile_sigmoid_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (RELU op): per-row float4 lanewise fmax(0, x).
    /// Same dispatch/params as `__tile_sigmoid_f32_4`.
    pub fn __tile_relu_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (TANH op): per-row float4 lanewise precise::tanh(x).
    /// Same dispatch/params as `__tile_sigmoid_f32_4`.
    pub fn __tile_tanh_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (GELU op): per-row float4 lanewise tanh-approx GELU,
    /// `0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))`.
    /// Same dispatch/params as `__tile_sigmoid_f32_4`.
    pub fn __tile_gelu_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (SQR op): per-row float4 lanewise x*x.
    pub fn __tile_sqr_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (NEG op): per-row float4 lanewise -x.
    pub fn __tile_neg_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (ABS op): per-row float4 lanewise fabs(x).
    pub fn __tile_abs_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (STEP op): per-row float4 lanewise (x>0)?1:0.
    pub fn __tile_step_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (EXP op): per-row float4 lanewise exp(x).
    pub fn __tile_exp_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (LOG op): per-row float4 lanewise log(x).
    pub fn __tile_log_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (SILU op): per-row float4 lanewise x/(1+exp(-x)).
    pub fn __tile_silu_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (HARDSIGMOID op): per-row float4 lanewise
    /// `fmax(0, fmin(1, x/6 + 0.5))`.
    pub fn __tile_hardsigmoid_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (HARDSWISH op): per-row float4 lanewise
    /// `x * fmax(0, fmin(1, x/6 + 0.5))`.
    pub fn __tile_hardswish_f32_4(
        src: u32, dst: u32,
        ne0_4: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (SIGMOID op): per-row scalar half lanewise
    /// `1 / (1 + exp(-x))`. Computation in float, store as half.
    /// Dispatch: tid + row + tcount (default), `ne0` = scalar half cols per row.
    pub fn __tile_sigmoid_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (RELU op): scalar half `max(x, 0)`.
    pub fn __tile_relu_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (TANH op): scalar half `precise::tanh(x)`.
    pub fn __tile_tanh_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (GELU op): scalar half tanh-approx GELU.
    pub fn __tile_gelu_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (SILU op): scalar half `x / (1 + exp(-x))`.
    pub fn __tile_silu_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (HARDSIGMOID op): scalar half `fmax(0, fmin(1, x/6 + 0.5))`.
    pub fn __tile_hardsigmoid_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (HARDSWISH op): scalar half `x * hardsigmoid(x)`.
    pub fn __tile_hardswish_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (SQR op): scalar half `x * x`.
    pub fn __tile_sqr_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (NEG op): scalar half `-x`.
    pub fn __tile_neg_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (ABS op): scalar half `fabs(x)`.
    pub fn __tile_abs_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (STEP op): scalar half `x > 0 ? 1 : 0`.
    pub fn __tile_step_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (EXP op): scalar half `exp(x)`.
    pub fn __tile_exp_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 (LOG op): scalar half `log(x)`.
    pub fn __tile_log_f16(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_swiglu_f32 (glu.metal): per-row SwiGLU activation
    /// `dst[i] = silu(src0[i]) * src1[i]` over `ne0` columns. Each row's
    /// src0/src1/dst pointers stride by `nb01`/`nb11`/`nb1` bytes; src0/src1
    /// start at `i00`/`i10` element offset within their row.
    pub fn __tile_swiglu_f32(
        src0: u32, src1: u32, dst: u32,
        ne0: u32, nb01: u32, nb11: u32, nb1: u32,
        i00: u32, i10: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32 (sigmoid op): per-row scalar float lanewise
    /// sigmoid. Buffers: src (const), dst (writable). Params: ne0 (scalar
    /// float cols per row), nb_src/nb_dst (row strides in bytes).
    pub fn __tile_sigmoid_f32_scalar(
        src: u32, dst: u32,
        ne0: u32, nb_src: u32, nb_dst: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32 (relu op): per-row scalar float fmax(0, x).
    pub fn __tile_relu_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (tanh op): per-row scalar float precise::tanh(x).
    pub fn __tile_tanh_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (gelu op): per-row scalar float tanh-approx GELU.
    pub fn __tile_gelu_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (silu op): per-row scalar float x/(1+exp(-x)).
    pub fn __tile_silu_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (hardsigmoid op): per-row scalar float fmax(0,fmin(1,x/6+0.5)).
    pub fn __tile_hardsigmoid_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (hardswish op): per-row scalar float x*hardsigmoid(x).
    pub fn __tile_hardswish_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (sqr op): per-row scalar float x*x.
    pub fn __tile_sqr_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (neg op): per-row scalar float -x.
    pub fn __tile_neg_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (abs op): per-row scalar float fabs(x).
    pub fn __tile_abs_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (step op): per-row scalar float x>0?1:0.
    pub fn __tile_step_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (exp op): per-row scalar float exp(x).
    pub fn __tile_exp_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (log op): per-row scalar float log(x).
    pub fn __tile_log_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;

    /// DS4 kernel_mul_mv_f32_f32_short: scalar dense matvec fallback for short
    /// rows. One simdgroup per tg, each lane = one output element. Buffers:
    /// src0 (const), src1 (const), dst (writable).
    pub fn __tile_mul_mv_f32_f32_short(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_f16_f32_short: half-precision src0 variant of
    /// `mul_mv_f32_f32_short`. Same dispatch and ABI; src0 is read as half
    /// and cast to float in the inner product. src1 stays float.
    pub fn __tile_mul_mv_f16_f32_short(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_t_t<float,float> (M88a setup): scaffolding-only variant
    /// of the full dense matvec. Same buffer/param ABI as `mul_mv_f32_f32_short`
    /// but dispatched with 4 simdgroups × 32 lanes per tg, with threadgroup
    /// shmem and dual simd attrs. M88a body zero-initializes the output row;
    /// M88b will replace it with the K-accumulator and helper_mv_reduce_and_write.
    pub fn __tile_mul_mv_f32_f32_setup(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_t_t<float,float> (M88b acc): extends M88a setup with
    /// the per-lane K-accumulator over `ne00`. NSG×NW=128 lanes stride-load
    /// src0/src1 and accumulate into a per-row partial sum, fanned through
    /// threadgroup shmem with a lane-0 finalize. Same ABI as M88a.
    pub fn __tile_mul_mv_f32_f32_acc(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_t_t<float,float> (M88c reduce): same K-accumulator as
    /// M88b but the finalize block uses the antirez `helper_mv_reduce_and_write`
    /// pattern — per-row simd_sum + sgitg-slot shmem + cross-simd simd_sum.
    /// Same ABI as M88a/M88b.
    pub fn __tile_mul_mv_f32_f32_reduce(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_t_t<half,float> M88d: half src0 sibling of
    /// __tile_mul_mv_f32_f32_reduce; same body with src0 read as half
    /// then cast to float in the K-accumulator.
    pub fn __tile_mul_mv_f16_f32_reduce(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_t_t_4<float,float4,float,float4> M89a: vectorized
    /// matvec with float4 loads + helper_mv_reduce_and_write finalize.
    pub fn __tile_mul_mv_f32_f32_4_reduce(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_t_t_4<half,half4,float,float4> M89b: half src0
    /// sibling of __tile_mul_mv_f32_f32_4_reduce.
    pub fn __tile_mul_mv_f16_f32_4_reduce(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_f16_f32_pair_4 M90: paired half-src0 vectorized
    /// matvec sharing one float src1 → two float dsts. Saves one y load
    /// vs two separate kernel_mul_mv_f16_f32_4 dispatches.
    pub fn __tile_mul_mv_f16_f32_pair_4(
        src0_a: u32, src0_b: u32, src1: u32, dst_a: u32, dst_b: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_q8_0_f32 M91: quantized Q8_0 × float matvec.
    /// src0 is laid out as block_q8_0 { half d; int8_t qs[32]; } (34 B per
    /// block of 32 elements). ne00 must be a multiple of QK8_0=32. Output is
    /// float. Baked NSG=4, NR0=2, NQ=8. Uses helper_mv_reduce_and_write.
    pub fn __tile_mul_mv_q8_0_f32(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32,
        ne12: u32, r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q8_0_f32 M92: MoE-routed Q8_0 matvec.
    /// Per (idx, iid1) the routed expert i02 is read from ids; src0s is
    /// offset by i02*nb02 and the M91 inner q8_0 dot runs. Grid is
    /// (tx, 1, idx + iid1 * nei0). Buffers: src0s, src1, ids, dst
    /// (dst is last so the writable-buf heuristic marks it correctly).
    pub fn __tile_mul_mv_id_q8_0_f32(
        src0s: u32, src1: u32, ids: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q2_K_f32 M111 (moe.metal:830): q2_K sibling of M92.
    /// Same 4-buf shell (src0s, src1, ids, dst) + 11-uint params; inner runs
    /// kernel_mul_mv_q2_K_f32_impl with NR0=N_R0_Q2_K=4, QK_K=256, block_q2_K=84B
    /// (uchar scales[16] + uchar qs[64] + half d + half dmin).
    pub fn __tile_mul_mv_id_q2_K_f32(
        src0s: u32, src1: u32, ids: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q4_K_f32 M112 (moe.metal:831): q4_K sibling of M111.
    /// Same 4-buf shell + 11-uint params; inner runs kernel_mul_mv_q4_K_f32_impl
    /// with NR0=N_R0_Q4_K=2, QK_K=256, block_q4_K=144B (half d + half dmin +
    /// uchar scales[12] + uchar qs[128]).
    pub fn __tile_mul_mv_id_q4_K_f32(
        src0s: u32, src1: u32, ids: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_iq2_xxs_f32 M113 (moe.metal:832): iq2_xxs sibling of
    /// M111/M112. Same 4-buf shell + 11-uint params; inner runs
    /// kernel_mul_mv_iq2_xxs_f32_impl (moe.metal:521-614) with
    /// NR0=N_R0_IQ2_XXS=4, block_iq2_xxs=66 B (half d + ushort qs[32]).
    /// Stages file-scope iq2xxs_grid[256] (ulong) + ksigns_iq2xs[128]
    /// into threadgroup shmem; per ib32 decodes 4 q2 ushorts → aux32 →
    /// grid lookup + ksigns + kmask_iq2xs sign bits; output *= 0.25f.
    pub fn __tile_mul_mv_id_iq2_xxs_f32(
        src0s: u32, src1: u32, ids: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_iq2_xxs_pair_f32 M128 (moe.metal:897): paired
    /// gate+up iq2_xxs MoE-routed matvec for fused gate+up. 6 char* bufs
    /// (src0_gate, src0_up, src1, dst_gate, dst_up, ids); same 11-uint
    /// params as M113. Shares y load + iq2xxs_grid/ksigns shmem tables
    /// across paired src0 streams.
    pub fn __tile_mul_mv_id_iq2_xxs_pair_f32(
        src0_gate: u32, src0_up: u32, src1: u32, dst_gate: u32, dst_up: u32, ids: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 M129 (moe.metal:959):
    /// SwiGLU-fused tri-output sibling of M128. 8 char* bufs (src0_gate,
    /// src0_up, src1, dst_gate, dst_up, dst_mid, ids, weights); adds 3 act
    /// uniforms (mid_row_stride, weight_stride, clamp_value). Final write
    /// produces gate, up, AND mid = silu(clamp(gate,c)) * clamp(up,-c,c) *
    /// route_weight where route_weight = weights[idx*weight_stride/4][0].
    pub fn __tile_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        src0_gate: u32, src0_up: u32, src1: u32, dst_gate: u32, dst_up: u32, dst_mid: u32, ids: u32, weights: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
        mid_row_stride: u32, weight_stride: u32, clamp_value: f32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q4_K_pair_f32 M130 (moe.metal:1106): paired
    /// gate+up q4_K MoE-routed matvec for fused gate+up. 6 char* bufs
    /// (src0_gate, src0_up, src1, dst_gate, dst_up, ids); same 11-uint
    /// params as M112/M128. Shares yl/yh/sumy loads across paired src0
    /// streams; no shmem table required (q4_K decode is self-contained).
    pub fn __tile_mul_mv_id_q4_K_pair_f32(
        src0_gate: u32, src0_up: u32, src1: u32, dst_gate: u32, dst_up: u32, ids: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q4_K_pair_swiglu_f32 M131 (moe.metal:1160):
    /// SwiGLU-fused tri-output sibling of M130. 8 char* bufs (src0_gate,
    /// src0_up, src1, dst_gate, dst_up, dst_mid, ids, weights); adds 3 act
    /// uniforms (mid_row_stride, weight_stride, clamp_value). Final write
    /// produces gate, up, AND mid = silu(clamp(gate,c)) * clamp(up,-c,c) *
    /// route_weight where route_weight = weights[idx*weight_stride/4][0].
    pub fn __tile_mul_mv_id_q4_K_pair_swiglu_f32(
        src0_gate: u32, src0_up: u32, src1: u32, dst_gate: u32, dst_up: u32, dst_mid: u32, ids: u32, weights: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
        mid_row_stride: u32, weight_stride: u32, clamp_value: f32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q2_K_sum6_f32 M132 (moe.metal:1245). q2_K MoE matvec
    /// summing 6 fixed-slot experts per token into one dst row. 4 char* bufs
    /// (src0s, src1, dst, ids); idx 2 (dst) is the only writable slot.
    /// 8 uniforms (ne00, ne0, nbi1, nb01, nb02, nb11, nb12, nb1). Outer loop
    /// over expert_slot ∈ 0..6 reads token_ids[expert_slot]; inner = M111 q2_K
    /// decode (NSG=2, NR0=N_R0_Q2_K=4, QK_K=256, block_q2_K=84B). simd_sum
    /// across lanes; tiisg==0 writes dst.
    pub fn __tile_mul_mv_id_q2_K_sum6_f32(
        src0s: u32, src1: u32, dst: u32, ids: u32,
        ne00: u32, ne0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32, nb1: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q4_K_sum6_f32 M133 (moe.metal:1336). q4_K sibling of
    /// M132 (q2_K sum6). Same 4-buf + 8-uniform shell + sum6 outer routing.
    /// Inner q4_K decode: NR0=N_R0_Q4_K=2, block_q4_K=144B (half d/dmin + scales[12] + qs[128]).
    pub fn __tile_mul_mv_id_q4_K_sum6_f32(
        src0s: u32, src1: u32, dst: u32, ids: u32,
        ne00: u32, ne0: u32, nbi1: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32, nb1: u32,
    ) -> u32;

    /// DS4 kernel_dsv4_attn_out_low_q8_0_f32 M93: stripped-down M92 with id=group.
    /// `i02 = idx` (no ids buffer). Buffers: src0s, src1, dst (dst is last so
    /// the writable-buf heuristic marks it correctly). Grid is
    /// (tx, 1, idx + iid1 * nei0). M92 minus the ids buffer + nbi1 param.
    pub fn __tile_dsv4_attn_out_low_q8_0_f32(
        src0s: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne11: u32,
        nei0: u32,
        nb01: u32, nb02: u32, nb11: u32, nb12: u32,
    ) -> u32;

    /// DS4 kernel_dsv4_shared_gate_up_swiglu_q8_0 M114: fused shared-expert
    /// q8_0 matvec from dense.metal:203. Two parallel gate + up streams share
    /// one y load; output triple (gate, up, mid=silu(gate)*up). Buffers:
    /// src0_gate, src0_up, src1 (const); dst_gate, dst_up, dst_mid (writable).
    /// 13 uints. NSG=2, NQ=8, NR0=N_R0_Q8_0=2.
    pub fn __tile_dsv4_shared_gate_up_swiglu_q8_0(
        src0_gate: u32, src0_up: u32, src1: u32,
        dst_gate: u32, dst_up: u32, dst_mid: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne12: u32,
        r2: u32, r3: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_f32_f32 M115: host-callable matvec dispatch wrapper
    /// from dense.metal:429 (instantiates kernel_mul_mv_t_t with float src0).
    /// Branches at runtime on `nr0 ∈ {2, 4}` over the same impl shell.
    /// Buffers: src0 (float), src1 (float), dst (float). 14 uints (nr0 at +7).
    /// NSG=4, NW=32.
    pub fn __tile_mul_mv_f32_f32(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne12: u32,
        r2: u32, r3: u32, nr0: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_f16_f32 M115: half-src0 sibling of mul_mv_f32_f32
    /// (dense.metal:430). Same dispatch shell with `device const half * x`.
    pub fn __tile_mul_mv_f16_f32(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne12: u32,
        r2: u32, r3: u32, nr0: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_f32_f32_4 M116: float4-vectorized host-callable
    /// dispatch wrapper from dense.metal:547. Same shape as M115 with
    /// float4/half4 inner loop + scalar tail. Branches on `nr0 ∈ {2, 4}`.
    pub fn __tile_mul_mv_f32_f32_4(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne12: u32,
        r2: u32, r3: u32, nr0: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_f16_f32_4 M116 (dense.metal:548): half-src0 sibling
    /// of mul_mv_f32_f32_4. Loads half4 weights, casts to float4 for dot.
    pub fn __tile_mul_mv_f16_f32_4(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne0: u32, ne1: u32, ne12: u32,
        r2: u32, r3: u32, nr0: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb11: u32, nb12: u32, nb13: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32 M117 (softmax.metal:240): host-callable
    /// unified row-softmax wrapper. 4 bufs (src0, src1=mask-or-self,
    /// src2=sink-or-self, dst); runtime branches on has_mask/has_sink/
    /// max_bias (ALiBi). ne00 = cols, ne01..ne13 = source/mask dims,
    /// nb01..nb13 = strides, nb1..nb3 = dst strides, scale/max_bias/
    /// m0/m1/n_head_log2 = softmax args.
    pub fn __tile_soft_max_f32(
        src0: u32, src1: u32, src2: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne11: u32, ne12: u32, ne13: u32,
        nb11: u32, nb12: u32, nb13: u32,
        nb1: u32, nb2: u32, nb3: u32,
        scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: u32,
        has_mask: u32, has_sink: u32,
    ) -> u32;

    /// DS4 kernel_soft_max_f32_4 M117 (softmax.metal:241): float4-
    /// vectorized sibling of soft_max_f32. ne00 must be a multiple of 4;
    /// mask is one float per i00 in [0, ne00/4), broadcast to all 4 lanes.
    pub fn __tile_soft_max_f32_4(
        src0: u32, src1: u32, src2: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne11: u32, ne12: u32, ne13: u32,
        nb11: u32, nb12: u32, nb13: u32,
        nb1: u32, nb2: u32, nb3: u32,
        scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: u32,
        has_mask: u32, has_sink: u32,
    ) -> u32;

    /// DS4 kernel_bin_fuse_f32_f32_f32 M118 (bin.metal:192): single host-callable
    /// wrapper for elementwise binary ops add/sub/mul/div on float src0/src1/dst.
    /// Scope: slow-path FC_RB=false + FC_F=1; runtime op (0=add,1=sub,2=mul,3=div)
    /// and cb_flag (column-broadcast modulo: i10 = cb ? i0%ne10 : i0).
    pub fn __tile_bin_fuse_f32(
        src0: u32, src1: u32, dst: u32,
        ne01: u32, ne02: u32, ne03: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32, ne13: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, nb1: u32, nb2: u32, nb3: u32,
        op: u32, cb_flag: u32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32 M119: host-callable elementwise unary op
    /// wrapper. Single dispatch covers ~26 unary ops via runtime `op` code
    /// (OP_UNARY_NUM_*: SCALE/FILL/CLAMP/SQR/SQRT/SIN/COS/LOG/LEAKY_RELU @10-18,
    /// TANH/RELU/SIGMOID/GELU/GELU_ERF/GELU_QUICK/SILU/ELU/NEG/ABS/SGN/STEP/
    /// HARDSWISH/HARDSIGMOID/EXP/SOFTPLUS/EXPM1/FLOOR/CEIL/ROUND/TRUNC/XIELU
    /// @100-121) and `cnt_flag` (1=contiguous fast path, 0=3D-grid strided).
    pub fn __tile_unary_f32(
        src0: u32, dst: u32,
        ne01: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne0: u32, nb1: u32, nb2: u32, nb3: u32,
        op: u32, cnt_flag: u32,
        slope: f32, scale: f32, bias: f32, val: f32,
        umin: f32, umax: f32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 M120: vec4 sibling of M119
    /// (unary.metal:311). T0=T=TC=float4. Same uniform layout as M119;
    /// driver dispatches over ne0/4 vec4 lanes per row.
    pub fn __tile_unary_f32_4(
        src0: u32, dst: u32,
        ne01: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne0: u32, nb1: u32, nb2: u32, nb3: u32,
        op: u32, cnt_flag: u32,
        slope: f32, scale: f32, bias: f32, val: f32,
        umin: f32, umax: f32,
    ) -> u32;

    /// DS4 kernel_unary_f16_f16 M121: half scalar sibling of M119
    /// (unary.metal:312). T0=T=half, TC=float (compute up-cast).
    pub fn __tile_unary_f16(
        src0: u32, dst: u32,
        ne01: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne0: u32, nb1: u32, nb2: u32, nb3: u32,
        op: u32, cnt_flag: u32,
        slope: f32, scale: f32, bias: f32, val: f32,
        umin: f32, umax: f32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_f16_f32_r1_2 M94: small-batch (r1ptg=2) matvec
    /// with half src0 and float src1. Baked NSG=2, NXPSG=8, CHPT=4 (chpb=1
    /// since f16 epb=4). Buffers: src0 (half), src1 (float), dst (float).
    /// Grid: (ceil(ne01/(NYPSG*NSG)), ceil(ne11/2), ne12*ne13);
    /// threads per group: 32 (one simdgroup).
    pub fn __tile_mul_mv_ext_f16_f32_r1_2(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_f16_f32_r1_3 M95: R1PTG=3 sibling of M94.
    pub fn __tile_mul_mv_ext_f16_f32_r1_3(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_f16_f32_r1_4 M96: R1PTG=4 sibling of M94.
    pub fn __tile_mul_mv_ext_f16_f32_r1_4(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_f16_f32_r1_5 M97: R1PTG=5 sibling of M94.
    pub fn __tile_mul_mv_ext_f16_f32_r1_5(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_q8_0_f32_r1_2 M98: small-batch q8_0 matvec, R1PTG=2;
    /// src0 is block_q8_0 weights (34-byte blocks { half d; int8_t qs[32]; }).
    pub fn __tile_mul_mv_ext_q8_0_f32_r1_2(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_q8_0_f32_r1_3 M99: R1PTG=3 sibling of M98.
    pub fn __tile_mul_mv_ext_q8_0_f32_r1_3(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_q8_0_f32_r1_4 M100: R1PTG=4 sibling of M98.
    pub fn __tile_mul_mv_ext_q8_0_f32_r1_4(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_ext_q8_0_f32_r1_5 M101: R1PTG=5 sibling of M98.
    pub fn __tile_mul_mv_ext_q8_0_f32_r1_5(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne01: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne10: u32, ne11: u32, ne12: u32,
        nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_f16_f32 M102a: tiled prompt-path GEMM scaffolding.
    /// Buffers: src0 (half [ne00 x ne01 x ne02 x ne03]),
    ///          src1 (half [ne00 x ne1  x ne12 x ne13]),
    ///          dst  (float[ne0  x ne1  x ne12 x ne13]).
    /// Params: 14 uints (ne00, ne02, nb01, nb02, nb03, ne12,
    ///                   nb10, nb11, nb12, nb13, ne0, ne1, r2, r3).
    /// M102a body zero-fills dst tile only; M102b adds K-loop + simdgroup matmul.
    pub fn __tile_mul_mm_f16_f32(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne12: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_q8_0_f32 M104a: id-routed mul_mm_id from moe.metal:1519.
    /// Buffers: src0 (block_q8_0 expert-stacked [ne00/32 × ne01 × ne02]),
    ///          src1 (half [ne00 × ne11 × ne12]),
    ///          htpe (uint32 [ne02] tokens-per-expert),
    ///          hids (int32 [ne02 × ne21] routing table; ids[im*ne21 + j] = idt*ne20 + ide),
    ///          dst  (float[ne0 × ne1 × ne12]).
    /// Params: 17 uints (ne00, ne02, nb01, nb02, nb03, nb10, nb11, nb12, nb13, ne0, ne1, ne11, ne12, ne20, ne21, r2, r3).
    /// M104a body: zero-fill routed dst region; full id-routed K-loop lands in M104b.
    pub fn __tile_mul_mm_id_q8_0_f32(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_q8_0_f16 M105: f16-dst sibling of M104b (moe.metal:1725).
    /// Same buffer layout and 17-uint params as M104b; only dst type differs (half instead of float).
    pub fn __tile_mul_mm_id_q8_0_f16(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_q2_K_f32 M106 (moe.metal:1722): q2_K sibling of M104b.
    /// Same 5 ptrs + 17 uints as M104b/M105; src0 is block_q2_K (84 B: uchar scales[16] + uchar qs[64] + half d + half dmin),
    /// nl=QK_NL=16 (one block covers 256 elements = 8 K-tiles of 32 each).
    pub fn __tile_mul_mm_id_q2_K_f32(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_q2_K_f16 M107 (moe.metal:1726): f16-dst sibling of M106.
    /// Identical buffer layout and params to M106; routed write casts the simdgroup_float8x8
    /// accumulator output to half on store.
    pub fn __tile_mul_mm_id_q2_K_f16(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_q4_K_f32 M108 (moe.metal:1723): q4_K sibling of M104b.
    /// Same 5 ptrs + 17 uints as M104b/M105/M106/M107; src0 is block_q4_K (144 B: half d + half dmin + uchar scales[12] + uchar qs[128]),
    /// nl=QK_NL=16 (one block covers 256 elements = 8 K-tiles of 32 each).
    /// Inlined dequantize_q4_K (with get_scale_min_k4_just2 helper inlined) replaces dequantize_q8_0/q2_K in the K-loop staging.
    pub fn __tile_mul_mm_id_q4_K_f32(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_q4_K_f16 M109 (moe.metal:1727): f16-dst sibling of M108.
    /// Identical buffer layout and params to M108; routed write casts the simdgroup_float8x8
    /// accumulator output to half on store.
    pub fn __tile_mul_mm_id_q4_K_f16(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_iq2_xxs_f32 M110 (moe.metal:1724): iq2_xxs sibling of M104b/M106/M108.
    /// Same 5 ptrs + 17 uints; src0 is block_iq2_xxs (66 B: half d + ushort qs[32]),
    /// nl=QK_NL=16 (one block covers 256 elements). Inlined dequantize_iq2_xxs reads from
    /// file-scope iq2xxs_grid[256] + ksigns_iq2xs[128] + kmask_iq2xs[8] tables.
    pub fn __tile_mul_mm_id_iq2_xxs_f32(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_id_iq2_xxs_f16 M110 (moe.metal:1728): f16-dst sibling of iq2_xxs_f32.
    pub fn __tile_mul_mm_id_iq2_xxs_f16(
        src0: u32, src1: u32, htpe: u32, hids: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32, ne11: u32, ne12: u32,
        ne20: u32, ne21: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 kernel_mul_mm_q8_0_f32 M103: tiled prompt-path GEMM with q8_0 src0.
    /// Buffers: src0 (block_q8_0 [ne00/32 x ne01 x ne02 x ne03] — 34 B blocks: half d at +0, int8 qs[32] at +2),
    ///          src1 (half [ne00 x ne1 x ne12 x ne13]),
    ///          dst  (float[ne0 x ne1 x ne12 x ne13]).
    /// Same 14-uint param shape as __tile_mul_mm_f16_f32.
    pub fn __tile_mul_mm_q8_0_f32(
        src0: u32, src1: u32, dst: u32,
        ne00: u32, ne02: u32,
        nb01: u32, nb02: u32, nb03: u32,
        ne12: u32,
        nb10: u32, nb11: u32, nb12: u32, nb13: u32,
        ne0: u32, ne1: u32,
        r2: u32, r3: u32,
    ) -> u32;

    /// DS4 dsv4_indexed_mixed_attention_heads8_rb4: decode specialization of the h8
    /// kernel that stages 4 raw/comp KV rows at once and consumes them sequentially.
    pub fn __tile_indexed_mixed_attention_h8_rb4_f32(
        q: u32, raw_kv: u32, comp_kv: u32, topk: u32, sinks: u32, dst: u32,
        n_tokens: u32, n_head: u32, n_raw: u32, n_comp: u32, top_k: u32, ratio: u32,
        window: u32, pos0: u32, raw_start: u32, raw_cap: u32,
        q_token_stride: u32, q_head_stride: u32, raw_row_stride: u32,
        comp_row_stride: u32, topk_token_stride: u32,
        dst_token_stride: u32, dst_head_stride: u32,
        scale: u32,
    ) -> u32;

    /// DS4 dsv4_indexer_scores_tiled (bf16/half K variant of the f32 tiled scorer).
    /// Same signature; K is staged through threadgroup half buffers and matmul uses
    /// simdgroup_half8x8 for mq/mk while the accumulator stays simdgroup_float8x8.
    pub fn __tile_indexer_scores_tiled_bf16(
        q: u32, weights: u32, index_comp: u32, scores: u32,
        n_comp: u32, n_tokens: u32, n_head: u32, pos0: u32, ratio: u32,
        q_token_stride: u32, q_head_stride: u32, weights_token_stride: u32,
        index_row_stride: u32, score_token_stride: u32, scale: u32,
    ) -> u32;

    /// Causal mask: fills upper-triangular portion of (S, S) tile with -inf.
    pub fn __tile_causal_mask_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Embedding lookup: gathers `count` rows from (V, D) weight table by index.
    pub fn __tile_embedding_f32(
        dst: u32, weight: u32, indices: *const u32,
        vocab_size: u32, embed_dim: u32, count: u32,
    ) -> u32;

    /// Cross-entropy loss: -log(softmax(logits)[target]) per row.
    pub fn __tile_cross_entropy_f32(
        dst: u32, logits: u32, targets: *const u32,
        rows: u32, cols: u32,
    ) -> u32;

    // ── Phase 0: foundational primitives for DeepSeek/LLM serving ───

    /// Transpose (ROWS × COLS) → (COLS × ROWS).
    pub fn __tile_transpose_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise 1/sqrt(x).
    pub fn __tile_rsqrt_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise natural logarithm.
    pub fn __tile_log_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise sigmoid: 1/(1+exp(-x)).
    pub fn __tile_sigmoid_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Cast f32 tile to f16 tile.
    pub fn __tile_cast_f32_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Cast f16 tile to f32 tile.
    pub fn __tile_cast_f16_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Extract sub-tile: (R2 × C2) from (R × C) at offset (row_off, col_off).
    pub fn __tile_slice_f32(
        dst: u32, src: u32,
        row_off: u32, col_off: u32,
        src_rows: u32, src_cols: u32,
        dst_rows: u32, dst_cols: u32,
    ) -> u32;

    /// Concatenate along columns: (R × C1) ++ (R × C2) → (R × C1+C2).
    pub fn __tile_concat_f32(
        dst: u32, a: u32, b: u32,
        rows: u32, cols_a: u32, cols_b: u32,
    ) -> u32;

    /// Indexed scatter: scatter src (N,D) rows into dst (M,D) at positions.
    pub fn __tile_scatter_f32(
        dst: u32, src: u32, indices: *const u32,
        n: u32, m: u32, d: u32,
    ) -> u32;

    /// Indexed gather: gather from src (M,D) at positions → out (N,D).
    pub fn __tile_gather_f32(
        dst: u32, src: u32, indices: *const u32,
        n: u32, m: u32, d: u32,
    ) -> u32;

    /// Top-K selection per row. Returns (R, K) values, writes indices to indices_out.
    pub fn __tile_topk_f32(
        dst: u32, src: u32, indices_out: *mut u32,
        rows: u32, cols: u32, k: u32,
    ) -> u32;

    /// Iota / arithmetic progression: emits `dst[i] = start + i` for `i in 0..valid_col`.
    /// Output is a 1×valid_col i32 tile. Used as the sort-index initializer
    /// for the topk_selector port (composes with `__tile_sort32_f32`).
    /// Lowers to `pto.tci` on Ascend.
    pub fn __tile_arith_progression_i32(
        dst: u32, start: u32, valid_col: u32,
    ) -> u32;

    /// Sort-buffer init: re-pad tile to next BLOCK boundary with sentinel.
    /// Output tile has same logical shape as src but is sentinel-padded
    /// for safe consumption by `__tile_sort32_f32`.
    /// Lowers to `pto.tfillpad` on Ascend.
    pub fn __tile_init_sort_buf_f32(
        dst: u32, src: u32, rows: u32, cols: u32,
    ) -> u32;

    /// 32-way bitonic sort over a 1×N f32 tile with paired ui32 indices.
    /// Output is a 1×(2*N) f32 tile of interleaved [value, idx] pairs
    /// (FLOAT_DST_STRIDE_COEF=2). Cols must be a positive multiple of 64.
    /// Lowers to `pto.tsort32` on Ascend.
    pub fn __tile_sort32_f32(
        dst: u32, values: u32, indices: u32, rows: u32, cols: u32,
    ) -> u32;

    /// 2-way bitonic merge: merges two 1×N sorted f32 tiles into a
    /// 1×(2N) sorted tile. `tmp` is a 1×(2N) scratch tile.
    /// Lowers to `pto.tmrgsort` (2-way form) on Ascend.
    pub fn __tile_mrgsort2_f32(
        dst: u32, src0: u32, src1: u32, tmp: u32, cols_each: u32,
    ) -> u32;

    /// Mask-pattern gather (lane select). 4-bit mask:
    ///   `0b1010 = 10` → P1010 (selects even lanes — value channel of
    ///   sort_result interleaved [val,idx] pairs).
    /// Lowers to `pto.tgather` (mask-pattern form) on Ascend.
    pub fn __tile_gather_mask_f32(
        dst: u32, src: u32, mask_pattern: u32, rows: u32, cols: u32,
    ) -> u32;

    /// f16 matrix multiply: C = A(M×K) @ B(K×N) → C(M×N).
    pub fn __tile_matmul_f16(
        dst: u32, a: u32, b: u32,
        m: u32, k: u32, n: u32,
    ) -> u32;

    /// int8 matmul with i32 accumulator and per-column f32 dequant → f16 GM.
    ///
    /// C_f16 = dequant(A(M×K, i8) @ B(K×N, i8) as i32, scale_per_col f32).
    ///
    /// PTO lowering (validated 2026-04-16, see
    /// memory/project_pto_i8_tmatmul_validated.md):
    ///   - L0A/L0B as i8 tiles, L0C as i32 acc
    ///   - `pto.tmatmul` / `pto.tmatmul.acc` dtype triple (i32, i8, i8)
    ///   - Scale tile allocated in `loc=scaling` (__fbuf__) f32
    ///   - `pto.tstore_fp ins(%acc, %scale) outs(%gm_f16)` — FixPipe folds
    ///     the per-column dequant into the L0C→GM DMA.
    ///
    /// Host must pre-quantize B offline to i8 with per-column f32 scales
    /// (absmax or percentile). A is per-token-quantized at runtime (cheap
    /// for M=1 decode).
    pub fn __tile_matmul_i8_acc_i32_dequant_f16(
        dst: u32, a: u32, b: u32, scale: *const f32,
        m: u32, k: u32, n: u32,
    ) -> u32;

    /// Load a `ROWS × COLS` i8 tile from global memory.
    ///
    /// PTO lowering: tload into a `loc=mat, dtype=i8` tile → tmov into `loc=left`
    /// (A) or `loc=right` (B) at matmul time.
    pub fn __tile_load_i8(gm: *const i8, rows: u32, cols: u32) -> u32;

    /// Store an i8 tile buffer to global memory.
    pub fn __tile_store_i8(gm: *mut i8, buf: u32, rows: u32, cols: u32);

    /// Matmul with transposed RHS: C[M,N] = A[M,K] @ B[N,K]^T, f32.
    /// B is stored in natural weight layout (out=N, in=K); the kernel
    /// transposes-on-the-fly. PTO path (mlir_to_pto) routes this to
    /// `pto.tmatmul` with a DN→ZN tload on a transposed tensor_view —
    /// avoiding the a5-only VEC→MAT tmov path.
    pub fn __tile_matmul_transposed_f32(
        dst: u32, a: u32, b: u32,
        m: u32, k: u32, n: u32,
    ) -> u32;

    /// f16 variant of matmul_transposed.
    pub fn __tile_matmul_transposed_f16(
        dst: u32, a: u32, b: u32,
        m: u32, k: u32, n: u32,
    ) -> u32;

    // ── Phase 3: Flash Attention primitives ──────────────────────────

    /// Fill tile with scalar value.
    pub fn __tile_fill_f32(dst: u32, scalar: f32, rows: u32, cols: u32) -> u32;

    /// Element-wise max of two tiles.
    pub fn __tile_max_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise min of two tiles.
    pub fn __tile_min_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// Hyperbolic tangent.
    pub fn __tile_tanh_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise absolute value.
    pub fn __tile_abs_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise square root.
    pub fn __tile_sqrt_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise ReLU: max(x, 0).
    pub fn __tile_relu_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise Softplus: log(1 + exp(x)).
    pub fn __tile_softplus_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    // __tile_sub_f32, __tile_reduce_max_f32, __tile_div_f32
    // are declared earlier in this same extern block.

    /// RMS normalization: x * rsqrt(mean(x^2) + eps).
    pub fn __tile_rms_norm_f32(dst: u32, src: u32, eps: f32, rows: u32, cols: u32) -> u32;

    // ── Phase 4: Quantization primitives ────────────────────────────

    /// Symmetric INT8 quantize: round(x / scale), clamped to [-128, 127].
    pub fn __tile_quantize_f32_i8(dst: u32, src: u32, scale: f32, rows: u32, cols: u32) -> u32;

    /// INT8 dequantize: x_i8 * scale → f32.
    pub fn __tile_dequantize_i8_f32(dst: u32, src: u32, scale: f32, rows: u32, cols: u32) -> u32;

    /// Row-wise absmax: max(|x|) per row → (R, 1).
    pub fn __tile_absmax_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    // ── Phase 6: Multi-token prediction / speculative decoding ─────

    /// Row-wise argmax: returns index of maximum value per row → (R, 1) u32 indices.
    pub fn __tile_argmax_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Top-p (nucleus) sampling: temperature-scaled softmax → cumsum → threshold at p.
    /// Returns sampled token indices per row → (R, 1) u32.
    /// `rng_seed` seeds the per-row uniform random draw.
    pub fn __tile_sample_top_p_f32(dst: u32, logits: u32, temperature: f32, top_p: f32, rng_seed: u32, rows: u32, cols: u32) -> u32;

    /// Draft verification: compare draft token indices against target logits.
    /// Returns per-position acceptance probabilities → (R, 1) f32.
    /// `draft_tokens`: (R, 1) u32 indices from draft model.
    /// `target_logits`: (R, COLS) f32 logits from target model.
    pub fn __tile_draft_verify_f32(dst: u32, draft_tokens: u32, target_logits: u32, rows: u32, cols: u32) -> u32;

    /// Token acceptance: apply acceptance mask to select between draft and target tokens.
    /// Where mask[i] >= threshold, keep draft_tokens[i]; else resample from target.
    /// Returns final token indices → (R, 1) u32.
    pub fn __tile_token_accept_f32(dst: u32, draft_tokens: u32, target_tokens: u32, accept_probs: u32, threshold: f32, rows: u32) -> u32;

    // ── Decode-optimized intrinsics (cooperative reduction) ──────────

    /// Cooperative matvec with f16 weights, f32 accumulation.
    /// One threadgroup per output row, simd_sum + shared mem reduction.
    /// a=activation(f32, K), b=weight(f16, N×K) → c=output(f32, N).
    pub fn __tile_matvec_f16(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// f32-weight matvec — out[n] = sum_k a[k] * w[n,k].
    /// a=activation(f32, K), b=weight(f32, N×K, row-major by N) → c=output(f32, N).
    /// rows=N (output width), cols=K (inner/contraction dim).
    /// Used by projections when weights haven't been cast to f16/bf16.
    pub fn __tile_matvec_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// Cooperative matvec with f16 weights and bias addition.
    pub fn __tile_matvec_f16_bias(dst: u32, a: u32, b: u32, bias: u32, rows: u32, cols: u32) -> u32;

    /// In-place RoPE for decode: applies rotary embedding to a single token.
    /// x is (num_heads × head_dim), modified in-place.
    pub fn __tile_rope_inplace_f32(dst: u32, x: u32, rows: u32, cols: u32) -> u32;

    /// KV cache update: copy vector (num_heads × head_dim) into cache at position.
    /// Cache layout: (num_heads, max_seq, head_dim).
    pub fn __tile_kv_cache_update_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Decode attention with cooperative threadgroup reduction.
    /// One threadgroup per head, supports GQA.
    /// 5-phase: score → max → exp+sum → normalize → weighted V sum.
    pub fn __tile_attention_decode_f32(dst: u32, q: u32, k: u32, v: u32, rows: u32, cols: u32) -> u32;

    // --- Prefill-path intrinsics (batched multi-position) ---

    /// Batched RoPE for prefill: applies RoPE to (seq_len, num_heads, head_dim) in-place.
    pub fn __tile_rope_prefill_f32(dst: u32, x: u32, rows: u32, cols: u32) -> u32;

    /// Batched KV cache update for prefill: copies seq_len vectors into cache.
    pub fn __tile_kv_cache_update_prefill_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Prefill attention: Q is (seq_len, num_heads, head_dim), K/V from cache.
    /// One threadgroup per (query_head, query_position) pair. Causal + GQA.
    pub fn __tile_attention_prefill_f32(dst: u32, q: u32, k: u32, v: u32, rows: u32, cols: u32) -> u32;

    // --- Fused decode intrinsics ---

    /// Cooperative matvec + residual add: out = A @ B^T + R.
    pub fn __tile_matvec_f16_add(dst: u32, a: u32, b: u32, r: u32, rows: u32, cols: u32) -> u32;

    /// Fused gate+up+silu: reads activation once, computes silu(A@W_gate) * (A@W_up).
    pub fn __tile_gate_up_silu_f16(dst: u32, a: u32, w_gate: u32, w_up: u32, rows: u32, cols: u32) -> u32;
}

// --- f16 tile operations --------------------------------------------------

extern "C" {
    /// Load a `ROWS × COLS` f16 tile from global memory (stored as u16).
    pub fn __tile_load_f16(gm: *const u16, rows: u32, cols: u32) -> u32;

    /// Store a local f16 tile buffer to global memory (as u16).
    pub fn __tile_store_f16(gm: *mut u16, buf: u32, rows: u32, cols: u32);

    /// Element-wise add two f16 tiles.
    pub fn __tile_add_f16(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise multiply two f16 tiles.
    pub fn __tile_mul_f16(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise softmax of an f16 tile.
    pub fn __tile_softmax_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
}

// --- Pipelining markers ---------------------------------------------------
//
// A `pipelined_for` loop advises the MLIR backend that the following loop
// body is a K-dim / tile-iteration loop where prologue-overlapped DMA and
// compute should be emitted. The marker is a zero-cost tag pair
// (`_begin(depth) ... _end()`) wrapped around the Rust `for` body. The cpp
// emitter consumes it to force-on (depth >= 2) or force-off (depth == 1)
// the existing double-buffer detection in `detect_tiling_loop`. The PTO
// backend ignores it (ptoas auto-inserts sync); GPU/MSL/NKI/Pallas emitters
// ignore it today but can honor the depth hint when they grow native
// scf.for handling.
//
// `depth` is the pipeline depth (2 = double-buffered, 3 = triple, etc.).
// `depth == 1` is an explicit opt-out of pipelining on this loop.

extern "C" {
    #[doc(hidden)]
    pub fn __tile_pipelined_for_begin(depth: u32);
    #[doc(hidden)]
    pub fn __tile_pipelined_for_end();
}

/// Mark the following tile-iteration loop as software-pipelined with the
/// given depth.
///
/// Place the marker *before* the loop — the cpp emitter's
/// `detect_tiling_loop` scans from the loop header through the latch back to
/// the header, so the marker only needs to be reachable from the header
/// block.
///
/// Usage (from `examples/tile_softmax_double_buf/kernels/src/lib.rs`):
/// ```ignore
/// pipelined_for(2);                       // depth=2 = double-buffered
/// let mut i = 0u32;
/// loop {
///     if i >= n_tiles { break; }
///     let offset = (i as usize) * COLS;
///     let t = tile_load_f32::<1, COLS>(input.wrapping_add(offset));
///     let y = safe::tile_softmax_f32(t);
///     tile_store_f32::<1, COLS>(output.wrapping_add(offset), y);
///     i = i + 1u32;
/// }
/// ```
///
/// The cpp backend pairs the `_begin` with the loop back-edge and forces the
/// tiling-loop heuristic on (for `depth >= 2`) or off (for `depth == 1`).
/// The PTO backend drops the marker entirely — `ptoas --enable-insert-sync`
/// derives the cross-pipe sync set from the tile-op DAG and produces the
/// same double-buffer overlap without the hint.
/// All other vendor emitters (GPU/NKI/MSL/SPIRV/BANG/Gaudi/AIE/TPU) also
/// drop the marker; they can grow native `scf.for` pipeline-hint support
/// later without breaking existing kernels.
#[inline(always)]
pub fn pipelined_for(depth: u32) {
    unsafe { __tile_pipelined_for_begin(depth) }
}

// --- Safe wrappers --------------------------------------------------------
//
// These wrappers encode const generic dimensions at the call site and
// pass them as runtime u32 values to the underlying extern "C" intrinsics.
// The codegen backend can inspect the call arguments to reconstruct the
// tile shape when emitting PTO assembly.

/// Load a `ROWS × COLS` f32 tile from global memory.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid f32 values.
#[inline(always)]
pub fn tile_load_f32<const ROWS: usize, const COLS: usize>(
    gm: *const f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_load_f32(gm, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store an f32 tile to global memory. Consumes the tile handle.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` writable f32 values.
#[inline(always)]
pub fn tile_store_f32<const ROWS: usize, const COLS: usize>(
    gm: *mut f32,
    tile: Tile<ROWS, COLS, f32>,
) {
    unsafe { __tile_store_f32(gm, tile.buf_id, ROWS as u32, COLS as u32) };
}

/// Element-wise add two f32 tiles, producing a new tile.
#[inline(always)]
pub fn tile_add_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_add_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise multiply two f32 tiles, producing a new tile.
#[inline(always)]
pub fn tile_mul_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_mul_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise exp of an f32 tile.
#[inline(always)]
pub fn tile_exp_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_exp_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise softmax of an f32 tile (reduces over columns).
#[inline(always)]
pub fn tile_softmax_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_softmax_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Matrix multiply f32: `(M × K) @ (K × N) → (M × N)`.
#[inline(always)]
pub fn tile_matmul_f32<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, f32>,
    b: Tile<K, N, f32>,
) -> Tile<M, N, f32> {
    let buf_id =
        unsafe { __tile_matmul_f32(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise subtract two f32 tiles: a - b.
#[inline(always)]
pub fn tile_sub_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_sub_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise divide two f32 tiles: a / b.
#[inline(always)]
pub fn tile_div_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_div_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise negate an f32 tile.
#[inline(always)]
pub fn tile_neg_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_neg_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Row-wise max reduction. Returns (ROWS × 1) tile.
#[inline(always)]
pub fn tile_reduce_max_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_reduce_max_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Row-wise sum reduction. Returns (ROWS × 1) tile.
#[inline(always)]
pub fn tile_reduce_sum_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_reduce_sum_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Scale all elements of a tile by a scalar.
#[inline(always)]
pub fn tile_scale_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scalar: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_scale_f32(0, src.buf_id, scalar, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Clamp all elements of a tile to `[lo, hi]`.
#[inline(always)]
pub unsafe fn tile_clamp_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    lo: f32,
    hi: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_clamp_f32(0, src.buf_id, lo, hi, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Fused scaled dot-product attention: softmax(Q @ K^T / sqrt(D)) @ V.
///
/// Fuses four operations into a single PTOAS-optimizable pipeline:
/// 1. matmul(Q, K^T) → scores (S × S)
/// 2. scale(scores, 1/sqrt(D))
/// 3. softmax(scores) → weights (S × S)
/// 4. matmul(weights, V) → output (S × D)
///
/// The Rust type system enforces dimension constraints:
/// Q(S×D) × K(S×D) → scores(S×S) × V(S×D) → out(S×D).
/// Move semantics ensure Q, K, V are each consumed exactly once.
#[inline(always)]
pub fn tile_attention_f32<const S: usize, const D: usize>(
    q: Tile<S, D, f32>,
    k: Tile<S, D, f32>,
    v: Tile<S, D, f32>,
) -> Tile<S, D, f32> {
    let buf_id = unsafe {
        __tile_attention_f32(0, q.buf_id, k.buf_id, v.buf_id, S as u32, D as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

// ── Transformer building-block wrappers ─────────────────────────────

/// SiLU/Swish: silu(x) = x * sigmoid(x).
#[inline(always)]
pub fn tile_silu_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_silu_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// RoPE: rotary position embedding at position `pos`.
#[inline(always)]
pub fn tile_rope_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    pos: u32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rope_f32(0, src.buf_id, pos, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Causal mask: upper triangle → -inf.
#[inline(always)]
pub fn tile_causal_mask_f32<const S: usize>(
    src: Tile<S, S, f32>,
) -> Tile<S, S, f32> {
    let buf_id = unsafe { __tile_causal_mask_f32(0, src.buf_id, S as u32, S as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Embedding: gather rows from (V, D) weight table by indices.
#[inline(always)]
pub fn tile_embedding_f32<const V: usize, const D: usize, const COUNT: usize>(
    weight: Tile<V, D, f32>,
    indices: *const u32,
) -> Tile<COUNT, D, f32> {
    let buf_id = unsafe {
        __tile_embedding_f32(0, weight.buf_id, indices, V as u32, D as u32, COUNT as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Cross-entropy: -log(softmax(logits)[target]) per row.
#[inline(always)]
pub fn tile_cross_entropy_f32<const ROWS: usize, const COLS: usize>(
    logits: Tile<ROWS, COLS, f32>,
    targets: *const u32,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe {
        __tile_cross_entropy_f32(0, logits.buf_id, targets, ROWS as u32, COLS as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

// ── Phase 0: foundational primitive wrappers ────────────────────────

/// Transpose: (ROWS × COLS) → (COLS × ROWS).
#[inline(always)]
pub fn tile_transpose_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<COLS, ROWS, f32> {
    let buf_id = unsafe { __tile_transpose_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise 1/sqrt(x).
#[inline(always)]
pub fn tile_rsqrt_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rsqrt_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise natural logarithm.
#[inline(always)]
pub fn tile_log_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_log_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise sigmoid: 1/(1+exp(-x)).
#[inline(always)]
pub fn tile_sigmoid_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_sigmoid_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}


/// Cast f32 tile to f16.
#[inline(always)]
pub fn tile_cast_f32_f16<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_cast_f32_f16(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Cast f16 tile to f32.
#[inline(always)]
pub fn tile_cast_f16_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_cast_f16_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Extract a sub-tile (slice).
#[inline(always)]
pub fn tile_slice_f32<
    const R: usize, const C: usize,
    const R2: usize, const C2: usize,
>(
    src: Tile<R, C, f32>,
    row_off: usize,
    col_off: usize,
) -> Tile<R2, C2, f32> {
    let buf_id = unsafe {
        __tile_slice_f32(
            0, src.buf_id,
            row_off as u32, col_off as u32,
            R as u32, C as u32, R2 as u32, C2 as u32,
        )
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Concatenate along columns: (R, C1) ++ (R, C2) → (R, C1+C2).
#[inline(always)]
pub fn tile_concat_f32<const R: usize, const C1: usize, const C2: usize>(
    a: Tile<R, C1, f32>,
    b: Tile<R, C2, f32>,
) -> Tile<R, { C1 + C2 }, f32> {
    let buf_id = unsafe {
        __tile_concat_f32(0, a.buf_id, b.buf_id, R as u32, C1 as u32, C2 as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Indexed scatter: scatter (N, D) rows into (M, D) at positions.
#[inline(always)]
pub fn tile_scatter_f32<const N: usize, const M: usize, const D: usize>(
    src: Tile<N, D, f32>,
    indices: *const u32,
) -> Tile<M, D, f32> {
    let buf_id = unsafe {
        __tile_scatter_f32(0, src.buf_id, indices, N as u32, M as u32, D as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Indexed gather: gather from (M, D) at positions → (N, D).
#[inline(always)]
pub fn tile_gather_f32<const N: usize, const M: usize, const D: usize>(
    src: Tile<M, D, f32>,
    indices: *const u32,
) -> Tile<N, D, f32> {
    let buf_id = unsafe {
        __tile_gather_f32(0, src.buf_id, indices, N as u32, M as u32, D as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Top-K selection per row: returns (R, K) values, writes indices.
#[inline(always)]
pub fn tile_topk_f32<const R: usize, const C: usize, const K: usize>(
    src: Tile<R, C, f32>,
    indices_out: *mut u32,
) -> Tile<R, K, f32> {
    let buf_id = unsafe {
        __tile_topk_f32(0, src.buf_id, indices_out, R as u32, C as u32, K as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// f16 matrix multiply: C = A(M×K) @ B(K×N).
#[inline(always)]
pub fn tile_matmul_f16<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u16>,
    b: Tile<K, N, u16>,
) -> Tile<M, N, u16> {
    let buf_id = unsafe {
        __tile_matmul_f16(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// int8 matmul with i32 accumulator and per-column f32 dequant → f16 GM tile.
///
/// C_f16 = dequant((A_i8 @ B_i8) as i32, scale_per_col_f32).
///
/// Returns an `f16` tile (u16 bit-pattern) that the PTO emitter lowers to a
/// `pto.tstore_fp` into GM — the scale tile is read from `scale` (a GM
/// pointer to the pre-computed per-column f32 scales) via `loc=scaling`.
///
/// See memory/project_pto_i8_tmatmul_validated.md.
#[inline(always)]
pub fn tile_matmul_i8_acc_i32_dequant_f16<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u32>,
    b: Tile<K, N, u32>,
    scale: *const f32,
) -> Tile<M, N, u16> {
    let buf_id = unsafe {
        __tile_matmul_i8_acc_i32_dequant_f16(
            0, a.buf_id, b.buf_id, scale, M as u32, K as u32, N as u32,
        )
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Load a `ROWS × COLS` i8 tile from global memory.
///
/// Per the existing i8 tile convention in this file (`tile_quantize_f32_i8`),
/// i8 tiles are typed `Tile<R, C, u32>` — the `u32` element slot is a marker
/// for the `i8` dtype at the emitter level, not the bit-width of each element.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid i8 values.
#[inline(always)]
pub fn tile_load_i8<const ROWS: usize, const COLS: usize>(
    gm: *const i8,
) -> Tile<ROWS, COLS, u32> {
    let buf_id = unsafe { __tile_load_i8(gm, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store an i8 tile to global memory. Consumes the tile handle.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` writable i8 values.
#[inline(always)]
pub fn tile_store_i8<const ROWS: usize, const COLS: usize>(
    gm: *mut i8,
    tile: Tile<ROWS, COLS, u32>,
) {
    unsafe { __tile_store_i8(gm, tile.buf_id, ROWS as u32, COLS as u32) };
}

/// f16 matrix multiply with f32 accumulator output: C = A(M×K) @ B(K×N).
///
/// Same FFI as `tile_matmul_f16` but the returned tile is typed `f32` so the
/// natural downstream op is `tile_store_f32` / `tile_add_f32`. This mirrors
/// the PTO emitter's registration of the L0C accumulator under the result
/// SSA as dtype=f32 (the only mixed-dtype triple ptoas on CANN 8.5 accepts
/// for f16 ops is `(f32, f16, f16)`). In-flight FixPipe handles any final
/// f32→target-dtype cast during the acc→GM DMA.
#[inline(always)]
pub fn tile_matmul_f16_acc_f32<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u16>,
    b: Tile<K, N, u16>,
) -> Tile<M, N, f32> {
    let buf_id = unsafe {
        __tile_matmul_f16(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Matmul with transposed RHS: `C[M,N] = A[M,K] @ B[N,K]^T`, f32.
///
/// `B` is shaped `Tile<N, K, f32>` in natural weight layout (rows=out, cols=in).
/// The kernel consumes the transpose implicitly via layout, so no explicit
/// transpose tile is materialised.
#[inline(always)]
pub fn tile_matmul_transposed_f32<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, f32>,
    b: Tile<N, K, f32>,
) -> Tile<M, N, f32> {
    let buf_id = unsafe {
        __tile_matmul_transposed_f32(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// f16 variant of `tile_matmul_transposed_f32`.
#[inline(always)]
pub fn tile_matmul_transposed_f16<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u16>,
    b: Tile<N, K, u16>,
) -> Tile<M, N, u16> {
    let buf_id = unsafe {
        __tile_matmul_transposed_f16(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

// ── Phase 3: Flash Attention primitive wrappers ────────────────────

/// Fill a tile with a scalar value.
#[inline(always)]
pub fn tile_fill_f32<const ROWS: usize, const COLS: usize>(
    scalar: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_fill_f32(0, scalar, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise max of two tiles.
#[inline(always)]
pub fn tile_max_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_max_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

// tile_sub_f32, tile_reduce_max_f32, tile_div_f32 are defined earlier in
// this module. The Phase-3 block previously redeclared them; duplicates
// have been removed.

/// RMS normalization: x * rsqrt(mean(x^2) + eps).
#[inline(always)]
pub fn tile_rms_norm_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    eps: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rms_norm_f32(0, src.buf_id, eps, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

// ── Phase 4: Quantization primitive wrappers ──────────────────────

/// Symmetric INT8 quantize: round(x / scale), clamped to [-128, 127].
#[inline(always)]
pub fn tile_quantize_f32_i8<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scale: f32,
) -> Tile<ROWS, COLS, u32> {
    let buf_id = unsafe { __tile_quantize_f32_i8(0, src.buf_id, scale, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// INT8 dequantize: x_i8 * scale → f32.
#[inline(always)]
pub fn tile_dequantize_i8_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u32>,
    scale: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_dequantize_i8_f32(0, src.buf_id, scale, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Row-wise absmax: max(|x|) per row → (R, 1).
#[inline(always)]
pub fn tile_absmax_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_absmax_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

// ── Phase 6: Multi-token prediction / speculative decoding wrappers ──

/// Row-wise argmax: returns index of max value per row → (R, 1) u32 indices.
#[inline(always)]
pub fn tile_argmax_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe { __tile_argmax_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Top-p (nucleus) sampling from logits.
///
/// Applies temperature scaling, softmax, sorts by probability, computes cumsum,
/// masks below top_p threshold, then samples from the remaining distribution.
/// Returns sampled token indices per row → (R, 1) u32.
#[inline(always)]
pub fn tile_sample_top_p_f32<const ROWS: usize, const COLS: usize>(
    logits: Tile<ROWS, COLS, f32>,
    temperature: f32,
    top_p: f32,
    rng_seed: u32,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe {
        __tile_sample_top_p_f32(0, logits.buf_id, temperature, top_p, rng_seed, ROWS as u32, COLS as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Draft verification: compare draft token indices against target model logits.
///
/// For each position, computes `p_target[draft_token] / p_draft[draft_token]` as
/// the acceptance ratio. Returns per-position acceptance probabilities → (R, 1).
#[inline(always)]
pub fn tile_draft_verify_f32<const ROWS: usize, const COLS: usize>(
    draft_tokens: Tile<ROWS, 1, u32>,
    target_logits: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe {
        __tile_draft_verify_f32(0, draft_tokens.buf_id, target_logits.buf_id, ROWS as u32, COLS as u32)
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Token acceptance: select between draft and resampled tokens based on acceptance mask.
///
/// Where `accept_probs[i] >= threshold`, keeps `draft_tokens[i]`;
/// otherwise uses `target_tokens[i]` (resampled from target distribution).
/// Returns final token sequence → (R, 1) u32.
#[inline(always)]
pub fn tile_token_accept_f32<const ROWS: usize>(
    draft_tokens: Tile<ROWS, 1, u32>,
    target_tokens: Tile<ROWS, 1, u32>,
    accept_probs: Tile<ROWS, 1, f32>,
    threshold: f32,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe {
        __tile_token_accept_f32(
            0, draft_tokens.buf_id, target_tokens.buf_id, accept_probs.buf_id,
            threshold, ROWS as u32,
        )
    };
    Tile { buf_id, _phantom: PhantomData }
}

/// Load two independent `ROWS × COLS` f32 tiles from separate GM pointers in one call.
///
/// Expresses two independent DMA loads so ptoas (via `--enable-insert-sync`) can
/// schedule them concurrently on the Mte2 pipe without an intervening barrier.
/// Semantically equivalent to two consecutive `tile_load_f32` calls, but signals
/// the programmer's intent that `gm0` and `gm1` are independent.
///
/// # Safety
/// Both `gm0` and `gm1` must point to at least `ROWS * COLS` valid f32 values.
#[inline(always)]
pub fn tile_join_load_f32<const ROWS: usize, const COLS: usize>(
    gm0: *const f32,
    gm1: *const f32,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    let t0 = unsafe { tile_load_f32::<ROWS, COLS>(gm0) };
    let t1 = unsafe { tile_load_f32::<ROWS, COLS>(gm1) };
    (t0, t1)
}

/// Issue a prefetch load of a `ROWS × COLS` f32 tile from global memory.
///
/// Identical to `tile_load_f32` at the hardware level. The different name signals
/// *double-buffering intent*: the caller intends to overlap this load with
/// compute on a previously loaded tile.  ptoas will schedule the resulting
/// `pto.tload` on the Mte2 pipe independently of the Vector-pipe ops that follow.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid f32 values.
#[inline(always)]
pub fn tile_prefetch_f32<const ROWS: usize, const COLS: usize>(
    gm: *const f32,
) -> Tile<ROWS, COLS, f32> {
    unsafe { tile_load_f32::<ROWS, COLS>(gm) }
}

/// Load a `ROWS × COLS` f16 tile from global memory (stored as u16).
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid u16 (f16) values.
#[inline(always)]
pub fn tile_load_f16<const ROWS: usize, const COLS: usize>(
    gm: *const u16,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_load_f16(gm, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store an f16 tile to global memory (as u16). Consumes the tile handle.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` writable u16 values.
#[inline(always)]
pub fn tile_store_f16<const ROWS: usize, const COLS: usize>(
    gm: *mut u16,
    tile: Tile<ROWS, COLS, u16>,
) {
    unsafe { __tile_store_f16(gm, tile.buf_id, ROWS as u32, COLS as u32) };
}

/// Element-wise add two f16 tiles.
#[inline(always)]
pub fn tile_add_f16<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, u16>,
    b: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_add_f16(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise softmax of an f16 tile.
#[inline(always)]
pub fn tile_softmax_f16<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_softmax_f16(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decode-specific tile operations
// ═══════════════════════════════════════════════════════════════════════════

/// Cooperative matrix-vector product with f16/bf16 weights.
///
/// Computes `out = A @ W^T` where A is (1, K) in f32 and W is (N, K) in f16.
/// Uses one threadgroup per output row with simd_sum + shared memory reduction.
/// Maps to `matvec_f16` kernel on Metal, cooperative matvec on AscendC.
#[inline(always)]
pub fn tile_matvec_f16<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w: Tile<N, K, f32>,  // stored as f16/bf16 on device
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_f16(0, a.buf_id, w.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Matrix-vector product with f32 weights.
///
/// Computes `out[n] = sum_k a[k] * w[n,k]` with full f32 precision on the
/// weight side (no f16/bf16 cast). For use when projection weights remain
/// f32 — the common case for DeepSeek decode on NPU before quantization.
#[inline(always)]
pub fn tile_matvec_f32<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w: Tile<N, K, f32>,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_f32(0, a.buf_id, w.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cooperative matrix-vector product with f16/bf16 weights + bias.
///
/// Computes `out = A @ W^T + bias`.
#[inline(always)]
pub fn tile_matvec_f16_bias<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w: Tile<N, K, f32>,
    bias: Tile<1, N, f32>,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_f16_bias(0, a.buf_id, w.buf_id, bias.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// In-place RoPE for decode (single position).
///
/// Applies rotary position embedding to x in-place.
/// Position is passed as a runtime parameter (not const generic).
#[inline(always)]
pub fn tile_rope_inplace_f32<const ROWS: usize, const COLS: usize>(
    x: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rope_inplace_f32(0, x.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// KV cache update: write a (num_heads, head_dim) vector into cache at current position.
///
/// Cache layout: (num_heads, max_seq, head_dim). Position is a runtime parameter.
#[inline(always)]
pub fn tile_kv_cache_update_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_kv_cache_update_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Single-token attention decode with KV cache.
///
/// For each Q head: score = Q @ K_cache^T / sqrt(head_dim), softmax, @ V_cache.
/// Supports GQA (multiple Q heads share one KV head).
/// Maps to `attention_decode` kernel on Metal.
#[inline(always)]
pub fn tile_attention_decode_f32<const NH: usize, const DH: usize>(
    q: Tile<NH, DH, f32>,
    k_cache: Tile<NH, DH, f32>,  // runtime: (NKV, max_seq, DH)
    v_cache: Tile<NH, DH, f32>,  // runtime: (NKV, max_seq, DH)
) -> Tile<NH, DH, f32> {
    let buf_id = unsafe { __tile_attention_decode_f32(0, q.buf_id, k_cache.buf_id, v_cache.buf_id, NH as u32, DH as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store a u32 tile (for argmax results).
#[inline(always)]
pub fn tile_store_u32<const ROWS: usize, const COLS: usize>(
    gm: *mut u32,
    tile: Tile<ROWS, COLS, u32>,
) {
    // Reuse the f32 store path — u32 and f32 have the same size
    unsafe { __tile_store_f32(gm as *mut f32, tile.buf_id, ROWS as u32, COLS as u32) };
}

// --- Prefill-path wrappers ---

/// Batched RoPE for prefill: applies rotary embeddings to (seq_len, num_heads, head_dim).
/// Each thread handles one (position, head, dim_pair) triple.
#[inline(always)]
pub fn tile_rope_prefill_f32<const NH: usize, const DH: usize>(
    x: Tile<NH, DH, f32>,
) -> Tile<NH, DH, f32> {
    let buf_id = unsafe { __tile_rope_prefill_f32(0, x.buf_id, NH as u32, DH as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Batched KV cache update for prefill: copies seq_len vectors into cache.
/// src is (seq_len, num_heads, head_dim), cache is (num_heads, max_seq, head_dim).
#[inline(always)]
pub fn tile_kv_cache_update_prefill_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_kv_cache_update_prefill_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Prefill attention: Q(seq_len, num_heads, head_dim) vs KV cache.
/// One threadgroup per (query_head, query_position) pair. Causal masking + GQA.
#[inline(always)]
pub fn tile_attention_prefill_f32<const NH: usize, const DH: usize>(
    q: Tile<NH, DH, f32>,
    k_cache: Tile<NH, DH, f32>,
    v_cache: Tile<NH, DH, f32>,
) -> Tile<NH, DH, f32> {
    let buf_id = unsafe { __tile_attention_prefill_f32(0, q.buf_id, k_cache.buf_id, v_cache.buf_id, NH as u32, DH as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cooperative matvec + residual add: out[row] = sum_k(A[k] * B[row,k]) + R[row].
/// Fuses matvec + elementwise add to eliminate one dispatch and buffer round-trip.
#[inline(always)]
pub fn tile_matvec_f16_add<const ROWS: usize, const COLS: usize>(
    a: Tile<1, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,   // runtime: bf16 weights
    r: Tile<ROWS, 1, f32>,      // residual vector
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_matvec_f16_add(0, a.buf_id, b.buf_id, r.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Fused gate+up+silu: out[i] = silu(A @ W_gate[i,:]) * (A @ W_up[i,:]).
/// Reads activation vector once, computes both gate and up projections, then fuses silu*mul.
#[inline(always)]
pub fn tile_gate_up_silu_f16<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w_gate: Tile<N, K, f32>,   // runtime: bf16 weights
    w_up: Tile<N, K, f32>,     // runtime: bf16 weights
) -> Tile<N, 1, f32> {
    let buf_id = unsafe { __tile_gate_up_silu_f16(0, a.buf_id, w_gate.buf_id, w_up.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ===========================================================================
// Safe view-based API
// ===========================================================================
//
// These wrappers accept typed `GmView` / `GmViewMut` instead of raw pointers
// and compute ops accept `Tile` by value. The shape check happens through
// const generics on both sides of the call. Everything below this line is
// safe Rust: the only `unsafe` hops are `GmDeviceCtx::new()` and
// `ctx.view{,_mut}`, which the kernel ABI wrapper does exactly once per
// argument at entry. Views borrow `&'a self` from the ctx, so they can't
// outlive it — kernels cannot leak device pointers back to the host.
//
// Raw-pointer intrinsics (`tile_load_f32`, `tile_add_f32`, etc.) remain
// available unchanged — they're how the in-tree kernels work today and how
// the codegen backends walk MIR. The view-based wrappers are additive.
//
// Naming: the view-based load/store use a `_view` suffix to disambiguate
// from the raw intrinsic of the same op. Compute ops are re-exported in a
// `safe` nested module so callers can write `safe::tile_add_f32(a, b)`
// without collision.

/// Safe load: a `GmView<R,C,T>` already promises `R*C` elements of `T`
/// backed by a valid device pointer, so the load itself is safe.
#[inline(always)]
pub fn tile_load_view_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_load_f32(view.as_ptr(), ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Safe store: consumes the tile handle (linearity) and writes through the
/// mutable view. The view enforces shape match at the call site.
#[inline(always)]
pub fn tile_store_view_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, f32>,
    tile: Tile<ROWS, COLS, f32>,
) {
    unsafe {
        __tile_store_f32(view.as_mut_ptr(), tile.buf_id, ROWS as u32, COLS as u32)
    };
}

/// Safe join-load: loads two copies of the same tile shape from two views
/// in one op. Enables move-only data flow when the same input is consumed
/// by two downstream paths (e.g. `x * sigmoid(x)`).
#[inline(always)]
pub fn tile_join_load_view_f32<'a, 'b, const ROWS: usize, const COLS: usize>(
    view_a: &GmView<'a, ROWS, COLS, f32>,
    view_b: &GmView<'b, ROWS, COLS, f32>,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    unsafe { tile_join_load_f32::<ROWS, COLS>(view_a.as_ptr(), view_b.as_ptr()) }
}

/// Load the same view's contents twice into two independent tile handles.
///
/// Unlocks shape-(B) migration for kernels that need to consume the same
/// input along two data paths (`x * sigmoid(x)`, `x² + x`, …) but only
/// have a single `GmView` param from the `#[tile_kernel]` macro. Semantically
/// equivalent to two consecutive `tile_load_view_f32` calls on the same
/// view — ptoas will fuse or pipeline them as the backend sees fit.
#[inline(always)]
pub fn tile_load_view_f32_twice<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    unsafe { tile_join_load_f32::<ROWS, COLS>(view.as_ptr(), view.as_ptr()) }
}

/// Load the same view's contents four times into four independent tile
/// handles. Used by fused reduce+apply kernels like RMS-norm where the
/// same input is squared, summed, and reapplied in the normalization step.
#[inline(always)]
pub fn tile_load_view_f32_quad<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    let ptr = view.as_ptr();
    let (t0, t1) = unsafe { tile_join_load_f32::<ROWS, COLS>(ptr, ptr) };
    let (t2, t3) = unsafe { tile_join_load_f32::<ROWS, COLS>(ptr, ptr) };
    (t0, t1, t2, t3)
}

/// Safe prefetch: identical in effect to `tile_load_view_f32` at the
/// intrinsic level (delegates to `__tile_load_f32`); the distinct
/// name lets ptoas's dependency analysis recognize the issue as a
/// double-buffer prefetch and overlap it with compute.
#[inline(always)]
pub fn tile_prefetch_view_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    unsafe { tile_prefetch_f32::<ROWS, COLS>(view.as_ptr()) }
}

/// Safe load / f16 variant.
#[inline(always)]
pub fn tile_load_view_f16<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id =
        unsafe { __tile_load_f16(view.as_ptr(), ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Safe store / f16 variant.
#[inline(always)]
pub fn tile_store_view_f16<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, u16>,
    tile: Tile<ROWS, COLS, u16>,
) {
    unsafe {
        __tile_store_f16(view.as_mut_ptr(), tile.buf_id, ROWS as u32, COLS as u32)
    };
}

/// Safe store / u32 variant. Used for argmax / token-index outputs.
/// Reuses the f32 store path because u32 and f32 share the same width.
#[inline(always)]
pub fn tile_store_view_u32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, u32>,
    tile: Tile<ROWS, COLS, u32>,
) {
    unsafe {
        __tile_store_f32(
            view.as_mut_ptr() as *mut f32,
            tile.buf_id,
            ROWS as u32,
            COLS as u32,
        )
    };
}

/// Safe re-exports of the pure compute ops. A `Tile` already carries the
/// full shape + dtype at the type level, so these are safe by construction
/// — the `unsafe` on the raw intrinsics is inherited from the extern "C"
/// block, not from any real safety invariant.
///
/// Example kernel body written against this module:
///
/// ```rust,ignore
/// use tile_std::tile::{GmView, GmViewMut, tile_load_view_f32, tile_store_view_f32, safe};
///
/// #[tile_std::tile_kernel]
/// pub fn softmax_tile_safe(
///     input:  GmView<'_, 1, 1024, f32>,
///     output: GmViewMut<'_, 1, 1024, f32>,
/// ) {
///     let x = tile_load_view_f32(&input);
///     let y = safe::tile_softmax_f32(x);
///     tile_store_view_f32(&output, y);
/// }
/// ```
///
/// No `unsafe` blocks inside the kernel body.
pub mod safe {
    use super::*;

    #[inline(always)]
    pub fn tile_add_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>, b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_add_f32(a, b) } }

    #[inline(always)]
    pub fn tile_sub_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>, b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_sub_f32(a, b) } }

    #[inline(always)]
    pub fn tile_mul_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>, b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_mul_f32(a, b) } }

    #[inline(always)]
    pub fn tile_div_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>, b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_div_f32(a, b) } }

    #[inline(always)]
    pub fn tile_neg_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_neg_f32(t) } }

    #[inline(always)]
    pub fn tile_exp_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_exp_f32(t) } }

    #[inline(always)]
    pub fn tile_softmax_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_softmax_f32(t) } }

    #[inline(always)]
    pub fn tile_matmul_f32<const M: usize, const K: usize, const N: usize>(
        a: Tile<M, K, f32>, b: Tile<K, N, f32>,
    ) -> Tile<M, N, f32> { unsafe { super::tile_matmul_f32(a, b) } }

    /// `C[M,N] = A[M,K] @ B[N,K]^T` — B is in natural weight layout.
    #[inline(always)]
    pub fn tile_matmul_transposed_f32<const M: usize, const K: usize, const N: usize>(
        a: Tile<M, K, f32>, b: Tile<N, K, f32>,
    ) -> Tile<M, N, f32> { unsafe { super::tile_matmul_transposed_f32(a, b) } }

    // ── Reductions ──────────────────────────────────────────────────
    #[inline(always)]
    pub fn tile_reduce_max_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> { unsafe { super::tile_reduce_max_f32(t) } }

    #[inline(always)]
    pub fn tile_reduce_sum_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> { unsafe { super::tile_reduce_sum_f32(t) } }

    #[inline(always)]
    pub fn tile_absmax_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> { unsafe { super::tile_absmax_f32(t) } }

    #[inline(always)]
    pub fn tile_argmax_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, 1, u32> { unsafe { super::tile_argmax_f32(t) } }

    // ── Elementwise scalar-param ────────────────────────────────────
    #[inline(always)]
    pub fn tile_scale_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>, scalar: f32,
    ) -> Tile<R, C, f32> { unsafe { super::tile_scale_f32(t, scalar) } }

    #[inline(always)]
    pub fn tile_rsqrt_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_rsqrt_f32(t) } }

    #[inline(always)]
    pub fn tile_silu_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_silu_f32(t) } }

    #[inline(always)]
    pub fn tile_rms_norm_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>, eps: f32,
    ) -> Tile<R, C, f32> { unsafe { super::tile_rms_norm_f32(t, eps) } }

    // ── Transformer building blocks ─────────────────────────────────
    #[inline(always)]
    pub fn tile_attention_f32<const S: usize, const D: usize>(
        q: Tile<S, D, f32>, k: Tile<S, D, f32>, v: Tile<S, D, f32>,
    ) -> Tile<S, D, f32> { unsafe { super::tile_attention_f32(q, k, v) } }

    #[inline(always)]
    pub fn tile_rope_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>, pos: u32,
    ) -> Tile<R, C, f32> { unsafe { super::tile_rope_f32(t, pos) } }

    #[inline(always)]
    pub fn tile_causal_mask_f32<const S: usize>(
        t: Tile<S, S, f32>,
    ) -> Tile<S, S, f32> { unsafe { super::tile_causal_mask_f32(t) } }

    // ── Shape / layout ──────────────────────────────────────────────
    #[inline(always)]
    pub fn tile_transpose_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<C, R, f32> { unsafe { super::tile_transpose_f32(t) } }

    #[inline(always)]
    pub fn tile_slice_f32<
        const R: usize, const C: usize,
        const R2: usize, const C2: usize,
    >(
        t: Tile<R, C, f32>, row_off: usize, col_off: usize,
    ) -> Tile<R2, C2, f32> { unsafe { super::tile_slice_f32(t, row_off, col_off) } }

    // ── Type conversions ────────────────────────────────────────────
    #[inline(always)]
    pub fn tile_cast_f32_f16<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, u16> { unsafe { super::tile_cast_f32_f16(t) } }

    #[inline(always)]
    pub fn tile_cast_f16_f32<const R: usize, const C: usize>(
        t: Tile<R, C, u16>,
    ) -> Tile<R, C, f32> { unsafe { super::tile_cast_f16_f32(t) } }

    #[inline(always)]
    pub fn tile_dequantize_i8_f32<const R: usize, const C: usize>(
        t: Tile<R, C, u32>, scale: f32,
    ) -> Tile<R, C, f32> { unsafe { super::tile_dequantize_i8_f32(t, scale) } }

    // ── Indexed / sampling ──────────────────────────────────────────
    //
    // These take raw `*const u32` / `*mut u32` index arrays. The raw-pointer
    // parts stay `unsafe` at the call site because they carry no shape info
    // and are passed from the host via the C ABI. The tile op itself is safe.

    /// # Safety
    /// `indices_out` must be a valid `*mut u32` buffer of at least `R * K` elements.
    #[inline(always)]
    pub unsafe fn tile_topk_f32<const R: usize, const C: usize, const K: usize>(
        t: Tile<R, C, f32>, indices_out: *mut u32,
    ) -> Tile<R, K, f32> { unsafe { super::tile_topk_f32(t, indices_out) } }

    /// # Safety
    /// `indices` must be a valid `*const u32` buffer of at least `N` elements.
    #[inline(always)]
    pub unsafe fn tile_scatter_f32<const N: usize, const M: usize, const D: usize>(
        src: Tile<N, D, f32>, indices: *const u32,
    ) -> Tile<M, D, f32> { unsafe { super::tile_scatter_f32(src, indices) } }

    /// # Safety
    /// `indices` must be a valid `*const u32` buffer of at least `COUNT` elements.
    #[inline(always)]
    pub unsafe fn tile_embedding_f32<const V: usize, const D: usize, const COUNT: usize>(
        weight: Tile<V, D, f32>, indices: *const u32,
    ) -> Tile<COUNT, D, f32> { unsafe { super::tile_embedding_f32(weight, indices) } }

    /// # Safety
    /// `targets` must be a valid `*const u32` buffer of at least `R` elements.
    #[inline(always)]
    pub unsafe fn tile_cross_entropy_f32<const R: usize, const C: usize>(
        logits: Tile<R, C, f32>, targets: *const u32,
    ) -> Tile<R, 1, f32> { unsafe { super::tile_cross_entropy_f32(logits, targets) } }

    #[inline(always)]
    pub fn tile_sample_top_p_f32<const R: usize, const C: usize>(
        logits: Tile<R, C, f32>, temperature: f32, top_p: f32, rng_seed: u32,
    ) -> Tile<R, 1, u32> {
        unsafe { super::tile_sample_top_p_f32(logits, temperature, top_p, rng_seed) }
    }

    #[inline(always)]
    pub fn tile_draft_verify_f32<const R: usize, const C: usize>(
        draft_tokens: Tile<R, 1, u32>, target_logits: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> {
        unsafe { super::tile_draft_verify_f32(draft_tokens, target_logits) }
    }

    #[inline(always)]
    pub fn tile_token_accept_f32<const R: usize>(
        draft_tokens: Tile<R, 1, u32>,
        target_tokens: Tile<R, 1, u32>,
        accept_probs: Tile<R, 1, f32>,
        threshold: f32,
    ) -> Tile<R, 1, u32> {
        unsafe { super::tile_token_accept_f32(draft_tokens, target_tokens, accept_probs, threshold) }
    }

}
