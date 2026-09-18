// q8_0_round_trip: per-32-elt-block activation quantization round-trip.
//
// Mirrors forward.rs::q8_0_round_trip verbatim:
// for each block of 32 (last block may be partial):
// amax = max(|x_i|)
// d    = amax / 127.0
// out_i = clamp(rint(x_i / d), -128, 127) * d         (banker's rounding)
// if amax == 0: out_i = 0
//
// This MUST be byte-exact against the CPU oracle because antirez uses
// it as the int8-dot pre-image; all gate on
// q8_0 fidelity. The kernel uses `rint(x)` (Metal IEEE round-half-to-
// even) which matches C `lrintf` under default FE_TONEAREST.
//
// Layout: input/output `[n_elts]`. One threadgroup per 32-elt block.
// Block size at threadgroup level is fixed at 32 (one simdgroup per block;
// simdgroup-max for amax, simdgroup-broadcast for d). Partial tail block
// is handled in-kernel via `last_block_len`.
//
// Uniforms:
// buffer(2) = n_elts          (uint)
// buffer(3) = n_full_blocks   (uint)  // n_elts / 32
// buffer(4) = last_block_len  (uint)  // n_elts % 32 (0 if exact)

#include <metal_stdlib>
using namespace metal;

kernel void ds4_q8_0_round_trip_f32(
    device const float* p_in        [[ buffer(0) ]],
    device       float* p_out       [[ buffer(1) ]],
    constant uint& n_elts           [[ buffer(2) ]],
    constant uint& n_full_blocks    [[ buffer(3) ]],
    constant uint& last_block_len   [[ buffer(4) ]],
    uint tid    [[ thread_position_in_threadgroup ]],
    uint bid    [[ threadgroup_position_in_grid ]]
)
{
    // 32 threads per block; one simdgroup per threadgroup.
    uint block_len = (bid < n_full_blocks) ? 32u : last_block_len;
    if (block_len == 0u) return;

    uint base = bid * 32u;
    bool active = tid < block_len;

    float v = active ? p_in[base + tid] : 0.0f;
    float a = active ? fabs(v) : 0.0f;

    // Intra-simdgroup amax over the (up to 32) lanes.
    float amax = simd_max(a);

    if (amax == 0.0f) {
        if (active) p_out[base + tid] = 0.0f;
        return;
    }

    float d  = amax / 127.0f;
    float id = 1.0f / d;

    if (active) {
        // `rint(x)` = round-half-to-even (matches C lrintf under
        // default FE_TONEAREST; matches Rust round_ties_even).
        float qf = rint(v * id);
        int   qi = (int)qf;
        qi = clamp(qi, -128, 127);
        p_out[base + tid] = (float)qi * d;
    }
}
