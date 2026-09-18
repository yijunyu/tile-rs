// head_rms_norm: per-head RMS over q_heads.
//
// Mirrors decode_step.rs:656-663:
// for h in 0..n_head {
// chunk = q_heads[h*head_dim..(h+1)*head_dim];
// ss = chunk.iter().map(|v| (v as f64)*(v as f64)).sum();
// scale = 1.0 / ((ss / head_dim as f64) as f32 + eps).sqrt();
// chunk *= scale;
// }
//
// Fidelity gate: DS4_HEAD_RMS_F64_FIDELITY toggles f64 accumulator.
// The Metal kernel uses f32 throughout (matches default OFF gate);
// when the env is ON the CPU fallback runs instead via dispatcher dispatch.
//
// Layout: input/output [n_head * head_dim], processed in-place style
// (separate input/output buffers to match the existing kernel style).
// One threadgroup per head, simdgroup reduction within the head.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_head_rms_norm_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& head_dim     [[ buffer(2) ]],
    constant float& eps         [[ buffer(3) ]],
    uint tid    [[ thread_position_in_threadgroup ]],
    uint head   [[ threadgroup_position_in_grid ]],
    uint tcount [[ threads_per_threadgroup ]]
)
{
    uint base = head * head_dim;

    // Sum of squares, float4-vectorized when divisible by 4.
    float local_sum = 0.0f;
    bool vec_ok = (head_dim % 4u) == 0u;
    if (vec_ok) {
        device const float4* p_in_v = (device const float4*)(p_in + base);
        uint n4 = head_dim / 4u;
        for (uint i = tid; i < n4; i += tcount) {
            float4 v = p_in_v[i];
            local_sum += dot(v, v);
        }
    } else {
        for (uint i = tid; i < head_dim; i += tcount) {
            float v = p_in[base + i];
            local_sum += v * v;
        }
    }

    // Intra-simdgroup reduction.
    local_sum = simd_sum(local_sum);

    // Cross-simdgroup reduction via threadgroup memory.
    constexpr uint MAX_SG = 1024 / 32;
    threadgroup float rms_shared[MAX_SG];
    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    uint num_sg     = (tcount + 31u) / 32u;
    if (simd_lane == 0) rms_shared[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum = 0.0f;
    if (tid == 0) {
        for (uint s = 0; s < num_sg; s++) total_sum += rms_shared[s];
        rms_shared[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Match the CPU oracle in decode_step.rs:659 exactly:
    // scale = 1.0 / ((ss / head_dim) + eps).sqrt()
    // i.e. eps is added AFTER the per-head mean, before sqrt.
    float scale = rsqrt(rms_shared[0] / float(head_dim) + eps);

    // Apply scale.
    if (vec_ok) {
        device const float4* p_in_v  = (device const float4*)(p_in + base);
        device       float4* p_out_v = (device       float4*)(p_out + base);
        uint n4 = head_dim / 4u;
        for (uint i = tid; i < n4; i += tcount)
            p_out_v[i] = p_in_v[i] * scale;
    } else {
        for (uint i = tid; i < head_dim; i += tcount)
            p_out[base + i] = p_in[base + i] * scale;
    }
}
