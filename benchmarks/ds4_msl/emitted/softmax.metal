// Iter3 (option E): cross-SIMD reduce via simd_max/simd_sum + vector fast::exp.
// Replaces iter2's "8 shared reads + 7 register maxes" tree with one SIMD intrinsic
// per reduction. Same 2-barrier structure as iter2.
//
// Why not option A (online softmax): the per-element exp values must still be
// produced for the output. Maintaining (m,d) across a custom-monoid SIMD reduce
// adds ~12 extra fast::exp() per thread (5 SIMD-combine + 7 TG-combine + 1
// rescale) to save a single threadgroup_barrier — almost certainly net-negative.
// Why not option D (mem_none): Metal mem_none gives sync without ordering;
// shared-mem writes from lane 0 are not guaranteed visible to other threads'
// reads, so the second barrier can't be downgraded safely.
// Why not option C: simd_* ops do not cross SIMD groups on Apple GPUs.
//
// Compile: xcrun metal -c <file>.metal -o <file>.air && xcrun metallib <file>.air -o <file>.metallib
#include <metal_stdlib>
using namespace metal;

kernel void ds4_softmax(
    device const  float* p0 [[ buffer(0) ]],
    device        float* p1 [[ buffer(1) ]],
    constant uint& num_elements [[ buffer(2) ]],
    uint tid    [[ thread_position_in_threadgroup ]],
    uint row    [[ threadgroup_position_in_grid ]],
    uint tcount [[ threads_per_threadgroup ]],
    uint sg_id  [[ simdgroup_index_in_threadgroup ]],
    uint lane   [[ thread_index_in_simdgroup ]]
) {
    threadgroup float buf_max[8];
    threadgroup float buf_sum[8];
    const uint base = row * num_elements;
    device const float4* in4  = (device const float4*)(p0 + base);
    device       float4* out4 = (device       float4*)(p1 + base);

    float4 v = in4[tid];
    float tmax = max(max(v.x, v.y), max(v.z, v.w));
    tmax = simd_max(tmax);
    if (lane == 0) buf_max[sg_id] = tmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every SIMD pulls the 8 per-SIMD maxes into its lanes 0..7 (others sentinel)
    // and reduces with a single simd_max instead of an 8-read 7-op register tree.
    float mval = (lane < 8) ? buf_max[lane] : -FLT_MAX;
    const float row_max = simd_max(mval);

    float4 e = fast::exp(v - row_max);
    float tsum = (e.x + e.y) + (e.z + e.w);
    tsum = simd_sum(tsum);
    if (lane == 0) buf_sum[sg_id] = tsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sval = (lane < 8) ? buf_sum[lane] : 0.0f;
    const float row_sum = simd_sum(sval);

    out4[tid] = e * (1.0f / row_sum);
}
