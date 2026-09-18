// sigmoid_stable (antirez ds4.c:4736) elementwise on Metal.
//
// Used by `output_hc_head_one` (ds4.c:7654-7681) — the gated sigmoid
// closed in . The encoder calls this as a primitive between
// the per-head `pre*scale + base` linear and the n_hc-wide weighted
// sum that closes the HC fold.
//
// Two parallel kernels covering both ways the gate is set:
// - `ds4_sigmoid_default_f32`  = 1/(1+exp(-x))      (positive-branch always)
// - `ds4_sigmoid_fidelity_f32` = antirez sign-branched
//
// As with silu, the split avoids a host-side branch inside the kernel
// body — the encoder binds the right symbol once at startup based on
// DS4_SILU_FIDELITY (the same env governs every sigmoid_stable site;
// see forward.rs::sigmoid_stable_antirez).

#include <metal_stdlib>
using namespace metal;

kernel void ds4_sigmoid_default_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& n            [[ buffer(2) ]],
    uint gid    [[ thread_position_in_grid ]]
)
{
    if (gid >= n) return;
    float x = p_in[gid];
    p_out[gid] = 1.0f / (1.0f + exp(-x));
}

kernel void ds4_sigmoid_fidelity_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& n            [[ buffer(2) ]],
    uint gid    [[ thread_position_in_grid ]]
)
{
    if (gid >= n) return;
    float x = p_in[gid];
    if (x >= 0.0f) {
        float e = exp(-x);
        p_out[gid] = 1.0f / (1.0f + e);
    } else {
        float e = exp(x);
        p_out[gid] = e / (1.0f + e);
    }
}
