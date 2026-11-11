// silu (antirez sigmoid_stable fidelity).
//
// Two parallel kernels covering both branches of the gate.
// Caller (host) decides which to dispatch based on DS4_SILU_FIDELITY:
//
// ds4_silu_default_f32   = x / (1 + exp(-x))            (positive-branch always)
// ds4_silu_fidelity_f32  = x * sigmoid_stable(x)        (antirez ds4.c:4736)
//
// The split avoids a host-side branch inside the kernel body. Both
// kernels operate elementwise; one thread per element. Output layout
// matches input.
//
// Used by the encoder when computing shared_expert SwiGLU and any
// routed-MoE mid (when activation == silu). Byte-exactness against
// `ds4_engine::forward::silu` is required at every callsite — the
// gate's purpose is bit-fidelity with antirez, so anything sloppier
// silently re-opens the very divergence closed.

#include <metal_stdlib>
using namespace metal;

kernel void ds4_silu_default_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& n            [[ buffer(2) ]],
    uint gid    [[ thread_position_in_grid ]]
)
{
    if (gid >= n) return;
    float x = p_in[gid];
    p_out[gid] = x / (1.0f + exp(-x));
}

kernel void ds4_silu_fidelity_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& n            [[ buffer(2) ]],
    uint gid    [[ thread_position_in_grid ]]
)
{
    if (gid >= n) return;
    float x = p_in[gid];
    float s;
    if (x >= 0.0f) {
        float e = exp(-x);
        s = 1.0f / (1.0f + e);
    } else {
        float e = exp(x);
        s = e / (1.0f + e);
    }
    p_out[gid] = x * s;
}
