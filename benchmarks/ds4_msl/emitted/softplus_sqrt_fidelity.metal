// softplus_sqrt (antirez f32 piecewise fidelity).
//
// Two parallel kernels covering both branches of the gate.
// Caller (host) decides which to dispatch based on DS4_MOE_ROUTER_FIDELITY:
//
// ds4_softplus_sqrt_default_f32   = sqrt(log1p(exp(x))) via stable identity
// `max(x,0) + log(1 + exp(-|x|))` then sqrt
// ds4_softplus_sqrt_fidelity_f32  = antirez ds4.c:4867 piecewise:
// if x > 20:  sqrt(x)
// if x < -20: sqrt(exp(x))
// else:       sqrt(log1p(exp(x)))
//
// The split mirrors silu_fidelity / sigmoid_fidelity — one symbol per
// fidelity flag, bound once at startup by the host. Each kernel processes
// one f32 element per thread (NOT float4 — antirez's piecewise branches
// don't vectorise without per-lane masking).
//
// Used at decode time inside MoE router logit transform; output Σ=1.5
// multiplies INTO every routed expert's down output, so router-side
// rounding amplifies through the residual stream.

#include <metal_stdlib>
using namespace metal;

// MSL has no log1pf; expand `log(1 + exp(-ax))` directly. For ax ≥ 0,
// `1 + exp(-ax)` ∈ (1, 2], so log() is well-behaved.
static inline float stable_softplus(float x) {
    float ax = fabs(x);
    return max(x, 0.0f) + log(1.0f + exp(-ax));
}

kernel void ds4_softplus_sqrt_default_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& n            [[ buffer(2) ]],
    uint gid    [[ thread_position_in_grid ]]
)
{
    if (gid >= n) return;
    float x = p_in[gid];
    p_out[gid] = sqrt(stable_softplus(x));
}

kernel void ds4_softplus_sqrt_fidelity_f32(
    device const float* p_in    [[ buffer(0) ]],
    device       float* p_out   [[ buffer(1) ]],
    constant uint& n            [[ buffer(2) ]],
    uint gid    [[ thread_position_in_grid ]]
)
{
    if (gid >= n) return;
    float x = p_in[gid];
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = exp(x);
    } else {
        // log1p(exp(x)) — Metal lacks log1p, but for x ∈ [-20, 20] the
        // argument `1 + exp(x)` is in (1+e^-20, 1+e^20] which is far from
        // catastrophic-cancellation territory. Match antirez `log1pf(expf(x))`
        // ordering exactly (f32, single exp then single log).
        sp = log(1.0f + exp(x));
    }
    p_out[gid] = sqrt(sp);
}
