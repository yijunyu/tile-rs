#include <metal_stdlib>
using namespace metal;

kernel void softmax(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    out[gid] = exp(in[gid]);
}
