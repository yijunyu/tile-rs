#include <musa_runtime.h>

__global__ void softmax(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}
