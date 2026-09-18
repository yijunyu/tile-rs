#include <cuda_runtime.h>

__global__ void softmax(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}
