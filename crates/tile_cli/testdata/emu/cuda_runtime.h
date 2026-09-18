#ifndef TILE_CUDA_EMU_H
#define TILE_CUDA_EMU_H
#include <math.h>
#include <stdint.h>
#include <string.h>
/* One thread at a time: `__global__` becomes an ordinary function and the launch
   coordinates become globals the driver sets between calls. */
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)
struct Dim3 { int x, y, z; };
static struct Dim3 threadIdx = {0,0,0}, blockIdx = {0,0,0};
static struct Dim3 blockDim  = {1,1,1}, gridDim  = {1,1,1};
static inline float rsqrtf_(float x) { return 1.0f / sqrtf(x); }
#ifndef rsqrtf
#define rsqrtf rsqrtf_
#endif
static inline float __expf(float x) { return expf(x); }
static inline float __logf(float x) { return logf(x); }
static inline int min_(int a, int b) { return a < b ? a : b; }
static inline int max_(int a, int b) { return a > b ? a : b; }
#define min min_
#define max max_
static inline int __float2int_rn(float x) { return (int)(x + (x >= 0.0f ? 0.5f : -0.5f)); }
/* `__half` is FLOAT here. The emitted prologue casts between them and a struct cannot be
   cast; the consequence is that f16 kernels would run at f32 precision, so only f32 ones
   are driven. */
typedef float __half;
static inline __half __float2half(float f) { return f; }
static inline float __half2float(__half h) { return h; }
/* Warp primitives CANNOT be emulated one thread at a time. These exist so the file
   compiles; every kernel that actually uses them is excluded from the op list below,
   because a stub returning the value unchanged computes a confident wrong answer. */
static inline float __shfl_down_sync(unsigned m, float v, int d) { (void)m;(void)d; return v; }
static inline float __shfl_xor_sync(unsigned m, float v, int d) { (void)m;(void)d; return v; }
static inline void __syncthreads(void) {}
#endif
