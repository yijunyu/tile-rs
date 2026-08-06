// ACL host launcher + device entry for tile-rs PTO kernels on Ascend.
//
// ptoas emits `AICORE void <name>(__gm__ T* ...)` — a device function, not a
// launchable entry. This file wraps it in a __global__ __aicore__ entry, then
// runs it from the host with ACL, timing N iterations on a stream.
//
// Build (on a 910B box):
// NOTE: link with -lm. This file's CPU reference softmax calls std::exp, and
// the CCE link does not pull libm -- the failure surfaces as a device-looking
// `ld.lld: undefined symbol: exp` when it is an ordinary host call.
//
//   ccec -O2 -std=c++17 -DMEMORY_BASE --cce-aicore-arch=dav-c220-cube -x cce \
//        -I<cann>/include -L<cann>/lib64 -lascendcl -lruntime
// The kernel .cpp from ptoas is #included so the device function is visible.
//
// Usage: ./pto_run <iters>
//   prints median kernel time in microseconds, plus an output checksum so a
//   zero-work run cannot masquerade as a fast one.

#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#ifndef PTO_KERNEL_HEADER
#define PTO_KERNEL_HEADER "attention.cpp"
#endif
#include PTO_KERNEL_HEADER

#ifndef PTO_M
#define PTO_M 16
#endif
#ifndef PTO_K
#define PTO_K 16
#endif
#ifndef PTO_N
#define PTO_N 16
#endif

// Launchable entry wrapping the ptoas-generated device function.
//
// Two attention variants exist and they do NOT have the same signature. The
// plain one is tile_attn(q,k,v,o); the scratch one is
// tile_attn_s(q,k,v,o,scratch) and routes the score matrix through GM instead
// of an Acc->Vec tmov -- which a2a3 rejects, so the plain variant does not
// compile at any shape. Build with -DPTO_ATTN_SCRATCH for the working one.
#ifdef PTO_ATTN_SCRATCH
extern "C" __global__ AICORE void pto_entry(__gm__ float *q, __gm__ float *k,
                                            __gm__ float *v, __gm__ float *o,
                                            __gm__ float *scr) {
  tile_attn_s(q, k, v, o, scr);
}
#else
extern "C" __global__ AICORE void pto_entry(__gm__ float *q, __gm__ float *k,
                                            __gm__ float *v, __gm__ float *o) {
  tile_attn(q, k, v, o);
}
#endif

#define CHECK(expr)                                                            \
  do {                                                                         \
    aclError _e = (expr);                                                      \
    if (_e != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "%s:%d: ACL error %d in %s\n", __FILE__, __LINE__,   \
                   (int)_e, #expr);                                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 20;
  const size_t na = (size_t)PTO_M * PTO_K, nb = (size_t)PTO_M * PTO_K,
               nc = (size_t)PTO_M * PTO_K;
  const size_t nv = (size_t)PTO_M * PTO_K;
  // Score matrix staged in GM: S x S.
  const size_t nscr = (size_t)PTO_M * PTO_M;

  CHECK(aclInit(nullptr));
  CHECK(aclrtSetDevice(0));
  aclrtStream stream;
  CHECK(aclrtCreateStream(&stream));

  std::vector<float> ha(na), hb(nb), hc(nc, 0.0f);
  unsigned long long seed = 0x9E3779B97F4A7C15ULL;
  auto rnd = [&]() {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
    return (float)(int)(seed >> 33) / (float)(1u << 30);
  };
  for (auto &x : ha) x = rnd();
  for (auto &x : hb) x = rnd();

  std::vector<float> hv(nv);
  for (auto &x : hv) x = rnd();
  void *dscr = nullptr;
  void *da, *db, *dc, *dv;
  CHECK(aclrtMalloc(&da, na * 4, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&db, nb * 4, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dc, nc * 4, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&dv, nv * 4, ACL_MEM_MALLOC_HUGE_FIRST));
#ifdef PTO_ATTN_SCRATCH
  CHECK(aclrtMalloc(&dscr, nscr * 4, ACL_MEM_MALLOC_HUGE_FIRST));
#endif
  CHECK(aclrtMemcpy(dv, nv * 4, hv.data(), nv * 4, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(da, na * 4, ha.data(), na * 4, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemcpy(db, nb * 4, hb.data(), nb * 4, ACL_MEMCPY_HOST_TO_DEVICE));

  // warm-up (first launch pays kernel load)
  pto_entry<<<1, nullptr, stream>>>((__gm__ float *)da, (__gm__ float *)db,
                                    (__gm__ float *)dv, (__gm__ float *)dc
#ifdef PTO_ATTN_SCRATCH
                                    , (__gm__ float *)dscr
#endif
                                    );
  CHECK(aclrtSynchronizeStream(stream));

  std::vector<double> us;
  for (int i = 0; i < iters; ++i) {
    aclrtEvent beg, end;
    CHECK(aclrtCreateEvent(&beg));
    CHECK(aclrtCreateEvent(&end));
    CHECK(aclrtRecordEvent(beg, stream));
    pto_entry<<<1, nullptr, stream>>>((__gm__ float *)da, (__gm__ float *)db,
                                      (__gm__ float *)dv, (__gm__ float *)dc
#ifdef PTO_ATTN_SCRATCH
                                      , (__gm__ float *)dscr
#endif
                                      );
    CHECK(aclrtRecordEvent(end, stream));
    CHECK(aclrtSynchronizeStream(stream));
    float ms = 0.f;
    CHECK(aclrtEventElapsedTime(&ms, beg, end));
    us.push_back((double)ms * 1000.0);
    CHECK(aclrtDestroyEvent(beg));
    CHECK(aclrtDestroyEvent(end));
  }

  CHECK(aclrtMemcpy(hc.data(), nc * 4, dc, nc * 4, ACL_MEMCPY_DEVICE_TO_HOST));
  double checksum = 0.0;
  for (float v : hc) checksum += v;

  // CPU reference: softmax(Q @ K^T / sqrt(D)) @ V  (non-causal)
  double max_rel = 0.0;
  const double sc = 1.0 / std::sqrt((double)PTO_K);
  for (int i = 0; i < PTO_M; ++i) {
    std::vector<double> row(PTO_M);
    double mx = -1e300;
    for (int j = 0; j < PTO_M; ++j) {
      double a = 0.0;
      for (int d = 0; d < PTO_K; ++d)
        a += (double)ha[(size_t)i * PTO_K + d] * (double)hb[(size_t)j * PTO_K + d];
      row[j] = a * sc;
      if (row[j] > mx) mx = row[j];
    }
    double den = 0.0;
    for (int j = 0; j < PTO_M; ++j) { row[j] = std::exp(row[j] - mx); den += row[j]; }
    for (int d = 0; d < PTO_K; ++d) {
      double acc = 0.0;
      for (int j = 0; j < PTO_M; ++j) acc += row[j] / den * (double)hv[(size_t)j * PTO_K + d];
      double got = (double)hc[(size_t)i * PTO_K + d];
      double dd = std::fabs(acc) > 1e-6 ? std::fabs(acc) : 1.0;
      double r = std::fabs(got - acc) / dd;
      if (r > max_rel) max_rel = r;
    }
  }

  std::sort(us.begin(), us.end());
  std::printf("M=%d K=%d N=%d iters=%d median=%.2f us checksum=%.4f max_rel_err=%.3e %s\n",
              PTO_M, PTO_K, PTO_N, iters, us[us.size() / 2], checksum, max_rel,
              max_rel < 1e-4 ? "MATCHES-CPU" : "MISMATCH");
  if (checksum == 0.0)
    std::printf("WARNING: zero checksum — kernel may not have computed anything\n");

  CHECK(aclrtFree(da)); CHECK(aclrtFree(db)); CHECK(aclrtFree(dc)); CHECK(aclrtFree(dv));
  CHECK(aclrtDestroyStream(stream));
  CHECK(aclrtResetDevice(0));
  CHECK(aclFinalize());
  return 0;
}
