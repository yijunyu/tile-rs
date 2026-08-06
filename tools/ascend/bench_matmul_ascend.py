#!/usr/bin/env python3
"""aclnnMatmul reference bench -- the vendor baseline for tile-rs PTO matmuls.

Why this exists: the generated PTO matmul is bandwidth-pinned at ~85 GB/s across
a 9x range of shapes, which "looks low" only against the part's PUBLISHED HBM
figure. That is a spec sheet, not a measurement. This gives the same shapes a
vendor-library number on the same box, turning "looks low" into a ratio.

It calls the vendor library; it does NOT hand-write an AscendC kernel. The point
of the project is generated kernels, so AscendC by hand is not a fallback we
want -- aclnn here is purely the yardstick, the way ggml-metal is on Apple.

Protocol matches the rest of the project: discard the first run, median of
repeats, and a correctness gate (max relative error against numpy) so a
configuration cannot look fast by computing less.

REQUIRES A SOURCED CANN ENVIRONMENT. Without ASCEND_OPP_PATH the workspace query
fails with 561103, whose surface text is only "Inner Error!"; aclGetRecentErrMsg
spells it out ("ASCEND_OPP_PATH is null", "opp kernel real path can not be
found"). Setting ASCEND_HOME_PATH and LD_LIBRARY_PATH alone is NOT enough --
run `source <cann>/set_env.sh` first.

Usage: source set_env.sh && bench_matmul_ascend.py [reps]
"""
import ctypes, os, sys, time, statistics
import numpy as np

CANN = os.environ.get("ASCEND_HOME_PATH",
       os.environ.get("ACLRS_CANN_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
LIB = os.path.join(CANN, "lib64")
acl = ctypes.CDLL(os.path.join(LIB, "libascendcl.so"))
api = ctypes.CDLL(os.path.join(LIB, "libopapi.so"))

ACL_FLOAT, ACL_FORMAT_ND = 0, 2
H2D, D2H, HUGE_FIRST = 1, 2, 0

for f, a, r in [
    ("aclInit", [ctypes.c_char_p], ctypes.c_int),
    ("aclrtSetDevice", [ctypes.c_int], ctypes.c_int),
    ("aclrtMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
    ("aclrtFree", [ctypes.c_void_p], ctypes.c_int),
    ("aclrtMemcpy", [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
    ("aclrtCreateStream", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
    ("aclrtSynchronizeStream", [ctypes.c_void_p], ctypes.c_int),
]:
    fn = getattr(acl, f); fn.argtypes, fn.restype = a, r

api.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p]
api.aclCreateTensor.restype = ctypes.c_void_p


acl.aclGetRecentErrMsg.restype = ctypes.c_char_p


def chk(ret, what):
    # Always surface aclGetRecentErrMsg: the numeric code alone sent me chasing
    # a tensor-layout bug for two rounds when the real cause was an unsourced
    # environment.
    if ret != 0:
        msg = (acl.aclGetRecentErrMsg() or b"").decode(errors="replace")
        raise AssertionError(f"{what} failed: {ret}\n{msg[:400]}")


def malloc(n):
    p = ctypes.c_void_p()
    chk(acl.aclrtMalloc(ctypes.byref(p), n, HUGE_FIRST), f"aclrtMalloc({n})")
    return p


def h2d(dev, arr):
    chk(acl.aclrtMemcpy(dev, arr.nbytes, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, H2D), "h2d")


def d2h(arr, dev):
    chk(acl.aclrtMemcpy(arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, dev, arr.nbytes, D2H), "d2h")


# aclCreateTensor RETAINS the shape and stride pointers it is given; it does not
# copy them. Letting the ctypes arrays go out of scope leaves the tensor holding
# freed memory and the workspace query segfaults. Keep them alive for the
# process lifetime.
_KEEPALIVE = []


def tensor(shape, dev):
    """Row-major ND tensor over an existing device allocation."""
    n = len(shape)
    shp = (ctypes.c_int64 * n)(*shape)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    strd = (ctypes.c_int64 * n)(*strides)
    _KEEPALIVE.extend((shp, strd))
    t = api.aclCreateTensor(shp, n, ACL_FLOAT, strd, 0, ACL_FORMAT_ND, shp, n, dev)
    assert t, "aclCreateTensor returned NULL"
    return ctypes.c_void_p(t)


# cubeMathType: 0 KEEP_DTYPE, 1 ALLOW_FP32_DOWN_PRECISION, 2 USE_FP16, 3 USE_HF32.
# KEEP_DTYPE is the like-for-like comparison against an f32 generated kernel; the
# others are reported because they show what the vendor library can do when
# allowed to drop precision, which the generated kernel is not doing.
MODES = [(0, "KEEP_DTYPE"), (1, "ALLOW_FP32_DOWN"), (3, "USE_HF32")]

SHAPES = [(16, 896, 4864), (16, 1536, 1536), (16, 2048, 11008)]


def bench(M, K, N, mode, reps):
    a_h = ((np.arange(M * K) % 31).astype(np.float32) - 15) * 0.031
    b_h = ((np.arange(K * N) % 17).astype(np.float32) - 8) * 0.05
    a_h = a_h.reshape(M, K); b_h = b_h.reshape(K, N)

    a_d, b_d, c_d = malloc(a_h.nbytes), malloc(b_h.nbytes), malloc(M * N * 4)
    h2d(a_d, a_h); h2d(b_d, b_h)
    ta, tb, tc = tensor([M, K], a_d), tensor([K, N], b_d), tensor([M, N], c_d)

    gw = api.aclnnMatmulGetWorkspaceSize
    gw.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int8,
                   ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    gw.restype = ctypes.c_int
    ws = ctypes.c_uint64(); ex = ctypes.c_void_p()
    chk(gw(ta, tb, tc, ctypes.c_int8(mode), ctypes.byref(ws), ctypes.byref(ex)),
        "aclnnMatmulGetWorkspaceSize")
    wsp = malloc(max(ws.value, 16)) if ws.value else ctypes.c_void_p(0)

    # An aclnn executor is SINGLE-USE by default: calling aclnnMatmul twice with
    # the same one segfaults. Timing needs repeats, so mark it repeatable.
    rep = getattr(api, "aclSetAclOpExecutorRepeatable", None)
    if rep is not None:
        rep.argtypes = [ctypes.c_void_p]
        rep.restype = ctypes.c_int
        chk(rep(ex), "aclSetAclOpExecutorRepeatable")

    run = api.aclnnMatmul
    run.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
    run.restype = ctypes.c_int

    stream = STREAM
    times = []
    for i in range(reps + 1):                      # +1: first run discarded
        t0 = time.perf_counter()
        chk(run(wsp, ws, ex, stream), "aclnnMatmul")
        chk(acl.aclrtSynchronizeStream(stream), "sync")
        dt = (time.perf_counter() - t0) * 1e6
        if i:
            times.append(dt)

    c_h = np.empty((M, N), dtype=np.float32)
    d2h(c_h, c_d)
    # Error RELATIVE TO THE MAGNITUDE OF THE PROBLEM, not elementwise. These
    # inputs are signed and sum with heavy cancellation, so individual outputs
    # land near zero and an elementwise relative error explodes there -- it
    # flagged correct f32 results as wrong at one shape and not another purely
    # from where the cancellation fell.
    ref = a_h.astype(np.float64) @ b_h.astype(np.float64)
    err = float(np.max(np.abs(c_h.astype(np.float64) - ref)) / max(np.max(np.abs(ref)), 1e-12))

    for p in (a_d, b_d, c_d):
        acl.aclrtFree(p)
    if ws.value:
        acl.aclrtFree(wsp)
    return statistics.median(times), err


chk(acl.aclInit(None), "aclInit")
chk(acl.aclrtSetDevice(0), "aclrtSetDevice")
STREAM = ctypes.c_void_p()
chk(acl.aclrtCreateStream(ctypes.byref(STREAM)), "aclrtCreateStream")

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
print(f"aclnnMatmul reference, f32, median of {reps} (first discarded)\n")
print(f"{'M x K x N':<22}{'mode':<18}{'us':>10}{'GB/s':>9}{'TFLOPS':>9}{'max_rel_err':>13}")
for (M, K, N) in SHAPES:
    for mode, name in MODES:
        try:
            us, err = bench(M, K, N, mode, reps)
        except AssertionError as e:
            print(f"{f'{M} x {K} x {N}':<22}{name:<18}{'--':>10}  {e}")
            continue
        byt = (M * K + K * N + M * N) * 4
        flop = 2 * M * K * N
        ok = "" if err < 1e-3 else "  <-- WRONG"
        print(f"{f'{M} x {K} x {N}':<22}{name:<18}{us:>10.2f}{byt/us/1e3:>9.1f}"
              f"{flop/us/1e6:>9.2f}{err:>13.2e}{ok}")
