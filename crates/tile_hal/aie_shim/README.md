# Why this C++ shim exists

XRT's **C** API only exposes the legacy Alveo/PL path
(`xrtDeviceLoadXclbinFile` + `xrtPLKernelOpen`). On XDNA2 that fails:

```
[XRT] ERROR: load_axlf: Operation not supported
xrtDeviceLoadXclbinFile failed: error -1
```

The NPU requires `register_xclbin` + `hw_context`, which exist **only in the C++
API** — `nm -D libxrt_coreutil.so` shows no C entry point for either. So a pure
`dlopen`-of-C-symbols backend cannot drive the NPU, however it is written.

This shim wraps the C++ flow behind an opaque C surface so `tile_hal` stays
ordinary Rust FFI. Verified: vecadd on the NPU from a Rust test, `max |diff| = 0`.

## Build

```sh
g++ -std=c++17 -O2 -fPIC -shared aie_shim.cpp -o libaie_shim.so -lxrt_coreutil
```

Needs `libxrt-dev` and `uuid-dev` (XRT's headers include `<uuid/uuid.h>`).

## Surface

| function | purpose |
|---|---|
| `aie_open(index, xclbin, kernel)` | device + register_xclbin + hw_context + kernel; empty `kernel` picks the first in the xclbin |
| `aie_bo_alloc(h, bytes, argno, flags)` | buffer in the right memory group; `flags=1` = cacheable, for the instruction stream |
| `aie_bo_write` / `aie_bo_read` | host<->device with the sync in the right direction |
| `aie_run(h, opcode, insts, words, bos, n)` | dispatch as `(opcode, insts, insts_len, buffers...)` — the convention confirmed on hardware |
| `aie_close` / `aie_bo_free` | teardown |
