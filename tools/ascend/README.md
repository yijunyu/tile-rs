# Running tile-rs PTO kernels on Ascend — WORKING RECIPE (2026-08-05)

Verified end to end on this box (910c, CANN 9.0.0, Ascend910):
  16x16x16 f32 matmul, median 6.14 us, max_rel_err 9.78e-08 vs CPU reference.

## Pipeline
    tile-rs  --(mlir_to_pto)-->  X.pto
             --(ptoas)-------->  X.cpp   (AscendC)
             --(ccec)--------->  pto_run (host+device, ACL)

## Commands
    ptoas --enable-insert-sync X.pto -o X.cpp

    CANN=/usr/local/Ascend/cann-9.0.0/aarch64-linux
    $CANN/ccec_compiler/bin/ccec -O2 -std=c++17 -DMEMORY_BASE \
      --cce-aicore-arch=dav-c220-cube -x cce pto_run.cpp -o pto_run \
      -I$CANN/include -I. -L$CANN/lib64 -lascendcl -lruntime -lstdc++

    LD_LIBRARY_PATH=$CANN/lib64 ./pto_run 20

## The launchability fix (this was the blocker)
ptoas emits `AICORE void name(__gm__ T*...)` — an aicore FUNCTION, not a
launchable kernel. Wrap it:

    extern "C" __global__ AICORE void pto_entry(__gm__ float*a, ...) { name(a, ...); }

Key points, each of which cost a debugging round:
  * Use `AICORE` (= [aicore]), NOT `__aicore__`. `__aicore__` is undefined in
    this mode, so the parser treats it as the return type and reports
    "kernel function type 'int (...)' must have void return type".
  * `__global__` IS valid and is what produces the launchable symbol pair
    (`T pto_entry` + `V pto_entry__`), matching reg_layer_*.elf.
  * Arch must be `dav-c220-cube` for matmul: it uses Acc (accumulator) tiles,
    which live on the cube core. `dav-c220-vec` fails with 14 errors.
  * `-lstdc++` is required; the CCE driver does not link libstdc++, so
    std::vector/std::sort produce undefined operator new / __gxx_personality_v0.
  * Host `main` and the device kernel CAN share one -x cce TU here (mix mode).

## Why NOT the 910b
CANN 8.5.2 has no `CompactMode` anywhere in its pto headers, but ptoas 0.55
always emits it in Tile<> template args (--cann-output-version does not
suppress it). CANN 9.0.0 defines it (pto/npu/a5/). So PTO kernels from this
ptoas build only on the 9.0.0 box. The 910b gets as far as assembling .pto
and compiling everything except the CompactMode references.

## Files
  pto_run.cpp   host launcher: ACL event timing, output checksum, CPU reference
                check (max_rel_err). Override shape with -DPTO_M/-DPTO_K/-DPTO_N
                and kernel with -DPTO_KERNEL_HEADER=\"X.cpp\".
  matmul.pto / matmul.cpp, attention.pto
