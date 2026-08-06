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

## Attention needs a5 hardware — cannot run on these boxes

`attn_run.cpp` is the attention counterpart of `pto_run.cpp`. It does NOT run
on either available box, and the reason is structural, not a build problem:

* `mlir_to_pto` marks any module containing attention with
  `pto.target_arch = "a5"` (see `module_uses_a5_ops`). Matmul carries no such
  attribute and runs fine.
* `translate_attention` emits `pto.tmov` from an **Acc** tile to a **Vec** tile
  (moving the S×S score matrix out of the accumulator for softmax). On the
  a2a3 path that is rejected at compile time:
  `pto/npu/a2a3/TMov.hpp: static assertion failed ... TMov: Invalid TileType`.
  tile-rs's own comment in `translate_attention_gqa` says the same thing —
  "we don't have working vec→mat / acc→vec tmov pairs on a2a3".
* Both boxes are a2a3 generation (dav-c220 / Ascend910), so the kernel is
  unrunnable here regardless of flags. Compiling for an a5 arch would produce
  a binary this hardware cannot execute.

Consequence for the causal-attention work (P1): it cannot be measured on
Ascend today. Its prerequisite is not the causal elision itself but an
a2a3-compatible attention lowering (avoiding acc→vec tmov), or a5 hardware.


## The vendor reference: `bench_matmul_ascend.py`

Times `aclnnMatmul` on the same shapes as `pto_run`, so the generated kernel has
a yardstick rather than a spec sheet. It calls the vendor library only -- it does
NOT hand-write an AscendC kernel, since generated kernels are the point and
hand-written AscendC is not a fallback we want.

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    python3 bench_matmul_ascend.py 20

Measured on 910c (Ascend910, CANN 9.0.0), f32, median of 20 with the first
discarded, every result checked against numpy:

| M x K x N | tile-rs PTO | aclnn KEEP_DTYPE | ratio |
|---|---|---|---|
| 16 x 896 x 4864 | 17.28 us / 1030 GB/s | 33.96 us / 524 GB/s | **1.97x faster** |
| 16 x 1536 x 1536 | 22.38 us / 430 GB/s | 30.71 us / 314 GB/s | **1.37x faster** |
| 16 x 2048 x 11008 | 56.64 us / 1607 GB/s | 81.97 us / 1110 GB/s | **1.45x faster** |

### The single-block trap -- read this before trusting any older PTO number

`pto_run` launched `<<<1>>>`. The generated kernel's outer N-loop has ALWAYS
been block-parallel -- it strides
`for n_i = get_block_idx(); n_i < n_iters; n_i += get_block_num()` -- so one
block ran the entire matmul on ONE AI core. Launching one block per N-tile is
worth **5.2x to 18.3x**:

| M x K x N | 1 block | N blocks | speedup |
|---|---|---|---|
| 16 x 896 x 4864 | 207.98 us | 17.28 us | 12.0x |
| 16 x 1536 x 1536 | 116.04 us | 22.38 us | 5.2x |
| 16 x 2048 x 11008 | 1036.34 us | 56.64 us | 18.3x |

Every PTO measurement taken before this was a single-core number. That includes
the "bandwidth-pinned at ~85 GB/s" reading -- which was simply one core's share
of the chip, and whose shape-independence was the giveaway I read as a kernel
property. It also includes the Kb/Nb sweep, so those constants were tuned in a
regime the kernel no longer runs in and need re-sweeping.

### Three traps, each of which cost a debugging round

* **An unsourced CANN environment fails as 561103 "Inner Error!"** and nothing
  more. `aclGetRecentErrMsg` spells it out: "ASCEND_OPP_PATH is null", "opp
  kernel real path can not be found". Setting `ASCEND_HOME_PATH` and
  `LD_LIBRARY_PATH` is NOT enough -- `set_env.sh` must be sourced. The bench now
  prints `aclGetRecentErrMsg` on every failure, because the numeric code alone
  sent me chasing a tensor-layout bug that did not exist.
* **An aclnn executor is single-use.** Calling `aclnnMatmul` twice with the same
  executor segfaults; timing needs `aclSetAclOpExecutorRepeatable`.
* **An elementwise relative error metric is wrong for GEMM.** These inputs are
  signed and sum with cancellation, so outputs land near zero and elementwise
  relative error explodes there -- it flagged correct f32 results as wrong at
  one shape and not another purely from where the cancellation fell. Compare
  against the magnitude of the problem: `max|c-ref| / max|ref|`.
