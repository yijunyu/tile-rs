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


## Block-parallelism audit of the generated PTO kernels

After the single-block launch was found, every generated kernel was checked for
whether it can use more than one AI core.

| kernel | loops | block-parallel | realistic shapes |
|---|---|---|---|
| matmul | K and N | yes (`get_block_idx`) | yes |
| attention | none | no | **no -- UB-rejected past 16x16** |

`emit_blocked_matmul_loops` is the only emitter that emits `get_block_idx`. The
attention path is fully unrolled straight-line code for one 16x16 tile, so at
that size one core genuinely is all the work -- but it does not scale:

    S=16  D=16   generated, 0 loops, not block-parallel
    S=128 D=128  REJECTED: tile of 65536B at offset 262144B would use 327680B
                 > UB_SIZE 262144B
    S=512 D=128  REJECTED: 524288B > UB_SIZE 262144B

So attention is a single-tile proof of concept, not a usable kernel -- but the
reason is NOT what it first looked like.

**The blocker is the UB allocator, not the algorithm.** `UbAllocator` bump-
allocates and never frees: its `free()` exists but is `#[allow(dead_code)]` and
is called from nowhere. The 16x16 attention emits **20 `alloc_tile` ops and zero
frees**, so every intermediate stays live for the whole kernel. That is
harmless at 16x16 (~20 KB) and fatal at 64x64, where each tile is 16 KB:
20 x 16 KB = 320 KB against a 256 KB Unified Buffer, and it dies at exactly the
offset the error reports (246784 B live before the last 16 KB tile).

At 64x64 the REAL data is Q+K+V = 48 KB plus a 16 KB score matrix. It fits four
times over. Both attention paths -- plain and the GM-scratch variant -- fail
identically, which is the giveaway that this is allocation, not dataflow.

Two things were wrong here, and a third is the real blocker.

**Wrong 1 -- the budget billed every tile to UB.** `vec` lives in UB, but `mat`
is L1, `left`/`right` are L0A/L0B and `acc` is L0C: separate physical buffers.
Charging them all to a 256 KB UB refused a 64x64 attention that uses 130 KB of
UB. `place_in` now bills each tile to the memory its `loc=` names, and 64x64
generates.

**Wrong 2 -- nothing is ever freed.** `UbAllocator::free()` is dead code, so a
16x16 attention's 20 `alloc_tile` ops all stay live. That is not what blocked
64x64, but it IS what blocks 128x128: eight `vec` tiles at 64 KB each is 512 KB
against a 256 KB UB. There is now a scratch pool (`acquire_scratch` /
`release_scratch`) wired into the softmax decomposition; it is inert for a
kernel containing a single softmax and will matter when the attention path
reuses its own temporaries.

**The real blocker: the attention kernel has never compiled.** `ptoas` accepts
it, then `ccec` rejects it at ANY shape, including the 16x16 that has always
"generated":

    TMov.hpp:160: static assertion failed ... TMov: Invalid TileType

It emits an Acc -> Vec `tmov`, which is not a legal move on a2a3. Verified
against the pre-change HEAD, so this is long-standing and independent of the
budget work. **The dump test only ever checked that .pto TEXT is produced --
nothing compiled it.** The matmul path is unaffected and still correct on device
(57.00 us at 16x2048x11008, MATCHES-CPU).

**The scratch variant is the one that works -- and it was never compiled either.**
`attention_scratch` routes the score matrix through GM (`tstore` then `tload`)
instead of an `Acc -> Vec` `tmov`, so it emits only the legal `mat->left` /
`mat->right` moves. It type-checks at both 16x16 and 64x64 where the plain
variant fails the static assert. It then stops one step later:

    ld.lld: error: undefined symbol: exp

`pto.texp` lowers to a call to `exp` that nothing provides in the CCE device
link. That is the current frontier: the kernel is type-correct and the remaining
gap is a math symbol, not a structural one.

`attn_run.cpp` builds either variant -- `-DPTO_ATTN_SCRATCH` selects the 5-arg
entry `tile_attn_s(q,k,v,o,scratch)` and allocates the S x S GM scratch buffer.
The two variants do NOT share a signature, which is why running the plain
harness against the scratch kernel silently built nothing.

### `undefined symbol: exp` -- it was HOST code all along

`llvm-objdump -dr` on the object shows the relocation is **`R_AARCH64_CALL26`**,
an AArch64 *host* relocation -- not device code at all. The call is
`attn_run.cpp:153`, `std::exp(row[j] - mx)`, in the harness's own CPU reference
softmax. The CCE link does not pull libm, so **`-lm` is the fix**.

`pto_run.cpp` (matmul) links fine because its CPU reference is multiply-add and
contains no `exp` -- which is exactly why matmul never hit this.

The device side was never implicated: preprocessing shows exactly ONE
`TEXP_IMPL` surviving, the a2a3 one, which calls the `vexp` hardware intrinsic.

Getting there took three wrong device-side hypotheses, recorded below because
each cost a round trip. The lesson is that a link error naming a libc symbol
should have been checked against the HOST object first -- `nm` and one
relocation dump answered in seconds what three rebuild cycles did not.

### The three device-side explanations that were wrong

Three plausible explanations, all falsified on device, recorded so nobody spends
the rounds again:

* **Not the CPU fallback.** `pto/cpu/TExp.hpp` calls libm `exp`, but the whole
  CPU section of `pto_instr_impl.hpp` is behind `#ifdef __CPU_SIM`, which this
  build does not define.
* **Not a missing arch macro.** The a2a3 section is behind
  `#ifdef PTO_NPU_ARCH_A2A3` and does include `npu/a2a3/TUnaryOp.hpp` (line 91),
  whose `TEXP_IMPL` uses the `vexp` hardware intrinsic -- no libm. Adding
  `-DPTO_NPU_ARCH_A2A3` changes nothing, and matmul builds and runs correctly
  WITHOUT it, which proves the a2a3 branch is already active by some other
  route.
* **Not cube-vs-vector core.** `vexp` is a vector intrinsic and attention mixes
  cube (`tmatmul`) with vector (softmax) work, so `--cce-aicore-arch` looked
  like the answer. Both `dav-c220-cube` and `dav-c220-vec` fail identically.

`ptoas` emits `TEXP(v72, v69)`; `TEXP` -> `MAP_INSTR_IMPL(TEXP, ...)` ->
`TEXP_IMPL`. Somewhere in that chain a `TEXP_IMPL` that calls scalar `exp` is
being selected over the `vexp` one. **The next diagnostic is to preprocess
(`ccec -E`) and see which `TEXP_IMPL` actually survives, or `nm` the object for
the referencing symbol** -- rather than guessing at another cause.

So the ordering is: identify which `TEXP_IMPL` is selected, then the plain
variant's Acc->Vec move (or retire it in favour of the scratch path). The budget
and reuse work above is necessary but was never sufficient.

The dump test takes `PTO_DUMP_S` / `PTO_DUMP_D` and records a rejection to
`attention.REJECTED` instead of panicking, so the UB boundary can be probed.


## Shared-box etiquette on 910c

`/usr/local/bin/task-submit` is a root job queue -- `--device N` locks an NPU for
the job, and other users submit through it. Work run directly, as everything
above was, contends with them.

Check `npu-smi` before timing anything: AICore has been observed at 0% (clean,
timings stable to ~1%) and at 100% with all eight devices held by another job.
A measurement taken during the latter is worthless and the run interferes with
whoever holds the device.
