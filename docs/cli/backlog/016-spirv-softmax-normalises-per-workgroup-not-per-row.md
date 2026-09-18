id: 16
area: lowering
title: spirv's softmax gave 224 non-finite values of 1024; fixed and verified on a Vulkan device
opened: 2026-09-03
closed: 2026-09-03

## wanted
To read the twelve `softmax` goldens the way the `reduce_sum` ones were read. Softmax
carries both a max-reduce and a sum-reduce, which is where the miscompile this repo already
shipped once lived.

## got
CUDA is sound and is the model. Its reduction is two-phase — warp reduce, per-warp partials
into `sdata`, `__syncthreads`, then a fold across the warps — and the three interdependent
numbers move together with the width:

    cols=1024:  // Launch: <<<num_tiles, 1024>>>   __shared__ float sdata[32]   (tid < 32)
    cols=256:   // Launch: <<<num_tiles, 256>>>    __shared__ float sdata[8]    (tid < 8)

`musa` shares that emitter. SPIR-V does not:

    cols=256:   layout(local_size_x = 256)   shared float sdata[8]
    cols=1024:  layout(local_size_x = 256)   shared float sdata[8]

The workgroup size is a constant. The reduction inside it is per-WORKGROUP —
`subgroupMax`, `sdata[gl_SubgroupID]`, `barrier()`, then a fold over
`tid < gl_NumSubgroups` — and the data index is `gl_GlobalInvocationID.x`. So a 1024-wide
row is covered by FOUR workgroups, each computing the max and the sum of its own 256
elements and normalising by them. The result is four independently-normalised chunks
presented as one softmax row.

At cols=256 it is correct, because one workgroup happens to cover the row. That is the same
accident the Metal reduction had: right at the width somebody tested, wrong above it.

**The frozen golden is the 1024 case**, so the reference output checked into this repository
for `spirv` softmax is a per-chunk softmax.

## second thing, latent
`barrier()` sits after `if (gid >= pc.num_elements) return;`. A GLSL compute barrier must be
reached by every invocation in the workgroup; threads that return early never do. It does
not bite at 1024 (a multiple of 256, so nobody returns) and does bite for any width that is
not, which is undefined behaviour rather than a wrong number.

## why filed and not fixed
No Vulkan device here — #001. The fix is not small either: covering a row wider than the
workgroup needs a strided loop and a per-row dispatch, which is the shape
`emit_reduce_sum_msl` has and this kernel does not. Writing that blind is what was declined
for CUDA, for the f16 tolerance, and for gaudi.

## how it was found, and the method that works
By comparing the emitted constants against the width the kernel was given, at two widths:

    grep -oE "local_size_x = [0-9]+"   at cols=256 and cols=1024

CUDA's three numbers move; SPIR-V's do not. This is the check that survives after the
"emit at two widths and diff the whole file" idea was withdrawn in #015 — diffing text
cannot tell a correct run-time-shaped kernel from a hardcoded one, but asking whether the
kernel's OWN declared sizes track the width can.

## workaround
`spirv` softmax is correct only for rows of exactly the workgroup size (256).


## resolved — and it was worse than filed, then measured

Filing this said the kernel was wrong for rows wider than the workgroup and correct at 256.
Reading it again for the fix showed it was wrong at 256 as well:

    row_max = (tid < gl_NumSubgroups) ? sdata[tid] : -inf;
    row_max = subgroupMax(row_max);

Only subgroup 0 spans `tid < gl_NumSubgroups`. Every OTHER subgroup reduces a full set of
`-inf` and takes `-inf` as the row maximum, so its threads compute `exp(val + inf)`. The
fold reached one subgroup and all eight read the result.

That predicts a precise number: 7 subgroups of 32 threads produce garbage, so 224 of 1024
outputs should be non-finite.

**This machine turned out to have a Vulkan device** — Apple M1 Ultra through Mesa
KosmicKrisp 26.2.0, Vulkan 1.4.354, with `glslangValidator` and `spirv-val` alongside. So it
was measured rather than argued. A 130-line compute harness, the same input generator the
Metal harness uses, `vkCmdDispatch(1,1,1)` with `num_elements = 1024`:

    old shader    max abs inf        non-finite 224 of 1024   sum inf
    fixed shader  max abs 7.229e-10  non-finite 0             sum 1.000000
                  max rel 4.689e-07

224, exactly as predicted from reading the code.

## the fix
The shape `emit_reduce_sum_msl` uses, which is verified on three GPUs: one workgroup per
row, a strided loop so every column is visited at any width, the fold BROADCAST through
shared memory so every thread reads the same row statistics, and no early return before any
barrier. The dispatch contract is stated in the emitted source the way CUDA states its
launch.

Two baked assumptions went with it. `sdata` was sized `local_x / 32`, assuming
`gl_SubgroupSize == 32`; Vulkan permits 4 to 64, and anything narrower makes
`gl_NumSubgroups` exceed that count and `sdata[gl_SubgroupID]` write off the end. It is one
slot per thread now, which is true for every subgroup size and costs 1KB. And `barrier()`
after `if (gid >= n) return;` is undefined for any width that is not a multiple of the
workgroup; there is no early return left.
