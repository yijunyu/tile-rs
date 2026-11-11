# Integration test: three Apple GPUs

Run 2026-09-01 from the installed release binary, against `mac` (M2 Max), `mini` (M4)
and `studio` (M1 Ultra, the development host).

## What it was testing

Requirement (3), "pure Rust", is not a build property — it is a claim about what a user
has to do before the tool works. The test is therefore deployment, not compilation:

    scp ~/.cargo/bin/tile mac:/tmp/tile && ssh mac /tmp/tile doctor

No install, no vendor library, no SDK bootstrap, no `brew`. It ran on both hosts and
detected each machine correctly:

| host | device | metal | detected via |
|---|---|---|---|
| studio | Apple M1 Ultra | 32023.883 | system_profiler |
| mac | Apple M2 Max | 32023.620 | system_profiler |
| mini | Apple M4 | 32023.864 | system_profiler |

Three different Metal versions, one binary, nothing installed.

## Numerics: identical across three GPU generations

`softmax` lowered to MSL and run with `-r`, checked against the CPU reference and against
PyTorch 2.8.0:

    kernel vs torch   max abs 3.03e-9   on all three
    ours   vs torch   max abs 3.49e-9   on all three
    verdict: all three agree

The kernel-versus-torch error is identical to three significant figures on M1 Ultra, M2
Max and M4. That is the expected result for IEEE-conformant operations in a fixed order,
and getting it is evidence that the emitted kernel does not depend on the GPU generation
it lands on. A difference here would have been the interesting outcome.

## Two reporting defects the run exposed

Neither would have been found on one machine, and both flattered the tool.

**1. A slowdown was labelled a speedup.** A 1024-element softmax is dominated by dispatch
overhead, so the device is genuinely slower than a scalar CPU loop — correctly measured,
and reported as `speedup: 0.1x`, which skims as a win. Now:

    SLOWDOWN : 0.12x against the reference — the device is 8.0x SLOWER here
               at this size the dispatch costs more than the work; try a
               larger problem before concluding anything about the kernel

Confirmed on all three: 8.0x slower (M1 Ultra), 8.7x (M2 Max), 6.2x (M4).

**2. A debug build inflates every ratio by ~11x.** The same 64x128x64 matmul, same commit,
same machine, same kernel:

    debug build      59.7x against the reference
    release build     5.2x against the reference

The device time is identical; the *baseline* is eleven times slower, because the reference
is a scalar Rust loop compiled without optimization. Nothing in the report hinted at it,
and a number quoted from a development build is not reproducible by anyone running an
installed one. The report now warns under `debug_assertions`.

The corrected matmul figures, from the release binary, order sensibly by generation:

| host | device | matmul speedup |
|---|---|---|
| studio | M1 Ultra | 5.2x |
| mac | M2 Max | 6.0x |
| mini | M4 | 7.4x |

## What this does not show

The reference is a naive single-threaded scalar loop written by this tool, so none of
these ratios is a GPU-versus-CPU figure, and the report says so every time. `-r` still
exists only for Metal (backlog #001).

## `-O4` finds a different answer on each GPU (2026-09-02)

The measured autotuner sweeps threadgroup widths on the device and writes the winner into
the artifact, with the caveat that it is one machine's measurement. That caveat is not
boilerplate — the machines disagree:

| host | device | best width | best | slowest tried | gain |
|---|---|---:|---:|---:|---:|
| studio | Apple M1 Ultra | **1024** | 17.25 us | 50.04 us @ 32 | 66% |
| mini | Apple M4 | **512** | 7.87 us | 21.25 us @ 32 | 63% |

On the M1 Ultra the widest configuration wins; on the M4 it does not — 1024 measures
8.29 us there, slower than 512. A tuning result copied from one to the other would pick a
configuration that is measurably worse on the machine it lands on, which is exactly what
the artifact's header warns against:

    // O4 (measured on Apple M4): dispatch this with a threadgroup of 512 threads.
    // This is a measurement of ONE machine, not a portable constant: re-run `-O4` on
    // the target you deploy to.

Both sweeps run in a single harness process; the whole six-width sweep takes about five
seconds, so re-running it per target is cheap enough that there is no excuse to copy one.

## From a clean clone (2026-09-02)

Incremental builds hide missing files and manifest mistakes, so the tree was cloned fresh
and built as a new user would:

```
git clone <repo> && cd crates/tile_cli && cargo build --features stats
```

It builds without `--features pico`, which is the point: the PICO emitter lives in a
sibling checkout that most people will not have, and the default path must not need it.
From that binary:

| check | result |
|---|---|
| `tile doctor` | detects the M1 Ultra and its Metal version |
| `tile k.mlir -t msl -o o.metal` | one `kernel void`, route and fidelity reported |
| `tile k.mlir -t pico` | **exit 4**, and the rebuild instruction is real — `pico` IS a declared feature |
| `tile k.mlir -t msl -r` | runs on the GPU, all three sources agree, and the debug-build warning fires |

**It found a real regression.** Adding `crates/tile_ui_wasm` — which declares its own
`[workspace]`, like `tile_cli` — without listing it in the root manifest's `exclude` broke
`cargo build` at the repository root for everyone:

    error: multiple workspace roots found in the same workspace

Nothing in `tile_cli`'s own tests would have caught it, because `tile_cli` is itself
excluded and builds in its own workspace, so the root manifest is never exercised there.
`every_self_workspacing_crate_is_excluded_from_the_root` now checks it on every run.

## Every referenced op, verified on the GPU (2026-09-02)

`--list-ops` says what LOWERS. That is not correctness, so every op the harness has a
reference for was run on the M1 Ultra against both the CPU reference and PyTorch 2.14.0:

| op | max abs vs torch |
|---|---:|
| relu, neg, abs | 0.00e0 (exact) |
| sigmoid | 1.79e-7 |
| log | 3.58e-7 |
| sqrt | 4.77e-7 |
| exp | 9.54e-7 |

Plus softmax (3.03e-9) and matmul (0.00e0) from earlier runs.

Four more references were then added, because `--list-ops` showed msl lowering seventeen
ops while only nine could be checked -- and an op nobody can check is one nobody has
checked:

| op | max abs vs torch |
|---|---:|
| tanh | 5.96e-8 |
| silu, softplus, rsqrt | 3.58e-7 to 4.77e-7 |

**Thirteen ops are now numerically verified on hardware** rather than merely lowered.
`rsqrt` is the interesting one: the harness feeds [-2, 2.25], so it takes the reciprocal
square root of negatives (NaN) and of zero (infinity), and kernel and reference agree on
every one of those positions as well as the finite ones. That agreement is now actually
checked rather than skipped.

**It exposed a hole in the comparison.** `sqrt` and `log` take the root and logarithm of
negative inputs -- the harness feeds [-2, 2.25] -- so both sides produce NaN. They agreed,
which is right. What was wrong is the mechanism: `if e > worst` is FALSE when `e` is NaN,
so every NaN difference was skipped while `checked` went on counting the element as
compared. A kernel producing NaN where the reference produced a finite number would have
passed with a clean maximum error.

Both-NaN is still agreement, because it is the same behaviour on both sides. A ONE-SIDED
NaN or infinity is now counted separately and reported before the tolerances, since no
magnitude describes it and "max abs 0.00e0" beside a thousand NaNs is the most misleading
line this tool could print.


## The same eleven ops on a second GPU (2026-09-02)

The three-GPU claim earlier rested on softmax alone. All eleven unary ops were re-run on
`mini` (Apple M4) against the same references:

| op | M1 Ultra | M4 |
|---|---:|---:|
| relu, neg, abs | 0.00e0 | 0.00e0 |
| sqrt | 4.77e-7 | 4.77e-7 |
| log | 3.58e-7 | 3.58e-7 |
| tanh | 5.96e-8 | 5.96e-8 |
| rsqrt | 4.77e-7 | 4.77e-7 |
| silu | 3.58e-7 | 3.58e-7 |
| softplus | 3.58e-7 | 3.58e-7 |
| **exp** | 9.54e-7 | **4.77e-7** |
| **sigmoid** | 1.79e-7 | **5.96e-8** |

Nine of eleven agree to the digit across two GPU generations. `exp` and `sigmoid` do not,
and the M4 is closer to the reference on both -- consistent with a different
implementation of the transcendental in the shader library rather than anything in the
emitted kernel, which is byte-identical on both machines.

This QUALIFIES the earlier line that the numerics are "identical to three significant
figures on three GPU generations". That was measured on softmax, and it is true of
softmax. It is not true of every op: two of thirteen differ by a factor of two in their
worst element, both far inside tolerance. The emitted kernel does not depend on the GPU
generation; the last bits of its transcendentals do.

## A real miscompile: row reductions covered 32 elements (2026-09-02)

`--list-ops` shows msl lowering `reduce_max` and `absmax`. Both were wrong.

    fn emit_reduce_max_msl:
        val = simd_max(val);
        if (tid == 0) p1[row] = val;

`simd_max` reduces across one SIMD GROUP -- 32 lanes on Apple silicon -- not across the
threadgroup. For any row wider than 32 the kernel returned the maximum of the first 32
elements and called it the row's. Measured on an M1 Ultra over a strictly increasing
1024-wide row: **0.31 where the true maximum was 10.23**.

`absmax` had the identical body, and it is the worse of the two: absmax feeds quantization
scales. A scale computed from a thirty-second of the row is too small, so every value
outside that window saturates on the way to int8 -- a quantized model that is quietly and
uniformly wrong rather than obviously broken.

**Nothing already built could have caught this.** The refusing arm does not fire, because
there IS an arm. The ignored-intrinsic detector does not fire, because the output really
does depend on the intrinsic. And the harness's own inputs could not have caught it
either: they are periodic with period 17, so the maximum always falls inside the first 32
elements and a correct kernel and this one agree. It took a hand-written input that rises
monotonically across the row.

`emit_sum_rows_msl` -- the next function in the same file -- already did it correctly:
a strided loop so every column is visited, then a cross-SIMD fold through threadgroup
memory. Both reductions now do the same, verified on the GPU at 10.23.

Two unit tests had to change, and they are worth recording: both asserted the emitted MSL
contained `uint gid = row * tcount + tid;`, which is the signature of the BROKEN version.
They were pinning the bug -- they would have failed on the fix and passed on the defect.
They now assert the strided loop and the barrier instead.

### What this says about the input generator

A fixed periodic input is fine for elementwise ops, where every element is its own test.
It is not enough for reductions, where the answer depends on which element is extreme and
a short period guarantees the extreme is early. Nothing in the harness knows that
difference yet; this note is the evidence for whoever adds a reduction-aware input.

### Closing that gap: the harness can now catch it

Two changes, and the second is the one that matters.

**Reductions are a shape.** `Shape::RowReduce { rows, cols }` sizes the output at one value
per row, and binds `num_elements` to the ROW WIDTH rather than the output size -- a
reduction told `num_elements = rows` reduces a sliver of each row. `reduce_max` and
`absmax` have references and torch expressions, so `-r` verifies them like any other op.
Both now report `max abs 0.00e0` against `torch.amax`.

**The input can now tell a partial reduction from a whole one.** It could not before:

    (i % 17) * 0.25 - 2.0          periodic; the maximum is in EVERY 32-element window
    ... + i * 1e-4                 the extreme lands near the end of the row

Reconstructing the pre-fix kernel byte for byte and running it under the new input:

    broken kernel   2.001600
    true row max    2.101900

A gap of 0.1, far outside any tolerance. Under the OLD input both returned exactly 2.0 --
which is why this shipped. `the_input_can_actually_discriminate_a_partial_reduction`
asserts the margin so the property cannot be lost again, and the ramp is small enough
(0.1 across 1024 elements against 0.25 steps) that every previously verified op still
verifies.

### The same defect in CUDA, and why it was refused rather than fixed

Auditing the class turned it up in `mlir_to_gpu`, where the emitted source says so itself:

    // block-wide max reduce — warp phase only (assumes N<=32)
    float _v1 = warp_reduce_max((float)_v0);
    p1[goff] = _v1;

`warp_reduce_max` folds across one warp -- 32 lanes. The kernel declares
`__shared__ float sdata[32]` for a cross-warp phase and never writes or reads it. `absmax`
has the same body in a separate arm, and prints `cols=1024` into the source it emits, so
the width was known and unchecked.

There is no NVIDIA device on this machine. Writing a block reduction nobody can run would
be exactly the plausible-looking work this project keeps finding, so the assumption the
source has always documented is now ENFORCED instead: a row wider than a warp is refused,
naming the width and what it would have returned. A row that fits in a warp lowers
unchanged.

`tile --list-ops` moved from `yes` to `-` for both ops on `gpu` as a result. That is the
report becoming true rather than the backend getting worse -- it could not lower those
rows before either, it just did not say so.

`argmax` was audited too and is correct: one thread scans the whole row serially. Slow,
but it is the row.

### The audit, completed: the same mistake in three backends

Scanning every backend's emitted `reduce_max` for the signature -- a subgroup/warp reduce
with no barrier and no fold -- found it three times:

| backend | what it emitted | now |
|---|---|---|
| msl | `simd_max`, lane 0 writes | **fixed and verified on the device** |
| gpu | `warp_reduce_max`, unused `__shared__` | refuses a row wider than a warp |
| spirv | `subgroupMax`, `p1[gl_SubgroupID] = val` | refuses a row wider than a subgroup |

SPIR-V is the worst of the three. It writes **one value per subgroup rather than per row**:
at `local_size_x = 256` with a subgroup of 32 it scatters eight partial maxima into
`p1[0..7]`, and a caller reading `p1[row]` gets subgroup 0's. And the same shape is in
`rms_norm`, which divides by `gl_SubgroupSize` -- so it normalised over 32 elements and
called that the row. rms_norm is not an exotic op.

32 is the bound because subgroup size is a DEVICE property -- 32 on most, 64 on AMD -- and
is not known at emit time, so only a row that fits the smallest common subgroup is safe.

Three of the emitters' own tests had to change: they used 256-wide rows and asserted the
kernels lowered. They were asserting the broken case works. Each now exercises a
subgroup-sized row AND asserts the wider one is refused.

Backends that do NOT have it: linalg and rvv reduce declaratively; aie, bang and gaudi
loop over the row; pico carries a barrier; argmax on Metal scans serially in one thread.
An audit that only ever finds problems is not an audit, so those are recorded too.

## A shared array sized at compile time, dispatched at run time (2026-09-02)

Found by pulling on the `parse_const_arg` trap from the specialization guard: if a value
that is not really known can be turned into a number, who *emits* that number?

`resolve_const` feeds `ctx.tile_width`, and the reduction kernels declared

    threadgroup float sdata[local_x];      // local_x = tile_width, an EMIT-time value

while the body writes

    sdata[tid] = tmax;                     // for every tid < tcount
    for (uint s = tcount/2; s > 0; s >>= 1)
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);

where `tcount` is `threads_per_threadgroup` — a RUN-time value the harness picks from the
run shape (`Shape::threads() = cols.clamp(1, 1024)`). Two independent sources for one
quantity. They agree whenever both come from the same kernel, which is why this survived:
the coupling was a coincidence, not an invariant.

A kernel emitted for a 64-wide tile and run over a 1024-wide row breaks it. Measured on
the M1 Ultra, before the fix:

    kernel vs torch 2.14.0     max abs 6.98e-2   max rel 1.78e1    rmse 2.56e-2
    ours   vs torch            max abs 1.86e-9   max rel 6.73e-7   rmse 6.65e-10
    verdict: the KERNEL disagrees with torch — look at the lowering

The three-way comparison did its job: our reference and torch agree to 6.7e-7, so the
disagreement is attributable to the kernel rather than to either reference.

`-O4` made it worse rather than catching it. The sweep runs one compiled source at
threadgroup widths 32…1024 on the argument that "the emitted kernels stride by
`threads_per_threadgroup`, so a different width is the same arithmetic in a different
number of steps" — true of the global loop, false of a fixed-size threadgroup array. It
measured 1024 as fastest and selected it: overrunning fastest is still fastest.

Plain `-r` reproduces it too, so this was never an autotuner bug — the base dispatch path
had it.

Fix: size the array by the dispatch ceiling rather than the tile width. `MAX_TG = 1024` is
the hard limit on `tcount` (the harness clamps to it and the autotuner's widest candidate
is 1024), so no dispatch this tool can issue overruns it. It costs 4KB of threadgroup
memory for f32, of 32KB available, and removes the coupling instead of documenting it.

    kernel vs torch 2.14.0     max abs 9.31e-10  max rel 6.11e-7   rmse 2.08e-10
    verdict: all three agree

Same case, and `-O4` still selects threadgroup 1024 at 17.33 us against 17.50 us before —
the correctness fix cost nothing measurable here.

`idx_data` was the same defect a few lines away, hardcoded at `[256]` instead of derived,
carrying the argmin half of the same reduction. Fixing only `sdata` would have left it
overrunning on exactly the dispatch that was just made safe.

Audited the rest: `scratch[64]` guards with `tid >= 64u`, `row_tmp[MAX_TOPK]` with
`tid >= top_k`, `ktg[128]` with `tid < 128u`, and `sel_scores[256]`/`idx[256]` with
`tid >= 256u`. Those four were written correctly; the two that were not are fixed.
Pinned device-free by `msl_threadgroup_bounds.rs`, so the invariant is checked on machines
with no GPU.

## The verdict blamed the reference for the kernel's error (2026-09-02)

Following the same question one more step: what else does the harness choose at run time
that the emitted kernel fixes at emit time? The grid.

`metal.swift` dispatches `MTLSize(width: grid, height: 1, depth: 1)` — one dimension. Seven
Metal kernels declare `uint2 tgpig [[ threadgroup_position_in_grid ]]` and index by both
components; the simdgroup matmul documents its own decomposition as
`grid = (ceil(N/8), ceil(M/8))`. Under a 1D dispatch `tgpig.y` is always 0, so every block
above the first row of the output is never written.

Running one produced this:

    kernel vs torch 2.14.0     max abs 4.04e1   max rel 1.00e0   rmse 1.71e1
    kernel vs ours             max abs 4.04e1   max rel 1.00e0   rmse 1.71e1
    ours   vs torch            max abs 9.54e-6  max rel 3.29e-4  rmse 1.74e-6
    verdict: this tool's own reference disagrees with torch too — the reference is
             wrong, and the kernel inherited it

The verdict is false, and it is the more serious of the two defects. The kernel is wrong by
100%. The reference sits 3.29e-4 from torch — ordinary f16 accumulation — and was named as
the cause. Anyone reading that line goes to audit the reference.

`attribute` took two of the three comparisons. It crossed "kernel within tolerance of
torch" with "ours within tolerance of torch" and called the (false, false) corner
inheritance. But "inherited" is a claim that the kernel matches the REFERENCE, and that is
the third comparison — computed, printed on the line directly above the verdict, and not
passed in. The one measurement that makes attribution possible was the one the attribution
could not see.

It now takes all three. When both differ from torch AND from each other, it says so
instead:

    verdict: all three disagree, and the kernel does not match the reference either —
             they are separate faults, so read the three rows above rather than this line

The 2D grid is refused rather than dispatched wrongly. The harness cannot infer an
intended 2D decomposition in general, so it declines and says which side is unable: "The
kernel is fine; the harness cannot infer the intended 2D decomposition."

That refusal exits 3, not 0. The old path printed "cannot read the emitted kernel's
signature" and returned OK on the reasoning that the conversion had succeeded and the file
was written. But the caller asked for `-r`. A CI step reading exit 0 from
`tile k.mlir -t msl -r` concludes the kernel ran and agreed with torch, which is exactly
the false report this tool exists to refuse.

## The other four ops the harness has references for (2026-09-02)

The tables above cover eleven unary ops. `RefOp` has fifteen variants, so four were never
in them: softmax, reduce_max, absmax, matmul. Run on the M1 Ultra:

| op | verdict |
|---|---|
| softmax | all three agree |
| reduce_max | all three agree |
| absmax | all three agree |
| matmul (f16) | disagrees — see below and backlog #010 |

That makes **14 of the 15 ops the harness can check** verified against both references, not
the "15 verified" this repo had been claiming (15 is how many it has references FOR) and
not the "thirteen" a heading claimed while the table below it listed eleven.

The matmul turned up a genuine defect: it accumulated in `half`. `float acc` took it from
6.46e-2 to 1.28e-2 max abs and 1.63e-2 to 4.38e-3 rmse. What remains is not the lowering
-- the harness rounds the kernel's inputs to half and computes both references on the f32
originals, so the two sides do not evaluate the same numbers. Filed as #010 rather than
papered over with a wider tolerance.

## reduce_sum: the last op msl did not lower (2026-09-02)

`tile --list-ops` had Metal at 17 of 18. The gap was `reduce_sum` — declared in `tile_std`,
lowered by seven other backends, and refused here rather than emitted as a copy:

    kernel @rsum uses a compute intrinsic (__tile_reduce_sum_f32) and this emitter has
    no arm for it, so it would have emitted a COPY of the first buffer instead

Written in the shape `reduce_max` had to be rewritten into after the miscompile: a strided
loop that visits every column at any threadgroup width, one partial per SIMD group, a
barrier, then a fold across the groups. `simd_sum` alone covers 32 lanes.

Verified on the M1 Ultra against both references:

    kernel vs torch 2.14.0     max abs 2.86e-6  max rel 2.26e-6  rmse 2.86e-6
    kernel vs ours             max abs 1.19e-6  max rel 9.43e-7  rmse 1.19e-6
    ours   vs torch            max abs 1.67e-6  max rel 1.32e-6  rmse 1.67e-6
    verdict: all three agree

And checked that the wrong version is catchable, which is the part the original reduction
bug taught. Emitting `simd_sum` with a lane-0 store — the miscompile this op could have
shipped — gives:

    kernel vs torch            max abs 4.96e0   max rel 3.93e0   rmse 4.96e0
    verdict: the KERNEL disagrees with torch — look at the lowering

4.96 against 2.86e-6, and the verdict names the kernel rather than the reference, which is
the attribution fix from earlier in the day doing its work on a live case.

**msl is now 18 of 18** — the first backend at full coverage on the probe. That is a
LOWERING count, not a correctness one, which is the distinction this whole document exists
to keep: `--list-ops` says what the emitters produce, and the fifteen verified ops are what
was run. With reduce_sum the harness now has sixteen references and fifteen of them agree
on hardware; the sixteenth is the f16 matmul of #010.

While writing the test: `test_msl_reduce_max` asserted only `msl.contains("simd_max")`,
which the BROKEN version satisfies too — it contained `simd_max`, reduced 32 lanes, and
stored lane 0's answer. Both reductions now assert the parts that tell the two apart: the
strided loop, the per-group partials, the barrier, and the cross-group fold.

## The probe was 18 of the 21 ops it could ask about (2026-09-03)

Having just made `--list-ops` state that its denominator is the probe's list rather than
the intrinsic surface, the obvious next question is whether the list is the right one.

The probe builds exactly one kernel shape: `__tile_<op>_f32(dst, src, rows, cols)`.
`tile_std` declares 21 intrinsics with that signature. The list held 18 — `cast_f16`,
`kv_cache_update` and `kv_cache_update_prefill` were absent, for no reason anyone recorded.
It was also called `PROBE_UNARY_OPS` while containing `argmax`, `transpose` and `softmax`.

Now defined by the rule instead of by taste: every intrinsic of that shape, all 21, and
named `PROBE_OPS`. What the three added show:

| op | backends that lower it |
|---|---|
| `cast_f16` | all eleven |
| `kv_cache_update` | linalg, rvv, msl |
| `kv_cache_update_prefill` | linalg, rvv, msl |

So the KV-cache pair was a gap in eight backends that the report simply never asked about.
That is the point of widening it: `--list-ops` can only be as honest as its questions.

msl stays at full marks, 21 of 21.

Adding the longer names broke the table — `kv_cache_updateyes`, the name running into the
first column, because the width was the constant 14 that happened to fit the old set. It is
computed from the longest name in the set now, which is the same fix as everything else this
week: derive it from the thing that knows rather than write it down beside.

## min, and the two-input path that made it checkable (2026-09-03)

Teaching the probe two-input ops surfaced an asymmetry nobody had noticed: Metal lowered
`max` and refused `min`. Nothing about that was deliberate — the arm was three lines and
had simply never been written.

Writing it was the small half. Neither `max` nor `min` had a REFERENCE, so lowering was all
that could be said about either. The harness's shapes each declare how many input buffers
they need, and every elementwise shape declared one:

    Shape::Rows      -> vec![rows * cols]
    Shape::Matmul    -> vec![m * k, k * n]

`Shape::Rows2` is the missing member: two tiles of one extent, one out. It is a variant
rather than a flag because `in_sizes` is what the harness allocates from and what the ABI
check compares against the kernel's buffer count — a two-input kernel driven by a one-input
shape now fails there, loudly, instead of reading a buffer nobody filled.

Verified on the M1 Ultra:

    min   kernel vs torch  max abs 0.00e0    max rel 0.00e0    (exact)
    max   kernel vs torch  max abs 9.31e-10  max rel 1.08e-7
    verdict: all three agree

Metal goes to 23 of 25 on the probe, and the harness has eighteen references with seventeen
verified — the odd one out still the f16 matmul of #010.

One thing worth recording rather than filing. On `min` the kernel matched torch EXACTLY
while differing from our own reference by 2.98e-8. The inputs are generated three times —
in Rust for the reference, in Swift for the device buffers, in Python for torch — and
`input_values_for`'s comment says they are "shared by construction rather than by two
implementations agreeing". They are not: Swift and Python evaluate the generator in double
and round once, Rust evaluates it in f32 throughout, so the Rust side sits about an ulp
away. It is inside tolerance everywhere and it is the same shape as #010, one level finer:
a comparison is only as good as the agreement about its inputs.

## matmul on Metal, and a tolerance the reference itself cannot meet (2026-09-03)

`__tile_matmul_f32` was the headline gap the matmul-shaped probe exposed: msl lowered the
f16 variants and refused the f32 one. The arm is the f16 emitter with `float` throughout,
so it carries neither the `half acc` precision trap nor #010's input rounding.

It lowers, and the run says something the earlier ops did not:

    kernel vs torch 2.14.0   max abs 1.14e-5  max rel 2.72e-4  rmse 2.40e-6
    kernel vs ours           max abs 1.14e-5  max rel 3.55e-4  rmse 2.66e-6
    ours   vs torch          max abs 9.54e-6  max rel 3.29e-4  rmse 1.74e-6

**Our own reference fails the tolerance against torch.** `ABS_TOL` is 1e-5 and `REL_TOL`
1e-4; the reference sits at 9.54e-6 and 3.29e-4 from torch on a K=64 product, computing the
same thing in a different summation order. So no matmul can pass this gate however correct
the kernel is, which is why matmul was never in the verified list and why "16 of 17" was
never going to become "17 of 17" by writing a better kernel.

That is #010's shape one level up: a fixed tolerance against an operation whose own
precision is not fixed. It is NOT closed by widening the number — that is fitting the gate
to the sample, rejected once already for f16 — so what changed is the sentence.

The verdict used to end "they are separate faults". That was an overreach in the opposite
direction from the "inherited" claim it replaced: here there are no faults at all, only
summation order, and the three magnitudes are all the same size. It now reads:

    all three disagree, including the kernel against the reference — compare the three
    magnitudes above: one much larger than the others is a defect, three of a size is the
    operation's own precision against this tolerance

which is the distinction a reader can actually act on, and the one the earlier findings
turned on: 17.8 against 6.7e-7 was a defect; 1.14e-5 against 9.54e-6 is arithmetic.

Metal reaches 24 of 25 on the probe. The one left is `matvec`, still a gap in every backend
but linalg and rvv.

## matvec: Metal reaches 25 of 25, via a bug the new verdict caught (2026-09-03)

The last gap. `__tile_matvec_f32` is `reduce_sum` with a multiply — a (rows, cols) matrix
against a (cols,) vector — so the emitter is the reduction shape again: strided loop,
per-SIMD-group partials, barrier, cross-group fold.

The first run was wrong, and wrong loudly:

    kernel vs torch   max abs 1.74e2   rmse 1.23e2
    ours   vs torch   max abs 3.05e-5  rmse 1.77e-5

One magnitude seven orders above the others — the reading the verdict was reworded to
name, on a defect introduced ten minutes after the rewording. `Shape::scalar` binds the
kernel's scalars by name, and its catch-all was:

    (Shape::RowReduce { cols, .. }, "num_elements") => cols,
    (_,                             "num_elements") => self.out_size(),

with a comment on the first line explaining that a reduction must bind `num_elements` to
the ROW WIDTH, because binding it to `out_size` "would make the kernel read `rows` elements
of a `rows * cols` row and reduce a sliver". Matvec is a reduction. Its `out_size` is
`rows`. The new shape fell into the catch-all and summed 4 terms of 256 — the exact failure
the comment describes, one shape later.

`num_elements` is now spelled out per shape with no `_` arm, so a shape added later has to
say what its kernels loop over rather than inherit an answer that happens to compile.
Fixed:

    kernel vs torch   max abs 1.53e-5  max rel 1.60e-7
    kernel vs ours    max abs 4.58e-5  max rel 3.20e-7
    ours   vs torch   max abs 3.05e-5  max rel 2.14e-7

Three of a size, and the reference is the middle one — a 256-term f32 dot product against
the same fixed tolerance matmul cannot meet. Arithmetic, not a fault, and the verdict says
so.

**msl is 25 of 25 on the probe.** The harness now has nineteen references and seventeen
that agree; the two that do not are matmul and matvec, both K-length reductions whose own
precision exceeds the fixed tolerance — not kernels anybody has shown to be wrong.

The refusal guard had to move a third time. It asserted msl refuses `matmul_f32`, then
`matvec_f32`; both were implemented within the hour. A test of the refusing DEFAULT ARM
should not be a hostage to coverage progress — each time its gap got filled the guard broke
and the easy move was to delete it. It now uses `__tile_frobnicate_f32`, a name nothing
declares and nobody can implement out from under it.

## The installed release binary, after all of it (2026-09-03)

Everything above was measured with `target/debug/tile`. The command the README gives is
`cargo install --path crates/tile_cli --features pico,stats`, which is a different build of
a crate that changed a great deal today, so it was run rather than assumed:

    install exit=0
    tile --list-ops       msl 25/25, unchanged from the debug build
    tile mv2.mlir -r      1.53e-5 / 4.58e-5 / 3.05e-5 — the same three figures

The numbers are identical because they are properties of the kernel and the references,
not of the optimizer level — which is the answer one wants and not one to take on faith.

The timing figures are NOT identical, and that is also right. The debug run prints

    WARNING: this is a debug build, so the reference is UNOPTIMIZED and every ratio
    above is inflated

and the release run does not; a grep for that warning finds it once in the debug binary's
output and zero times in the release binary's. The tool's claim about its own build is
true in both directions, which is a small thing to check and exactly the kind of
self-referential claim that goes stale unnoticed.

On the release build the matvec reports SLOWDOWN 0.18x — the device is 5.6x slower than an
optimized CPU loop on a 4-row problem. That is the honest number for that size, and the
report says so rather than quoting the debug build's flattering one.

## Every backend's softmax, read (2026-09-03)

The `reduce_sum` goldens were worth reading — they turned up #014 and #015. `softmax` is
frozen for sixteen backends and carries both a max-reduce and a sum-reduce, which is where
the miscompile this repo shipped once lived. So: all sixteen.

The question asked of each was not "does the text change with the width" — that check was
withdrawn in #015, because width-independent text is the RIGHT answer for a kernel that
takes its width at run time. The question is **where the kernel gets the extent it reduces
over**, and there are only three honest answers:

* **It names the operation.** linalg (`linalg.reduce`), rvv, nki (`nisa.tensor_reduce`),
  pto (`pto.trowsum`), ttmetal (`softmax_tile`), pico (a `vsum` opcode), aie. The extent
  belongs to the operand; a partial reduction is not expressible.
* **It takes the extent at run time.** msl (`num_elements` at dispatch — and verified on
  three GPUs), tpu (`jax.ShapeDtypeStruct(x0.shape)`), hexagon (an `int n` parameter handed
  to `hvx_softmax_f32`), csl (`@range(i16, 0, N, 1)`).
* **It derives its own sizes from the width.** CUDA and musa, and they are the model:

      cols=1024   <<<num_tiles, 1024>>>   __shared__ float sdata[32]   (tid < 32)
      cols=256    <<<num_tiles, 256>>>    __shared__ float sdata[8]    (tid <  8)

  Three interdependent numbers, all moving with the width. Nobody has to trust that kernel;
  you can read whether it tracks.

**One backend answers none of the three.** SPIR-V emits `local_size_x = 256` and
`shared float sdata[8]` at both widths, and reduces per WORKGROUP while indexing with
`gl_GlobalInvocationID`. A 1024-wide row is therefore four workgroups each normalising its
own 256 elements. Correct at 256 by accident of the width somebody tested; wrong above it.
Filed as #016, with a latent second problem — `barrier()` after a conditional `return`.

Sixteen kernels read, one defect. The ratio is the argument for freezing emitted text: none
of this needed hardware, and thirteen of these backends have none here.

## This machine has a Vulkan device, and the spirv softmax was wrong on it (2026-09-03)

#016 was filed as unfixable here for want of a Vulkan device. That was not checked. This
box has `glslangValidator`, `spirv-val`, the Vulkan loader and Mesa's Vulkan-on-Metal
driver:

    deviceName  Apple M1 Ultra
    driverName  KosmicKrisp (Mesa 26.2.0)
    apiVersion  1.4.354

So the kernel was measured instead of argued about.

Reading it again for the fix showed it was worse than filed — wrong at 256 as well as
above it:

    row_max = (tid < gl_NumSubgroups) ? sdata[tid] : -inf;
    row_max = subgroupMax(row_max);

Only subgroup 0 spans `tid < gl_NumSubgroups`. Every other subgroup reduces a full set of
`-inf`, takes `-inf` as the row maximum, and evaluates `exp(val + inf)`. That predicts a
number: 7 subgroups of 32 threads, so **224 of 1024 outputs non-finite**.

A 130-line compute harness (`assets/harness/vulkan.c`, not yet wired into `-r`) using the
same input generator as the Metal harness, `vkCmdDispatch(1,1,1)`, `num_elements = 1024`:

    old shader    max abs inf        non-finite 224 of 1024   sum inf
    fixed shader  max abs 7.229e-10  non-finite 0             sum 1.000000
                  max rel 4.689e-07

224, exactly as predicted from reading the source.

The fix is the shape `emit_reduce_sum_msl` uses and three GPUs have verified: one workgroup
per row, a strided loop, the fold broadcast through shared memory, no early return before a
barrier, and the dispatch contract stated in the emitted file the way CUDA states its
launch. Two baked assumptions went with it — `sdata` sized `local_x / 32` (Vulkan permits
subgroups of 4 to 64, and a narrower one overruns that array) is now one slot per thread.

**What this costs the backlog.** #001 says "no run harness for any target but Metal". That
is now true only of CUDA, Cambricon and Gaudi; SPIR-V has one, and it found a defect the
first time it ran. The `validated on Vulkan/MoltenVK` fidelity class that #012 questioned is
substantiated for this kernel and this driver, and for nothing else yet.

## SPIR-V joins Metal as a backend `-r` can measure (2026-09-03)

Finding a Vulkan device here made the harness worth wiring in rather than keeping as a
one-off script. `assets/harness/vulkan.c` now satisfies the same contract
`assets/harness/metal.swift` does — `#device`, `#threadgroup`, `#warmup`, `#us` per
iteration, the values, and the `#c` control arm — so `run.rs` parses one output format
whichever device it drove.

Three things had to become form-aware rather than Metal-shaped:

* `KernelAbi::parse` dispatches to `parse_msl` or the new `parse_glsl`. One entry point,
  because binding a Metal signature's buffer order onto a GLSL shader is how a harness ends
  up comparing two unrelated numbers.
* `harness_plan` names the tools each path needs — `swift`, or `glslangValidator` and
  `clang` — and a missing one is reported BY NAME rather than surfacing as an exec error,
  so "the run did not happen" cannot be mistaken for "the run disagreed".
* The dispatch width. Metal takes its threadgroup at dispatch, so `-O4` may sweep it;
  SPIR-V compiles `local_size_x` into the module, so the shader's width wins. `KernelAbi`
  carries `local_x: Option<usize>` to say which case a kernel is, and the harness is told a
  width it can actually honour rather than one that would make the reported threadgroup a
  fiction.

Every SPIR-V op the harness has a reference for, run on Mesa KosmicKrisp / Apple M1 Ultra:

| op | verdict |
|---|---|
| exp | all three agree |
| log | all three agree |
| rsqrt | all three agree |
| sigmoid | all three agree |
| silu | all three agree |
| softmax | all three agree |

`relu`, `sqrt` and `tanh` refuse — spirv scores 11 of 25 on the probe and those are honest
gaps, reported as refusals rather than as failures.

**A message that had gone false.** `-t spirv` printed "this is cross generation, and the
result cannot be run or measured here" — and then ran and measured it. The check compared
FAMILY names, and `vulkan` against `apple-gpu` looked disjoint; Mesa's driver puts them on
the same device. "Cannot be run here" is a claim about a harness, so it asks about the
harness now, and says which of the two facts holds instead of inferring the stronger from
the weaker.

## The SPIR-V reductions were refused for want of a device; now they run (2026-09-03)

`ReduceMax` and `Absmax` were made to refuse a row wider than 32 earlier in this work, with
the reasoning written into the source: the kernel wrote `p1[gl_SubgroupID] = subgroupMax(v)`
— one value per SUBGROUP into an output the caller indexes by row — and "there is no Vulkan
device here either, so the same rule: refuse rather than write a reduction nobody can run."

The premise was wrong. Both are rewritten in the shape `emit_reduce_sum_msl` uses, and both
were checked on Mesa KosmicKrisp / Apple M1 Ultra over a 256-wide row:

    reduce_max   all three agree
    absmax       all three agree

`RmsNorm` still refuses, and that is not an oversight: it has the same scatter shape and NO
reference in the harness, so a rewrite could be emitted but not checked. Fixing the two that
can be verified and leaving the one that cannot is the whole distinction this repo keeps
drawing.

spirv goes from 11 of 25 on the probe to 13, and from six verified ops to eight.

Two things the rewrite exposed:

* `needs_subgroup` listed `Softmax` alone, so the reductions emitted `subgroupMax` without
  requesting `GL_KHR_shader_subgroup_arithmetic` and glslang rejected the shader outright.
  That was invisible while they refused before reaching the compiler.
* `-O4` still called `parse_msl` directly, so `-t spirv -O4` reported "no `kernel void`
  entry point" — a Metal-shaped complaint about a GLSL file. It parses by form now, and
  when a kernel fixes its own workgroup it says there is nothing to sweep rather than
  measuring one width and calling it the best of a set of one.

## rms_norm: two errors in one line, both measured (2026-09-03)

`RmsNorm` was left refusing when reduce_max and absmax were fixed, on the grounds that it
had no reference in the harness to check a rewrite against. That is a reason to ADD the
reference, not to leave the kernel broken, so both were done.

The line it emitted was:

    float rms = inversesqrt(sum_sq / float(gl_SubgroupSize) + 1e-6);

with `sum_sq = subgroupAdd(sq)` above it. Two errors in one statement: the sum covered a
SUBGROUP rather than the row, and the divisor was the subgroup width rather than the row
width. A 256-wide row got eight different scales, each computed from a different 32 of its
elements.

Measured on Mesa KosmicKrisp / Apple M1 Ultra over a 256-wide row, against a CPU reference:

    old   max abs 7.940e-02
    new   max abs 5.784e-07

and through `-r`, with all three sources: **all three agree**, max abs 4.77e-7.

`RefOp::RmsNorm` and its torch expression are new — `x * rsqrt(mean(x^2) + eps)`, as
tile_std documents it. The harness now has twenty references.

**One thing left baked in, recorded rather than widened.** `__tile_rms_norm_f32` declares an
`eps` operand and the emitter ignores it, hardcoding 1e-6; the reference and the torch
expression match that constant so the comparison means something. It is the same shape as
#011 — an emitter given a value and using its own — and closing it needs the harness to pass
a float scalar, which it cannot do yet. Written down here rather than quietly papered over
by picking whatever eps made the numbers agree.

## eps stops being the emitter's number (2026-09-03)

`__tile_rms_norm_f32(dst, src, eps, rows, cols)` declares an eps operand. The SPIR-V emitter
hardcoded 1e-6 and ignored it, and — worse — the reference and the torch expression
hardcoded 1e-6 to match. Three constants agreeing with each other is not a check: change the
kernel's eps and the comparison would have gone on passing.

All three read it from the source now:

* The emitter resolves the operand through a float constant map. `const_map` takes the
  leading digits and stops, so `1.000000e-06` came back as `1` — floats needed their own map
  rather than a cast.
* `RefOp::apply`, `reference_output`, `compare_detailed` and `compare` take eps, read once in
  `run_kernel` by `Shape::rms_eps_from_mlir`, next to where the matmul dimensions are read
  from the same source for the same reason.
* The torch expression is formatted with it rather than being a static string.

A call whose eps is NOT a compile-time constant is refused, naming the two ways out. Giving
it a default would be a silent change to what the kernel computes, which is what was already
happening.

Checked with three values rather than one, because a single value cannot tell "uses eps"
from "happens to match":

    eps = 1e-3   all three agree
    eps = 1e-6   all three agree
    eps = 5e-2   all three agree

**The comparison caught my own half-finished change twice.** After threading eps through the
emitter and the Rust reference, `eps = 1e-3` reported "the KERNEL disagrees with torch" —
the torch expression still said 1e-6. Fixing that gave "the kernel matches torch but this
tool's reference does not": a blanket search-and-replace had put a literal `1e-6` into
`reference_output` as well as into the timing path where eps genuinely does not matter. Both
were found by running, not by reading.

The test fixture had `%e = llvm.mlir.constant(0 : i32)` passed where the declaration says
`f32`. It exercises the real signature now.

## SPIR-V: 13 of 25 to 21 of 25, every new op checked on the device (2026-09-03)

Being able to RUN a backend changes what may honestly be added to it. Eight arms went in —
`abs`, `neg`, `relu`, `sqrt`, `tanh`, `softplus`, `reduce_sum`, `min` — chosen because every
one has a reference in the harness, so none of them is a kernel written and hoped for.

All fifteen SPIR-V ops the harness can check now report **all three agree** on Mesa
KosmicKrisp / Apple M1 Ultra: exp, log, rsqrt, sigmoid, silu, softmax, abs, neg, relu, sqrt,
tanh, softplus, reduce_sum, reduce_max, absmax — plus `min` and `rms_norm`, seventeen in all.

`softplus` carries the same guard the CPU reference does — `v > 20.0 ? v : log(1+exp(v))` —
because above about 20 the formula IS `x` in f32 while `exp(x)` overflows to inf, and
without it the comparison would have been about f32 range rather than about the kernel.

**The extension list is derived now, not maintained.** `needs_subgroup` decided which
`#extension` lines to emit, and it said `Softmax` alone; it gained ReduceMax and Absmax, then
RmsNorm, then ReduceSum — four drifts in a day, each one emitting `subgroupAdd` without the
extension that provides it, which glslang rejects outright. The body is generated into a
buffer first and the flag is `body.contains("subgroup")`, so the set cannot drift from the
text it describes. That is the same rule the coverage figure and the op-count needed: derive
it from the thing that knows.

spirv is 21 of 25 on the probe. What is left is `cast_f16`-adjacent and the two
`kv_cache_update` variants, which no backend but linalg, rvv and msl lowers.

## matmul and matvec on SPIR-V, and the control arm earning its keep (2026-09-03)

spirv reaches **23 of 25**. The two left are the `kv_cache_update` pair, which only linalg,
rvv and msl lower.

Both new kernels are written one workgroup per output ROW rather than as the tiled f16 GEMM
already in this backend, which indexes with `gl_WorkGroupID.y` and needs a 2D dispatch. The
harness issues `vkCmdDispatch(grid, 1, 1)`. A 2D kernel could have been emitted here and
never run — which is the state this backend's reductions were already in, and the reason it
took a device to notice.

Measured on Mesa KosmicKrisp / Apple M1 Ultra:

    matmul   kernel vs torch 1.14e-5   kernel vs ours 1.14e-5   ours vs torch 9.54e-6
    matvec   kernel vs torch 1.53e-5   kernel vs ours 4.58e-5   ours vs torch 3.05e-5

Both are the #013 tolerance wall — three magnitudes of a size, which the verdict names as
the operation's own precision rather than a fault. And both sets are **identical to Metal's,
digit for digit**. Two independently written backends landing on the same numbers is a
better argument that each is right than either measurement alone.

**The control arm caught a bug in the harness I wrote.** The first matmul run reported:

    the control arm AGREED: dispatching over different inputs produced identical output
    across all 4096

`KernelAbi::parse_glsl` read the push-constant block from `{...}` on ONE line. Softmax emits
`{ uint num_elements; } pc;` on one line and parsed fine; matmul emits M, N and K on separate
lines, so no scalars were found, a single wrong value was pushed, `N` arrived as 0, and the
kernel's loop never executed. A kernel that writes nothing writes the same nothing whatever
it is given — which is exactly what the control arm exists to detect, and the only reason
this was a caught bug rather than a silent one. It parses multi-line blocks now.

## #013: a tolerance that is derived rather than chosen (2026-09-03)

`ABS_TOL = 1e-5` judged a 1024-term sum and a `relu` by the same number. An elementwise
op's error does not grow with anything; a K-term reduction's does, so no constant serves
both, and `matmul`, `matvec` and a 1024-wide `reduce_sum` could not pass — nor could this
tool's own CPU reference, which is how you know the gate had stopped measuring the kernel.

The issue called this a numerics decision. It is not, once the number stops being chosen:

    error_budget(op, shape, dtype) -> Option<Budget>

    magnitude  = max over outputs of the sum of the ABSOLUTE values of its terms
    guaranteed = (u_dtype + gamma_K) * magnitude,  gamma_K = Ku/(1 - Ku),  u = EPS/2
    typical    = (u_dtype + sqrt(K) * u) * magnitude

`guaranteed` is Wilkinson's bound: no summation order can exceed it, so a kernel outside
it is wrong whatever order it used. That is the property a fitted threshold never has —
#010's f16 threshold was fitted, and reverted for exactly this reason. `typical` is not a
gate; it is printed so that "admissible, but 8x above typical" — a needlessly bad
summation order — is visible rather than hidden inside a pass.

    matmul f32   msl and spirv   1.14e-5 against 3.12e-4    all three agree
    matvec f32   msl             1.53e-5 against 3.54e-4    all three agree
    reduce_sum   K=1024          1.14e-5 against 6.65e-2    all three agree

**The first version of this fix was wrong, in the way the tool exists to catch.** It used
f32's roundoff for every kernel. The msl f16 matmul errs by 2.18e-2 — correctly, because
`half * half` rounds each product before the f32 accumulator sees it — and the new tighter
gate promoted its verdict from a vague "all three disagree" to a confident "the KERNEL
disagrees with torch, look at the lowering". A tighter tolerance pointing confidently at
the wrong place is worse than the loose one it replaced. The budget takes the dtype now
(`u = 2^-11` for f16), and a kernel that accumulated in half would still be 10x outside it.

And #013 was three constants, not one: `RunReport.tolerance` was the literal `1e-5`, and
the exit code was `worst > 1e-4` — an order looser, so a kernel could be reported outside
tolerance and still exit 0. All three read the one budget.

## The msl rms_norm ignored the epsilon it was handed

Turned up by the same sweep, on a kernel whose MLIR says `1.000000e-03`:

    kernel vs torch   max abs 5.44e-4
    ours   vs torch   max abs 1.19e-7

Not a tolerance question — a factor of 4500. The emitted Metal read
`rsqrt(... + (float)1e-6)`: the constant scan stops at the first non-digit, so
`1.000000e-03` arrived as the integer 1, the lookup failed, and the emitter wrote its own
default. The kernel computed a different function from the one it was given, and nothing
in its output said so.

This is the SAME bug that was fixed in `mlir_to_spirv.rs` a day earlier. Fixing it there
did not prompt anyone to look here, and neither backend had a test — the spirv fix went in
without one. Both have one now, asserting the stated eps reaches the kernel and the
emitter's default does not survive it. Both backends now report 4.77e-7 on that kernel.

## The harness was wrong and the kernel was blamed

The sweep that verified #013 left one kernel disagreeing: `rn.mlir`, the FOUR-argument
`rms_norm`, at 3.70e-1 against a reference sitting 2.98e-8 from torch. By the rule this
tool teaches — one magnitude far larger than the others is a defect — that is a broken
kernel, and the verdict said so: *the KERNEL disagrees with torch, look at the lowering*.

The kernel was right. Diffing it against its five-argument twin, which agrees, showed the
two emitted files differ in exactly one character run: `1e-6` against `1e-3`.

`Shape::rms_eps_from_mlir` reads argument index 2 of the call. In the five-argument form
that is eps. In the four-argument form — `(tile, tile, rows, cols)` — it is the ROW COUNT,
whose constant is `1`. So both references computed `rsqrt(mean + 1.0)` while the kernel
computed `rsqrt(mean + 1e-6)`.

**This is the blind spot of a three-way comparison.** Its power comes from two independent
references agreeing; when the harness feeds both of them the same wrong parameter they
still agree, and the kernel — the only side that is right — is the outlier. No amount of
adding references fixes that. Two guards, both read from the call's own form: the
five-argument arity, and the constant's type, since an `i32` in the eps position is not an
epsilon whatever its position.

### And the general check the two eps bugs share

The epsilon is the one parameter the references are handed and the kernel is not: it is
baked into the emitted source at lowering time. So the two can differ silently. Within two
days that produced two bugs from opposite directions — the msl emitter ignoring a stated
eps, and the harness inventing one from the wrong operand.

`run::eps_reaches_the_kernel(src, eps)` now scans every float literal in the emitted source
and compares numerically, so `1e-3`, `0.001` and `1.000000e-03` all satisfy it. When the
number the references are using appears nowhere in the kernel, `-r` says so ABOVE the
accuracy figures and warns that the verdict may name the side that is right.

All 60 kernels in the scratch corpus now report "all three agree" on Metal.

## #014 and #015: the blocked fix was the OPTIMAL one, not the only one

Both were filed as unfixable here, on reasoning that was sound: no Cambricon or Habana
hardware, and this repo has twice rejected writing plausible kernels it cannot run.

Both were fixed here, without acquiring anything.

**bang (#014)** had three defects — a write-back copying `per_task * sizeof(float)` out of
a ONE-element NRAM buffer, an output count of a whole row where one number was due, and
each task summing its own slice with nothing combining them. The issue said they had to be
fixed in one edit or the result would be "differently wrong and look fixed".

The fix is not the MLU cross-task reduction the issue assumed was needed. **A row that fits
in NRAM does not need splitting.** When the kernel reduces, the prologue emits
`per_task = TILE_SIZE_X; offset = 0;`, every task reduces the whole row, and the write-back
becomes `if (taskId == 0) __memcpy(p1 + offset, nram_1, sizeof(float), NRAM2GDRAM);`.
Correct at any taskDim, no cross-task communication, all three defects at once. It
duplicates work across tasks — a performance cost, not a correctness one, and the emitted
source says so rather than leaving the next reader to infer it.

`absmax` went with them: it broadcast its result across the whole tile, so its output had a
different SHAPE from every other lowering of the same intrinsic, and from `RefOp::Absmax`,
which this repo folds to one value per row exactly like `reduce_max`.

**gaudi (#015)** was the elementwise template with a reduce in the middle: it stored inside
the loop and carried no accumulator, so each pass wrote one vector's sum at its own index.
The accumulator is hoisted above the loop and the single store moved below it. Every call
in the new shape is one this emitter already emitted — nothing was invented for a machine
that isn't here.

The test checks POSITIONS, not substrings: the accumulator must be declared before the loop
and the store must come after the loop's advance. A test that only looked for `_acc` would
pass on a kernel that still stored every iteration, which is the bug.

**The pattern worth keeping.** In both cases the blocked thing was the OPTIMAL fix, and a
correct one was available without it. "No hardware to verify a cross-task reduction" is
true, and it was allowed to stand for "no fix is possible", which it never implied. A slower
kernel that computes the right answer beats a fast one that does not, and unlike the fast
one it can be verified from here.

Still not claimed for either: that it runs. Both route lines print
`[exact, unvalidated on hardware]`, and that stays until someone with the device says
otherwise.

## #010: an f16 kernel is finally compared against the numbers it was given

`metal.swift` writes an f16 kernel's inputs as `Float16(v[i])` while both references used
the f32 originals. So part of every reported f16 gap was that rounding rather than the
lowering, and nothing separated the two.

All three sides round now: `quantize_inputs` on the Rust side, `.half().float()` in the
torch script, and the harness unchanged because it already did it. On the msl f16 matmul:

    before   kernel vs torch  2.18e-2 abs / 2.86e-1 rel      ours vs torch  9.54e-6
    after    kernel vs torch  1.56e-2 abs / 4.81e-4 rel      ours vs torch  0.00e0

**`ours vs torch` at exactly zero is the check on the fix** — the two references agree bit
for bit, which they can only do if they compute on the same numbers. The kernel's max
relative error fell by a factor of 600; that is how much of the old figure was the harness.

### A comment that asserted a fix nobody had made

`RunReport::render` said, for a day, that "the inputs are quantized at generation time now,
so both sides see identical values". `input_values_for` has no dtype parameter and never
had one. The sentence sat in the one place a reader would check before trusting an f16
number, and it contradicted — in the same file — the note that same function printed to the
user. A wrong comment is worse than no comment where it describes a guarantee someone would
otherwise go and verify.

### And the conversion had a real bug in it

`f16_round` is hand-written; there is no `f16` in stable Rust and this crate has no
dependencies. The subnormal reconstruction used an exponent one too low and **halved every
f16 subnormal**. It was caught by checking against numpy rather than by re-reading the
code — 598 values over the harness's own range plus the edges.

That sweep needs numpy and cannot live in the suite, so its equivalent does: every one of
the 63488 finite f16 bit patterns must be a fixed point of the rounding, each decoded
independently of the code under test. The halved-subnormal bug fails 2048 of them.

## #012: a fidelity class that says where it was measured

`route: mlir -> mlir -> msl [exact, validated on Apple GPU]` was printed on the last line of
every conversion, and by the design doc's own account it is "worth more than anything else
the tool prints". Nine backends carried `Validated`. This repo holds a run for three.

The other six came from projects that DO have the hardware, so those claims may well be
true — the issue was right that downgrading them here would replace a possibly-true claim
with a definitely-wrong one, and that it is not this repo's call which were really run.

So the claim is not changed. It is **attributed**:

    route: ... -> msl   [exact, validated on Apple GPU (docs/cli/INTEGRATION.md, 3 machines)]
    route: ... -> nki   [exact, validated on Trainium trn1 — inherited, no run recorded here]

`tile doctor --targets` was two groups with all nine validated backends in the first, under
"measured — emitted AND checked against a reference on that hardware". It is three now:
**measured HERE** (linalg, msl, spirv), **claimed UPSTREAM** (pto, gpu, nki, tpu, gaudi and
the second Ascend route), and **unmeasured**.

A multi-hop route degrades honestly: the hardware name comes from the last hop, but the
evidence is `Here` only if EVERY validated hop is. Taking the last hop's evidence would let
a route passing through an unattributed backend cite a file that says nothing about it.

**An evidence pointer must name a file that exists**, and a test opens every one — a
pointer that cannot be followed is worse than no pointer, because it looks checkable. That
test failed on its first run, on a path this change itself had got wrong:
`assets/harness/vulkan.c` is relative to the crate, not the repo root.

## Running the whole corpus on both backends, and what that caught (2026-09-03)

Every kernel in the scratch corpus, lowered to Metal AND to SPIR-V, verdicts compared:

    msl    60 of 60  all three agree
    spirv  58 of 60  all three agree; 2 refused as unmeasurable here

Three of the five initial mismatches were real defects, and one was mine.

**A regression I had introduced that morning.** Fixing the multi-line push-constant block
(matmul emits M, N and K on separate lines) broke the single-line form it replaced:
splitting the whole line on `;` makes the first field
`layout(push_constant) uniform PushConstants { uint num_elements`, which does not start
with `uint `. Single-line blocks yielded NOTHING, the harness fell back to pushing
`in_sizes[0]`, and matvec was handed **256 for a 64-wide row** and read off the end of it —
max rel 1.00e0, where the same kernel on Metal read 2.30e-7.

Softmax survived the whole time only because at one row `rows * cols == cols` and the wrong
number happened to be right. Neither form had a test; that is how one fix could silently
undo the other, and both are pinned now.

**A kernel the harness cannot drive was being measured anyway.** The pre-existing SPIR-V
f16 GEMM declares `local_size_y = 16`, indexes with `gl_WorkGroupID.y` and holds packed
halves in `uint p0[]`. The harness issues `vkCmdDispatch(grid, 1, 1)` and fills buffers
with f32, so it read f32 bits as pairs of halves across a grid it never got — and reported
*"the KERNEL disagrees with torch, look at the lowering"*. That is the third confident
wrong pointer in a day, all three from measuring something the harness could not measure.
`parse_glsl` refuses both shapes now and names which.

**And the four-argument `rms_norm` was refused on SPIR-V.** Same arity mistake as the
harness bug, from the opposite side: the emitter read operand 2 as eps regardless, saw the
row count, found no float constant and turned a good kernel away. It at least failed SAFE —
the harness made the identical mistake and took the row count as an eps of 1.0. The
four-argument form uses the documented default now; a *stated* eps that is not constant is
still refused, which is the refusal worth keeping.

The two that remain refused are the f16 GEMM pair, and refusal is the right answer for
them: this harness cannot dispatch them, and the alternative is a number describing neither
the kernel nor the reference.

## A shape sweep, and the three bugs it found (2026-09-03)

The corpus was 32, 64, 256, 1024 and 4096 wide — all powers of two — and every kernel in it
passed. 192 kernels were generated instead at widths chosen to break assumptions rather
than confirm them: 17, 127, 129, 1000, 1023, 4096, and matmuls at non-square shapes.

    before   msl 176/192   spirv 148/192
    after    msl 191/192   spirv 191/192

### The tree fold was correct only for powers of two

Six emitters in `mlir_to_msl.rs` wrote the textbook threadgroup reduction:

    for (uint s = tcount/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] = ...; }

The host picks the threadgroup from the row width, so a 127-wide row got `tcount = 127`,
the loop started at `s = 63`, and **`sdata[126]` was never read** — the row's last element
took no part in its own softmax. Softmax failed at 17, 127, 129, 1000 and 1023 and agreed
at every power of two, which is exactly the pattern the corpus could not show.

The fold halves by CEILING now and bounds against the LIVE count (`tid + s < n`), not the
original one — with `n = 63, s = 32`, index 31 is left alone this round and folded the
next, which a bound against `tcount` would get wrong in the other direction.

### Neither backend could cover a row wider than one workgroup

`msl` emitted `uint gid = base + tid; if (gid < num_elements) ...` — one element per
thread, with the threadgroup clamped at 1024. `spirv` emitted the same shape with
`local_size_x` compiled in at 256. So every unary op failed at 4096 wide on Metal, and at
1000 and beyond on Vulkan, with three quarters of the row never written and the harness
reading back whatever the buffer held.

Two independently written backends, the same wrong assumption, neither caught by a corpus
whose rows all fitted in one workgroup. Both stride now.

### A gate that required both bounds at once

`rsqrt` over a row containing a value near zero produces a huge result, so a relative error
of 1.64e-5 shows up as an absolute 6.7e-4. `within` demanded `max_abs <= 1e-5` AND
`max_rel <= 1e-4` — stricter than `numpy.allclose`, and **this tool's own reference failed
it against torch**, which is the same signal #013 gave for reductions. Each element is now
judged by the measure that means something for it: relative where the reference is large
enough for a ratio, absolute where it is not.

### What was NOT chased

`log` at 4096 wide still reports a disagreement, and it is not a defect. `log(x)` passes
through zero near x = 1, so a relative error against it is meaningless in exactly the way
the ratio floor exists to prevent; the kernel matches our reference to 9.5e-7 and our
reference differs from torch's `log` by 3.2e-5. That is one libm against another. Widening
a number to make it green is what #013 was filed about, and the verdict already describes
the situation accurately.

## #017: the harness drives the shape the source states

`tile.rs` built every non-matmul shape with `rows: 1` and `cols` set to the total element
count — literals, never read from the source. So a kernel written `%r = 3, %c = 129` was
driven as one row of 387, `base = row * num_elements` was always 0, and **no kernel in this
repo had ever had its row indexing exercised**. The reference was handed the same flattened
shape, so both sides agreed about a question the MLIR had not asked.

`Shape::rows_cols_from_mlir` reads `(rows, cols)` from the first `__tile_load_*`, the way
`matmul_from_mlir` already read its three dimensions — which is why matmul was the one
shape this never affected. When the source states no constant dimensions the old flattening
stays: correct for one row, and honest about not knowing.

Two things moved with it:

* **`num_elements` is the row WIDTH**, not the total. Kernels use it both as the row stride
  and as the per-row bound, so the total was wrong in two places at once. It was only ever
  right because `rows` was pinned to 1.
* **The SPIR-V elementwise path became row-aware.** It had just been given a flat
  grid-stride loop to fix the wide-row bug — correct for one row, wrong for several. It is
  one workgroup per row, striding within it, which fixes both.

### How the fix is known to have worked

Agreement could not show it. Before the change the kernel was flat too, so kernel and
reference agreed about the wrong problem; they agree about the right one now and the
verdict reads the same either way. What distinguishes them is that the two shapes are
DIFFERENT computations, and the test says so: a softmax over four rows of 256 is not a
softmax over one row of 1024, and `reference_output` must now return different numbers for
the two. If it did not, `rows` would not be reaching the reference.

The 192-kernel shape sweep is **191/192 on both backends** with the row counts real.

## An f16 sweep: the dtype axis the corpus barely covered (2026-09-03)

108 half-precision kernels across the same shapes. **29 of 108 agreed at the start, 81 at
the end**, and every step was a different kind of wrong.

### A whole kernel family had never been run

`exp`, `log`, `neg` and friends in f16 select a ggml-derived emitter: `char*` buffers with
the element type stated in a CAST inside the body, and byte strides (`ne0`, `nb_src`,
`nb_dst`) instead of an element count. The harness had no binding for those names, so it
refused — correctly, since a wrong stride reads the row at the wrong offset, but it meant
that family had never executed. `ne0` is the row width and `nb_*` are that width in bytes;
both are now bound, with the element size taken from the kernel's dtype.

### The dtype was read from a declaration that does not carry it

Those buffers are `device const char*`, so `parse_msl` reported `float` and the harness
filled them with f32 while the kernel read pairs of halves. `exp` came back wrong by
**5.85e4**. The type is stated in the cast — `(device const half *)(p0 + row * nb_src)` —
so that is where it is read from now, and a `char*` buffer with no recognisable cast is
refused rather than guessed at.

### A relative gate tighter than the format

With the data right, `exp` still "disagreed" at 4.76e-4 relative. f16's unit roundoff is
**2⁻¹¹ = 4.88e-4**: a result stored in half cannot be closer than that however good the
lowering is, and `REL_TOL` was 1e-4 — five times tighter than the format allows. This is
#013 in the dtype dimension and takes the same answer: the bound is derived from the
format, not chosen. f32's roundoff is 5.96e-8, far below `REL_TOL`, so the f32 path is
unchanged.

### The control arm fired on a correct kernel

`absmax` and `reduce_max` in f16 reported *"the control arm AGREED: dispatching over
different inputs produced identical output"*. The kernel was right. The generator is
periodic in `(i + seed) % 17`, so reseeding **permutes** the row — and a max over 17 or
more elements sees the same set. The only difference comes from the `i * 1e-4` ramp, which
at f16 precision near 2.25 is below one ulp, so the two dispatches really did produce
identical bytes.

The control arm's input now scales and offsets as well as reseeding, in both harnesses, so
every value moves by far more than any format's ulp. The range stays modest so `exp` does
not overflow f16.

### The same wide-row bug, in the family that had never run

`if (tid >= ne0) return;` — one element per thread, threadgroup clamped at 1024, so a
4096-wide row lost three quarters of itself. The third place today this exact assumption
turned up, and the only one that could not be found before because the harness could not
dispatch these kernels at all.

### What is NOT claimed

`softmax` and `rms_norm` in f16 still report a difference — 2.85e-3 and 5.93e-4 relative,
against references that match torch to 1.2e-7. They reduce, but their error is not
`gamma_K * sum|terms|`, so `error_budget` deliberately excludes them and **this tool has no
derived bound for them in half**.

The verdict used to say "the KERNEL disagrees with torch — look at the lowering". It does
not any more: pointing at the lowering is a confident claim, and it must not be printed
where there is no bound to justify it. It now says the kernel differs from both references
by its own f16 arithmetic, that no bound exists for this operation in half, and gives the
reader 2⁻¹¹ as the scale to read the figures against. That is the third time today a
confident pointer had to be withdrawn, and all three were the same mistake: measuring
something the tool could not measure.

## Two dtypes, two backends, 600 measurements (2026-09-03)

The f16 sweep had run on Metal only. Running it on Vulkan too, and adding the two-input
elementwise shapes that had never been swept at all:

    f32   msl 191/192   spirv 191/192
    f16   msl 107/108   spirv 107/108      (0/108 on spirv when this started)

### SPIR-V could lower f16 and had never compiled it

`-t spirv` on an f16 kernel wrote a shader to disk and reported a successful lowering. The
shader did not compile. Nothing noticed because the harness refused `float16_t` buffers at
the signature check, so glslang was never asked.

Once the Vulkan harness learned to fill and read half buffers, three faults surfaced at
once:

* `max(v, 0.0)` with `v` a `float16_t` — GLSL will not mix the two, so **f16 `relu` had
  never compiled**. Widening to float and narrowing on the store makes every literal in
  every expression a non-question, which is what the Metal f16 family already did.
* `subgroupAdd(float16_t)` needs `GL_EXT_shader_subgroup_extended_types_float16`, and
  `max(float, float16_t)` is ambiguous. Every f16 reduction accumulates in float now — more
  accurate, and it needs no optional extension.
* **Every reduction was a one-thread workgroup.** `tile_width` was taken from the STORE,
  and a row reduction stores one value per row, so `local_size_x = 1` and `shared sdata[1]`.
  In f32 that was merely serial; in f16 it also produced the `subgroupAdd` glslang refused.

### Then the two backends disagreed, which is the point of having two

With SPIR-V's f16 softmax and rms_norm widened to float, they **agreed with both
references** — while Metal's still did not. Metal was summing 256 exponentials in `half`,
and its rms_norm narrowed through a `threadgroup half rms_shared`, a `half rms` scale, and
a vectorised `dot(half4, half4)` branch that a row takes whenever its width is a multiple
of 4. That last one is why `3x129` agreed while `1x256` and `1x4096` did not — a split no
amount of reading the code would have suggested looking for.

None of this was found by inspection. It was found because one backend was fixed for an
unrelated reason and the other stopped matching it.

### And a fourth place with the same wide-row assumption

The two-input elementwise family indexes `gid = row * tcount + tid` — block indexing, so
one workgroup over a 4096-wide row wrote only the first 1024 elements. Six emitters, the
fourth distinct prologue form in this file to carry the same assumption, each found in a
different sweep.

### Two ops were missing an arm, five characters each

`__tile_sqrt_f16` was absent from Metal's sqrt arm and `__tile_log_f16` from SPIR-V's log
arm, while their siblings name both dtypes on one line. Neither was deliberate; both were
invisible until a sweep asked for every op at every dtype.

### What is still not claimed

`log` at 4096 wide in f32, on both backends: one libm against another, not a defect, and
deliberately not chased. `softmax` in f16 at 4096 wide on Metal: the outputs are near
2.4e-4 with a wide spread, so the smallest fall into f16 subnormals where relative
precision degrades — the report says the tool has no derived bound for softmax in half and
gives 2⁻¹¹ as the scale to read against, rather than naming the lowering.

## The optimizer changes no result, and a near-miss on a documented ABI (2026-09-03)

### 192 kernels × three optimizer levels

Every kernel in the shape corpus, run at `-O0`, `-O2` and `-O4` and compared:

    191 of 192 identical at all three levels

The one exception is `log` at 4096 wide, which is *equally* non-agreeing at every level —
the libm difference already recorded, not an optimizer divergence. **No kernel's verdict
changes with the optimizer**, which is the property an optimizer has to have and which
nothing here had checked.

### The mixed-precision matvec had never been measured either

`Shape::Matvec` could not bind `M`, `K` or `N`, so a kernel taking a GEMM-shaped signature
was refused — correctly, since an unbound scalar would be passed as 0, but the shape knows
these numbers: a matvec IS a matmul with N = 1. With that bound, the mixed-precision matvec
(f32 activation, f16 weights, f32 output) verifies at 1x64, 4x256 and 7x129.

### And a near-miss worth recording

Writing an f16 matvec for the sweep, I read `__tile_matvec_f16` as "matvec with f16
buffers" and called it with all-f16 loads. The emitted kernel then read the matrix as the
vector and returned after row 0 — it agreed at 1x64 by coincidence, one row, and was wrong
at every other shape.

The conclusion I was about to draw was that two emitters of the same intrinsic disagreed
about operand order, and the fix was to route the tile-level intrinsic to the tile-level
shape. **That would have broken a documented ABI.** `__tile_matvec_f16` is the
mixed-precision kernel: f32 activation in p0, f16 weights in p1, f32 output in p2, stated
on its `KernelType` and pinned by `test_msl_matvec_f16_cooperative_reduction`, which
asserts each buffer's type by name.

The malformed thing was my caller. The pinned test is what said so — the change compiled,
the sweep went green, and only the assertion `"p0 (activation) must be float*"` stood
between a plausible-looking fix and a broken contract. Called the way its ABI actually
reads, that kernel agrees at every shape.

That is the fourth time in this session a confident conclusion had to be withdrawn, and the
first where the thing that caught it was a test somebody had written earlier rather than a
measurement taken later.

## -O4 recommended a dispatch width without checking the answer (2026-09-03)

`-O4` is the MEASURED level: it sweeps threadgroup widths on the real device, picks the
fastest, and writes it into the emitted source as a dispatch recommendation.

It never read the output buffer. The harness's sweep mode times each width and `exit(0)`s
before printing a value, so the ranking was on speed alone.

**That is the wrong test for this axis.** Threadgroup width is the parameter every
reduction in this repo folds across, and the power-of-two tree fold produced *wrong answers*
at thread counts that were not powers of two. A sweep that only times would have found such
a width fastest and recommended it, in a comment, in the artifact the host code is written
from.

Each candidate is now re-run through the ordinary verified comparison at its own width
(`run::verify_at_width`, which calls the same `compare_detailed` the single run uses, so the
two cannot drift). A width that disagrees is dropped and says why; if none survive, no width
is recommended at all — a fast configuration that computes the wrong answer is worth less
than no recommendation.

An op with no reference cannot be checked. That does not silently remove the
recommendation, it qualifies it: the widths are still ranked, with a note that the ranking
rests on timing alone.

**Honestly stated: this dropped nothing on the current corpus.** The sweep's candidates are
all powers of two, and the tree-fold bug needed a thread count that was not — so `-O4`
would not have caught that particular defect either. What changed is that the recommendation
now has a correctness precondition instead of none.

### And -O4 was tuning a shape the kernel is not verified under

The `-O4` path built its own `Shape` with `rows: 1` and `cols` = the total element count —
the flattening #017 removed from the `-r` path and only from there. So the level that tunes
the dispatch geometry was using a different geometry from the level that verifies it. Both
read the shape from the source now.

## Reading the backends nobody here can run, at the shapes that broke the ones we can

The shape and dtype sweeps found bugs on Metal and Vulkan. The same shapes were then
emitted for the fourteen backends with no harness, and the text read — the technique that
found #014 and #015, now aimed by knowing exactly which shapes break kernels.

What came back is mostly reassuring, and the reasons are worth recording:

| backend | at a 4096-wide row | why |
|---|---|---|
| `linalg`, `rvv`, `nki`, `tpu`, `pico` | correct by construction | they NAME the reduction (`linalg.reduce`, `nisa.tensor_reduce`, `jnp.sum`, a single `vsum` opcode) and the width belongs to the operand |
| `bang`, `gaudi` | correct | fixed in this session (#014, #015) |
| `gpu`, `musa` | **refuse** | "this emitter reduces within one block"; the CUDA block-reduce lesson holding |
| `pto` | gated | needs a feature not built here, and says so |
| `aie` | one latent issue, below | hand-rolled |

An emitter that cannot express a partial reduction is worth more than one that expresses it
correctly, and five of the fourteen are in that category by construction.

### The one finding: `aie` could silently drop a tail

    TILE_WIDTH   = 129
    PROBLEM_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 387
    N_TILES      = PROBLEM_SIZE // TILE_WIDTH

`PROBLEM_SIZE` is overridable at run time and `N_TILES` is integer division. At the emitted
default it is exact — 387 // 129 = 3, and multi-row is handled correctly. Pass anything
that is not a multiple and the remainder is never processed: a partial answer with a
successful exit, which is the shape of defect this repo keeps finding, in a backend nothing
here can run.

The emitted source now asserts the precondition and names the consequence:

    assert PROBLEM_SIZE % TILE_WIDTH == 0, (
        f"PROBLEM_SIZE {PROBLEM_SIZE} is not a multiple of TILE_WIDTH "
        f"{TILE_WIDTH}: N_TILES truncates and the last "
        f"{PROBLEM_SIZE % TILE_WIDTH} elements would never be processed."
    )

Emitted as an assertion rather than fixed in the emitter, because how a remainder tile
SHOULD be handled is a question for whoever owns the AIE dataflow. Refusing loudly is the
part that can be decided from here.

Verified by executing the emitted preamble: it accepts 129 and 387, and refuses 130 and 200
naming the count that would be lost.

## The sweep is in the repo now, not in the session that wrote it

Everything the shape and dtype sweeps found — a threadgroup fold correct only for powers of
two, two backends unable to cover a row wider than one workgroup, a fourth prologue with
the same assumption, an accuracy gate stricter than `numpy.allclose`, and `rows` never
leaving 1 — came from generated kernels that lived in a scratch directory. The findings
were written down; the ability to find them again was not.

`crates/tile_cli/tests/shape_sweep.rs` generates the corpus and runs it:

    TILE_SWEEP=1 cargo test --test shape_sweep -- --nocapture
    shape_sweep: 550 of 552 measured comparisons agree

Twelve shapes chosen to break assumptions (17, 127, 1000, 1023, 4096, and row counts above
1), twelve ops, two dtypes, two backends. It needs a GPU and a torch, so it is gated behind
`TILE_SWEEP=1` and skips with a note otherwise; `TILE_SWEEP_TARGETS` narrows it.

Three things it deliberately does NOT do:

* **It does not pin a count.** Asserting "191 of 192" fails the day someone adds an op,
  which makes adding ops expensive and teaches people to delete the test. It asserts that
  the ops with a reference all agree, and that the sweep actually ran (`ran > 100`) so a
  configuration error cannot pass as a clean sweep.
* **It does not count "no derived bound" as a failure.** `softmax` in f16 at 4096 wide is
  the only case today, on BOTH backends and by the same amount — two independently written
  kernels deviating identically, which is evidence about f16 subnormals rather than about
  either lowering. Calling it a failure would pressure someone into widening a number to
  make the suite green, which is what #013 was filed about. It is listed by name, and the
  list is capped so a new one cannot appear unnoticed.
* **It does not treat a refusal as a failure.** A kernel this harness cannot drive is
  skipped, because refusing is the answer this repo prefers to a plausible number.

The first version of the classifier got the second point wrong: the "no derived bound"
verdict spans several lines and it only matched the first, so the two honest cases were
reported as defects. Fixed by matching what the verdict's first line actually says.

## The sweep now covers the families that found the other half of the bugs

`shape_sweep.rs` covered row-shaped ops only. But the two-input family had its own
block-indexing prologue (`gid = row * tcount + tid`, six emitters, only the first `tcount`
elements of a wide row written), and matvec's `M`/`K`/`N` could not be bound at all, which
is why the mixed-precision matvec had never been measured on any backend. Neither was in
the codified sweep, so both could regress silently.

    TILE_SWEEP=1 cargo test --test shape_sweep -- --nocapture
    shape_sweep: 606 of 608 measured comparisons agree

Added: `min`/`max` at five shapes in both dtypes, `matmul` at four non-square shapes, and
`matvec` at four. The two that do not agree are still the same `softmax` f16 at 4096 wide
on each backend, reported honestly rather than counted.

### And the -O4 verification is in the spec now

The check that a threadgroup width must be CORRECT before it can be fastest was code with
no scenario. It is four steps in `04_profile_and_optimize.feature` now: that each candidate
is checked before ranking, that a disagreeing width is dropped with its reason, that no
width is recommended when none reproduces the reference, and that an op with no reference
leaves the ranking *qualified* rather than removed.

Two of those assert against the source text rather than by producing a wrong kernel.
Demonstrating the drop would mean shipping a width-sensitive emitter to test the guard,
which trades a real defect for a test; what is checked instead is that the guard's
explanation exists and says why, because a guard whose explanation has drifted is one
nobody will act on.

### The coverage doc has two counters and they disagreed

Adding the scenario made `the_committed_coverage_doc_counts_the_scenarios_that_exist` fail
with "the doc claims 23, the file has 24" — correct, and it named the fix. But regenerating
produced 23 as well: the generator counts scenarios it can EXECUTE, the checker counts what
is in the file. A scenario with no step definitions is invisible to one and visible to the
other.

That is the same two-counters-of-one-thing shape as the drifting figures elsewhere in this
project, and it resolved itself the right way: the scenario needed implementing, not
tagging. Spec coverage is **175 of 180 (97%)**, up from 174 of 179.

Worth recording: regenerating the doc without the full configuration first produced
"170 of 179 (94%)" and I nearly committed it. The doc's own opening warns that a number
quoted without its configuration is not reproducible, and I had just proved it.

## `ok` is not the answer to a question you did not ask (#009)

`tile <kernel> -i` printed this, two lines apart:

    tile plan: not computable (no declared extent)
    bounds (n/a): UB ok; repeat ok; DMA stride ok;

Three capacity checks answering `ok`, directly under the admission that there was no tile
to check. They ran against a DEFAULT tile size. #009 says it plainly — "the worst of the
three possible answers: a planner reading this concludes the tiling is fine".

There are three answers and the third one now exists:

    bounds (n/a): UNKNOWN — nothing to check
      the source declares no extent, so there is no tile size to check a capacity against
      unknown is not ok: a check with no input has not passed, and a tiling
      that was never examined must not read as one that was

Two tests, deliberately opposed: no declared extent must give `UNKNOWN` and never `UB ok`;
a source WITH an extent must still be `Checked`. The second exists because a change making
everything answer "unknown" would satisfy the first and destroy the feature — the same
reason the sweep asserts `ran > 100`.

#009 stays open. Its substance is two Ascend cost models that need the reader #004 asks
for; this was the one line in it that needed neither hardware nor a reader.

## Emitted source that nothing compiles is source nobody has checked

The SPIR-V f16 path made this concrete: `-t spirv` on an f16 kernel wrote a shader to disk
and reported a successful lowering, and the shader did not compile. `max(v, 0.0)` on a
`float16_t` is a type error, so f16 `relu` had **never once compiled** — invisible because
the harness refused `float16_t` buffers at the signature check, so glslang was never asked.

So: which of the sixteen emitted forms is checked by anything at all?

| form | checked by | when |
|---|---|---|
| `msl` | `xcrun metal` | every `-r` run and every golden |
| `spirv` | `glslangValidator` | every `-r` run |
| `aie`, `nki`, `tpu` | **nothing, until now** | they emit Python |
| the vendor C backends | nothing | `cncc`, `mcc`, `nvcc`, the TPC compiler are not on this machine |

The Python three have no excuse: Python parses with no vendor SDK at all.
`crates/tile_cli/tests/emitted_parses.rs` emits each of them at five shapes chosen to
break kernels and runs `ast.parse` over the result. 645 emissions across the whole corpus
parsed before the test was written, so it found nothing today — the point is that a
backend which starts emitting invalid Python tomorrow can no longer do it silently, which
is precisely what SPIR-V did for as long as nobody compiled its f16 output.

It guards against the vacuous pass the same way the sweep does: if every backend started
refusing, `checked` would be 0 and every file would trivially parse, so the test asserts a
floor on how many were emitted at all. And a refusal is not a failure — only text that WAS
emitted has to be a program.

This is a syntax gate, not a semantic one. It cannot say whether a kernel is correct, only
that it is a program. That is exactly the check the f16 `relu` needed: it was not subtly
wrong, it did not parse.

## The CUDA backends had never been read by a compiler, and one of them wasn't a program

`nvcc` is not on this machine, so nothing had ever compiled `gpu` or `musa` output. But
CUDA is close enough to C++ that a stub header — `__global__` defined away, `threadIdx` a
struct, `__shfl_down_sync` an identity, `__half` a float wrapper — lets clang answer the
only question that matters here: **is this a program?**

    270 of 277 emitted CUDA files parse; 7 fail, all one bug

The `gpu` matmul arm allocated a result variable, registered it as the matmul's output, and
then emitted **only a comment**:

    float _v0 = p0[goff];
    float _v1 = p1[goff];
    // TODO: matmul 16x16x16 — use cublasSgemm(_v0, _v1, out)
    p2[goff] = _v2;

`_v2` is never declared. The kernel does not compile. And `tile --list-ops` reported
`matmul: yes` for `gpu` and `musa` throughout — **the coverage matrix was counting a file
that is not a program as a lowering.** That is #005 and #006 in a place neither had looked:
#006's detector renames the intrinsic and asks whether the output changed, and a TODO
comment *naming* the intrinsic changes when it is renamed.

It refuses now, naming what a real one would need. `gpu` and `musa` drop from 11/25 to
10/25, which is the truth.

### The test was pinning the broken behaviour

`test_gpu_matmul_f16` asserted the output contains `__global__ void tile_matmul_f16` and
`matmul 32x64x32`. Both were satisfied by a kernel that does not compile: the first is the
emitted kernel named after the MLIR function in the fixture, and the second was matching
the text of the TODO comment. It asserts the refusal now.

### And most of the first number was my own instrument

The first run reported **194 of 308 failing**. 154 of those were a missing `musa_runtime.h`
in my stub and ~21 were `rsqrtf`, a real CUDA intrinsic the stub didn't define. Only 19 were
the emitter. Reporting "194 broken kernels" would have been the same error this tool exists
to catch — a measurement whose instrument is the thing that is broken — so the stub was
completed first and the number recomputed.

Both gates are now `tests/emitted_parses.rs`, with the stub written into the test rather
than kept as an asset so it cannot drift from the code that uses it. Its doc comment says
what the failure mode is: forgetting an intrinsic produces a FALSE ALARM, which is why the
first run looked catastrophic.

## Two more backends read by a compiler for the first time, and gaudi wasn't a program either

`cncc` and the TPC compiler are not on this machine, so nothing had ever read `bang` or
`gaudi` output. Both are close enough to C++ that a stub header reaches them.

    bang    229 of 229 parse
    gaudi   208 of 208 parse   (190 of 220 before the fix below)

**`bang` is clean.** Its `TILE_SIZE_K` is properly `#define`d, and every kernel compiles as
C++ against a stub of `__mlu_entry__`, `__nram__`, `__memcpy` and the `__bang_*` family.

**`gaudi` was not.** Its two scalar-loop arms emitted

    float256 v_0 = v_f32_ld_tnsr(index, input0);
    for (int _li = 0; _li < 256; _li++) {
        v_1[_li] = logf(v_0[_li]);      // v_1 is never declared
    }
    v_f32_st_tnsr(index, output0, v_1);

`log` and `rsqrt`, both writing into a destination nobody declares — and `--list-ops`
reported `log: yes` and `rsqrt: yes` for this backend throughout. **The identical defect to
the CUDA matmul's undeclared `_v2`, in a different backend, found the same way.** Every
other arm in that file declares its intermediates; these two were simply written without it.

### Stub discipline, twice more

The gaudi check first reported 127 of 220 failing. Five causes, four of them mine:
`short512` and `half` undefined, `float256` with no subscript operator, and a
float/vector assignment my stub's types rejected. Rewriting the stub around ONE universal
value type — convertible to and from float, subscriptable, arithmetic-combining — removed
all four without guessing a single real signature. 30 failures remained; 24 were `v_1`.

`void main(tensor, tensor)` is TPC-C's legitimate entry point and C++ reserves `main`, so
the check renames it in a copy before parsing. That is my compiler's restriction, not the
emitter's fault, and it is worth saying which is which.

### The count so far

Of sixteen emitted forms, seven are now read by something: `msl` and `spirv` by real
compilers, `aie`/`nki`/`tpu` by a Python parser, `gpu`/`musa` and `bang`/`gaudi` by clang
with stubs. Three of those seven were emitting text that was not a program, and each was
invisible for exactly as long as nobody asked a compiler.

### The vendor-C gates are in the repo now, stubs and all

The bang and gaudi findings came from stub headers in the session scratchpad — which dies
with the session, so the check was not durable even though the fix was. Both stubs live
inside `tests/emitted_parses.rs` now, beside the CUDA one, for the same reason: a stub kept
as an asset can drift from the test that uses it.

    cargo test --test emitted_parses -- --nocapture
    -> 45 Python, 22 CUDA, 50 vendor-C (bang + gaudi) files parsed

Codifying it introduced one more instrument bug, which is worth recording because it is the
fourth of exactly this kind. `bang` emits `#include <bang.h>` itself, so force-including the
stub as well gave two paths to the same text, `#pragma once` did not dedupe them, and every
bang kernel failed on a redefinition **the harness had caused**. 25 failures, none of them
the emitter's. The stub goes in the include path under the name the emitter asks for, and
only `gaudi` — which includes nothing — is force-included.

## Every emitted form is now read by something

`hexagon` and `ttmetal` include vendor headers by path; stubbing those paths lets clang
read them. `linalg`, `rvv` and `pto` emit MLIR and `mlir-opt` is not here — but SSA has one
invariant that needs no compiler: **every `%value` used must be defined earlier.**

That is not an arbitrary check. It is exactly the defect found three times today in
backends that DO have compilers — the CUDA matmul writing an undeclared `_v2`, and gaudi's
`log` and `rsqrt` writing an undeclared `v_1`. A backend nobody can compile is the likeliest
place for a fourth.

    hexagon + ttmetal   140 of 140 parse
    linalg + rvv        416 of 416 with no use-before-def

Both clean. All five gates are in `tests/emitted_parses.rs`:

    cargo test --test emitted_parses -- --nocapture
    -> 45 Python, 22 CUDA, 50 vendor-C, 20 header-including, 30 MLIR

| form | read by |
|---|---|
| `msl`, `spirv` | real compilers, on every `-r` run |
| `aie`, `nki`, `tpu` | `ast.parse` |
| `gpu`, `musa`, `bang`, `gaudi` | clang + an inline stub |
| `hexagon`, `ttmetal` | clang + stubbed header paths |
| `linalg`, `rvv`, `pto` | an SSA use-before-def check |
| `csl`, `pico` | still nothing — CSL is Zig-like and pico emits its own assembly |

Fourteen of sixteen. Three of them were emitting text that was not a program.

### Five rounds of fixing the instrument

The hexagon/ttmetal number went 0 → 0 → 112 → 121 → 140 of 140, and **every one of those
jumps was a fix to my stub, not to an emitter**: missing header paths, `#pragma once`
failing to dedupe across copies written to several paths, `mm_init` absent, `matmul_tiles`
stubbed only in the singular. Before that, `bang` failed 25 of 25 because the stub was both
force-included and included by name.

That is five false alarms in one afternoon from the same cause, and the reason each was
caught is that a uniform result — everything failing, or everything failing the same way —
is a broken instrument rather than a discovery about independently written emitters. The
same tell that exposed `rtk proxy diff` reporting every backend identical.

The stubs live inside the test, with a shared `#ifndef` guard rather than `#pragma once`,
and the doc comments say which failures were the harness's.

## All sixteen. Every emitted form is checked by something

`csl` and `pico` were the last two, and neither has a compiler here.

**CSL** is Zig-like — clang will not parse it — but it declares its names explicitly
(`param X:`, `var X:`, `task X()`, loop captures `|i|`), so the same invariant that reached
MLIR reaches it: every identifier used must be declared, and the delimiters must balance.
70 emissions clean.

**pico** emits its own assembly listing and the assembler is not here either. Three
invariants it has anyway: the embedded `@manifest` is JSON that downstream tooling parses,
**one mnemonic carries one opcode** (a mnemonic with two is a table bug), and the program
ends with `end`. Checked against the committed goldens, since the emitter is feature-gated.

Seven gates, `cargo test --test emitted_parses`:

    45 Python, 22 CUDA, 50 vendor-C, 20 header-including, 30 MLIR, 10 CSL, 2 pico listings

| form | read by | how |
|---|---|---|
| `msl`, `spirv` | real compilers | every `-r` run |
| `aie`, `nki`, `tpu` | `ast.parse` | Python needs no vendor SDK |
| `gpu`, `musa`, `bang`, `gaudi` | clang | inline stub headers |
| `hexagon`, `ttmetal` | clang | stubbed header paths |
| `linalg`, `rvv`, `pto` | SSA use-before-def | no compiler needed |
| `csl` | declared-name check | no compiler exists here |
| `pico` | listing self-consistency | no assembler exists here |

**Three of the sixteen were emitting text that was not a program** — the SPIR-V f16 path,
the CUDA matmul, and gaudi's `log`/`rsqrt` — and each was invisible for exactly as long as
nobody asked.

### The instrument was wrong six times

Every one of these checks reported a confident failure that was its own fault first:

| what it claimed | what it was |
|---|---|
| 194 of 308 CUDA files broken | 175 were stub gaps (`musa_runtime.h`, `rsqrtf`) |
| 127 of 220 gaudi files broken | four of five causes were stub type gaps |
| 25 of 25 bang files broken | the stub was force-included AND included by name |
| 140 of 140 hexagon/ttmetal broken | missing header paths, then `#pragma once` not deduping |
| 19 ttmetal matmuls broken | `matmul_tiles` stubbed only in the singular |
| 5 CSL identifiers undeclared | my own string surgery mangled `compute_task` |

Six false alarms, one afternoon, one cause: **a measurement whose instrument is the broken
thing.** Each was caught by the same tell — a uniform result is a broken instrument, not a
discovery about independently written emitters — the tell that first exposed
`rtk proxy diff` reporting every backend identical.

That tell is the most reusable thing in this document.

## The -O4 guard now has something to catch, and it was proved to catch it

`-O4`'s correctness guard was added and dropped nothing, which is a weak place to leave a
safety check. The reason was structural: the sweep's candidate widths were **powers of two
only**, and the threadgroup fold that produced wrong answers did so only at counts that are
NOT powers of two. The guard could never have had a width to drop.

The candidate list is `32, 64, 96, 128, 192, 256, 384, 512, 768, 1024` now — the multiples
of the 32-wide SIMD group between the powers of two. That is real tuning space, not just
test material: on an M1 Ultra, 384 and 768 measure within 10% of the best.

**And the guard was then proved to work, rather than assumed to.** With the fold
temporarily reverted to its old power-of-two form, `-O4` on a 1024-wide softmax reported:

    threadgroup 96: DROPPED — it computes a different answer at this width
    threadgroup 192: DROPPED …
    threadgroup 384: DROPPED …
    threadgroup 768: DROPPED …
    6 of 10 widths verified; the rest are excluded from the ranking.
    best: threadgroup 1024 at 17.88 us

Exactly the four non-power-of-two widths, exactly the six powers of two kept, and a correct
recommendation still made. That is the precise signature of that defect and nothing else.
The fold was restored immediately and all ten verify again.

Two things worth separating. The guard existing is not the same as the guard working, and
the difference was one experiment. And a safety check that has never fired is not
reassuring — it is untested, and the honest response is to arrange for it to have something
to catch rather than to record that it found nothing.

## Emulating a backend nobody can run, and what it found

Every check so far has been SYNTAX. The handoff named the gap: nothing verifies what a
backend *computes* except the two with a device.

The stubs that make `bang` parse can be made to **emulate** instead of no-op — `__memcpy`
as `memcpy`, `taskId = 0`, the `__bang_*` family implemented for real. The emitted MLU
kernel is then a plain C function this machine can RUN, against the same input generator
and the same reference the device path uses.

    emulated bang exp:  1.353352815e-01   reference: 1.353352832e-01

A 2e-9 f32 difference. **That is semantic verification of a backend with no hardware here.**

### It found that bang reduces the whole TILE, not each row

`reduce_sum`, `reduce_max`, `absmax` and `softmax` all used `per_task`, which is
`TILE_SIZE` — rows*cols. A 4x256 kernel summed all 1024 elements and wrote **one** number
where four were due:

    emulated  4.587758636e+01  0  0  0
    reference 1.2640e+00  8.0676e+00  1.4871e+01  2.1675e+01

Identical to correct when rows == 1, which is all the run harness ever dispatched before
#017. Worse, **the #014 fix that introduced this shape says "reduces the WHOLE row" while
reducing the whole tile** — I wrote that comment, and it was wrong in the same way the code
was. All four now loop over rows; the write-back stores `rows` values rather than one.

### Two more instrument bugs, and one of them inflated a real finding

The first emulation sweep reported 87 of 96 and the second 115 of 165. Both were wrong:

* `${base%%_*}` splits at the FIRST underscore, so `reduce_sum_4x256` became op `reduce`,
  which has no reference — those kernels were **silently skipped**. The sweep measured less
  than it claimed while reporting a clean-looking ratio.
* The second sweep dropped the `rm -f` between iterations, so a REFUSED emission left the
  previous kernel's file behind and it was compared against the wrong op's reference.
  `relu`, `tanh`, `sqrt` and `softplus` are refused by `bang`; 48 of the 50 "failures" were
  that.

Seven and eight. The lesson has not changed and the tell has not either: a failure that
lands on every shape of one op, or a ratio that looks too clean, is the instrument.

### What this does and does not establish

The emulator runs the emitted C with `__memcpy` as a memcpy and one task. It therefore
checks the ARITHMETIC and the indexing — which is where all four defects were — and says
nothing about DMA behaviour, bank conflicts, multi-task scheduling, or whether `cncc`
accepts the source. A Cambricon engineer would still have to run it. But "four values were
expected and one was written" needed no Cambricon at all.

### The emulator is in the repo now, and proved to bite

It lived in the session scratchpad — the finding was durable, the tool that found it was
not, which is the exact gap `HANDOFF.md` exists to close.

    cargo test --test emulate_bang -- --nocapture
    -> emulate_bang: 35 kernels run and compared, 5 refused

Both stub and driver are inside `tests/emulate_bang.rs`, and the reference comes from
`run::reference_output` — the same one the Metal and Vulkan paths use, so the three cannot
drift.

**Proved to bite, not assumed to.** With `reduce_sum` temporarily reverted to its
whole-tile form, the test failed on exactly the three multi-row shapes and passed both
single-row ones:

    reduce_sum 4x256: max rel 7.32e27
    reduce_sum 3x129: max rel 1.12e0
    reduce_sum 8x64:  max rel 1.91e27
    3 emulated kernels disagree with the reference. This is not a syntax complaint:
    the kernel ran and computed a different answer.

That signature — multi-row fails, single-row passes — is the defect and nothing else. Same
discipline as the `-O4` guard: a check that has never fired is untested, and one
experiment is what separates "the check exists" from "the check works".

It guards the vacuous pass too: if `bang` started refusing everything, nothing would be
compared and the test would pass by checking nothing, so it asserts a floor on how many
kernels actually ran.

## gaudi could not be emulated, and finding out why was the finding

`bang` was emulated because its emitted C is arithmetic a stub can implement. `gaudi` was
the obvious next candidate — and it is not emulatable, for a reason worth having.

Its kernels are index-space based: `v_f32_ld_tnsr(index, input0)` loads a 256-lane vector
at `index`, and the loop steps `index[0]` by `get_index_space_stride()[0]`. To emulate
that, one has to know whether `index[0]` counts elements or vectors, and what the stride
returns. **The emitter documents the loop's shape and not its units.**

Guessing would have produced a ninth false alarm. The units are the HOST's, and no amount
of reading this repo settles them.

But the ambiguity is not harmless, because a reduction here accumulates across the whole
index space and stores **one** value. Whether that is one value per row or one for the
entire tensor is decided entirely by a convention stated nowhere — **which is the exact
defect `bang` had**, where the reduction covered the whole tile and wrote one number where
`rows` were due, and survived two fixes including one whose own comment said it reduced
"the WHOLE row".

So the emitted source states the contract now, as `msl` and `spirv` already do:

    // DISPATCH CONTRACT -- the host owns the index space and this kernel
    // cannot check it:
    //   * one iteration processes ONE 256-lane vector at `index`;
    //   * stepping index[0] by get_index_space_stride()[0] must visit every vector
    //     of the operand exactly once.
    //   * THIS IS A REDUCTION: it accumulates across the whole index space and
    //     stores ONE value. For a 4x256 tensor the host must therefore
    //     dispatch ONE INDEX SPACE PER ROW -- 4 of them, each covering 256
    //     elements. An index space covering the whole tensor reduces across rows
    //     and writes one number where 4 are due.

An elementwise kernel gets the weaker contract and does not claim the stronger one; a
contract that overstates is as useless as none, and the test checks both directions.

**What was decidable from here was not the answer but the question.** The host owns the
index space; this repo cannot fix that. It can refuse to leave the requirement implicit,
which is what let the same defect hide in `bang` through two attempts to fix it.

## The same distinction, in the last two backends: width-independent for a good reason, or a bad one

The handoff guessed that `hexagon` and `ttmetal` were "probably statable, not emulatable".
Checking rather than leaving it a guess found something sharper.

Both emit **byte-identical text** for a 256-element tensor and a 4096-element one. #015 is
exactly about not reading that as one thing:

* **`hexagon` is width-independent for a GOOD reason.** `execute_k(in0, out, n)` takes its
  length at run time, the way `msl` reads `num_elements` at dispatch. Identical text at
  every shape is correct here. What was unstated is what `n` MEANS for more than one row —
  a row-wise op called once with `n = rows*cols` spans the rows.
* **`ttmetal` is width-independent for a BAD reason.** It takes no length at all:
  `cb_wait_front(cb_in0, 1)` waits for exactly one 32x32 tile and pushes one. A
  4096-element tensor is four tiles and needs four invocations that nothing mentioned.

Both refuse `reduce_sum` outright, which is the honest answer and was already right.

The contracts are in the emitted source now:

    // CALLER CONTRACT -- this kernel takes its length at run time and
    // cannot check it:
    //   * `n` is the element count of ONE ROW, not of the tensor;
    //   * a rows x cols tensor needs ONE CALL PER ROW …

    // HOST CONTRACT -- this kernel processes exactly ONE 32x32 tile per
    // invocation and takes no length parameter:
    //   * a tensor of N elements needs ceil(N / 1024) invocations …
    //   * for a ROW-WISE op the tiling must not split a row across tiles,
    //     or each tile is normalised against its own fragment.

### Where this leaves the sixteen

Four backends now state a dispatch contract in the emitted source — `msl`, `spirv`,
`gaudi`, `hexagon`, `ttmetal` — because in each the correctness of the kernel depends on
something the host does and the kernel cannot check. That is not a documentation exercise:
the identical ambiguity was a REAL defect in `bang`, which reduced the whole tile and wrote
one number where `rows` were due, and which survived two fixes including one whose own
comment said it reduced "the WHOLE row".

The pattern the session keeps returning to: **when a kernel cannot check something, the
next best thing is to say so where the person who can check it will read it.**

## Checking what the MLIR backends DECLARE, with no toolchain at all

`linalg`, `rvv` and `pto` emit MLIR and `mlir-opt` is not here. The SSA check says no value
is used before it is defined — structure. This says meaning.

It is possible only because these backends NAME the operation instead of hand-rolling it:
`linalg.reduce … dimensions = [1]` with an `arith.addf` body, rather than a loop. That is
the same property which makes them unable to express the partial-reduction bug `bang` and
`gaudi` both had.

**But the property cuts both ways.** A named operation cannot reduce the wrong NUMBER of
elements — and can perfectly well name the wrong OPERATION or the wrong DIMENSION:

* `arith.addf` where `arith.maximumf` was meant is a `reduce_max` that sums;
* `dimensions = [0]` instead of `[1]` reduces ACROSS rows rather than along each — the bang
  defect arriving by a different route.

Neither a parser, nor a use-before-def check, nor a syntax gate would notice either.

    cargo test --test declared_semantics -- --nocapture
    -> declared_semantics: 80 emissions checked

Ten ops across four shapes and two backends, each checked for the primitive it must name,
`dimensions = [1]` where it reduces, and a declared `tensor<rows x cols x f32>` matching
what was asked. All clean.

### Proved to bite — after a first attempt that proved nothing

Swapping the operator in what I took to be the reduction emitter changed nothing: the gate
still passed. **The edit had landed on `UnaryKind::Relu`, not the reduction** — so the gate
was never given a wrong `reduce_max` to catch, and the proof attempt was invalid rather
than the gate being weak. Worth separating: "the check did not fire" and "the check cannot
fire" look identical from the outside and are not the same claim.

Targeting the real one — `ReduceKind::Max | ReduceKind::AbsMax => (…, "arith.addf")` —
failed `reduce_max` and `absmax` on every shape of both backends while every other op
passed. That is a module which parses, defines every value it uses, and computes a sum
where a maximum was asked for.

## The compiler was on this machine the whole time, and `topk` did not compile

`xcrun metal` is installed here. Two goldens and the `-r` path invoked it; **274 KernelType
variants and 23 MLIR fixtures did not.**

`topk` — a kernel with a fixture in this repo and a passing test — does not compile:

    error: read-only variable is not assignable
        p1[out_base + k] = -MAXFLOAT;

Its three buffers are `(input, out_values, out_indices)`, which its own `KernelType`
comment states, so the **last two** are written. Only the last was declared writable, and
the kernel assigned through a `device const` pointer. The mechanism for this already
existed — `last_two_writable` — and `TopK` simply was not in it.

**Its test passed throughout.** It converts the MLIR and greps the result for an idiom, and
a string containing the right idiom is still a string. That is exactly the gap that hid an
f16 `relu` in the SPIR-V backend which had never once compiled, and a CUDA matmul that
wrote an undeclared variable.

    cargo test --test emitted_parses -- --nocapture
    -> emitted_parses: 30 Metal kernels compiled

Fifty emissions across the ops and shapes now compile clean with the real compiler, and the
gate is in the suite. The regression test additionally asserts that `topk` still WRITES to
`p1`, so it cannot pass by the kernel having quietly stopped writing.

### What this says about the other 274

The gate covers what a `__tile_*` intrinsic can reach. The DS4 families — 97 of the 274
variants — are reached by their own intrinsics with their own arities, and constructing
those by hand is exactly the guessing that produced eight false alarms today. **That
surface remains uncompiled, and it is now the largest unchecked thing in this repo.** It is
recorded in the handoff rather than estimated: the honest statement is that nobody knows
whether those kernels compile, and finding out needs their real call signatures rather than
my reconstruction of them.
