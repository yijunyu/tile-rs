# The `lowering` cluster, and what it forced

`tile backlog` stopped on three open issues in one area. This is what reading them
together produced, recorded because the point of the rule is to change something, not to
note that a threshold was crossed.

## The three

* **#003** — no lifter exists for any target-source form.
* **#004** — cannot read AscendC, so a kernel that already exists is outside the tool.
* **#005** — an unhandled intrinsic emits a plausible copy instead of refusing.

The first two are obviously one problem. The third looks unrelated, and is not.

## One design problem

**tile-rs models lowering as a total function when it is partial**, in both directions at
once:

* **Partial in the domain.** An emitter's dispatch ends in an arm that produces output.
  An intrinsic with no arm therefore yields a kernel that compiles, passes the emit-purity
  check, and computes something else — #005. The generality matrix cannot catch this: it
  only ever feeds canonical snippets, so every arm it exercises is one that exists.
* **Partial in direction.** Nineteen forms are destinations and three are sources, so
  every kernel that already exists in a target language is outside the tool — #003, #004.

Both show up as the same user experience: *tile answered, and the answer was not a
lowering of what I gave it.* In #005 it emitted a copy; in #003/#004 it emitted a refusal
whose shape implies the pair is impossible rather than unwritten.

## What changed

**The partiality is now checkable, from inside `tile_cli`.** `emit::ignored_intrinsics`
emits twice — once as written, once with the intrinsic's *name* replaced by one no emitter
can handle. Everything else about the module is identical, so if the output does not
change, the emitter never looked at that operation. A conversion that hits this now says
so and exits 1:

```
tile: IGNORED: __tile_matmul_f32 appears 1 time(s) and changed nothing in the output
tile: the msl emitter produced output without using 1 of this kernel's operations.
  What it wrote is not a lowering of what you gave it.
```

This deliberately does **not** edit the emitters. They live in a tree this crate
`#[path]`-includes and does not own, and a refusing default arm there is the right fix but
belongs to `rustc_codegen_tile` — where it should also come with a test that feeds a
deliberately unhandled intrinsic, which is the thing that was missing.

## What is still open, and why it is not a fourth patch

Reading remains a decision rather than a bug: either commit to one reader — `mlir` is the
obvious one, and `.cce`/`.h` is what people actually have on disk — or stop implying
symmetry the tool does not have. `--list-forms` already prints "tile-rs reads 3 forms and
writes 19", which is the honest version; the docs should not say "swissknife" without it.

#003 and #004 stay open because that decision has not been made. #005's *detector* is
built, so the class cannot recur silently; its *fix* stays open with the emitters.

## Evidence from the PICO session (2026-09-01), which reframes the cluster

`tilers-pico` was asked whether `pico_lift` refutes the claim above that no lifter exists.
The answer was more useful than a yes or no, and it came with a better diagnosis of what
these four issues have in common.

**`pico_lift` is not a decompiler, and a general lifter does not follow from it.** It
splits a `.om` into programs, computes a *signature* — the multiset of vector/cube
intrinsics — and matches that against a fixed recipe catalog to recover a `tile_std`
kernel. So it is a signature lookup, not source recovery. It works only because PICO's
target language **is** the intrinsic set: a closed 91-symbol alphabet with an
op→intrinsic map small enough to invert by table. AscendC has no equivalent, because its
source is a C++ program rather than an alphabet — which is exactly why #004 is harder
than #003 and not merely another instance of it.

The inverse is also known-ambiguous, and documented as such on their side:
`vvmax` is produced by both `max(a,b)` and `relu(x)`; `vsemad` by `scale`, `adds` and
`fill`. Lift therefore returns **candidates**, and anything reasoning backwards from an
instruction mix has to carry that rather than pick one. A "lifter" that returned a single
answer here would be wrong in a way nobody could see.

**The transferable part is not the lifter — it is the validation shape.**

    lift -> lower -> compare instruction mixes

That is a compiler self-check which does not require the lifter to be complete *or*
unambiguous: if tile-rs can both read and write a form, it can compare signatures across a
round trip and catch a wrong lowering, without ever claiming to have recovered the source.
It is a much smaller target than #003 and it would have caught #005 — a copy kernel has
the wrong instruction mix by construction. Worth pricing before the general reader.

**Their diagnosis of the cluster, which is better than the one above.** tile-rs models a
target as ONE form. A real backend is a *stack* of them — op program, physical plan,
instruction words, container — with different owners and different validity conditions at
each level. #003 and #004 are the read direction of that; #008 is the write direction
(`.pico.s` stops at the op program, and nothing in `tile` turns it into instruction
words). On that reading the cluster is not "reading is unwritten" but "a form is the wrong
granularity", and the open question becomes **what a form should be** before a fifth issue
lands in the area.

Two concrete constraints they hit that any writer at the lower levels must respect:

* `instr_size` must be 16-byte aligned — an EVEN instruction-word count. The loader
  rejects an odd one *before running*, with `Error, instr_size(10232) should be 16 bytes
  aligned`, and nothing in the container hints at it. If `tile` ever writes a `.om`, it
  needs the same guard.
* Of svp's 13 native schedules, only 2 are shape-parameterised; the other 11 replay an
  envelope qualified from one specific container, and anything the emitter does not
  overwrite keeps *that* container's constants. The distinction between a **derived** and
  a **qualified** schedule is the one to build in from the start rather than retrofit,
  because binding a qualified one silently imports its model into tile-rs.

  **Corrected by measurement, and the correction is the important part.** The PICO
  session measured this after sending it, and "11 of 13 import their model" was half
  wrong:

      gemm       distance 0   5 models, 4 shapes, 32/32 varying words derived
      conv124    distance 0   8 instances, 3 shapes, 49/49
      nop70      distance 2   28 instances,        41/42
      eltwise87  distance 4   3 instances, 3 shapes, 30/34

  `conv124` **is** an envelope replay, qualified from YOLOv8s -- and it derives all 49
  varying positions on RT-DETR, a model it never saw. So "replays a recipe" does not
  imply "imports its model". Only measurement separates the two, and the source form
  does not predict it.

  The design consequence for tile-rs: if this distinction is built in, the label must be
  **UNMEASURED** rather than a verdict, with a measured distance of 0 across several
  shapes promoting it to derived. That is an evidence ladder, the same shape as
  `HardwareParams.measured` refusing to answer bound queries until someone has measured
  the chip -- and it is the invariant this crate already inherited. Calling an unmeasured
  schedule "qualified" because of how it was written would be the same error as approving
  a tiling from another chip's capacities: a plausible label standing in for a
  measurement, in the one place where being wrong is invisible.

  (The two schedules that miss, miss for the same reason -- `ldgp.xraddr` unclassified in
  both. One field, not two bugs.)

This also corrects a caveat I sent them: `-O3` does have PICO equivalents that need no
device — `PICO_SIM_CMD` with libinstsim for cycles, `validate_reemittable`/`audit_*` for
structural word and event counts, and `pico_transform::gates()` for six necessary
conditions on a rewritten stream.


## The rule that came out of this, now enforced in `-r`

Two independent findings turned out to be one shape, and the general form is worth stating
because it is not about matmul or about PICO:

> Every comparison run must include one arm that MUST disagree.
> If it agrees, the harness is broken, not the kernel.

Mine: `RunReport.checked` was recorded and never rendered, and `compare` maxes the error
over the values that came back -- so an empty result set gives zero error and the verdict
line reads "all three agree". Theirs: a board rig returned the previous model's output
*and timings* for a model that had failed to load, so two variants "agreed with baseline"
while one of them had never executed. Absence of evidence rendering as evidence of
agreement, one level apart.

The Metal harness now re-fills its inputs from a different seed, dispatches once more, and
prints the result as `#c` lines; `run::check_control` refuses the run if that output is
identical. Verified both ways: a Metal kernel that writes a constant and ignores its
inputs produces byte-identical control output over all 4096 elements and is refused, while
the real f16 matmul passes. The refusal says it is a harness fault rather than a numerical
one, because that is the distinction the number alone cannot make.

---

# The second `lowering` cluster (2026-09-03)

The area crossed the limit again, with three that have nothing to do with the first three.
Recorded the same way, because the rule asks for an analysis and not for a longer list.

## The three

* **#011** — two Metal specializations bake in a shape that is not an operand.
* **#014** — bang's row reductions write a whole row and read past a one-element buffer.
* **#015** — gaudi's row reduction reuses the elementwise loop and never accumulates.
* **#016** — spirv's softmax reduces per workgroup, so a row wider than 256 is normalised
  in chunks.

## One design problem

**An emitter bakes in a size, or takes it from the wrong place, and nothing compares it
with the size it was handed.**

Each is that sentence with a different noun:

* #011 bakes in a head dimension. The intrinsic passes `dk` as an operand and the emitter
  never reads it. Caught because the probe asked; fixed for four kernels by
  `reject_specialization_mismatch`, which does the comparison and refuses on a proven
  mismatch.
* #015 reuses the elementwise loop for a reduction: it stores inside the loop with no
  accumulator across iterations, so each pass writes one vector's sum rather than the row's.
  Whether that is wrong turns on an index-space configuration the emitter does not produce,
  so it is filed as a question. (The first version of this bullet said the two widths gave
  byte-identical output. They do not — they differ in a comment — and the check that
  produced that claim was broken; see the corollary below.)
* #014 bakes in the wrong size relationship: it copies `per_task * sizeof(float)` out of a
  one-element buffer, sizing the write-back from the INPUT extent when a reduction's output
  extent is one.
* #016 bakes in a workgroup size. `local_size_x = 256` and `shared float sdata[8]` are the
  same at cols=256 and cols=1024, while the reduction they serve is per-workgroup — so
  above 256 the row is split across workgroups and each normalises its own chunk. It is
  correct at exactly the width somebody tested, which is the Metal reduction's story again.

The failure they share is not "hardcoding". A specialization is a legitimate thing to
write, and `emit_flash_attn_ext_vec_score_msl` is allowed to be a head-dim-64 kernel. The
failure is that the baked-in number and the given number never meet, so the mismatch has no
place to be noticed. Every one of these emitters HAS the operand it should be checking
against.

## The remedy exists and is demonstrated

`reject_specialization_mismatch` is the shape: resolve the operand, compare it with what
the emitter assumes, refuse a proven mismatch, and leave an unprovable one alone. It runs
between classification and emission, it refuses only what it can demonstrate, and it is
pinned by tests that include the case where the operand is dynamic and nothing may be
concluded.

What #014, #015 and #016 need is not a new idea. It is that same check plus hardware to
verify the corrected kernel on, and this machine has no Cambricon card, no Gaudi and no
Vulkan device (#001). Writing the reductions blind is the thing this repo has now declined
four times, for CUDA, for the f16 tolerance, for gaudi and for spirv.

CUDA is the counter-example that makes the rule concrete rather than pious. Its softmax
emits `<<<num_tiles, 1024>>>`, `sdata[32]` and `(tid < 32)` at one width, and
`<<<num_tiles, 256>>>`, `sdata[8]`, `(tid < 8)` at another. Three interdependent numbers,
all derived from the width it was handed. Nobody has to trust that kernel; you can read
whether it tracks.

## The rule this cluster produces

**A number an emitter assumes must be compared with the operand that carries it, in the
emitter, at emit time.** Where the operand does not exist the assumption belongs in the
refusal message rather than in silence — which is the open half of #011, whose remaining
two kernels take byte strides and no head dimension at all.

A corollary I proposed here and then had to withdraw, kept because the withdrawal is the
lesson: *emit a reduction at two row widths and diff the output; identical text means a
hardcoded width.* It does not. `msl` and `tpu` are identical at both widths and both are
right — msl reads `num_elements` at dispatch, tpu takes its shape from the input array at
run time. Width-independent TEXT is the correct answer for a kernel that obtains its width
at run time, and the diff cannot tell that from a baked-in constant.

Worse, I ran it as `rtk proxy diff`, and `diff` is one of rtk's own subcommands rather than
the diff program — it printed nothing and exited 0 on every pair, including `linalg`, whose
two outputs differ in plain sight. `/usr/bin/diff` gets both right. A check that passes
everything is not evidence, and the uniformity should have been the tell before the
conclusion was.

What survives is the question, not the test: **where does this kernel get the extent it
reduces over?** An operand, an input shape, a dispatch scalar — or a literal. That is read
from the emitter, not from a diff of its output.

And there is a check that does work, which found #016: **emit at two widths and compare the
kernel's OWN declared sizes** — its launch, its shared array, its guard bound. Not the whole
text, which is noise; the numbers the kernel states about itself. CUDA's move together and
SPIR-V's do not, and one `grep -oE "local_size_x = [0-9]+"` separates them.

## The cluster dissolved (2026-09-03)

Down to one open issue in this area, so `tile backlog` no longer stops on it. What closed
the two most recent ones is worth keeping, because both were filed as *cannot be fixed
here* and both were fixed here — without acquiring any hardware.

* **#014 (bang)** — the fix was not to write the MLU cross-task reduction the issue said
  was needed. It was to notice that **a row that fits in NRAM does not need splitting**:
  every task reduces the whole row and task 0 stores. Correct at any taskDim, no cross-task
  communication, and it settles the over-read, the output count and the uncombined partials
  in one edit, which is what the issue said had to happen together.
* **#015 (gaudi)** — the accumulator hoisted above the loop and the store moved below it.
  Every call in the new shape is one the emitter already emitted; nothing was invented.

The pattern in both: **the blocked thing was the OPTIMAL fix, and a correct one was
available without it.** "No hardware to verify a cross-task reduction" is true and was
allowed to stand for "no fix is possible", which it never implied. A slower kernel that
computes the right answer is worth more than a fast one that does not, and it is
verifiable here.

What is still not claimed for either: that it runs. Both route lines still print
`[exact, unvalidated on hardware]`, which stays true until someone with the device says
otherwise.

---

# The third `lowering` cluster (2026-09-04)

Four open again — #011, #020, #022, #023 — and the rule asks for the design problem rather
than four patches. This one is not the same as either earlier cluster: cluster 1 was
*lowering is partial and pretends to be total*, cluster 2 was *a baked-in number never
meets the operand that carries it*. This is a third thing.

## The four

* **#011** — two Metal specializations bake in a shape that is not an operand. (The
  remaining half: kernels that take byte strides and no head dimension at all.)
* **#020** — scatter and gather write and read at an index no backend checks.
* **#022** — 64 intrinsics are lowered and declared nowhere.
* **#023** — the f16 matvec reads its operands in the opposite order from the f32 one.

And **#019**, filed under `transform`, is the same shape: `__tile_attention_gqa_*` is
lowered by five backends, declared by none, and read two ways.

## One design problem

**The declaration is not the contract.**

`crates/tile_std/src/tile.rs` declares 423 intrinsics, and every declaration is a list of
`u32`s. What actually separates a correct lowering from an incorrect one is none of that:

| what differs | where it lives today |
|---|---|
| which operand is the matrix and which the vector (#023) | each emitter, separately |
| whether an index may exceed the extent (#020) | nowhere |
| which operand carries a dimension the kernel bakes in (#011) | the kernel's own literals |
| that the intrinsic exists at all (#022, #019) | only in the emitters that lower it |

So each emitter invents the contract it needs, and **two emitters can disagree with nothing
in the repository able to detect it**. #023 is the cleanest instance because the two
disagreeing emitters are for the *same target*: Metal's f32 matvec reads
`p0[base + i] * p1[i]`, its f16 matvec reads `p0[i] * p1[base + i]`, and both compile,
both emit, and both pass every gate that does not run them.

That is why these four look unrelated and are not. They are all the same sentence: *the
thing that had to be agreed was never written down anywhere that a check could reach.*

## The remedy is already half-built, in this crate, for a different purpose

`emitted_parses.rs` classifies operands **from the wrapper call sites** in `tile.rs`:

```
.buf_id      -> a tile
X as u32     -> a dimension
literal 0    -> the destination
```

`wrapper_call_sites` and `classify_params` already read this and it is already trusted —
it is what builds a callable module for each of 236 declared intrinsics. So the operand
roles ARE recorded, in the typed Rust wrappers, and they ARE machine-readable.

**Nothing compares an emitter's reading against them.** The classification is used to
GENERATE inputs and never to CHECK output. A gate that did compare would have caught #023
without running anything: `tile_matvec_f16`'s wrapper says which argument is the matrix,
and the emitted kernel indexes the other one.

That is the concrete next move for this area, and it is smaller than any of the four
fixes: not "declare the 64" (#022, worth doing but larger), but **make the existing
classification an assertion**. #022 then becomes the statement of how much of the corpus
that assertion can cover — 64 intrinsics have no wrapper to compare against, which is
exactly why they are the coverage gap that entry describes.

## #024 joined it, and it sharpens the thesis rather than adding to the list

**#024** — four backends accept an f16 intrinsic and emit a kernel byte-identical to the
f32 one. It is the same sentence as the other four, with the contract term being the
WIDTH: `__tile_exp_f16` says the buffers are f16, the emitter matches the name, finds an
arm, and drops the width. Nothing in the repository compares what the intrinsic said with
what the kernel does about it.

It also lands squarely in cluster ONE. That cluster's #005 was *an unhandled intrinsic
emits a plausible copy instead of refusing*, and its detector — `emit::ignored_intrinsics`,
which re-emits with the intrinsic's NAME replaced and fails if nothing changes — **cannot
see this**, because the name is handled and only the width is ignored. `musa` is the pure
case: zero mentions of `half`, `__half` or `"f16"` anywhere in its emitter, and it emits an
f32 kernel rather than refusing.

So the two clusters are one thing seen from two sides. Cluster 1: *lowering is partial and
presents as total*. Cluster 3: *the declaration is not the contract*. **An emitter presents
as total precisely because the contract is too thin to say what it would be failing to
do.** `__tile_exp_f16` and `__tile_exp_f32` are, to the dispatcher, two names — nothing
records that the suffix is a promise about buffer width, so an arm matching one and
behaving like the other is not a detectable error.

The detector generalises the same way `ignored_intrinsics` did, and
`tests/dtype_is_not_ignored.rs` is it: emit the same kernel at both widths and require the
output to differ, unless the backend names the run-time mechanism that makes it correctly
width-independent. Two do — `tpu` reads `x0.dtype` from the array it is handed, `ttmetal`
names no dtype because its circular buffers carry their own format — and a bare diff would
have reported both as broken. **That distinction is the reason the check takes a named
mechanism and not a clean diff**, and it is the corollary this file already had to withdraw
once, in the row-width dimension, for exactly the same reason.

`bang` was on that list until this session, and its fix is the shape the other four do not
have available: one flag. `ctx.dtype = "f16"` was set inside the matmul branch only, while
the entire `half` path had been present the whole time. Eleven of its twelve f16 operations
emitted byte-identical f32. That is a bug. `musa` having no f16 code at all is a capability
gap, and the honest response there is to REFUSE — which is cluster 1's remedy, arriving
where cluster 3 predicted the hole would be.

## What is deliberately not being done

Fixing #023 by making the f16 matvec match the f32 one. The f16 kernel is DS4-derived and
its emitter declares it `activation(f32), weight(f16), output(f32)`; that is a convention,
and overruling it may break the kernel it was written for. Choosing between the intrinsic's
order and DS4's is the same class of decision as #019, and both are waiting on the same
missing thing — a declaration that says which is meant.

Guessing here would produce a fifth entry in this area rather than close a fourth.

## #025 was filed here, fired the stop, and did not belong (2026-09-05)

`tile backlog add` took `lowering` to six and said what it is supposed to say:
six in one area is one design problem wearing six hats. It was wrong, and the
rule worked anyway -- it forced the question of what the area should have been.

#025 is that the backend's `AscendMetadataLoader::get_rlib_metadata` opens an
`.rlib` and walks it with `tar::Archive::entries()`. Every other entry in this
file is the same shape as the cluster thesis: **an emitter treats a partial
function as total and produces plausible output instead of refusing.** #025 is
the opposite in every respect that matters here:

* it fails in the RUST FRONTEND, before any emitter is reached;
* it REFUSES, loudly, with a compiler error naming the file it could not read;
* nothing plausible-but-wrong is emitted, because nothing is emitted;
* no intrinsic, no dtype, no shape and no arm is involved.

Reading it together with the other five produces nothing, which is the test.
Counting it with them would have made the sixth hat an argument for a redesign
of the emitters, and the defect is a wrong archive parser in a metadata loader.

So the area vocabulary gains **`toolchain`**: the tool driving rustc and a
provisioned backend, as distinct from what an emitter does once it is running.
That is the "decide what the area should have been" the stop rule asks for, and
it leaves `lowering` at five -- the number this file has actually accounted for.

The cost of the choice is worth naming: a sixth area weakens the clustering
signal, and the vocabulary is deliberately small (`backlog.rs` says so at the
`area` field). It is justified here only because the defect class is genuinely
new to this backlog and a seventh entry of the same kind -- anything about the
toolchain rather than the lowering -- now has a home that does not distort this
one.

---

# Third reading: six open, and every one is checked by a bespoke differential

The rule fired again at six: **#011, #020, #022, #023, #024, #027**. The earlier
readings still hold, but they do not explain this set, and the set has something
the first three did not: *workarounds that are all the same shape.*

## What the workarounds say

Read them instead of the titles.

* **#024** — "`tests/dtype_is_not_ignored.rs` emits every kernel at both widths
  and compares." A differential on **dtype**.
* **#023** — "`emulate_msl.rs` drives f16 for every other operation Metal lowers
  and SKIPS matvec." A differential on **operand order**, with a hole at exactly
  the operation that was wrong.
* **#011** — two specializations bake in a shape that is not an operand, and
  nothing checks it **at selection**.
* **#020** — an index no backend checks: no **bounds** obligation anywhere.
* **#022** — 64 intrinsics are lowered and **declared nowhere**.
* **#027** — the host dispatch has no obligation to reach the device, so it is
  checked by a lint in another repository entirely.

Five hand-built harnesses, each covering one axis, each with a hole where nobody
thought to look. #023's hole is not an oversight in the harness; it is the
harness's design: it can only compare an emission against *another emission*.

## One design problem

**An intrinsic has no contract, so an emitted kernel can only ever be checked
against another emission, never against a specification.**

That is why every one of these is found by a differential and why every
differential is partial. A differential answers "did these two emissions
agree?", which is a proxy for "is this emission right" and fails in exactly one
direction: when both sides are wrong the same way, or when the axis under test
was not the axis that broke. #024 is both sides wrong the same way — four
backends emit f32 for f16 *byte for byte*, so any check comparing backends to
each other passes.

#022 is not a coverage gap sitting next to these. It is the root: 64 intrinsics
are **lowered and declared nowhere**, so for those there is nothing a contract
could even be written against.

## What to change

Give the intrinsic a declaration, and make it carry the obligations rather than
just the name and arity:

* operand order and arity;
* dtype behaviour — what an f16 intrinsic must emit, so #024 is a check and not
  a test someone remembered to write;
* the reduction axis, if it reduces (which is #014, #015 and #016, now closed,
  all three);
* an index-bounds obligation (#020);
* which operands are shapes, so a specialization cannot bake in one that is not
  (#011);
* and, at the host layer, the launch obligation (#027).

Then the bespoke harnesses collapse into one mechanism: emit, and check the
emission against the declaration. Where a device exists, `-r` already runs and
verifies and becomes the executable half of the same contract; where one does
not, the structural obligations still apply.

This is also the answer to a question asked from the cannbench side: whether
`lint_launch.py` could be lifted into the type system. It can, and the shape it
wants is this declaration. The lint discovered three categories the hard way --
a helper returning an ARGUMENT carries no obligation, a helper returning an
output it ALLOCATED must discharge one, and a deferred launch is a token the
caller consumes exactly once -- and those are a borrow, a produced value and a
linear token. They are not lint heuristics that happen to work; they are the
contract, written in the only place it currently exists, which is a regex in
another repository.

**Not proposed:** that the contract can decide everything. Whether a grader's
profiler observes a launch is empirical, and whether an identity permutation
should alias or copy is a semantic choice a person makes. A declaration
enforces a decision; it does not make one.

---

# The contract, written out: five obligations, each learned by losing

The third reading proposed that an intrinsic declaration should carry its
obligations. This is what those obligations actually are, taken from a
competition tree where each was learned the expensive way rather than designed.
They are listed here because a contract invented at a desk is a guess, and these
are not: every one has a measured cost next to it.

**1. REACHES THE DEVICE.** Every path out of the dispatch performs at least one
launch. Not derivable from the dataflow -- an identity permutation has no work
to do and returning the input is semantically correct -- so it is a property of
the emitted trace and wants a linear witness that `launch()` mints.
*Cost of not having it: two operators scored 0.00 with answers that were exact.
129 points.*

**2. EVERY OFFSET IS BOUNDED BY THE ALLOCATION, NOT BY THE DATA.** A value read
from the input can be anything the input can make it. Out of range it is not a
wrong answer, it is an out-of-bounds store that traps the core -- and on this
hardware a trapped core skips every case after it, so one costs forty.
*Cost: unique 52 cases skipped, moe_gating 44.* Neither was reachable by any
shape sweep, because the trigger is the data. That is the part a type system
buys you that testing does not.

**3. EVERY LIMIT IS DERIVED, NEVER FITTED.** A bound equal to the largest shape
the operator was shown is a sample, not a bound. It is invisible to the visible
set by construction, because that set is what it was fitted to.
*Cost: dilation_2d refused 16x16 filters its kernel sized every buffer to
handle; kMaxTaps was declared and never used.*

**4. A FAULT IS SCOPED TO ITS CASE.** Where an obligation cannot be met, refuse
the case cleanly rather than fault the device. This is worth roughly forty to
one and it does not require understanding the defect: engram converted a
device-killing allocation into a TORCH_CHECK and recovered 29 skipped cases with
the bug entirely unmoved.

**5. A GENERATED ARTIFACT IS REPRODUCIBLE FROM ITS GENERATOR.** Otherwise the
"do not edit by hand" header is an instruction to destroy work, and the build
will not object. *Cost: 21 of 48 generators diverged; one would have deleted
every python wrapper in the package, which is a zero per operator with every
case still computing correctly.*

## Why this belongs in tile_spec and not in five linters

It already IS five linters -- lint_launch, lint_index, lint_capacity,
lint_ub_ceiling and lint_generated, gating a build in another repository, each
a regex over C++ that reconstructs by pattern what the emitter knew and threw
away. They work, and they found 31 no-launch returns, 8 unchecked indices, 5
fitted bounds and 21 stale generators. But a regex cannot see an intrinsic's
arity, its reduction axis, or which operand is a shape, so each one had to be
calibrated by deliberately breaking it, and two of them shipped false-clean
first (lint_capacity classified by magnitude; lint_launch blanked the string
delimiters it needed).

An emitter that declared these would not need any of them, and would carry them
to every backend rather than the one where they were paid for.

---

# Measured: the cpp lift runs on 64 of 65 kernels and none of them round-trips

`lift_cpp.rs` and the `ascendc-to-rs` toolchain are both real and installed, and
`tile k.cce --via tile -t tile` produces tile DSL for essentially the whole
cannbench tree. That is further than #003 and #004 left it -- both were closed
on "rewrite it by hand", and the reader has been built since.

Run over all 65 kernel translation units in cannbench-tilers:

    lifted (produced output)          64 / 65
    lifted with NO holes               0 / 65
    `// verify:` markers            1805
    `TODO`                          6398
    `__tile_buf_alloc((0) as u32)`  1557

The last line is the one that decides it. A zero-size allocation is not a
warning, it is a kernel that allocates nothing, so nothing lifted here can be
re-emitted and run. The markers are honest -- "allocation for `buf_o` was not
lifted; size inferred from its first use", "TODO untranslated size" -- and the
route prints `[synthesised]` and says it is only as good as the lift. Nothing
here is claiming more than it does.

**Why this belongs in this cluster rather than as a seventh entry.** It is the
same defect the other six are: there is no contract saying what a lifted kernel
must satisfy, so the lifter emits what it could parse and marks the rest, and
the only thing that would catch a wrong lift is comparing it against another
emission. A declaration that carried buffer extents -- which the ORIGINAL kernel
states, in `pipe_.InitBuffer(buf, n * sizeof(T))` -- would make "size not
lifted" a checkable failure instead of a comment.

**The single highest-value fix**, if the goal is a working round trip: lift the
InitBuffer extents. 1557 of the holes are that one thing, and every other marker
is survivable in a kernel that at least allocates.

## ...and the return half needs a build this repository cannot make

Lifting is only one direction. Asking for the other one:

    tile gelu_lifted.rs -t cpp -o gelu.cce
    tile: the route tile -> tile -> mlir -> cpp needs feature:ascend, which this
      build does not have.
      "ascend" gates code that is not in this repository, so rebuilding here
      cannot turn it on.

So `cpp -> tile -> cpp` has two halves and they fail differently. The READ half
is here, runs, and is now mostly faithful (the `bytes_to_count_expr` truncation
took 709 untranslated sizes to 61). The WRITE half is not here at all, and no
amount of work in this tree turns it on.

That is worth stating because it reorders the remaining work. Finishing the
lift's last 893 inferred allocations makes the lifted DSL correct, and correct
is worth having on its own -- but it does not by itself let anything be
re-emitted, and a plan that assumed "fix the lift, then regenerate the kernels"
would find that out at the end rather than the start.

---

# Eight open, and the eighth says the seventh reading was wrong

The reading below this line concluded that #028 -- a Fixpipe descriptor slot
silently dropped by an arch-conditional -- was "the same design problem, seen
from the target end", and kept it in `lowering`. Then #029 arrived: the Ascend
cube has an im2col instruction (`LoadData3DParamsV2`) and nothing in tile-rs
can reach it, so a convolution lowers to a materialised column matrix instead
-- a 419 MB GM intermediate against a 16 MB input, at 0.17x of the reference
where the field leader is at 1.63x.

Reading #029 with #011, #020, #022, #023, #024 and #027 produces nothing. It is
not an operand order, a dtype, a bound, a reduction axis or a launch. Reading it
with #028 produces a sentence immediately:

**tile-rs has no model of what a target CAN DO -- only of what its emitter
happens to write.**

* #028: a descriptor's slots, and which of them exist on which architecture.
  Present on both, at different addresses, and an `#if` around three field
  assignments deleted them with no diagnostic.
* #029: an instruction that exists and is never issued, so the lowering picks
  the portable shape and pays a K-fold memory blowup for it.

Both are the target's contract, not the intrinsic's. The six that remain are
all the intrinsic's -- *what the caller promised* -- and the fix the third
reading proposed for them (put the obligations in the declaration) does not
touch either of these, because neither is about the caller.

## What the area should have been

Split, on the same reasoning that gave `toolchain` its own area at #025 and by
the same test: read the candidate with the incumbents and see whether anything
comes out.

    lowering   what the CALLER promised and the emitter must honour
               #011 #020 #022 #023 #024 #027
    target     what the TARGET offers and the emitter must know
               #028 #029

`#028` and `#029` are refiled under **`target`**. That leaves `lowering` at six
-- the number the three readings above actually account for -- and gives the
next entry of this kind somewhere to go that does not distort them.

The cost is the same one #025 named: a seventh area weakens the clustering
signal, and the vocabulary is deliberately small. It is justified here because
two entries already fit it, they arrived from different directions (a struct
layout and a missing instruction), and putting either in `lowering` would have
made the eighth hat an argument for redesigning the emitters when the defect is
that nobody wrote down what the chip does.

## What `target` should hold, so it does not become a bag

A target declaration, in the same evidence-graded shape `HardwareParams.measured`
already uses for capacities:

* **instructions** the target has, with the operand shapes they take -- so a
  lowering can ASK for load3d and be refused with a name, rather than never
  asking. #029.
* **descriptors** as named slot sets, per architecture and per layout, with
  presence as part of the declaration -- so an unplaced slot is an emit-time
  error and a slot that does not exist on this arch is a refusal rather than
  dead text. #028.
* and the capacities that are already there, which is the part that works: the
  cannbench tree asserts L0_A / L0_B / L0_C from `GetCoreMemSize` on every run
  and derives its tile from them, which is exactly the shape the other two
  want.

The measured payoff for having asked the target ONE question, in the tree where
this was found: `Mmad` accepts `Tuple<float, float, float>`, which is four
lines above the assertion text naming it. Nobody had asked, so every float32
matmul went down a vector Axpy loop -- 12x to 70x slower than the cube, with
identical accuracy, for a year.

---

# Seven open, and #028 is the target side of the same missing contract

The rule fired again on `tile backlog add`, at **#011, #020, #022, #023, #024,
#027, #028**. Six of those the readings above already account for. This is about
whether the seventh belongs, and the #025 precedent says that is the question to
ask rather than assume.

## #028

An AscendC matmul writes its result through `Fixpipe`, which takes a parameter
struct. The struct has a **different shape on each architecture**:

```cpp
struct FixpipeParamsV220 { ... uint16_t ndNum = 1; uint16_t srcNdStride = 0;
                               uint16_t dstNdStride = 0; ... };

template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsArch3510 {
    __aicore__ FixpipeParamsArch3510() {}                  // empty body
    typename TransformParams<format>::PARAMS params;       // the same three,
};                                                         // one level down
```

The codegen was arch-conditional in the obvious way, and the obvious way is
wrong:

```cpp
#if !(__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
    fp.ndNum = 1; fp.srcNdStride = 0; fp.dstNdStride = 0;
#endif
```

On c310 that programs `LOOP3_PARA` -- a hardware register, written on every
Fixpipe -- from an uninitialised `params`. It compiles clean, because the `#if`
removed the field access AND the compiler's chance to say the field was gone.

## It is the same design problem, seen from the target end

The third reading's thesis is *an intrinsic has no contract, so an emission can
only be checked against another emission*. #028 is that sentence with the
subject changed: **the TARGET's descriptor has no contract either.** `Fixpipe`
is a named set of slots -- n, m, srcStride, dstStride, quantPre, ndNum,
srcNdStride, dstNdStride -- and every one of those slots exists on both
architectures. Only their ADDRESSES differ. Nothing anywhere records that the
set is the same, so a per-arch branch could drop three of them and no check in
either repository could notice.

Two details make it worse than a rename, and both have to survive into whatever
replaces it:

* `TransformParams<CO2Layout::NZ>::PARAMS` is `uint8_t` -- for that layout the
  nd parameters **do not exist**, and setting them is the error rather than the
  fix. The target layout decides whether the slot is there.
* `CO2Layout::COLUMN_MAJOR` has the slots under **different names** --
  `dnNum`, `srcNzMatrixStride`, `dstDnMatrixStride`.

So the mapping is `slot -> (exists?, path)`, per architecture and per layout.
That is exactly the declaration the third reading asked for, applied to a target
builtin instead of a tile intrinsic, and it is why this is a seventh hat rather
than a seventh problem.

## The sixth obligation

The contract written out above has five, each with a measured cost. This adds
one, with the same provenance -- it was paid for, not designed:

**6. EVERY SLOT OF A TARGET DESCRIPTOR IS FILLED, ON EVERY ARCHITECTURE THAT
HAS IT.** An arch-conditional may choose a *path* to a slot. It may not choose
whether the slot gets a value. Where an architecture genuinely lacks the slot,
that is a fact about the target and belongs in the descriptor's declaration, so
that a value supplied for it is a refusal instead of dead text.
*Cost: one benchmark credit for a submission whose only change was this, which
came back byte-identical to the run it was supposed to fix -- and the byte-for-
byte match is still unexplained, because the toolkit that would settle it is the
grader's and not this machine's.*

Note what obligations 1-5 have in common with this one and what they do not.
All six are properties an emitter knows and throws away. But 1-5 are about the
kernel's own behaviour and can, in principle, be recovered by reading the
emitted C++ -- which is what the five linters do. **This one cannot**, by
construction: the text `fp.ndNum = 1` is present in the file, and whether it is
compiled is a property of the preprocessor and the toolkit's headers together.
A regex over the source is the wrong instrument in a way it was not for the
other five, and that is an argument for the declaration rather than a sixth
linter.

## What is deliberately not claimed

That fixing the descriptor fixes the c310 matmul. It does not: the submitted
result was identical with and without the struct swap, so the struct may never
have been the defect. What #028 establishes is that the check could not have
existed -- not that the bug it would have caught is the bug that is costing
points. NOTES-ascendc.md 206 keeps both readings open.

---

# Seven again with #031, and it is the OTHER half of #027

`lowering` is #011, #020, #022, #023, #024, #027 and now #031 -- a blocked loop
that clamps its trip count and then advances by the unclamped one, silently
dropping every row in between.

## Reading it with the six

Five of the six are the third reading's thesis: *the declaration is not the
contract*, so an emitter's reading of an intrinsic cannot be checked against
anything but another emission. #031 is not that. Nothing about an intrinsic is
wrong in it. The loop is CODEGEN STRUCTURE -- "iterate in blocks of N, clamp at
a boundary, advance" -- and it was written by hand, in a kernel, for the
fourteenth time.

**#027 is the same sentence.** "No host dispatch is emitted, so *the operator
must reach the device* cannot be a codegen invariant" -- the emitter does not
own the dispatch, so every kernel re-implements it and a regex in another
repository has to check it afterwards. #031: the emitter does not own the block
loop, so every kernel re-implements it and nothing checks it at all.

    #027   the emitter does not own the DISPATCH   -> checked by a lint, elsewhere
    #031   the emitter does not own the BLOCK LOOP -> checked by nothing

So the area now holds two theses, not one:

    (a) the declaration is not the contract        #011 #020 #022 #023 #024
    (b) the emitter does not own the structure,    #027 #031
        so each kernel re-derives it and can get it wrong

## Why (b) is worth separating

(a) is fixed by giving the intrinsic obligations -- operand roles, dtype,
bounds, reduction axis. That work does not touch (b) at all: a perfectly
declared intrinsic still has to be *emitted into* a loop nest, and the nest is
where #031 lives.

(b) has a different remedy: the emitter should own a small set of structural
templates -- the block loop, the dispatch, the double-buffered stage -- and
emit them, rather than every kernel writing its own. That is the difference
between a bug that can recur and one that cannot.

**The evidence that (b) is real and not theoretical**, all from one session in
the cannbench tree:

  * #031 itself: `rb += kRow` where `rb += rc` was meant. Needs `M % kRow != 0`
    AND a block spanning a boundary, so batch-1, M-divisible and one-row-per-core
    shapes all pass. Shipped, and found only because a downstream operator fell
    to 2/20.
  * The same session's `prep_split`/`prep_weave` processed ONE row of 2H
    elements per iteration with three full barriers around it, so the cost was
    per ITERATION and nearly independent of H -- 46 ns and 32 ns per row across
    two shapes whose H differs by 2x. Batching kT/H rows into one pass gave 6.9x
    and 10.9x. Nothing about the intrinsic was wrong; the loop was.
  * And `tile_mm`'s inner loop is three `PipeBarrier<PIPE_ALL>` per k-step with
    no double buffering -- the same class, unfixed, and bounded at ~2x because
    the matmul turned out to be within 2.4x of `torch.bmm` anyway.

Three instances, three kernels, one missing abstraction.

## What changes

Nothing is refiled yet -- (b) has two members and the `toolchain`/`target`
precedent was set at a clearer boundary. But the next entry of shape (b) makes
three, and at that point the area should split into `lowering` (the caller's
contract) and something like `emit-structure` (the loop nests, the dispatch,
the staging), the same way `target` was split out for what the CHIP offers.

Recorded now so that decision starts from three data points instead of being
re-derived.

---

# The split, executed (2026-09-12)

#033 is the third member of (b), so the condition this document set for itself
is met and the area is split. `#027`, `#031`, `#033` are now `emit-structure`;
`lowering` keeps `#011 #020 #022 #023 #024`.

## Why #033 is shape (b) and not shape (a)

It reads like a transform bug -- an invariance analysis that is per-expression
where it should be per-factor -- and #030 in `transform` is genuinely that. But
the reason the factor got dropped is structural. Hoisting a loop-invariant
subexpression into a table is a STRUCTURAL template: build the table once,
index it at the use site, emit whatever did not go into the table. The emitter
does not own that template, so the table was hand-written in one function and
its use hand-written in another, and nothing related the two. The residual had
no place to be emitted because no component owned the transform.

    #027   the emitter does not own the DISPATCH        -> checked by a lint, elsewhere
    #031   the emitter does not own the BLOCK LOOP      -> checked by nothing
    #033   the emitter does not own the HOIST+RESIDUAL  -> checked by nothing

Same sentence three times. That is an area.

## What each area's redesign is

**`lowering` -- give the intrinsic obligations.** An intrinsic declaration
currently carries a name and an arity. It must carry operand ROLES (which is
the matrix and which the vector: #023), the DTYPE as an obligation the backend
must honour rather than a hint it may ignore (#024), the INDEX BOUNDS it reads
and writes (#020), and the SHAPE facts a specialization depends on (#011). #022
is the gate: 64 intrinsics are declared nowhere, so there is no place to hang
any of this until declaration is mandatory.

The single sentence: **a backend must not be able to accept a call it does not
implement the contract of.**

**`emit-structure` -- the emitter owns a small set of structural templates**,
and kernels instantiate them instead of rewriting them. The three that have
already cost measured score:

  * the **blocked loop** -- take N, clamp at a boundary, process, advance by
    the CLAMPED count (#031; shipped wrong, found only when a downstream
    operator fell to 2/20)
  * the **host dispatch** -- the operator reaches the device (#027; currently
    checked by a regex in another repository)
  * the **hoist with residual** -- tabulate the invariant factors, emit the
    rest at the use site (#033; silently correct on every case where the
    dropped factor is 1)

and two more with measured evidence but no entry yet, because the workaround
was cheaper than the filing at the time:

  * the **batched pass** -- `prep_split`/`prep_weave` did one row per iteration
    with three full barriers around it, making the cost per-ITERATION and
    nearly independent of H. Batching gave 6.9x and 10.9x.
  * the **double-buffered stage** -- `tile_mm`'s inner loop is three
    `PipeBarrier<PIPE_ALL>` per k-step with no double buffering. Bounded at
    ~2x and still unfixed.

The single sentence: **a structure that every kernel needs should be emitted
once, not re-derived per kernel.**

## The count is still over

Splitting does not reduce 16 open, and it is not meant to. What it buys is
that each cluster now names ONE design problem with ONE sentence, which is the
precondition for fixing an area rather than its members. `lowering` at 5 and
`emit-structure` at 3 both still fire the per-area rule, correctly: neither is
five bugs or three bugs.

---

# 2026-09-17: fired again at 7. The intrinsic signature is not authoritative.

The rule fired on `lowering` (7) and `emit-structure` (3), 10 overall, when
#038 was added. Reading the seven together.

| # | symptom |
|---|---|
| 011 | two Metal specializations bake in a shape that is not an operand |
| 020 | scatter and gather write and read at an index no backend checks |
| 022 | 64 intrinsics are lowered and declared nowhere; 46 of them by MSL |
| 023 | the f16 matvec reads its operands in the opposite order from the f32 one, and out of bounds |
| 024 | four backends accept an f16 intrinsic and emit an f32 kernel, byte for byte |
| 034 | an intrinsic's declared scalar operand is ignored and a constant baked in its place |
| 038 | bounds check refuses a cube matmul by measuring it against VECTOR limits |

## One design problem

**Nothing in the tree says what an intrinsic IS.** There is no table giving,
for `__tile_matmul_f16`:

* its operands, and for each whether it carries DATA, a SHAPE, a SCALAR or an
  INDEX — 011 bakes a shape because nothing said it was an operand, 034 bakes a
  constant over a declared scalar, 020 cannot check an index because nothing
  marks one;
* the ELEMENT TYPE its own name announces — 024 emits f32 for an f16 intrinsic
  in four backends and 023 gets f16 operand ORDER wrong, because the `f16` in
  the name is a string each emitter re-reads by hand;
* which EXECUTION UNIT it runs on — 038 measures a cube matmul against the
  vector unit's 8-bit repeat counter and against UB, and refuses a kernel
  proven correct on two targets (rel 2.2e-07 on 910B; 20/20, 0 anti-cheat on
  950pr/c310, cannbench-tilers NOTES 270);
* that it exists at all — 022 is 64 intrinsics lowered by emitters and declared
  in no one place, which is this same absence counted a different way.

Fifteen emitters each re-derive all of it from the intrinsic's NAME, by
substring match, independently. They disagree, and every disagreement is a row
above. 022 states the cluster plainly: the declarations are missing, so there
is nothing for the other six to have been checked against.

This is the previous cluster one layer down. That one said lowering is partial
and must be able to REFUSE. Refusing requires knowing what was asked for, and
that is what is missing here — a name is not a signature.

It is also NOTES 268 in cannbench-tilers, reached from the other end: "the
intrinsic layer has no type for a reduced tile and no broadcast-back, so every
composable use is malformed." Same root. Each caller patches around the part
of the description it personally needs.

## What the area should have been

One declarative registry — the single source of truth:

    __tile_matmul_f16:
      operands: [acc_mode: scalar_i32, a: data<f16>, b: data<f16>,
                 m: shape, k: shape, n: shape]
      result:   data<f32>        # accumulator width, not operand width
      unit:     cube             # <- 038 needs exactly this field

* **Emitters read it** instead of string-matching a name. One that cannot serve
  an entry refuses (the previous cluster's rule) rather than silently emitting
  the f32 kernel (024), the reversed operand order (023), a baked constant
  (034) or a baked shape (011).
* **The bounds model dispatches on `unit`.** A cube op is checked against
  L1/L0A/L0B/L0C, a vector op against UB and the repeat field. Today there is
  one check and it is the vector one. That is 038 — and the knock-on is worse
  than a false refusal: the ONLY way to reach the cube matmul today is to hide
  the extents from the checker by leaving M/K/N undefined, which also defeats
  `detect_blocked_matmul_loads`, leaves a dead full-shape `loc=vec` tile per
  operand, and so caps the reachable shape at about 128x128 and keeps the
  K/N-blocked path — the one that parallelises over N via `get_block_idx` —
  from ever being selected.
* **Coverage enumerates it.** 022 becomes a diff between registry and backend
  rather than a survey. 020's index checks hang off the operand kinds.

## Cost of not doing it

Measured. #038 alone is what stands between the emitter and the cube-bound half
of the CANN Bench board — GQA, MHA, SparseFlashAttention, MLA, MlaProlog,
GroupedMatmul, QuantMatmul, Conv2D, WeightQuantBatchMatmul,
Conv3DBackpropFilter, GroupedMatmulSwigluQuant, together about half the gap to
top-3. The PTO cube is proven to work on c310. What is missing is the ability
to ask the emitter for it at a shape the bounds checker will accept.

## Order

1. The registry, with `unit` and per-operand kinds. Populate the ~30 intrinsics
   the Ascend path uses, not all 64.
2. Bounds model dispatches on `unit`. Closes 038, unblocks real matmul shapes.
3. `parse_u32_from_arg` accepts `arith.constant` as well as
   `llvm.mlir.constant`, so a declared extent stops being a liability.
4. Emitters read the registry for dtype and operand order. Closes 024, 023,
   034, 011.
5. Coverage diffs against it (022); index checks hang off it (020).

# 2026-09-19: seven again. The registry as drafted would not have caught #039.

#038 closed. #039 took its place, so the area is at seven with a different
seventh, and the six it joins are unchanged:

| # | symptom |
|---|---|
| 011 | two Metal specializations bake in a shape that is not an operand |
| 020 | scatter and gather write and read at an index no backend checks |
| 022 | 64 intrinsics are lowered and declared nowhere; 46 of them by MSL |
| 023 | the f16 matvec reads its operands in the opposite order from the f32 one, and out of bounds |
| 024 | four backends accept an f16 intrinsic and emit an f32 kernel, byte for byte |
| 034 | an intrinsic's declared scalar operand is ignored and a constant baked in its place |
| 039 | the transposed matmul cannot block, so every runtime-shaped transB refuses |

The previous reading is not superseded. #039 is the same mechanism a fourth
time — an emitter reads an operand position, does not find what it expects, and
substitutes a value it invented rather than refusing. 011 invented a shape, 024
a dtype, 034 a scalar; 039 invented an *extent*, reading `PTO_MM_KB` (256) and
`PTO_MM_NB` (64) for a K and an N the caller supplies at runtime, emitting one
fixed `M x 256 x 64` tile, and computing that corner of a much larger output.

## What is new, and it is a hole in the proposed fix

The registry sketched above gives `__tile_matmul_f16` the operands
`[..., m: shape, k: shape, n: shape]`. **That is not enough to have prevented
#039.** K and N *are* declared as shape operands there, and the emitter *did*
read those positions. What it could not express is that a shape operand has two
states — known now, or supplied at runtime — and that they are not
interchangeable. `resolve_const` collapses both to a `u32`:

    fn resolve_const(&self, s: &str) -> u32 {
        if let Some(&n) = self.const_map.get(s.trim()) { return n; }
        parse_const_arg(s)                       // and this returns 0
    }

There is no `None`. A runtime extent cannot be *represented*, only guessed at,
and `generate_func_pto` makes the guess worse than a guess by seeding
`const_map` with the block stand-ins — so the runtime case does not even reach
`parse_const_arg`'s zero, which at least looks wrong. It gets 256 and 64, which
look like shapes.

So the registry needs a third axis alongside kind, dtype and unit:

    k: shape<dynamic_ok>        # and the emitter must handle BOTH states

and `resolve_const` must become `Option<u32>`, with every caller made to say
what it does when the answer is "not known here". That is the change that turns
039 from a defect into a type error.

## Why this one was invisible for so long, and what it costs

A *static* transposed matmul was always correct — verified here, and
independently on a peer's tree at M=16 K=512 N=32 (9.611e-08); a second
anchor at M=32 K=4096 N=512 was withdrawn 2026-09-20 as measuring a commit
outside that tree's history. Every test of the
route used static extents, so every test passed. The accumulator-dtype test
added to this same route days earlier lowered a *dynamic* module and asserted
only on the L0C dtype; it passed, correctly, while certifying a module that
computed a corner. It is the fourth entry in this cluster caught only by a
bespoke differential and the first caught by a device measurement instead.

Cost is unchanged in kind from #038 and smaller in size: `transB == 1` is the
most frequent single reason the matmul family misses the cube on 950PR, and
until the transposed path blocks, every one of those shapes falls to the
vector path. That is a bounded, nameable loss rather than half the board — but
it is the same root, and it will keep producing entries at this rate until
step 1 of the order above is actually done.

## Order — unchanged, with one insertion

The five steps below "What the area should have been" stand. Insert, between
1 and 2:

1b. `resolve_const -> Option<u32>`, and a `shape` operand carries whether it
    may be dynamic. Every emitter that reads an extent is then forced to say
    what it does when the extent is not known at compile time — block, or
    refuse. Today the default is neither: it is to invent one.

# 2026-09-19, later: #039 closed, #040 replaces it, and the shape-operand hole is now measured

#039 ("the transposed matmul cannot block") is CLOSED -- eaaa625d blocks it,
M=16..512 all lower, ptoas assembles all ten kernels. #040 takes its place, and
the count is unchanged at seven because the defect moved rather than went away.

That movement is the finding. The 2026-09-19 reading above said the proposed
registry would not have caught #039, because it declares K and N as `shape`
operands and the emitter DID read those positions; what it could not express is
that a shape has two states, known-now and supplied-at-runtime. Step 1b was
added: `resolve_const -> Option<u32>`, so absence is in the type.

**#040 says 1b is necessary and not sufficient.** I implemented the dynamic
shape correctly -- the transposed tensor_view now carries `%kk` instead of the
block stand-in, which is exactly what 1b is for -- and ccec refused every
kernel:

    static K    strides = [%c1, %c1024]  ->  Stride<64,64,64,1,1024>  DN   ok
    runtime K   strides = [%c1, %kk]     ->  Stride<64,64,64,1,  -1>  ND   refused

`Layout::DN` is INFERRED by the vendor's C++ templates from the stride, and the
inference needs a compile-time constant. So a shape operand does not have two
states, it has THREE, and the third is the one that bites:

    static            known to us and to the target
    runtime           known to neither
    runtime-to-us,
    static-to-target  must reach the generated code as a template parameter

The third is not an emitter convenience. For a row-major view the unknown
extent lands in the stride that is not 1, so the view stays classifiable and
nobody notices the distinction exists. For the TRANSPOSED view the unknown
extent IS the non-unit stride, and the same shape becomes untypeable. Whether a
runtime extent is expressible therefore depends on the LAYOUT it is used
through, which no amount of `Option<u32>` at the call site can tell you.

## What this adds to the order

Step 1b stands. Insert after it:

1c. A shape operand records whether the target can accept it dynamically
    THROUGH THE LAYOUT IT IS USED IN, and the emitter can hoist one into a
    kernel template parameter when it cannot. Without the hoist the only
    options are to specialise per extent (combinatorial) or to bake a constant
    and be wrong -- and the tree has already done the second: make_tv_transposed
    wrote `strides = [%c1, %c256]`, always typed as DN, and those constants were
    the block stand-ins. It compiled and computed one corner of the caller's
    output for months.

That last sentence is the cluster in one line. **Correctness and typeability
were being traded against each other, silently, because nothing in the tree
could represent the trade.** #011 bakes a shape, #034 bakes a scalar, #039 read
a stand-in as an extent -- every one of them is an emitter choosing the value it
can NAME over the value it was GIVEN. The registry fixes naming. 1c is what
stops the trade.
