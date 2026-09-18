id: 5
area: lowering
title: an unhandled intrinsic emits a plausible COPY kernel instead of refusing
opened: 2026-09-01
closed: 2026-09-02

## wanted
Lower a matmul to Metal and run it. `tile mm.mlir -t msl -o mm.metal`.

## got
It "succeeded". The emitted kernel compiles, `xcrun metal` accepts it, and the
body is:

    uint gid = base + tid;
    if (gid < num_elements) p1[gid] = p0[gid];

That is a copy. It ignores the third buffer entirely and computes nothing like a
matmul. No warning, exit 0, fidelity reported as [exact, validated on Apple GPU].

Isolated by holding the module shape fixed and varying one thing at a time:

    op         dtype  emitted
    matmul     f16    real matmul (acc += p0[m*K+kk] * p1[kk*N+n])
    matmul     f32    COPY
    relu       f16    real relu
    relu       f32    COPY
    softmax/exp/sigmoid/rms_norm/layernorm  f16 and f32   correct

Two different causes, one symptom. `__tile_relu_f32` does not exist in tile_std at
all -- only `__tile_relu_f32_4` and `__tile_relu_f32_scalar` -- so that is an
unknown name. `__tile_matmul_f32` IS declared (tile_std/src/tile.rs, in the same
extern block as the f16 one) and IS referenced twice in mlir_to_msl.rs, so that is
a handled name whose arm does not fire on this module.

The unifying defect is neither of those: it is that the emitter has no refusing
default arm. An intrinsic it does not handle falls through to a copy that compiles.

## workaround
Used f16 for matmul, which lowers correctly. Found the f32 case only by emitting
both and diffing the bodies -- nothing in the tool's own output distinguishes a
correct lowering from this one, which is the part that matters.

Evidence for the fix, in priority order:

1. The default arm should REFUSE, not copy. `tile_spec`'s emit-purity and generality
   matrices cannot catch this: they only ever feed canonical snippets, so every arm
   they exercise is one that exists. A test that feeds a deliberately unhandled
   intrinsic and asserts an Err would have caught both cases on the first run.
2. `-r` catches it the moment there is a matmul reference (backlog #002) -- the
   copy's numbers are nowhere near a matmul's. That makes #002 worth doing FIRST:
   it is the detector for this whole class.
3. The f16/f32 split is worth auditing across all 15 emitters, not just msl. The
   probe above is four lines of shell and can be run per target.

## fixed at the root (2026-09-02)

The note above named the unifying defect correctly -- "the emitter has no refusing default
arm" -- and then filed the fix as belonging to a tree this crate does not own. That was
wrong: `crates/rustc_codegen_tile/` IS in this repository. It is excluded from the root
workspace and has no Cargo.toml, which is why it reads as external, but its sources are
here and `tile_cli` `#[path]`-includes them.

`reject_unhandled_compute` now runs after `classify_body` in `mlir_to_msl.rs`:

    kernel @mm uses a compute intrinsic (__tile_matmul_f32) and this emitter has no
    arm for it, so it would have emitted a COPY of the first buffer instead -- source
    that compiles, runs, and computes something else. Refusing.

The test is STRUCTURAL rather than a second list of handled names, which would drift out
of step with `classify_body` the first time anyone added an intrinsic. `ctx.kernel_type`
starts at `KernelType::Copy` and every recognised intrinsic overwrites it, so:

  * a non-structural intrinsic is present, AND
  * the kernel type is still the default

means, by construction, that nothing claimed it. `is_structural_intrinsic` already existed
for the chain guard, so the two refusals share one definition of "compute op".

Both causes the note distinguished are now refused, and they did not need to be told
apart: `__tile_relu_f32` (a name tile_std does not declare) and `__tile_matmul_f32` (a
declared, handled name whose arm did not fire) both reach the same guard, because the
guard asks what the emitter UNDERSTOOD rather than what the source said.

A genuine copy -- load then store, no compute intrinsic -- is unaffected and still lowers;
a guard that refused it would have been worse than the bug. Both directions are asserted
in `mlir_to_msl.rs`'s own tests, so the backend carries them rather than only this CLI.

Verified: 246 emitter tests pass (244 before, plus these two), all 16 golden files are
byte-identical, and the full suite is green.

What this does NOT do: make the arm fire for `__tile_matmul_f32`. That kernel is still not
lowerable to MSL at f32 -- but it now says so instead of handing back a copy, which is the
part that mattered. Use f16, which lowers correctly.

## the same defect in the detector, found by surveying every emitter (2026-09-02)

Fixing msl raised the obvious question: what do the other fifteen do? Measured rather than
assumed, by lowering a kernel whose only compute op is `__tile_relu_f32` to every writable
form. Six emitted output and said nothing:

    gpu  musa  nki  tpu  bang  spirv

They were not escaping the emitter's guard -- msl is the only one with an arm. They were
escaping `emit::ignored_intrinsics`, and for a reason worth recording. The detector renames
the intrinsic and asks whether the output changed. `mlir_to_gpu` writes

    // TODO: unhandled intrinsic: __tile_relu_f32
    p1[goff] = %y;

so renaming it DID change the output -- by exactly the one word inside that comment -- and
the detector concluded the emitter had depended on the name. It had only quoted it while
ignoring it, which is the case the detector exists to find. An emitter that mentions what
it is about to drop was invisible to the check written to catch emitters that drop things.

Normalising the name out of BOTH outputs before comparing is the fix: the question is
"identical apart from where the name is echoed", not "identical". All six came back.

The state now, for an intrinsic nobody handles:

| outcome | forms |
|---|---|
| refuses at emit | msl |
| caught by the detector, exit 1 | gpu, musa, spirv, nki, aie, tpu, bang, gaudi, hexagon, ttmetal, csl |
| genuinely lowers it | pico — folds relu to `vvmax` and records it in the manifest |

No writable form can now emit a wrong kernel for an unhandled intrinsic without saying so.
`no_emitter_can_silently_ignore_an_intrinsic_any_more` asserts exactly that, over the form
table rather than a list, so a seventeenth backend is covered the day it is added.

One thing this surfaced and did not fix: `mlir_to_gpu` leaks an MLIR SSA name (`%y`) into
its CUDA output, which cannot compile. The detector now catches the module, `tile` exits 1
and says the output is not a lowering, so nobody is misled -- but the emitter should refuse
rather than write source it knows is broken. That is a `mlir_to_gpu` change, and this note
is the evidence for whoever makes it.

## giving the arm to the emitters that produced a plausible copy (2026-09-02)

The CLI-level guarantee was complete once the detector was fixed, but it is a guarantee of
`tile`, not of the emitters. The real rustc backend calls `convert_mlir_to_*` directly and
gets no detector, so a library user still received a wrong kernel in silence.

Worked through the ones whose fallthrough produced something PLAUSIBLE -- output that
compiles and runs and looks like a kernel -- because those are the dangerous shape:

| emitter | what it wrote for an unhandled op | now |
|---|---|---|
| msl | `p1[gid] = p0[gid];` (MSL copy) | refuses |
| spirv | `p1[gid] = p0[gid];` (GLSL copy) | refuses |
| hexagon | `hvx_vcopy_f32(in0, out, n)` (identity copy) | refuses |
| gaudi | `v_f32_ld_tnsr` then `v_f32_st_tnsr`, nothing between | refuses |
| gpu, nki, tpu, bang | a `TODO` comment, then the SSA name leaked into code | refuses |

Nine of the thirteen emitters now refuse at the library level, where before this thread one
did. `musa` came along with the shared change rather than needing its own.

Three were left detector-only at first -- `aie`, `ttmetal` and `csl` -- on the grounds that
none emits a plausible COPY and each has an unfamiliar structure. That reasoning was
sound and the conclusion was still wrong: "obviously incomplete on inspection" assumes
somebody inspects it, and the whole point of this issue is that nobody does. They were
finished the same day.

    csl       task compute_task() void {}          an EMPTY task
    ttmetal   MAIN with its buffers and no compute
    aie       an IRON program whose core body omits the op

Each is source the vendor toolchain accepts and the device will run. All three shared the
`Op::Unknown`-discarded or `_ => { /* skip */ }` shape after all, so the change was the
mechanical one; only reading enough of each file to be sure of that was the work.

**All twelve in-tree emitters now refuse at the library level.** `pico`, the thirteenth,
lowers the op properly. `no_emitter_can_silently_ignore_an_intrinsic_any_more` still holds
the line for whatever is added next.

`is_structural_intrinsic` is now shared rather than copied. Two definitions of "which
intrinsics do not select a kernel body" would drift, and the first symptom would be one
emitter refusing a kernel another accepts.


## correction: `__tile_relu_f32` IS declared (2026-09-02)

The note that opened this issue says:

> `__tile_relu_f32` does not exist in tile_std at all -- only `__tile_relu_f32_4` and
> `__tile_relu_f32_scalar` -- so that is an unknown name.

That is wrong. It is declared, in the same extern block as the rest of the tile-level
unary ops:

    crates/tile_std/src/tile.rs:2073
        pub fn __tile_relu_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

The `_4` and `_scalar` variants exist too; the tile-level name was there all along and I
did not look past the first two matches.

The fix is unaffected -- no emitter has an arm for it, so refusing is right either way --
but the CHARACTERISATION changes, and for the worse. This was filed as "two different
causes, one symptom": an unknown name and a handled name whose arm did not fire. There is
only one cause. Both `__tile_relu_f32` and `__tile_matmul_f32` are **declared intrinsics
that the emitters do not implement**, and the surface tile_std advertises is wider than
what any backend delivers.

That gap is now measurable, because refusing is what makes it visible. For the declared
tile-level unary ops, lowering a minimal kernel for each:

| op | msl | gpu | pico |
|---|---|---|---|
| sigmoid, exp, log, silu | yes | yes | yes |
| sqrt | yes | no | yes |
| softplus | yes | no | no |
| relu, tanh | no | no | yes |
| neg | no | no | yes |
| abs | no | no | no |

Six of ten on Metal, four on CUDA, eight on PICO. Every "no" in that table used to be a
silent copy, which is why nobody had counted them.

`abs` is declared and implemented by none of the three.


## closing the Metal half of the gap (2026-09-02)

Measuring it was the point of the refusal work; four of the six gaps on Metal were then
one arm each, because `emit_unary_msl` already existed:

    relu   max(x, 0)      written out: not a one-argument call
    abs    fabs(x)        NOT `abs`, which is MSL's integer overload
    tanh   tanh(x)
    neg    -x             an operator, not a call

`__tile_neg_f16` had an arm and `__tile_neg_f32` did not, which is what made neg lower at
one dtype and refuse at the other. Metal is now 10/10 on the declared tile-level unary ops.

`abs` and `neg` are verified on the GPU against the CPU reference and torch -- all three
agree. `relu` too, after `-r` caught a real bug in the arm I had just written:

    error: C-style cast from rvalue to reference type 'decltype(p0[gid])'
           (aka 'const device float &')

`decltype(p0[gid])` is a REFERENCE, so casting a literal to it is invalid. Binding the
load to a value first fixes it. That is the run harness doing exactly what it is for --
the source looked right, and only compiling it said otherwise. `tanh` has no reference in
the harness, so it is lowered and unverified, and the report says so rather than implying
otherwise.

CUDA is still 4/10 and the same four arms would close most of it. Not done here: this
change is worth being sure about per backend, and Metal is the one with a device attached
to check against.
