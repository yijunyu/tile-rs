id: 12
area: run
title: "validated on <hw>" is printed for hardware this repo has no way to reach
opened: 2026-09-02
closed: 2026-09-03

## wanted
To check the poster's "3 GPUs verified on", which holds -- `mac` (M2 Max), `mini` (M4) and
`studio` (M1 Ultra) are all in INTEGRATION.md with detected chip names and timings. While
there, I read what the neighbouring fidelity claims assert.

## got
`design.md` defines the class exactly:

> `validated` | additionally checked against a CPU reference on that hardware -- the 8
> On-HW backends

and `forms.rs` carries nine of them:

    CPU, Ascend 910B (x2), Apple GPU, NVIDIA, Vulkan/MoltenVK, Trainium trn1,
    TPU v5e, Gaudi

`tile --list-forms` prints these, and so does the last line of every single conversion:
`route: tile -> mlir -> msl [exact, validated on Apple GPU]`. It is, by the design doc's
own account, "worth more than anything else the tool prints".

Backlog #001 says: "no run harness for any target but Metal... this Mac has no CUDA device
to write one against." INTEGRATION.md says the same in the CUDA section: "There is no
NVIDIA device on this machine." So this repo cannot have run a kernel on a TPU v5e, a
Trainium trn1, a Gaudi or an NVIDIA card, and there is no record in it of anyone having
done so. The evidence it does hold is for Apple GPU (fifteen ops, three machines) and CPU
(linalg).

## why this is filed and not changed
The claims may well be TRUE. These kernels came from projects that do have the hardware --
910B work in particular is real and lives in another tree -- so "validated on Ascend 910B"
may be a fact this repo simply does not carry the evidence for. Downgrading nine fidelity
classes on the strength of my not finding proof HERE would replace a possibly-true claim
with a definitely-wrong one, and it is not my call which of the eight were actually run.

What is wrong either way is that the claim is unattributed. The tool's own D4 rule is
"unmeasured refuses -- the CLI propagates that refusal instead of printing a plausible
number", and a fidelity class printed on every conversion is exactly a measurement claim.
It should say where the measurement lives.

## two ways out, for whoever owns the hardware record
  (a) Point each `Validated(hw)` at its evidence -- a date, a machine, a run -- the way
      INTEGRATION.md does for the three Apple GPUs. Then the claim is checkable.
  (b) Split the class: validated-here versus validated-upstream, so a reader can tell a
      claim this repo can defend from one it inherited.

## workaround
Trust `validated on Apple GPU` and `validated on CPU`; those have runs recorded in
docs/cli/INTEGRATION.md. For the other six, read the class as inherited rather than
demonstrated.

## one of the eight is now substantiated (2026-09-03)

`Validated("Vulkan/MoltenVK")` was in the list of claims this repo carried no evidence for.
It now has some: the spirv softmax was run on Mesa's Vulkan-on-Metal driver on this machine
and checked against a CPU reference at max abs 7.229e-10 (#016, and the run that found the
defect it fixed).

That is one kernel, on one driver, on an Apple GPU — not the general claim the fidelity
class makes, and the class still says nothing about where it was measured. The evidence is
in docs/cli/INTEGRATION.md and reproducible with assets/harness/vulkan.c, which is exactly
the attribution option (a) above asks for. Seven backends still have none.

## closed (2026-09-03) — by option (a), attribution

Not by deciding which of the eight were really run. That judgement is still not mine and
the issue was right to say so. The claim is unchanged; it now says where its evidence is.

    Fidelity::Validated(&'static str, Evidence)
    Evidence::Here(&'static str)   a run recorded in this repo, at the named place
    Evidence::Inherited           validated by a project with the hardware; nothing here

What the route line prints:

    route: mlir -> mlir -> msl   [exact, validated on Apple GPU (docs/cli/INTEGRATION.md, 3 machines)]
    route: mlir -> mlir -> nki   [exact, validated on Trainium trn1 — inherited, no run recorded here]

`tile doctor --targets` was two groups with all nine validated backends in the first, under
the heading "measured — emitted AND checked against a reference on that hardware". It is
three now: **measured HERE** (linalg, msl, spirv), **claimed UPSTREAM** (pto, gpu, nki,
tpu, gaudi and the second Ascend route), and **unmeasured**.

A multi-hop route degrades: its hardware name comes from the last hop, but its evidence is
`Here` only if EVERY validated hop is. Taking the last hop's evidence would let a route
passing through an unattributed backend end up citing a file that says nothing about it.

An evidence pointer must name a file that exists, and a test opens every one — a pointer
that cannot be followed is worse than no pointer, because it looks checkable. That test
failed on its first run, on a path this very change had got wrong
(`assets/harness/vulkan.c` is relative to the crate, not the repo root).
