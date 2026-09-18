# The `run` cluster, and what it forced

`tile backlog` stopped on three open issues in one area. This is what reading them
together produced, recorded because the point of the rule is to change something, not to
note that a threshold was crossed.

## The issues

* **#001** — no run harness for any target but Metal. *Partly wrong when written: a
  Vulkan device was on this machine the whole time, and `-t spirv -r` runs on it now. Open
  for CUDA and the rest, which this Mac genuinely cannot reach.*
* **#010** — f16 kernels are compared against inputs they were never given.
* **#012** — "validated on <hw>" is printed for hardware this repo cannot reach.
* **#013** — a fixed tolerance cannot judge an operation whose own precision is not fixed.
  **Closed 2026-09-03.**

(The heading said "The three" over four bullets. Kept as a note rather than quietly
corrected, because a count restated beside the thing that knows it is the exact failure
this cluster is about.)

The first looks like a missing feature and the third like a documentation slip. They are
the same problem, and #010 and #013 are that problem again one level down: the report is
pitched wider than the measurement in target, in dtype, and in operation.

## One design problem

**The tool reports verification at a granularity its harness cannot deliver.**

`run` can measure exactly one configuration: a Metal kernel, on an Apple GPU, with f32
buffers, against a CPU reference and torch. Everything the tool SAYS about verification is
pitched wider than that, and each issue is one axis of the gap:

* **Wider in target.** The fidelity class `validated on <hw>` asserts an on-hardware check
  for eight backends. The harness reaches one of them — #001 states this plainly, and #012
  is what happens when the report does not.
* **Wider in operation.** Within f32, the comparison is only sound for ops whose error does
  not grow with their input. A K-term reduction's does, and the fixed tolerance is one this
  tool's OWN reference fails on a matmul — so for those ops the gate measures nothing about
  the kernel, which is #013.
* **Wider in dtype.** Within Metal, the comparison is only sound for f32. For f16 the
  harness quantizes the kernel's inputs and computes both references at full precision, so
  the verdict "the KERNEL disagrees with torch — look at the lowering" is produced by the
  harness's own input handling — #010.

Both come out as the same user experience, and it is the one this tool exists to prevent:
*tile told me something was verified, and the verification did not cover what I ran.* The
three-way comparison was built precisely so absence of evidence could not be rendered as
evidence of agreement, and these are three places where the reporting layer does it anyway
— above the comparison rather than inside it.

## What changes, and what cannot here

What can be fixed without hardware has been:

* The f16 report now states that the kernel's inputs were rounded and the references' were
  not, so no one is sent to audit a lowering for arithmetic it did not do. The underlying
  mismatch stays open in #010 because closing it is a decision about which inputs are
  canonical, not a bug fix.
* `--run` on a kernel the harness cannot dispatch exits 3 instead of 0, so a caller cannot
  read "the run did not happen" as "the run agreed".
* The op count is now measured — 14 of the 15 the harness has references for — rather than
  quoted as the number of references.

What cannot be fixed here is the shape of the claim. A fidelity class is a measurement
claim printed on every conversion, and the project's own D4 rule says an unmeasured bound
must refuse rather than print a plausible number. The same rule applied to fidelity gives
the answer: a `validated` class should carry where it was measured, the way
INTEGRATION.md does for the three Apple GPUs, or say that it was inherited. Choosing
between those needs whoever owns the hardware record for the other seven backends.

**So the rule this cluster produces:** a claim about verification names the configuration
it was verified in. It is the same rule the coverage figure needed — "175 of 178" drifted
because it was quoted without the host that produced it — arriving a second time from a
different direction, which is a reasonable sign it is the right rule.

## What closing #013 settled, and what it did not

The cluster's one sentence was *the tool reports verification at a granularity its harness
cannot deliver*. #013 was that in the OPERATION dimension, and it is now closed by making
the tolerance a property of the operation and its dtype rather than a constant:
`error_budget(op, shape, dtype)` returns Wilkinson's bound, computed from `f32::EPSILON`
and the terms actually summed.

Two of the three dimensions are now closed by measurement rather than by wording:

* **target** — #001's Vulkan half, by building the harness that was claimed impossible.
* **operation** — #013, by deriving the bound instead of choosing it.

**dtype remains open (#010).** And the first attempt at #013 got dtype wrong in a new
place: the budget used f32's roundoff for an f16 kernel and promoted a correct lowering to
"the KERNEL disagrees with torch". The budget takes the dtype now. That this cluster's own
remaining dimension is where the fix for another dimension failed is worth stating plainly
rather than filing as a fourth issue.
