# Single-backend data movement, and why the CLI cannot describe it

Three things `cannbench-tilers` needed from `tile` on 2026-10-24. One of them is
**already implemented at four layers and announces its own gap**; one has no form; one has
no surface at all. The common thread is not that tile-rs lacks the capability — it is that
the CLI's vocabulary and its *discovery* surface are both organised around portable
arithmetic, while all three of these are single-backend data movement.

**Filed as a spec and not as backlog entries, deliberately.** `tile backlog` exits 3 with
`STOP AND RECONSIDER: "emit-structure" (3), "lowering" (7)`, and `backlog/README.md` says
the right response is to propose the redesign rather than add a fourth entry.

## The three

    1  an int8 matmul with the dequant folded into its OUTPUT TRANSFER
    2  a dedup whose output EXTENT is computed on the device
    3  a question about what a kernel SHOULD cost

## One design problem

**`--list-ops` describes a portable subset, and the capability that matters for Ascend is
single-backend.** The table's own footer says so: *"The denominator is this probe's 32
ops, NOT the whole intrinsic surface, which is much larger. The set is the portable
tile-level ops; single-backend kernel families ... are deliberately outside it, not
overlooked."* That is a correct design decision for a portability table and it leaves no
surface that answers **"what can `pto` do for me"**.

The cost is measured, on me, in one session: I read the 32-op table, found no `quant`,
`dequant`, `scale`, `fixpipe` or `epilogue` row, and concluded the fused int8 path was not
expressible. **It is declared in `tile_std` as
`__tile_matmul_i8_acc_i32_dequant_f16`** — the dequant is in the name — with
`__tile_dequantize_i8_f32` and `__tile_quantize_f32_i8` beside it, `FixPipeDequant` at
four sites in `mlir_to_pto.rs`, and `pto.tdequant` / `set_quant_vector` among the **624**
`pto.*` names in `ptoas.so` on 910b. Four layers of implementation, invisible from the
one command a caller runs to ask.

The second half of the problem is the vocabulary rather than the table: **a transfer is
not an op, and an extent is not a value.** §2 and §3 are both that, and so are four
already-filed issues, in three different areas:

    #020  lowering    scatter and gather write and read at an index no backend checks
    #028  target      an arch-conditional field assignment is deleted, not diagnosed
                      (the fields are `fp.params.*` -- Fixpipe IS the transfer config)
    #029  target      a target's im2col instruction is unreachable, so conv lowers to a
                      materialised column matrix
    #040  lowering    a transposed matmul operand cannot carry a runtime K -- "DN is
                      inferred from a static stride"

`#029` is the clearest statement of it: the op is expressible, the *movement* is not, and
the fallback materialises the thing the instruction existed to avoid.

## 1. The fused int8 dequant matmul: implemented, undiscoverable, and refused on small shapes

**What is MEASURED, through the installed binary (release, `~/ascend-rs-priv/crates/tile_cli`,
features `[]`, installed 16:00 2026-10-23):**

    __tile_dequantize_i8_f32              pto   LOWERED   output differs from a
                                                          load/store-only module
    __tile_matmul_i8_acc_i32_dequant_f16  pto   REFUSED   matmul_i8: single-block path
        unsupported (M=256 K=0 N=0); extend detect_blocked_matmul_loads +
        translate_matmul_i8 to cover small-shape i8 matmul

So the standalone dequant reaches the backend, and the fused form refuses **by name, with
the two functions to extend printed in the message**. That is the standard the emitters are
held to — a gap that announces itself — and it is the opposite of a silent copy.

**Caveat that must travel with that refusal: the probe passed `K=0 N=0`.**
`op-leaves-a-trace.sh` builds a generic module, so part of "single-block path unsupported"
is the probe's shape and not ours. What the message does establish is that
`translate_matmul_i8` **exists** and has a blocked path. Whether `quant_matmul`'s real
shapes take it is **UNKNOWN from this machine** — `doctor` reports Apple M1 Ultra with no
Ascend accelerator, so `pto` here is cross-generation emission with no harness.

**Why it is worth the work, priced from silicon rather than estimated.** `quant_matmul` on
the 950pr 2.0 board scores **32.4916** = compile 5.0 + function 25.0 + performance
**2.4916 of 70**, at 20/20 with `avg_speedup` **0.1364** (landed 2026-10-24). Our pipeline
is **three kernels where the vendor has one**, and the job record names them with their
microseconds — case 12 reads `qm_i8h_kernel` 78.87, `mm_dyn_512_f32` 354.68,
`qm_post_kernel` 21.16, summing to 454.71 which IS that case's `elapsed_us`. The first is
literally `I8ToHalf`, the third applies the scales afterwards, so the "int8 quant matmul"
never touches the int8 cube. Implied cube throughput **41-63 TMAC/s against the vendor's
217-362**. Eliminating the two extra passes is 22% of that case and models at **+2.07
points**; both run at 797 and 1191 GB/s, i.e. bandwidth-saturated and therefore eliminable
rather than optimisable.

**What this spec asks for, in order of cost:**

* **A discovery surface for single-backend families.** `tile --list-ops --backend pto`, or
  a `--list-intrinsics <form>` that prints the whole surface for one target with its
  lowering status, so the answer to *"can pto fold a dequant"* is one command and not four
  greps across three repositories. This is the item that actually cost a session.
* **Extend `detect_blocked_matmul_loads` + `translate_matmul_i8`** to the small-shape
  path, which the refusal already names. Not specified further here: the message is more
  precise than anything this document could add.
* **A composition gate, not a cell.** `--list-ops` probes each cell with a single op, and
  `SKILL.md` trap 3 records 42 of 42 `spirv` intrinsics emitting `p1[gid] = p0[gid]` at
  exit 0. The acceptance test for the fusion is
  `op-leaves-a-trace.sh pto __tile_dequantize_i8_f32 -b __tile_matmul_i8_acc_i32_dequant_f16`
  — the `-b` form — and it must additionally assert **one** kernel in the output, because
  emitting two is precisely the defect being removed.

## 2. An output extent computed on the device

**Measured.** `unique`'s float32 path holds **99.5%** of that operator's device time —
3,621,425 of 3,639,255 us across four of twenty cases — at `sum(ps)` **0.0001**, which is
**15,980x to 25,087x** off the vendor baseline, or **0.4-0.63 us per element** on a dedup.
The int/fp16 path of the same operator scores 0.16-0.94 on the same metric, so this is the
float32 route specifically: float32 does not fit one bitmap, so it partitions on the key's
top 12 bits and runs a 2^20 bitmap per bucket.

**The extent is the point.** The unique count is known only after the kernel runs, so the
plugin reads it back (`cnt.cpu().item<int32_t>()`) and *then* shapes the output with
`narrow`. No op in the portable set has an output extent that is a device-computed scalar,
and the nearest neighbours — `argmax`, `absmax`, `reduce_max`, `reduce_sum`,
`init_sort_buf`, `mxfp4_value` — all produce a statically-known shape. `#020` is the same
gap from the index side; `#040` is it from the stride side.

**This spec does NOT propose a `dedup` op.** The achievable rate is unknown, so an op
specified now would bake in our two-level bitmap, which is the thing under suspicion. What
is specified is the **capability**: an output whose extent is a value the kernel produces,
with the host-side readback part of the contract rather than re-invented per plugin. The
budget constraint is real and belongs in that contract — the harness zeroes an operator
that makes more than five `aclrtMemcpy` per case, so a readback is affordable and not free.

**One thing that is NOT the lever, recorded so it is not offered as one.** That path runs
on a single AI core — `nb32 = 1` with the comment `// DEBUG: cores out of the picture`.
Flipping it to the core count is worth **+0.0129 points** against **+7.00** for reaching
parity on those four cases, because twenty cores removes 20x of a 25,000x deficit.

## 3. A question about what a kernel should cost

There is no surface. The full flag list is `--cross --daemon --force --from --help
--info-only --iterations --json --keep --keep-dir --list-forms --list-ops --model
--offline --no-install --route --run --stats --tile --to --ui --ui-port --verbose
--version --via --yes`; `-m/--model` provisions a ds4-rs engine and is unrelated. `-i`
reports what a kernel **is** — dtypes, extents, tile plan, hazards, bounds — never what it
**should cost**, and `--cross` answers *will this fit*, which is a different question.

**Why that cost something concrete.** Both items above were nearly mis-prioritised and
only arithmetic done outside the tool caught it: the single-core stub reads as the obvious
fix and is worth 0.2% of the prize, and `unique`'s four cases hold 99.5% of the device time
while being worth +7.00 of 70 — because the score subtracts the hardware floor from both
sides, so **device-time share is not score share**. A tool that ranked by device time would
have sent the work to the wrong place twice in one session.

**Requirements rather than a design, because the design is in flight.**
`crates/tile_codegen/src/roofline.rs` and `hardware.rs` are in this tree **untracked** —
another lane's live work, 5 Ascend references in `roofline.rs` and 0 in `hardware.rs`. So:

* it must **refuse** rather than extrapolate when the arch's numbers are not held. The
  precedent to avoid is `--cross`, which accepts any string and checks nothing for
  anything but `ascend`;
* it must report a **score** delta and not only a time delta, or it reproduces the
  device-time-share error above. That makes the objective an input, so it is a parameter
  and not a constant;
* it must distinguish **bandwidth-saturated** (eliminable) from **compute-bound**
  (optimisable), because that distinction is what made §1's two extra passes a certain
  +2.07 rather than a hope.

## What cannot be specified yet, and why

* **The 950 cube's real MAC rate.** 41-63 against 217-362 TMAC/s are both *implied* from
  job-record timings, not measured peaks. An earlier claim that the K-step was "26x off the
  cube's arithmetic peak" was withdrawn precisely because that denominator was never
  measured on this hardware.
* **What a 1M-element float32 dedup should cost.** The only reference is the platform's
  `baseline_perf_us`, which is a stored constant and on some cases a proxy cap
  (`max(10 * t_hw_us, 10)`), so it cannot be read as the vendor's time.
* **Whether `pto.tdequant`'s signature matches `DequantSpec`.** The name is in `ptoas.so`
  on 910b; the signature was not probed, and `ptoas 0.58` removed `pto.tstore_fp`, so this
  lowering is assembler-version dependent.
* **Whether `quant_matmul`'s shapes take the blocked `matmul_i8` path.** Needs a
  well-formed i8 matmul module at our real shapes, emitted with `--cross ascend`, and
  ideally run where an Ascend part is.

## Order

1. **The discovery surface in §1.** It is the cheapest item, it is the one that actually
   cost a session, and until it exists every future reader repeats the mistake this
   document opens with.
2. **`translate_matmul_i8`'s small-shape path**, which the refusal already names, with the
   composition gate as its test.
3. **§3's requirements**, which cost nothing to adopt and would have prevented two
   mis-prioritisations.
4. **§2** last, and as a capability rather than an op — its achievable rate is the one
   number nobody holds.

**The single sentence: a portability table cannot answer a single-backend question, and
for Ascend work that is the only question being asked.**
