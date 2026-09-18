# `tile` on the AMD box

For the `tilers-amd` session on the Ryzen AI Max+ 395 (Strix Halo). Written here because
that session was not reachable when this was prepared — `~/amd/PLAN.md` has sshd down as
B1 — so this is what it needs on the way back up.

## The correction that matters most

**AMD is two devices with two different answers, and only one of them is a tile-rs
target.**

| device | tile-rs target | status |
|---|---|---|
| **XDNA2 NPU (NPU2)** | `aie` — IRON / MLIR-AIE Python | **lowers today, on any machine** |
| **Radeon 8060S (gfx1151)** | none | there is no HIP/ROCm backend in tile-rs at all |

Asking `tile` for a HIP kernel is not a gap in the CLI; it is a backend nobody has
written. `tile doctor` reports an `amd-gpu` as present, defaults the output form to the
CPU `linalg` bridge, and says why — deliberately, because defaulting a Radeon to `aie`
would emit NPU source for a GPU.

## What works right now, with no hardware and no dual-boot

Verified on an M1 Ultra, which has no AMD anything:

```
tile k.mlir -t aie -o k.py        # real IRON: ObjectFifo, Runtime, SequentialPlacer
```

The emitted program **defaults to `NPU2Col1()`** — XDNA2, which is this box — and takes
`npu` as its first argument to select `NPU1Col1()` (Phoenix) instead. softmax and the f16
matmul both lower to real IRON; they are not the fall-through copy described in backlog
#005.

So every part of the skill except running is usable before the box is fixed: identify,
profile (tile plan, RAW/WAR/WAW edges, barrier points, resource bounds), `--route`,
lower, `-O0..-O2`, `-s`, and the backlog.

## What is blocked, and on what

* **`-r` (run + measure)** — there is no AIE harness; the harness is Metal-only today
  (backlog #001). This is blocked twice over: on the harness, and on B1/B2.
* **`-O3` (toolchain feedback)** — `aiecc.py` / MLIR-AIE is Linux-only, so there is
  nothing to ask until the Ubuntu dual-boot in `~/amd/PLAN.md` is done. `tile` reports
  the compiler as *unavailable* and continues; it never reports the kernel as rejected
  because the machine is missing a tool.
* **The ROCm baseline** (`ds4.c`) — same dual-boot dependency, and unrelated to tile-rs
  since there is no HIP target to compare against.

## First three things when the box is up

1. `tile doctor` on Linux. It should find `amd-gpu` via `rocm-smi` and `amd-npu` via
   `xrt-smi`, report them as **two separate families**, and default to `linalg`. If it
   reports one, or defaults to `aie`, the probe is wrong — that is `platform/linux.rs`
   and it has fixture tests but has never met the real hardware.
2. `tile k.mlir -t aie -O3 -o k.py` once `aiecc.py` is installed. That is the first
   real check that what tile-rs emits for this NPU actually compiles.
3. An AIE harness for `-r`, if there is a way to dispatch and read back a buffer. The
   Metal one (`crates/tile_cli/assets/harness/metal.swift`, ~60 lines) is the shape: the
   harness dispatches, prints values and `#us` device times on stdout, and `tile` owns
   the reference and the comparison — so a second harness is a transcription rather than
   a second numerics implementation.

## The protocol, in one line

Try `tile` first. When it genuinely cannot do the job, `tile backlog add` **before** you
work around it, solve the problem by hand, then write the workaround back into the entry —
that is what the eventual fix starts from. `~/.claude/skills/tile-rs/SKILL.md` has the
detail. Off this machine, set `TILE_BACKLOG` to a shared path or `tile backlog` will tell
you it is reading somewhere else.
