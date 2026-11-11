id: 1
area: run
title: no run harness for any target but Metal
opened: 2026-09-01

## wanted
`tile k.cu -t gpu -r` — run a lowered CUDA kernel and check it, the way the Metal
path does.

## got
"-r has no harness for \"gpu\" yet." The harness is Swift over Metal; nothing
equivalent exists for CUDA, Ascend, Vulkan or the rest.

## workaround
None available in-session: this Mac has no CUDA device to write one against. The
Metal harness in assets/harness/metal.swift is the shape to copy — it reports values
and device times on stdout and lets `tile` own the reference and the comparison, so a
CUDA version is a host program with the same output protocol, not a second numerics
implementation.

## narrowed 2026-09-03 — SPIR-V has one now

This said "no run harness for any target but Metal", and nobody had checked whether the
other targets were really out of reach. One was not: this machine has the Vulkan loader,
Mesa's Vulkan-on-Metal driver (KosmicKrisp 26.2.0, Apple M1 Ultra, Vulkan 1.4.354),
`glslangValidator` and `spirv-val`.

`assets/harness/vulkan.c` runs SPIR-V compute against the same input generator the Metal
harness uses. It found a real defect the first time it ran — the softmax returned 224
non-finite values of 1024 — and verified the fix at max abs 7.229e-10 (#016).

It is NOT wired into `-r` yet; `run.rs` still drives metal.swift only. So what remains true
of this issue is narrower than what it claimed: no harness for CUDA, Cambricon or Gaudi,
for want of those cards. The claim that Metal was the only reachable target was untested,
and it was wrong.

## correction (2026-09-03): "nothing equivalent exists for Vulkan" was wrong
Not narrowed by new work -- wrong when written. This machine had `glslangValidator`,
`spirv-val`, the Vulkan loader and Mesa's KosmicKrisp driver (Vulkan 1.4.354 on an Apple
M1 Ultra) the whole time. Nobody checked before writing "nothing equivalent exists".

`crates/tile_cli/assets/harness/vulkan.c` is that harness, satisfying the same argv and
stdout contract as `metal.swift` down to the input generator, and `-t spirv -r` runs on the
device. 19 SPIR-V ops are verified there.

What remains true is the CUDA half: this Mac has no NVIDIA device, and no amount of looking
changes that. The issue stays open for `gpu`, `musa`, `nki`, `tpu`, `aie` and the rest --
but its title and its body both claimed more than the facts supported, and the same
sentence was used to refuse work that was in fact possible. That is the failure this
backlog exists to record, and it happened in the backlog itself.
