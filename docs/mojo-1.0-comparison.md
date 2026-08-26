# tile-rs next to Mojo 1.0

Mojo's compiler and toolchain were open-sourced on 18 August 2026, which makes a
like-for-like comparison possible for the first time. Apple GPU is the one target
both projects support, so that is where the head-to-head runs.

Sixteen kernels: five primitives, and eleven from the decode path of
DeepSeek-R1-Distill-Qwen-1.5B.

## Apple M1 Ultra — tile-rs → Metal vs Mojo

Timed on the GPU's own clock (`MTLCommandBuffer.gpuStartTime`/`gpuEndTime`), so
host submit-and-wait is outside the measurement. Median of 5 blocks of 40
launches, minimum within each block. Mojo built with optimisation on.

| kernel | tile-rs → Metal (µs) | Mojo (µs) | ratio |
|---|---:|---:|---:|
| `vec_add` | 44.1 | 45.4 | 1.03 |
| `vec_mul` | 44.1 | 45.4 | 1.03 |
| `vec_sub` | 44.1 | 45.3 | 1.03 |
| `vec_exp` | 34.2 | 35.5 | 1.04 |
| `residual_add` | 44.1 | 45.4 | 1.03 |
| `rms_norm` | 49.3 | 64.6 | 1.31 |
| `rope` | 29.9 | 33.1 | 1.10 |
| `argmax` | 274.7 | 281.5 | 1.02 |
| `matmul` | 244.6 | 272.0 | 1.11 |
| `q_proj` | 30.5 | 34.2 | 1.12 |
| `k_proj` | 30.2 | 32.2 | 1.07 |
| `v_proj` | 30.5 | 32.2 | 1.05 |
| `o_proj` | 30.6 | 33.6 | 1.10 |
| `down_proj` | 45.4 | 45.9 | 1.01 |
| `gate_up_silu` | 147.6 | 152.5 | 1.03 |
| `attn_gqa` | 8616.9 | 9801.1 | 1.14 |

**15 of 16 faster, 1 tie, median gap 4.7%.**

Read the corpus total with care: 9.74 ms against 11.00 ms is 1.13x, but
`attn_gqa` is 88% of that total, so the corpus figure is close to a single
kernel's result. The per-kernel column is the honest view.

## Ascend 910B2 — five backends from one source

Every column below is generated *from tile-rs*. The question is which backend
tile-rs should lower to for a given kernel, not whose language is faster.

| | tile-rs (best-of) | Ascend C | PTO | TileLang | Triton |
|---|---:|---:|---:|---:|---:|
| total-time ratio | 51.08x | 1.62x | 2.28x | 1.42x | 22.33x |
| lines of code, all 16 | **223** | 1276 | 1605 | 498 | 283 |

Three caveats, none of them small:

* **It is a total-time ratio**, not a geometric mean or a median of speedups, so
  the expensive kernels dominate. Triton reads 22.33x while being fastest on only
  5 of 16. TileLang is fastest on **8 of 16** — more than any other backend — and
  has the largest total, because its launch path is the leanest on the part while
  its scans are single-core.
* **The tile-rs column is a best-of**, therefore theoretical: the fastest
  generated kernel per row against the slowest. Achievable in principle, since
  each kernel is compiled independently; not achievable if a deployment needs one
  backend throughout.
* **Triton and TileLang reach the NPU through their own Ascend C generation**, so
  those columns measure a longer path than the direct Ascend C column.

## What the resource-safety claim looks like in practice

All four Ascend backends must respect a 192 KB vector unified buffer. Three of
the four got it wrong, each differently — and each was caught at a different
distance from the hardware:

| backend | how the limit was missed | who caught it |
|---|---|---|
| PTO | budget constant set *above* the hardware's | the assembler |
| Triton | gated on Triton's representational limit, not the buffer | the compiler |
| TileLang | no check at all | the device, mid-run |
| Ascend C | — | it asks the hardware |

A budget that is merely wrong is caught by the next tool down. A budget that is
absent is caught by the chip. That ordering is the argument for putting the
constraint in the type system, where it is checked before anything is emitted.

## Reproducing

Scope for every figure above: Ascend 910B2 with CANN 8.5.2, and Apple M1 Ultra
(128 GB), model DeepSeek-R1-Distill-Qwen-1.5B. Each kernel is verified against a
host reference before it is timed; the Apple figures additionally carry a
correctness check per kernel.

Benchmark entries and the kernel corpus are published at
[pu-rs.org](https://pu-rs.org).
