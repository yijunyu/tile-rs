id: 19
area: transform
title: __tile_attention_gqa_* is lowered by five backends, declared by none, and read two ways
opened: 2026-09-04

## wanted
One statement of what `__tile_attention_gqa_f32` takes, so that the five emitters
lowering it are lowering the same thing.

## got
No declaration. It is not in `crates/tile_std/src/tile.rs`, which declares the
other 423 intrinsics, and it appears in no file outside `mlir_to_*.rs`. Every
emitter therefore documents the signature for itself, and two of them disagree:

    mlir_to_nki.rs:1741   (dst, q, k, v, heads, head_dim, seq_len, kv_heads)
    mlir_to_tpu.rs:1368   (dst, q, k, v, heads, head_dim, seq_len, kv_heads)
    mlir_to_gpu.rs:1771   (dst, q, k, v, num_heads, num_kv_heads, seq_len, head_dim)

Positions 5, 6 and 8 mean different things in the two readings. One of them is
building kernels from the wrong operands, and nothing in the tree says which.

Separately, and independent of the order question: `nki` and `tpu` do not
IMPLEMENT the grouping at all. nki resolves `kv_heads` and never uses it; tpu
writes `let _kv_heads = ...`, which silences the compiler rather than answering
it. Both then emit ordinary attention. `msl` and `gpu` do implement it --
`const uint group_size = num_heads / num_kv_heads;`.

The nki arm's own comment says what it does not do:

    // Grouped-Query Attention: Q has `heads` heads, K/V have `kv_heads` heads.
    // Decomposed into: for each group, scores = Q @ K^T, ...

There is no "for each group" in the code below it.

## workaround
`nki` and `tpu` now REFUSE a GQA call whose `kv_heads` differs from `heads`,
rather than emitting ungrouped attention that runs and computes something else.
Where the two are equal, GQA degenerates to multi-head attention and the existing
lowering is correct, so those calls still emit.

That is deliberately the smaller half of the fix. The refusal is well defined
inside nki and tpu's own reading of the signature; it does not settle which
reading is right, and this issue stays open until something declares the
intrinsic.

## what the fix needs
* A declaration in `tile_std`, or a statement from whoever wrote these arms, of
  the argument order. There is no caller to infer it from.
* Then: either implement grouping in nki and tpu, or have them refuse GQA
  outright rather than only in the unequal case.
