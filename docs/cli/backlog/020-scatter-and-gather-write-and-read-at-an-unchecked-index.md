id: 20
area: lowering
title: scatter and gather write and read at an index no backend checks
opened: 2026-09-04

## wanted
`__tile_scatter_f32` and `__tile_gather_f32` take an index array of arbitrary
runtime `u32`. The emitted kernel should not write or read outside the buffer
when one of those indices is out of range.

## got
No bounds check in any of the nine backends that lower them. The Metal body is
representative:

    uint idx = (uint)p1[gid / stride];
    uint col = gid % stride;
    p2[idx * stride + col] = p0[gid];      // scatter: unchecked WRITE
    p2[gid] = p0[idx * stride + col];      // gather:  unchecked READ

On a GPU an out-of-range `idx` here is an out-of-bounds device write, not a wrong
answer.

The parameter that would bound it is already being passed and is read by nobody.
`tile_std` declares

    __tile_scatter_f32(dst, src, indices: *const u32, n, m, d)

and its safe wrapper states what `m` means, which the declaration does not:

    pub fn tile_scatter_f32<const N, const M, const D>(
        src: Tile<N, D, f32>, indices: *const u32,
    ) -> Tile<M, D, f32>

Source is N x D, destination is **M x D**. So `m` is the destination row count --
exactly the bound the check needs. `aie` and `gaudi` carry it in their op structs
and never read it (rustc: "field `m` is never read"); `gpu` writes
`let _m = ...`, silencing the question.

The repo already knows the pattern. `__tile_topk_mask_scatter_f32`'s own doc
comment reads: "for gid in [0, num_elements), idx = topk[gid]; if 0<=idx<dst_len,
dst[idx] = 0.0." That one checks. The general scatter does not.

## workaround
None that does not change the kernel ABI, which is why this is filed rather than
fixed. The Metal scatter kernel's parameters are `num_elements` and `stride`; the
destination extent `m` is NOT among them, so the guard cannot be written without
adding a buffer binding, and the host that binds those buffers is outside this
repo. Gather is the same in mirror: it loops over the OUTPUT extent, so the
source bound it would need is likewise absent.

## the same class, elsewhere
Two more found the same way, both in `mlir_to_gaudi.rs`, both reported by rustc as
a never-read field:

* `Embedding { vocab_size }` -- the token from a runtime indices buffer indexed
  the weight table with no check, and `vocab_size` was destructured as
  `vocab_size: _`, carried into the op and explicitly thrown away. **Fixed**, not
  filed: unlike scatter this needs no ABI change, because the bound is already a
  compile-time constant in hand beside `count` and `embed_dim`. An out-of-range
  token now reads as zero, which is what an embedding of an invalid token should
  be rather than whatever was in memory.
* `Slice { src_r }` -- the source row extent, carried and never read. **Fixed**,
  and more cheaply than either: the slice reads
  `src[(_si + row_off) * src_c + (_sj + col_off)]` for `_si < dst_r`, so it touches
  row `row_off + dst_r - 1`, and EVERY term is a compile-time constant. A slice
  that provably runs off its source is now refused at emit, at no cost in the
  emitted kernel.

## where the evidence lives (this changed, and why)
`field \`m\` is never read`, twice in `mlir_to_aie.rs` and twice in
`mlir_to_gaudi.rs`, is the standing evidence for THIS issue.

It was first left as a bare warning, with an instruction here NOT to silence it.
That was wrong once the cost was measured: CI compiles `tile_spec` with
`-D warnings`, and the coverage workflow on `main` was RED because the crate did
not compile. A warning left standing to be seen is worth a line of output; it is
not worth a red build, which hides far more.

So each of the four now carries `#[allow(dead_code)]` **with a doc comment saying
the field is carried and never read, that this is the finding, and pointing
here**. The evidence is preserved in a form that does not cost a build.

Remove the `allow` when the bound is threaded through and read -- not before, and
not to tidy up.

## what the fix needs
* Thread `m` (scatter) and `n` (gather) into the kernel signature for each
  backend, and guard the index against it -- the same shape as
  `topk_mask_scatter`'s existing check.
* That is a signature change on nine emitters plus their host bindings, so it
  wants a decision rather than a patch.
