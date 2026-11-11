id: 11
area: lowering
title: two Metal specializations bake in a shape that is not an operand
opened: 2026-09-02

## wanted

(Revised 2026-09-02 after the first version of this issue reached the wrong conclusion;
the correction is below. The date lives here because the frontmatter takes no `updated`
field -- and filing one anyway is how this issue spent a day invisible to `tile backlog`.)
An audit of preconditions the emitters document and do not enforce, after the CUDA
reduction bug turned out to be labelled `(assumes N<=32)` in the source it emitted.

## got
Fifteen emitters scanned. Eleven have none. `gpu` had the CUDA one (fixed). `aie`'s single
hit tells the code to use the runtime width constant, which it does. `pto` is the
disciplined case: it refuses a matmul whose M is not 16-aligned and names the requirement.

`msl` had a family of shape-specialized kernels that baked in `head_dim` 64 or 128,
`n_embd` 4096, `n_hc` 4, `n_tokens` 1, or `K % 8 == 0`, and checked none of it.

## correction: the first version of this issue was wrong
It said the shapes could not be checked, because every one of these emitters has the
signature `fn emit_..._msl(out: &mut String)` -- no shape arguments. That is true of the
emitters and false of the problem. The INTRINSICS carry the shape:

    __tile_flash_attn_ext_vec_score_f32(q, k, v, mask, sinks, pad, dst,
                                        dk, dv, ne01, ne11, nb01, nb11, scale)

`dk` is right there, and `classify_body` was already resolving operands through
`const_map` for other kernels. The emitter was handed the head dimension and ignored it.
Filing this as unfixable was a wrong conclusion reached by looking only at the emitter's
signature.

Four kernels are now guarded by `reject_specialization_mismatch`, between classification
and emission: `FlashAttnExtVecScore` and `FlashAttnExtVecOut` (dk=dv=64),
`Dsv4HcSplitWeightedSumNorm4` (n_embd=4096, n_hc=4) and `Dsv4Q8HcExpand4Q8_0` (n_hc=4,
n_tokens=1). A proven mismatch refuses and names the specialization.

Only a PROVEN mismatch. `resolve_const` cannot be used for a guard: its `parse_const_arg`
fallback strips the `%` off an SSA name and returns the digits, so `%42` comes back as 42
-- a number that looks like a value and is not one. `known_const` consults `const_map` and
integer literals only, so a dynamic shape still lowers. `msl_specialization_guard.rs` pins
that case as well as the refusals, because a guard that refused dynamic operands would
break working callers while looking thorough.

## what is still open, and why
Two, both because the shape is genuinely not an operand:

`emit_flash_attn_ext_pad_msl` bakes in n_head=64, head_dim=128, ntg=128. Its intrinsic
takes only byte strides -- `nb11`, `nb12`, ... -- and no head dimension. head_dim could be
INFERRED (`nb11 == head_dim * sizeof(half)`, so 256), but that guard would rest on my
reading of the layout convention, and if K is ever f32 rather than f16 it refuses a
correct caller. Not guessed at, for the reason the CUDA episode established.

`emit_matmul_f16_simdgroup_msl` documents `K must be a multiple of 8`. The intrinsic it is
selected by, `__tile_matmul_simdgroup_f16`, is not declared in `tile_std` at all -- no
Rust caller can reach it, only hand-written MLIR, and K is not among its operands.

Both need someone who knows the intended layout contract. Neither is reachable from the
Rust surface with a wrong shape today.

## workaround
For those two: use them only at the shape each was specialized for.
