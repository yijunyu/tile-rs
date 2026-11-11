id: 22
area: lowering
title: 64 intrinsics are lowered and declared nowhere; 46 of them by MSL, and they are the coverage gap
opened: 2026-09-04

## wanted
Every intrinsic an emitter lowers to be declared somewhere, so that a caller can
be type-checked against it and a test corpus can be built from it.

## got
They are not. The MSL emitter matches 301 `__tile_*` names outside its test
module; `crates/` declares 406; **46** of the 301 appear in no declaration
anywhere. Across ALL backends the unique figure is **64**, of which 46 are
reachable through MSL and **18 are lowered only by other backends**:

    backend     names  undeclared
    msl           301          46
    bang           82          19
    tpu            72          15
    nki            72          15
    gpu            72          15
    aie            71          13
    gaudi          73          12
    pto            69           8
    spirv          76           6
    linalg         50           5

`hexagon`, `ttmetal` and `csl` are absent because they match only PREFIXES, not
names -- see the note on the number. Among the 46:

    __tile_attention_gqa_f32        __tile_attn_decode_splitk
    __tile_attn_decode_batched      __tile_attn_decode_splitk_v2
    __tile_attn_decode_combine      __tile_attn_decode_v4_batched
    __tile_add_inplace_batched      __tile_kv_write_batched
    __tile_absmax_f16               __tile_argmin_f32
    __tile_cast_bf16_f32            __tile_const
    __tile_fill_f16                 __tile_get_rows_f32          ... and 30 more

Note `__tile_get_rows_f32` is undeclared while `__tile_get_rows_f32_strided` IS
declared -- they are different intrinsics, not a spelling of one.

## why it matters, with the measurement
This is the same surface as #021's coverage gap, seen from the other side.

`mlir_to_msl.rs` holds 6223 of the 9031 uncovered emitter lines, and **3746 of
those 6223 are in one function**, `generate_func_msl` -- whose uncovered lines are
the `match ctx.kernel_type` arms for kernel types nothing constructs. The next
worst are `emit_attention_gqa_msl` (381), and the `attn_decode_*` batched arms.

Those are exactly the names above. The declared-intrinsic corpus added this
session is built FROM the declarations, so it reaches 241 intrinsics and cannot
reach these 46 at all. They are uncovered because they are undeclared.

And #019 is one instance of the consequence: with no declaration, each emitter
documents the signature for itself, and two of them already disagree about
`__tile_attention_gqa_f32`'s argument order.

## workaround
None. A corpus can be extended to these 46 only by taking the arity and operand
kinds from the emitter's own match arms -- which tests the emitter against input
derived from the emitter. That is legitimate for a COMPILE check (does it emit a
program?) and worthless for a correctness one, and the distinction must be stated
wherever it is done.

## what the fix needs
Declare them in `tile_std`, or say why they should not be. Then the corpus reaches
them for free and #021's gap closes from the same work.

Two of the 46 had NEVER COMPILED, found by asking only "is this a program?":
`fill` wrote through a `device const` pointer, and `MatmulF16Simdgroup` passed
`float*` buffers to `simdgroup_load` against `simdgroup_half8x8` accumulators.
Both fixed; `undeclared_intrinsics_still_emit_a_program` keeps them checked.

## a note on the number
It was 54, then 53, then 44, then 47, and it is **46**. Every move was the
instrument, never the code:

* 54 counted string literals inside the emitter's own test module, where
  `__tile_frobnicate_f32` is a placeholder rather than an arm.
* 53 checked declarations with an unanchored `grep`, which matched
  `__tile_get_rows_f32` as a prefix of the genuinely-declared
  `__tile_get_rows_f32_strided` and called it declared.
* 44 used a `[a-z0-9_]+` character class for intrinsic names, which silently
  excluded the 13 K-quant kernels -- `__tile_mul_mm_id_q2_K_f32`,
  `__tile_mul_mv_gate_up_swiglu_q4_K_f32` and their siblings. **Uppercase.** They
  are the DS4 quantized matmuls, which is to say the exact family this issue is
  about.

* 47 counted `__tile_matvec_quant_` as a name. It is a PREFIX, matched with
  `name.starts_with(..)` rather than `==`, and nothing declares a prefix. The tell
  is the trailing underscore. It is also why `hexagon`, `ttmetal` and `csl` each
  appeared to lower seven undeclared intrinsics: all seven of their literals are
  prefixes, and their true count is zero.

46 is what the committed gate measures, and it prints its own inputs -- 301 names
matched, 406 declared -- so the next person can watch the number move rather than
wonder about it.

## the next rung
18 of the 64 are lowered only by backends other than MSL, so
`undeclared_intrinsics_still_emit_a_program` cannot reach them. The C-family,
SPIR-V and mlir-opt gates could, on the same terms: is it a program, not is it
right.
