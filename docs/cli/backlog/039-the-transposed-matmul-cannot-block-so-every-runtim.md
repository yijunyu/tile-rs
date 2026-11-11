id: 39
area: lowering
title: the transposed matmul cannot block, so every runtime-shaped transB refuses
opened: 2026-09-19
closed: 2026-09-19

## wanted
C[M,N] = A[M,K] * B^T[N,K] on ascend a5, with K and N arriving as kernel
arguments. transB == 1 is the single most frequent reason the matmul family
misses the cube on 950PR, and on that part nothing else serves those shapes.

## got
translate_matmul_transposed has no blocked path. translate_matmul routes a
dynamic shape to translate_matmul_blocked; the transposed twin read the
BLOCK stand-ins that generate_func_pto puts in const_map (PTO_MM_KB=256,
PTO_MM_NB=64) as if they were the caller's extents and emitted one fixed
M x 256 x 64 tile. That lowered and returned a matrix wrong everywhere except
one corner (MERE 1.109 / 1.154 at M=256/512, K=1024, N=2048) until commit
4107af47 made it refuse:

  matmul_transposed: K and N are a runtime kernel argument, and the
  transposed lowering does not block.

## workaround
Refuse, and fall to the vector path. The alternative -- materialising B^T in
GM and using the non-transposed route -- costs a full transpose pass and
gives back most of what the cube won.

What the fix needs, concretely:
  - translate_matmul_blocked takes its operands as `deferred` loads with a
    tv_ssa + elem_offset and rebuilds a partition_view per K/N block. The
    transposed B does not arrive that way: translate_matmul_transposed calls
    make_tv_transposed(b_gm, n, k) to build a [K,N]-shaped, [1,K]-strided
    tensor_view over the N x K GM buffer, then make_pv on that. Blocking it
    means offsetting WITHIN the transposed view -- block (kb, nb) is rows
    kb*KB..+KB and cols nb*NB..+NB of the transposed view, which are cols and
    rows respectively of the underlying buffer.
  - The DN->ZN tload is the only supported transposed-MAT TLoad combo (see
    the make_tv_transposed doc at mlir_to_pto.rs:1374), so the blocked B tile
    must stay ZN while the blocked A tile stays NZ. The existing blocked
    emitter assumes both are NZ.
  - The accumulator is f32 for f16 operands (acc_dtype_for); the blocked
    emitter already does a per-block FixPipe store, which is what lets the
    f32 L0C work across blocks.
  - Known-good anchor to regress against: a peer tree's blocked transposed
    path validates M=16 K=512 N=32 at 9.611e-08, STATIC. (A second anchor,
    2.573e-07 at M=32 K=4096 N=512, was WITHDRAWN by that fork's owner on
    2026-09-20: its commit is not an ancestor of their HEAD and emits a
    different function shape.) Static is the
    discriminator -- a static transposed matmul was always correct here too,
    which is why the defect stayed invisible for so long.
