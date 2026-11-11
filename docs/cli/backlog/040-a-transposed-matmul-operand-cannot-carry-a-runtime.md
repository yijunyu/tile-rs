id: 40
area: lowering
title: a transposed matmul operand cannot carry a runtime K -- DN is inferred from a static stride
opened: 2026-09-19

## wanted
C[M,N] = A[M,K] * B^T[N,K] on ascend a5 with K a kernel argument. transB == 1
is the most frequent single reason the matmul family misses the cube on 950PR.

## got
SUPERSEDES the "cannot block" half of #039, which is done: eaaa625d blocks the
transposed path, M=16..512 all lower, and ptoas assembles all ten kernels
(rc=0) including the M=256 pair that previously refused on L0A.

ccec then refuses every one of them:

  tload_common.hpp:350: static assertion failed
    GlobalTensor<half, Shape<1,1,1,64,64>, Stride<64,64,64,1,-1>, Layout::ND>
    Tile<Mat, half, 64,64, BLayout::RowMajor, SLayout::ColMajor, ...>
  Fix: TLOAD(MatTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ/ND2NZ/DN2ZN!

A transposed B reaches CBUF through DN->ZN, the only supported transposed-MAT
combo, and Layout::DN is INFERRED from the emitted stride template. Measured on
one emitter, one day, two variants of one kernel:

    static K    strides = [%c1, %c1024]   ->  Stride<64,64,64,1,1024>  DN   ok
    runtime K   strides = [%c1, %kk]      ->  Stride<64,64,64,1,  -1>  ND   refused

-1 is ptoas's dynamic stride. For a row-major view the unknown extent lands in
the stride that is not 1, so the view stays classifiable; for the TRANSPOSED
view the unknown extent IS the non-unit stride.

## workaround
None that ships. The route is backed out of cannbench-tilers (kernels, CMake,
header, plugin); the regenerated PTO and a new bench/wrap_pto_kernel.py are
kept, so re-landing it is three commands once K is expressible.

What the fix needs:
  - The transposed operand's K must reach the generated C++ as a COMPILE-TIME
    constant, so the Stride<> template is fully known and DN is inferred.
    Either specialise kernels per (M,K) -- grouped_matmul's transposed failures
    are K=1024 at M=256 and M=512, so two entries cover the measured cases, but
    it does not generalise to the other eleven matmul operators -- or teach the
    emitter to hoist a transposed operand's K into a template parameter on the
    kernel rather than an argument. The second is the one that generalises.
  - Note the historical trap this explains: make_tv_transposed wrote
    strides = [%c1, %c256] and therefore always typed as DN. Those constants
    were the BLOCK STAND-INS, so the old route compiled and computed one
    M x 256 x 64 corner (NOTES 312). Correctness and typeability were being
    traded against each other and nothing in the tree said so.
