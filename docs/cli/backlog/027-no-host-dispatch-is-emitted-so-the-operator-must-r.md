id: 27
area: emit-structure
title: no host dispatch is emitted, so "the operator must reach the device" cannot be a codegen invariant
opened: 2026-09-08

## wanted
tile-rs to own the HOST side of an operator (the torch dispatch that computes a
tile plan, allocates outputs and launches), not just the kernel. Two properties
would then hold by construction instead of being audited afterwards:

  1. TOTALITY OF THE TILE PLAN. tiles = ceil(n/T) is 0 when n is 0, and a
     zero-tile launch is a launch. The defect only exists because a human wrote
     `if (n == 0) return out;` AROUND the plan. If the emitter owns the
     dispatch there is nowhere to write that.

  2. A LAUNCH OBLIGATION. Every path out of the dispatch performs at least one
     launch. This is NOT derivable from the dataflow: an identity permutation
     has no work to do and returning the input is semantically correct. It is a
     property of the emitted TRACE, so it wants an effect/linear encoding --
     `launch()` mints a witness, the dispatch's return type demands one, and a
     path that returns without launching fails to typecheck.

## got
tile-rs emits kernels only. crates/rustc_codegen_tile/src/ has mlir_to_pto.rs
for the Ascend path and no TORCH_LIBRARY / RunOpApi / op_plugin emission
anywhere in crates/ (grep finds none). tile_spec and tile_codegen carry no
witness, linear or effect machinery to hang the obligation on.

## workaround
bench/lint_launch.py in the cannbench-tilers tree: it finds the function
TORCH_LIBRARY_IMPL registers, plus any same-file helper that itself launches,
and flags a tensor-shaped return reached before the first RunOpApi. It gates
syncall. Found 31 real instances across three widenings.

What the fix will need, learned from those widenings -- these are exactly the
distinctions a linear/effect type would make STRUCTURALLY, and each one cost a
separate round of false negatives to discover by hand:

  * a helper that returns one of its ARGUMENTS is a pass-through and carries no
    obligation; its caller still launches. (castF32 returning an already-f32
    weight.) -> a borrow, not a produced value.
  * a helper that ALLOCATES an output and returns it early WAS the device path
    and skipped it. -> must discharge the obligation itself.
  * the deferred pattern, `*deferred = acl_call; return out;`, hands the launch
    to the caller to run in a batch. -> a linear token the caller must consume
    exactly once. This is the case a naive "did it launch here" check gets
    wrong in both directions.
  * multi-output operators return std::make_tuple(...) or {a, b}; matching only
    `return <name>;` missed every one of them.

Cost of not having it: an operator that answers correctly without launching is
scored ZERO in full by the grader's anti-cheat -- all 80 hidden cases, compile
and function scores included. Transpose lost 63.31 and StridedSlice 65.64 to
one such return each.

What does NOT lift, and should not be promised: whether the grader's profiler
OBSERVES the launch is empirical and external. A type system can guarantee a
launch is emitted, not that it is counted. And the identity-transpose policy
(alias the input, or copy it on device) is a semantic choice a human makes; the
type system enforces it once chosen.
