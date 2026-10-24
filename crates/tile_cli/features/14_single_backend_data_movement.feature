Feature: Single-backend data movement, which a portability table cannot describe
  `--list-ops` measures 32 PORTABLE tile-level ops across every backend, and its own
  footer says single-backend kernel families are deliberately outside it. That is right
  for a portability table and it leaves the Ascend question unanswerable: a reader read
  that table, found no `quant`, `dequant`, `scale`, `fixpipe` or `epilogue` row, and
  concluded a fused int8 dequant matmul was not expressible -- while `tile_std` declares
  `__tile_matmul_i8_acc_i32_dequant_f16` with the dequant in its name and the `pto`
  emitter carries arms for it. Four layers of implementation, invisible from the one
  command a caller runs to ask.

  The other three scenarios here are the rest of `docs/cli/spec-data-movement.md`: a
  transfer is not an op and an extent is not a value, so the fused epilogue, the
  device-computed output extent and "what should this cost" each need a surface the tool
  does not have yet. They carry @planned, and the tag comes off in the change that makes
  each one pass.

  Scenario: the whole intrinsic surface for ONE backend is one command away
    When I ask "pto" for its intrinsic surface
    Then the fused int8 dequant matmul is listed with an arm
    And the surface is far larger than the portable op probe
    And the report says it is derived from source and names the measured instrument
    And asking does not change what --list-ops reports
    And a backend whose emitter source this build cannot read is UNKNOWN, not zero

  # ptoas consumes the pto dialect and emits C++ -- but over pto/pto-inst.hpp, not over
  # kernel_operator.h. Folding that into `cpp` would be wrong twice: a different intrinsic
  # layer, and one `cpp`'s reader cannot parse, so forcing the form yields a confident
  # profile of nothing instead of a refusal.
  Scenario: the C++ ptoas emits is a form of its own, not AscendC
    When I identify a kernel that ptoas emitted
    Then its form is "ptoc" and not "cpp"
    And a real AscendC kernel is still identified as "cpp"
    And there is no direct route to it yet, and asking says so rather than crashing

  # The edge belongs in the graph the day tile invokes ptoas as a toolchain step, and not
  # before: `emit.rs`'s own gate refuses an edge with neither an emitter nor a feature
  # gate, because that is "a route the planner offers and nothing can take, and the user
  # would meet it as a crash rather than a refusal". Today the only thing that performs
  # pto -> ptoc is ptoas on the command line.
  @planned
  Scenario: a pto kernel lowers to the C++ ptoas emits, through the toolchain
    Given a kernel in the pto dialect and a provisioned ptoas
    When I lower it to "ptoc"
    Then the route taken is pto -> ptoc and names toolchain:ptoas
    And on a machine without ptoas it is refused as ACQUIRABLE, not as unwritten

  @planned
  Scenario: the fused int8 dequant matmul emits ONE kernel, not a matmul and two passes
    Given an i8 matmul module at a shape the single-block path refuses today
    When I lower it to "pto" asking for the dequant folded into the output transfer
    Then the emitted source contains exactly one kernel
    And the dequantisation appears in that kernel's output transfer, not in a second pass
    And a shape the blocked path cannot take is REFUSED by name, with the functions to extend

  @planned
  Scenario: an output whose extent is a value the kernel computes
    Given a dedup kernel whose output length is known only after it runs
    When I lower it to "pto"
    Then the output extent is carried as a device-computed scalar, not as a static shape
    And the host-side readback is part of the lowering contract rather than per-plugin code
    And the contract states the per-case memcpy budget it must stay inside

  @planned
  Scenario: what a kernel SHOULD cost, reported as a score delta and not a time delta
    Given a kernel and a target whose measured capacities this build holds
    When I ask what it should cost
    Then the answer separates bandwidth-saturated work from compute-bound work
    And it reports a score delta against a stated objective, not device-time share alone
    And for a target whose numbers are not held it REFUSES rather than extrapolating
