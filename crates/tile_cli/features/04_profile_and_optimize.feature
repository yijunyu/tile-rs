Feature: One input with no output means profile, then optimize for this machine
  The single-argument invocation is the tool's front door. It must tell the user
  what the kernel IS before it changes anything, and it must produce something
  runnable on the machine in front of them — not a cross-compiled artifact they
  cannot execute.

  Background:
    Given a clean working directory
    And a tile-rs kernel "softmax.rs"

  Scenario: a single input profiles and then optimizes by default
    Given a tile-rs kernel in a fresh directory
    When I convert it with no output named
    Then the profile reports the kernel, its dtypes and its tile plan
    And the artifact has a derived name beside the input
    And an optimized artifact for the detected native target is produced
    And the optimization level used is reported as "O2"

  # The default invocation must not clobber. The derived name is the input stem plus
  # ".opt" plus the output form's extension, written beside the input.
  Scenario: an existing artifact is refused, and --force says that it overwrote
    Given a tile-rs kernel in a fresh directory
    When I convert it with no output named
    And I convert it again
    Then the existing artifact is refused rather than overwritten

  Scenario: --force overwrites, and says that it did
    Given a tile-rs kernel in a fresh directory
    When I convert it with no output named
    And I convert it again with --force
    Then the report says an existing file was overwritten

  Scenario: -i stops after the profile
    When I profile with --info-only
    Then a profile is printed and nothing is written

  Scenario: -i and -o together are a usage error, not a silent precedence rule
    When I run "tile softmax.rs -i -o out.metal"
    Then the command fails with exit code 2
    And the error states that --info-only produces no output

  # The levels are separated by WHAT THEY MAY TOUCH, so cost and reproducibility are
  # predictable from the number alone.
  Scenario Outline: each -O level runs the passes it is permitted and no others
    When the pipeline runs at level <level>
    Then the passes that ran are "<passes>"
    And running it twice gives byte-identical output

    Examples:
      | level | passes                                                                  |
      | 0     |                                                                         |
      | 1     | dead-op-elimination                                                     |
      | 2     | dead-op-elimination, constant-dedup, common-subexpression, redundant-load |
      | 3     | dead-op-elimination, constant-dedup, common-subexpression, redundant-load |

  Scenario: a level names what it does not do, and why
    When the pipeline runs at level 2
    Then the report names fusion as the emitters' job, not as missing work

  Scenario: the rewrite layer parses and prints without changing a byte
    Given the canonical kernel
    When it is parsed into the IR and printed back
    Then the result is byte-identical to the input

  Scenario: a pass that separated a fusion pair reverts to the input
    Given a kernel whose silu and mul have been separated
    When the pipeline runs at level 2
    Then the input is returned unchanged
    And the report says the optimization was reverted

  Scenario: -O2 refuses a tiling the target cannot hold
    Given a kernel whose tile exceeds what "ascend" can hold
    When the bounds are checked
    Then a bound violation is reported
    And the violation names the rule and the operation

  Scenario: -O2 refuses to judge a tiling on an unmeasured architecture
    Given the target is "ascend-950", which nobody has measured
    When the bounds are checked
    Then the check refuses rather than approving or rejecting

  @requires-toolchain
  Scenario: -O3 asks the target's own compiler
    Given the target's compiler is installed
    When I run "tile softmax.mlir -t msl -O3 -o out.metal"
    Then the compiler's verdict is reported
    And the conversion still produced its output

  Scenario: -O3 without the target's compiler degrades rather than failing
    Given the target's compiler is not installed
    When the toolchain check runs
    Then it reports the compiler as unavailable, not the kernel as rejected
    And it states that the conversion still ran

  Scenario: bare -O means -O2
    When I run "tile softmax.rs -O -o out.rs"
    Then the optimization level used is reported as "O2"

  Scenario: the default level is reproducible
    When I convert the same input twice at the default level
    Then the two outputs are byte-identical

  Scenario: -O4 requires a device and refuses honestly without one
    When I ask for -O4 on a machine with no accelerator
    Then it fails with the device code and offers -O3 instead

  # The simulation must apply what it announces, or every test above it is vacuous.
  Scenario: a forced absence is applied, not merely printed
    Then a simulated absence is actually applied, not just announced

  # The axis is threadgroup width: the one thing this layer can vary without changing
  # what the kernel COMPUTES. Re-chunking a tile is not in that class and is still only
  # CHECKED, never rewritten.
  @requires-device
  Scenario: -O4 records what it measured and invents nothing
    Then every candidate O4 tried is recorded with its own measurement
    And a single timing leaves the headroom unassessed, and only the sweep writes one

  # A width must be CORRECT before it can be fastest. The sweep used to time each
  # threadgroup width and never read the output buffer -- it exits before printing a
  # value -- so the recommendation written into the emitted source rested on speed
  # alone. That is the wrong test for this axis: every reduction here folds across
  # `tcount`, and the power-of-two tree fold produced WRONG ANSWERS at thread counts
  # that were not powers of two.
  @requires-device
  Scenario: -O4 verifies a width before recommending it
    Then each candidate width is checked against the reference before it is ranked
    And a width that computes a different answer is dropped, with the reason said
    And when no width reproduces the reference, none is recommended at all
    And an op with no reference leaves the ranking stated as timing-only, not removed
    # -O4 output is deliberately NOT byte-reproducible: it carries a measurement, and a
    # measurement is not a pure function of its input. The default level still is.

  # `f10513c` — HardwareParams/TargetSemantics carry `measured`, and every query
  # refuses when it is false, because a wrong lowering is silent. The CLI is the
  # first user-facing surface for that refusal, so it must propagate it, not
  # smooth it over into a plausible-looking number.
  Scenario: an unmeasured architecture makes the profile refuse, not guess
    Given the selected target's HardwareParams are not measured
    When I run "tile softmax.rs -i -t cpp --cross ascend-950"
    Then the resource-bound section reports "unmeasured" for every bound
    And no UB, repeat, stride or cube-tile verdict is printed
    And the error explains that the architecture has not been measured

  # 950PR and 950DT are different machines behind one ISA target (NOTES-dt-pr-diff).
  # `GetCoreNumAiv()` has never been printed from a graded job on either, so the
  # profile must not invent 910B's 48-core grid — and the DT refusal names the SKU
  # so a PR-measured ceiling cannot be mistaken for a DT measurement.
  Scenario: the 950 SKUs refuse bounds and invent no core count
    Given the selected target is "ascend-950dt", which nobody has measured
    When I run "tile softmax.rs -i -t cpp --cross ascend-950dt"
    Then the resource-bound section reports "unmeasured" for every bound
    And the tile plan reports an unknown core count rather than a 910B grid
    And the error names the DT silicon id

  # Exit 3, not the 1 this originally asked for: measuring that chip is work nobody has
  # done, which is what 3 means. 1 would file it as a failure of this run.
  Scenario: an unmeasured architecture blocks optimization above O1
    Then an unmeasured architecture refuses to report a bound rather than borrowing one

  Scenario: a target with genuinely no such limit answers Ok rather than refusing
    Given the selected target is "msl", which has no unified buffer and no repeat field
    When I run "tile softmax.rs -i -t msl"
    Then every resource-bound check passes
    And the bounds are reported as measured with a limit of zero, not as unmeasured

  # On MLIR, where the SSA form makes the dependences readable. The same report for a
  # `.rs` input needs the frontend that lands in M6, so that variant stays @planned.
  Scenario: the profile surfaces the hazard model
    When I run "tile softmax.mlir -i"
    Then the profile lists the RAW, WAR and WAW edges between vector ops
    And it lists the barrier points required to make the schedule safe
    And any edge left unsynchronised is reported as a defect, not a note
    And the profile reports the hazard model with its unsynchronised count

  Scenario: the same hazard report is produced for a tile-rs input
    Then the hazard model is reported for a tile-rs input too
