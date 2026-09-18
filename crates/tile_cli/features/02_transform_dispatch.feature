Feature: What the tool does is a function of the input and output forms
  Different forms mean a lowering (down the stack) or a lifting (up it);
  identical forms mean an optimization. All three are edges in one
  transformation graph, so "no route" is a first-class, explained answer
  rather than a crash or a silent copy.

  Background:
    Given a clean working directory
    And a tile-rs kernel "softmax.rs"

  @requires-backend
  Scenario: a lower is chosen when the output form sits below the input
    Given the codegen backend is provisioned
    When I run "tile softmax.rs -o out.metal"
    Then the route taken descends from the tile-rs source to the target
    And the output contains "kernel void"

  @requires-lifter
  Scenario: a lift is chosen when the output form sits above the input
    Given a Metal kernel "softmax.metal"
    And a lifter exists for "msl -> tile"
    When I run "tile softmax.metal -o lifted.rs"
    Then the transformation kind is "lift"
    And "lifted.rs" parses as Rust
    And "lifted.rs" carries the tile-rs kernel attribute
    And re-lowering "lifted.rs" to "msl" succeeds
    And the fidelity class is "synthesised"

  Scenario: with no reader for the input form, a lift says which work is missing
    Given no lifter exists for "msl"
    When I run "tile softmax.metal -o lifted.rs"
    Then the command fails with exit code 3
    And the error states that there is no lifter for "msl -> tile"
    And the error says the frontend has not been written yet, not that it cannot exist
    And the error points at "tile --list-forms" for what can be read

  Scenario: identical forms mean optimization, not a copy
    Given a kernel with a dead definition in it
    When I run "tile softmax_dead.mlir -o out.mlir"
    Then the transformation kind is "optimize"
    And the output has form "mlir"
    And the command does not merely copy the input bytes
    And the report names the passes that fired

  Scenario: -o wins over any extension inference
    When I name an output whose extension says nothing
    Then the named file holds the requested form

  # Not "at least 2": for most pairs today the graph offers exactly one route, and a
  # scenario demanding two would be asserting against a graph nobody has built. What
  # --route owes is that every candidate is listed with what it needs, and that asking
  # costs nothing.
  Scenario: routes are enumerable with their requirements, and asking writes nothing
    When I ask for the routes without converting
    Then each candidate route is listed with its requirements, and nothing is written

  Scenario: --via pins the intermediates, and a pin nothing satisfies is refused
    When I pin an intermediate that is not on the route
    Then the contradiction is refused rather than ignored

  # Lift edges do NOT compose into lowering routes automatically. A synthesised
  # intermediate silently feeding a lowering is precisely the invisible-wrongness
  # failure this tool exists to prevent, so a route that passes through a lift must
  # be asked for by name.
  # Written against the ONE lift that exists today (pto -> tile). The msl variant is
  # the same rule with a lifter nobody has built; it stays @planned rather than being
  # asserted against a premise the graph cannot satisfy.
  Scenario: a lift does not silently compose into a lowering route
    Given a lifter exists for "pto -> tile"
    When I run "tile softmax_pto.mlir -o out.metal"
    Then the command fails with exit code 2
    And the error states that no direct route exists from "pto" to "msl"
    And the error offers the composed route "pto -> tile -> mlir -> msl" behind "--via"
    And the error states that the composed route is only as good as the lift
    And the refusal reaches the command line too, not only the planner
    And asking for it by name still refuses, because no lifter is written

  Scenario: the composed route runs when it is named explicitly
    Given a lifter exists for "pto -> tile"
    When I run "tile softmax_pto.mlir --via tile -o out.metal"
    Then the route taken is "pto -> tile -> mlir -> msl"
    And the fidelity class is "synthesised"

  # @requires-lifter, not @planned: the form rule says a SOURCE form owes no reader, so
  # an msl frontend is a product somebody would have to decide to build. Filing it as
  # planned work would keep it on the ledger as debt forever.
  @requires-lifter
  Scenario: the same rule holds for a target-source lifter once one exists
    Given a lifter exists for "msl -> tile"
    When I run "tile softmax.metal -o out.cce"
    Then the command fails with exit code 2
    And the error offers the composed route "msl -> tile -> cpp" behind "--via"

  # Backlog #005, fixed at its root in mlir_to_msl.rs. An unrecognised intrinsic left the
  # kernel type at its Copy default and the emitter wrote a copy of the first buffer:
  # source that compiles, that xcrun metal accepts, and that computes something else --
  # reported as [exact, validated on Apple GPU] with exit 0.
  Scenario: an intrinsic the emitter does not understand is refused, never copied
    Then an intrinsic the emitter does not handle is refused, not copied
    And a kernel that really is a copy still lowers

  Scenario: an unreadable input names the missing capability, never a silent no-op
    When I convert a target source to another target source
    Then it names the missing frontend in the tense of work not yet done

  Scenario: a target missing from this build fails distinguishably from one nobody wrote
    When I ask for a target this build was not compiled with
    Then it says so as an acquirable absence and names the build feature

  Scenario: two inputs of different forms are two jobs, not a route request
    When I profile two inputs of different forms
    Then each is profiled on its own and no conversion between them is attempted

  # tile-rs has ~1.5 readers (tile-rs source, MLIR text) and 16 writers. The tool is a
  # fan-out, not yet a swissknife, and the sparse graph must be DISCOVERABLE or users
  # will read "edge does not exist" as "tool is broken".
  Scenario: the reader/writer matrix is one command away
    When I run "tile --list-forms"
    Then every form is listed with whether it can be read, written, optimized and lifted from
    And the output makes clear that most forms are write-only

  # "Unsupported", not "impossible" — and the two are different exit codes. A missing
  # frontend is a piece of work nobody has done; the refusal has to carry that tense, or
  # a reader concludes the pair can never work and stops asking.
  Scenario: same-form optimization names the missing reader, in the right tense
    When I run "tile softmax.metal -o softmax2.metal"
    Then the command fails with exit code 3
    And the error states that tile-rs can emit "msl" but cannot read it
    And the error says the frontend has not been written yet, not that it cannot exist
    And the error shows the route that would work if one existed
    And the error names the forms for which optimization does exist

  Scenario: a target absent from THIS build is a different answer from one nobody wrote
    Given this build was compiled without the "cpp" emitter
    When I run "tile k.mlir -t cpp -o out"
    Then the command fails with exit code 4
    And the error names the feature that would enable it
    And the exit code differs from the one used when no frontend exists

  Scenario Outline: same-form optimization exists only for the forms tile-rs owns
    When I run "tile k.<ext> -o k2.<ext>"
    Then the outcome is "<outcome>"

    Examples:
      | ext   | outcome                      |
      | rs    | optimized                    |
      | mlir  | optimized                    |
      | metal | refused, no reader for msl   |
      | cu    | refused, no reader for gpu   |
      | cce   | refused, no reader for cpp   |

  # A kernel that is numerically wrong is invisible until it corrupts a run, so every
  # conversion states how much the result should be trusted.
  #
  # And "validated" says WHERE it was validated (backlog #012). Nine backends carried the
  # class and this repo holds a run for three; the rest came from projects that do have the
  # hardware, so the claims may be true. They are not downgraded -- they are attributed, so
  # a reader can tell a claim this repo can defend from one it inherited.
  Scenario Outline: every route reports a fidelity class, and says where it was measured
    When I run "tile softmax.rs -t <form> -o out --route"
    Then the route's fidelity class is "<fidelity>"

    Examples:
      | form   | fidelity                                                              |
      | msl    | exact, validated on Apple GPU (docs/cli/INTEGRATION.md, 3 machines)   |
      | gpu    | exact, validated on NVIDIA — inherited, no run recorded here          |
      | csl    | exact, unvalidated on hardware                                        |
      | ttmetal| exact, unvalidated on hardware                                        |

  Scenario: the fidelity class is printed on every conversion, not only with --route
    Then every conversion prints its route and fidelity, not only --route

  @requires-lifter
  Scenario: a lifted result is marked as synthesised and requires review
    Given a lifter exists for "pto -> tile"
    When I run "tile k.mlir -f pto -o k.rs"
    Then the fidelity class is "synthesised"
    And the output carries a header stating it was lifted and needs review
