Feature: -r runs the generated kernel and says what it measured
  Emitting a kernel proves it compiles. Running it proves it computes the right answer,
  and is the only way to say anything about speed that is not a guess.

  Two comparisons, because they answer different questions and are constantly confused:
  against a reference (what the accelerator bought you) and against -O0 (what the
  optimizer bought you). And a third opinion — PyTorch on the CPU — because a kernel
  and a reference from the same repository can agree perfectly and both be wrong.

  Scenario: a speedup is never reported without both measurements
    Given a run where the optimizer changed nothing
    Then no optimizer ratio is reported
    And the report says the two sources were identical

  Scenario: a reported speedup names what it is measured against
    Given a completed run
    Then the report states that the baseline is a naive single-threaded scalar loop
    And the report states that this is not a GPU-versus-CPU figure

  Scenario: every timing carries its sample count and its spread
    Given a completed run
    Then each timing reports the number of runs, the minimum and the maximum
    And it reports how many warmup runs preceded them

  Scenario: the reference does not allocate inside the timed region
    When the reference timing loop is inspected
    Then no allocation happens between the clock starting and stopping

  Scenario: the kernel, the reference and torch all see the same input
    When the three input generators are compared
    Then they compute the same values by the same rule

  Scenario: a kernel whose operation has no reference is refused, not run
    When a kernel using an operation with no reference is offered
    Then it is refused with the operations that do have one
    And the refusal says numbers with nothing to compare them against are worse

  Scenario: a kernel using two referenceable operations is refused rather than guessed at
    When a kernel using two referenceable operations is offered
    Then it is refused because the composition order would have to be guessed

  Scenario: without torch the weaker comparison is still made, and labelled
    Given torch is not installed
    Then the report still compares the kernel against this tool's reference
    And it states that torch was unavailable and the claim is weaker

  Scenario Outline: three sources attribute a disagreement to the right side
    Given the kernel is "<kernel>" to torch and this tool's reference is "<ours>"
    Then the verdict is "<verdict>"

    Examples:
      | kernel | ours  | verdict                |
      | close  | close | all three agree        |
      | far    | close | the KERNEL disagrees   |
      | far    | far   | reference is wrong     |
      | close  | far   | does not follow it     |

  Scenario: a kernel that matches nothing is not called an inherited error
    Given the kernel is "far" to torch and this tool's reference is "far"
    And the kernel does not match this tool's reference either
    Then the verdict does not blame the reference alone

  Scenario: two sources cannot make the attribution three sources can
    Then the verdict differs between a wrong kernel and a wrong reference

  Scenario: a relative error near zero does not dominate the accuracy figure
    When accuracy is computed against a reference value near zero
    Then the relative error is not reported as enormous

  Scenario: a non-finite mismatch is counted, not averaged away
    When one side produces NaN where the other produces a number
    Then the mismatch is counted separately
    And the summary figure stays finite
    And the result is never within tolerance

  Scenario: --run and --info-only together are a usage error
    When I run "tile softmax.mlir -r -i"
    Then the command fails with exit code 2

  Scenario: --run and --cross together are a usage error
    When I run "tile softmax.mlir -r --cross ascend"
    Then the command fails with exit code 2

  Scenario: a target with no harness says so rather than pretending
    When a run is asked for on a target with no harness
    Then it is refused by name

  @requires-device
  Scenario: the kernel runs on this machine and matches the reference
    Given a Metal device and the emitted kernel
    When I lower it and run it on this machine
    Then the kernel's output matches the reference within tolerance
    And a device time is reported

  Scenario: uv is pinned in the manifest rather than piped from a shell
    Then uv has a verified entry for every platform this tool ships to
    And each entry carries a real digest and unpacks

  Scenario: the torch path needs no second command from the user
    Then the reference is asked for through "uv run --with torch"
    And uv is provisioned first when it is not present

  @requires-device
  Scenario: torch is consulted and the three sources are compared
    Given a Metal device and torch reachable
    When I lower it and run it on this machine
    Then the report names the torch version
    And it reports the kernel against torch, the kernel against ours, and ours against torch
    And it states a verdict
