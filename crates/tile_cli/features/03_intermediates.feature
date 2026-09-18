Feature: Intermediate representations are real files, discarded unless asked for
  A multi-hop route materialises every hop. Keeping them is how a user debugs a
  lowering; discarding them by default is how the tool stays quiet. The scratch
  directory is removed even when the run fails, so a failed conversion never
  litters the tree.

  At -O1 and above a readable input gains an optimize hop BEFORE the lowering, which
  is what makes the optimization inspectable: it is a step on the route, not a claim
  in a report, so -k writes it out and a user can diff what the passes did.

  Background:
    Given a clean working directory

  Scenario: intermediates are removed by default
    When I run "tile softmax.mlir -t msl -o out.metal"
    Then "out.metal" exists
    And no intermediate files remain beside the input
    And the scratch directory has been removed

  Scenario: -k keeps every hop, named by step index and form
    When I run "tile softmax.mlir -t msl -o out.metal -k"
    Then the kept file is "softmax.1.mlir.mlir"
    And the kept file is the input as the optimizer left it

  Scenario: the optimize hop is a step on the route, not a claim in a report
    When I run "tile softmax.mlir -t msl -o out.metal --route"
    Then the route taken is "mlir -> mlir -> msl"

  Scenario: -O0 has no optimize hop to keep
    When I run "tile softmax.mlir -t msl -o out.metal -O0 -k"
    Then no intermediate files remain beside the input

  Scenario: --keep-dir implies -k and places the hops there
    When I keep the intermediates in a named directory
    Then the hops are there and nowhere else

  Scenario: a failed hop still cleans up, and says which hop failed
    When a hop fails
    Then it says which hop failed and leaves nothing behind

  Scenario: -k preserves the intermediates of a failed run for debugging
    When a hop fails and I asked to keep the intermediates
    Then the intermediates of the failed run survive for debugging
