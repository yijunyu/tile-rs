Feature: The mundane cases a real user hits in the first ten minutes
  Every one of these is a bug report waiting to be filed. They are specified here
  because a swissknife is judged on its edges, not its centre.

  Scenario: "-" reads stdin, and the form must then be given
    When I pipe a module into "tile - -f mlir -t msl -o out.metal" and keep the output
    Then the module is read from stdin
    And "out.metal" is written

  Scenario: stdin with no -f is a usage error, because there is no name to sniff from
    When I run "tile - -t msl -o out.metal" with a module on stdin
    Then the command fails with exit code 2
    And the error states that stdin requires "-f"

  Scenario: "-o -" writes the result to stdout and nothing else does
    When I run "tile k.mlir -t msl -o -"
    Then the emitted MSL is on stdout
    And the profile, the route and any warning are on stderr

  Scenario: several inputs are converted independently, and one failure does not lose the rest
    Given "a.mlir", "b.mlir" and "c.mlir" where "b.mlir" is malformed
    When I run "tile a.mlir b.mlir c.mlir -t msl" over those
    Then "a.opt.metal" and "c.opt.metal" are written
    And the failure for "b.mlir" is reported with its reason
    And the exit code is 1

  Scenario: several inputs with a single -o is a usage error
    When I run "tile a.mlir b.mlir -o out.metal"
    Then the command fails with exit code 2
    And the error suggests an output directory instead

  Scenario: a directory input is refused with the glob that would have worked
    When I run "tile" on a directory
    Then the error shows the equivalent shell glob

  Scenario: an output directory that does not exist is an error, not a silent mkdir
    When I run "tile k.mlir -o ./nope/out.metal"
    Then the command fails with exit code 2
    And the error names the missing directory

  Scenario: a file containing several kernels reports all of them
    Given "multi.mlir" contains two kernels
    When I profile it
    Then both kernels are profiled and named

  Scenario: converting a multi-kernel module emits all of them into one output
    Given "multi.mlir" contains two kernels
    When I convert it to one output
    Then "out.metal" contains both kernel entry points

  Scenario: a non-UTF8 input is refused before sniffing
    Given "k.mlir" contains invalid UTF-8
    When I run "tile k.mlir -i"
    Then the command fails with exit code 2
    And the error states that kernel sources must be UTF-8
    And the byte offset of the first invalid sequence is given

  Scenario: an empty input is refused, matching the emitters' own contract
    Given "k.mlir" is empty
    When I run "tile k.mlir -t msl -o out.metal"
    Then the command fails with exit code 1
    And the error states that the module is empty

  Scenario: a symlinked input is read through, and the output is named after the link
    Given "link.mlir" is a symlink to "real.mlir"
    When I convert the symlink
    Then "real.mlir" is read
    And the output is written beside "link.mlir", not beside "real.mlir"

  # fs::write follows a symlink, so this silently replaced a file the user never named.
  # --force is permission to replace the file that WAS named, not to leave the tree.
  Scenario: an output that is a symlink is refused rather than written through
    Given "out.metal" is a symlink to a file outside the output directory
    When I convert with -o naming that symlink and --force
    Then the link target is untouched and the error names it

  # Exit 4, not 3. This capability EXISTS — it is one download away — and telling the
  # caller "unsupported" would send them away from something that works elsewhere.
  Scenario Outline: a form that exists but is not compiled into this build says so
    Given this build was compiled without the "<form>" emitter
    When I run "tile k.mlir -t <form> -o out"
    Then the command fails with exit code 4
    And the error distinguishes "not compiled into this build" from "no such form"
    And it names the feature that would enable it

    Examples:
      | form |
      | cpp  |
      | pto  |

  Scenario: two concurrent runs do not corrupt the shared state directory
    Then two concurrent runs do not corrupt the shared state directory

  Scenario: an interrupted run cleans up on the next run
    When a previous run was killed and left its scratch
    Then the next run sweeps it and says so at verbosity
