Feature: One pure-Rust binary, one option grammar, on every supported host
  Requirements (2) and (3). Platform specifics are confined to one module behind
  cfg gates; everything else compiles identically everywhere. "Pure Rust" is a
  claim about our build graph — no C or C++ is compiled to build `tile` — not a
  claim that vendor runtimes do not exist; those are dlopened at runtime when,
  and only when, they are present.

  Scenario: no C or C++ is compiled to build the tool
    When the dependency graph of "tile" is inspected
    Then no crate in it runs a C or C++ compiler in its build script
    And the binary links no vendor library at build time

  # Opt-in because it compiles the crate for another triple, which takes minutes. That
  # is real work rather than absent work, so it is conditional, not @planned: set
  # TILE_SPEC_CROSS=1 and it runs. A triple whose std is not installed is skipped as an
  # unconfigured machine rather than reported as a portability failure.
  #
  # musl rather than gnu: `cargo check` needs no linker for it, so the check is about
  # the SOURCE being portable, which is the claim.
  @requires-cross
  Scenario Outline: the tool builds for every supported target triple
    When I build "tile" for "<triple>"
    Then the build succeeds with no platform-specific source outside the platform module

    Examples:
      | triple                       |
      | aarch64-apple-darwin         |
      | aarch64-unknown-linux-musl   |
      | x86_64-unknown-linux-musl    |

  Scenario: cfg gates are confined to the platform module
    When the source tree is scanned for "cfg(target_os" and "cfg(target_arch"
    Then every occurrence is inside "tile_core::platform" or a UI backend selector

  Scenario Outline: the documented options are accepted with both spellings
    When I run "tile <invocation>"
    Then the command is accepted by the parser

    Examples:
      | invocation                        |
      | --help                            |
      | -h                                |
      | --version                         |
      | -v                                |
      | softmax.rs -V                     |
      | softmax.rs --verbose --verbose    |
      | softmax.rs -VV                    |
      | softmax.rs -i                     |
      | softmax.rs --info-only            |
      | softmax.rs -k -t msl              |
      | softmax.rs --keep --to msl        |
      | softmax.rs -s                     |
      | softmax.rs --stats                |
      | softmax.rs -O3 -o out.metal       |
      | -d --daemon                       |

  Scenario: -v is version and -V is verbose, as specified
    When I run "tile -v"
    Then the version string is printed and the tool exits 0
    When I run "tile softmax.rs -V"
    Then diagnostic output is produced on stderr

  Scenario: verbosity is repeatable and additive
    When I raise the verbosity one step at a time
    Then each step says strictly more

  Scenario: --help documents every option, its default, and its exit codes
    When I run "tile --help"
    Then every option in the grammar appears with a one-line description
    And the default optimization level is shown
    And the exit code table is reachable from the help text

  # Codes are grouped by WHO CAN FIX IT AND HOW, because that is the only grouping a
  # script or an agent can act on. The pair that matters most is 3 against 4: "nobody
  # has written a Metal frontend" and "this binary lacks the Ascend targets" are not
  # the same answer. Collapsing them tells a caller to retry forever, or to give up one
  # download early.
  Scenario Outline: exit codes are grouped by remedy, not by subsystem
    Given the condition "<condition>"
    When the command runs
    Then the exit code is <code>

    Examples:
      | condition                                  | code |
      | success                                    | 0    |
      | a transformation failed                    | 1    |
      | a usage error                              | 2    |
      | the route needs a --via the caller omitted | 2    |
      | the capability has not been written yet    | 3    |
      | no route between the forms has been built  | 3    |
      | a required toolchain is unavailable        | 4    |
      | the target is not compiled into this build | 4    |
      | a license is required                      | 5    |
      | a device is required but absent            | 6    |
      | the UI or daemon cannot start              | 7    |

  Scenario: exit 3 never claims that something is impossible
    When I run "tile --help"
    Then the help text states that 3 is not a claim of impossibility

  Scenario: diagnostics never go to stdout when stdout carries a result
    When stdout carries the result
    Then stdout holds only the artifact and every diagnostic is on stderr
