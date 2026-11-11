Feature: The same capabilities are reachable by a human and by an agent
  Requirement (7). The CLI and the MCP daemon are two front-ends over one core,
  so an agent can do exactly what a user can do and no more. Protocol hygiene
  matters: in stdio mode, stdout belongs to MCP alone.

  Scenario: the daemon exposes the CLI's verbs as MCP tools
    When I run "tile -d"
    Then an MCP server is served over stdio
    And it advertises the tools "doctor, list_forms, identify, profile, routes, convert, install"
    And every advertised tool carries an input schema

  Scenario: an MCP conversion and the equivalent CLI invocation agree
    Given a tile-rs kernel "softmax.rs"
    When the MCP client calls "convert" with input "softmax.rs" and to "msl"
    And I separately run "tile softmax.rs -t msl -o out.metal"
    Then both outputs are byte-identical

  Scenario: in stdio mode all logging goes to stderr
    When I run "tile -d -VVV"
    Then no diagnostic byte is written to stdout
    And the MCP framing on stdout stays parseable

  # There is no TCP daemon at all. Serving licensed corpus data over a socket needs an
  # auth token nobody has built, so loopback is refused on the same grounds as
  # 0.0.0.0 -- and specifying "0.0.0.0 needs an opt-in" would describe a door that is
  # not there.
  Scenario: a socket daemon is refused until it can authenticate its callers
    When I ask the daemon to listen on a socket
    Then both loopback and non-loopback are refused for the same stated reason

  Scenario: agent-initiated work is bounded
    When an MCP client requests a conversion needing a toolchain
    Then no toolchain is auto-installed
    And the reply names the install tool the client may call explicitly

  Scenario: -m selects the engine matching the detected accelerator
    Given the detected accelerator family is "apple-gpu"
    When an engine is chosen for that family
    Then the engine "ds4-rs-metal" is selected

  Scenario: an engine that is not on the machine names everywhere it looked
    When an engine that is not installed is located
    Then the error names every path it searched
    And it states that nothing is fetched on the user's behalf

  Scenario: a failure to provision a model never costs the kernel tools
    Given the detected accelerator family is "apple-gpu"
    When I run "tile -d -m qwen3"
    Then the daemon still answers a request

  Scenario Outline: the engine follows the accelerator family
    Given the detected accelerator family is "<family>"
    When an engine is chosen for that family
    Then the engine "<engine>" is selected

    Examples:
      | family    | engine        |
      | apple-gpu | ds4-rs-metal  |
      | nvidia    | ds4-rs-cuda   |
      | amd-gpu   | ds4-rs-amd    |
      | amd-npu   | ds4-rs-amd    |
      | ascend    | ds4-rs-ascend |

  # The daemon serves agents. An agent must not be able to make the machine download
  # gigabytes by asking a question.
  Scenario: the daemon never binds TCP without an explicit token
    When I run "tile -d --listen 127.0.0.1:7777"
    Then the command fails with exit code 2
    And the error explains that a TCP daemon serving licensed data requires a token

  Scenario: stdio is the default transport and needs no token
    When I run "tile -d"
    Then the transport is stdio
    And no port is opened

  Scenario: a killed run leaves no scratch behind
    When a previous run was killed and left its scratch
    Then the next run sweeps it and says so at verbosity

  Scenario: -m outside daemon mode is a usage error
    When I run "tile softmax.mlir -m qwen3"
    Then the command fails with exit code 2
    And the error states that -m requires -d

  # The fetch mechanics -- size reported up front, an interrupted download resuming --
  # are not built and stay @planned below. This clause IS built, and it is the one that
  # decides whether asking for a model can cost you the tools you already had.
  Scenario: a model that cannot be provisioned does not take the kernel tools with it
    Then a model that cannot be provisioned leaves the daemon serving the kernel tools

  # Weights are not fetched: a multi-gigabyte download needs a pinned entry with a
  # digest, the rule every toolchain here obeys, and no such entry exists for a model.
  # What IS built is the rest of it -- build, locate, start, supervise, stop.
  Scenario: provisioning an engine is explicit, reported, and cleaned up after
    Then provisioning an engine announces the build before it starts
