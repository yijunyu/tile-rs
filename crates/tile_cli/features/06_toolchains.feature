Feature: Toolchains arrive on demand, verified, user-scoped, and never as a scavenger hunt
  Requirements (1) and (6). The tool acquires exactly the toolchain the route in
  front of it needs, at the moment it needs it, from a pinned manifest with a
  checksum, into the user's own prefix. What it must NOT do is claim to install
  things it legally or technically cannot — a vendor SDK behind a EULA, a login
  or a root install is refused with the exact remedy, which is what "never leave
  the user hunting for setup docs" actually requires.

  Background:
    Given an empty toolchain prefix at "~/.tile-rs/toolchains"

  # Driven against a local artifact, so the check is hermetic and needs no network.
  # The transport is not what is being tested -- the VERIFICATION is, and that is the
  # part that must hold whatever the bytes arrived over.
  Scenario: a pinned artifact is fetched, verified and installed without prompting
    Given a pinned artifact and an empty toolchain prefix
    When the toolchain is acquired
    Then the artifact is installed under the user prefix
    And the fetch was announced before it began
    And its sha256 was checked against the manifest
    And nothing was asked of the user

  Scenario: acquiring the same toolchain twice does nothing the second time
    Given a pinned artifact and an empty toolchain prefix
    When the toolchain is acquired
    And the toolchain is acquired again
    Then the second acquisition reports it was already present
    And the second acquisition says nothing

  Scenario: a corrupted download is rejected before it is unpacked
    Given an artifact whose bytes do not match the manifest
    When the toolchain is acquired
    Then the acquisition fails with a checksum mismatch
    And nothing was unpacked
    And no toolchain directory was left behind

  Scenario: only the toolchain the route needs is installed
    Given a pinned artifact and an empty toolchain prefix
    When the toolchain is acquired
    Then no other toolchain in the manifest was fetched

  Scenario: installs are user-scoped and never touch the system
    Given a pinned artifact and an empty toolchain prefix
    When the toolchain is acquired
    Then no path outside the user prefix is written
    And a layout version is recorded so a later move can migrate

  Scenario Outline: a toolchain that cannot be automated is refused with the exact remedy
    When I ask to install "<id>"
    Then the command fails with exit code 4
    And the error states the barrier "<barrier>"
    And the error gives the exact command or URL to obtain it
    And the error states what the tool will do once it is present

    Examples:
      | id        | barrier      |
      | cann      | vendor-login |
      | cuda      | eula         |
      | xcode-clt | os-level     |

  Scenario: what this platform can be given is one command away
    When I run "tile install"
    Then every manifest entry for this platform is listed with its state
    And the report names the prefix everything installs under

  Scenario: --no-install turns acquisition into a diagnosis
    Given a pinned artifact and an empty toolchain prefix
    When the toolchain is acquired with "--no-install"
    Then the acquisition is declined
    And the error prints the install command that would work
    And nothing was written under the user prefix

  Scenario: --offline refuses a network URL but still reads a local one
    Given a pinned artifact and an empty toolchain prefix
    When the toolchain is acquired with "--offline"
    Then a local artifact is still read
    And a network URL is refused without being attempted

  Scenario: an unpinned entry will not be downloaded at all
    Then an entry with no recorded digest is refused before anything is fetched

  Scenario: a tool with no entry for this platform is refused, not guessed at
    When I ask to install a tool this platform has no entry for
    Then it is refused as acquirable-elsewhere, naming this platform

  Scenario: the daemon does not install on an agent's behalf by default
    Then the daemon acquires nothing on an agent's behalf, and says what it would take

  # The three ways a `.rs` lowering fails silently, each refused before the build rather
  # than discovered after it. Each of these cost a debugging session and each points at
  # the wrong component when it happens.
  Scenario: a set RUSTFLAGS is refused, because it would silently disable the backend
    Given RUSTFLAGS is set in the environment
    When a kernel is lowered from tile-rs source
    Then the command is refused before the build starts
    And the error explains that the backend would never load
    And the error gives the exact way to re-run it

  Scenario: the codegen path is translated from the form id, not passed through
    Then the codegen path for "msl" is "metal"
    And the codegen path for "gpu" is "cuda"
    And an unrecognised value is never sent to the backend
