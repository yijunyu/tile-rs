Feature: Prior attempts inform the current one, and the licensed corpus degrades gracefully
  -s answers "has anyone tried this before, and how did it go" for the kernels
  named in the arguments. Two databases back it: the user's own attempts, always
  readable, and the curated cross-target corpus, which is licensed. A missing
  license removes a source of knowledge; it never removes a capability.

  Background:
    Given a tile-rs kernel "softmax.rs"

  Scenario: -s reports prior conversions relevant to the kernels in the arguments
    When I convert a kernel twice and then ask for its history
    Then both conversions are listed against that kernel with their routes
    And the headroom of a conversion stays unassessed rather than zero
    And only kernels named by the arguments are reported

  Scenario: the kernel identity is resolved, not string-matched on the filename
    Then a kernel is identified by its contents, not by its filename

  Scenario: an unassessed headroom is displayed as unassessed, never as zero
    Given a corpus where "argmax" has headroom NULL and "rope" has headroom 0.0
    Then the headroom cell for "argmax" reads "unassessed"
    And the headroom cell for "rope" reads "closed"
    And the two cells are not the same text

  Scenario: without a license, -s still works on local attempts and explains the gap
    Given no license key is installed
    And the local store holds an attempt for "softmax"
    When the stats report is produced for "softmax"
    Then the local attempt is reported
    And one line states what the licensed corpus would add and how to obtain a key

  Scenario: a missing license degrades a feature but never blocks the conversion
    Given no license key is installed
    When I run "tile softmax.mlir -s -t msl -o out.metal"
    Then the conversion still succeeds
    And the stats section carries the unlicensed notice

  @requires-stats-build
  Scenario: the corpus is unreadable without the key, not merely unlisted
    Given a sealed corpus
    Then its contents are not readable as text
    And opening it with the wrong key fails whole, not partially

  @requires-stats-build
  Scenario: a license is verified offline against an embedded public key
    Given a license token signed by a known key
    When the token is verified
    Then the subject, feature set and expiry are reported
    And no network request was needed

  Scenario: license status with no key installed is not a failure
    Given no license key is installed
    When I run "tile license status"
    Then the exit code is 0
    And the report states that everything else works unchanged

  @requires-stats-build
  Scenario Outline: an invalid license is rejected with the reason
    Given a license token that is "<defect>"
    When the token is verified
    Then the verification fails
    And the reason is "<reason>"

    Examples:
      | defect                   | reason   |
      | expired                  | expired  |
      | signed by an unknown key | tampered |
      | tampered with            | tampered |

  @requires-stats-build
  Scenario: a forged token is never reported as merely expired
    Given a license token that is "expired and signed by an unknown key"
    When the token is verified
    Then the reason is "tampered"

  Scenario: writes to the local store are never license-gated
    Given no license key is installed
    When an attempt is recorded locally
    Then it is readable again without any license

  Scenario: the tool never fabricates a measurement to fill a column
    When a record carries a headroom with no stated basis
    Then it is refused rather than stored

  # The corpus is exported and sealed at release time, so it is a snapshot. Answering
  # from month-old measurements without saying so violates the same doctrine as
  # inventing a headroom.
  Scenario: every stats report states how old the corpus is
    Given a corpus snapshot with an export timestamp
    When the stats report is produced for "softmax"
    Then the report names the corpus export timestamp and source revision

  Scenario: the corpus is read by a pure-Rust reader, not a database binding
    When the dependency graph of the "stats" feature is inspected
    Then no crate in it runs a C or C++ compiler
    And no SQLite implementation is linked

  Scenario: a corpus that fails its seal is reported, not silently ignored
    Then a sealed corpus that fails authentication is reported, not ignored
