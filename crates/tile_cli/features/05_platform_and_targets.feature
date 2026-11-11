Feature: The tool knows what machine it is on and defaults to something runnable
  Requirements (4) and (5). Detection is runtime probing of vendor libraries and
  CLIs — never a build-time link — which is what lets one pure-Rust binary run
  on macOS, x86_64, aarch64 and arm. The default output target is the native one
  so the artifact can actually be executed here; with no accelerator at all the
  default is the CPU `linalg` bridge, which is runnable everywhere.

  Scenario: doctor reports platform, accelerators, and SDKs separately
    When I ask the doctor about this machine
    Then the report names the platform, the accelerators and the SDKs in separate sections
    And a family whose SDK is absent is distinguished from a family that is absent

  Scenario Outline: the default output target follows the detected accelerator
    Given the detected accelerator family is "<family>"
    When I run "tile softmax.rs"
    Then the default output form is "<form>"

    Examples:
      | family    | form   |
      | apple-gpu | msl    |
      | nvidia    | gpu    |
      | amd-npu   | aie    |
      | ascend    | cpp    |
      | none      | linalg |

  # There is no HIP/ROCm target in the registry. A Radeon or Instinct box is NOT an
  # AMD Ryzen AI NPU, and defaulting one to the `aie` form would emit IRON Python for
  # a GPU. Until a ROCm target exists, an AMD GPU falls back to linalg and says why.
  Scenario: an AMD GPU is not an AMD NPU
    Given the detected accelerator family is "amd-gpu"
    When I run "tile softmax.rs"
    Then the default output form is "linalg"
    And the report states that tile-rs has no ROCm/HIP target yet
    And the report does not offer the "aie" form as if it would run there

  Scenario: doctor names the AMD device class it found
    Then doctor lists an AMD GPU as present with no tile-rs target

  Scenario: --cross permits a target with no local device
    When I convert with --cross on a machine with no accelerator
    Then the absence of a device is not itself a failure

  Scenario: naming a non-native target is cross generation and says so
    When I convert to a target that does not run on this machine
    Then the run is reported as cross generation and still succeeds

  Scenario: -O4 is rejected under cross generation
    When I ask for -O4 under --cross
    Then the error states that measurement requires the native target

  # Requirement (3) is not a build property. It is a claim about what a user has to do
  # before the tool works, so the test is DEPLOYMENT -- copy the binary, run it, read
  # what it says. Opt-in because it needs other machines: TILE_SPEC_HOSTS="mac mini".
  @requires-hosts
  Scenario: the same binary reports correctly on every host it is copied to
    Then the same binary reports correctly on every host it is copied to

  Scenario: no vendor library is required to start the tool
    When I ask for the version on a machine with nothing installed
    Then the tool still reports its version

  Scenario: doctor distinguishes "measured" from "supported"
    When I ask the doctor which targets are trustworthy
    Then measured targets are listed apart from unmeasured ones
    And the report says that unmeasured is not unlimited
