Feature: Adding a target via the CodegenTarget trait + registry
  The whole extension surface of tile-rs is: implement CodegenTarget, then
  register(Box::new(MyTarget)) on a TargetRegistry. Selection is by name, the
  value of TILERS_CODEGEN_PATH. No enum to extend, no dispatch match arm.

  Scenario: the built-in registry exposes the debug reference target
    Given a registry pre-populated with the built-in targets
    Then the registry is not empty
    And selecting "debug" routes to a target
    And selecting "does-not-exist" routes to nothing

  Scenario: registering a custom target makes it selectable by name
    Given an empty registry
    When I register a custom target named "myaccel"
    Then the registry has 1 target
    And selecting "myaccel" routes to a target
    And the custom target emits its signature for a non-empty module

  Scenario: TILERS_CODEGEN_PATH selection resolves through the registry
    Given a registry pre-populated with the built-in targets
    And a custom target named "myaccel" registered on top
    When TILERS_CODEGEN_PATH is "myaccel"
    Then the registry selects the matching target by that name

  Scenario: a plain convert fn lifts into the trait via the EmitterTarget adapter
    Given an empty registry
    When I register the plain "cuda" convert fn as an EmitterTarget
    Then selecting "cuda" routes to a target
    And emitting a non-empty module through it yields the adapter output
    And emitting an empty module through it fails
