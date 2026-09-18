Feature: The UI is an optional front-end, and its absence is never fatal
  -u serves the view of the same core: the machine, the profile, the routes and the
  form matrix. The server half is the part that has to exist either way and the part
  that can be TESTED — a headless box can fetch a page and assert what is on it; it
  cannot look at a window. The egui-wasm bundle slots in behind this same server as a
  static asset, and needs a wasm toolchain this tool will not acquire silently.

  Because a headless machine is the normal case for a compiler tool — ssh, CI, a build
  box — "no windowing subsystem" degrades to printing the URL, not to an error.

  Scenario: -u serves the view on loopback and prints its address
    When I run "tile softmax.mlir -u"
    Then the address is printed on stdout
    And fetching it returns the profile, the routes and the form matrix

  Scenario: the listener binds loopback and nothing else
    When the UI listener is bound
    Then its address is a loopback address

  Scenario: the page needs nothing from the network
    When the page is rendered
    Then it references no external host
    And a viewer with no connectivity sees everything on it

  Scenario: kernel source cannot inject markup into the page
    Given a kernel whose text contains markup
    When the page is rendered
    Then the markup is shown as text, not interpreted

  Scenario: the UI is a view over the core, not a second implementation
    When the view is built for a kernel
    Then its profile is the one the profiler produced
    And its routes are the ones the planner produced

  Scenario: a headless host prints the address rather than failing
    Given no windowing subsystem is available
    When I run "tile softmax.mlir -u"
    Then the address is printed on stdout
    And the exit code is 0

  # Deliberately not built, and the reason is in ui.rs: a window cannot be asserted by a
  # headless CI box, and the server half has to exist either way. What matters is that
  # asking for it does not silently get you the other subsystem.
  Scenario: --ui native is refused rather than silently served as a web view
    Then --ui native refuses in the tense of work not done, and names what does work
    And a kernel really named native is still readable

  # NOT egui: eframe means wasm-bindgen, which is a build prerequisite `tile` refuses to
  # acquire silently, and a few hundred crates against a dependency budget that is the
  # point of this repository. The bundle is dependency-free Rust over linear memory --
  # four extern "C" functions, which is all a browser needs. It buys the one thing a
  # server-rendered page cannot do: the form matrix filters as you type.
  Scenario: the wasm bundle is served from the same address
    Then the wasm bundle is served from the same address as the page
    And the page still stands with no wasm at all

  # Shape is not behaviour. A bundle that loads and returns nothing useful passes every
  # structural check and leaves a filter box that silently does nothing -- so the
  # behaviour is asserted where a runtime already exists, rather than giving `tile` a
  # wasm interpreter it has no other reason to carry.
  @requires-wasm-runtime
  Scenario: the bundle does what it claims, executed
    Then the bundle's filter returns the right rows when actually executed

  # Demonstrated rather than assumed: an off-by-one that drops the first matching row
  # leaves the magic number, the version and every export name intact. The structural
  # test passed on that bundle; this one failed. That is why both exist.
  @requires-wasm-target
  Scenario: the checked-in bundle has not drifted from its source
    Then the checked-in bundle is what its source builds
