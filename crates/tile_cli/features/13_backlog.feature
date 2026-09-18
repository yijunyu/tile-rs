Feature: The backlog records what tile could not do, and says when to stop patching
  The useful signal about a tool is not what it does — it is what it was asked for and
  could not do. That signal is normally lost: an agent works around a gap, the session
  ends, and the next one rediscovers it. Here the workaround is written down beside the
  gap, so the eventual fix starts from evidence rather than a fresh guess.

  Only gaps hit while doing real work belong here. `plan.md` already says what is
  scheduled; this says what actually bit somebody, and an entry written speculatively
  is noise in the one place that should be nothing but evidence.

  Scenario: an entry survives being written and read back
    Given an entry recording a gap and how it was worked around
    When it is written to the backlog and read again
    Then every field comes back unchanged

  Scenario: an entry with no workaround says so rather than leaving a blank
    Given an entry with no workaround recorded
    When it is rendered
    Then it states that an entry without one is a report, not evidence

  Scenario: each entry is its own file, so two sessions do not conflict
    When two entries are recorded
    Then each has its own file named by its number and title

  Scenario: a handful of unrelated gaps is healthy
    Given open issues in "run" and "lowering"
    Then the verdict is healthy
    And the exit code would be 0

  Scenario: three issues in one area is a design problem, not three bugs
    Given three open issues all in "form-detection"
    Then the verdict says to stop and reconsider
    And it names the area
    And it says that fixing them individually is three ways of not addressing it

  Scenario: a cluster is reported ahead of a raw count
    Given many open issues, three of them in one area
    Then the verdict names the area rather than the total

  Scenario: enough scattered issues eventually say stop as well
    Given twelve open issues in twelve different areas
    Then the verdict says the tool is costing more than it saves

  Scenario: closing issues brings the verdict back, so the rule is not a ratchet
    Given three open issues all in "form-detection"
    When one of them is closed
    Then the verdict is healthy

  Scenario: the listing groups by area and shows each workaround
    Given open issues in "run" and "lowering"
    When the backlog is listed
    Then the entries are grouped by area
    And each shows the first line of its workaround
    And the listing ends with a verdict

  Scenario: an empty backlog says so rather than printing a bare zero
    Given no issues have been recorded
    When the backlog is listed
    Then it states that nothing has been recorded yet

  Scenario: the exit code carries the verdict so a session can branch on it
    Then a healthy backlog exits 0
    And a backlog that should stop the tool exits 3

  Scenario: an unknown field in an entry is an error, not something to skip
    When an entry carrying an unknown field is read
    Then the unknown field is named in the refusal
