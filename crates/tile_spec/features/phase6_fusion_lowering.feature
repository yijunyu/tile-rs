Feature: Phase-6 speculative-decoding and SiLU->Mul fusion lower through the open emitters
  Beyond the single-op core set, tile-rs lowers fused multi-op kernels (the
  SwiGLU SiLU->Mul fusion) and the Phase-6 speculative-decoding primitives
  (argmax / sample_top_p / token_accept). Each must produce a recognizable,
  math-specific lowering through a representative open backend's pure emitter.
  This pins the fused / spec-decode dispatch arms that the single-op matrix
  scenarios never exercise.

  Scenario Outline: the fused/spec-decode <op> op lowers to <target>
    Given the canonical "<op>" MLIR snippet
    When I emit the snippet to target "<target>"
    Then the emit succeeds
    And the output contains "<evidence>"
    And emitting the same snippet again yields byte-identical output

    Examples: SiLU->Mul fusion (SwiGLU)
      | op       | target | evidence       |
      | silu_mul | tpu    | fused silu_mul |

    Examples: argmax (greedy token select)
      | op     | target | evidence       |
      | argmax | tpu    | jnp.argmax     |
      | argmax | linalg | linalg.generic |

    Examples: sample_top_p (temperature softmax then sample)
      | op           | target | evidence       |
      | sample_top_p | tpu    | jax.nn.softmax |
      | sample_top_p | linalg | sample_top_p   |

    Examples: token_accept (spec-decode acceptance)
      | op           | target | evidence          |
      | token_accept | tpu    | astype(jnp.int32) |
