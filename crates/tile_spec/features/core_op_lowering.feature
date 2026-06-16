Feature: Every open backend lowers the core tile-op set
  tile-rs claims generality across the accelerator long tail: the same core
  ops (rms_norm, matmul, softmax, rope, elementwise add) must lower through
  every open backend's pure emitter. This pins one scenario per op, checking a
  representative backend emits a recognizable lowering for it. It ties to the
  generality-matrix unit tests but expresses the claim as a use case.

  Scenario Outline: the <op> op lowers to <target>
    Given the canonical "<op>" MLIR snippet
    When I emit the snippet to target "<target>"
    Then the emit succeeds
    And the output contains "<evidence>"

    Examples: rms_norm
      | op       | target | evidence     |
      | rms_norm | spirv  | inversesqrt  |
      | rms_norm | gpu    | rsqrt        |
      | rms_norm | msl    | kernel void  |
      | rms_norm | linalg | math.rsqrt   |
      | rms_norm | tpu    | rsqrt        |

    Examples: matmul
      | op     | target | evidence    |
      | matmul | spirv  | layout(set  |
      | matmul | gpu    | __global__  |
      | matmul | pto    | func.func   |

    Examples: softmax
      | op      | target | evidence            |
      | softmax | spirv  | layout(set          |
      | softmax | nki    | @nki.jit            |
      | softmax | aie    | from aie            |
      | softmax | linalg | linalg.softmax      |
      | softmax | tpu    | jnp.exp             |

    Examples: rope
      | op   | target | evidence    |
      | rope | spirv  | layout(set  |
      | rope | gpu    | __global__  |

    Examples: elementwise add
      | op  | target | evidence    |
      | add | spirv  | layout(set  |
      | add | msl    | kernel void |
