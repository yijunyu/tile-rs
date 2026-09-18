Feature: A kernel's form is decided by magic first, extension second, never by guesswork
  `tile` is pandoc for kernels: the user names files, not formats. But four
  extensions in this domain are overloaded — `.py` by nki/aie/tpu, `.mlir` by
  mlir/linalg/pto/rvv, `.c` by gaudi/hexagon, `.cpp` by ttmetal/cpp — so an
  extension is a hint, not a decision procedure. Resolution is: explicit -f,
  then content magic, then extension, then an error that NAMES the candidates.
  `.rs` is reserved for tile-rs and is never sniffed for anything else.

  Background:
    Given a clean working directory

  Scenario Outline: content magic disambiguates a shared extension
    Given a file "k<ext>" whose content contains "<magic>"
    When I run "tile k<ext> --info-only"
    Then the reported input form is "<form>"

    Examples:
      | ext    | magic            | form    |
      | .py    | @nki.jit         | nki     |
      | .py    | from aie.iron    | aie     |
      | .py    | pallas           | tpu     |
      | .mlir  | linalg.          | linalg  |
      | .mlir  | riscv64          | rvv     |
      | .c     | tpc-clang        | gaudi   |
      | .c     | hvx_             | hexagon |
      | .cpp   | void MAIN        | ttmetal |
      | .cpp   | AscendC::        | cpp     |
      | .metal | kernel void      | msl     |
      | .cu    | __global__       | gpu     |
      | .comp  | layout(set = 0   | spirv   |
      | .mu    | musa_runtime.h   | musa    |
      | .mlu   | __mlu_entry__    | bang    |
      | .csl   | comptime         | csl     |

  Scenario: .rs is reserved for tile-rs unconditionally
    Given a file "k.rs" whose content contains "__global__"
    When I run "tile k.rs --info-only"
    Then the reported input form is "tile"
    And no sniffing of "k.rs" is attempted

  Scenario: an unrecognisable file is refused with the candidate forms named
    Given a file "k.py" with no recognisable magic
    When I run "tile k.py --info-only"
    Then the command fails with exit code 2
    And the error names the candidate forms "nki, aie, tpu"
    And the error shows how to force one with "-f"

  Scenario: -f overrides both magic and extension
    Given a file "k.py" whose content contains "@nki.jit"
    When I run "tile k.py -f tpu --info-only"
    Then the reported input form is "tpu"

  Scenario: a form claimed by -f that contradicts strong magic is warned about, not silently accepted
    Given a file "k.metal" whose content contains "kernel void"
    When I run "tile k.metal -f gpu --info-only"
    Then a warning states that the content looks like form "msl"
    And the reported input form is "gpu"

  # The literal version -- add a row for a hypothetical form -- cannot run: FORMS is a
  # const table. What it asserts is that resolution is DRIVEN by the table rather than
  # by per-form code, and that is checkable exhaustively over the rows that exist.
  Scenario: the form table is data, so resolution is uniform across every row
    Then every form in the table resolves from its own extension and magic
