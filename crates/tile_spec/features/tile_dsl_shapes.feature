Feature: Tile DSL enforces shapes via const generics
  The tile IR (Tile<ROWS, COLS, T> + ~80 tile_* intrinsics in tile_std::tile)
  is shape-checked at compile time through const-generic dimensions: an
  elementwise op binds both operands to the same <R, C>, and matmul threads the
  contracted <M, K, N> so mismatched inner dims fail to type-check. tile_std is
  a no_core kernel-side crate (not host-runnable), so this use case is pinned by
  introspecting the published intrinsic signatures rather than executing them.

  Background:
    Given the tile_std tile DSL source

  Scenario: the tile type carries its dimensions in the type
    Then a "Tile" type is declared with const ROWS and COLS dimensions

  Scenario Outline: the <op> intrinsic is generic over its shape
    Then the intrinsic "<fn>" is generic over "<dims>"

    Examples:
      | op          | fn               | dims              |
      | load        | tile_load_f32    | const ROWS, const COLS |
      | add         | tile_add_f32     | const ROWS, const COLS |
      | mul         | tile_mul_f32     | const ROWS, const COLS |
      | softmax     | tile_softmax_f32 | const ROWS, const COLS |
      | matmul      | tile_matmul_f32  | const M, const K, const N |

  Scenario: matmul threads the contracted dimension distinctly from the outputs
    Then the intrinsic "tile_matmul_f32" binds three distinct dims "M", "K", "N"
