; tile-rs PICO intrinsic program (SVP_NNN / Hi3403). The target language IS the
; intrinsic set: 91 named primary opcodes, emitted with a JSON manifest. Alone among
; the targets it ends in no vendor compiler -- "linking" rebuilds the .om container.
.kernel softmax_1d
.ub_geometry 8x16x1024x16      ; 2 MiB, per-profile, never keyed off the V100/V101 name
  ld.ub      u0, gm0, 1, 1024
  reduce.max u1, u0
  sub.bcast  u2, u0, u1
  exp        u3, u2
  reduce.sum u4, u3
  div.bcast  u5, u3, u4
  st.gm      gm1, u5, 1, 1024
.end
