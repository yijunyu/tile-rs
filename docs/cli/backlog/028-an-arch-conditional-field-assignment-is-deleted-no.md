id: 28
area: target
title: An arch-conditional field assignment is deleted, not diagnosed, when the target struct moves the field
opened: 2026-09-10

## wanted
Emit an AscendC Fixpipe descriptor for two architectures whose parameter struct
has the SAME fields in DIFFERENT PLACES. On c220 the NZ2ND parameters are
top-level (`p.ndNum`); on c310 `FixpipeParamsArch3510<CO2Layout::ROW_MAJOR>`
moves them one level down into `p.params` (an `Nz2NdParams`). I wanted the
lowering to be told "these three values go into the descriptor" and have the
backend place them, so that adding an architecture cannot lose them.

## got
Hand-written codegen instead, in the shape everyone writes it:

    #if !(__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
        fp.ndNum = 1; fp.srcNdStride = 0; fp.dstNdStride = 0;
    #endif

This compiles clean on both architectures and silently programs LOOP3_PARA from
an uninitialised `params` on c310 -- `FixpipeParamsArch3510()` has an empty body,
and dav_3510's `SetLoop3Para` packs all three fields into a hardware register on
every call. The `#if` removed the assignment AND the compiler's ability to
notice that the field it named no longer exists. One benchmark credit was spent
on a submission whose only change was this, and it came back byte-identical.

## workaround
Set `fp.params.ndNum` / `fp.params.srcNdStride` / `fp.params.dstNdStride` on the
3510 branch (bench/gen_mm.py in cannbench-tilers, NOTES-ascendc.md 206).

What the fix will need, concretely:
  * `FixpipeParamsV220` -- fields `ndNum`, `srcNdStride`, `dstNdStride` at top
    level, defaults 1/0/0.
  * `FixpipeParamsC310<format = CO2Layout::ROW_MAJOR> : FixpipeParamsArch3510<format>`
    -- same three fields under `.params`, whose type is
    `TransformParams<format>::PARAMS` = `Nz2NdParams` for ROW_MAJOR, `uint8_t`
    for `CO2Layout::NZ` (no nd parameters at all), `Nz2DnParams` for
    COLUMN_MAJOR (different field NAMES: `dnNum`, `srcNzMatrixStride`,
    `dstDnMatrixStride`).
  * So the mapping is not a rename: the target layout DECIDES WHETHER THE
    PARAMETER EXISTS. A descriptor model that carries `{nd_num, src_nd_stride,
    dst_nd_stride}` as optional values and lets the backend reject or place them
    is the shape that works; a per-arch `#if` in a template is not.
  * `Nz2NdParams` is declared but never defined in the CANN 8.x toolkit shipped
    for 910b, so `FixpipeParamsC310<>` cannot be instantiated there. Any test has
    to be conditional on the toolkit, not just on the arch flag.

What would have made the tool able to do it: a lowering that treats a target
struct as a NAMED SET OF SLOTS rather than as text, so an unplaced value is an
error at emit time on every target that has the slot.
