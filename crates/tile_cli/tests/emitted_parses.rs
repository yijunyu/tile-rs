//! Emitted source that nothing ever compiles is source nobody has checked.
//!
//! The SPIR-V f16 path made the cost of this concrete: `-t spirv` on an f16 kernel wrote a
//! shader to disk and reported a successful lowering, and the shader did not compile.
//! `max(v, 0.0)` on a `float16_t` is a type error, so f16 `relu` had NEVER compiled --
//! invisible because the harness refused `float16_t` buffers at the signature check, so
//! glslang was never asked.
//!
//! Two of sixteen emitted forms are compiled by something here: Metal by `xcrun metal`,
//! SPIR-V by `glslangValidator`. The vendor C backends need toolchains this machine does
//! not have. But `aie`, `nki` and `tpu` emit PYTHON, and Python can be parsed with no
//! vendor SDK at all -- so for those three there is no excuse for emitting text nobody
//! has checked.
//!
//! This is a syntax gate, not a semantic one. It cannot tell whether the kernel is
//! correct, only that it is a program. That is exactly the check that would have caught
//! the f16 `relu`, which was not subtly wrong -- it did not parse.

use std::process::Command;

/// Forms whose output is Python.
const PYTHON_FORMS: &[&str] = &["aie", "nki", "tpu"];

/// Forms whose output is CUDA-flavoured C++. `musa` delegates to the `gpu` emitter.
const CUDA_FORMS: &[&str] = &["gpu", "musa"];

/// Enough of the CUDA vocabulary to PARSE an emitted kernel with a host C++ compiler.
///
/// No semantics, no runtime, no device -- its only job is to let clang answer "is this a
/// program?" for kernels that nothing on a machine without `nvcc` has ever read.
///
/// Written down here rather than kept as an asset so it cannot drift from the test that
/// uses it. Every name in it appears in emitted output; when the emitters learn a new
/// intrinsic this stub has to learn it too, and the failure mode of forgetting is a FALSE
/// ALARM -- which is why the first run of this check reported 194 broken kernels when 175
/// of them were the stub's own gaps.
const CUDA_STUB: &str = r#"
#pragma once
#include <math.h>
#include <stdint.h>
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)
struct dim3 { unsigned x, y, z; };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
static inline void __syncthreads(void) {}
static inline void __syncwarp(unsigned = 0u) {}
static inline float __shfl_down_sync(unsigned, float v, int, int = 32) { return v; }
static inline float __shfl_sync(unsigned, float v, int, int = 32) { return v; }
static inline float __shfl_xor_sync(unsigned, float v, int, int = 32) { return v; }
static inline int   __shfl_down_sync(unsigned, int v, int, int = 32) { return v; }
static inline unsigned __ballot_sync(unsigned, int) { return 0u; }
struct __half {
  float v;
  __half() : v(0) {}
  __half(float f) : v(f) {}
  operator float() const { return v; }
};
static inline __half __float2half(float f) { return __half(f); }
static inline float  __half2float(__half h) { return (float)h; }
static inline __half hexp(__half h) { return __half(expf((float)h)); }
static inline __half hlog(__half h) { return __half(logf((float)h)); }
static inline __half hsqrt(__half h) { return __half(sqrtf((float)h)); }
/* Real CUDA conversion intrinsics. Missing from this stub is why `quantize` looked
   broken -- the emitter is right to call it. */
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
static inline int __float2int_rn(float x) { return (int)(x + (x >= 0.0f ? 0.5f : -0.5f)); }
static inline int __float2int_rd(float x) { return (int)x; }
static inline int __float2int_ru(float x) { return (int)x + 1; }
static inline int __float2int_rz(float x) { return (int)x; }
static inline float __int2float_rn(int x) { return (float)x; }
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static inline float __expf(float x) { return expf(x); }
static inline float __logf(float x) { return logf(x); }
static inline float __fdividef(float a, float b) { return a / b; }
"#;

/// Forms whose output is vendor C/C++ that a host compiler can be coaxed into parsing.
/// `bang` is MLU C++, `gaudi` is TPC-C; neither compiler is on this machine.
const VENDOR_C_FORMS: &[(&str, &str)] = &[("bang", BANG_STUB), ("gaudi", TPC_STUB)];

/// MLU C++, enough to parse.
const BANG_STUB: &str = r#"
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
#define __mlu_entry__
#define __mlu_func__
#define __mlu_device__
#define __nram__
#define __wram__
#define __ldram__
#define __mlu_shared__
enum MemDir { GDRAM2NRAM, NRAM2GDRAM, NRAM2NRAM, GDRAM2GDRAM, NRAM2WRAM };
static int taskId, taskDim, clusterId, clusterDim, coreId, coreDim;
static inline void __memcpy(void*, const void*, size_t, MemDir) {}
static inline void __sync_all(void) {}
static inline void __sync_cluster(void) {}
template <class... A> static inline void __bang_add(A...) {}
template <class... A> static inline void __bang_sub(A...) {}
template <class... A> static inline void __bang_mul(A...) {}
template <class... A> static inline void __bang_div(A...) {}
template <class... A> static inline void __bang_maxequal(A...) {}
template <class... A> static inline void __bang_minequal(A...) {}
template <class... A> static inline void __bang_abs(A...) {}
template <class... A> static inline void __bang_active_exp(A...) {}
template <class... A> static inline void __bang_active_relu(A...) {}
template <class... A> static inline void __bang_active_sigmoid(A...) {}
template <class... A> static inline void __bang_active_tanh(A...) {}
template <class... A> static inline void __bang_active_sqrt(A...) {}
template <class... A> static inline void __bang_log(A...) {}
template <class... A> static inline void __bang_half2float(A...) {}
template <class... A> static inline void __bang_float2half_dn(A...) {}
template <class... A> static inline void __bang_mul_scalar(A...) {}
template <class... A> static inline void __bang_add_scalar(A...) {}
template <class... A> static inline void __bang_write_value(A...) {}
template <class... A> static inline void __bang_max(A...) {}
template <class... A> static inline void __bang_min(A...) {}
template <class... A> static inline void __bang_sumpool(A...) {}
template <class... A> static inline void __bang_transpose(A...) {}
typedef __fp16 half;
"#;

/// TPC-C, enough to parse.
///
/// ONE universal value type rather than exact signatures: `tpcv` converts to and from
/// float, subscripts and arithmetic-combines. Guessing each intrinsic's real argument list
/// would turn every gap in the stub into a false alarm about the emitter — the first
/// version of this stub reported 127 failures of which four causes out of five were its
/// own.
const TPC_STUB: &str = r#"
#pragma once
#include <math.h>
struct tpcv {
  float v[512];
  tpcv() { for (int i = 0; i < 512; ++i) v[i] = 0.f; }
  tpcv(float f) { for (int i = 0; i < 512; ++i) v[i] = f; }
  operator float() const { return v[0]; }
  float& operator[](int i) { return v[i]; }
  const float& operator[](int i) const { return v[i]; }
  tpcv operator+(const tpcv&) const { return *this; }
  tpcv operator-(const tpcv&) const { return *this; }
  tpcv operator*(const tpcv&) const { return *this; }
  tpcv operator/(const tpcv&) const { return *this; }
  tpcv& operator+=(const tpcv&) { return *this; }
};
typedef tpcv float256; typedef tpcv float64; typedef tpcv short512;
typedef tpcv bfloat128; typedef tpcv half256; typedef float half;
struct int5 { int v[5]; int& operator[](int i) { return v[i]; } };
typedef struct { int h; } tensor;
static inline int5 get_index_space_offset(void) { int5 r{}; return r; }
static inline int5 get_index_space_size(void)   { int5 r{}; return r; }
static inline int5 get_index_space_stride(void) { int5 r{}; return r; }
#define TPC_V(name) template <class... A> static inline tpcv name(A...) { return tpcv(); }
#define TPC_VOID(name) template <class... A> static inline void name(A...) {}
TPC_V(v_f32_ld_tnsr) TPC_V(v_f32_mov_scalar) TPC_V(v_f32_mov_b) TPC_V(v_f32_add)
TPC_V(v_f32_sub) TPC_V(v_f32_mul) TPC_V(v_f32_neg) TPC_V(v_f32_abs) TPC_V(v_f32_exp)
TPC_V(v_f32_rcp) TPC_V(v_f32_max_b) TPC_V(v_f32_reduce_add) TPC_V(v_f32_reduce_max)
/* Also called by mlir_to_gaudi and previously absent here, which reported five
   kernels broken for a gap in this stub. Same `_scalar` / `_b` convention as the
   entries above. */
TPC_V(v_f32_div) TPC_V(v_f32_exp_scalar) TPC_V(v_f32_log_scalar) TPC_V(v_f32_mul_b)
TPC_V(v_f16_ld_tnsr) TPC_V(v_f16_mov_scalar) TPC_V(v_f16_mov_b) TPC_V(v_f16_sub)
TPC_V(v_f16_mul) TPC_V(v_f16_mul_b) TPC_V(v_f16_neg) TPC_V(v_f16_abs) TPC_V(v_f16_exp)
TPC_V(v_f16_rcp) TPC_V(v_f16_max_b) TPC_V(v_f16_reduce_add) TPC_V(v_f16_reduce_max)
TPC_V(v_i32_ld_tnsr) TPC_V(v_i32_mov_scalar) TPC_V(v_bf16_ld_tnsr)
TPC_VOID(v_f32_st_tnsr) TPC_VOID(v_f16_st_tnsr) TPC_VOID(v_i32_st_tnsr)
TPC_VOID(v_bf16_st_tnsr)
"#;

/// Forms whose output includes vendor headers by PATH. Each entry is
/// (form, stub body, the header paths the emitted source asks for).
///
/// The stub bodies use a shared `#ifndef` guard rather than `#pragma once`, because the
/// same text is written to several paths and `#pragma once` is per-FILE: without a shared
/// guard every kernel fails on a redefinition the harness caused. That mistake was made
/// twice before it was understood -- once here and once with `bang`, which includes
/// `bang.h` itself and must therefore not ALSO be force-included.
const HEADER_FORMS: &[(&str, &str, &[&str])] = &[
    (
        "hexagon",
        HEX_STUB,
        &[
            "HTP/core/ops.h",
            "HTP/hvx_exp.h",
            "HTP/hvx_matmul.h",
            "HTP/hvx_softmax.h",
            "HTP/hvx_rms_norm.h",
            "HTP/hvx_utils.h",
        ],
    ),
    (
        "ttmetal",
        TT_STUB,
        &[
            "compute_kernel_api/common.h",
            "compute_kernel_api/matmul.h",
            "compute_kernel_api/softmax.h",
            "compute_kernel_api/rms_norm.h",
            "compute_kernel_api/eltwise_unary/exp.h",
        ],
    ),
];

/// Qualcomm HVX, enough to parse.
const HEX_STUB: &str = r#"
#ifndef TILE_HEX_STUB_H
#define TILE_HEX_STUB_H
#include <math.h>
template <class... A> static inline void hvx_vexp_f32(A...) {}
template <class... A> static inline void hvx_exp(A...) {}
template <class... A> static inline void hvx_softmax(A...) {}
template <class... A> static inline void hvx_softmax_f32(A...) {}
template <class... A> static inline void hvx_rms_norm(A...) {}
template <class... A> static inline void hvx_rms_norm_f32(A...) {}
template <class... A> static inline void hvx_matmul(A...) {}
template <class... A> static inline void hvx_matmul_f32(A...) {}
/* Called by mlir_to_hexagon and previously absent, which reported working kernels as
   broken. Same naming convention as the entries above. */
template <class... A> static inline void hvx_vadd_f32(A...) {}
template <class... A> static inline void hvx_vcopy_f32(A...) {}
template <class... A> static inline void hvx_vmpy_f(A...) {}
template <class... A> static inline void hvx_vload_f32_aligned(A...) {}
template <class... A> static inline void hvx_vstore_f32_aligned(A...) {}
#endif
"#;

/// Tenstorrent Metalium, enough to parse. `matmul_tiles` is plural in the real API; a
/// stub with only the singular reported 19 false failures.
const TT_STUB: &str = r#"
#ifndef TILE_TT_STUB_H
#define TILE_TT_STUB_H
#define MAIN main_kernel()
namespace tt { struct CB { static constexpr int c_in0 = 0, c_in1 = 1, c_out0 = 2; }; }
template <class... A> static inline void init_sfpu(A...) {}
template <class... A> static inline void exp_tile_init(A...) {}
template <class... A> static inline void exp_tile(A...) {}
template <class... A> static inline void softmax_tile_init(A...) {}
template <class... A> static inline void softmax_tile(A...) {}
template <class... A> static inline void rms_norm_tile_init(A...) {}
template <class... A> static inline void rms_norm_tile(A...) {}
template <class... A> static inline void matmul_tile(A...) {}
/* Called by mlir_to_ttmetal and previously absent. `binary_op_init_common` and
   `add_tiles` are real TT-Metal compute-kernel API. */
template <class... A> static inline void binary_op_init_common(A...) {}
template <class... A> static inline void add_tiles(A...) {}
template <class... A> static inline void add_tiles_init(A...) {}
template <class... A> static inline void copy_tile_init(A...) {}
template <class... A> static inline void matmul_tiles(A...) {}
template <class... A> static inline void mm_init(A...) {}
template <class... A> static inline void mm_init_short(A...) {}
template <class... A> static inline void cb_wait_front(A...) {}
template <class... A> static inline void cb_pop_front(A...) {}
template <class... A> static inline void cb_push_back(A...) {}
template <class... A> static inline void cb_reserve_back(A...) {}
template <class... A> static inline void acquire_dst(A...) {}
template <class... A> static inline void release_dst(A...) {}
template <class... A> static inline void pack_tile(A...) {}
template <class... A> static inline void copy_tile(A...) {}
#endif
"#;

/// Shapes that broke kernels elsewhere: a prime, a non-multiple of the workgroup, one
/// wider than any threadgroup, and more than one row.
const SHAPES: &[(usize, usize)] = &[(1, 17), (1, 127), (1, 256), (1, 4096), (3, 129)];

fn kernel(op: &str, rows: usize, cols: usize, reduces: bool) -> String {
    let out_cols = if reduces { 1 } else { cols };
    format!(
        "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
         attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant({rows} : i32) : i32\n\
         \x20   %c = llvm.mlir.constant({cols} : i32) : i32\n\
         \x20   %o = llvm.mlir.constant({out_cols} : i32) : i32\n\
         \x20   %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
         \x20   %y = llvm.call @__tile_{op}_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %o) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n"
    )
}

fn have_python() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
fn every_python_backend_emits_python_that_parses() {
    if !have_python() {
        eprintln!("emitted_parses: skipped, no python3 on PATH");
        return;
    }
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-emitted-parses-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for &(rows, cols) in SHAPES {
        for (op, reduces) in [("exp", false), ("reduce_sum", true), ("softmax", false)] {
            let src = dir.join(format!("{op}_{rows}x{cols}.mlir"));
            std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
            for form in PYTHON_FORMS {
                let out = dir.join(format!("{form}_{op}_{rows}x{cols}.py"));
                let emitted = Command::new(exe)
                    .arg(&src)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                // A refusal is a legitimate answer and not this test's business. Only
                // text that WAS emitted has to be a program.
                if !emitted.status.success() || !out.exists() {
                    continue;
                }
                let parse = Command::new("python3")
                    .arg("-c")
                    .arg("import ast,sys; ast.parse(open(sys.argv[1]).read())")
                    .arg(&out)
                    .output()
                    .expect("python3 runs");
                checked += 1;
                if !parse.status.success() {
                    let why = String::from_utf8_lossy(&parse.stderr)
                        .lines()
                        .last()
                        .unwrap_or("")
                        .to_string();
                    bad.push(format!("{form} {op} {rows}x{cols}: {why}"));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitted Python files parsed");
    for b in &bad {
        eprintln!("  {b}");
    }
    // A vacuous pass is the failure mode to guard against: if the emitters all started
    // refusing, `checked` would be 0 and every file would trivially parse.
    assert!(
        checked >= PYTHON_FORMS.len() * 3,
        "only {checked} files were emitted at all -- the backends are refusing, and this \
         test would otherwise pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emitted files are not valid Python. Emitted source that nothing compiles is \
         source nobody has checked, which is how an f16 `relu` that never once compiled \
         sat in the SPIR-V backend reporting successful lowerings.",
        bad.len()
    );
}

/// The CUDA-flavoured backends, syntax-checked with a host C++ compiler and a stub.
///
/// Nothing on a machine without `nvcc` had ever read these. The first run found the `gpu`
/// matmul arm emitting a TODO comment and registering a result variable it never declared,
/// so the kernel wrote `p2[goff] = _v2;` for an undeclared `_v2` -- while `--list-ops`
/// reported `matmul: yes` for that backend. The generality matrix was counting a file that
/// is not a program as a lowering.
///
/// It refuses now, and this test is what keeps it refusing rather than going back to
/// emitting something that reads like a kernel.
#[test]
fn every_cuda_backend_emits_cxx_that_parses() {
    let Some(cxx) = which("clang++").or_else(|| which("g++")) else {
        eprintln!("emitted_parses: skipped, no host C++ compiler");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-cuda-parses-{}", std::process::id()));
    let inc = dir.join("stub");
    std::fs::create_dir_all(&inc).expect("a scratch directory");
    // Both names, same contents: the emitters use one vocabulary and differ in the header
    // they include.
    for h in ["cuda_runtime.h", "musa_runtime.h"] {
        std::fs::write(inc.join(h), CUDA_STUB).expect("write stub");
    }
    // The emitters include both the runtime and the fp16 header; the stub text is
    // identical, so writing it twice under two names would redefine `dim3` and friends.
    // One name includes the other.
    std::fs::write(
        inc.join("cuda_fp16.h"),
        "#pragma once\n#include <cuda_runtime.h>\n",
    )
    .expect("write stub");
    std::fs::write(
        inc.join("musa_fp16.h"),
        "#pragma once\n#include <musa_runtime.h>\n",
    )
    .expect("write stub");

    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for &(rows, cols) in SHAPES {
        for (op, reduces) in [("exp", false), ("reduce_sum", true), ("softmax", false)] {
            let src = dir.join(format!("{op}_{rows}x{cols}.mlir"));
            std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
            for form in CUDA_FORMS {
                let out = dir.join(format!("{form}_{op}_{rows}x{cols}.cu"));
                let emitted = Command::new(exe)
                    .arg(&src)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                if !emitted.status.success() || !out.exists() {
                    continue; // a refusal is an answer, not a failure
                }
                let parse = Command::new(&cxx)
                    .args(["-std=c++17", "-fsyntax-only", "-x", "c++", "-I"])
                    .arg(&inc)
                    .arg(&out)
                    .output()
                    .expect("the C++ compiler runs");
                checked += 1;
                if !parse.status.success() {
                    let why = String::from_utf8_lossy(&parse.stderr)
                        .lines()
                        .find(|l| l.contains("error:"))
                        .unwrap_or("")
                        .to_string();
                    bad.push(format!("{form} {op} {rows}x{cols}: {why}"));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitted CUDA files parsed");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(
        checked >= CUDA_FORMS.len() * 3,
        "only {checked} CUDA files were emitted -- the backends are refusing everything \
         and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emitted CUDA files are not valid C++. An emitter that produces a non-program \
         while `--list-ops` reports the op as lowered is the coverage matrix counting a \
         file nobody can compile.",
        bad.len()
    );
}

fn which(tool: &str) -> Option<String> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if p.is_empty() {
        None
    } else {
        Some(p)
    }
}

/// True when `xcrun metal` can actually compile. `xcrun` being on PATH is not enough:
/// Xcode ships the driver without the MetalToolchain component, and then every compile
/// fails with "cannot execute tool 'metal' due to missing Metal Toolchain" — a machine
/// problem, not a kernel problem (same distinction as `Feedback::Unavailable` in
/// `toolchain.rs`). Probing once keeps the real-compiler gates honest on machines where
/// the component was never downloaded.
fn metal_toolchain_works(xcrun: &str) -> bool {
    let dir = std::env::temp_dir().join(format!("tile-metal-probe-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let src = dir.join("probe.metal");
    let air = dir.join("probe.air");
    let _ = std::fs::write(
        &src,
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void k(device float* p [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n\
         \x20   p[i] = 1.0;\n}\n",
    );
    let ok = Command::new(xcrun)
        .args(["metal", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&air)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

/// After `which("xcrun")`, skip unless the Metal toolchain component is present.
macro_rules! require_metal_toolchain {
    ($xcrun:expr) => {
        if !metal_toolchain_works(&$xcrun) {
            eprintln!(
                "emitted_parses: skipped, xcrun is present but Metal Toolchain is not \
                 (xcodebuild -downloadComponent MetalToolchain)"
            );
            return;
        }
    };
}

/// `bang` and `gaudi`, syntax-checked with a host C++ compiler and a stub per form.
///
/// Neither `cncc` nor the TPC compiler is on this machine, so nothing had ever read these.
/// `bang` came back clean at 229 of 229 across the full corpus. `gaudi` did not: its two
/// scalar-loop arms wrote `v_1[_li] = logf(v_0[_li]);` and never declared `v_1`, while
/// `--list-ops` reported `log: yes` and `rsqrt: yes` -- the identical defect to the CUDA
/// matmul's undeclared `_v2`, in a different backend, found the same way.
#[test]
fn every_vendor_c_backend_emits_cxx_that_parses() {
    let Some(cxx) = which("clang++").or_else(|| which("g++")) else {
        eprintln!("emitted_parses: skipped, no host C++ compiler");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-vendor-parses-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for (form, stub) in VENDOR_C_FORMS {
        // `bang` emits `#include <bang.h>` itself, so the stub goes in the include path
        // under that name and must NOT also be force-included: two paths to the same text
        // defeat `#pragma once` and every kernel fails on a redefinition that is the
        // harness's doing, not the emitter's. `gaudi` includes nothing, so it is
        // force-included.
        let stub_path = dir.join(if *form == "bang" {
            "bang.h".to_string()
        } else {
            format!("{form}_stub.h")
        });
        std::fs::write(&stub_path, stub).expect("write stub");
        let force_include = *form != "bang";
        for &(rows, cols) in SHAPES {
            for (op, reduces) in [
                ("exp", false),
                ("log", false),
                ("rsqrt", false),
                ("reduce_sum", true),
                ("softmax", false),
            ] {
                let src = dir.join(format!("{form}_{op}_{rows}x{cols}.mlir"));
                std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
                let out = dir.join(format!("{form}_{op}_{rows}x{cols}.c"));
                let emitted = Command::new(exe)
                    .arg(&src)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                if !emitted.status.success() || !out.exists() {
                    continue; // a refusal is an answer
                }
                // `void main(tensor, tensor)` is TPC-C's legitimate entry point and C++
                // reserves `main`. Renamed in a COPY: that is the host compiler's
                // restriction, not the emitter's fault, and conflating the two would
                // report a defect that is not there.
                let text = std::fs::read_to_string(&out).unwrap_or_default();
                let chk = dir.join(format!("{form}_{op}_{rows}x{cols}.chk.cpp"));
                std::fs::write(&chk, text.replace("\nvoid main(", "\nvoid tpc_main("))
                    .expect("write");
                let mut cmd = Command::new(&cxx);
                cmd.args(["-std=c++17", "-fsyntax-only"]);
                if force_include {
                    cmd.arg("-include").arg(&stub_path);
                }
                let parse = cmd
                    .args(["-I"])
                    .arg(&dir)
                    .args(["-x", "c++"])
                    .arg(&chk)
                    .output()
                    .expect("the C++ compiler runs");
                checked += 1;
                if !parse.status.success() {
                    let why = String::from_utf8_lossy(&parse.stderr)
                        .lines()
                        .find(|l| l.contains("error:"))
                        .unwrap_or("")
                        .to_string();
                    bad.push(format!("{form} {op} {rows}x{cols}: {why}"));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitted vendor-C files parsed");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(
        checked >= VENDOR_C_FORMS.len() * 3,
        "only {checked} vendor-C files were emitted -- the backends are refusing \
         everything and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emitted vendor-C files are not valid C++. An emitter that writes into a \
         variable it never declares, while `--list-ops` reports the op as lowered, is the \
         coverage matrix counting a file nobody can compile.",
        bad.len()
    );
}

/// `hexagon` and `ttmetal`, which include vendor headers by path.
///
/// Both came back clean at 140 of 140 -- but only after five rounds of fixing the STUB:
/// missing header paths, `#pragma once` failing to dedupe across copies, `mm_init` and
/// `matmul_tiles` absent. Every one of those produced a confident-looking failure count
/// that was entirely the harness's. The count that matters is the one taken after the
/// instrument stops being the broken thing.
#[test]
fn every_header_including_backend_emits_cxx_that_parses() {
    let Some(cxx) = which("clang++").or_else(|| which("g++")) else {
        eprintln!("emitted_parses: skipped, no host C++ compiler");
        return;
    };
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-header-parses-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for (form, stub, headers) in HEADER_FORMS {
        let inc = dir.join(format!("{form}_inc"));
        for h in *headers {
            let path = inc.join(h);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("stub dirs");
            }
            std::fs::write(&path, stub).expect("write stub");
        }
        for &(rows, cols) in SHAPES {
            for (op, reduces) in [("exp", false), ("softmax", false), ("reduce_sum", true)] {
                let src = dir.join(format!("{form}_{op}_{rows}x{cols}.mlir"));
                std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
                let out = dir.join(format!("{form}_{op}_{rows}x{cols}.cpp"));
                let emitted = Command::new(exe)
                    .arg(&src)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                if !emitted.status.success() || !out.exists() {
                    continue;
                }
                let parse = Command::new(&cxx)
                    .args(["-std=c++17", "-fsyntax-only", "-I"])
                    .arg(&inc)
                    .args(["-x", "c++"])
                    .arg(&out)
                    .output()
                    .expect("the C++ compiler runs");
                checked += 1;
                if !parse.status.success() {
                    let why = String::from_utf8_lossy(&parse.stderr)
                        .lines()
                        .find(|l| l.contains("error:"))
                        .unwrap_or("")
                        .to_string();
                    bad.push(format!("{form} {op} {rows}x{cols}: {why}"));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitted header-including files parsed");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(
        checked >= HEADER_FORMS.len() * 2,
        "only {checked} files were emitted -- the backends are refusing everything"
    );
    assert!(
        bad.is_empty(),
        "{} emitted files are not valid C++",
        bad.len()
    );
}

/// The MLIR-emitting backends, checked against the one invariant SSA text has.
///
/// `mlir-opt` is not on this machine, so `linalg`, `rvv` and `pto` are read by nothing.
/// But SSA has a property that needs no compiler: every `%value` used as an operand must
/// be defined earlier, by a `%value =` or as a block/function argument.
///
/// That is not an arbitrary check. It is EXACTLY the defect found three times today in
/// backends that do have compilers -- the CUDA matmul writing an undeclared `_v2`, and
/// gaudi's `log` and `rsqrt` writing an undeclared `v_1`. A backend nobody can compile is
/// the most likely place for a fourth, and this finds it without one.
///
/// 416 emissions across the corpus were clean when this was written; it is here so the
/// fourth cannot arrive unnoticed.
#[test]
fn mlir_backends_use_no_value_before_defining_it() {
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-ssa-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for form in ["linalg", "rvv"] {
        for &(rows, cols) in SHAPES {
            for (op, reduces) in [("exp", false), ("softmax", false), ("reduce_sum", true)] {
                let src = dir.join(format!("{form}_{op}_{rows}x{cols}.in.mlir"));
                std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
                let out = dir.join(format!("{form}_{op}_{rows}x{cols}.out.mlir"));
                let emitted = Command::new(exe)
                    .arg(&src)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                if !emitted.status.success() || !out.exists() {
                    continue;
                }
                checked += 1;
                let text = std::fs::read_to_string(&out).unwrap_or_default();
                for (n, v) in ssa_use_before_def(&text) {
                    bad.push(format!(
                        "{form} {op} {rows}x{cols} line {n}: {v} used before any definition"
                    ));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitted MLIR files checked for use-before-def");
    for b in bad.iter().take(5) {
        eprintln!("  {b}");
    }
    assert!(checked >= 6, "only {checked} MLIR files were emitted");
    assert!(
        bad.is_empty(),
        "{} uses of an undefined SSA value. That is the defect the CUDA matmul and \
         gaudi's log both had, in a backend no compiler here can read.",
        bad.len()
    );
}

/// Every `(line number, %value)` used before anything defines it.
///
/// Definitions are what stands left of the first `=`, plus typed arguments. This is the
/// one invariant that needs no toolchain, and it is not arbitrary: using a value that was
/// never defined is exactly the defect found in the CUDA matmul and in gaudi's `log`.
fn ssa_use_before_def(text: &str) -> Vec<(usize, String)> {
    let mut defined: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split("//").next().unwrap_or("");
        if let Some((lhs, _)) = line.split_once('=') {
            for v in ssa_names(lhs) {
                defined.push(v);
            }
        }
        for v in typed_args(line) {
            defined.push(v);
        }
        let rhs = line.split_once('=').map(|x| x.1).unwrap_or(line);
        for v in ssa_names(rhs) {
            if !defined.contains(&v) {
                out.push((n + 1, v));
            }
        }
    }
    out
}

/// Every `%name` in a fragment.
fn ssa_names(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let start = i;
            i += 1;
            while i < b.len()
                && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.' || b[i] == b'$')
            {
                i += 1;
            }
            if i > start + 1 {
                out.push(s[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

/// `%arg0: tensor<...>` — a name introduced by a signature rather than an assignment.
fn typed_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (idx, _) in s.match_indices(':') {
        let before = &s[..idx];
        if let Some(last) = ssa_names(before).pop() {
            if before.trim_end().ends_with(&last) {
                out.push(last);
            }
        }
    }
    out
}

/// `csl`, the last form with no compiler here at all.
///
/// Cerebras CSL is Zig-like and clang will not parse it. But it declares its names
/// explicitly -- `param X:`, `var X:`, `task X()`, and loop captures `|i|` -- so the same
/// invariant that reached MLIR reaches this: every identifier used must be declared, and
/// the delimiters must balance.
///
/// 70 emissions were clean when this was written. It exists so that the fourth
/// undeclared-variable bug, if it comes, does not come here unnoticed.
#[test]
fn csl_declares_every_name_it_uses() {
    const KEYWORDS: &[&str] = &[
        "param", "var", "const", "task", "fn", "comptime", "for", "if", "else", "while", "return",
        "void", "true", "false", "and", "or", "not", "export", "layout", "i16", "i32", "u16",
        "u32", "f16", "f32", "bool", "struct", "union", "enum", "switch", "break", "continue",
    ];
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-csl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for &(rows, cols) in SHAPES {
        for (op, reduces) in [("exp", false), ("softmax", false), ("reduce_sum", true)] {
            let src = dir.join(format!("{op}_{rows}x{cols}.mlir"));
            std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
            let out = dir.join(format!("{op}_{rows}x{cols}.csl"));
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "csl", "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !out.exists() {
                continue;
            }
            checked += 1;
            let text = std::fs::read_to_string(&out).unwrap_or_default();
            let mut declared: Vec<String> = Vec::new();
            for w in ["param", "var", "const", "task", "fn"] {
                let mut rest = text.as_str();
                while let Some(i) = rest.find(w) {
                    rest = &rest[i + w.len()..];
                    let name: String = rest
                        .trim_start()
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        declared.push(name);
                    }
                }
            }
            // Loop captures: `|i|`
            let mut rest = text.as_str();
            while let Some(i) = rest.find('|') {
                let after = &rest[i + 1..];
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && after[name.len()..].starts_with('|') {
                    declared.push(name);
                }
                rest = &rest[i + 1..];
            }
            let (mut par, mut brk, mut brc) = (0i32, 0i32, 0i32);
            for (n, raw) in text.lines().enumerate() {
                let line = raw.split("//").next().unwrap_or("");
                for c in line.chars() {
                    match c {
                        '(' => par += 1,
                        ')' => par -= 1,
                        '[' => brk += 1,
                        ']' => brk -= 1,
                        '{' => brc += 1,
                        '}' => brc -= 1,
                        _ => {}
                    }
                }
                // Walk the line word by word. A declaring keyword consumes the NEXT
                // word (the name it introduces) and `@builtins` are skipped whole.
                //
                // The first version did this by string surgery -- stripping the keyword
                // and then the name out of the line -- and mangled `compute_task` into
                // `compute_`, reporting five identifiers the emitter had declared
                // perfectly well. String surgery on the thing being measured is the same
                // mistake as a stub with a gap in it.
                let mut words: Vec<String> = Vec::new();
                let mut chars = line.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '@' {
                        while chars
                            .peek()
                            .is_some_and(|c| c.is_alphanumeric() || *c == '_')
                        {
                            chars.next();
                        }
                    } else if c.is_alphabetic() || c == '_' {
                        let mut w = String::from(c);
                        while chars
                            .peek()
                            .is_some_and(|c| c.is_alphanumeric() || *c == '_')
                        {
                            w.push(chars.next().unwrap());
                        }
                        words.push(w);
                    }
                }
                let mut skip_next = false;
                for word in &words {
                    if skip_next {
                        skip_next = false;
                        continue;
                    }
                    if ["param", "var", "const", "task", "fn"].contains(&word.as_str()) {
                        skip_next = true;
                        continue;
                    }
                    if KEYWORDS.contains(&word.as_str()) || declared.iter().any(|d| d == word) {
                        continue;
                    }
                    bad.push(format!(
                        "{op} {rows}x{cols} line {}: `{word}` not declared",
                        n + 1
                    ));
                }
            }
            if par != 0 || brk != 0 || brc != 0 {
                bad.push(format!(
                    "{op} {rows}x{cols}: delimiters unbalanced ({par},{brk},{brc})"
                ));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitted CSL files checked");
    for b in bad.iter().take(5) {
        eprintln!("  {b}");
    }
    assert!(checked >= 3, "only {checked} CSL files were emitted");
    assert!(
        bad.is_empty(),
        "{} undeclared names or unbalanced delimiters in CSL",
        bad.len()
    );
}

/// The `pico` listing, which no assembler here can read.
///
/// Three invariants it has anyway: the embedded `@manifest` is JSON that downstream
/// tooling parses, one mnemonic carries one opcode, and the program ends with `end`.
/// Checked against the committed goldens, since the emitter itself is feature-gated.
#[test]
fn pico_listings_are_self_consistent() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/golden");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("emitted_parses: skipped, no golden directory");
        return;
    };
    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        if !path.to_string_lossy().ends_with(".pico.s") {
            continue;
        }
        checked += 1;
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // One mnemonic, one opcode -- a mnemonic carrying two is a table bug.
        let mut seen: Vec<(String, String)> = Vec::new();
        let mut last = String::new();
        for line in text.lines() {
            let Some((lhs, rhs)) = line.split_once("; op=") else {
                continue;
            };
            let mnemonic = lhs.trim().to_string();
            if mnemonic.is_empty() || !mnemonic.chars().all(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            let opcode: String = rhs.chars().take_while(|c| !c.is_whitespace()).collect();
            if let Some((_, prev)) = seen.iter().find(|(m, _)| *m == mnemonic) {
                if *prev != opcode {
                    bad.push(format!(
                        "{name}: `{mnemonic}` carries both {prev} and {opcode}"
                    ));
                }
            } else {
                seen.push((mnemonic.clone(), opcode));
            }
            last = mnemonic;
        }
        if seen.is_empty() {
            bad.push(format!("{name}: no instructions"));
        } else if last != "end" {
            bad.push(format!("{name}: ends with `{last}`, not `end`"));
        }
        // The manifest is JSON in comments; downstream tooling reads it.
        if let Some(i) = text.find("@manifest") {
            let body: String = text[i..]
                .lines()
                .skip(1)
                .take_while(|l| l.trim_start().starts_with(';'))
                .map(|l| l.trim_start().trim_start_matches(';'))
                .collect::<Vec<_>>()
                .join("\n");
            let opens = body.matches('{').count();
            let closes = body.matches('}').count();
            if opens == 0 || opens != closes {
                bad.push(format!(
                    "{name}: @manifest braces do not balance ({opens}/{closes})"
                ));
            }
        } else {
            bad.push(format!("{name}: no @manifest block"));
        }
    }
    eprintln!("emitted_parses: {checked} pico listings checked");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(checked >= 1, "no pico goldens were found to check");
    assert!(bad.is_empty(), "{} pico listing defects", bad.len());
}

/// Metal, compiled by the real compiler rather than parsed by a stub.
///
/// `xcrun metal` is on this machine and only two goldens plus the `-r` path ever invoked
/// it. `topk` — a kernel with a fixture in this repo and a passing test — did not compile:
/// it wrote through a `device const` pointer, because only the last of its three buffers
/// was declared writable and it has two outputs.
///
/// A test that converts MLIR and greps the result for an idiom cannot see that. A string
/// containing the right idiom is still a string.
#[test]
fn metal_kernels_compile_with_the_real_compiler() {
    let Some(xcrun) = which("xcrun") else {
        eprintln!("emitted_parses: skipped, no xcrun");
        return;
    };
    require_metal_toolchain!(xcrun);
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-metal-cc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for &(rows, cols) in SHAPES {
        for (op, reduces) in [
            ("exp", false),
            ("softmax", false),
            ("reduce_sum", true),
            ("rms_norm", false),
            ("transpose", false),
            ("layernorm", false),
        ] {
            let src = dir.join(format!("{op}_{rows}x{cols}.mlir"));
            std::fs::write(&src, kernel(op, rows, cols, reduces)).expect("write");
            let out = dir.join(format!("{op}_{rows}x{cols}.metal"));
            let _ = std::fs::remove_file(&out);
            let emitted = Command::new(exe)
                .arg(&src)
                .args(["-t", "msl", "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !emitted.status.success() || !out.exists() {
                continue; // a refusal is an answer
            }
            checked += 1;
            let air = dir.join(format!("{op}_{rows}x{cols}.air"));
            let cc = Command::new(&xcrun)
                .args(["metal", "-c"])
                .arg(&out)
                .arg("-o")
                .arg(&air)
                .output()
                .expect("xcrun runs");
            if !cc.status.success() {
                let why = String::from_utf8_lossy(&cc.stderr)
                    .lines()
                    .find(|l| l.contains("error:"))
                    .unwrap_or("")
                    .to_string();
                bad.push(format!("{op} {rows}x{cols}: {why}"));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} Metal kernels compiled");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(checked >= 10, "only {checked} Metal kernels were emitted");
    assert!(
        bad.is_empty(),
        "{} emitted Metal kernels do not compile. xcrun metal is on this machine; a \
         string check that greps for an idiom cannot replace it.",
        bad.len()
    );
}

/// Every MLIR fixture the MSL emitter carries in its OWN unit tests, compiled by the real
/// Metal compiler.
///
/// `metal_kernels_compile_with_the_real_compiler` above hand-lists six operations. The
/// emitter's test module holds seventy-odd MLIR modules written by whoever wrote each
/// lowering — the authoritative call form for each intrinsic, including arities no caller
/// in this repo exercises. Their tests assert on SUBSTRINGS of the emitted text
/// (`msl.contains("threadgroup")`), which is why nine of them emitted Metal that does not
/// compile while every test passed:
///
///   * `argmax`, `draft_verify` and `sample_top_p` each got a second `uint base` on top of
///     the shared row-striding prologue — a redefinition;
///   * `causal_mask` and both `kv_cache_update` kernels got a prologue reading a
///     `num_elements` their signature does not declare;
///   * `quantize` and `dequantize` emitted a three-statement loop body with no braces, so
///     `gid` was out of scope for the two statements that used it;
///   * `cast_bf16_f32` declared its 16-bit source buffer as `float*` and then read it with
///     `as_type<ushort>`, a cast between types of different size.
///
/// This gate needs no list to maintain: a fixture added to the emitter is compiled here
/// the moment it appears.
///
/// Needs `xcrun`; skips with a note otherwise.
#[test]
fn every_msl_fixture_in_the_emitter_compiles() {
    let Some(xcrun) = which("xcrun") else {
        eprintln!("emitted_parses: skipped, no xcrun");
        return;
    };
    require_metal_toolchain!(xcrun);
    let emitter = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rustc_codegen_tile/src/mlir_to_msl.rs");
    let Ok(src) = std::fs::read_to_string(&emitter) else {
        eprintln!(
            "emitted_parses: skipped, {} is not readable",
            emitter.display()
        );
        return;
    };

    // Every raw string in the emitter that holds a module with an entry point.
    let mut fixtures: Vec<String> = Vec::new();
    let mut rest = src.as_str();
    while let Some(open) = rest.find("r#\"") {
        let after = &rest[open + 3..];
        let Some(close) = after.find("\"#") else {
            break;
        };
        let body = &after[..close];
        rest = &after[close + 2..];
        if !body.contains("llvm.func") || !body.contains("hacc.entry") {
            continue;
        }
        // `format!` TEMPLATES are not modules: their `{{` escapes and `@{}` placeholder
        // reach the emitter verbatim, which then classifies the kernel by fallback and
        // emits a body for the wrong buffer count. Treating one as a fixture cost a
        // false "read-only variable is not assignable" that read exactly like the real
        // ones — the instrument, again, being the broken thing.
        if body.contains("{{") || body.contains("@{}") {
            continue;
        }
        fixtures.push(body.trim_start_matches('\n').to_string());
    }

    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-msl-fixtures-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let mut checked = 0usize;
    let mut refused = 0usize;
    let mut bad: Vec<String> = Vec::new();

    for (i, mlir) in fixtures.iter().enumerate() {
        let stem = format!("fixture{i:03}");
        let srcf = dir.join(format!("{stem}.mlir"));
        std::fs::write(&srcf, mlir).expect("write");
        let out = dir.join(format!("{stem}.metal"));
        let _ = std::fs::remove_file(&out);
        let emitted = Command::new(exe)
            .arg(&srcf)
            .args(["-t", "msl", "-o"])
            .arg(&out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !out.exists() {
            refused += 1; // two fixtures exist to BE refused; a refusal is an answer
            continue;
        }
        checked += 1;
        let cc = Command::new(&xcrun)
            .args(["metal", "-c"])
            .arg(&out)
            .arg("-o")
            .arg(dir.join(format!("{stem}.air")))
            .output()
            .expect("xcrun runs");
        if !cc.status.success() {
            let entry = mlir
                .lines()
                .find_map(|l| l.trim().strip_prefix("llvm.func @"))
                .and_then(|l| l.split('(').next())
                .unwrap_or(&stem)
                .to_string();
            bad.push(format!(
                "{entry}: {}",
                String::from_utf8_lossy(&cc.stderr)
                    .lines()
                    .find(|l| l.contains("error:"))
                    .map(|l| l.split("metal:").last().unwrap_or(l).trim().to_string())
                    .unwrap_or_default()
            ));
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("emitted_parses: {checked} emitter fixtures compiled, {refused} refused");
    for b in &bad {
        eprintln!("  {b}");
    }
    // Guard the vacuous pass: if the extraction stopped matching, or the emitter began
    // refusing, this would pass by compiling nothing.
    assert!(
        checked >= 50,
        "only {checked} fixtures were emitted -- the extraction or the emitter changed \
         shape and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} of the emitter's own fixtures emit Metal that does not compile. Their unit \
         tests pass because they assert on substrings of the emitted text.",
        bad.len()
    );
}

/// Every intrinsic that `tile_std` DECLARES and the MSL emitter HANDLES, compiled by the
/// real Metal compiler. Slow (a few hundred `xcrun metal` invocations); set `TILE_SWEEP=1`.
///
/// This answers the question the handoff called the largest unchecked thing in the repo:
/// 97 of the 274 `KernelType` variants are reached only by their own intrinsics with their
/// own arities, and nobody knew whether they compile. They could not be reached by hand
/// because reconstructing a call signature is guessing — but they do not have to be
/// reconstructed. `crates/tile_std/src/tile.rs` declares all 423 of them, every parameter
/// a `u32` or an `f32`, which is exactly enough to build a call.
///
/// What it does NOT know, and this is the honest limit: the declarations give each
/// parameter's TYPE but not whether a `u32` is a tile handle or a dimension. That is
/// inferred from parameter names, and the inference is a guess. It is a load-bearing one —
/// giving every name-classified tile parameter its own buffer instead of sharing one took
/// the failures from 7 to 62, `__tile_exp_f32` among them, which is how we know the
/// classification and not the emitter was wrong in those cases.
///
/// So this test asserts a RATCHET, not a verdict. It pins how many compile today and fails
/// if that number falls. A defect in one of the eleven it cannot classify would not be
/// reported as an emitter bug on this evidence alone.
#[test]
fn declared_intrinsics_compile_with_the_real_compiler() {
    if std::env::var("TILE_SWEEP").is_err() {
        eprintln!("emitted_parses: declared-intrinsic sweep skipped, set TILE_SWEEP=1");
        return;
    }
    let Some(xcrun) = which("xcrun") else {
        eprintln!("emitted_parses: skipped, no xcrun");
        return;
    };
    require_metal_toolchain!(xcrun);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (Ok(decls_src), Ok(emitter_src)) = (
        std::fs::read_to_string(root.join("../tile_std/src/tile.rs")),
        std::fs::read_to_string(root.join("../rustc_codegen_tile/src/mlir_to_msl.rs")),
    ) else {
        eprintln!("emitted_parses: skipped, the declarations or the emitter are not here");
        return;
    };

    // Names the MSL emitter has an arm for. A declaration it does not handle would be
    // refused, which says nothing about whether its lowering compiles.
    let handled: std::collections::HashSet<&str> = emitter_src
        .match_indices("\"__tile_")
        .filter_map(|(i, _)| {
            let rest = &emitter_src[i + 1..];
            rest.find('"').map(|e| &rest[..e])
        })
        .collect();

    let cases: Vec<(String, String)> = modules_from_declarations(&decls_src)
        .into_iter()
        .filter(|(name, _)| handled.contains(name.as_str()))
        .collect();

    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-declared-cc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let (mut ok, mut refused) = (0usize, 0usize);
    let (mut refused_no_arm, mut refused_specialized) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    for (name, mlir) in &cases {
        let srcf = dir.join(format!("{name}.mlir"));
        std::fs::write(&srcf, mlir).expect("write");
        let out = dir.join(format!("{name}.metal"));
        let _ = std::fs::remove_file(&out);
        let emitted = Command::new(exe)
            .arg(&srcf)
            .args(["-t", "msl", "-o"])
            .arg(&out)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !out.exists() {
            // A refusal is an answer -- but only if it is the RIGHT answer, and that had
            // never been looked at. Every one falls into exactly two kinds, and both are
            // the emitter behaving correctly:
            //
            //   * it has no arm for the intrinsic. These reach the corpus because the
            //     `handled` filter matches the NAME anywhere in the emitter source,
            //     comments and doc tables included, so a documented-but-unimplemented
            //     intrinsic looks handled right up until it is asked to lower.
            //   * a specialization guard fired: the kernel is baked to n_embd=4096, dk=64
            //     or n_hc=4 and this corpus passes a generic 256. Refusing beats emitting a
            //     kernel that reads out of bounds, and it is backlog #011's family.
            //
            // Anything OUTSIDE those two is a refusal nobody has explained, so it fails.
            let why = String::from_utf8_lossy(&emitted.stderr).to_string();
            if why.contains("has no arm for it") {
                refused_no_arm += 1;
            } else if why.contains("is specialized to") {
                refused_specialized += 1;
            } else {
                bad.push(format!(
                    "{name}: refused for a reason this test does not recognise -- {}",
                    why.lines()
                        .find(|l| l.contains("failed"))
                        .unwrap_or("(no message)")
                        .chars()
                        .take(120)
                        .collect::<String>()
                ));
            }
            refused += 1;
            continue;
        }
        let cc = Command::new(&xcrun)
            .args(["metal", "-c"])
            .arg(&out)
            .arg("-o")
            .arg(dir.join(format!("{name}.air")))
            .output()
            .expect("xcrun runs");
        if cc.status.success() {
            ok += 1;
        } else {
            bad.push(name.clone());
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: {ok} of {} declared+handled intrinsics compile, {refused} refused \
         ({refused_no_arm} no arm, {refused_specialized} specialization guard), {} that do \
         not compile",
        cases.len(),
        bad.len()
    );
    for b in &bad {
        eprintln!("  does not compile: {b}");
    }
    assert!(
        cases.len() >= 200,
        "only {} intrinsics were both declared and handled -- the extraction changed shape",
        cases.len()
    );
    // The ratchet. 236 of 241 compile and NOTHING is left unclassified.
    //
    // The road here is the point. It read 233/245 with a generator that classified operands
    // by parameter NAME and handed integer constants to the twelve declarations taking a
    // pointer; correcting the corpus made it 229/241 -- lower, and better. Then the safe
    // wrappers in tile_std turned out to state, at their call sites, exactly what the
    // declarations do not: `__tile_matvec_f16_bias(0, a.buf_id, w.buf_id, bias.buf_id, ..)`
    // says which operands are tiles, which are dimensions, and that dst is a literal 0.
    // That is an independent contract -- written by the safe API's author, not an emitter's
    // -- so using it is reading rather than reconstructing.
    //
    // With it, three of the seven remaining failures were the GENERATOR being wrong, and
    // four were the EMITTER: two matvec arms writing the addend and reading the output
    // (`p2[row] = total + p3[row]`, backwards), rope_prefill writing its own const input,
    // and topk_mask_scatter with a third brace-less multi-statement loop. All four assigned
    // through a `device const` pointer or used a variable out of scope; none compiled.
    //
    // 225 compiled before the fixture defects were fixed, on the OLD corpus, and is not
    // comparable with this figure. It is kept only as the record of that experiment.
    const RATCHET: usize = 236;
    assert!(
        ok >= RATCHET,
        "{ok} intrinsics compile, and {RATCHET} did when this ratchet was set. A lowering \
         that used to produce a Metal program no longer does."
    );
}

/// Every intrinsic `tile_std` declares, as a single-call MLIR module.
///
/// The declarations are the only non-guessed description of these call signatures in the
/// tree — 423 of them, every parameter a `u32` or an `f32`. Reconstructing them by hand is
/// what the handoff warned against; reading them is not.
fn modules_from_declarations(decls_src: &str) -> Vec<(String, String)> {
    let sites = wrapper_call_sites(decls_src);
    let mut cases: Vec<(String, String)> = Vec::new();
    for decl in decls_src.split("pub fn __tile_").skip(1) {
        let Some(paren) = decl.find('(') else {
            continue;
        };
        let name = format!("__tile_{}", &decl[..paren]);
        // The load/store primitives are the SCAFFOLDING every module here already uses to
        // get a tile in and a result out. A module whose "operation" is another load has no
        // compute in it and is not a kernel, so it says nothing about a lowering.
        //
        // Excluding them did NOT dismiss what they exposed, and that distinction mattered.
        // Fed such a module (three buffers -- two loaded, one stored) the MSL emitter
        // classified it as Copy and emitted `p1[gid] = p0[gid]`, writing to a buffer it had
        // declared `device const`. It was recorded here as unestablished, because a defect
        // found only through a degenerate input is not reported as one until you know a
        // real input reaches it. A real one does: `Copy` is the emitter's DEFAULT kernel
        // type, not a special case, so any module whose compute it does not recognise
        // arrives there with whatever buffer count it has. That arm now refuses.
        if name.starts_with("__tile_load_") || name.starts_with("__tile_store_") {
            continue;
        }
        let Some(close) = decl.find(')') else {
            continue;
        };
        let mut params: Vec<(String, String)> = Vec::new();
        for p in decl[paren + 1..close].split(',') {
            let p = p.split("//").next().unwrap_or("").trim();
            let Some((pn, ty)) = p.split_once(':') else {
                continue;
            };
            params.push((pn.trim().to_string(), ty.trim().to_string()));
        }
        if params.is_empty() {
            continue;
        }
        let site = sites.get(&name);
        let (kinds, from_wrapper) = classify_params(&params, site);
        let module = build_call_module(&name, &params, &kinds, from_wrapper);
        cases.push((name, module));
    }
    cases
}

/// What one declared parameter actually is.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Tile,
    Dim,
    Float,
    Zero,
    Ptr,
}

/// The safe wrappers' own call sites, which say what each argument IS.
///
/// `tile_std` declares the intrinsics AND wraps most of them in typed Rust, and the wrapper
/// bodies are an independent statement of the thing the declarations do not give:
///
/// ```text
///     __tile_matvec_f16_bias(0, a.buf_id, w.buf_id, bias.buf_id, N as u32, K as u32)
/// ```
///
/// `.buf_id` is a tile handle, `X as u32` is a dimension, and a literal `0` is the
/// destination placeholder. This is written by whoever wrote the safe API, not by an
/// emitter author, so using it to build a test input is reading a contract rather than
/// reconstructing one — and unlike the emitter's own source, it is not circular.
fn wrapper_call_sites(src: &str) -> std::collections::HashMap<String, Vec<String>> {
    let mut out: std::collections::HashMap<String, Vec<String>> = Default::default();
    let b = src.as_bytes();
    for (i, _) in src.match_indices("__tile_") {
        // The declarations themselves are `pub fn __tile_x(`; skip those.
        if src[..i].trim_end().ends_with("fn") {
            continue;
        }
        let mut j = i;
        while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
            j += 1;
        }
        if j >= b.len() || b[j] != b'(' {
            continue;
        }
        let name = src[i..j].to_string();
        if out.contains_key(&name) {
            continue;
        }
        let mut depth = 0i32;
        let mut args: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut k = j;
        while k < b.len() {
            let ch = b[k] as char;
            match ch {
                '(' | '[' => {
                    depth += 1;
                    if depth > 1 {
                        cur.push(ch);
                    }
                }
                ')' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    cur.push(ch);
                }
                ',' if depth == 1 => {
                    args.push(cur.trim().to_string());
                    cur.clear();
                }
                ';' => break,
                _ => cur.push(ch),
            }
            k += 1;
        }
        if !cur.trim().is_empty() {
            args.push(cur.trim().to_string());
        }
        if !args.is_empty() {
            out.insert(name, args);
        }
    }
    out
}

/// Classify each declared parameter, preferring the wrapper's own call site.
///
/// Returns the kinds and whether the wrapper supplied them. Where it did, the tile operands
/// are DISTINCT buffers, because the wrapper names distinct tiles; where it did not, the
/// name heuristic is used and every tile shares one loaded buffer, which is how the
/// emitter's own fixtures pass them. Both halves were measured: on the wrapper-classified
/// set, distinct buffers compile 48 and shared 45; applying distinct buffers to the
/// name-classified set instead takes the whole sweep from 229 to 202.
fn classify_params(params: &[(String, String)], site: Option<&Vec<String>>) -> (Vec<Kind>, bool) {
    if let Some(site) = site {
        if site.len() == params.len() {
            let kinds = site
                .iter()
                .zip(params)
                .map(|(arg, (_, ty))| {
                    if ty.starts_with('*') {
                        Kind::Ptr
                    } else if arg.contains(".buf_id") {
                        Kind::Tile
                    } else if ty == "f32" {
                        Kind::Float
                    } else if arg == "0" || arg == "0u32" {
                        Kind::Zero
                    } else {
                        Kind::Dim
                    }
                })
                .collect();
            return (kinds, true);
        }
    }
    const TILE_NAMES: &[&str] = &[
        "src", "dst", "out", "q", "k", "v", "mask", "sinks", "pad", "blk", "block", "state", "ids",
        "weights", "x", "w", "a", "b", "c", "inp", "shared", "gate", "up",
    ];
    let kinds = params
        .iter()
        .map(|(pn, ty)| {
            if ty.starts_with('*') {
                Kind::Ptr
            } else if ty == "f32" {
                Kind::Float
            } else if TILE_NAMES.iter().any(|t| {
                // An exact name, or one with a NUMERIC suffix (`src0`, `src_1`). The rule
                // used to admit any `_`-suffixed name, which made `dst_len` -- a length --
                // into a tile handle. Names like `k_cache` that this drops are supplied by
                // a wrapper call site anyway, which is the better source.
                pn == t
                    || pn.strip_prefix(t).is_some_and(|r| {
                        let r = r.strip_prefix('_').unwrap_or(r);
                        !r.is_empty() && r.chars().all(|ch| ch.is_ascii_digit())
                    })
            }) {
                Kind::Tile
            } else {
                Kind::Dim
            }
        })
        .collect();
    (kinds, false)
}

/// Build a call to `name` from its declared parameters and their classified kinds.
fn build_call_module(
    name: &str,
    params: &[(String, String)],
    kinds: &[Kind],
    distinct_tiles: bool,
) -> String {
    let mut order: Vec<&str> = Vec::new();
    for ((pn, _), k) in params.iter().zip(kinds) {
        if *k == Kind::Tile && !order.contains(&pn.as_str()) {
            order.push(pn);
        }
    }
    // The SOURCE dtype of a cast is in the intrinsic's NAME and nowhere else -- the
    // declaration types every operand `u32`. Handing an f32 tile to `cast_f16_f32` builds a
    // module that is malformed rather than a lowering that is wrong, and `mlir-opt` says so
    // in those terms: "use of value '%arg0' expects different type than prior uses". The
    // `cast_<from>_<to>` convention is the emitter's own -- it selects the kernel type from
    // the same name -- so reading it here is reading a stated convention, not guessing.
    let load = if name.starts_with("__tile_cast_f16_") {
        "__tile_load_f16"
    } else {
        "__tile_load_f32"
    };
    let tile_bufs = if distinct_tiles { order.len() } else { 1 };
    // The STORE DESTINATION goes LAST. Every emitter here marks the final buffer as the
    // writable one, so putting the destination before the pointer operands made the
    // INDICES buffer the output and the real output `const` -- and `scatter`, `gather` and
    // `topk` then "failed" with `read-only variable is not assignable`, which reads exactly
    // like the genuine const-pointer defects found in msl.
    let ptr_count = kinds.iter().filter(|k| **k == Kind::Ptr).count();
    let store_arg = tile_bufs + ptr_count;
    let mut next_ptr = tile_bufs;

    let mut loads = String::new();
    let mut tile_var: std::collections::HashMap<&str, String> = Default::default();
    if distinct_tiles {
        for (i, t) in order.iter().enumerate() {
            tile_var.insert(t, format!("%t_{t}"));
            loads.push_str(&format!(
                "    %t_{t} = llvm.call @{load}(%arg{i}, %r, %c) :                  (!llvm.ptr<1>, i32, i32) -> i32\n"
            ));
        }
    } else {
        loads.push_str(&format!(
            "    %t = llvm.call @{load}(%arg0, %r, %c) :              (!llvm.ptr<1>, i32, i32) -> i32\n"
        ));
        for t in &order {
            tile_var.insert(t, "%t".to_string());
        }
    }

    let mut args: Vec<String> = Vec::new();
    let mut tys: Vec<&str> = Vec::new();
    for ((pn, _), k) in params.iter().zip(kinds) {
        match k {
            Kind::Ptr => {
                args.push(format!("%arg{next_ptr}"));
                tys.push("!llvm.ptr<1>");
                next_ptr += 1;
            }
            Kind::Tile => {
                args.push(tile_var[pn.as_str()].clone());
                tys.push("i32");
            }
            Kind::Float => {
                args.push("%f".to_string());
                tys.push("f32");
            }
            Kind::Zero => {
                args.push("%z".to_string());
                tys.push("i32");
            }
            Kind::Dim => {
                args.push("%c".to_string());
                tys.push("i32");
            }
        }
    }
    let sig = (0..=store_arg)
        .map(|i| format!("%arg{i}: !llvm.ptr<1>"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "module {{\n  llvm.func @k({sig}) attributes {{hacc.entry}} {{\n    ^bb0:\n\
         \x20   %r = llvm.mlir.constant(256 : i32) : i32\n\
         \x20   %c = llvm.mlir.constant(256 : i32) : i32\n\
         \x20   %z = llvm.mlir.constant(0 : i32) : i32\n\
         \x20   %f = llvm.mlir.constant(1.000000e+00 : f32) : f32\n\
         {loads}\
         \x20   %y = llvm.call @{name}({}) : ({}) -> i32\n\
         \x20   llvm.call @__tile_store_f32(%arg{store_arg}, %y, %r, %c) :          (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
         \x20   llvm.return\n  }}\n}}\n",
        args.join(", "),
        tys.join(", ")
    )
}

/// The declared-intrinsic corpus, pointed at the backends that can be checked here with no
/// vendor toolchain at all. Slow; set `TILE_SWEEP=1`.
///
/// The gates above test a hand-listed handful of ops — `exp`, `softmax`, `reduce_sum` —
/// across a few shapes. That is a narrow slice of what these emitters claim: `tile_std`
/// declares 423 intrinsics, and the DS4 families among them are reached by nothing else in
/// this suite. `declared_intrinsics_compile_with_the_real_compiler` closed that gap for
/// `msl`, which is the only backend with a compiler on this machine. This closes as much of
/// it as can be closed for the rest, using the two checks that need nothing installed:
///
///   * for the MLIR-emitting backends, the SSA invariant — every `%value` used must be
///     defined earlier. Not arbitrary: it is precisely the defect found in the CUDA matmul
///     and in gaudi's `log`, both in backends no compiler here can read.
///   * for the Python-emitting backends, the interpreter's own parser.
///
/// A refusal is an answer, not a failure — most of this corpus is Metal-specific and the
/// other backends decline most of it. What would NOT be an answer is emitting something
/// that is not a program, and that is what this looks for.
#[test]
fn declared_intrinsics_pass_the_toolchain_free_gates() {
    if std::env::var("TILE_SWEEP").is_err() {
        eprintln!("emitted_parses: toolchain-free corpus sweep skipped, set TILE_SWEEP=1");
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(decls_src) = std::fs::read_to_string(root.join("../tile_std/src/tile.rs")) else {
        eprintln!("emitted_parses: skipped, the declarations are not in this tree");
        return;
    };
    let cases = modules_from_declarations(&decls_src);
    assert!(
        cases.len() >= 200,
        "only {} declarations parsed -- the extraction changed shape",
        cases.len()
    );

    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-corpus-free-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let python = have_python();
    let mut bad: Vec<String> = Vec::new();
    let mut summary: Vec<String> = Vec::new();

    for form in ["linalg", "rvv", "pto", "aie", "nki", "tpu"] {
        let is_python = PYTHON_FORMS.contains(&form);
        if is_python && !python {
            continue;
        }
        let ext = if is_python { "py" } else { "mlir" };
        let (mut emitted_n, mut checked) = (0usize, 0usize);

        for (name, mlir) in &cases {
            let srcf = dir.join(format!("{form}_{name}.in.mlir"));
            std::fs::write(&srcf, mlir).expect("write");
            // A FRESH output path per kernel. Reusing one meant a refused emission left the
            // previous kernel's file to be checked against this one -- 48 false failures the
            // first time this suite did it.
            let out = dir.join(format!("{form}_{name}.{ext}"));
            let _ = std::fs::remove_file(&out);
            let run = Command::new(exe)
                .arg(&srcf)
                .args(["-t", form, "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !run.status.success() || !out.exists() {
                continue; // a refusal is an answer
            }
            emitted_n += 1;
            if is_python {
                let cc = Command::new("python3")
                    .args(["-m", "py_compile"])
                    .arg(&out)
                    .output()
                    .expect("python3 runs");
                checked += 1;
                if !cc.status.success() {
                    bad.push(format!(
                        "{form} {name}: {}",
                        String::from_utf8_lossy(&cc.stderr)
                            .lines()
                            .last()
                            .unwrap_or("")
                            .trim()
                            .chars()
                            .take(110)
                            .collect::<String>()
                    ));
                }
            } else {
                let text = std::fs::read_to_string(&out).unwrap_or_default();
                checked += 1;
                if let Some((line, v)) = ssa_use_before_def(&text).into_iter().next() {
                    bad.push(format!(
                        "{form} {name} line {line}: {v} used before any definition"
                    ));
                }
            }
        }
        summary.push(format!("{form} {emitted_n}/{}", cases.len()));
        let _ = checked;
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: declared corpus vs toolchain-free gates — {}",
        summary.join(", ")
    );
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    // Guard the vacuous pass. If every backend began refusing everything, or the corpus
    // stopped being built, this would pass by checking nothing.
    let total: usize = summary
        .iter()
        .filter_map(|s| s.split_whitespace().nth(1))
        .filter_map(|f| f.split('/').next())
        .filter_map(|n| n.parse::<usize>().ok())
        .sum();
    assert!(
        total >= 50,
        "only {total} kernels were emitted across all six backends -- they are refusing \
         nearly everything and this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emissions from the declared-intrinsic corpus are not programs.",
        bad.len()
    );
}

/// The declared-intrinsic corpus against every backend that emits C or C++ here. Slow;
/// set `TILE_SWEEP=1`.
///
/// The three C-family gates above each test a hand-listed handful of ops. This points the
/// same corpus that found five defects in `msl` at `gpu`, `musa`, `bang`, `gaudi`,
/// `hexagon` and `ttmetal`, reusing their stubs exactly as those gates set them up —
/// `bang` includes its own header so the stub goes on the include path, `gaudi` needs
/// force-include, the header-including pair need their vendor paths created, and `gaudi`
/// emits `void main(` which is not legal C++ at file scope.
///
/// A word on what a failure here means. These stubs are MINE, and they are where eight
/// false alarms came from in one afternoon: the CUDA check first reported 194 broken
/// kernels of which 175 were gaps in my own header, and the hexagon/ttmetal count went
/// 0 → 0 → 112 → 121 → 140 as I fixed the stub rather than an emitter. So a failure is
/// triaged before it is believed, and the assertion message says so.
#[test]
fn declared_intrinsics_pass_the_c_family_gates() {
    if std::env::var("TILE_SWEEP").is_err() {
        eprintln!("emitted_parses: C-family corpus sweep skipped, set TILE_SWEEP=1");
        return;
    }
    let Some(cxx) = which("clang++").or_else(|| which("g++")) else {
        eprintln!("emitted_parses: skipped, no C++ compiler");
        return;
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(decls_src) = std::fs::read_to_string(root.join("../tile_std/src/tile.rs")) else {
        eprintln!("emitted_parses: skipped, the declarations are not in this tree");
        return;
    };
    let cases = modules_from_declarations(&decls_src);
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-corpus-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    // Stubs, laid out the way each existing gate lays them out. Getting this wrong is the
    // usual first result: the CUDA emitters `#include <cuda_runtime.h>` themselves, so the
    // stub has to answer to THAT NAME on the include path -- force-including it under some
    // other name leaves the include unresolved, and every gpu/musa kernel then "fails" at
    // line 5 with a fatal error that is the harness's, not the emitter's.
    for h in ["cuda_runtime.h", "musa_runtime.h"] {
        std::fs::write(dir.join(h), CUDA_STUB).expect("write");
    }
    std::fs::write(
        dir.join("cuda_fp16.h"),
        "#pragma once\n#include <cuda_runtime.h>\n",
    )
    .expect("write");
    std::fs::write(
        dir.join("musa_fp16.h"),
        "#pragma once\n#include <musa_runtime.h>\n",
    )
    .expect("write");
    for (form, stub) in VENDOR_C_FORMS {
        let name = if *form == "bang" {
            "bang.h".to_string()
        } else {
            format!("{form}_stub.h")
        };
        std::fs::write(dir.join(name), stub).expect("write");
    }
    for (_form, stub, headers) in HEADER_FORMS {
        for h in *headers {
            let p = dir.join(h);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).expect("vendor include dir");
            }
            std::fs::write(&p, stub).expect("write");
        }
    }

    let mut summary: Vec<String> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    // gaudi is held to a RATCHET rather than to zero, and the reason is specific rather
    // than a shrug. Its remaining failures are casts of a `float256` to `float*`
    // (`((float *)&v_1)[0]`), and whether tpc-clang accepts those is not knowable from
    // here -- this file's TPC stub types them as a plain vector typedef. The handoff
    // already records that gaudi cannot be emulated here for the same reason, and that
    // guessing at its semantics once produced a false alarm. So: the count may not GROW.
    let mut gaudi_fail = 0usize;
    let mut gaudi_why: Vec<String> = Vec::new();

    for form in ["gpu", "musa", "bang", "gaudi", "hexagon", "ttmetal"] {
        let (mut emitted_n, mut checked) = (0usize, 0usize);
        for (name, mlir) in &cases {
            let srcf = dir.join(format!("{form}_{name}.in.mlir"));
            std::fs::write(&srcf, mlir).expect("write");
            let out = dir.join(format!("{form}_{name}.c"));
            let _ = std::fs::remove_file(&out);
            let run = Command::new(exe)
                .arg(&srcf)
                .args(["-t", form, "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !run.status.success() || !out.exists() {
                continue; // a refusal is an answer
            }
            emitted_n += 1;
            // `gaudi` emits `void main(`, which is not legal at C++ file scope. The
            // existing gate renames it; so does this one, for the same reason.
            let text = std::fs::read_to_string(&out).unwrap_or_default();

            // Create any vendor header this emission asks for that is not already there.
            // HEADER_FORMS lists the paths by hand, and the list was short two of them --
            // `HTP/hvx_eltwise.h` and `compute_kernel_api/eltwise_binary.h` -- which
            // reported four kernels broken for a gap in the harness. Reading the includes
            // out of the emitted file cannot miss one, and a header the emitter never asks
            // for is never created.
            let stub_for = |f: &str| -> Option<&str> {
                HEADER_FORMS
                    .iter()
                    .find(|(n, _, _)| *n == f)
                    .map(|(_, st, _)| *st)
            };
            if let Some(stub) = stub_for(form) {
                for line in text.lines() {
                    let Some(rest) = line.trim().strip_prefix("#include \"") else {
                        continue;
                    };
                    let Some(h) = rest.split('"').next() else {
                        continue;
                    };
                    let hp = dir.join(h);
                    if !hp.exists() {
                        if let Some(parent) = hp.parent() {
                            std::fs::create_dir_all(parent).expect("vendor include dir");
                        }
                        std::fs::write(&hp, stub).expect("write");
                    }
                }
            }
            let chk = dir.join(format!("{form}_{name}.chk.cpp"));
            std::fs::write(&chk, text.replace("\nvoid main(", "\nvoid tpc_main(")).expect("write");

            let mut cmd = Command::new(&cxx);
            cmd.args(["-std=c++17", "-fsyntax-only"]);
            if form == "gaudi" {
                cmd.arg("-include").arg(dir.join("gaudi_stub.h"));
            }
            let parse = cmd
                .args(["-I"])
                .arg(&dir)
                .args(["-x", "c++"])
                .arg(&chk)
                .output()
                .expect("the C++ compiler runs");
            checked += 1;
            if !parse.status.success() {
                let why = String::from_utf8_lossy(&parse.stderr)
                    .lines()
                    .find(|l| l.contains("error:"))
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(110)
                    .collect::<String>();
                if form == "gaudi" {
                    gaudi_fail += 1;
                    gaudi_why.push(format!("{name}: {why}"));
                } else {
                    bad.push(format!("{form} {name}: {why}"));
                }
            }
        }
        summary.push(format!("{form} {checked}/{emitted_n}"));
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: declared corpus vs C-family gates — {}",
        summary.join(", ")
    );
    for b in bad.iter().take(15) {
        eprintln!("  {b}");
    }
    let total: usize = summary
        .iter()
        .filter_map(|s| s.split_whitespace().nth(1))
        .filter_map(|f| f.split('/').next())
        .filter_map(|n| n.parse::<usize>().ok())
        .sum();
    assert!(
        total >= 100,
        "only {total} kernels were emitted across six backends -- they are refusing nearly \
         everything and this test would pass by checking nothing"
    );
    // The gaudi ratchet, measured BY THIS TEST ON THIS CORPUS. It was first set to 7 from
    // a hand-run sweep over a different, smaller corpus, and this gate promptly reported 9
    // -- the same "measure it with the instrument that will be asserting on it" mistake
    // this suite keeps finding in emitters, made here in a test.
    //
    // Body-derived declarations took it down from a larger number; what is left is casts of
    // a `float256` to `float*` (`((float *)&v_1)[0]`), which this file's TPC stub cannot
    // adjudicate. A change that takes the count HIGHER is a regression even though these
    // nine are unresolved.
    const GAUDI_KNOWN_UNRESOLVED: usize = 9;
    for w in gaudi_why.iter().take(8) {
        eprintln!("  gaudi (stub fidelity, not asserted): {w}");
    }
    assert!(
        gaudi_fail <= GAUDI_KNOWN_UNRESOLVED,
        "gaudi now fails {gaudi_fail} of the corpus, up from {GAUDI_KNOWN_UNRESOLVED}. \
         Those are held open because this file's TPC stub cannot say whether tpc-clang \
         accepts a `float256` cast to `float*`; a NEW failure is a different matter."
    );
    assert!(
        bad.is_empty(),
        "{} emissions from the declared-intrinsic corpus are not valid C++. TRIAGE BEFORE \
         BELIEVING: the stubs in this file are hand-written, and the last time this check \
         was extended, 175 of 194 reported failures were gaps in the stub rather than \
         defects in an emitter. Check that the missing name is one the vendor really \
         provides before touching a backend.",
        bad.len()
    );
}

/// SPIR-V, through the real Vulkan shader compiler, and then through the SPIR-V validator.
///
/// The same story as `xcrun metal`: `glslangValidator` was on this machine the whole time
/// and nothing in this suite used it, so the SPIR-V backend had no compiler gate at all —
/// only the use-before-def check that `mlir_backends_use_no_value_before_defining_it`
/// applies to the MLIR-emitting forms. It found four defects on its first run:
///
///   * `rope`, `causal_mask` and `embedding` emitted `gid %% half_cols`. In a Rust format
///     string `%` needs no escaping (`{{` and `}}` do), so writing `%%` on the printf
///     convention put a literal doubled percent into the GLSL. None of the three ever
///     compiled.
///   * `topk` declared both of its output buffers `writeonly`, then read them back — the
///     rank scan reads `p0`, and the insertion shift reads both one slot down. The
///     qualifier described each buffer's role rather than its use.
///
/// TWO tools, deliberately. `glslangValidator` checks the GLSL and produces a module;
/// `spirv-val` then checks that module against the SPIR-V spec, which is a different
/// question and the one a driver will actually ask.
///
/// The invocation matters and was got wrong first: without `--target-env vulkan1.1` every
/// kernel using a subgroup op is rejected for "requires SPIR-V 1.3", which reported seven
/// false failures. The emitted files' own `// Compile:` line still names the plain
/// invocation, so it does not work for those kernels.
///
/// Needs `glslangValidator`; skips with a note otherwise.
#[test]
fn spirv_kernels_compile_with_the_real_shader_compiler() {
    let Some(glslang) = which("glslangValidator") else {
        eprintln!("emitted_parses: skipped, no glslangValidator");
        return;
    };
    let spirv_val = which("spirv-val");
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-spirv-cc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    // The hand-listed core ops across shapes, plus every declared intrinsic.
    let mut cases: Vec<(String, String)> = Vec::new();
    for &(rows, cols) in SHAPES {
        for (op, reduces) in [
            ("exp", false),
            ("softmax", false),
            ("reduce_sum", true),
            ("rms_norm", false),
        ] {
            cases.push((
                format!("{op}_{rows}x{cols}"),
                kernel(op, rows, cols, reduces),
            ));
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Ok(decls) = std::fs::read_to_string(root.join("../tile_std/src/tile.rs")) {
        cases.extend(modules_from_declarations(&decls));
    }

    let (mut checked, mut refused, mut validated) = (0usize, 0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    for (name, mlir) in &cases {
        let srcf = dir.join(format!("{name}.mlir"));
        std::fs::write(&srcf, mlir).expect("write");
        let comp = dir.join(format!("{name}.comp"));
        let _ = std::fs::remove_file(&comp);
        let emitted = Command::new(exe)
            .arg(&srcf)
            .args(["-t", "spirv", "-o"])
            .arg(&comp)
            .arg("--force")
            .current_dir(&dir)
            .output()
            .expect("the tile binary runs");
        if !emitted.status.success() || !comp.exists() {
            refused += 1; // a refusal is an answer
            continue;
        }
        checked += 1;
        let spv = dir.join(format!("{name}.spv"));
        let cc = Command::new(&glslang)
            .args(["--target-env", "vulkan1.1", "-V"])
            .arg(&comp)
            .arg("-o")
            .arg(&spv)
            .output()
            .expect("glslangValidator runs");
        if !cc.status.success() {
            bad.push(format!(
                "{name}: {}",
                String::from_utf8_lossy(&cc.stdout)
                    .lines()
                    .chain(String::from_utf8_lossy(&cc.stderr).lines())
                    .find(|l| l.contains("ERROR"))
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(110)
                    .collect::<String>()
            ));
            continue;
        }
        // The module compiled. Whether it is a VALID SPIR-V module is a separate question,
        // and it is the one a driver asks.
        if let Some(val) = &spirv_val {
            let v = Command::new(val)
                .arg(&spv)
                .output()
                .expect("spirv-val runs");
            validated += 1;
            if !v.status.success() {
                bad.push(format!(
                    "{name}: compiled but spirv-val rejects it — {}",
                    String::from_utf8_lossy(&v.stderr)
                        .lines()
                        .next()
                        .unwrap_or("")
                        .trim()
                        .chars()
                        .take(110)
                        .collect::<String>()
                ));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: {checked} SPIR-V shaders compiled ({validated} also spirv-val'd), \
         {refused} refused"
    );
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    assert!(
        checked >= 40,
        "only {checked} shaders were emitted -- spirv is refusing nearly everything and \
         this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emitted SPIR-V shaders do not compile or do not validate.",
        bad.len()
    );
}

/// The MLIR-emitting backends, through the real MLIR parser and verifier. Slow;
/// set `TILE_SWEEP=1`.
///
/// `mlir-opt` is on this machine at `/opt/homebrew/opt/llvm/bin/mlir-opt` — present, just
/// absent from `PATH`, which is why a header in `declared_semantics.rs` said for a long
/// time that it was not here at all. The emitted files carry
/// `// Verify: mlir-opt --verify-diagnostics <this_file>` in their own preamble, so this
/// gate is running the check the emitter itself documents.
///
/// What it adds over `mlir_backends_use_no_value_before_defining_it`: that one enforces one
/// invariant by hand. This one type-checks. A `linalg.matmul` whose operands are
/// `tensor<256x0xf32>` because a dimension resolved to zero satisfies use-before-def
/// perfectly and is still not a program.
///
/// Needs `mlir-opt`; skips with a note otherwise.
#[test]
fn mlir_backends_parse_with_the_real_mlir_parser() {
    let mlir_opt = which("mlir-opt").or_else(|| {
        let p = "/opt/homebrew/opt/llvm/bin/mlir-opt";
        std::path::Path::new(p).exists().then(|| p.to_string())
    });
    let Some(mlir_opt) = mlir_opt else {
        eprintln!("emitted_parses: skipped, no mlir-opt");
        return;
    };
    if std::env::var("TILE_SWEEP").is_err() {
        eprintln!("emitted_parses: mlir-opt corpus sweep skipped, set TILE_SWEEP=1");
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(decls_src) = std::fs::read_to_string(root.join("../tile_std/src/tile.rs")) else {
        eprintln!("emitted_parses: skipped, the declarations are not in this tree");
        return;
    };
    let cases = modules_from_declarations(&decls_src);
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-mlir-parse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut summary: Vec<String> = Vec::new();
    let mut bad: Vec<String> = Vec::new();

    for form in ["linalg", "rvv", "pto"] {
        let mut parsed = 0usize;
        for (name, mlir) in &cases {
            let srcf = dir.join(format!("{form}_{name}.in.mlir"));
            std::fs::write(&srcf, mlir).expect("write");
            let out = dir.join(format!("{form}_{name}.out.mlir"));
            let _ = std::fs::remove_file(&out);
            let run = Command::new(exe)
                .arg(&srcf)
                .args(["-t", form, "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if !run.status.success() || !out.exists() {
                continue; // a refusal is an answer
            }
            let check = Command::new(&mlir_opt)
                .arg(&out)
                .args(["-o", "/dev/null"])
                .output()
                .expect("mlir-opt runs");
            if check.status.success() {
                parsed += 1;
            } else {
                bad.push(format!(
                    "{form} {name}: {}",
                    String::from_utf8_lossy(&check.stderr)
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("")
                        .trim()
                        .chars()
                        .take(110)
                        .collect::<String>()
                ));
            }
        }
        summary.push(format!("{form} {parsed}"));
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: declared corpus through mlir-opt — {}",
        summary.join(", ")
    );
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    let total: usize = summary
        .iter()
        .filter_map(|s| s.split_whitespace().nth(1))
        .filter_map(|n| n.parse::<usize>().ok())
        .sum();
    assert!(
        total >= 40,
        "only {total} modules parsed -- the backends are refusing nearly everything and \
         this test would pass by checking nothing"
    );
    assert!(
        bad.is_empty(),
        "{} emitted MLIR modules do not survive the real parser and verifier.",
        bad.len()
    );
}

/// The intrinsics the MSL emitter lowers that NOTHING DECLARES, compiled by the real Metal
/// compiler. Slow; set `TILE_SWEEP=1`.
///
/// `declared_intrinsics_compile_with_the_real_compiler` above is built from `tile_std`'s
/// declarations, so it reaches 241 intrinsics and cannot reach these at all — they are
/// uncovered because they are undeclared (backlog #022). Between them they account for
/// 3746 of the 6223 uncovered lines in `mlir_to_msl.rs`, nearly all inside
/// `generate_func_msl`'s `match ctx.kernel_type`.
///
/// WHAT THIS CHECKS, AND WHAT IT CANNOT. The call shape here is not read from a contract,
/// because there is no contract: it is a generic call whose arity is widened until the
/// emitter accepts it. So this asks only **"is the emitted text a program?"** — the same
/// question `xcrun metal` answers for the declared corpus. It says nothing about whether
/// the lowering is CORRECT, and it cannot, because the input was shaped to fit the emitter
/// rather than to fit a stated signature. Declare the 44 and this becomes a real test.
///
/// It earned its place on the first run: `fill` wrote through a `device const` pointer, and
/// `MatmulF16Simdgroup` passed `float*` buffers to `simdgroup_load` against
/// `simdgroup_half8x8` accumulators — a kernel that had never compiled.
#[test]
fn undeclared_intrinsics_still_emit_a_program() {
    if std::env::var("TILE_SWEEP").is_err() {
        eprintln!("emitted_parses: undeclared-intrinsic sweep skipped, set TILE_SWEEP=1");
        return;
    }
    let Some(xcrun) = which("xcrun") else {
        eprintln!("emitted_parses: skipped, no xcrun");
        return;
    };
    require_metal_toolchain!(xcrun);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(emitter) =
        std::fs::read_to_string(root.join("../rustc_codegen_tile/src/mlir_to_msl.rs"))
    else {
        eprintln!("emitted_parses: skipped, the emitter is not in this tree");
        return;
    };
    // Names the emitter matches OUTSIDE its test module. `__tile_frobnicate_f32` lives in
    // that module and is a placeholder, not an arm; counting it put this list at 54.
    let production = emitter
        .find("#[cfg(test)]")
        .map(|i| &emitter[..i])
        .unwrap_or(&emitter[..]);
    let mut handled: Vec<String> = Vec::new();
    for (i, _) in production.match_indices("\"__tile_") {
        let rest = &production[i + 1..];
        if let Some(e) = rest.find('"') {
            let n = rest[..e].to_string();
            // A trailing underscore means the literal is a PREFIX matched with
            // `starts_with`, not a name matched with `==` -- `name if
            // name.starts_with("__tile_matvec_quant_")`. Nothing declares a prefix, so
            // counting it made the list one longer than the number of real intrinsics. It
            // is also how `hexagon`, `ttmetal` and `csl` each appeared to lower seven
            // undeclared intrinsics when in truth they match seven prefixes.
            if !n.ends_with('_') && !handled.contains(&n) {
                handled.push(n);
            }
        }
    }
    // Declared ANYWHERE in crates/, matched as a whole name. An unanchored match calls
    // `__tile_get_rows_f32` declared because `__tile_get_rows_f32_strided` is; that put
    // this list at 53.
    let mut declared: Vec<String> = Vec::new();
    for entry in walk_rs(&root.join("..")) {
        let Ok(text) = std::fs::read_to_string(&entry) else {
            continue;
        };
        for (i, _) in text.match_indices("pub fn __tile_") {
            let rest = &text[i + "pub fn ".len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            declared.push(rest[..end].to_string());
        }
    }
    let undeclared: Vec<&String> = handled.iter().filter(|n| !declared.contains(n)).collect();
    assert!(
        undeclared.len() >= 30,
        "only {} undeclared intrinsics found -- the extraction changed shape",
        undeclared.len()
    );

    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-undeclared-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let (mut ok, mut never) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();

    for name in &undeclared {
        let mut emitted = None;
        // Widen the arity until the emitter accepts one. Most of these arms read no
        // operands at all -- they only select a kernel type -- so any shape reaches them.
        for nargs in [8usize, 6, 5, 4, 10, 12] {
            let mut args: Vec<&str> = vec!["%z"];
            for i in 1..nargs {
                args.push(if i < 4 { "%t" } else { "%c" });
            }
            let tys = vec!["i32"; nargs].join(", ");
            let mlir = format!(
                "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                 attributes {{hacc.entry}} {{\n    ^bb0:\n\
                 \x20   %r = llvm.mlir.constant(256 : i32) : i32\n\
                 \x20   %c = llvm.mlir.constant(256 : i32) : i32\n\
                 \x20   %z = llvm.mlir.constant(0 : i32) : i32\n\
                 \x20   %t = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
                 \x20   %y = llvm.call @{name}({}) : ({tys}) -> i32\n\
                 \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                 \x20   llvm.return\n  }}\n}}\n",
                args.join(", ")
            );
            let srcf = dir.join(format!("{name}.mlir"));
            std::fs::write(&srcf, &mlir).expect("write");
            let out = dir.join(format!("{name}.metal"));
            let _ = std::fs::remove_file(&out);
            let run = Command::new(exe)
                .arg(&srcf)
                .args(["-t", "msl", "-o"])
                .arg(&out)
                .arg("--force")
                .current_dir(&dir)
                .output()
                .expect("the tile binary runs");
            if run.status.success() && out.exists() {
                emitted = Some(out);
                break;
            }
        }
        let Some(out) = emitted else {
            never += 1;
            continue;
        };
        let cc = Command::new(&xcrun)
            .args(["metal", "-c"])
            .arg(&out)
            .arg("-o")
            .arg(dir.join(format!("{name}.air")))
            .output()
            .expect("xcrun runs");
        if cc.status.success() {
            ok += 1;
        } else {
            bad.push(format!(
                "{name}: {}",
                String::from_utf8_lossy(&cc.stderr)
                    .lines()
                    .find(|l| l.contains("error:"))
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(110)
                    .collect::<String>()
            ));
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: {ok} of {} undeclared intrinsics compile, {never} never emitted \
         (from {} names the emitter matches and {} declared across crates/)",
        undeclared.len(),
        handled.len(),
        declared.len()
    );
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    assert!(
        bad.is_empty(),
        "{} intrinsics the emitter lowers, and nothing declares, emit Metal that does not \
         compile.",
        bad.len()
    );
}

/// The `__tile_*` names an emitter matches on, minus everything declared anywhere.
///
/// Two rules that are not obvious and both cost a wrong number:
///   * skip the emitter's own `#[cfg(test)]` module -- its fixtures name intrinsics that
///     are not arms (`__tile_frobnicate_f32` is a placeholder);
///   * skip literals ending in `_` -- those are PREFIXES matched with `starts_with`, not
///     names matched with `==`, and nothing declares a prefix. `hexagon`, `ttmetal` and
///     `csl` match ONLY prefixes, so their true undeclared count is zero rather than seven.
fn undeclared_names_of(emitter_src: &str, declared: &[String]) -> Vec<String> {
    let production = emitter_src
        .find("#[cfg(test)]")
        .map(|i| &emitter_src[..i])
        .unwrap_or(emitter_src);
    let mut out: Vec<String> = Vec::new();
    for (i, _) in production.match_indices("\"__tile_") {
        let rest = &production[i + 1..];
        if let Some(e) = rest.find('"') {
            let n = rest[..e].to_string();
            // An intrinsic name is identifier characters and nothing else. Taking
            // everything up to the closing quote also catches literals like
            // `"__tile_rms_norm_*: an eps operand is given but ..."` -- an ERROR MESSAGE
            // that begins with a name. Those produced "names" containing spaces and a
            // slash, and the slash turned `dir.join(name)` into a path whose parent does
            // not exist: the sweep died on `write: NotFound` while the directory it named
            // was plainly there.
            let looks_like_a_name = n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if looks_like_a_name && !n.ends_with('_') && !declared.contains(&n) && !out.contains(&n)
            {
                out.push(n);
            }
        }
    }
    out
}

/// Every `pub fn __tile_*` declared anywhere under `crates/`.
fn all_declared_intrinsics(root: &std::path::Path) -> Vec<String> {
    let mut declared = Vec::new();
    for entry in walk_rs(root) {
        let Ok(text) = std::fs::read_to_string(&entry) else {
            continue;
        };
        for (i, _) in text.match_indices("pub fn __tile_") {
            let rest = &text[i + "pub fn ".len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            declared.push(rest[..end].to_string());
        }
    }
    declared
}

/// Every `.rs` under a directory, for the declaration scan.
fn walk_rs(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target" || n == ".git") {
                continue;
            }
            out.extend(walk_rs(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out
}

/// The undeclared intrinsics of the OTHER backends, each through its own real checker.
/// Slow; set `TILE_SWEEP=1`.
///
/// `undeclared_intrinsics_still_emit_a_program` covers MSL, which lowers 46 of the 64
/// intrinsics that no declaration in this tree mentions. **18 are lowered only elsewhere**,
/// so that gate cannot see them at all. This one takes each remaining backend to the
/// checker that suits it: `glslangValidator` for SPIR-V, the Python interpreter for `aie`,
/// `nki` and `tpu`.
///
/// Same terms as the MSL gate, stated for the same reason: the call shape is widened until
/// the emitter accepts it, so this asks **"is the emitted text a program?"** and not "is the
/// lowering right?" — there is no contract to be right against, which is the whole point of
/// backlog #022.
///
/// It found `aie` writing `elem_out0[i] = elem_in0[i] * %t`, the seventh backend able to put
/// an MLIR token in its output. That is invisible to the DECLARED corpus, where aie is
/// clean.
///
/// TWO EXCLUSIONS, both learned the hard way:
///   * `linalg` and `pto` emit MLIR. Their SSA values are SPELLED `%t0`, so a leak check
///     over their output fires on every line, and the `reject_leaked_ssa_names` guard was
///     briefly added to them before twenty-odd tests said otherwise.
///   * the SPIR-V output must be written to a `.comp` file. `glslangValidator` infers the
///     shader stage from the EXTENSION, and a sweep that wrote `.out` reported six false
///     failures that looked exactly like real ones.
#[test]
fn undeclared_intrinsics_of_the_other_backends_still_emit_a_program() {
    if std::env::var("TILE_SWEEP").is_err() {
        eprintln!("emitted_parses: other-backend undeclared sweep skipped, set TILE_SWEEP=1");
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let declared = all_declared_intrinsics(&root.join(".."));
    let exe = env!("CARGO_BIN_EXE_tile");
    let dir = std::env::temp_dir().join(format!("tile-und-others-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let glslang = which("glslangValidator");
    let python = have_python();
    let mut summary: Vec<String> = Vec::new();
    let mut bad: Vec<String> = Vec::new();

    for (form, ext) in [
        ("spirv", "comp"),
        ("aie", "py"),
        ("nki", "py"),
        ("tpu", "py"),
    ] {
        if ext == "comp" && glslang.is_none() {
            continue;
        }
        if ext == "py" && !python {
            continue;
        }
        let src_path = root.join(format!("../rustc_codegen_tile/src/mlir_to_{form}.rs"));
        let Ok(emitter) = std::fs::read_to_string(&src_path) else {
            continue;
        };
        let names = undeclared_names_of(&emitter, &declared);
        let (mut ok, mut refused) = (0usize, 0usize);

        for name in &names {
            let mut emitted = None;
            for nargs in [8usize, 6, 5, 4, 10, 12] {
                let mut args: Vec<&str> = vec!["%z"];
                for i in 1..nargs {
                    args.push(if i < 4 { "%t" } else { "%c" });
                }
                let mlir = format!(
                    "module {{\n  llvm.func @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) \
                     attributes {{hacc.entry}} {{\n    ^bb0:\n\
                     \x20   %r = llvm.mlir.constant(256 : i32) : i32\n\
                     \x20   %c = llvm.mlir.constant(256 : i32) : i32\n\
                     \x20   %z = llvm.mlir.constant(0 : i32) : i32\n\
                     \x20   %t = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32\n\
                     \x20   %y = llvm.call @{name}({}) : ({}) -> i32\n\
                     \x20   llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n\
                     \x20   llvm.return\n  }}\n}}\n",
                    args.join(", "),
                    vec!["i32"; nargs].join(", ")
                );
                let srcf = dir.join(format!("{form}_{name}.mlir"));
                std::fs::write(&srcf, &mlir).expect("write");
                // The EXTENSION is load-bearing for glslang; see the note above.
                let out = dir.join(format!("{form}_{name}.{ext}"));
                let _ = std::fs::remove_file(&out);
                let run = Command::new(exe)
                    .arg(&srcf)
                    .args(["-t", form, "-o"])
                    .arg(&out)
                    .arg("--force")
                    .current_dir(&dir)
                    .output()
                    .expect("the tile binary runs");
                if run.status.success() && out.exists() {
                    emitted = Some(out);
                    break;
                }
            }
            let Some(out) = emitted else {
                refused += 1; // a refusal is an answer
                continue;
            };
            let check = if ext == "comp" {
                Command::new(glslang.as_ref().unwrap())
                    .args(["--target-env", "vulkan1.1", "-V"])
                    .arg(&out)
                    .arg("-o")
                    .arg(dir.join(format!("{form}_{name}.spv")))
                    .output()
                    .expect("glslangValidator runs")
            } else {
                Command::new("python3")
                    .args(["-m", "py_compile"])
                    .arg(&out)
                    .output()
                    .expect("python3 runs")
            };
            if check.status.success() {
                ok += 1;
            } else {
                let msg = String::from_utf8_lossy(&check.stderr)
                    .lines()
                    .chain(String::from_utf8_lossy(&check.stdout).lines())
                    .find(|l| l.contains("error") || l.contains("ERROR"))
                    .unwrap_or("")
                    .trim()
                    .chars()
                    .take(100)
                    .collect::<String>();
                bad.push(format!("{form} {name}: {msg}"));
            }
        }
        summary.push(format!("{form} {ok}/{}", names.len()));
        let _ = refused;
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "emitted_parses: undeclared intrinsics of the other backends — {}",
        summary.join(", ")
    );
    for b in bad.iter().take(10) {
        eprintln!("  {b}");
    }
    assert!(
        bad.is_empty(),
        "{} intrinsics lowered by a non-MSL backend, and declared by nobody, emit text that \
         is not a program.",
        bad.len()
    );
}
