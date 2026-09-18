//! Joining the chain: a routed-MoE step whose experts are evaluated partly on
//! the NPU, verified against a pure-CPU reference.
//!
//! The NPU primitive is the already-verified `matvec.xclbin` (1x64 @ 64x64).
//! Note the layout difference: the emitted kernel computes `x @ W`
//! (`out[j] = Σ_i x[i]·W[i][j]`) whereas ds4's `matvec_f32` is `W · x`
//! (`out[j] = Σ_i w[j·d_in+i]·x[i]`), so the weight matrix is transposed on the
//! way in.
#![cfg(feature = "aie")]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};

#[link(name = "aie_shim")]
extern "C" {
    fn aie_open(index: c_uint, xclbin: *const c_char, kernel: *const c_char) -> *mut c_void;
    fn aie_close(h: *mut c_void);
    fn aie_bo_alloc(h: *mut c_void, bytes: usize, argno: c_int, flags: c_int) -> *mut c_void;
    fn aie_bo_free(b: *mut c_void);
    fn aie_bo_write(b: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn aie_bo_read(b: *mut c_void, dst: *mut c_void, bytes: usize) -> c_int;
    fn aie_run(h: *mut c_void, opcode: c_uint, insts: *mut c_void, words: c_uint,
               bos: *const *mut c_void, n: c_int) -> c_int;
}

const D: usize = 64;          // both d_in and d_out for this kernel
const TILE: usize = 4096;     // the kernel's buffer extent
const OPCODE: c_uint = 3;

/// Minimal NPU-backed matvec, holding the xclbin open across calls.
struct AieMatvec {
    h: *mut c_void,
    insts: *mut c_void,
    words: c_uint,
}

impl AieMatvec {
    fn new() -> Option<Self> {
        let xclbin = std::env::var("AIE_MATVEC")
            .unwrap_or_else(|_| "/data/aie-kernels/matvec.xclbin".into());
        let insts_bytes = std::fs::read(xclbin.replace(".xclbin", ".insts.bin")).ok()?;
        let cx = CString::new(xclbin).ok()?;
        let ck = CString::new("").ok()?;
        let h = unsafe { aie_open(0, cx.as_ptr(), ck.as_ptr()) };
        if h.is_null() { return None; }
        let insts = unsafe { aie_bo_alloc(h, insts_bytes.len(), 1, 1) };
        unsafe { aie_bo_write(insts, insts_bytes.as_ptr() as *const c_void, insts_bytes.len()) };
        Some(Self { h, insts, words: (insts_bytes.len() / 4) as c_uint })
    }

    /// `w` is row-major [d_out][d_in], as ds4 stores it.
    fn matvec(&self, w: &[f32], x: &[f32]) -> Vec<f32> {
        assert_eq!(w.len(), D * D);
        assert_eq!(x.len(), D);
        let mut a = vec![0f32; TILE];
        a[..D].copy_from_slice(x);
        // transpose into the kernel's x@W convention
        let mut b = vec![0f32; TILE];
        for j in 0..D { for i in 0..D { b[i * D + j] = w[j * D + i]; } }

        let nb = TILE * 4;
        let ba = unsafe { aie_bo_alloc(self.h, nb, 3, 0) };
        let bb = unsafe { aie_bo_alloc(self.h, nb, 4, 0) };
        let bc = unsafe { aie_bo_alloc(self.h, nb, 5, 0) };
        unsafe {
            aie_bo_write(ba, a.as_ptr() as *const c_void, nb);
            aie_bo_write(bb, b.as_ptr() as *const c_void, nb);
        }
        let bos = [ba, bb, bc];
        let rc = unsafe { aie_run(self.h, OPCODE, self.insts, self.words, bos.as_ptr(), 3) };
        assert_eq!(rc, 0, "aie_run failed: {rc}");
        let mut out = vec![0f32; TILE];
        unsafe { aie_bo_read(bc, out.as_mut_ptr() as *mut c_void, nb) };
        unsafe { aie_bo_free(ba); aie_bo_free(bb); aie_bo_free(bc); }
        out.truncate(D);
        out
    }
}

impl Drop for AieMatvec {
    fn drop(&mut self) {
        unsafe { aie_bo_free(self.insts); aie_close(self.h); }
    }
}

fn cpu_matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..D).map(|j| (0..D).map(|i| w[j * D + i] * x[i]).sum()).collect()
}

fn silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }

/// One expert, parameterised by which matvec implementation to use.
fn expert<F: Fn(&[f32], &[f32]) -> Vec<f32>>(
    mv: &F, x: &[f32], wg: &[f32], wu: &[f32], wd: &[f32], w_e: f32,
) -> Vec<f32> {
    const CLAMP: f32 = 10.0;
    let gate = mv(wg, x);
    let up = mv(wu, x);
    let mid: Vec<f32> = gate.iter().zip(up.iter())
        .map(|(&g, &u)| silu(g.min(CLAMP)) * u.clamp(-CLAMP, CLAMP) * w_e)
        .collect();
    mv(wd, &mid)
}

#[test]
fn moe_step_split_across_cpu_and_npu() {
    let npu = match AieMatvec::new() {
        Some(n) => n,
        None => { eprintln!("skipping: matvec.xclbin unavailable"); return; }
    };
    let g = |i: usize| ((i as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
    let x: Vec<f32> = (0..D).map(g).collect();
    let n_exp = 6usize;
    let weights: Vec<f32> = (0..n_exp).map(|i| 0.1 + 0.03 * i as f32).collect();
    let mk = |seed: usize| -> Vec<Vec<f32>> {
        (0..n_exp).map(|e| (0..D * D).map(|i| g(i + e * 977 + seed) * 0.1).collect()).collect()
    };
    let (wg, wu, wd) = (mk(7), mk(101), mk(907));

    let cpu_mv = |w: &[f32], v: &[f32]| cpu_matvec(w, v);
    let npu_mv = |w: &[f32], v: &[f32]| npu.matvec(w, v);

    // reference: all six experts on the CPU
    let mut reference = vec![0f32; D];
    for e in 0..n_exp {
        let y = expert(&cpu_mv, &x, &wg[e], &wu[e], &wd[e], weights[e]);
        for (a, &v) in reference.iter_mut().zip(y.iter()) { *a += v; }
    }

    // co-executed: last expert on the NPU, the rest on the CPU (share ~1/6,
    // matching the measured ~12.6 / ~93 GB/s bandwidth ratio)
    let n_secondary = 1usize;
    let split = n_exp - n_secondary;
    let mut got = vec![0f32; D];
    for e in 0..split {
        let y = expert(&cpu_mv, &x, &wg[e], &wu[e], &wd[e], weights[e]);
        for (a, &v) in got.iter_mut().zip(y.iter()) { *a += v; }
    }
    for e in split..n_exp {
        let y = expert(&npu_mv, &x, &wg[e], &wu[e], &wd[e], weights[e]);
        for (a, &v) in got.iter_mut().zip(y.iter()) { *a += v; }
    }

    let mut max_rel = 0f32;
    for (a, b) in got.iter().zip(reference.iter()) {
        max_rel = max_rel.max((a - b).abs() / b.abs().max(1e-3));
    }
    println!("MoE step, {split} experts on CPU + {n_secondary} on NPU: max_rel = {max_rel:e}");
    assert!(max_rel < 1e-3, "co-executed MoE diverged: max_rel = {max_rel}");
}
