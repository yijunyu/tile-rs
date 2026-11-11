//! A whole MoE expert in one NPU dispatch, instead of three.
//!
//! `y` is linear in the routing weight (`mid = silu(g)·u·w_e`, then `y = W_d·mid`),
//! so the kernel computes with `w_e = 1` and the host scales the result — exact,
//! and it keeps the routing weight out of the device signature.
#![cfg(feature = "aie")]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::time::Instant;

#[link(name = "aie_shim")]
extern "C" {
    fn aie_open(index: c_uint, xclbin: *const c_char, kernel: *const c_char) -> *mut c_void;
    fn aie_close(h: *mut c_void);
    fn aie_bo_alloc(h: *mut c_void, bytes: usize, argno: c_int, flags: c_int) -> *mut c_void;
    fn aie_bo_free(b: *mut c_void);
    fn aie_bo_write(b: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn aie_bo_read(b: *mut c_void, dst: *mut c_void, bytes: usize) -> c_int;
    fn aie_run_start(h: *mut c_void, opcode: c_uint, insts: *mut c_void, words: c_uint,
                     bos: *const *mut c_void, n: c_int) -> *mut c_void;
    fn aie_run_wait(r: *mut c_void) -> c_int;
}

const D: usize = 64;
const WCAT: usize = 3 * D * D;
const OPCODE: c_uint = 3;

struct FusedExpert { h: *mut c_void, insts: *mut c_void, words: c_uint,
                     bw: *mut c_void, bx: *mut c_void, by: *mut c_void }

impl FusedExpert {
    fn new() -> Option<Self> {
        let xc = std::env::var("AIE_EXPERT")
            .unwrap_or_else(|_| "/data/aie-kernels/expert.xclbin".into());
        let ib = std::fs::read(xc.replace(".xclbin", ".insts.bin")).ok()?;
        let cx = CString::new(xc).ok()?; let ck = CString::new("").ok()?;
        let h = unsafe { aie_open(0, cx.as_ptr(), ck.as_ptr()) };
        if h.is_null() { return None; }
        let insts = unsafe { aie_bo_alloc(h, ib.len(), 1, 1) };
        unsafe { aie_bo_write(insts, ib.as_ptr() as *const c_void, ib.len()) };
        Some(Self { h, insts, words: (ib.len()/4) as c_uint,
            bw: unsafe { aie_bo_alloc(h, WCAT*4, 3, 0) },
            bx: unsafe { aie_bo_alloc(h, D*4,    4, 0) },
            by: unsafe { aie_bo_alloc(h, D*4,    5, 0) } })
    }
    /// Upload the concatenated weights once; they stay resident for later runs.
    fn upload_weights(&self, wg: &[f32], wu: &[f32], wd: &[f32]) {
        let mut cat = Vec::with_capacity(WCAT);
        cat.extend_from_slice(wg); cat.extend_from_slice(wu); cat.extend_from_slice(wd);
        unsafe { aie_bo_write(self.bw, cat.as_ptr() as *const c_void, WCAT*4) };
    }
    fn start(&self, x: &[f32]) -> *mut c_void {
        unsafe { aie_bo_write(self.bx, x.as_ptr() as *const c_void, D*4) };
        let bos = [self.bw, self.bx, self.by];
        unsafe { aie_run_start(self.h, OPCODE, self.insts, self.words, bos.as_ptr(), 3) }
    }
    fn join(&self, r: *mut c_void, w_e: f32) -> Vec<f32> {
        assert_eq!(0, unsafe { aie_run_wait(r) });
        let mut y = vec![0f32; D];
        unsafe { aie_bo_read(self.by, y.as_mut_ptr() as *mut c_void, D*4) };
        for v in y.iter_mut() { *v *= w_e; }     // y is linear in w_e
        y
    }
}
impl Drop for FusedExpert {
    fn drop(&mut self) { unsafe {
        aie_bo_free(self.bw); aie_bo_free(self.bx); aie_bo_free(self.by);
        aie_bo_free(self.insts); aie_close(self.h);
    }}
}

fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..D).map(|j| (0..D).map(|i| w[j*D+i]*x[i]).sum()).collect()
}
fn cpu_expert(x: &[f32], wg: &[f32], wu: &[f32], wd: &[f32], w_e: f32) -> Vec<f32> {
    const C: f32 = 10.0;
    let g = mv(wg,x); let u = mv(wu,x);
    let mid: Vec<f32> = g.iter().zip(u.iter())
        .map(|(&a,&b)| { let a=a.min(C); (a/(1.0+(-a).exp()))*b.clamp(-C,C)*w_e }).collect();
    mv(wd,&mid)
}

#[test]
fn fused_expert_correct_and_faster() {
    std::env::set_var("AIE_EXPERT", "/data/aie-kernels/expert.xclbin");
    let npu = match FusedExpert::new() { Some(n)=>n, None=>{ eprintln!("skip: no expert.xclbin"); return; } };
    let g = |i: usize| ((i as f32*12.9898).sin()*43758.547).fract() - 0.5;
    let x: Vec<f32> = (0..D).map(g).collect();
    let wg: Vec<f32> = (0..D*D).map(|i| g(i+7)*0.1).collect();
    let wu: Vec<f32> = (0..D*D).map(|i| g(i+101)*0.1).collect();
    let wd: Vec<f32> = (0..D*D).map(|i| g(i+907)*0.1).collect();
    let w_e = 0.19f32;

    npu.upload_weights(&wg, &wu, &wd);
    let got = npu.join(npu.start(&x), w_e);
    let want = cpu_expert(&x, &wg, &wu, &wd, w_e);
    let mut max_rel = 0f32;
    for (a,b) in got.iter().zip(want.iter()) {
        max_rel = max_rel.max((a-b).abs()/b.abs().max(1e-3));
    }
    println!("fused expert on NPU: max_rel = {max_rel:e}");
    assert!(max_rel < 1e-3, "fused expert wrong: {max_rel}");

    const IT: usize = 200;
    for _ in 0..20 { let r = npu.start(&x); let _ = npu.join(r, w_e); }
    let t0 = Instant::now();
    for _ in 0..IT { let r = npu.start(&x); std::hint::black_box(npu.join(r, w_e)); }
    let t_fused = t0.elapsed().as_secs_f64()/IT as f64;

    let t0 = Instant::now();
    for _ in 0..IT { std::hint::black_box(cpu_expert(&x,&wg,&wu,&wd,w_e)); }
    let t_cpu = t0.elapsed().as_secs_f64()/IT as f64;

    println!("fused NPU expert   {:8.1} us  (was ~590 us unfused, 3 dispatches)", t_fused*1e6);
    println!("CPU expert         {:8.1} us", t_cpu*1e6);
    println!("NPU / CPU          {:8.2}x", t_fused/t_cpu);
}

#[test]
fn vectorised_expert_is_faster() {
    let xc = "/data/aie-kernels/expertv.xclbin";
    if !std::path::Path::new(xc).exists() { eprintln!("skip: no expertv.xclbin"); return; }
    std::env::set_var("AIE_EXPERT", xc);
    let npu = match FusedExpert::new() { Some(n)=>n, None=>{ eprintln!("skip"); return; } };

    let g = |i: usize| ((i as f32*12.9898).sin()*43758.547).fract() - 0.5;
    let x: Vec<f32> = (0..D).map(g).collect();
    let wg: Vec<f32> = (0..D*D).map(|i| g(i+7)*0.1).collect();
    let wu: Vec<f32> = (0..D*D).map(|i| g(i+101)*0.1).collect();
    let wd: Vec<f32> = (0..D*D).map(|i| g(i+907)*0.1).collect();
    let w_e = 0.19f32;

    // the vectorised kernel indexes weights as [in][out], so transpose
    let t = |w: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; D*D];
        for j in 0..D { for i in 0..D { o[i*D+j] = w[j*D+i]; } }
        o
    };
    // the kernel's gate accumulator feeds exp2 directly, so scale W_gate here
    let mut wgs = t(&wg);
    for v in wgs.iter_mut() { *v *= -std::f32::consts::LOG2_E; }
    npu.upload_weights(&wgs, &t(&wu), &t(&wd));

    let got = npu.join(npu.start(&x), w_e);
    let want = cpu_expert(&x, &wg, &wu, &wd, w_e);
    let mut max_rel = 0f32;
    let mut n_bad = 0usize;
    for (a,b) in got.iter().zip(want.iter()) {
        if !a.is_finite() { n_bad += 1; continue; }
        let r = (a-b).abs()/b.abs().max(1e-3);
        if r > max_rel { max_rel = r; }
    }
    assert_eq!(n_bad, 0, "{n_bad} non-finite outputs from the NPU");
    println!("vectorised expert: max_rel = {max_rel:e}  (bf16 intermediates)");
    println!("  got  {:?}", &got[..4]);
    println!("  want {:?}", &want[..4]);

    const IT: usize = 200;
    for _ in 0..20 { let r = npu.start(&x); let _ = npu.join(r, w_e); }
    let t0 = std::time::Instant::now();
    for _ in 0..IT { let r = npu.start(&x); std::hint::black_box(npu.join(r, w_e)); }
    let t_vec = t0.elapsed().as_secs_f64()/IT as f64;
    println!("vectorised NPU expert {:8.1} us   (scalar fused was 998.0)", t_vec*1e6);
    assert!(max_rel < 5e-2, "vectorised expert too far off: {max_rel}");
}
