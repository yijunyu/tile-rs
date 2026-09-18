//! Does overlapping NPU expert work with host work actually save wall clock?
//!
//! The earlier co-execution test ran the engines sequentially, so it proved
//! correctness but not benefit. This one starts the NPU expert asynchronously,
//! does the host experts while it is in flight, then joins — and times all
//! three arrangements.
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
const TILE: usize = 4096;
const OPCODE: c_uint = 3;

struct Npu { h: *mut c_void, insts: *mut c_void, words: c_uint,
             ba: *mut c_void, bb: *mut c_void, bc: *mut c_void }

impl Npu {
    fn new() -> Option<Self> {
        let xclbin = std::env::var("AIE_MATVEC")
            .unwrap_or_else(|_| "/data/aie-kernels/matvec.xclbin".into());
        let ib = std::fs::read(xclbin.replace(".xclbin", ".insts.bin")).ok()?;
        let cx = CString::new(xclbin).ok()?; let ck = CString::new("").ok()?;
        let h = unsafe { aie_open(0, cx.as_ptr(), ck.as_ptr()) };
        if h.is_null() { return None; }
        let insts = unsafe { aie_bo_alloc(h, ib.len(), 1, 1) };
        unsafe { aie_bo_write(insts, ib.as_ptr() as *const c_void, ib.len()) };
        let nb = TILE * 4;
        Some(Self { h, insts, words: (ib.len()/4) as c_uint,
                    ba: unsafe { aie_bo_alloc(h, nb, 3, 0) },
                    bb: unsafe { aie_bo_alloc(h, nb, 4, 0) },
                    bc: unsafe { aie_bo_alloc(h, nb, 5, 0) } })
    }
    /// Upload operands and start the run without waiting.
    fn start(&self, w: &[f32], x: &[f32]) -> *mut c_void {
        let mut a = vec![0f32; TILE]; a[..D].copy_from_slice(x);
        let mut b = vec![0f32; TILE];
        for j in 0..D { for i in 0..D { b[i*D+j] = w[j*D+i]; } }
        let nb = TILE * 4;
        unsafe {
            aie_bo_write(self.ba, a.as_ptr() as *const c_void, nb);
            aie_bo_write(self.bb, b.as_ptr() as *const c_void, nb);
        }
        let bos = [self.ba, self.bb, self.bc];
        unsafe { aie_run_start(self.h, OPCODE, self.insts, self.words, bos.as_ptr(), 3) }
    }
    fn join(&self, r: *mut c_void) -> Vec<f32> {
        assert_eq!(0, unsafe { aie_run_wait(r) });
        let mut o = vec![0f32; TILE];
        unsafe { aie_bo_read(self.bc, o.as_mut_ptr() as *mut c_void, TILE*4) };
        o.truncate(D); o
    }
}
impl Drop for Npu {
    fn drop(&mut self) { unsafe {
        aie_bo_free(self.ba); aie_bo_free(self.bb); aie_bo_free(self.bc);
        aie_bo_free(self.insts); aie_close(self.h);
    }}
}

fn cpu_matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..D).map(|j| (0..D).map(|i| w[j*D+i]*x[i]).sum()).collect()
}
fn silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }

fn cpu_expert(x: &[f32], wg: &[f32], wu: &[f32], wd: &[f32], w_e: f32) -> Vec<f32> {
    const C: f32 = 10.0;
    let g = cpu_matvec(wg, x); let u = cpu_matvec(wu, x);
    let mid: Vec<f32> = g.iter().zip(u.iter())
        .map(|(&a,&b)| silu(a.min(C))*b.clamp(-C,C)*w_e).collect();
    cpu_matvec(wd, &mid)
}

#[test]
fn concurrent_beats_sequential() {
    let npu = match Npu::new() { Some(n)=>n, None=>{ eprintln!("skip: no matvec.xclbin"); return; } };
    let g = |i: usize| ((i as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
    let x: Vec<f32> = (0..D).map(g).collect();
    let n_exp = 6usize;
    let wt: Vec<f32> = (0..n_exp).map(|i| 0.1 + 0.03*i as f32).collect();
    let mk = |s: usize| -> Vec<Vec<f32>> {
        (0..n_exp).map(|e| (0..D*D).map(|i| g(i+e*977+s)*0.1).collect()).collect() };
    let (wg, wu, wd) = (mk(7), mk(101), mk(907));
    let host_experts = n_exp - 1;      // last one goes to the NPU

    let run_host = |acc: &mut Vec<f32>| {
        for e in 0..host_experts {
            let y = cpu_expert(&x, &wg[e], &wu[e], &wd[e], wt[e]);
            for (a,&v) in acc.iter_mut().zip(y.iter()) { *a += v; }
        }
    };

    const IT: usize = 200;
    // warm both paths
    for _ in 0..20 { let r = npu.start(&wg[5], &x); let _ = npu.join(r); }

    // sequential: host first, then NPU
    let t0 = Instant::now();
    for _ in 0..IT {
        let mut acc = vec![0f32; D];
        run_host(&mut acc);
        let r = npu.start(&wg[5], &x);
        let y = npu.join(r);
        for (a,&v) in acc.iter_mut().zip(y.iter()) { *a += v; }
        std::hint::black_box(&acc);
    }
    let t_seq = t0.elapsed().as_secs_f64()/IT as f64;

    // concurrent: start the NPU, do the host experts while it is in flight
    let t0 = Instant::now();
    for _ in 0..IT {
        let r = npu.start(&wg[5], &x);
        let mut acc = vec![0f32; D];
        run_host(&mut acc);
        let y = npu.join(r);
        for (a,&v) in acc.iter_mut().zip(y.iter()) { *a += v; }
        std::hint::black_box(&acc);
    }
    let t_con = t0.elapsed().as_secs_f64()/IT as f64;

    // host-only, all six experts, for reference
    let t0 = Instant::now();
    for _ in 0..IT {
        let mut acc = vec![0f32; D];
        for e in 0..n_exp {
            let y = cpu_expert(&x, &wg[e], &wu[e], &wd[e], wt[e]);
            for (a,&v) in acc.iter_mut().zip(y.iter()) { *a += v; }
        }
        std::hint::black_box(&acc);
    }
    let t_host = t0.elapsed().as_secs_f64()/IT as f64;

    println!("host-only (6 experts)     {:8.1} us", t_host*1e6);
    println!("sequential (5 host + NPU) {:8.1} us", t_seq*1e6);
    println!("concurrent (5 host || NPU){:8.1} us", t_con*1e6);
    println!("overlap saves             {:8.1} us  ({:.2}x vs sequential)",
             (t_seq-t_con)*1e6, t_seq/t_con);
    assert!(t_con < t_seq, "overlap did not help: {t_con} vs {t_seq}");
}
