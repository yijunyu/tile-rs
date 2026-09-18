//! AMD Ryzen AI / AIE (XDNA2) backend for the HAL.
//!
//! Runtime abstraction over XRT via dlopen-based FFI, mirroring [`crate::cuda`].
//! Enabled by the `aie` feature.
//!
//! # Status
//!
//! The XRT **C** API used below drives buffers and runs correctly, but it
//! **cannot load an xclbin on XDNA2**. `xrtDeviceLoadXclbinFile` is the legacy
//! Alveo/PL path and the NPU rejects it:
//!
//! ```text
//! [XRT] ERROR: load_axlf: Operation not supported
//! xrtDeviceLoadXclbinFile failed: error -1
//! ```
//!
//! The NPU requires `register_xclbin` + `hw_context`, which exist **only in
//! XRT's C++ API** — `nm -D libxrt_coreutil.so` shows no C entry point for
//! either. This was verified on hardware: `AieDevice::new` succeeds (the device
//! opens), and the load then fails.
//!
//! The working path is therefore `../aie_shim/aie_shim.cpp`, a small C++ wrapper
//! exposing that flow behind an opaque C surface, exercised by
//! `tests/aie_shim_hw.rs` — vecadd on the NPU from Rust, `max |diff| = 0`.
//!
//! What remains valid here: the buffer/stream/launcher trait surface, and the
//! dispatch convention `(opcode, insts_bo, insts_len, ...data BOs)`, which was
//! confirmed on hardware (a bare `(a, b, c)` call returns
//! `ERT_CMD_STATE_ERROR`). Migrating this module onto the shim is mechanical:
//! the Rust structure does not change, only which symbols it calls.

use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::backend::DynDevice;
use crate::buffer::DeviceRepr;
use crate::device::{Device, MemInfo};
use crate::error::{HalError, HalResult};
use crate::kernel::{KernelLauncher, KernelMode, LaunchGrid};
use crate::stream::Stream;

// ── XRT FFI types ──

type XrtResult = i32;
type XrtDeviceHandle = *mut c_void;
type XrtKernelHandle = *mut c_void;
type XrtRunHandle = *mut c_void;
type XrtBufferHandle = *mut c_void;

/// `xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE`
const SYNC_TO_DEVICE: i32 = 0;
/// `xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE`
const SYNC_FROM_DEVICE: i32 = 1;

/// `XRT_BO_FLAGS_HOST_ONLY` — NPU buffers are host-visible.
const BO_FLAGS_HOST_ONLY: u32 = 1 << 2;

/// NPU run opcode for a transaction (instruction-stream) dispatch.
pub const OPCODE_TRANSACTION: u32 = 3;

/// `ert_cmd_state::ERT_CMD_STATE_COMPLETED`
const ERT_CMD_STATE_COMPLETED: i32 = 4;

struct XrtDriver {
    _lib: libloading::Library,
    xrt_device_open: unsafe extern "C" fn(u32) -> XrtDeviceHandle,
    xrt_device_close: unsafe extern "C" fn(XrtDeviceHandle) -> XrtResult,
    xrt_device_load_xclbin_file: unsafe extern "C" fn(XrtDeviceHandle, *const u8) -> XrtResult,
    xrt_device_get_xclbin_uuid: unsafe extern "C" fn(XrtDeviceHandle, *mut u8) -> XrtResult,
    xrt_bo_alloc: unsafe extern "C" fn(XrtDeviceHandle, usize, u32, u32) -> XrtBufferHandle,
    xrt_bo_free: unsafe extern "C" fn(XrtBufferHandle) -> XrtResult,
    xrt_bo_write: unsafe extern "C" fn(XrtBufferHandle, *const c_void, usize, usize) -> XrtResult,
    xrt_bo_read: unsafe extern "C" fn(XrtBufferHandle, *mut c_void, usize, usize) -> XrtResult,
    xrt_bo_sync: unsafe extern "C" fn(XrtBufferHandle, i32, usize, usize) -> XrtResult,
    xrt_bo_map: unsafe extern "C" fn(XrtBufferHandle) -> *mut c_void,
    xrt_kernel_open: unsafe extern "C" fn(XrtDeviceHandle, *const u8, *const u8) -> XrtKernelHandle,
    xrt_kernel_close: unsafe extern "C" fn(XrtKernelHandle) -> XrtResult,
    xrt_kernel_arg_group_id: unsafe extern "C" fn(XrtKernelHandle, i32) -> i32,
    xrt_run_open: unsafe extern "C" fn(XrtKernelHandle) -> XrtRunHandle,
    xrt_run_set_arg: unsafe extern "C" fn(XrtRunHandle, i32, ...) -> XrtResult,
    xrt_run_start: unsafe extern "C" fn(XrtRunHandle) -> XrtResult,
    xrt_run_wait: unsafe extern "C" fn(XrtRunHandle) -> i32,
    xrt_run_close: unsafe extern "C" fn(XrtRunHandle) -> XrtResult,
}

unsafe impl Send for XrtDriver {}
unsafe impl Sync for XrtDriver {}

static XRT_DRIVER: OnceLock<Result<XrtDriver, String>> = OnceLock::new();

fn xrt_driver() -> HalResult<&'static XrtDriver> {
    XRT_DRIVER
        .get_or_init(|| unsafe {
            let lib = libloading::Library::new("libxrt_coreutil.so")
                .or_else(|_| libloading::Library::new("libxrt_coreutil.so.2"))
                .map_err(|e| format!("cannot load libxrt_coreutil.so: {}", e))?;

            macro_rules! sym {
                ($name:ident, $sym:expr) => {
                    let $name = *lib
                        .get::<unsafe extern "C" fn() -> XrtResult>($sym)
                        .map_err(|e| format!("symbol {} not found: {}", stringify!($name), e))?;
                    #[allow(clippy::transmute_ptr_to_ptr)]
                    let $name = std::mem::transmute($name);
                };
            }

            sym!(xrt_device_open, b"xrtDeviceOpen\0");
            sym!(xrt_device_close, b"xrtDeviceClose\0");
            sym!(xrt_device_load_xclbin_file, b"xrtDeviceLoadXclbinFile\0");
            sym!(xrt_device_get_xclbin_uuid, b"xrtDeviceGetXclbinUUID\0");
            sym!(xrt_bo_alloc, b"xrtBOAlloc\0");
            sym!(xrt_bo_free, b"xrtBOFree\0");
            sym!(xrt_bo_write, b"xrtBOWrite\0");
            sym!(xrt_bo_read, b"xrtBORead\0");
            sym!(xrt_bo_sync, b"xrtBOSync\0");
            sym!(xrt_bo_map, b"xrtBOMap\0");
            sym!(xrt_kernel_open, b"xrtPLKernelOpen\0");
            sym!(xrt_kernel_close, b"xrtKernelClose\0");
            sym!(xrt_kernel_arg_group_id, b"xrtKernelArgGroupId\0");
            sym!(xrt_run_open, b"xrtRunOpen\0");
            sym!(xrt_run_set_arg, b"xrtRunSetArg\0");
            sym!(xrt_run_start, b"xrtRunStart\0");
            sym!(xrt_run_wait, b"xrtRunWait\0");
            sym!(xrt_run_close, b"xrtRunClose\0");

            Ok(XrtDriver {
                _lib: lib,
                xrt_device_open,
                xrt_device_close,
                xrt_device_load_xclbin_file,
                xrt_device_get_xclbin_uuid,
                xrt_bo_alloc,
                xrt_bo_free,
                xrt_bo_write,
                xrt_bo_read,
                xrt_bo_sync,
                xrt_bo_map,
                xrt_kernel_open,
                xrt_kernel_close,
                xrt_kernel_arg_group_id,
                xrt_run_open,
                xrt_run_set_arg,
                xrt_run_start,
                xrt_run_wait,
                xrt_run_close,
            })
        })
        .as_ref()
        .map_err(|e| HalError::BackendNotAvailable(e.clone()))
}

fn check_xrt(ret: XrtResult, op: &str) -> HalResult<()> {
    if ret == 0 {
        Ok(())
    } else {
        Err(HalError::RuntimeError(format!(
            "{} failed: error {}",
            op, ret
        )))
    }
}

// ── Stream ──

/// XRT has no stream object; ordering is per-run. Synchronisation happens in
/// [`KernelLauncher::launch`] via `xrtRunWait`, so this is a marker type that
/// keeps the HAL surface uniform across backends.
pub struct AieStream {
    _priv: (),
}

impl AieStream {
    fn new() -> HalResult<Self> {
        Ok(Self { _priv: () })
    }
}

impl Stream for AieStream {
    fn synchronize(&self) -> HalResult<()> {
        Ok(())
    }
}

unsafe impl Send for AieStream {}

// ── Buffer ──

/// Mapping state of an [`AieBuffer`].
///
/// XRT allocates a **local copy** of an argument buffer when the compute unit's
/// bank connectivity does not match the allocation, and says so only in a
/// warning:
///
/// ```text
/// [XRT] WARNING: ... the argument is allocated in bank 0, the compute unit is
/// connected to bank 65537. Allocating local copy of argument buffer in connected bank.
/// ```
///
/// A host mapping of the *original* buffer may then not see what the kernel
/// wrote into that copy.
///
/// **What this typestate does and does not buy you.** It forces a caller to
/// demonstrate the mapping agrees with XRT's own read path before taking a
/// zero-copy view, which is a cheap necessary condition. It is *not* proof of
/// soundness: in the bug that motivated this — reads through a cached mapped
/// pointer went stale after the first call — an isolated reproduction with the
/// same access pattern showed all views agreeing, so this check would have
/// passed. The guard that actually caught it was numerical: recompute a sample
/// of the kernel's output on the CPU and compare, on a call *other* than the
/// first. Use both.
pub struct Unverified;
/// The mapped view agrees with what the kernel wrote: zero-copy access is sound.
pub struct Direct;
/// XRT keeps a copy; only [`DeviceBuffer::copy_to_host`] /
/// [`DeviceBuffer::copy_from_host`] are valid.
pub struct Shadowed;

pub struct AieBuffer<T: DeviceRepr, S = Unverified> {
    bo: XrtBufferHandle,
    mapped: *mut c_void,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
    _state: std::marker::PhantomData<S>,
}

impl<T: DeviceRepr, S> crate::buffer::DeviceBuffer<T> for AieBuffer<T, S> {
    fn len(&self) -> usize {
        self.count
    }

    fn copy_from_host(&mut self, src: &[T], _stream: &dyn Stream) -> HalResult<()> {
        let drv = xrt_driver()?;
        let size = src.len() * std::mem::size_of::<T>();
        if src.len() > self.count {
            return Err(HalError::MemoryError(format!(
                "copy_from_host: {} elements into a {}-element buffer",
                src.len(),
                self.count
            )));
        }
        check_xrt(
            unsafe { (drv.xrt_bo_write)(self.bo, src.as_ptr() as *const c_void, size, 0) },
            "xrtBOWrite",
        )?;
        check_xrt(
            unsafe { (drv.xrt_bo_sync)(self.bo, SYNC_TO_DEVICE, size, 0) },
            "xrtBOSync(TO_DEVICE)",
        )
    }

    fn copy_to_host(&self, _stream: &dyn Stream) -> HalResult<Vec<T>> {
        let drv = xrt_driver()?;
        let size = self.count * std::mem::size_of::<T>();
        check_xrt(
            unsafe { (drv.xrt_bo_sync)(self.bo, SYNC_FROM_DEVICE, size, 0) },
            "xrtBOSync(FROM_DEVICE)",
        )?;
        let mut dst = vec![unsafe { std::mem::zeroed::<T>() }; self.count];
        check_xrt(
            unsafe { (drv.xrt_bo_read)(self.bo, dst.as_mut_ptr() as *mut c_void, size, 0) },
            "xrtBORead",
        )?;
        Ok(dst)
    }

    fn as_raw_ptr(&self) -> *const c_void {
        self.bo as *const c_void
    }

    fn as_raw_mut_ptr(&mut self) -> *mut c_void {
        self.bo
    }
}

/// Outcome of [`AieBuffer::verify_mapping_after_run`].
pub enum Mapping<T: DeviceRepr> {
    Direct(AieBuffer<T, Direct>),
    Shadowed(AieBuffer<T, Shadowed>),
}

impl<T: DeviceRepr> AieBuffer<T, Unverified> {
    /// Decide whether this buffer's host mapping is a valid view of what the
    /// kernel writes, by comparing it against XRT's own read path.
    ///
    /// **This is only meaningful once a dispatch has written the buffer.** A
    /// buffer that only the host has touched will compare equal through both
    /// paths whether or not a shadow copy exists — that host-only round trip
    /// was measured to pass on a buffer that was in fact shadowed, so it proves
    /// nothing. Run the kernel, then call this.
    pub fn verify_mapping_after_run(self) -> HalResult<Mapping<T>> {
        if self.mapped.is_null() {
            return Ok(Mapping::Shadowed(self.into_state()));
        }
        let drv = xrt_driver()?;
        let size = self.count * std::mem::size_of::<T>();
        check_xrt(
            unsafe { (drv.xrt_bo_sync)(self.bo, SYNC_FROM_DEVICE, size, 0) },
            "xrtBOSync(FROM_DEVICE)",
        )?;
        let mut via_read = vec![0u8; size];
        check_xrt(
            unsafe { (drv.xrt_bo_read)(self.bo, via_read.as_mut_ptr() as *mut c_void, size, 0) },
            "xrtBORead",
        )?;
        let via_map = unsafe { std::slice::from_raw_parts(self.mapped as *const u8, size) };
        if via_map == via_read.as_slice() {
            Ok(Mapping::Direct(self.into_state()))
        } else {
            Ok(Mapping::Shadowed(self.into_state()))
        }
    }
}

impl<T: DeviceRepr, S> AieBuffer<T, S> {
    fn into_state<S2>(self) -> AieBuffer<T, S2> {
        let out = AieBuffer {
            bo: self.bo,
            mapped: self.mapped,
            count: self.count,
            _phantom: std::marker::PhantomData,
            _state: std::marker::PhantomData,
        };
        std::mem::forget(self);          // ownership of `bo` moves with it
        out
    }
}

impl<T: DeviceRepr> AieBuffer<T, Direct> {
    /// Zero-copy view. Reachable only after the mapping has been shown to agree
    /// with what the kernel writes, so the stale-read failure is unrepresentable.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mapped as *const T, self.count) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mapped as *mut T, self.count) }
    }
}

unsafe impl<T: DeviceRepr, S> Send for AieBuffer<T, S> {}

impl<T: DeviceRepr, S> Drop for AieBuffer<T, S> {
    fn drop(&mut self) {
        if let Ok(drv) = xrt_driver() {
            unsafe {
                (drv.xrt_bo_free)(self.bo);
            }
        }
    }
}

// ── Kernel Launcher ──

pub struct AieLauncher {
    device: XrtDeviceHandle,
    kernel: Option<XrtKernelHandle>,
    /// Instruction stream BO (`insts.bin`) and its length in 32-bit words.
    insts: Option<(XrtBufferHandle, usize)>,
}

unsafe impl Send for AieLauncher {}

impl AieLauncher {
    /// Locate the instruction stream that belongs to an xclbin.
    ///
    /// `aiecc.py --aie-generate-npu-insts --npu-insts-name=<n>` writes it
    /// separately; by convention we look for `<xclbin-stem>.insts.bin`, and fall
    /// back to `insts.bin` in the same directory.
    fn insts_path_for(xclbin: &Path) -> Option<PathBuf> {
        let stem_sibling = xclbin.with_extension("insts.bin");
        if stem_sibling.exists() {
            return Some(stem_sibling);
        }
        let generic = xclbin.parent()?.join("insts.bin");
        generic.exists().then_some(generic)
    }
}

impl KernelLauncher for AieLauncher {
    type Stream = AieStream;

    fn load(&mut self, path: &Path, _mode: KernelMode) -> HalResult<()> {
        let drv = xrt_driver()?;
        let path_cstr = CString::new(path.to_str().unwrap_or(""))
            .map_err(|_| HalError::KernelError("invalid xclbin path".into()))?;
        check_xrt(
            unsafe { (drv.xrt_device_load_xclbin_file)(self.device, path_cstr.as_ptr() as *const u8) },
            "xrtDeviceLoadXclbinFile",
        )?;

        let mut uuid = [0u8; 16];
        check_xrt(
            unsafe { (drv.xrt_device_get_xclbin_uuid)(self.device, uuid.as_mut_ptr()) },
            "xrtDeviceGetXclbinUUID",
        )?;

        // MLIR-AIE names the design's entry point MLIR_AIE by default.
        let kname = CString::new("MLIR_AIE").unwrap();
        let kern = unsafe {
            (drv.xrt_kernel_open)(self.device, uuid.as_ptr(), kname.as_ptr() as *const u8)
        };
        if kern.is_null() {
            return Err(HalError::KernelError(
                "xrtPLKernelOpen returned null — see the module docs on hw_context".into(),
            ));
        }
        self.kernel = Some(kern);

        // Upload the instruction stream, if one sits next to the xclbin.
        if let Some(ip) = Self::insts_path_for(path) {
            let bytes = std::fs::read(&ip)
                .map_err(|e| HalError::KernelError(format!("reading {}: {}", ip.display(), e)))?;
            let grp = unsafe { (drv.xrt_kernel_arg_group_id)(kern, 1) };
            let bo = unsafe {
                (drv.xrt_bo_alloc)(
                    self.device,
                    bytes.len(),
                    BO_FLAGS_HOST_ONLY,
                    if grp < 0 { 0 } else { grp as u32 },
                )
            };
            if bo.is_null() {
                return Err(HalError::MemoryError("xrtBOAlloc(insts) returned null".into()));
            }
            check_xrt(
                unsafe { (drv.xrt_bo_write)(bo, bytes.as_ptr() as *const c_void, bytes.len(), 0) },
                "xrtBOWrite(insts)",
            )?;
            check_xrt(
                unsafe { (drv.xrt_bo_sync)(bo, SYNC_TO_DEVICE, bytes.len(), 0) },
                "xrtBOSync(insts)",
            )?;
            self.insts = Some((bo, bytes.len() / 4));
        }
        Ok(())
    }

    unsafe fn launch(
        &self,
        _name: &str,
        _grid: LaunchGrid,
        args: &mut [*mut c_void],
        _stream: &Self::Stream,
    ) -> HalResult<()> {
        let drv = xrt_driver()?;
        let kern = self
            .kernel
            .ok_or_else(|| HalError::KernelError("no xclbin loaded".into()))?;
        let (insts_bo, insts_len) = self
            .insts
            .ok_or_else(|| HalError::KernelError("no instruction stream loaded".into()))?;

        let run = unsafe { (drv.xrt_run_open)(kern) };
        if run.is_null() {
            return Err(HalError::KernelError("xrtRunOpen returned null".into()));
        }

        // NPU transaction convention: (opcode, insts, insts_len, data BOs...).
        let set = |idx: i32, f: &dyn Fn() -> XrtResult| -> HalResult<()> {
            check_xrt(f(), &format!("xrtRunSetArg({})", idx))
        };
        set(0, &|| unsafe {
            (drv.xrt_run_set_arg)(run, 0, OPCODE_TRANSACTION)
        })?;
        set(1, &|| unsafe {
            (drv.xrt_run_set_arg)(run, 1, insts_bo)
        })?;
        set(2, &|| unsafe {
            (drv.xrt_run_set_arg)(run, 2, insts_len as u32)
        })?;
        for (i, a) in args.iter().enumerate() {
            let idx = i as i32 + 3;
            set(idx, &|| unsafe { (drv.xrt_run_set_arg)(run, idx, *a) })?;
        }

        let start = unsafe { (drv.xrt_run_start)(run) };
        if start != 0 {
            unsafe { (drv.xrt_run_close)(run) };
            return Err(HalError::KernelError(format!("xrtRunStart failed: {}", start)));
        }
        let state = unsafe { (drv.xrt_run_wait)(run) };
        unsafe { (drv.xrt_run_close)(run) };
        if state != ERT_CMD_STATE_COMPLETED {
            return Err(HalError::KernelError(format!(
                "run did not complete: ert state {}",
                state
            )));
        }
        Ok(())
    }
}

impl Drop for AieLauncher {
    fn drop(&mut self) {
        if let Ok(drv) = xrt_driver() {
            unsafe {
                if let Some((bo, _)) = self.insts {
                    (drv.xrt_bo_free)(bo);
                }
                if let Some(k) = self.kernel {
                    (drv.xrt_kernel_close)(k);
                }
            }
        }
    }
}

// ── Device ──

pub struct AieDevice {
    ordinal: u32,
    name: String,
    handle: XrtDeviceHandle,
}

impl AieDevice {
    pub fn new(ordinal: u32) -> HalResult<Self> {
        let drv = xrt_driver()?;
        let handle = unsafe { (drv.xrt_device_open)(ordinal) };
        if handle.is_null() {
            return Err(HalError::DeviceError(format!(
                "xrtDeviceOpen({}) returned null — is the amdxdna driver loaded and \
                 /dev/accel/accel0 present?",
                ordinal
            )));
        }
        Ok(Self {
            ordinal,
            // XRT's C API has no simple name query; xrt-smi is the source of
            // truth. Keep it descriptive rather than wrong.
            name: format!("AMD XDNA NPU (device {})", ordinal),
            handle,
        })
    }

    /// Raw XRT device handle, for callers that need the C API directly.
    pub fn raw(&self) -> XrtDeviceHandle {
        self.handle
    }
}

impl Device for AieDevice {
    type Stream = AieStream;
    type Buffer<T: DeviceRepr> = AieBuffer<T>;
    type Launcher = AieLauncher;

    fn ordinal(&self) -> u32 {
        self.ordinal
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn create_stream(&self) -> HalResult<AieStream> {
        AieStream::new()
    }

    fn alloc<T: DeviceRepr>(&self, count: usize) -> HalResult<AieBuffer<T>> {
        let drv = xrt_driver()?;
        let size = count * std::mem::size_of::<T>();
        let bo = unsafe { (drv.xrt_bo_alloc)(self.handle, size, BO_FLAGS_HOST_ONLY, 0) };
        if bo.is_null() {
            return Err(HalError::MemoryError(format!(
                "xrtBOAlloc({} bytes) returned null",
                size
            )));
        }
        let mapped = unsafe { (drv.xrt_bo_map)(bo) };
        Ok(AieBuffer {
            bo,
            mapped,
            count,
            _phantom: std::marker::PhantomData,
            _state: std::marker::PhantomData,
        })
    }

    fn alloc_from_slice<T: DeviceRepr>(
        &self,
        data: &[T],
        stream: &AieStream,
    ) -> HalResult<AieBuffer<T>> {
        let mut buf = self.alloc(data.len())?;
        <AieBuffer<T> as crate::buffer::DeviceBuffer<T>>::copy_from_host(&mut buf, data, stream)?;
        Ok(buf)
    }

    fn mem_info(&self) -> HalResult<MemInfo> {
        // The NPU shares the system's unified memory; XRT's C API exposes no
        // free/total for XDNA. Reporting a fabricated number would be worse
        // than saying so.
        Err(HalError::DeviceError(
            "mem_info is not available for XDNA via the XRT C API; \
             the NPU shares host unified memory (query the host instead)"
                .into(),
        ))
    }

    fn create_launcher(&self) -> HalResult<AieLauncher> {
        Ok(AieLauncher {
            device: self.handle,
            kernel: None,
            insts: None,
        })
    }
}

impl DynDevice for AieDevice {
    fn ordinal(&self) -> u32 {
        Device::ordinal(self)
    }

    fn name(&self) -> &str {
        Device::name(self)
    }

    fn mem_info(&self) -> HalResult<MemInfo> {
        Device::mem_info(self)
    }

    fn alloc_f32(&self, count: usize) -> HalResult<Box<dyn crate::buffer::DeviceBuffer<f32>>> {
        let buf = Device::alloc::<f32>(self, count)?;
        Ok(Box::new(buf))
    }
}

unsafe impl Send for AieDevice {}
unsafe impl Sync for AieDevice {}

impl Drop for AieDevice {
    fn drop(&mut self) {
        if let Ok(drv) = xrt_driver() {
            unsafe {
                (drv.xrt_device_close)(self.handle);
            }
        }
    }
}
