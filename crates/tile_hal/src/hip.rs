//! HIP GPU backend (AMD) for the HAL.
//!
//! Provides a runtime abstraction over HIP using dlopen-based FFI.
//! Enabled by the `hip` feature.

use std::ffi::{c_void, CString};
use std::path::Path;
use std::sync::OnceLock;

use crate::backend::DynDevice;
use crate::buffer::DeviceRepr;
use crate::device::{Device, MemInfo};
use crate::error::{HalError, HalResult};
use crate::kernel::{KernelLauncher, KernelMode, LaunchGrid};
use crate::stream::Stream;

// ── HIP FFI types ──

type HipResult = i32;
type HipDev = i32;
type HipCtxPtr = *mut c_void;
type HipStreamPtr = *mut c_void;
type HipModulePtr = *mut c_void;
type HipFuncPtr = *mut c_void;

const HIP_SUCCESS: HipResult = 0;

struct HipDriver {
    _lib: libloading::Library,
    cu_init: unsafe extern "C" fn(u32) -> HipResult,
    cu_set_device: unsafe extern "C" fn(i32) -> HipResult,
    cu_device_get: unsafe extern "C" fn(*mut HipDev) -> HipResult,
    #[allow(dead_code)]
    cu_device_get_count: unsafe extern "C" fn(*mut i32) -> HipResult,
    cu_device_get_name: unsafe extern "C" fn(*mut c_void, HipDev) -> HipResult,
    cu_ctx_create: unsafe extern "C" fn(*mut HipCtxPtr) -> HipResult,
    cu_ctx_destroy: unsafe extern "C" fn(HipCtxPtr) -> HipResult,
    cu_stream_create: unsafe extern "C" fn(*mut HipStreamPtr) -> HipResult,
    cu_stream_destroy: unsafe extern "C" fn(HipStreamPtr) -> HipResult,
    cu_stream_synchronize: unsafe extern "C" fn(HipStreamPtr) -> HipResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult,
    cu_mem_free: unsafe extern "C" fn(*mut c_void) -> HipResult,
    cu_memcpy_htod: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> HipResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> HipResult,
    cu_module_load: unsafe extern "C" fn(*mut HipModulePtr, *const u8) -> HipResult,
    cu_module_get_function: unsafe extern "C" fn(*mut HipFuncPtr, HipModulePtr, *const u8) -> HipResult,
    cu_launch_kernel: unsafe extern "C" fn(
        HipFuncPtr,
        u32,
        u32,
        u32, // grid
        u32,
        u32,
        u32, // block
        u32, // shared mem
        HipStreamPtr,
        *mut *mut c_void, // args
        *mut *mut c_void, // extra
    ) -> HipResult,
    cu_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> HipResult,
}

unsafe impl Send for HipDriver {}
unsafe impl Sync for HipDriver {}

static HIP_DRIVER: OnceLock<Result<HipDriver, String>> = OnceLock::new();

fn hip_driver() -> HalResult<&'static HipDriver> {
    HIP_DRIVER
        .get_or_init(|| unsafe {
            let lib = libloading::Library::new("libamdhip64.so")
                .or_else(|_| libloading::Library::new("libamdhip64.so.7"))
                .map_err(|e| format!("cannot load libamdhip64.so: {}", e))?;

            macro_rules! sym {
                ($name:ident, $sym:expr) => {
                    let $name = *lib
                        .get::<unsafe extern "C" fn() -> HipResult>($sym)
                        .map_err(|e| format!("symbol {} not found: {}", stringify!($name), e))?;
                    #[allow(clippy::transmute_ptr_to_ptr)]
                    let $name = std::mem::transmute($name);
                };
            }

            sym!(cu_init, b"hipInit\0");
            sym!(cu_set_device, b"hipSetDevice\0");
            sym!(cu_device_get, b"hipGetDevice\0");
            sym!(cu_device_get_count, b"hipGetDeviceCount\0");
            sym!(cu_device_get_name, b"hipGetDeviceProperties\0");
            sym!(cu_ctx_create, b"hipDevicePrimaryCtxRetain\0");
            sym!(cu_ctx_destroy, b"hipDevicePrimaryCtxRelease\0");
            sym!(cu_stream_create, b"hipStreamCreate\0");
            sym!(cu_stream_destroy, b"hipStreamDestroy\0");
            sym!(cu_stream_synchronize, b"hipStreamSynchronize\0");
            sym!(cu_mem_alloc, b"hipMalloc\0");
            sym!(cu_mem_free, b"hipFree\0");
            sym!(cu_memcpy_htod, b"hipMemcpyHtoD\0");
            sym!(cu_memcpy_dtoh, b"hipMemcpyDtoH\0");
            sym!(cu_module_load, b"hipModuleLoad\0");
            sym!(cu_module_get_function, b"hipModuleGetFunction\0");
            sym!(cu_launch_kernel, b"hipModuleLaunchKernel\0");
            sym!(cu_mem_get_info, b"hipMemGetInfo\0");

            Ok(HipDriver {
                _lib: lib,
                cu_init,
cu_set_device,
                cu_device_get,
                cu_device_get_count,
                cu_device_get_name,
                cu_ctx_create,
                cu_ctx_destroy,
                cu_stream_create,
                cu_stream_destroy,
                cu_stream_synchronize,
                cu_mem_alloc,
                cu_mem_free,
                cu_memcpy_htod,
                cu_memcpy_dtoh,
                cu_module_load,
                cu_module_get_function,
                cu_launch_kernel,
                cu_mem_get_info,
            })
        })
        .as_ref()
        .map_err(|e| HalError::BackendNotAvailable(e.clone()))
}

fn check_hip(ret: HipResult, op: &str) -> HalResult<()> {
    if ret == HIP_SUCCESS {
        Ok(())
    } else {
        Err(HalError::RuntimeError(format!(
            "{} failed: error {}",
            op, ret
        )))
    }
}

// ── Stream ──

pub struct HipStream {
    raw: HipStreamPtr,
}

impl HipStream {
    fn new() -> HalResult<Self> {
        let drv = hip_driver()?;
        let mut raw: HipStreamPtr = std::ptr::null_mut();
        check_hip(
            unsafe { (drv.cu_stream_create)(&mut raw) },
            "hipStreamCreate",
        )?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> HipStreamPtr {
        self.raw
    }
}

impl Stream for HipStream {
    fn synchronize(&self) -> HalResult<()> {
        let drv = hip_driver()?;
        check_hip(
            unsafe { (drv.cu_stream_synchronize)(self.raw) },
            "hipStreamSynchronize",
        )
    }
}

unsafe impl Send for HipStream {}

impl Drop for HipStream {
    fn drop(&mut self) {
        if let Ok(drv) = hip_driver() {
            unsafe {
                (drv.cu_stream_destroy)(self.raw);
            }
        }
    }
}

// ── Buffer ──

pub struct HipBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceRepr> crate::buffer::DeviceBuffer<T> for HipBuffer<T> {
    fn len(&self) -> usize {
        self.count
    }

    fn copy_from_host(&mut self, src: &[T], _stream: &dyn Stream) -> HalResult<()> {
        let drv = hip_driver()?;
        let size = src.len() * std::mem::size_of::<T>();
        check_hip(
            unsafe { (drv.cu_memcpy_htod)(self.ptr, src.as_ptr() as *const c_void, size) },
            "cuMemcpyHtoD",
        )
    }

    fn copy_to_host(&self, _stream: &dyn Stream) -> HalResult<Vec<T>> {
        let drv = hip_driver()?;
        let mut dst = vec![unsafe { std::mem::zeroed::<T>() }; self.count];
        let size = self.count * std::mem::size_of::<T>();
        check_hip(
            unsafe {
                (drv.cu_memcpy_dtoh)(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    size,
                )
            },
            "cuMemcpyDtoH",
        )?;
        Ok(dst)
    }

    fn as_raw_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }

    fn as_raw_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }
}

unsafe impl<T: DeviceRepr> Send for HipBuffer<T> {}

impl<T: DeviceRepr> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        if let Ok(drv) = hip_driver() {
            unsafe {
                (drv.cu_mem_free)(self.ptr);
            }
        }
    }
}

// ── Kernel Launcher ──

pub struct HipLauncher {
    module: Option<HipModulePtr>,
}

unsafe impl Send for HipLauncher {}

impl KernelLauncher for HipLauncher {
    type Stream = HipStream;

    fn load(&mut self, path: &Path, _mode: KernelMode) -> HalResult<()> {
        let drv = hip_driver()?;
        let path_cstr = CString::new(path.to_str().unwrap_or(""))
            .map_err(|_| HalError::KernelError("invalid path".into()))?;
        let mut module: HipModulePtr = std::ptr::null_mut();
        check_hip(
            unsafe { (drv.cu_module_load)(&mut module, path_cstr.as_ptr() as *const u8) },
            "hipModuleLoad",
        )?;
        self.module = Some(module);
        Ok(())
    }

    unsafe fn launch(
        &self,
        name: &str,
        grid: LaunchGrid,
        args: &mut [*mut c_void],
        stream: &Self::Stream,
    ) -> HalResult<()> {
        let drv = hip_driver()?;
        let module = self
            .module
            .ok_or_else(|| HalError::KernelError("no module loaded".into()))?;
        let name_cstr =
            CString::new(name).map_err(|_| HalError::KernelError("invalid kernel name".into()))?;
        let mut func: HipFuncPtr = std::ptr::null_mut();
        check_hip(
            unsafe {
                (drv.cu_module_get_function)(&mut func, module, name_cstr.as_ptr() as *const u8)
            },
            "hipModuleGetFunction",
        )?;
        check_hip(
            unsafe {
                (drv.cu_launch_kernel)(
                    func,
                    grid.blocks,
                    1,
                    1,
                    grid.threads_per_block,
                    1,
                    1,
                    grid.shared_mem_bytes,
                    stream.raw(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )
    }
}

// ── Device ──

pub struct HipDevice {
    ordinal: u32,
    name: String,
    ctx: HipCtxPtr,
}

impl HipDevice {
    pub fn new(ordinal: u32) -> HalResult<Self> {
        let drv = hip_driver()?;
        check_hip(unsafe { (drv.cu_init)(0) }, "hipInit")?;

        let mut dev: HipDev = 0;
        check_hip(
            unsafe { (drv.cu_device_get)(&mut dev) },
            "hipGetDevice",
        )?;

        let name = format!("HIP Device {dev}");

        unsafe { (drv.cu_set_device)(0) };
        let ctx: HipCtxPtr = std::ptr::null_mut();
        Ok(Self { ordinal, name, ctx })
    }
}

impl Device for HipDevice {
    type Stream = HipStream;
    type Buffer<T: DeviceRepr> = HipBuffer<T>;
    type Launcher = HipLauncher;

    fn ordinal(&self) -> u32 {
        self.ordinal
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn create_stream(&self) -> HalResult<HipStream> {
        HipStream::new()
    }

    fn alloc<T: DeviceRepr>(&self, count: usize) -> HalResult<HipBuffer<T>> {
        let drv = hip_driver()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let size = count * std::mem::size_of::<T>();
        check_hip(unsafe { (drv.cu_mem_alloc)(&mut ptr, size) }, "cuMemAlloc")?;
        Ok(HipBuffer {
            ptr,
            count,
            _phantom: std::marker::PhantomData,
        })
    }

    fn alloc_from_slice<T: DeviceRepr>(
        &self,
        data: &[T],
        stream: &HipStream,
    ) -> HalResult<HipBuffer<T>> {
        let mut buf = self.alloc(data.len())?;
        <HipBuffer<T> as crate::buffer::DeviceBuffer<T>>::copy_from_host(&mut buf, data, stream)?;
        Ok(buf)
    }

    fn mem_info(&self) -> HalResult<MemInfo> {
        let drv = hip_driver()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        check_hip(
            unsafe { (drv.cu_mem_get_info)(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok(MemInfo { free, total })
    }

    fn create_launcher(&self) -> HalResult<HipLauncher> {
        Ok(HipLauncher { module: None })
    }
}

impl DynDevice for HipDevice {
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

unsafe impl Send for HipDevice {}
unsafe impl Sync for HipDevice {}

impl Drop for HipDevice {
    fn drop(&mut self) {
        if let Ok(drv) = hip_driver() {
            unsafe {

            }
        }
    }
}
