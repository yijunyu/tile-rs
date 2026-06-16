//! CUDA GPU backend for the HAL.
//!
//! Provides a runtime abstraction over NVIDIA CUDA using dlopen-based FFI.
//! Enabled by the `cuda` feature.

use std::ffi::{c_void, CString};
use std::path::Path;
use std::sync::OnceLock;

use crate::backend::DynDevice;
use crate::buffer::DeviceRepr;
use crate::device::{Device, MemInfo};
use crate::error::{HalError, HalResult};
use crate::kernel::{KernelLauncher, KernelMode, LaunchGrid};
use crate::stream::Stream;

// ── CUDA FFI types ──

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuStream = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;

struct CudaDriver {
    _lib: libloading::Library,
    cu_init: unsafe extern "C" fn(u32) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult,
    #[allow(dead_code)]
    cu_device_get_count: unsafe extern "C" fn(*mut i32) -> CuResult,
    cu_device_get_name: unsafe extern "C" fn(*mut u8, i32, CuDevice) -> CuResult,
    cu_ctx_create: unsafe extern "C" fn(*mut CuContext, u32, CuDevice) -> CuResult,
    cu_ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_stream_create: unsafe extern "C" fn(*mut CuStream, u32) -> CuResult,
    cu_stream_destroy: unsafe extern "C" fn(CuStream) -> CuResult,
    cu_stream_synchronize: unsafe extern "C" fn(CuStream) -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(*mut c_void) -> CuResult,
    cu_memcpy_htod: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> CuResult,
    cu_module_load: unsafe extern "C" fn(*mut CuModule, *const u8) -> CuResult,
    cu_module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const u8) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        CuFunction,
        u32, u32, u32, // grid
        u32, u32, u32, // block
        u32,           // shared mem
        CuStream,
        *mut *mut c_void, // args
        *mut *mut c_void, // extra
    ) -> CuResult,
    cu_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

static CUDA_DRIVER: OnceLock<Result<CudaDriver, String>> = OnceLock::new();

fn cuda_driver() -> HalResult<&'static CudaDriver> {
    CUDA_DRIVER
        .get_or_init(|| {
            unsafe {
                let lib = libloading::Library::new("libcuda.so")
                    .or_else(|_| libloading::Library::new("libcuda.so.1"))
                    .map_err(|e| format!("cannot load libcuda.so: {}", e))?;

                macro_rules! sym {
                    ($name:ident, $sym:expr) => {
                        let $name = *lib.get::<unsafe extern "C" fn() -> CuResult>($sym)
                            .map_err(|e| format!("symbol {} not found: {}", stringify!($name), e))?;
                        #[allow(clippy::transmute_ptr_to_ptr)]
                        let $name = std::mem::transmute($name);
                    };
                }

                sym!(cu_init, b"cuInit\0");
                sym!(cu_device_get, b"cuDeviceGet\0");
                sym!(cu_device_get_count, b"cuDeviceGetCount\0");
                sym!(cu_device_get_name, b"cuDeviceGetName\0");
                sym!(cu_ctx_create, b"cuCtxCreate_v2\0");
                sym!(cu_ctx_destroy, b"cuCtxDestroy_v2\0");
                sym!(cu_stream_create, b"cuStreamCreate\0");
                sym!(cu_stream_destroy, b"cuStreamDestroy_v2\0");
                sym!(cu_stream_synchronize, b"cuStreamSynchronize\0");
                sym!(cu_mem_alloc, b"cuMemAlloc_v2\0");
                sym!(cu_mem_free, b"cuMemFree_v2\0");
                sym!(cu_memcpy_htod, b"cuMemcpyHtoD_v2\0");
                sym!(cu_memcpy_dtoh, b"cuMemcpyDtoH_v2\0");
                sym!(cu_module_load, b"cuModuleLoad\0");
                sym!(cu_module_get_function, b"cuModuleGetFunction\0");
                sym!(cu_launch_kernel, b"cuLaunchKernel\0");
                sym!(cu_mem_get_info, b"cuMemGetInfo_v2\0");

                Ok(CudaDriver {
                    _lib: lib,
                    cu_init,
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
            }
        })
        .as_ref()
        .map_err(|e| HalError::BackendNotAvailable(e.clone()))
}

fn check_cu(ret: CuResult, op: &str) -> HalResult<()> {
    if ret == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(HalError::RuntimeError(format!("{} failed: error {}", op, ret)))
    }
}

// ── Stream ──

pub struct CudaStream {
    raw: CuStream,
}

impl CudaStream {
    fn new() -> HalResult<Self> {
        let drv = cuda_driver()?;
        let mut raw: CuStream = std::ptr::null_mut();
        check_cu(unsafe { (drv.cu_stream_create)(&mut raw, 0) }, "cuStreamCreate")?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> CuStream {
        self.raw
    }
}

impl Stream for CudaStream {
    fn synchronize(&self) -> HalResult<()> {
        let drv = cuda_driver()?;
        check_cu(unsafe { (drv.cu_stream_synchronize)(self.raw) }, "cuStreamSynchronize")
    }
}

unsafe impl Send for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let Ok(drv) = cuda_driver() {
            unsafe { (drv.cu_stream_destroy)(self.raw); }
        }
    }
}

// ── Buffer ──

pub struct CudaBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceRepr> crate::buffer::DeviceBuffer<T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        self.count
    }

    fn copy_from_host(&mut self, src: &[T], _stream: &dyn Stream) -> HalResult<()> {
        let drv = cuda_driver()?;
        let size = src.len() * std::mem::size_of::<T>();
        check_cu(
            unsafe { (drv.cu_memcpy_htod)(self.ptr, src.as_ptr() as *const c_void, size) },
            "cuMemcpyHtoD",
        )
    }

    fn copy_to_host(&self, _stream: &dyn Stream) -> HalResult<Vec<T>> {
        let drv = cuda_driver()?;
        let mut dst = vec![unsafe { std::mem::zeroed::<T>() }; self.count];
        let size = self.count * std::mem::size_of::<T>();
        check_cu(
            unsafe { (drv.cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, size) },
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

unsafe impl<T: DeviceRepr> Send for CudaBuffer<T> {}

impl<T: DeviceRepr> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if let Ok(drv) = cuda_driver() {
            unsafe { (drv.cu_mem_free)(self.ptr); }
        }
    }
}

// ── Kernel Launcher ──

pub struct CudaLauncher {
    module: Option<CuModule>,
}

unsafe impl Send for CudaLauncher {}

impl KernelLauncher for CudaLauncher {
    type Stream = CudaStream;

    fn load(&mut self, path: &Path, _mode: KernelMode) -> HalResult<()> {
        let drv = cuda_driver()?;
        let path_cstr = CString::new(path.to_str().unwrap_or("")).map_err(|_| {
            HalError::KernelError("invalid path".into())
        })?;
        let mut module: CuModule = std::ptr::null_mut();
        check_cu(
            unsafe { (drv.cu_module_load)(&mut module, path_cstr.as_ptr() as *const u8) },
            "cuModuleLoad",
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
        let drv = cuda_driver()?;
        let module = self
            .module
            .ok_or_else(|| HalError::KernelError("no module loaded".into()))?;
        let name_cstr = CString::new(name).map_err(|_| {
            HalError::KernelError("invalid kernel name".into())
        })?;
        let mut func: CuFunction = std::ptr::null_mut();
        check_cu(
            unsafe {
                (drv.cu_module_get_function)(
                    &mut func,
                    module,
                    name_cstr.as_ptr() as *const u8,
                )
            },
            "cuModuleGetFunction",
        )?;
        check_cu(
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
            "cuLaunchKernel",
        )
    }
}

// ── Device ──

pub struct CudaDevice {
    ordinal: u32,
    name: String,
    ctx: CuContext,
}

impl CudaDevice {
    pub fn new(ordinal: u32) -> HalResult<Self> {
        let drv = cuda_driver()?;
        check_cu(unsafe { (drv.cu_init)(0) }, "cuInit")?;

        let mut dev: CuDevice = 0;
        check_cu(
            unsafe { (drv.cu_device_get)(&mut dev, ordinal as i32) },
            "cuDeviceGet",
        )?;

        let mut name_buf = [0u8; 256];
        check_cu(
            unsafe { (drv.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev) },
            "cuDeviceGetName",
        )?;
        let name = unsafe {
            std::ffi::CStr::from_ptr(name_buf.as_ptr() as *const _)
                .to_string_lossy()
                .into_owned()
        };

        let mut ctx: CuContext = std::ptr::null_mut();
        check_cu(
            unsafe { (drv.cu_ctx_create)(&mut ctx, 0, dev) },
            "cuCtxCreate",
        )?;

        Ok(Self { ordinal, name, ctx })
    }
}

impl Device for CudaDevice {
    type Stream = CudaStream;
    type Buffer<T: DeviceRepr> = CudaBuffer<T>;
    type Launcher = CudaLauncher;

    fn ordinal(&self) -> u32 {
        self.ordinal
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn create_stream(&self) -> HalResult<CudaStream> {
        CudaStream::new()
    }

    fn alloc<T: DeviceRepr>(&self, count: usize) -> HalResult<CudaBuffer<T>> {
        let drv = cuda_driver()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let size = count * std::mem::size_of::<T>();
        check_cu(
            unsafe { (drv.cu_mem_alloc)(&mut ptr, size) },
            "cuMemAlloc",
        )?;
        Ok(CudaBuffer {
            ptr,
            count,
            _phantom: std::marker::PhantomData,
        })
    }

    fn alloc_from_slice<T: DeviceRepr>(
        &self,
        data: &[T],
        stream: &CudaStream,
    ) -> HalResult<CudaBuffer<T>> {
        let mut buf = self.alloc(data.len())?;
        <CudaBuffer<T> as crate::buffer::DeviceBuffer<T>>::copy_from_host(&mut buf, data, stream)?;
        Ok(buf)
    }

    fn mem_info(&self) -> HalResult<MemInfo> {
        let drv = cuda_driver()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        check_cu(
            unsafe { (drv.cu_mem_get_info)(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok(MemInfo { free, total })
    }

    fn create_launcher(&self) -> HalResult<CudaLauncher> {
        Ok(CudaLauncher { module: None })
    }
}

impl DynDevice for CudaDevice {
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

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if let Ok(drv) = cuda_driver() {
            unsafe { (drv.cu_ctx_destroy)(self.ctx); }
        }
    }
}
