use crate::error::{HalError, HalResult};

/// Identifies a hardware backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Huawei Ascend NPU (910B/910C).
    Ascend,
    /// NVIDIA GPU (via CUDA).
    Cuda,
    /// Moore Threads MTT S4000 (via MUSA — CUDA-compatible).
    Musa,
    /// AMD Ryzen AI / AIE2P (via IRON/Peano).
    Aie,
    /// AWS Trainium (via NKI).
    Nki,
    /// Vulkan/Metal (via SPIR-V).
    Spirv,
    /// Apple GPU (via Metal Shading Language).
    Msl,
    /// Cambricon MLU (via BANG-C).
    Bang,
    /// Intel Gaudi (via TPC-C).
    Gaudi,
}

impl BackendKind {
    /// Parse from the `TILERS_CODEGEN_PATH` environment variable value.
    pub fn from_codegen_path(s: &str) -> Option<Self> {
        match s {
            "cpp" | "pto" => Some(BackendKind::Ascend),
            "gpu" => Some(BackendKind::Cuda),
            "musa" => Some(BackendKind::Musa),
            "aie" => Some(BackendKind::Aie),
            "nki" => Some(BackendKind::Nki),
            "spirv" => Some(BackendKind::Spirv),
            "msl" => Some(BackendKind::Msl),
            "bang" => Some(BackendKind::Bang),
            "gaudi" => Some(BackendKind::Gaudi),
            _ => None,
        }
    }

    /// Returns the codegen path string for this backend.
    pub fn codegen_path(&self) -> &'static str {
        match self {
            BackendKind::Ascend => "cpp",
            BackendKind::Cuda => "gpu",
            BackendKind::Musa => "musa",
            BackendKind::Aie => "aie",
            BackendKind::Nki => "nki",
            BackendKind::Spirv => "spirv",
            BackendKind::Msl => "msl",
            BackendKind::Bang => "bang",
            BackendKind::Gaudi => "gaudi",
        }
    }
}

/// Selects and initializes a backend at runtime.
///
/// Reads `TILERS_CODEGEN_PATH` to determine which backend to use,
/// then creates the appropriate device.
pub struct BackendSelector {
    kind: BackendKind,
}

impl BackendSelector {
    /// Create a selector from the `TILERS_CODEGEN_PATH` environment variable.
    /// Defaults to `Ascend` if not set.
    pub fn from_env() -> HalResult<Self> {
        let path = std::env::var("TILERS_CODEGEN_PATH")
            .or_else(|_| std::env::var("ACLRS_CODEGEN_PATH"))
            .unwrap_or_else(|_| "cpp".to_string());
        let kind = BackendKind::from_codegen_path(&path).ok_or_else(|| {
            HalError::BackendNotAvailable(format!("unknown codegen path: {}", path))
        })?;
        Ok(Self { kind })
    }

    /// Create a selector for a specific backend.
    pub fn new(kind: BackendKind) -> Self {
        Self { kind }
    }

    /// Which backend is selected.
    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    /// Create a device for the selected backend.
    ///
    /// Returns a boxed trait object. For static dispatch, use the
    /// backend-specific modules directly (e.g., `tile_hal::ascend`).
    #[allow(unused_variables)]
    pub fn device(&self, ordinal: u32) -> HalResult<Box<dyn DynDevice>> {
        match self.kind {
            #[cfg(feature = "ascend")]
            BackendKind::Ascend => {
                let dev = crate::ascend::AscendDevice::new(ordinal)?;
                Ok(Box::new(dev))
            }
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => {
                let dev = crate::cuda::CudaDevice::new(ordinal)?;
                Ok(Box::new(dev))
            }
            other => Err(HalError::BackendNotAvailable(format!(
                "{:?} backend not compiled (enable the feature flag)",
                other
            ))),
        }
    }
}

/// Object-safe subset of [`Device`] for dynamic dispatch via `BackendSelector`.
///
/// For full type-level control, use the concrete device types directly.
pub trait DynDevice: Send + Sync {
    fn ordinal(&self) -> u32;
    fn name(&self) -> &str;
    fn mem_info(&self) -> HalResult<crate::device::MemInfo>;

    /// Allocate `count * size_of::<f32>()` bytes, returning a type-erased buffer.
    fn alloc_f32(&self, count: usize) -> HalResult<Box<dyn crate::buffer::DeviceBuffer<f32>>>;
}
