use std::fmt;

/// Unified error type for all HAL operations.
#[derive(Debug)]
pub enum HalError {
    /// Device initialization or query failed.
    DeviceError(String),
    /// Memory allocation or transfer failed.
    MemoryError(String),
    /// Kernel loading or launch failed.
    KernelError(String),
    /// Stream or synchronization failed.
    StreamError(String),
    /// The requested backend is not available.
    BackendNotAvailable(String),
    /// Generic error from an underlying runtime.
    RuntimeError(String),
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HalError::DeviceError(msg) => write!(f, "device error: {}", msg),
            HalError::MemoryError(msg) => write!(f, "memory error: {}", msg),
            HalError::KernelError(msg) => write!(f, "kernel error: {}", msg),
            HalError::StreamError(msg) => write!(f, "stream error: {}", msg),
            HalError::BackendNotAvailable(msg) => write!(f, "backend not available: {}", msg),
            HalError::RuntimeError(msg) => write!(f, "runtime error: {}", msg),
        }
    }
}

impl std::error::Error for HalError {}

pub type HalResult<T> = Result<T, HalError>;
