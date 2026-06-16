use std::ffi::c_void;
use std::path::Path;

use crate::error::HalResult;
use crate::stream::Stream;

/// Selects the execution engine for a kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KernelMode {
    /// Vector / element-wise pipeline (default).
    #[default]
    Vector,
    /// Matrix / cube engine (matmul, GEMM).
    Matrix,
    /// General-purpose (both vector and matrix).
    General,
}

/// Loads and launches compiled kernels on a device.
///
/// Each backend translates kernel names and launch parameters into the
/// appropriate runtime calls (rtKernelLaunch, cuLaunchKernel, etc.).
pub trait KernelLauncher: Send {
    /// The stream type used for ordered execution.
    type Stream: Stream;

    /// Load a kernel binary (device object file or shared library).
    fn load(&mut self, path: &Path, mode: KernelMode) -> HalResult<()>;

    /// Launch a kernel by name with the given arguments.
    ///
    /// # Safety
    /// The caller must ensure that `args` pointers are valid device pointers
    /// and that the kernel signature matches the provided arguments.
    unsafe fn launch(
        &self,
        name: &str,
        grid: LaunchGrid,
        args: &mut [*mut c_void],
        stream: &Self::Stream,
    ) -> HalResult<()>;
}

/// Kernel launch dimensions.
#[derive(Debug, Clone, Copy)]
pub struct LaunchGrid {
    /// Number of blocks (Ascend: block_dim, CUDA: gridDim).
    pub blocks: u32,
    /// Threads per block (CUDA only; ignored on Ascend).
    pub threads_per_block: u32,
    /// Shared memory bytes per block (CUDA only; ignored on Ascend).
    pub shared_mem_bytes: u32,
}

impl LaunchGrid {
    /// Create a simple 1D grid with N blocks (for Ascend-style dispatch).
    pub fn blocks(n: u32) -> Self {
        Self {
            blocks: n,
            threads_per_block: 1,
            shared_mem_bytes: 0,
        }
    }

    /// Create a CUDA-style grid with blocks and threads.
    pub fn cuda(blocks: u32, threads: u32) -> Self {
        Self {
            blocks,
            threads_per_block: threads,
            shared_mem_bytes: 0,
        }
    }

    /// Set shared memory bytes (builder pattern).
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }
}

// ── Tile Operations ──

/// Element type for tile operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileDtype {
    F32,
    F16,
    BF16,
    I8,
    I32,
}

/// Describes a tile operation for backend-agnostic kernel composition.
///
/// Each variant maps 1:1 to an `__tile_*` intrinsic. Backends translate
/// these into their native op representations at codegen time.
#[derive(Debug, Clone)]
pub enum TileOp {
    // Memory
    Load { rows: u32, cols: u32, dtype: TileDtype },
    Store { rows: u32, cols: u32, dtype: TileDtype },

    // Element-wise arithmetic
    Add { rows: u32, cols: u32 },
    Sub { rows: u32, cols: u32 },
    Mul { rows: u32, cols: u32 },
    Div { rows: u32, cols: u32 },
    Max { rows: u32, cols: u32 },
    Neg { rows: u32, cols: u32 },

    // Element-wise math
    Exp { rows: u32, cols: u32 },
    Log { rows: u32, cols: u32 },
    Rsqrt { rows: u32, cols: u32 },
    Sigmoid { rows: u32, cols: u32 },
    Silu { rows: u32, cols: u32 },

    // Scalar ops
    Scale { rows: u32, cols: u32, scalar: f32 },
    Fill { rows: u32, cols: u32, scalar: f32 },
    Clamp { rows: u32, cols: u32, lo: f32, hi: f32 },

    // Reductions
    ReduceMax { rows: u32, cols: u32 },
    ReduceSum { rows: u32, cols: u32 },
    Softmax { rows: u32, cols: u32 },
    RmsNorm { rows: u32, cols: u32, eps: f32 },

    // Matrix
    Matmul { m: u32, k: u32, n: u32 },
    Transpose { rows: u32, cols: u32 },

    // Shape
    Slice { src_rows: u32, src_cols: u32, dst_rows: u32, dst_cols: u32, row_off: u32, col_off: u32 },
    Concat { rows1: u32, cols1: u32, rows2: u32, cols2: u32 },

    // Transformer-specific
    Rope { rows: u32, cols: u32, pos: u32 },
    CausalMask { rows: u32, cols: u32 },
    Attention { b: u32, s: u32, d: u32 },
    Embedding { vocab: u32, dim: u32 },
    CrossEntropy { batch: u32, classes: u32 },

    // Quantization
    Absmax { rows: u32, cols: u32 },
    Quantize { rows: u32, cols: u32, scale: f32 },
    Dequantize { rows: u32, cols: u32, scale: f32 },

    // Type casting
    CastF32ToF16 { rows: u32, cols: u32 },
    CastF16ToF32 { rows: u32, cols: u32 },

    // Indexing
    Scatter { rows: u32, cols: u32 },
    Gather { rows: u32, cols: u32 },
    TopK { rows: u32, cols: u32, k: u32 },

    // Multi-token prediction / speculative decoding
    ArgMax { rows: u32, cols: u32 },
    SampleTopP { rows: u32, cols: u32, temperature: f32, top_p: f32 },
    DraftVerify { rows: u32, cols: u32 },
    TokenAccept { rows: u32 },
}

/// Describes a complete kernel as a sequence of tile operations.
///
/// This is the bridge between the tile API (device code) and the HAL (host code).
/// The compiler emits a `TileKernel` descriptor alongside the compiled binary,
/// enabling the HAL to:
/// 1. Select the right backend binary to load
/// 2. Configure launch parameters based on tile dimensions
/// 3. Validate buffer sizes at launch time
pub struct TileKernel {
    /// Kernel function name (matches the Rust function name).
    pub name: String,
    /// Ordered sequence of tile operations in the kernel.
    pub ops: Vec<TileOp>,
    /// Number of input buffers.
    pub num_inputs: u32,
    /// Number of output buffers.
    pub num_outputs: u32,
    /// Preferred execution mode.
    pub mode: KernelMode,
}

impl TileKernel {
    /// Check if this kernel uses matrix (cube/GEMM) operations.
    pub fn uses_matmul(&self) -> bool {
        self.ops.iter().any(|op| matches!(op, TileOp::Matmul { .. }))
    }

    /// Check if this kernel uses quantization operations.
    pub fn uses_quantization(&self) -> bool {
        self.ops.iter().any(|op| matches!(op, TileOp::Quantize { .. } | TileOp::Dequantize { .. } | TileOp::Absmax { .. }))
    }

    /// Check if this kernel uses speculative decoding operations.
    pub fn uses_speculative_decoding(&self) -> bool {
        self.ops.iter().any(|op| matches!(op,
            TileOp::DraftVerify { .. } | TileOp::TokenAccept { .. } | TileOp::SampleTopP { .. }
        ))
    }

    /// Infer the preferred kernel mode from the operations.
    pub fn inferred_mode(&self) -> KernelMode {
        if self.uses_matmul() {
            KernelMode::General
        } else {
            KernelMode::Vector
        }
    }
}
