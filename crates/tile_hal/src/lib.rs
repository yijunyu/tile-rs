//! # tile_hal — Hardware Abstraction Layer
//!
//! Provides backend-agnostic traits for accelerator programming, enabling
//! the same host code to target Ascend NPU, NVIDIA GPU, AMD AIE, and other
//! backends supported by the tile-rs compiler.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use tile_hal::prelude::*;
//!
//! let backend = BackendSelector::from_env()?; // reads TILERS_CODEGEN_PATH
//! let device = backend.device(0)?;
//! let stream = device.create_stream()?;
//! let buf = device.alloc::<f32>(1024)?;
//! buf.copy_from_host(&host_data, &stream)?;
//! stream.synchronize()?;
//! ```

pub mod backend;
pub mod buffer;
pub mod device;
pub mod error;
pub mod kernel;
pub mod stream;

#[cfg(feature = "ascend")]
pub mod ascend;

#[cfg(feature = "cuda")]
pub mod cuda;

pub mod prelude {
    pub use crate::backend::*;
    pub use crate::buffer::*;
    pub use crate::device::*;
    pub use crate::error::*;
    pub use crate::kernel::*;
    pub use crate::stream::*;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use std::sync::Mutex;

    /// Serializes the tests that mutate the process-wide `TILERS_CODEGEN_PATH`
    /// env var, so cargo's parallel test runner can't interleave them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── BackendKind tests ──

    #[test]
    fn test_backend_kind_from_codegen_path() {
        assert_eq!(
            BackendKind::from_codegen_path("cpp"),
            Some(BackendKind::Ascend)
        );
        assert_eq!(
            BackendKind::from_codegen_path("pto"),
            Some(BackendKind::Ascend)
        );
        assert_eq!(
            BackendKind::from_codegen_path("gpu"),
            Some(BackendKind::Cuda)
        );
        assert_eq!(
            BackendKind::from_codegen_path("musa"),
            Some(BackendKind::Musa)
        );
        assert_eq!(
            BackendKind::from_codegen_path("aie"),
            Some(BackendKind::Aie)
        );
        assert_eq!(
            BackendKind::from_codegen_path("nki"),
            Some(BackendKind::Nki)
        );
        assert_eq!(
            BackendKind::from_codegen_path("spirv"),
            Some(BackendKind::Spirv)
        );
        assert_eq!(
            BackendKind::from_codegen_path("msl"),
            Some(BackendKind::Msl)
        );
        assert_eq!(
            BackendKind::from_codegen_path("bang"),
            Some(BackendKind::Bang)
        );
        assert_eq!(
            BackendKind::from_codegen_path("gaudi"),
            Some(BackendKind::Gaudi)
        );
        assert_eq!(BackendKind::from_codegen_path("unknown"), None);
    }

    #[test]
    fn test_backend_kind_roundtrip() {
        let kinds = [
            BackendKind::Ascend,
            BackendKind::Cuda,
            BackendKind::Musa,
            BackendKind::Aie,
            BackendKind::Nki,
            BackendKind::Spirv,
            BackendKind::Msl,
            BackendKind::Bang,
            BackendKind::Gaudi,
        ];
        for kind in &kinds {
            let path = kind.codegen_path();
            let roundtrip = BackendKind::from_codegen_path(path).unwrap();
            assert_eq!(*kind, roundtrip);
        }
    }

    #[test]
    fn test_backend_selector_default() {
        // With no env var set, defaults to Ascend
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("TILERS_CODEGEN_PATH");
        let sel = BackendSelector::from_env().unwrap();
        assert_eq!(sel.kind(), BackendKind::Ascend);
    }

    #[test]
    fn test_backend_selector_from_env_unknown_path_errors() {
        // The error arm of from_env: an unrecognized TILERS_CODEGEN_PATH yields
        // BackendNotAvailable. Uses a value no BackendKind maps to. (Env-var
        // mutation is serialized with the other from_env test below.)
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var("TILERS_CODEGEN_PATH", "definitely-not-a-backend");
        let result = BackendSelector::from_env();
        std::env::remove_var("TILERS_CODEGEN_PATH");
        match result {
            Err(HalError::BackendNotAvailable(msg)) => {
                assert!(msg.contains("definitely-not-a-backend"), "msg was: {msg}");
            }
            Err(e) => panic!("expected BackendNotAvailable, got error: {e}"),
            Ok(_) => panic!("expected BackendNotAvailable, got Ok"),
        }
    }

    #[test]
    fn test_backend_selector_unavailable() {
        // Without feature flags, requesting a device should return BackendNotAvailable
        let sel = BackendSelector::new(BackendKind::Bang);
        let result = sel.device(0);
        match result {
            Err(HalError::BackendNotAvailable(_)) => {} // expected
            Err(e) => panic!("unexpected error: {}", e),
            Ok(_) => panic!("expected BackendNotAvailable"),
        }
    }

    // ── LaunchGrid tests ──

    #[test]
    fn test_launch_grid_blocks() {
        let grid = LaunchGrid::blocks(8);
        assert_eq!(grid.blocks, 8);
        assert_eq!(grid.threads_per_block, 1);
        assert_eq!(grid.shared_mem_bytes, 0);
    }

    #[test]
    fn test_launch_grid_cuda() {
        let grid = LaunchGrid::cuda(16, 256).with_shared_mem(4096);
        assert_eq!(grid.blocks, 16);
        assert_eq!(grid.threads_per_block, 256);
        assert_eq!(grid.shared_mem_bytes, 4096);
    }

    // ── TileKernel tests ──

    #[test]
    fn test_tile_kernel_vec_add() {
        let kernel = TileKernel {
            name: "vec_add".to_string(),
            ops: vec![
                TileOp::Load {
                    rows: 1,
                    cols: 1024,
                    dtype: TileDtype::F32,
                },
                TileOp::Load {
                    rows: 1,
                    cols: 1024,
                    dtype: TileDtype::F32,
                },
                TileOp::Add {
                    rows: 1,
                    cols: 1024,
                },
                TileOp::Store {
                    rows: 1,
                    cols: 1024,
                    dtype: TileDtype::F32,
                },
            ],
            num_inputs: 2,
            num_outputs: 1,
            mode: KernelMode::Vector,
        };
        assert!(!kernel.uses_matmul());
        assert!(!kernel.uses_quantization());
        assert_eq!(kernel.inferred_mode(), KernelMode::Vector);
    }

    #[test]
    fn test_tile_kernel_matmul() {
        let kernel = TileKernel {
            name: "gemm".to_string(),
            ops: vec![
                TileOp::Load {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
                TileOp::Load {
                    rows: 64,
                    cols: 32,
                    dtype: TileDtype::F32,
                },
                TileOp::Matmul {
                    m: 32,
                    k: 64,
                    n: 32,
                },
                TileOp::Store {
                    rows: 32,
                    cols: 32,
                    dtype: TileDtype::F32,
                },
            ],
            num_inputs: 2,
            num_outputs: 1,
            mode: KernelMode::Matrix,
        };
        assert!(kernel.uses_matmul());
        assert_eq!(kernel.inferred_mode(), KernelMode::General);
    }

    #[test]
    fn test_tile_kernel_quantize_pipeline() {
        let kernel = TileKernel {
            name: "quantize_weights".to_string(),
            ops: vec![
                TileOp::Load {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
                TileOp::Absmax { rows: 32, cols: 64 },
                TileOp::Quantize {
                    rows: 32,
                    cols: 64,
                    scale: 0.0,
                },
                TileOp::Store {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::I8,
                },
            ],
            num_inputs: 1,
            num_outputs: 1,
            mode: KernelMode::Vector,
        };
        assert!(kernel.uses_quantization());
        assert!(!kernel.uses_matmul());
    }

    #[test]
    fn test_tile_kernel_flash_attention() {
        let kernel = TileKernel {
            name: "flash_attention".to_string(),
            ops: vec![
                TileOp::Load {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
                TileOp::Load {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
                TileOp::Load {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
                TileOp::Transpose { rows: 32, cols: 64 },
                TileOp::Matmul {
                    m: 32,
                    k: 64,
                    n: 32,
                },
                TileOp::Scale {
                    rows: 32,
                    cols: 32,
                    scalar: 0.125,
                },
                TileOp::ReduceMax { rows: 32, cols: 32 },
                TileOp::Exp { rows: 32, cols: 32 },
                TileOp::Softmax { rows: 32, cols: 32 },
                TileOp::Matmul {
                    m: 32,
                    k: 32,
                    n: 64,
                },
                TileOp::Store {
                    rows: 32,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
            ],
            num_inputs: 3,
            num_outputs: 1,
            mode: KernelMode::General,
        };
        assert!(kernel.uses_matmul());
        assert!(!kernel.uses_quantization());
        assert_eq!(kernel.inferred_mode(), KernelMode::General);
    }

    // ── MemInfo tests ──

    #[test]
    fn test_mem_info_used() {
        let info = MemInfo {
            free: 1024,
            total: 4096,
        };
        assert_eq!(info.used(), 3072);
    }

    // ── Error tests ──

    #[test]
    fn test_hal_error_display() {
        let err = HalError::DeviceError("test".to_string());
        assert_eq!(format!("{}", err), "device error: test");

        let err = HalError::BackendNotAvailable("cuda".to_string());
        assert_eq!(format!("{}", err), "backend not available: cuda");
    }

    // ── DeviceRepr tests ──

    #[test]
    fn test_f16_newtype() {
        let val = f16(0x3C00); // 1.0 in IEEE 754 half
        assert_eq!(val.0, 0x3C00);
    }

    // ── TileOp coverage ──

    #[test]
    fn test_tile_op_all_variants() {
        // Ensure all TileOp variants can be constructed (compile-time check)
        let ops: Vec<TileOp> = vec![
            TileOp::Load {
                rows: 1,
                cols: 1,
                dtype: TileDtype::F32,
            },
            TileOp::Store {
                rows: 1,
                cols: 1,
                dtype: TileDtype::F32,
            },
            TileOp::Add { rows: 1, cols: 1 },
            TileOp::Sub { rows: 1, cols: 1 },
            TileOp::Mul { rows: 1, cols: 1 },
            TileOp::Div { rows: 1, cols: 1 },
            TileOp::Max { rows: 1, cols: 1 },
            TileOp::Neg { rows: 1, cols: 1 },
            TileOp::Exp { rows: 1, cols: 1 },
            TileOp::Log { rows: 1, cols: 1 },
            TileOp::Rsqrt { rows: 1, cols: 1 },
            TileOp::Sigmoid { rows: 1, cols: 1 },
            TileOp::Silu { rows: 1, cols: 1 },
            TileOp::Scale {
                rows: 1,
                cols: 1,
                scalar: 1.0,
            },
            TileOp::Fill {
                rows: 1,
                cols: 1,
                scalar: 0.0,
            },
            TileOp::Clamp {
                rows: 1,
                cols: 1,
                lo: -1.0,
                hi: 1.0,
            },
            TileOp::ReduceMax { rows: 1, cols: 1 },
            TileOp::ReduceSum { rows: 1, cols: 1 },
            TileOp::Softmax { rows: 1, cols: 1 },
            TileOp::RmsNorm {
                rows: 1,
                cols: 1,
                eps: 1e-6,
            },
            TileOp::Matmul { m: 1, k: 1, n: 1 },
            TileOp::Transpose { rows: 1, cols: 1 },
            TileOp::Slice {
                src_rows: 2,
                src_cols: 2,
                dst_rows: 1,
                dst_cols: 1,
                row_off: 0,
                col_off: 0,
            },
            TileOp::Concat {
                rows1: 1,
                cols1: 1,
                rows2: 1,
                cols2: 1,
            },
            TileOp::Rope {
                rows: 1,
                cols: 1,
                pos: 0,
            },
            TileOp::CausalMask { rows: 1, cols: 1 },
            TileOp::Attention { b: 1, s: 1, d: 1 },
            TileOp::Embedding {
                vocab: 100,
                dim: 64,
            },
            TileOp::CrossEntropy {
                batch: 1,
                classes: 10,
            },
            TileOp::Absmax { rows: 1, cols: 1 },
            TileOp::Quantize {
                rows: 1,
                cols: 1,
                scale: 1.0,
            },
            TileOp::Dequantize {
                rows: 1,
                cols: 1,
                scale: 1.0,
            },
            TileOp::CastF32ToF16 { rows: 1, cols: 1 },
            TileOp::CastF16ToF32 { rows: 1, cols: 1 },
            TileOp::Scatter { rows: 1, cols: 1 },
            TileOp::Gather { rows: 1, cols: 1 },
            TileOp::TopK {
                rows: 1,
                cols: 1,
                k: 5,
            },
            TileOp::ArgMax { rows: 1, cols: 1 },
            TileOp::SampleTopP {
                rows: 1,
                cols: 1,
                temperature: 0.7,
                top_p: 0.9,
            },
            TileOp::DraftVerify { rows: 1, cols: 1 },
            TileOp::TokenAccept { rows: 1 },
        ];
        assert_eq!(ops.len(), 41);
    }

    // ── KernelMode tests ──

    #[test]
    fn test_kernel_mode_default() {
        let mode: KernelMode = Default::default();
        assert_eq!(mode, KernelMode::Vector);
    }

    // ── Speculative decoding TileKernel ──

    #[test]
    fn test_tile_kernel_speculative_decode() {
        let kernel = TileKernel {
            name: "speculative_decode".to_string(),
            ops: vec![
                TileOp::Load {
                    rows: 4,
                    cols: 1,
                    dtype: TileDtype::I32,
                }, // draft tokens
                TileOp::Load {
                    rows: 4,
                    cols: 256,
                    dtype: TileDtype::F32,
                }, // target logits
                TileOp::DraftVerify { rows: 4, cols: 256 },
                TileOp::ArgMax { rows: 4, cols: 256 },
                TileOp::TokenAccept { rows: 4 },
                TileOp::Store {
                    rows: 4,
                    cols: 1,
                    dtype: TileDtype::I32,
                },
            ],
            num_inputs: 2,
            num_outputs: 1,
            mode: KernelMode::Vector,
        };
        assert!(kernel.uses_speculative_decoding());
        assert!(!kernel.uses_matmul());
        assert!(!kernel.uses_quantization());
    }

    #[test]
    fn test_tile_kernel_mtp_draft_head() {
        let kernel = TileKernel {
            name: "mtp_draft_head".to_string(),
            ops: vec![
                TileOp::Load {
                    rows: 1,
                    cols: 64,
                    dtype: TileDtype::F32,
                },
                TileOp::Load {
                    rows: 64,
                    cols: 256,
                    dtype: TileDtype::F32,
                },
                TileOp::Matmul {
                    m: 1,
                    k: 64,
                    n: 256,
                },
                TileOp::SampleTopP {
                    rows: 1,
                    cols: 256,
                    temperature: 0.7,
                    top_p: 0.9,
                },
                TileOp::Store {
                    rows: 1,
                    cols: 1,
                    dtype: TileDtype::I32,
                },
            ],
            num_inputs: 2,
            num_outputs: 1,
            mode: KernelMode::General,
        };
        assert!(kernel.uses_speculative_decoding());
        assert!(kernel.uses_matmul());
        assert_eq!(kernel.inferred_mode(), KernelMode::General);
    }

    // ── HalError: every Display arm + the std::error::Error impl ──

    #[test]
    fn test_hal_error_display_all_arms() {
        let cases: [(HalError, &str); 6] = [
            (HalError::DeviceError("d".into()), "device error: d"),
            (HalError::MemoryError("m".into()), "memory error: m"),
            (HalError::KernelError("k".into()), "kernel error: k"),
            (HalError::StreamError("s".into()), "stream error: s"),
            (
                HalError::BackendNotAvailable("b".into()),
                "backend not available: b",
            ),
            (HalError::RuntimeError("r".into()), "runtime error: r"),
        ];
        for (err, want) in cases {
            assert_eq!(format!("{err}"), want);
            // Debug is derived; exercise it so the variant's debug arm is covered.
            assert!(!format!("{err:?}").is_empty());
        }
    }

    #[test]
    fn test_hal_error_is_std_error() {
        // Exercise the `impl std::error::Error` path: usable as a trait object
        // and (default) source() returns None.
        let err = HalError::RuntimeError("boom".into());
        let dyn_err: &dyn std::error::Error = &err;
        assert!(dyn_err.source().is_none());
        // HalResult<T> propagation through `?` over the Error trait.
        fn fails() -> HalResult<u32> {
            Err(HalError::MemoryError("oom".into()))
        }
        assert!(fails().is_err());
    }

    // ── DeviceBuffer default method + DeviceRepr marker + f16 newtype ──

    /// Minimal in-memory DeviceBuffer impl to exercise the trait's default
    /// `is_empty()` (which dispatches to `len()`), with no device present.
    struct MockBuf {
        data: Vec<f32>,
    }
    impl DeviceBuffer<f32> for MockBuf {
        fn len(&self) -> usize {
            self.data.len()
        }
        fn copy_from_host(&mut self, src: &[f32], _s: &dyn Stream) -> HalResult<()> {
            self.data = src.to_vec();
            Ok(())
        }
        fn copy_to_host(&self, _s: &dyn Stream) -> HalResult<Vec<f32>> {
            Ok(self.data.clone())
        }
        fn as_raw_ptr(&self) -> *const std::ffi::c_void {
            self.data.as_ptr() as *const std::ffi::c_void
        }
        fn as_raw_mut_ptr(&mut self) -> *mut std::ffi::c_void {
            self.data.as_mut_ptr() as *mut std::ffi::c_void
        }
    }

    struct NoopStream;
    impl Stream for NoopStream {
        fn synchronize(&self) -> HalResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_device_buffer_default_is_empty_and_roundtrip() {
        let mut buf = MockBuf { data: vec![] };
        assert_eq!(buf.len(), 0);
        assert!(
            buf.is_empty(),
            "empty buffer must report is_empty via default"
        );

        let stream = NoopStream;
        buf.copy_from_host(&[1.0, 2.0, 3.0], &stream).unwrap();
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        assert_eq!(buf.copy_to_host(&stream).unwrap(), vec![1.0, 2.0, 3.0]);
        // Raw pointers are non-null for a populated buffer.
        assert!(!buf.as_raw_ptr().is_null());
        assert!(!buf.as_raw_mut_ptr().is_null());
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_f16_newtype_repr_and_clone() {
        let one = f16(0x3C00); // 1.0 in IEEE-754 half
        let copy = one; // Copy
        let cloned = one.clone();
        assert_eq!(one.0, 0x3C00);
        assert_eq!(copy.0, cloned.0);
        // Debug derive is exercised.
        assert!(format!("{one:?}").contains("3C00") || format!("{one:?}").contains("15360"));
    }

    /// `DeviceRepr` is a `Copy + Send + Sync + 'static` marker; assert the
    /// primitive impls and the `f16` newtype satisfy it (compile-time bound).
    #[test]
    fn test_device_repr_marker_impls() {
        fn assert_repr<T: DeviceRepr>() {}
        assert_repr::<f32>();
        assert_repr::<f64>();
        assert_repr::<f16>();
        assert_repr::<u8>();
        assert_repr::<u16>();
        assert_repr::<u32>();
        assert_repr::<u64>();
        assert_repr::<i8>();
        assert_repr::<i16>();
        assert_repr::<i32>();
        assert_repr::<i64>();
    }
}
