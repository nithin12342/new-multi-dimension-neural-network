//! Native Zero-Overhead Rust Runtime Layer for MultimodalNFMNet (nfm-core)
//!
//! Realizes the core mathematical invariants, low-level tile contractions,
//! anti-collapse losses, and lock-free Arrow telemetry buffering in native code.

pub mod kernels;
pub mod losses;
pub mod telemetry;
pub mod checkpoint;
pub mod localization;
pub mod engine;

// Re-exports
pub use kernels::{PoincareBall, PoincareManifold, eval_chebyshev_order2_tile, evaluate_chebyshev_tile, contract_tile_16x16};
pub use losses::{clamp_contrastive_logits, clamped_infonce_loss, compute_infonce_loss_from_logits, vicreg_variance_hinge};
pub use telemetry::{ArrowTelemetryBuffer, ParquetTelemetrySink, ArrowTelemetrySink, TerminalSafetyMonitor, SafetyStatus};
pub use checkpoint::{CheckpointValidator, StateDictRemapper};
pub use localization::DualStageLocalizer;
pub use engine::PinnedBufferPool;


// =========================================================================
// Zero-Overhead C-ABI Foreign Function Interface (FFI) for Python / C interop
// =========================================================================

/// C-ABI entrypoint: Evaluate Order-2 Chebyshev basis expansion
#[no_mangle]
pub unsafe extern "C" fn nfm_chebyshev_order2(src: *const f32, len: usize, dst: *mut f32) -> i32 {
    if src.is_null() || dst.is_null() {
        return -1;
    }
    let src_slice = std::slice::from_raw_parts(src, len);
    let dst_slice = std::slice::from_raw_parts_mut(dst, len * 3);
    eval_chebyshev_order2_tile(src_slice, dst_slice);
    0
}

/// C-ABI entrypoint: Poincaré radius clipping ||x|| <= 1 - eps
#[no_mangle]
pub unsafe extern "C" fn nfm_poincare_project(data: *mut f32, total_elements: usize, dim: usize, eps: f32) -> i32 {
    if data.is_null() || dim == 0 || total_elements % dim != 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(data, total_elements);
    let ball = PoincareBall { c: 1.0, eps };
    ball.project(slice, dim);
    0
}

/// C-ABI entrypoint: Clamped InfoNCE similarity matrix scaling
#[no_mangle]
pub unsafe extern "C" fn nfm_clamped_infonce(sim_matrix: *mut f32, count: usize, tau: f32) -> i32 {
    if sim_matrix.is_null() || count == 0 || tau <= 0.0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(sim_matrix, count);
    clamped_infonce_loss(slice, tau);
    0
}
