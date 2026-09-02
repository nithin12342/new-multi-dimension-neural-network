//! End-to-End Integration Tests for crates/nfm-core Native Runtime Layer

use nfm_core::kernels::{PoincareBall, contract_tile_16x16};
use nfm_core::losses::{clamped_infonce_loss, compute_infonce_loss_from_logits, vicreg_variance_hinge};
use nfm_core::telemetry::{ArrowTelemetryBuffer, ParquetTelemetrySink, TerminalSafetyMonitor, SafetyStatus};
use nfm_core::checkpoint::CheckpointValidator;
use std::collections::HashMap;
use tempfile::NamedTempFile;

#[test]
fn test_e2e_chebyshev_tile_computation() {
    let mut tile = [0.0f32; 256];
    for i in 0..256 {
        tile[i] = ((i as f32) / 256.0) * 2.0 - 1.0;
    }

    let mut weights = [0.0f32; 3 * 256];
    for i in 0..(3 * 256) {
        weights[i] = 1.0 / 256.0;
    }

    let mut out = [0.0f32; 256];
    contract_tile_16x16(&tile, &weights, &mut out);

    // Verify all output values are finite
    for val in out {
        assert!(val.is_finite());
    }
}

#[test]
fn test_e2e_poincare_gyro_projection() {
    let ball = PoincareBall::new(1.0);
    let mut coords = [0.99999f32, 0.99999f32]; // norm > 1.0
    ball.project(&mut coords, 2);

    let norm = (coords[0] * coords[0] + coords[1] * coords[1]).sqrt();
    assert!(norm <= 1.0 - 1e-4 + 1e-6);

    let lambda = ball.lambda(&coords);
    assert!(lambda <= 1000.0);
}

#[test]
fn test_e2e_clamped_infonce_and_vicreg() {
    let mut sim = [500.0f32, -200.0f32, 10.0f32, 0.0f32];
    clamped_infonce_loss(&mut sim, 0.07);

    assert_eq!(sim[0], 10.8);
    assert_eq!(sim[1], -10.8);

    let loss = compute_infonce_loss_from_logits(&sim, 2, 0.07);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);

    let z = [0.0f32; 8];
    let vicreg_loss = vicreg_variance_hinge(&z, 4, 2, 1.0, 1e-4);
    assert!(vicreg_loss > 0.9);
}

#[test]
fn test_e2e_arrow_and_parquet_telemetry_pipeline() {
    let mut buffer = ArrowTelemetryBuffer::new();

    for i in 1..=10 {
        buffer.append_step(i, "omni_stream", 0.5 - (i as f32) * 0.01, 18.0 + (i as f32), 0.75);
    }
    assert_eq!(buffer.len(), 10);

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    ParquetTelemetrySink::flush_buffer(path, &mut buffer).unwrap();
    assert_eq!(buffer.len(), 0);
    assert!(path.exists());
    assert!(std::fs::metadata(path).unwrap().len() > 100);
}

#[test]
fn test_e2e_terminal_safety_monitor() {
    let mut monitor = TerminalSafetyMonitor::new(15.0, 2);

    assert_eq!(monitor.inspect(1, 1, "stream_1", 2.0, 15.0, 0.4), SafetyStatus::Ok);

    // First spike -> Warning
    match monitor.inspect(1, 2, "stream_1", 25.0, 45.0, 0.4) {
        SafetyStatus::Warning(msg) => assert!(msg.contains("surged")),
        _ => panic!("Expected warning on spike"),
    }

    // Second spike -> CriticalAbort
    match monitor.inspect(1, 3, "stream_1", 30.0, 60.0, 0.4) {
        SafetyStatus::CriticalAbort(msg) => assert!(msg.contains("CRITICAL ABORT")),
        _ => panic!("Expected critical abort on second spike"),
    }
}

#[test]
fn test_e2e_checkpoint_validator() {
    let mut expected = HashMap::new();
    expected.insert("decoder.gyroplane.centroids".to_string(), vec![10, 256]);

    let validator = CheckpointValidator::new(expected);
    // Verifies validator initialized and alias map loaded
    assert!(validator.validate_and_remap(&[]).is_err());
}
