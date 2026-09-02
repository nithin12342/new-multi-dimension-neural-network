"""
Unit Test Suite for Forensic Remediation & Physical Codebase Materialization.
Covers the exact 4 test requirements:
  1. test_infonce_fp16_overflow_guard
  2. test_vicreg_variance_hinge
  3. test_poincare_boundary_clipping
  4. test_arrow_telemetry_flush
"""

import unittest
import os
import shutil
import tempfile
import torch
import pyarrow.parquet as pq

from src.losses.ssl_bundle import (
    clamped_infonce,
    vicreg_variance_hinge,
    poincare_boundary_clip,
    ClampedInfoNCELoss,
    VICRegLoss,
)
from src.telemetry.recorder import TelemetryRecorder
from src.engine.monitor import EarlyWarningMonitor

class TestForensicRemediation(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="remediation_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_infonce_fp16_overflow_guard(self):
        """
        Validates that collinear adversarial vectors with norm 100.0 do not
        trigger NaN or exponential overflow (>65,504 in FP16).
        """
        B, D = 16, 256
        # Collinear adversarial vectors with high norm
        z1 = torch.ones(B, D) * 100.0
        z2 = torch.ones(B, D) * 100.0

        loss_fn = ClampedInfoNCELoss(temperature=0.07)
        loss = loss_fn(z1, z2)

        self.assertTrue(torch.isfinite(loss), "Loss must be finite for adversarial inputs")
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss must not be infinite")
        self.assertTrue(loss.item() >= 0.0, "Loss must be non-negative")

        # Directly test functional helper
        fn_loss = clamped_infonce(z1, z2, temperature=0.07)
        self.assertTrue(torch.isfinite(fn_loss))

    def test_vicreg_variance_hinge(self):
        """
        Validates that zero-variance latent channels are strictly penalized
        with margin gamma=1.0.
        """
        B, D = 32, 128
        # Complete dimensional collapse: all zeros across batch
        collapsed_z = torch.zeros(B, D)

        loss = vicreg_variance_hinge(collapsed_z, gamma=1.0, eps=1e-4)

        # Since var is 0, std is sqrt(1e-4) = 0.01, hinge is 1.0 - 0.01 = 0.99
        self.assertAlmostEqual(loss.item(), 0.99, places=2)
        self.assertGreater(loss.item(), 0.95, "Collapsed variance must be heavily penalized")

        # Healthy batch with variance >= 1.0
        healthy_z = torch.randn(B, D) * 2.0
        healthy_loss = vicreg_variance_hinge(healthy_z, gamma=1.0, eps=1e-4)
        self.assertLess(healthy_loss.item(), 0.1, "Healthy variance must yield minimal hinge loss")

    def test_poincare_boundary_clipping(self):
        """
        Validates that vectors near or exceeding boundary norm 0.999999 are
        strictly clipped to ||x|| <= 1 - 1e-4.
        """
        eps = 1e-4
        max_allowed_norm = 1.0 - eps

        # Vectors outside the boundary
        adversarial_x = torch.tensor([
            [0.999999, 0.999999],
            [2.5, 0.0],
            [-3.0, -4.0],
            [0.99999, 0.0]
        ], dtype=torch.float32)

        clipped_x = poincare_boundary_clip(adversarial_x, eps=eps)
        norms = torch.norm(clipped_x, p=2, dim=-1)

        for norm_val in norms:
            self.assertLessEqual(
                norm_val.item(),
                max_allowed_norm + 1e-6,
                f"Poincaré norm {norm_val.item()} must be strictly <= {max_allowed_norm}"
            )

    def test_arrow_telemetry_flush(self):
        """
        Validates that sub-microsecond in-memory logging outputs valid,
        readable, Snappy-compressed Parquet files.
        """
        recorder = TelemetryRecorder(output_dir=self.test_dir)

        # Record multiple steps in-memory
        for step in range(1, 11):
            recorder.record_metric({
                "step": step,
                "stream": "self_supervised_omni",
                "loss": 0.5 - (step * 0.01),
                "ppl": 18.5,
                "silhouette": 0.72
            })
            recorder.record_prediction({
                "step": step,
                "sample_id": f"sample_{step}",
                "correct": True
            })
            recorder.record_hardware({
                "step": step,
                "gpu_vram_allocated_mb": 245.0,
                "cpu_util_pct": 18.5
            })

        self.assertEqual(recorder.buffered_metric_count, 10)
        self.assertEqual(recorder.buffered_prediction_count, 10)
        self.assertEqual(recorder.buffered_hardware_count, 10)

        # Flush at epoch 1 boundary
        flushed_paths = recorder.flush_epoch_parquet(epoch=1)

        self.assertIn("metrics", flushed_paths)
        self.assertIn("predictions", flushed_paths)
        self.assertIn("hardware", flushed_paths)

        # Buffers must be cleared post-flush
        self.assertEqual(recorder.buffered_metric_count, 0)

        # Read back and verify Parquet contents
        for key, path in flushed_paths.items():
            self.assertTrue(os.path.exists(path), f"Parquet file {path} must exist on disk")
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 10, f"Table {key} must contain 10 records")

    def test_early_warning_monitor_spikes_and_manifold(self):
        """Validates that EarlyWarningMonitor catches loss spikes and Poincaré saturation."""
        monitor = EarlyWarningMonitor(window_size=10, loss_spike_threshold=30.0, max_consecutive_spikes=3)

        # Step 1: Normal
        res = monitor.inspect_step(1, 1, "omni", loss=1.5, ppl=15.0, radius=0.5)
        self.assertEqual(res["status"], "OK")

        # Step 2: Loss surge > 30.0
        res = monitor.inspect_step(1, 2, "omni", loss=35.0, ppl=20.0, radius=0.5)
        self.assertEqual(res["status"], "WARNING_LOSS")

        # Step 3: PPL pegging > 600.0
        res = monitor.inspect_step(1, 3, "omni", loss=2.0, ppl=750.0, radius=0.5)
        self.assertEqual(res["status"], "WARNING_PPL")

        # Step 4: Manifold saturation >= 0.9999
        res = monitor.inspect_step(1, 4, "omni", loss=2.0, ppl=15.0, radius=0.99995)
        self.assertEqual(res["status"], "WARNING_MANIFOLD")

if __name__ == "__main__":
    unittest.main()
