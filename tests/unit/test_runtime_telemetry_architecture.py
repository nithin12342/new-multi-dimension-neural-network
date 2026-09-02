"""
Unit Test Suite for Decoupled Inline Runtime Telemetry Architecture.
Validates:
  1. In-memory Apache Arrow buffering (zero disk/database I/O during inner loop)
  2. Proactive Early Warning Terminal Monitor with instant critical abort triggers
  3. Epoch-boundary Snappy Parquet flush
  4. Vectorized offline DuckDB post-mortem analytical queries via read_parquet()
"""

import unittest
import os
import shutil
import tempfile
import pyarrow.parquet as pq

from src.telemetry.recorder import TelemetryRecorder
from src.engine.monitor import EarlyWarningMonitor
from src.telemetry.duckdb_post_mortem import DuckDBPostMortem

class TestRuntimeTelemetryArchitecture(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="runtime_telemetry_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_in_memory_arrow_buffering_zero_disk_io(self):
        """Validates that step-level appends occur strictly in host RAM without creating files."""
        recorder = TelemetryRecorder(output_dir=self.test_dir)

        # Append 50 steps rapidly
        for i in range(1, 51):
            recorder.record_metric({
                "step": i,
                "epoch": 1,
                "stream": "self_supervised_omni",
                "loss": 0.45 - (i * 0.002),
                "ppl": 15.2,
                "radius": 0.35
            })

        # Verify no files created yet in output directory
        self.assertEqual(len(os.listdir(self.test_dir)), 0, "No disk files should be written during inner loop")
        self.assertEqual(recorder.buffered_metric_count, 50, "All 50 records must reside in RAM")

    def test_inline_safety_monitor_critical_abort(self):
        """
        Validates that EarlyWarningMonitor actively intercepts catastrophic divergences
        and triggers a critical RuntimeError abort before compute is wasted.
        """
        monitor = EarlyWarningMonitor(
            loss_spike_threshold=30.0,
            ppl_stall_threshold=600.0,
            radius_boundary_threshold=0.9999,
            max_consecutive_spikes=2
        )

        # Step 1: Normal step
        res1 = monitor.inspect_step(1, 1, "vicreg", loss=1.2, ppl=12.0, radius=0.4)
        self.assertEqual(res1["status"], "OK")

        # Step 2: First divergence spike
        res2 = monitor.inspect_step(1, 2, "vicreg", loss=45.0, ppl=15.0, radius=0.4)
        self.assertEqual(res2["status"], "WARNING_LOSS")

        # Step 3: Second consecutive spike -> must raise critical abort if requested
        with self.assertRaises(RuntimeError) as ctx:
            monitor.inspect_step(1, 3, "vicreg", loss=52.0, ppl=20.0, radius=0.4, raise_on_critical=True)

        self.assertIn("CRITICAL ABORT", str(ctx.exception))

    def test_poincare_boundary_early_warning(self):
        """Validates that Poincare manifold saturation is caught immediately."""
        monitor = EarlyWarningMonitor(radius_boundary_threshold=0.9999)

        # Safe norm
        res = monitor.inspect_step(1, 1, "ntp", loss=1.5, ppl=10.0, radius=0.85)
        self.assertEqual(res["status"], "OK")

        # Saturated boundary norm
        res_sat = monitor.inspect_step(1, 2, "ntp", loss=1.5, ppl=10.0, radius=0.99998)
        self.assertEqual(res_sat["status"], "WARNING_MANIFOLD")

    def test_epoch_boundary_parquet_flush_and_duckdb_post_mortem(self):
        """
        Validates that buffered metrics flush to Snappy Parquet at epoch boundaries
        and can be queried offline via DuckDB read_parquet without locks.
        """
        recorder = TelemetryRecorder(output_dir=self.test_dir)

        # Populate two epochs of telemetry
        for ep in [1, 2]:
            for st in range(1, 21):
                recorder.record_metric({
                    "step": st,
                    "epoch": ep,
                    "stream": "omni",
                    "loss": 1.0 / (ep + st * 0.05),
                    "ppl": 20.0 - ep,
                    "radius": 0.4
                })
                recorder.record_hardware({
                    "step": st,
                    "epoch": ep,
                    "gpu_vram_allocated_mb": 450.0 + ep * 10,
                    "cpu_util_pct": 22.5
                })
            # Flush strictly at epoch boundary
            recorder.flush_epoch_parquet(epoch=ep)

        # Verify Parquet files exist on disk
        files = os.listdir(self.test_dir)
        self.assertTrue(any("telemetry_epoch_001_metrics.parquet" in f for f in files))
        self.assertTrue(any("telemetry_epoch_002_metrics.parquet" in f for f in files))

        # Perform offline post-mortem queries using DuckDB
        analyzer = DuckDBPostMortem(telemetry_dir=self.test_dir)
        trajectories = analyzer.query_macro_loss_trajectory()
        self.assertEqual(len(trajectories), 2, "Should have 2 epoch summary records")
        self.assertEqual(trajectories[0]["num_steps"], 20)

        hw_metrics = analyzer.query_hardware_efficiency()
        self.assertEqual(len(hw_metrics), 2)
        self.assertAlmostEqual(hw_metrics[0]["avg_vram_mb"], 460.0, places=1)

        anomalies = analyzer.query_anomalies_and_divergences()
        self.assertEqual(len(anomalies), 0, "No anomalies expected in clean run")

        analyzer.close()

if __name__ == "__main__":
    unittest.main()
