"""
Unit Test: Pillar 5 - Arrow Telemetry Sink
Validates recording 10,000 synthetic metrics and committing to Snappy Parquet with row-count and column fidelity.
"""

import unittest
import os
import shutil
import tempfile
import pyarrow.parquet as pq

from src.telemetry.recorder import TelemetryRecorder

class TestArrowTelemetrySink(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="arrow_sink_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_10k_metrics_and_flush_parquet(self):
        recorder = TelemetryRecorder(output_dir=self.test_dir)
        num_records = 10000

        for step in range(1, num_records + 1):
            recorder.record_metric({
                "step": step,
                "stream": "omni_ssl",
                "loss": 0.5 - (step * 0.00001),
                "ppl": 15.0 + (step % 5),
                "radius": 0.45
            })

        self.assertEqual(recorder.buffered_metric_count, num_records)

        # Commit to Snappy Parquet
        flushed_paths = recorder.flush_epoch_parquet(epoch=1)
        self.assertIn("metrics", flushed_paths)

        metrics_file = flushed_paths["metrics"]
        self.assertTrue(os.path.exists(metrics_file))

        # Read back via PyArrow Parquet
        table = pq.read_table(metrics_file)
        self.assertEqual(table.num_rows, num_records)
        self.assertIn("step", table.column_names)
        self.assertIn("stream", table.column_names)
        self.assertIn("loss", table.column_names)
        self.assertIn("ppl", table.column_names)
        self.assertIn("radius", table.column_names)

        # Ensure buffer is cleared
        self.assertEqual(recorder.buffered_metric_count, 0)

if __name__ == "__main__":
    unittest.main()
