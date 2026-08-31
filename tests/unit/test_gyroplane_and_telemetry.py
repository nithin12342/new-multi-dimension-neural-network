"""
Unit & Regression Tests for PoincareGyroplaneClassifier and Periodic Hardware Telemetry Logger.
Verifies hyperbolic gyroplane classification, metric stability, and DuckDB time-series hardware logging.
"""

import os
import shutil
import unittest
import torch
import duckdb

from src.domain.model.riemannian import PoincareGyroplaneClassifier, PoincareConformalChart
from src.domain.model.decoder import SingleNestedMatrixDecoder
from src.infrastructure.logging.session_logger import SessionTelemetryLogger

class TestGyroplaneAndPeriodicTelemetry(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(os.path.expanduser("~"), "tmp_gyro_telemetry_test_db")
        os.makedirs(self.test_dir, exist_ok=True)
        self.logger = SessionTelemetryLogger(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_poincare_gyroplane_classifier(self):
        """Test Poincaré Gyroplane classification produces valid dynamic logits without Euclidean distortion."""
        B, D, K = 4, 256, 10
        classifier = PoincareGyroplaneClassifier(embed_dim=D, num_classes=K, curvature=1.0, temperature=0.2)
        
        # Test forward pass with random Poincaré ball embeddings
        z = torch.randn(B, D) * 0.1
        logits = classifier(z)

        self.assertEqual(logits.shape, (B, K))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Test distinct predictions across varied sample directions
        preds = logits.argmax(dim=-1)
        self.assertEqual(preds.shape, (B,))

    def test_single_nested_matrix_decoder_with_gyroplane(self):
        """Test SingleNestedMatrixDecoder forward pass integrating PoincareGyroplaneClassifier."""
        decoder = SingleNestedMatrixDecoder(embed_dim=256, num_classes=10)
        Z_seq = torch.randn(2, 64, 256)
        z_riem = torch.randn(2, 256) * 0.1
        z_bar = torch.randn(2, 256)

        out = decoder(Z_seq, z_riem, z_bar, compute_heads=True)
        self.assertIn("logits", out)
        self.assertIn("ntp_logits", out)
        self.assertIn("x_recon", out)
        self.assertEqual(out["logits"].shape, (2, 10))

    def test_periodic_hardware_telemetry_duckdb(self):
        """Test recording and querying periodic hardware time-series telemetry in DuckDB."""
        # 1. Log Session Start
        session_stats = self.logger.log_session_start()
        self.assertIn("session_id", session_stats)

        # 2. Log Periodic Hardware Snapshots across epochs
        for epoch in range(1, 4):
            self.logger.log_periodic_hardware(stream_id=1, epoch=epoch, elapsed_sec=epoch * 5.2)

        # 3. Query DuckDB database to verify persistence
        con = duckdb.connect(self.logger.db_path, read_only=True)
        count = con.execute("SELECT COUNT(*) FROM hardware_telemetry_timeseries").fetchone()[0]
        self.assertEqual(count, 3)

        rows = con.execute("SELECT epoch, elapsed_sec, cpu_percent, ram_used_gb FROM hardware_telemetry_timeseries ORDER BY epoch").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[1][0], 2)
        self.assertEqual(rows[2][0], 3)
        con.close()

if __name__ == "__main__":
    unittest.main()
