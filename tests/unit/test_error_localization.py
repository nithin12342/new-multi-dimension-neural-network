"""
Unit & Regression Tests for MultimodalErrorLocalizationEngine and DuckDB Error Telemetry.
Verifies all 5 modality failure coordinate detectors and prefix rollback functions.
"""

import os
import shutil
import unittest
import torch
import duckdb

from src.domain.model.error_localization import MultimodalErrorLocalizationEngine
from src.infrastructure.logging.prediction_logger import PredictionLogExporter

class TestMultimodalErrorLocalization(unittest.TestCase):

    def setUp(self):
        self.engine = MultimodalErrorLocalizationEngine(
            text_loss_threshold=3.0,
            patch_loss_threshold=0.20,
            audio_spectral_threshold=0.25
        )
        self.test_dir = os.path.join(os.path.expanduser("~"), "tmp_err_loc_test_db")
        os.makedirs(self.test_dir, exist_ok=True)
        self.exporter = PredictionLogExporter(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_text_failure_localization(self):
        """Test token-level surprisal spike detection and first erroneous step identification."""
        B, S, V = 2, 16, 1000
        ntp_logits = torch.randn(B, S, V)
        target_tokens = torch.randint(0, V, (B, S))

        # Intentionally inject high cross-entropy anomaly at token index 7 for sample 0
        ntp_logits[0, 7, target_tokens[0, 7]] = -50.0

        results = self.engine.locate_text_failure(ntp_logits, target_tokens)
        self.assertEqual(len(results), B)
        self.assertEqual(results[0]["first_error_step"], 0)
        self.assertIn(7, results[0]["all_failed_tokens"])
        self.assertGreater(results[0]["worst_token_loss"], 3.0)

    def test_visual_patch_failure_localization(self):
        """Test spatial patch grid (h*, w*) error coordinate localization on 14x14 grid."""
        B, N, D = 2, 196, 256
        x_recon = torch.zeros(B, N, D)
        target_features = torch.zeros(B, N, D)

        # Inject patch distortion at patch 32 (row 2, col 4 on 14x14 grid)
        target_features[0, 32] = 5.0

        results = self.engine.locate_visual_patch_failure(x_recon, target_features)
        self.assertEqual(len(results), B)
        self.assertEqual(results[0]["worst_patch_coord"], [2, 4])
        self.assertGreater(results[0]["worst_patch_mse"], 0.20)

    def test_video_spatiotemporal_failure_localization(self):
        """Test spatiotemporal (t*, h*, w*) failure pinpointing across frames."""
        B, C, T, H, W = 2, 3, 4, 32, 32
        video_recon = torch.zeros(B, C, T, H, W)
        video_target = torch.zeros(B, C, T, H, W)

        # Inject distortion at frame t=2, coordinate (10, 15)
        video_target[0, :, 2, 10, 15] = 10.0

        results = self.engine.locate_video_spatiotemporal_failure(video_recon, video_target)
        self.assertEqual(len(results), B)
        self.assertEqual(results[0]["worst_frame_idx"], 2)
        self.assertEqual(results[0]["worst_spatiotemporal_coord"][0], 2)
        self.assertEqual(results[0]["worst_spatiotemporal_coord"][1], 10)
        self.assertEqual(results[0]["worst_spatiotemporal_coord"][2], 15)

    def test_audio_spectral_failure_localization(self):
        """Test time-frequency bin (f*, t*) localization in Mel-spectrogram."""
        B, C, F, T = 2, 1, 64, 64
        audio_tensor = torch.zeros(B, C, F, T)

        # Inject energy concentration at frequency 25, time 40
        audio_tensor[0, 0, 25, 40] = 8.5

        results = self.engine.locate_audio_spectral_failure(audio_tensor)
        self.assertEqual(len(results), B)
        self.assertEqual(results[0]["worst_freq_bin"], 25)
        self.assertEqual(results[0]["worst_time_bin"], 40)

    def test_tabular_feature_failure_localization(self):
        """Test tabular column feature index error localization."""
        B, D_tab = 2, 15
        tab_recon = torch.zeros(B, D_tab)
        tab_target = torch.zeros(B, D_tab)

        # Inject error at column 8
        tab_target[0, 8] = 4.0

        results = self.engine.locate_tabular_feature_failure(tab_recon, tab_target)
        self.assertEqual(len(results), B)
        self.assertEqual(results[0]["worst_feature_idx"], 8)
        self.assertEqual(results[0]["worst_feature_error"], 16.0)

    def test_prefix_kv_cache_rollback(self):
        """Test key-value prefix preservation and future state truncation."""
        B, H, S, D = 2, 4, 32, 64
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)

        # Rollback to step 12
        p_keys, p_vals = self.engine.rollback_prefix_kv_cache(keys, values, rollback_step=12)
        self.assertEqual(p_keys.shape, (B, H, 12, D))
        self.assertEqual(p_vals.shape, (B, H, 12, D))
        self.assertTrue(torch.equal(p_keys, keys[:, :, :12, :]))

    def test_duckdb_error_localization_export_and_query(self):
        """Test inserting error localization records into DuckDB and verifying schema queryability."""
        records = [
            {
                "timestamp": "2026-08-31_19-10-00",
                "epoch": 300,
                "stream_id": 1,
                "sample_id": "stream1_ep300_sample0",
                "overall_status": "FAIL_TEXT",
                "text_first_error_step": 3,
                "text_error_token_idx": 22,
                "text_worst_loss": 6.84,
                "image_failed_patch_coords": [[2, 4], [3, 4]],
                "image_worst_patch_coord": [2, 4],
                "image_max_residual": 0.45,
                "audio_worst_freq_bin": 25,
                "audio_worst_time_bin": 40
            },
            {
                "timestamp": "2026-08-31_19-10-00",
                "epoch": 300,
                "stream_id": 1,
                "sample_id": "stream1_ep300_sample1",
                "overall_status": "PASS",
                "text_first_error_step": -1,
                "text_error_token_idx": 0,
                "text_worst_loss": 0.12,
                "image_failed_patch_coords": [],
                "image_worst_patch_coord": [0, 0],
                "image_max_residual": 0.01,
                "audio_worst_freq_bin": 0,
                "audio_worst_time_bin": 0
            }
        ]
        db_path = self.exporter.export_error_localization_logs(records)
        self.assertTrue(os.path.exists(db_path))

        con = duckdb.connect(db_path, read_only=True)
        count = con.execute("SELECT COUNT(*) FROM sample_error_localization").fetchone()[0]
        self.assertEqual(count, 2)

        row = con.execute("SELECT text_first_error_step, image_worst_patch_coord FROM sample_error_localization WHERE sample_id='stream1_ep300_sample0'").fetchone()
        self.assertEqual(row[0], 3)
        self.assertEqual(row[1], "[2, 4]")
        con.close()

if __name__ == "__main__":
    unittest.main()
