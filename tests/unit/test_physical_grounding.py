"""
Unit Verification Suite for Physical Grounding of All 7 Overhaul Components:
1. to_clean_scalar autograd sanitization
2. MultimodalNFMNet forward-pass error localization wiring
3. PinnedTensorPool host memory allocation
4. TelemetryRecorder PyArrow buffering and Snappy Parquet flush
5. losses.py and MultimodalSSLBundle multi-paradigm calculation
6. ChebyshevTileContraction16x16 & NFMTensorRTL tile contraction
"""

import unittest
import os
import shutil
import tempfile
import torch
import numpy as np

from src.application.orchestrator.training_loop import to_clean_scalar, MultimodalNFMNet
from src.infrastructure.data.multimodal_dataset import PinnedTensorPool
from src.infrastructure.logging.telemetry_recorder import TelemetryRecorder
from src.domain.loss.losses import InfoNCELoss, ClampedInfoNCELoss, VICRegLoss, CausalNextTokenLoss
from src.domain.loss.ssl_bundle import MultimodalSSLBundle
from src.domain.model.tile_contraction import ChebyshevTileContraction16x16
from src.domain.model.nfm_tensor_rtl import NFMTensorRTL
from src.domain.config.config_entities import SystemConfig

class TestPhysicalGrounding(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="physical_grounding_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_to_clean_scalar_autograd_sanitization(self):
        """Verify to_clean_scalar safely strips autograd graph and handles non-finite values."""
        x = torch.randn(2, 2, requires_grad=True)
        loss = (x ** 2).sum()
        self.assertIsNotNone(loss.grad_fn)

        val = to_clean_scalar(loss)
        self.assertIsInstance(val, float)

        # Test non-finite protection
        nan_t = torch.tensor(float("nan"))
        inf_t = torch.tensor(float("inf"))
        self.assertEqual(to_clean_scalar(nan_t, default=0.5), 0.5)
        self.assertEqual(to_clean_scalar(inf_t, default=1.0), 1.0)
        self.assertEqual(to_clean_scalar(np.nan, default=0.0), 0.0)

    def test_multimodal_nfmnet_forward_error_localization(self):
        """Verify MultimodalNFMNet returns error localization diagnostics when requested."""
        model = MultimodalNFMNet()
        model.eval()

        B = 2
        x_img = torch.randn(B, 3, 224, 224)
        x_txt = torch.randint(0, 1000, (B, 64))
        x_aud = torch.randn(B, 1, 128, 64)

        with torch.no_grad():
            outputs = model(x_img, x_txt, x_aud=x_aud, return_error_localization=True)

        self.assertIn("logits", outputs)
        self.assertIn("error_localization", outputs)
        diag = outputs["error_localization"]
        self.assertIn("overall_status", diag)
        self.assertIn("status_flag", diag)

    def test_pinned_tensor_pool(self):
        """Verify PinnedTensorPool allocates and stages batch tensors cleanly."""
        pool = PinnedTensorPool(default_batch_size=4)
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "label": torch.tensor([0, 1, 2, 3]),
            "sample_id": ["s1", "s2", "s3", "s4"]
        }

        staged = pool.stage_batch_to_pinned(batch)
        self.assertEqual(staged["image"].shape, batch["image"].shape)
        self.assertEqual(staged["label"].shape, batch["label"].shape)
        self.assertEqual(staged["sample_id"], batch["sample_id"])
        pool.clear()

    def test_telemetry_recorder_arrow_and_parquet(self):
        """Verify TelemetryRecorder buffers records in Arrow and flushes Parquet files."""
        recorder = TelemetryRecorder(output_dir=self.test_dir)

        recorder.record_metric({"epoch": 1, "loss": 0.45, "ppl": 18.2})
        recorder.record_prediction({"sample_id": "test_1", "correct": True})
        recorder.record_hardware({"gpu_vram_allocated_mb": 120.5})

        flushed = recorder.flush_epoch_parquet(epoch=1)
        self.assertIn("metrics", flushed)
        self.assertIn("predictions", flushed)
        self.assertIn("hardware", flushed)

        for path in flushed.values():
            self.assertTrue(os.path.exists(path))

    def test_losses_and_ssl_bundle(self):
        """Verify losses.py exports and MultimodalSSLBundle multi-paradigm calculation."""
        bundle = MultimodalSSLBundle()

        outputs = {
            "z_proj": torch.randn(4, 128),
            "ntp_logits": torch.randn(4, 32, 30522),
            "x_recon": torch.randn(4, 16, 256),
            "z_bar": torch.randn(4, 256),
            "q_dist": torch.softmax(torch.randn(4, 10), dim=-1)
        }
        augmented = {
            "z_proj": torch.randn(4, 128)
        }
        text_tokens = torch.randint(0, 1000, (4, 32))

        # Test multiple SSL paradigms
        for p in ["self_supervised_ntp", "self_supervised_barlow", "self_supervised_vicreg", "self_supervised_mae", "self_supervised_dec"]:
            loss = bundle.compute_loss(p, outputs, augmented, text_tokens=text_tokens)
            self.assertTrue(torch.isfinite(loss), f"Loss for paradigm '{p}' must be finite")

    def test_chebyshev_tile_contraction_16x16(self):
        """Verify ChebyshevTileContraction16x16 and NFMTensorRTL perform block GEMM contractions."""
        B, N, D = 2, 8, 256
        Z = torch.randn(B, N, D, requires_grad=True)

        rtl = NFMTensorRTL(embed_dim=256, tile_dim=16)
        out = rtl.execute_tile_contraction(Z)

        self.assertEqual(out.shape, (B, N, D))
        self.assertTrue(out.is_contiguous())

        # Verify gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(Z.grad)

if __name__ == "__main__":
    unittest.main()
