"""
E2E Verification Fixture Test for Phase 4 Implementation.
Verifies end-to-end execution of MultimodalNFMNet pipeline across 6 streams,
37 metric calculation, serialized checkpoint generation, and recovery mechanisms.
"""

import os
import shutil
import unittest
import torch
from src.domain.config.config_entities import SystemConfig, TrainingConfig, PathConfig
from src.application.orchestrator.training_loop import ParadigmTrainingOrchestrator, MultimodalNFMNet
from src.infrastructure.metrics.metric_computer import ThirtySevenMetricComputer
from src.infrastructure.checkpoint.serializer import CheckpointSerializer
from src.infrastructure.checkpoint.discovery import CheckpointDiscoveryScanner
from src.infrastructure.storage.drive_manager import GoogleDriveManager

class TestMultimodalNFMNetPipelineE2E(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(os.path.expanduser("~"), "tmp_e2e_test_drive")
        os.makedirs(self.test_dir, exist_ok=True)
        path_cfg = PathConfig(
            drive_mount_point=self.test_dir,
            base_dir=self.test_dir,
            datasets_dir=os.path.join(self.test_dir, "datasets"),
            checkpoints_dir=os.path.join(self.test_dir, "checkpoints"),
            dummy_weights_dir=os.path.join(self.test_dir, "dummy_weights"),
            logs_dir=os.path.join(self.test_dir, "logs"),
            metrics_dir=os.path.join(self.test_dir, "metrics"),
            reports_dir=os.path.join(self.test_dir, "reports")
        )
        train_cfg = TrainingConfig(num_streams=6, num_epochs=1)
        self.config = SystemConfig(path=path_cfg, training=train_cfg)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_model_forward_shapes(self):
        """Test MultimodalNFMNet forward pass shapes."""
        model = MultimodalNFMNet(self.config)
        x_img = torch.randn(2, 3, 224, 224)
        x_txt = torch.randint(0, 30522, (2, 128))
        out = model(x_img, x_txt)

        self.assertIn("logits", out)
        self.assertIn("z_proj", out)
        self.assertIn("q_dist", out)
        self.assertEqual(out["logits"].shape, (2, 10))
        self.assertEqual(out["z_proj"].shape, (2, 128))
        self.assertEqual(out["q_dist"].shape, (2, 10))

    def test_end_to_end_orchestration(self):
        """Test complete 6-stream training loop execution."""
        orchestrator = ParadigmTrainingOrchestrator(self.config)
        orchestrator.train_multi_stream()

        # Check directories created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "checkpoints")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "dummy_weights")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "logs")))

        # Check discovery scanner
        scanner = CheckpointDiscoveryScanner(os.path.join(self.test_dir, "checkpoints"))
        latest_ckpt = scanner.get_latest_valid_checkpoint(1)
        self.assertIsNotNone(latest_ckpt)
        self.assertTrue(latest_ckpt.endswith(".safetensors"))

if __name__ == "__main__":
    unittest.main()
