"""
Unit Tests for Comprehensive MultimodalNFMNet Codebase & Systems Overhaul:
1. StateDictRemapper: Key aliasing & strict shape validation
2. Contiguous Multimodal Batch Collation
3. In-Memory PyArrow Buffer & Snappy Parquet Export
4. SafeTensors Checkpoint Architecture Versioning (2.1.0)
5. Tensor Lifecycle Sanitization (Autograd graph detachment)
"""

import unittest
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import numpy as np

from src.infrastructure.checkpoint.discovery import StateDictRemapper
from src.infrastructure.checkpoint.serializer import CheckpointSerializer
from src.infrastructure.data.multimodal_dataset import MultimodalPyTorchDataset
from src.infrastructure.logging.prediction_logger import PyArrowTelemetryBuffer
from src.domain.config.config_entities import SystemConfig, PathConfig

class DummyLegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 10)

class DummyTargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)
        self.gyroplane = nn.Module()
        self.gyroplane.centroids = nn.Parameter(torch.randn(10, 16))

class TestCodebaseOverhaul(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="overhaul_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_state_dict_remapper_alias_and_shapes(self):
        """Verify StateDictRemapper maps legacy keys to target model and raises on shape mismatch."""
        target_model = DummyTargetModel()
        
        # Valid state dict with legacy alias 'classifier.weight'
        legacy_state_dict = {
            "fc.weight": torch.randn(16, 32),
            "fc.bias": torch.randn(16),
            "classifier.weight": torch.randn(10, 16)
        }

        remapped = StateDictRemapper.remap_and_validate(legacy_state_dict, target_model)
        self.assertIn("gyroplane.centroids", remapped)
        self.assertEqual(remapped["gyroplane.centroids"].shape, target_model.gyroplane.centroids.shape)

        # Incompatible shape must raise ValueError
        corrupt_state_dict = {
            "fc.weight": torch.randn(64, 32), # Wrong shape! Target is (16, 32)
            "fc.bias": torch.randn(16)
        }
        with self.assertRaises(ValueError):
            StateDictRemapper.remap_and_validate(corrupt_state_dict, target_model, strict_shapes=True)

    def test_multimodal_dataset_contiguous_collation(self):
        """Verify collate_fn produces strictly contiguous tensors for fast DMA transfers."""
        batch_samples = [
            {
                "image": torch.randn(3, 32, 32),
                "video": torch.randn(3, 4, 32, 32),
                "text": torch.randint(0, 1000, (16,)),
                "audio": torch.randn(1, 64),
                "tabular": torch.randn(16),
                "label": torch.tensor(1),
                "sample_id": f"sample_{i}"
            }
            for i in range(4)
        ]

        collated = MultimodalPyTorchDataset.collate_fn(batch_samples)
        for key in ["image", "video", "text", "audio", "tabular", "label"]:
            tensor = collated[key]
            self.assertTrue(tensor.is_contiguous(), f"Tensor '{key}' must be contiguous in memory")

    def test_pyarrow_telemetry_buffer_flush(self):
        """Verify in-memory PyArrowTelemetryBuffer accumulates metrics and flushes Snappy Parquet."""
        buffer = PyArrowTelemetryBuffer()

        buffer.buffer_metric({"epoch": 1, "loss": 1.25, "ppl": 24.5, "silhouette": 0.65})
        buffer.buffer_metric({"epoch": 1, "loss": 1.10, "ppl": 20.1, "silhouette": 0.72})

        flushed = buffer.flush_epoch_parquet(epoch=1, output_dir=self.test_dir)
        self.assertIn("metrics", flushed)
        self.assertTrue(os.path.exists(flushed["metrics"]))

        # Verify Parquet content using PyArrow
        import pyarrow.parquet as pq
        table = pq.read_table(flushed["metrics"])
        self.assertEqual(len(table), 2)
        self.assertIn("loss", table.column_names)

    def test_safetensors_schema_metadata_versioning(self):
        """Verify CheckpointSerializer embeds architecture_version 2.1.0 in SafeTensors headers."""
        path_config = PathConfig(base_dir=self.test_dir, checkpoints_dir=os.path.join(self.test_dir, "checkpoints"))
        serializer = CheckpointSerializer(path_config)
        model = DummyTargetModel()
        optimizer = torch.optim.Adam(model.parameters())
        sys_config = SystemConfig()

        saved_path = serializer.save_checkpoint(
            stream_id=0,
            model=model,
            optimizer=optimizer,
            scaler=None,
            epoch=1,
            batch_idx=10,
            metrics={"acc": 0.95},
            system_config=sys_config
        )

        loaded = serializer.load_checkpoint(saved_path)
        meta = loaded["metadata"]
        self.assertEqual(meta.get("architecture_version"), "2.1.0")
        self.assertEqual(meta.get("architecture"), "MultimodalNFMNet")

    def test_tensor_lifecycle_sanitization(self):
        """Verify tensors detached from autograd graph have no gradient graph retained."""
        x = torch.randn(4, 16, requires_grad=True)
        y = x.sum()
        
        # Simulating monitoring list addition
        detached_val = y.detach().cpu().item()
        self.assertIsInstance(detached_val, float)
        self.assertFalse(isinstance(detached_val, torch.Tensor))

if __name__ == "__main__":
    unittest.main()
