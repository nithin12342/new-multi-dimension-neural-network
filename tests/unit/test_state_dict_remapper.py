"""
Unit Test: Pillar 3 - State Dict Remapper
Validates alias resolution and strict shape mismatch rejection.
"""

import unittest
import torch
import torch.nn as nn

from src.infrastructure.checkpoint.discovery import StateDictRemapper

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(128, 64)
        self.gyroplane = nn.Module()
        self.gyroplane.centroids = nn.Parameter(torch.randn(10, 64))

class TestStateDictRemapper(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()

    def test_legacy_key_alias_resolution(self):
        # Legacy key 'classifier.weight' should map to 'gyroplane.centroids'
        legacy_state_dict = {
            "encoder.weight": torch.randn(64, 128),
            "encoder.bias": torch.randn(64),
            "classifier.weight": torch.randn(10, 64),
        }

        remapped = StateDictRemapper.remap_and_validate(
            loaded_state_dict=legacy_state_dict,
            target_model=self.model,
            strict_shapes=True
        )

        self.assertIn("gyroplane.centroids", remapped)
        self.assertNotIn("classifier.weight", remapped)
        self.assertEqual(remapped["gyroplane.centroids"].shape, (10, 64))

    def test_shape_mismatch_rejection(self):
        # Incompatible shape (e.g. 5 centroids instead of 10)
        invalid_state_dict = {
            "encoder.weight": torch.randn(64, 128),
            "encoder.bias": torch.randn(64),
            "classifier.weight": torch.randn(5, 64), # Wrong shape!
        }

        with self.assertRaises(ValueError) as ctx:
            StateDictRemapper.remap_and_validate(
                loaded_state_dict=invalid_state_dict,
                target_model=self.model,
                strict_shapes=True
            )

        self.assertIn("Shape mismatch", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
