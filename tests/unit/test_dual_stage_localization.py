"""
Unit Test: Pillar 2 - Dual-Stage Error Localization
Validates Stage 1 (representation variance collapse) and Stage 2 (fused gradient norm explosion).
"""

import unittest
import torch
import numpy as np

from src.domain.model.error_localization import MultimodalErrorLocalizationEngine

class TestDualStageLocalization(unittest.TestCase):

    def setUp(self):
        self.engine = MultimodalErrorLocalizationEngine(
            variance_floor=1e-4,
            grad_norm_cap=100.0
        )

    def test_stage1_variance_collapse_detection(self):
        # Stage 1: Inject collapsed feature vector with zero variance (sigma^2 < 1e-5)
        collapsed_features = torch.full((1, 16, 256), 0.42, dtype=torch.float32)
        
        # Confirm variance collapse is identified by Stage 1 auditor
        is_healthy = self.engine.localizer.audit_intermediate_features(collapsed_features)
        self.assertFalse(is_healthy, "Collapsed features must fail Stage 1 audit")

        # Healthy features with high variance
        healthy_features = torch.randn(1, 16, 256) * 2.0
        self.assertTrue(self.engine.localizer.audit_intermediate_features(healthy_features))

    def test_stage2_gradient_norm_explosion_detection(self):
        # Stage 2: Inject exploded gradient vector with norm > 100.0
        exploded_grads = torch.ones(500) * 10.0 # L2 norm = sqrt(500 * 100) = sqrt(50000) = 223.6 > 100.0
        norm_val = float(torch.norm(exploded_grads, p=2).item())
        
        self.assertGreater(norm_val, 100.0)
        
        # Verify gradient auditor detects the violation
        is_safe = self.engine.localizer.audit_fused_gradients(exploded_grads)
        self.assertFalse(is_safe, "Exploded gradient norm must be rejected by Stage 2 audit")

        # Safe gradients with low norm
        safe_grads = torch.ones(10) * 0.5 # L2 norm = sqrt(10 * 0.25) = 1.58 < 100.0
        self.assertTrue(self.engine.localizer.audit_fused_gradients(safe_grads))

if __name__ == "__main__":
    unittest.main()
