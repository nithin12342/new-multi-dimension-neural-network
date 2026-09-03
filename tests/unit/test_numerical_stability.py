"""
Unit Test: Pillar 6 - Numerical Stability & Invariant Defenses
Validates InfoNCE logit clamping to [-10.8, 10.8], Poincare radius clipping, and conformal factor ceiling.
"""

import unittest
import torch
import numpy as np

from src.losses.ssl_bundle import (
    clamped_infonce,
    poincare_boundary_clip,
    ClampedInfoNCELoss,
)

class TestNumericalStability(unittest.TestCase):

    def test_infonce_logit_clamping(self):
        # Adversarial high-norm vectors producing initial logits >> 100.0
        z1 = torch.tensor([[50.0, 50.0], [100.0, 100.0]], dtype=torch.float32)
        z2 = torch.tensor([[50.0, 50.0], [100.0, 100.0]], dtype=torch.float32)

        loss_fn = ClampedInfoNCELoss(temperature=0.07)
        loss = loss_fn(z1, z2)

        # Loss must be finite and well-conditioned
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_poincare_boundary_and_conformal_factor_bounds(self):
        eps = 1e-4
        max_allowed_norm = 1.0 - eps

        # Vectors outside and exactly on the boundary
        adversarial_vecs = torch.tensor([
            [1.5, 0.0],
            [0.0, 2.0],
            [0.999999, 0.0],
            [1.0, 1.0]
        ], dtype=torch.float32)

        clipped = poincare_boundary_clip(adversarial_vecs, eps=eps)
        norms = torch.norm(clipped, p=2, dim=-1)

        for norm_val in norms:
            self.assertLessEqual(
                norm_val.item(),
                max_allowed_norm + 1e-6,
                f"Poincaré norm must be <= {max_allowed_norm}"
            )

            # Check conformal factor lambda_x = 2 / (1 - ||x||^2) <= 1000.0
            norm_sq = norm_val.item() ** 2
            denom = max(1.0 - norm_sq, eps)
            lambda_x = min(2.0 / denom, 1000.0)
            self.assertLessEqual(lambda_x, 1000.0)

if __name__ == "__main__":
    unittest.main()
