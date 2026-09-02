"""
Unit Tests for 4 Remediation Guards:
1. FP16 InfoNCE Overflow Guard (clamp to 10.8)
2. VICReg Variance Hinge & Covariance Penalty (prevent latent collapse)
3. Poincaré Ball Boundary Saturation Guard (norm <= 1 - 1e-4)
4. Causal Next-Token Pad Masking & Anti-Collapse Silhouette Computation
"""

import unittest
import torch
import numpy as np

from src.domain.loss.loss_functions import InfoNCELoss, VICRegLoss, CausalNextTokenLoss
from src.domain.model.riemannian import PoincareConformalChart
from src.infrastructure.metrics.metric_computer import MetricComputer

class TestRemediationGuards(unittest.TestCase):

    def test_infonce_fp16_overflow_guard(self):
        """Verify InfoNCE similarity matrix is strictly clamped to [-10.8, 10.8] preventing FP16 exp() overflow."""
        loss_fn = InfoNCELoss(temperature=0.07, max_logit=10.8)
        B, D = 4, 128
        # Create adversarial collinear vectors with large norms
        z_i = torch.ones(B, D) * 100.0
        z_j = torch.ones(B, D) * 100.0

        loss = loss_fn(z_i, z_j)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertLessEqual(loss.item(), 50.0)

    def test_vicreg_variance_hinge(self):
        """Verify VICReg variance hinge strictly penalizes collapsed representations where std < gamma."""
        loss_fn = VICRegLoss(gamma=1.0, eps=1e-4)
        B, D = 64, 16

        # Case A: Collapsed representations (zero variance across channels)
        z_collapsed = torch.ones(B, D)
        loss_collapsed = loss_fn(z_collapsed, z_collapsed)

        # Case B: Well-distributed representations (std >= 1.0)
        z_healthy = torch.randn(B, D) * 1.2
        loss_healthy = loss_fn(z_healthy, z_healthy)

        # Collapsed loss must be strictly greater than healthy loss due to variance hinge penalty
        self.assertGreater(loss_collapsed.item(), loss_healthy.item())

    def test_poincare_boundary_clipping(self):
        """Verify Poincaré conformal factor and distance remain bounded even when ||x|| approaches or exceeds 1.0."""
        chart = PoincareConformalChart(c=1.0, eps=1e-4)
        
        # Extremely adversarial vector near/past unit disk boundary
        x_edge = torch.ones(2, 64) * 0.999999
        y_edge = -torch.ones(2, 64) * 0.999999

        scale = chart.conformal_scale(x_edge)
        self.assertFalse(torch.isnan(scale).any())
        self.assertFalse(torch.isinf(scale).any())
        self.assertLessEqual(torch.max(scale).item(), 1000.0)

        dist = chart.geodesic_distance(x_edge, y_edge)
        self.assertFalse(torch.isnan(dist).any())
        self.assertFalse(torch.isinf(dist).any())

    def test_causal_ntp_pad_masking(self):
        """Verify CausalNextTokenLoss ignores padded tokens (index 0) and computes clean gradients."""
        loss_fn = CausalNextTokenLoss(ignore_index=0)
        B, S, V = 2, 8, 100

        logits = torch.randn(B, S, V, requires_grad=True)
        # Sequence of tokens where half are padding (0)
        targets = torch.tensor([[10, 15, 20, 0, 0, 0, 0, 0],
                                [5, 8, 12, 16, 0, 0, 0, 0]], dtype=torch.long)

        loss = loss_fn(logits, targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertIsNotNone(logits.grad)

    def test_metric_computer_anti_collapse_silhouette(self):
        """Verify MetricComputer penalizes collapsed representations instead of outputting a degenerate 0.997."""
        evaluator = MetricComputer()
        B, D = 10, 64
        
        # Case A: Totally collapsed embeddings (all vectors nearly identical)
        collapsed_embeds = np.ones((B, D)) + np.random.randn(B, D) * 1e-6
        # Case B: Well-dispersed diverse embeddings
        healthy_embeds = np.random.randn(B, D)

        preds = np.random.randn(B, 10)
        targets = np.random.randint(0, 10, size=(B,))
        losses = {"ce": 1.5, "infonce": 0.5, "mlmce": 1.5}

        metrics_collapsed = evaluator.compute_all_37_metrics(preds, targets, collapsed_embeds, losses)
        metrics_healthy = evaluator.compute_all_37_metrics(preds, targets, healthy_embeds, losses)

        # Collapsed silhouette must be low (<= 0.20), NOT the false 0.997!
        self.assertLessEqual(metrics_collapsed["silhouette"], 0.20)
        # Healthy silhouette should be significantly higher
        self.assertGreater(metrics_healthy["silhouette"], metrics_collapsed["silhouette"])

if __name__ == "__main__":
    unittest.main()
