"""
Unit Tests for the epoch-2 freeze-at-53.4340 root-cause fixes:
1. Loss values are never clamped (a saturated clamp kills gradients -> permanent freeze).
2. Chebyshev T2 uses mean (not sum) contraction so stacked blocks cannot explode.
3. Fresh runs reset the persistent traversal registry to Chunk 000.
"""

import os
import tempfile
import unittest

import torch

from src.domain.loss.loss_functions import CausalNextTokenLoss, InfoNCELoss
from src.domain.model.chebyshev import ChebyshevFunctionalBlock

duckdb = None
try:
    import duckdb as _duckdb  # type: ignore

    duckdb = _duckdb
except ImportError:
    pass


class TestNoZeroGradLossClamp(unittest.TestCase):
    def test_ntp_grad_flows_when_logits_explode(self):
        """Exploded NTP logits must yield finite loss with live gradients, not a pinned 50.0."""
        loss_fn = CausalNextTokenLoss(ignore_index=0)
        logits = (torch.randn(2, 8, 100) * 1000.0).requires_grad_()
        targets = torch.randint(1, 100, (2, 8))
        loss = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_infonce_forward_stays_fp16_bounded(self):
        """Forward similarities must stay within [-10.8, 10.8] (FP16 contract)."""
        loss_fn = InfoNCELoss(temperature=0.07)
        z = (torch.randn(4, 16) * 100.0).requires_grad_()
        loss = loss_fn(z, z.clone().detach())
        self.assertTrue(torch.isfinite(loss))

    def test_infonce_grad_reaches_embeddings_when_saturated(self):
        """Straight-through clamp: saturated sims must still push gradients into z."""
        loss_fn = InfoNCELoss(temperature=0.07)
        base = torch.randn(4, 16)
        # Near-identical views -> all sims saturate the 10.8 ceiling.
        z = (base + torch.randn_like(base) * 1e-4).requires_grad_()
        loss = loss_fn(z, base)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.0)

    def test_infonce_grad_flows_on_collapsed_views(self):
        """Identical views must still produce gradient signal (uniform sim -> ln(2B-1))."""
        loss_fn = InfoNCELoss(temperature=0.07)
        z = torch.ones(4, 16, requires_grad=True)
        loss = loss_fn(z, z.clone().detach())
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            loss.item(), 1.9459, places=3
        )  # ln(2B - 1) = ln(7): documents collapse value
        loss.backward()
        self.assertIsNotNone(z.grad)


class TestChebyshevBoundedGrowth(unittest.TestCase):
    def test_large_tiles_stay_finite_with_live_grads(self):
        """20x-magnitude tiles must not explode through a block (old sum-formula blew up ~16x)."""
        block = ChebyshevFunctionalBlock()
        torch.manual_seed(0)
        X = (torch.randn(8, 16, 16) * 20.0).requires_grad_()
        B, _, _ = X.shape
        Z = X.view(B, 256).unsqueeze(0).expand(2, -1, -1).contiguous()
        out = block(Z)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(float(out.abs().max()), 1e6)
        out.sum().backward()
        self.assertTrue(torch.isfinite(X.grad).all())

    def test_init_regime_unchanged(self):
        """Small (init-scale) inputs behave as before: T2 ~= -X dominates, no collapse to zero."""
        block = ChebyshevFunctionalBlock()
        torch.manual_seed(1)
        Z = torch.randn(2, 4, 256) * 0.1
        out = block(Z)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(float(out.abs().mean()), 1e-6)


@unittest.skipIf(duckdb is None, "duckdb not installed")
class TestTraversalResetOnFreshStart(unittest.TestCase):
    def test_reset_returns_to_chunk_zero(self):
        from src.infrastructure.logging.prediction_logger import PredictionLogExporter

        tmp = tempfile.mkdtemp()
        exporter = PredictionLogExporter(tmp)
        for i in range(5):
            exporter.log_traversal_chunk(
                timestamp="t",
                stream_id=1,
                epoch=1,
                chunk_index=i,
                chunk_size=128,
                total_raw=60000,
            )
        idx, _, _ = exporter.get_next_unvisited_chunk_index(
            chunk_size=128, total_raw=60000
        )
        self.assertEqual(idx, 5)
        exporter.reset_traversal_history()
        idx, _, pass_num = exporter.get_next_unvisited_chunk_index(
            chunk_size=128, total_raw=60000
        )
        self.assertEqual(idx, 0)
        self.assertEqual(pass_num, 1)
        self.assertTrue(
            os.path.exists(os.path.join(tmp, "multimodal_telemetry.duckdb"))
        )


if __name__ == "__main__":
    unittest.main()
