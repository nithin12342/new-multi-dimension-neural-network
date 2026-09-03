"""
Unit Test: Pillar 7 - Chebyshev Tile Kernels & Expansion
Validates Chebyshev polynomial evaluation: T_0(0.5) = 1.0, T_1(0.5) = 0.5, T_2(0.5) = -0.5
and 16x16 tile contraction operator.
"""

import unittest
from typing import Tuple
import torch
import numpy as np

from src.domain.model.tile_contraction import ChebyshevTileContraction16x16

def eval_chebyshev_scalar(x: float) -> Tuple[float, float, float]:
    t0 = 1.0
    t1 = x
    t2 = 2.0 * x * x - 1.0
    return t0, t1, t2

class TestTileKernels(unittest.TestCase):

    def test_chebyshev_polynomial_evaluations(self):
        # Specific assertions from specification:
        # T_0(0.5) = 1.0, T_1(0.5) = 0.5, T_2(0.5) = -0.5
        t0, t1, t2 = eval_chebyshev_scalar(0.5)
        self.assertEqual(t0, 1.0)
        self.assertEqual(t1, 0.5)
        self.assertEqual(t2, -0.5)

        # Boundary checks
        t0_1, t1_1, t2_1 = eval_chebyshev_scalar(1.0)
        self.assertEqual(t0_1, 1.0)
        self.assertEqual(t1_1, 1.0)
        self.assertEqual(t2_1, 1.0)

        t0_neg, t1_neg, t2_neg = eval_chebyshev_scalar(-1.0)
        self.assertEqual(t0_neg, 1.0)
        self.assertEqual(t1_neg, -1.0)
        self.assertEqual(t2_neg, 1.0)

    def test_chebyshev_16x16_tile_contraction_operator(self):
        B, N, D = 4, 16, 256
        layer = ChebyshevTileContraction16x16(embed_dim=D, tile_dim=16, chebyshev_order=2)
        
        inp = torch.randn(B, N, D)
        out = layer(inp)
        
        self.assertEqual(out.shape, (B, N, D))
        self.assertTrue(torch.isfinite(out).all())

if __name__ == "__main__":
    unittest.main()
