"""
FILE-021 | FOLDER-002 | src/domain/model/core_model.py
Owning Aggregate: FunctionalCoreModel
Responsibility: execute chebyshev matrix contractions trace scaling and poincaré hyperbolic chart mapping
Must Never: allow feature norms to exceed unit disk boundary
"""

import torch
import torch.nn as nn
from typing import Tuple

from src.domain.model.chebyshev import ChebyshevFunctionalBlock
from src.domain.model.trace_activation import TraceInvariantGate
from src.domain.model.riemannian import PoincareConformalChart


class FunctionalCoreModel(nn.Module):
    """
    Functional Core Model Backbone Aggregate.
    Executes Stage 1 & Stage 2 Order-2 Chebyshev Functional Matrix Tile Contractions (16x16),
    Trace-Invariant Activation Scaling, Global Token Pooling, and Poincaré Hyperbolic Conformal Chart Mapping.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        tile_dim: int = 16,
        chebyshev_order: int = 2,
        poincare_curvature: float = 1.0,
    ):
        super().__init__()
        self.chebyshev1 = ChebyshevFunctionalBlock(embed_dim, tile_dim, chebyshev_order)
        self.trace_gate1 = TraceInvariantGate(tile_dim)
        self.chebyshev2 = ChebyshevFunctionalBlock(embed_dim, tile_dim, chebyshev_order)
        self.trace_gate2 = TraceInvariantGate(tile_dim)
        self.riemannian_chart = PoincareConformalChart(poincare_curvature)

    def forward(
        self, Z0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Functional Core Model.
        Z0: [B, N_total, 256] -> Returns (Z2_scaled, z_riemannian, z_bar).
        """
        # Stage 1
        Z1 = self.chebyshev1(Z0)
        Z1_scaled = self.trace_gate1(Z1)

        # Stage 2
        Z2 = self.chebyshev2(Z1_scaled)
        Z2_scaled = self.trace_gate2(Z2)

        # Global Sequence Pooling
        z_bar = Z2_scaled.mean(dim=1)  # [B, 256]

        # Poincare Conformal Riemannian Hyperbolic Mapping.
        # z_bar norms grow with dim (||z|| ~ sqrt(D)) so a hard clip pins
        # EVERY sample to the shell (radius == 1 - eps) and the manifold
        # alert fires each step. Apply a parameter-free adaptive radial
        # squash (direction-preserving, monotonic, bounded by target < 1)
        # then exp_map_zero, so radii distribute inside the ball.
        z_norm = torch.norm(z_bar, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        tangent_scale = 1.0 / (1.0 + z_norm)
        z_tangent = z_bar * tangent_scale
        z_riemannian = self.riemannian_chart.exp_map_zero(z_tangent)

        return Z2_scaled, z_riemannian, z_bar
