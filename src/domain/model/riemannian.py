"""
FILE-005 | FOLDER-002 | src/domain/model/riemannian.py
Owning Aggregate: ConformalRiemannianChart
Responsibility: map features to poincaré ball conformal charts
Must Never: allow feature norms to exceed unit disk boundary
"""

import torch
import torch.nn as nn

class PoincareConformalChart(nn.Module):
    """
    Metric Deformation via Conformal Riemannian Charting on the Poincaré Ball manifold.
    Enforces norm ||x|| < 1, computes scale factor lambda_x, geodesic distance, and Möbius addition.
    """
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.c = c
        self.eps = eps

    def project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce hyperbolic constraint ||x|| < 1 - eps."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 - self.eps
        cond = norm > max_norm
        projected = x / (norm + 1e-8) * max_norm
        return torch.where(cond, projected, x)

    def conformal_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute conformal scale factor lambda_x = 2 / (1 - c * ||x||^2)."""
        x_proj = self.project_to_ball(x)
        norm_sq = torch.sum(x_proj ** 2, dim=-1, keepdim=True)
        lambda_x = 2.0 / (1.0 - self.c * norm_sq + 1e-7)
        return lambda_x

    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Möbius vector addition x (+) c y."""
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)
        c = self.c
        x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1.0 + 2.0 * c * xy + c * y_sq) * x + (1.0 - c * x_sq) * y
        denom = 1.0 + 2.0 * c * xy + (c ** 2) * x_sq * y_sq + 1e-7
        return self.project_to_ball(num / denom)

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic geodesic distance d_M(x, y)."""
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)
        diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
        x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)

        arg = 1.0 + 2.0 * diff_sq / ((1.0 - x_sq) * (1.0 - y_sq) + 1e-7)
        arg = torch.clamp(arg, min=1.0 + 1e-7)
        dist = torch.acosh(arg)
        return dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to Poincaré ball manifold and apply conformal scale."""
        x_proj = self.project_to_ball(x)
        scale = self.conformal_scale(x_proj)
        return x_proj * scale
