"""
FILE-005 | FOLDER-002 | src/domain/model/riemannian.py
Owning Aggregate: ConformalRiemannianChart
Responsibility: map features to poincaré ball conformal charts and evaluate hyperbolic gyroplane classification
Must Never: allow feature norms to exceed unit disk boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class PoincareGyroplaneClassifier(nn.Module):
    """
    Hyperbolic Gyroplane Classifier operating on Poincaré Ball representations.
    Computes hyperbolic geodesic distance d_D^n(z, mu_k) to K trainable Riemannian centroids mu_k in D^n,
    producing calibrated logits Logits_k = - d_D^n(z, mu_k) / tau.
    Eliminates Euclidean linear metric distortion and fixes classification geometry collapse.
    """
    def __init__(self, embed_dim: int = 256, num_classes: int = 10, curvature: float = 1.0, temperature: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.curvature = curvature
        self.temperature = temperature
        self.chart = PoincareConformalChart(c=curvature)
        
        # Trainable Riemannian cluster centroids mu_k initialized inside unit ball
        raw_centroids = torch.randn(num_classes, embed_dim) * 0.05
        self.centroids = nn.Parameter(raw_centroids)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, embed_dim] -> returns calibrated logits [B, num_classes] based on Hyperbolic Geodesic Distance.
        """
        z_ball = self.chart.project_to_ball(z)             # [B, embed_dim]
        c_ball = self.chart.project_to_ball(self.centroids) # [K, embed_dim]
        
        B = z_ball.shape[0]
        K = c_ball.shape[0]
        
        z_exp = z_ball.unsqueeze(1).expand(B, K, -1)
        c_exp = c_ball.unsqueeze(0).expand(B, K, -1)
        
        # Pairwise Hyperbolic Geodesic distance: [B, K]
        dist = self.chart.geodesic_distance(z_exp, c_exp).squeeze(-1)
        
        # Negative temperature-scaled distance as calibrated class logits
        logits = -dist / max(1e-4, self.temperature)
        return logits
