"""
FILE-005 | FOLDER-002 | src/domain/model/riemannian.py
Owning Aggregate: ConformalRiemannianChart
Responsibility: map features to poincaré ball conformal charts with strict boundary saturation clipping (||x|| <= 1 - 1e-4) and evaluate hyperbolic gyroplane classification
Must Never: allow feature norms to exceed unit disk boundary or conformal scale to blow up to infinity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_poincare(
    x: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    r"""Projects unconstrained Euclidean vectors into the Poincare ball.

    Enforces ||x|| < (1 - eps) / sqrt(c).
    """
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    max_norm = (1.0 - eps) / torch.sqrt(c_t.clamp_min(1e-8))
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    cond = norm >= max_norm
    projected = (max_norm / norm) * x
    return torch.where(cond, projected, x)


def exp_map_zero(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at origin: T_0 D -> D. Bounded output via project_poincare."""
    c_t = torch.as_tensor(c, dtype=v.dtype, device=v.device)
    sqrt_c = torch.sqrt(c_t.clamp_min(1e-8))
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    gamma = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return project_poincare(gamma * v, c=c, eps=eps)


def clip_manifold_grads(model: nn.Module, max_norm: float = 1.0) -> None:
    """Clips gradients to prevent hyperbolic boundary divergence."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def project_manifold_params_(model: nn.Module) -> None:
    """Projects trainable hyperbolic params back inside the ball after optimizer step."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, PoincareGyroplaneClassifier):
                m.centroids.copy_(
                    project_poincare(m.centroids.data, c=m.curvature, eps=m.chart.eps)
                )


class PoincareConformalChart(nn.Module):
    """
    Metric Deformation via Conformal Riemannian Charting on the Poincare Ball manifold.
    Enforces strict curvature-aware clipping ||x|| <= (1 - eps) / sqrt(c)
    to prevent boundary saturation where lambda_x -> infinity.
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.c = c
        self.eps = eps

    def project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce strict hyperbolic constraint ||x|| <= (1 - eps) / sqrt(c)."""
        return project_poincare(x, c=self.c, eps=self.eps)

    def _max_norm_sq(self) -> float:
        max_norm = (1.0 - self.eps) / max(1e-8, self.c) ** 0.5
        return max_norm * max_norm

    def conformal_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute conformal scale factor lambda_x = 2 / (1 - c * ||x||^2).
        Strictly clamped to <= 1000.0 to prevent infinite gradients.
        """
        x_proj = self.project_to_ball(x)
        norm_sq = torch.sum(x_proj**2, dim=-1, keepdim=True)
        norm_sq = torch.clamp(norm_sq, max=self._max_norm_sq())
        lambda_x = 2.0 / (1.0 - self.c * norm_sq + 1e-7)
        return torch.clamp(lambda_x, min=1.0, max=1000.0)

    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Möbius vector addition x (+) c y with boundary saturation defense."""
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)
        c = self.c
        x_sq = torch.clamp(
            torch.sum(x**2, dim=-1, keepdim=True), max=self._max_norm_sq()
        )
        y_sq = torch.clamp(
            torch.sum(y**2, dim=-1, keepdim=True), max=self._max_norm_sq()
        )
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1.0 + 2.0 * c * xy + c * y_sq) * x + (1.0 - c * x_sq) * y
        denom = 1.0 + 2.0 * c * xy + (c**2) * x_sq * y_sq + 1e-7
        denom = torch.clamp(denom, min=1e-5)
        return self.project_to_ball(num / denom)

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic geodesic distance d_M(x, y).
        Strictly clamps argument to acosh to [1.0 + 1e-7, 1e4] to eliminate numerical explosions.
        """
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)
        diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
        x_sq = torch.clamp(
            torch.sum(x**2, dim=-1, keepdim=True), max=self._max_norm_sq()
        )
        y_sq = torch.clamp(
            torch.sum(y**2, dim=-1, keepdim=True), max=self._max_norm_sq()
        )

        denom = torch.clamp((1.0 - x_sq) * (1.0 - y_sq), min=1e-7)
        arg = 1.0 + 2.0 * diff_sq / denom
        arg = torch.clamp(arg, min=1.0 + 1e-7, max=1e4)
        dist = torch.acosh(arg)
        return dist

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Map tangent vector at origin onto the ball (tanh keeps output inside)."""
        return exp_map_zero(v, c=self.c, eps=self.eps)

    def forward(self, x: torch.Tensor, use_exp_map: bool = False) -> torch.Tensor:
        """Project input features into Poincare ball interior.

        Must Never multiply by conformal scale: x_proj * lambda_x blows the
        norm to ~0.9999 * 1000 = 999.9, breaching D^n. lambda_x is a metric
        tensor for distances/gradients only; expose via conformal_scale().
        """
        if use_exp_map:
            return self.exp_map_zero(x)
        return self.project_to_ball(x)


class PoincareGyroplaneClassifier(nn.Module):
    """
    Hyperbolic Gyroplane Classifier operating on Poincaré Ball representations.
    Computes hyperbolic geodesic distance d_D^n(z, mu_k) to K trainable Riemannian centroids mu_k in D^n,
    producing calibrated logits Logits_k = - d_D^n(z, mu_k) / tau.
    Eliminates Euclidean linear metric distortion and fixes classification geometry collapse.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 10,
        curvature: float = 1.0,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.curvature = curvature
        self.temperature = temperature
        self.chart = PoincareConformalChart(c=curvature, eps=1e-4)

        # Trainable Riemannian cluster centroids mu_k initialized inside unit ball
        raw_centroids = torch.randn(num_classes, embed_dim) * 0.05
        self.centroids = nn.Parameter(raw_centroids)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, embed_dim] -> returns calibrated logits [B, num_classes] based on Hyperbolic Geodesic Distance.
        """
        z_ball = self.chart.project_to_ball(z)  # [B, embed_dim]
        c_ball = self.chart.project_to_ball(self.centroids)  # [K, embed_dim]

        B = z_ball.shape[0]
        K = c_ball.shape[0]

        z_exp = z_ball.unsqueeze(1).expand(B, K, -1)
        c_exp = c_ball.unsqueeze(0).expand(B, K, -1)

        # Pairwise Hyperbolic Geodesic distance: [B, K]
        dist = self.chart.geodesic_distance(z_exp, c_exp).squeeze(-1)

        # Negative temperature-scaled distance as calibrated class logits
        logits = -dist / max(1e-4, self.temperature)
        return logits
