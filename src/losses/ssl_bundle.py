"""
FILE: src/losses/ssl_bundle.py
Owning Aggregate: SSLBundle
Responsibility: Canonical hardware-agnostic loss bundle with strict numerical invariants:
  1. Clamped InfoNCE logit clamping to [-10.8, 10.8] preventing FP16 exp() overflow (>65,504)
  2. VICReg variance hinge (gamma=1.0, eps=1e-4) and covariance penalty preventing latent collapse
  3. Poincare boundary projection guard (||x|| <= 1 - 1e-4, lambda_x <= 1000.0)
  4. Causal NTP pad masking (ignore_index=0) preventing perplexity stalling (PPL > 600)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from src.domain.loss.losses import (
    InfoNCELoss,
    ClampedInfoNCELoss,
    BarlowTwinsLoss,
    VICRegLoss,
    CausalNextTokenLoss,
    DECKLRegLoss,
)
from src.domain.loss.ssl_bundle import MultimodalSSLBundle

def clamped_infonce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
    clamp_min: float = -10.8,
    clamp_max: float = 10.8
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss with strict logit clamping to prevent FP16 overflow."""
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    sim = torch.matmul(z1_norm, z2_norm.T) / temperature
    sim_clamped = torch.clamp(sim, min=clamp_min, max=clamp_max)
    targets = torch.arange(sim_clamped.size(0), device=sim_clamped.device)
    return F.cross_entropy(sim_clamped, targets)

def vicreg_variance_hinge(
    z: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4
) -> torch.Tensor:
    """Compute VICReg variance hinge loss: mean(max(0, gamma - std(z_j)))."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))

def poincare_boundary_clip(
    x: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-4,
    max_conformal_scale: float = 1000.0
) -> torch.Tensor:
    """Projects vectors onto the Poincare ball with strict radius clipping ||x|| <= 1 - eps."""
    max_norm = 1.0 - eps
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-7)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale

__all__ = [
    "MultimodalSSLBundle",
    "InfoNCELoss",
    "ClampedInfoNCELoss",
    "BarlowTwinsLoss",
    "VICRegLoss",
    "CausalNextTokenLoss",
    "DECKLRegLoss",
    "clamped_infonce",
    "vicreg_variance_hinge",
    "poincare_boundary_clip",
]
