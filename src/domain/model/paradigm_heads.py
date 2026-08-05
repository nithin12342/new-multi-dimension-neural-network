"""
FILE-006 | FOLDER-002 | src/domain/model/paradigm_heads.py
Owning Aggregate: ParadigmHeads
Responsibility: project pooled representations into paradigm output heads
Must Never: share learnable parameters across paradigm head instances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class SSLProjectionHead(nn.Module):
    """Self-Supervised Learning Projection Head (2-layer MLP z_proj = W2 * ReLU(W1 * Z_bar + b1))."""
    def __init__(self, in_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, proj_dim)

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        """Project pooled representation [B, 256] -> L2 normalized [B, 128]."""
        h = self.relu(self.fc1(z_bar))
        z_proj = self.fc2(h)
        return F.normalize(z_proj, dim=-1)

class MaskedReconstructionHead(nn.Module):
    """Masked Autoencoder Reconstruction Head (linear decoder X_hat = W_recon * Z^(L) + b_recon)."""
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.decoder = nn.Linear(in_dim, in_dim)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Decode sequence tokens [B, N, 256] -> [B, N, 256]."""
        return self.decoder(Z)

class NextTokenPredictionHead(nn.Module):
    """Self-Supervised Causal Next-Token Prediction Head for Thought Process Sequences."""
    def __init__(self, in_dim: int = 256, vocab_size: int = 30522):
        super().__init__()
        self.lm_head = nn.Linear(in_dim, vocab_size)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Project sequence representations [B, N, 256] -> Causal Vocabulary Logits [B, N, 30522]."""
        return self.lm_head(Z)

class SupervisedClassificationHead(nn.Module):
    """Supervised Classification Head (y_cls = W_cls * Z_bar + b_cls)."""
    def __init__(self, in_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        """Compute classification logits [B, 256] -> [B, K]."""
        return self.classifier(z_bar)

class SupervisedRegressionHead(nn.Module):
    """Supervised Regression Head (y_reg = W_reg * Z_bar + b_reg)."""
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.regressor = nn.Linear(in_dim, 1)

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        """Compute regression output [B, 256] -> [B, 1]."""
        return self.regressor(z_bar)

class DECClusteringHead(nn.Module):
    """Deep Embedded Clustering Head using Student's t-distribution soft cluster assignments q_ij."""
    def __init__(self, in_dim: int = 256, num_clusters: int = 10, alpha: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.randn(num_clusters, in_dim) * 0.1)

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignment distribution q_ij of shape [B, Num_Clusters].
        q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_j' (...)
        """
        dist_sq = torch.sum((z_bar.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2, dim=-1) # [B, K]
        q_num = (1.0 + dist_sq / self.alpha) ** (- (self.alpha + 1.0) / 2.0)
        q = q_num / torch.sum(q_num, dim=1, keepdim=True)
        return q
