"""
FILE-022 | FOLDER-002 | src/domain/model/decoder.py
Owning Aggregate: SingleNestedMatrixDecoder
Responsibility: project lower-dimensional core representations into all multi-task outputs using a single nested matrix decoder
Must Never: bypass nested matrix polynomial contractions during decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from src.domain.model.chebyshev import ChebyshevFunctionalBlock
from src.domain.model.trace_activation import TraceInvariantGate

class SingleNestedMatrixDecoder(nn.Module):
    """
    Single Unified Nested Matrix Decoder Aggregate.
    Combines all multi-task decoder head functionalities (Next-Token Prediction LM, Masked Reconstruction,
    SSL Contrastive Projection, Supervised Classification, Supervised Regression, DEC Soft Clustering)
    into ONE single decoder engine.
    Uses Order-2 Chebyshev Functional Nested Matrix Contractions (16x16 tiles) + Trace Scaling to map lower-dimensional
    manifold representations back into target output spaces.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        tile_dim: int = 16,
        chebyshev_order: int = 2,
        proj_dim: int = 128,
        vocab_size: int = 30522,
        num_classes: int = 10,
        num_clusters: int = 10,
        alpha: float = 1.0
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha

        # Decoder Nested Matrix Dimension Transformation Core
        self.decoder_chebyshev = ChebyshevFunctionalBlock(embed_dim, tile_dim, chebyshev_order)
        self.decoder_trace_gate = TraceInvariantGate(tile_dim)

        # Single Combined Multi-Task Decoder Projections
        self.ntp_projection = nn.Linear(embed_dim, vocab_size)       # Next-Token Thought LM Logits
        self.recon_projection = nn.Linear(embed_dim, embed_dim)      # Masked Autoencoder Reconstruction
        self.ssl_projection = nn.Linear(embed_dim, proj_dim)         # L2-Normalized Contrastive Projection
        self.cls_projection = nn.Linear(embed_dim, num_classes)      # Classification Logits
        self.reg_projection = nn.Linear(embed_dim, 1)                # Regression Scalar
        self.centroids = nn.Parameter(torch.randn(num_clusters, embed_dim) * 0.1) # DEC Cluster Centroids

        # Anti-Mode-Collapse Initializer: Xavier Uniform on classification projection head to eliminate index 8 attractor bias
        nn.init.xavier_uniform_(self.cls_projection.weight)
        nn.init.zeros_(self.cls_projection.bias)

    def forward(
        self,
        Z_sequence: torch.Tensor,
        z_riemannian: torch.Tensor,
        z_bar: torch.Tensor,
        compute_heads: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Single Decoder Forward Pass using Nested Matrix Contractions.
        Z_sequence: [B, N_total, 256], z_riemannian: [B, 256], z_bar: [B, 256]
        If compute_heads=False, computes only z_proj to preserve VRAM during secondary view contrastive passes.
        """
        # 1. Apply Decoder Nested Matrix Contraction & Trace Activation to Riemannian pooled state
        z_riem_seq = z_riemannian.unsqueeze(1) # [B, 1, 256]
        z_riem_contracted = self.decoder_trace_gate(self.decoder_chebyshev(z_riem_seq)).squeeze(1) # [B, 256]
        z_proj = F.normalize(self.ssl_projection(z_riem_contracted), dim=-1) # [B, 128]

        if not compute_heads:
            return {
                "z_bar": z_bar,
                "z_riemannian": z_riemannian,
                "z_proj": z_proj
            }

        # 2. Apply Decoder Nested Matrix Polynomial Contraction & Trace Activation to sequence state
        Z_dec_raw = self.decoder_chebyshev(Z_sequence)
        Z_dec = self.decoder_trace_gate(Z_dec_raw)

        # 3. Compute Single Unified Decoder Multi-Task Outputs
        ntp_logits = self.ntp_projection(Z_dec)                      # [B, N_total, 30522]
        x_recon = self.recon_projection(Z_dec)                       # [B, N_total, 256]
        logits = self.cls_projection(z_riem_contracted)              # [B, Num_Classes]
        reg_out = self.reg_projection(z_riem_contracted)             # [B, 1]

        # Student's t-distribution soft cluster assignments q_ij
        dist_sq = torch.sum((z_riem_contracted.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2, dim=-1) # [B, K]
        q_num = (1.0 + dist_sq / self.alpha) ** (- (self.alpha + 1.0) / 2.0)
        q_dist = q_num / torch.sum(q_num, dim=1, keepdim=True)       # [B, Num_Clusters]

        return {
            "z_bar": z_bar,
            "z_riemannian": z_riemannian,
            "z_proj": z_proj,
            "ntp_logits": ntp_logits,
            "x_recon": x_recon,
            "logits": logits,
            "reg_out": reg_out,
            "q_dist": q_dist
        }

class MultiTaskOmniDecoder(SingleNestedMatrixDecoder):
    """Backward compatibility alias for SingleNestedMatrixDecoder."""
    pass
