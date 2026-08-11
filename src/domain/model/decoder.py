"""
FILE-022 | FOLDER-002 | src/domain/model/decoder.py
Owning Aggregate: MultiTaskOmniDecoder
Responsibility: project core representations into multi-task paradigm decoder head outputs
Must Never: share learnable weights across paradigm decoder head instances
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from src.domain.model.paradigm_heads import (
    SSLProjectionHead, MaskedReconstructionHead, NextTokenPredictionHead,
    SupervisedClassificationHead, SupervisedRegressionHead, DECClusteringHead
)

class MultiTaskOmniDecoder(nn.Module):
    """
    Multi-Task Omni Decoder Aggregate.
    Combines all 6 paradigm output heads (Next-Token Prediction LM Head, Masked Autoencoder Decoder,
    SSL Projection Head, Classification Head, Regression Head, DEC Clustering Head) into a single decoder unit.
    """

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128, vocab_size: int = 30522, num_classes: int = 10, num_clusters: int = 10):
        super().__init__()
        self.ssl_projector = SSLProjectionHead(embed_dim, proj_dim)
        self.masked_recon = MaskedReconstructionHead(embed_dim)
        self.ntp_head = NextTokenPredictionHead(embed_dim, vocab_size)
        self.classifier = SupervisedClassificationHead(embed_dim, num_classes)
        self.regressor = SupervisedRegressionHead(embed_dim)
        self.dec_clustering = DECClusteringHead(embed_dim, num_clusters)

    def forward(self, Z_sequence: torch.Tensor, z_riemannian: torch.Tensor, z_bar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project core model outputs through multi-task decoder heads.
        Z_sequence: [B, N_total, 256], z_riemannian: [B, 256], z_bar: [B, 256]
        """
        z_proj = self.ssl_projector(z_riemannian)
        x_recon = self.masked_recon(Z_sequence)
        ntp_logits = self.ntp_head(Z_sequence)
        logits = self.classifier(z_riemannian)
        reg_out = self.regressor(z_riemannian)
        q_dist = self.dec_clustering(z_riemannian)

        return {
            "z_bar": z_bar,
            "z_riemannian": z_riemannian,
            "z_proj": z_proj,
            "x_recon": x_recon,
            "ntp_logits": ntp_logits,
            "logits": logits,
            "reg_out": reg_out,
            "q_dist": q_dist
        }
