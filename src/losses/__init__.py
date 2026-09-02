"""
Package: src.losses
Canonical shortcut forwarding to guarded loss functions and SSL bundles.
"""

from src.domain.loss.losses import (
    InfoNCELoss,
    ClampedInfoNCELoss,
    BarlowTwinsLoss,
    VICRegLoss,
    CausalNextTokenLoss,
    MaskedReconstructionLoss,
    DECKLRegLoss,
)
from src.domain.loss.ssl_bundle import MultimodalSSLBundle
from src.domain.loss.matryoshka_loss import MatryoshkaLoss

__all__ = [
    "InfoNCELoss",
    "ClampedInfoNCELoss",
    "BarlowTwinsLoss",
    "VICRegLoss",
    "CausalNextTokenLoss",
    "MaskedReconstructionLoss",
    "DECKLRegLoss",
    "MultimodalSSLBundle",
    "MatryoshkaLoss",
]
