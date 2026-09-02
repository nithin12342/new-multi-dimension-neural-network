"""
FILE-024 | FOLDER-004 | src/domain/loss/losses.py
Owning Aggregate: LossAggregation
Responsibility: export verified numerical stability loss functions with FP16 clamping and anti-collapse variance hinges
Must Never: allow unclamped similarity logits or unmasked pad tokens
"""

from src.domain.loss.loss_functions import (
    InfoNCELoss,
    BarlowTwinsLoss,
    VICRegLoss,
    CausalNextTokenLoss,
    CrossEntropyParadigmLoss,
    DECKLRegLoss,
    ClampedInfoNCELoss
)

__all__ = [
    "InfoNCELoss",
    "ClampedInfoNCELoss",
    "BarlowTwinsLoss",
    "VICRegLoss",
    "CausalNextTokenLoss",
    "CrossEntropyParadigmLoss",
    "DECKLRegLoss"
]
