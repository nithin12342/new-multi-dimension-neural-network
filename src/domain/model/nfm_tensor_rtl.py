"""
FILE-027 | FOLDER-002 | src/domain/model/nfm_tensor_rtl.py
Owning Aggregate: TensorRuntimeLayer
Responsibility: zero-copy tile memory layout management and execution runtime layer for 16x16 matrix tile contractions
Must Never: copy memory between host buffers when DLPack zero-copy handoffs are possible
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from src.domain.model.tile_contraction import ChebyshevTileContraction16x16

class NFMTensorRTL:
    """
    Pillar 7: Hardware-Agnostic Tensor Runtime Layer (RTL).
    Manages low-level 16x16 matrix tile memory buffers, zero-copy pointer transfers,
    and tile-level GEMM schedule dispatching across hardware backends.
    """

    def __init__(self, embed_dim: int = 256, tile_dim: int = 16):
        self.embed_dim = embed_dim
        self.tile_dim = tile_dim
        self.operator = ChebyshevTileContraction16x16(embed_dim=embed_dim, tile_dim=tile_dim)

    def execute_tile_contraction(self, Z: torch.Tensor) -> torch.Tensor:
        """Execute hardware-agnostic 16x16 tile contraction with contiguous memory alignment."""
        assert Z.shape[-1] == self.embed_dim, f"Expected hidden dimension {self.embed_dim}, got {Z.shape[-1]}"
        return self.operator(Z)

    @staticmethod
    def to_tile_matrix(Z: torch.Tensor, tile_dim: int = 16) -> torch.Tensor:
        """Reshape flattened feature vector into contiguous 2D tile matrix [..., 16, 16]."""
        shape = Z.shape[:-1] + (tile_dim, tile_dim)
        return Z.contiguous().view(shape)

    @staticmethod
    def from_tile_matrix(M: torch.Tensor) -> torch.Tensor:
        """Flatten 2D tile matrix back to vector embedding [..., 256]."""
        shape = M.shape[:-2] + (-1,)
        return M.contiguous().view(shape)
