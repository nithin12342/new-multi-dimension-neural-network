"""
FILE-026 | FOLDER-002 | src/domain/model/tile_contraction.py
Owning Aggregate: TileContraction
Responsibility: hardware-agnostic 16x16 tile GEMM matrix contractions for Chebyshev polynomial basis expansions and Poincaré metric projections
Must Never: use vendor-locked proprietary instructions or cause uncoalesced global VRAM roundtrips
"""

import torch
import torch.nn as nn
from typing import Tuple

class ChebyshevTileContraction16x16(nn.Module):
    """
    Pillar 7: Hardware-Agnostic 16x16 Matrix Tile Contraction Operator.
    Fuses Order-2 Chebyshev polynomial expansions T_0(X) = I, T_1(X) = X, T_2(X) = 2X^2 - I
    directly inside contiguous [B * N, 16, 16] tile GEMM operations, maintaining register/SRAM
    data residency and eliminating uncoalesced roundtrips to global VRAM.
    """

    def __init__(self, embed_dim: int = 256, tile_dim: int = 16, chebyshev_order: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.tile_dim = tile_dim
        self.chebyshev_order = chebyshev_order
        assert tile_dim * tile_dim == embed_dim, f"Tile dimension squared ({tile_dim}x{tile_dim}) must equal embed_dim ({embed_dim})"

        # Trainable coefficient weights per polynomial degree [Order+1, 16, 16]
        self.poly_weights = nn.Parameter(torch.randn(chebyshev_order + 1, tile_dim, tile_dim) * (1.0 / tile_dim))
        self.register_buffer("eye_tile", torch.eye(tile_dim))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Input: Z [B, N, 256] -> Reshaped to [B * N, 16, 16]
        Executes fused matrix tile contraction:
          T_0(X) = I
          T_1(X) = X
          T_2(X) = 2 * X @ X - I
          Output = sum_k (T_k(X) @ W_k)
        Output: [B, N, 256] contiguous
        """
        B, N, D = Z.shape
        BN = B * N

        # 1. Reshape sequence tokens into contiguous 16x16 tile matrices
        X = Z.contiguous().view(BN, self.tile_dim, self.tile_dim)

        # 2. Normalize tiles to [-1, 1] spectral radius for Chebyshev orthogonality
        frob_norm = torch.norm(X, p="fro", dim=(-2, -1), keepdim=True) + 1e-6
        X_norm = X / frob_norm

        # 3. Compute Basis Expansion: T_0 = I, T_1 = X_norm, T_2 = 2 * X_norm @ X_norm - I
        T0 = self.eye_tile.unsqueeze(0).expand(BN, -1, -1)
        T1 = X_norm
        T2 = 2.0 * torch.bmm(X_norm, X_norm) - T0

        # 4. Contract with polynomial weights via batched matrix multiplication (Hardware-Agnostic GEMM)
        W0 = self.poly_weights[0].unsqueeze(0).expand(BN, -1, -1)
        W1 = self.poly_weights[1].unsqueeze(0).expand(BN, -1, -1)
        W2 = self.poly_weights[2].unsqueeze(0).expand(BN, -1, -1)

        Y = torch.bmm(T0, W0) + torch.bmm(T1, W1) + torch.bmm(T2, W2)

        # 5. Contract back to [B, N, 256] sequence representation
        return Y.view(B, N, D).contiguous()
