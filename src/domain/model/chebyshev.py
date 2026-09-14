"""
FILE-002 | FOLDER-002 | src/domain/model/chebyshev.py
Owning Aggregate: ChebyshevFunctionalBlock
Responsibility: compute order-2 chebyshev functional matrix polynomial contractions
Must Never: flatten matrix tiles into 1d vectors
"""

import torch
import torch.nn as nn
from typing import Tuple


class ChebyshevFunctionalBlock(nn.Module):
    """
    Order-2 Chebyshev Functional Matrix Expansion Block operating over atomic 16x16 matrix tiles.
    Evaluates matrix polynomial bases T0(X)=X, T1(X)=X, T2(X)=2*(X*X^T)-X and contracts with C0, C1, C2.
    """

    def __init__(self, embed_dim: int = 256, tile_dim: int = 16, order: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.tile_dim = tile_dim
        self.order = order
        assert embed_dim == tile_dim * tile_dim, (
            f"embed_dim ({embed_dim}) must equal tile_dim^2 ({tile_dim * tile_dim})"
        )

        # Trainable 16x16 coefficient matrices C0, C1, C2.
        # Gain-2 Xavier init over the 3 concatenated bases: each block must
        # output ~2x its input RMS because the TraceInvariantGate that always
        # follows halves it (sigmoid(Tr/16) ~= 0.5 at init), netting ~1.0 per
        # stage. The old 0.02 shrank ~7x per block, collapsing the network to
        # a constant function (z_bar -> 0, all views identical, silhouette 0).
        init_std = 2.0 / (3 * tile_dim) ** 0.5
        self.C0 = nn.Parameter(torch.randn(tile_dim, tile_dim) * init_std)
        self.C1 = nn.Parameter(torch.randn(tile_dim, tile_dim) * init_std)
        self.C2 = nn.Parameter(torch.randn(tile_dim, tile_dim) * init_std)

    def compute_chebyshev_bases(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute T0(X), T1(X), T2(X) for input matrix tiles X of shape [B_N, 16, 16].
        T0(X) = X
        T1(X) = X
        T2(X) = 2 * (X @ X^T) - X
        Returns tuple of 3 tensors each of shape [B_N, 16, 16].
        """
        T0 = X
        T1 = X
        # Batch matrix multiplication: X @ X^T -> [B_N, 16, 16], averaged
        # over 4 contraction dims. The gain-2 init above raises the quadratic
        # term with it, so T2 is tamed 4x harder to keep large-X behavior
        # stable (per-stage map ~= 0.6 a^2, fixed point well above the
        # operating range). At init (small X) T2 ~= -X, as before.
        XXT = torch.bmm(X, X.transpose(1, 2)) / (4 * self.tile_dim)
        T2 = 2.0 * XXT - X
        return T0, T1, T2

    def contract_tensor_cores(
        self, T0: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor
    ) -> torch.Tensor:
        """
        Contract Chebyshev bases with trainable coefficient matrices C0, C1, C2.
        Y = T0 @ C0 + T1 @ C1 + T2 @ C2
        Returns output tensor Y of shape [B_N, 16, 16].
        """
        Y = (
            torch.matmul(T0, self.C0)
            + torch.matmul(T1, self.C1)
            + torch.matmul(T2, self.C2)
        )
        return Y

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        Input Z: [B, N, 256]
        Reshape: [B*N, 16, 16]
        Bases -> Contraction -> Reshape back: [B, N, 256]
        """
        B, N, D = Z.shape
        X = Z.view(-1, self.tile_dim, self.tile_dim)
        T0, T1, T2 = self.compute_chebyshev_bases(X)
        Y = self.contract_tensor_cores(T0, T1, T2)
        Z_out = Y.view(B, N, D)
        return Z_out
