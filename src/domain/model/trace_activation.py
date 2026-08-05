"""
FILE-003 | FOLDER-002 | src/domain/model/trace_activation.py
Owning Aggregate: ChebyshevFunctionalBlock
Responsibility: apply trace invariant activation scaling to matrix tiles
Must Never: cause warp divergence across matrix dimensions
"""

import torch
import torch.nn as nn

class TraceInvariantGate(nn.Module):
    """
    In-Register Trace-Invariant Activation Gate.
    Scales matrix elements using normalized matrix trace: scale(Y) = Sigmoid(Tr(Y) / 16).
    """
    def __init__(self, tile_dim: int = 16):
        super().__init__()
        self.tile_dim = tile_dim

    def compute_matrix_trace(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix trace Tr(Y) for batched 16x16 tiles [B_N, 16, 16].
        Returns trace scaling scalar tensor of shape [B_N, 1, 1].
        """
        # Sum diagonal elements along dim 1 and 2
        trace = torch.diagonal(Y, dim1=1, dim2=2).sum(dim=-1, keepdim=True).unsqueeze(-1)
        return trace

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute scale factor Sigmoid(Tr(Y)/16) and apply elementwise scaling.
        Supports both 3D [B_N, 16, 16] and 3D token tensors [B, N, D].
        """
        if Y.ndim == 3 and Y.shape[1] != self.tile_dim:
            # Shape is [B, N, D] -> view as [B*N, 16, 16]
            B, N, D = Y.shape
            tiles = Y.view(-1, self.tile_dim, self.tile_dim)
            trace = self.compute_matrix_trace(tiles) # [B*N, 1, 1]
            scale = torch.sigmoid(trace / float(self.tile_dim)) # [B*N, 1, 1]
            scaled_tiles = tiles * scale
            return scaled_tiles.view(B, N, D)
        else:
            trace = self.compute_matrix_trace(Y)
            scale = torch.sigmoid(trace / float(self.tile_dim))
            return Y * scale
