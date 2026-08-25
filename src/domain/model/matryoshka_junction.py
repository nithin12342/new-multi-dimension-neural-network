"""
FILE-028 | FOLDER-002 | src/domain/model/matryoshka_junction.py
Owning Aggregate: MatryoshkaJunction
Responsibility: perform L2 norm-rescaled feature concatenation and projection between nested sub-models per Godey & Artzi (Cornell 2026)
Must Never: allow magnitude mismatch or dimension misalignment during feature concatenation
"""

import torch
import torch.nn as nn

class InterModelMatryoshkaJunction(nn.Module):
    """
    Inter-Model Junction with L2 Norm Rescaling (Godey & Artzi, Cornell 2026).
    Rescales lower-exit representations to match fresh input embedding norm magnitude before concatenation and linear projection.
    """

    def __init__(self, lower_dim: int = 256, fresh_dim: int = 256, out_dim: int = 256, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.proj = nn.Linear(lower_dim + fresh_dim, out_dim)

    def forward(self, lower_output: torch.Tensor, fresh_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lower_output: Output tensor from sub-model m [B, L, lower_dim]
            fresh_embedding: Fresh input embedding for sub-model m+1 [B, L, fresh_dim]
        Returns:
            Concatenated, rescaled, and projected input for sub-model m+1 [B, L, out_dim]
        """
        # 1. Compute L2 Norms along feature dimension (D)
        lower_norm = torch.norm(lower_output, p=2, dim=-1, keepdim=True) + self.eps
        fresh_norm = torch.norm(fresh_embedding, p=2, dim=-1, keepdim=True) + self.eps

        # 2. Rescale lower output to match fresh embedding norm magnitude
        rescaled_lower = lower_output * (fresh_norm / lower_norm)

        # 3. Concatenate fresh embedding and rescaled lower output
        concat_state = torch.cat([fresh_embedding, rescaled_lower], dim=-1)

        # 4. Linear projection to out_dim
        return self.proj(concat_state)
