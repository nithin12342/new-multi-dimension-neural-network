"""
FILE-004 | FOLDER-002 | src/domain/model/tokenizers.py
Owning Aggregate: ModalityTokenizers
Responsibility: tokenize and project image and text inputs
Must Never: mix patch dimensions across sequence boundaries
"""

import torch
import torch.nn as nn

class VisionPatchTokenizer(nn.Module):
    """Vision Token Projection: projects images [B, 3, H, W] to patch embeddings [B, N_img, 256]."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """Project image tensor [B, 3, H, W] -> [B, N_img, 256]."""
        raise NotImplementedError("Stubbed for Phase 3 Code Skeleton")

class TextEmbeddingTokenizer(nn.Module):
    """Text Token Projection: maps text token IDs [B, S] to embeddings [B, S, 256]."""
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x_txt: torch.Tensor) -> torch.Tensor:
        """Map text tokens [B, S] -> [B, S, 256]."""
        raise NotImplementedError("Stubbed for Phase 3 Code Skeleton")

class MultimodalTokenFusion(nn.Module):
    """Multimodal Fusion: concatenates vision tokens and text tokens along sequence dimension."""
    def __init__(self):
        super().__init__()

    def fuse(self, E_img: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        """Concatenate E_img [B, N_img, D] and E_txt [B, S, D] -> Z^(0) [B, N, D]."""
        raise NotImplementedError("Stubbed for Phase 3 Code Skeleton")
