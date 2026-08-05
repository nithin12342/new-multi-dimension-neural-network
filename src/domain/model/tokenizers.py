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
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """Project image tensor [B, 3, H, W] -> [B, N_img, 256]."""
        # x_img: [B, 3, H, W]
        feat = self.proj(x_img) # [B, D, H/16, W/16]
        B, D, H_p, W_p = feat.shape
        # Flatten spatial dimensions into sequence dimension: [B, H_p * W_p, D]
        E_img = feat.flatten(2).transpose(1, 2)
        return E_img

class TextEmbeddingTokenizer(nn.Module):
    """Text Token Projection: maps text token IDs [B, S] to embeddings [B, S, 256]."""
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x_txt: torch.Tensor) -> torch.Tensor:
        """Map text tokens [B, S] -> [B, S, 256]."""
        return self.embedding(x_txt)

class MultimodalTokenFusion(nn.Module):
    """Multimodal Fusion: concatenates vision tokens and text tokens along sequence dimension."""
    def __init__(self):
        super().__init__()

    def fuse(self, E_img: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        """Concatenate E_img [B, N_img, D] and E_txt [B, S, D] -> Z^(0) [B, N_img + S, D]."""
        return torch.cat([E_img, E_txt], dim=1)

    def forward(self, E_img: torch.Tensor, E_txt: torch.Tensor) -> torch.Tensor:
        """Forward pass wrapper calling fuse."""
        return self.fuse(E_img, E_txt)
