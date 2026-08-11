"""
FILE-020 | FOLDER-002 | src/domain/model/encoder.py
Owning Aggregate: CombinedOmniEncoder
Responsibility: encode 5-modality inputs using gigatokenizer engine into unified sequence embeddings
Must Never: drop active modality tokens or mix sequence dimensions
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from src.domain.model.tokenizers import (
    VisionPatchTokenizer, VideoSpatiotemporalTokenizer, TextEmbeddingTokenizer,
    AudioSpectrogramTokenizer, TabularGraphTokenizer, OmniTokenFusion
)

class CombinedOmniEncoder(nn.Module):
    """
    Combined 5-Modality Encoder Aggregate.
    Combines GigaTokenizer-accelerated text embedding, spatiotemporal video Conv3D,
    visual Conv2D patch projection, audio Mel-spectrogram projection, and tabular feature projection
    into a single unified token sequence tensor Z^(0) [B, N_total, 256].
    """

    def __init__(self, embed_dim: int = 256, patch_size: int = 16, vocab_size: int = 30522, num_tab_features: int = 15):
        super().__init__()
        self.vision_tokenizer = VisionPatchTokenizer(3, embed_dim, patch_size)
        self.video_tokenizer = VideoSpatiotemporalTokenizer(3, embed_dim, patch_size)
        self.text_tokenizer = TextEmbeddingTokenizer(vocab_size, embed_dim)
        self.audio_tokenizer = AudioSpectrogramTokenizer(1, embed_dim, patch_size)
        self.tabular_tokenizer = TabularGraphTokenizer(num_tab_features, embed_dim, num_tokens=4)
        self.fusion = OmniTokenFusion()

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        x_vid: Optional[torch.Tensor] = None,
        x_aud: Optional[torch.Tensor] = None,
        x_tab: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode 5 modalities into unified embedding sequence Z^(0) [B, N_total, 256]."""
        E_img = self.vision_tokenizer(x_img)
        E_txt = self.text_tokenizer(x_txt)
        E_vid = self.video_tokenizer(x_vid) if x_vid is not None else None
        E_aud = self.audio_tokenizer(x_aud) if x_aud is not None else None
        E_tab = self.tabular_tokenizer(x_tab) if x_tab is not None else None

        Z0 = self.fusion(E_img, E_txt, E_vid=E_vid, E_aud=E_aud, E_tab=E_tab)
        return Z0
