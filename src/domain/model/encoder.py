"""
FILE-020 | FOLDER-002 | src/domain/model/encoder.py
Owning Aggregate: CombinedOmniEncoder
Responsibility: encode and contract higher-dimensional 5-modality inputs into lower-dimensional nested matrix sequence embeddings
Must Never: drop active modality tokens or bypass nested matrix dimension reduction
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from src.domain.model.tokenizers import (
    VisionPatchTokenizer, VideoSpatiotemporalTokenizer, TextEmbeddingTokenizer,
    AudioSpectrogramTokenizer, TabularGraphTokenizer, OmniTokenFusion
)
from src.domain.model.chebyshev import ChebyshevFunctionalBlock
from src.domain.model.trace_activation import TraceInvariantGate

class CombinedOmniEncoder(nn.Module):
    """
    Combined 5-Modality Nested Matrix Encoder Aggregate.
    Fuses high-dimensional video, image, text, audio, and tabular inputs, and applies
    Order-2 Chebyshev Functional Nested Matrix Polynomial Contractions (16x16 tiles) + Trace Scaling
    to map high-dimensional raw modal inputs into lower-dimensional sequence embeddings Z^(0) [B, N_total, 256].
    """

    def __init__(self, embed_dim: int = 256, patch_size: int = 16, vocab_size: int = 30522, num_tab_features: int = 15, tile_dim: int = 16, chebyshev_order: int = 2):
        super().__init__()
        self.vision_tokenizer = VisionPatchTokenizer(3, embed_dim, patch_size)
        self.video_tokenizer = VideoSpatiotemporalTokenizer(3, embed_dim, patch_size)
        self.text_tokenizer = TextEmbeddingTokenizer(vocab_size, embed_dim)
        self.audio_tokenizer = AudioSpectrogramTokenizer(1, embed_dim, patch_size)
        self.tabular_tokenizer = TabularGraphTokenizer(num_tab_features, embed_dim, num_tokens=4)
        self.fusion = OmniTokenFusion()

        # Encoder Nested Matrix Dimension Reduction Block
        self.encoder_chebyshev = ChebyshevFunctionalBlock(embed_dim, tile_dim, chebyshev_order)
        self.encoder_trace_gate = TraceInvariantGate(tile_dim)

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        x_vid: Optional[torch.Tensor] = None,
        x_aud: Optional[torch.Tensor] = None,
        x_tab: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode and contract high-dimensional 5-modality inputs into lower-dimensional sequence tensor Z^(0) [B, N_total, 256]."""
        E_img = self.vision_tokenizer(x_img)
        E_txt = self.text_tokenizer(x_txt)
        E_vid = self.video_tokenizer(x_vid) if x_vid is not None else None
        E_aud = self.audio_tokenizer(x_aud) if x_aud is not None else None
        E_tab = self.tabular_tokenizer(x_tab) if x_tab is not None else None

        # Fused raw token sequence
        Z_raw = self.fusion(E_img, E_txt, E_vid=E_vid, E_aud=E_aud, E_tab=E_tab)

        # Apply Encoder Nested Matrix Polynomial Contraction & Trace Activation
        Z_contracted = self.encoder_chebyshev(Z_raw)
        Z0 = self.encoder_trace_gate(Z_contracted)
        return Z0
