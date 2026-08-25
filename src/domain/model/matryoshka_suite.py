"""
FILE-030 | FOLDER-002 | src/domain/model/matryoshka_suite.py
Owning Aggregate: MultimodalMatryoshkaSuite
Responsibility: execute nested multi-exit 5-modality forward passes using L2 norm-rescaled inter-model junctions per Godey & Artzi (Cornell 2026)
Must Never: allow disconnected exits or un-rescaled intermediate representation concatenations
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

from src.domain.model.encoder import CombinedOmniEncoder
from src.domain.model.core_model import FunctionalCoreModel
from src.domain.model.decoder import SingleNestedMatrixDecoder
from src.domain.model.matryoshka_junction import InterModelMatryoshkaJunction

class MultimodalMatryoshkaSuite(nn.Module):
    """
    Multimodal Matryoshka Language Model Suite Architecture (Godey & Artzi, Cornell 2026).
    Nests smaller sub-models into a single 5-modality backbone trained end-to-end.
    Provides multi-exit outputs for zero-cost online distillation and instant speculative decoding.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        tile_dim: int = 16,
        chebyshev_order: int = 2,
        vocab_size: int = 30522,
        num_classes: int = 10,
        num_exits: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_exits = num_exits

        # 1. Shared 5-Modality Combined Omni-Encoder
        self.encoder = CombinedOmniEncoder(embed_dim=embed_dim, tile_dim=tile_dim, chebyshev_order=chebyshev_order)

        # 2. Nested Functional Core Models for Exit 1, Exit 2, and Master Exit M
        self.core_blocks = nn.ModuleList([
            FunctionalCoreModel(embed_dim=embed_dim, tile_dim=tile_dim, chebyshev_order=chebyshev_order)
            for _ in range(num_exits)
        ])

        # 3. Inter-Model Matryoshka Junctions with L2 Norm Rescaling & Projection
        self.junctions = nn.ModuleList([
            InterModelMatryoshkaJunction(lower_dim=embed_dim, fresh_dim=embed_dim, out_dim=embed_dim)
            for _ in range(num_exits - 1)
        ])

        # 4. Multi-Exit Single Nested Matrix Decoders
        self.decoders = nn.ModuleList([
            SingleNestedMatrixDecoder(embed_dim=embed_dim, tile_dim=tile_dim, chebyshev_order=chebyshev_order, vocab_size=vocab_size, num_classes=num_classes)
            for _ in range(num_exits)
        ])

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        x_vid: torch.Tensor = None,
        x_aud: torch.Tensor = None,
        x_tab: torch.Tensor = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward Pass for Matryoshka Suite returning output dictionaries for all nested exits.
        Returns:
            List[Dict[str, torch.Tensor]] containing decoder outputs for Exit 1, Exit 2, ..., Exit M.
        """
        # Step 1: Shared 5-Modality Token Encoding
        Z_seq = self.encoder(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab) # [B, N_total, 256]

        exit_outputs = []
        current_state = Z_seq

        for m in range(self.num_exits):
            # Step 2: Core Model Processing for Exit m
            Z_core, z_riemannian, z_bar = self.core_blocks[m](current_state)

            # Step 3: Decode Outputs for Exit m
            out_m = self.decoders[m](Z_core, z_riemannian, z_bar)
            exit_outputs.append(out_m)

            # Step 4: Inter-Model Junction to Next Exit m+1 (if applicable)
            if m < self.num_exits - 1:
                current_state = self.junctions[m](Z_core, Z_seq)

        return exit_outputs
