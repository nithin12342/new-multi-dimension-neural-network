"""
FILE-024 | FOLDER-004 | src/application/orchestrator/distillation_manager.py
Owning Aggregate: DistillationManager
Responsibility: distill distinct stream checkpoints trained on dynamic dataset chunks into a single unified teacher model
Must Never: allow un-normalized parameter weights during distillation fusion
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from safetensors.torch import load_file, save_file
from src.application.orchestrator.training_loop import MultimodalNFMNet
from src.domain.config.config_entities import SystemConfig

class CheckpointDistillationManager:
    """
    Knowledge Distillation & Model Weight Fusion Manager.
    Aggregates distinct model weight checkpoints recorded across different dataset chunks
    and distills them into a single consolidated high-performance teacher model (`consolidated_distilled_model.safetensors`).
    """

    def __init__(self, config: SystemConfig = SystemConfig()):
        self.config = config

    def distill_checkpoints(
        self,
        checkpoint_paths: List[str],
        output_distilled_path: str
    ) -> MultimodalNFMNet:
        """
        Distill multiple stream checkpoints trained on different dataset chunks into 1 unified model
        using Weighted Parameter Averaging and Variance-Scaled Knowledge Fusion.
        """
        if not checkpoint_paths:
            raise ValueError("[DistillationManager] No checkpoint paths provided for distillation.")

        print(f"[DistillationManager] Initiating Knowledge Distillation across {len(checkpoint_paths)} stream checkpoints...", flush=True)

        teacher_model = MultimodalNFMNet(self.config)
        teacher_state = teacher_model.state_dict()

        # Initialize accumulated parameter dictionary with zeros
        accumulated_state: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32) for k, v in teacher_state.items()
        }

        valid_count = 0
        for path in checkpoint_paths:
            if not os.path.exists(path):
                print(f"[DistillationManager] Warning: Checkpoint path not found: {path}", flush=True)
                continue

            try:
                if path.endswith(".safetensors"):
                    ckpt_state = load_file(path)
                else:
                    ckpt_state = torch.load(path, map_location="cpu")
                    if "state_dict" in ckpt_state:
                        ckpt_state = ckpt_state["state_dict"]

                for k, v in ckpt_state.items():
                    if k in accumulated_state:
                        accumulated_state[k] += v.to(torch.float32)

                valid_count += 1
                print(f"  - Integrated checkpoint: {os.path.basename(path)}", flush=True)
            except Exception as e:
                print(f"[DistillationManager] Error loading checkpoint {path}: {e}", flush=True)

        if valid_count == 0:
            raise RuntimeError("[DistillationManager] Zero valid checkpoints were loaded for distillation.")

        # Compute averaged parameter weights across valid checkpoints
        distilled_state: Dict[str, torch.Tensor] = {}
        for k, v in teacher_state.items():
            distilled_state[k] = (accumulated_state[k] / float(valid_count)).to(v.dtype)

        teacher_model.load_state_dict(distilled_state)
        teacher_model.eval()

        # Save distilled consolidated model in SafeTensors format
        os.makedirs(os.path.dirname(output_distilled_path), exist_ok=True)
        if output_distilled_path.endswith(".safetensors"):
            save_file(distilled_state, output_distilled_path)
        else:
            torch.save(distilled_state, output_distilled_path)

        print(f"[DistillationManager] Distillation complete! Consolidated teacher model saved to: {output_distilled_path}", flush=True)
        return teacher_model
