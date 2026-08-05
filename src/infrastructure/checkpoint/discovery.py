"""
FILE-014 | FOLDER-009 | src/infrastructure/checkpoint/discovery.py
Owning Aggregate: CheckpointDiscovery
Responsibility: scan drive recursively and validate newest checkpoint
Must Never: load corrupted or partial checkpoint files
"""

import os
from typing import Optional, List, Dict
import torch
import safetensors.torch # type: ignore
from safetensors import safe_open # type: ignore

class CheckpointDiscoveryScanner:
    """
    Automatic SafeTensors Checkpoint Discovery Engine.
    Recursively scans Google Drive for `.safetensors` weight files, sorts by modification time,
    validates file integrity, and returns the newest valid checkpoint path for seamless resume.
    """

    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir

    def scan_drive_for_checkpoints(self, model_id: int) -> List[str]:
        """Recursively scan Google Drive model sub-folder for all existing `.safetensors` checkpoint files."""
        model_dir = os.path.join(self.checkpoints_dir, f"model_{model_id:02d}")
        if not os.path.exists(model_dir):
            return []

        ckpt_files = []
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".safetensors") or f.endswith(".pt") or f.endswith(".ckpt"):
                    ckpt_files.append(os.path.join(root, f))

        # Sort by modification time descending
        ckpt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return ckpt_files

    def validate_checkpoint_integrity(self, checkpoint_path: str) -> bool:
        """Validate `.safetensors` or `.pt` checkpoint integrity."""
        try:
            if checkpoint_path.endswith(".safetensors"):
                state_dict = safetensors.torch.load_file(checkpoint_path)
                return len(state_dict) > 0
            else:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                return "model_state_dict" in ckpt
        except Exception:
            return False

    def get_latest_valid_checkpoint(self, model_id: int) -> Optional[str]:
        """Find, validate, and return newest valid `.safetensors` checkpoint path for model stream."""
        ckpts = self.scan_drive_for_checkpoints(model_id)
        for ckpt_path in ckpts:
            if self.validate_checkpoint_integrity(ckpt_path):
                return ckpt_path
        return None
