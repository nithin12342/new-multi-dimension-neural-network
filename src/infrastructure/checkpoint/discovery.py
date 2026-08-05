"""
FILE-014 | FOLDER-009 | src/infrastructure/checkpoint/discovery.py
Owning Aggregate: CheckpointDiscovery
Responsibility: scan drive recursively and validate newest checkpoint
Must Never: load corrupted or partial checkpoint files
"""

import os
from typing import Optional, List, Dict
import torch

class CheckpointDiscoveryScanner:
    """
    Automatic Checkpoint Discovery Engine.
    Recursively scans Google Drive, sorts checkpoints by timestamp, validates file integrity,
    and returns the newest valid checkpoint path for seamless resume.
    """

    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir

    def scan_drive_for_checkpoints(self, model_id: int) -> List[str]:
        """Recursively scan Google Drive model sub-folder for all existing checkpoint files."""
        model_dir = os.path.join(self.checkpoints_dir, f"model_{model_id:02d}")
        if not os.path.exists(model_dir):
            return []

        ckpt_files = []
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".pt") or f.endswith(".ckpt"):
                    ckpt_files.append(os.path.join(root, f))

        # Sort by modification time descending
        ckpt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return ckpt_files

    def validate_checkpoint_integrity(self, checkpoint_path: str) -> bool:
        """Validate checkpoint integrity by attempting safe torch load of required keys."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            required_keys = ["model_state_dict", "epoch"]
            for key in required_keys:
                if key not in ckpt:
                    return False
            return True
        except Exception:
            return False

    def get_latest_valid_checkpoint(self, model_id: int) -> Optional[str]:
        """Find, validate, and return newest valid checkpoint path for model stream."""
        ckpts = self.scan_drive_for_checkpoints(model_id)
        for ckpt_path in ckpts:
            if self.validate_checkpoint_integrity(ckpt_path):
                return ckpt_path
        return None
