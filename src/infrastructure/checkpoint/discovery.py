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

from src.infrastructure.storage.drive_manager import GoogleDriveManager
from src.domain.config.config_entities import PathConfig

class CheckpointDiscoveryScanner:
    """
    Automatic SafeTensors Checkpoint Discovery Engine.
    Recursively scans storage for `.safetensors` weight files, sorts by modification time,
    validates file integrity, and returns the newest valid checkpoint path for seamless resume.
    """

    def __init__(self, checkpoints_dir: str = None, path_config: PathConfig = PathConfig()):
        if checkpoints_dir is not None:
            self.checkpoints_dir = checkpoints_dir
        else:
            self.checkpoints_dir = GoogleDriveManager(path_config).resolve_path("checkpoints")

    def scan_drive_for_checkpoints(self, model_id: int) -> List[str]:
        """Recursively scan model sub-folder for all existing `.safetensors` checkpoint files."""
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

    def read_checkpoint_metadata(self, checkpoint_path: str) -> Dict[str, str]:
        """Read header metadata embedded inside a .safetensors checkpoint."""
        try:
            if checkpoint_path.endswith(".safetensors"):
                with safe_open(checkpoint_path, framework="pt") as f:
                    return f.metadata() or {}
            return {}
        except Exception:
            return {}


class StateDictRemapper:
    """
    Deterministic State Dictionary Remapper & Shape Validation Engine.
    Resolves legacy architectural aliases and validates tensor dimensions
    to prevent silent random re-initialization during model refactoring.
    """

    # Known parameter migration aliases across refactored architectures
    ALIAS_MAP = {
        "decoder.classifier.weight": "decoder.gyroplane.centroids",
        "model.decoder.classifier.weight": "model.decoder.gyroplane.centroids",
        "classifier.weight": "gyroplane.centroids",
        "decoder.classifier.bias": None # Bias eliminated in hyperbolic gyroplane distance
    }

    @classmethod
    def remap_and_validate(
        cls,
        loaded_state_dict: Dict[str, torch.Tensor],
        target_model: torch.nn.Module,
        strict_shapes: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Remap loaded state dict keys to match target model, validate tensor shapes,
        and raise fatal error if incompatible shapes are encountered.
        """
        target_state_dict = target_model.state_dict()
        remapped_state_dict: Dict[str, torch.Tensor] = {}

        # Strip or add 'model.' prefix if needed
        has_model_prefix_loaded = any(k.startswith("model.") for k in loaded_state_dict.keys())
        has_model_prefix_target = any(k.startswith("model.") for k in target_state_dict.keys())

        for key, tensor in loaded_state_dict.items():
            mapped_key = key

            # Apply prefix adjustment
            if has_model_prefix_loaded and not has_model_prefix_target:
                mapped_key = mapped_key[6:]
            elif not has_model_prefix_loaded and has_model_prefix_target:
                mapped_key = f"model.{mapped_key}"

            # Check explicit alias table
            if mapped_key in cls.ALIAS_MAP:
                alias = cls.ALIAS_MAP[mapped_key]
                if alias is None:
                    continue # Discard obsolete parameter
                mapped_key = alias

            if mapped_key in target_state_dict:
                target_shape = target_state_dict[mapped_key].shape
                if tensor.shape != target_shape:
                    if strict_shapes:
                        raise ValueError(
                            f"[StateDictRemapper Fatal] Shape mismatch for parameter '{mapped_key}': "
                            f"target model requires {list(target_shape)}, but checkpoint provides {list(tensor.shape)}."
                        )
                    else:
                        continue # Skip incompatible shape
                remapped_state_dict[mapped_key] = tensor
            else:
                # Key not present in target model; retained if not strictly constrained
                remapped_state_dict[mapped_key] = tensor

        return remapped_state_dict
