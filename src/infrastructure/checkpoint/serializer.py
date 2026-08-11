"""
FILE-013 | FOLDER-009 | src/infrastructure/checkpoint/serializer.py
Owning Aggregate: CheckpointSerializer
Responsibility: serialize checkpoints using safetensors file format with 37 metric metadata
Must Never: allow path separators in serialized checkpoint filenames
"""

import os
import glob
import time
import json
import torch
from typing import Dict, Any, List

import safetensors.torch # type: ignore
from safetensors import safe_open # type: ignore

from src.domain.config.config_entities import PathConfig, SystemConfig
from src.infrastructure.storage.drive_manager import GoogleDriveManager
from src.infrastructure.metrics.metric_computer import ThirtySevenMetricComputer

class CheckpointSerializer:
    """
    Consolidated SafeTensors Checkpoint Manager.
    Saves weight checkpoints directly in HuggingFace `.safetensors` format with JSON metric headers.
    Maintains ONLY ONE consolidated FP16 `.safetensors` checkpoint per stream on Google Drive or non-blocking storage.
    """

    def __init__(self, path_config: PathConfig = PathConfig()):
        self.path_config = path_config
        self.drive_mgr = GoogleDriveManager(path_config)
        self.metric_computer = ThirtySevenMetricComputer()
        self.local_dir = os.path.join(os.path.expanduser("~"), ".cache", "local_checkpoints")
        os.makedirs(self.local_dir, exist_ok=True)

    def create_dummy_weights(self, models: List[torch.nn.Module], system_config: SystemConfig) -> List[str]:
        """Create initial lightweight dummy `.safetensors` weight files in local runtime storage."""
        saved_paths = []
        dummy_base_dir = self.drive_mgr.resolve_path("dummy_weights")

        for i, model in enumerate(models, 1):
            local_target_dir = os.path.join(dummy_base_dir, f"model_{i:02d}")
            os.makedirs(local_target_dir, exist_ok=True)
            dummy_path = os.path.join(local_target_dir, "dummy_v1.safetensors")

            # Convert weights to FP16 half precision
            fp16_state = {k: v.half() if torch.is_floating_point(v) else v for k, v in model.state_dict().items()}
            metadata = {
                "model_id": str(i),
                "model_version": system_config.version,
                "dataset_version": system_config.data.dataset_name,
                "created_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "architecture": "MultimodalNFMNet",
                "is_dummy": "true"
            }
            safetensors.torch.save_file(fp16_state, dummy_path, metadata=metadata)
            saved_paths.append(dummy_path)
        return saved_paths

    def save_checkpoint(
        self,
        stream_id: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        epoch: int,
        batch_idx: int,
        metrics: Dict[str, Any],
        system_config: SystemConfig,
        is_best: bool = False
    ) -> str:
        """
        Save lightweight FP16 consolidated checkpoint in `.safetensors` format to resolved storage.
        Purges older checkpoint files in the target folder so ONLY 1 single `.safetensors` file exists per stream.
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"model_{stream_id + 1:02d}"

        # 1. Save Full Checkpoint to Local Cache Storage
        local_target_dir = os.path.join(self.local_dir, model_name)
        os.makedirs(local_target_dir, exist_ok=True)
        local_latest_path = os.path.join(local_target_dir, "latest_local.pt")

        checkpoint = {
            "stream_id": stream_id + 1,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "timestamp": timestamp,
            "model_version": system_config.version,
            "dataset_version": system_config.data.dataset_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "is_best": is_best
        }
        torch.save(checkpoint, local_latest_path)

        # 2. Export Consolidated FP16 Weights in .safetensors Format to Resolved Checkpoints Dir
        resolved_checkpoints_dir = self.drive_mgr.resolve_path("checkpoints")
        drive_target_dir = os.path.join(resolved_checkpoints_dir, model_name)
        os.makedirs(drive_target_dir, exist_ok=True)

        raw_signature_name = self.metric_computer.format_serialized_signature(
            stream_id=stream_id + 1,
            timestamp=timestamp,
            epoch=epoch,
            model_version=system_config.version,
            dataset_version=system_config.data.dataset_name,
            metrics=metrics
        )
        # Double-guard filename against slashes or path separators
        safe_filename = os.path.basename(raw_signature_name).replace(".pt", ".safetensors").replace("/", "_").replace("\\", "_")
        drive_filepath = os.path.join(drive_target_dir, safe_filename)

        # Compress state dict to FP16 half precision
        fp16_state_dict = {k: v.half() if torch.is_floating_point(v) else v for k, v in model.state_dict().items()}

        # Build string metadata dictionary for SafeTensors header
        metadata = {
            "stream_id": str(stream_id + 1),
            "epoch": str(epoch),
            "timestamp": timestamp,
            "model_version": system_config.version,
            "dataset_version": system_config.data.dataset_name,
            "metrics": json.dumps(metrics),
            "consolidated": "true",
            "is_best": "true" if is_best else "false"
        }

        # Purge ALL previous checkpoint files in this stream's target folder
        for old_file in glob.glob(os.path.join(drive_target_dir, "*")):
            try:
                os.remove(old_file)
            except Exception:
                pass

        # Save SafeTensors weight file
        safetensors.torch.save_file(fp16_state_dict, drive_filepath, metadata=metadata)
        return drive_filepath

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load and deserialize `.safetensors` or legacy `.pt` checkpoint file."""
        if checkpoint_path.endswith(".safetensors"):
            model_state_dict = safetensors.torch.load_file(checkpoint_path)
            metadata: Dict[str, str] = {}
            with safe_open(checkpoint_path, framework="pt") as f:
                raw_meta = f.metadata()
                if raw_meta:
                    metadata = raw_meta

            metrics = json.loads(metadata.get("metrics", "{}")) if "metrics" in metadata else {}
            epoch = int(metadata.get("epoch", 1))
            stream_id = int(metadata.get("stream_id", 1))

            return {
                "model_state_dict": model_state_dict,
                "epoch": epoch,
                "stream_id": stream_id,
                "metrics": metrics,
                "metadata": metadata
            }
        else:
            return torch.load(checkpoint_path, map_location="cpu")
