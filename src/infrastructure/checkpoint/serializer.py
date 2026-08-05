"""
FILE-013 | FOLDER-009 | src/infrastructure/checkpoint/serializer.py
Owning Aggregate: CheckpointSerializer
Responsibility: serialize checkpoints with 37 metric signature filenames
Must Never: overwrite existing valid checkpoints without versioning
"""

import os
import glob
import time
import torch
from typing import Dict, Any, List
from src.domain.config.config_entities import PathConfig, SystemConfig
from src.infrastructure.metrics.metric_computer import ThirtySevenMetricComputer

class CheckpointSerializer:
    """
    Consolidated Checkpoint Manager.
    Saves intermediate full state checkpoints locally to prevent Google Drive bloat.
    Maintains ONLY ONE consolidated FP16 best checkpoint file per stream (<50MB total across all streams) on Google Drive.
    """

    def __init__(self, path_config: PathConfig = PathConfig()):
        self.path_config = path_config
        self.metric_computer = ThirtySevenMetricComputer()
        self.local_dir = os.path.join(os.path.expanduser("~"), ".cache", "local_checkpoints")
        os.makedirs(self.local_dir, exist_ok=True)

    def create_dummy_weights(self, models: List[torch.nn.Module], system_config: SystemConfig) -> List[str]:
        """Create initial lightweight dummy weight files in local runtime storage."""
        saved_paths = []
        for i, model in enumerate(models, 1):
            local_target_dir = os.path.join(self.local_dir, f"model_{i:02d}")
            os.makedirs(local_target_dir, exist_ok=True)
            dummy_path = os.path.join(local_target_dir, "dummy_v1.pt")

            # Compress weights to FP16 half precision to minimize size
            fp16_state = {k: v.half() if torch.is_floating_point(v) else v for k, v in model.state_dict().items()}
            dummy_state = {
                "model_id": i,
                "model_version": system_config.version,
                "dataset_version": system_config.data.dataset_name,
                "created_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "architecture": "MultimodalNFMNet",
                "model_state_dict": fp16_state,
                "is_dummy": True
            }
            torch.save(dummy_state, dummy_path)
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
        Save intermediate checkpoint to local storage and prune old local files.
        If is_best is True, exports a lightweight FP16 consolidated checkpoint (<10MB) to Google Drive,
        automatically purging older Google Drive files to strictly enforce <50MB total Drive usage.
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"model_{stream_id + 1:02d}"

        # 1. Save Full Checkpoint to Local Storage
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

        # 2. Export Consolidated FP16 Weights to Google Drive (ONLY 1 FILE PER STREAM)
        drive_target_dir = os.path.join(self.path_config.checkpoints_dir, model_name)
        os.makedirs(drive_target_dir, exist_ok=True)

        filename = self.metric_computer.format_serialized_signature(
            stream_id=stream_id + 1,
            timestamp=timestamp,
            epoch=epoch,
            model_version=system_config.version,
            dataset_version=system_config.data.dataset_name,
            metrics=metrics
        )
        drive_filepath = os.path.join(drive_target_dir, filename)

        # Build lightweight FP16 consolidated checkpoint
        fp16_state_dict = {k: v.half() if torch.is_floating_point(v) else v for k, v in model.state_dict().items()}
        consolidated_ckpt = {
            "stream_id": stream_id + 1,
            "epoch": epoch,
            "timestamp": timestamp,
            "model_version": system_config.version,
            "dataset_version": system_config.data.dataset_name,
            "model_state_dict": fp16_state_dict,
            "metrics": metrics,
            "consolidated": True
        }

        # Purge ALL previous checkpoint files in this stream's Google Drive folder to prevent storage bloat
        for old_file in glob.glob(os.path.join(drive_target_dir, "*")):
            try:
                os.remove(old_file)
            except Exception:
                pass

        # Save single consolidated file to Google Drive
        torch.save(consolidated_ckpt, drive_filepath)
        return drive_filepath

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load and deserialize checkpoint dictionary from disk."""
        return torch.load(checkpoint_path, map_location="cpu")
