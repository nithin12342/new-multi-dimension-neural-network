"""
FILE-013 | FOLDER-009 | src/infrastructure/checkpoint/serializer.py
Owning Aggregate: CheckpointSerializer
Responsibility: serialize checkpoints with 37 metric signature filenames
Must Never: overwrite existing valid checkpoints without versioning
"""

import os
import time
import torch
from typing import Dict, Any, List
from src.domain.config.config_entities import PathConfig, SystemConfig
from src.infrastructure.metrics.metric_computer import ThirtySevenMetricComputer

class CheckpointSerializer:
    """
    Checkpoint Manager responsible for saving, loading, dummy weight initialization,
    and 37-metric serialized filename generation.
    """

    def __init__(self, path_config: PathConfig = PathConfig()):
        self.path_config = path_config
        self.metric_computer = ThirtySevenMetricComputer()

    def create_dummy_weights(self, models: List[torch.nn.Module], system_config: SystemConfig) -> List[str]:
        """Create and save 6 initial dummy weight files directly in Google Drive."""
        saved_paths = []
        for i, model in enumerate(models, 1):
            target_dir = os.path.join(self.path_config.dummy_weights_dir, f"model_{i:02d}")
            os.makedirs(target_dir, exist_ok=True)
            dummy_path = os.path.join(target_dir, "dummy_v1.pt")

            dummy_state = {
                "model_id": i,
                "model_version": system_config.version,
                "dataset_version": system_config.data.dataset_name,
                "created_at": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "architecture": "MultimodalNFMNet",
                "model_state_dict": model.state_dict(),
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
        """Serialize complete model state, optimizer state, scaler, and 37 metrics to Google Drive."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        target_dir = os.path.join(self.path_config.checkpoints_dir, f"model_{stream_id + 1:02d}")
        os.makedirs(target_dir, exist_ok=True)

        filename = self.metric_computer.format_serialized_signature(
            stream_id=stream_id + 1,
            timestamp=timestamp,
            epoch=epoch,
            model_version=system_config.version,
            dataset_version=system_config.data.dataset_name,
            metrics=metrics
        )
        filepath = os.path.join(target_dir, filename)

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
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(target_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

        latest_path = os.path.join(target_dir, "latest_model.pt")
        torch.save(checkpoint, latest_path)

        return filepath

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load and deserialize checkpoint dictionary from disk."""
        return torch.load(checkpoint_path, map_location="cpu")
