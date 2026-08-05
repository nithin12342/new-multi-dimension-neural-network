"""
FILE-009 | FOLDER-005 | src/infrastructure/storage/drive_manager.py
Owning Aggregate: DriveManager
Responsibility: mount google drive and resolve persistent directories
Must Never: write outputs to colab local temporary storage
"""

import os
from typing import Dict
from src.domain.config.config_entities import PathConfig

class GoogleDriveManager:
    """Manager for Google Drive mounting, persistent directory creation, and path resolution."""

    def __init__(self, path_config: PathConfig = PathConfig()):
        self.config = path_config

    def mount_drive(self) -> bool:
        """Mount Google Drive in Colab environment if available, or fallback to local directory."""
        if os.path.exists(self.config.drive_mount_point):
            return True
        try:
            from google.colab import drive # type: ignore
            drive.mount(self.config.drive_mount_point, force_remount=False)
            return True
        except (ImportError, Exception):
            # Fallback for non-Colab or local environments
            os.makedirs(self.config.base_dir, exist_ok=True)
            return False

    def initialize_directory_structure(self) -> Dict[str, str]:
        """Create full directory hierarchy matching spec §14 on Google Drive."""
        base = self.config.base_dir
        subdirs = {
            "datasets": os.path.join(base, "datasets"),
            "checkpoints": os.path.join(base, "checkpoints"),
            "dummy_weights": os.path.join(base, "dummy_weights"),
            "logs": os.path.join(base, "logs"),
            "session_logs": os.path.join(base, "logs", "session_logs"),
            "prediction_logs": os.path.join(base, "logs", "prediction_logs"),
            "training_logs": os.path.join(base, "logs", "training_logs"),
            "validation_logs": os.path.join(base, "logs", "validation_logs"),
            "recovery_logs": os.path.join(base, "logs", "recovery_logs"),
            "metrics": os.path.join(base, "metrics"),
            "reports": os.path.join(base, "reports"),
            "confusion_matrices": os.path.join(base, "confusion_matrices"),
            "classification_reports": os.path.join(base, "classification_reports"),
            "tensorboard": os.path.join(base, "tensorboard"),
            "visualizations": os.path.join(base, "visualizations"),
        }

        # Add 6 model checkpoint sub-directories
        for i in range(1, 7):
            subdirs[f"model_{i:02d}_ckpt"] = os.path.join(base, "checkpoints", f"model_{i:02d}")
            subdirs[f"model_{i:02d}_weights"] = os.path.join(base, "dummy_weights", f"model_{i:02d}")

        for path in subdirs.values():
            os.makedirs(path, exist_ok=True)

        return subdirs

    def resolve_path(self, category: str) -> str:
        """Resolve absolute path for specified storage category."""
        dirs = self.initialize_directory_structure()
        if category in dirs:
            return dirs[category]
        target_path = os.path.join(self.config.base_dir, category)
        os.makedirs(target_path, exist_ok=True)
        return target_path
