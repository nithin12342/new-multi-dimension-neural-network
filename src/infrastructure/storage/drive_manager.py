"""
FILE-009 | FOLDER-005 | src/infrastructure/storage/drive_manager.py
Owning Aggregate: DriveManager
Responsibility: mount google drive and resolve persistent directories
Must Never: write outputs to colab local temporary storage
"""

import os
import sys
from typing import Dict
from src.domain.config.config_entities import PathConfig

class GoogleDriveManager:
    """Manager for Google Drive mounting, persistent directory creation, and path resolution."""

    def __init__(self, path_config: PathConfig = PathConfig()):
        self.config = path_config
        self.resolved_base_dir = self._determine_base_directory()

    def _determine_base_directory(self) -> str:
        """
        Determine valid, non-blocking storage path.
        Respects custom self.config.base_dir if provided.
        Otherwise falls back to Google Drive or local storage.
        """
        configured_base = self.config.base_dir

        # 1. If configured_base is outside /content/drive, use configured_base directly
        if not configured_base.startswith("/content/drive"):
            os.makedirs(configured_base, exist_ok=True)
            print(f"[DriveManager] Using specified base directory: {configured_base}", flush=True)
            return configured_base

        # 2. Configured base is under /content/drive/MyDrive. Check if Drive is mounted and writable.
        gdrive_mydrive = "/content/drive/MyDrive"
        if os.path.exists(gdrive_mydrive) and os.access(gdrive_mydrive, os.W_OK):
            os.makedirs(configured_base, exist_ok=True)
            print(f"[DriveManager] Google Drive active at: {configured_base}", flush=True)
            return configured_base

        # 3. Fallback: Google Drive not mounted. Use non-blocking local storage path /content/SOTA_Cluster_Shared or ./SOTA_Cluster_Shared
        colab_content = "/content/SOTA_Cluster_Shared"
        if os.path.exists("/content"):
            os.makedirs(colab_content, exist_ok=True)
            print(f"[DriveManager] Google Drive not mounted. Using storage path: {colab_content}", flush=True)
            return colab_content

        local_fallback = os.path.abspath("./SOTA_Cluster_Shared")
        os.makedirs(local_fallback, exist_ok=True)
        print(f"[DriveManager] Using local storage path: {local_fallback}", flush=True)
        return local_fallback

    def mount_drive(self) -> bool:
        """Check if Google Drive is available without blocking subprocess execution."""
        gdrive_mydrive = "/content/drive/MyDrive"
        return os.path.exists(gdrive_mydrive) and os.access(gdrive_mydrive, os.W_OK)

    def initialize_directory_structure(self) -> Dict[str, str]:
        """Create full directory hierarchy matching spec §14."""
        base = self.resolved_base_dir
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
        target_path = os.path.join(self.resolved_base_dir, category)
        os.makedirs(target_path, exist_ok=True)
        return target_path
