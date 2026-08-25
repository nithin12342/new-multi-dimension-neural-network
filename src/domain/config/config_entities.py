"""
FILE-001 | FOLDER-001 | src/domain/config/config_entities.py
Owning Aggregate: ConfigRegistry
Responsibility: define immutable 5-modality self-supervised training and model configuration data structures
Must Never: modify config values after initialization
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration data structure for MultimodalNFMNet model architecture."""
    image_channels: int = 3
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    vocab_size: int = 30522
    max_seq_len: int = 128
    embed_dim: int = 256
    tile_dim: int = 16
    num_chebyshev_blocks: int = 2
    chebyshev_order: int = 2
    num_classes: int = 10
    num_clusters: int = 10
    projection_dim: int = 128
    poincare_curvature: float = 1.0

@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for 5-modality E-MM1 dataset loading and preprocessing."""
    dataset_name: str = "encord-team/E-MM1-1M"
    batch_size: int = 16
    num_workers: int = 2
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    image_height: int = 224
    image_width: int = 224
    max_text_len: int = 64
    chunk_size: int = 128
    use_chunk_indexing: bool = True

@dataclass(frozen=True)
class PathConfig:
    """Immutable configuration for Google Drive storage paths."""
    drive_mount_point: str = "/content/drive"
    base_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared"
    datasets_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared/datasets"
    checkpoints_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared/checkpoints"
    dummy_weights_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared/dummy_weights"
    logs_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared/logs"
    metrics_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared/metrics"
    reports_dir: str = "/content/drive/MyDrive/SOTA_Cluster_Shared/reports"

@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for 6-stream unified self-supervised omni-pretraining orchestrator."""
    num_streams: int = 6
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    use_amp: bool = False
    seed: int = 42
    stream_paradigms: Tuple[str, ...] = (
        "self_supervised_ntp",
        "self_supervised_barlow",
        "self_supervised_vicreg",
        "self_supervised_mae",
        "self_supervised_dec",
        "self_supervised_omni"
    )

@dataclass(frozen=True)
class SystemConfig:
    """Master immutable system configuration binding all sub-configs."""
    version: str = "v1.0.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    path: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
