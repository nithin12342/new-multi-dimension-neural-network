"""
FILE-012 | FOLDER-008 | src/infrastructure/streams/stream_manager.py
Owning Aggregate: StreamManager
Responsibility: isolate 6 cuda execution streams and optimizers
Must Never: share cuda streams or scalers across models
"""

import torch
from typing import Dict, Any, List
from src.domain.config.config_entities import TrainingConfig

class SixStreamManager:
    """
    Manager for maintaining 6 isolated CUDA execution streams, optimizers, and AMP GradScalers
    to maximize T4 GPU compute utilization.
    """

    def __init__(self, config: TrainingConfig = TrainingConfig()):
        self.config = config
        self.num_streams = config.num_streams
        self.streams: List[Any] = []
        self.scalers: List[Any] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_streams(self, models: List[torch.nn.Module]) -> None:
        """Initialize 6 isolated CUDA streams, AdamW optimizers, and GradScalers."""
        assert len(models) == self.num_streams, f"Expected {self.num_streams} models, got {len(models)}"

        self.streams.clear()
        self.scalers.clear()
        self.optimizers.clear()

        for i, model in enumerate(models):
            model.to(self.device)
            # 1. Create CUDA stream if CUDA available
            if torch.cuda.is_available():
                stream = torch.cuda.Stream()
            else:
                stream = None
            self.streams.append(stream)

            # 2. Create independent AdamW optimizer per stream model
            opt = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.optimizers.append(opt)

            # 3. Create independent AMP GradScaler per stream
            enabled = self.config.use_amp and torch.cuda.is_available()
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=enabled)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=enabled)
            self.scalers.append(scaler)

    def get_stream_context(self, stream_id: int):
        """Return CUDA stream for specified stream index (0 to 5)."""
        stream = self.streams[stream_id]
        if stream is not None and torch.cuda.is_available():
            return torch.cuda.stream(stream)
        else:
            # Dummy context manager for CPU execution
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()

    def synchronize_all(self) -> None:
        """Synchronize all 6 CUDA streams before metric collection."""
        if torch.cuda.is_available():
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()
            torch.cuda.synchronize()
