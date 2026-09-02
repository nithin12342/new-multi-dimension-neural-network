"""
Package: src.engine
Canonical shortcut forwarding to the training orchestrator, early warning monitor, model, and sanitization boundary.
"""

from src.application.orchestrator.training_loop import (
    ParadigmTrainingOrchestrator,
    to_clean_scalar,
    MultimodalNFMNet,
    train_multi_stream,
)
from src.application.fault_tolerance.recovery_manager import FaultToleranceManager
from src.engine.monitor import EarlyWarningMonitor

__all__ = [
    "ParadigmTrainingOrchestrator",
    "to_clean_scalar",
    "MultimodalNFMNet",
    "train_multi_stream",
    "FaultToleranceManager",
    "EarlyWarningMonitor",
]
