"""
Package: src.engine
Canonical shortcut forwarding to the training orchestrator, early warning monitor, model, and sanitization boundary.
"""

from src.engine.autograd import to_clean_scalar
from src.engine.trainer import ProductionTrainerHook
from src.engine.monitor import EarlyWarningMonitor
from src.application.orchestrator.training_loop import (
    ParadigmTrainingOrchestrator,
    MultimodalNFMNet,
    train_multi_stream,
)
from src.application.fault_tolerance.recovery_manager import FaultToleranceManager

__all__ = [
    "to_clean_scalar",
    "ProductionTrainerHook",
    "EarlyWarningMonitor",
    "ParadigmTrainingOrchestrator",
    "MultimodalNFMNet",
    "train_multi_stream",
    "FaultToleranceManager",
]
