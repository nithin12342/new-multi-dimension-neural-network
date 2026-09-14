"""
Package: src.engine
Canonical shortcut forwarding to the training orchestrator, early warning monitor, model, and sanitization boundary.
Uses dynamic lazy loading for orchestrator/training_loop to prevent circular import locks during initialization.
"""

from src.engine.autograd import to_clean_scalar
from src.engine.trainer import ProductionTrainerHook
from src.engine.monitor import EarlyWarningMonitor
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

def __getattr__(name: str):
    if name in ("ParadigmTrainingOrchestrator", "MultimodalNFMNet", "train_multi_stream"):
        import src.application.orchestrator.training_loop as tl
        return getattr(tl, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(__all__)
