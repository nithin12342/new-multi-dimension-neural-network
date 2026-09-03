"""
FILE: src/engine/trainer.py
Owning Aggregate: TrainingOrchestration
Responsibility: High-level training loop integration hook connecting autograd detachment,
                pinned memory pools, and the inline EarlyWarningMonitor.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from src.engine.autograd import to_clean_scalar
from src.engine.monitor import EarlyWarningMonitor
from src.telemetry.recorder import TelemetryRecorder

class ProductionTrainerHook:
    """
    Training loop integration hook applying autograd sanitization (to_clean_scalar)
    and inline EarlyWarningMonitor checks across every training iteration.
    """

    def __init__(
        self,
        monitor: Optional[EarlyWarningMonitor] = None,
        recorder: Optional[TelemetryRecorder] = None
    ):
        self.monitor = monitor or EarlyWarningMonitor()
        self.recorder = recorder

    def process_step(
        self,
        epoch: int,
        step: int,
        stream: str,
        loss_tensor: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        raise_on_critical: bool = False
    ) -> Dict[str, Any]:
        """
        Processes step metrics by sanitizing loss through to_clean_scalar,
        computing manifold radius, and performing inline safety inspection.
        """
        loss_val = to_clean_scalar(loss_tensor)
        
        radius_val = 0.0
        if embeddings is not None:
            with torch.no_grad():
                radius_val = to_clean_scalar(torch.norm(embeddings, p=2, dim=-1).max())

        ppl_val = float(torch.exp(torch.clamp(torch.tensor(loss_val), max=7.0)).item())

        # Inline safety inspection
        monitor_result = self.monitor.inspect_step(
            epoch=epoch,
            step=step,
            stream=stream,
            loss=loss_val,
            ppl=ppl_val,
            radius=radius_val,
            raise_on_critical=raise_on_critical
        )

        # In-memory Arrow telemetry recording
        if self.recorder is not None:
            self.recorder.record_metric({
                "step": step,
                "epoch": epoch,
                "stream": stream,
                "loss": loss_val,
                "ppl": ppl_val,
                "radius": radius_val,
                "status": monitor_result["status"]
            })

        return {
            "loss": loss_val,
            "ppl": ppl_val,
            "radius": radius_val,
            "monitor": monitor_result
        }
