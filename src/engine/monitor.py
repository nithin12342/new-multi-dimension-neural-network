"""
FILE: src/engine/monitor.py
Owning Aggregate: EarlyWarningMonitor
Responsibility: Inline Terminal Early Warning Monitor with sliding-window Z-score and moving IQR calculations
Must Never: allow silent numerical blowouts, non-finite loss compounding, or unmonitored manifold saturation
"""

import sys
import numpy as np
from typing import Dict, Any, List, Optional

class EarlyWarningMonitor:
    """
    Inline Terminal Early Warning Monitor for MultimodalNFMNet.
    Tracks step metrics, computes moving IQR and rolling Z-scores, and raises
    terminal alerts or triggers critical execution aborts if:
      1. Loss exceeds 30.0 or becomes non-finite (NaN/Inf)
      2. Perplexity (PPL) stalls or pegs (>600)
      3. Poincaré radius approaches boundary norm >= 0.9999
    """

    def __init__(
        self,
        window_size: int = 50,
        loss_spike_threshold: float = 30.0,
        ppl_stall_threshold: float = 600.0,
        radius_boundary_threshold: float = 0.9999,
        max_consecutive_spikes: int = 3,
    ):
        self.window_size = window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.ppl_stall_threshold = ppl_stall_threshold
        self.radius_boundary_threshold = radius_boundary_threshold
        self.max_consecutive_spikes = max_consecutive_spikes

        self._loss_history: List[float] = []
        self._ppl_history: List[float] = []
        self._radius_history: List[float] = []
        self._consecutive_spikes: int = 0

    def compute_z_score(self, values: List[float], current_value: float) -> float:
        """Compute rolling Z-score across recent historical values."""
        if len(values) < 5:
            return 0.0
        mean = float(np.mean(values[-self.window_size:]))
        std = float(np.std(values[-self.window_size:]))
        if std < 1e-6:
            return 0.0
        return (current_value - mean) / std

    def compute_iqr_anomaly(self, values: List[float], current_value: float) -> bool:
        """Check if current value is an outlier based on rolling 1.5 * IQR rule."""
        if len(values) < 10:
            return False
        window = values[-self.window_size:]
        q25, q75 = np.percentile(window, [25, 75])
        iqr = q75 - q25
        return (current_value > q75 + 1.5 * iqr) or (current_value < q25 - 1.5 * iqr)

    def inspect_step(
        self,
        epoch: int,
        step: int,
        stream: str,
        loss: float,
        ppl: float,
        radius: float = 0.0,
        raise_on_critical: bool = False
    ) -> Dict[str, Any]:
        """
        Inspect step metrics in real-time and print color-coded terminal alerts
        or trigger execution abort if numerical limits are breached.
        """
        status = "OK"
        alerts: List[str] = []

        # 1. Non-finite or catastrophic loss check
        if np.isnan(loss) or np.isinf(loss) or loss > self.loss_spike_threshold:
            self._consecutive_spikes += 1
            msg = (
                f"[MONITOR ALERT] Epoch {epoch:03d} | Step {step:04d} | Stream: {stream} -> "
                f"Loss surged to {loss:.4f} (Consecutive Spikes: {self._consecutive_spikes}/{self.max_consecutive_spikes})"
            )
            alerts.append(msg)
            print(msg, flush=True)

            if self._consecutive_spikes >= self.max_consecutive_spikes:
                abort_msg = (
                    f"[CRITICAL ABORT] Stream '{stream}' diverged after "
                    f"{self._consecutive_spikes} consecutive catastrophic loss spikes (Loss: {loss:.4f})!"
                )
                print(abort_msg, file=sys.stderr, flush=True)
                status = "CRITICAL_ABORT"
                if raise_on_critical:
                    raise RuntimeError(abort_msg)
            else:
                status = "WARNING_LOSS"
        else:
            if self._consecutive_spikes > 0:
                self._consecutive_spikes -= 1

        # 2. Perplexity Stall / Pegging Check
        if ppl > self.ppl_stall_threshold:
            msg = (
                f"[PPL ALERT] Epoch {epoch:03d} | Step {step:04d} | Stream: {stream} -> "
                f"Perplexity pegged at {ppl:.2f} (Threshold: {self.ppl_stall_threshold:.1f})"
            )
            alerts.append(msg)
            print(msg, flush=True)
            if status == "OK":
                status = "WARNING_PPL"

        # 3. Poincare Manifold Boundary Check
        if radius >= self.radius_boundary_threshold:
            msg = (
                f"[MANIFOLD ALERT] Epoch {epoch:03d} | Step {step:04d} | Stream: {stream} -> "
                f"Poincaré radius saturated to {radius:.6f} >= {self.radius_boundary_threshold}!"
            )
            alerts.append(msg)
            print(msg, flush=True)
            if status == "OK":
                status = "WARNING_MANIFOLD"

        # Record histories
        if not np.isnan(loss) and not np.isinf(loss):
            self._loss_history.append(float(loss))
        if not np.isnan(ppl) and not np.isinf(ppl):
            self._ppl_history.append(float(ppl))
        if not np.isnan(radius) and not np.isinf(radius):
            self._radius_history.append(float(radius))

        z_loss = self.compute_z_score(self._loss_history, loss) if not np.isnan(loss) else 0.0
        iqr_loss = self.compute_iqr_anomaly(self._loss_history, loss) if not np.isnan(loss) else False

        return {
            "status": status,
            "loss_z_score": z_loss,
            "loss_iqr_outlier": iqr_loss,
            "consecutive_spikes": self._consecutive_spikes,
            "alerts": alerts
        }

    def reset(self) -> None:
        """Reset historical window tracking."""
        self._loss_history.clear()
        self._ppl_history.clear()
        self._radius_history.clear()
        self._consecutive_spikes = 0
