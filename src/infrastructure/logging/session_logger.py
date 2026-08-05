"""
FILE-015 | FOLDER-010 | src/infrastructure/logging/session_logger.py
Owning Aggregate: SessionLogger
Responsibility: profile hardware stats and log session telemetry
Must Never: block training execution during logging disk writes
"""

import os
import sys
import time
import json
import torch
import psutil
from typing import Dict, Any

class SessionTelemetryLogger:
    """Detailed Session & Hardware Utilization Logger for Google Colab environment."""

    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        self.session_logs_dir = os.path.join(logs_dir, "session_logs")
        os.makedirs(self.session_logs_dir, exist_ok=True)

    def log_session_start(self) -> Dict[str, Any]:
        """Record session start time, GPU specs, CUDA/PyTorch versions, and memory state."""
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"

        session_stats = {
            "session_id": f"session_{start_time}",
            "start_time": start_time,
            "gpu_name": gpu_name,
            "cuda_version": cuda_ver,
            "pytorch_version": torch.__version__,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        log_file = os.path.join(self.session_logs_dir, f"{session_stats['session_id']}_start.json")
        with open(log_file, "w") as f:
            json.dump(session_stats, f, indent=2)

        return session_stats

    def profile_hardware(self) -> Dict[str, float]:
        """Query real-time CPU, RAM, GPU memory, and GPU utilization metrics."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
            "ram_percent": psutil.virtual_memory().percent
        }
        if torch.cuda.is_available():
            stats["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
            stats["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
        return stats

    def log_session_end(self, session_start_stats: Dict[str, Any]) -> None:
        """Record final session summary, total runtime duration, and append to Drive log."""
        end_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        session_summary = {
            **session_start_stats,
            "end_time": end_time,
            "final_hardware": self.profile_hardware()
        }
        log_file = os.path.join(self.session_logs_dir, f"{session_start_stats['session_id']}_summary.json")
        with open(log_file, "w") as f:
            json.dump(session_summary, f, indent=2)
