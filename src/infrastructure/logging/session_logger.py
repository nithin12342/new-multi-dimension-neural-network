"""
FILE-015 | FOLDER-010 | src/infrastructure/logging/session_logger.py
Owning Aggregate: SessionLogger
Responsibility: profile hardware stats and log session telemetry in single consolidated duckdb database
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
    """Detailed Session & Hardware Utilization Logger storing telemetry in single `multimodal_telemetry.duckdb` database."""

    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        self.db_path = os.path.join(logs_dir, "multimodal_telemetry.duckdb")
        self._init_db_schema()

    def _init_db_schema(self) -> None:
        """Initialize DuckDB table schema for session telemetry in single consolidated database."""
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            con.execute("""
                CREATE TABLE IF NOT EXISTS session_telemetry (
                    session_id VARCHAR,
                    start_time VARCHAR,
                    end_time VARCHAR,
                    gpu_name VARCHAR,
                    cuda_version VARCHAR,
                    pytorch_version VARCHAR,
                    python_version VARCHAR,
                    cpu_count INTEGER,
                    ram_total_gb DOUBLE,
                    gpu_count INTEGER,
                    cpu_percent DOUBLE,
                    ram_used_gb DOUBLE
                )
            """)
            con.close()
            print(f"[DuckDB Logger] Consolidated session telemetry initialized: {self.db_path}", flush=True)
        except Exception as e:
            print(f"[DuckDB Logger] Warning initializing session telemetry schema at {self.db_path}: {e}", flush=True)

    def log_session_start(self) -> Dict[str, Any]:
        """Record session start time, GPU specs, CUDA/PyTorch versions, and memory state into DuckDB."""
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"

        session_stats = {
            "session_id": f"session_{start_time}",
            "start_time": start_time,
            "gpu_name": gpu_name,
            "cuda_version": cuda_ver,
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "cpu_count": psutil.cpu_count() or 1,
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            con.execute("""
                INSERT INTO session_telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_stats["session_id"],
                session_stats["start_time"],
                "",
                session_stats["gpu_name"],
                session_stats["cuda_version"],
                session_stats["pytorch_version"],
                session_stats["python_version"],
                session_stats["cpu_count"],
                session_stats["ram_total_gb"],
                session_stats["gpu_count"],
                psutil.cpu_percent(),
                round(psutil.virtual_memory().used / (1024 ** 3), 2)
            ))
            con.close()
        except Exception as e:
            print(f"[DuckDB Logger] Error logging session start to {self.db_path}: {e}", flush=True)

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
        """Record final session summary and update DuckDB session record."""
        end_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            con.execute("""
                UPDATE session_telemetry SET end_time = ? WHERE session_id = ?
            """, (end_time, session_start_stats["session_id"]))
            con.close()
            print(f"[DuckDB Logger] Session telemetry record updated: {self.db_path}", flush=True)
        except Exception as e:
            print(f"[DuckDB Logger] Error updating session end at {self.db_path}: {e}", flush=True)
