"""
FILE: src/telemetry/recorder.py
Owning Aggregate: TelemetryPipeline
Responsibility: In-memory Apache Arrow table aggregator and epoch-boundary Snappy Parquet flush
Must Never: perform blocking row-by-row disk I/O inside the inner training loop
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

class TelemetryRecorder:
    """
    In-memory Apache Arrow table aggregator that accumulates step metrics,
    predictions, and hardware time-series directly in host RAM, flushing
    to Snappy-compressed Parquet files once per epoch boundary.
    """

    def __init__(self, output_dir: str = "./telemetry_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._metric_records: List[Dict[str, Any]] = []
        self._prediction_records: List[Dict[str, Any]] = []
        self._hardware_records: List[Dict[str, Any]] = []

    def record_metric(self, record: Dict[str, Any]) -> None:
        """Sub-microsecond append of training step metrics in host RAM."""
        rec = dict(record)
        rec.setdefault("timestamp", time.time())
        self._metric_records.append(rec)

    def record_prediction(self, record: Dict[str, Any]) -> None:
        """Append inference prediction log in host RAM."""
        rec = dict(record)
        rec.setdefault("timestamp", time.time())
        self._prediction_records.append(rec)

    def record_hardware(self, record: Dict[str, Any]) -> None:
        """Append hardware time-series telemetry record."""
        rec = dict(record)
        rec.setdefault("timestamp", time.time())
        self._hardware_records.append(rec)

    def flush_epoch_parquet(self, epoch: int) -> Dict[str, str]:
        """
        Consolidate in-memory buffers into Arrow Tables and write directly to
        Snappy-compressed Parquet files at epoch boundaries with zero SQLite/DuckDB locks.
        """
        out_paths = {}

        if self._metric_records:
            table = pa.Table.from_pylist(self._metric_records)
            path = os.path.join(self.output_dir, f"telemetry_epoch_{epoch:03d}_metrics.parquet")
            pq.write_table(table, path, compression="snappy")
            out_paths["metrics"] = path
            self._metric_records.clear()

        if self._prediction_records:
            table = pa.Table.from_pylist(self._prediction_records)
            path = os.path.join(self.output_dir, f"telemetry_epoch_{epoch:03d}_predictions.parquet")
            pq.write_table(table, path, compression="snappy")
            out_paths["predictions"] = path
            self._prediction_records.clear()

        if self._hardware_records:
            table = pa.Table.from_pylist(self._hardware_records)
            path = os.path.join(self.output_dir, f"telemetry_epoch_{epoch:03d}_hardware.parquet")
            pq.write_table(table, path, compression="snappy")
            out_paths["hardware"] = path
            self._hardware_records.clear()

        return out_paths

    def clear(self) -> None:
        """Clear all in-memory buffers without writing."""
        self._metric_records.clear()
        self._prediction_records.clear()
        self._hardware_records.clear()

    @property
    def buffered_metric_count(self) -> int:
        return len(self._metric_records)

    @property
    def buffered_prediction_count(self) -> int:
        return len(self._prediction_records)

    @property
    def buffered_hardware_count(self) -> int:
        return len(self._hardware_records)
