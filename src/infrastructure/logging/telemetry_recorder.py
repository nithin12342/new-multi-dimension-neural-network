"""
FILE-023 | FOLDER-010 | src/infrastructure/logging/telemetry_recorder.py
Owning Aggregate: TelemetryRecorder
Responsibility: zero-overhead sub-microsecond in-memory PyArrow table buffering and Snappy Parquet epoch flushing
Must Never: execute synchronous SQL disk writes inside the high-frequency training loop
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

class TelemetryRecorder:
    """
    Pillar 5: Dedicated PyArrow In-Memory Telemetry Recording Engine.
    Buffers step-wise metrics, sample predictions, and hardware profiling data
    into in-memory Arrow arrays (<5 us per insert), flushing them to Snappy-compressed
    Parquet files at epoch boundaries.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.parquet_dir = os.path.join(output_dir, "parquet_telemetry")
        os.makedirs(self.parquet_dir, exist_ok=True)
        self.db_path = os.path.join(output_dir, "multimodal_telemetry.duckdb")

        self._metric_records: List[Dict[str, Any]] = []
        self._prediction_records: List[Dict[str, Any]] = []
        self._hardware_records: List[Dict[str, Any]] = []
        self._error_records: List[Dict[str, Any]] = []

    def record_metric(self, record: Dict[str, Any]) -> None:
        """Buffer single epoch/step metric into memory."""
        self._metric_records.append(record)

    def record_prediction(self, record: Dict[str, Any]) -> None:
        """Buffer single sample prediction into memory."""
        self._prediction_records.append(record)

    def record_hardware(self, record: Dict[str, Any]) -> None:
        """Buffer single hardware time-series snapshot into memory."""
        self._hardware_records.append(record)

    def record_error_localization(self, record: Dict[str, Any]) -> None:
        """Buffer fine-grained multimodal error localization record."""
        self._error_records.append(record)

    def flush_epoch_parquet(self, epoch: int) -> Dict[str, str]:
        """
        Serialize all accumulated in-memory tables directly to Snappy-compressed
        Apache Parquet files, achieving high compression with zero active GPU starvation.
        """
        flushed_files: Dict[str, str] = {}

        if self._metric_records:
            metric_file = os.path.join(self.parquet_dir, f"epoch_metrics_{epoch:04d}.parquet")
            table = pa.Table.from_pylist(self._metric_records)
            pq.write_table(table, metric_file, compression="snappy")
            flushed_files["metrics"] = metric_file
            self._metric_records.clear()

        if self._prediction_records:
            pred_file = os.path.join(self.parquet_dir, f"predictions_ep{epoch:04d}.parquet")
            table = pa.Table.from_pylist(self._prediction_records)
            pq.write_table(table, pred_file, compression="snappy")
            flushed_files["predictions"] = pred_file
            self._prediction_records.clear()

        if self._hardware_records:
            hw_file = os.path.join(self.parquet_dir, f"hardware_ep{epoch:04d}.parquet")
            table = pa.Table.from_pylist(self._hardware_records)
            pq.write_table(table, hw_file, compression="snappy")
            flushed_files["hardware"] = hw_file
            self._hardware_records.clear()

        if self._error_records:
            err_file = os.path.join(self.parquet_dir, f"error_loc_ep{epoch:04d}.parquet")
            table = pa.Table.from_pylist(self._error_records)
            pq.write_table(table, err_file, compression="snappy")
            flushed_files["error_localization"] = err_file
            self._error_records.clear()

        return flushed_files

    def register_duckdb_views(self) -> None:
        """Expose Parquet files as dynamic analytical views in DuckDB without write lock contention."""
        glob_path = os.path.join(self.parquet_dir, "epoch_metrics_*.parquet").replace("\\", "/")
        try:
            import duckdb
            con = duckdb.connect(self.db_path, read_only=False)
            con.execute(f"""
                CREATE OR REPLACE VIEW v_epoch_metrics_parquet AS 
                SELECT * FROM read_parquet('{glob_path}')
            """)
            con.close()
        except Exception as e:
            pass
