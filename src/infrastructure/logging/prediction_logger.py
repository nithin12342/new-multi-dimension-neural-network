"""
FILE-016 | FOLDER-010 | src/infrastructure/logging/prediction_logger.py
Owning Aggregate: PredictionLogger
Responsibility: export per sample predictions in compressed duckdb database
Must Never: drop sample predictions or misalign target labels
"""

import os
import json
import subprocess
from typing import List, Dict, Any

class PredictionLogExporter:
    """
    Detailed Per-Epoch Sample Prediction Logger using DuckDB columnar database storage.
    Stores all epoch predictions in a single, highly compressed `predictions.duckdb` file
    directly on Google Drive / persistent storage, appending and updating after every epoch.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.pred_dir = os.path.join(output_dir, "prediction_logs")
        os.makedirs(self.pred_dir, exist_ok=True)
        self.db_path = os.path.join(self.pred_dir, "predictions.duckdb")
        self._ensure_duckdb_installed()
        self._init_db_schema()

    def _ensure_duckdb_installed(self) -> None:
        """Ensure duckdb python package is available in the current environment."""
        try:
            import duckdb # type: ignore
        except ImportError:
            try:
                subprocess.check_call(["pip", "install", "duckdb", "--quiet"])
            except Exception:
                pass

    def _init_db_schema(self) -> None:
        """Initialize DuckDB table schema for prediction logs on persistent storage."""
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            con.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    timestamp VARCHAR,
                    epoch INTEGER,
                    sample_id VARCHAR,
                    input_file VARCHAR,
                    ground_truth VARCHAR,
                    predicted VARCHAR,
                    confidence DOUBLE,
                    prob_dist VARCHAR,
                    correct BOOLEAN,
                    loss_contribution DOUBLE
                )
            """)
            con.close()
            print(f"[DuckDB Logger] Persistent prediction database initialized: {self.db_path}", flush=True)
        except Exception as e:
            print(f"[DuckDB Logger] Warning initializing schema at {self.db_path}: {e}", flush=True)

    def record_prediction(
        self,
        timestamp: str,
        sample_id: str,
        input_file: str,
        ground_truth: Any,
        predicted: Any,
        confidence: float,
        prob_dist: List[float],
        correct: bool,
        loss_contribution: float
    ) -> Dict[str, Any]:
        """Format individual sample prediction dictionary."""
        return {
            "timestamp": timestamp,
            "sample_id": sample_id,
            "input_file": input_file,
            "ground_truth": str(ground_truth),
            "predicted": str(predicted),
            "confidence": round(float(confidence), 4),
            "prob_dist": json.dumps([round(float(p), 4) for p in prob_dist]),
            "correct": bool(correct),
            "loss_contribution": round(float(loss_contribution), 4)
        }

    def export_epoch_logs(self, epoch: int, predictions: List[Dict[str, Any]]) -> str:
        """Appends epoch predictions directly into compressed `predictions.duckdb` database on Google Drive."""
        if not predictions:
            return self.db_path

        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)

            rows = [
                (
                    p["timestamp"],
                    epoch,
                    p["sample_id"],
                    p["input_file"],
                    p["ground_truth"],
                    p["predicted"],
                    p["confidence"],
                    p.get("prob_dist", "[]") if isinstance(p.get("prob_dist"), str) else json.dumps(p.get("prob_dist", [])),
                    p["correct"],
                    p["loss_contribution"]
                )
                for p in predictions
            ]

            con.executemany("""
                INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

            con.close()
        except Exception as e:
            print(f"[DuckDB Logger] Error appending to {self.db_path}: {e}", flush=True)
            # Fallback JSON logging if DuckDB connection fails
            json_fallback = os.path.join(self.pred_dir, f"epoch_{epoch:03d}_predictions.json")
            with open(json_fallback, "w") as f:
                json.dump(predictions, f, indent=2)

        return self.db_path
