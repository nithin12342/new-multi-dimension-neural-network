"""
FILE-016 | FOLDER-010 | src/infrastructure/logging/prediction_logger.py
Owning Aggregate: PredictionLogger
Responsibility: export sample predictions and 37 metrics to consolidated duckdb database
Must Never: drop sample predictions or misalign target labels
"""

import os
import json
import subprocess
from typing import List, Dict, Any

class PredictionLogExporter:
    """
    Detailed Per-Epoch Sample Prediction & 37-Metric Logger using a single consolidated DuckDB database.
    Stores sample predictions and complete 37 evaluation metrics in `multimodal_telemetry.duckdb`
    directly on Google Drive / persistent storage.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.db_path = os.path.join(output_dir, "multimodal_telemetry.duckdb")
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
        """Initialize DuckDB table schemas for sample predictions and all 37 evaluation metrics."""
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            
            # 1. Predictions Table
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

            # 2. 37-Metrics Table
            con.execute("""
                CREATE TABLE IF NOT EXISTS epoch_metrics (
                    timestamp VARCHAR,
                    stream_id INTEGER,
                    epoch INTEGER,
                    paradigm VARCHAR,
                    acc DOUBLE, prec DOUBLE, rec DOUBLE, f1 DOUBLE, ce DOUBLE,
                    classification_report VARCHAR, confmat VARCHAR,
                    mse DOUBLE, mae DOUBLE, r2 DOUBLE, evr DOUBLE,
                    infonce DOUBLE, ntxent DOUBLE, barlow DOUBLE, vicreg DOUBLE,
                    mlmce DOUBLE, ppl DOUBLE,
                    maerecon DOUBLE, recon DOUBLE, chamfer DOUBLE,
                    linprobe DOUBLE, knn DOUBLE,
                    silhouette DOUBLE, dbi DOUBLE, chi DOUBLE, dunn DOUBLE, ari DOUBLE, nmi DOUBLE, homog DOUBLE, compl DOUBLE, vmeasure DOUBLE,
                    trust DOUBLE, cont DOUBLE,
                    loglik DOUBLE, loglik_score DOUBLE, aic DOUBLE, bic DOUBLE
                )
            """)
            con.close()
            print(f"[DuckDB Logger] Consolidated database initialized: {self.db_path}", flush=True)
        except Exception as e:
            print(f"[DuckDB Logger] Warning initializing consolidated schema: {e}", flush=True)

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
        """Appends epoch predictions directly into `predictions` table in `multimodal_telemetry.duckdb`."""
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
            print(f"[DuckDB Logger] Error appending predictions to {self.db_path}: {e}", flush=True)

        return self.db_path

    def export_epoch_metrics(
        self,
        stream_id: int,
        epoch: int,
        paradigm: str,
        timestamp: str,
        metrics: Dict[str, Any]
    ) -> str:
        """Appends all 37 calculated metrics directly into `epoch_metrics` table in `multimodal_telemetry.duckdb`."""
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)

            cls_report = f"Accuracy: {metrics.get('acc', 0.0):.4f}, F1: {metrics.get('f1', 0.0):.4f}, Precision: {metrics.get('prec', 0.0):.4f}, Recall: {metrics.get('rec', 0.0):.4f}"

            row = (
                timestamp,
                stream_id,
                epoch,
                paradigm,
                float(metrics.get("acc", 0.0)),
                float(metrics.get("prec", 0.0)),
                float(metrics.get("rec", 0.0)),
                float(metrics.get("f1", 0.0)),
                float(metrics.get("ce", 0.0)),
                cls_report,
                str(metrics.get("confmat", "TP0_FP0_FN0")),
                float(metrics.get("mse", 0.0)),
                float(metrics.get("mae", 0.0)),
                float(metrics.get("r2", 0.0)),
                float(metrics.get("evr", 0.0)),
                float(metrics.get("infonce", 0.0)),
                float(metrics.get("ntxent", 0.0)),
                float(metrics.get("barlow", 0.0)),
                float(metrics.get("vicreg", 0.0)),
                float(metrics.get("mlmce", 0.0)),
                float(metrics.get("ppl", 0.0)),
                float(metrics.get("maerecon", 0.0)),
                float(metrics.get("recon", 0.0)),
                float(metrics.get("chamfer", 0.0)),
                float(metrics.get("linprobe", 0.0)),
                float(metrics.get("knn", 0.0)),
                float(metrics.get("silhouette", 0.0)),
                float(metrics.get("dbi", 0.0)),
                float(metrics.get("chi", 0.0)),
                float(metrics.get("dunn", 0.0)),
                float(metrics.get("ari", 0.0)),
                float(metrics.get("nmi", 0.0)),
                float(metrics.get("homog", 0.0)),
                float(metrics.get("compl", 0.0)),
                float(metrics.get("vmeasure", 0.0)),
                float(metrics.get("trust", 0.0)),
                float(metrics.get("cont", 0.0)),
                float(metrics.get("loglik", 0.0)),
                float(metrics.get("loglik_score", 0.0)),
                float(metrics.get("aic", 0.0)),
                float(metrics.get("bic", 0.0))
            )

            con.execute("""
                INSERT INTO epoch_metrics VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?
                )
            """, row)

            con.close()
        except Exception as e:
            print(f"[DuckDB Logger] Error exporting epoch metrics to {self.db_path}: {e}", flush=True)

        return self.db_path
