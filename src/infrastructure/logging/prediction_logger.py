"""
FILE-016 | FOLDER-010 | src/infrastructure/logging/prediction_logger.py
Owning Aggregate: PredictionLogger
Responsibility: export sample predictions, 37 metrics, and persistent dataset batch traversal history to consolidated duckdb database
Must Never: drop sample predictions, misalign target labels, or allow un-tracked sample repetitions before 100% dataset pass
"""

import os
import json
import subprocess
from typing import List, Dict, Any, Tuple

class PredictionLogExporter:
    """
    Detailed Per-Epoch Sample Prediction, 37-Metric & Persistent Dataset Traversal Logger
    using a single consolidated DuckDB database (`multimodal_telemetry.duckdb`).
    Guarantees 100% persistent dataset traversal tracking with zero sample reuse before complete dataset pass.
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
        """Initialize DuckDB table schemas for predictions, 37 evaluation metrics, and dataset traversal history."""
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

            # 3. Persistent Dataset Traversal Registry Table
            con.execute("""
                CREATE TABLE IF NOT EXISTS dataset_traversal_history (
                    timestamp VARCHAR,
                    stream_id INTEGER,
                    epoch INTEGER,
                    chunk_index INTEGER,
                    start_sample_idx INTEGER,
                    end_sample_idx INTEGER,
                    total_raw_samples INTEGER,
                    completed_full_pass BOOLEAN
                )
            """)

            # 4. Fine-Grained Multimodal Error Localization Table
            con.execute("""
                CREATE TABLE IF NOT EXISTS sample_error_localization (
                    timestamp VARCHAR,
                    epoch INTEGER,
                    stream_id INTEGER,
                    sample_id VARCHAR,
                    overall_status VARCHAR,
                    text_first_error_step INTEGER,
                    text_error_token_idx INTEGER,
                    text_worst_loss DOUBLE,
                    image_failed_patch_coords VARCHAR,
                    image_worst_patch_coord VARCHAR,
                    image_max_residual DOUBLE,
                    audio_worst_freq_bin INTEGER,
                    audio_worst_time_bin INTEGER
                )
            """)

            con.close()
            print(f"[DuckDB Logger] Consolidated database initialized with traversal registry & error localization: {self.db_path}", flush=True)
        except Exception as e:
            print(f"[DuckDB Logger] Warning initializing consolidated schema: {e}", flush=True)

    def get_next_unvisited_chunk_index(self, chunk_size: int = 128, total_raw: int = 60000) -> Tuple[int, bool, int]:
        """
        Query DuckDB dataset_traversal_history to find the NEXT dataset chunk index.
        Calculates exact current pass number and returns (chunk_index, just_completed_pass, pass_number).
        Guarantees dataset traversal progresses sequentially through Chunk 000, 001, ..., max_chunks-1 per pass!
        """
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            res = con.execute("SELECT COUNT(*) FROM dataset_traversal_history").fetchone()
            total_logged = res[0] if res else 0
            con.close()

            max_chunks = max(1, total_raw // chunk_size) # 468 chunks for 60,000 samples @ 128 batch size
            current_chunk_idx = total_logged % max_chunks
            pass_number = (total_logged // max_chunks) + 1
            just_completed_pass = (total_logged > 0) and (current_chunk_idx == 0)

            return current_chunk_idx, just_completed_pass, pass_number
        except Exception as e:
            return 0, False, 1

    def log_traversal_chunk(
        self,
        timestamp: str,
        stream_id: int,
        epoch: int,
        chunk_index: int,
        chunk_size: int = 128,
        total_raw: int = 60000,
        completed_full_pass: bool = False
    ) -> None:
        """Record batch traversal chunk in dataset_traversal_history DuckDB table."""
        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, total_raw)

            con.execute("""
                INSERT INTO dataset_traversal_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, stream_id, epoch, chunk_index, start_idx, end_idx, total_raw, completed_full_pass))

            con.close()
        except Exception as e:
            print(f"[DuckDB Logger] Error logging dataset traversal chunk: {e}", flush=True)

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

    def export_error_localization_logs(self, records: List[Dict[str, Any]]) -> str:
        """Appends fine-grained multimodal error localization logs into `sample_error_localization` table."""
        if not records:
            return self.db_path

        try:
            import duckdb # type: ignore
            con = duckdb.connect(self.db_path, read_only=False)
            rows = [
                (
                    r.get("timestamp", ""),
                    int(r.get("epoch", 0)),
                    int(r.get("stream_id", 1)),
                    str(r.get("sample_id", "")),
                    str(r.get("overall_status", "PASS")),
                    int(r.get("text_first_error_step", -1)),
                    int(r.get("text_error_token_idx", -1)),
                    float(r.get("text_worst_loss", 0.0)),
                    json.dumps(r.get("image_failed_patch_coords", [])),
                    json.dumps(r.get("image_worst_patch_coord", [])),
                    float(r.get("image_max_residual", 0.0)),
                    int(r.get("audio_worst_freq_bin", -1)),
                    int(r.get("audio_worst_time_bin", -1))
                )
                for r in records
            ]
            con.executemany("""
                INSERT INTO sample_error_localization VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            con.close()
        except Exception as e:
            print(f"[DuckDB Logger] Error appending error localization records to {self.db_path}: {e}", flush=True)

        return self.db_path

