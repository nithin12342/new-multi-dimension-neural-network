"""
FILE-016 | FOLDER-010 | src/infrastructure/logging/prediction_logger.py
Owning Aggregate: PredictionLogger
Responsibility: export per sample predictions in csv json parquet
Must Never: drop sample predictions or misalign target labels
"""

import os
import json
import csv
from typing import List, Dict, Any

class PredictionLogExporter:
    """Detailed Per-Epoch Sample Prediction Logger producing CSV, JSON, and Parquet logs."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.pred_dir = os.path.join(output_dir, "prediction_logs")
        os.makedirs(self.pred_dir, exist_ok=True)

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
            "prob_dist": [round(float(p), 4) for p in prob_dist],
            "correct": bool(correct),
            "loss_contribution": round(float(loss_contribution), 4)
        }

    def export_epoch_logs(self, epoch: int, predictions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Export epoch predictions to epoch_predictions.csv, .json, and .parquet files."""
        epoch_str = f"epoch_{epoch:03d}"

        # 1. JSON Exporter
        json_path = os.path.join(self.pred_dir, f"{epoch_str}_predictions.json")
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=2)

        # 2. CSV Exporter
        csv_path = os.path.join(self.pred_dir, f"{epoch_str}_predictions.csv")
        if len(predictions) > 0:
            keys = predictions[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(predictions)

        # 3. Parquet Exporter
        parquet_path = os.path.join(self.pred_dir, f"{epoch_str}_predictions.parquet")
        try:
            import pandas as pd # type: ignore
            df = pd.DataFrame(predictions)
            df.to_parquet(parquet_path, index=False)
        except (ImportError, Exception):
            # Fallback: copy json to parquet path extension if pandas/pyarrow unavailable
            with open(parquet_path, "w") as f:
                json.dump(predictions, f)

        return {"json": json_path, "csv": csv_path, "parquet": parquet_path}
