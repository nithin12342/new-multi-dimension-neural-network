"""
FILE: src/telemetry/duckdb_post_mortem.py
Owning Aggregate: OfflineAnalytics
Responsibility: Decoupled offline SQL analysis over Snappy-compressed Parquet telemetry files using DuckDB
Must Never: execute synchronous queries or locks during the active training loop
"""

import os
import glob
import duckdb
from typing import Dict, Any, List, Optional

class DuckDBPostMortem:
    """
    Decoupled offline analytical query engine for MultimodalNFMNet telemetry.
    Queries Snappy-compressed Parquet files (`telemetry_epoch_*.parquet`) post-mortem
    using vectorized DuckDB SQL without any write-lock contention on active training.
    """

    def __init__(self, telemetry_dir: str = "./telemetry_output"):
        self.telemetry_dir = telemetry_dir
        self.con = duckdb.connect(database=":memory:")
        self._register_views()

    def _register_views(self) -> None:
        """Register Parquet glob views if files exist."""
        metrics_pattern = os.path.join(self.telemetry_dir, "*metrics.parquet").replace("\\", "/")
        pred_pattern = os.path.join(self.telemetry_dir, "*predictions.parquet").replace("\\", "/")
        hw_pattern = os.path.join(self.telemetry_dir, "*hardware.parquet").replace("\\", "/")

        if glob.glob(metrics_pattern):
            self.con.execute(f"CREATE OR REPLACE VIEW v_metrics AS SELECT * FROM read_parquet('{metrics_pattern}')")
        if glob.glob(pred_pattern):
            self.con.execute(f"CREATE OR REPLACE VIEW v_predictions AS SELECT * FROM read_parquet('{pred_pattern}')")
        if glob.glob(hw_pattern):
            self.con.execute(f"CREATE OR REPLACE VIEW v_hardware AS SELECT * FROM read_parquet('{hw_pattern}')")

    def query_macro_loss_trajectory(self) -> List[Dict[str, Any]]:
        """Compute epoch-level loss statistics across all streams."""
        try:
            res = self.con.execute("""
                SELECT 
                    epoch,
                    stream,
                    COUNT(*) as num_steps,
                    ROUND(AVG(loss), 4) as avg_loss,
                    ROUND(MIN(loss), 4) as min_loss,
                    ROUND(MAX(loss), 4) as max_loss,
                    ROUND(AVG(ppl), 2) as avg_ppl,
                    ROUND(MAX(radius), 6) as max_radius
                FROM v_metrics
                GROUP BY epoch, stream
                ORDER BY epoch, stream
            """).fetchdf()
            return res.to_dict(orient="records")
        except Exception as e:
            return []

    def query_anomalies_and_divergences(self, loss_spike_threshold: float = 30.0) -> List[Dict[str, Any]]:
        """Identify any steps where loss spiked, became NaN/Inf, or Poincaré radius saturated."""
        try:
            res = self.con.execute(f"""
                SELECT 
                    epoch,
                    step,
                    stream,
                    loss,
                    ppl,
                    radius
                FROM v_metrics
                WHERE loss > {loss_spike_threshold}
                   OR isnan(loss)
                   OR isinf(loss)
                   OR radius >= 0.9999
                ORDER BY epoch, step
            """).fetchdf()
            return res.to_dict(orient="records")
        except Exception as e:
            return []

    def query_hardware_efficiency(self) -> List[Dict[str, Any]]:
        """Compute average GPU VRAM and CPU utilization per epoch."""
        try:
            res = self.con.execute("""
                SELECT 
                    epoch,
                    ROUND(AVG(gpu_vram_allocated_mb), 2) as avg_vram_mb,
                    ROUND(MAX(gpu_vram_allocated_mb), 2) as max_vram_mb,
                    ROUND(AVG(cpu_util_pct), 2) as avg_cpu_pct
                FROM v_hardware
                GROUP BY epoch
                ORDER BY epoch
            """).fetchdf()
            return res.to_dict(orient="records")
        except Exception as e:
            return []

    def query_raw(self, sql: str) -> List[Dict[str, Any]]:
        """Execute arbitrary SQL query across the offline Parquet views."""
        return self.con.execute(sql).fetchdf().to_dict(orient="records")

    def close(self) -> None:
        self.con.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DuckDB Offline Telemetry Analyzer")
    parser.add_argument("--dir", type=str, default="./telemetry_output", help="Directory containing Parquet files")
    args = parser.parse_args()

    analyzer = DuckDBPostMortem(telemetry_dir=args.dir)
    print("=== Macro Loss Trajectory ===")
    print(analyzer.query_macro_loss_trajectory())
    print("\n=== Anomalies and Divergences ===")
    print(analyzer.query_anomalies_and_divergences())
    analyzer.close()
