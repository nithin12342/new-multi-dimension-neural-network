import duckdb
import pandas as pd
import numpy as np

db_path = r'c:\Users\thela\Downloads\new multi dimension neural network\multimodal_telemetry.duckdb'
con = duckdb.connect(db_path, read_only=True)

print("=================================================================")
print("  EXACT 8:55 PM RUN ADVERSARIAL & NUMERICAL DUCKDB AUDIT        ")
print("=================================================================")

# Get recent timestamps from epoch_metrics
df_ts = con.execute("SELECT DISTINCT timestamp FROM epoch_metrics ORDER BY timestamp DESC LIMIT 20").df()
print("Recent Metric Timestamps in DuckDB:")
print(df_ts.to_string())

# Find latest timestamp in epoch_metrics
latest_ts = con.execute("SELECT MAX(timestamp) FROM epoch_metrics").fetchone()[0]
print(f"\nLatest Epoch Metric Timestamp: {latest_ts}")

# Also check max timestamp in dataset_traversal_history
latest_trav_ts = con.execute("SELECT MAX(timestamp) FROM dataset_traversal_history").fetchone()[0]
print(f"Latest Traversal History Timestamp: {latest_trav_ts}")

# Audit all epoch metric records for the latest run session (e.g. today's latest records)
df_recent_metrics = con.execute(f"SELECT * FROM epoch_metrics WHERE timestamp >= '{latest_ts[:10]}' ORDER BY timestamp DESC, epoch DESC").df()
print(f"\nTotal Recent Epoch Metric Records: {len(df_recent_metrics)}")

if len(df_recent_metrics) > 0:
    numeric_cols = ['acc', 'prec', 'rec', 'f1', 'ce', 'mse', 'mae', 'r2', 'evr', 'infonce', 'barlow', 'vicreg', 'ppl', 'silhouette', 'aic', 'bic']
    print("\nRecent Run Metrics Summary:")
    print(df_recent_metrics[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].to_string())

    print("\nLatest 10 Epoch Metric Entries:")
    print(df_recent_metrics[['timestamp', 'stream_id', 'epoch', 'paradigm', 'acc', 'ce', 'mse', 'ppl', 'silhouette']].head(10).to_string())

# Audit predictions for latest timestamp
df_preds = con.execute(f"SELECT * FROM predictions WHERE timestamp = '{latest_ts}'").df()
print(f"\nTotal Prediction Records at Latest Timestamp '{latest_ts}': {len(df_preds)}")

if len(df_preds) > 0:
    print("\nSample Prediction Batch:")
    print(df_preds[['epoch', 'sample_id', 'ground_truth', 'predicted', 'confidence', 'correct', 'loss_contribution']].to_string())

    print("\nPredicted Label Distribution:")
    print(df_preds['predicted'].value_counts())

    print("\nGround Truth Label Distribution:")
    print(df_preds['ground_truth'].value_counts())

    print("\nConfidence Summary:")
    print(df_preds['confidence'].describe())

    print("\nLoss Contribution Summary:")
    print(df_preds['loss_contribution'].describe())

    print("\nAccuracy Rate at Latest Timestamp:", df_preds['correct'].mean())

# Audit dataset traversal history
df_trav = con.execute("SELECT * FROM dataset_traversal_history ORDER BY timestamp DESC LIMIT 15").df()
print(f"\nLatest 15 Dataset Traversal History Entries:")
print(df_trav.to_string())

con.close()
