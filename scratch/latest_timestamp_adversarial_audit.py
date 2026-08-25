import duckdb
import pandas as pd
import numpy as np

db_path = r'c:\Users\thela\Downloads\new multi dimension neural network\multimodal_telemetry.duckdb'
con = duckdb.connect(db_path, read_only=True)

print("=================================================================")
print("  EXACT LATEST TIMESTAMP ADVERSARIAL & NUMERICAL DUCKDB AUDIT    ")
print("=================================================================")

# 1. Get exact latest timestamp from epoch_metrics and predictions
latest_metric_ts = con.execute("SELECT MAX(timestamp) FROM epoch_metrics").fetchone()[0]
latest_pred_ts = con.execute("SELECT MAX(timestamp) FROM predictions").fetchone()[0]
latest_trav_ts = con.execute("SELECT MAX(timestamp) FROM dataset_traversal_history").fetchone()[0]

print(f"Latest Metric Timestamp:     {latest_metric_ts}")
print(f"Latest Prediction Timestamp: {latest_pred_ts}")
print(f"Latest Traversal Timestamp:  {latest_trav_ts}\n")

# Filter strictly by the latest timestamp: latest_metric_ts
target_ts = latest_metric_ts

print(f"-----------------------------------------------------------------")
print(f"  1. EPOCH METRICS FOR LATEST TIMESTAMP: {target_ts}            ")
print(f"-----------------------------------------------------------------")

df_metrics = con.execute(f"SELECT * FROM epoch_metrics WHERE timestamp = '{target_ts}'").df()
print(f"Total Epoch Records at Timestamp '{target_ts}': {len(df_metrics)}")
if len(df_metrics) > 0:
    cols = ['stream_id', 'epoch', 'paradigm', 'acc', 'prec', 'rec', 'f1', 'ce', 'mse', 'mae', 'r2', 'evr', 'infonce', 'barlow', 'vicreg', 'ppl', 'silhouette', 'aic', 'bic']
    print(df_metrics[cols].to_string())

print(f"\n-----------------------------------------------------------------")
print(f"  2. PREDICTIONS FOR LATEST TIMESTAMP: {target_ts}                ")
print(f"-----------------------------------------------------------------")

df_preds = con.execute(f"SELECT * FROM predictions WHERE timestamp = '{target_ts}'").df()
print(f"Total Prediction Records at Timestamp '{target_ts}': {len(df_preds)}")
if len(df_preds) > 0:
    print("\nSample Records:")
    print(df_preds[['epoch', 'sample_id', 'input_file', 'ground_truth', 'predicted', 'confidence', 'correct', 'loss_contribution']].to_string())

    print("\nConfidence Summary:")
    print(df_preds['confidence'].describe())

    print("\nLoss Contribution Summary:")
    print(df_preds['loss_contribution'].describe())

    print("\nPredicted Label Counts:")
    print(df_preds['predicted'].value_counts())

    print("\nGround Truth Label Counts:")
    print(df_preds['ground_truth'].value_counts())

    print("\nAccuracy at Timestamp:", df_preds['correct'].mean())

print(f"\n-----------------------------------------------------------------")
print(f"  3. DATASET TRAVERSAL RECORD FOR LATEST TIMESTAMP: {target_ts}   ")
print(f"-----------------------------------------------------------------")

df_trav = con.execute(f"SELECT * FROM dataset_traversal_history WHERE timestamp = '{target_ts}'").df()
print(df_trav.to_string())

con.close()
