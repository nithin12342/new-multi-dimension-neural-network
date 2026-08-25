import duckdb
import pandas as pd
import numpy as np

db_path = r'c:\Users\thela\Downloads\new multi dimension neural network\multimodal_telemetry.duckdb'
con = duckdb.connect(db_path, read_only=True)

print("=== 1. EPOCH METRICS NUMERICAL ANALYSIS (AUG 25 RUN) ===")
df_metrics = con.execute("SELECT * FROM epoch_metrics WHERE timestamp LIKE '2026-08-25%' ORDER BY epoch ASC").df()
print(f"Total Epoch Records in Aug 25 Run: {len(df_metrics)}")
if len(df_metrics) > 0:
    numeric_cols = ['acc', 'prec', 'rec', 'f1', 'ce', 'mse', 'mae', 'r2', 'evr', 'infonce', 'ntxent', 'barlow', 'vicreg', 'mlmce', 'ppl', 'maerecon', 'recon', 'chamfer', 'linprobe', 'knn', 'silhouette', 'dbi', 'chi', 'dunn', 'ari', 'nmi', 'homog', 'compl', 'vmeasure', 'trust', 'cont', 'loglik', 'loglik_score', 'aic', 'bic']
    stats = df_metrics[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    print(stats.to_string())

print("\n=== 2. SAMPLE PREDICTIONS NUMERICAL ANALYSIS (AUG 25 RUN) ===")
df_preds = con.execute("SELECT * FROM predictions WHERE timestamp LIKE '2026-08-25%' ORDER BY epoch ASC").df()
print(f"Total Predictions Logged: {len(df_preds)}")
if len(df_preds) > 0:
    print("\nConfidence Distribution:")
    print(df_preds['confidence'].describe())
    print("\nLoss Contribution Distribution:")
    print(df_preds['loss_contribution'].describe())
    print("\nPredicted Label Distribution:")
    print(df_preds['predicted'].value_counts())
    print("\nGround Truth Label Distribution:")
    print(df_preds['ground_truth'].value_counts())
    print("\nSample Correctness Rate:", df_preds['correct'].mean())

print("\n=== 3. ADVERSARIAL TRAVERSAL HISTORY AUDIT ===")
df_trav = con.execute("SELECT * FROM dataset_traversal_history WHERE timestamp LIKE '2026-08-25%' ORDER BY epoch ASC").df()
print(f"Total Traversal Logged Chunks: {len(df_trav)}")
if len(df_trav) > 0:
    print("Unique Chunks Traversed:", df_trav['chunk_index'].nunique())
    print("Chunk Index Range:", df_trav['chunk_index'].min(), "to", df_trav['chunk_index'].max())
    print("Full Pass Completed Count:", df_trav['completed_full_pass'].sum())

con.close()
