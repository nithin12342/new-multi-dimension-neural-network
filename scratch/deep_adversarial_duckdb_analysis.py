import duckdb
import pandas as pd
import numpy as np

db_path = r'c:\Users\thela\Downloads\new multi dimension neural network\multimodal_telemetry.duckdb'
con = duckdb.connect(db_path, read_only=True)

print("=================================================================")
print("  INTENTION ENGINEERING ADVERSARIAL & NUMERICAL DUCKDB AUDIT     ")
print("=================================================================")

# 1. Inspect Tables
tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
print(f"Tables Found: {tables}\n")

for t in tables:
    count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"Table '{t}': {count} total rows")

# 2. Detailed Audit of epoch_metrics (Last Run: August 25, 2026)
print("\n-----------------------------------------------------------------")
print("  1. EPOCH METRICS NUMERICAL PROFILING (LAST RUN: AUG 25, 2026)  ")
print("-----------------------------------------------------------------")

df_metrics = con.execute("SELECT * FROM epoch_metrics WHERE timestamp LIKE '2026-08-25%' ORDER BY epoch ASC").df()
print(f"Total Epoch Records in Aug 25 Run: {len(df_metrics)}")

if len(df_metrics) > 0:
    numeric_cols = [
        'acc', 'prec', 'rec', 'f1', 'ce', 'mse', 'mae', 'r2', 'evr', 
        'infonce', 'ntxent', 'barlow', 'vicreg', 'mlmce', 'ppl', 
        'maerecon', 'recon', 'chamfer', 'linprobe', 'knn', 'silhouette', 
        'dbi', 'chi', 'dunn', 'ari', 'nmi', 'homog', 'compl', 'vmeasure', 
        'trust', 'cont', 'loglik', 'loglik_score', 'aic', 'bic'
    ]
    
    stats_df = df_metrics[numeric_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats_df['skewness'] = df_metrics[numeric_cols].skew()
    stats_df['kurtosis'] = df_metrics[numeric_cols].kurt()
    stats_df['dynamic_variance'] = df_metrics[numeric_cols].var() > 0
    print(stats_df.to_string())

# 3. Detailed Audit of predictions (Last Run: August 25, 2026)
print("\n-----------------------------------------------------------------")
print("  2. PREDICTION LOG & CONFIDENCE ADVERSARIAL AUDIT (AUG 25 RUN) ")
print("-----------------------------------------------------------------")

df_preds = con.execute("SELECT * FROM predictions WHERE timestamp LIKE '2026-08-25%' ORDER BY epoch ASC").df()
print(f"Total Predictions Logged: {len(df_preds)}")

if len(df_preds) > 0:
    print("\n[A] Confidence Score Distribution:")
    print(df_preds['confidence'].describe())
    
    print("\n[B] Sample Loss Contribution Distribution:")
    print(df_preds['loss_contribution'].describe())
    
    print("\n[C] Predicted Label Frequency (Mode Collapse Test):")
    pred_counts = df_preds['predicted'].value_counts()
    for cls_val, cnt in pred_counts.items():
        pct = (cnt / len(df_preds)) * 100
        print(f"  Class {cls_val:>2}: {cnt:>5} samples ({pct:>5.2f}%)")

    print("\n[D] Ground Truth Label Frequency:")
    gt_counts = df_preds['ground_truth'].value_counts()
    for cls_val, cnt in gt_counts.items():
        pct = (cnt / len(df_preds)) * 100
        print(f"  Class {cls_val:>2}: {cnt:>5} samples ({pct:>5.2f}%)")

    print("\n[E] Confusion Matrix Analysis (Ground Truth vs Predicted):")
    conf_mat = pd.crosstab(df_preds['ground_truth'], df_preds['predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(conf_mat.to_string())

    print("\n[F] Overall Sample Accuracy Rate:", df_preds['correct'].mean())

# 4. Detailed Audit of dataset_traversal_history
print("\n-----------------------------------------------------------------")
print("  3. DATASET TRAVERSAL REGISTRY & COVERAGE AUDIT (AUG 25 RUN)   ")
print("-----------------------------------------------------------------")

df_trav = con.execute("SELECT * FROM dataset_traversal_history WHERE timestamp LIKE '2026-08-25%' ORDER BY epoch ASC").df()
print(f"Total Traversal Logged Chunks: {len(df_trav)}")

if len(df_trav) > 0:
    unique_chunks = df_trav['chunk_index'].nunique()
    min_chunk = df_trav['chunk_index'].min()
    max_chunk = df_trav['chunk_index'].max()
    sample_coverage = (unique_chunks * 128) / 60000.0 * 100.0
    print(f"Unique Chunks Traversed: {unique_chunks} (Chunk range: {min_chunk} to {max_chunk})")
    print(f"Total Unique Samples Traversed: {unique_chunks * 128} / 60,000 ({sample_coverage:.2f}% coverage)")
    print(f"Full Pass Completed Flag Count: {df_trav['completed_full_pass'].sum()}")

# 5. Session Hardware Telemetry Audit
print("\n-----------------------------------------------------------------")
print("  4. HARDWARE & SESSION TELEMETRY AUDIT                          ")
print("-----------------------------------------------------------------")
if 'session_telemetry' in tables:
    df_sess = con.execute("SELECT * FROM session_telemetry ORDER BY start_time DESC LIMIT 5").df()
    print(df_sess.to_string())

con.close()
