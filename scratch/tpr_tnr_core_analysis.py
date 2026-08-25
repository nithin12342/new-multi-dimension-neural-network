import duckdb
import pandas as pd
import numpy as np

db_path = r'c:\Users\thela\Downloads\new multi dimension neural network\multimodal_telemetry.duckdb'
con = duckdb.connect(db_path, read_only=True)

print("=================================================================")
print("  INTENTION ENGINEERING CORE NETWORK & TPR/TNR AUDIT             ")
print("=================================================================")

# 1. Fetch all predictions from 8:55 PM run
latest_ts = con.execute("SELECT MAX(timestamp) FROM predictions").fetchone()[0]
df_preds = con.execute(f"SELECT * FROM predictions WHERE timestamp >= '2026-08-25_15-00-00'").df()

print(f"Total Predictions Audited in Recent Run: {len(df_preds)}")

if len(df_preds) > 0:
    # 2. Confusion Matrix across 10 classes
    conf_mat = pd.crosstab(df_preds['ground_truth'], df_preds['predicted'], rownames=['Actual'], colnames=['Predicted'])
    print("\nConfusion Matrix (Ground Truth vs Predicted):")
    print(conf_mat.to_string())

    # 3. Calculate Per-Class TPR (Sensitivity/Recall) and TNR (Specificity)
    all_classes = sorted(list(set(df_preds['ground_truth'].unique()).union(set(df_preds['predicted'].unique()))))
    
    tpr_list = []
    tnr_list = []
    prec_list = []
    
    print("\nPer-Class True Positive Rate (TPR) & True Negative Rate (TNR):")
    print(f"{'Class':<6} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<5} | {'TPR (Recall)':<14} | {'TNR (Specificity)':<18} | {'Precision':<10}")
    print("-" * 90)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for c in range(10):
        tp = conf_mat.loc[c, c] if (c in conf_mat.index and c in conf_mat.columns) else 0
        fp = conf_mat[c].sum() - tp if c in conf_mat.columns else 0
        fn = conf_mat.loc[c].sum() - tp if c in conf_mat.index else 0
        
        # TN = all predictions not in row c and not in col c
        total_samples = len(df_preds)
        tn = total_samples - (tp + fp + fn)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        tpr_list.append(tpr)
        tnr_list.append(tnr)
        prec_list.append(prec)

        print(f"Class {c:<2} | {tp:<5} | {fp:<5} | {fn:<5} | {tn:<5} | {tpr*100:>6.2f}%        | {tnr*100:>8.2f}%          | {prec*100:>6.2f}%")

    macro_tpr = np.mean(tpr_list)
    macro_tnr = np.mean(tnr_list)
    macro_prec = np.mean(prec_list)

    print("-" * 90)
    print(f"MACRO AVG  | Macro TPR (Sensitivity): {macro_tpr*100:.2f}% | Macro TNR (Specificity): {macro_tnr*100:.2f}% | Macro Precision: {macro_prec*100:.2f}%")

# 4. Stream Metrics Analysis across streams 1 to 6
print("\n-----------------------------------------------------------------")
print("  PER-STREAM LOSS & PARADIGM BREAKDOWN                           ")
print("-----------------------------------------------------------------")
df_metrics = con.execute("SELECT stream_id, paradigm, MAX(epoch) as max_epoch, AVG(acc) as avg_acc, AVG(ce) as avg_ce, AVG(infonce) as avg_infonce, AVG(barlow) as avg_barlow, AVG(vicreg) as avg_vicreg FROM epoch_metrics WHERE timestamp >= '2026-08-25_15-00-00' GROUP BY stream_id, paradigm ORDER BY stream_id").df()
print(df_metrics.to_string())

con.close()
