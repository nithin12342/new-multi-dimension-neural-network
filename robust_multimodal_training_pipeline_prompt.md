# Robust Multimodal Training Pipeline Prompt

## Objective

Develop a **complete production-grade multimodal deep learning training framework** using **open-source multimodal datasets** that supports **supervised**, **self-supervised**, and **unsupervised** learning.

The framework must be designed for **Google Colab (T4 GPU)** and optimized for maximum GPU utilization while maintaining reliability, resumability, and detailed experiment tracking.

---

# Core Requirements

## 1. Complete End-to-End Implementation

Provide the complete source code including:

- Data downloading
- Dataset preprocessing
- Data augmentation
- Model architecture
- Training pipeline
- Validation pipeline
- Testing pipeline
- Evaluation pipeline
- Visualization
- Checkpoint management
- Logging
- Google Drive integration
- Automatic recovery
- Experiment tracking

The implementation should require minimal manual intervention.

---

# 2. Open Multimodal Dataset Support

Use only publicly available datasets.

The framework should automatically download and prepare datasets.

Support multimodal data including:

- Images
- Text
- Audio
- Video (optional)
- Metadata (optional)

The dataset loader should be modular so additional datasets can easily be added.

---

# 3. Google Drive Integration

All outputs must be stored directly inside Google Drive.

This includes:

- Checkpoints
- Logs
- Metrics
- Visualizations
- Reports
- Confusion matrices
- Classification reports
- Training history
- TensorBoard logs (optional)

Nothing should be permanently stored inside Colab runtime.

---

# 4. Robust Checkpoint Mechanism

Implement a production-quality checkpoint manager.

It must automatically:

- Save checkpoints after every epoch
- Save best checkpoints
- Save latest checkpoints
- Save emergency checkpoints
- Recover after Colab disconnects
- Recover after runtime crashes
- Recover after KeyboardInterrupt
- Recover after CUDA Out-of-Memory
- Resume training automatically

---

# 5. Automatic Checkpoint Discovery

When training begins:

- Recursively search Google Drive
- Detect every previous checkpoint
- Sort by timestamp
- Validate checkpoint integrity
- Resume from the newest valid checkpoint automatically

No manual checkpoint selection should be required.

---

# 6. Checkpoint Naming Convention

Every checkpoint filename must include:

- Date
- Time
- Epoch
- Training step
- Model version
- Dataset version
- Accuracy
- Precision
- Recall
- F1 Score
- Validation Loss

Example:

```
2026-08-04_21-45-13
Epoch_032
Acc_97.84
Prec_97.63
Recall_97.58
F1_97.60
ValLoss_0.0214
Model_v12
Dataset_v5.ckpt
```

---

# 7. Metrics to Save

Every checkpoint must include all available metrics.

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Cross-Entropy Loss
- Classification Report
- Confusion Matrix

---

## Regression Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Explained Variance Ratio

---

## Contrastive / Self-Supervised Metrics

- InfoNCE Loss
- NT-Xent Loss
- Barlow Twins Loss
- VICReg Loss

---

## Language Modeling Metrics

- Masked Language Modeling Cross-Entropy Loss
- Perplexity (PPL)

---

## Reconstruction Metrics

- Masked Autoencoder Reconstruction Loss
- Reconstruction Loss
- Chamfer Distance

---

## Representation Learning Metrics

- Linear Probing Accuracy
- K-NN Classifier Accuracy

---

## Clustering Metrics

- Silhouette Coefficient
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dunn Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneity
- Completeness
- V-Measure

---

## Statistical Metrics

- Log-Likelihood
- Log-Likelihood Score
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)

---

# 8. Detailed Prediction Logging

For every epoch, generate detailed prediction logs.

Each prediction should include:

- Timestamp
- Sample ID
- Input file
- Ground truth label
- Predicted label
- Prediction confidence
- Probability distribution
- Correct / Incorrect
- Loss contribution

Store logs as:

```
epoch_predictions.csv
epoch_predictions.json
epoch_predictions.parquet
```

---

# 9. Detailed Session Logging

Every Colab session must generate detailed logs containing:

- Session start time
- Session end time
- Runtime duration
- GPU information
- CUDA version
- PyTorch version
- Dataset version
- Model version
- Git commit hash (if applicable)
- Random seed
- Memory usage
- CPU usage
- GPU utilization
- Disk usage
- Batch processing speed
- Samples per second
- Learning rate schedule
- Optimizer state

Each log entry must include precise timestamps.

---

# 10. Six Parallel Weight Files

Initialize and maintain **six independent model weight files** to maximize T4 GPU utilization.

Each weight file represents an independent training stream.

Suggested configuration:

| Weight File | Training Strategy |
|-------------|-------------------|
| Model 1 | Supervised |
| Model 2 | Self-Supervised |
| Model 3 | Unsupervised |
| Model 4 | Supervised |
| Model 5 | Self-Supervised |
| Model 6 | Unsupervised |

Each model should:

- Train independently
- Maintain separate checkpoints
- Maintain separate logs
- Maintain separate metrics
- Maintain separate optimizer state
- Maintain separate learning schedules

Training should proceed sequentially across learning paradigms for each weight file to encourage rapid generalization.

---

# 11. Dummy Weight Initialization

Before training begins:

Create six dummy model files containing only:

- Neural network architecture
- Randomly initialized weights
- Metadata
- Version information

Store them immediately in Google Drive.

Example:

```
weights/
    model_01/
        dummy_v1.pt

    model_02/
        dummy_v1.pt

    ...

    model_06/
        dummy_v1.pt
```

These files serve as initialization placeholders and simplify monitoring and recovery.

---

# 12. Automatic Weight Recovery

Implement a robust recovery mechanism that:

- Searches Google Drive recursively
- Finds the latest checkpoint for each model
- Validates checkpoint integrity
- Restores:
  - Model weights
  - Optimizer state
  - Scheduler state
  - AMP scaler state
  - Random seeds
  - Training history
  - Epoch
  - Batch index
- Continues training automatically from the most recent valid checkpoint

No manual intervention should be required.

---

# 13. Versioning

Maintain automatic version control for:

- Models
- Datasets
- Checkpoints
- Configurations
- Metrics
- Reports

Every saved artifact should include:

- Version number
- Creation timestamp
- Parent checkpoint (if applicable)

---

# 14. Logging Directory Structure

Use a well-organized directory structure in Google Drive:

```text
GoogleDrive/
│
├── datasets/
├── checkpoints/
│   ├── model_01/
│   ├── model_02/
│   ├── model_03/
│   ├── model_04/
│   ├── model_05/
│   └── model_06/
│
├── dummy_weights/
├── logs/
│   ├── session_logs/
│   ├── prediction_logs/
│   ├── training_logs/
│   ├── validation_logs/
│   └── recovery_logs/
│
├── metrics/
├── reports/
├── confusion_matrices/
├── classification_reports/
├── tensorboard/
└── visualizations/
```

---

# 15. Fault Tolerance

The framework must recover gracefully from:

- Google Colab disconnects
- Runtime resets
- CUDA Out-of-Memory errors
- Keyboard interrupts
- Power failures
- Corrupted checkpoints (fallback to previous valid checkpoint)
- Google Drive synchronization delays

---

# 16. Overall Goal

Produce a **fully automated, production-grade, fault-tolerant multimodal training framework** that:

- Uses open-source multimodal datasets
- Maximizes T4 GPU utilization
- Maintains six independent training pipelines
- Saves detailed checkpoints and metrics
- Logs every prediction with timestamps
- Automatically resumes training from the latest valid checkpoint
- Stores all artifacts in Google Drive
- Requires minimal manual intervention throughout the training lifecycle
