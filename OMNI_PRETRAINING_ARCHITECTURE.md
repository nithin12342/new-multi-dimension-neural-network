# 🌌 `MultimodalNFMNet-OmniPretrain` Blueprint

> **Master Architecture Specification:** Tri-Aggregate Architecture (Combined Encoder, Functional Core Model, Multi-Task Decoder) for Self-Supervised Omni-Pretraining across **5 Fundamental Modalities** (Video, Image, Text, Audio, Structured Tabular/Point-Cloud) Ingesting the Open-Source **E-MM1 Dataset (`encord-team/E-MM1-1M`)**. Powered by the **GigaTokenizer High-Throughput Tokenization Engine**.

---

## 1. Tri-Aggregate Architectural Decomposition

`MultimodalNFMNet-OmniPretrain` is explicitly decomposed into 3 core architectural aggregates:

```
[5-Modality Input Tensors] (Video, Image, Text, Audio, Tabular)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. CombinedOmniEncoder (FILE-020: src/domain/model/encoder.py)  │
│    - GigaTokenizerEngine (Zero-copy byte SIMD tokenization)     │
│    - VideoSpatiotemporalTokenizer (Conv3D 16x16x2 patches)      │
│    - VisionPatchTokenizer (Conv2D 16x16 patches)                │
│    - AudioSpectrogramTokenizer (Mel-spectrogram projection)     │
│    - TabularGraphTokenizer (Feature vector projection)          │
│    - OmniTokenFusion -> Outputs Z^(0) [B, N_total, 256]          │
└─────────────────────────────────────────────────────────────────┘
         │ Z^(0) Sequence Tensor [B, N_total, 256]
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FunctionalCoreModel (FILE-021: src/domain/model/core_model.py)│
│    - Order-2 Chebyshev Functional Matrix Blocks (16x16 Tiles)   │
│    - Trace-Invariant Activation Scaling sigma(Tr(Y)/16)         │
│    - Global Sequence Pooling z_bar                              │
│    - Poincaré Conformal Hyperbolic Chart -> z_riemannian        │
└─────────────────────────────────────────────────────────────────┘
         │ Core Outputs: Z2_sequence, z_riemannian, z_bar
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. MultiTaskOmniDecoder (FILE-022: src/domain/model/decoder.py) │
│    - NextTokenPredictionHead (Causal Thought LM Logits y_ntp)   │
│    - MaskedReconstructionHead (MAE Reconstruction X_recon)      │
│    - SSLProjectionHead (L2-normalized z_proj [B, 128])         │
│    - SupervisedClassificationHead (Classification Logits)       │
│    - SupervisedRegressionHead (Regression Scalar Output)        │
│    - DECClusteringHead (Student-t Soft Cluster Assignments q)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. E-MM1 Open-Source Dataset Breakdown (Encord Team)

| Modality | Selected Open-Source Dataset Source | Hugging Face Dataset ID | Thought Process Target |
|---|---|---|---|
| 🎬 **Video** | **Encord E-MM1 Video Streams** | `encord-team/E-MM1-1M` | Situated Causal Video Reasoning & Action Sequence Prediction |
| 🖼️ **Images** | **Encord E-MM1 Visual Diagrams** | `encord-team/E-MM1-1M` | Architecture Diagram Analysis, Visual Logic & Chart Decomposition |
| 📝 **Text** | **Encord E-MM1 Text Thought Chains** | `encord-team/E-MM1-1M` | Mathematical Deduction, Step-by-Step Thought Chains & Code Invariants |
| 🎧 **Audio** | **Encord E-MM1 Audio Waveforms** | `encord-team/E-MM1-1M` | Spoken Thought Telemetry, Acoustic Waveforms & Speech Spectrum |
| 📊 **Tabular** | **Encord E-MM1 Graph & Feature Tensors** | `encord-team/E-MM1-1M` | Relational Financial Graph Metrics, Supply Chain Topology & Anomaly Risk |

---

## 3. Storage & Detailed DuckDB Database Logging

All pretraining telemetry, E-MM1 5-modality sample predictions, 37 evaluation metrics, and GPU session stats are logged in a single compressed file on Google Drive:  
`/content/drive/MyDrive/SOTA_Cluster_Shared/logs/multimodal_telemetry.duckdb`

### DuckDB Unified Table Schemas:
- **`predictions`**: Sample ID, 5-modality input tag, ground-truth label, prediction logit, confidence score, loss contribution.
- **`epoch_metrics`**: Complete 37 evaluation metrics (Perplexity `ppl`, InfoNCE `infonce`, Barlow loss, Classification Accuracy `acc`, Silhouette score, AIC/BIC) per stream per epoch.
- **`session_telemetry`**: Hardware GPU utilization (Tesla T4), VRAM allocated, CPU percent, system RAM used.
