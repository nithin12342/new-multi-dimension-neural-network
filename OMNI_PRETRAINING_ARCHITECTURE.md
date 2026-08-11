# 🌌 `MultimodalNFMNet-OmniPretrain` Blueprint

> **Master Architecture Specification:** Tri-Aggregate Architecture (Encoder, Core Model, **Single Nested Matrix Decoder**) using **Order-2 Chebyshev Functional Nested Matrix Polynomial Contractions** across **Encoder**, **Core Model**, and **Decoder** for Self-Supervised Omni-Pretraining across **5 Fundamental Modalities** (Video, Image, Text, Audio, Structured Tabular/Point-Cloud) Ingesting the Open-Source **E-MM1 Dataset (`encord-team/E-MM1-1M`)**. Powered by the **GigaTokenizer High-Throughput Tokenization Engine**.

---

## 1. Nested Matrix Tri-Aggregate Architectural Breakdown

The core functionality of mapping higher dimensions into lower dimensions using **Nested Matrix Contractions** is strictly applied across **Encoder**, **Core Model**, and **Single Decoder**:

```
[5-Modality Input Tensors] (Video, Image, Text, Audio, Tabular)
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. CombinedOmniEncoder (src/domain/model/encoder.py)             │
│    - GigaTokenizerEngine (Zero-copy byte SIMD tokenization)      │
│    - 5-Modality Tokenizers (Video, Image, Text, Audio, Tabular)  │
│    - ENCODER CHEBYSHEV NESTED MATRIX CONTRACTION BLOCK (16x16)   │
│    - Trace-Invariant Activation Scaling -> Z^(0) [B, N, 256]     │
└──────────────────────────────────────────────────────────────────┘
         │ Z^(0) Sequence Tensor [B, N_total, 256]
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. FunctionalCoreModel (src/domain/model/core_model.py)          │
│    - Stage 1 & 2 Chebyshev Functional Matrix Blocks (16x16 Tiles)│
│    - Trace-Invariant Activation Scaling sigma(Tr(Y)/16)          │
│    - Poincaré Conformal Hyperbolic Chart -> z_riemannian         │
└──────────────────────────────────────────────────────────────────┘
         │ Core Outputs: Z2_sequence, z_riemannian, z_bar
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. SingleNestedMatrixDecoder (src/domain/model/decoder.py)       │
│    - DECODER CHEBYSHEV NESTED MATRIX CONTRACTION BLOCK (16x16)   │
│    - Trace-Invariant Activation Scaling                          │
│    - Single Combined Decoder Projection Engine -> Output Dict:   │
│      * ntp_logits: Causal Thought LM Logits [B, N, 30522]       │
│      * x_recon: Masked Autoencoder Reconstruction [B, N, 256]    │
│      * z_proj: L2-Normalized Contrastive Projection [B, 128]    │
│      * logits: Classification Logits [B, Num_Classes]           │
│      * reg_out: Regression Scalar Output [B, 1]                  │
│      * q_dist: Student-t Soft Cluster Assignments [B, Clusters]  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Equations for Nested Matrix Contractions

Across Encoder, Core, and Single Decoder, $16 \times 16$ tile matrix polynomial contractions compress and transform features:

1. **Polynomial Basis Calculation:**
   $$T_0(X) = X, \quad T_1(X) = X, \quad T_2(X) = 2 X X^T - X$$
2. **Matrix Core Contraction:**
   $$Y = T_0(X) \cdot C_0 + T_1(X) \cdot C_1 + T_2(X) \cdot C_2 = X C_0 + X C_1 + \left(2 X X^T - X\right) C_2$$
3. **Trace-Invariant Activation Scaling:**
   $$\text{scale}(Y) = \sigma\left( \frac{\text{Tr}(Y)}{16} \right), \quad Z_{\text{out}} = Y \odot \text{scale}(Y)$$

---

## 3. E-MM1 Open-Source Dataset Breakdown (Encord Team)

| Modality | Selected Open-Source Dataset Source | Hugging Face Dataset ID | Thought Process Target |
|---|---|---|---|
| 🎬 **Video** | **Encord E-MM1 Video Streams** | `encord-team/E-MM1-1M` | Situated Causal Video Reasoning & Action Sequence Prediction |
| 🖼️ **Images** | **Encord E-MM1 Visual Diagrams** | `encord-team/E-MM1-1M` | Architecture Diagram Analysis, Visual Logic & Chart Decomposition |
| 📝 **Text** | **Encord E-MM1 Text Thought Chains** | `encord-team/E-MM1-1M` | Mathematical Deduction, Step-by-Step Thought Chains & Code Invariants |
| 🎧 **Audio** | **Encord E-MM1 Audio Waveforms** | `encord-team/E-MM1-1M` | Spoken Thought Telemetry, Acoustic Waveforms & Speech Spectrum |
| 📊 **Tabular** | **Encord E-MM1 Graph & Feature Tensors** | `encord-team/E-MM1-1M` | Relational Financial Graph Metrics, Supply Chain Topology & Anomaly Risk |

---

## 4. Storage & Detailed DuckDB Database Logging

All pretraining telemetry, E-MM1 5-modality sample predictions, 37 evaluation metrics, and GPU session stats are logged in a single compressed file on Google Drive:  
`/content/drive/MyDrive/SOTA_Cluster_Shared/logs/multimodal_telemetry.duckdb`

### DuckDB Unified Table Schemas:
- **`predictions`**: Sample ID, 5-modality input tag, ground-truth label, prediction logit, confidence score, loss contribution.
- **`epoch_metrics`**: Complete 37 evaluation metrics (Perplexity `ppl`, InfoNCE `infonce`, Barlow loss, Classification Accuracy `acc`, Silhouette score, AIC/BIC) per stream per epoch.
- **`session_telemetry`**: Hardware GPU utilization (Tesla T4), VRAM allocated, CPU percent, system RAM used.
