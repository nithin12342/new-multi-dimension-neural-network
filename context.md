# 🏗️ PROJECT CONTEXT: Multi-Domain Supply Chain & Finance AI Platform

> **Purpose of this file:** This is a comprehensive briefing document for any AI agent, developer, or collaborator. It contains everything needed to understand the project architecture, data sources, preprocessing requirements, and build a SOTA feature extraction pipeline.

---

## 1. Project Overview

A production-grade **Multi-Domain AI Platform** combining Supply Chain Management and Financial Fraud Detection. The system uses Graph Neural Networks to detect financial fraud rings, time-series models for demand forecasting, and federated learning for privacy-preserving multi-party training.

**Repository:** `https://github.com/nithin12342/multi-domain-project-finance-and-supply-chain-management-`

---

## 2. Repository Structure

```
Clean-SupplyChain-Finance/
├── backend/
│   ├── ai-ml/                    # 🧠 ALL AI/ML CODE LIVES HERE
│   │   ├── notebooks/            # Colab notebooks for training & data prep
│   │   │   ├── Data_Prep_Rebuild_Colab.ipynb   # Data preprocessing pipeline
│   │   │   └── Modular_Training_Colab.ipynb     # Master training loop
│   │   └── scripts/training/     # Modular Python training scripts
│   │       ├── config.py          # Paths (Google Drive + GitHub repo)
│   │       ├── dataloaders.py     # PyTorch DataLoaders + StandardScaler
│   │       ├── federated_trainer.py  # Training loop + Hard Sample Mining
│   │       ├── mlops_telemetry.py    # WandB + hardware monitoring
│   │       └── models.py            # Neural network architectures
│   ├── data/processed/           # 🗄️ PROCESSED CSV FILES (model inputs)
│   │   ├── structural_fraud_features.csv    # Graph features for FraudMLP
│   │   └── features_spectral_*.csv          # FFT features for NasaRulPredictor
│   ├── ai-service/               # Java Spring Boot AI microservice
│   ├── auth-service/             # Authentication microservice
│   ├── finance-service/          # Finance domain microservice
│   ├── supplychain-service/      # Supply chain domain microservice
│   ├── blockchain-service/       # Blockchain integration
│   └── iot-service/              # IoT sensor integration
├── frontend/                     # React/Next.js UI
├── k8s/                          # Kubernetes deployment manifests
├── docker-compose.yml            # Local orchestration
└── pom.xml                       # Maven parent POM
```

---

## 3. Raw Datasets (Kaggle Sources & Key Authentication)

- **Kaggle API Key:** `KGAT_c0234fe2d5a9d53f6c18baf6fbe983b4`

### 3.1 IEEE-CIS Fraud Detection
- **Kaggle:** `https://www.kaggle.com/c/ieee-fraud-detection`
- **Size:** ~590MB (train_transaction.csv ~680K rows, train_identity.csv ~144K rows)
- **Key Columns:** `TransactionID`, `isFraud`, `TransactionAmt`, `ProductCD`, `card1-6`, `addr1-2`, `P_emaildomain`, `R_emaildomain`, `C1-C14`, `D1-D15`, `M1-M9`, `V1-V339`
- **Use Case:** Primary fraud detection dataset. Build transaction graphs where accounts (card holders) are nodes and transactions are edges. Detect circular fraud rings and layered money laundering.

### 3.2 PaySim Synthetic Financial Dataset
- **Kaggle:** `https://www.kaggle.com/datasets/ealaxi/paysim1`
- **Size:** ~487MB (~6.3M rows)
- **Key Columns:** `step` (time), `type` (CASH_IN/OUT, DEBIT, PAYMENT, TRANSFER), `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- **Use Case:** Large-scale transaction network. Build directed weighted graphs where `nameOrig → nameDest` are edges, `amount` is edge weight. Detect transfer chains (A→B→C→D→A circular flows).

### 3.3 DataCo Supply Chain Dataset
- **Kaggle:** `https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`
- **Size:** ~48MB (~180K rows)
- **Key Columns:** `Order Id`, `Customer Id`, `Product Name`, `Sales`, `Order Profit Per Order`, `Shipping Mode`, `Late_delivery_risk`, `Delivery Status`, `Order Region`, `Market`, `Order Item Quantity`
- **Use Case:** Supply chain network topology. Build supplier-warehouse-customer graphs. Predict delivery risks and supply chain bottlenecks.

---

## 4. Current Model Architectures

### 4.1 FraudDetectionMLP (Binary Classification)
```python
# Input: 15 graph centrality features per transaction chunk
# Output: 1 logit (BCEWithLogitsLoss)
# Architecture: Linear(15→128) → BN → LeakyReLU → Dropout(0.3)
#             → Linear(128→64) → BN → LeakyReLU → Dropout(0.3)
#             → Linear(64→32) → BN → LeakyReLU → Dropout(0.2)
#             → Linear(32→1)
```

### 4.2 NasaRulPredictor (Regression)
```python
# Input: 120 spectral FFT features (24 sensors × 5 features)
# Output: 1 continuous RUL value (MSELoss)
# Architecture: 4 layers, GELU activation, hidden_dim=256
```

---

## 5. Current Graph Feature Extraction (NEEDS UPGRADE)

The current pipeline extracts these **15 features** per transaction chunk from a directed weighted graph:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `pagerank_max` | Highest PageRank score in the chunk's graph |
| 2 | `pagerank_mean` | Average PageRank across all nodes |
| 3 | `pagerank_variance` | Variance of PageRank distribution |
| 4 | `pagerank_entropy` | Shannon entropy of PageRank (concentration measure) |
| 5 | `cluster_coeff_avg` | Average local clustering coefficient |
| 6 | `cluster_coeff_max` | Maximum clustering coefficient |
| 7 | `cluster_coeff_std` | Std dev of clustering coefficients |
| 8 | `edge_density` | Graph density (edges / max possible edges) |
| 9 | `degree_max` | Maximum node degree (in + out) |
| 10 | `degree_mean` | Average node degree |
| 11 | `degree_std` | Std dev of degree distribution |
| 12 | `largest_wcc_ratio` | Ratio of largest weakly connected component to total |
| 13 | `amount_std` | Std dev of transaction amounts |
| 14 | `amount_skew` | Skewness of amount distribution |
| 15 | `reciprocity` | Fraction of edges with a reverse edge (A↔B) |

---

## 6. ADVANCED Feature Extraction Requirements (What Needs Building)

> **This is the critical section.** The current 15 features are a baseline. A production-grade system should extract **50-100 features** using advanced mathematical techniques. Here is what an expert agent should implement:

### 6.1 Topological Data Analysis (TDA)
- **Persistent Homology:** Compute Betti numbers (β₀, β₁, β₂) for each transaction sub-graph. Track topological "holes" that represent financial bottlenecks.
- **Persistence Barcodes:** Generate fixed-size persistence barcode vectors per chunk, encoding the birth/death of topological features across filtration thresholds.
- **Simplicial Complexes:** Extract 2-simplices (transaction triangles: A→B→C→A) and 3-simplices. Count closed-loop structures that indicate fraud rings.

### 6.2 Spectral Graph Features
- **Laplacian Eigenvalues:** Compute the top-k eigenvalues of the graph Laplacian. The spectral gap (λ₂ - λ₁) indicates network connectivity.
- **Fiedler Vector:** The eigenvector corresponding to the second smallest eigenvalue reveals the natural clustering structure.
- **Cheeger Constant:** Measures the bottleneck of information flow in the graph.

### 6.3 Centrality Measures (Beyond PageRank)
- **Betweenness Centrality:** Identifies bridge nodes that control fund flows.
- **Eigenvector Centrality:** Identifies nodes connected to other important nodes.
- **Katz Centrality:** Consider all paths (not just shortest) between nodes.
- **HITS (Hubs & Authorities):** Separate hub nodes (many outgoing) from authority nodes (many incoming).

### 6.4 Community Detection Features
- **Louvain Modularity:** Detect communities. The modularity score indicates how well the graph decomposes.
- **Number of Communities:** Count of detected communities per chunk.
- **Community Size Distribution:** Gini coefficient of community sizes (fraud rings create unusually tight clusters).

### 6.5 Temporal/Sequential Features (PaySim specific)
- **Transaction Velocity:** Number of transactions per time unit per account.
- **Amount Velocity:** Rate of change of transaction amounts.
- **Balance Anomaly Score:** Deviation of post-transaction balance from expected value.
- **Time-to-Reversal:** Time between a TRANSFER and a corresponding CASH_OUT (layering pattern).

### 6.6 Rough Path Signatures (For Time-Series Domains)
- **Log-Signatures:** Encode the entire multi-dimensional path of IoT telemetry into a fixed-length tensor without losing high-frequency data.
- **Natural Cubic Spline Interpolation:** For missing sensor data, providing smooth differentiable functions for Neural SDE integration.

---

## 7. Training Infrastructure

- **Compute:** Google Colab Free Tier (T4 GPU, 15GB VRAM, 12GB RAM)
- **MLOps:** Weights & Biases (WandB) for metric tracking
- **Checkpoints:** Google Drive at `/content/drive/MyDrive/SOTA_Cluster_Shared/`
- **Version Control:** GitHub (`master` branch)
- **Training Loop:** Federated-style with Hard Sample Mining, confusion matrices, hardware profiling

---

## 8. Execution Environment

### For Data Preprocessing:
- **GitHub Codespaces** (cloud VM with pip/conda, direct Git access)
- Download Kaggle datasets → preprocess → commit processed CSVs to repo
- No GPU needed for preprocessing

### For Model Training:
- **Google Colab** (free T4 GPU)
- Clone repo → import scripts → mount Drive → train
- Save checkpoints to Google Drive

---

## 9. Files That Must Be Produced by Preprocessing

The preprocessing pipeline must output these CSVs to `backend/data/processed/`:

```
structural_fraud_features.csv
├── 1000+ rows (one per transaction chunk)
├── 15-100 graph topology feature columns
├── chunk_id column (integer identifier)
└── isFraud column (0 or 1, derived from raw labels)
```

**Critical Rule:** The `isFraud` label MUST come from the real dataset labels, NOT random generation. Each chunk's label = `max(isFraud)` across all transactions in that chunk.

---

## 10. Quality Benchmarks

A correctly preprocessed dataset should yield these training metrics within 20 epochs:

| Metric | Minimum Acceptable | Target |
|--------|-------------------|--------|
| Train AUC | > 0.75 | > 0.85 |
| Validation AUC | > 0.70 | > 0.82 |
| Train F1 | > 0.60 | > 0.75 |
| Precision-Recall Balance | No majority-class collapse | Both > 0.60 |

---

## 11. Single Source of Truth: Multimodal Nested Functional Matrix Network (`MultimodalNFMNet`) Specification

> **Reference Source File:** `Blackwell B200 and Riemannian Networks - Google Gemini (8_4_2026 9：39：02 PM).html`  
> **Architecture Name:** `MultimodalNFMNet` (Hardware-Co-Designed Nested Functional Matrix Network on Riemannian Conformal Charts with NVIDIA Tensor Cores / Blackwell B200 / T4 Hardware Acceleration)

The `MultimodalNFMNet` is a hardware-co-designed neural network architecture optimized for matrix compute engines (NVIDIA Tensor Cores, AMD WMMA, Intel XMX). It replaces standard point-wise activation layers ($W \cdot x + b$) with Chebyshev Functional Matrix Contractions operating over atomic $16 \times 16$ tile embeddings, coupled with an in-register Trace-Invariant Activation Gate.

---

### 11.1 Complete Mathematical Formulation

#### A. Modality Tokenization & Tile Projection
Given a multimodal input tuple consisting of an image $x_{\text{img}} \in \mathbb{R}^{B \times 3 \times H \times W}$ and a text sequence $x_{\text{txt}} \in \mathbb{Z}^{B \times S}$:

1. **Vision Token Projection:** The image is projected into patch embeddings of size $16 \times 16$ via a non-overlapping 2D convolution:
   $$E_{\text{img}} = \text{Reshape}\left(\text{Conv2D}(x_{\text{img}}, W_{\text{patch}}, \text{stride}=16)\right) \in \mathbb{R}^{B \times N_{\text{img}} \times D}$$
   where $N_{\text{img}} = \frac{H \cdot W}{256}$ and $D = 256$ (divisible into 16-element sub-vectors).

2. **Text Token Projection:** Text tokens are mapped through an embedding lookup table:
   $$E_{\text{txt}} = \text{Embedding}(x_{\text{txt}}, W_{\text{vocab}}) \in \mathbb{R}^{B \times S \times D}$$

3. **Multimodal Fusion:** Tokens are concatenated along the sequence dimension:
   $$Z^{(0)} = \left[ E_{\text{img}} \; \Vert \; E_{\text{txt}} \right] \in \mathbb{R}^{B \times N \times D}, \quad \text{where } N = N_{\text{img}} + S$$

---

#### B. Functional Matrix Block (Order-2 Chebyshev Expansion)
For any intermediate layer state $Z^{(l)} \in \mathbb{R}^{B \times N \times D}$, the state tensor is reshaped into atomic $16 \times 16$ matrix blocks $X \in \mathbb{R}^{16 \times 16}$.  
Instead of standard scalar weights, the block evaluates an orthogonal Chebyshev Polynomial Basis up to degree $P=2$:

1. **Chebyshev Matrix Polynomial Bases:**
   $$T_0(X) = X$$
   $$T_1(X) = X$$
   $$T_2(X) = 2 \cdot \left( X \cdot X^T \right) - X$$

2. **Tensor Core Matrix Contraction:** The basis matrices are contracted with trainable coefficient matrices $C_0, C_1, C_2 \in \mathbb{R}^{D \times D}$:
   $$Y = \sum_{p=0}^{2} T_p(X) \cdot C_p = X \cdot C_0 + X \cdot C_1 + \left( 2 X X^T - X \right) \cdot C_2$$

3. **Trace-Invariant Activation Gate:** To preserve frame covariance without warp divergence, the activation scales output matrix elements using the normalized matrix trace $\text{Tr}(Y) = \sum_{i=1}^{16} Y_{ii}$:
   $$\text{scale}(Y) = \frac{1}{1 + \exp\left( -\frac{\text{Tr}(Y)}{16} \right)}$$
   $$Z^{(l+1)} = Y \odot \text{scale}(Y)$$

---

#### C. Metric Deformation via Conformal Riemannian Charting
The manifold $\mathcal{M}$ (Poincaré Disc/Ball) is mapped to flat 2D coordinate grid $x \in \mathbb{D}^n$ via conformal distortion:
1. **Conformal Scale Factor ($\lambda_x$):**
   $$\lambda_x = \frac{2}{1 - \|x\|^2}$$
2. **Riemannian Metric Tensor ($g_x$):**
   $$g_x = \lambda_x^2 I_n = \left(\frac{2}{1 - \|x\|^2}\right)^2 I_n$$
3. **Geodesic Distance in Conformal Chart:**
   $$d_{\mathcal{M}}(x, y) = \text{arcosh}\left(1 + 2 \frac{\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)$$
4. **Möbius Vector Addition / Transport:**
   $$x \oplus_c y = \frac{(1 + 2c \langle x, y \rangle + c \|y\|^2) x + (1 - c \|x\|^2) y}{1 + 2c \langle x, y \rangle + c^2 \|x\|^2 \|y\|^2}$$

---

#### D. Multi-Task Paradigm Heads
The output representations from the backbone $Z^{(L)} \in \mathbb{R}^{B \times N \times D}$ are aggregated via Global Token Pooling:
$$\bar{Z} = \frac{1}{N} \sum_{i=1}^{N} Z^{(L)}_i \in \mathbb{R}^{B \times D}$$

1. **Self-Supervised Head (SSL)**
   - Contrastive Projection: $z_{\text{proj}} = \mathbf{W}_2 \cdot \text{ReLU}\left( \mathbf{W}_1 \bar{Z} + b_1 \right)$
   - Masked Reconstruction: $\hat{X}_{\text{recon}} = \mathbf{W}_{\text{recon}} \cdot Z^{(L)} + b_{\text{recon}}$
   - InfoNCE Contrastive Loss:
     $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$
   - Barlow Twins Cross-Correlation Loss:
     $$\mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,j}^B)^2}}, \quad \mathcal{L}_{\text{Barlow}} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda_{\text{BT}} \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2$$
   - VICReg Loss:
     $$\mathcal{L}_{\text{VICReg}} = \lambda_{\text{vic}} s(Z^A, Z^B) + \mu_{\text{vic}} \big[v(Z^A) + v(Z^B)\big] + \nu_{\text{vic}} \big[c(Z^A) + c(Z^B)\big]$$

2. **Supervised Head**
   - Classification Logits: $y_{\text{cls}} = \mathbf{W}_{\text{cls}} \bar{Z} + b_{\text{cls}} \in \mathbb{R}^{B \times K}$
   - Regression Output: $y_{\text{reg}} = \mathbf{W}_{\text{reg}} \bar{Z} + b_{\text{reg}} \in \mathbb{R}^{B \times 1}$
   - Cross-Entropy Loss:
     $$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_{k} \log \left( \text{Softmax}(y_{\text{cls}})_k \right)$$
   - Regression Losses:
     $$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_{\text{reg}, i})^2, \quad \mathcal{L}_{\text{MAE}} = \frac{1}{B} \sum_{i=1}^B |y_i - \hat{y}_{\text{reg}, i}|, \quad R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

3. **Unsupervised Head (Deep Embedded Clustering - DEC)**
   - Soft cluster assignment $q_{ij}$ between pooled features $\bar{Z}_i$ and cluster centroid $\mu_j$ using Student's $t$-distribution kernel:
     $$q_{ij} = \frac{\left( 1 + \Vert{}\bar{Z}_i - \mu_j\Vert{}^2 \right)^{-1}}{\sum_{j'} \left( 1 + \Vert{}\bar{Z}_i - \mu_{j'}\Vert{}^2 \right)^{-1}}$$
   - KL-Divergence Regularization Loss:
     $$\mathcal{L}_{\text{DEC}} = \text{KL}(P \parallel Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}, \quad \text{where } p_{ij} = \frac{q_{ij}^2 / \sum_i q_{ij}}{\sum_{j'} (q_{ij'}^2 / \sum_i q_{ij'})}$$

---

### 11.2 Structural Block Diagram

```
===================================================================================
                                MULTIMODAL INPUTS
===================================================================================
     Image Input: [B, 3, H, W]                    Text Input: [B, S]
                 │                                            │
                 ▼                                            ▼
   ┌───────────────────────────┐                ┌───────────────────────────┐
   │    Conv2D Patch Proj      │                │     Embedding Lookup      │
   │  (Kernel=16x16, Stride=16) │                │    (Vocab=30522, D=256)   │
   └─────────────┬─────────────┘                └─────────────┬─────────────┘
                 │                                            │
                 ▼ [B, N_img, 256]                            ▼ [B, S, 256]
                 └──────────────────────┬─────────────────────┘
                                        │
                                        ▼ (Sequence Concatenation)
                            Merged Tokens: [B, N, 256]
                                        │
===================================================================================
                      SHARED FUNCTIONAL BACKBONE (STAGE 1 & 2)
===================================================================================
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │      Reshape to 16x16 Matrix Tiles        │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │         Chebyshev Basis Generator         │
                  │   T0 = X | T1 = X | T2 = 2*(X·X^T) - X    │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │      Batched 2D Tensor Core GEMM          │
                  │       Y = T0·C0 + T1·C1 + T2·C2           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │       In-Register Trace Activation        │
                  │     scale = Sigmoid( Tr(Y) / 16 )         │
                  │               Z = Y * scale               │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼ (Repeated across L=2 Functional Blocks)
                             Backbone Output: [B, N, 256]
                                        │
                                        ▼
                           Global Token Pooling: [B, 256]
                                        │
===================================================================================
                             PARADIGM OUTPUT HEADS
===================================================================================
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
             ▼                          ▼                          ▼
┌─────────────────────────┐┌─────────────────────────┐┌─────────────────────────┐
│  1. SELF-SUPERVISED     ││  2. SUPERVISED HEAD     ││  3. UNSUPERVISED HEAD   │
│  - SSL Projector MLP    ││  - Linear Classifier    ││  - Student-t Cluster q  │
│  - Recon Linear Head    ││  - Linear Regressor     ││  - Trainable Centroids  │
└────────────┬────────────┘└────────────┬────────────┘└────────────┬────────────┘
             │                          │                          │
             ▼                          ▼                          ▼
  Outputs: {z, recon}        Outputs: {logits, reg}       Outputs: {q_dist}
  Loss: InfoNCE / MSE        Loss: CE / MAE / R²          Loss: DEC KL-Div
```

---

### 11.3 Step-by-Step Layer Pipeline & Dimension Flow

| Layer Index | Component Name | Mathematical Transformation | Input Shape | Output Shape | Hardware Execution Engine |
|---|---|---|---|---|---|
| **0a** | Image Patch Tokenizer | $\text{Conv2d}(x, W, s=16)$ | `[B, 3, H, W]` | `[B, 256, H/16, W/16]` | FP16 Standard CUDA Cores |
| **0b** | Text Tokenizer | $\text{Embedding}(x, W)$ | `[B, S]` | `[B, S, 256]` | VRAM Memory Lookup |
| **1** | Token Sequence Fusion | $[E_{\text{img}} \Vert E_{\text{txt}}]$ | `[B, N_img, 256]`, `[B, S, 256]` | `[B, N, 256]` | Memory Coalesced Concatenation |
| **2** | Tile Reshaper | $\text{View}(B \cdot N, 16, 16)$ | `[B, N, 256]` | `[B*N, 16, 16]` | Register Re-indexing |
| **3** | Chebyshev Generator (Block 1) | $T_2(X) = 2(X \cdot X^T) - X$ | `[B*N, 16, 16]` | $3 \times$ `[B*N, 16, 16]` | Tensor Core FP16 Matrix Unit |
| **4** | Polynomial GEMM (Block 1) | $\sum_{p=0}^2 T_p(X) \cdot C_p$ | $3 \times$ `[B*N, 16, 16]` | `[B*N, 16, 16]` | Native `m16n16k16` Tensor Cores |
| **5** | Trace Scale Gate (Block 1) | $Y \cdot \sigma\left(\frac{\text{Tr}(Y)}{16}\right)$ | `[B*N, 16, 16]` | `[B, N, 256]` | Intra-Warp Register Shuffles |
| **6** | Functional Layer (Block 2) | Repeat Steps 2–5 | `[B, N, 256]` | `[B, N, 256]` | Tensor Core FP16 Matrix Unit |
| **7** | Global Average Pooling | $\bar{Z} = \frac{1}{N} \sum_{i=1}^N Z_i$ | `[B, N, 256]` | `[B, 256]` | Vector Reduction ALU |
| **8a** | SSL Projection Head | $W_2 \cdot \text{ReLU}(W_1 \bar{Z})$ | `[B, 256]` | `[B, 128]` | Standard GEMM / Tensor Cores |
| **8b** | Classification Head | $W_{\text{cls}} \bar{Z} + b_{\text{cls}}$ | `[B, 256]` | `[B, Num_Classes]` | Standard GEMM / Tensor Cores |
| **8c** | Clustering Head (DEC) | $(1 + \Vert{}\bar{Z} - \mu_j\Vert{}^2)^{-1}$ | `[B, 256]` | `[B, Num_Clusters]` | Vector Distance Reduction |

---

### 11.4 Multi-Stream CUDA Execution & 37-Metric Serializer Signature

#### 1. Concurrent CUDA Stream Execution
6 isolated CUDA streams run simultaneously over VRAM without GIL blocking:
$$\text{Stream } s: \mathcal{S}_s = \Big\{ \text{Model}_s, \text{Optimizer}_s, \text{GradScaler}_s \Big\}, \quad s \in \{1, 2, 3, 4, 5, 6\}$$
$$\text{CUDA\_Sync: } \text{torch.cuda.synchronize}() \implies \text{Collect Metrics } \forall s \in \{1..6\}$$

#### 2. 37-Metric Serialized Checkpoint Signature Format
All evaluation metrics are computed at epoch completion and compiled into strict checkpoint filenames saved directly to Google Drive:
```
CKPT_Stream{s}_{timestamp}_ACC-{acc}_PREC-{prec}_REC-{rec}_F1-{f1}_CE-{ce}_MSE-{mse}_MAE-{mae}_R2-{r2}_INFONCE-{infonce}_NTXENT-{ntxent}_BARLOW-{barlow}_VICREG-{vicreg}_MLMCE-{mlmce}_PPL-{ppl}_MAERECON-{maerecon}_RECON-{recon}_CHAMFER-{chamfer}_LINPROBE-{linprobe}_KNN-{knn}_SILHOUETTE-{silhouette}_DBI-{dbi}_CHI-{chi}_DUNN-{dunn}_ARI-{ari}_NMI-{nmi}_HOMOG-{homog}_COMPL-{compl}_VMESG-{vmeasure}_EVR-{evr}_TRUST-{trust}_CONT-{cont}_LOGLIK-{loglik}_LOGLIK_SCORE-{loglik_score}_AIC-{aic}_BIC-{bic}_CONFMAT-{conf_str}.pt
```

---

## 12. Strict Data Integrity Rule: No Mock Data Fallouts or Mock Fallbacks

- **Invariant:** There is only one way: authentic, real data only.
- **Enforcement:** Zero synthetic/mock data generation (`torch.randn`, `torch.randint`), no dummy tensor fallbacks, and no mock fallouts under any circumstances. All datasets must be downloaded, preprocessed, and loaded directly from real authentic dataset sources (torchvision datasets, Kaggle API via `KGAT_c0234fe2d5a9d53f6c18baf6fbe983b4`).
- **Failure Contract:** If data downloading or loading fails, the pipeline fails hard with an explicit `RuntimeError` — mock fallbacks are strictly forbidden.
