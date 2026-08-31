---
license: mit
---

<div align="center">

# 🧠 MultimodalNFMNet

### Nested Functional Matrix Network for 5-Modality Self-Supervised Omni-Pretraining

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![SafeTensors](https://img.shields.io/badge/Weights-SafeTensors-FF6F00)](https://huggingface.co/docs/safetensors)
[![DuckDB](https://img.shields.io/badge/Telemetry-DuckDB-FFF000?logo=duckdb)](https://duckdb.org/)
[![Google Colab](https://img.shields.io/badge/Runtime-Google%20Colab%20T4-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

*A novel neural network architecture that replaces standard flat dense layers with **Order-2 Chebyshev Functional Matrix Polynomial Contractions** over 16×16 atomic matrix tiles, combined with **Poincaré Hyperbolic Conformal Charting** and **Matryoshka Multi-Exit Nested Sub-Models** for unified multimodal representation learning.*

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Training](#-training) · [Metrics](#-metrics--telemetry) · [Documentation](#-documentation) · [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Architecture](#-architecture)
  - [Tri-Aggregate Pipeline](#tri-aggregate-pipeline)
  - [Chebyshev Functional Matrix Blocks](#chebyshev-functional-matrix-blocks)
  - [Poincaré Hyperbolic Charting](#poincaré-hyperbolic-charting)
  - [Matryoshka Multi-Exit Suite](#matryoshka-multi-exit-suite)
  - [Loss Functions](#loss-functions)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start (Google Colab)](#quick-start-google-colab)
- [Project Structure](#-project-structure)
- [Training](#-training)
  - [6-Stream Self-Supervised Pretraining](#6-stream-self-supervised-pretraining)
  - [Dataset](#dataset)
  - [Checkpointing & Auto-Resume](#checkpointing--auto-resume)
- [Metrics & Telemetry](#-metrics--telemetry)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Highlights

| Feature | Description |
|---|---|
| **🔢 Nested Matrix Computation** | Replaces flat dense layers with Order-2 Chebyshev Polynomial Contractions over 16×16 matrix tiles — preserving 2D structural information throughout the network |
| **🌀 Hyperbolic Geometry** | Poincaré Ball conformal charting with Möbius addition and geodesic distance — naturally encodes hierarchical reasoning structures |
| **🪆 Matryoshka Multi-Exit** | Nested sub-models (Exit 1 → Exit 2 → Master) with L2 norm-rescaled junctions enabling zero-cost online distillation and speculative decoding |
| **🎯 5-Modality Fusion** | Unified processing of Image, Video, Text, Audio, and Tabular data through modality-specific tokenizers fused into a shared sequence |
| **📊 37-Metric Telemetry** | Comprehensive DuckDB-backed tracking across 8 metric families with persistent dataset traversal registry |
| **🎯 Fine-Grained Error Localization** | Coordinate-accurate failure pinpointing (Token $t^*$, Image Patch $(h^*, w^*)$, Video $(t^*, h^*, w^*)$, Audio $(f^*, t^*)$) with prefix-preserving rollback (PRM & Step-DPO) |
| **💾 Efficient Storage** | FP16 SafeTensors checkpoints (<16 MB per stream) with automatic Google Drive checkpoint discovery and auto-resume |
| **🔄 6 Parallel Streams** | Six independent CUDA streams exploring complementary SSL objectives (InfoNCE, Barlow Twins, VICReg, MAE, DEC, Omni) |
| **🛡️ Fault Tolerant** | Graceful recovery from Colab disconnects, CUDA OOM, and runtime resets with emergency state preservation |

---

## 🏗️ Architecture

### Tri-Aggregate Pipeline

MultimodalNFMNet follows a clean **Encoder → Core → Decoder** tri-aggregate architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MultimodalNFMNet Forward Pass                            │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │   ENCODER     │    │    CORE MODEL     │    │         DECODER              │   │
│  │              │    │                  │    │                              │   │
│  │ Image  ──┐  │    │ Chebyshev Stage 1│    │  NTP Logits [B, 64, 30522]  │   │
│  │ Video  ──┤  │    │   16×16 Tiles    │    │  Reconstruction [B, N, 256] │   │
│  │ Text   ──┼──┼───►│ Trace Scaling    │───►│  Contrastive z  [B, 128]    │   │
│  │ Audio  ──┤  │    │ Chebyshev Stage 2│    │  Classification [B, K]      │   │
│  │ Tabular──┘  │    │ Poincaré Chart   │    │  Regression     [B, 1]      │   │
│  │              │    │                  │    │  DEC Clustering  [B, K]     │   │
│  └──────────────┘    └──────────────────┘    └──────────────────────────────┘   │
│                                                                                 │
│  Z_raw [B,N,256] ──► Z_core [B,N,256] + z_riem [B,256] ──► Multi-Task Outputs  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Chebyshev Functional Matrix Blocks

The core computational primitive reshapes 256-D feature vectors into **16×16 matrix tiles** and evaluates Order-2 Chebyshev matrix polynomials:

$$T_0(X) = X, \quad T_1(X) = X, \quad T_2(X) = 2(XX^\top) - X$$

$$Y = T_0(X)\,C_0 + T_1(X)\,C_1 + T_2(X)\,C_2$$

where $C_0, C_1, C_2 \in \mathbb{R}^{16 \times 16}$ are trainable coefficient matrices. This preserves the **2D structural geometry** of features — unlike standard dense layers that flatten everything into 1D vectors.

**Trace-Invariant Activation** then modulates the output via the normalized matrix trace:

$$Y_{\text{scaled}} = Y \odot \sigma\!\left(\frac{\mathrm{Tr}(Y)}{16}\right)$$

### Poincaré Hyperbolic Charting

After global sequence pooling, representations are projected onto the **Poincaré Ball** $\mathbb{D}^n$ (a model of hyperbolic geometry) with conformal metric scaling:

$$\lambda_x = \frac{2}{1 - c\,\|x\|^2}$$

This space supports **Möbius addition** $(\oplus_c)$ and **geodesic distance**:

$$d_{\mathbb{D}^n}(x, y) = \operatorname{arcosh}\!\left(1 + \frac{2\,\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)$$

Hyperbolic space naturally represents **hierarchical and tree-structured** reasoning patterns with exponentially more room near the boundary of the disk.

### Matryoshka Multi-Exit Suite

Inspired by [Matryoshka Language Model Suites (Godey & Artzi, Cornell 2026)](https://arxiv.org/abs/2608.09703), the architecture nests smaller sub-models inside a single backbone:

```
                    ┌─────────────────────────────────────┐
                    │       Shared Omni-Encoder            │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │     Core Block 1 → Decoder 1         │  ← Exit 1 (Smallest / Fastest)
                    └────────────────┬────────────────────┘
                            Junction │ (L2 Norm Rescaling)
                    ┌────────────────▼────────────────────┐
                    │     Core Block 2 → Decoder 2         │  ← Exit 2 (Medium)
                    └────────────────┬────────────────────┘
                            Junction │ (L2 Norm Rescaling)
                    ┌────────────────▼────────────────────┐
                    │     Core Block 3 → Decoder 3         │  ← Exit 3 (Master / Full)
                    └─────────────────────────────────────┘
```

**Inter-Model Junctions** use L2 norm rescaling to prevent representation magnitude drift:

$$\hat{h}^{(m)} = h^{(m)} \cdot \frac{\|e^{(m+1)}\|_2}{\|h^{(m)}\|_2}, \qquad h_{\text{in}}^{(m+1)} = W_{\text{proj}}\!\left[e^{(m+1)} \,\|\, \hat{h}^{(m)}\right]$$

### Fine-Grained Multimodal Error Localization & Prefix Rollback

Instead of binary pass/fail retries that discard whole responses and restart from Step 0, MultimodalNFMNet integrates **coordinate-level error localization** and **prefix-preserving branching** across all 5 modalities:

```
CONVENTIONAL OUTCOME RE-RUN (WASTEFUL):
[Step 1: OK] ──► [Step 2: OK] ──► [Step 3: FAILS ❌] ──► [Step 4: INVALID] ──► RESTART FROM STEP 0

FINE-GRAINED LOCALIZATION & PREFIX ROLLBACK (ORNet / PRM / Step-DPO):
[Step 1: OK] ──► [Step 2: OK] ──► [Step 3: FAILS ❌ at (t*)] ──► Freeze Prefix & Rollback to Step 2
       │                │                                         │
       └── CACHED ──────┴─────────────────────────────────────────┴──► [Step 3' (Corrected)] ──► Success ✅
```

- **📝 Text:** Pinpoints First Erroneous Token ($t^*$) & Thought Step ($s^*$) via token-level surprisal and Process Reward Model ($V_\phi$) scoring.
- **🖼️ Image:** Locates failing spatial patch grid coordinates $(h^*, w^*)$ via MAE reconstruction residuals.
- **🎥 Video:** Detects temporal frame disruption index $t^*$ and spatiotemporal patch $(t^*, h^*, w^*)$.
- **🎵 Audio:** Identifies time-frequency anomaly coordinates $(f^*, t^*)$ in Mel-spectrograms.
- **🧊 3D / Tabular:** Pinpoints point cloud Cartesian error coordinates $(x^*, y^*, z^*)$ and graph column indices.

### Loss Functions

| Loss | Type | Formula |
|---|---|---|
| **InfoNCE** | Contrastive SSL | $\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}$ |
| **Barlow Twins** | Redundancy Reduction | $\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2$ |
| **VICReg** | Variance-Invariance-Covariance | MSE + Hinge Variance + Off-Diagonal Covariance |
| **Causal NTP** | Next-Token Prediction | Auto-regressive Cross-Entropy over text thought tokens |
| **DEC KL** | Deep Embedded Clustering | $\mathcal{L} = \mathrm{KL}(P \,\|\, Q)$ with Student-t soft assignments |
| **Matryoshka Distillation** | Online Multi-Exit | $(1-\alpha)\mathcal{L}_{\text{CE}} + \alpha\,\mathcal{L}_{\text{distill}}(l^m, \mathrm{sg}(l^M))$ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Google Colab with T4 GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/nithin12342/new-multi-dimension-neural-network.git
cd new-multi-dimension-neural-network

# Install dependencies
pip install torch torchvision safetensors duckdb numpy scipy psutil transformers
```

### Quick Start (Google Colab)

```python
# In a Google Colab cell:
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/nithin12342/new-multi-dimension-neural-network.git
%cd new-multi-dimension-neural-network

!pip install -q safetensors duckdb psutil

!python train_omni.py
```

The pipeline will automatically:
1. Mount Google Drive for persistent storage
2. Download the E-MM1 multimodal dataset
3. Auto-resume from the latest valid checkpoint (if any)
4. Execute 6-stream self-supervised pretraining
5. Log all 37 metrics to DuckDB telemetry database
6. Save consolidated FP16 SafeTensors checkpoints to Google Drive

---

## 📁 Project Structure

```
new-multi-dimension-neural-network/
│
├── train_omni.py                           # 🚀 Entry point for Colab execution
│
├── src/
│   ├── domain/                             # Pure domain logic (no I/O)
│   │   ├── config/
│   │   │   └── config_entities.py          #   Frozen dataclasses (Model, Data, Path, Training)
│   │   ├── model/
│   │   │   ├── encoder.py                  #   CombinedOmniEncoder (5-modality fusion)
│   │   │   ├── core_model.py               #   FunctionalCoreModel (Chebyshev + Poincaré)
│   │   │   ├── decoder.py                  #   SingleNestedMatrixDecoder (multi-task heads)
│   │   │   ├── chebyshev.py                #   ChebyshevFunctionalBlock (16×16 tile contractions)
│   │   │   ├── trace_activation.py         #   TraceInvariantGate (matrix trace scaling)
│   │   │   ├── riemannian.py               #   PoincareConformalChart (hyperbolic mapping)
│   │   │   ├── tokenizers.py               #   GigaTokenizerEngine + 5 modality tokenizers
│   │   │   ├── matryoshka_suite.py         #   MultimodalMatryoshkaSuite (multi-exit)
│   │   │   ├── matryoshka_junction.py      #   InterModelMatryoshkaJunction (L2 rescaling)
│   │   │   └── paradigm_heads.py           #   SSL/Supervised/Clustering projection heads
│   │   ├── loss/
│   │   │   ├── loss_functions.py           #   InfoNCE, BarlowTwins, VICReg, NTP, DEC losses
│   │   │   └── matryoshka_loss.py          #   Multi-exit online distillation loss
│   │   └── data/
│   │       └── dataset_interface.py        #   AbstractMultimodalDataset interface
│   │
│   ├── application/                        # Orchestration & fault tolerance
│   │   ├── orchestrator/
│   │   │   ├── training_loop.py            #   ParadigmTrainingOrchestrator (6-stream engine)
│   │   │   └── distillation_manager.py     #   CheckpointDistillationManager (teacher fusion)
│   │   └── fault_tolerance/
│   │       └── recovery_manager.py         #   FaultToleranceManager (CUDA OOM recovery)
│   │
│   ├── infrastructure/                     # External systems & hardware
│   │   ├── data/
│   │   │   └── multimodal_dataset.py       #   E-MM1 5-modality dataset loader
│   │   ├── checkpoint/
│   │   │   ├── serializer.py               #   FP16 SafeTensors checkpoint serializer
│   │   │   └── discovery.py                #   Recursive Drive checkpoint scanner
│   │   ├── metrics/
│   │   │   └── metric_computer.py          #   37-metric computer (8 metric families)
│   │   ├── logging/
│   │   │   ├── prediction_logger.py        #   DuckDB prediction & metric exporter
│   │   │   └── session_logger.py           #   Hardware telemetry profiler
│   │   ├── storage/
│   │   │   └── drive_manager.py            #   Google Drive mount & directory manager
│   │   └── streams/
│   │       └── stream_manager.py           #   6 isolated CUDA stream manager
│   │
│   └── interfaces/
│       └── cli/
│           └── main.py                     #   Pipeline CLI runner
│
├── tests/
│   └── e2e/
│       └── test_full_pipeline.py           #   End-to-end forward & training tests
│
├── SKELETON.md                             #   Master architectural blueprint
├── OMNI_PRETRAINING_ARCHITECTURE.md        #   5-modality pretraining specification
├── HUMAN_CRITICAL_THINKING_ARCHITECTURE.md #   Critical thinking & NTP architecture
└── MATRYOSHKA_MULTIMODAL_SUITE_SPECIFICATION.md  # Matryoshka suite specification
```

---

## 🏋️ Training

### 6-Stream Self-Supervised Pretraining

The training pipeline executes **6 independent CUDA streams sequentially**, each exploring a different self-supervised learning objective:

| Stream | Paradigm | Loss Combination |
|--------|----------|-----------------|
| 1 | `self_supervised_ntp` | NTP + InfoNCE |
| 2 | `self_supervised_barlow` | Barlow Twins + NTP |
| 3 | `self_supervised_vicreg` | VICReg + MAE |
| 4 | `self_supervised_mae` | Masked Autoencoder |
| 5 | `self_supervised_dec` | DEC KL-Divergence Clustering |
| 6 | `self_supervised_omni` | NTP + InfoNCE + MAE |

Each stream maintains its own model weights, optimizer state, and gradient scaler. Only the active stream's model resides on GPU — completed streams are offloaded to CPU to preserve VRAM.

### Dataset

The pipeline uses [Encord E-MM1](https://huggingface.co/datasets/encord-team/E-MM1-1M), an open-source multimodal reasoning dataset providing aligned 5-modality samples:

| Modality | Input Shape | Tokenizer | Output Tokens |
|----------|-------------|-----------|---------------|
| Image | `[B, 3, 224, 224]` | Conv2d Patch (16×16) | 196 |
| Video | `[B, 3, 4, 224, 224]` | Conv3d Spatiotemporal | 196 |
| Text | `[B, 64]` | GigaTokenizer Embedding | 64 |
| Audio | `[B, 1, 64, 64]` | Mel-Spectrogram Patch | 16 |
| Tabular | `[B, 15]` | Linear Graph Projection | 4 |

A **Persistent Dataset Traversal Registry** (backed by DuckDB) ensures 100% dataset coverage before any sample repetition, tracking chunk indices across Colab session restarts.

### Checkpointing & Auto-Resume

- **Format:** HuggingFace [SafeTensors](https://huggingface.co/docs/safetensors) (FP16, <16 MB per stream)
- **Storage:** Google Drive (`/content/drive/MyDrive/SOTA_Cluster_Shared/checkpoints/`)
- **Auto-Resume:** On startup, the pipeline recursively scans Google Drive for the newest valid `.safetensors` checkpoint per stream, validates integrity, and resumes training from the exact epoch
- **Legacy Compatibility:** Automatic state dict key remapping from single-exit to multi-exit Matryoshka format

---

## 📊 Metrics & Telemetry

All telemetry is stored in a **single consolidated DuckDB database** (`multimodal_telemetry.duckdb`) with 5 tables:

### `epoch_metrics` — 37 Metrics Across 8 Families

| Family | Metrics |
|--------|---------|
| **Classification** | Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa, Log Loss |
| **Regression** | MSE, MAE, RMSE, R², Explained Variance Ratio |
| **Clustering** | Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index |
| **Information Theory** | Mutual Information, Normalized MI, Adjusted MI, Entropy |
| **Statistical** | Spearman ρ, Kendall τ, Pearson r, Mean Absolute Deviation |
| **Distributional** | Wasserstein Distance, KL Divergence, JS Divergence, Skewness, Kurtosis |
| **Calibration** | Brier Score, Expected Calibration Error (ECE) |
| **Language Modeling** | Perplexity, Cross-Entropy, Top-5 Accuracy, Top-10 Accuracy |

### `sample_error_localization` — Coordinate-Level Failure Localization

Per-sample fine-grained error coordinates:
- `text_first_error_step` & `text_error_token_idx`: Exact step $s^*$ and token position $t^*$ of reasoning divergence
- `image_failed_patch_coords` & `image_worst_patch_coord`: $14 \times 14$ spatial patch coordinates $[h^*, w^*]$ of reconstruction residual spikes
- `audio_worst_freq_bin` & `audio_worst_time_bin`: $64 \times 64$ Mel-spectrogram time-frequency anomaly coordinates

### `predictions` — Per-Sample Prediction Logs

Per-epoch sample-level predictions with softmax probabilities, predicted vs. true class, and sample IDs.

### `hardware_telemetry_timeseries` — Continuous Periodic Hardware Profiling

Real-time per-epoch system metrics:
- `gpu_vram_allocated_mb`, `gpu_vram_reserved_mb`, `gpu_vram_peak_mb`
- `cpu_percent`, `ram_used_gb`, `ram_percent`, `elapsed_sec`

### `session_telemetry` — Hardware & Environment Profiling

GPU name, VRAM total capacity, CPU cores, RAM total, PyTorch/CUDA versions, and session launch/end timestamps.

---

## ⚙️ Configuration

All configurations are **frozen dataclasses** in [`config_entities.py`](src/domain/config/config_entities.py):

```python
@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int = 256          # Feature embedding dimension (= 16² tile)
    tile_dim: int = 16            # Atomic matrix tile dimension
    chebyshev_order: int = 2      # Chebyshev polynomial order
    vocab_size: int = 30522       # Text vocabulary size
    num_classes: int = 10         # Classification head classes
    num_clusters: int = 10        # DEC clustering centroids
    projection_dim: int = 128     # Contrastive projection dimension
    poincare_curvature: float = 1.0  # Poincaré ball curvature

@dataclass(frozen=True)
class TrainingConfig:
    num_streams: int = 6          # Parallel CUDA training streams
    num_epochs: int = 50          # Epochs per stream per session
    learning_rate: float = 3e-4   # AdamW learning rate
    weight_decay: float = 1e-4    # AdamW weight decay
    seed: int = 42                # Reproducibility seed
```

---

## 📖 Documentation

The repository includes extensive architectural documentation:

| Document | Description |
|----------|-------------|
| [`SKELETON.md`](SKELETON.md) | Master architectural blueprint with 23 requirements and 25 DIP file nodes |
| [`POINCARE_GYROPLANE_AND_PERIODIC_TELEMETRY_SPECIFICATION.md`](POINCARE_GYROPLANE_AND_PERIODIC_TELEMETRY_SPECIFICATION.md) | Poincaré Gyroplane Geodesic Classifier & Periodic Hardware Telemetry |
| [`DUCKDB_TIMESTAMPED_ERROR_LOCALIZATION_ANALYSIS.md`](DUCKDB_TIMESTAMPED_ERROR_LOCALIZATION_ANALYSIS.md) | Timestamped DuckDB telemetry audit & 4-step diagnostic workflow |
| [`FINE_GRAINED_MULTIMODAL_ERROR_LOCALIZATION_ARCHITECTURE.md`](FINE_GRAINED_MULTIMODAL_ERROR_LOCALIZATION_ARCHITECTURE.md) | Fine-grained failure localization & prefix rollback across all 5 modalities |
| [`DUAL_STAGE_ERROR_LOCALIZATION_IMPLEMENTATION_REPORT.md`](DUAL_STAGE_ERROR_LOCALIZATION_IMPLEMENTATION_REPORT.md) | Implementation report for live pre-training & post-training (PRM / Step-DPO) |
| [`OMNI_PRETRAINING_ARCHITECTURE.md`](OMNI_PRETRAINING_ARCHITECTURE.md) | 5-modality pretraining pipeline specification |
| [`HUMAN_CRITICAL_THINKING_ARCHITECTURE.md`](HUMAN_CRITICAL_THINKING_ARCHITECTURE.md) | Critical thinking & NTP design |
| [`MATRYOSHKA_MULTIMODAL_SUITE_SPECIFICATION.md`](MATRYOSHKA_MULTIMODAL_SUITE_SPECIFICATION.md) | Matryoshka multi-exit suite specification |
| [`OMNI_DATASET_COMMERCIAL_CATALOG.md`](OMNI_DATASET_COMMERCIAL_CATALOG.md) | Commercial license audit for all datasets |
| [`NATLOG_HYPERGRAPH_SIMULATION_ARCHITECTURE.md`](NATLOG_HYPERGRAPH_SIMULATION_ARCHITECTURE.md) | Natural Logic hypergraph & symbolic reasoning |
| [`AUTONOMOUS_SANDBOX_PIPELINE_ARCHITECTURE.md`](AUTONOMOUS_SANDBOX_PIPELINE_ARCHITECTURE.md) | Autonomous sandbox execution architecture |
| [`CONVERSATION_HISTORY_LOG.md`](CONVERSATION_HISTORY_LOG.md) | Complete development chronology |

---

## 🗺️ Roadmap

- [x] **Stage 1:** 5-Modality Self-Supervised Omni-Pretraining (InfoNCE, Barlow Twins, VICReg, NTP, DEC)
- [x] **Matryoshka Integration:** Multi-exit nested sub-models with online distillation
- [x] **Telemetry:** 37-metric DuckDB tracking with persistent traversal registry
- [x] **Fine-Grained Error Localization:** Coordinate-accurate failure localization & DuckDB telemetry
- [ ] **Stage 2:** Supervised Fine-Tuning (SFT) with Prefix-Preserving Self-Correction & Hyperbolic Gyroplanes
- [ ] **Stage 3:** Post-Training Alignment (Step-DPO / Process Reward Models / MCTS)
- [ ] **Natural Logic Engine:** Z3 SMT-backed hypergraph reasoning
- [ ] **Autonomous Sandbox:** Self-healing Docker/gVisor execution environment
- [ ] **Inference API:** FastAPI serving with speculative decoding via Matryoshka exits

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

Please ensure your code follows the existing Domain-Driven Design (DDD) bounded context structure and includes appropriate docstrings.

---

## 📝 Citation

If you use MultimodalNFMNet in your research, please cite:

```bibtex
@software{multimodalnfmnet2026,
  title     = {MultimodalNFMNet: Nested Functional Matrix Network for 5-Modality Self-Supervised Omni-Pretraining},
  author    = {Nithin},
  year      = {2026},
  url       = {https://github.com/nithin12342/new-multi-dimension-neural-network},
  license   = {MIT}
}
```

### Related Works

- Godey, N. & Artzi, Y. (2026). *Matryoshka Language Model Suites.* [arXiv:2608.09703](https://arxiv.org/abs/2608.09703)
- Encord Team. *E-MM1-1M: Open-Source Multimodal Reasoning Dataset.* [HuggingFace](https://huggingface.co/datasets/encord-team/E-MM1-1M)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**Built with ❤️ using PyTorch, Riemannian Geometry, and Chebyshev Polynomials**

</div>