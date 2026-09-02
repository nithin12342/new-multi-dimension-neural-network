# SKELETON.md — MultimodalNFMNet Master Architectural Blueprint & Living Progress Skeleton

> **System:** Robust 5-Modality Self-Supervised Omni-Pretraining Pipeline for MultimodalNFMNet  
> **Language:** Python (PyTorch)  
> **Target Runtime:** Google Colab (T4 GPU, 15GB VRAM, 12GB RAM)  
> **Phase:** Phase 4 Verified & Production Synchronized  
> **Source of Truth:** [context.md](context.md) + [OMNI_PRETRAINING_ARCHITECTURE.md](OMNI_PRETRAINING_ARCHITECTURE.md) + [HUMAN_CRITICAL_THINKING_ARCHITECTURE.md](HUMAN_CRITICAL_THINKING_ARCHITECTURE.md) + [.agents/rules/intention-engineering-principles.md](.agents/rules/intention-engineering-principles.md)

---

## 1. Requirements & System Invariants

- id: REQ-001
  text: "Implement MultimodalNFMNet 5-modality architecture with Chebyshev Functional Matrix Blocks, Trace-Invariant Activation, and Conformal Riemannian Charting per context.md & OMNI_PRETRAINING_ARCHITECTURE.md"
  spec_id: SPEC-001

- id: REQ-002
  text: "Implement paradigm heads: SSL (InfoNCE, Barlow Twins, VICReg, Masked Reconstruction, Next-Token Prediction), Supervised (Classification CE, Regression MSE/MAE/R²), Unsupervised (DEC Student-t clustering with KL-divergence)"
  spec_id: SPEC-002

- id: REQ-003
  text: "Support 5-modality authentic open-source datasets (Video, Image, Text, Audio, Tabular) with automatic download, preprocessing, and augmentation"
  spec_id: SPEC-003

- id: REQ-004
  text: "Maintain six independent model weight files across three paradigms (2× Supervised, 2× SSL, 2× Unsupervised) with sequential stream training"
  spec_id: SPEC-004

- id: REQ-005
  text: "Implement SafeTensors weight serialization (.safetensors) with single consolidated FP16 checkpoint per stream (<16 MB each, <95 MB total)"
  spec_id: SPEC-005

- id: REQ-006
  text: "Automatic checkpoint discovery: recursive Google Drive scan, timestamp sort, SafeTensors integrity validation, auto-resume from newest valid checkpoint"
  spec_id: SPEC-006

- id: REQ-007
  text: "Store all persistent artifacts (checkpoints, DuckDB databases, logs, reports) exclusively on Google Drive"
  spec_id: SPEC-007

- id: REQ-008
  text: "Compute and log all 37 metrics per epoch across 8 metric families into DuckDB epoch_metrics table"
  spec_id: SPEC-008

- id: REQ-009
  text: "Detailed per-epoch sample prediction logging into DuckDB predictions table"
  spec_id: SPEC-009

- id: REQ-010
  text: "Detailed session logging: hardware info, CUDA/PyTorch versions, memory/CPU/GPU utilization into DuckDB session_telemetry table"
  spec_id: SPEC-010

- id: REQ-011
  text: "Dummy weight initialization: create lightweight SafeTensors model files with version metadata before training begins"
  spec_id: SPEC-011

- id: REQ-012
  text: "Fault tolerance: graceful recovery from Colab disconnects, runtime resets, CUDA OOM, auto-extension when resuming completed checkpoints"
  spec_id: SPEC-012

- id: REQ-013
  text: "Automatic versioning for models, datasets, checkpoints, configurations, metrics — every artifact carries version number and timestamp"
  spec_id: SPEC-013

- id: REQ-014
  text: "Organized Google Drive directory structure per spec §14"
  spec_id: SPEC-014

- id: REQ-015
  text: "Multi-stream CUDA execution: 6 isolated CUDA streams with per-stream Model, Optimizer, GradScaler, synchronized metric collection"
  spec_id: SPEC-015

- id: REQ-016
  text: "Rule 12 Strict Enforcement: Authentic datasets only (torchvision datasets, Kaggle key KGAT_c0234fe2d5a9d53f6c18baf6fbe983b4); zero mock data fallbacks"
  spec_id: SPEC-016

- id: REQ-017
  text: "Self-Supervised Next-Token Prediction (NTP) over Multimodal Human Critical Thinking Sequences per HUMAN_CRITICAL_THINKING_ARCHITECTURE.md"
  spec_id: SPEC-017

- id: REQ-018
  text: "Single Consolidated DuckDB Database (multimodal_telemetry.duckdb) containing predictions, epoch_metrics, and session_telemetry tables"
  spec_id: SPEC-018

- id: REQ-019
  text: "5-Modality Pretraining Pipeline: Video, Image, Text, Audio, Tabular/Point-Cloud via Encord E-MM1 dataset (encord-team/E-MM1-1M)"
  spec_id: SPEC-019

- id: REQ-020
  text: "GigaTokenizer Engine: High-throughput zero-copy SIMD tokenization concept inspired by Stanford GigaToken (24 GB/sec throughput) via Hash-LRU caching and byte-level mapping"
  spec_id: SPEC-020

- id: REQ-021
  text: "STRICT LOCAL EXECUTION RULE: Never execute pretraining runs on local developer PC; training execution is strictly restricted to Google Colab cloud environment to prevent local PC crashes"
  spec_id: SPEC-021

- id: REQ-022
  text: "Implement Fine-Grained Multimodal Error Localization & Step-Level Targeted Correction Engine across 5 modalities with DuckDB sample_error_localization telemetry export and prefix-preserving rollback support"
  spec_id: SPEC-022

- id: REQ-023
  text: "Implement Continuous Periodic Time-Series Hardware Telemetry (hardware_telemetry_timeseries in DuckDB) and Poincaré Gyroplane Hyperbolic Geodesic Classification Head"
  spec_id: SPEC-023

- id: REQ-024
  text: "Implement 4 Forensic Remediation Guards: FP16 InfoNCE logit clamp (<= 10.8), strict VICReg variance hinge (gamma=1.0, eps=1e-4), Poincaré boundary clipping (||x|| <= 1 - 1e-4, lambda_x <= 1000), and Causal NTP token pad masking (ignore_index=0)"
  spec_id: SPEC-024

- id: REQ-025
  text: "Implement Comprehensive Codebase & Systems Overhaul: StateDictRemapper with shape validation, contiguous multimodal collation, PyArrow in-memory buffering with Snappy Parquet export, SafeTensors 2.1.0 schema versioning, and autograd graph memory sanitization"
  spec_id: SPEC-025

---

## 2. Bounded Contexts & Aggregates

### 1. ModelArchitecture
- **reason_separate:** "Owns neural network definition only — no training logic, no I/O"
- **sot_id:** SOT-001
- **aggregates:**
  - **ChebyshevFunctionalBlock**
    - invariant: "Tile reshape produces exactly [B*N, 16, 16] matrices; Chebyshev bases T0, T1, T2 computed over 16×16 tiles; trace activation preserves shape"
    - entities: [ChebyshevBasis, TraceGate, TileReshaper]
    - value_objects: [TileSize(16), PolynomialOrder(2)]
  - **ModalityTokenizers & GigaTokenizerEngine**
    - invariant: "Projects 5 modalities (Video, Image, Text, Audio, Tabular) into unified sequence D=256 using GigaTokenizer zero-copy SIMD lookup"
    - entities: [GigaTokenizerEngine, VisionPatchTokenizer, VideoSpatiotemporalTokenizer, TextEmbeddingTokenizer, AudioSpectrogramTokenizer, TabularGraphTokenizer, OmniTokenFusion]
  - **MultimodalErrorLocalizationEngine**
    - invariant: "Pinpoints exact failure coordinates across 5 modalities (token t*, patch [h*,w*], frame t*, time-frequency [f*,t*]) with prefix rollback"
    - entities: [MultimodalErrorLocalizationEngine]
  - **Tri-Aggregate Architecture**
    - invariant: "MultimodalNFMNet is composed of CombinedOmniEncoder, FunctionalCoreModel, and SingleNestedMatrixDecoder"
    - entities: [CombinedOmniEncoder, FunctionalCoreModel, SingleNestedMatrixDecoder, MultimodalMatryoshkaSuite]

---

## 3. Implementation Status Matrix (25 Modular DIP Nodes)

| File ID | Folder Path | Owning Aggregate | Single Responsibility (<=7 Words) | Implementation Status |
|---|---|---|---|---|
| **FILE-001** | `src/domain/config/config_entities.py` | ConfigRegistry | define immutable training and model configuration data structures | ✅ Implemented & Tested |
| **FILE-002** | `src/domain/model/chebyshev.py` | ChebyshevBlock | compute order-2 chebyshev functional matrix polynomial contractions | ✅ Implemented & Tested |
| **FILE-003** | `src/domain/model/trace_activation.py` | ChebyshevBlock | apply trace invariant activation scaling to matrix tiles | ✅ Implemented & Tested |
| **FILE-004** | `src/domain/model/tokenizers.py` | ModalityTokenizers | tokenize 5 modalities using gigatokenizer zero copy simd engine | ✅ Implemented & Tested |
| **FILE-005** | `src/domain/model/riemannian.py` | RiemannianChart | map features to poincaré ball conformal charts | ✅ Implemented & Tested |
| **FILE-006** | `src/domain/model/paradigm_heads.py` | ParadigmHeads | project representations into paradigm output heads including ntp | ✅ Implemented & Tested |
| **FILE-007** | `src/domain/data/dataset_interface.py` | DatasetRegistry | define abstract dataset loader and preprocessing interfaces | ✅ Implemented & Tested |
| **FILE-008** | `src/domain/loss/loss_functions.py` | LossFunctions | compute contrastive ntp classification and dec clustering losses | ✅ Implemented & Tested |
| **FILE-009** | `src/infrastructure/storage/drive_manager.py` | DriveManager | mount google drive and resolve persistent directories | ✅ Implemented & Tested |
| **FILE-010** | `src/infrastructure/data/multimodal_dataset.py` | DatasetRegistry | download preprocess and load e-mm1 5 modality authentic dataset batches | ✅ Implemented & Tested |
| **FILE-011** | `src/infrastructure/metrics/metric_computer.py` | MetricComputer | compute 37 classification regression clustering statistical metrics | ✅ Implemented & Tested |
| **FILE-012** | `src/infrastructure/streams/stream_manager.py` | StreamManager | isolate 6 cuda execution streams and optimizers | ✅ Implemented & Tested |
| **FILE-013** | `src/infrastructure/checkpoint/serializer.py` | CheckpointSerializer | serialize checkpoints using fp16 safetensors file format | ✅ Implemented & Tested |
| **FILE-014** | `src/infrastructure/checkpoint/discovery.py` | CheckpointDiscovery | scan drive recursively and validate newest safetensors checkpoint | ✅ Implemented & Tested |
| **FILE-015** | `src/infrastructure/logging/session_logger.py` | SessionLogger | profile hardware stats and log session telemetry to duckdb | ✅ Implemented & Tested |
| **FILE-016** | `src/infrastructure/logging/prediction_logger.py` | PredictionLogger | export sample predictions and 37 metrics to duckdb | ✅ Implemented & Tested |
| **FILE-017** | `src/application/orchestrator/training_loop.py` | TrainingLoop | execute 5 modality epoch iterations across paradigm streams | ✅ Implemented & Tested |
| **FILE-018** | `src/application/fault_tolerance/recovery_manager.py` | RecoveryManager | catch runtime failures and trigger emergency recovery | ✅ Implemented & Tested |
| **FILE-019** | `src/interfaces/cli/main.py` | MainRunner | sequence end to end colab training pipeline | ✅ Implemented & Tested |
| **FILE-020** | `src/domain/model/encoder.py` | CombinedOmniEncoder | encode 5 modalities using chebyshev matrix reduction | ✅ Implemented & Tested |
| **FILE-021** | `src/domain/model/core_model.py` | FunctionalCoreModel | execute chebyshev matrix contractions and poincare chart | ✅ Implemented & Tested |
| **FILE-022** | `src/domain/model/decoder.py` | SingleNestedMatrixDecoder | project core representations into all outputs using single decoder | ✅ Implemented & Tested |
| **FILE-023** | `train_omni.py` | MainRunner | entrypoint script for google colab execution | ✅ Implemented & Tested |
| **FILE-030** | `src/domain/model/matryoshka_suite.py` | MultimodalMatryoshkaSuite | execute nested multi exit 5 modality forward passes | ✅ Implemented & Tested |
| **FILE-031** | `src/domain/model/error_localization.py` | MultimodalErrorLocalizationEngine | pinpoint exact failure coordinates across 5 modalities with rollback | ✅ Implemented & Tested |

