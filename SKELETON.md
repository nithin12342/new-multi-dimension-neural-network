# SKELETON.md — MultimodalNFMNet Master Architectural Blueprint & Living Progress Skeleton

> **System:** Robust Multimodal Training Pipeline for MultimodalNFMNet  
> **Language:** Python (PyTorch)  
> **Target Runtime:** Google Colab (T4 GPU, 15GB VRAM, 12GB RAM)  
> **Phase:** Phase 4 Verified & Production Synchronized (Git Main Sync: `70feba0`)  
> **Source of Truth:** [context.md](context.md) §11 + [HUMAN_CRITICAL_THINKING_ARCHITECTURE.md](HUMAN_CRITICAL_THINKING_ARCHITECTURE.md) + [.agents/rules/intention-engineering-principles.md](.agents/rules/intention-engineering-principles.md)

---

## 1. Requirements & System Invariants

- id: REQ-001
  text: "Implement MultimodalNFMNet architecture with Chebyshev Functional Matrix Blocks, Trace-Invariant Activation, and Conformal Riemannian Charting per context.md §11.1"
  spec_id: SPEC-001

- id: REQ-002
  text: "Implement paradigm heads: SSL (InfoNCE, Barlow Twins, VICReg, Masked Reconstruction, Next-Token Prediction), Supervised (Classification CE, Regression MSE/MAE/R²), Unsupervised (DEC Student-t clustering with KL-divergence)"
  spec_id: SPEC-002

- id: REQ-003
  text: "Support multimodal authentic open-source datasets (image+text) with automatic download, preprocessing, and augmentation"
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
  - **ModalityTokenizers**
    - invariant: "Vision patch Conv2D produces [B, N_img, 256]; text embedding produces [B, S, 256]; fusion concatenation preserves D=256"
    - entities: [VisionPatchTokenizer, TextEmbeddingTokenizer, TokenFusion]
    - value_objects: [PatchSize(16), EmbeddingDim(256), VocabSize(30522)]
  - **ConformalRiemannianChart**
    - invariant: "Poincaré ball constraint ‖x‖ < 1 enforced; conformal scale λ_x = 2/(1−‖x‖²) always positive"
    - entities: [PoincareBall, MobiusAddition, GeodesicDistance]
    - value_objects: [Curvature(c)]
  - **ParadigmHeads**
    - invariant: "Each head receives sequence/pooled representations and produces paradigm-specific output (including NextTokenPredictionHead for auto-regressive thought modeling)"
    - entities: [SSLProjectionHead, MaskedReconstructionHead, NextTokenPredictionHead, SupervisedClassificationHead, SupervisedRegressionHead, DECClusteringHead]
    - value_objects: [NumClasses(10), NumClusters(10), ProjectionDim(128), VocabSize(30522)]

---

### 2. DataLoading
- **reason_separate:** "Owns dataset download, preprocessing, augmentation — never touches model weights or training state"
- **sot_id:** SOT-002
- **aggregates:**
  - **DatasetRegistry**
    - invariant: "Rule 12 strictly enforced: loads real authentic open-source datasets; zero mock data generation"
    - entities: [DatasetDownloader, DatasetPreprocessor, AugmentationPipeline, DatasetConfig]
    - value_objects: [DatasetVersion, SplitRatio]

---

### 3. LossComputation
- **reason_separate:** "Owns all loss functions and 37-metric calculations"
- **sot_id:** SOT-003
- **aggregates:**
  - **LossFunctions**
    - invariant: "Computes InfoNCE, Barlow Twins, VICReg, CausalNextTokenLoss, CrossEntropy, MSE, MAE, DECKLLoss"
    - entities: [InfoNCELoss, BarlowTwinsLoss, VICRegLoss, CausalNextTokenLoss, CrossEntropyParadigmLoss, DECKLRegLoss]
  - **MetricComputer**
    - invariant: "Computes all 37 metrics across 8 metric families; returns flat dict with standardized keys"
    - entities: [ThirtySevenMetricComputer]

---

### 4. TrainingOrchestrator
- **reason_separate:** "Owns training loop, validation loop, and multi-stream coordination"
- **sot_id:** SOT-004
- **aggregates:**
  - **StreamManager**
    - invariant: "Exactly 6 CUDA streams maintained; each stream owns one (Model, Optimizer, GradScaler) triple"
    - entities: [SixStreamManager]
  - **TrainingLoop**
    - invariant: "Auto-extending training target budget when resuming from completed epoch checkpoints"
    - entities: [ParadigmTrainingOrchestrator]

---

### 5. CheckpointManager
- **reason_separate:** "Owns SafeTensors checkpoint I/O, discovery, and consolidation"
- **sot_id:** SOT-005
- **aggregates:**
  - **CheckpointSerializer**
    - invariant: "Saves FP16 model weights in HuggingFace .safetensors format; maintains EXACTLY 1 consolidated file per stream (<16 MB)"
    - entities: [CheckpointSerializer]
  - **CheckpointDiscovery**
    - invariant: "Scans .safetensors files; validates header metadata; returns newest valid checkpoint"
    - entities: [CheckpointDiscoveryScanner]

---

### 6. Logging
- **reason_separate:** "Owns all DuckDB database logging output"
- **sot_id:** SOT-007
- **aggregates:**
  - **PredictionLogger & SessionLogger**
    - invariant: "All predictions, 37 metrics, and session telemetry recorded into single multimodal_telemetry.duckdb database on Google Drive"
    - entities: [PredictionLogExporter, SessionTelemetryLogger]

---

## 3. Data Flow Order

```
Configuration → GoogleDriveStorage → DataLoading → ModelArchitecture → LossComputation
    → TrainingOrchestrator → CheckpointManager → Logging (DuckDB) → FaultTolerance → Entrypoint
```

---

## 4. Implementation Status Matrix (19 Modular DIP Nodes)

| File ID | Folder Path | Owning Aggregate | Single Responsibility (<=7 Words) | Implementation Status |
|---|---|---|---|---|
| **FILE-001** | `src/domain/config/config_entities.py` | ConfigRegistry | define immutable training and model configuration data structures | ✅ Implemented & Tested |
| **FILE-002** | `src/domain/model/chebyshev.py` | ChebyshevBlock | compute order-2 chebyshev functional matrix polynomial contractions | ✅ Implemented & Tested |
| **FILE-003** | `src/domain/model/trace_activation.py` | ChebyshevBlock | apply trace invariant activation scaling to matrix tiles | ✅ Implemented & Tested |
| **FILE-004** | `src/domain/model/tokenizers.py` | ModalityTokenizers | tokenize and project image and text inputs | ✅ Implemented & Tested |
| **FILE-005** | `src/domain/model/riemannian.py` | RiemannianChart | map features to poincaré ball conformal charts | ✅ Implemented & Tested |
| **FILE-006** | `src/domain/model/paradigm_heads.py` | ParadigmHeads | project representations into paradigm output heads including ntp | ✅ Implemented & Tested |
| **FILE-007** | `src/domain/data/dataset_interface.py` | DatasetRegistry | define abstract dataset loader and preprocessing interfaces | ✅ Implemented & Tested |
| **FILE-008** | `src/domain/loss/loss_functions.py` | LossFunctions | compute contrastive ntp classification and dec clustering losses | ✅ Implemented & Tested |
| **FILE-009** | `src/infrastructure/storage/drive_manager.py` | DriveManager | mount google drive and resolve persistent directories | ✅ Implemented & Tested |
| **FILE-010** | `src/infrastructure/data/multimodal_dataset.py` | DatasetRegistry | download preprocess and load authentic dataset batches | ✅ Implemented & Tested |
| **FILE-011** | `src/infrastructure/metrics/metric_computer.py` | MetricComputer | compute 37 classification regression clustering statistical metrics | ✅ Implemented & Tested |
| **FILE-012** | `src/infrastructure/streams/stream_manager.py` | StreamManager | isolate 6 cuda execution streams and optimizers | ✅ Implemented & Tested |
| **FILE-013** | `src/infrastructure/checkpoint/serializer.py` | CheckpointSerializer | serialize checkpoints using fp16 safetensors file format | ✅ Implemented & Tested |
| **FILE-014** | `src/infrastructure/checkpoint/discovery.py` | CheckpointDiscovery | scan drive recursively and validate newest safetensors checkpoint | ✅ Implemented & Tested |
| **FILE-015** | `src/infrastructure/logging/session_logger.py` | SessionLogger | profile hardware stats and log session telemetry to duckdb | ✅ Implemented & Tested |
| **FILE-016** | `src/infrastructure/logging/prediction_logger.py` | PredictionLogger | export sample predictions and 37 metrics to duckdb | ✅ Implemented & Tested |
| **FILE-017** | `src/application/orchestrator/training_loop.py` | TrainingLoop | execute epoch iterations across paradigm streams with ntp | ✅ Implemented & Tested |
| **FILE-018** | `src/application/fault_tolerance/recovery_manager.py` | RecoveryManager | catch runtime failures and trigger emergency recovery | ✅ Implemented & Tested |
| **FILE-019** | `src/interfaces/cli/main.py` | MainRunner | sequence end to end colab training pipeline | ✅ Implemented & Tested |

---

## 5. Traceability ID Chain

```
REQ-001 -> SPEC-001 -> SOT-001 (ModelArchitecture) -> FILE-002, FILE-003, FILE-004, FILE-005
REQ-002 -> SPEC-002 -> SOT-001 & SOT-003           -> FILE-006, FILE-008
REQ-003 -> SPEC-003 -> SOT-002 (DataLoading)       -> FILE-007, FILE-010
REQ-004 -> SPEC-004 -> SOT-004 (TrainingOrchestrator) -> FILE-012
REQ-005 -> SPEC-005 -> SOT-005 (CheckpointManager) -> FILE-013 (.safetensors)
REQ-006 -> SPEC-006 -> SOT-005 (CheckpointManager) -> FILE-014
REQ-007 -> SPEC-007 -> SOT-006 (GoogleDriveStorage)-> FILE-009
REQ-008 -> SPEC-008 -> SOT-003 (LossComputation)   -> FILE-011
REQ-009 -> SPEC-009 -> SOT-007 (Logging)           -> FILE-016 (multimodal_telemetry.duckdb)
REQ-010 -> SPEC-010 -> SOT-007 (Logging)           -> FILE-015 (multimodal_telemetry.duckdb)
REQ-011 -> SPEC-011 -> SOT-010 & SOT-005           -> FILE-019, FILE-013
REQ-012 -> SPEC-012 -> SOT-008 (FaultTolerance)    -> FILE-018
REQ-013 -> SPEC-013 -> SOT-009 (Configuration)     -> FILE-001
REQ-014 -> SPEC-014 -> SOT-006 (GoogleDriveStorage)-> FILE-009
REQ-015 -> SPEC-015 -> SOT-004 (TrainingOrchestrator) -> FILE-012, FILE-017
REQ-016 -> SPEC-016 -> SOT-002 (DataLoading)       -> FILE-010 (Rule 12 Authentic Data)
REQ-017 -> SPEC-017 -> SOT-001 & SOT-003           -> FILE-006, FILE-008, FILE-017 (NTP Thought Sequences)
REQ-018 -> SPEC-018 -> SOT-007 (Logging)           -> FILE-015, FILE-016 (Single DuckDB Consolidation)
```

---

## 6. Strategic Expansion Roadmap & Future Plan

1. **Phase 5: High-Scale Dataset Integration (HuggingFace Streams):**
   - Connect `MultimodalPyTorchDataset` to stream `MMMU`, `ScienceQA`, `ChartQA`, and `CodeContests` datasets directly from HuggingFace datasets hub.
2. **Phase 6: Large-Scale Epoch Scaling (1,000+ Epochs on Multi-GPU / Distributed DDP):**
   - Scale `ParadigmTrainingOrchestrator` to support Distributed Data Parallel (DDP) multi-GPU clusters while maintaining single consolidated `.safetensors` weight exports (<16 MB).
3. **Phase 7: Interactive Reasoning & Thought Visualization UI:**
   - Query `multimodal_telemetry.duckdb` directly from Next.js / Streamlit web interface to render live 3D Poincaré ball geodesic embeddings, confidence heatmaps, and next-token thought generation logs.
