# SKELETON.md — MultimodalNFMNet Training Framework

> **System:** Robust Multimodal Training Pipeline for MultimodalNFMNet  
> **Language:** Python (PyTorch)  
> **Target Runtime:** Google Colab (T4 GPU, 15GB VRAM, 12GB RAM)  
> **Phase:** Implementation Complete (Phase 4 Verified by E2E Execution Fixture)  
> **Source of Truth:** [context.md](context.md) §11 + [robust_multimodal_training_pipeline_prompt.md](robust_multimodal_training_pipeline_prompt.md)

---

## 1. Requirements

- id: REQ-001
  text: "Implement MultimodalNFMNet architecture with Chebyshev Functional Matrix Blocks, Trace-Invariant Activation, and Conformal Riemannian Charting per context.md §11.1"
  spec_id: SPEC-001

- id: REQ-002
  text: "Implement three paradigm heads: SSL (InfoNCE, Barlow Twins, VICReg, Masked Reconstruction), Supervised (Classification CE, Regression MSE/MAE/R²), Unsupervised (DEC Student-t clustering with KL-divergence)"
  spec_id: SPEC-002

- id: REQ-003
  text: "Support multimodal open-source datasets (image+text minimum) with automatic download, preprocessing, and augmentation"
  spec_id: SPEC-003

- id: REQ-004
  text: "Maintain six independent model weight files across three paradigms (2× Supervised, 2× SSL, 2× Unsupervised) with sequential training"
  spec_id: SPEC-004

- id: REQ-005
  text: "Implement production-quality checkpoint manager: per-epoch saves, best/latest/emergency checkpoints, 37-metric serialized filename convention"
  spec_id: SPEC-005

- id: REQ-006
  text: "Automatic checkpoint discovery: recursive Google Drive scan, timestamp sort, integrity validation, auto-resume from newest valid checkpoint"
  spec_id: SPEC-006

- id: REQ-007
  text: "Store all artifacts (checkpoints, logs, metrics, visualizations, reports) exclusively on Google Drive — nothing permanently on Colab runtime"
  spec_id: SPEC-007

- id: REQ-008
  text: "Compute and log 37 metrics per epoch across all metric families: classification, regression, contrastive/SSL, language modeling, reconstruction, representation learning, clustering, statistical"
  spec_id: SPEC-008

- id: REQ-009
  text: "Detailed per-epoch prediction logging (timestamp, sample ID, ground truth, prediction, confidence, probability distribution, loss contribution) in CSV/JSON/Parquet"
  spec_id: SPEC-009

- id: REQ-010
  text: "Detailed session logging: hardware info, CUDA/PyTorch versions, memory/CPU/GPU utilization, batch speed, LR schedule, optimizer state, all with precise timestamps"
  spec_id: SPEC-010

- id: REQ-011
  text: "Dummy weight initialization: create six model files with random weights, metadata, and version info, stored to Google Drive before training begins"
  spec_id: SPEC-011

- id: REQ-012
  text: "Fault tolerance: graceful recovery from Colab disconnects, runtime resets, CUDA OOM, KeyboardInterrupt, power failures, corrupted checkpoints, GDrive sync delays"
  spec_id: SPEC-012

- id: REQ-013
  text: "Automatic versioning for models, datasets, checkpoints, configurations, metrics, reports — every artifact carries version number, timestamp, parent checkpoint"
  spec_id: SPEC-013

- id: REQ-014
  text: "Organized Google Drive directory structure per spec §14"
  spec_id: SPEC-014

- id: REQ-015
  text: "Multi-stream CUDA execution: 6 isolated CUDA streams with per-stream Model, Optimizer, GradScaler, synchronized metric collection"
  spec_id: SPEC-015

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
    - ports_in: ["Fused token tensor [B, N, 256] ← TokenFusion"]
    - ports_out: ["Backbone output [B, N, 256] → ParadigmHeads"]
  - **ModalityTokenizers**
    - invariant: "Vision patch Conv2D produces [B, N_img, 256]; text embedding produces [B, S, 256]; fusion concatenation preserves D=256"
    - entities: [VisionPatchTokenizer, TextEmbeddingTokenizer, TokenFusion]
    - value_objects: [PatchSize(16), EmbeddingDim(256), VocabSize(30522)]
    - ports_in: ["Raw image [B, 3, H, W] ← DataLoading", "Raw text tokens [B, S] ← DataLoading"]
    - ports_out: ["Fused tokens [B, N, 256] → ChebyshevFunctionalBlock"]
  - **ConformalRiemannianChart**
    - invariant: "Poincaré ball constraint ‖x‖ < 1 enforced; conformal scale λ_x = 2/(1−‖x‖²) always positive"
    - entities: [PoincareBall, MobiusAddition, GeodesicDistance]
    - value_objects: [Curvature(c)]
    - ports_in: ["Backbone features [B, N, 256] ← ChebyshevFunctionalBlock"]
    - ports_out: ["Riemannian-mapped features → ParadigmHeads"]
  - **ParadigmHeads**
    - invariant: "Each head receives pooled [B, 256] and produces paradigm-specific output; heads are independent and never share learnable parameters"
    - entities: [SSLProjectionHead, SupervisedClassificationHead, SupervisedRegressionHead, DECClusteringHead, MaskedReconstructionHead]
    - value_objects: [NumClasses(K), NumClusters, ProjectionDim(128)]
    - ports_in: ["Pooled features [B, 256] ← GlobalPooling"]
    - ports_out: ["Logits/projections/cluster assignments → LossComputation"]

---

### 2. DataLoading
- **reason_separate:** "Owns dataset download, preprocessing, augmentation — never touches model weights or training state"
- **sot_id:** SOT-002
- **aggregates:**
  - **DatasetRegistry**
    - invariant: "Every registered dataset implements a common interface returning (image_tensor, text_tokens, label, metadata) tuples"
    - entities: [DatasetDownloader, DatasetPreprocessor, AugmentationPipeline, DatasetConfig]
    - value_objects: [DatasetVersion, SplitRatio]
    - ports_in: ["Dataset name/config ← Configuration"]
    - ports_out: ["PyTorch DataLoaders → TrainingOrchestrator"]
  - **MultimodalCollator**
    - invariant: "Collated batch always produces dict with keys {image: [B,3,H,W], text: [B,S], label: [...], metadata: {...}}"
    - entities: [PadCollator, BatchAssembler]
    - value_objects: [MaxSeqLen, ImageSize]
    - ports_in: ["Raw samples ← DatasetRegistry"]
    - ports_out: ["Collated batches → TrainingOrchestrator"]

---

### 3. LossComputation
- **reason_separate:** "Owns all loss functions and metric calculations — no model parameters, no training loop logic"
- **sot_id:** SOT-003
- **aggregates:**
  - **LossFunctions**
    - invariant: "Every loss function returns a scalar tensor; combined loss is a weighted sum of active paradigm losses"
    - entities: [InfoNCELoss, BarlowTwinsLoss, VICRegLoss, CrossEntropyLoss, MSELoss, MAELoss, DECKLLoss, ReconstructionLoss]
    - value_objects: [LossWeights, Temperature(τ)]
    - ports_in: ["Model outputs ← ParadigmHeads", "Ground truth ← DataLoading"]
    - ports_out: ["Loss scalar → TrainingOrchestrator", "Per-sample losses → PredictionLogger"]
  - **MetricComputer**
    - invariant: "Computes all 37 metrics; returns a flat dict with standardized keys matching the checkpoint filename convention"
    - entities: [ClassificationMetrics, RegressionMetrics, SSLMetrics, ClusteringMetrics, StatisticalMetrics, RepresentationMetrics]
    - value_objects: [MetricNames(37)]
    - ports_in: ["Predictions + ground truth ← TrainingOrchestrator"]
    - ports_out: ["Metrics dict → CheckpointManager", "Metrics dict → SessionLogger"]

---

### 4. TrainingOrchestrator
- **reason_separate:** "Owns the training loop, validation loop, and multi-stream coordination — never defines model layers or loss math"
- **sot_id:** SOT-004
- **aggregates:**
  - **StreamManager**
    - invariant: "Exactly 6 CUDA streams maintained; each stream owns one (Model, Optimizer, GradScaler) triple; streams never share parameters"
    - entities: [CUDAStream, StreamConfig, StreamScheduler]
    - value_objects: [StreamID(1..6), ParadigmAssignment]
    - ports_in: ["Model architecture ← ModelArchitecture", "DataLoaders ← DataLoading", "Loss functions ← LossComputation"]
    - ports_out: ["Training state → CheckpointManager", "Metrics → MetricComputer"]
  - **TrainingLoop**
    - invariant: "One epoch = one full pass over the assigned DataLoader for the active paradigm; gradient scaling (AMP FP16) always enabled on T4"
    - entities: [EpochRunner, GradientScaler, LRScheduler, OptimizerFactory]
    - value_objects: [LearningRate, BatchSize, NumEpochs, WarmupSteps]
    - ports_in: ["Stream assignments ← StreamManager", "Batches ← DataLoading"]
    - ports_out: ["Epoch results → MetricComputer", "Model state → CheckpointManager"]
  - **ValidationLoop**
    - invariant: "Validation always runs in eval mode with torch.no_grad(); uses same metric pipeline as training"
    - entities: [Validator, EvalModeContext]
    - value_objects: []
    - ports_in: ["Validation DataLoader ← DataLoading", "Model ← StreamManager"]
    - ports_out: ["Validation metrics → MetricComputer"]

---

### 5. CheckpointManager
- **reason_separate:** "Owns all checkpoint I/O, discovery, naming, versioning, and integrity validation — never runs training or computes metrics"
- **sot_id:** SOT-005
- **aggregates:**
  - **CheckpointSerializer**
    - invariant: "Every checkpoint file contains: model state_dict, optimizer state_dict, scheduler state_dict, scaler state_dict, epoch, batch_idx, random seeds, training history, metrics dict, version info"
    - entities: [CheckpointSaver, CheckpointLoader, CheckpointNamer, IntegrityValidator]
    - value_objects: [CheckpointVersion, MetricSignature(37)]
    - ports_in: ["Training state ← TrainingOrchestrator", "Metrics ← MetricComputer"]
    - ports_out: ["Checkpoint files → GoogleDriveStorage"]
  - **CheckpointDiscovery**
    - invariant: "Discovery scans recursively; sorts by timestamp descending; validates by attempting torch.load with weights_only; returns newest valid or None"
    - entities: [RecursiveScanner, TimestampSorter, IntegrityChecker]
    - value_objects: []
    - ports_in: ["Google Drive paths ← GoogleDriveStorage"]
    - ports_out: ["Restored state → TrainingOrchestrator"]

---

### 6. GoogleDriveStorage
- **reason_separate:** "Owns all Google Drive mounting, path management, directory creation, and sync — never touches model logic or training state"
- **sot_id:** SOT-006
- **aggregates:**
  - **DriveManager**
    - invariant: "Drive is mounted before any read/write; all paths resolve under /content/drive/MyDrive/SOTA_Cluster_Shared/; directory structure matches spec §14"
    - entities: [DriveMounter, DirectoryInitializer, PathResolver, SyncMonitor]
    - value_objects: [BasePath, DirectoryLayout]
    - ports_in: ["Mount request ← Entrypoint"]
    - ports_out: ["Resolved paths → CheckpointManager, SessionLogger, PredictionLogger, ReportGenerator"]

---

### 7. Logging
- **reason_separate:** "Owns all logging output — session logs, prediction logs, training logs — never modifies model state"
- **sot_id:** SOT-007
- **aggregates:**
  - **SessionLogger**
    - invariant: "Every session log entry includes ISO-8601 timestamp, GPU info, CUDA version, PyTorch version, memory/CPU/GPU utilization"
    - entities: [SessionRecorder, HardwareProfiler, TimestampFormatter]
    - value_objects: [SessionID, LogFormat]
    - ports_in: ["Hardware stats ← Runtime", "Training progress ← TrainingOrchestrator"]
    - ports_out: ["Session log files → GoogleDriveStorage"]
  - **PredictionLogger**
    - invariant: "Every prediction logged with: timestamp, sample_id, input_file, ground_truth, predicted, confidence, probability_distribution, correct_flag, loss_contribution"
    - entities: [PredictionRecorder, FormatConverter]
    - value_objects: [OutputFormats(CSV, JSON, Parquet)]
    - ports_in: ["Per-sample predictions ← TrainingOrchestrator", "Per-sample losses ← LossComputation"]
    - ports_out: ["Prediction log files → GoogleDriveStorage"]

---

### 8. FaultTolerance
- **reason_separate:** "Owns error handling, emergency checkpointing, and recovery logic — wraps other contexts but never replaces their logic"
- **sot_id:** SOT-008
- **aggregates:**
  - **RecoveryManager**
    - invariant: "Any unhandled exception triggers emergency checkpoint save before propagating; CUDA OOM triggers gradient accumulation halving and retry"
    - entities: [ExceptionHandler, EmergencyCheckpointer, OOMRecovery, GracefulShutdown]
    - value_objects: [MaxRetries(3), BackoffStrategy]
    - ports_in: ["Exceptions ← TrainingOrchestrator", "Checkpoint state ← CheckpointManager"]
    - ports_out: ["Emergency checkpoints → GoogleDriveStorage", "Recovery state → TrainingOrchestrator"]

---

### 9. Configuration
- **reason_separate:** "Owns all hyperparameters, paths, and version tracking — single place to change any tunable"
- **sot_id:** SOT-009
- **aggregates:**
  - **ConfigRegistry**
    - invariant: "Config is immutable after initialization; every config instance carries a version string and creation timestamp"
    - entities: [TrainingConfig, ModelConfig, DataConfig, PathConfig, VersionTracker]
    - value_objects: [ConfigVersion, Seed]
    - ports_in: ["User overrides ← Entrypoint"]
    - ports_out: ["Config → all other contexts"]

---

### 10. Entrypoint
- **reason_separate:** "Owns the top-level orchestration sequence: mount drive → init config → init directories → discover checkpoints → init models → train — nothing else"
- **sot_id:** SOT-010
- **aggregates:**
  - **MainRunner**
    - invariant: "Execution follows strict sequence: mount → config → directories → dummy weights → checkpoint discovery → training loop; any step failure halts the pipeline"
    - entities: [ColabInitializer, PipelineSequencer]
    - value_objects: []
    - ports_in: ["User trigger ← Colab cell execution"]
    - ports_out: ["Initialized pipeline → TrainingOrchestrator"]

---

## 3. Data Flow Order

```
Configuration → GoogleDriveStorage → DataLoading → ModelArchitecture → LossComputation
    → TrainingOrchestrator → CheckpointManager → Logging → FaultTolerance → Entrypoint
```

---

## 4. Phase 1 Architecture: Folder Structure & DIP Layers

| Folder ID | Folder Path | Purpose Statement | Owning Bounded Context | DIP Layer |
|-----------|-------------|-------------------|------------------------|-----------|
| **FOLDER-001** | `src/domain/config` | Configuration domain models and immutability rules | Configuration | Domain |
| **FOLDER-002** | `src/domain/model` | MultimodalNFMNet core mathematical layers and heads | ModelArchitecture | Domain |
| **FOLDER-003** | `src/domain/data` | Abstract dataset interfaces and batch contracts | DataLoading | Domain |
| **FOLDER-004** | `src/domain/loss` | Loss function mathematical formulations | LossComputation | Domain |
| **FOLDER-005** | `src/infrastructure/storage` | Google Drive mounting and path resolution | GoogleDriveStorage | Infrastructure |
| **FOLDER-006** | `src/infrastructure/data` | Concrete dataset downloaders and PyTorch loaders | DataLoading | Infrastructure |
| **FOLDER-007** | `src/infrastructure/metrics` | 37-metric computer and evaluator implementations | LossComputation | Infrastructure |
| **FOLDER-008** | `src/infrastructure/streams` | 6 parallel CUDA execution stream handlers | TrainingOrchestrator | Infrastructure |
| **FOLDER-009** | `src/infrastructure/checkpoint` | Checkpoint serialization and discovery scanners | CheckpointManager | Infrastructure |
| **FOLDER-010** | `src/infrastructure/logging` | Session telemetry and prediction log exporters | Logging | Infrastructure |
| **FOLDER-011** | `src/application/orchestrator` | Epoch training and validation loop controllers | TrainingOrchestrator | Application |
| **FOLDER-012** | `src/application/fault_tolerance` | Exception recovery and emergency checkpointing | FaultTolerance | Application |
| **FOLDER-013** | `src/interfaces/cli` | Top-level Colab execution pipeline entrypoint | Entrypoint | Interfaces |

---

## 5. Phase 2 & Phase 4 Implementation Status

| File ID | Folder ID | File Path | Owning Aggregate | Single Responsibility (<=7 Words) | Must Never Clause | Implementation Status |
|---------|-----------|-----------|------------------|-----------------------------------|-------------------|-----------------------|
| **FILE-001** | FOLDER-001 | `src/domain/config/config_entities.py` | ConfigRegistry | define immutable training and model configuration data structures | modify config values after initialization | ✅ Implemented & Tested |
| **FILE-002** | FOLDER-002 | `src/domain/model/chebyshev.py` | ChebyshevFunctionalBlock | compute order-2 chebyshev functional matrix polynomial contractions | flatten matrix tiles into 1d vectors | ✅ Implemented & Tested |
| **FILE-003** | FOLDER-002 | `src/domain/model/trace_activation.py` | ChebyshevFunctionalBlock | apply trace invariant activation scaling to matrix tiles | cause warp divergence across matrix dimensions | ✅ Implemented & Tested |
| **FILE-004** | FOLDER-002 | `src/domain/model/tokenizers.py` | ModalityTokenizers | tokenize and project image and text inputs | mix patch dimensions across sequence boundaries | ✅ Implemented & Tested |
| **FILE-005** | FOLDER-002 | `src/domain/model/riemannian.py` | ConformalRiemannianChart | map features to poincaré ball conformal charts | allow feature norms to exceed unit disk boundary | ✅ Implemented & Tested |
| **FILE-006** | FOLDER-002 | `src/domain/model/paradigm_heads.py` | ParadigmHeads | project pooled representations into paradigm output heads | share learnable parameters across paradigm head instances | ✅ Implemented & Tested |
| **FILE-007** | FOLDER-003 | `src/domain/data/dataset_interface.py` | DatasetRegistry | define abstract dataset loader and preprocessing interfaces | execute concrete download network requests | ✅ Implemented & Tested |
| **FILE-008** | FOLDER-004 | `src/domain/loss/loss_functions.py` | LossFunctions | compute supervised contrastive and dec clustering losses | mutate model gradients directly inside loss calculations | ✅ Implemented & Tested |
| **FILE-009** | FOLDER-005 | `src/infrastructure/storage/drive_manager.py` | DriveManager | mount google drive and resolve persistent directories | write outputs to colab local temporary storage | ✅ Implemented & Tested |
| **FILE-010** | FOLDER-006 | `src/infrastructure/data/multimodal_dataset.py` | DatasetRegistry | download preprocess and load multimodal dataset batches | return un-collated variable length sequence batches | ✅ Implemented & Tested |
| **FILE-011** | FOLDER-007 | `src/infrastructure/metrics/metric_computer.py` | MetricComputer | compute 37 classification regression clustering statistical metrics | omit any metric key from evaluation dictionary | ✅ Implemented & Tested |
| **FILE-012** | FOLDER-008 | `src/infrastructure/streams/stream_manager.py` | StreamManager | isolate 6 cuda execution streams and optimizers | share cuda streams or scalers across models | ✅ Implemented & Tested |
| **FILE-013** | FOLDER-009 | `src/infrastructure/checkpoint/serializer.py` | CheckpointSerializer | serialize checkpoints with 37 metric signature filenames | overwrite existing valid checkpoints without versioning | ✅ Implemented & Tested |
| **FILE-014** | FOLDER-009 | `src/infrastructure/checkpoint/discovery.py` | CheckpointDiscovery | scan drive recursively and validate newest checkpoint | load corrupted or partial checkpoint files | ✅ Implemented & Tested |
| **FILE-015** | FOLDER-010 | `src/infrastructure/logging/session_logger.py` | SessionLogger | profile hardware stats and log session telemetry | block training execution during logging disk writes | ✅ Implemented & Tested |
| **FILE-016** | FOLDER-010 | `src/infrastructure/logging/prediction_logger.py` | PredictionLogger | export per sample predictions in compressed duckdb database | drop sample predictions or misalign target labels | ✅ Implemented & Tested |
| **FILE-017** | FOLDER-011 | `src/application/orchestrator/training_loop.py` | TrainingLoop | execute epoch iterations across paradigm training streams | skip gradient scaling step during fp16 training | ✅ Implemented & Tested |
| **FILE-018** | FOLDER-012 | `src/application/fault_tolerance/recovery_manager.py` | RecoveryManager | catch runtime failures and trigger emergency recovery | swallow exceptions without saving emergency state | ✅ Implemented & Tested |
| **FILE-019** | FOLDER-013 | `src/interfaces/cli/main.py` | MainRunner | sequence end to end colab training pipeline | start training before validating storage and checkpoints | ✅ Implemented & Tested |

---

## 6. Traceability ID Chain (REQ → SPEC → SOT → FOLDER → FILE)

```
REQ-001 -> SPEC-001 -> SOT-001 (ModelArchitecture) -> FOLDER-002 -> FILE-002, FILE-003, FILE-004, FILE-005
REQ-002 -> SPEC-002 -> SOT-001 & SOT-003           -> FOLDER-002 & FOLDER-004 -> FILE-006, FILE-008
REQ-003 -> SPEC-003 -> SOT-002 (DataLoading)       -> FOLDER-003 & FOLDER-006 -> FILE-007, FILE-010
REQ-004 -> SPEC-004 -> SOT-004 (TrainingOrchestrator) -> FOLDER-008 -> FILE-012
REQ-005 -> SPEC-005 -> SOT-005 (CheckpointManager) -> FOLDER-009 -> FILE-013
REQ-006 -> SPEC-006 -> SOT-005 (CheckpointManager) -> FOLDER-009 -> FILE-014
REQ-007 -> SPEC-007 -> SOT-006 (GoogleDriveStorage)-> FOLDER-005 -> FILE-009
REQ-008 -> SPEC-008 -> SOT-003 (LossComputation)   -> FOLDER-007 -> FILE-011
REQ-009 -> SPEC-009 -> SOT-007 (Logging)           -> FOLDER-010 -> FILE-016
REQ-010 -> SPEC-010 -> SOT-007 (Logging)           -> FOLDER-010 -> FILE-015
REQ-011 -> SPEC-011 -> SOT-010 & SOT-005           -> FOLDER-013 & FOLDER-009 -> FILE-019, FILE-013
REQ-012 -> SPEC-012 -> SOT-008 (FaultTolerance)    -> FOLDER-012 -> FILE-018
REQ-013 -> SPEC-013 -> SOT-009 (Configuration)     -> FOLDER-001 -> FILE-001
REQ-014 -> SPEC-014 -> SOT-006 (GoogleDriveStorage)-> FOLDER-005 -> FILE-009
REQ-015 -> SPEC-015 -> SOT-004 (TrainingOrchestrator) -> FOLDER-008 & FOLDER-011 -> FILE-012, FILE-017
```

---

## 7. Quality Gates & Verification Evidence

- **Phase 0 Planning:** ✅ PASSED (15 REQs mapped to 10 Bounded Contexts).
- **Phase 1 Architecture:** ✅ PASSED (13 Folders mapped to DIP layers).
- **Phase 2 File Design:** ✅ PASSED (19 Files with $\le 7$-word SRP responsibilities).
- **Phase 3 Code Skeleton:** ✅ PASSED (100% clean `py_compile` across all files).
- **Phase 4 Implementation:** ✅ PASSED (All 19 nodes implemented and verified by E2E test fixture `tests/e2e/test_full_pipeline.py`).
