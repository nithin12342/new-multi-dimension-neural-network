# SKELETON.md — MultimodalNFMNet Training Framework

> **System:** Robust Multimodal Training Pipeline for MultimodalNFMNet  
> **Language:** Python (PyTorch)  
> **Target Runtime:** Google Colab (T4 GPU, 15GB VRAM, 12GB RAM)  
> **Phase:** Planning (Phase 0)  
> **Source of Truth:** [context.md](context.md) §11 + [robust_multimodal_training_pipeline_prompt.md](robust_multimodal_training_pipeline_prompt.md)

---

## Requirements

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

## Bounded Contexts

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

## Data Flow Order

```
Configuration → GoogleDriveStorage → DataLoading → ModelArchitecture → LossComputation
    → TrainingOrchestrator → CheckpointManager → Logging → FaultTolerance → Entrypoint
```

**Total order (producers before consumers):**
1. Configuration
2. GoogleDriveStorage
3. DataLoading
4. ModelArchitecture
5. LossComputation
6. TrainingOrchestrator
7. CheckpointManager
8. Logging
9. FaultTolerance
10. Entrypoint

---

## Validation (Phase 0 Exit Gate)

| Check | Status |
|-------|--------|
| Every REQ maps to ≥1 aggregate | ✅ REQ-001→ModelArchitecture, REQ-002→ParadigmHeads+LossComputation, REQ-003→DataLoading, REQ-004→StreamManager, REQ-005→CheckpointSerializer, REQ-006→CheckpointDiscovery, REQ-007→GoogleDriveStorage, REQ-008→MetricComputer, REQ-009→PredictionLogger, REQ-010→SessionLogger, REQ-011→MainRunner+CheckpointSerializer, REQ-012→RecoveryManager, REQ-013→ConfigRegistry+VersionTracker, REQ-014→DriveManager, REQ-015→StreamManager |
| Every aggregate has non-empty invariant | ✅ All 16 aggregates carry invariant statements |
| data_flow_order is total with no cycles | ✅ Linear chain: Config→GDrive→Data→Model→Loss→Training→Checkpoint→Logging→Fault→Entry |

---

## REQ → SPEC → SOT Traceability

| REQ | SPEC | SOT (Bounded Context) | Aggregate(s) |
|-----|------|----------------------|--------------|
| REQ-001 | SPEC-001 | SOT-001 ModelArchitecture | ChebyshevFunctionalBlock, ModalityTokenizers, ConformalRiemannianChart |
| REQ-002 | SPEC-002 | SOT-001 + SOT-003 | ParadigmHeads, LossFunctions |
| REQ-003 | SPEC-003 | SOT-002 DataLoading | DatasetRegistry, MultimodalCollator |
| REQ-004 | SPEC-004 | SOT-004 TrainingOrchestrator | StreamManager |
| REQ-005 | SPEC-005 | SOT-005 CheckpointManager | CheckpointSerializer |
| REQ-006 | SPEC-006 | SOT-005 CheckpointManager | CheckpointDiscovery |
| REQ-007 | SPEC-007 | SOT-006 GoogleDriveStorage | DriveManager |
| REQ-008 | SPEC-008 | SOT-003 LossComputation | MetricComputer |
| REQ-009 | SPEC-009 | SOT-007 Logging | PredictionLogger |
| REQ-010 | SPEC-010 | SOT-007 Logging | SessionLogger |
| REQ-011 | SPEC-011 | SOT-010 + SOT-005 | MainRunner, CheckpointSerializer |
| REQ-012 | SPEC-012 | SOT-008 FaultTolerance | RecoveryManager |
| REQ-013 | SPEC-013 | SOT-009 Configuration | ConfigRegistry |
| REQ-014 | SPEC-014 | SOT-006 GoogleDriveStorage | DriveManager |
| REQ-015 | SPEC-015 | SOT-004 TrainingOrchestrator | StreamManager |

---

## Next Phase: Architecture (Phase 1)

Phase 0 exit gate satisfied. Ready for Phase 1: map each bounded context to a folder structure following DIP layering. Awaiting user approval to proceed.
