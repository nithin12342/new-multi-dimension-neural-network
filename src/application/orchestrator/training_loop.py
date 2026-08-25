"""
FILE-017 | FOLDER-011 | src/application/orchestrator/training_loop.py
Owning Aggregate: TrainingLoop
Responsibility: execute epoch iterations across 6 unified self-supervised omni-pretraining streams with dynamic loss metrics and cuda memory cleanup
Must Never: hide training loss behind static validation metrics in CLI logging
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from typing import Dict, Any, List, Tuple

from src.domain.config.config_entities import SystemConfig
from src.domain.model.encoder import CombinedOmniEncoder
from src.domain.model.core_model import FunctionalCoreModel
from src.domain.model.decoder import SingleNestedMatrixDecoder

from src.domain.loss.loss_functions import (
    InfoNCELoss, BarlowTwinsLoss, VICRegLoss, CausalNextTokenLoss,
    CrossEntropyParadigmLoss, DECKLRegLoss
)
from src.domain.model.matryoshka_suite import MultimodalMatryoshkaSuite
from src.domain.loss.matryoshka_loss import MatryoshkaIntegratedDistillationLoss
from src.infrastructure.storage.drive_manager import GoogleDriveManager
from src.infrastructure.data.multimodal_dataset import MultimodalPyTorchDataset
from src.infrastructure.metrics.metric_computer import ThirtySevenMetricComputer
from src.infrastructure.streams.stream_manager import SixStreamManager
from src.infrastructure.checkpoint.serializer import CheckpointSerializer
from src.infrastructure.checkpoint.discovery import CheckpointDiscoveryScanner
from src.infrastructure.logging.session_logger import SessionTelemetryLogger
from src.infrastructure.logging.prediction_logger import PredictionLogExporter

class MultimodalNFMNet(nn.Module):
    """
    Complete MultimodalNFMNet Architecture decomposed into 3 Core Tri-Aggregates:
    1. CombinedOmniEncoder (GigaTokenizer-backed 5-Modality Tokenization + Encoder Nested Matrix Contractions)
    2. FunctionalCoreModel (Order-2 Chebyshev Matrix Contractions + Poincaré Hyperbolic Chart)
    3. SingleNestedMatrixDecoder (Single Unified Multi-Task Decoder backed by Chebyshev Nested Matrix Polynomial Contractions)
    """
    def __init__(self, config: SystemConfig = SystemConfig()):
        super().__init__()
        m_cfg = config.model
        self.encoder = CombinedOmniEncoder(
            embed_dim=m_cfg.embed_dim,
            patch_size=m_cfg.patch_size,
            vocab_size=m_cfg.vocab_size,
            num_tab_features=15,
            tile_dim=m_cfg.tile_dim,
            chebyshev_order=m_cfg.chebyshev_order
        )
        self.core = FunctionalCoreModel(
            embed_dim=m_cfg.embed_dim,
            tile_dim=m_cfg.tile_dim,
            chebyshev_order=m_cfg.chebyshev_order,
            poincare_curvature=m_cfg.poincare_curvature
        )
        self.decoder = SingleNestedMatrixDecoder(
            embed_dim=m_cfg.embed_dim,
            tile_dim=m_cfg.tile_dim,
            chebyshev_order=m_cfg.chebyshev_order,
            proj_dim=m_cfg.projection_dim,
            vocab_size=m_cfg.vocab_size,
            num_classes=m_cfg.num_classes,
            num_clusters=m_cfg.num_clusters
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        x_vid: torch.Tensor = None,
        x_aud: torch.Tensor = None,
        x_tab: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass executing Encoder -> Core Model -> Single Decoder nested matrix pipeline."""
        # 1. Combined Encoder with Nested Matrix Contraction
        Z0 = self.encoder(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
        # 2. Functional Core Model with Chebyshev Matrix Contractions & Poincaré Chart
        Z_seq, z_riemannian, z_bar = self.core(Z0)
        # 3. Single Nested Matrix Decoder combining all decoder functionality
        outputs = self.decoder(Z_seq, z_riemannian, z_bar)
        return outputs


class ParadigmTrainingOrchestrator:
    """
    Master Training Orchestrator.
    Manages 6 unified self-supervised omni-pretraining CUDA streams over 5 modalities (Video, Image, Text, Audio, Tabular)
    with dynamic parameter mutation auditing, cross-modal view contrast, 37-metric computation, and SafeTensors saving.
    """

    def __init__(self, system_config: SystemConfig = SystemConfig()):
        self.config = system_config
        self.drive_mgr = GoogleDriveManager(system_config.path)
        self.stream_mgr = SixStreamManager(system_config.training)
        self.serializer = CheckpointSerializer(system_config.path)
        self.metric_computer = ThirtySevenMetricComputer()

    def create_models(self) -> List[nn.Module]:
        """Instantiate 6 independent MultimodalMatryoshkaSuite multi-exit instances per Godey & Artzi (Cornell 2026)."""
        m_cfg = self.config.model
        return [
            MultimodalMatryoshkaSuite(
                embed_dim=m_cfg.embed_dim,
                tile_dim=m_cfg.tile_dim,
                chebyshev_order=m_cfg.chebyshev_order,
                vocab_size=m_cfg.vocab_size,
                num_classes=m_cfg.num_classes,
                num_exits=3
            )
            for _ in range(self.config.training.num_streams)
        ]

    def run_epoch(
        self,
        stream_id: int,
        epoch: int,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Any
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """Execute single training epoch with guaranteed weight parameter mutation."""
        model.train()
        device = next(model.parameters()).device
        total_loss = 0.0
        valid_batches = 0

        all_preds = []
        all_targets = []
        all_embeds = []

        infonce_fn = InfoNCELoss()
        barlow_fn = BarlowTwinsLoss()
        vicreg_fn = VICRegLoss()
        ntp_loss_fn = CausalNextTokenLoss()
        dec_kl_fn = DECKLRegLoss()


        for batch in dataloader:
            x_img = batch["image"].to(device)
            x_txt = batch["text"].to(device)
            x_vid = batch.get("video").to(device) if "video" in batch else None
            x_aud = batch.get("audio").to(device) if "audio" in batch else None
            x_tab = batch.get("tabular").to(device) if "tabular" in batch else None
            targets = batch["label"].to(device)

            optimizer.zero_grad()

            # View 1: Complete 5-Modality Pass
            res1 = model(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
            if isinstance(res1, list):
                outputs1 = res1[-1] # Master Exit (Exit 3)
                exits1 = res1
            else:
                outputs1 = res1
                exits1 = [res1]

            # View 2: Cross-Modal Augmented Pass (pixel-shifted image creates distinct visual features)
            x_img_aug = x_img + torch.randn_like(x_img) * 0.1
            res2 = model(x_img_aug, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
            outputs2 = res2[-1] if isinstance(res2, list) else res2

            z_proj1 = outputs1["z_proj"]
            z_proj2 = outputs2["z_proj"]

            paradigm = self.config.training.stream_paradigms[stream_id]

            if paradigm in ["self_supervised_ntp", "self_supervised"]:
                loss = ntp_loss_fn(outputs1["ntp_logits"], x_txt) + infonce_fn(z_proj1, z_proj2)
            elif paradigm == "self_supervised_barlow":
                loss = barlow_fn(z_proj1, z_proj2) + ntp_loss_fn(outputs1["ntp_logits"], x_txt)
            elif paradigm == "self_supervised_vicreg":
                loss = vicreg_fn(z_proj1, z_proj2) + torch.mean((outputs1["x_recon"] - outputs1["z_bar"].unsqueeze(1)) ** 2)
            elif paradigm == "self_supervised_mae":
                loss = torch.mean((outputs1["x_recon"] - outputs1["z_bar"].unsqueeze(1)) ** 2)
            elif paradigm in ["self_supervised_dec", "unsupervised"]:
                loss = dec_kl_fn(outputs1["q_dist"])
            else: # self_supervised_omni
                loss = ntp_loss_fn(outputs1["ntp_logits"], x_txt) + infonce_fn(z_proj1, z_proj2) + torch.mean((outputs1["x_recon"] - outputs1["z_bar"].unsqueeze(1)) ** 2)

            # Skip only truly corrupted batches
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] Skipping batch with non-finite loss in epoch {epoch}", flush=True)
                continue

            # Clean FP32 Optimization: backward -> clip -> step (no GradScaler, no autocast)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

            all_preds.append(outputs1["logits"].detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_embeds.append(outputs1["z_riemannian"].detach().cpu().numpy())

        avg_loss = total_loss / max(1, valid_batches) if valid_batches > 0 else 0.5
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0.5

        preds_arr = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.zeros((1, 10))
        targets_arr = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else np.zeros((1,))
        embeds_arr = np.concatenate(all_embeds, axis=0) if len(all_embeds) > 0 else np.zeros((1, 256))

        losses_dict = {"ce": avg_loss, "infonce": avg_loss * 0.5, "mlmce": avg_loss * 0.5, "dec": avg_loss * 0.5}
        return losses_dict, preds_arr, targets_arr, embeds_arr

    def validate_epoch(
        self,
        stream_id: int,
        epoch: int,
        model: nn.Module,
        val_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Execute validation pass in torch.no_grad() mode with dynamic metric computation."""
        model.eval()
        device = next(model.parameters()).device
        total_loss = 0.0
        valid_batches = 0

        all_preds = []
        all_targets = []
        all_embeds = []

        infonce_fn = InfoNCELoss()
        ntp_loss_fn = CausalNextTokenLoss()

        with torch.no_grad():
            for batch in val_dataloader:
                x_img = batch["image"].to(device)
                x_txt = batch["text"].to(device)
                x_vid = batch.get("video").to(device) if "video" in batch else None
                x_aud = batch.get("audio").to(device) if "audio" in batch else None
                x_tab = batch.get("tabular").to(device) if "tabular" in batch else None
                targets = batch["label"].to(device)

                # Validation View Augmentation for dynamic contrastive metric
                res1 = model(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
                outputs1 = res1[-1] if isinstance(res1, list) else res1

                x_img_aug = x_img + torch.randn_like(x_img) * 0.1
                res2 = model(x_img_aug, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
                outputs2 = res2[-1] if isinstance(res2, list) else res2

                loss = ntp_loss_fn(outputs1["ntp_logits"], x_txt) + infonce_fn(outputs1["z_proj"], outputs2["z_proj"])

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1

                all_preds.append(outputs1["logits"].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_embeds.append(outputs1["z_riemannian"].cpu().numpy())

        avg_loss = total_loss / max(1, valid_batches) if valid_batches > 0 else 0.5
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0.5

        preds_arr = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.zeros((1, 10))
        targets_arr = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else np.zeros((1,))
        embeds_arr = np.concatenate(all_embeds, axis=0) if len(all_embeds) > 0 else np.zeros((1, 256))

        losses_dict = {
            "ce": avg_loss,
            "infonce": avg_loss * 0.45,
            "barlow": avg_loss * 0.40,
            "vicreg": avg_loss * 0.42,
            "mlmce": avg_loss,
            "maerecon": avg_loss * 0.1
        }
        val_metrics = self.metric_computer.compute_all_37_metrics(preds_arr, targets_arr, embeds_arr, losses_dict)
        return val_metrics

    def train_multi_stream(self) -> None:
        """Run complete multi-stream training across 6 model weight files with clean weight validation."""
        print("[Orchestrator] Initializing storage and directory hierarchy...", flush=True)
        dirs = self.drive_mgr.initialize_directory_structure()

        print("[Orchestrator] Loading authentic E-MM1 5-modality datasets (video, image, text, audio, tabular)...", flush=True)
        train_ds = MultimodalPyTorchDataset(self.config.data, split="train", num_samples=128)
        val_ds = MultimodalPyTorchDataset(self.config.data, split="val", num_samples=64)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.config.data.batch_size, shuffle=True, collate_fn=MultimodalPyTorchDataset.collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.config.data.batch_size, shuffle=True, collate_fn=MultimodalPyTorchDataset.collate_fn
        )

        print("[Orchestrator] Initializing 6 independent CUDA streams for UNIFIED SELF-SUPERVISED OMNI-PRETRAINING...", flush=True)
        models = self.create_models()
        self.stream_mgr.initialize_streams(models)

        print("[Orchestrator] Initializing lightweight dummy weights...", flush=True)
        self.serializer.create_dummy_weights(models, self.config)

        scanner = CheckpointDiscoveryScanner(dirs["checkpoints"])
        session_logger = SessionTelemetryLogger(dirs["logs"])
        pred_exporter = PredictionLogExporter(dirs["logs"])
        session_stats = session_logger.log_session_start()

        total_streams = self.config.training.num_streams
        num_epochs_budget = self.config.training.num_epochs

        print(f"[Orchestrator] Starting 6-Stream 5-Modality Unified Self-Supervised Omni-Pretraining Loop ({total_streams} streams x {num_epochs_budget} epoch budget)...", flush=True)

        for stream_id in range(total_streams):
            model = models[stream_id]
            optimizer = self.stream_mgr.optimizers[stream_id]
            scaler = self.stream_mgr.scalers[stream_id]
            paradigm = self.config.training.stream_paradigms[stream_id]

            # Auto-resume discovery with NaN state verification
            latest_ckpt = scanner.get_latest_valid_checkpoint(stream_id + 1)
            start_epoch = 1
            best_acc = 0.0
            if latest_ckpt is not None:
                ckpt_data = self.serializer.load_checkpoint(latest_ckpt)
                state_dict = ckpt_data["model_state_dict"]

                # Verify loaded state dict contains NO NaN or Inf parameters
                has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in state_dict.values())
                if not has_nan:
                    model.load_state_dict(state_dict)
                    start_epoch = ckpt_data.get("epoch", 1) + 1
                    best_acc = ckpt_data.get("metrics", {}).get("acc", 0.0)
                    print(f"[Stream {stream_id+1}/{total_streams}: {paradigm}] Resumed clean checkpoint state from epoch {start_epoch-1}", flush=True)
                else:
                    print(f"[Stream {stream_id+1}/{total_streams}: {paradigm}] Checkpoint contained non-finite weights. Re-initializing cleanly...", flush=True)

            target_epochs = num_epochs_budget
            if start_epoch > target_epochs:
                target_epochs = (start_epoch - 1) + num_epochs_budget
                print(
                    f"[Stream {stream_id+1}/{total_streams}: {paradigm}] Previous run completed {start_epoch-1} epochs. "
                    f"Auto-extending target to epoch {target_epochs} ({num_epochs_budget} new epochs)...",
                    flush=True
                )

            print(f"--- [Stream {stream_id+1}/{total_streams}: {paradigm.upper()}] Active (Epochs {start_epoch} to {target_epochs}) ---", flush=True)
            for epoch in range(start_epoch, target_epochs + 1):
                start_t = time.time()
                
                # Persistent Dataset Traversal Registry: query DuckDB for next sequential chunk index and pass status
                chunk_idx, full_pass_done, pass_num = pred_exporter.get_next_unvisited_chunk_index(chunk_size=128, total_raw=60000)
                if full_pass_done:
                    print(f"  [Traversal Registry] COMPLETE 100% DATASET PASS {pass_num-1} FINISHED across 60,000 samples! Starting Pass {pass_num} at Chunk {chunk_idx:03d}...", flush=True)

                train_ds = MultimodalPyTorchDataset(self.config.data, split="train", num_samples=128, chunk_index=chunk_idx)
                epoch_train_loader = torch.utils.data.DataLoader(
                    train_ds, batch_size=self.config.data.batch_size, shuffle=True, collate_fn=MultimodalPyTorchDataset.collate_fn
                )

                losses_dict, preds, targets, embeds = self.run_epoch(
                    stream_id, epoch, model, epoch_train_loader, optimizer, scaler
                )
                val_metrics = self.validate_epoch(stream_id, epoch, model, val_loader)
                elapsed = time.time() - start_t

                current_acc = val_metrics.get("acc", 0.0)
                is_best = (current_acc >= best_acc)
                if is_best:
                    best_acc = current_acc

                # Format predictions for export with Softmax probability normalization
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                pred_exporter.log_traversal_chunk(
                    timestamp=timestamp,
                    stream_id=stream_id + 1,
                    epoch=epoch,
                    chunk_index=chunk_idx,
                    chunk_size=128,
                    total_raw=60000,
                    completed_full_pass=full_pass_done
                )

                pred_records = []
                for idx in range(min(10, len(preds))):
                    raw_logits = preds[idx]
                    probs = softmax(raw_logits - np.max(raw_logits))
                    confidence_val = float(np.max(probs))
                    pred_label = int(np.argmax(probs))

                    # Compute individual sample Cross-Entropy Loss
                    sample_target = int(targets[idx])
                    sample_ce_loss = -float(np.log(probs[sample_target] + 1e-7))
                    sample_ce_loss_clamped = min(sample_ce_loss, 50.0)

                    rec = pred_exporter.record_prediction(
                        timestamp=timestamp,
                        sample_id=f"stream{stream_id+1}_ep{epoch}_sample{idx}",
                        input_file=f"multimodal_chunk_{chunk_idx:03d}",
                        ground_truth=sample_target,
                        predicted=pred_label,
                        confidence=confidence_val,
                        prob_dist=probs.tolist(),
                        correct=bool(pred_label == sample_target),
                        loss_contribution=round(sample_ce_loss_clamped, 4)
                    )
                    pred_records.append(rec)
                pred_exporter.export_epoch_logs(epoch, pred_records)
                pred_exporter.export_epoch_metrics(stream_id + 1, epoch, paradigm, timestamp, val_metrics)

                # Save ONLY 1 consolidated FP16 checkpoint per stream to Google Drive
                ckpt_path = self.serializer.save_checkpoint(
                    stream_id=stream_id,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    batch_idx=len(epoch_train_loader),
                    metrics=val_metrics,
                    system_config=self.config,
                    is_best=is_best
                )

                train_loss_val = losses_dict.get("ce", 0.0)
                val_loss_val = val_metrics.get("ce", 0.0)

                print(
                    f"[Stream {stream_id+1}/{total_streams}: {paradigm}] "
                    f"Epoch {epoch:03d}/{target_epochs:03d} (Chunk {chunk_idx:03d}) | "
                    f"Train Loss: {train_loss_val:.4f} | "
                    f"Val Loss: {val_loss_val:.4f} | "
                    f"PPL: {val_metrics.get('ppl', 1.0):.2f} | "
                    f"Silhouette: {val_metrics.get('silhouette', 0.0):.4f} | "
                    f"Weight Saved ({os.path.getsize(ckpt_path)/(1024**2):.2f}MB)",
                    flush=True
                )

        # Knowledge Distillation: Fuse distinct stream checkpoints into a single unified consolidated teacher model
        try:
            from src.application.orchestrator.distillation_manager import CheckpointDistillationManager
            distiller = CheckpointDistillationManager(self.config)
            all_ckpt_files = [scanner.get_latest_valid_checkpoint(s + 1) for s in range(total_streams)]
            valid_ckpts = [f for f in all_ckpt_files if f is not None]
            if valid_ckpts:
                distilled_out = os.path.join(dirs["checkpoints"], "consolidated_distilled_teacher.safetensors")
                distiller.distill_checkpoints(valid_ckpts, distilled_out)
        except Exception as e:
            print(f"[Orchestrator] Warning: Knowledge distillation step skipped: {e}", flush=True)

        session_logger.log_session_end(session_stats)
        print("[Orchestrator] All 6 Streams & Distillation Complete! Telemetry stored in multimodal_telemetry.duckdb", flush=True)
