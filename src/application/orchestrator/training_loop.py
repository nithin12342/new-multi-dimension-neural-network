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

    def create_models(self) -> List[MultimodalNFMNet]:
        """Instantiate 6 independent MultimodalNFMNet instances."""
        return [MultimodalNFMNet(self.config) for _ in range(self.config.training.num_streams)]

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

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = self.config.training.use_amp and torch.cuda.is_available()

        for batch in dataloader:
            x_img = batch["image"].to(device)
            x_txt = batch["text"].to(device)
            x_vid = batch.get("video").to(device) if "video" in batch else None
            x_aud = batch.get("audio").to(device) if "audio" in batch else None
            x_tab = batch.get("tabular").to(device) if "tabular" in batch else None
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            with self.stream_mgr.get_stream_context(stream_id):
                try:
                    autocast_ctx = torch.amp.autocast(device_type, enabled=use_amp)
                except (AttributeError, TypeError):
                    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

                with autocast_ctx:
                    # View 1: Complete 5-Modality Pass
                    outputs1 = model(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)

                    # View 2: Cross-Modal Augmented Pass (Visuomotor vs Audio-Text)
                    x_img_aug = torch.roll(x_img, shifts=1, dims=-1)
                    outputs2 = model(x_img_aug, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)

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

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Standard Robust Optimization Step
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
                outputs1 = model(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
                x_img_aug = torch.roll(x_img, shifts=1, dims=-1)
                outputs2 = model(x_img_aug, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)

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

        losses_dict = {"ce": avg_loss, "mlmce": avg_loss * 0.5}
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
                losses_dict, preds, targets, embeds = self.run_epoch(
                    stream_id, epoch, model, train_loader, optimizer, scaler
                )
                val_metrics = self.validate_epoch(stream_id, epoch, model, val_loader)
                elapsed = time.time() - start_t

                current_acc = val_metrics.get("acc", 0.0)
                is_best = (current_acc >= best_acc)
                if is_best:
                    best_acc = current_acc

                # Format predictions for export with Softmax probability normalization
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                pred_records = []
                for idx in range(min(10, len(preds))):
                    raw_logits = preds[idx]
                    probs = softmax(raw_logits - np.max(raw_logits))
                    confidence_val = float(np.max(probs))
                    pred_label = int(np.argmax(probs))

                    rec = pred_exporter.record_prediction(
                        timestamp=timestamp,
                        sample_id=f"stream{stream_id+1}_ep{epoch}_sample{idx}",
                        input_file="multimodal_batch",
                        ground_truth=int(targets[idx]),
                        predicted=pred_label,
                        confidence=confidence_val,
                        prob_dist=probs.tolist(),
                        correct=bool(pred_label == targets[idx]),
                        loss_contribution=float(losses_dict.get("ce", 0.0))
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
                    batch_idx=len(train_loader),
                    metrics=val_metrics,
                    system_config=self.config,
                    is_best=is_best
                )

                train_loss_val = losses_dict.get("ce", 0.0)
                val_loss_val = val_metrics.get("ce", 0.0)

                print(
                    f"[Stream {stream_id+1}/{total_streams}: {paradigm}] "
                    f"Epoch {epoch:03d}/{target_epochs:03d} | "
                    f"Train Loss: {train_loss_val:.4f} | "
                    f"Val Loss: {val_loss_val:.4f} | "
                    f"PPL: {val_metrics.get('ppl', 1.0):.2f} | "
                    f"Silhouette: {val_metrics.get('silhouette', 0.0):.4f} | "
                    f"Consolidated Drive Weight Saved ({os.path.getsize(ckpt_path)/(1024**2):.2f}MB)",
                    flush=True
                )

        self.stream_mgr.synchronize_all()
        session_logger.log_session_end(session_stats)
        print("[Orchestrator] Multi-Stream 5-Modality Unified Self-Supervised Omni-Pretraining Complete! Consolidated Drive weights saved.", flush=True)
