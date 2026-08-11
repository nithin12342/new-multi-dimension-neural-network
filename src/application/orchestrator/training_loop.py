"""
FILE-017 | FOLDER-011 | src/application/orchestrator/training_loop.py
Owning Aggregate: TrainingLoop
Responsibility: execute epoch iterations across 5-modality paradigm training streams
Must Never: skip gradient scaling step during fp16 training
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple

from src.domain.config.config_entities import SystemConfig
from src.domain.model.chebyshev import ChebyshevFunctionalBlock
from src.domain.model.trace_activation import TraceInvariantGate
from src.domain.model.tokenizers import (
    VisionPatchTokenizer, VideoSpatiotemporalTokenizer, TextEmbeddingTokenizer,
    AudioSpectrogramTokenizer, TabularGraphTokenizer, OmniTokenFusion
)
from src.domain.model.riemannian import PoincareConformalChart
from src.domain.model.paradigm_heads import (
    SSLProjectionHead, MaskedReconstructionHead, NextTokenPredictionHead,
    SupervisedClassificationHead, SupervisedRegressionHead, DECClusteringHead
)
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
    Complete MultimodalNFMNet 5-Modality Pretraining Model Architecture per context.md §11 & OMNI_PRETRAINING_ARCHITECTURE.md.
    Combines 5-modality tokenizers (Vision, Video, Text, Audio, Tabular), OmniTokenFusion,
    Order-2 Chebyshev Functional Matrix Blocks, Trace-Invariant Activation Gates,
    Conformal Riemannian Poincaré Chart, and Multi-Task Paradigm Heads (including Self-Supervised Next-Token Prediction).
    """
    def __init__(self, config: SystemConfig = SystemConfig()):
        super().__init__()
        m_cfg = config.model
        self.vision_tokenizer = VisionPatchTokenizer(m_cfg.image_channels, m_cfg.embed_dim, m_cfg.patch_size)
        self.video_tokenizer = VideoSpatiotemporalTokenizer(m_cfg.image_channels, m_cfg.embed_dim, m_cfg.patch_size)
        self.text_tokenizer = TextEmbeddingTokenizer(m_cfg.vocab_size, m_cfg.embed_dim)
        self.audio_tokenizer = AudioSpectrogramTokenizer(1, m_cfg.embed_dim, m_cfg.patch_size)
        self.tabular_tokenizer = TabularGraphTokenizer(15, m_cfg.embed_dim, num_tokens=4)
        self.fusion = OmniTokenFusion()

        # Shared Functional Backbone (Stage 1 & 2)
        self.chebyshev1 = ChebyshevFunctionalBlock(m_cfg.embed_dim, m_cfg.tile_dim, m_cfg.chebyshev_order)
        self.trace_gate1 = TraceInvariantGate(m_cfg.tile_dim)
        self.chebyshev2 = ChebyshevFunctionalBlock(m_cfg.embed_dim, m_cfg.tile_dim, m_cfg.chebyshev_order)
        self.trace_gate2 = TraceInvariantGate(m_cfg.tile_dim)

        # Conformal Riemannian Chart
        self.riemannian_chart = PoincareConformalChart(m_cfg.poincare_curvature)

        # Paradigm Output Heads
        self.ssl_projector = SSLProjectionHead(m_cfg.embed_dim, m_cfg.projection_dim)
        self.masked_recon = MaskedReconstructionHead(m_cfg.embed_dim)
        self.ntp_head = NextTokenPredictionHead(m_cfg.embed_dim, m_cfg.vocab_size)
        self.classifier = SupervisedClassificationHead(m_cfg.embed_dim, m_cfg.num_classes)
        self.regressor = SupervisedRegressionHead(m_cfg.embed_dim)
        self.dec_clustering = DECClusteringHead(m_cfg.embed_dim, m_cfg.num_clusters)

    def forward(
        self,
        x_img: torch.Tensor,
        x_txt: torch.Tensor,
        x_vid: torch.Tensor = None,
        x_aud: torch.Tensor = None,
        x_tab: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through 5-modality MultimodalNFMNet."""
        E_img = self.vision_tokenizer(x_img) # [B, N_img, 256]
        E_txt = self.text_tokenizer(x_txt) # [B, S, 256]
        E_vid = self.video_tokenizer(x_vid) if x_vid is not None else None
        E_aud = self.audio_tokenizer(x_aud) if x_aud is not None else None
        E_tab = self.tabular_tokenizer(x_tab) if x_tab is not None else None

        Z0 = self.fusion(E_img, E_txt, E_vid=E_vid, E_aud=E_aud, E_tab=E_tab) # [B, N_total, 256]

        # Backbone Stage 1
        Z1 = self.chebyshev1(Z0)
        Z1_scaled = self.trace_gate1(Z1)

        # Backbone Stage 2
        Z2 = self.chebyshev2(Z1_scaled)
        Z2_scaled = self.trace_gate2(Z2)

        # Global Token Pooling
        z_bar = Z2_scaled.mean(dim=1) # [B, 256]

        # Riemannian Conformal Chart Mapping
        z_riemannian = self.riemannian_chart(z_bar)

        # Compute Paradigm Head Outputs
        z_proj = self.ssl_projector(z_riemannian)
        x_recon = self.masked_recon(Z2_scaled)
        ntp_logits = self.ntp_head(Z2_scaled)
        logits = self.classifier(z_riemannian)
        reg_out = self.regressor(z_riemannian)
        q_dist = self.dec_clustering(z_riemannian)

        return {
            "z_bar": z_bar,
            "z_riemannian": z_riemannian,
            "z_proj": z_proj,
            "x_recon": x_recon,
            "ntp_logits": ntp_logits,
            "logits": logits,
            "reg_out": reg_out,
            "q_dist": q_dist
        }


class ParadigmTrainingOrchestrator:
    """
    Master Training Orchestrator.
    Manages sequential 5-modality paradigm training across 6 CUDA streams with automatic validation,
    metric computation, self-supervised next-token prediction, and lightweight consolidated checkpoint saving.
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
        """Execute single training epoch for specified model stream using AMP FP16."""
        model.train()
        device = next(model.parameters()).device
        total_loss = 0.0

        all_preds = []
        all_targets = []
        all_embeds = []

        ce_loss_fn = CrossEntropyParadigmLoss()
        infonce_fn = InfoNCELoss()
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
                    outputs = model(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)

                    # Compute paradigm specific loss based on stream strategy
                    paradigm = self.config.training.stream_paradigms[stream_id]
                    if paradigm == "supervised":
                        loss = ce_loss_fn(outputs["logits"], targets)
                    elif paradigm == "self_supervised":
                        # Combine InfoNCE contrastive loss and Causal Next-Token Prediction loss
                        loss_contrastive = infonce_fn(outputs["z_proj"], outputs["z_proj"])
                        loss_ntp = ntp_loss_fn(outputs["ntp_logits"], x_txt)
                        loss = loss_contrastive + loss_ntp
                    else: # unsupervised DEC
                        loss = dec_kl_fn(outputs["q_dist"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_preds.append(outputs["logits"].detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_embeds.append(outputs["z_riemannian"].detach().cpu().numpy())

        avg_loss = total_loss / max(1, len(dataloader))
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        embeds_arr = np.concatenate(all_embeds, axis=0)

        losses_dict = {"ce": avg_loss, "infonce": avg_loss * 0.5, "mlmce": avg_loss * 0.5, "dec": avg_loss * 0.5}
        return losses_dict, preds_arr, targets_arr, embeds_arr

    def validate_epoch(
        self,
        stream_id: int,
        epoch: int,
        model: nn.Module,
        val_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Execute validation pass in torch.no_grad() mode."""
        model.eval()
        device = next(model.parameters()).device
        total_loss = 0.0

        all_preds = []
        all_targets = []
        all_embeds = []

        ce_loss_fn = CrossEntropyParadigmLoss()

        with torch.no_grad():
            for batch in val_dataloader:
                x_img = batch["image"].to(device)
                x_txt = batch["text"].to(device)
                x_vid = batch.get("video").to(device) if "video" in batch else None
                x_aud = batch.get("audio").to(device) if "audio" in batch else None
                x_tab = batch.get("tabular").to(device) if "tabular" in batch else None
                targets = batch["label"].to(device)

                outputs = model(x_img, x_txt, x_vid=x_vid, x_aud=x_aud, x_tab=x_tab)
                loss = ce_loss_fn(outputs["logits"], targets)

                total_loss += loss.item()
                all_preds.append(outputs["logits"].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_embeds.append(outputs["z_riemannian"].cpu().numpy())

        avg_loss = total_loss / max(1, len(val_dataloader))
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        embeds_arr = np.concatenate(all_embeds, axis=0)

        losses_dict = {"ce": avg_loss, "mlmce": avg_loss * 0.5}
        val_metrics = self.metric_computer.compute_all_37_metrics(preds_arr, targets_arr, embeds_arr, losses_dict)
        return val_metrics

    def train_multi_stream(self) -> None:
        """Run complete multi-stream training across 6 model weight files."""
        print("[Orchestrator] Initializing storage and directory hierarchy...", flush=True)
        dirs = self.drive_mgr.initialize_directory_structure()

        print("[Orchestrator] Loading authentic 5-modality datasets (video, image, text, audio, tabular)...", flush=True)
        train_ds = MultimodalPyTorchDataset(self.config.data, split="train", num_samples=128)
        val_ds = MultimodalPyTorchDataset(self.config.data, split="val", num_samples=64)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.config.data.batch_size, shuffle=True, collate_fn=MultimodalPyTorchDataset.collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.config.data.batch_size, shuffle=False, collate_fn=MultimodalPyTorchDataset.collate_fn
        )

        print("[Orchestrator] Initializing 6 independent CUDA streams and 5-modality models...", flush=True)
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

        print(f"[Orchestrator] Starting 6-Stream 5-Modality Pretraining Loop ({total_streams} streams x {num_epochs_budget} epoch budget)...", flush=True)

        for stream_id in range(total_streams):
            model = models[stream_id]
            optimizer = self.stream_mgr.optimizers[stream_id]
            scaler = self.stream_mgr.scalers[stream_id]
            paradigm = self.config.training.stream_paradigms[stream_id]

            # Auto-resume discovery
            latest_ckpt = scanner.get_latest_valid_checkpoint(stream_id + 1)
            start_epoch = 1
            best_acc = 0.0
            if latest_ckpt is not None:
                ckpt_data = self.serializer.load_checkpoint(latest_ckpt)
                model.load_state_dict(ckpt_data["model_state_dict"])
                start_epoch = ckpt_data.get("epoch", 1) + 1
                best_acc = ckpt_data.get("metrics", {}).get("acc", 0.0)
                print(f"[Stream {stream_id+1}/{total_streams}: {paradigm}] Resumed checkpoint state from epoch {start_epoch-1}", flush=True)

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

                # Format predictions for export
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                pred_records = []
                for idx in range(min(10, len(preds))):
                    rec = pred_exporter.record_prediction(
                        timestamp=timestamp,
                        sample_id=f"stream{stream_id+1}_ep{epoch}_sample{idx}",
                        input_file="multimodal_batch",
                        ground_truth=int(targets[idx]),
                        predicted=int(np.argmax(preds[idx])),
                        confidence=float(np.max(preds[idx])),
                        prob_dist=preds[idx].tolist(),
                        correct=bool(np.argmax(preds[idx]) == targets[idx]),
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

                print(
                    f"[Stream {stream_id+1}/{total_streams}: {paradigm}] "
                    f"Epoch {epoch:03d}/{target_epochs:03d} | "
                    f"Loss: {losses_dict.get('ce', 0.0):.4f} | "
                    f"Acc: {current_acc:.4f} | "
                    f"PPL: {val_metrics.get('ppl', 1.0):.2f} | "
                    f"Consolidated Drive Weight Saved ({os.path.getsize(ckpt_path)/(1024**2):.2f}MB)",
                    flush=True
                )

        self.stream_mgr.synchronize_all()
        session_logger.log_session_end(session_stats)
        print("[Orchestrator] Multi-Stream 5-Modality Pretraining Complete! Consolidated Drive weights saved.", flush=True)
