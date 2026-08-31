"""
FILE-031 | FOLDER-002 | src/domain/model/error_localization.py
Owning Aggregate: MultimodalErrorLocalizationEngine
Responsibility: pinpoint exact token, patch, frame, frequency, and tabular failure coordinates across 5 modalities with prefix rollback
Must Never: smear localized errors across entire sequences or discard valid prefix key-value states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

class MultimodalErrorLocalizationEngine(nn.Module):
    """
    Fine-Grained Failure Localization & Prefix Rollback Engine for MultimodalNFMNet.
    Identifies exact failure coordinates across 5 modalities:
      1. Text: Token index t* and reasoning step s* where CE loss spikes
      2. Image: Spatial patch grid coordinates (h*, w*) where MAE reconstruction residual exceeds threshold
      3. Video: Spatiotemporal coordinate (t*, h*, w*) and temporal transition loss
      4. Audio: Time-frequency bin (f*, t*) in Mel-spectrogram
      5. Tabular: Column feature index d* with maximal projection error
    """

    def __init__(
        self,
        text_loss_threshold: float = 4.0,
        patch_loss_threshold: float = 0.25,
        audio_spectral_threshold: float = 0.30
    ):
        super().__init__()
        self.text_loss_thresh = text_loss_threshold
        self.patch_loss_thresh = patch_loss_threshold
        self.audio_spectral_thresh = audio_spectral_threshold

    def locate_text_failure(
        self,
        ntp_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        prm_step_values: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify exact token index t* and reasoning step s* where cross-entropy loss spikes or PRM value drops.
        ntp_logits: [B, S, VocabSize], target_tokens: [B, S]
        """
        B, S, V = ntp_logits.shape
        target_clamped = torch.clamp(target_tokens[:, :S], 0, V - 1)
        
        token_losses = F.cross_entropy(
            ntp_logits.view(-1, V), target_clamped.contiguous().view(-1), reduction='none'
        ).view(B, S)

        results = []
        for b in range(B):
            losses_b = token_losses[b]
            max_val, max_idx = torch.max(losses_b, dim=0)
            failed_indices = (losses_b > self.text_loss_thresh).nonzero(as_tuple=True)[0].tolist()
            first_error_step = failed_indices[0] if len(failed_indices) > 0 else -1
            
            prm_score = 1.0
            if prm_step_values is not None and b < prm_step_values.shape[0]:
                prm_score = float(prm_step_values[b].mean().item())

            results.append({
                "first_error_step": first_error_step,
                "worst_token_idx": int(max_idx.item()),
                "worst_token_loss": float(max_val.item()),
                "all_failed_tokens": failed_indices,
                "prm_step_score": prm_score
            })
        return results

    def locate_visual_patch_failure(
        self,
        x_recon: torch.Tensor,
        target_features: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Identify exact (h*, w*) patch grid coordinates where reconstruction fails.
        x_recon: [B, N_patches, D], target_features: [B, N_patches, D]
        """
        if x_recon.shape != target_features.shape:
            min_len = min(x_recon.size(1), target_features.size(1))
            x_recon = x_recon[:, :min_len, :]
            target_features = target_features[:, :min_len, :]

        patch_mse = torch.mean((x_recon - target_features) ** 2, dim=-1) # [B, N_patches]
        B, N = patch_mse.shape
        grid_size = max(1, int(N ** 0.5))

        results = []
        for b in range(B):
            mse_b = patch_mse[b]
            failed_p = (mse_b > self.patch_loss_thresh).nonzero(as_tuple=True)[0].tolist()
            coords = [[p // grid_size, p % grid_size] for p in failed_p]
            worst_p = int(torch.argmax(mse_b).item())
            
            results.append({
                "failed_patch_coords": coords,
                "worst_patch_coord": [worst_p // grid_size, worst_p % grid_size],
                "worst_patch_mse": float(mse_b[worst_p].item()),
                "total_failed_patches": len(failed_p)
            })
        return results

    def locate_video_spatiotemporal_failure(
        self,
        video_recon: torch.Tensor,
        video_target: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Identify spatiotemporal coordinates (t*, h*, w*) where video frame reconstruction diverges.
        video_recon, video_target: [B, 3, T, H, W]
        """
        # Pixel-wise squared error across channels: [B, T, H, W]
        err = torch.mean((video_recon - video_target) ** 2, dim=1)
        B, T, H, W = err.shape

        results = []
        for b in range(B):
            err_b = err[b] # [T, H, W]
            frame_losses = torch.mean(err_b, dim=(1, 2)) # [T]
            worst_frame = int(torch.argmax(frame_losses).item())
            
            # Find worst patch in worst frame
            flat_spatial = err_b[worst_frame].view(-1)
            worst_spatial_idx = int(torch.argmax(flat_spatial).item())
            worst_h = worst_spatial_idx // W
            worst_w = worst_spatial_idx % W

            results.append({
                "worst_frame_idx": worst_frame,
                "worst_spatiotemporal_coord": [worst_frame, worst_h, worst_w],
                "worst_frame_mse": float(frame_losses[worst_frame].item())
            })
        return results

    def locate_audio_spectral_failure(
        self,
        audio_recon: torch.Tensor,
        audio_target: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify time-frequency coordinates (f*, t*) in audio Mel-spectrogram.
        audio_recon: [B, 1, F, T]
        """
        B = audio_recon.shape[0]
        if audio_target is not None:
            res = torch.abs(audio_recon - audio_target).squeeze(1) # [B, F, T]
        else:
            res = audio_recon.squeeze(1)

        results = []
        for b in range(B):
            spec = res[b]
            max_flat = int(torch.argmax(spec).item())
            W = spec.shape[-1]
            f_star = max_flat // W
            t_star = max_flat % W
            results.append({
                "worst_freq_bin": f_star,
                "worst_time_bin": t_star,
                "spectral_energy": float(spec[f_star, t_star].item())
            })
        return results

    def locate_tabular_feature_failure(
        self,
        tab_recon: torch.Tensor,
        tab_target: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Identify tabular column feature index d* with maximum reconstruction error.
        tab_recon, tab_target: [B, D_tab]
        """
        err = (tab_recon - tab_target) ** 2 # [B, D_tab]
        B, D = err.shape

        results = []
        for b in range(B):
            err_b = err[b]
            worst_dim = int(torch.argmax(err_b).item())
            results.append({
                "worst_feature_idx": worst_dim,
                "worst_feature_error": float(err_b[worst_dim].item())
            })
        return results

    def rollback_prefix_kv_cache(
        self,
        cached_keys: torch.Tensor,
        cached_values: torch.Tensor,
        rollback_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preserve valid prefix key-value states < rollback_step, discarding invalid future states.
        cached_keys, cached_values: [B, NumHeads, SeqLen, HeadDim]
        """
        valid_step = max(0, rollback_step)
        prefix_keys = cached_keys[:, :, :valid_step, :]
        prefix_values = cached_values[:, :, :valid_step, :]
        return prefix_keys, prefix_values
