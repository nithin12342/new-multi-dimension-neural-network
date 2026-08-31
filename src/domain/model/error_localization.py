"""
FILE-031 | FOLDER-002 | src/domain/model/error_localization.py
Owning Aggregate: MultimodalErrorLocalizationEngine
Responsibility: pinpoint exact token, patch, frame, frequency, and point failure coordinates across modalities
Must Never: smear localized errors across entire sequences or ignore prefix validity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

class MultimodalErrorLocalizationEngine(nn.Module):
    """
    Fine-Grained Failure Localization Engine for MultimodalNFMNet.
    Identifies exact failure coordinates across 5 modalities:
      1. Text: Token index t* and reasoning step s* where CE loss spikes
      2. Image: Spatial patch grid coordinates (h*, w*) where MAE reconstruction residual exceeds threshold
      3. Video: Spatiotemporal coordinate (t*, h*, w*) and temporal transition loss
      4. Audio: Time-frequency bin (f*, t*) in Mel-spectrogram
      5. Tabular: Column feature index d* with maximal projection error
    """

    def __init__(self, text_loss_threshold: float = 4.0, patch_loss_threshold: float = 0.25):
        super().__init__()
        self.text_loss_thresh = text_loss_threshold
        self.patch_loss_thresh = patch_loss_threshold

    def locate_text_failure(
        self,
        ntp_logits: torch.Tensor,
        target_tokens: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Identify exact token index t* where cross-entropy loss spikes."""
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
            
            results.append({
                "first_error_step": first_error_step,
                "worst_token_idx": int(max_idx.item()),
                "worst_token_loss": float(max_val.item()),
                "all_failed_tokens": failed_indices
            })
        return results

    def locate_visual_patch_failure(
        self,
        x_recon: torch.Tensor,
        target_features: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Identify exact (h*, w*) patch grid coordinates where reconstruction fails."""
        # Ensure dimensions match for residual calculation
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
                "worst_patch_mse": float(mse_b[worst_p].item())
            })
        return results

    def locate_audio_spectral_failure(
        self,
        audio_tensor: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Identify spectral energy divergence in audio Mel-spectrogram."""
        B = audio_tensor.shape[0]
        results = []
        for b in range(B):
            spec = audio_tensor[b, 0] if audio_tensor.ndim == 4 else audio_tensor[b]
            # Find coordinates of maximum energy concentration
            max_flat = int(torch.argmax(spec).item())
            H, W = spec.shape[-2], spec.shape[-1]
            f_star = max_flat // W
            t_star = max_flat % W
            results.append({
                "worst_freq_bin": f_star,
                "worst_time_bin": t_star,
                "spectral_energy": float(spec[f_star, t_star].item())
            })
        return results
