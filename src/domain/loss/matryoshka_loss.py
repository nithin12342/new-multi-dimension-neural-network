"""
FILE-029 | FOLDER-004 | src/domain/loss/matryoshka_loss.py
Owning Aggregate: MatryoshkaIntegratedDistillationLoss
Responsibility: compute zero-cost online distillation loss from master sub-model M to all smaller nested sub-models m < M
Must Never: allow stop_grad failure or unweighted loss imbalance across exit points
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class MatryoshkaIntegratedDistillationLoss(nn.Module):
    """
    Integrated Zero-Cost Online Distillation Loss for Matryoshka LM Suites (Godey & Artzi, Cornell 2026).
    Distills logits from the largest master exit M to all smaller sub-models m < M during the single forward pass.
    """

    def __init__(self, alpha_d: float = 0.3):
        super().__init__()
        self.alpha_d = alpha_d
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, exit_logits: List[torch.Tensor], target_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            exit_logits: List of logit tensors [l^1, l^2, ..., l^M] where each l^m is [B, V] or [B, N_total, V]
            target_tokens: Ground-truth target token indices [B] or [B, S]
        Returns:
            Dictionary containing total_loss, per_exit_losses, and distillation_losses
        """
        M = len(exit_logits)
        assert M >= 1, "At least 1 exit point is required for Matryoshka Loss."

        total_loss = torch.tensor(0.0, device=exit_logits[0].device)
        per_exit_losses = []
        distill_losses = []

        vocab_size = exit_logits[0].size(-1)
        targets_clamped = torch.clamp(target_tokens.long(), min=0, max=vocab_size - 1)

        for m in range(M):
            logits_m = exit_logits[m]
            if logits_m.dim() == 3 and target_tokens.dim() == 2:
                seq_len = target_tokens.size(1)
                text_logits = logits_m[:, -seq_len:, :] # [B, S, V]
                shift_logits = text_logits[:, :-1, :].contiguous().view(-1, vocab_size)
                shift_targets = targets_clamped[:, 1:].contiguous().view(-1)
            else:
                shift_logits = logits_m.view(-1, vocab_size)
                shift_targets = targets_clamped.view(-1)

            # 1. Supervised Cross-Entropy Loss for exit m
            l_ce = self.ce_loss_fn(shift_logits, shift_targets)

            if m < M - 1:
                # 2. Online Distillation Loss from Master M to exit m
                master_logits_m = exit_logits[-1]
                if master_logits_m.dim() == 3 and target_tokens.dim() == 2:
                    seq_len = target_tokens.size(1)
                    master_text_logits = master_logits_m[:, -seq_len:, :]
                    master_shift_logits = master_text_logits[:, :-1, :].contiguous().view(-1, vocab_size)
                else:
                    master_shift_logits = master_logits_m.view(-1, vocab_size)

                master_probs_flat = F.softmax(master_shift_logits.detach(), dim=-1)
                log_probs_m = F.log_softmax(shift_logits, dim=-1)

                l_distill = -torch.sum(master_probs_flat * log_probs_m, dim=-1).mean()
                distill_losses.append(l_distill)

                # Convex Combination: (1 - alpha_d) * L_ce + alpha_d * L_distill
                l_m = (1.0 - self.alpha_d) * l_ce + self.alpha_d * l_distill
            else:
                distill_losses.append(torch.tensor(0.0, device=logits_m.device))
                l_m = l_ce

            per_exit_losses.append(l_m)
            total_loss = total_loss + l_m

        total_loss = torch.clamp(total_loss / M, min=0.0, max=50.0)

        return {
            "total_loss": total_loss,
            "per_exit_losses": per_exit_losses,
            "distill_losses": distill_losses
        }

# Backward compatibility alias
MatryoshkaLoss = MatryoshkaIntegratedDistillationLoss

