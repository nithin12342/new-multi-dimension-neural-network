"""
FILE-008 | FOLDER-004 | src/domain/loss/loss_functions.py
Owning Aggregate: LossFunctions
Responsibility: compute supervised contrastive VICReg and dec clustering losses with FP16 temperature clamping and target token bounds
Must Never: allow un-clamped similarity matrix to cause FP16 overflow or NaN/Inf values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss for Self-Supervised Learning with FP16 overflow clamping."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss over L2-normalized representations z_i, z_j [B, D]."""
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        batch_size = z_i.shape[0]

        representations = torch.cat([z_i, z_j], dim=0) # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature # [2B, 2B]

        # FP16 AMP Overflow Guard: Clamp similarity matrix to [-50, 50] to prevent exp() overflow (>65,504)
        similarity_matrix = torch.clamp(similarity_matrix, min=-50.0, max=50.0)

        # Labels for positive pairs bounded strictly within [0, 2B - 1]
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)], dim=0).to(z_i.device)
        labels = torch.clamp(labels.long(), min=0, max=2 * batch_size - 1)

        # Mask out self-contrastive similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -50.0)

        loss = F.cross_entropy(similarity_matrix, labels)
        return torch.clamp(loss, min=0.0, max=50.0)

class BarlowTwinsLoss(nn.Module):
    """Barlow Twins Cross-Correlation Loss with FP16 clamping."""
    def __init__(self, lambda_param: float = 5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute Barlow Twins cross-correlation loss between views z_a and z_b."""
        N, D = z_a.shape
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-5)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-5)

        C = torch.matmul(z_a_norm.T, z_b_norm) / max(1, N)
        C = torch.clamp(C, min=-50.0, max=50.0)

        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        off_diag = C.flatten()[:-1].view(D - 1, D + 1)[:, 1:].pow_(2).sum()

        loss = on_diag + self.lambda_param * off_diag
        return torch.clamp(loss, min=0.0, max=50.0)

class VICRegLoss(nn.Module):
    """VICReg (Variance-Invariance-Covariance Regularization) Loss with Normalized Weights."""
    def __init__(self, sim_coeff: float = 1.0, std_coeff: float = 1.0, cov_coeff: float = 0.04):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute numerically stable VICReg loss."""
        N, D = z_a.shape

        sim_loss = F.mse_loss(z_a, z_b)

        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1.0 - std_z_a)) + torch.mean(F.relu(1.0 - std_z_b))

        z_a_cent = z_a - z_a.mean(dim=0)
        z_b_cent = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a_cent.T @ z_a_cent) / max(1, N - 1)
        cov_z_b = (z_b_cent.T @ z_b_cent) / max(1, N - 1)
        cov_loss = (cov_z_a.pow(2).sum() - torch.diagonal(cov_z_a).pow(2).sum()) / D + \
                   (cov_z_b.pow(2).sum() - torch.diagonal(cov_z_b).pow(2).sum()) / D

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return torch.clamp(loss, min=0.0, max=50.0)

class CausalNextTokenLoss(nn.Module):
    """Causal Next-Token Prediction Loss over Auto-Regressive Thought Sequences with target token bounds."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, ntp_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute causal next-token cross-entropy loss with target index bounds.
        ntp_logits: [B, N, V] (where N >= S text tokens)
        target_tokens: [B, S]
        """
        batch_size, seq_len = target_tokens.shape
        vocab_size = ntp_logits.size(-1)

        text_seq_len = min(ntp_logits.size(1), seq_len)
        text_logits = ntp_logits[:, -text_seq_len:, :] # [B, S_text, V]
        matched_targets = target_tokens[:, -text_seq_len:]

        shift_logits = text_logits[:, :-1, :].contiguous().view(-1, vocab_size) # [B*(S-1), V]
        shift_targets = matched_targets[:, 1:].contiguous().view(-1).long() # [B*(S-1)]

        shift_targets = torch.clamp(shift_targets, min=0, max=vocab_size - 1)

        loss = self.loss_fn(shift_logits, shift_targets)
        return torch.clamp(loss, min=0.0, max=50.0)

class CrossEntropyParadigmLoss(nn.Module):
    """Cross-Entropy Classification Loss with logit temperature scaling and target class index bounds."""
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with logit temperature scaling and target class bounds."""
        num_classes = logits.size(-1)
        targets_clamped = torch.clamp(targets.long(), min=0, max=num_classes - 1)
        scaled_logits = logits / self.temperature
        loss = self.loss_fn(scaled_logits, targets_clamped)
        return torch.clamp(loss, min=0.0, max=50.0)

class DECKLRegLoss(nn.Module):
    """Deep Embedded Clustering KL-Divergence Loss KL(P || Q)."""
    def __init__(self):
        super().__init__()

    def compute_target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Compute target distribution p_ij from soft assignments q_ij."""
        weight = (q ** 2) / (q.sum(0) + 1e-7)
        p = (weight.T / (weight.sum(1) + 1e-7)).T
        return p.detach()

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute KL-divergence loss KL(P || Q)."""
        p = self.compute_target_distribution(q)
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        return torch.clamp(loss, min=0.0, max=50.0)
