"""
FILE-008 | FOLDER-004 | src/domain/loss/loss_functions.py
Owning Aggregate: LossFunctions
Responsibility: compute supervised contrastive VICReg and dec clustering losses with FP16 temperature clamping, variance hinge, and target token masking
Must Never: allow un-clamped similarity matrix to cause FP16 overflow or NaN/Inf values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss with strict FP16 overflow clamping (sim <= 10.8 to ensure exp(sim) < 65,504)."""
    def __init__(self, temperature: float = 0.07, max_logit: float = 10.8):
        super().__init__()
        self.temperature = temperature
        self.max_logit = max_logit

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss over L2-normalized representations z_i, z_j [B, D]."""
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        batch_size = z_i.shape[0]

        representations = torch.cat([z_i, z_j], dim=0) # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature # [2B, 2B]

        # FP16 AMP Overflow Guard: Clamp similarity matrix to [-10.8, 10.8] (ln(65504) ~= 11.09)
        # Prevents exp() overflow beyond 65,504 in FP16 arithmetic during chunk transitions (Epochs 21-23 spike fix)
        similarity_matrix = torch.clamp(similarity_matrix, min=-self.max_logit, max=self.max_logit)

        # Labels for positive pairs bounded strictly within [0, 2B - 1]
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)], dim=0).to(z_i.device)
        labels = torch.clamp(labels.long(), min=0, max=2 * batch_size - 1)

        # Mask out self-contrastive similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -self.max_logit)

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
    """
    VICReg Loss with strict variance hinge (gamma=1.0, eps=1e-4) and off-diagonal covariance penalty.
    Maintains active channel variance across all latent dimensions, preventing latent dimensional collapse (evr -> 0).
    """
    def __init__(
        self,
        sim_coeff: float = 1.0,
        std_coeff: float = 5.0,
        cov_coeff: float = 0.01,
        gamma: float = 1.0,
        eps: float = 1e-4
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute numerically stable VICReg loss with variance hinge."""
        N, D = z_a.shape

        sim_loss = F.mse_loss(z_a, z_b)

        # Variance Hinge: penalize any channel whose standard deviation falls below gamma=1.0
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(self.gamma - std_z_a)) + torch.mean(F.relu(self.gamma - std_z_b))

        # Covariance Penalty: decorrelate all off-diagonal channel pairs
        z_a_cent = z_a - z_a.mean(dim=0)
        z_b_cent = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a_cent.T @ z_a_cent) / max(1, N - 1)
        cov_z_b = (z_b_cent.T @ z_b_cent) / max(1, N - 1)
        cov_loss = (cov_z_a.pow(2).sum() - torch.diagonal(cov_z_a).pow(2).sum()) / D + \
                   (cov_z_b.pow(2).sum() - torch.diagonal(cov_z_b).pow(2).sum()) / D

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return torch.clamp(loss, min=0.0, max=50.0)

class CausalNextTokenLoss(nn.Module):
    """
    Causal Next-Token Prediction Loss with padded token masking (ignore_index=0).
    Prevents unpadded token dilution from stalling real sequence perplexity.
    """
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, ntp_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute causal next-token cross-entropy loss with target index bounds and pad masking.
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

        # Mask out out-of-vocabulary indices to ignore_index so they do not corrupt gradients
        valid_mask = (shift_targets >= 0) & (shift_targets < vocab_size)
        shift_targets = torch.where(valid_mask, shift_targets, torch.tensor(self.ignore_index, device=shift_targets.device, dtype=torch.long))

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

# Backward compatibility & explicit naming alias
ClampedInfoNCELoss = InfoNCELoss

