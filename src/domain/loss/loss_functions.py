"""
FILE-008 | FOLDER-004 | src/domain/loss/loss_functions.py
Owning Aggregate: LossFunctions
Responsibility: compute supervised contrastive and dec clustering losses
Must Never: mutate model gradients directly inside loss calculations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss for Self-Supervised Learning."""
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

        # Labels for positive pairs
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)], dim=0).to(z_i.device)

        # Mask out self-contrastive similarities (-1e4 fits safely in FP16/AMP without overflow)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e4)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class BarlowTwinsLoss(nn.Module):
    """Barlow Twins Cross-Correlation Loss."""
    def __init__(self, lambda_param: float = 5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute Barlow Twins cross-correlation loss between views z_a and z_b."""
        N, D = z_a.shape
        # Normalize along batch dimension
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-5)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-5)

        # Cross-correlation matrix C [D, D]
        C = torch.matmul(z_a_norm.T, z_b_norm) / N

        # Invariance loss: diagonal terms set to 1
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        # Reduction loss: off-diagonal terms set to 0
        off_diag = C.flatten()[:-1].view(D - 1, D + 1)[:, 1:].pow_(2).sum()

        loss = on_diag + self.lambda_param * off_diag
        return loss

class VICRegLoss(nn.Module):
    """VICReg (Variance-Invariance-Covariance Regularization) Loss."""
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute VICReg loss."""
        N, D = z_a.shape

        # Invariance (MSE)
        sim_loss = F.mse_loss(z_a, z_b)

        # Variance regularization
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1.0 - std_z_a)) + torch.mean(F.relu(1.0 - std_z_b))

        # Covariance regularization
        z_a_cent = z_a - z_a.mean(dim=0)
        z_b_cent = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a_cent.T @ z_a_cent) / (N - 1)
        cov_z_b = (z_b_cent.T @ z_b_cent) / (N - 1)
        cov_loss = (cov_z_a.pow(2).sum() - torch.diagonal(cov_z_a).pow(2).sum()) / D + \
                   (cov_z_b.pow(2).sum() - torch.diagonal(cov_z_b).pow(2).sum()) / D

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

class CausalNextTokenLoss(nn.Module):
    """Causal Next-Token Prediction Loss over Auto-Regressive Thought Sequences."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, ntp_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute causal next-token cross-entropy loss.
        ntp_logits: [B, N, V] (where N >= S text tokens)
        target_tokens: [B, S]
        """
        batch_size, seq_len = target_tokens.shape
        # Extract text portion of sequence logits
        text_logits = ntp_logits[:, -seq_len:, :] # [B, S, V]

        # Shift logits and targets for causal next-token prediction
        shift_logits = text_logits[:, :-1, :].contiguous().view(-1, text_logits.size(-1)) # [B*(S-1), V]
        shift_targets = target_tokens[:, 1:].contiguous().view(-1) # [B*(S-1)]

        return self.loss_fn(shift_logits, shift_targets)

class CrossEntropyParadigmLoss(nn.Module):
    """Cross-Entropy Classification Loss."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self.loss_fn(logits, targets)

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
        return loss
