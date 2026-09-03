"""
FILE: src/engine/autograd.py
Owning Aggregate: AutogradBoundary
Responsibility: Strict computational graph detachment boundary preventing tensor memory leaks.
"""

import torch

@torch.jit.script
def to_clean_scalar(tensor: torch.Tensor) -> float:
    """
    Strips tensor computational graph attachments and transfers scalars
    directly to host CPU memory as native Python floats.
    """
    if tensor.requires_grad:
        return float(tensor.detach().cpu().item())
    return float(tensor.cpu().item())
