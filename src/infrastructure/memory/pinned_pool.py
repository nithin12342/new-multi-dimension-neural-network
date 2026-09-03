"""
FILE: src/infrastructure/memory/pinned_pool.py
Owning Aggregate: MemoryManagement
Responsibility: Pre-allocated host-device pinned tensor buffer pools for zero-copy DMA streaming.
"""

import torch
from typing import Dict, Tuple

class PinnedTensorPool:
    """
    Host-Device Pinned Tensor Pool.
    Eliminates dynamic CUDA host-side page locking stalls during asynchronous DMA transfers.
    """

    def __init__(self, capacity: int, tensor_specs: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]):
        self.pool = {
            name: [
                torch.empty(shape, dtype=dtype, pin_memory=torch.cuda.is_available())
                for _ in range(capacity)
            ]
            for name, (shape, dtype) in tensor_specs.items()
        }
        self.indices = {name: 0 for name in tensor_specs}
        self.capacity = capacity
        self.tensor_specs = tensor_specs

    def acquire(self, name: str) -> torch.Tensor:
        """Acquire the next pre-allocated pinned tensor in the circular pool."""
        if name not in self.pool:
            raise KeyError(f"Tensor spec '{name}' not found in PinnedTensorPool")
        idx = self.indices[name]
        tensor = self.pool[name][idx]
        self.indices[name] = (idx + 1) % self.capacity
        return tensor
