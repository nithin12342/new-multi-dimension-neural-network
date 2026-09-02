"""
FILE-028 | FOLDER-002 | src/domain/model/rust_interop.py
Owning Aggregate: RustInteropLayer
Responsibility: bridge PyTorch tensors with native zero-overhead Rust runtime layer (crates/nfm-core) via ctypes/C-ABI
Must Never: allow un-sanitized pointer access or non-contiguous memory slices to cross the FFI boundary
"""

import os
import sys
import ctypes
import torch
import numpy as np
from typing import Optional

class RustNFMRuntime:
    """
    Python Bridge for crates/nfm-core.
    Provides direct zero-copy C-ABI calls into compiled native Rust kernels:
    - Chebyshev Order-2 expansions
    - Poincaré boundary projections (||x|| <= 1 - eps)
    - InfoNCE logit clamping ([-10.8, 10.8])
    """

    _lib: Optional[ctypes.CDLL] = None

    @classmethod
    def load_library(cls) -> Optional[ctypes.CDLL]:
        """Attempt to locate and load compiled nfm-core native library."""
        if cls._lib is not None:
            return cls._lib

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        lib_candidates = [
            os.path.join(project_root, "crates/nfm-core/target/release/nfm_core.dll"),
            os.path.join(project_root, "crates/nfm-core/target/debug/nfm_core.dll"),
            os.path.join(project_root, "crates/nfm-core/target/release/libnfm_core.so"),
            os.path.join(project_root, "crates/nfm-core/target/debug/libnfm_core.so"),
            os.path.join(project_root, "crates/nfm-core/target/release/libnfm_core.dylib"),
        ]

        for path in lib_candidates:
            if os.path.exists(path):
                try:
                    cls._lib = ctypes.CDLL(path)
                    break
                except Exception:
                    pass
        return cls._lib

    @classmethod
    def poincare_project(cls, x: torch.Tensor, dim: int = 256, eps: float = 1e-4) -> torch.Tensor:
        """Projects tensor onto Poincare ball using Rust runtime if available, else PyTorch."""
        lib = cls.load_library()
        if lib is not None and hasattr(lib, "nfm_poincare_project") and not x.is_cuda:
            x_contig = x.contiguous().float()
            ptr = ctypes.cast(x_contig.data_ptr(), ctypes.POINTER(ctypes.c_float))
            lib.nfm_poincare_project(ptr, x_contig.numel(), dim, ctypes.c_float(eps))
            return x_contig

        # Pure PyTorch fallback
        max_norm = 1.0 - eps
        norms = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-7)
        scale = torch.clamp(max_norm / norms, max=1.0)
        return x * scale

    @classmethod
    def clamp_infonce(cls, sim_matrix: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        """Clamps InfoNCE logits to [-10.8, 10.8] using Rust runtime if available, else PyTorch."""
        lib = cls.load_library()
        if lib is not None and hasattr(lib, "nfm_clamped_infonce") and not sim_matrix.is_cuda:
            sim_contig = (sim_matrix / tau).contiguous().float()
            ptr = ctypes.cast(sim_contig.data_ptr(), ctypes.POINTER(ctypes.c_float))
            lib.nfm_clamped_infonce(ptr, sim_contig.numel(), ctypes.c_float(1.0))
            return sim_contig

        # Pure PyTorch fallback
        return torch.clamp(sim_matrix / tau, min=-10.8, max=10.8)
