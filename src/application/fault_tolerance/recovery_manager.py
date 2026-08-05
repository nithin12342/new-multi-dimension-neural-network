"""
FILE-018 | FOLDER-012 | src/application/fault_tolerance/recovery_manager.py
Owning Aggregate: RecoveryManager
Responsibility: catch runtime failures and trigger emergency recovery
Must Never: swallow exceptions without saving emergency state
"""

import sys
import torch
from typing import Callable, Any, Optional

class FaultToleranceManager:
    """
    Fault Tolerance & Recovery Engine.
    Handles CUDA OOM, KeyboardInterrupt, Colab disconnects, and triggers emergency checkpointing.
    """

    def __init__(self, emergency_save_fn: Optional[Callable[[str], None]] = None):
        self.emergency_save_fn = emergency_save_fn

    def handle_oom(self, current_batch_size: int) -> int:
        """Clear CUDA cache and return halved batch size upon CUDA OOM."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        new_batch_size = max(1, current_batch_size // 2)
        return new_batch_size

    def trigger_emergency_checkpoint(self, reason: str) -> None:
        """Invoke emergency checkpoint serializer before process exit."""
        if self.emergency_save_fn is not None:
            try:
                self.emergency_save_fn(reason)
            except Exception as e:
                print(f"[FaultToleranceManager] Emergency save failed: {e}", file=sys.stderr)

    def execute_with_recovery(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Wrap execution in try-except block capturing all fatal exceptions."""
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[FaultToleranceManager] CUDA OOM Detected: {e}", file=sys.stderr)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.trigger_emergency_checkpoint("CUDA_OOM")
            else:
                self.trigger_emergency_checkpoint("RuntimeError")
            raise e
        except KeyboardInterrupt:
            print("[FaultToleranceManager] KeyboardInterrupt detected. Saving emergency checkpoint...", file=sys.stderr)
            self.trigger_emergency_checkpoint("KeyboardInterrupt")
            raise
        except Exception as e:
            print(f"[FaultToleranceManager] Unexpected Failure: {e}", file=sys.stderr)
            self.trigger_emergency_checkpoint(f"Exception_{type(e).__name__}")
            raise e
