"""
FILE: pretrain.py (Canonical Self-Supervised Pretraining Runner)
Owning Aggregate: PipelineOrchestration
Responsibility: Direct alias to launch 6-stream self-supervised pretraining with full CLI arguments
"""

import sys
from train import main

if __name__ == "__main__":
    sys.exit(main())
