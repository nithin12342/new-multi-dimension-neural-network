"""
FILE-023 | train_omni.py
Owning Aggregate: MainRunner
Responsibility: launch end-to-end 5-modality self-supervised omni-pretraining pipeline
"""

import os
import sys

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.interfaces.cli.main import main

if __name__ == "__main__":
    print("==========================================================================")
    print(" STARTING MULTIMODAL NFMNET 5-MODALITY OMNI-PRETRAINING PIPELINE ")
    print("==========================================================================")
    main()
