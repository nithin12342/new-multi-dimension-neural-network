"""
FILE-019 | FOLDER-013 | src/interfaces/cli/main.py
Owning Aggregate: MainRunner
Responsibility: sequence end to end colab training pipeline
Must Never: start training before validating storage and checkpoints
"""

import sys
from src.domain.config.config_entities import SystemConfig
from src.application.orchestrator.training_loop import ParadigmTrainingOrchestrator
from src.application.fault_tolerance.recovery_manager import FaultToleranceManager

class MultimodalNFMNetPipelineCLI:
    """
    Top-Level Colab Execution Pipeline Entrypoint.
    Executes 6-step initialization: Mount Drive -> Init Directories -> Config -> Dummy Weights -> Discovery -> Training.
    """

    def __init__(self, config: SystemConfig = SystemConfig()):
        self.config = config
        self.orchestrator = ParadigmTrainingOrchestrator(config)
        self.fault_manager = FaultToleranceManager()

    def run_full_pipeline(self) -> None:
        """Sequence end-to-end training framework execution with fault tolerance."""
        print("==========================================================================")
        print(" 🚀 Starting MultimodalNFMNet Robust Training Framework (Google Colab T4) ")
        print("==========================================================================")

        def _execute():
            self.orchestrator.train_multi_stream()

        self.fault_manager.execute_with_recovery(_execute)

def main():
    """Main CLI entrypoint function."""
    cli = MultimodalNFMNetPipelineCLI()
    cli.run_full_pipeline()

if __name__ == "__main__":
    main()
