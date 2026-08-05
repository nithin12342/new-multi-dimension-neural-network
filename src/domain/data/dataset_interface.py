"""
FILE-007 | FOLDER-003 | src/domain/data/dataset_interface.py
Owning Aggregate: DatasetRegistry
Responsibility: define abstract dataset loader and preprocessing interfaces
Must Never: execute concrete download network requests
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch

class AbstractMultimodalDataset(ABC):
    """Abstract base class for all multimodal dataset loaders."""

    @abstractmethod
    def __len__(self) -> int:
        """Return total dataset sample count."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample dict containing image, text, label, and metadata."""
        pass

    @abstractmethod
    def download(self) -> None:
        """Abstract method for downloading raw dataset files."""
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """Abstract method for preprocessing raw dataset files."""
        pass
