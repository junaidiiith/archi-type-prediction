"""
Abstract interfaces for text-classification backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any
from datasets import DatasetDict


class TextClassificationModel(ABC):
    """
    Base class exposing a unified API across BERT, Unsloth, and API LLM flows.
    """

    def __init__(
        self,
        model_name: str,
        *,
        output_dir: str = "runs/artifacts",
        seed: Optional[int] = 3407,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    @abstractmethod
    def train(
        self,
        dataset: DatasetDict,
        *,
        evaluation_dataset: Optional[DatasetDict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on the provided dataset.

        Returns a dictionary containing training statistics (loss, runtime, etc.).
        """

    @abstractmethod
    def evaluate(
        self,
        dataset: DatasetDict,
        *,
        split: str = "test",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run inference on the selected split and return metrics plus per-sample predictions.
        """


__all__ = ["TextClassificationModel"]
