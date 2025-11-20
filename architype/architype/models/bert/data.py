"""
Torch dataset wrappers for BERT-style fine-tuning. Ported from the legacy
``data_loading.encoding`` module.
"""

from __future__ import annotations

from typing import List, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_max_length(tokenizer: AutoTokenizer) -> int:
    tokenizer_name = tokenizer.name_or_path.lower()
    if "modernbert" in tokenizer_name:
        return 8000
    return 512


class EncodingDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        labels: List[Union[str, int]] | None = None,
        max_length: int = 512,
        remove_duplicates: bool = False,
    ):
        max_length = get_max_length(tokenizer) if max_length is None else max_length

        if remove_duplicates:
            # print(f"Dataset with {len(texts)} samples before removing duplicates")
            texts_to_id = {text: i for i, text in enumerate(texts)}
            texts = list(texts_to_id.keys())
            labels = [labels[i] for i in texts_to_id.values()] if labels else None

        self.inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        if labels is not None:
            self.inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs["input_ids"])

    def __getitem__(self, index: int):
        return {key: value[index] for key, value in self.inputs.items()}


class GPTTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def oversample_dataset(dataset: Dataset, oversampling_ratio: float = 0.7):
    """
    Oversample minority classes to balance the dataset.
    """

    class_occurrences = dataset[:]["labels"].numpy()
    unique_classes, counts = np.unique(class_occurrences, return_counts=True)
    max_count = counts.max()
    indices_with_oversamples: list[int] = []
    for class_idx, count in zip(unique_classes, counts):
        class_indices = np.where(class_occurrences == class_idx)[0]
        indices_with_oversamples.extend(class_indices)
        oversample_count = int(oversampling_ratio * (max_count - count))
        indices_with_oversamples.extend(np.random.choice(class_indices, oversample_count))

    return indices_with_oversamples


__all__ = ["EncodingDataset", "GPTTextDataset", "get_max_length", "oversample_dataset"]

