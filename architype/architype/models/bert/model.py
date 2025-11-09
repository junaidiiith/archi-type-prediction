"""
Fine-tuning and evaluation utilities for BERT-style classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict  # type: ignore[import]
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..base import TextClassificationModel


def _prepare_label_encoding(
    dataset: DatasetDict,
) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    unique_labels = sorted(
        {sample for split in dataset for sample in set(dataset[split]["label"])}
    )
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    label2id: Dict[Any, int],
) -> Dataset:
    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        tokenized = tokenizer(batch["text"], truncation=True)
        tokenized["labels"] = [label2id[label] for label in batch["label"]]
        return tokenized

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
    precision, recall, _, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    metrics["macro_precision"] = precision
    metrics["macro_recall"] = recall
    return metrics


@dataclass
class BertTrainingConfig:
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_length: Optional[int] = None
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    gradient_accumulation_steps: int = 1


class BertTextClassifier(TextClassificationModel):
    """
    Wrapper around Hugging Face Transformers for BERT-style classification.
    """

    def __init__(
        self,
        model_name: str,
        *,
        output_dir: str = "runs/artifacts/bert",
        seed: Optional[int] = 3407,
    ) -> None:
        super().__init__(model_name, output_dir=output_dir, seed=seed)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.label2id: Dict[Any, int] = {}
        self.id2label: Dict[int, Any] = {}

    def _build_tokenizer(self, max_length: Optional[int]) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if max_length:
            tokenizer.model_max_length = max_length
        return tokenizer

    def train(
        self,
        dataset: DatasetDict,
        *,
        evaluation_dataset: Optional[DatasetDict] = None,
        config: Optional[BertTrainingConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if config is None:
            config = BertTrainingConfig()

        if evaluation_dataset is None:
            evaluation_dataset = dataset

        self.label2id, self.id2label = _prepare_label_encoding(dataset)
        self.tokenizer = self._build_tokenizer(config.max_length)

        tokenized_train = _tokenize_dataset(
            dataset["train"], self.tokenizer, self.label2id
        )
        tokenized_eval = _tokenize_dataset(
            evaluation_dataset.get(
                "validation", evaluation_dataset.get("test", dataset["train"])
            ),
            self.tokenizer,
            self.label2id,
        )

        num_labels = len(self.label2id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, pad_to_multiple_of=8
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
            logging_steps=config.logging_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            seed=self.seed,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics

    def evaluate(
        self,
        dataset: DatasetDict,
        *,
        split: str = "test",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.tokenizer is None or self.model is None:
            num_labels = len(self.label2id) if self.label2id else None
            self.tokenizer = self._build_tokenizer(kwargs.get("max_length"))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=self.id2label if self.id2label else None,
                label2id=self.label2id if self.label2id else None,
            )

        if not self.label2id:
            self.label2id, self.id2label = _prepare_label_encoding(dataset)

        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not present in dataset. Available: {list(dataset.keys())}"
            )

        tokenized_ds = _tokenize_dataset(dataset[split], self.tokenizer, self.label2id)

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, pad_to_multiple_of=8
        )

        args = TrainingArguments(
            output_dir=str(self.output_dir / "eval"),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 32),
            seed=self.seed,
            do_train=False,
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        predictions = trainer.predict(tokenized_ds)
        logits = predictions.predictions
        label_ids = predictions.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_ids = probs.argmax(axis=-1)

        metrics = predictions.metrics
        metrics.update(_compute_metrics((logits, label_ids)))

        prediction_records = []
        original_texts = dataset[split]["text"]
        for idx, (text, pred_idx, true_idx, prob_vec) in enumerate(
            zip(original_texts, pred_ids, label_ids, probs)
        ):
            prediction_records.append(
                {
                    "text": text,
                    "true_label": self.id2label.get(true_idx, true_idx),
                    "predicted_label": self.id2label.get(pred_idx, pred_idx),
                    "confidence": float(np.max(prob_vec)),
                }
            )

        return {"metrics": metrics, "predictions": prediction_records}


__all__ = ["BertTextClassifier", "BertTrainingConfig"]
