"""
Fine-tuning and evaluation utilities for BERT-style classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict  # type: ignore[import]
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..base import TextClassificationModel


@dataclass
class BertTrainingConfig:
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_length: Optional[int] = None
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    logging_steps: int = 100
    eval_steps: int = 100
    gradient_accumulation_steps: int = 1
    early_stopping_patience: Optional[int] = 50


def _prepare_label_encoding(
    dataset: DatasetDict,
) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    unique_labels = sorted(
        {sample for split in dataset for sample in set(
            dataset[split]["label"])}
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


def _make_compute_metrics(id2label: Optional[Dict[int, Any]] = None):
    """
    Factory function to create a compute_metrics function with access to id2label.

    Args:
        id2label: Optional mapping from class ID to label name for per-class metrics.

    Returns:
        A compute_metrics function compatible with Transformers Trainer.
    """
    def _compute_metrics(eval_pred) -> Dict[str, Any]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Convert logits to probabilities for ROC AUC
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        # Determine number of classes
        num_classes = len(np.unique(labels))

        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

        # ROC AUC requires probabilities and multi_class parameter for multi-class
        try:
            if num_classes == 2:
                # Binary classification
                metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
            elif num_classes > 2:
                # Multi-class classification - requires multi_class parameter
                metrics["roc_auc"] = roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro"
                )
            else:
                # Only one class present - cannot compute ROC AUC
                metrics["roc_auc"] = 0.0
        except (ValueError, TypeError) as e:
            # Skip ROC AUC if there are issues (e.g., only one class present, invalid input)
            metrics["roc_auc"] = 0.0

        # Macro-averaged metrics
        precision, recall, _, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        metrics["macro_precision"] = precision
        metrics["macro_recall"] = recall

        # Per-class metrics - flattened for TensorBoard compatibility
        if id2label is not None:
            per_class_precision, per_class_recall, per_class_f1, per_class_support = (
                precision_recall_fscore_support(
                    labels,
                    preds,
                    average=None,
                    zero_division=0,
                )
            )

            unique_labels = sorted(set(labels) | set(preds))

            # Flatten per-class metrics into scalar values for TensorBoard
            # Format: per_class/{label_name}/{metric_name}
            for idx, class_id in enumerate(unique_labels):
                label_name = id2label.get(class_id, f"class_{class_id}")
                # Sanitize label name for TensorBoard (replace spaces/special chars with underscores)
                safe_label_name = str(label_name).replace(
                    " ", "_").replace("/", "_")

                metrics[f"per_class/{safe_label_name}/precision"] = float(
                    per_class_precision[idx])
                metrics[f"per_class/{safe_label_name}/recall"] = float(
                    per_class_recall[idx])
                metrics[f"per_class/{safe_label_name}/f1"] = float(
                    per_class_f1[idx])
                metrics[f"per_class/{safe_label_name}/support"] = int(
                    per_class_support[idx])

        return metrics

    return _compute_metrics


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

        # Determine metric for best model selection
        # When early stopping is enabled, use eval_loss (monitored by EarlyStoppingCallback)
        # Otherwise, use macro_f1 for best model selection
        metric_for_best = "eval_loss" if config.early_stopping_patience is not None else "macro_f1"

        # Determine evaluation strategy
        # If eval_steps is set, use "steps" strategy; otherwise use the configured strategy
        eval_strategy = config.eval_strategy
        eval_steps_value = None
        if hasattr(config, "eval_steps") and config.eval_steps is not None:
            eval_strategy = "steps"
            eval_steps_value = config.eval_steps

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps_value,
            save_strategy=config.save_strategy,
            logging_steps=config.logging_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            seed=self.seed,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model=metric_for_best,
            greater_is_better=(metric_for_best == "macro_f1"),
            logging_dir=str(self.output_dir / "logs"),
            remove_unused_columns=True,
        )

        callbacks = []
        if config.early_stopping_patience is not None:
            # Early stopping monitors eval_loss
            # It will stop if eval_loss doesn't improve (decrease) for patience evaluations
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_threshold=0.0,
                )
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            compute_metrics=_make_compute_metrics(self.id2label),
            callbacks=callbacks if callbacks else None,
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

        tokenized_ds = _tokenize_dataset(
            dataset[split], self.tokenizer, self.label2id)

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, pad_to_multiple_of=8
        )

        args = TrainingArguments(
            output_dir=str(self.output_dir / "eval"),
            per_device_eval_batch_size=kwargs.get(
                "per_device_eval_batch_size", 32),
            seed=self.seed,
            do_train=False,
            report_to=[],
        )

        compute_metrics_fn = _make_compute_metrics(self.id2label)
        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )

        predictions = trainer.predict(tokenized_ds)
        logits = predictions.predictions
        label_ids = predictions.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_ids = probs.argmax(axis=-1)

        metrics = predictions.metrics
        metrics.update(compute_metrics_fn((logits, label_ids)))

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
