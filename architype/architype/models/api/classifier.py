"""
Hosted LLM API classification wrapper.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from architype.architype.llm.base import LLMService
from architype.architype.llm.prompts.base import Prompt
from architype.architype.llm.prompts.llm_text_classification import (
    APIClassificationResponse,
    ZeroShotTextClassificationPrompt,
)
from architype.architype.models.base import TextClassificationModel


def _chunk_list(items: List[Tuple[str, str]], size: int) -> List[List[Dict[str, Any]]]:
    size = max(size, 1)
    return [
        [{"id": k, "text": txt, "label": label} for k, (txt, label) in enumerate(items[i : i + size], start=1)]
        for i in range(0, len(items), size)
    ]


class APILLMTextClassifier(TextClassificationModel):
    """
    Evaluation helper that delegates classification to a hosted LLM via the `LLMService`.
    """

    def __init__(
        self,
        model_name: str,
        *,
        prompt: Prompt = ZeroShotTextClassificationPrompt(),
        output_dir: str = "runs/artifacts/api",
        seed: Optional[int] = 3407,
    ) -> None:
        super().__init__(model_name, output_dir=output_dir, seed=seed)
        self.prompt = prompt

    def train(
        self,
        dataset: DatasetDict,
        *,
        evaluation_dataset: Optional[DatasetDict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError("API-based LLM models do not support fine-tuning.")

    def _build_prompt(
        self,
        class_labels: List[str],
        batched_inputs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        class_lines = "\n".join(f"- {label}" for label in class_labels)
        texts_str = json.dumps(batched_inputs, indent=2)
        return self.prompt.build_messages(class_lines=class_lines, text=texts_str)

    def evaluate(
        self,
        dataset: DatasetDict,
        *,
        split: str = "test",
        class_labels: Optional[List[str]] = None,
        batch_size: int = 50,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not present in dataset. Available: {list(dataset.keys())}"
            )

        texts = dataset[split]["text"]
        labels = dataset[split]["label"]
        if class_labels is None:
            class_labels = sorted(
                set(dataset["train"]["label"] + dataset["test"]["label"])
            )

        predictions: List[str] = []
        rationales: List[str] = []
        batched_inputs = _chunk_list(list(zip(texts, labels)), batch_size)
        batched_responses = []
        message_batches = [
            self._build_prompt(class_labels, [{"id": bi['id'], "text": bi['text']} for bi in batched_input])
            for batched_input in batched_inputs
        ]

        responses = LLMService.get_llm_response_parallel(
            message_batches,
            response_format=self.prompt.response_model,
        )
        batched_responses: List[APIClassificationResponse] = sum(responses, [])
        predictions, expected = [], []
        for batch, response in zip(batched_inputs, batched_responses):
            input_map = {bi['id']: bi for bi in batch}
            prediction = response.prediction
            pred_id = response.id
            expected_label = input_map[pred_id]['label']
            predictions.append(prediction)
            expected.append(expected_label)
            rationales.append(response.rationale)
            
        metrics: Dict[str, float] = {}
        metrics["accuracy"] = accuracy_score(expected, predictions)
        metrics["macro_f1"] = f1_score(expected, predictions, average="macro")
        precision, recall, _, _ = precision_recall_fscore_support(
            expected, predictions, average="macro", zero_division=0
        )
        metrics["macro_precision"] = precision
        metrics["macro_recall"] = recall

        prediction_records = []
        for text, true_label, pred_label, rationale in zip(
            texts, expected, predictions, rationales
        ):
            prediction_records.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "rationale": rationale,
                }
            )

        return {"metrics": metrics, "predictions": prediction_records}


__all__ = ["APILLMTextClassifier"]
