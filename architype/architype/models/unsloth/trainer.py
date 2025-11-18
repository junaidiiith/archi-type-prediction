"""
Unsloth-based fine-tuning for text classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer


from ..base import TextClassificationModel
from .prompts import build_eval_prompt, build_train_prompt


@dataclass
class UnslothTrainingConfig:
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    warmup_steps: int = 5
    logging_steps: int = 10


class UnslothTextClassifier(TextClassificationModel):
    """
    Text classification via QLoRA fine-tuning using Unsloth.
    """

    def __init__(
        self,
        model_name: str,
        *,
        output_dir: str = "runs/artifacts/unsloth",
        seed: Optional[int] = 3407,
    ) -> None:
        super().__init__(model_name, output_dir=output_dir, seed=seed)
        self.model = None
        self.tokenizer = None
        self.class_labels: List[str] = []

    def _format_prompts(
        self,
        dataset: Dataset,
        class_labels: List[str],
        include_label: bool = True,
    ) -> Dataset:
        class_lines = "\n".join(f"- {label}" for label in class_labels)

        def _format(example: Dict[str, Any]) -> Dict[str, str]:
            prompt = (
                build_train_prompt(
                    class_lines=class_lines,
                    text=example["text"],
                    label=example["label"],
                )
                if include_label
                else build_eval_prompt(
                    class_lines=class_lines,
                    text=example["text"],
                )
            )
            return {"text": prompt, "label": example["label"]}

        return dataset.map(_format, remove_columns=dataset.column_names)

    def train(
        self,
        dataset: DatasetDict,
        *,
        evaluation_dataset: Optional[DatasetDict] = None,
        config: Optional[UnslothTrainingConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if config is None:
            config = UnslothTrainingConfig()
        if evaluation_dataset is None:
            evaluation_dataset = dataset

        self.class_labels = sorted(set(dataset["train"]["label"]))
        formatted_train = self._format_prompts(
            dataset["train"], self.class_labels, include_label=True
        )
        eval_split = evaluation_dataset.get(
            "validation", evaluation_dataset.get("test", dataset["train"])
        )
        formatted_eval = self._format_prompts(
            eval_split, self.class_labels, include_label=True
        )

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_train,
            eval_dataset=formatted_eval,
            args=SFTConfig(
                dataset_text_field="text",
                output_dir=str(self.output_dir),
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                num_train_epochs=config.num_train_epochs,
                logging_steps=config.logging_steps,
                learning_rate=config.learning_rate,
                report_to="none",
                seed=self.seed,
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        train_result = trainer.train()
        trainer.save_model(str(self.output_dir / "lora"))
        self.tokenizer.save_pretrained(str(self.output_dir / "lora"))

        return train_result.metrics

    def _predict_single(self, text: str) -> Tuple[str, float]:
        assert self.model is not None and self.tokenizer is not None
        class_lines = "\n".join(f"- {label}" for label in self.class_labels)
        prompt = build_eval_prompt(class_lines=class_lines, text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
            )
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        for label in self.class_labels:
            if label.lower() in decoded.lower():
                return label, 1.0
        return decoded.strip().splitlines()[-1], 0.0

    def evaluate(
        self,
        dataset: DatasetDict,
        *,
        split: str = "test",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=kwargs.get("max_seq_length", 2048),
                load_in_4bit=True,
            )
        if not self.class_labels:
            self.class_labels = sorted(set(dataset["train"]["label"]))

        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not present in dataset. Available: {list(dataset.keys())}"
            )

        texts = dataset[split]["text"]
        labels = dataset[split]["label"]

        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        predictions: List[str] = []
        confidences: List[float] = []
        for text in tqdm(texts, desc="Unsloth inference"):
            pred, conf = self._predict_single(text)
            predictions.append(pred)
            confidences.append(conf)

        metrics: Dict[str, float] = {}
        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["macro_f1"] = f1_score(labels, predictions, average="macro")
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["macro_precision"] = precision
        metrics["macro_recall"] = recall

        prediction_records = []
        for text, true_label, pred_label, conf in zip(
            texts, labels, predictions, confidences
        ):
            prediction_records.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": conf,
                }
            )

        return {"metrics": metrics, "predictions": prediction_records}


__all__ = ["UnslothTextClassifier", "UnslothTrainingConfig"]
