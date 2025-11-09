"""
Prompt builders used by the Unsloth text classifier.
"""

from __future__ import annotations

TRAIN_PROMPT = """<|im_start|>system
You are an enterprise architecture assistant that classifies elements using the provided class list.
Respond with the class label exactly as written.
<|im_end|>
<|im_start|>user
Classes:
{class_lines}

Classify the following text:
{text}
<|im_end|>
<|im_start|>assistant
{label}<|im_end|>
"""

EVAL_PROMPT = """<|im_start|>system
You are an enterprise architecture assistant that classifies elements using the provided class list.
Respond with only the class label.
<|im_end|>
<|im_start|>user
Classes:
{class_lines}

Classify the following text:
{text}
<|im_end|>
<|im_start|>assistant
"""


def build_train_prompt(*, class_lines: str, text: str, label: str) -> str:
    return TRAIN_PROMPT.format(class_lines=class_lines, text=text, label=label)


def build_eval_prompt(*, class_lines: str, text: str) -> str:
    return EVAL_PROMPT.format(class_lines=class_lines, text=text)


__all__ = ["build_train_prompt", "build_eval_prompt"]

