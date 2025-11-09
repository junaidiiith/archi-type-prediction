"""
LLM prompt definitions for text classification tasks.
"""

from __future__ import annotations
from typing import List

from pydantic import BaseModel, Field

from .base import Prompt


DEFAULT_SYSTEM_PROMPT = (
    "You are a domain expert that classifies enterprise architecture elements."
    " Respond with the class label verbatim."
)

DEFAULT_PROMPT_TEMPLATE = """Classes:
{class_lines}

Text:
{text}

Predict the label of each text in the batch."""


class APIClassificationResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the input text.")
    text: str = Field(..., description="Original input text.")
    rationale: str = Field(
        ...,
        description="Model rationale explaining the predicted class.",
    )
    prediction: str = Field(..., description="Predicted class label.")


class APIClassificationBatchResponse(BaseModel):
    items: List[APIClassificationResponse]


class ZeroShotTextClassificationPrompt(Prompt):
    def __init__(self) -> None:
        super().__init__(
            name="zero_shot_text_classification",
            template=DEFAULT_PROMPT_TEMPLATE,
            response_model=APIClassificationBatchResponse,
            system=DEFAULT_SYSTEM_PROMPT,
        )