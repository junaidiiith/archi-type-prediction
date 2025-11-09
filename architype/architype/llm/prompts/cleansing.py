"""
LLM prompt definitions for model cleansing tasks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import Prompt


class ModelMeaningfulnessVerdict(BaseModel):
    """
    Structured response describing whether a model is meaningful.
    """

    is_meaningful: bool = Field(
        ...,
        description="True when the provided edges describe a meaningful enterprise model.",
    )
    rationale: str = Field(
        ...,
        description="Short explanation referencing the supplied edges.",
    )
    confidence: str = Field(
        "medium",
        description="Confidence in the judgement (e.g., low, medium, high).",
    )


CLEANSING_TEMPLATE = """You are validating whether an enterprise architecture model
describes meaningful business structure or is a dummy placeholder.

You are given up to {edge_count} edges from the model. Each edge is provided in the
format `Source -> Target`. Review the edges holistically. Answer these points:
1. Determine if the model appears meaningful (business-like names, consistent relations).
2. Flag edges that clearly show placeholder content (e.g., `Class1 -> Class2`, `Foo -> Bar`).
3. Provide a concise rationale that references problematic or meaningful edges.

Respond strictly in JSON using the provided schema."""


class ModelMeaningfulnessPrompt(Prompt):
    def __init__(self) -> None:
        super().__init__(
            name="model_meaningfulness",
            template=CLEANSING_TEMPLATE + "\n\nEdges:\n{edges}\n",
            response_model=ModelMeaningfulnessVerdict,
            system=(
                "You are an enterprise modeling expert who identifies placeholder "
                "or meaningless diagrams. Be critical and prefer flagging uncertain "
                "models when evidence is weak."
            ),
        )


__all__ = ["ModelMeaningfulnessPrompt", "ModelMeaningfulnessVerdict"]
