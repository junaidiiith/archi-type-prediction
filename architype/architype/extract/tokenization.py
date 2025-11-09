"""
Lightweight tokenization helpers used for text extraction. These functions are
lifted from the legacy ``tokenization`` package and kept verbatim until the
refactor wires them into the new pipeline.
"""

from __future__ import annotations

from re import finditer
from typing import List

from transformers import AutoTokenizer

from .special_tokens import (
    EDGE_END,
    EDGE_START,
    NODE_BEGIN,
    NODE_END,
    escape_keywords,
)


def get_special_tokens() -> dict:
    return {"additional_special_tokens": [EDGE_START, EDGE_END, NODE_BEGIN, NODE_END]}


def get_tokenizer(model_name: str, use_special_tokens: bool = False, max_length: int = 512):
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_special_tokens:
        tokenizer.add_special_tokens(get_special_tokens())

    tokenizer.model_max_length = max_length
    return tokenizer


def camel_case_split(identifier: str) -> List[str]:
    if any(ek in identifier for ek in escape_keywords):
        return [identifier]
    matches = finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier
    )
    return [m.group(0) for m in matches]


def doc_tokenizer(doc: str, lower: bool = False) -> str:
    words = doc.split()
    snake_words: List[str] = []
    for w1 in words:
        if any(ek in w1 for ek in escape_keywords):
            snake_words.append(w1)
        else:
            snake_words.extend([w2 for w2 in w1.split("_") if w2 != ""])

    final_words: List[str] = []
    for word in snake_words:
        if any(ek in word for ek in escape_keywords):
            final_words.append(word)
        else:
            final_words.extend(camel_case_split(word))

    if lower:
        final_words = [w.lower() for w in final_words]

    return " ".join(final_words)


__all__ = ["get_special_tokens", "get_tokenizer", "camel_case_split", "doc_tokenizer"]

