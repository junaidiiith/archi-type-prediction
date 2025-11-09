"""
Prompt abstractions for LLM-powered cleansing and evaluation tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


@dataclass
class Prompt:
    """
    Base prompt definition.

    Parameters
    ----------
    name:
        Identifier used when caching prompt/response pairs.
    template:
        String template rendered with ``format`` using the keyword arguments
        passed to :meth:`build_messages`.
    response_model:
        Optional Pydantic model describing the expected JSON response.
    system:
        Optional system instruction injected ahead of the user message.
    """

    name: str
    template: str
    response_model: Optional[Type[BaseModel]] = None
    system: Optional[str] = None

    def render(self, **kwargs: Any) -> str:
        """Render the template using the supplied keyword arguments."""

        return self.template.format(**kwargs)

    def build_messages(self, **kwargs: Any) -> List[Dict[str, str]]:
        """
        Convert the prompt into an OpenAI-style chat message list.
        """

        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.render(**kwargs)})
        return messages


__all__ = ["Prompt"]
