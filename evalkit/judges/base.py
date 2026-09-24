"""Judge abstract base class.

A Judge turns a prompt into a text completion. One method, so backends are
trivially swappable and easy to fake in tests.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class Judge(ABC):
    #: identity used for cache keys; concrete backends set provider/model.
    provider: str = "base"
    model: str = "unknown"
    temperature: float = 0.0

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's text completion for the prompt."""

    def cache_id(self) -> str:
        return f"{self.provider}:{self.model}:{self.temperature}"

    # -- parsing helpers shared by evaluators -----------------------------

    @staticmethod
    def parse_json_array(text: str) -> list:
        """Best-effort extraction of a JSON array from a model response."""
        text = text.strip()
        # Strip markdown code fences if present.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "claims" in data:
                return list(data["claims"])
        except json.JSONDecodeError:
            pass
        # Fallback: grab the first [...] block.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    @staticmethod
    def parse_yes_no(text: str) -> bool:
        return text.strip().lower().startswith("y")
