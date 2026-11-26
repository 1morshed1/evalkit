"""Anthropic judge backend (optional; requires the `anthropic` package)."""

from __future__ import annotations

import os

from evalkit.judges.base import Judge
from evalkit.schemas import JudgeConfig


class AnthropicJudge(Judge):
    def __init__(self, config: JudgeConfig):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Anthropic backend requires the anthropic package: pip install anthropic"
            ) from exc
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.provider = "anthropic"
        self.model = config.model or "claude-haiku-5-5"
        self.temperature = config.temperature
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            # No `temperature`: the Anthropic Messages API rejects sampling
            # parameters on current models. Reproducibility comes from the
            # on-disk judge cache instead (see judges/cache.py).
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
        return "".join(parts)
