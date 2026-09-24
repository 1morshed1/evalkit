"""OpenAI judge backend (optional; requires `pip install evalkit-py[openai]`)."""

from __future__ import annotations

import os

from evalkit.judges.base import Judge
from evalkit.schemas import JudgeConfig


class OpenAIJudge(Judge):
    def __init__(self, config: JudgeConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAI backend requires the openai package: pip install 'evalkit-py[openai]'"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.provider = "openai"
        self.model = config.model or "gpt-4o-mini"
        self.temperature = config.temperature
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
