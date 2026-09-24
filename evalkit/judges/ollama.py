"""Ollama judge backend (default). Local, free, no API key."""

from __future__ import annotations

import httpx

from evalkit.judges.base import Judge
from evalkit.schemas import JudgeConfig


class OllamaJudge(Judge):
    def __init__(self, config: JudgeConfig, timeout: float = 120.0):
        self.provider = "ollama"
        self.model = config.model or "llama3.2"
        self.temperature = config.temperature
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        self._timeout = timeout

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
