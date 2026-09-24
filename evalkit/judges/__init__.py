"""LLM-as-judge backends."""

from __future__ import annotations

from evalkit.judges.base import Judge
from evalkit.judges.cache import CachedJudge
from evalkit.schemas import JudgeConfig


def build_judge(config: JudgeConfig, *, cache: bool = True) -> Judge:
    provider = config.provider.lower()
    if provider == "ollama":
        from evalkit.judges.ollama import OllamaJudge

        judge: Judge = OllamaJudge(config)
    elif provider == "openai":
        from evalkit.judges.openai import OpenAIJudge

        judge = OpenAIJudge(config)
    elif provider == "anthropic":
        from evalkit.judges.anthropic import AnthropicJudge

        judge = AnthropicJudge(config)
    else:
        raise ValueError(f"Unknown judge provider: {config.provider}")

    if cache:
        return CachedJudge(judge)
    return judge


__all__ = ["Judge", "CachedJudge", "build_judge"]
