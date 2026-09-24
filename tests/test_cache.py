"""Judge cache: second identical call is a hit and skips the inner judge."""

from __future__ import annotations

from evalkit.judges.base import Judge
from evalkit.judges.cache import CachedJudge


class CountingJudge(Judge):
    def __init__(self):
        self.provider = "count"
        self.model = "m"
        self.temperature = 0.0
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return f"resp-{self.calls}"


def test_cache_hit(tmp_path):
    inner = CountingJudge()
    judge = CachedJudge(inner, cache_dir=tmp_path)
    first = judge.complete("hello")
    second = judge.complete("hello")
    assert first == second == "resp-1"
    assert inner.calls == 1  # inner called only once
    assert judge.hits == 1 and judge.misses == 1


def test_cache_distinguishes_prompts(tmp_path):
    inner = CountingJudge()
    judge = CachedJudge(inner, cache_dir=tmp_path)
    judge.complete("a")
    judge.complete("b")
    assert inner.calls == 2
