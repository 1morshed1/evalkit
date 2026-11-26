"""Judge backends: request shape, with the provider SDK faked out."""

from __future__ import annotations

import sys
import types

import pytest

from evalkit.schemas import JudgeConfig


class _FakeMessages:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text="ok")
        return types.SimpleNamespace(content=[block])


@pytest.fixture
def fake_anthropic(monkeypatch):
    messages = _FakeMessages()
    module = types.ModuleType("anthropic")
    module.Anthropic = lambda api_key: types.SimpleNamespace(messages=messages)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return messages


def test_anthropic_judge_defaults_and_omits_sampling_params(fake_anthropic):
    from evalkit.judges.anthropic import AnthropicJudge

    judge = AnthropicJudge(JudgeConfig(provider="anthropic", model=""))
    assert judge.model == "claude-haiku-5-5"
    assert judge.complete("Is the sky blue?") == "ok"

    sent = fake_anthropic.calls[0]
    assert sent["model"] == "claude-haiku-5-5"
    # The Messages API rejects temperature/top_p/top_k on current models.
    for param in ("temperature", "top_p", "top_k"):
        assert param not in sent


def test_anthropic_judge_respects_configured_model(fake_anthropic):
    from evalkit.judges.anthropic import AnthropicJudge

    judge = AnthropicJudge(JudgeConfig(provider="anthropic", model="claude-sonnet-5-5"))
    judge.complete("hi")
    assert fake_anthropic.calls[0]["model"] == "claude-sonnet-5-5"


def test_anthropic_judge_requires_api_key(fake_anthropic, monkeypatch):
    from evalkit.judges.anthropic import AnthropicJudge

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicJudge(JudgeConfig(provider="anthropic", model=""))
