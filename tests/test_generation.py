"""Faithfulness / relevance tests using the FakeJudge (no LLM)."""

from __future__ import annotations

from evalkit.evaluators.generation import GenerationEvaluator
from evalkit.schemas import Prediction, TestCase
from tests.conftest import FakeJudge


def _case():
    return TestCase(id="c1", question="What is the refund window?")


def test_faithfulness_all_supported():
    judge = FakeJudge(claims=["Refunds within 30 days", "Receipt required"])
    pred = Prediction(answer="...", contexts=["Refunds within 30 days with receipt."])
    out = GenerationEvaluator().evaluate(_case(), pred, judge)
    assert out["faithfulness"].score == 1.0
    assert out["hallucination_rate"].score == 0.0
    assert out["faithfulness"].detail["unsupported_claims"] == []


def test_faithfulness_with_hallucination():
    judge = FakeJudge(
        claims=["Refunds within 30 days", "Free lifetime warranty"],
        unsupported=["Free lifetime warranty"],
    )
    pred = Prediction(answer="...", contexts=["Refunds within 30 days."])
    out = GenerationEvaluator().evaluate(_case(), pred, judge)
    assert out["faithfulness"].score == 0.5
    assert out["hallucination_rate"].score == 0.5
    assert out["faithfulness"].detail["unsupported_claims"] == ["Free lifetime warranty"]


def test_relevance_labels():
    for label, expected in [("FULLY", 1.0), ("PARTIAL", 0.5), ("NONE", 0.0)]:
        judge = FakeJudge(claims=[], relevance=label)
        pred = Prediction(answer="some answer", contexts=[])
        out = GenerationEvaluator().evaluate(_case(), pred, judge)
        assert out["relevance"].score == expected
    # No contexts -> faithfulness not emitted.
    assert "faithfulness" not in out


def test_no_claims_is_vacuously_faithful():
    judge = FakeJudge(claims=[])
    pred = Prediction(answer="hello", contexts=["irrelevant"])
    out = GenerationEvaluator().evaluate(_case(), pred, judge)
    assert out["faithfulness"].score == 1.0
