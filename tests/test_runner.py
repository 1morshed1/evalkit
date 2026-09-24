"""Runner smoke + skip-logic + gate tests, using the demo target + FakeJudge."""

from __future__ import annotations

from evalkit.config import parse_suite
from evalkit.evaluators import GenerationEvaluator, RetrievalEvaluator
from evalkit.runner import run_suite
from tests.conftest import FakeJudge

SUITE = {
    "suite_name": "test",
    "target": {"type": "python", "import": "demo.rag:answer"},
    "judge": {"provider": "ollama", "model": "llama3.2", "concurrency": 2},
    "test_cases": [
        {
            "id": "refund",
            "question": "What is your refund policy?",
            "relevant_doc_ids": ["doc-policy-returns-v3"],
            "tags": ["policy"],
        },
        {
            # No relevant_doc_ids -> retrieval must be skipped for this case.
            "id": "no-groundtruth",
            "question": "How long does standard shipping take?",
            "tags": ["shipping"],
        },
    ],
    "thresholds": {"faithfulness": 0.5, "relevance": 0.5},
}


def _fake_judge():
    return FakeJudge(claims=["Refunds within 30 days"], relevance="FULLY")


def test_runner_end_to_end():
    suite = parse_suite(SUITE)
    result = run_suite(
        suite,
        evaluators=[RetrievalEvaluator(), GenerationEvaluator()],
        judge=_fake_judge(),
    )
    assert result.suite_name == "test"
    assert len(result.cases) == 2
    # First case has ground truth -> retrieval metrics present.
    c0 = next(c for c in result.cases if c.id == "refund")
    assert "retrieval_recall_at_5" in c0.metrics
    assert "relevance" in c0.metrics
    # Second case lacks ground truth -> retrieval skipped, generation still runs.
    c1 = next(c for c in result.cases if c.id == "no-groundtruth")
    assert "retrieval" in c1.skipped
    assert "relevance" in c1.metrics
    assert "retrieval_recall_at_5" not in c1.metrics


def test_gate_pass():
    suite = parse_suite(SUITE)
    result = run_suite(suite, judge=_fake_judge())
    assert result.gate.passed is True


def test_gate_fail_on_high_threshold():
    spec = dict(SUITE)
    spec["thresholds"] = {"relevance": 0.99}
    suite = parse_suite(spec)
    # Relevance PARTIAL = 0.5 < 0.99 -> violation.
    result = run_suite(suite, judge=FakeJudge(claims=["x"], relevance="PARTIAL"))
    assert result.gate.passed is False
    assert any("relevance" in v for v in result.gate.violations)
