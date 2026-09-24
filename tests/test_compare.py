"""Comparison + regression-detection tests, including ceiling-metric direction."""

from __future__ import annotations

from evalkit.report.compare import compare_results, regressions


def _result(summary):
    return {"summary": summary}


def test_improvement_and_regression_directions():
    baseline = _result({"faithfulness": 0.80, "hallucination_rate": 0.12})
    candidate = _result({"faithfulness": 0.90, "hallucination_rate": 0.04})
    rows = {r["metric"]: r for r in compare_results(baseline, candidate)}
    # Higher faithfulness = better.
    assert rows["faithfulness"]["better"] is True
    # Lower hallucination = better even though value went down.
    assert rows["hallucination_rate"]["better"] is True
    assert regressions(list(rows.values())) == []


def test_detects_regression():
    baseline = _result({"relevance": 0.90})
    candidate = _result({"relevance": 0.70})
    rows = compare_results(baseline, candidate)
    assert rows[0]["better"] is False
    assert len(regressions(rows)) == 1


def test_ceiling_regression():
    baseline = _result({"hallucination_rate": 0.05})
    candidate = _result({"hallucination_rate": 0.20})  # worse (higher)
    rows = compare_results(baseline, candidate)
    assert rows[0]["better"] is False
    assert len(regressions(rows)) == 1


def test_flat_is_not_regression():
    rows = compare_results(_result({"relevance": 0.9}), _result({"relevance": 0.9}))
    assert rows[0]["direction"] == "flat"
    assert regressions(rows) == []
