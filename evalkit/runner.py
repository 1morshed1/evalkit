"""Runner: wire Target -> Evaluators -> aggregated RunResult.

Per-case work runs in a thread pool (judge calls are IO-bound), so cases are
evaluated concurrently up to `judge.concurrency`. The runner never hardcodes a
concrete target/judge/evaluator type — it depends only on the interfaces.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from evalkit import __version__
from evalkit.evaluators import (
    AgentEvaluator,
    Evaluator,
    GenerationEvaluator,
    RetrievalEvaluator,
)
from evalkit.evaluators.base import Evaluator as EvaluatorBase
from evalkit.judges import build_judge
from evalkit.judges.base import Judge
from evalkit.schemas import (
    CaseResult,
    Gate,
    MetricResult,
    RunResult,
    Suite,
    TestCase,
)
from evalkit.targets import build_target
from evalkit.targets.base import Target

# Metrics where lower is better (ceilings), not higher.
CEILING_METRICS = {"hallucination_rate"}


def default_evaluators(suite: Suite | None = None) -> list[Evaluator]:
    """Pick the evaluator set for a suite based on its target type.

    Agent suites (``target.type: agent``) score recorded traces, so they use the
    AgentEvaluator; everything else runs the RAG retrieval + generation pair.
    """
    if suite is not None and suite.target.type in ("agent", "trace"):
        return [AgentEvaluator()]
    return [RetrievalEvaluator(), GenerationEvaluator()]


def run_suite(
    suite: Suite,
    *,
    evaluators: list[Evaluator] | None = None,
    use_cache: bool = True,
    judge: Judge | None = None,
    target: Target | None = None,
) -> RunResult:
    evaluators = evaluators if evaluators is not None else default_evaluators(suite)
    target = target or build_target(suite.target)
    needs_judge = any(e.needs_judge for e in evaluators)
    if judge is None and needs_judge:
        judge = build_judge(suite.judge, cache=use_cache)

    concurrency = max(1, suite.judge.concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        case_results = list(
            pool.map(lambda c: _run_case(c, target, evaluators, judge), suite.test_cases)
        )

    summary = _aggregate(case_results)
    gate = _check_thresholds(summary, suite.thresholds)
    return RunResult(
        evalkit_version=__version__,
        suite_name=suite.suite_name,
        run_id=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        judge={"provider": suite.judge.provider, "model": suite.judge.model},
        summary=summary,
        thresholds=suite.thresholds,
        gate=gate,
        cases=case_results,
    )


def _run_case(
    case: TestCase,
    target: Target,
    evaluators: list[Evaluator],
    judge: Judge | None,
) -> CaseResult:
    pred = target.predict(case.question)
    metrics: dict[str, MetricResult] = {}
    skipped: list[str] = []
    for ev in evaluators:
        assert isinstance(ev, EvaluatorBase)
        if not ev.applies_to(case):
            skipped.append(ev.name)
            continue
        metrics.update(ev.evaluate(case, pred, judge))
    return CaseResult(id=case.id, tags=case.tags, prediction=pred, metrics=metrics, skipped=skipped)


def _aggregate(cases: list[CaseResult]) -> dict[str, float]:
    """Mean of each metric across the cases that reported it."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for c in cases:
        for name, result in c.metrics.items():
            totals[name] = totals.get(name, 0.0) + result.score
            counts[name] = counts.get(name, 0) + 1
    summary = {name: round(totals[name] / counts[name], 4) for name in totals}

    latencies = [c.prediction.latency_ms for c in cases if c.prediction.latency_ms is not None]
    if latencies:
        summary["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
    costs = [c.prediction.cost_usd for c in cases if c.prediction.cost_usd is not None]
    if costs:
        summary["avg_cost_usd"] = round(sum(costs) / len(costs), 6)
    return summary


def _check_thresholds(summary: dict[str, float], thresholds: dict[str, float]) -> Gate:
    violations: list[str] = []
    for metric, limit in thresholds.items():
        if metric not in summary:
            continue
        value = summary[metric]
        if is_ceiling(metric):
            if value > limit:
                violations.append(f"{metric}={value} > max {limit}")
        else:
            if value < limit:
                violations.append(f"{metric}={value} < min {limit}")
    return Gate(passed=not violations, violations=violations)


def is_ceiling(metric: str) -> bool:
    """True for metrics where lower is better (hallucination, latency, cost)."""
    return (
        metric in CEILING_METRICS
        or "hallucination" in metric
        or "cost" in metric
        or metric.startswith("avg_latency")
    )
