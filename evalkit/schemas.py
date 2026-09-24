"""Data contracts for EvalKit.

Every subsystem speaks in these dataclasses. They are defined before the code
that uses them (see docs/01-architecture.md), because the schema is the stable
contract and the code is replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Input: test suite
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """One evaluation case loaded from the YAML suite."""

    __test__ = False  # not a pytest test class

    id: str
    question: str
    expected_answer: str | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Optional inline agent trace (Phase 3); dict form, parsed lazily.
    agent_trace: dict[str, Any] | None = None
    expected_result: dict[str, Any] | None = None

    def has(self, field_name: str) -> bool:
        """Whether ground-truth field is present and non-empty (drives skip logic)."""
        value = getattr(self, field_name, None)
        return bool(value)


@dataclass
class TargetConfig:
    type: str  # "http" | "python"
    endpoint: str | None = None
    import_path: str | None = None  # "module:function"
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    provider: str = "ollama"  # ollama | openai | anthropic
    model: str = "llama3.2"
    base_url: str | None = None
    temperature: float = 0.0
    concurrency: int = 4


@dataclass
class Suite:
    suite_name: str
    target: TargetConfig
    judge: JudgeConfig
    test_cases: list[TestCase]
    thresholds: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Target output
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    """Normalized output of any Target adapter, regardless of transport."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    cost_usd: float | None = None


# ---------------------------------------------------------------------------
# Metric results
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Result of one evaluator on one case."""

    name: str
    score: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"score": self.score}
        out.update(self.detail)
        return out


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    prediction: Prediction
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tags": self.tags,
            "prediction": {
                "answer": self.prediction.answer,
                "contexts": self.prediction.contexts,
                "doc_ids": self.prediction.doc_ids,
                "latency_ms": self.prediction.latency_ms,
                "cost_usd": self.prediction.cost_usd,
            },
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "skipped": self.skipped,
        }


@dataclass
class Gate:
    passed: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Canonical result object; serialized to the result JSON (Contract 3)."""

    evalkit_version: str
    suite_name: str
    run_id: str
    judge: dict[str, str]
    summary: dict[str, float]
    thresholds: dict[str, float]
    gate: Gate
    cases: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evalkit_version": self.evalkit_version,
            "suite_name": self.suite_name,
            "run_id": self.run_id,
            "judge": self.judge,
            "summary": self.summary,
            "thresholds": self.thresholds,
            "gate": {"passed": self.gate.passed, "violations": self.gate.violations},
            "cases": [c.to_dict() for c in self.cases],
        }


# ---------------------------------------------------------------------------
# Agent trace (Phase 3, defined now)
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None


@dataclass
class Step:
    type: str  # "llm" | "tool_call" | "tool_error" | "retry_after_error"
    cost_usd: float = 0.0
    tool_call: ToolCall | None = None


@dataclass
class AgentTrace:
    steps: list[Step]
    final_output: str = ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]
