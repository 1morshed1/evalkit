"""Data contracts for EvalKit.

Every subsystem speaks in these dataclasses. They are defined before the code
that uses them (see docs/01-architecture.md), because the schema is the stable
contract and the code is replaceable.
"""

from __future__ import annotations

import json
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


#: Step types that carry a ToolCall (i.e. represent an actual tool invocation).
TOOL_STEP_TYPES = {"tool_call", "tool_error", "retry_after_error"}


@dataclass
class AgentTrace:
    steps: list[Step]
    final_output: str = ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool the agent actually invoked, in order (errors included)."""
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def llm_steps(self) -> list[Step]:
        return [s for s in self.steps if s.type == "llm"]

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    # -- adapters -------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTrace:
        """Parse a trace from a plain dict.

        Two shapes are accepted:
        * **native** — ``{"steps": [...], "final_output": "..."}`` where each step
          is ``{"type": ..., "cost_usd": ..., "tool_call": {...}}``.
        * **openai** — ``{"messages": [...]}`` (or ``{"format": "openai", ...}``),
          a list of OpenAI chat-completion messages; delegated to ``from_openai``.
        """
        if "messages" in data or data.get("format") == "openai":
            return cls.from_openai(list(data.get("messages") or []))
        steps = [_step_from_dict(s) for s in (data.get("steps") or [])]
        return cls(steps=steps, final_output=str(data.get("final_output", "")))

    @classmethod
    def from_openai(cls, messages: list[dict[str, Any]]) -> AgentTrace:
        """Build a trace from OpenAI-style chat messages (Contract 4).

        Each assistant message becomes an ``llm`` step; every ``tool_calls`` entry
        on it becomes a tool step whose outcome (success / error / retry) is
        resolved from the matching ``role: tool`` message. The last assistant
        message with text content is the ``final_output``.
        """
        steps: list[Step] = []
        final_output = ""
        by_call_id: dict[str, tuple[Step, ToolCall]] = {}
        errored_tools: set[str] = set()

        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                steps.append(Step(type="llm", cost_usd=_message_cost(msg)))
                for raw_tc in msg.get("tool_calls") or []:
                    fn = raw_tc.get("function", raw_tc) or {}
                    name = str(fn.get("name", ""))
                    call = ToolCall(name=name, arguments=_parse_arguments(fn.get("arguments")))
                    step_type = "retry_after_error" if name in errored_tools else "tool_call"
                    step = Step(type=step_type, tool_call=call)
                    steps.append(step)
                    call_id = raw_tc.get("id")
                    if call_id is not None:
                        by_call_id[str(call_id)] = (step, call)
                content = msg.get("content")
                if content:
                    final_output = _content_text(content)
            elif role == "tool":
                call_id = msg.get("tool_call_id")
                content = _content_text(msg.get("content"))
                pair = by_call_id.get(str(call_id)) if call_id is not None else None
                is_error = bool(msg.get("is_error")) or _looks_like_error(content)
                if pair is not None:
                    step, call = pair
                    if is_error:
                        call.error = content
                        errored_tools.add(call.name)
                        # A retry that then errors again stays a retry; a first
                        # failure becomes a tool_error the recovery metric counts.
                        if step.type == "tool_call":
                            step.type = "tool_error"
                    else:
                        call.result = content
        return cls(steps=steps, final_output=final_output)


def _step_from_dict(data: dict[str, Any]) -> Step:
    tc = data.get("tool_call")
    tool_call = None
    if isinstance(tc, dict):
        tool_call = ToolCall(
            name=str(tc.get("name", "")),
            arguments=dict(tc.get("arguments") or {}),
            result=tc.get("result"),
            error=tc.get("error"),
        )
    return Step(
        type=str(data.get("type", "llm")),
        cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
        tool_call=tool_call,
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """OpenAI encodes tool arguments as a JSON string; accept dicts too."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_": parsed}
        except json.JSONDecodeError:
            return {}
    return {}


def _content_text(content: Any) -> str:
    """Flatten OpenAI content, which may be a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts)
    return str(content)


def _message_cost(msg: dict[str, Any]) -> float:
    """Read a per-message cost if the trace carries one (else 0)."""
    for key in ("cost_usd", "cost"):
        if key in msg and msg[key] is not None:
            try:
                return float(msg[key])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _looks_like_error(content: str) -> bool:
    head = content.strip().lower()[:40]
    return head.startswith(("error", "exception", "traceback", "failed"))
