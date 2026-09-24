"""Agent metrics: task completion, tool precision/recall, step efficiency,
error recovery, and cost.

Unlike the RAG evaluators, this one scores an *execution trace* rather than a
retrieved-and-generated answer. The trace lives inline on the case
(``case.agent_trace``) and is parsed into an :class:`AgentTrace`; the ground
truth lives on ``case.expected_result``. The judge is **optional** — every
metric here is deterministic unless ``expected_result.expected_output`` asks for
an LLM to rule on task completion.

Expected-result schema (all keys optional)::

    expected_result:
      success: true                       # explicit pass/fail, wins over everything
      success_contains: "1024"            # substring that must appear in final_output
      expected_output: "the disk is full" # LLM-judged completion (needs a judge)
      expected_tool_calls:                # names, or name + argument subset to match
        - {name: "search", arguments: {q: "refund policy"}}
        - "read_file"
      optimal_step_count: 3               # for step efficiency (LLM calls)
"""

from __future__ import annotations

from typing import Any

from evalkit.evaluators.base import Evaluator
from evalkit.judges.base import Judge
from evalkit.schemas import AgentTrace, MetricResult, Prediction, TestCase, ToolCall

COMPLETION_PROMPT = """You are grading whether an AGENT accomplished a TASK.

TASK: {task}
EXPECTED OUTCOME: {expected}
AGENT'S FINAL OUTPUT: {actual}

Did the agent accomplish the task and produce the expected outcome?
Reply with a single word: "yes" or "no"."""


class AgentEvaluator(Evaluator):
    name = "agent"
    requires = ["agent_trace"]
    needs_judge = False  # judge is used only if available and expected_output is set

    def evaluate(
        self, case: TestCase, pred: Prediction, judge: Judge | None
    ) -> dict[str, MetricResult]:
        trace = AgentTrace.from_dict(case.agent_trace or {})
        expected = case.expected_result or {}
        out: dict[str, MetricResult] = {}

        out["agent_task_completion"] = self._completion(case, trace, expected, judge)

        pr = self._tool_precision_recall(expected.get("expected_tool_calls"), trace.tool_calls)
        if pr is not None:
            precision, recall, detail = pr
            out["agent_tool_precision"] = MetricResult("agent_tool_precision", precision, detail)
            out["agent_tool_recall"] = MetricResult("agent_tool_recall", recall, detail)

        efficiency = self._step_efficiency(expected.get("optimal_step_count"), trace)
        if efficiency is not None:
            out["agent_step_efficiency"] = efficiency

        out["agent_error_recovery"] = self._error_recovery(trace)
        out["agent_cost_usd"] = MetricResult(
            "agent_cost_usd", round(trace.total_cost_usd, 6), {"steps": len(trace.steps)}
        )
        return out

    # -- task completion ------------------------------------------------------

    def _completion(
        self,
        case: TestCase,
        trace: AgentTrace,
        expected: dict[str, Any],
        judge: Judge | None,
    ) -> MetricResult:
        if "success" in expected:
            score = 1.0 if expected["success"] else 0.0
            return MetricResult("agent_task_completion", score, {"basis": "explicit"})

        contains = expected.get("success_contains")
        if contains is not None:
            needles = contains if isinstance(contains, list) else [contains]
            haystack = trace.final_output.lower()
            hit = all(str(n).lower() in haystack for n in needles)
            return MetricResult(
                "agent_task_completion", 1.0 if hit else 0.0, {"basis": "success_contains"}
            )

        expected_output = expected.get("expected_output")
        if expected_output and judge is not None:
            raw = judge.complete(
                COMPLETION_PROMPT.format(
                    task=case.question, expected=expected_output, actual=trace.final_output
                )
            )
            score = 1.0 if judge.parse_yes_no(raw) else 0.0
            return MetricResult("agent_task_completion", score, {"basis": "judge"})

        # No completion criterion given: fall back to "produced a final answer".
        score = 1.0 if trace.final_output.strip() else 0.0
        return MetricResult("agent_task_completion", score, {"basis": "final_output_present"})

    # -- tool precision / recall ---------------------------------------------

    def _tool_precision_recall(
        self, expected_raw: Any, actual: list[ToolCall]
    ) -> tuple[float, float, dict[str, Any]] | None:
        if not expected_raw:
            return None
        expected = [_normalize_expected_tool(e) for e in expected_raw]

        matched = [False] * len(actual)
        correct = 0
        missing: list[str] = []
        for exp in expected:
            for i, act in enumerate(actual):
                if not matched[i] and _tool_matches(exp, act):
                    matched[i] = True
                    correct += 1
                    break
            else:
                missing.append(exp["name"])

        recall = correct / len(expected) if expected else 0.0
        precision = correct / len(actual) if actual else 0.0
        detail = {
            "expected_tools": len(expected),
            "actual_tools": len(actual),
            "matched": correct,
            "missing": missing,
            "unexpected": [a.name for i, a in enumerate(actual) if not matched[i]],
        }
        return round(precision, 4), round(recall, 4), detail

    # -- step efficiency ------------------------------------------------------

    def _step_efficiency(self, optimal: Any, trace: AgentTrace) -> MetricResult | None:
        if optimal is None:
            return None
        optimal = float(optimal)
        # "Steps" = reasoning steps (LLM calls); fall back to total if none tagged.
        actual = len(trace.llm_steps) or len(trace.steps)
        if actual <= 0:
            return MetricResult("agent_step_efficiency", 0.0, {"actual_steps": 0})
        efficiency = min(optimal / actual, 1.0)
        return MetricResult(
            "agent_step_efficiency",
            round(efficiency, 4),
            {"actual_steps": actual, "optimal_steps": optimal},
        )

    # -- error recovery -------------------------------------------------------

    def _error_recovery(self, trace: AgentTrace) -> MetricResult:
        errors = sum(1 for s in trace.steps if s.type == "tool_error")
        recoveries = sum(1 for s in trace.steps if s.type == "retry_after_error")
        # No errors -> nothing to recover from -> perfect by convention.
        rate = 1.0 if errors == 0 else min(recoveries / errors, 1.0)
        return MetricResult(
            "agent_error_recovery",
            round(rate, 4),
            {"tool_errors": errors, "recoveries": recoveries},
        )


def _normalize_expected_tool(entry: Any) -> dict[str, Any]:
    """Accept either a bare tool name or ``{name, arguments}``."""
    if isinstance(entry, str):
        return {"name": entry, "arguments": None}
    if isinstance(entry, dict):
        return {"name": str(entry.get("name", "")), "arguments": entry.get("arguments")}
    return {"name": str(entry), "arguments": None}


def _tool_matches(expected: dict[str, Any], actual: ToolCall) -> bool:
    if expected["name"] != actual.name:
        return False
    exp_args = expected.get("arguments")
    if not exp_args:
        return True
    # Every specified argument must be present and equal (subset match).
    return all(actual.arguments.get(k) == v for k, v in exp_args.items())
