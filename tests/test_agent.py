"""Agent evaluation: trace adapters, metric math, and an end-to-end run."""

from __future__ import annotations

from evalkit.config import parse_suite
from evalkit.evaluators import AgentEvaluator
from evalkit.runner import run_suite
from evalkit.schemas import AgentTrace, Prediction, TestCase

# --- trace adapters --------------------------------------------------------


def test_from_dict_native():
    trace = AgentTrace.from_dict(
        {
            "final_output": "done",
            "steps": [
                {"type": "llm", "cost_usd": 0.001},
                {"type": "tool_call", "tool_call": {"name": "search", "arguments": {"q": "x"}}},
            ],
        }
    )
    assert trace.final_output == "done"
    assert len(trace.llm_steps) == 1
    assert [tc.name for tc in trace.tool_calls] == ["search"]
    assert round(trace.total_cost_usd, 4) == 0.001


def test_from_openai_parses_calls_args_and_final_output():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "cost_usd": 0.002,
            "tool_calls": [{"id": "c1", "function": {"name": "get", "arguments": '{"id": "5"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "the answer", "cost_usd": 0.003},
    ]
    trace = AgentTrace.from_openai(messages)
    assert trace.final_output == "the answer"
    assert len(trace.llm_steps) == 2
    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.name == "get"
    assert call.arguments == {"id": "5"}
    assert call.result == "ok"
    assert round(trace.total_cost_usd, 4) == 0.005


def test_from_openai_marks_error_then_retry():
    messages = [
        {"role": "user", "content": "status?"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "a", "function": {"name": "fetch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "Error: timeout"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "b", "function": {"name": "fetch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "b", "content": "shipped"},
        {"role": "assistant", "content": "it shipped"},
    ]
    trace = AgentTrace.from_openai(messages)
    types = [s.type for s in trace.steps if s.tool_call is not None]
    assert types == ["tool_error", "retry_after_error"]
    assert trace.tool_calls[0].error == "Error: timeout"


# --- metric math -----------------------------------------------------------


def _case(agent_trace, expected_result):
    return TestCase(id="t", question="q", agent_trace=agent_trace, expected_result=expected_result)


def _eval(case):
    return AgentEvaluator().evaluate(case, Prediction(answer=""), judge=None)


def test_tool_precision_recall_and_arg_matching():
    trace = {
        "final_output": "ok",
        "steps": [
            {"type": "tool_call", "tool_call": {"name": "search", "arguments": {"q": "cats"}}},
            {"type": "tool_call", "tool_call": {"name": "noise", "arguments": {}}},
        ],
    }
    expected = {"expected_tool_calls": [{"name": "search", "arguments": {"q": "cats"}}]}
    m = _eval(_case(trace, expected))
    assert m["agent_tool_recall"].score == 1.0  # the one expected call was made
    assert m["agent_tool_precision"].score == 0.5  # 1 of 2 actual calls was expected


def test_tool_arg_mismatch_fails_match():
    trace = {
        "steps": [
            {"type": "tool_call", "tool_call": {"name": "search", "arguments": {"q": "dogs"}}}
        ]
    }
    expected = {"expected_tool_calls": [{"name": "search", "arguments": {"q": "cats"}}]}
    m = _eval(_case(trace, expected))
    assert m["agent_tool_recall"].score == 0.0


def test_step_efficiency_capped_at_one():
    trace = {
        "final_output": "x",
        "steps": [{"type": "llm"}, {"type": "llm"}, {"type": "llm"}, {"type": "llm"}],
    }
    m = _eval(_case(trace, {"optimal_step_count": 2}))
    assert m["agent_step_efficiency"].score == 0.5  # 2 optimal / 4 actual
    # An over-achieving trace never scores above 1.0.
    m2 = _eval(_case({"steps": [{"type": "llm"}]}, {"optimal_step_count": 3}))
    assert m2["agent_step_efficiency"].score == 1.0


def test_error_recovery_rate():
    trace = {
        "steps": [
            {"type": "tool_error"},
            {"type": "retry_after_error"},
            {"type": "tool_error"},
        ]
    }
    m = _eval(_case(trace, {}))
    assert m["agent_error_recovery"].score == 0.5  # 1 recovery / 2 errors
    # No errors -> perfect by convention.
    m2 = _eval(_case({"steps": [{"type": "llm"}]}, {}))
    assert m2["agent_error_recovery"].score == 1.0


def test_completion_basis_explicit_and_substring():
    m = _eval(_case({"final_output": "anything"}, {"success": False}))
    assert m["agent_task_completion"].score == 0.0

    m2 = _eval(_case({"final_output": "The price is $499/mo"}, {"success_contains": "$499"}))
    assert m2["agent_task_completion"].score == 1.0

    m3 = _eval(_case({"final_output": "no idea"}, {"success_contains": "$499"}))
    assert m3["agent_task_completion"].score == 0.0


def test_cost_metric_sums_steps():
    trace = {"steps": [{"type": "llm", "cost_usd": 0.001}, {"type": "llm", "cost_usd": 0.002}]}
    m = _eval(_case(trace, {}))
    assert m["agent_cost_usd"].score == 0.003


# --- end-to-end through the runner (agent mode, no judge, no target) --------

AGENT_SUITE = {
    "suite_name": "agent-e2e",
    "target": {"type": "agent"},
    "judge": {"provider": "ollama", "model": "llama3.2"},
    "test_cases": [
        {
            "id": "good",
            "question": "return window?",
            "expected_result": {
                "success_contains": "90",
                "expected_tool_calls": [{"name": "search_kb"}],
                "optimal_step_count": 2,
            },
            "agent_trace": {
                "format": "openai",
                "messages": [
                    {"role": "user", "content": "return window?"},
                    {
                        "role": "assistant",
                        "cost_usd": 0.001,
                        "tool_calls": [
                            {"id": "1", "function": {"name": "search_kb", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "1", "content": "90-day window"},
                    {"role": "assistant", "content": "You have 90 days.", "cost_usd": 0.001},
                ],
            },
        }
    ],
    "thresholds": {"agent_task_completion": 0.9, "agent_cost_usd": 0.01},
}


def test_agent_run_end_to_end_no_judge():
    suite = parse_suite(AGENT_SUITE)
    # No judge or target passed: agent mode must build a no-op trace target and
    # skip judge construction entirely.
    result = run_suite(suite)
    assert result.summary["agent_task_completion"] == 1.0
    assert result.summary["agent_tool_recall"] == 1.0
    assert result.summary["agent_error_recovery"] == 1.0
    assert result.gate.passed is True
    case = result.cases[0]
    assert "agent" not in case.skipped
    assert "agent_cost_usd" in case.metrics


def test_from_openai_via_demo_module():
    from demo.agent import ORDER_STATUS_WITH_RETRY

    trace = AgentTrace.from_openai(ORDER_STATUS_WITH_RETRY)
    assert "Tuesday" in trace.final_output
    error_steps = [s for s in trace.steps if s.type == "tool_error"]
    retry_steps = [s for s in trace.steps if s.type == "retry_after_error"]
    assert len(error_steps) == 1 and len(retry_steps) == 1
