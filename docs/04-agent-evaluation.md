# EvalKit — Agent Evaluation

The RAG evaluators score an *answer*. Agents are different: the answer alone
doesn't tell you whether the agent was reliable — it might have reached a correct
answer by luck after five wrong tool calls and two crashes. So EvalKit scores the
**execution trace**: what the agent did, in what order, at what cost.

This dimension is fully **offline and deterministic** — no live target, no judge,
and no API keys — because the trace is already recorded on the case. (The judge is
used only if a case opts into LLM-judged completion via `expected_output`.)

---

## Running an agent suite

Set `target.type: agent` and attach a trace to each case. The runner then selects
the `AgentEvaluator` instead of the retrieval + generation pair, and builds a no-op
"trace target" (there is nothing live to call).

```yaml
target:
  type: agent
judge:
  provider: ollama        # unused unless a case sets expected_output
  model: "llama3.2"
test_cases:
  - id: "..."
    question: "..."
    expected_result: { ... }   # ground truth (all keys optional)
    agent_trace: { ... }       # what the agent actually did
```

```bash
evalkit run demo/agent_suite.yaml --html report.html
evalkit init my_agents.yaml --kind agent   # scaffold a starter
```

---

## Trace formats

Two shapes are accepted; both parse into the same `AgentTrace`
(`evalkit/schemas.py`). The evaluator only ever sees the parsed `AgentTrace`.

### OpenAI messages (recommended — paste from your logs)

```yaml
agent_trace:
  format: openai            # or simply include a top-level `messages:` key
  messages:
    - {role: user, content: "What's the status of order 5591?"}
    - role: assistant
      cost_usd: 0.001       # optional per-message cost, summed into agent_cost_usd
      tool_calls:
        - {id: c1, function: {name: get_order, arguments: '{"order_id": "5591"}'}}
    - {role: tool, tool_call_id: c1, content: "Order 5591: shipped."}
    - {role: assistant, content: "Order 5591 has shipped.", cost_usd: 0.001}
```

`AgentTrace.from_openai(messages)` maps the messages as follows:

| OpenAI message | Becomes |
|---|---|
| `assistant` (any) | an `llm` step (carrying `cost_usd`/`cost` if present) |
| each `tool_calls[]` entry | a tool step; `arguments` JSON-string is parsed to a dict |
| `tool` message | resolves the matching tool step to success or **error** |
| last `assistant` with text | the trace's `final_output` |

**Error and retry detection is automatic.** A `tool` message whose content is an
error (`is_error: true`, or content starting with `error`/`exception`/`traceback`/
`failed`) flips its tool step from `tool_call` to `tool_error`. If the agent then
calls the *same tool name* again, that later step is tagged `retry_after_error` —
which is exactly what the error-recovery metric counts.

`content` may be a plain string or the OpenAI list-of-parts form; both are
flattened. It's also importable directly:

```python
from evalkit import AgentTrace
trace = AgentTrace.from_openai(messages)
```

### Native form

If you don't have OpenAI logs, describe the steps directly:

```yaml
agent_trace:
  final_output: "Order 5591 has shipped."
  steps:
    - {type: llm, cost_usd: 0.001}
    - {type: tool_call, tool_call: {name: get_order, arguments: {order_id: "5591"}, result: "..."}}
    - {type: llm, cost_usd: 0.001}
```

Step `type` is one of `llm`, `tool_call`, `tool_error`, `retry_after_error`. Any
step with a `tool_call` counts as a tool invocation (errors included); `llm` steps
are the "reasoning steps" used for step efficiency.

---

## The `expected_result` (ground truth)

Every key is optional — measure only what you've labelled.

```yaml
expected_result:
  success: true                       # explicit pass/fail — wins over everything
  success_contains: "shipped"         # substring(s) that must appear in final_output
  expected_output: "the order shipped"# LLM-judged completion (needs a judge)
  expected_tool_calls:                # names, or name + an argument subset to match
    - {name: "get_order", arguments: {order_id: "5591"}}
    - "notify_user"
  optimal_step_count: 2               # baseline for step efficiency
```

---

## The metrics

All are per-case, then averaged into the run summary. Higher is better except
`agent_cost_usd`, which is a **ceiling** (lower is better) and gates accordingly.

### `agent_task_completion`
Did the agent achieve the goal? Resolved by the first applicable rule:
1. `success` (explicit bool) → 1.0 / 0.0
2. `success_contains` → 1.0 if every substring appears in `final_output` (case-insensitive)
3. `expected_output` **and** a judge available → the judge answers yes/no
4. otherwise → 1.0 if a non-empty `final_output` was produced, else 0.0

The chosen basis is recorded in the metric detail (`explicit`, `success_contains`,
`judge`, `final_output_present`) so a score is always explainable.

### `agent_tool_precision` / `agent_tool_recall`
Greedy one-to-one match between `expected_tool_calls` and the trace's actual tool
calls. A match requires the same tool `name`; if the expected entry lists
`arguments`, each specified key must be present and equal in the actual call (a
**subset** match, so you needn't pin every argument).

- **recall** = matched / expected — *did the agent make the calls it should have?*
- **precision** = matched / actual — *did it avoid calls it shouldn't have?*

Detail lists `missing` and `unexpected` tools. Both metrics are skipped when a
case has no `expected_tool_calls`.

### `agent_step_efficiency`
`min(optimal_step_count / actual_steps, 1.0)`, where `actual_steps` is the number
of `llm` steps (falling back to total steps if none are tagged `llm`). Capped at
1.0 so an under-budget trace never scores above a perfect run. Skipped when
`optimal_step_count` is absent.

### `agent_error_recovery`
`recoveries / tool_errors`, i.e. `retry_after_error` steps over `tool_error` steps.
With **no errors there is nothing to recover from**, so the score is 1.0 by
convention (a clean run is not penalized).

### `agent_cost_usd`
Sum of `cost_usd` across all steps. Rendered as currency in every report and
treated as a ceiling by the gate and by comparison mode (a drop is an improvement).

---

## Worked example (the demo suite)

`demo/agent_suite.yaml` has three cases: a clean lookup, a call that times out and
is retried successfully, and a case where the agent picks the wrong tool and never
finds the answer. That last case is deliberate — it gives the gate something to
catch:

```
│ agent_task_completion │ 0.667 │  (2 of 3 tasks completed)
│ agent_tool_precision  │ 0.500 │
│ agent_tool_recall     │ 0.667 │
│ agent_step_efficiency │ 0.778 │
│ agent_error_recovery  │ 1.000 │  (the one timeout was retried and recovered)
│ agent_cost_usd        │ $0.002600 │
```

---

## Where it lives

| Concern | File |
|---|---|
| Trace types + adapters (`from_openai`, `from_dict`) | `evalkit/schemas.py` |
| Metric logic | `evalkit/evaluators/agent.py` |
| Evaluator selection by `target.type` | `evalkit/runner.py` (`default_evaluators`) |
| No-op trace target | `evalkit/targets/trace.py` |
| Demo traces (OpenAI form) | `demo/agent.py`, `demo/agent_suite.yaml` |
| Tests | `tests/test_agent.py` |
