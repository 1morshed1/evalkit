# EvalKit

**Evaluate LLM applications — retrieval quality, generation quality, and agent reliability — locally, reproducibly, and in CI.**

Every team building with LLMs has the same problem: they can't tell if their system is getting better or worse. They tweak a chunking strategy or swap a model, users complain or don't, and nobody has the numbers. EvalKit gives you the numbers.

```bash
pip install evalkit-py
```

> Import name is `evalkit`; the PyPI distribution is `evalkit-py`.

---

## Quick start (3 lines)

```python
from evalkit import load_suite, run_suite

result = run_suite(load_suite("eval_suite.yaml"))
print(result.summary)          # {'faithfulness': 0.91, 'retrieval_recall_at_5': 0.85, ...}
```

Or from the terminal:

```bash
evalkit run eval_suite.yaml --output results.json --html report.html
```

---

## What it measures

| Dimension | Metrics | Needs ground truth? | Needs a judge LLM? |
|---|---|---|---|
| **Retrieval** | recall@k, precision@k, MRR, nDCG@5, hit-rate | Yes (`relevant_doc_ids`) | No — pure math |
| **Generation** | faithfulness, relevance, hallucination rate | No (reference-free) | Yes |
| **Agent** | task completion, tool precision/recall, step efficiency, error recovery, cost | Partial | Optional |

**Metric definitions:**

- **recall@k** — of the documents that *should* have been retrieved, what fraction appeared in the top *k*.
- **precision@k** — of the top *k* retrieved documents, what fraction were relevant.
- **MRR** — mean reciprocal rank of the first relevant document (rewards putting the right doc first).
- **nDCG@5** — rank-weighted retrieval quality; earlier hits count more.
- **hit-rate** — did *any* relevant document make the top *k* (1 or 0 per case).
- **faithfulness** — fraction of the answer's atomic claims that are supported by the retrieved context. `1.0` means fully grounded.
- **hallucination rate** — `1 − faithfulness`; the fraction of claims *not* supported by context.
- **relevance** — how directly the answer addresses the question (LLM-judged, 0–1).
- **agent task completion** — did the agent achieve the goal (explicit flag, a required substring, or LLM-judged).
- **agent tool precision / recall** — of the tools the agent called, how many were expected (precision); of the expected tools, how many were called (recall). Arguments can be matched too.
- **agent step efficiency** — `optimal_steps / actual_steps`, capped at 1.0 (rewards getting there in fewer LLM calls).
- **agent error recovery** — of the tool calls that errored, what fraction the agent retried and recovered from.
- **agent cost (USD)** — total cost summed across the trace's steps.

**How faithfulness works:** the answer is decomposed into atomic claims by an LLM, and each claim is checked against the retrieved context. Unsupported claims are reported *by name*, so a low score is debuggable — you see the exact hallucinated sentence.

---

## The suite file

```yaml
suite_name: "Customer Support RAG - v2.1"

target:
  type: python                 # or: http
  import: "myapp.rag:answer"   # a callable question -> {answer, contexts, doc_ids}
  # endpoint: "http://localhost:8000/api/query"   # for type: http

judge:
  provider: ollama             # ollama (default, local, free) | openai | anthropic
  model: "llama3.2"

test_cases:
  - id: "refund-policy"
    question: "What is your refund policy?"
    expected_answer: "Full refund within 30 days with receipt."
    relevant_doc_ids: ["doc-policy-returns-v3"]   # optional (retrieval only)
    tags: ["policy", "refund"]

thresholds:
  retrieval_recall_at_5: 0.80
  faithfulness: 0.85
  relevance: 0.80
  hallucination_rate: 0.15     # a ceiling: lower is better
```

Your target must accept `{"question": str}` and return `{"answer": str, "contexts": [str], "doc_ids": [str]?}`. Two adapters ship in the box: **`python`** (import a callable) and **`http`** (POST to an endpoint). Metrics that need ground truth are **skipped and recorded** for cases that lack it — you don't have to label everything to measure anything.

---

## Judges: local and free by default

The LLM-as-judge (used for faithfulness and relevance) is pluggable. The default is **Ollama** running locally, so the whole suite runs with **zero API cost and no secrets** — ideal for CI and iteration. Swap in OpenAI or Anthropic when you want a stronger judge:

```yaml
judge: { provider: openai, model: gpt-4o-mini }      # needs OPENAI_API_KEY
judge: { provider: anthropic, model: claude-3-5-haiku-latest }  # needs ANTHROPIC_API_KEY
```

Every judge call runs at `temperature=0` and is **cached on disk** (keyed by prompt + model). Re-running an unchanged suite makes **zero** LLM calls — fast and cheap.

---

## Comparison mode (the killer feature)

When you change your chunking strategy or swap models, compare two runs:

```bash
evalkit compare --baseline results-v1.json --candidate results-v2.json --output comparison.html
```

```
                    EvalKit — Baseline vs Candidate
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric                  ┃ Baseline ┃ Candidate ┃ Change    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ retrieval_recall_at_5   │ 0.720    │ 0.850     │ +18.1% ✅ │
│ faithfulness            │ 0.810    │ 0.910     │ +12.3% ✅ │
│ relevance               │ 0.880    │ 0.920     │ +4.5%  ✅ │
│ hallucination_rate      │ 0.120    │ 0.040     │ -66.7% ✅ │
└─────────────────────────┴──────────┴───────────┴───────────┘
```

Comparison is **ceiling-aware**: a drop in `hallucination_rate` is scored as an improvement, not a regression.

---

## CI/CD integration

Two independent gates, because they answer different questions:

```bash
# Absolute gate — "is quality acceptable at all?" (no baseline needed)
evalkit run eval_suite.yaml --fail-below-thresholds

# Relative gate — "did this change make it worse?" (needs a baseline)
evalkit run eval_suite.yaml --fail-on-regression --baseline main-results.json
```

Either gate exits with code `1` on failure, so a GitHub Action fails the PR. See [`.github/workflows/eval.yml`](.github/workflows/eval.yml) for a complete example.

---

## Try the demo (offline, no API keys)

A tiny synthetic RAG ships in [`demo/`](demo/):

```bash
ollama pull llama3.2                       # one-time
evalkit run demo/eval_suite.yaml --html report.html
```

To see comparison mode, run it twice — once with the strong retriever (`demo.rag:answer`) and once with the weak one (`demo.rag:answer_weak`) — then `evalkit compare` the two result files. See [`docs/03-case-study.md`](docs/03-case-study.md).

---

## Agent evaluation

For tool-using agents, EvalKit scores the **execution trace** — not just the final
answer, but *how* the agent got there. Set `target.type: agent` and attach a trace
to each case; nothing live is called, and no judge or API key is needed (the judge is
used only if a case sets `expected_output`).

Traces come in two shapes. Paste an **OpenAI** message list straight from your logs:

```yaml
target:
  type: agent

test_cases:
  - id: "order-status"
    question: "What's the status of order 5591?"
    expected_result:
      success_contains: "shipped"
      expected_tool_calls:
        - {name: "get_order", arguments: {order_id: "5591"}}
      optimal_step_count: 2
    agent_trace:
      format: openai            # a list of OpenAI chat-completion messages
      messages:
        - {role: user, content: "What's the status of order 5591?"}
        - role: assistant
          cost_usd: 0.001
          tool_calls:
            - {id: c1, function: {name: get_order, arguments: '{"order_id": "5591"}'}}
        - {role: tool, tool_call_id: c1, content: "Order 5591: shipped."}
        - {role: assistant, content: "Order 5591 has shipped.", cost_usd: 0.001}
```

…or use the **native** form (`{steps: [...], final_output: "..."}`) if you don't
have OpenAI logs. `AgentTrace.from_openai(messages)` is also importable directly.
A tool call that errors and is retried is detected automatically and feeds the
error-recovery metric. Run it like any other suite:

```bash
evalkit run demo/agent_suite.yaml --html report.html
```

```
          EvalKit — Support Agent — reliability
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                ┃     Value ┃ Threshold ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ agent_task_completion │     0.667 │       0.8 │  FAIL  │
│ agent_tool_precision  │     0.500 │           │   —    │
│ agent_tool_recall     │     0.667 │       0.8 │  FAIL  │
│ agent_step_efficiency │     0.778 │           │   —    │
│ agent_error_recovery  │     1.000 │       0.9 │  PASS  │
│ agent_cost_usd        │ $0.002600 │     0.005 │  PASS  │
└───────────────────────┴───────────┴───────────┴────────┘
```

(The demo suite deliberately includes a failing case so the gate has something to
catch — one agent picks the wrong tool and never finds the answer.)

---

## Scaffold a suite

```bash
evalkit init eval_suite.yaml              # a RAG starter
evalkit init agent_suite.yaml --kind agent   # an agent starter
```

---

## Design & decisions

This project is documented as it was built. If you want the *why* behind every choice:

- [`docs/00-decisions.md`](docs/00-decisions.md) — every decision, with rationale and rejected alternatives.
- [`docs/01-architecture.md`](docs/01-architecture.md) — module layout and the data contracts.
- [`docs/02-roadmap.md`](docs/02-roadmap.md) — the phased build plan.
- [`docs/03-case-study.md`](docs/03-case-study.md) — a reproducible before/after on the demo RAG.
- [`docs/04-agent-evaluation.md`](docs/04-agent-evaluation.md) — trace formats, adapters, and every agent metric.

---

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest        # tests use a fake judge — no LLM required
uv run ruff check
```

## License

MIT.
