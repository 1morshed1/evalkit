# EvalKit — Architecture & Data Contracts

This document defines the module layout and the **data contracts** (schemas) that hold the system together. The contracts matter more than the code: if the schemas are right, the code is replaceable.

---

## Package layout

```
evalkit/
├── __init__.py            # public API: load_suite(), run_suite(), AgentTrace
├── cli.py                 # `evalkit run|compare|init`
├── config.py              # load/validate YAML suite -> Suite dataclass
├── targets/
│   ├── base.py            # Target ABC -> returns Prediction
│   ├── http.py            # POST endpoint adapter
│   ├── python.py          # import "module:fn" adapter
│   └── trace.py           # no-op target for agent suites (trace is inline)
├── judges/
│   ├── base.py            # Judge ABC: complete(prompt) -> str
│   ├── ollama.py          # default
│   ├── openai.py
│   ├── anthropic.py
│   └── cache.py           # on-disk prompt->response cache
├── evaluators/
│   ├── base.py            # Evaluator ABC: requires[], evaluate(case, pred, judge)
│   ├── retrieval.py       # recall@k, precision@k, MRR, nDCG, hit-rate
│   ├── generation.py      # faithfulness, relevance, hallucination rate
│   └── agent.py           # task completion, tool P/R, step efficiency, recovery, cost
├── runner.py              # orchestrates: target -> evaluators -> results
├── report/
│   ├── console.py         # Rich table
│   ├── json_report.py     # canonical result JSON (see below)
│   ├── html.py            # self-contained Jinja2 report
│   └── compare.py         # baseline vs candidate
└── schemas.py             # all dataclasses + trace adapters in one place
```

The `runner` chooses the evaluator set from `target.type`: an `agent` suite runs
the `AgentEvaluator` (offline, trace-scoring); everything else runs the RAG
retrieval + generation pair. See [`04-agent-evaluation.md`](04-agent-evaluation.md).

**Design principle:** the three subsystems — **Target** (produces answers), **Judge** (scores with an LLM), **Evaluator** (computes a metric) — are each an abstract base class with one job. The **runner** wires them together and never hardcodes a concrete type. Swap Ollama for OpenAI, HTTP for Python, without touching the runner.

---

## Contract 1 — Test suite (YAML input)

Matches the plan's format, with two clarifications from the decisions:
- `target` picks an adapter (`http` or `python`).
- `relevant_doc_ids` is **optional** per case (D7).

```yaml
suite_name: "Customer Support RAG - v2.1"
target:
  type: python                       # http | python | agent
  import: "demo.rag:answer"          # for python
  # endpoint: "http://localhost:8000/api/query"   # for http

judge:
  provider: ollama                   # ollama | openai | anthropic
  model: "llama3.2"

test_cases:
  - id: "refund-policy-basic"
    question: "What is your refund policy?"
    expected_answer: "Full refund within 30 days with receipt."
    relevant_doc_ids: ["doc-policy-returns-v3"]   # optional
    tags: ["policy", "refund"]
    # Agent suites (target.type: agent) add two per-case fields instead:
    # agent_trace: {...}        # what the agent did (see Contract 4)
    # expected_result: {...}    # agent ground truth (see 04-agent-evaluation.md)

thresholds:
  retrieval_recall_at_5: 0.80
  faithfulness: 0.85
  relevance: 0.90
  hallucination_rate: 0.05           # max allowed (a ceiling, not floor)
```

---

## Contract 2 — Prediction (Target output, normalized)

Every target adapter returns this, regardless of transport:

```python
@dataclass
class Prediction:
    answer: str
    contexts: list[str]              # retrieved passages (RAG); [] for plain LLM
    doc_ids: list[str] = field(default_factory=list)  # retrieved ids, if known
    latency_ms: float | None = None
    cost_usd: float | None = None
```

---

## Contract 3 — Result JSON (THE canonical output) — design-first (D9)

Written by `run`, read by `compare`, `html`, `console`. This is the contract users store in git.

```json
{
  "evalkit_version": "0.1.0",
  "suite_name": "Customer Support RAG - v2.1",
  "run_id": "2026-09-24T14:03:00Z",
  "judge": { "provider": "ollama", "model": "llama3.2" },
  "summary": {
    "retrieval_recall_at_5": 0.85,
    "faithfulness": 0.91,
    "relevance": 0.92,
    "hallucination_rate": 0.04,
    "avg_latency_ms": 1890,
    "avg_cost_usd": 0.0
  },
  "thresholds": { "faithfulness": 0.85, "hallucination_rate": 0.05 },
  "gate": { "passed": true, "violations": [] },
  "cases": [
    {
      "id": "refund-policy-basic",
      "tags": ["policy", "refund"],
      "prediction": { "answer": "...", "contexts": ["..."], "doc_ids": ["..."] },
      "metrics": {
        "faithfulness": {
          "score": 1.0, "total_claims": 3, "supported_claims": 3,
          "unsupported_claims": []
        },
        "relevance": { "score": 0.9 },
        "retrieval_recall_at_5": { "score": 1.0, "status": "ok" }
      },
      "skipped": []                    // metrics skipped for missing ground truth
    }
  ]
}
```

**Why per-case detail is kept:** the summary is for the CV table; the per-case block (especially `unsupported_claims`) is what lets you *debug* a bad score — it names the exact hallucinated sentence.

---

## Contract 4 — AgentTrace (D4)

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: str | None = None
    error: str | None = None

@dataclass
class Step:
    type: str                # "llm" | "tool_call" | "tool_error" | "retry_after_error"
    cost_usd: float = 0.0
    tool_call: ToolCall | None = None

@dataclass
class AgentTrace:
    steps: list[Step]
    final_output: str = ""

    @property
    def tool_calls(self) -> list[ToolCall]:   # every invocation, errors included
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def llm_steps(self) -> list[Step]: ...     # reasoning steps (for efficiency)
    @property
    def total_cost_usd(self) -> float: ...     # sum over steps

    @classmethod
    def from_openai(cls, messages) -> "AgentTrace": ...   # Contract 4 adapter
    @classmethod
    def from_dict(cls, data) -> "AgentTrace": ...         # native or openai shape
```

Adapters translate a framework's raw trace into this — the evaluator only ever
sees `AgentTrace`. `from_openai` also detects tool errors and retries so error
recovery is measurable from raw logs. Full field/metric reference:
[`04-agent-evaluation.md`](04-agent-evaluation.md).

---

## Contract 5 — Evaluator interface

```python
class Evaluator(ABC):
    name: str
    requires: list[str]              # e.g. ["relevant_doc_ids"]; runner skips if missing
    needs_judge: bool = False        # runner only builds a judge if some evaluator needs one

    @abstractmethod
    def evaluate(
        self, case: TestCase, pred: Prediction, judge: Judge | None
    ) -> dict[str, MetricResult]: ...   # one evaluator may emit several named metrics
```

`requires` is how D7 (skip-and-record) is enforced generically: the runner checks
each evaluator's `requires` against the case's available fields before calling it
(e.g. `agent` requires `agent_trace`). `needs_judge` lets an agent-only suite run
with no judge constructed at all.

---

## Data flow (one test case)

```
runner picks evaluators from target.type (RAG pair, or AgentEvaluator)
TestCase.question
   -> Target.predict()            -> Prediction {answer, contexts, doc_ids}
                                     (agent suites: a no-op trace target; the
                                      trace + ground truth ride on the TestCase)
   -> for each Evaluator:
        if requires satisfied:
            Evaluator.evaluate()   -> {name: MetricResult}   (may call Judge)
        else:
            record as skipped
   -> aggregate into case result
runner aggregates all cases -> summary -> gate check -> Result JSON
                                                     -> console / html / compare
```

---

## The faithfulness pipeline (the hard metric, D6 applied)

```
answer --extract_claims(judge)--> [claim1, claim2, claim3]
for each claim:
    check_claim_support(claim, contexts) via judge   # cached, temp=0, async
score = supported / total_claims
unsupported_claims -> reported as potential hallucinations
hallucination_rate = 1 - faithfulness (per case), averaged for summary
```

Both judge calls go through `judges/cache.py`, so the second run of an unchanged suite makes **zero** LLM calls.
