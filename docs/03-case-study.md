# Case Study — Measuring a RAG Improvement with EvalKit

This walks through evaluating two versions of a RAG system and proving, with
numbers, that one is better. It uses the synthetic demo that ships in
[`demo/`](../demo/), so you can reproduce every number offline with Ollama.

> The demo is deliberately small and synthetic. The point is to show the
> *method* — the same commands work against a real service via the `http` target.

---

## The two systems

Both answer customer-support questions over the same 5-document corpus
([`demo/rag.py`](../demo/rag.py)):

| System | Retrieval | Answer |
|---|---|---|
| **Baseline** (`answer_weak`) | top-1 document only | grounded answer **+ an ungrounded marketing sentence** ("lifetime money-back guarantee") |
| **Candidate** (`answer`) | top-3 documents | grounded answer, nothing extra |

The baseline models two very common RAG failure modes: **thin retrieval** (only
one document, so multi-document questions lose recall) and **ungrounded
embellishment** (the model adds a plausible sentence the sources never say).

---

## Reproduce it

```bash
ollama pull llama3.2

# Baseline
evalkit run demo/eval_suite_baseline.yaml -o examples/baseline.json

# Candidate
evalkit run demo/eval_suite.yaml -o examples/candidate.json --html examples/report.html

# Compare
evalkit compare --baseline examples/baseline.json --candidate examples/candidate.json \
  --output examples/comparison.html
```

Judge = `llama3.2` via Ollama (local, free). All calls are cached, so the second
run is instant.

---

## Result

```
                 EvalKit — Baseline vs Candidate
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                    ┃ Baseline ┃ Candidate ┃ Change     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ retrieval_recall_at_5     │ 0.917    │ 1.000     │ +9.1%   ✅ │
│ retrieval_ndcg_at_5       │ 0.935    │ 1.000     │ +6.9%   ✅ │
│ faithfulness              │ 0.745    │ 1.000     │ +34.3%  ✅ │
│ relevance                 │ 0.500    │ 0.750     │ +50.0%  ✅ │
│ hallucination_rate        │ 0.256    │ 0.000     │ -100.0% ✅ │
│ avg_latency_ms            │ ~0       │ ~0        │ +25.0%  ❌ │
└───────────────────────────┴──────────┴───────────┴────────────┘
```

**Reading the table:**

- **Faithfulness 0.745 → 1.000, hallucination 0.256 → 0.000.** The baseline's
  extra marketing sentence is decomposed into a claim ("a lifetime money-back
  guarantee is offered") that no retrieved passage supports, so it is flagged as
  a hallucination. The candidate adds nothing ungrounded, so every claim is
  supported. This is the metric that matters most for RAG, and EvalKit names the
  exact offending sentence in the per-case report — open
  [`examples/report.html`](../examples/report.html) to see it.
- **Recall@5 0.917 → 1.000.** The one multi-document question
  (`refund-and-returns`, which needs *both* the refund and warranty docs) costs
  the top-1 baseline half its recall on that case; the top-3 candidate retrieves
  both.
- **Relevance 0.50 → 0.75.** The baseline's off-topic promo sentence drags the
  judged relevance down.
- **Latency +25% ❌.** The honest trade-off: retrieving three documents instead
  of one costs a little more. In this in-memory demo the absolute numbers are
  sub-millisecond (≈0), but EvalKit still flags the *direction* correctly —
  latency and cost are treated as "lower is better", so an increase is a ❌, not
  a ✅. On a real system this is exactly the signal you want when weighing a
  quality gain against its cost.

---

## Why this is the point

Without EvalKit, the story is "I switched to top-3 retrieval and removed some
boilerplate; I think it's better." With EvalKit, the story is the table above:
**+34% faithfulness, hallucinations eliminated, +9% recall, for a measured
latency cost.** One is an opinion; the other is evidence.

The same `evalkit run ... --fail-on-regression --baseline candidate.json` in CI
means the day someone reintroduces that marketing sentence, the pull request
goes red before it ships.
