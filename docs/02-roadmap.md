# EvalKit — Build Roadmap

Phased plan. Each phase ends with something **runnable and committed**. Order chosen so the contracts (schemas) exist before the code that depends on them, and so a real evaluation runs as early as possible.

Legend: ☐ todo · ✅ done · each phase = one or more commits.

**Status:** Phases 0–3 are complete. Phase 4 is complete except publishing to PyPI
(`evalkit-py`), which needs registry credentials.

---

## Phase 0 — Scaffold (foundation)

Goal: `uv`-managed package that imports and lints clean.

- ✅ `pyproject.toml` (PEP 621): name `evalkit-py`, import pkg `evalkit`, Python ≥3.11
- ✅ Dev deps: `pytest`, `ruff`; runtime deps: `pyyaml`, `rich`, `jinja2`, `httpx`
- ✅ `evalkit/__init__.py` with version + public API stubs
- ✅ `evalkit/schemas.py` — **all dataclasses** (Prediction, TestCase, MetricResult, Result, AgentTrace). Contracts first (D9).
- ✅ MIT `LICENSE`, `.gitignore`, minimal `README.md`
- ✅ `ruff check` + `pytest` (empty) green

**Done when:** `uv run python -c "import evalkit"` works; CI-able skeleton exists.

---

## Phase 1 — Core loop: run a suite (RAG retrieval + generation)

Goal: `evalkit run eval_suite.yaml` produces a console table + result JSON on a **real** (synthetic) RAG.

- ✅ `config.py` — parse + validate YAML suite into dataclasses
- ✅ `targets/base.py` + `targets/python.py` (import `module:fn`) + `targets/http.py`
- ✅ `judges/base.py` + `judges/ollama.py` + `judges/cache.py` (temp=0, on-disk cache — D6)
- ✅ `evaluators/retrieval.py`: recall@k, precision@k, MRR, hit-rate, nDCG (pure math, no LLM — start here, easiest + deterministic)
- ✅ `evaluators/generation.py`: faithfulness (claim decomposition), relevance (LLM-as-judge)
- ✅ `runner.py`: wire target → evaluators → aggregate; enforce `requires`/skip (D7); async judge pool
- ✅ `report/console.py` (Rich) + `report/json_report.py` (canonical schema)
- ✅ `demo/rag.py` — tiny in-memory RAG + `demo/eval_suite.yaml` (10–20 cases) so the whole thing runs offline
- ✅ Tests: retrieval metrics unit-tested against known values; runner smoke test with a fake judge

**Done when:** `evalkit run demo/eval_suite.yaml` prints a metrics table and writes `results.json`, using Ollama, zero API cost.

**Why retrieval before generation:** retrieval metrics are pure math with exact expected values — perfect for locking the runner + schema down with deterministic tests before adding the noisy LLM-judge parts.

---

## Phase 2 — Reports, comparison, CI gate

Goal: the outputs that prove the framework works.

- ✅ `report/html.py` — self-contained Jinja2 report, inline CSS, inline SVG charts (D8)
- ✅ `report/compare.py` + `evalkit compare --baseline --candidate` → the change table (the killer feature)
- ✅ CI gate flags: `--fail-below-thresholds` (absolute) and `--fail-on-regression --baseline` (relative) → exit code 1 (D10)
- ✅ `.github/workflows/eval.yml` example
- ✅ README: metrics table + definitions, 3-line quick start, comparison-table screenshot

**Done when:** `evalkit compare` emits the before/after table (the README centerpiece) and CI exits non-zero on a real regression.

---

## Phase 3 — Agent evaluation

Goal: evaluate tool-using agents.

- ✅ `evaluators/agent.py`: task completion, tool precision/recall, step efficiency, error recovery, cost
- ✅ `AgentTrace` adapter: `from_openai(messages)` (Contract 4)
- ✅ `demo/agent.py` trace + suite; tests
- ✅ README: agent metrics section

**Done when:** an OpenAI-style trace scores through `evalkit run` with agent metrics.

---

## Phase 4 — Polish & publish

- ✅ Case study section: evaluate a RAG (the synthetic demo suite) showing a measured improvement
- ✅ `openai.py` + `anthropic.py` judge backends (parity with Ollama)
- ☐ Package metadata, `README` badges, publish to PyPI as `evalkit-py`
- ✅ Optional: `evalkit init` scaffolds a starter suite

---

## Critical path

```
Phase 0 schemas ──> Phase 1 runner ──> Phase 2 compare/CI ──> release-ready
                                   └──> Phase 3 agent (parallel-ish, independent)
```

Everything hinges on the **schemas (Phase 0)** and the **result JSON (Contract 3)**. Those are built first and changed reluctantly.

---

## Remaining

The framework is feature-complete for local and CI use. The only open item is
publishing the package to PyPI (the name `evalkit-py` has since been taken, so the
distribution needs a new name first — see `docs/00` D2);
until then, install from source with `uv pip install -e ".[dev]"`.
