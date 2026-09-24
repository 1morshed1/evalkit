# EvalKit — Decision Record

This document answers every open question from the plan review with a **recommended default** and, more importantly, **the reasoning behind it**. It is written to be read top-to-bottom as a learning resource: each decision explains the trade-off, not just the verdict.

> Format: each decision states the **Question**, the **Decision**, the **Why**, and (where useful) **What we rejected**.

---

## D1 — Scope: what ships in the MVP

**Question:** Build all three evaluators (retrieval + generation + agent) at once, or an MVP first?

**Decision:** Ship in phases. **Phase 1 = Retrieval + Generation.** **Agent evaluator = Phase 3.**

**Why:** Retrieval and generation cover the RAG use case, which is the plan's headline ("Customer Support RAG"), the case study, and the comparison table that goes on the CV. They also share the same simple data shape (question → answer → contexts). Agent evaluation needs an `AgentTrace` (tool calls, per-step cost, retries) that has *no standard format* — it drags in adapter work for LangChain / OpenAI / etc. Doing it last means the core framework is already proven when we tackle the messy part.

**What we rejected:** "Build everything then test." Greenfield frameworks that try to be complete before they are *usable* tend to die before the first real evaluation runs. We want a runnable `evalkit run` by end of Phase 1.

---

## D2 — Package name

**Question:** `pip install evalkit` — is the name free?

**Decision:** **Import name stays `evalkit`.** **Distribution (PyPI) name = `evalkit-py`.** Fallback single clean brand = `evallens` (free as both).

**Why:** Checked PyPI live: `evalkit` returns HTTP 200 → already taken, cannot publish under it. But the *import* name (what you type in `import evalkit`) does **not** have to match the PyPI distribution name. Keeping `import evalkit` preserves the repo name, the CV narrative, and every code example. We only rename the thing users type after `pip install`.

Availability checked (404 = free): `evalkit-py`, `evallens`, `probeval`, `evalbench-kit`, `gradekit`, `llmeval-kit`.

**What we rejected:** Renaming the whole project to force a matching name. Not worth rewriting every doc; the dist-name/import-name split is a normal Python pattern (e.g. `beautifulsoup4` → `import bs4`).

---

## D3 — How EvalKit talks to the target system (the most important interface)

**Question:** The plan shows two contradictory target modes — a YAML `endpoint: http://...` (HTTP) and `evalkit.evaluate(questions, ...)` (Python import). Which is primary?

**Decision:** Support **both**, behind a `Target` abstraction. Two built-in adapters:
- `type: http` + `endpoint:` → POST each question, read JSON response.
- `type: python` + `import: "mymodule:my_rag_fn"` → import and call a function.

Both adapters return the same normalized object: `{ answer: str, contexts: list[str] }`.

**Why:** They serve different users. HTTP is language-agnostic and mirrors production (your RAG already runs as a service). Python-callable is zero-setup for demos and unit-test-style evals. Hiding both behind one `Target` interface means the evaluators never know or care which was used — that is the whole point of an abstraction. Adding a third adapter later (e.g. a CLI subprocess) is then a drop-in.

**Contract the target must satisfy** (documented once, enforced by the adapter):
```
input:  { "question": str }
output: { "answer": str, "contexts": [str, ...], "doc_ids": [str, ...]? }
```
`doc_ids` is optional and only needed for retrieval metrics.

---

## D4 — Agent trace format (Phase 3, but decided now)

**Question:** `AgentEvaluator` needs an `AgentTrace` with tool calls, steps, per-step cost. Where does that come from? No standard exists.

**Decision:** Define **our own minimal `AgentTrace` schema** (a plain dataclass / JSON shape). Ship **one adapter** to start — from the **OpenAI tool-calling message list** — because it is the most common raw shape. LangChain/LlamaIndex adapters = optional later.

**Why:** We cannot wait for an industry standard that does not exist. Owning the schema keeps the evaluator clean; adapters translate *into* our schema at the edges. Picking OpenAI's format first covers the largest number of users with the least code. The schema is in `01-architecture.md`.

**What we rejected:** Depending directly on a framework's internal trace objects — that couples EvalKit to one framework's version churn.

---

## D5 — Judge LLM provider + default

**Question:** Which LLM-as-judge provider is the default?

**Decision:** **Provider-agnostic `Judge` interface** with three built-in backends: **Ollama (default)**, OpenAI, Anthropic. Default model = `llama3.2` (already installed locally).

**Why:** Ollama runs locally and free. That matters enormously here because (a) the demo/case study can run with zero API spend, (b) CI can run without secrets, and (c) you can iterate without a bill. OpenAI/Anthropic backends exist for users who want stronger judges. The interface is one method — `complete(prompt) -> str` — so swapping is trivial.

**What we rejected:** Hardcoding OpenAI. That would make the "free local demo" story impossible and add an API-key barrier to first run.

---

## D6 — Judge determinism, cost, and speed

**Question:** Faithfulness = (N claims × checks) LLM calls per test case. Expensive, slow, non-deterministic. Handle from the start?

**Decision:** From day one: **`temperature=0`**, **on-disk response cache** (hash of prompt+model → response), and **async concurrency** with a bounded worker pool.

**Why:**
- `temperature=0` → as reproducible as the model allows; needed so re-running an eval doesn't shift scores by noise.
- **Cache** → the same claim/context pair is judged once, not every run. Turns a slow 5-minute suite into seconds on re-run, and makes CI cheap. This is the single highest-value engineering detail in the project.
- **Async** → judge calls are IO-bound; running them concurrently (with a semaphore cap) cuts wall-clock time massively without hammering the provider.

**What we rejected:** Naive synchronous loop with no cache. Works in a toy demo, unusable on a real suite.

---

## D7 — Ground truth: mixed test suites

**Question:** Retrieval metrics need `relevant_doc_ids` ground truth; faithfulness/relevance need none (reference-free). Do suites mix both?

**Decision:** Yes. Each metric **declares what fields it requires**; the runner **skips a metric for a test case that lacks the required fields** and records it as `skipped` (not failed).

**Why:** Real datasets are uneven — you'll have doc-id labels for some questions, not others. Skipping-with-record is honest (the report shows coverage) and avoids forcing users to label everything before they can measure anything. Reference-free metrics (faithfulness, relevance, toxicity) run on every case; retrieval metrics run only where labels exist.

---

## D8 — HTML report: self-contained

**Question:** Single-file self-contained HTML — inline everything, no CDN?

**Decision:** **Yes, fully inline** — Jinja2 template with CSS and any JS embedded, no external network requests.

**Why:** The report is meant to be emailed to a stakeholder or uploaded as a CI artifact. It must render offline, in an airplane, in five years. A CDN link is a future 404. Charts, if any, use inline SVG rather than a charting CDN.

---

## D9 — Result JSON schema first

**Question:** Comparison mode reads two result JSONs. Design the schema early?

**Decision:** **Yes — the result JSON schema is designed before any evaluator is written.** Both `run` (writes it) and `compare` (reads it) depend on it. See `01-architecture.md`.

**Why:** The schema is the contract between subsystems. Nailing it first prevents a rewrite when `compare` discovers `run` didn't record what it needed. It also becomes the stable format users store in git to track quality over time.

---

## D10 — What counts as a "regression" for CI exit code

**Question:** `--fail-on-regression`: is a regression *below an absolute threshold*, or *below a baseline*? The plan shows both.

**Decision:** Support **both**, independently:
- `--fail-below-thresholds` → exit 1 if any metric is under its `thresholds:` value (absolute gate; needs no baseline).
- `--fail-on-regression --baseline old.json` → exit 1 if any metric dropped vs the baseline run (relative gate; needs a baseline).

**Why:** They answer different questions. Absolute thresholds = "is quality acceptable at all?" (good for a first PR). Relative regression = "did this change make it worse?" (good for ongoing PRs). A real CI wants both, and forcing one interpretation would surprise half of users.

---

## D11 — Case study dataset

**Question:** The README case study evaluates your own Shothik AI RAG (anonymized). Do we have exportable labeled data, or synthesize?

**Decision:** **Open question for you (see Queries below).** Default plan: build a small **synthetic but realistic** demo suite (10–20 cases) so the framework is provable end-to-end without your private data; swap in anonymized real data if/when available.

**Why:** The framework must be demonstrable regardless. A synthetic suite unblocks all development and gives the README its comparison table today. Real anonymized data makes the case study stronger later but is not on the critical path.

---

## D12 — Python version + tooling

**Question:** Python version and dependency manager?

**Decision:** **Python 3.11+** (dev on 3.12, which is installed). **`uv`** for env + dependency management (installed). **`pyproject.toml`** (PEP 621), **`ruff`** for lint/format, **`pytest`** for tests.

**Why:** 3.11 gives modern type syntax (`list[str]`, `X | None`) without shims and is widely available. `uv` is fast and already on this machine. `ruff` replaces black+flake8+isort with one fast tool — less config to explain, good for a learning project.

---

## D13 — License

**Question:** Open-source license?

**Decision:** **MIT.**

**Why:** Maximally permissive → maximum adoption, which is the point of a portfolio tool ("engineers use my tool"). No copyleft friction for anyone evaluating it at work.

---

## D14 — How an agent suite runs (Phase 3, as built)

**Question:** D4 fixed the `AgentTrace` schema and the OpenAI adapter, but not how a
suite of traces actually flows through a runner that was built around
`target.predict(question)`. There is no live system to call — the trace is the
input. How do agent cases run without bolting a second code path onto the runner?

**Decision:** Keep the one runner and express agent mode through the existing
abstractions:
- **A `target.type: agent`** selects a no-op **trace target** (`targets/trace.py`)
  that returns an empty `Prediction`. The runner's `target → evaluators` flow is
  unchanged; the trace and ground truth ride on the `TestCase`
  (`agent_trace`, `expected_result`), which the `AgentEvaluator` reads directly.
- **Evaluator selection is by target type** (`runner.default_evaluators`): `agent`
  suites run the `AgentEvaluator`; everything else runs the RAG pair.
- **The judge is optional.** Every agent metric is deterministic, so
  `AgentEvaluator.needs_judge = False` and no judge is constructed — an agent suite
  runs with zero API cost and no Ollama. A case may still opt into LLM-judged
  completion with `expected_output`, used only if a judge happens to be configured.
- **Error recovery is derived, not hand-authored.** `from_openai` inspects `tool`
  messages: an error flips a step to `tool_error`, and a subsequent call to the same
  tool becomes `retry_after_error`. So recovery is measurable straight from raw logs.

**Why:** Reusing the `Target`/`Evaluator`/runner contracts (D3, D7) kept agent
support additive — no branch in the hot loop, no second orchestration path. Offline
determinism preserves the D5/D6 "free local demo, cheap CI" story for agents too.
Deriving errors/retries from the trace means users paste logs rather than
hand-labelling recovery.

**What we rejected:** A separate `run_agent_suite` entry point (duplicates
aggregation, gating, and reporting), and requiring a judge for task completion
(would reintroduce the API-key barrier D5 removed). Full reference:
[`04-agent-evaluation.md`](04-agent-evaluation.md).

---

## Summary table

| # | Topic | Decision |
|---|---|---|
| D1 | MVP scope | Retrieval+Generation first; Agent in Phase 3 |
| D2 | Name | import `evalkit`, PyPI `evalkit-py` (fallback `evallens`) |
| D3 | Target interface | `Target` abstraction: `http` + `python` adapters |
| D4 | Agent trace | Own minimal schema; OpenAI adapter first |
| D5 | Judge provider | Pluggable; **Ollama/llama3.2 default**; OpenAI/Anthropic optional |
| D6 | Judge quality | temp=0 + on-disk cache + async concurrency |
| D7 | Ground truth | Metrics declare required fields; skip-and-record when missing |
| D8 | HTML report | Fully self-contained, inline, no CDN |
| D9 | Result schema | Design before evaluators |
| D10 | CI regression | Both absolute thresholds and relative-to-baseline |
| D11 | Case study | Synthetic demo suite now; anonymized real data later |
| D12 | Tooling | Python 3.11+, uv, pyproject, ruff, pytest |
| D13 | License | MIT |
| D14 | Agent mode | No-op trace target + evaluator-selection by type; judge optional; errors/retries derived from the trace |
