# Contributing to EvalKit

Thanks for your interest. EvalKit is deliberately small and built around three
one-job abstractions — **Target**, **Judge**, **Evaluator** — wired by a runner
that never hardcodes a concrete type. Most contributions are a new implementation
of one of those interfaces, and shouldn't require touching the runner at all.

## Development setup

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest          # tests use a fake judge — no LLM required
uv run ruff check      # lint + import order
uv run ruff format     # apply formatting
```

Please keep both `pytest` and `ruff check` green in every commit. Tests are
offline and deterministic (see `tests/conftest.py`'s `FakeJudge`); new tests
should stay that way — never require a live LLM or network.

## Project layout

See [`docs/01-architecture.md`](docs/01-architecture.md) for the full map and the
data contracts. The short version: `schemas.py` holds every dataclass (change it
reluctantly — it is the contract), `runner.py` orchestrates, and the three plugin
folders (`targets/`, `judges/`, `evaluators/`) are where most work happens.

## The three extension points

### Add a Target (a new way to reach the system under test)

Subclass `Target` and return a normalized `Prediction`. The base class's
`normalize()` accepts a `Prediction`, a plain string, or a dict.

```python
# evalkit/targets/mything.py
from evalkit.targets.base import Target
from evalkit.schemas import Prediction


class MyTarget(Target):
    def predict(self, question: str) -> Prediction:
        ...  # call your system
        return self.normalize({"answer": ..., "contexts": [...], "doc_ids": [...]})
```

Register it in `targets/__init__.py::build_target` under a new `target.type`, and
teach `config.py::_parse_target` to accept that type.

### Add a Judge (a new LLM-as-judge backend)

One method. Set `provider`/`model` so the on-disk cache key is stable.

```python
# evalkit/judges/mybackend.py
from evalkit.judges.base import Judge


class MyJudge(Judge):
    def __init__(self, config):
        self.provider, self.model = "mybackend", config.model

    def complete(self, prompt: str) -> str: ...  # return the model's text, temperature=0
```

Register it in `judges/__init__.py::build_judge`. The cache and concurrency are
handled for you by the runner and `judges/cache.py`.

### Add an Evaluator (a new metric or metric family)

Declare which case fields you need in `requires` (the runner skips-and-records a
case that lacks them, per D7), and whether you need a judge. One evaluator may
emit several named metrics.

```python
# evalkit/evaluators/mymetric.py
from evalkit.evaluators.base import Evaluator
from evalkit.schemas import MetricResult, Prediction, TestCase


class MyEvaluator(Evaluator):
    name = "mymetric"
    requires = ["expected_answer"]
    needs_judge = False

    def evaluate(self, case: TestCase, pred: Prediction, judge) -> dict[str, MetricResult]:
        return {"my_score": MetricResult("my_score", 0.0, {"detail": "..."})}
```

Add it to `evaluators/__init__.py` and to the appropriate list in
`runner.default_evaluators`. If lower is better for your metric (a ceiling like
cost or hallucination rate), make sure `runner.is_ceiling` recognizes its name so
gating and comparison score it correctly.

## Conventions

- Type hints everywhere; modern syntax (`list[str]`, `X | None`, `from __future__
  import annotations`).
- Keep docstrings that explain *why*, matching the surrounding style.
- New metrics get exact-value unit tests where they're deterministic (see
  `tests/test_retrieval.py` and `tests/test_agent.py` for the pattern).
- Update the relevant `docs/` page and the README when you add a user-facing
  feature — this project is documented as it is built.

## Pull requests

Small, focused PRs are easiest to review. Describe what changed and why, and note
any new interface or config surface. CI runs `pytest` and `ruff check`.
