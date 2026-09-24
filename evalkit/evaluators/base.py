"""Evaluator abstract base class.

An evaluator may emit *several* named metrics (e.g. retrieval emits recall@k and
precision@k), so `evaluate` returns a dict of name -> MetricResult.

`requires` lists the case fields that must be present for this evaluator to run.
The runner checks it and records a skip when a field is missing (D7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from evalkit.judges.base import Judge
from evalkit.schemas import MetricResult, Prediction, TestCase


class Evaluator(ABC):
    name: str = "evaluator"
    #: TestCase fields that must be non-empty for this evaluator to apply.
    requires: list[str] = []
    #: whether this evaluator needs a Judge (LLM) to run.
    needs_judge: bool = False

    def applies_to(self, case: TestCase) -> bool:
        return all(case.has(field) for field in self.requires)

    @abstractmethod
    def evaluate(
        self, case: TestCase, pred: Prediction, judge: Judge | None
    ) -> dict[str, MetricResult]:
        """Return a mapping of metric-name -> MetricResult."""
