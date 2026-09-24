"""Trace target: for agent suites whose traces are already recorded inline.

There is nothing live to call — the execution trace lives on each case
(``agent_trace``) and the :class:`~evalkit.evaluators.agent.AgentEvaluator`
reads it directly. This adapter exists only to satisfy the runner's
target contract, returning an empty Prediction per case.
"""

from __future__ import annotations

from evalkit.schemas import Prediction
from evalkit.targets.base import Target


class TraceTarget(Target):
    def predict(self, question: str) -> Prediction:
        return Prediction(answer="")
