"""Target abstract base class.

A Target is anything that answers a question. The evaluators never know which
concrete transport produced the Prediction — that is the point of the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evalkit.schemas import Prediction


class Target(ABC):
    @abstractmethod
    def predict(self, question: str) -> Prediction:
        """Answer a question, returning a normalized Prediction."""

    @staticmethod
    def normalize(payload: Any, latency_ms: float | None = None) -> Prediction:
        """Coerce an arbitrary target response into a Prediction.

        Accepts either a Prediction, a plain string (answer only), or a dict with
        keys: answer, contexts, doc_ids, cost_usd.
        """
        if isinstance(payload, Prediction):
            if latency_ms is not None and payload.latency_ms is None:
                payload.latency_ms = latency_ms
            return payload
        if isinstance(payload, str):
            return Prediction(answer=payload, latency_ms=latency_ms)
        if isinstance(payload, dict):
            return Prediction(
                answer=str(payload.get("answer", "")),
                contexts=list(payload.get("contexts") or []),
                doc_ids=list(payload.get("doc_ids") or []),
                latency_ms=latency_ms,
                cost_usd=payload.get("cost_usd"),
            )
        raise TypeError(
            f"Target returned unsupported type {type(payload)!r}; "
            "expected Prediction, str, or dict."
        )
