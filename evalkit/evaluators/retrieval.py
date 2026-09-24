"""Retrieval metrics: recall@k, precision@k, MRR, hit-rate, nDCG.

Pure math, no LLM. Deterministic — ideal for locking down the runner with exact
unit tests. Ground truth = case.relevant_doc_ids; prediction = pred.doc_ids
(ordered by rank, most relevant first).
"""

from __future__ import annotations

import math

from evalkit.evaluators.base import Evaluator
from evalkit.judges.base import Judge
from evalkit.schemas import MetricResult, Prediction, TestCase

DEFAULT_KS = (1, 3, 5, 10)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & relevant)
    return hits / len(top_k)


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # rank i is position i+1
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


class RetrievalEvaluator(Evaluator):
    name = "retrieval"
    requires = ["relevant_doc_ids"]
    needs_judge = False

    def __init__(self, ks: tuple[int, ...] = DEFAULT_KS):
        self.ks = ks

    def evaluate(
        self, case: TestCase, pred: Prediction, judge: Judge | None
    ) -> dict[str, MetricResult]:
        retrieved = pred.doc_ids
        relevant = set(case.relevant_doc_ids)
        out: dict[str, MetricResult] = {}
        for k in self.ks:
            out[f"retrieval_recall_at_{k}"] = MetricResult(
                f"retrieval_recall_at_{k}", recall_at_k(retrieved, relevant, k)
            )
            out[f"retrieval_precision_at_{k}"] = MetricResult(
                f"retrieval_precision_at_{k}", precision_at_k(retrieved, relevant, k)
            )
        out["retrieval_mrr"] = MetricResult("retrieval_mrr", mrr(retrieved, relevant))
        out["retrieval_hit_rate"] = MetricResult(
            "retrieval_hit_rate", hit_rate(retrieved, relevant, max(self.ks))
        )
        out["retrieval_ndcg_at_5"] = MetricResult(
            "retrieval_ndcg_at_5", ndcg_at_k(retrieved, relevant, 5)
        )
        return out
