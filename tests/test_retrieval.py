"""Exact-value tests for the deterministic retrieval metrics."""

from __future__ import annotations

import math

from evalkit.evaluators.retrieval import (
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "x"}
    assert recall_at_k(retrieved, relevant, 5) == 0.5  # 1 of 2 relevant found
    assert recall_at_k(retrieved, relevant, 1) == 0.0  # 'a' not relevant
    assert recall_at_k(retrieved, relevant, 2) == 0.5


def test_precision_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_hit_rate():
    assert hit_rate(["a", "b"], {"b"}, 5) == 1.0
    assert hit_rate(["a", "b"], {"z"}, 5) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5  # first relevant at rank 2
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["a", "b"], {"z"}) == 0.0


def test_ndcg():
    # Relevant at rank 1 only -> dcg = 1/log2(2)=1; idcg (1 relevant) = 1 -> 1.0
    assert ndcg_at_k(["a", "b"], {"a"}, 5) == 1.0
    # Relevant at rank 2 -> dcg = 1/log2(3); idcg = 1 -> that ratio
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["a", "b"], {"b"}, 5), expected)
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_empty_relevant():
    assert recall_at_k(["a"], set(), 5) == 0.0
