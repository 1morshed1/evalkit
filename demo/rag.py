"""A tiny synthetic RAG system.

Sparse keyword retrieval over a small in-memory corpus, plus a naive answer
composer. Each document carries an explicit ``keywords`` index (a stand-in for a
sparse/BM25 index) so retrieval is deterministic and question->doc mapping is
unambiguous — the point of the demo is to exercise EvalKit, not to be a good
retriever.

Two entry points drive EvalKit's comparison mode (see docs/03-case-study.md):

- ``answer``      : the CANDIDATE — top-3 retrieval, grounded answer.
- ``answer_weak`` : the BASELINE  — top-1 retrieval, and it appends an
                    ungrounded marketing sentence, so faithfulness drops and
                    the hallucination rate rises.

Everything runs offline; no external services.
"""

from __future__ import annotations

import re

CORPUS: list[dict] = [
    {
        "id": "doc-policy-returns-v3",
        "keywords": {"refund", "refunds", "policy", "receipt", "money", "30"},
        "text": "Customers may request a full refund within 30 days of purchase. "
        "A valid receipt is required for all returns.",
        "answer": "You can get a full refund within 30 days of purchase, and a "
        "receipt is required.",
    },
    {
        "id": "doc-policy-warranty-v2",
        "keywords": {
            "defective",
            "broken",
            "damaged",
            "warranty",
            "replacement",
            "faulty",
            "return",
            "returns",
            "90",
        },
        "text": "Defective or damaged products are covered by the warranty and can "
        "be returned within 90 days of purchase for a replacement or refund.",
        "answer": "Defective products can be returned within 90 days under the warranty policy.",
    },
    {
        "id": "doc-pricing-2024",
        "keywords": {"price", "cost", "enterprise", "plan", "pricing", "month", "seats"},
        "text": "The enterprise plan starts at $499 per month. Custom pricing is "
        "available for large teams with more than 100 seats.",
        "answer": "The enterprise plan starts at $499 per month, with custom pricing "
        "for large teams.",
    },
    {
        "id": "doc-shipping-1",
        "keywords": {"shipping", "delivery", "ship", "arrive", "express", "standard"},
        "text": "Standard shipping takes 3 to 5 business days. Express shipping "
        "delivers within 1 business day for an extra fee.",
        "answer": "Standard shipping takes 3 to 5 business days; express shipping "
        "arrives in 1 business day.",
    },
    {
        "id": "doc-support-hours",
        "keywords": {"support", "contact", "help", "hours", "available", "chat", "email"},
        "text": "Customer support is available Monday to Friday, 9am to 6pm, via "
        "email and live chat.",
        "answer": "Support is available Monday to Friday, 9am to 6pm, via email and live chat.",
    },
]

_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "do",
    "does",
    "you",
    "your",
    "i",
    "can",
    "get",
    "how",
    "much",
    "what",
    "of",
    "for",
    "to",
    "after",
    "if",
    "and",
    "my",
    "when",
    "on",
    "about",
    "me",
    "tell",
    "long",
    "take",
    "will",
    "it",
    "in",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in _STOP}


def _rank(question: str) -> list[dict]:
    """Stable rank by keyword-overlap score (ties keep corpus order)."""
    q = _tokens(question)
    scored = [(len(q & doc["keywords"]), i, doc) for i, doc in enumerate(CORPUS)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    hits = [doc for score, _, doc in scored if score > 0]
    return hits or [CORPUS[0]]


def answer(question: str) -> dict:
    """Candidate: top-3 retrieval, grounded answer from the top document."""
    ranked = _rank(question)[:3]
    return {
        "answer": ranked[0]["answer"],
        "contexts": [d["text"] for d in ranked],
        "doc_ids": [d["id"] for d in ranked],
    }


_HALLUCINATION = " We also offer a lifetime money-back guarantee on every product."


def answer_weak(question: str) -> dict:
    """Baseline: top-1 retrieval, plus an ungrounded marketing sentence."""
    ranked = _rank(question)[:1]
    return {
        "answer": ranked[0]["answer"] + _HALLUCINATION,
        "contexts": [ranked[0]["text"]],
        "doc_ids": [ranked[0]["id"]],
    }
