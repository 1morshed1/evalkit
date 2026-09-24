"""Generation metrics via LLM-as-judge: faithfulness, relevance.

Faithfulness (the hard metric): decompose the answer into atomic claims, then
check each claim against the retrieved contexts. Score = supported / total.
Hallucination rate = 1 - faithfulness. Unsupported claims are surfaced by name
so a bad score is debuggable.

Relevance: does the answer actually address the question? Judge rates 0-100.
"""

from __future__ import annotations

from evalkit.evaluators.base import Evaluator
from evalkit.judges.base import Judge
from evalkit.schemas import MetricResult, Prediction, TestCase

CLAIMS_PROMPT = """Decompose the following answer into individual, atomic factual claims.
Each claim must be a single, self-contained, verifiable statement.
Return ONLY a JSON array of strings, nothing else.

Answer: {answer}"""

SUPPORT_PROMPT = """You are checking whether a CLAIM is supported by CONTEXT passages.
The claim is SUPPORTED if any passage states or clearly implies it, even if the
wording differs (paraphrases count). Unrelated passages do not matter.
The claim is UNSUPPORTED only if no passage backs it up or a passage contradicts it.

CONTEXT:
{context}

CLAIM: {claim}

Answer with a single word: "yes" if supported, "no" if not."""

RELEVANCE_PROMPT = """Judge how well the ANSWER addresses the QUESTION. Consider only
whether it responds to what was asked, not factual accuracy.

Choose exactly one label:
- FULLY: directly and completely answers the question.
- PARTIAL: addresses the question but is incomplete or partly off-topic.
- NONE: does not address the question at all.

QUESTION: {question}
ANSWER: {answer}

Reply with ONLY one word: FULLY, PARTIAL, or NONE."""

_RELEVANCE_SCORES = {"fully": 1.0, "partial": 0.5, "none": 0.0}


class GenerationEvaluator(Evaluator):
    name = "generation"
    requires: list[str] = []  # reference-free
    needs_judge = True

    def evaluate(
        self, case: TestCase, pred: Prediction, judge: Judge | None
    ) -> dict[str, MetricResult]:
        if judge is None:
            raise ValueError("GenerationEvaluator requires a judge.")
        out: dict[str, MetricResult] = {}

        # Relevance (needs only question + answer).
        out["relevance"] = self._relevance(case.question, pred.answer, judge)

        # Faithfulness (needs contexts). Skip cleanly if none retrieved.
        if pred.contexts:
            faith = self._faithfulness(pred.answer, pred.contexts, judge)
            out["faithfulness"] = faith
            out["hallucination_rate"] = MetricResult(
                "hallucination_rate", round(1.0 - faith.score, 4)
            )
        return out

    def _relevance(self, question: str, answer: str, judge: Judge) -> MetricResult:
        raw = judge.complete(RELEVANCE_PROMPT.format(question=question, answer=answer))
        label = raw.strip().lower()
        score = 0.0
        for key, value in _RELEVANCE_SCORES.items():
            if key in label:
                score = value
                break
        return MetricResult("relevance", score, {"raw": raw.strip()[:60]})

    def _faithfulness(
        self, answer: str, contexts: list[str], judge: Judge
    ) -> MetricResult:
        claims = self._extract_claims(answer, judge)
        if not claims:
            # No verifiable claims -> vacuously faithful.
            return MetricResult(
                "faithfulness", 1.0, {"total_claims": 0, "supported_claims": 0,
                                       "unsupported_claims": []}
            )
        context_text = "\n---\n".join(contexts)
        supported = 0
        unsupported: list[str] = []
        for claim in claims:
            prompt = SUPPORT_PROMPT.format(claim=claim, context=context_text)
            if judge.parse_yes_no(judge.complete(prompt)):
                supported += 1
            else:
                unsupported.append(claim)
        score = supported / len(claims)
        return MetricResult(
            "faithfulness",
            round(score, 4),
            {
                "total_claims": len(claims),
                "supported_claims": supported,
                "unsupported_claims": unsupported,
            },
        )

    def _extract_claims(self, answer: str, judge: Judge) -> list[str]:
        raw = judge.complete(CLAIMS_PROMPT.format(answer=answer))
        claims = judge.parse_json_array(raw)
        return [str(c).strip() for c in claims if str(c).strip()]
