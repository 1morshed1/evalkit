"""Shared test fixtures: a deterministic fake judge (no LLM needed)."""

from __future__ import annotations

import json

from evalkit.judges.base import Judge


class FakeJudge(Judge):
    """Scriptable judge for tests.

    - CLAIMS prompt -> returns configured claims as a JSON array.
    - SUPPORT prompt -> 'yes' unless the claim appears in `unsupported`.
    - RELEVANCE prompt -> configured relevance integer.
    """

    def __init__(self, claims=None, unsupported=None, relevance="FULLY"):
        self.provider = "fake"
        self.model = "fake"
        self.temperature = 0.0
        self._claims = claims or []
        self._unsupported = set(unsupported or [])
        self._relevance = relevance
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if "Decompose" in prompt:
            return json.dumps(self._claims)
        if prompt.strip().startswith("Judge how well"):
            return str(self._relevance)
        if "CLAIM:" in prompt:
            claim_line = ""
            for line in prompt.splitlines():
                if line.startswith("CLAIM:"):
                    claim_line = line[len("CLAIM:"):].strip()
                    break
            return "no" if claim_line in self._unsupported else "yes"
        return ""
