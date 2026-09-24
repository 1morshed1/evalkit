"""HTTP target: POST the question to an endpoint and read a JSON response."""

from __future__ import annotations

import time

import httpx

from evalkit.schemas import Prediction, TargetConfig
from evalkit.targets.base import Target


class HttpTarget(Target):
    def __init__(self, config: TargetConfig, timeout: float = 60.0):
        if not config.endpoint:
            raise ValueError("http target requires 'endpoint'.")
        self._endpoint = config.endpoint
        self._headers = config.headers
        self._timeout = timeout

    def predict(self, question: str) -> Prediction:
        start = time.perf_counter()
        resp = httpx.post(
            self._endpoint,
            json={"question": question},
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        return self.normalize(resp.json(), latency_ms=latency_ms)
