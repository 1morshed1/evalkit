"""On-disk cache wrapper for any Judge.

Keyed by hash(judge identity + prompt). The second run of an unchanged suite
makes zero LLM calls — the single highest-value engineering detail (see D6).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from evalkit.judges.base import Judge

DEFAULT_CACHE_DIR = Path(".evalkit_cache")


class CachedJudge(Judge):
    def __init__(self, inner: Judge, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self._inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.temperature = inner.temperature
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str) -> str:
        raw = f"{self._inner.cache_id()}\x00{prompt}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def complete(self, prompt: str) -> str:
        key = self._key(prompt)
        path = self._path(key)
        if path.exists():
            with self._lock:
                self.hits += 1
            try:
                return json.loads(path.read_text())["response"]
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt entry; fall through and recompute
        response = self._inner.complete(prompt)
        with self._lock:
            self.misses += 1
        path.write_text(json.dumps({"prompt": prompt, "response": response}))
        return response
