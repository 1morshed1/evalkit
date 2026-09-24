"""Python target: import a callable by 'module:function' and call it."""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable

from evalkit.schemas import Prediction, TargetConfig
from evalkit.targets.base import Target


class PythonTarget(Target):
    def __init__(self, config: TargetConfig):
        if not config.import_path or ":" not in config.import_path:
            raise ValueError("python target requires import as 'module:function'.")
        # Make modules in the current working directory importable (e.g. demo.rag).
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module_name, func_name = config.import_path.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name, None)
        if not callable(fn):
            raise ValueError(f"{config.import_path} is not callable.")
        self._fn: Callable = fn

    def predict(self, question: str) -> Prediction:
        start = time.perf_counter()
        payload = self._fn(question)
        latency_ms = (time.perf_counter() - start) * 1000
        return self.normalize(payload, latency_ms=latency_ms)
