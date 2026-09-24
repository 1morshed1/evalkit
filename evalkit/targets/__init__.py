"""Target adapters: turn a question into a normalized Prediction."""

from __future__ import annotations

from evalkit.schemas import TargetConfig
from evalkit.targets.base import Target
from evalkit.targets.http import HttpTarget
from evalkit.targets.python import PythonTarget


def build_target(config: TargetConfig) -> Target:
    if config.type == "http":
        return HttpTarget(config)
    if config.type == "python":
        return PythonTarget(config)
    raise ValueError(f"Unknown target type: {config.type}")


__all__ = ["Target", "HttpTarget", "PythonTarget", "build_target"]
