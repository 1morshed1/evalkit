"""EvalKit — evaluate LLM applications (retrieval, generation, agents)."""

from __future__ import annotations

__version__ = "0.1.0"

from evalkit.config import load_suite
from evalkit.runner import run_suite

__all__ = ["__version__", "load_suite", "run_suite"]
