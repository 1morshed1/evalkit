"""Canonical result JSON: the contract between run, compare, and html."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalkit.schemas import RunResult


def write_json(result: RunResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2))


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
