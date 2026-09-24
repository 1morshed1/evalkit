"""Load and validate a YAML evaluation suite into typed dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evalkit.schemas import JudgeConfig, Suite, TargetConfig, TestCase


class SuiteError(ValueError):
    """Raised when a suite file is malformed."""


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    if not path.exists():
        raise SuiteError(f"Suite file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SuiteError("Suite root must be a mapping.")
    return parse_suite(raw)


def parse_suite(raw: dict[str, Any]) -> Suite:
    suite_name = raw.get("suite_name")
    if not suite_name:
        raise SuiteError("Missing required field: suite_name")

    target = _parse_target(raw.get("target") or {})
    judge = _parse_judge(raw.get("judge") or {})

    cases_raw = raw.get("test_cases")
    if not cases_raw:
        raise SuiteError("Suite has no test_cases.")
    cases = [_parse_case(c, i) for i, c in enumerate(cases_raw)]

    thresholds = raw.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        raise SuiteError("thresholds must be a mapping of metric -> number.")

    return Suite(
        suite_name=str(suite_name),
        target=target,
        judge=judge,
        test_cases=cases,
        thresholds={k: float(v) for k, v in thresholds.items()},
    )


def _parse_target(raw: dict[str, Any]) -> TargetConfig:
    ttype = raw.get("type")
    if ttype not in ("http", "python"):
        raise SuiteError("target.type must be 'http' or 'python'.")
    tc = TargetConfig(
        type=ttype,
        endpoint=raw.get("endpoint"),
        import_path=raw.get("import"),
        headers=raw.get("headers") or {},
    )
    if tc.type == "http" and not tc.endpoint:
        raise SuiteError("target.type=http requires 'endpoint'.")
    if tc.type == "python" and not tc.import_path:
        raise SuiteError("target.type=python requires 'import' (module:function).")
    return tc


def _parse_judge(raw: dict[str, Any]) -> JudgeConfig:
    return JudgeConfig(
        provider=raw.get("provider", "ollama"),
        model=raw.get("model", "llama3.2"),
        base_url=raw.get("base_url"),
        temperature=float(raw.get("temperature", 0.0)),
        concurrency=int(raw.get("concurrency", 4)),
    )


def _parse_case(raw: dict[str, Any], idx: int) -> TestCase:
    if not isinstance(raw, dict):
        raise SuiteError(f"test_cases[{idx}] must be a mapping.")
    cid = raw.get("id") or f"case-{idx}"
    question = raw.get("question")
    if not question:
        raise SuiteError(f"test_cases[{idx}] ({cid}) missing 'question'.")
    return TestCase(
        id=str(cid),
        question=str(question),
        expected_answer=raw.get("expected_answer"),
        relevant_doc_ids=list(raw.get("relevant_doc_ids") or []),
        tags=list(raw.get("tags") or []),
        agent_trace=raw.get("agent_trace"),
        expected_result=raw.get("expected_result"),
    )
