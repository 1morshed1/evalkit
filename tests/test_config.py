"""Suite parsing + validation tests."""

from __future__ import annotations

import pytest

from evalkit.config import SuiteError, parse_suite


def test_parse_minimal():
    suite = parse_suite(
        {
            "suite_name": "s",
            "target": {"type": "python", "import": "demo.rag:answer"},
            "test_cases": [{"id": "a", "question": "q?"}],
        }
    )
    assert suite.suite_name == "s"
    assert suite.judge.provider == "ollama"  # default
    assert suite.test_cases[0].id == "a"


def test_http_requires_endpoint():
    with pytest.raises(SuiteError):
        parse_suite(
            {"suite_name": "s", "target": {"type": "http"}, "test_cases": [{"question": "q"}]}
        )


def test_python_requires_import():
    with pytest.raises(SuiteError):
        parse_suite(
            {"suite_name": "s", "target": {"type": "python"}, "test_cases": [{"question": "q"}]}
        )


def test_bad_target_type():
    with pytest.raises(SuiteError):
        parse_suite(
            {"suite_name": "s", "target": {"type": "grpc"}, "test_cases": [{"question": "q"}]}
        )


def test_missing_question():
    with pytest.raises(SuiteError):
        parse_suite(
            {
                "suite_name": "s",
                "target": {"type": "python", "import": "m:f"},
                "test_cases": [{"id": "x"}],
            }
        )
