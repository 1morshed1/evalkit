"""CLI-level tests: `evalkit init` and an agent `evalkit run`."""

from __future__ import annotations

from evalkit.cli import main
from evalkit.config import load_suite
from evalkit.runner import run_suite


def test_init_writes_runnable_rag_suite(tmp_path):
    path = tmp_path / "eval_suite.yaml"
    assert main(["init", str(path)]) == 0
    assert path.exists()
    suite = load_suite(path)  # parses without error
    assert suite.target.type == "python"


def test_init_agent_suite_runs_green(tmp_path):
    path = tmp_path / "agent.yaml"
    assert main(["init", str(path), "--kind", "agent"]) == 0
    result = run_suite(load_suite(path))  # no judge, no live target
    assert result.summary["agent_task_completion"] == 1.0
    assert result.gate.passed is True


def test_init_refuses_overwrite_without_force(tmp_path):
    path = tmp_path / "eval_suite.yaml"
    assert main(["init", str(path)]) == 0
    assert main(["init", str(path)]) == 1  # exists -> refuse
    assert main(["init", str(path), "--force"]) == 0  # explicit overwrite
