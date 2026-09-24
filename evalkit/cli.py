"""EvalKit command-line interface: `evalkit run` and `evalkit compare`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from evalkit import __version__
from evalkit.config import load_suite
from evalkit.report import (
    compare_results,
    load_result,
    render_comparison_console,
    render_console,
    render_html,
    write_json,
)
from evalkit.report.compare import regressions, render_comparison_html
from evalkit.runner import run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Evaluate LLM applications.")
    parser.add_argument("--version", action="version", version=f"evalkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run an evaluation suite.")
    p_run.add_argument("suite", help="Path to the YAML suite file.")
    p_run.add_argument("--output", "-o", default="results.json", help="Result JSON path.")
    p_run.add_argument("--html", default=None, help="Also write a self-contained HTML report.")
    p_run.add_argument("--no-cache", action="store_true", help="Disable the judge cache.")
    p_run.add_argument(
        "--fail-below-thresholds",
        action="store_true",
        help="Exit 1 if any metric is below (or above, for ceilings) its threshold.",
    )
    p_run.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 if any metric regressed vs --baseline.",
    )
    p_run.add_argument("--baseline", default=None, help="Baseline result JSON for regression check.")

    p_init = sub.add_parser("init", help="Write a starter eval_suite.yaml.")
    p_init.add_argument(
        "path", nargs="?", default="eval_suite.yaml", help="Where to write the starter suite."
    )
    p_init.add_argument(
        "--kind", choices=["rag", "agent"], default="rag", help="Starter template to write."
    )
    p_init.add_argument("--force", action="store_true", help="Overwrite an existing file.")

    p_cmp = sub.add_parser("compare", help="Compare two result JSON files.")
    p_cmp.add_argument("--baseline", required=True, help="Baseline result JSON.")
    p_cmp.add_argument("--candidate", required=True, help="Candidate result JSON.")
    p_cmp.add_argument("--output", "-o", default=None, help="Write comparison HTML.")
    p_cmp.add_argument(
        "--fail-on-regression", action="store_true", help="Exit 1 if any metric regressed."
    )

    args = parser.parse_args(argv)
    console = Console()

    if args.command == "run":
        return _cmd_run(args, console)
    if args.command == "compare":
        return _cmd_compare(args, console)
    if args.command == "init":
        return _cmd_init(args, console)
    return 2


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    suite = load_suite(args.suite)
    result = run_suite(suite, use_cache=not args.no_cache)
    write_json(result, args.output)
    render_console(result, console)
    console.print(f"[dim]Wrote {args.output}[/dim]")
    if args.html:
        render_html(result, args.html)
        console.print(f"[dim]Wrote {args.html}[/dim]")

    exit_code = 0
    if args.fail_below_thresholds and not result.gate.passed:
        console.print("[red]Threshold gate failed.[/red]")
        exit_code = 1
    if args.fail_on_regression:
        if not args.baseline:
            console.print("[red]--fail-on-regression requires --baseline[/red]")
            return 2
        rows = compare_results(load_result(args.baseline), result.to_dict())
        regs = regressions(rows)
        if regs:
            console.print("[red]Regressions vs baseline:[/red]")
            for r in regs:
                console.print(f"  [red]- {r['metric']}: {r['baseline']} -> {r['candidate']}[/red]")
            exit_code = 1
    return exit_code


def _cmd_compare(args: argparse.Namespace, console: Console) -> int:
    baseline = load_result(args.baseline)
    candidate = load_result(args.candidate)
    rows = compare_results(baseline, candidate)
    render_comparison_console(rows, console)
    if args.output:
        render_comparison_html(rows, args.output)
        console.print(f"[dim]Wrote {args.output}[/dim]")
    if args.fail_on_regression and regressions(rows):
        console.print("[red]Regressions detected.[/red]")
        return 1
    return 0


_RAG_STARTER = """\
suite_name: "My RAG suite"

target:
  type: python                 # or: http
  import: "myapp.rag:answer"   # callable: question -> {answer, contexts, doc_ids}
  # endpoint: "http://localhost:8000/api/query"   # for type: http

judge:
  provider: ollama             # ollama (local, free) | openai | anthropic
  model: "llama3.2"

test_cases:
  - id: "example"
    question: "What is your refund policy?"
    expected_answer: "Full refund within 30 days with receipt."
    relevant_doc_ids: ["doc-policy-returns-v3"]   # optional (retrieval only)
    tags: ["policy"]

thresholds:
  retrieval_recall_at_5: 0.80
  faithfulness: 0.85
  relevance: 0.80
  hallucination_rate: 0.15     # ceiling: lower is better
"""

_AGENT_STARTER = """\
suite_name: "My agent suite"

target:
  type: agent                  # scores recorded traces; nothing live is called

judge:
  provider: ollama             # only used if a case sets expected_output
  model: "llama3.2"

test_cases:
  - id: "example"
    question: "What's the status of order 5591?"
    expected_result:
      success_contains: "shipped"
      expected_tool_calls:
        - {name: "get_order", arguments: {order_id: "5591"}}
      optimal_step_count: 2
    agent_trace:
      format: openai            # a list of OpenAI chat-completion messages
      messages:
        - {role: user, content: "What's the status of order 5591?"}
        - role: assistant
          cost_usd: 0.001
          tool_calls:
            - {id: c1, function: {name: get_order, arguments: '{"order_id": "5591"}'}}
        - {role: tool, tool_call_id: c1, content: "Order 5591: shipped."}
        - {role: assistant, content: "Order 5591 has shipped.", cost_usd: 0.001}

thresholds:
  agent_task_completion: 0.80
  agent_tool_recall: 0.80
  agent_cost_usd: 0.01         # ceiling: lower is better
"""


def _cmd_init(args: argparse.Namespace, console: Console) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        console.print(f"[red]{path} already exists (use --force to overwrite).[/red]")
        return 1
    starter = _AGENT_STARTER if args.kind == "agent" else _RAG_STARTER
    path.write_text(starter)
    console.print(f"[green]Wrote {args.kind} starter suite to {path}[/green]")
    console.print(f"Next: [bold]evalkit run {path}[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
