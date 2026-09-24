"""EvalKit command-line interface: `evalkit run` and `evalkit compare`."""

from __future__ import annotations

import argparse
import sys

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


if __name__ == "__main__":
    sys.exit(main())
