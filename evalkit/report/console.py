"""Rich console rendering of a RunResult."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from evalkit.runner import is_ceiling
from evalkit.schemas import RunResult

# Metric display order: put the headline RAG metrics first.
PREFERRED_ORDER = [
    "retrieval_recall_at_5",
    "retrieval_precision_at_5",
    "retrieval_mrr",
    "retrieval_ndcg_at_5",
    "retrieval_hit_rate",
    "faithfulness",
    "relevance",
    "hallucination_rate",
    "avg_latency_ms",
    "avg_cost_usd",
]


def _ordered(summary: dict[str, float]) -> list[str]:
    keys = list(summary.keys())
    front = [k for k in PREFERRED_ORDER if k in summary]
    rest = sorted(k for k in keys if k not in front)
    return front + rest


def render_console(result: RunResult, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"EvalKit — {result.suite_name}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="bold")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for name in _ordered(result.summary):
        value = result.summary[name]
        limit = result.thresholds.get(name)
        status = "—"
        if limit is not None:
            ok = value <= limit if is_ceiling(name) else value >= limit
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        limit_str = "" if limit is None else f"{limit}"
        table.add_row(name, _fmt(name, value), limit_str, status)

    console.print(table)
    console.print(
        f"Cases: {len(result.cases)}  |  Judge: {result.judge['provider']}/{result.judge['model']}"
    )
    if result.gate.passed:
        console.print("[green]Gate: PASSED[/green]")
    else:
        console.print("[red]Gate: FAILED[/red]")
        for v in result.gate.violations:
            console.print(f"  [red]- {v}[/red]")


def _fmt(name: str, value: float) -> str:
    if name.startswith("avg_latency"):
        return f"{value:,.0f}"
    if name.startswith("avg_cost"):
        return f"${value:.6f}"
    return f"{value:.3f}"
