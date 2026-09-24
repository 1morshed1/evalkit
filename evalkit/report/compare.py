"""Comparison mode: baseline vs candidate result JSONs (the killer feature)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from evalkit.runner import is_ceiling

EPS = 1e-9


def compare_results(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build one row per metric present in either summary."""
    b_sum = baseline.get("summary", {})
    c_sum = candidate.get("summary", {})
    metrics = list(dict.fromkeys(list(b_sum) + list(c_sum)))
    rows: list[dict[str, Any]] = []
    for m in metrics:
        b = b_sum.get(m)
        c = c_sum.get(m)
        row: dict[str, Any] = {"metric": m, "baseline": b, "candidate": c}
        if b is None or c is None:
            row.update(change_pct=None, better=None, direction="n/a")
            rows.append(row)
            continue
        change_pct = ((c - b) / abs(b) * 100) if abs(b) > EPS else (0.0 if abs(c) < EPS else 100.0)
        raw_up = c > b + EPS
        raw_down = c < b - EPS
        direction = "up" if raw_up else "down" if raw_down else "flat"
        # "better" accounts for ceiling metrics (lower is better).
        if direction == "flat":
            better: bool | None = None
        elif is_ceiling(m):
            better = raw_down
        else:
            better = raw_up
        row.update(change_pct=round(change_pct, 1), better=better, direction=direction)
        rows.append(row)
    return rows


def regressions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows where the candidate got worse (excludes latency/cost by default? no —
    include all; caller decides). A regression = better is False."""
    return [r for r in rows if r.get("better") is False]


def _fmt(metric: str, value: float | None) -> str:
    if value is None:
        return "—"
    if metric.startswith("avg_latency"):
        return f"{value:,.0f}"
    if metric.startswith("avg_cost"):
        return f"${value:.6f}"
    return f"{value:.3f}"


def render_comparison_console(
    rows: list[dict[str, Any]], console: Console | None = None
) -> None:
    console = console or Console()
    table = Table(title="EvalKit — Baseline vs Candidate")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Change", justify="right")
    for r in rows:
        change = ""
        if r["change_pct"] is not None:
            sign = "+" if r["change_pct"] >= 0 else ""
            mark = "✅" if r["better"] else "❌" if r["better"] is False else "➖"
            change = f"{sign}{r['change_pct']}%  {mark}"
        table.add_row(
            r["metric"],
            _fmt(r["metric"], r["baseline"]),
            _fmt(r["metric"], r["candidate"]),
            change,
        )
    console.print(table)


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>EvalKit Comparison</title><style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e6e6;margin:0}
.wrap{max-width:820px;margin:0 auto;padding:32px 20px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid #232a35;text-align:right}
th:first-child,td:first-child{text-align:left}
.up{color:#4ade80}.down{color:#f87171}
</style></head><body><div class="wrap"><h1>Baseline vs Candidate</h1>
<table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Change</th></tr></thead><tbody>
{rows}
</tbody></table></div></body></html>"""


def render_comparison_html(rows: list[dict[str, Any]], path: str | Path) -> None:
    lines = []
    for r in rows:
        change = "—"
        cls = ""
        if r["change_pct"] is not None:
            sign = "+" if r["change_pct"] >= 0 else ""
            mark = "✅" if r["better"] else "❌" if r["better"] is False else "➖"
            change = f"{sign}{r['change_pct']}% {mark}"
            cls = "up" if r["better"] else "down" if r["better"] is False else ""
        lines.append(
            f"<tr><td>{r['metric']}</td><td>{_fmt(r['metric'], r['baseline'])}</td>"
            f"<td>{_fmt(r['metric'], r['candidate'])}</td>"
            f"<td class='{cls}'>{change}</td></tr>"
        )
    Path(path).write_text(_HTML.format(rows="\n".join(lines)))
