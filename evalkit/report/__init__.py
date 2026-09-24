"""Report generators: console, JSON, HTML, comparison."""

from __future__ import annotations

from evalkit.report.compare import compare_results, render_comparison_console
from evalkit.report.console import render_console
from evalkit.report.html import render_html
from evalkit.report.json_report import load_result, write_json

__all__ = [
    "render_console",
    "write_json",
    "load_result",
    "render_html",
    "compare_results",
    "render_comparison_console",
]
