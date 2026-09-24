"""Self-contained HTML report (inline CSS, no CDN, renders offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment

from evalkit.runner import is_ceiling
from evalkit.schemas import RunResult

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvalKit Report — {{ r.suite_name }}</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #0f1115; color: #e6e6e6; }
  .wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: #9aa4b2; font-size: 13px; margin-bottom: 24px; }
  .gate { display: inline-block; padding: 4px 12px; border-radius: 999px;
          font-weight: 600; font-size: 13px; }
  .pass { background: #10331f; color: #4ade80; }
  .fail { background: #3a1416; color: #f87171; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0 32px; font-size: 14px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #232a35; }
  th { color: #9aa4b2; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .tag-pass { color: #4ade80; font-weight: 600; }
  .tag-fail { color: #f87171; font-weight: 600; }
  .bar { height: 8px; border-radius: 4px; background: #232a35; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: #3b82f6; }
  details { border: 1px solid #232a35; border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
  summary { cursor: pointer; font-weight: 600; }
  .claim { color: #f87171; font-size: 13px; margin: 2px 0; }
  code { background: #1a2030; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>EvalKit Report — {{ r.suite_name }}</h1>
  <div class="sub">
    Run {{ r.run_id }} · Judge {{ r.judge.provider }}/{{ r.judge.model }} ·
    {{ r.cases|length }} cases ·
    <span class="gate {{ 'pass' if r.gate.passed else 'fail' }}">
      Gate {{ 'PASSED' if r.gate.passed else 'FAILED' }}</span>
  </div>

  <h2>Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th class="num">Value</th><th>Level</th>
      <th class="num">Threshold</th><th>Status</th></tr></thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row.name }}</td>
        <td class="num">{{ row.display }}</td>
        <td>{% if row.is_ratio %}<div class="bar"><i style="width: {{ row.pct }}%"></i></div>{% endif %}</td>
        <td class="num">{{ row.threshold }}</td>
        <td>{% if row.status == 'PASS' %}<span class="tag-pass">PASS</span>
            {% elif row.status == 'FAIL' %}<span class="tag-fail">FAIL</span>
            {% else %}—{% endif %}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  {% if not r.gate.passed %}
  <h2>Gate violations</h2>
  <ul>{% for v in r.gate.violations %}<li class="tag-fail">{{ v }}</li>{% endfor %}</ul>
  {% endif %}

  <h2>Cases</h2>
  {% for c in r.cases %}
  <details>
    <summary>{{ c.id }}
      {%- if c.metrics.faithfulness %} · faithfulness {{ '%.2f'|format(c.metrics.faithfulness.score) }}{% endif %}
      {%- if c.metrics.relevance %} · relevance {{ '%.2f'|format(c.metrics.relevance.score) }}{% endif %}</summary>
    <p><strong>Answer:</strong> {{ c.prediction.answer }}</p>
    {% set unsupported = c.metrics.get('faithfulness', {}).get('unsupported_claims') %}
    {% if unsupported %}
      <p><strong>Potential hallucinations:</strong></p>
      {% for claim in unsupported %}<div class="claim">✗ {{ claim }}</div>{% endfor %}
    {% endif %}
    {% if c.skipped %}<p class="sub">Skipped: {{ c.skipped|join(', ') }}</p>{% endif %}
  </details>
  {% endfor %}
</div>
</body>
</html>
"""


def _build_rows(result: RunResult) -> list[dict[str, Any]]:
    from evalkit.report.console import _ordered

    rows: list[dict[str, Any]] = []
    for name in _ordered(result.summary):
        value = result.summary[name]
        limit = result.thresholds.get(name)
        is_ratio = 0.0 <= value <= 1.0 and not name.startswith("avg_") and "cost" not in name
        status = "NA"
        if limit is not None:
            ok = value <= limit if is_ceiling(name) else value >= limit
            status = "PASS" if ok else "FAIL"
        if name.startswith("avg_latency"):
            display = f"{value:,.0f} ms"
        elif "cost" in name:
            display = f"${value:.6f}"
        else:
            display = f"{value:.3f}"
        rows.append(
            {
                "name": name,
                "display": display,
                "is_ratio": is_ratio,
                "pct": round(value * 100, 1) if is_ratio else 0,
                "threshold": "" if limit is None else limit,
                "status": status,
            }
        )
    return rows


def render_html(result: RunResult, path: str | Path) -> None:
    env = Environment(autoescape=True)
    template = env.from_string(_TEMPLATE)
    html = template.render(r=result.to_dict(), rows=_build_rows(result))
    Path(path).write_text(html)
