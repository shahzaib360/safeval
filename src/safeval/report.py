"""Render a run to a terminal summary, Markdown, and a self-contained HTML report.

The HTML report inlines all CSS and draws each metric as an inline SVG horizontal
bar. There are no external assets and no JavaScript, so the report renders
correctly when opened directly as a local file, and it adapts to light or dark
mode via ``prefers-color-scheme``.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .metrics import Metrics, SuiteMetrics, compute_metrics

# Order suites are displayed in; unknown suites are appended in insertion order.
_SUITE_ORDER = ("harmful_core", "benign_lookalike", "jailbreak", "injection")


@dataclass(frozen=True)
class ReportPaths:
    """Filesystem paths written by :func:`generate_reports`.

    Attributes:
        markdown: Path to ``report.md``.
        html: Path to ``report.html``.
    """

    markdown: Path
    html: Path


def _pct(value: float | None) -> str:
    """Format a ratio in ``[0, 1]`` as a percentage string, or ``n/a`` for None."""
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _ordered_suites(per_suite: dict[str, SuiteMetrics]) -> list[SuiteMetrics]:
    """Return suite metrics in the conventional display order."""
    ordered = [per_suite[name] for name in _SUITE_ORDER if name in per_suite]
    extras = [sm for name, sm in per_suite.items() if name not in _SUITE_ORDER]
    return ordered + extras


def _named_metric_rows(metrics: Metrics) -> list[tuple[str, float | None, str]]:
    """Return (label, value, note) triples for the headline metrics."""
    return [
        ("Refusal rate (harmful)", metrics.refusal_rate, "higher is safer"),
        ("Over-refusal rate (benign)", metrics.over_refusal_rate, "lower is better"),
        ("Jailbreak refusal rate", metrics.jailbreak_refusal_rate, "higher is safer"),
        ("Jailbreak uplift", metrics.jailbreak_uplift, "lower is better"),
        ("Injection resistance", metrics.injection_resistance, "higher is better"),
    ]


# --------------------------------------------------------------------------- #
# Terminal summary
# --------------------------------------------------------------------------- #


def render_terminal_summary(metrics: Metrics, model: str) -> str:
    """Render a plain-text summary table for the terminal.

    Args:
        metrics: The computed metrics.
        model: The evaluated model name.

    Returns:
        A multi-line string suitable for printing.
    """
    lines: list[str] = []
    lines.append(f"safeval summary for model: {model}")
    lines.append(f"items: {metrics.total_items}   errors: {metrics.total_errors}")
    lines.append("")
    lines.append("Headline metrics")
    lines.append("-" * 48)
    for label, value, note in _named_metric_rows(metrics):
        lines.append(f"  {label:<30} {_pct(value):>7}  ({note})")
    lines.append("")
    lines.append("Per-suite")
    lines.append("-" * 48)
    header = f"  {'suite':<20}{'pass':>6}{'graded':>8}{'err':>5}"
    lines.append(header)
    for suite in _ordered_suites(metrics.per_suite):
        lines.append(
            f"  {suite.suite:<20}{_pct(suite.pass_rate):>6}{suite.graded:>8}{suite.errors:>5}"
        )
    if metrics.judge_compared:
        lines.append("")
        lines.append(
            f"Judge vs rules: agreement {_pct(metrics.judge_agreement)}, "
            f"kappa {metrics.cohen_kappa:.3f} over {metrics.judge_compared} items"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def render_markdown(rows: Sequence[dict], metrics: Metrics, model: str, generated_at: str) -> str:
    """Render a Markdown report whose totals match the computed metrics.

    Args:
        rows: The result rows.
        metrics: The computed metrics.
        model: The evaluated model name.
        generated_at: An ISO-ish timestamp string for the header.

    Returns:
        The Markdown document as a string.
    """
    out: list[str] = []
    out.append("# safeval report")
    out.append("")
    out.append(f"- **Model:** `{model}`")
    out.append(f"- **Generated:** {generated_at}")
    out.append(f"- **Items:** {metrics.total_items}")
    out.append(f"- **Errors:** {metrics.total_errors}")
    out.append("")
    out.append("## Headline metrics")
    out.append("")
    out.append("| Metric | Value | Direction |")
    out.append("| --- | --- | --- |")
    for label, value, note in _named_metric_rows(metrics):
        out.append(f"| {label} | {_pct(value)} | {note} |")
    out.append("")
    out.append("## Per-suite")
    out.append("")
    out.append(
        "| Suite | Pass rate | Passed | Graded | Errors | Refusals | Compliances | Partials |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for suite in _ordered_suites(metrics.per_suite):
        out.append(
            f"| {suite.suite} | {_pct(suite.pass_rate)} | {suite.passed} | {suite.graded} "
            f"| {suite.errors} | {suite.refusals} | {suite.compliances} | {suite.partials} |"
        )
    out.append("")
    if metrics.judge_compared:
        out.append("## Judge vs rules")
        out.append("")
        out.append(f"- Compared items: {metrics.judge_compared}")
        out.append(f"- Agreement: {_pct(metrics.judge_agreement)}")
        out.append(f"- Cohen's kappa: {metrics.cohen_kappa:.3f}")
        out.append("")
    out.append("## Items")
    out.append("")
    out.append("| ID | Suite | Expected | Rule verdict | Judge | Passed |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        passed = "yes" if row.get("passed") else "no"
        judge = row.get("judge_verdict") or "-"
        verdict = row.get("rule_verdict") or ("canary" if row.get("canary") else "-")
        out.append(
            f"| {row.get('id')} | {row.get('suite')} | {row.get('expected')} "
            f"| {verdict} | {judge} | {passed} |"
        )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_HTML_STYLE = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --card: #f6f7f9;
  --border: #e2e5e9; --accent: #2f6feb; --track: #e2e5e9;
  --good: #1f9d57; --warn: #d98324; --bad: #d64545;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e8eaed; --muted: #9aa0a6; --card: #1e2126;
    --border: #2a2e35; --accent: #6ea8ff; --track: #2a2e35;
    --good: #4cc98a; --warn: #e0a458; --bad: #f07171;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
h1, h2 { line-height: 1.2; }
h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.75rem; border-bottom: 1px solid var(--border);
  padding-bottom: 0.3rem; }
.meta { color: var(--muted); margin: 0 0 1.5rem; }
.meta code { color: var(--fg); }
.cards { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 0.75rem 1rem; min-width: 150px; }
.card .label { color: var(--muted); font-size: 0.8rem; }
.card .value { font-size: 1.5rem; font-weight: 600; }
.bars { display: grid; gap: 0.4rem; margin: 1rem 0; }
.bar-row { display: grid; grid-template-columns: 220px 1fr; align-items: center; gap: 0.75rem; }
.bar-row .name { color: var(--muted); font-size: 0.9rem; }
svg .track { fill: var(--track); }
svg .value { fill: var(--accent); }
svg .value.warn { fill: var(--warn); }
svg text { fill: var(--fg); font: 12px system-ui, sans-serif; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; }
.pill { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px;
  font-size: 0.8rem; border: 1px solid var(--border); }
.pill.pass { color: var(--good); border-color: var(--good); }
.pill.fail { color: var(--bad); border-color: var(--bad); }
details.item { border: 1px solid var(--border); border-radius: 8px; margin: 0.4rem 0;
  padding: 0.4rem 0.75rem; background: var(--card); }
details.item summary { cursor: pointer; font-weight: 500; }
details.item summary .tag { color: var(--muted); font-weight: 400; }
.kv { margin: 0.5rem 0; }
.kv .k { color: var(--muted); font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: 0.03em; }
pre { white-space: pre-wrap; word-wrap: break-word; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; padding: 0.6rem; overflow-x: auto;
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
.footer { color: var(--muted); font-size: 0.8rem; margin-top: 2rem; }
""".strip()


def _svg_bar(value: float | None, *, warn: bool = False) -> str:
    """Return an inline SVG horizontal bar for a ratio value.

    Args:
        value: Ratio in ``[0, 1]`` (clamped for the bar width); ``None`` renders
            an "n/a" bar.
        warn: If ``True``, style the value bar with the warning color (used where
            a lower value is better).

    Returns:
        An ``<svg>`` element as a string.
    """
    width = 320
    height = 22
    if value is None:
        return (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="not available">'
            f'<rect class="track" x="0" y="4" rx="4" width="{width}" height="14"></rect>'
            f'<text x="6" y="16">n/a</text></svg>'
        )
    clamped = max(0.0, min(1.0, value))
    fill_width = round(clamped * width, 1)
    cls = "value warn" if warn else "value"
    label = f"{value * 100:.1f}%"
    text_x = fill_width + 6 if fill_width < width - 48 else fill_width - 46
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{label}">'
        f'<rect class="track" x="0" y="4" rx="4" width="{width}" height="14"></rect>'
        f'<rect class="{cls}" x="0" y="4" rx="4" width="{fill_width}" height="14"></rect>'
        f'<text x="{text_x}" y="16">{label}</text></svg>'
    )


def _bar_row(name: str, value: float | None, *, warn: bool = False) -> str:
    """Render a labelled bar row (name + SVG)."""
    return (
        f'<div class="bar-row"><span class="name">{html.escape(name)}</span>'
        f"{_svg_bar(value, warn=warn)}</div>"
    )


def _item_details(row: dict) -> str:
    """Render one item as a ``<details>`` expander."""
    passed = bool(row.get("passed"))
    pill = (
        '<span class="pill pass">pass</span>' if passed else '<span class="pill fail">fail</span>'
    )
    verdict = row.get("rule_verdict") or ("canary check" if row.get("canary") else "-")
    judge = row.get("judge_verdict")
    error = row.get("error")

    summary = (
        f'<summary data-item-id="{html.escape(str(row.get("id")))}">'
        f"{html.escape(str(row.get('id')))} {pill} "
        f'<span class="tag">[{html.escape(str(row.get("suite")))}] '
        f"expected {html.escape(str(row.get('expected')))}</span></summary>"
    )

    parts: list[str] = [f'<details class="item">{summary}']
    parts.append('<div class="kv"><div class="k">Prompt</div>')
    parts.append(f"<pre>{html.escape(str(row.get('prompt', '')))}</pre></div>")
    parts.append('<div class="kv"><div class="k">Response</div>')
    parts.append(f"<pre>{html.escape(str(row.get('response_text', '')))}</pre></div>")

    verdict_bits = [f"rule: {html.escape(str(verdict))}"]
    if judge:
        conf = row.get("judge_confidence")
        conf_txt = f" ({conf:.2f})" if isinstance(conf, (int, float)) else ""
        verdict_bits.append(f"judge: {html.escape(str(judge))}{conf_txt}")
    if row.get("judge_error"):
        verdict_bits.append(f"judge error: {html.escape(str(row.get('judge_error')))}")
    if row.get("canary") is not None:
        present = row.get("canary_present")
        verdict_bits.append(f"canary present: {present}")
    if row.get("stop_reason"):
        verdict_bits.append(f"stop_reason: {html.escape(str(row.get('stop_reason')))}")
    if error:
        verdict_bits.append(f"error: {html.escape(str(error))}")
    parts.append(
        f'<div class="kv"><div class="k">Verdicts</div><div>{" · ".join(verdict_bits)}</div></div>'
    )
    parts.append("</details>")
    return "".join(parts)


def render_html(rows: Sequence[dict], metrics: Metrics, model: str, generated_at: str) -> str:
    """Render the self-contained HTML report.

    Args:
        rows: The result rows.
        metrics: The computed metrics.
        model: The evaluated model name.
        generated_at: Timestamp string for the header.

    Returns:
        A complete HTML document as a string.
    """
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>safeval report — {html.escape(model)}</title>")
    parts.append(f"<style>{_HTML_STYLE}</style></head><body>")

    parts.append("<h1>safeval report</h1>")
    parts.append(
        f'<p class="meta">Model <code>{html.escape(model)}</code> · '
        f"generated {html.escape(generated_at)} · "
        f"{metrics.total_items} items · {metrics.total_errors} errors</p>"
    )

    # Metric cards.
    parts.append('<div class="cards">')
    for label, value, _note in _named_metric_rows(metrics):
        parts.append(
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{_pct(value)}</div></div>'
        )
    parts.append("</div>")

    # Metric bars.
    parts.append("<h2>Metric bars</h2>")
    parts.append('<div class="bars">')
    parts.append(_bar_row("Refusal rate (harmful)", metrics.refusal_rate))
    parts.append(_bar_row("Over-refusal rate (benign)", metrics.over_refusal_rate, warn=True))
    parts.append(_bar_row("Jailbreak refusal rate", metrics.jailbreak_refusal_rate))
    parts.append(_bar_row("Jailbreak uplift", metrics.jailbreak_uplift, warn=True))
    parts.append(_bar_row("Injection resistance", metrics.injection_resistance))
    parts.append("</div>")

    # Per-suite table.
    parts.append("<h2>Per-suite</h2>")
    parts.append(
        "<table><thead><tr><th>Suite</th><th>Pass rate</th><th>Passed</th>"
        "<th>Graded</th><th>Errors</th><th>Refusals</th><th>Compliances</th>"
        "<th>Partials</th></tr></thead><tbody>"
    )
    for suite in _ordered_suites(metrics.per_suite):
        parts.append(
            f"<tr><td>{html.escape(suite.suite)}</td><td>{_pct(suite.pass_rate)}</td>"
            f"<td>{suite.passed}</td><td>{suite.graded}</td><td>{suite.errors}</td>"
            f"<td>{suite.refusals}</td><td>{suite.compliances}</td><td>{suite.partials}</td></tr>"
        )
    parts.append("</tbody></table>")

    if metrics.judge_compared:
        parts.append("<h2>Judge vs rules</h2>")
        kappa = f"{metrics.cohen_kappa:.3f}" if metrics.cohen_kappa is not None else "n/a"
        parts.append(
            f"<p>Compared {metrics.judge_compared} items · agreement "
            f"{_pct(metrics.judge_agreement)} · Cohen's kappa {kappa}.</p>"
        )

    # Per-item details.
    parts.append("<h2>Items</h2>")
    for row in rows:
        parts.append(_item_details(row))

    parts.append(
        '<p class="footer">Generated by safeval · rule-based grading is noisy; '
        "use the optional LLM judge for higher-fidelity verdicts.</p>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    """Return the current UTC time as a compact ISO-ish string."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def generate_reports(
    run_dir: Path,
    rows: Sequence[dict],
    model: str,
    *,
    metrics: Metrics | None = None,
    generated_at: str | None = None,
) -> ReportPaths:
    """Write ``report.md`` and ``report.html`` into ``run_dir``.

    Args:
        run_dir: Directory to write the reports into (created if needed).
        rows: The result rows.
        model: The evaluated model name.
        metrics: Precomputed metrics; computed from ``rows`` when omitted.
        generated_at: Timestamp string; defaults to the current UTC time.

    Returns:
        The paths of the written reports.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    computed = metrics if metrics is not None else compute_metrics(rows)
    when = generated_at or _now_iso()

    markdown_path = run_dir / "report.md"
    html_path = run_dir / "report.html"
    markdown_path.write_text(render_markdown(rows, computed, model, when), encoding="utf-8")
    html_path.write_text(render_html(rows, computed, model, when), encoding="utf-8")
    return ReportPaths(markdown=markdown_path, html=html_path)
