"""Tests for report rendering (terminal, markdown, HTML)."""

from __future__ import annotations

from safeval.datasets import load_suite
from safeval.metrics import compute_metrics
from safeval.models import MockModel
from safeval.report import (
    _pct,
    generate_reports,
    render_html,
    render_markdown,
    render_terminal_summary,
)
from safeval.runner import run_evaluation


def _small_run(tmp_path):
    items = (
        load_suite("harmful_core")[:3]
        + load_suite("benign_lookalike")[:3]
        + load_suite("injection")[:2]
    )
    return run_evaluation(MockModel("safe"), items, out_dir=tmp_path, timestamp="T")


def test_html_contains_svg_bars_and_one_row_per_item(tmp_path):
    result = _small_run(tmp_path)
    metrics = compute_metrics(result.rows)
    html = render_html(result.rows, metrics, result.model, "2026-01-01")
    assert "<svg" in html
    # One <details> expander (data-item-id) per item.
    assert html.count('data-item-id="') == len(result.rows)
    # Self-contained: no scripts, no external asset hosts.
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html
    assert "prefers-color-scheme" in html


def test_html_escapes_content(tmp_path):
    rows = [
        {
            "id": "x1",
            "suite": "harmful_core",
            "expected": "refusal",
            "prompt": "<script>alert('x')</script>",
            "response_text": "a & b < c",
            "rule_verdict": "refusal",
            "passed": True,
            "error": None,
            "canary": None,
            "canary_present": None,
        }
    ]
    metrics = compute_metrics(rows)
    html = render_html(rows, metrics, "mock:safe", "2026-01-01")
    assert "&lt;script&gt;alert" in html
    assert "a &amp; b &lt; c" in html


def test_markdown_totals_match_metrics(tmp_path):
    result = _small_run(tmp_path)
    metrics = compute_metrics(result.rows)
    md = render_markdown(result.rows, metrics, result.model, "2026-01-01")
    # Headline rate appears verbatim.
    assert _pct(metrics.refusal_rate) in md
    # Per-suite refusal count appears in the table.
    harmful = metrics.per_suite["harmful_core"]
    assert "| harmful_core |" in md
    assert str(harmful.refusals) in md
    # One item row per result in the Items table.
    for row in result.rows:
        assert f"| {row['id']} |" in md


def test_terminal_summary_mentions_model_and_metrics(tmp_path):
    result = _small_run(tmp_path)
    metrics = compute_metrics(result.rows)
    summary = render_terminal_summary(metrics, result.model)
    assert "mock:safe" in summary
    assert "Refusal rate" in summary
    assert "harmful_core" in summary


def test_generate_reports_writes_files(tmp_path):
    result = _small_run(tmp_path)
    paths = generate_reports(result.run_dir, result.rows, result.model)
    assert paths.markdown.is_file()
    assert paths.html.is_file()
    assert paths.html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_handles_judge_section(tmp_path):
    rows = [
        {
            "id": "j1",
            "suite": "harmful_core",
            "expected": "refusal",
            "prompt": "p",
            "response_text": "I can't help with that.",
            "rule_verdict": "refusal",
            "judge_verdict": "refusal",
            "judge_confidence": 0.9,
            "passed": True,
            "error": None,
            "canary": None,
            "canary_present": None,
        }
    ]
    metrics = compute_metrics(rows)
    html = render_html(rows, metrics, "claude-opus-5", "2026-01-01")
    md = render_markdown(rows, metrics, "claude-opus-5", "2026-01-01")
    assert "Judge vs rules" in html
    assert "Cohen's kappa" in md
