"""Command-line interface for safeval.

Subcommands:
    ``suites``  List available suites and their item counts.
    ``run``     Run one or more suites against a model and write results + reports.
    ``report``  Regenerate reports from a stored run directory or results file.

The CLI is the only component that prints. It requires an explicit model — via
``--model`` or the ``SAFEVAL_MODEL`` environment variable — so a real (billable)
model is never selected by accident; the mock presets (``mock:safe``,
``mock:unsafe``, ``mock:over-refuser``) run fully offline. For the same reason
``--judge`` requires an explicit ``--judge-model``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .datasets import EvalItem, SuiteError, available_suites, load_suite
from .metrics import compute_metrics
from .models import DEFAULT_ANTHROPIC_MODEL, make_model
from .report import generate_reports, render_terminal_summary
from .runner import run_evaluation

# Exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

_DEFAULT_SUITES = "harmful_core,benign_lookalike,jailbreak,injection"


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated option value into a trimmed, non-empty list."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _suite_dir_arg(value: str | None) -> Path | None:
    """Convert an optional ``--suite-dir`` string into a Path."""
    return Path(value) if value else None


def _load_selected(
    names: Sequence[str], suite_dir: Path | None, limit: int | None
) -> list[EvalItem]:
    """Load the requested suites, optionally truncating each to ``limit`` items.

    Args:
        names: Suite names to load.
        suite_dir: Optional suite directory override.
        limit: If given, keep only the first ``limit`` items of each suite.

    Returns:
        The concatenated items.
    """
    items: list[EvalItem] = []
    for name in names:
        suite_items = load_suite(name, suite_dir=suite_dir)
        if limit is not None:
            suite_items = suite_items[:limit]
        items.extend(suite_items)
    return items


def _load_rows(path: Path) -> tuple[list[dict], list[int]]:
    """Load result rows from a JSONL file, tolerating blank and corrupt lines.

    A run that crashes mid-write can leave a truncated final line; the report
    should still work with the surviving rows (that is the whole point of the
    incremental format), so unparsable lines are skipped and reported rather
    than aborting with a traceback.

    Args:
        path: Path to a ``results.jsonl`` file.

    Returns:
        A tuple of (parsed rows, 1-based line numbers that were skipped).

    Raises:
        SuiteError: If the file does not exist.
    """
    if not path.is_file():
        raise SuiteError(f"results file not found: {path}")
    rows: list[dict] = []
    skipped: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped.append(line_no)
    return rows, skipped


def _cmd_suites(args: argparse.Namespace) -> int:
    """Handle ``safeval suites``."""
    suite_dir = _suite_dir_arg(args.suite_dir)
    suites = available_suites(suite_dir)
    if not suites:
        print("No suites found.")
        return EXIT_OK
    print(f"Available suites ({len(suites)}):")
    for name in sorted(suites):
        try:
            count = len(load_suite(name, suite_dir=suite_dir))
        except SuiteError as exc:
            print(f"  {name:<20} (error: {exc})")
            continue
        print(f"  {name:<20} {count:>3} items")
    return EXIT_OK


def _make_progress(stream) -> object:
    """Return a progress callback that writes ``i/N`` lines to ``stream``."""

    def progress(index: int, total: int, item: EvalItem) -> None:
        stream.write(f"\r  running {index}/{total}: {item.suite}/{item.id}".ljust(60))
        stream.flush()
        if index == total:
            stream.write("\n")
            stream.flush()

    return progress


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle ``safeval run``."""
    model_spec = args.model or os.environ.get("SAFEVAL_MODEL")
    if not model_spec:
        print(
            "error: --model is required (no default, to avoid accidental API spend).\n"
            "\n"
            "Run fully offline with a mock model, e.g.:\n"
            "  safeval run --model mock:safe --suites harmful_core,benign_lookalike\n"
            "\n"
            "Mock presets: mock:safe, mock:unsafe, mock:over-refuser\n"
            "Or pass a real Anthropic model id (needs ANTHROPIC_API_KEY), e.g. "
            f"{DEFAULT_ANTHROPIC_MODEL}.\n"
            "The SAFEVAL_MODEL environment variable can stand in for --model.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.judge and not args.judge_model:
        print(
            "error: --judge requires an explicit --judge-model (no default judge, "
            f"to avoid accidental API spend), e.g. --judge-model {DEFAULT_ANTHROPIC_MODEL}.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Fail fast, before any items run, if a key is needed but absent.
    if (not model_spec.startswith("mock:") or args.judge) and not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        print(
            "error: ANTHROPIC_API_KEY is not set. A real model (and --judge) needs a key;\n"
            "set the environment variable, or run an offline mock:* model without --judge.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    suite_dir = _suite_dir_arg(args.suite_dir)
    names = _split_csv(args.suites)
    try:
        model = make_model(model_spec)
        items = _load_selected(names, suite_dir, args.limit)
    except (SuiteError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not items:
        print("error: no items to run (check --suites and --limit).", file=sys.stderr)
        return EXIT_ERROR

    try:
        result = run_evaluation(
            model,
            items,
            out_dir=Path(args.out),
            use_judge=args.judge,
            judge_model=args.judge_model,
            resume=args.resume,
            progress=_make_progress(sys.stderr),
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
    except Exception as exc:  # noqa: BLE001 - surface any run failure to the user
        print(f"error: run failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    metrics = compute_metrics(result.rows)
    print(render_terminal_summary(metrics, result.model))
    reports = generate_reports(result.run_dir, result.rows, result.model, metrics=metrics)
    print()
    print(f"Results: {result.results_path}")
    print(f"Report (markdown): {reports.markdown}")
    print(f"Report (html): {reports.html}")

    if args.judge:
        judge_failures = sum(1 for row in result.rows if row.get("judge_error"))
        if judge_failures:
            first_judge_error = next(
                row["judge_error"] for row in result.rows if row.get("judge_error")
            )
            print(
                f"warning: the judge failed on {judge_failures} item(s); those items "
                f"fall back to rule-based verdicts (first error: {first_judge_error})",
                file=sys.stderr,
            )

    if metrics.total_errors:
        first_error = next(row["error"] for row in result.rows if row.get("error"))
        print(
            f"warning: {metrics.total_errors} of {metrics.total_items} item(s) errored "
            f"(first error: {first_error})",
            file=sys.stderr,
        )
        if metrics.total_errors == metrics.total_items:
            print("error: every item errored; nothing was graded.", file=sys.stderr)
            return EXIT_ERROR
    return EXIT_OK


def _resolve_run_target(target: Path) -> tuple[Path, Path]:
    """Resolve a run target into (results_path, run_dir).

    Args:
        target: A run directory or a ``results.jsonl`` file.

    Returns:
        A tuple of (results file path, run directory).

    Raises:
        SuiteError: If neither a directory with results nor a file is found.
    """
    if target.is_dir():
        return target / "results.jsonl", target
    if target.is_file():
        return target, target.parent
    raise SuiteError(f"run target not found: {target}")


def _cmd_report(args: argparse.Namespace) -> int:
    """Handle ``safeval report``."""
    try:
        results_path, run_dir = _resolve_run_target(Path(args.target))
        rows, skipped = _load_rows(results_path)
    except SuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    for line_no in skipped:
        print(
            f"warning: {results_path.name}:{line_no}: skipping unparsable line "
            "(truncated by a crash?)",
            file=sys.stderr,
        )
    if not rows:
        print("error: no result rows found to report.", file=sys.stderr)
        return EXIT_ERROR

    model = str(rows[0].get("model", "unknown"))
    metrics = compute_metrics(rows)
    print(render_terminal_summary(metrics, model))
    reports = generate_reports(run_dir, rows, model, metrics=metrics)
    print()
    print(f"Report (markdown): {reports.markdown}")
    print(f"Report (html): {reports.html}")
    if args.open:
        webbrowser.open(reports.html.resolve().as_uri())
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="safeval",
        description="A lightweight, reproducible harness for evaluating LLM safety behavior.",
    )
    parser.add_argument("--version", action="version", version=f"safeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_suites = sub.add_parser("suites", help="list available suites and item counts")
    p_suites.add_argument("--suite-dir", default=None, help="directory of .jsonl suites to use")
    p_suites.set_defaults(func=_cmd_suites)

    p_run = sub.add_parser("run", help="run suites against a model")
    p_run.add_argument(
        "--model",
        default=None,
        help="model spec: mock:safe|mock:unsafe|mock:over-refuser or an Anthropic "
        "model id (required; the SAFEVAL_MODEL env var can stand in for the flag)",
    )
    p_run.add_argument("--suites", default=_DEFAULT_SUITES, help="comma-separated suite names")
    p_run.add_argument("--suite-dir", default=None, help="directory of .jsonl suites to use")
    p_run.add_argument("--judge", action="store_true", help="also grade with an LLM judge")
    p_run.add_argument(
        "--judge-model",
        default=None,
        help="judge model id (required with --judge; needs a real key)",
    )
    p_run.add_argument("--out", default="runs", help="base output directory (default: runs)")
    p_run.add_argument("--run-dir", default=None, help="explicit run directory (overrides naming)")
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="continue the most recent run for this model under --out "
        "(or the one given by --run-dir), skipping ids already on disk",
    )
    p_run.add_argument("--limit", type=int, default=None, help="keep only first N items per suite")
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="regenerate reports from a run")
    p_report.add_argument("target", help="a run directory or a results.jsonl file")
    p_report.add_argument("--open", action="store_true", help="open the HTML report in a browser")
    p_report.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
