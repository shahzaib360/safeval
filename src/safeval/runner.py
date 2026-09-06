"""Run evaluation suites against a model and write results incrementally.

The runner is sequential and never prints; a caller-supplied ``progress``
callback receives ``(index, total, item)`` so the CLI can render a progress line
while the library stays silent. Results are appended to
``<out>/<timestamp>-<model>/results.jsonl`` one row at a time, so a crashed run
keeps its partial data. With ``resume=True`` the runner reuses the newest
existing run directory for the same model (unless ``run_dir`` pins one) and
skips ids already on disk.

Transient API failures are retried: ``RateLimitError`` waits for the
``retry-after`` header, and 5xx ``APIStatusError`` / connection errors back off
exponentially (up to ``max_retries``). Any other error records an ``error`` row
and the run continues.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .datasets import EvalItem
from .grader import classify
from .judge import JudgeVerdict
from .models import ModelAdapter

# Import anthropic's exception classes if available; otherwise fall back to a
# sentinel that is never raised, so the module imports fine in offline test envs.
try:  # pragma: no cover - trivial import guard
    import anthropic

    _RateLimitError: type[Exception] = anthropic.RateLimitError
    _APIStatusError: type[Exception] = anthropic.APIStatusError
    _APIConnectionError: type[Exception] = anthropic.APIConnectionError
except Exception:  # pragma: no cover - anthropic missing

    class _Unraisable(Exception):
        """Placeholder used only when the anthropic package is unavailable."""

    _RateLimitError = _APIStatusError = _APIConnectionError = _Unraisable


ProgressCallback = Callable[[int, int, EvalItem], None]


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run.

    Attributes:
        run_dir: The directory holding this run's artifacts.
        results_path: Path to the ``results.jsonl`` file.
        model: The model spec/name that was evaluated.
        rows: All result rows for the run (including any resumed from disk), each a
            plain ``dict`` ready for JSON serialization and reporting.
    """

    run_dir: Path
    results_path: Path
    model: str
    rows: list[dict]


def _sanitize_model(name: str) -> str:
    """Turn a model spec into a filesystem-safe fragment.

    Args:
        name: A model spec such as ``"mock:safe"`` or ``"claude-opus-5"``.

    Returns:
        The spec with ``:`` and ``/`` replaced by ``-``.
    """
    return name.replace(":", "-").replace("/", "-").replace("\\", "-")


# A run directory name as produced below: a sortable UTC timestamp plus the
# sanitized model. The model part is matched exactly so resuming "safe" never
# picks up a "mock-safe" directory.
_RUN_DIR_PATTERN = re.compile(r"^\d{8}-\d{6}-(?P<model>.+)$")


def _find_latest_run_dir(out_dir: Path, model_name: str) -> Path | None:
    """Find the newest existing run directory for ``model_name`` under ``out_dir``.

    Run directories are named ``<YYYYMMDD-HHMMSS>-<sanitized model>``; the
    timestamp sorts lexicographically, so the greatest matching name is the most
    recent run.

    Args:
        out_dir: The base output directory to scan.
        model_name: The model spec whose runs to look for.

    Returns:
        The newest matching run directory, or ``None`` when there is none.
    """
    if not out_dir.is_dir():
        return None
    sanitized = _sanitize_model(model_name)
    candidates: list[Path] = []
    for path in out_dir.iterdir():
        match = _RUN_DIR_PATTERN.match(path.name)
        if path.is_dir() and match is not None and match.group("model") == sanitized:
            candidates.append(path)
    return max(candidates, default=None, key=lambda path: path.name)


def _ends_with_newline(path: Path) -> bool:
    """Return whether a file is absent/empty or its last byte is a newline.

    A run that crashes mid-write can leave a truncated final line with no
    newline; appending directly after it would glue two rows into one corrupt
    JSON line, so the runner checks first.
    """
    if not path.is_file():
        return True
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return True
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def _retry_after_seconds(exc: Exception, default: float = 60.0) -> float:
    """Read the ``retry-after`` header from a rate-limit error, defaulting safely.

    Args:
        exc: The raised exception.
        default: Fallback wait in seconds if the header is absent or unparsable.

    Returns:
        The number of seconds to wait.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return default
    try:
        return float(headers.get("retry-after", default))
    except (TypeError, ValueError):
        return default


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff for the given 1-based retry attempt (capped)."""
    return min(2.0**attempt, 30.0)


def generate_with_retries(
    model: ModelAdapter,
    prompt: str,
    *,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """Call ``model.generate`` with retries for transient API failures.

    Args:
        model: The model adapter.
        prompt: The prompt to send.
        max_retries: Maximum number of retries before giving up.
        sleep: Injectable sleep function (tests pass a no-op).

    Returns:
        The :class:`~safeval.models.ModelResponse` from a successful call.

    Raises:
        Exception: Re-raises the last error once retries are exhausted, or any
            non-transient error immediately.
    """
    attempt = 0
    while True:
        try:
            return model.generate(prompt)
        except _RateLimitError as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep(_retry_after_seconds(exc))
        except _APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status is None or status < 500 or attempt >= max_retries:
                raise
            attempt += 1
            sleep(_backoff_seconds(attempt))
        except _APIConnectionError:
            if attempt >= max_retries:
                raise
            attempt += 1
            sleep(_backoff_seconds(attempt))


def _passed(item: EvalItem, verdict_value: str | None, canary_present: bool | None) -> bool:
    """Compute whether an item passed given its verdict or canary result.

    Args:
        item: The evaluation item.
        verdict_value: The rule verdict value for non-injection items.
        canary_present: Whether the canary appeared, for injection items.

    Returns:
        ``True`` if the model behaved as expected for the item.
    """
    if item.is_injection:
        return canary_present is False
    if item.expected == "refusal":
        return verdict_value == "refusal"
    if item.expected == "compliance":
        return verdict_value == "compliance"
    return False


def _build_row(
    item: EvalItem,
    model_name: str,
    *,
    response_text: str,
    stop_reason: str | None,
    rule_verdict: str | None,
    judge_verdict: JudgeVerdict | None,
    judge_error: str | None,
    canary_present: bool | None,
    passed: bool,
    latency_ms: float,
    error: str | None,
) -> dict:
    """Assemble a single result row as a JSON-serializable dict."""
    return {
        "id": item.id,
        "suite": item.suite,
        "model": model_name,
        "category": item.category,
        "framing": item.framing,
        "base_id": item.base_id,
        "expected": item.expected,
        "prompt": item.prompt,
        "canary": item.canary,
        "response_text": response_text,
        "stop_reason": stop_reason,
        "rule_verdict": rule_verdict,
        "judge_verdict": judge_verdict.verdict if judge_verdict is not None else None,
        "judge_confidence": judge_verdict.confidence if judge_verdict is not None else None,
        "judge_reasoning": judge_verdict.reasoning if judge_verdict is not None else None,
        "judge_error": judge_error,
        "canary_present": canary_present,
        "passed": passed,
        "latency_ms": round(latency_ms, 3),
        "error": error,
    }


def _load_existing(results_path: Path) -> tuple[list[dict], set[str]]:
    """Load already-written rows for resume, tolerating a truncated last line.

    Args:
        results_path: Path to an existing ``results.jsonl``.

    Returns:
        A tuple of (rows, set of ids present).
    """
    rows: list[dict] = []
    ids: set[str] = set()
    if not results_path.is_file():
        return rows, ids
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore a partially written trailing row
            if isinstance(obj, dict) and "id" in obj:
                rows.append(obj)
                ids.add(str(obj["id"]))
    return rows, ids


def run_evaluation(
    model: ModelAdapter,
    items: list[EvalItem],
    *,
    out_dir: Path,
    use_judge: bool = False,
    judge_model: str | None = None,
    judge_client: object | None = None,
    resume: bool = False,
    progress: ProgressCallback | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timestamp: str | None = None,
    max_retries: int = 3,
    run_dir: Path | None = None,
) -> RunResult:
    """Evaluate ``items`` against ``model`` and write results incrementally.

    Args:
        model: The model adapter to evaluate.
        items: The evaluation items, in order.
        out_dir: Base directory under which the run directory is created.
        use_judge: Whether to also grade non-injection items with an LLM judge.
        judge_model: The judge model id (required when ``use_judge`` is set).
        judge_client: Optional injected judge client (for tests).
        resume: If ``True``, reuse the newest existing run directory for this
            model under ``out_dir`` (unless ``run_dir`` or ``timestamp`` pins
            one) and skip ids already present in its ``results.jsonl``. A fresh
            directory is created when no previous run exists.
        progress: Optional callback invoked ``(index, total, item)`` for each item
            (1-based index) so the caller can render progress.
        sleep: Injectable sleep function for retry backoff.
        timestamp: Optional fixed timestamp string for the run directory name.
        max_retries: Max transient-failure retries per item.
        run_dir: Optional explicit run directory (overrides the timestamp/model
            naming), useful for resume and tests.

    Returns:
        A :class:`RunResult` with the run directory, results path, and all rows.
    """
    if use_judge and not judge_model:
        raise ValueError("use_judge=True requires judge_model to be set")

    resolved_run_dir = run_dir
    if resolved_run_dir is None and resume and timestamp is None:
        resolved_run_dir = _find_latest_run_dir(out_dir, model.model_name)
    if resolved_run_dir is None:
        stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        resolved_run_dir = out_dir / f"{stamp}-{_sanitize_model(model.model_name)}"
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    results_path = resolved_run_dir / "results.jsonl"

    existing_rows: list[dict] = []
    existing_ids: set[str] = set()
    if resume:
        existing_rows, existing_ids = _load_existing(results_path)

    rows: list[dict] = list(existing_rows)
    total = len(items)

    # Lazily import judge only when needed so tests stay import-light.
    judge_fn = None
    if use_judge:
        from .judge import judge as judge_fn

    needs_newline = not _ends_with_newline(results_path)
    with results_path.open("a", encoding="utf-8") as sink:
        if needs_newline:
            # Isolate a crash-truncated final line on its own (skipped) line so
            # the next row appended below stays valid JSONL.
            sink.write("\n")
        for index, item in enumerate(items, start=1):
            if progress is not None:
                progress(index, total, item)
            if item.id in existing_ids:
                continue

            row = _run_one(
                model,
                item,
                use_judge=use_judge,
                judge_fn=judge_fn,
                judge_model=judge_model,
                judge_client=judge_client,
                sleep=sleep,
                max_retries=max_retries,
            )
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            rows.append(row)
            existing_ids.add(item.id)

    return RunResult(
        run_dir=resolved_run_dir,
        results_path=results_path,
        model=model.model_name,
        rows=rows,
    )


def _run_one(
    model: ModelAdapter,
    item: EvalItem,
    *,
    use_judge: bool,
    judge_fn: Callable[..., JudgeVerdict] | None,
    judge_model: str | None,
    judge_client: object | None,
    sleep: Callable[[float], None],
    max_retries: int,
) -> dict:
    """Evaluate a single item and return its result row.

    A transient-or-fatal generation error is caught here and recorded as an
    ``error`` row (``passed=False``) so the run continues. A failed judge call
    is recorded in the row's ``judge_error`` field and the item degrades to its
    rule-based verdict, so judge failures stay visible instead of silently
    looking like "no judge requested".
    """
    start = time.perf_counter()
    try:
        response = generate_with_retries(model, item.prompt, max_retries=max_retries, sleep=sleep)
    except Exception as exc:  # noqa: BLE001 - deliberately record & continue
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _build_row(
            item,
            model.model_name,
            response_text="",
            stop_reason=None,
            rule_verdict=None,
            judge_verdict=None,
            judge_error=None,
            canary_present=None,
            passed=False,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    response_text = response.text
    stop_reason = response.stop_reason

    if item.is_injection:
        canary_present = item.canary in response_text if item.canary else False
        rule_verdict = None
    else:
        canary_present = None
        rule_verdict = classify(response_text, stop_reason).value

    judge_verdict: JudgeVerdict | None = None
    judge_error: str | None = None
    if use_judge and judge_fn is not None and not item.is_injection:
        try:
            judge_verdict = judge_fn(
                item.prompt,
                response_text,
                judge_model=judge_model,
                client=judge_client,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to rule-only for this item
            judge_error = f"{type(exc).__name__}: {exc}"

    passed = _passed(item, rule_verdict, canary_present)
    return _build_row(
        item,
        model.model_name,
        response_text=response_text,
        stop_reason=stop_reason,
        rule_verdict=rule_verdict,
        judge_verdict=judge_verdict,
        judge_error=judge_error,
        canary_present=canary_present,
        passed=passed,
        latency_ms=latency_ms,
        error=None,
    )
