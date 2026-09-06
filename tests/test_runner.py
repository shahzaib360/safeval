"""End-to-end runner tests using offline mock models."""

from __future__ import annotations

import json
import types

import pytest

from safeval import runner
from safeval.datasets import load_suite
from safeval.metrics import compute_metrics
from safeval.models import MockModel
from safeval.runner import generate_with_retries, run_evaluation


class CountingModel:
    """Wraps a MockModel and counts generate calls (to verify resume)."""

    def __init__(self, preset: str) -> None:
        self._inner = MockModel(preset)
        self.model_name = self._inner.model_name
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        return self._inner.generate(prompt)


class RaisingModel:
    """Always raises the given exception (to produce error rows)."""

    def __init__(self, exc: Exception) -> None:
        self.model_name = "raising"
        self._exc = exc

    def generate(self, prompt: str):
        raise self._exc


def test_safe_model_passes_harmful_suite(tmp_path):
    items = load_suite("harmful_core")
    result = run_evaluation(MockModel("safe"), items, out_dir=tmp_path, timestamp="T")
    metrics = compute_metrics(result.rows)
    assert metrics.refusal_rate == pytest.approx(1.0)
    assert metrics.per_suite["harmful_core"].pass_rate == pytest.approx(1.0)
    assert result.results_path.is_file()


def test_unsafe_model_fails_harmful_suite(tmp_path):
    items = load_suite("harmful_core")
    result = run_evaluation(MockModel("unsafe"), items, out_dir=tmp_path, timestamp="T")
    metrics = compute_metrics(result.rows)
    assert metrics.refusal_rate == pytest.approx(0.0)
    assert metrics.per_suite["harmful_core"].pass_rate == pytest.approx(0.0)


def test_over_refuser_fails_benign_suite(tmp_path):
    items = load_suite("benign_lookalike")
    result = run_evaluation(MockModel("over-refuser"), items, out_dir=tmp_path, timestamp="T")
    metrics = compute_metrics(result.rows)
    assert metrics.over_refusal_rate is not None
    assert metrics.over_refusal_rate >= 0.5
    assert metrics.per_suite["benign_lookalike"].pass_rate < 0.5


def test_safe_model_resists_all_injection(tmp_path):
    items = load_suite("injection")
    result = run_evaluation(MockModel("safe"), items, out_dir=tmp_path, timestamp="T")
    metrics = compute_metrics(result.rows)
    assert metrics.injection_resistance == pytest.approx(1.0)
    assert all(row["canary_present"] is False for row in result.rows)


def test_safe_model_zero_jailbreak_uplift(tmp_path):
    items = load_suite("harmful_core") + load_suite("jailbreak")
    result = run_evaluation(MockModel("safe"), items, out_dir=tmp_path, timestamp="T")
    metrics = compute_metrics(result.rows)
    assert metrics.jailbreak_refusal_rate == pytest.approx(1.0)
    assert metrics.jailbreak_uplift == pytest.approx(0.0)


def test_progress_callback_invoked(tmp_path):
    items = load_suite("harmful_core")[:3]
    seen = []
    run_evaluation(
        MockModel("safe"),
        items,
        out_dir=tmp_path,
        timestamp="T",
        progress=lambda i, n, item: seen.append((i, n, item.id)),
    )
    assert seen[0][0] == 1
    assert seen[-1] == (3, 3, items[2].id)


def test_resume_skips_all_existing(tmp_path):
    run_dir = tmp_path / "run"
    items = load_suite("harmful_core")[:3]
    first = CountingModel("safe")
    run_evaluation(first, items, out_dir=tmp_path, run_dir=run_dir)
    assert first.calls == 3

    second = CountingModel("safe")
    result = run_evaluation(second, items, out_dir=tmp_path, run_dir=run_dir, resume=True)
    assert second.calls == 0
    assert len(result.rows) == 3


def test_resume_runs_only_missing(tmp_path):
    run_dir = tmp_path / "run"
    items = load_suite("harmful_core")[:3]
    # First run only the first two items.
    run_evaluation(CountingModel("safe"), items[:2], out_dir=tmp_path, run_dir=run_dir)

    third = CountingModel("safe")
    result = run_evaluation(third, items, out_dir=tmp_path, run_dir=run_dir, resume=True)
    assert third.calls == 1  # only the missing third item
    assert len(result.rows) == 3
    ids = {row["id"] for row in result.rows}
    assert ids == {item.id for item in items}


def test_resume_without_run_dir_reuses_latest(tmp_path):
    """The documented CLI path: --resume alone finds the previous run directory."""
    items = load_suite("harmful_core")[:3]
    first_result = run_evaluation(CountingModel("safe"), items[:2], out_dir=tmp_path)

    second = CountingModel("safe")
    result = run_evaluation(second, items, out_dir=tmp_path, resume=True)
    assert result.run_dir == first_result.run_dir
    assert second.calls == 1  # only the missing third item
    assert len(result.rows) == 3


def test_resume_ignores_other_models_run_dirs(tmp_path):
    """Resume must not continue into a different model's run directory."""
    items = load_suite("harmful_core")[:1]
    other = run_evaluation(MockModel("unsafe"), items, out_dir=tmp_path)

    model = CountingModel("safe")
    result = run_evaluation(model, items, out_dir=tmp_path, resume=True)
    assert model.calls == 1  # nothing to resume for this model
    assert result.run_dir != other.run_dir
    assert result.run_dir.name.endswith("-mock-safe")


def test_resume_appends_cleanly_after_truncated_line(tmp_path):
    """A crash-truncated final line must not corrupt the next appended row."""
    run_dir = tmp_path / "run"
    items = load_suite("harmful_core")[:2]
    run_evaluation(MockModel("safe"), items[:1], out_dir=tmp_path, run_dir=run_dir)

    results = run_dir / "results.jsonl"
    with results.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "trunc')  # simulate a crash mid-write, no newline

    result = run_evaluation(
        MockModel("safe"), items, out_dir=tmp_path, run_dir=run_dir, resume=True
    )
    assert len(result.rows) == 2

    parsed_ids = []
    unparsable = 0
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed_ids.append(json.loads(line)["id"])
        except json.JSONDecodeError:
            unparsable += 1
    assert unparsable == 1  # the fragment stays isolated on its own line
    assert parsed_ids == [items[0].id, items[1].id]


class _ExplodingJudgeClient:
    """A judge client whose every call fails, to exercise judge error paths."""

    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("judge down")

    def __init__(self) -> None:
        self.messages = self._Messages()


def test_judge_failure_recorded_per_item(tmp_path):
    """A failed judge call must be visible in the row, not silently dropped."""
    items = load_suite("harmful_core")[:2]
    result = run_evaluation(
        MockModel("safe"),
        items,
        out_dir=tmp_path,
        timestamp="T",
        use_judge=True,
        judge_model="fake-judge",
        judge_client=_ExplodingJudgeClient(),
    )
    for row in result.rows:
        assert row["judge_verdict"] is None
        assert "judge down" in row["judge_error"]
        assert row["error"] is None  # generation itself succeeded


def test_error_rows_recorded_and_counted(tmp_path):
    items = load_suite("harmful_core")[:2]
    result = run_evaluation(
        RaisingModel(ValueError("kaboom")), items, out_dir=tmp_path, timestamp="T"
    )
    metrics = compute_metrics(result.rows)
    assert metrics.total_errors == 2
    assert all(row["error"] and "kaboom" in row["error"] for row in result.rows)
    assert all(row["passed"] is False for row in result.rows)


def test_results_written_incrementally(tmp_path):
    items = load_suite("harmful_core")[:3]
    result = run_evaluation(MockModel("safe"), items, out_dir=tmp_path, timestamp="T")
    lines = result.results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


# --- Retry behavior -------------------------------------------------------- #


class _FakeRateLimit(Exception):
    def __init__(self):
        super().__init__("rate limited")
        self.response = types.SimpleNamespace(headers={"retry-after": "0"})


class _FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _FlakyModel:
    def __init__(self, exc_sequence):
        self.model_name = "flaky"
        self._sequence = list(exc_sequence)
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        if self._sequence:
            exc = self._sequence.pop(0)
            if exc is not None:
                raise exc
        return MockModel("safe").generate(prompt)


def test_rate_limit_is_retried(monkeypatch, no_sleep):
    monkeypatch.setattr(runner, "_RateLimitError", _FakeRateLimit)
    model = _FlakyModel([_FakeRateLimit(), None])
    response = generate_with_retries(model, "hi", sleep=no_sleep)
    assert model.calls == 2
    assert response.text


def test_server_error_is_retried(monkeypatch, no_sleep):
    monkeypatch.setattr(runner, "_APIStatusError", _FakeStatusError)
    model = _FlakyModel([_FakeStatusError(503), None])
    generate_with_retries(model, "hi", sleep=no_sleep)
    assert model.calls == 2


def test_client_error_not_retried(monkeypatch, no_sleep):
    monkeypatch.setattr(runner, "_APIStatusError", _FakeStatusError)
    model = _FlakyModel([_FakeStatusError(400), None])
    with pytest.raises(_FakeStatusError):
        generate_with_retries(model, "hi", sleep=no_sleep)
    assert model.calls == 1


def test_retries_give_up_and_record_error(monkeypatch, no_sleep, tmp_path):
    monkeypatch.setattr(runner, "_RateLimitError", _FakeRateLimit)
    items = load_suite("harmful_core")[:1]
    model = _FlakyModel([_FakeRateLimit()] * 10)
    result = run_evaluation(
        model, items, out_dir=tmp_path, timestamp="T", sleep=no_sleep, max_retries=3
    )
    assert result.rows[0]["error"] is not None
    assert "_FakeRateLimit" in result.rows[0]["error"]
