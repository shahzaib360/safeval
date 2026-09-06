"""Tests for suite loading and validation."""

from __future__ import annotations

import pytest

from safeval.datasets import (
    SuiteError,
    available_suites,
    load_suite,
    load_suite_file,
    load_suites,
)

SHIPPED_SUITES = ["harmful_core", "benign_lookalike", "jailbreak", "injection"]
EXPECTED_COUNTS = {
    "harmful_core": 15,
    "benign_lookalike": 15,
    "jailbreak": 10,
    "injection": 8,
}


def test_all_shipped_suites_discoverable():
    suites = available_suites()
    for name in SHIPPED_SUITES:
        assert name in suites


@pytest.mark.parametrize("name", SHIPPED_SUITES)
def test_shipped_suite_loads_and_validates(name):
    items = load_suite(name)
    assert len(items) == EXPECTED_COUNTS[name]
    assert all(item.id for item in items)
    assert all(item.suite == name for item in items)


def test_injection_items_have_canary_and_are_flagged():
    items = load_suite("injection")
    assert all(item.is_injection for item in items)
    assert all(item.canary for item in items)
    assert all(item.expected == "resist" for item in items)


def test_jailbreak_items_link_back_to_harmful_core():
    harmful_ids = {item.id for item in load_suite("harmful_core")}
    for item in load_suite("jailbreak"):
        assert item.base_id in harmful_ids


def test_load_suites_concatenates_in_order():
    items = load_suites(["harmful_core", "injection"])
    assert len(items) == EXPECTED_COUNTS["harmful_core"] + EXPECTED_COUNTS["injection"]
    assert items[0].suite == "harmful_core"
    assert items[-1].suite == "injection"


def test_unknown_suite_raises_naming_available(tmp_path):
    with pytest.raises(SuiteError) as excinfo:
        load_suite("does_not_exist")
    assert "does_not_exist" in str(excinfo.value)


def test_malformed_line_error_includes_line_number(tmp_path):
    bad = tmp_path / "broken.jsonl"
    bad.write_text(
        '{"id": "a", "prompt": "hi", "expected": "refusal"}\n'
        '{"id": "b", "expected": "refusal"}\n',  # missing 'prompt'
        encoding="utf-8",
    )
    with pytest.raises(SuiteError) as excinfo:
        load_suite_file(bad)
    message = str(excinfo.value)
    assert "broken.jsonl:2" in message
    assert "prompt" in message


def test_invalid_json_error_includes_line_number(tmp_path):
    bad = tmp_path / "notjson.jsonl"
    bad.write_text('{"id": "a"} this is not json\n', encoding="utf-8")
    with pytest.raises(SuiteError) as excinfo:
        load_suite_file(bad)
    assert "notjson.jsonl:1" in str(excinfo.value)


def test_invalid_expected_value_rejected(tmp_path):
    bad = tmp_path / "expected.jsonl"
    bad.write_text('{"id": "a", "prompt": "hi", "expected": "maybe"}\n', encoding="utf-8")
    with pytest.raises(SuiteError) as excinfo:
        load_suite_file(bad)
    assert "expected" in str(excinfo.value)


def test_duplicate_id_rejected(tmp_path):
    bad = tmp_path / "dupes.jsonl"
    bad.write_text(
        '{"id": "x", "prompt": "a", "expected": "refusal"}\n'
        '{"id": "x", "prompt": "b", "expected": "refusal"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SuiteError) as excinfo:
        load_suite_file(bad)
    assert "duplicate id" in str(excinfo.value)


def test_empty_suite_rejected(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(SuiteError) as excinfo:
        load_suite_file(empty)
    assert "empty" in str(excinfo.value)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(SuiteError) as excinfo:
        load_suite_file(tmp_path / "nope.jsonl")
    assert "not found" in str(excinfo.value)


def test_suite_dir_override(tmp_path):
    custom = tmp_path / "custom.jsonl"
    custom.write_text(
        '{"id": "c1", "prompt": "hello éü中文", "expected": "compliance"}\n',
        encoding="utf-8",
    )
    suites = available_suites(tmp_path)
    assert "custom" in suites
    items = load_suite("custom", suite_dir=tmp_path)
    assert len(items) == 1
    assert "中文" in items[0].prompt


def test_suite_dir_not_found(tmp_path):
    with pytest.raises(SuiteError):
        available_suites(tmp_path / "missing")
