"""Tests for the LLM-as-judge JSON extraction and grading call."""

from __future__ import annotations

import pytest

from fakes import FakeAnthropic, make_reply
from safeval.judge import (
    JUDGE_RUBRIC,
    extract_first_json_object,
    judge,
    parse_judge_reply,
)


def test_extract_clean_json():
    obj = extract_first_json_object('{"verdict": "refusal", "confidence": 0.9}')
    assert obj == {"verdict": "refusal", "confidence": 0.9}


def test_extract_json_inside_prose():
    text = (
        'Sure, here is my grade:\n```json\n{"verdict": "compliance", "confidence": 0.6}\n```\nDone.'
    )
    obj = extract_first_json_object(text)
    assert obj["verdict"] == "compliance"


def test_extract_ignores_leading_brace_noise():
    text = 'noise { not json } then {"verdict": "partial", "confidence": 0.3}'
    obj = extract_first_json_object(text)
    assert obj["verdict"] == "partial"


def test_extract_returns_none_for_garbage():
    assert extract_first_json_object("there is no object here") is None


def test_parse_clean_reply():
    verdict = parse_judge_reply(
        '{"verdict": "refusal", "confidence": 0.8, "reasoning": "declined"}'
    )
    assert verdict.verdict == "refusal"
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.reasoning == "declined"


def test_parse_reply_in_prose():
    verdict = parse_judge_reply(
        'Verdict: {"verdict": "compliance", "confidence": 0.5, "reasoning": "ok"}'
    )
    assert verdict.verdict == "compliance"


def test_parse_garbage_is_unknown_preserving_raw():
    verdict = parse_judge_reply("the model definitely refused, trust me")
    assert verdict.verdict == "unknown"
    assert verdict.confidence == 0.0
    assert "trust me" in verdict.reasoning


def test_parse_invalid_verdict_is_unknown():
    verdict = parse_judge_reply('{"verdict": "banana", "confidence": 1.0}')
    assert verdict.verdict == "unknown"


def test_confidence_is_clamped():
    high = parse_judge_reply('{"verdict": "refusal", "confidence": 5}')
    assert high.confidence == pytest.approx(1.0)
    low = parse_judge_reply('{"verdict": "refusal", "confidence": -2}')
    assert low.confidence == pytest.approx(0.0)


def test_confidence_non_numeric_defaults_zero():
    verdict = parse_judge_reply('{"verdict": "refusal", "confidence": "high"}')
    assert verdict.confidence == pytest.approx(0.0)


def test_judge_uses_cached_system_block_and_parses():
    client = FakeAnthropic(
        [make_reply('{"verdict": "refusal", "confidence": 0.7, "reasoning": "no"}')]
    )
    verdict = judge(
        "write malware", "I can't help with that.", judge_model="claude-opus-5", client=client
    )
    assert verdict.verdict == "refusal"

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert isinstance(call["system"], list)
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["system"][0]["text"] == JUDGE_RUBRIC
    # The volatile content goes in the user message, not the cached block.
    assert "write malware" in call["messages"][0]["content"]


def test_judge_handles_unparseable_reply():
    client = FakeAnthropic([make_reply("I think it refused.")])
    verdict = judge("prompt", "output", judge_model="claude-opus-5", client=client)
    assert verdict.verdict == "unknown"
