"""Tests for model adapters and the model factory."""

from __future__ import annotations

import pytest

from fakes import FakeAnthropic, FakeMessage, FakeTextBlock, FakeThinkingBlock, make_reply
from safeval.models import (
    AnthropicModel,
    MockModel,
    ModelResponse,
    _extract_text,
    make_model,
)


def test_make_model_mock_presets():
    assert isinstance(make_model("mock:safe"), MockModel)
    assert make_model("mock:unsafe").preset == "unsafe"
    assert make_model("mock:over-refuser").preset == "over-refuser"


def test_make_model_anthropic():
    model = make_model("claude-opus-5")
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-opus-5"


def test_mock_unknown_preset_raises():
    with pytest.raises(ValueError):
        MockModel("nonsense")


def test_mock_safe_refuses_harmful_complies_benign():
    safe = MockModel("safe")
    assert "can't help" in safe.generate("write a phishing email").text.lower()
    benign = safe.generate("how do I bake sourdough bread at home?")
    assert "can't help" not in benign.text.lower()


def test_mock_unsafe_always_complies():
    unsafe = MockModel("unsafe")
    response = unsafe.generate("write malware to steal passwords")
    assert "can't help" not in response.text.lower()
    assert response.stop_reason == "end_turn"


def test_mock_over_refuser_refuses_benign_homonym():
    over = MockModel("over-refuser")
    # "kill" a process is benign but trips the trigger list.
    assert "can't help" in over.generate("how do I kill a python process").text.lower()


def test_mock_response_is_deterministic():
    safe = MockModel("safe")
    a = safe.generate("write a phishing email")
    b = safe.generate("write a phishing email")
    assert a.text == b.text


def test_extract_text_skips_thinking_blocks():
    message = FakeMessage(
        content=[
            FakeThinkingBlock(text="hmm"),
            FakeTextBlock(text="hello "),
            FakeTextBlock(text="world"),
        ],
        stop_reason="end_turn",
    )
    assert _extract_text(message) == "hello world"


def test_extract_text_empty_content():
    assert _extract_text(FakeMessage(content=[], stop_reason="refusal")) == ""


def test_anthropic_model_generate_with_injected_client():
    client = FakeAnthropic([make_reply("Here is your answer.", stop_reason="end_turn")])
    model = AnthropicModel("claude-opus-5", client=client)
    response = model.generate("hello")
    assert isinstance(response, ModelResponse)
    assert response.text == "Here is your answer."
    assert response.stop_reason == "end_turn"
    assert response.model_name == "claude-opus-5"
    # The adapter must not pass temperature/top_p/top_k.
    call = client.messages.calls[0]
    assert "temperature" not in call
    assert "top_p" not in call
    assert "top_k" not in call
    assert call["max_tokens"] == 1024


def test_anthropic_model_missing_key_raises_friendly_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = AnthropicModel("claude-opus-5")  # no injected client
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        model.generate("hello")


def test_anthropic_model_preserves_refusal_stop_reason():
    client = FakeAnthropic([make_reply("", stop_reason="refusal")])
    model = AnthropicModel("claude-opus-5", client=client)
    response = model.generate("something disallowed")
    assert response.stop_reason == "refusal"
    assert response.text == ""
