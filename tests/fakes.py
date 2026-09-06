"""Offline fakes for the Anthropic SDK surface the code touches.

These are plain dataclasses/objects — no network, no API key. They mimic just
enough of the SDK (``client.messages.create`` returning a message with a list of
typed content blocks and a ``stop_reason``) for the adapters and judge to work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeTextBlock:
    """A stand-in for an SDK text content block."""

    text: str
    type: str = "text"


@dataclass
class FakeThinkingBlock:
    """A stand-in for a non-text (thinking) content block."""

    text: str = ""
    type: str = "thinking"


@dataclass
class FakeMessage:
    """A stand-in for a Messages API response."""

    content: list
    stop_reason: str = "end_turn"


class FakeMessages:
    """Fake ``client.messages`` exposing ``create`` returning queued replies."""

    def __init__(self, replies: list[FakeMessage]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> FakeMessage:
        self.calls.append(kwargs)
        if self._replies:
            return self._replies.pop(0)
        return FakeMessage(content=[FakeTextBlock(text="")], stop_reason="end_turn")


class FakeAnthropic:
    """Fake Anthropic client with a ``messages`` attribute."""

    def __init__(self, replies: list[FakeMessage]) -> None:
        self.messages = FakeMessages(replies)


def make_reply(text: str, stop_reason: str = "end_turn") -> FakeMessage:
    """Build a fake reply with a single text block."""
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason=stop_reason)
