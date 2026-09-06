"""Shared pytest fixtures.

All fixtures here are offline: no network, no API key. Reusable fake objects for
the Anthropic SDK live in ``tests/fakes.py``.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def no_sleep():
    """A no-op sleep function for retry tests."""

    def _sleep(_seconds: float) -> None:
        return None

    return _sleep
