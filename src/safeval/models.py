"""Model adapters: a small protocol plus an Anthropic-backed and a mock backend.

The rest of the harness only depends on the :class:`ModelAdapter` protocol, so
tests run entirely offline against :class:`MockModel` and real evaluations run
against :class:`AnthropicModel`. A model is selected from a spec string by
:func:`make_model` (e.g. ``"mock:safe"`` or ``"claude-opus-5"``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# The Anthropic model id suggested in CLI help and error messages. It is never
# applied silently — the user must always name a model explicitly (via --model,
# --judge-model, or the SAFEVAL_MODEL env var).
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"

# Naive harm-word list used only by the *mock* presets to fake a safety policy.
# This is intentionally crude: it exists to give deterministic, offline behavior
# for demos and tests, not to be a real classifier.
_HARM_WORDS = (
    "phishing",
    "hotwire",
    "counterfeit",
    "malware",
    "ransomware",
    "keylogger",
    "botnet",
    "steal",
    "shoplift",
    "break into",
    "cheat on my taxes",
    "hide income",
    "without their permission",
    "without permission",
    "clone",
    "scam",
    "fake",
    "impersonating",
)

# Extra trigger words for the "over-refuser" preset. These include benign
# homonyms so the preset over-refuses on the benign look-alike suite.
_OVER_REFUSAL_TRIGGERS = _HARM_WORDS + (
    "kill",
    "shoot",
    "terminate",
    "chemical",
    "lock",
    "attack",
    "weapon",
    "password",
)

_MOCK_REFUSAL_TEXT = (
    "I can't help with that. This request looks like it could facilitate harm, "
    "so I won't provide it. If you have a legitimate, lawful goal I'm happy to "
    "help with a safer alternative."
)


@dataclass(frozen=True)
class ModelResponse:
    """A single model completion.

    Attributes:
        text: The concatenated text of the response. May be empty when the model
            refuses via a dedicated stop reason.
        stop_reason: The provider stop reason (e.g. ``"end_turn"``,
            ``"max_tokens"``, ``"refusal"``), or ``None`` if unknown.
        model_name: The concrete model that produced the response.
    """

    text: str
    stop_reason: str | None
    model_name: str


@runtime_checkable
class ModelAdapter(Protocol):
    """A minimal text-in/text-out model interface.

    Implementations must be safe to call sequentially and should surface provider
    stop reasons so the grader can use them.
    """

    model_name: str

    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response for a single user prompt.

        Args:
            prompt: The user prompt.

        Returns:
            The model's response.
        """
        ...


def _looks_harmful(prompt: str, triggers: tuple[str, ...]) -> bool:
    """Return whether any trigger word appears in the prompt (case-insensitive)."""
    lowered = prompt.lower()
    return any(word in lowered for word in triggers)


class MockModel:
    """Deterministic offline model with three presets.

    Presets:
        ``"safe"``: refuses prompts matching a naive harm-word list, complies
            otherwise. Approximates a well-aligned model for demos and tests.
        ``"unsafe"``: complies with everything (never refuses).
        ``"over-refuser"``: refuses anything containing trigger words, including
            benign homonyms, so it over-refuses on the benign suite.

    The mock never touches the network and produces stable output for a prompt.
    """

    _PRESETS = frozenset({"safe", "unsafe", "over-refuser"})

    def __init__(self, preset: str = "safe") -> None:
        """Initialize the mock.

        Args:
            preset: One of ``"safe"``, ``"unsafe"``, ``"over-refuser"``.

        Raises:
            ValueError: If ``preset`` is not a known preset.
        """
        if preset not in self._PRESETS:
            raise ValueError(f"unknown mock preset {preset!r}; choose from {sorted(self._PRESETS)}")
        self.preset = preset
        self.model_name = f"mock:{preset}"

    def _comply(self, prompt: str) -> ModelResponse:
        """Produce a compliant, benign-looking response with numbered steps."""
        text = (
            "Sure, here's how you can approach that:\n"
            "1. First, understand what you're trying to accomplish.\n"
            "2. Then gather the tools or information you need.\n"
            "3. Finally, carry out the steps carefully and check the result.\n"
            "Let me know if you'd like more detail on any step."
        )
        return ModelResponse(text=text, stop_reason="end_turn", model_name=self.model_name)

    def _refuse(self) -> ModelResponse:
        """Produce a refusal response."""
        return ModelResponse(
            text=_MOCK_REFUSAL_TEXT, stop_reason="end_turn", model_name=self.model_name
        )

    def generate(self, prompt: str) -> ModelResponse:
        """Generate a response according to the configured preset.

        Args:
            prompt: The user prompt.

        Returns:
            A refusal or a compliance response depending on the preset and prompt.
        """
        if self.preset == "unsafe":
            return self._comply(prompt)
        if self.preset == "over-refuser":
            return (
                self._refuse()
                if _looks_harmful(prompt, _OVER_REFUSAL_TRIGGERS)
                else self._comply(prompt)
            )
        # "safe"
        return self._refuse() if _looks_harmful(prompt, _HARM_WORDS) else self._comply(prompt)


class AnthropicModel:
    """Anthropic-backed model adapter.

    The client is constructed with no arguments, so it reads ``ANTHROPIC_API_KEY``
    from the environment. The key is never logged. A missing key surfaces as a
    friendly :class:`RuntimeError` on first use.
    """

    def __init__(
        self, model_id: str, *, max_tokens: int = 1024, client: object | None = None
    ) -> None:
        """Initialize the adapter.

        Args:
            model_id: The Anthropic model id (no date suffix), e.g.
                ``"claude-opus-5"``.
            max_tokens: Max tokens for each completion (1024 by default).
            client: Optional pre-built client, primarily for tests. When ``None``
                the client is created lazily on first :meth:`generate`.
        """
        self.model_name = model_id
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self) -> object:
        """Return the Anthropic client, constructing it lazily.

        Returns:
            The client instance.

        Raises:
            RuntimeError: If the ``anthropic`` package is not installed, or no
                API key is set (the SDK would otherwise fail at request time
                with an unhelpful ``TypeError``).
        """
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'anthropic' package is required for real model runs. "
                "Install it with: pip install anthropic"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "No API key found. Set the ANTHROPIC_API_KEY environment variable "
                "to run against a real model (the mock:* models run offline)."
            )
        # No arguments: the SDK reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic()
        return self._client

    def generate(self, prompt: str) -> ModelResponse:
        """Call the Anthropic Messages API for a single prompt.

        Args:
            prompt: The user prompt.

        Returns:
            The model response, with ``stop_reason`` preserved. When the API
            signals a refusal via ``stop_reason == "refusal"`` the text may be
            empty; that is still a valid refusal signal for the grader.

        Raises:
            RuntimeError: If the API key is missing.
        """
        import anthropic  # local import keeps the module import-light for tests

        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Authentication failed. Set the ANTHROPIC_API_KEY environment "
                "variable to a valid key before running against a real model."
            ) from exc

        text = _extract_text(response)
        stop_reason = getattr(response, "stop_reason", None)
        return ModelResponse(text=text, stop_reason=stop_reason, model_name=self.model_name)


def _extract_text(response: object) -> str:
    """Concatenate the text of all text blocks in a Messages API response.

    The ``content`` attribute is a list of typed blocks; only blocks whose
    ``type == "text"`` contribute (thinking blocks, if any, are skipped).

    Args:
        response: A Messages API response object.

    Returns:
        The concatenated text, possibly empty.
    """
    content = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


_MOCK_SPEC = re.compile(r"^mock:(.+)$")


def make_model(spec: str, *, max_tokens: int = 1024, client: object | None = None) -> ModelAdapter:
    """Build a model adapter from a spec string.

    Args:
        spec: ``"mock:safe"``, ``"mock:unsafe"``, ``"mock:over-refuser"``, or an
            Anthropic model id such as ``"claude-opus-5"``.
        max_tokens: Max tokens for Anthropic completions.
        client: Optional injected Anthropic client (ignored for mock specs).

    Returns:
        A configured :class:`ModelAdapter`.

    Raises:
        ValueError: If a ``mock:`` spec names an unknown preset.
    """
    match = _MOCK_SPEC.match(spec.strip())
    if match:
        return MockModel(match.group(1))
    return AnthropicModel(spec.strip(), max_tokens=max_tokens, client=client)
