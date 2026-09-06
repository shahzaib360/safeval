"""Load and validate evaluation suites from JSONL files.

A *suite* is a `.jsonl` file where every line is one JSON object describing a
single evaluation item. Suites either ship inside the installed package (under
``safeval/suites``) or come from a user-supplied directory via ``--suite-dir``.

Each line is parsed into a typed :class:`EvalItem`. Malformed lines raise
:class:`SuiteError` with the file name, the 1-based line number, and the offending
field so the problem is easy to locate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

# The subset of ``expected`` values the harness understands.
_VALID_EXPECTED = frozenset({"refusal", "compliance", "resist"})


class SuiteError(Exception):
    """Raised when a suite file is missing or a line fails validation."""


@dataclass(frozen=True)
class EvalItem:
    """A single evaluation item loaded from a suite.

    Attributes:
        id: Stable unique identifier for the item.
        suite: Name of the suite the item belongs to (the file stem).
        prompt: The prompt sent to the model. For injection items this is the
            ``task_prompt`` (the legitimate task plus its embedded untrusted text).
        expected: One of ``"refusal"``, ``"compliance"``, or ``"resist"``.
        category: Optional coarse category label.
        canary: For injection items, the string whose presence in the output means
            the injection succeeded. ``None`` for non-injection items.
        base_id: For jailbreak items, the id of the plain ``harmful_core`` item the
            jailbreak is derived from. ``None`` otherwise.
        framing: For jailbreak items, the framing used (e.g. ``"roleplay"``).
        raw: The original decoded JSON object, preserved for reporting.
    """

    id: str
    suite: str
    prompt: str
    expected: str
    category: str | None = None
    canary: str | None = None
    base_id: str | None = None
    framing: str | None = None
    raw: dict = None  # type: ignore[assignment]

    @property
    def is_injection(self) -> bool:
        """Whether this item is graded by canary containment rather than by verdict."""
        return self.canary is not None


def _require(obj: dict, field: str, *, file: str, line: int) -> object:
    """Return ``obj[field]`` or raise a :class:`SuiteError` naming the location.

    Args:
        obj: The decoded JSON object for the line.
        field: The required field name.
        file: The suite file name, for the error message.
        line: The 1-based line number, for the error message.

    Returns:
        The field's value.

    Raises:
        SuiteError: If the field is missing.
    """
    if field not in obj:
        raise SuiteError(f"{file}:{line}: missing required field '{field}'")
    return obj[field]


def _parse_line(text: str, *, suite: str, file: str, line: int) -> EvalItem:
    """Parse a single JSONL line into an :class:`EvalItem`.

    Args:
        text: The raw line text (without trailing newline).
        suite: The suite name (file stem).
        file: The suite file name, for error messages.
        line: The 1-based line number, for error messages.

    Returns:
        The parsed item.

    Raises:
        SuiteError: If the line is not valid JSON, is not an object, or is missing
            a required field or has an invalid ``expected`` value.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SuiteError(f"{file}:{line}: invalid JSON ({exc.msg})") from exc

    if not isinstance(obj, dict):
        raise SuiteError(f"{file}:{line}: expected a JSON object, got {type(obj).__name__}")

    item_id = _require(obj, "id", file=file, line=line)
    expected = _require(obj, "expected", file=file, line=line)
    if expected not in _VALID_EXPECTED:
        raise SuiteError(
            f"{file}:{line}: invalid 'expected' value {expected!r}; "
            f"must be one of {sorted(_VALID_EXPECTED)}"
        )

    if suite == "injection" or "task_prompt" in obj:
        prompt = _require(obj, "task_prompt", file=file, line=line)
        canary = _require(obj, "canary", file=file, line=line)
    else:
        prompt = _require(obj, "prompt", file=file, line=line)
        canary = None

    return EvalItem(
        id=str(item_id),
        suite=suite,
        prompt=str(prompt),
        expected=str(expected),
        category=obj.get("category"),
        canary=str(canary) if canary is not None else None,
        base_id=obj.get("base_id"),
        framing=obj.get("framing"),
        raw=obj,
    )


def load_suite_file(path: Path, *, suite_name: str | None = None) -> list[EvalItem]:
    """Load and validate one suite file from disk.

    Args:
        path: Path to a ``.jsonl`` suite file.
        suite_name: Override for the suite name; defaults to the file stem.

    Returns:
        The list of validated items, in file order.

    Raises:
        SuiteError: If the file does not exist, is empty, contains a duplicate id,
            or has a malformed line.
    """
    if not path.is_file():
        raise SuiteError(f"suite file not found: {path}")

    suite = suite_name or path.stem
    items: list[EvalItem] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue  # tolerate blank lines
            item = _parse_line(stripped, suite=suite, file=path.name, line=line_no)
            if item.id in seen:
                raise SuiteError(f"{path.name}:{line_no}: duplicate id {item.id!r}")
            seen.add(item.id)
            items.append(item)

    if not items:
        raise SuiteError(f"{path.name}: suite is empty")
    return items


def _package_suites_dir() -> Path:
    """Return the directory of suites shipped inside the installed package.

    The data directory is named ``suites`` (not ``datasets``) on purpose: a
    ``safeval/datasets/`` package directory would shadow this very module the
    moment anyone dropped an ``__init__.py`` into it.
    """
    return Path(str(resources.files("safeval") / "suites"))


def available_suites(suite_dir: Path | None = None) -> dict[str, Path]:
    """Discover suites by name.

    Args:
        suite_dir: If given, discover ``.jsonl`` files here instead of the
            package's bundled datasets directory.

    Returns:
        A mapping of suite name (file stem) to its path, sorted by name.

    Raises:
        SuiteError: If ``suite_dir`` is given but is not a directory.
    """
    base = suite_dir if suite_dir is not None else _package_suites_dir()
    if suite_dir is not None and not base.is_dir():
        raise SuiteError(f"suite directory not found: {base}")
    found = {path.stem: path for path in sorted(base.glob("*.jsonl"))}
    return found


def load_suite(name: str, *, suite_dir: Path | None = None) -> list[EvalItem]:
    """Load one suite by name.

    Args:
        name: The suite name (file stem), e.g. ``"harmful_core"``.
        suite_dir: Optional directory override; defaults to the bundled datasets.

    Returns:
        The validated items for that suite.

    Raises:
        SuiteError: If no suite of that name exists.
    """
    suites = available_suites(suite_dir)
    if name not in suites:
        known = ", ".join(sorted(suites)) or "(none)"
        raise SuiteError(f"unknown suite {name!r}; available suites: {known}")
    return load_suite_file(suites[name], suite_name=name)


def load_suites(names: list[str], *, suite_dir: Path | None = None) -> list[EvalItem]:
    """Load several suites and concatenate their items in the given order.

    Args:
        names: Suite names to load, in order.
        suite_dir: Optional directory override.

    Returns:
        The concatenated list of items across all requested suites.
    """
    items: list[EvalItem] = []
    for name in names:
        items.extend(load_suite(name, suite_dir=suite_dir))
    return items
