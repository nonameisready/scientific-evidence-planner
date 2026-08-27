"""Load scientific questions from JSON files or dicts.

Questions produced by upstream projects only need the two required keys
``question_id`` and ``question``; any extra keys are preserved in
``Question.metadata`` so no information from other repositories is lost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Question

_KNOWN_KEYS = {"question_id", "question", "domain", "metadata"}


class QuestionLoadError(ValueError):
    """Raised when a question payload is malformed."""


def load_question_dict(data: dict[str, Any]) -> Question:
    """Build a :class:`Question` from a dict, validating required keys.

    Unknown top-level keys are folded into ``metadata`` so datasets from
    previous projects load without modification.
    """
    for key in ("question_id", "question"):
        if key not in data or not data[key]:
            raise QuestionLoadError(f"question payload missing required key: {key!r}")

    extra = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    metadata = dict(data.get("metadata", {}))
    metadata.update(extra)

    return Question(
        question_id=str(data["question_id"]),
        question=str(data["question"]),
        domain=data.get("domain", "generic"),
        metadata=metadata,
    )


def load_question(path: str | Path) -> Question:
    """Load a single question from a JSON file."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise QuestionLoadError(f"{path}: expected a JSON object")
    return load_question_dict(payload)


def load_questions(path: str | Path) -> list[Question]:
    """Load a question dataset: a JSON list, or a JSONL file (one object per line)."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise QuestionLoadError(f"{path}: expected a JSON object or list")
    return [load_question_dict(row) for row in rows]
