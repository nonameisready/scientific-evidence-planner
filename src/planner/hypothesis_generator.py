"""Generate structured competing hypotheses for a question.

The framework is offline, so hypothesis generation is pluggable rather
than model-driven: the default generator loads curated hypothesis sets
from JSON (produced by hand or by upstream projects), and a template
generator produces generic methodological alternatives when no curated
set exists. Both implement :class:`HypothesisGenerator`, so a future
LLM-backed generator can drop in without touching other modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .schemas import Hypothesis, Question


class HypothesisGenerator(Protocol):
    """Interface: produce competing hypotheses for a question."""

    def generate(self, question: Question) -> list[Hypothesis]:
        ...


def normalize_priors(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Normalize priors to sum to 1; assign uniform priors if all are 0."""
    total = sum(h.prior for h in hypotheses)
    if total <= 0:
        uniform = 1.0 / len(hypotheses) if hypotheses else 0.0
        for h in hypotheses:
            h.prior = uniform
    else:
        for h in hypotheses:
            h.prior = h.prior / total
    return hypotheses


class StaticHypothesisGenerator:
    """Load curated hypothesis sets from a JSON file keyed by question_id.

    File format::

        {
          "astro_001": [
            {"hypothesis_id": "H1", "description": "...", "prior": 0.4, ...},
            ...
          ]
        }
    """

    def __init__(self, path: str | Path) -> None:
        self._sets: dict[str, list[dict]] = json.loads(Path(path).read_text())

    def generate(self, question: Question) -> list[Hypothesis]:
        rows = self._sets.get(question.question_id)
        if rows is None:
            raise KeyError(f"no hypothesis set for question {question.question_id!r}")
        return normalize_priors([Hypothesis.from_dict(r) for r in rows])


class TemplateHypothesisGenerator:
    """Fallback generator with generic, domain-independent alternatives.

    For any empirical claim there are recurring methodological
    explanations; these templates encode them without reference to any
    particular field.
    """

    TEMPLATES = [
        "The reported finding reflects a real underlying effect.",
        "The reported finding is an artifact of the analysis method.",
        "The reported finding is confounded by an unmodeled factor.",
        "The reported finding is caused by an instrument or measurement systematic.",
    ]

    def generate(self, question: Question) -> list[Hypothesis]:
        hypotheses = [
            Hypothesis(
                hypothesis_id=f"{question.question_id}_H{i + 1}",
                description=description,
                prior=0.0,
                metadata={"generated_by": "template"},
            )
            for i, description in enumerate(self.TEMPLATES)
        ]
        return normalize_priors(hypotheses)


def save_hypotheses(hypotheses: list[Hypothesis], path: str | Path) -> None:
    """Persist hypotheses as structured JSON."""
    Path(path).write_text(json.dumps([h.to_dict() for h in hypotheses], indent=2))


def load_hypotheses(path: str | Path) -> list[Hypothesis]:
    """Load hypotheses previously saved with :func:`save_hypotheses`."""
    return [Hypothesis.from_dict(r) for r in json.loads(Path(path).read_text())]
