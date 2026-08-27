"""Import the Paper 3 question/outcome dataset into planner objects.

Paper 3 (`scientific-question-outcome-prediction-indicators`) released a
temporally grounded dataset of 980 astronomy research questions frozen at
five historical cutoffs, each with a future-outcome label derived from
the five years of literature that followed. Its records already carry
``question_id`` and ``question`` keys, so the planner's question loader
ingests them directly; everything else (source, cutoff, provenance
cluster) is preserved in ``Question.metadata``.

Usage::

    python experiments/import_paper3.py \
        [path/to/questions_v1.jsonl] [path/to/labels_v1.jsonl]

With no arguments, runs on the three-record sample vendored under
``experiments/data/paper3_questions_sample.jsonl`` so the demo works
offline. Point it at Paper 3's released ``data/questions/questions_v1.jsonl``
and ``data/labels/labels_v1.jsonl`` for the full dataset.

The outcome labels make each question a candidate benchmark unit for this
framework: the cutoff gives the historical-simulation boundary, and the
label (addressed / answer_status / premise_status, plus follow-up papers)
is exactly the hidden post-cutoff ground truth the evaluation protocol
needs. This script performs the load and pairing; graph construction from
Paper 3's provenance clusters is left to the experiment that needs it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from planner import Question, load_questions

SAMPLE = REPO_ROOT / "experiments" / "data" / "paper3_questions_sample.jsonl"


def load_paper3(
    questions_path: str | Path,
    labels_path: str | Path | None = None,
) -> list[tuple[Question, dict | None]]:
    """Load Paper 3 questions, each paired with its outcome label (if given).

    Returns (question, label) tuples; label is None when no labels file is
    supplied or the question has no label row.
    """
    questions = load_questions(questions_path)
    labels: dict[str, dict] = {}
    if labels_path is not None:
        for line in Path(labels_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["question_id"]] = row
    return [(q, labels.get(q.question_id)) for q in questions]


def main() -> None:
    questions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    labels_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    pairs = load_paper3(questions_path, labels_path)
    print(f"Loaded {len(pairs)} Paper 3 questions from {questions_path}")
    for question, label in pairs:
        source = question.metadata.get("source", "?")
        cutoff = question.metadata.get("cutoff_year", "?")
        line = f"  {question.question_id} [{source}, cutoff {cutoff}]"
        if label is not None:
            line += (
                f" -> addressed={label.get('addressed')}"
                f", status={label.get('answer_status')}"
            )
        print(line)
        print(f"    {question.question[:100]}...")


if __name__ == "__main__":
    main()
