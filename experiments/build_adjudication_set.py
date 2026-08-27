"""Build the blinded annotation package for the high-specificity stratum.

Why this exists
---------------
The scaled benchmark (``paper3_benchmark.py``) finds planner utility
anti-correlated with Paper 3's ``addressed`` label, with the residual
effect concentrated in the high-specificity stratum. Paper 3 documents
that this label is retrieval-bounded and under-counts engagement
exactly for narrow, object-level questions (its Paper 1 import scores
2/10 under abstract retrieval vs 10/10 under full-text adjudication).

So the anti-correlation has two competing explanations:

    (A) genuine neglect  -- the community really did skip these
    (B) measurement bias -- the label missed engagement that happened

This package produces the human adjudication that separates them. For
each of the 62 high-specificity tension questions, an annotator searches
the post-cutoff literature *by hand* (full text, not abstract TF-IDF)
and records whether the question was substantively engaged.

Design
------
Blinded: the existing label is withheld from the annotation file and
kept in a separate key, so the annotator cannot anchor on it. Order is
shuffled with a fixed seed for reproducibility. The 26 questions the
label calls addressed act as a control for false positives; the 36 it
calls not-addressed are where under-counting would appear.

Rubric (deliberately identical to Paper 3's, only the search differs):
a question counts as engaged if at least one paper published within
5 years after its cutoff investigates *this question's own* objects,
quantities, or claimed relationship -- not merely the same topic.

Outputs (in experiments/data/adjudication/):
    questions_blind.json  -- annotation items, no labels
    label_key.json        -- the withheld labels, for scoring only
    worksheet.csv         -- spreadsheet fallback
    bench_built.html      -- the annotation tool, data injected into
                             bench.html (which holds an __ITEMS__ slot)

Usage::

    python experiments/build_adjudication_set.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TENSION = REPO_ROOT / "experiments" / "data" / "paper3_tension"
RESULTS = REPO_ROOT / "results" / "paper3_benchmark.json"
OUT = REPO_ROOT / "experiments" / "data" / "adjudication"

SPECIFICITY_THRESHOLD = 3
SHUFFLE_SEED = 20260827


def main() -> None:
    benchmark = json.loads(RESULTS.read_text())
    questions = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in (TENSION / "questions.jsonl").read_text().splitlines()
        if line.strip()
    }
    labels = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in (TENSION / "labels.jsonl").read_text().splitlines()
        if line.strip()
    }

    stratum = [
        r
        for r in benchmark["per_question"]
        if (r.get("txt_specificity") or 0) >= SPECIFICITY_THRESHOLD
    ]

    items, key = [], {}
    for record in stratum:
        qid = record["question_id"]
        question = questions[qid]
        label = labels[qid]
        cutoff_year = int(question["cutoff_year"])

        items.append(
            {
                "question_id": qid,
                "question": question["question"],
                "subfield": question["subfield"],
                "cutoff_date": question["cutoff_date"],
                "search_window": f"{cutoff_year + 1}-01-01 to {cutoff_year + 5}-12-31",
                "evidence": [
                    {
                        "arxiv_id": c["arxiv_id"],
                        "published": c.get("published", "")[:10],
                        "sentence": c.get("sentence", ""),
                    }
                    for c in question["source_material"]["cluster"]
                ],
            }
        )
        key[qid] = {
            "paper3_addressed": bool(label.get("addressed")),
            "paper3_answer_status": label.get("answer_status"),
            "paper3_future_paper_count": label.get("future_paper_count"),
            "planner_top_k_utility": record["metrics"]["top_k_utility"],
        }

    random.Random(SHUFFLE_SEED).shuffle(items)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions_blind.json").write_text(json.dumps(items, indent=2))
    (OUT / "label_key.json").write_text(json.dumps(key, indent=2))

    with (OUT / "worksheet.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question_id", "subfield", "cutoff_date", "search_window",
             "question", "engaged", "evidence_ids", "notes"]
        )
        for item in items:
            writer.writerow(
                [item["question_id"], item["subfield"], item["cutoff_date"],
                 item["search_window"], item["question"], "", "", ""]
            )

    # Inject the items into the annotation tool template.
    template = OUT / "bench.html"
    if template.exists():
        payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        built = template.read_text().replace("__ITEMS__", payload)
        (OUT / "bench_built.html").write_text(built)

    n_addressed = sum(1 for v in key.values() if v["paper3_addressed"])
    print(f"Built adjudication set: {len(items)} questions")
    print(f"  withheld labels: {n_addressed} addressed, "
          f"{len(items) - n_addressed} not_addressed")
    print(f"  shuffled with seed {SHUFFLE_SEED}")
    print(f"  wrote {OUT.relative_to(REPO_ROOT)}/"
          "{questions_blind.json, label_key.json, worksheet.csv}")


if __name__ == "__main__":
    main()
