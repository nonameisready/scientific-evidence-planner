"""Render a compact judging dossier for each adjudication question.

Harvesting gives every question its in-window citing papers; most are
irrelevant (they cite the source paper for background). This script
ranks candidates by lexical overlap with the question's distinctive
terms so the plausible engagements surface first, and prints a dossier
short enough to judge from.

Ranking is a reading aid only -- it decides presentation order, not
verdicts. The verdict is a judgement about whether a candidate
investigates *this question's own* objects, quantities, or claimed
relationship.

Usage::

    python experiments/prepare_judgments.py [start] [end]   # 1-indexed
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADJ = REPO_ROOT / "experiments" / "data" / "adjudication"

STOP = set("""a an the and or of in on for to from with by is are was were be been
this that these those it its as at we our their there which what how does do
can could may might will would should has have had not no than then when where
between among during study studies results result observed observations show
shows shown suggest suggests using used use new such more most other others
both same different first second two three about into over under within very
some any all each other paper we find found indicate indicates""".split())


def terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
    return {w for w in words if w not in STOP}


def score(candidate: dict, key_terms: set[str]) -> float:
    hay = terms(candidate["title"] + " " + (candidate.get("abstract") or ""))
    if not hay:
        return 0.0
    overlap = key_terms & hay
    # Title hits weigh more: engagement usually shows in the title.
    title_hits = key_terms & terms(candidate["title"])
    return len(overlap) + 2.0 * len(title_hits)


def main() -> None:
    items = json.loads((ADJ / "questions_blind.json").read_text())
    citations = json.loads((ADJ / "citations.json").read_text())

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(items)

    for i, item in enumerate(items, 1):
        if not (start <= i <= end):
            continue
        qid = item["question_id"]
        if qid not in citations:
            # Not harvested yet. Distinct from "harvested, found nothing" --
            # conflating the two would invite a false "not engaged" verdict.
            print("=" * 78)
            print(f"[{i}] {qid}  -- NOT YET HARVESTED, do not judge")
            print()
            continue
        cands = citations[qid]
        key_terms = terms(item["question"])
        ranked = sorted(cands, key=lambda c: score(c, key_terms), reverse=True)

        print("=" * 78)
        print(f"[{i}] {qid}  ({item['subfield']}, window {item['search_window'][:4]}"
              f"-{item['search_window'][14:18]})")
        print(f"Q: {item['question']}")
        print(f"-- {len(cands)} in-window citing papers; top {min(6, len(ranked))} by term overlap:")
        if not ranked:
            print("   (none -- no in-window citation found for any source paper)")
        for c in ranked[:6]:
            ident = c.get("arxiv") or c.get("doi") or ""
            print(f"   * [{c['year']}] {c['title']}   <{ident}>")
            abstract = (c.get("abstract") or "").replace("\n", " ")
            if abstract:
                print(f"     {abstract[:300]}")
        print()


if __name__ == "__main__":
    main()
