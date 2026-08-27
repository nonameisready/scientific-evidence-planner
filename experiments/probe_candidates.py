"""Search one question's candidate papers for specific concepts.

The ranked dossier (``prepare_judgments.py``) orders candidates by
lexical overlap, which can bury a real engagement whose wording differs
from the question's. Before recording a "not engaged" verdict, probe the
full candidate list for the concepts that would constitute engagement.

Usage::

    python experiments/probe_candidates.py <question_index> <term> [term...]

Terms are OR-ed, matched case-insensitively against title and abstract.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADJ = REPO_ROOT / "experiments" / "data" / "adjudication"


def main() -> None:
    items = json.loads((ADJ / "questions_blind.json").read_text())
    citations = json.loads((ADJ / "citations.json").read_text())

    index = int(sys.argv[1])
    patterns = [re.compile(re.escape(t), re.I) for t in sys.argv[2:]]
    item = items[index - 1]
    cands = citations.get(item["question_id"], [])

    hits = [
        c for c in cands
        if any(p.search(c["title"]) or p.search(c.get("abstract") or "")
               for p in patterns)
    ]
    print(f"[{index}] {item['question_id']}: {len(hits)}/{len(cands)} candidates match")
    for c in hits[:12]:
        ident = c.get("arxiv") or c.get("doi") or ""
        title_hit = any(p.search(c["title"]) for p in patterns)
        print(f"  {'T' if title_hit else ' '} [{c['year']}] {c['title']}  <{ident}>")
        if title_hit:
            abstract = (c.get("abstract") or "").replace("\n", " ")
            print(f"      {abstract[:260]}")


if __name__ == "__main__":
    main()
