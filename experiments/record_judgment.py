"""Append one adjudication verdict to annotations.json.

Usage::

    python experiments/record_judgment.py <index> <yes|partial|no|unsure> \
        "<evidence ids>" "<notes>"

Index is the 1-based position in questions_blind.json (the same numbering
the dossier prints). Re-recording an index overwrites its previous entry,
so a revised judgment replaces rather than duplicates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADJ = REPO_ROOT / "experiments" / "adjudication_out"
ADJ.mkdir(exist_ok=True)
STORE = ADJ / "annotations.json"
ITEMS = REPO_ROOT / "experiments" / "data" / "adjudication" / "questions_blind.json"

VALID = {"yes", "partial", "no", "unsure"}


def main() -> None:
    items = json.loads(ITEMS.read_text())
    index = int(sys.argv[1])
    verdict = sys.argv[2].lower()
    evidence = sys.argv[3] if len(sys.argv) > 3 else ""
    notes = sys.argv[4] if len(sys.argv) > 4 else ""

    if verdict not in VALID:
        raise SystemExit(f"verdict must be one of {sorted(VALID)}")

    qid = items[index - 1]["question_id"]
    payload = json.loads(STORE.read_text()) if STORE.exists() else {"annotations": []}
    rows = [r for r in payload["annotations"] if r["question_id"] != qid]
    rows.append({
        "question_id": qid,
        "engaged": verdict,
        "evidence_ids": evidence,
        "notes": notes,
    })
    payload["annotations"] = rows
    STORE.write_text(json.dumps(payload, indent=1))
    print(f"[{index}] {qid} -> {verdict}  ({len(rows)}/62 recorded)")


if __name__ == "__main__":
    main()
