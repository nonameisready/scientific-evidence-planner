"""Harvest post-cutoff citing papers for the adjudication set.

Paper 3's ``addressed`` label comes from abstract-level TF-IDF retrieval
over a subfield corpus, keeping the top-8 candidates. That retrieval is
documented to miss engagement with narrow, object-level questions. This
script builds the alternative retrieval channel used for adjudication:
the **citation graph**.

For each question, every paper in its generating evidence cluster is
looked up on Semantic Scholar and its citing papers are collected, then
filtered to the question's 5-year post-cutoff window. Work that engages
a question grounded in specific papers almost always cites at least one
of them, so this surfaces candidates that keyword retrieval over
abstracts cannot reach.

The output is candidate *evidence*, not verdicts: judging whether a
citing paper substantively engages the question is a separate step.

Usage::

    python experiments/harvest_citations.py

Writes experiments/data/adjudication/citations.json incrementally, so
the run can be interrupted and resumed without refetching.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADJ = REPO_ROOT / "experiments" / "data" / "adjudication"
OUT = ADJ / "citations.json"

API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/citations"
FIELDS = "title,abstract,year,publicationDate,externalIds,venue"
PAGE = 100
MAX_PAGES = 2          # up to 200 citing papers per source paper
SLEEP = 1.1            # unauthenticated rate limit courtesy
MAX_RETRIES = 10

# The unauthenticated endpoint throttles hard but briefly: a 429 clears in
# ~1-2s. Exponential backoff therefore idles long after the limit lifts, so
# 429s get many short retries instead; other errors still back off.
THROTTLE_WAIT = 2.0


def fetch_page(arxiv_id: str, offset: int) -> dict | None:
    url = f"{API.format(arxiv_id)}?fields={FIELDS}&limit={PAGE}&offset={offset}"
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "evidence-planner-adjudication/1.0"}
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(THROTTLE_WAIT)
                continue
            time.sleep(SLEEP * (2 ** min(attempt, 3)))
        except Exception:
            time.sleep(SLEEP * (2 ** min(attempt, 3)))
    return None


def citing_papers(arxiv_id: str) -> list[dict]:
    out, offset = [], 0
    for _ in range(MAX_PAGES):
        payload = fetch_page(arxiv_id, offset)
        time.sleep(SLEEP)
        if not payload or not payload.get("data"):
            break
        for row in payload["data"]:
            paper = row.get("citingPaper") or {}
            if paper.get("title"):
                out.append(paper)
        if payload.get("next") is None:
            break
        offset = payload["next"]
    return out


def main() -> None:
    items = json.loads((ADJ / "questions_blind.json").read_text())
    store = json.loads(OUT.read_text()) if OUT.exists() else {}

    for i, item in enumerate(items, 1):
        qid = item["question_id"]
        if qid in store:
            continue

        lo_year = int(item["search_window"][:4])
        hi_year = int(item["search_window"][14:18])
        seen, candidates = set(), []

        for source in item["evidence"]:
            for paper in citing_papers(source["arxiv_id"]):
                year = paper.get("year")
                if year is None or not (lo_year <= year <= hi_year):
                    continue
                key = paper.get("paperId")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "title": paper.get("title", ""),
                        "abstract": (paper.get("abstract") or "")[:1200],
                        "year": year,
                        "date": paper.get("publicationDate"),
                        "arxiv": (paper.get("externalIds") or {}).get("ArXiv"),
                        "doi": (paper.get("externalIds") or {}).get("DOI"),
                        "cites": source["arxiv_id"],
                    }
                )

        store[qid] = candidates
        OUT.write_text(json.dumps(store, indent=1))
        print(f"[{i}/{len(items)}] {qid}: {len(candidates)} in-window citing papers",
              flush=True)

    total = sum(len(v) for v in store.values())
    empty = sum(1 for v in store.values() if not v)
    print(f"\nDone. {len(store)} questions, {total} candidate papers, "
          f"{empty} questions with no in-window citing paper.")


if __name__ == "__main__":
    main()
