"""Scaled benchmark: run the planner over Paper 3's 374 tension questions.

Paper 3's evidence-tension questions each store their generating evidence
cluster verbatim (sentences, arXiv ids, publication dates) plus a
future-outcome label and cluster-level structure features. This script
converts every cluster into a planner evidence graph with transparent
rules, runs the full pipeline at each question's own historical cutoff,
and reports the evaluation metrics as *distributions* over 374 questions
instead of a single worked example.

Vendored inputs (verbatim subsets of Paper 3's released files) live in
``experiments/data/paper3_tension/``.

Graph construction rules (deliberately simple and auditable)
------------------------------------------------------------
Hypotheses come from the framework's generic template generator (the
reported effect is real / analysis artifact / unmodeled confounder /
measurement systematic). Priors are seeded from the cluster's structure
flags rather than uniform, so each question's evidence configuration
carries into the information-gain computation (weights below are
normalized): quantified tensions shift mass toward "real effect"
(scaled by ``str_max_sigma``), method-challenge and contradiction
language toward "analysis artifact", detection-vs-limit stance
opposition toward "unmodeled confounder", and cross-facility clusters
toward "measurement systematic".

For each sentence in the cluster: one paper node (timestamped with its
arXiv publication date) and one claim node derived from it. Edge rules,
driven by the same structure flags:

* every claim supports "real effect" --- base 0.5, +0.2 when the
  cluster quantifies the tension (``str_quantified``), -0.1 when it is
  method-typed;
* contradiction-typed clusters (``str_type_contradiction``): each claim
  also supports "analysis artifact" at 0.4 --- explicit disagreement is
  prima facie evidence that at least one analysis is off (0.5 when
  method-typed);
* stance-typed clusters (``str_type_stance``, detection vs.\ limit):
  each claim supports "unmodeled confounder" at 0.35 --- opposed
  detection regimes commonly reflect differing observing conditions;
* cross-facility clusters (``str_cross_facility``): each claim supports
  "measurement systematic" at 0.35 --- when instruments disagree, an
  instrument-level explanation is live; single-facility tensions leave
  that hypothesis evidence-free for gap analysis to surface.

Provenance: distinct arXiv ids are distinct roots. This *understates*
dependence (two papers can share an underlying dataset), so
independence-driven novelty is an upper bound here.

What is measured at scale
-------------------------
Five metrics are computable for every question: top-K utility, average
expected information gain, action diversity, hypothesis coverage, and
expected utility. Entropy reduction is additionally computed for the
premise-refuted cases, where the label fixes the direction of the hidden
evidence (against "real effect"). Resolution efficiency is *not*
computable at scale --- the labels record that and when a question was
addressed, not through which action type --- and is reported as such.

Outcome linkage: planner metrics are stratified by Paper 3's
``addressed`` label, with a seeded permutation test on the difference in
means, and rank-correlated with future paper counts.

Usage::

    python experiments/paper3_benchmark.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from planner import (
    ActionGenerator,
    BayesianBeliefUpdater,
    EdgeType,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    GapAnalyzer,
    NewEvidence,
    NodeType,
    Planner,
    Question,
    TemplateHypothesisGenerator,
    UtilityEstimator,
    load_question_dict,
    load_utility_config,
)
from planner.evaluation import (
    action_diversity,
    average_information_gain,
    expected_utility,
    hypothesis_coverage,
    hypothesis_entropy,
    top_k_utility,
)

DATA = REPO_ROOT / "experiments" / "data" / "paper3_tension"
RESULTS = REPO_ROOT / "results"
TOP_K = 5

#: Likelihoods replayed for premise-refuted questions: the label fixes
#: only the direction (the "real effect" hypothesis lost); the three
#: alternatives stay symmetric.
REFUTED_LIKELIHOODS = {"H1": 0.15, "H2": 0.5, "H3": 0.5, "H4": 0.5}


def load_rows(name: str) -> dict[str, dict]:
    rows = {}
    for line in (DATA / name).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["question_id"]] = row
    return rows


def make_hypotheses(question: Question, structure: dict):
    """Template hypotheses with structure-seeded priors (rules above)."""
    hypotheses = TemplateHypothesisGenerator().generate(question)
    sigma = min(float(structure.get("str_max_sigma") or 0.0), 5.0)
    weights = [
        1.0 + 0.5 * (sigma / 5.0),  # H1 real effect
        1.0
        + 0.3 * bool(structure.get("str_type_contradiction"))
        + 0.4 * bool(structure.get("str_type_method")),  # H2 artifact
        1.0 + 0.4 * bool(structure.get("str_type_stance")),  # H3 confounder
        1.0 + 0.4 * bool(structure.get("str_cross_facility")),  # H4 systematic
    ]
    for h, w in zip(hypotheses, weights):
        h.prior = w
    total = sum(h.prior for h in hypotheses)
    for h in hypotheses:
        h.prior /= total
    return hypotheses


def build_graph(
    question: Question, hypothesis_ids: list[str], structure: dict
) -> EvidenceGraph:
    """Convert one tension cluster into an evidence graph (rules above)."""
    h1, h2, h3, h4 = hypothesis_ids
    graph = EvidenceGraph()
    for hid in hypothesis_ids:
        graph.add_node(EvidenceNode(hid, NodeType.HYPOTHESIS))

    contradiction = bool(structure.get("str_type_contradiction"))
    method = bool(structure.get("str_type_method"))
    quantified = bool(structure.get("str_quantified"))
    stance = bool(structure.get("str_type_stance"))
    cross_facility = bool(structure.get("str_cross_facility"))

    real_conf = 0.5 + (0.2 if quantified else 0.0) - (0.1 if method else 0.0)
    artifact_conf = 0.5 if method else (0.4 if contradiction else None)

    cluster = question.metadata["source_material"]["cluster"]
    for i, item in enumerate(cluster):
        paper_id = f"paper_{item['arxiv_id']}"
        claim_id = f"claim_{question.question_id}_{i}"
        published = item.get("published", "")
        if not graph.has_node(paper_id):
            graph.add_node(
                EvidenceNode(paper_id, NodeType.PAPER, timestamp=published)
            )
        graph.add_node(
            EvidenceNode(
                claim_id,
                NodeType.CLAIM,
                label=item.get("sentence", "")[:200],
                timestamp=published,
            )
        )
        graph.add_edge(EvidenceEdge(claim_id, paper_id, EdgeType.DERIVED_FROM))
        graph.add_edge(
            EvidenceEdge(claim_id, h1, EdgeType.SUPPORTS, confidence=real_conf)
        )
        if artifact_conf is not None:
            graph.add_edge(
                EvidenceEdge(claim_id, h2, EdgeType.SUPPORTS, confidence=artifact_conf)
            )
        if stance:
            graph.add_edge(
                EvidenceEdge(claim_id, h3, EdgeType.SUPPORTS, confidence=0.35)
            )
        if cross_facility:
            graph.add_edge(
                EvidenceEdge(claim_id, h4, EdgeType.SUPPORTS, confidence=0.35)
            )
    return graph


def run_question(
    question: Question, structure: dict, label: dict, planner: Planner
) -> dict:
    hypotheses = make_hypotheses(question, structure)
    hyp_ids = [h.hypothesis_id for h in hypotheses]

    graph = build_graph(question, hyp_ids, structure)
    visible = graph.filter_before(question.metadata["cutoff_date"])

    report = GapAnalyzer().analyze(question, hypotheses, visible)
    actions = ActionGenerator().generate(report)
    recommendations = planner.rank(actions, hypotheses, visible, top_k=TOP_K)

    record = {
        "question_id": question.question_id,
        "subfield": question.metadata.get("subfield"),
        "cutoff_year": question.metadata.get("cutoff_year"),
        "n_cluster_papers": structure.get("str_n_papers"),
        "n_gaps": len(report.gaps),
        "addressed": bool(label.get("addressed")),
        "answer_status": label.get("answer_status"),
        "future_paper_count": label.get("future_paper_count"),
        "metrics": {
            "top_k_utility": top_k_utility(recommendations, TOP_K),
            "average_information_gain": average_information_gain(recommendations),
            "action_diversity": action_diversity(recommendations),
            "hypothesis_coverage": hypothesis_coverage(recommendations, hypotheses),
            "expected_utility": expected_utility(recommendations),
        },
    }

    # Direction is only knowable for refuted premises: replay evidence
    # against the "real effect" hypothesis and measure the entropy drop.
    if label.get("premise_status") == "refuted":
        likelihoods = {
            hid: REFUTED_LIKELIHOODS[f"H{i + 1}"]
            for i, hid in enumerate(hyp_ids)
        }
        evidence = NewEvidence(
            evidence_id=f"{question.question_id}_refutation",
            description="historical premise refutation (direction from label)",
            likelihoods=likelihoods,
        )
        posterior = BayesianBeliefUpdater().update(hypotheses, evidence)
        record["metrics"]["entropy_reduction"] = hypothesis_entropy(
            hypotheses
        ) - hypothesis_entropy(posterior)
    return record


def distribution(values: list[float]) -> dict:
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n

    def quantile(q: float) -> float:
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return values[lo] + (values[hi] - values[lo]) * (idx - lo)

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(var**0.5, 4),
        "min": round(values[0], 4),
        "q25": round(quantile(0.25), 4),
        "median": round(quantile(0.5), 4),
        "q75": round(quantile(0.75), 4),
        "max": round(values[-1], 4),
    }


def permutation_pvalue(
    group_a: list[float], group_b: list[float], n_perm: int = 5000, seed: int = 0
) -> float:
    """Two-sided permutation test on the difference in means."""
    rng = random.Random(seed)
    observed = abs(
        sum(group_a) / len(group_a) - sum(group_b) / len(group_b)
    )
    pooled = group_a + group_b
    n_a = len(group_a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(
            sum(pooled[:n_a]) / n_a - sum(pooled[n_a:]) / (len(pooled) - n_a)
        )
        if diff >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rank = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rank[order[k]] = avg
            i = j + 1
        return rank

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main() -> None:
    questions = load_rows("questions.jsonl")
    labels = load_rows("labels.jsonl")
    structures = load_rows("structure_features.jsonl")

    config = load_utility_config(REPO_ROOT / "configs" / "default.yaml")
    planner = Planner(UtilityEstimator(config=config))

    records = []
    for qid, row in sorted(questions.items()):
        question = load_question_dict(row)
        records.append(
            run_question(question, structures.get(qid, {}), labels[qid], planner)
        )

    metric_names = [
        "top_k_utility",
        "average_information_gain",
        "action_diversity",
        "hypothesis_coverage",
        "expected_utility",
    ]
    distributions = {
        name: distribution([r["metrics"][name] for r in records])
        for name in metric_names
    }

    addressed = [r for r in records if r["addressed"]]
    unaddressed = [r for r in records if not r["addressed"]]
    stratified = {}
    for name in metric_names:
        a = [r["metrics"][name] for r in addressed]
        b = [r["metrics"][name] for r in unaddressed]
        stratified[name] = {
            "addressed_mean": round(sum(a) / len(a), 4),
            "unaddressed_mean": round(sum(b) / len(b), 4),
            "permutation_p": round(permutation_pvalue(a, b), 4),
        }

    correlations = {
        name: round(
            spearman(
                [r["metrics"][name] for r in records],
                [float(r["future_paper_count"] or 0) for r in records],
            ),
            4,
        )
        for name in metric_names
    }

    refuted = [r for r in records if "entropy_reduction" in r["metrics"]]

    summary = {
        "n_questions": len(records),
        "top_k": TOP_K,
        "metric_distributions": distributions,
        "stratified_by_addressed": {
            "n_addressed": len(addressed),
            "n_unaddressed": len(unaddressed),
            "metrics": stratified,
        },
        "spearman_vs_future_paper_count": correlations,
        "refuted_premises": {
            "n": len(refuted),
            "entropy_reduction": [
                {
                    "question_id": r["question_id"],
                    "bits": round(r["metrics"]["entropy_reduction"], 4),
                }
                for r in refuted
            ],
        },
        "not_computable_at_scale": {
            "resolution_efficiency": "labels record whether/when a question "
            "was addressed, not through which action type; requires "
            "action-typed ground truth mining"
        },
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "paper3_benchmark.json"
    out.write_text(json.dumps({"summary": summary, "per_question": records}, indent=2))

    print(f"Ran planner over {len(records)} tension questions (top-{TOP_K})\n")
    print(f"{'metric':<28}{'mean':>8}{'std':>8}{'median':>8}{'IQR':>16}")
    for name, d in distributions.items():
        iqr = f"[{d['q25']:.3f}, {d['q75']:.3f}]"
        print(f"{name:<28}{d['mean']:>8.3f}{d['std']:>8.3f}{d['median']:>8.3f}{iqr:>16}")
    print(f"\nStratified by future outcome (addressed {len(addressed)} vs not {len(unaddressed)}):")
    for name, s in stratified.items():
        print(
            f"  {name:<26} addressed {s['addressed_mean']:.3f} vs "
            f"{s['unaddressed_mean']:.3f}  (perm. p={s['permutation_p']})"
        )
    print("\nSpearman vs future paper count:")
    for name, rho in correlations.items():
        print(f"  {name:<26} rho={rho:+.3f}")
    print(f"\nPremise-refuted cases (direction from label): {len(refuted)}")
    for r in refuted:
        print(f"  {r['question_id']}: entropy reduction {r['metrics']['entropy_reduction']:.3f} bits")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
