"""Tests for the Paper 3 tension-cluster conversion and scaled benchmark."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from paper3_benchmark import (
    REFUTED_LIKELIHOODS,
    build_graph,
    distribution,
    make_hypotheses,
    permutation_pvalue,
    run_question,
    spearman,
)
from planner import EdgeType, Planner, Question


def make_question(n_papers=3):
    return Question(
        question_id="t1",
        question="What causes the reported discrepancy?",
        metadata={
            "cutoff_date": "2020-12-31",
            "source_material": {
                "cluster": [
                    {
                        "arxiv_id": f"1234.{i:05d}",
                        "published": f"201{5 + i}-01-01",
                        "sentence": f"claim sentence {i}",
                    }
                    for i in range(n_papers)
                ]
            },
        },
    )


STRUCTURE = {
    "str_type_contradiction": 1,
    "str_type_method": 0,
    "str_type_stance": 1,
    "str_quantified": 1,
    "str_max_sigma": 4.0,
    "str_cross_facility": 0,
    "str_n_papers": 3,
}


def test_priors_seeded_from_structure_and_normalized():
    hyps = make_hypotheses(make_question(), STRUCTURE)
    assert len(hyps) == 4
    assert sum(h.prior for h in hyps) == pytest.approx(1.0)
    # Quantified 4-sigma tension shifts mass toward the real-effect
    # hypothesis; stance shifts toward the confounder; cross_facility=0
    # leaves the systematic hypothesis at base weight (the minimum).
    assert hyps[0].prior > hyps[3].prior
    assert hyps[2].prior > hyps[3].prior


def test_graph_structure_follows_rules():
    question = make_question(n_papers=3)
    hyps = make_hypotheses(question, STRUCTURE)
    graph = build_graph(question, [h.hypothesis_id for h in hyps], STRUCTURE)
    # 4 hypothesis + 3 paper + 3 claim nodes.
    assert len(graph) == 10
    h1, h2, h3, h4 = [h.hypothesis_id for h in hyps]
    # Quantified: real-effect support at 0.7 from every claim.
    assert [e.confidence for e in graph.evidence_for(h1)] == [0.7] * 3
    # Contradiction-typed: artifact supported at 0.4.
    assert [e.confidence for e in graph.evidence_for(h2)] == [0.4] * 3
    # Stance-typed: confounder supported; single-facility: systematic bare.
    assert len(graph.evidence_for(h3)) == 3
    assert graph.evidence_for(h4) == []
    # Distinct papers are distinct provenance roots.
    roots = set()
    for edge in graph.evidence_for(h1):
        roots |= graph.provenance_roots(edge.source)
    assert len(roots) == 3
    # Claims derive from their papers.
    assert all(
        graph.out_edges(f"claim_t1_{i}", EdgeType.DERIVED_FROM) for i in range(3)
    )


def test_run_question_produces_metrics():
    question = make_question()
    label = {"addressed": True, "answer_status": "partially_addressed",
             "future_paper_count": 7, "premise_status": "intact"}
    record = run_question(question, STRUCTURE, label, Planner())
    for name in (
        "top_k_utility",
        "average_information_gain",
        "action_diversity",
        "hypothesis_coverage",
        "expected_utility",
    ):
        assert name in record["metrics"]
    assert "entropy_reduction" not in record["metrics"]  # premise intact


def test_refuted_premise_reduces_entropy():
    question = make_question()
    label = {"addressed": True, "answer_status": "premise_refuted",
             "future_paper_count": 3, "premise_status": "refuted"}
    record = run_question(question, STRUCTURE, label, Planner())
    assert record["metrics"]["entropy_reduction"] > 0
    assert set(REFUTED_LIKELIHOODS) == {"H1", "H2", "H3", "H4"}


def test_distribution_stats():
    d = distribution([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4
    assert d["mean"] == pytest.approx(2.5)
    assert d["median"] == pytest.approx(2.5)
    assert d["min"] == 1.0 and d["max"] == 4.0


def test_permutation_pvalue_detects_separation():
    a, b = [1.0] * 20, [2.0] * 20
    assert permutation_pvalue(a, b) < 0.01
    same = permutation_pvalue([1.0, 2.0] * 10, [1.0, 2.0] * 10)
    assert same > 0.5


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
