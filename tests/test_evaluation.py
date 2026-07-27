import pytest

from planner import (
    ActionGenerator,
    ActionType,
    CandidateAction,
    GapAnalyzer,
    GroundTruthEvidence,
    HistoricalBenchmark,
    Hypothesis,
    Planner,
    Recommendation,
    UtilityBreakdown,
)
from planner.evaluation import (
    action_diversity,
    average_information_gain,
    entropy_reduction,
    expected_utility,
    hypothesis_coverage,
    hypothesis_entropy,
    resolution_efficiency,
    top_k_utility,
)


def make_rec(action_type, target, total, ig=0.5):
    action = CandidateAction(
        action_id=f"a_{action_type.value}_{target}",
        action_type=action_type,
        description="x",
        estimated_cost=0.1,
        estimated_time=0.1,
        target_hypothesis=target,
    )
    utility = UtilityBreakdown(
        information_gain=ig, novelty=0.5, tractability=0.5, cost=0.1, total=total
    )
    return Recommendation(action=action, utility=utility, reasoning="")


@pytest.fixture
def recs():
    return [
        make_rec(ActionType.COLLECT_OBSERVATION, "H1", 2.0, ig=0.8),
        make_rec(ActionType.COMPARE_DATASETS, "H1", 1.5, ig=0.6),
        make_rec(ActionType.SEARCH_ARCHIVE, "H2", 1.0, ig=0.4),
        make_rec(ActionType.SEARCH_ARCHIVE, "H3", 0.5, ig=0.2),
    ]


def test_top_k_utility(recs):
    assert top_k_utility(recs, 2) == pytest.approx(1.75)
    assert top_k_utility([], 3) == 0.0


def test_entropy_metrics():
    before = [Hypothesis("H1", "", prior=0.5), Hypothesis("H2", "", prior=0.5)]
    after = [Hypothesis("H1", "", prior=0.9), Hypothesis("H2", "", prior=0.1)]
    assert hypothesis_entropy(before) == pytest.approx(1.0)
    assert entropy_reduction(before, after) > 0


def test_average_information_gain(recs):
    assert average_information_gain(recs) == pytest.approx(0.5)


def test_action_diversity(recs):
    assert action_diversity(recs) == pytest.approx(3 / 4)
    assert action_diversity([]) == 0.0


def test_hypothesis_coverage(recs):
    hyps = [Hypothesis(h, "") for h in ("H1", "H2", "H3", "H4")]
    assert hypothesis_coverage(recs, hyps) == pytest.approx(3 / 4)


def test_resolution_efficiency(recs):
    ground_truth = [
        GroundTruthEvidence("e1", "H1", "collect_observation"),
        GroundTruthEvidence("e2", "H4", "run_simulation"),
    ]
    assert resolution_efficiency(recs, ground_truth) == pytest.approx(0.5)
    assert resolution_efficiency(recs, []) == 0.0


def test_expected_utility_discounts_by_rank(recs):
    value = expected_utility(recs)
    # Rank-discounted mean must be above the plain mean because the best
    # recommendation is first.
    plain_mean = sum(r.utility.total for r in recs) / len(recs)
    assert value > plain_mean


class SimplePipeline:
    """Minimal concrete PlannerPipeline for benchmark tests."""

    def __init__(self, question, hypotheses):
        self.question = question
        self.hypotheses = hypotheses

    def plan(self, graph, top_k):
        report = GapAnalyzer().analyze(self.question, self.hypotheses, graph)
        actions = ActionGenerator().generate(report)
        recs = Planner().rank(actions, self.hypotheses, graph, top_k=top_k)
        return recs, self.hypotheses


def test_historical_benchmark_end_to_end(question, hypotheses, graph):
    ground_truth = [
        GroundTruthEvidence(
            "c3",
            "H1",
            "collect_observation",
            likelihoods={"H1": 0.9, "H2": 0.2, "H3": 0.2},
        )
    ]
    benchmark = HistoricalBenchmark(graph, ground_truth, cutoff="2020")
    result = benchmark.run(SimplePipeline(question, hypotheses), "q1", top_k=20)

    assert result.cutoff == "2020"
    assert set(result.metrics) == {
        "top_k_utility",
        "entropy_reduction",
        "average_information_gain",
        "action_diversity",
        "hypothesis_coverage",
        "resolution_efficiency",
        "expected_utility",
    }
    # The plan recommends collect_observation for H1 (independence gap),
    # which matches the hidden evidence: entropy must drop.
    assert result.metrics["resolution_efficiency"] == 1.0
    assert result.metrics["entropy_reduction"] > 0
    assert 0 < result.metrics["hypothesis_coverage"] <= 1
