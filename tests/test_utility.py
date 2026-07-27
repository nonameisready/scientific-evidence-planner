import math

import pytest

from planner import (
    ActionType,
    CandidateAction,
    ExpectedEntropyReductionEstimator,
    ProvenanceNoveltyEstimator,
    ResourceTractabilityEstimator,
    SimpleCostEstimator,
    UtilityConfig,
    UtilityEstimator,
    entropy,
    load_utility_config,
)


def make_action(action_type=ActionType.COLLECT_OBSERVATION, target="H1", **kwargs):
    defaults = dict(
        action_id="a1",
        action_type=action_type,
        description="test action",
        estimated_cost=0.5,
        estimated_time=0.4,
        required_resources=["instrument time"],
        target_hypothesis=target,
    )
    defaults.update(kwargs)
    return CandidateAction(**defaults)


def test_entropy():
    assert entropy([1.0]) == 0.0
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([0.5, 0.5, 0.0]) == pytest.approx(1.0)


def test_information_gain_positive_and_bounded(hypotheses, graph):
    est = ExpectedEntropyReductionEstimator()
    gain = est.estimate(make_action(), hypotheses, graph)
    prior_entropy = entropy([h.prior for h in hypotheses])
    assert 0.0 < gain < prior_entropy


def test_information_gain_zero_without_target(hypotheses, graph):
    est = ExpectedEntropyReductionEstimator()
    assert est.estimate(make_action(target=None), hypotheses, graph) == 0.0
    assert est.estimate(make_action(target="unknown"), hypotheses, graph) == 0.0


def test_stronger_actions_gain_more(hypotheses, graph):
    est = ExpectedEntropyReductionEstimator()
    weak = est.estimate(make_action(ActionType.READ_PAPER), hypotheses, graph)
    strong = est.estimate(make_action(ActionType.COLLECT_OBSERVATION), hypotheses, graph)
    assert strong > weak


def test_novelty_boosted_for_single_root_hypothesis(hypotheses, graph):
    est = ProvenanceNoveltyEstimator()
    # H1's evidence all descends from one dataset -> full boost.
    concentrated = est.estimate(make_action(ActionType.RERUN_ANALYSIS, "H1"), hypotheses, graph)
    base = est.source_novelty[ActionType.RERUN_ANALYSIS]
    assert concentrated == pytest.approx(min(base * 2.0, 1.0))
    # Unknown target falls back to the base score.
    no_target = est.estimate(make_action(ActionType.RERUN_ANALYSIS, None), hypotheses, graph)
    assert no_target == base


def test_tractability_penalizes_missing_resources(hypotheses, graph):
    action = make_action()
    unconstrained = ResourceTractabilityEstimator()
    assert unconstrained.estimate(action, hypotheses, graph) == pytest.approx(0.6)
    constrained = ResourceTractabilityEstimator(available_resources={"compute"})
    assert constrained.estimate(action, hypotheses, graph) == pytest.approx(0.3)


def test_cost_passthrough(hypotheses, graph):
    assert SimpleCostEstimator().estimate(make_action(), hypotheses, graph) == 0.5


def test_utility_combination_respects_weights(hypotheses, graph):
    action = make_action()
    neutral = UtilityEstimator().estimate(action, hypotheses, graph)
    assert neutral.total == pytest.approx(
        neutral.information_gain + neutral.novelty + neutral.tractability - neutral.cost
    )
    cost_averse = UtilityEstimator(config=UtilityConfig(cost_weight=10.0))
    assert cost_averse.estimate(action, hypotheses, graph).total < neutral.total


def test_load_utility_config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("utility:\n  cost_weight: 2.5\n")
    cfg = load_utility_config(path)
    assert cfg.cost_weight == 2.5
    assert cfg.information_gain_weight == 1.0
