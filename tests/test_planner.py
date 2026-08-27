from planner import (
    ActionGenerator,
    GapAnalyzer,
    Planner,
    UtilityConfig,
    UtilityEstimator,
)


def make_recommendations(question, hypotheses, graph, top_k=None, config=None):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    actions = ActionGenerator().generate(report)
    planner = Planner(UtilityEstimator(config=config or UtilityConfig()))
    return planner.rank(actions, hypotheses, graph, top_k=top_k)


def test_ranking_is_sorted_and_ranked(question, hypotheses, graph):
    recs = make_recommendations(question, hypotheses, graph)
    totals = [r.utility.total for r in recs]
    assert totals == sorted(totals, reverse=True)
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))


def test_top_k_truncates(question, hypotheses, graph):
    all_recs = make_recommendations(question, hypotheses, graph)
    top3 = make_recommendations(question, hypotheses, graph, top_k=3)
    assert len(top3) == 3
    assert [r.action.action_id for r in top3] == [
        r.action.action_id for r in all_recs[:3]
    ]


def test_recommendations_carry_reasoning_and_target(question, hypotheses, graph):
    recs = make_recommendations(question, hypotheses, graph, top_k=5)
    for rec in recs:
        assert rec.action.target_hypothesis in {h.hypothesis_id for h in hypotheses}
        assert "information gain" in rec.reasoning
        assert rec.action.target_hypothesis in rec.reasoning


def test_weights_change_ranking(question, hypotheses, graph):
    neutral = make_recommendations(question, hypotheses, graph)
    cheap_first = make_recommendations(
        question, hypotheses, graph, config=UtilityConfig(cost_weight=50.0)
    )
    # With extreme cost aversion the cheapest action must win.
    assert cheap_first[0].action.estimated_cost == min(
        r.action.estimated_cost for r in neutral
    )
