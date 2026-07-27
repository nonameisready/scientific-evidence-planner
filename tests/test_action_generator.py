from planner import ActionGenerator, ActionType, GapAnalyzer, load_catalog


def test_actions_generated_per_gap(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    actions = ActionGenerator().generate(report)
    assert actions
    gap_ids = {g.hypothesis_id for g in report.gaps}
    assert {a.target_hypothesis for a in actions} == gap_ids
    # Unique ids.
    ids = [a.action_id for a in actions]
    assert len(ids) == len(set(ids))


def test_independence_gap_gets_independent_source_actions(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    actions = ActionGenerator().generate(report)
    h1_types = {a.action_type for a in actions if a.target_hypothesis == "H1"}
    assert ActionType.COLLECT_OBSERVATION in h1_types
    assert ActionType.COMPARE_DATASETS in h1_types


def test_actions_carry_cost_time_resources(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    for action in ActionGenerator().generate(report):
        assert 0.0 <= action.estimated_cost <= 1.0
        assert 0.0 <= action.estimated_time <= 1.0
        assert action.required_resources
        assert action.metadata["gap_reason"]


def test_catalog_loads_from_yaml(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "actions:\n"
        "  - action_type: run_simulation\n"
        "    description: simulate\n"
        "    estimated_cost: 0.2\n"
        "    estimated_time: 0.3\n"
        "    required_resources: [compute]\n"
    )
    catalog = load_catalog(path)
    assert len(catalog) == 1
    assert catalog[0].action_type == ActionType.RUN_SIMULATION
    assert catalog[0].estimated_cost == 0.2
