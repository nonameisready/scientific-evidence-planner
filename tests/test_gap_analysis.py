from planner import GapAnalyzer, GapAnalyzerConfig


def test_untouched_hypothesis_is_max_confidence_gap(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    by_id = {g.hypothesis_id: g for g in report.gaps}
    assert by_id["H3"].confidence == 1.0
    assert "No evidence" in by_id["H3"].reason


def test_single_root_hypothesis_flagged_for_independence(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    by_id = {g.hypothesis_id: g for g in report.gaps}
    # H1 has two claims but both descend from dataset d1.
    assert "independent" in by_id["H1"].missing_evidence
    assert by_id["H1"].confidence == 0.78


def test_sparse_evidence_flagged(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    by_id = {g.hypothesis_id: g for g in report.gaps}
    # H2 has a single evidence edge, below min_evidence_count=2.
    assert "additional" in by_id["H2"].missing_evidence


def test_gaps_sorted_by_confidence(question, hypotheses, graph):
    report = GapAnalyzer().analyze(question, hypotheses, graph)
    confidences = [g.confidence for g in report.gaps]
    assert confidences == sorted(confidences, reverse=True)


def test_thresholds_configurable(question, hypotheses, graph):
    lenient = GapAnalyzer(
        GapAnalyzerConfig(
            min_evidence_count=1,
            min_support_confidence=0.1,
            require_independent_sources=False,
        )
    )
    report = lenient.analyze(question, hypotheses, graph)
    flagged = {g.hypothesis_id for g in report.gaps}
    # With lenient thresholds only the evidence-free H3 remains a gap.
    assert flagged == {"H3"}
