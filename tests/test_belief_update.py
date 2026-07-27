import pytest

from planner import (
    BayesianBeliefUpdater,
    Hypothesis,
    NewEvidence,
    apply_evidence_sequence,
)


def test_bayes_update_shifts_mass_toward_likely_hypothesis(hypotheses):
    evidence = NewEvidence(
        evidence_id="e1",
        description="independent confirmation",
        likelihoods={"H1": 0.9, "H2": 0.1, "H3": 0.1},
    )
    updated = BayesianBeliefUpdater().update(hypotheses, evidence)
    posteriors = {h.hypothesis_id: h.prior for h in updated}
    assert posteriors["H1"] > 0.4
    assert posteriors["H2"] < 0.3
    assert sum(posteriors.values()) == pytest.approx(1.0)


def test_update_is_non_destructive_and_records_evidence(hypotheses):
    evidence = NewEvidence("e1", "x", likelihoods={"H1": 0.9, "H2": 0.1, "H3": 0.1})
    updated = BayesianBeliefUpdater().update(hypotheses, evidence)
    assert hypotheses[0].prior == 0.4  # input untouched
    assert "e1" in updated[0].supporting_evidence  # H1 rose
    assert "e1" in updated[1].contradicting_evidence  # H2 fell


def test_missing_likelihood_is_neutral(hypotheses):
    evidence = NewEvidence("e1", "x", likelihoods={})
    updated = BayesianBeliefUpdater().update(hypotheses, evidence)
    assert [h.prior for h in updated] == pytest.approx([0.4, 0.3, 0.3])


def test_impossible_evidence_keeps_priors(hypotheses):
    evidence = NewEvidence("e1", "x", likelihoods={"H1": 0.0, "H2": 0.0, "H3": 0.0})
    updated = BayesianBeliefUpdater().update(hypotheses, evidence)
    assert [h.prior for h in updated] == [0.4, 0.3, 0.3]


def test_negative_likelihood_rejected(hypotheses):
    evidence = NewEvidence("e1", "x", likelihoods={"H1": -0.5})
    with pytest.raises(ValueError):
        BayesianBeliefUpdater().update(hypotheses, evidence)


def test_sequence_application_compounds():
    hypotheses = [Hypothesis("H1", "real", prior=0.5), Hypothesis("H2", "artifact", prior=0.5)]
    items = [
        NewEvidence("e1", "a", likelihoods={"H1": 0.8, "H2": 0.2}),
        NewEvidence("e2", "b", likelihoods={"H1": 0.8, "H2": 0.2}),
    ]
    updated = apply_evidence_sequence(hypotheses, items)
    # Two 4:1 likelihood ratios from even priors -> 16:17 odds ... i.e. 16/17.
    assert updated[0].prior == pytest.approx(16 / 17)
