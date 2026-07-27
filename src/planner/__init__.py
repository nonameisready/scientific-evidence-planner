"""Scientific Evidence Planner: offline planning of the next
evidence-acquisition step under uncertainty."""

from .action_generator import ActionGenerator, ActionTemplate, load_catalog
from .belief_update import (
    BayesianBeliefUpdater,
    BeliefUpdater,
    apply_evidence_sequence,
)
from .evaluation import (
    BenchmarkResult,
    GroundTruthEvidence,
    HistoricalBenchmark,
    PlannerPipeline,
)
from .evidence_graph import EvidenceGraph
from .gap_analysis import GapAnalyzer, GapAnalyzerConfig
from .hypothesis_generator import (
    HypothesisGenerator,
    StaticHypothesisGenerator,
    TemplateHypothesisGenerator,
    load_hypotheses,
    normalize_priors,
    save_hypotheses,
)
from .planner import Planner
from .question_loader import (
    QuestionLoadError,
    load_question,
    load_question_dict,
    load_questions,
)
from .schemas import (
    ActionType,
    CandidateAction,
    EdgeType,
    EvidenceEdge,
    EvidenceGap,
    EvidenceNode,
    GapReport,
    Hypothesis,
    NewEvidence,
    NodeType,
    Question,
    Recommendation,
    UtilityBreakdown,
)
from .utility import (
    ExpectedEntropyReductionEstimator,
    ProvenanceNoveltyEstimator,
    ResourceTractabilityEstimator,
    SimpleCostEstimator,
    UtilityConfig,
    UtilityEstimator,
    entropy,
    load_utility_config,
)

__all__ = [
    "ActionGenerator",
    "ActionTemplate",
    "ActionType",
    "BayesianBeliefUpdater",
    "BeliefUpdater",
    "BenchmarkResult",
    "CandidateAction",
    "EdgeType",
    "EvidenceEdge",
    "EvidenceGap",
    "EvidenceGraph",
    "EvidenceNode",
    "ExpectedEntropyReductionEstimator",
    "GapAnalyzer",
    "GapAnalyzerConfig",
    "GapReport",
    "GroundTruthEvidence",
    "HistoricalBenchmark",
    "Hypothesis",
    "HypothesisGenerator",
    "NewEvidence",
    "NodeType",
    "Planner",
    "PlannerPipeline",
    "ProvenanceNoveltyEstimator",
    "Question",
    "QuestionLoadError",
    "Recommendation",
    "ResourceTractabilityEstimator",
    "SimpleCostEstimator",
    "StaticHypothesisGenerator",
    "TemplateHypothesisGenerator",
    "UtilityBreakdown",
    "UtilityConfig",
    "UtilityEstimator",
    "apply_evidence_sequence",
    "entropy",
    "load_catalog",
    "load_hypotheses",
    "load_question",
    "load_question_dict",
    "load_questions",
    "load_utility_config",
    "normalize_priors",
    "save_hypotheses",
]
