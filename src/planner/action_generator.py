"""Generate candidate evidence-acquisition actions.

Actions are generic (read a paper, search an archive, re-run an
analysis, ...) and are instantiated per evidence gap: for each
under-evidenced hypothesis the generator proposes the action types that
could plausibly close its gap. Cost/time/resource defaults come from an
action catalog that is fully configurable via YAML — nothing here is
domain-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schemas import ActionType, CandidateAction, EvidenceGap, GapReport


@dataclass
class ActionTemplate:
    """Catalog entry describing one generic action type."""

    action_type: ActionType
    description: str
    estimated_cost: float
    estimated_time: float
    required_resources: list[str] = field(default_factory=list)


DEFAULT_CATALOG: list[ActionTemplate] = [
    ActionTemplate(ActionType.READ_PAPER, "Read a relevant paper in depth", 0.1, 0.1, ["literature access"]),
    ActionTemplate(ActionType.SEARCH_ARCHIVE, "Search an archive for existing data or literature", 0.15, 0.15, ["archive access"]),
    ActionTemplate(ActionType.RERUN_ANALYSIS, "Re-run an existing analysis with alternative settings", 0.4, 0.35, ["compute", "original data"]),
    ActionTemplate(ActionType.COLLECT_OBSERVATION, "Collect an additional independent observation", 0.85, 0.9, ["instrument time"]),
    ActionTemplate(ActionType.RUN_SIMULATION, "Run a simulation to test the hypothesis mechanism", 0.5, 0.5, ["compute", "simulation code"]),
    ActionTemplate(ActionType.COMPARE_DATASETS, "Compare independent datasets for consistency", 0.3, 0.3, ["multiple datasets"]),
    ActionTemplate(ActionType.SEARCH_REVIEW, "Search review papers for consolidated evidence", 0.1, 0.1, ["literature access"]),
]

# Which action types plausibly address which kind of missing evidence.
# Keyed by keywords appearing in the gap's ``missing_evidence`` text; the
# fallback is the full catalog.
_GAP_ACTION_MAP: dict[str, list[ActionType]] = {
    "independent": [
        ActionType.COLLECT_OBSERVATION,
        ActionType.COMPARE_DATASETS,
        ActionType.SEARCH_ARCHIVE,
    ],
    "higher-confidence": [
        ActionType.RERUN_ANALYSIS,
        ActionType.RUN_SIMULATION,
        ActionType.READ_PAPER,
    ],
    "any evidence": [
        ActionType.SEARCH_ARCHIVE,
        ActionType.SEARCH_REVIEW,
        ActionType.READ_PAPER,
        ActionType.RUN_SIMULATION,
    ],
}


def load_catalog(path: str | Path) -> list[ActionTemplate]:
    """Load an action catalog from YAML.

    Expected structure::

        actions:
          - action_type: read_paper
            description: ...
            estimated_cost: 0.1
            estimated_time: 0.1
            required_resources: [literature access]
    """
    data = yaml.safe_load(Path(path).read_text())
    templates = []
    for row in data.get("actions", []):
        templates.append(
            ActionTemplate(
                action_type=ActionType(row["action_type"]),
                description=row["description"],
                estimated_cost=float(row["estimated_cost"]),
                estimated_time=float(row["estimated_time"]),
                required_resources=list(row.get("required_resources", [])),
            )
        )
    return templates


class ActionGenerator:
    """Instantiate candidate actions from a gap report."""

    def __init__(self, catalog: list[ActionTemplate] | None = None) -> None:
        self.catalog = catalog or DEFAULT_CATALOG
        self._by_type = {t.action_type: t for t in self.catalog}

    def generate(self, report: GapReport) -> list[CandidateAction]:
        actions: list[CandidateAction] = []
        counter = 0
        for gap in report.gaps:
            for template in self._templates_for_gap(gap):
                counter += 1
                actions.append(
                    CandidateAction(
                        action_id=f"{report.question_id}_a{counter:03d}",
                        action_type=template.action_type,
                        description=(
                            f"{template.description} to address gap "
                            f"'{gap.missing_evidence}' for hypothesis "
                            f"{gap.hypothesis_id}"
                        ),
                        estimated_cost=template.estimated_cost,
                        estimated_time=template.estimated_time,
                        required_resources=list(template.required_resources),
                        target_hypothesis=gap.hypothesis_id,
                        metadata={"gap_reason": gap.reason, "gap_confidence": gap.confidence},
                    )
                )
        return actions

    def _templates_for_gap(self, gap: EvidenceGap) -> list[ActionTemplate]:
        for keyword, action_types in _GAP_ACTION_MAP.items():
            if keyword in gap.missing_evidence:
                return [self._by_type[t] for t in action_types if t in self._by_type]
        return list(self.catalog)
