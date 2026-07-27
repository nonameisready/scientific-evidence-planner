# Data schemas

All schemas are dataclasses in `src/planner/schemas.py` with
`to_dict`/`from_dict` for JSON round-tripping. They are the only coupling
between modules.

## Question

| Field | Type | Description |
| --- | --- | --- |
| `question_id` | str | Unique id (e.g. `astro_001`) |
| `question` | str | The scientific question text |
| `domain` | str | Free-form domain tag, default `generic` |
| `metadata` | dict | Anything else; unknown loader keys land here |

## Hypothesis

| Field | Type | Description |
| --- | --- | --- |
| `hypothesis_id` | str | Unique id (e.g. `H1`) |
| `description` | str | Statement of the hypothesis |
| `prior` | float | Current probability mass (normalized across the set) |
| `supporting_evidence` | list[str] | Evidence-graph node ids supporting it |
| `contradicting_evidence` | list[str] | Node ids contradicting it |
| `metadata` | dict | Free-form |

## EvidenceNode

| Field | Type | Description |
| --- | --- | --- |
| `node_id` | str | Unique id |
| `node_type` | enum | `paper`, `observation`, `dataset`, `claim`, `hypothesis` |
| `label` | str | Human-readable label |
| `timestamp` | str \| None | ISO date or year; `None` = structural node, never filtered by the cutoff |
| `metadata` | dict | Free-form |

## EvidenceEdge

| Field | Type | Description |
| --- | --- | --- |
| `source`, `target` | str | Node ids (must exist in the graph) |
| `edge_type` | enum | `supports`, `contradicts`, `derived_from`, `observes` |
| `confidence` | float | In [0, 1], validated on insert |
| `metadata` | dict | Free-form |

Edge direction conventions: evidence points *at* the hypothesis it
supports/contradicts; a derived node points *at* its origin
(`claim --derived_from--> paper --derived_from--> dataset`); an observation
points at what it observes.

## EvidenceGap / GapReport

`GapReport` holds a `question_id` and a confidence-sorted list of
`EvidenceGap` entries:

| Field | Type | Description |
| --- | --- | --- |
| `hypothesis_id` | str | The under-evidenced hypothesis |
| `missing_evidence` | str | What kind of evidence is missing |
| `confidence` | float | How certain the analyzer is that the gap is real |
| `reason` | str | Human-readable explanation |

## CandidateAction

| Field | Type | Description |
| --- | --- | --- |
| `action_id` | str | Unique id |
| `action_type` | enum | `read_paper`, `search_archive`, `rerun_analysis`, `collect_observation`, `run_simulation`, `compare_datasets`, `search_review` |
| `description` | str | What the action does and which gap it addresses |
| `estimated_cost` | float | Relative cost in [0, 1] |
| `estimated_time` | float | Relative time in [0, 1] |
| `required_resources` | list[str] | Resource tags (e.g. `compute`) |
| `target_hypothesis` | str \| None | Hypothesis the action informs |
| `metadata` | dict | Includes `gap_reason`, `gap_confidence` |

## UtilityBreakdown

Per-component scores: `information_gain` (bits), `novelty`, `tractability`,
`cost` (each [0, 1]-ish) and the weighted `total`.

## Recommendation

| Field | Type | Description |
| --- | --- | --- |
| `action` | CandidateAction | The recommended action |
| `utility` | UtilityBreakdown | Its component scores |
| `reasoning` | str | Human-readable justification |
| `rank` | int | 1-based rank in the returned list |

## NewEvidence

| Field | Type | Description |
| --- | --- | --- |
| `evidence_id` | str | Id, recorded on updated hypotheses |
| `description` | str | What was observed |
| `likelihoods` | dict[str, float] | `hypothesis_id -> P(evidence \| hypothesis)`; missing entries default to 1.0 |
| `metadata` | dict | Free-form |

## GroundTruthEvidence (evaluation)

| Field | Type | Description |
| --- | --- | --- |
| `evidence_id` | str | Id of the hidden post-cutoff evidence |
| `hypothesis_id` | str | Hypothesis it bears on |
| `action_type` | str | Action type that historically produced it |
| `likelihoods` | dict | For replaying through the belief updater |
| `metadata` | dict | Free-form (description, timestamp, ...) |
