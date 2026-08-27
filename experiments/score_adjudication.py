"""Score human adjudications against Paper 3's retrieval-bounded label.

Answers the question the scaled benchmark could not: is the planner's
anti-correlation with measured attention genuine neglect, or an artifact
of a label that under-counts engagement for specific questions?

The decisive quantity is the **recovery rate**: among questions Paper 3
labels ``not_addressed``, the fraction a human finds were actually
engaged by post-cutoff literature. Paper 3's own Paper 1 import implies
this rate can be high for object-level questions (it recovered 8/10).

    recovery rate ~ 0      -> label is sound here; neglect reading survives
    recovery rate high     -> label under-counts; anti-correlation is
                              substantially artifact, as suspected

The script also reports the false-positive rate on the control arm
(questions labeled addressed that the human finds unengaged), and
re-runs the stratum's utility comparison under the *corrected* labels,
which is the number that actually updates the paper.

Usage::

    python experiments/score_adjudication.py [annotations.json]

Annotation format (list, or the export from the annotation tool)::

    [{"question_id": "...", "engaged": "yes|partial|no|unsure",
      "evidence_ids": "...", "notes": "..."}]

Partial counts as engaged by default (matching Paper 3's ``addressed``,
which includes ``partially_addressed``); ``unsure`` rows are excluded
from rates and reported separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from paper3_benchmark import permutation_pvalue  # noqa: E402

ADJ = REPO_ROOT / "experiments" / "data" / "adjudication"
ENGAGED = {"yes", "partial"}
UNENGAGED = {"no"}


def load_annotations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("annotations", payload) if isinstance(payload, dict) else payload
    return [r for r in rows if r.get("engaged")]


def run_sensitivity(scored: list[dict], key: dict) -> dict:
    """Recompute the headline numbers under a lenient and a strict bar.

    Counting ``partial`` as engaged matches how the original label treats
    ``partially_addressed``, but it is also the permissive choice, and a
    recovery rate is only convincing if it survives the strict reading
    where solely unambiguous direct engagement counts.
    """
    out = {}
    for name, engaged_set in (
        ("lenient_yes_partial", {"yes", "partial"}),
        ("strict_yes_only", {"yes"}),
    ):
        verdict = {r["question_id"]: r["engaged"].lower() for r in scored}
        arm = [q for q in verdict if not key[q]["paper3_addressed"]]
        recovered = [q for q in arm if verdict[q] in engaged_set]
        control = [q for q in verdict if key[q]["paper3_addressed"]]
        contested = [q for q in control if verdict[q] not in engaged_set]
        a = [key[q]["planner_top_k_utility"] for q in verdict if verdict[q] in engaged_set]
        b = [key[q]["planner_top_k_utility"] for q in verdict if verdict[q] not in engaged_set]
        entry = {
            "recovery": f"{len(recovered)}/{len(arm)}",
            "recovery_rate": round(len(recovered) / len(arm), 3) if arm else None,
            "control_contested": f"{len(contested)}/{len(control)}",
        }
        if len(a) >= 5 and len(b) >= 5:
            entry["utility_engaged_mean"] = round(sum(a) / len(a), 4)
            entry["utility_unengaged_mean"] = round(sum(b) / len(b), 4)
            entry["permutation_p"] = round(permutation_pvalue(a, b), 4)
        out[name] = entry
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ADJ / "annotations.json"
    if not path.exists():
        print(f"No annotations at {path}.")
        print("Run experiments/build_adjudication_set.py, annotate, then re-run.")
        return

    key = json.loads((ADJ / "label_key.json").read_text())
    annotations = load_annotations(path)

    unknown = [r["question_id"] for r in annotations if r["question_id"] not in key]
    if unknown:
        print(f"warning: {len(unknown)} unknown question_id(s), ignored")
    annotations = [r for r in annotations if r["question_id"] in key]

    unsure = [r for r in annotations if r["engaged"].lower() not in ENGAGED | UNENGAGED]
    scored = [r for r in annotations if r["engaged"].lower() in ENGAGED | UNENGAGED]

    sensitivity = run_sensitivity(scored, key)

    # Recovery arm: label says not_addressed.
    recovery_arm = [r for r in scored if not key[r["question_id"]]["paper3_addressed"]]
    recovered = [r for r in recovery_arm if r["engaged"].lower() in ENGAGED]
    # Control arm: label says addressed.
    control_arm = [r for r in scored if key[r["question_id"]]["paper3_addressed"]]
    false_pos = [r for r in control_arm if r["engaged"].lower() in UNENGAGED]

    print(f"Adjudicated {len(scored)} of 62 high-specificity questions"
          f" ({len(unsure)} unsure, excluded)\n")

    if recovery_arm:
        rate = len(recovered) / len(recovery_arm)
        print(f"RECOVERY RATE  {len(recovered)}/{len(recovery_arm)} = {rate:.0%}")
        print("  (labeled not_addressed, human finds post-cutoff engagement)")
        if rate >= 0.30:
            verdict = ("label under-counts substantially here -- the "
                       "anti-correlation is substantially a measurement artifact")
        elif rate >= 0.15:
            verdict = ("label under-counts moderately -- the neglect reading is "
                       "weakened but not eliminated")
        else:
            verdict = ("label holds up here -- the neglect reading survives this "
                       "check")
        print(f"  verdict: {verdict}\n")

    if control_arm:
        fp = len(false_pos) / len(control_arm)
        print(f"CONTROL ARM    {len(false_pos)}/{len(control_arm)} = {fp:.0%} "
              "false positives")
        print("  (labeled addressed, human finds no real engagement)\n")

    # Re-run the stratum comparison under corrected labels.
    corrected = {
        r["question_id"]: r["engaged"].lower() in ENGAGED for r in scored
    }
    a = [key[q]["planner_top_k_utility"] for q, eng in corrected.items() if eng]
    b = [key[q]["planner_top_k_utility"] for q, eng in corrected.items() if not eng]
    if len(a) >= 5 and len(b) >= 5:
        diff = sum(a) / len(a) - sum(b) / len(b)
        p = permutation_pvalue(a, b)
        print("UTILITY COMPARISON UNDER CORRECTED LABELS")
        print(f"  engaged (n={len(a)})   mean top-K utility {sum(a)/len(a):.3f}")
        print(f"  unengaged (n={len(b)}) mean top-K utility {sum(b)/len(b):.3f}")
        print(f"  difference {diff:+.3f}, permutation p={p:.4f}")
        print("  (original stratum result: 1.830 vs 1.925, p=0.0044)")
        if p >= 0.05:
            print("  -> effect does NOT survive correction: report as artifact")
        else:
            print("  -> effect survives correction: neglect reading is supported")
    else:
        print("Not enough adjudicated rows in both arms for the utility test yet.")

    print("\nSENSITIVITY TO THE ENGAGEMENT BAR")
    for name, entry in sensitivity.items():
        line = (f"  {name:<22} recovery {entry['recovery']} "
                f"({entry['recovery_rate']:.0%})")
        if "permutation_p" in entry:
            line += (f"  utility {entry['utility_engaged_mean']:.3f} vs "
                     f"{entry['utility_unengaged_mean']:.3f}, "
                     f"p={entry['permutation_p']}")
        print(line)

    out = ADJ / "adjudication_result.json"
    out.write_text(json.dumps({
        "n_scored": len(scored),
        "n_unsure": len(unsure),
        "recovery": {"n": len(recovery_arm), "recovered": len(recovered)},
        "control": {"n": len(control_arm), "contested": len(false_pos)},
        "sensitivity": sensitivity,
        "corrected_labels": corrected,
    }, indent=2))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
