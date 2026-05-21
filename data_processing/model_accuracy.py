"""
data_processing/model_accuracy.py
─────────────────────────────────────────────────────────────────────────────
Compute recognition accuracy for each model, broken down by:
  • Overall (all images)
  • Per pose  (frontal / profile / pitch)
  • Per pattern number  (1–5)

"Accuracy" = fraction of adversarial images where the model still recognises
the attacker (similarity_score ≥ per-model threshold).  Lower accuracy means
the adversarial attacks are more effective.

Produces: model_accuracy.csv
  One row per (model, grouping, group_value) with columns:
    model, group_by, group_value, n_images, n_recognised, accuracy,
    mean_score, std_score, min_score, max_score, median_score

Usage
─────
    python data_processing/model_accuracy.py [--db DB_PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    DEFAULT_DB, RESULTS_DIR,
    MODEL_ORDER, MODEL_SHORT, MODEL_THRESHOLDS, DEFAULT_THRESHOLD,
    POSE_LABELS, POSE_COLUMNS,
    open_db, load_all_attack_rows, threshold_for, is_match, pose_label,
    ensure_results_dir,
)


def _compute_accuracy(rows: list[dict]) -> list[dict]:
    """Build accuracy table from raw attack rows.

    Each attack row has one model + one image.  We group by model and then
    by several facets (overall / pose / pattern).
    """
    # model → group_by → group_value → [scores]
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for r in rows:
        model = r["model_name"]
        score = r["similarity_score"]

        buckets[model]["overall"]["all"].append(score)
        buckets[model]["pose"][pose_label(r)].append(score)

        pid = r.get("pattern_id", 0)
        if pid > 0:
            buckets[model]["pattern"][str(pid)].append(score)
        else:
            buckets[model]["pattern"]["unclassified"].append(score)

        # joint pattern × pose breakdown
        pat_key = str(pid) if pid and pid > 0 else "unclassified"
        joint_key = f"{pat_key}/{pose_label(r)}"
        buckets[model]["pattern_pose"][joint_key].append(score)

    results: list[dict] = []
    for model in MODEL_ORDER:
        if model not in buckets:
            continue
        thr = threshold_for(model)
        for group_by in ("overall", "pose", "pattern", "pattern_pose"):
            items = buckets[model][group_by]
            for group_value in sorted(items.keys()):
                scores = np.array(items[group_value])
                n = len(scores)
                recognised = int(np.sum(scores >= thr))
                results.append({
                    "model":        model,
                    "model_short":  MODEL_SHORT.get(model, model),
                    "threshold":    f"{thr:.4f}",
                    "group_by":     group_by,
                    "group_value":  group_value,
                    "n_images":     n,
                    "n_recognised": recognised,
                    "accuracy":     f"{recognised / n:.6f}" if n else "",
                    "mean_score":   f"{float(np.mean(scores)):.6f}" if n else "",
                    "std_score":    f"{float(np.std(scores)):.6f}" if n else "",
                    "min_score":    f"{float(np.min(scores)):.6f}" if n else "",
                    "max_score":    f"{float(np.max(scores)):.6f}" if n else "",
                    "median_score": f"{float(np.median(scores)):.6f}" if n else "",
                })

    # ── ALL_MODELS combined rows ──────────────────────────────────────────
    # Each (model, image) pair is one observation; "recognised" is judged
    # using that model's own threshold.
    combined: dict[str, dict[str, tuple[list[float], int]]] = defaultdict(
        lambda: defaultdict(lambda: ([], 0))
    )
    for model in MODEL_ORDER:
        if model not in buckets:
            continue
        thr = threshold_for(model)
        for group_by in ("overall", "pose", "pattern", "pattern_pose"):
            for group_value, scores_list in buckets[model][group_by].items():
                existing_scores, existing_rec = combined[group_by][group_value]
                existing_scores.extend(scores_list)
                recognised = sum(1 for s in scores_list if s >= thr)
                combined[group_by][group_value] = (
                    existing_scores, existing_rec + recognised
                )

    for group_by in ("overall", "pose", "pattern", "pattern_pose"):
        for group_value in sorted(combined[group_by].keys()):
            scores_list, recognised = combined[group_by][group_value]
            if not scores_list:
                continue
            scores = np.array(scores_list)
            n = len(scores)
            results.append({
                "model":        "ALL_MODELS",
                "model_short":  "All Models",
                "threshold":    "",
                "group_by":     group_by,
                "group_value":  group_value,
                "n_images":     n,
                "n_recognised": recognised,
                "accuracy":     f"{recognised / n:.6f}" if n else "",
                "mean_score":   f"{float(np.mean(scores)):.6f}" if n else "",
                "std_score":    f"{float(np.std(scores)):.6f}" if n else "",
                "min_score":    f"{float(np.min(scores)):.6f}" if n else "",
                "max_score":    f"{float(np.max(scores)):.6f}" if n else "",
                "median_score": f"{float(np.median(scores)):.6f}" if n else "",
            })

    return results


FIELDNAMES = [
    "model", "model_short", "threshold",
    "group_by", "group_value",
    "n_images", "n_recognised", "accuracy",
    "mean_score", "std_score", "min_score", "max_score", "median_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Model accuracy by pose and pattern.")
    parser.add_argument("--db",  default=DEFAULT_DB)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_results_dir(Path(args.out))
    con = open_db(db_path)
    rows = load_all_attack_rows(con)
    con.close()

    if not rows:
        print("No attack records found.", file=sys.stderr)
        sys.exit(1)

    results = _compute_accuracy(rows)

    out_path = out_dir / "model_accuracy.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {out_path}  ({len(results)} rows)")

    # ── Print a quick summary to stdout ───────────────────────────────────
    print("\n── Overall accuracy per model ──")
    for r in results:
        if r["group_by"] == "overall":
            acc = float(r["accuracy"]) * 100 if r["accuracy"] else 0
            print(f"  {r['model_short']:<18} {acc:5.1f}%  "
                  f"({r['n_recognised']}/{r['n_images']} recognised)  "
                  f"mean={r['mean_score']}  std={r['std_score']}")


if __name__ == "__main__":
    main()
