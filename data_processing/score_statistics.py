"""
data_processing/score_statistics.py
─────────────────────────────────────────────────────────────────────────────
Compute model-independent score statistics for each pose and pattern.

For every adversarial image we first average its similarity scores across all
models to get a single model-independent score.  We then aggregate those
averages by pose and by pattern.

Attack success is evaluated per-model using each model's own LFW threshold:
an image fools a model when its score for that model is below the model's
threshold.  The CSV and stdout summary report both per-model and overall
success counts.

Produces: score_statistics.csv
  One row per (group_by, group_value) with columns for score stats,
  per-model success counts/rates, and combined totals.

Usage
─────
    python data_processing/score_statistics.py [--db DB_PATH] [--out DIR]
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
    MODEL_ORDER, MODEL_SHORT,
    open_db, load_all_attack_rows, pose_label, threshold_for,
    ensure_results_dir,
)


# ── Per-image record with model-level detail ─────────────────────────────────

def _build_image_records(rows: list[dict]) -> list[dict]:
    """Collapse per-model attack rows into one record per image.

    Each record contains:
      - avg_score: mean similarity across models
      - model_scores: {model_name: score}
      - model_fooled: {model_name: bool}  (score < threshold)
      - n_models_fooled: int
      - pose / pattern metadata
    """
    img_scores: dict[str, dict[str, float]] = defaultdict(dict)
    img_meta: dict[str, dict] = {}

    for r in rows:
        path = r["image_path"]
        model = r["model_name"]
        img_scores[path][model] = r["similarity_score"]
        if path not in img_meta:
            img_meta[path] = {
                "image_path": path,
                "identity":   r["identity_name"],
                "pattern_id": r.get("pattern_id", 0),
                "is_frontal": r.get("is_frontal", 0),
                "is_profile": r.get("is_profile", 0),
                "is_pitch":   r.get("is_pitch", 0),
            }

    records = []
    for path, scores in img_scores.items():
        meta = img_meta[path]
        fooled = {
            m: scores[m] < threshold_for(m)
            for m in scores
        }
        meta["avg_score"]       = float(np.mean(list(scores.values())))
        meta["model_scores"]    = scores
        meta["model_fooled"]    = fooled
        meta["n_models_fooled"] = sum(fooled.values())
        meta["n_models"]        = len(scores)
        records.append(meta)
    return records


# ── Aggregation ──────────────────────────────────────────────────────────────

def _compute_stats(records: list[dict]) -> list[dict]:
    """Group records by pose and pattern, compute statistics."""

    # Discover which models are present
    models_present = []
    for m in MODEL_ORDER:
        if any(m in rec["model_scores"] for rec in records):
            models_present.append(m)

    # group_by → group_value → [records]
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        buckets["overall"]["all"].append(rec)
        buckets["pose"][pose_label(rec)].append(rec)

        pid = rec.get("pattern_id", 0)
        if pid > 0:
            buckets["pattern"][str(pid)].append(rec)
        else:
            buckets["pattern"]["unclassified"].append(rec)

    results: list[dict] = []
    for group_by in ("overall", "pose", "pattern"):
        items = buckets[group_by]
        for gv in sorted(items.keys()):
            recs = items[gv]
            n = len(recs)
            avg_scores = np.array([r["avg_score"] for r in recs])

            row: dict = {
                "group_by":    group_by,
                "group_value": gv,
                "n_images":    n,
                "mean":        f"{float(np.mean(avg_scores)):.6f}" if n else "",
                "std":         f"{float(np.std(avg_scores)):.6f}" if n else "",
                "min":         f"{float(np.min(avg_scores)):.6f}" if n else "",
                "q25":         f"{float(np.percentile(avg_scores, 25)):.6f}" if n else "",
                "median":      f"{float(np.median(avg_scores)):.6f}" if n else "",
                "q75":         f"{float(np.percentile(avg_scores, 75)):.6f}" if n else "",
                "max":         f"{float(np.max(avg_scores)):.6f}" if n else "",
            }

            # Per-model success counts
            total_fooled = 0
            total_evaluated = 0
            for m in models_present:
                short = MODEL_SHORT.get(m, m)
                fooled = sum(1 for r in recs if r["model_fooled"].get(m, False))
                evaluated = sum(1 for r in recs if m in r["model_scores"])
                row[f"success_{short}"] = fooled
                row[f"n_{short}"]       = evaluated
                row[f"rate_{short}"]    = (
                    f"{fooled / evaluated:.6f}" if evaluated else ""
                )
                total_fooled += fooled
                total_evaluated += evaluated

            row["total_successes"]  = total_fooled
            row["total_evaluated"]  = total_evaluated
            row["attack_success_rate"] = (
                f"{total_fooled / total_evaluated:.6f}"
                if total_evaluated else ""
            )

            results.append(row)

    return results, models_present


def _build_fieldnames(models_present: list[str]) -> list[str]:
    fields = [
        "group_by", "group_value", "n_images",
        "mean", "std", "min", "q25", "median", "q75", "max",
    ]
    for m in models_present:
        short = MODEL_SHORT.get(m, m)
        fields.extend([f"success_{short}", f"n_{short}", f"rate_{short}"])
    fields.extend(["total_successes", "total_evaluated", "attack_success_rate"])
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-independent score statistics per pose and pattern.")
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

    records = _build_image_records(rows)
    results, models_present = _compute_stats(records)

    fieldnames = _build_fieldnames(models_present)
    out_path = out_dir / "score_statistics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {out_path}  ({len(results)} rows)")

    # ── Quick summary ─────────────────────────────────────────────────────
    short_names = [MODEL_SHORT.get(m, m) for m in models_present]
    model_header = "|".join(f"{s[:6]:>6}" for s in short_names)

    print(f"\n── Score statistics (per-model thresholds) ──")
    print(f"  {'group':<25} {'n':>5}  {'mean':>10}  {'std':>10}  "
          f"success  ({model_header}| total)")
    print(f"  {'─' * 25} {'─' * 5}  {'─' * 10}  {'─' * 10}  "
          f"{'─' * (9 + 7 * len(models_present) + 7)}")

    for r in results:
        tag = f"{r['group_by']}/{r['group_value']}"
        n = r["n_images"]

        # Per-model breakdown: "x/y"
        per_model_parts = []
        for m in models_present:
            short = MODEL_SHORT.get(m, m)
            s = r.get(f"success_{short}", 0)
            e = r.get(f"n_{short}", 0)
            per_model_parts.append(f"{s:>3}/{e:<3}")

        model_detail = "|".join(per_model_parts)

        total_s = r["total_successes"]
        total_e = r["total_evaluated"]
        asr = float(r["attack_success_rate"]) * 100 if r["attack_success_rate"] else 0

        print(f"  {tag:<25} {n:>5}  {r['mean']:>10}  {r['std']:>10}  "
              f"{asr:5.1f}%  ({model_detail}|{total_s:>4}/{total_e:<4})")


if __name__ == "__main__":
    main()
