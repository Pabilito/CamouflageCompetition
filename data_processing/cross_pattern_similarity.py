"""
data_processing/cross_pattern_similarity.py
─────────────────────────────────────────────────────────────────────────────
Compute pairwise cosine similarity between adversarial images that belong to
different patterns, for every identity and every model.

Pattern assignment is read from the ``pattern_id`` column in the attacks
table (manually annotated in the UI).  Images with pattern_id == 0
(unclassified) are skipped.

For each identity that has images in at least two distinct patterns, every
cross-pattern (A ↔ B) image pair is scored using pre-computed embeddings
from the DB.  All ordered pairs of distinct pattern IDs are evaluated,
e.g. if an identity has patterns {1, 2, 3} the script computes 1↔2, 1↔3,
and 2↔3.

Produces two files:

  cross_pattern_pairs.csv
    One row per (identity, model, pattern_a, pattern_b, image_a, image_b)
    with the raw cosine similarity and whether it exceeds the per-model
    threshold.

  cross_pattern_summary.csv
    Aggregated statistics per (identity, model) and overall totals:
    n_pairs, mean_sim, std_sim, min_sim, max_sim, n_matched, accuracy.

Usage
─────
    python data_processing/cross_pattern_similarity.py [--db DB_PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    DEFAULT_DB, RESULTS_DIR,
    MODEL_ORDER, MODEL_SHORT,
    open_db, load_all_embeddings, cosine_similarity, threshold_for,
    ensure_results_dir,
)


# ── Build pattern groups from the DB ─────────────────────────────────────────

def _load_pattern_groups(con) -> dict[str, dict[int, list[str]]]:
    """Return {identity: {pattern_id: [image_path, ...]}} from the attacks
    table.  Only includes images with pattern_id > 0 (annotated)."""
    rows = con.execute(
        "SELECT DISTINCT identity_name, image_path, pattern_id "
        "FROM attacks "
        "WHERE identity_name IS NOT NULL AND identity_name != '' "
        "  AND pattern_id > 0"
    ).fetchall()

    groups: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, int]] = set()
    for r in rows:
        key = (r["identity_name"], r["image_path"], r["pattern_id"])
        if key not in seen:
            seen.add(key)
            groups[r["identity_name"]][r["pattern_id"]].append(r["image_path"])

    # Sort paths within each pattern for deterministic output
    for identity in groups:
        for pid in groups[identity]:
            groups[identity][pid].sort(key=lambda p: Path(p).name)

    return dict(groups)


# ── Core comparison ──────────────────────────────────────────────────────────

def _compare_all(con, emb_cache: dict) -> tuple[list[dict], list[dict]]:
    """Return (pair_rows, summary_rows)."""

    pattern_groups = _load_pattern_groups(con)

    # Keep only identities with ≥ 2 distinct patterns
    identities = sorted(
        (i for i, pats in pattern_groups.items() if len(pats) >= 2),
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else 0, x),
    )

    if not identities:
        print("No identities with ≥ 2 annotated patterns found.", file=sys.stderr)
        return [], []

    models = [m for m in MODEL_ORDER if m in emb_cache]
    if not models:
        print("No model embeddings found.", file=sys.stderr)
        return [], []

    print(f"Identities with cross-pattern data: {len(identities)}  |  "
          f"Models: {len(models)}")

    pair_rows: list[dict] = []
    # Per (identity, model)
    summary_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    match_acc:   dict[tuple[str, str], list[bool]]  = defaultdict(list)
    # Per (pattern_pair, model)  — pattern_pair is e.g. "1↔2"
    pp_sim_acc:  dict[tuple[str, str], list[float]] = defaultdict(list)
    pp_match_acc: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for identity in identities:
        pats = pattern_groups[identity]
        pat_ids = sorted(pats.keys())
        n_images = sum(len(v) for v in pats.values())
        print(f"  {identity}: patterns {pat_ids}  ({n_images} images)")

        # All ordered pairs of distinct pattern IDs
        for pid_a, pid_b in itertools.combinations(pat_ids, 2):
            cross_pairs = list(itertools.product(pats[pid_a], pats[pid_b]))
            pp_key = f"{pid_a}↔{pid_b}"

            for model in models:
                thr = threshold_for(model)
                model_embs = emb_cache.get(model, {})

                for path_a, path_b in cross_pairs:
                    ea = model_embs.get(path_a)
                    eb = model_embs.get(path_b)
                    if ea is None or eb is None:
                        continue

                    sim = cosine_similarity(ea, eb)
                    matched = sim >= thr

                    pair_rows.append({
                        "identity":    identity,
                        "model":       model,
                        "model_short": MODEL_SHORT.get(model, model),
                        "threshold":   f"{thr:.4f}",
                        "pattern_a":   pid_a,
                        "pattern_b":   pid_b,
                        "image_a":     Path(path_a).name,
                        "image_b":     Path(path_b).name,
                        "similarity":  f"{sim:.6f}",
                        "matched":     matched,
                        "path_a":      path_a,
                        "path_b":      path_b,
                    })

                    summary_acc[(identity, model)].append(sim)
                    match_acc[(identity, model)].append(matched)
                    pp_sim_acc[(pp_key, model)].append(sim)
                    pp_match_acc[(pp_key, model)].append(matched)

    # ── Build summary rows ────────────────────────────────────────────────
    summary_rows: list[dict] = []

    for identity in identities:
        for model in models:
            key = (identity, model)
            sims = summary_acc.get(key, [])
            matches = match_acc.get(key, [])
            if not sims:
                continue
            arr = np.array(sims)
            n_matched = sum(matches)
            summary_rows.append({
                "identity":    identity,
                "model":       model,
                "model_short": MODEL_SHORT.get(model, model),
                "threshold":   f"{threshold_for(model):.4f}",
                "n_pairs":     len(sims),
                "mean_sim":    f"{float(np.mean(arr)):.6f}",
                "std_sim":     f"{float(np.std(arr)):.6f}",
                "min_sim":     f"{float(np.min(arr)):.6f}",
                "max_sim":     f"{float(np.max(arr)):.6f}",
                "median_sim":  f"{float(np.median(arr)):.6f}",
                "n_matched":   n_matched,
                "accuracy":    f"{n_matched / len(sims):.6f}",
            })

    # ── Overall per-model totals ──────────────────────────────────────────
    for model in models:
        all_sims = [s for (i, m), ss in summary_acc.items()
                    if m == model for s in ss]
        all_matches = [b for (i, m), bs in match_acc.items()
                       if m == model for b in bs]
        if all_sims:
            arr = np.array(all_sims)
            n_matched = sum(all_matches)
            summary_rows.append({
                "identity":    "ALL",
                "model":       model,
                "model_short": MODEL_SHORT.get(model, model),
                "threshold":   f"{threshold_for(model):.4f}",
                "n_pairs":     len(all_sims),
                "mean_sim":    f"{float(np.mean(arr)):.6f}",
                "std_sim":     f"{float(np.std(arr)):.6f}",
                "min_sim":     f"{float(np.min(arr)):.6f}",
                "max_sim":     f"{float(np.max(arr)):.6f}",
                "median_sim":  f"{float(np.median(arr)):.6f}",
                "n_matched":   n_matched,
                "accuracy":    f"{n_matched / len(all_sims):.6f}",
            })

    # Grand total
    all_sims    = [s for ss in summary_acc.values() for s in ss]
    all_matches = [b for bs in match_acc.values() for b in bs]
    if all_sims:
        arr = np.array(all_sims)
        n_matched = sum(all_matches)
        summary_rows.append({
            "identity":    "ALL",
            "model":       "ALL",
            "model_short": "ALL",
            "threshold":   "",
            "n_pairs":     len(all_sims),
            "mean_sim":    f"{float(np.mean(arr)):.6f}",
            "std_sim":     f"{float(np.std(arr)):.6f}",
            "min_sim":     f"{float(np.min(arr)):.6f}",
            "max_sim":     f"{float(np.max(arr)):.6f}",
            "median_sim":  f"{float(np.median(arr)):.6f}",
            "n_matched":   n_matched,
            "accuracy":    f"{n_matched / len(all_sims):.6f}",
        })

    # ── Pattern-pair summary ──────────────────────────────────────────────
    # One row per (pattern_pair, model), plus per-pattern-pair totals,
    # per-model totals, and a grand total.
    pp_rows: list[dict] = []
    pattern_pairs = sorted({k[0] for k in pp_sim_acc.keys()})

    def _pp_summary_row(label: str, model: str, model_short: str,
                        threshold: str, sims: list[float],
                        matches: list[bool]) -> dict:
        arr = np.array(sims)
        n_matched = sum(matches)
        n = len(sims)
        return {
            "pattern_pair": label,
            "model":        model,
            "model_short":  model_short,
            "threshold":    threshold,
            "n_pairs":      n,
            "mean_sim":     f"{float(np.mean(arr)):.6f}",
            "std_sim":      f"{float(np.std(arr)):.6f}",
            "min_sim":      f"{float(np.min(arr)):.6f}",
            "max_sim":      f"{float(np.max(arr)):.6f}",
            "median_sim":   f"{float(np.median(arr)):.6f}",
            "n_matched":    n_matched,
            "accuracy":     f"{n_matched / n:.6f}" if n else "",
        }

    for pp in pattern_pairs:
        for model in models:
            sims    = pp_sim_acc.get((pp, model), [])
            matches = pp_match_acc.get((pp, model), [])
            if not sims:
                continue
            pp_rows.append(_pp_summary_row(
                pp, model, MODEL_SHORT.get(model, model),
                f"{threshold_for(model):.4f}", sims, matches))

        # Per-pattern-pair total (all models combined)
        all_pp_sims    = [s for m in models for s in pp_sim_acc.get((pp, m), [])]
        all_pp_matches = [b for m in models for b in pp_match_acc.get((pp, m), [])]
        if all_pp_sims:
            pp_rows.append(_pp_summary_row(
                pp, "ALL", "All Models", "", all_pp_sims, all_pp_matches))

    # Per-model totals across all pattern pairs
    for model in models:
        all_m_sims    = [s for pp in pattern_pairs for s in pp_sim_acc.get((pp, model), [])]
        all_m_matches = [b for pp in pattern_pairs for b in pp_match_acc.get((pp, model), [])]
        if all_m_sims:
            pp_rows.append(_pp_summary_row(
                "ALL", model, MODEL_SHORT.get(model, model),
                f"{threshold_for(model):.4f}", all_m_sims, all_m_matches))

    # Grand total
    if all_sims:
        pp_rows.append(_pp_summary_row(
            "ALL", "ALL", "All Models", "", all_sims, all_matches))

    return pair_rows, summary_rows, pp_rows, models, pattern_pairs


PAIR_FIELDS = [
    "identity", "model", "model_short", "threshold",
    "pattern_a", "pattern_b",
    "image_a", "image_b", "similarity", "matched",
    "path_a", "path_b",
]

SUMMARY_FIELDS = [
    "identity", "model", "model_short", "threshold",
    "n_pairs", "mean_sim", "std_sim", "min_sim", "max_sim", "median_sim",
    "n_matched", "accuracy",
]

PP_SUMMARY_FIELDS = [
    "pattern_pair", "model", "model_short", "threshold",
    "n_pairs", "mean_sim", "std_sim", "min_sim", "max_sim", "median_sim",
    "n_matched", "accuracy",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-pattern adversarial ↔ adversarial similarity.")
    parser.add_argument("--db",  default=DEFAULT_DB)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_results_dir(Path(args.out))
    con = open_db(db_path)

    print("Loading all embeddings from DB…")
    emb_cache = load_all_embeddings(con)
    n_models  = len(emb_cache)
    n_entries = sum(len(v) for v in emb_cache.values())
    print(f"Loaded {n_entries} embeddings across {n_models} model(s).\n")

    pair_rows, summary_rows, pp_rows, models, pattern_pairs = _compare_all(con, emb_cache)
    con.close()

    # ── Write pair-level CSV ──────────────────────────────────────────────
    pairs_path = out_dir / "cross_pattern_pairs.csv"
    with open(pairs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pair_rows)
    print(f"Wrote {pairs_path}  ({len(pair_rows)} rows)")

    # ── Write per-identity summary CSV ────────────────────────────────────
    summary_path = out_dir / "cross_pattern_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}  ({len(summary_rows)} rows)")

    # ── Write pattern-pair summary CSV ────────────────────────────────────
    pp_path = out_dir / "cross_pattern_by_pair.csv"
    with open(pp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PP_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(pp_rows)
    print(f"Wrote {pp_path}  ({len(pp_rows)} rows)")

    if not pair_rows:
        print("\nNo cross-pattern pairs found. Make sure at least some images "
              "have pattern_id annotated (≥ 2 distinct patterns per identity).")
        return

    # ── Stdout: per-model accuracy per pattern pair ───────────────────────
    short_names = [MODEL_SHORT.get(m, m) for m in models]
    model_header = "|".join(f"{s[:6]:>6}" for s in short_names)

    print(f"\n── Cross-pattern accuracy per model (per-model thresholds) ──")
    print(f"  {'pair':<10} {'n':>6}  {'mean':>10}  {'std':>10}  "
          f"accuracy ({model_header}| total)")
    print(f"  {'─' * 10} {'─' * 6}  {'─' * 10}  {'─' * 10}  "
          f"{'─' * (9 + 7 * len(models) + 7)}")

    # Pattern-pair rows (the "ALL" model rows give the combined view)
    for pp in pattern_pairs:
        _print_pp_line(pp, pp_rows, models)

    # Grand total line
    _print_pp_line("ALL", pp_rows, models)

    # ── Stdout: score statistics per pattern pair ─────────────────────────
    print(f"\n── Cross-pattern score statistics (per-model thresholds) ──")
    print(f"  {'pair':<10} {'n':>6}  {'mean':>10}  {'std':>10}  "
          f"{'min':>10}  {'median':>10}  {'max':>10}  "
          f"success  ({model_header}| total)")
    print(f"  {'─' * 10} {'─' * 6}  {'─' * 10}  {'─' * 10}  "
          f"{'─' * 10}  {'─' * 10}  {'─' * 10}  "
          f"{'─' * (9 + 7 * len(models) + 7)}")

    for pp in pattern_pairs:
        _print_pp_stats_line(pp, pp_rows, models)
    _print_pp_stats_line("ALL", pp_rows, models)


def _print_pp_line(pp_label: str, pp_rows: list[dict],
                   models: list[str]) -> None:
    """Print one accuracy summary line for a pattern pair."""
    # Per-model cells
    per_model_parts = []
    for model in models:
        r = next((r for r in pp_rows
                  if r["pattern_pair"] == pp_label and r["model"] == model), None)
        if r:
            per_model_parts.append(f"{r['n_matched']:>3}/{r['n_pairs']:<3}")
        else:
            per_model_parts.append(f"{'—':>7}")
    model_detail = "|".join(per_model_parts)

    # Combined row
    combined = next((r for r in pp_rows
                     if r["pattern_pair"] == pp_label and r["model"] == "ALL"), None)
    if combined:
        acc = float(combined["accuracy"]) * 100 if combined["accuracy"] else 0
        print(f"  {pp_label:<10} {combined['n_pairs']:>6}  "
              f"{combined['mean_sim']:>10}  {combined['std_sim']:>10}  "
              f"{acc:5.1f}%  ({model_detail}|"
              f"{combined['n_matched']:>4}/{combined['n_pairs']:<4})")


def _print_pp_stats_line(pp_label: str, pp_rows: list[dict],
                         models: list[str]) -> None:
    """Print one score-statistics line for a pattern pair."""
    per_model_parts = []
    total_matched = 0
    total_n = 0
    for model in models:
        r = next((r for r in pp_rows
                  if r["pattern_pair"] == pp_label and r["model"] == model), None)
        if r:
            per_model_parts.append(f"{r['n_matched']:>3}/{r['n_pairs']:<3}")
            total_matched += r["n_matched"]
            total_n += r["n_pairs"]
        else:
            per_model_parts.append(f"{'—':>7}")
    model_detail = "|".join(per_model_parts)

    combined = next((r for r in pp_rows
                     if r["pattern_pair"] == pp_label and r["model"] == "ALL"), None)
    if combined:
        asr_val = float(combined["accuracy"]) * 100 if combined["accuracy"] else 0
        # "accuracy" here means matched (recognised) rate; attack success = 1 - accuracy
        attack_success = 100.0 - asr_val
        print(f"  {pp_label:<10} {combined['n_pairs']:>6}  "
              f"{combined['mean_sim']:>10}  {combined['std_sim']:>10}  "
              f"{combined['min_sim']:>10}  {combined['median_sim']:>10}  "
              f"{combined['max_sim']:>10}  "
              f"{asr_val:5.1f}%  ({model_detail}|"
              f"{total_matched:>4}/{total_n:<4})")


if __name__ == "__main__":
    main()
