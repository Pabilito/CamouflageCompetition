"""
data_processing/gallery_search_stats.py
─────────────────────────────────────────────────────────────────────────────
Measure and visualise gallery search performance from the attacks table.

"Gallery position" is the rank assigned to the target identity in the gallery
search: position 1 means the model immediately recognised the attacker (worst
attack); high position means the model was confused with many other identities
(best attack).  Higher is always better for the attacker.

Statistics computed
───────────────────
  • Rank-N accuracy (N = 1 … 10)
        Fraction of (image, model) pairs where gallery_position ≥ N.
        "Position ≥ N" means the attack moved the true identity below N
        other gallery entries, i.e. the model was fooled by at least N − 1
        alternative identities.
  • Mean / std / median / min / max gallery position
  • Attack success rate (position > 1 = model did NOT immediately identify)

All three breakdowns:
  1. Overall   — one row per model + one ALL_MODELS aggregate
  2. Per model × pose   (frontal / profile / pitch)
  3. Per model × pattern

Two output formats per breakdown: heatmap PNG and table PNG.

Produces CSV:
  gallery_search_stats.csv   — all computed metrics, one row per
                                (model, group_by, group_value)

Produces PNGs (6 files):
  fig_gs_rank_n_heatmap.png / fig_gs_rank_n_table.png
  fig_gs_positions_heatmap.png / fig_gs_positions_table.png
  fig_gs_breakdown_heatmap.png / fig_gs_breakdown_table.png

Usage
─────
    python data_processing/gallery_search_stats.py [--db DB_PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    DEFAULT_DB, RESULTS_DIR,
    MODEL_ORDER, MODEL_SHORT,
    open_db, load_all_attack_rows, pose_label,
    ensure_results_dir,
)

RANK_N_VALUES = list(range(1, 11))   # Rank-1 … Rank-10

# ── Colour theme ──────────────────────────────────────────────────────────────
# Higher gallery position = better for attacker → use reversed map
# (red = position 1 / bad, green = high position / good)
CMAP_POS  = "RdYlGn"       # for absolute position heat
CMAP_RATE = "RdYlGn_r"     # for rank-N accuracy (lower % = greener = better attack)

BG   = "#0f1117"
FG   = "white"
MUTED = "#aaaaaa"

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold"}
FONT_HEADER = {"fontsize": 9, "fontweight": "bold"}


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading & metric computation
# ══════════════════════════════════════════════════════════════════════════════

def _build_records(rows: list[dict]) -> list[dict]:
    """
    Flatten the raw attack rows into one record per (image_path, model_name)
    that has a gallery_position.  Attach pose / pattern metadata.
    """
    records = []
    for r in rows:
        pos = r.get("gallery_position")
        if pos is None:
            continue
        gt = r.get("gallery_total") or 0
        records.append({
            "image_path":  r["image_path"],
            "model":       r["model_name"],
            "identity":    r.get("identity_name", ""),
            "position":    int(pos),
            "gallery_total": int(gt),
            "pose":        pose_label(r),
            "pattern_id":  r.get("pattern_id", 0) or 0,
        })
    return records


def _compute_metrics(records: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Compute gallery-search metrics for every (model, group_by, group_value).
    Also appends an ALL_MODELS aggregate row for each group.

    Returns (result_rows, models_present).
    """
    # Discover models present
    models_present = [m for m in MODEL_ORDER
                      if any(r["model"] == m for r in records)]

    # ── bucket: model → group_by → group_value → [positions]
    buckets: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in records:
        m   = r["model"]
        pos = r["position"]
        buckets[m]["overall"]["all"].append(pos)
        buckets[m]["pose"][r["pose"]].append(pos)
        pid = r["pattern_id"]
        buckets[m]["pattern"][str(pid) if pid > 0 else "unclassified"].append(pos)

    results: list[dict] = []

    def _row(model: str, model_short: str, group_by: str,
             group_value: str, positions: list[int]) -> dict:
        arr = np.array(positions, dtype=float)
        n   = len(arr)
        # Rank-N accuracy: fraction where position >= N (attack fooled >= N-1 ids)
        rank_n = {f"rank{k}_acc": float(np.mean(arr <= k)) if n else np.nan
                  for k in RANK_N_VALUES}
        # Attack success: position > 1 (model did NOT immediately recognise)
        asr = float(np.mean(arr > 1)) if n else np.nan
        row = {
            "model":        model,
            "model_short":  model_short,
            "group_by":     group_by,
            "group_value":  group_value,
            "n_images":     n,
            "mean_pos":     float(np.mean(arr))   if n else np.nan,
            "std_pos":      float(np.std(arr))    if n else np.nan,
            "median_pos":   float(np.median(arr)) if n else np.nan,
            "min_pos":      float(np.min(arr))    if n else np.nan,
            "max_pos":      float(np.max(arr))    if n else np.nan,
            "attack_success_rate": asr,
        }
        row.update(rank_n)
        return row

    # Per-model rows
    for model in models_present:
        short = MODEL_SHORT.get(model, model)
        for group_by in ("overall", "pose", "pattern"):
            for group_value, positions in sorted(buckets[model][group_by].items()):
                results.append(_row(model, short, group_by, group_value, positions))

    # ALL_MODELS aggregate: pool positions across models
    all_buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for model in models_present:
        for group_by in ("overall", "pose", "pattern"):
            for group_value, positions in buckets[model][group_by].items():
                all_buckets[group_by][group_value].extend(positions)

    for group_by in ("overall", "pose", "pattern"):
        for group_value, positions in sorted(all_buckets[group_by].items()):
            results.append(_row("ALL_MODELS", "All Models",
                                group_by, group_value, positions))

    return results, models_present


def _write_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    fieldnames = (
        ["model", "model_short", "group_by", "group_value", "n_images",
         "mean_pos", "std_pos", "median_pos", "min_pos", "max_pos",
         "attack_success_rate"]
        + [f"rank{k}_acc" for k in RANK_N_VALUES]
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            # Format floats
            row = dict(r)
            for key, val in row.items():
                if isinstance(val, float):
                    row[key] = "" if np.isnan(val) else f"{val:.6f}"
            w.writerow(row)
    print(f"Wrote {out_path}  ({len(results)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Rank-N accuracy heatmap / table
#  Rows = models (no ALL_MODELS),  Cols = Rank-1 … Rank-10
#  Cell = rank-N accuracy % (overall only)
# ══════════════════════════════════════════════════════════════════════════════

def _load_rank_n_overall(results: list[dict]) -> tuple[list, list, np.ndarray]:
    """
    Return (model_labels, rank_labels, values) for the rank-N overall heatmap.
    Rows = real models (not ALL_MODELS), Cols = Rank-1 … Rank-10.
    """
    model_labels = [m for m in MODEL_ORDER
                    if any(r["model"] == m and r["group_by"] == "overall"
                           for r in results)]
    rank_labels  = [f"Rank ≤ {k}" for k in RANK_N_VALUES]
    values       = np.full((len(model_labels), len(RANK_N_VALUES)), np.nan)

    mi = {m: i for i, m in enumerate(model_labels)}
    for r in results:
        if r["group_by"] != "overall" or r["model"] == "ALL_MODELS":
            continue
        i = mi.get(r["model"])
        if i is None:
            continue
        for j, k in enumerate(RANK_N_VALUES):
            v = r.get(f"rank{k}_acc")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values[i, j] = float(v) * 100

    return model_labels, rank_labels, values


def plot_rank_n_heatmap(results: list[dict], out_path: Path,
                        gallery_total: int = 0) -> None:
    model_labels, rank_labels, values = _load_rank_n_overall(results)
    n_rows, n_cols = values.shape

    fig_w = max(10, n_cols * 1.1 + 2.5)
    fig_h = max(3.5, n_rows * 0.85 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _dark(fig, ax)

    im = ax.imshow(values, aspect="auto", cmap=CMAP_RATE, vmin=0, vmax=100)

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v):
                continue
            color = "black" if 25 < v < 75 else "white"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(rank_labels, rotation=30, ha="right",
                       fontsize=9, color=FG)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=9, color=FG)
    ax.tick_params(colors=FG, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("Rank-N Accuracy (%)", color=FG, fontsize=9)
    _style_cbar(cbar)

    gt_note = f"  (gallery size ≈ {gallery_total})" if gallery_total else ""
    ax.set_title(f"Gallery Search — Rank-N Accuracy per Model{gt_note}",
                 color=FG, **FONT_TITLE, pad=12)
    _subtitle(ax, "Rank ≤ N accuracy = fraction where target ranked within top N  |  "
              "green = lower recognition rate = better attack")
    fig.tight_layout()
    _save(fig, out_path)


def plot_rank_n_table(results: list[dict], out_path: Path,
                      gallery_total: int = 0) -> None:
    model_labels, rank_labels, values_pct = _load_rank_n_overall(results)
    n_rows, n_cols = values_pct.shape
    values = values_pct / 100.0

    annots = [[""] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if not np.isnan(v):
                annots[i][j] = f"{v:.3f}"

    gt_note = f"  (gallery size ≈ {gallery_total})" if gallery_total else ""
    _plain_table(
        title=f"Gallery Search — Rank-N Accuracy per Model{gt_note}",
        subtitle="Rank ≤ N accuracy (0–1) — fraction where target ranked within top N  |  lower = better attack",
        row_labels=model_labels,
        col_labels=rank_labels,
        values=values,
        annots=annots,
        out_path=out_path,
        row_label_header="Model",
    )
def _load_position_summary(results: list[dict]) -> tuple[list, list, np.ndarray, list]:
    """
    Rows = all models + ALL_MODELS (overall only).
    Cols = Mean Pos / Median Pos / ASR% / Rank1 / Rank5 / Rank10.
    """
    row_labels = [m for m in MODEL_ORDER
                  if any(r["model"] == m and r["group_by"] == "overall"
                         for r in results)] + ["ALL_MODELS"]
    col_keys   = ["mean_pos", "median_pos", "attack_success_rate",
                  "rank1_acc", "rank5_acc", "rank10_acc"]
    col_labels = ["Mean Pos", "Median Pos", "ASR %",
                  "Rank ≤ 1 %", "Rank ≤ 5 %", "Rank ≤ 10 %"]

    # For normalisation direction: pos → lower is worse (model still recognises)
    # We flip for colour: want green = good for attacker
    n_rows = len(row_labels)
    n_cols = len(col_keys)
    raw    = np.full((n_rows, n_cols), np.nan)
    annots = [[""] * n_cols for _ in range(n_rows)]

    ri = {m: i for i, m in enumerate(row_labels)}
    for r in results:
        if r["group_by"] != "overall":
            continue
        i = ri.get(r["model"])
        if i is None:
            continue
        for j, key in enumerate(col_keys):
            v = r.get(key)
            if v is None:
                continue
            fv = float(v) if not isinstance(v, float) else v
            if np.isnan(fv):
                continue
            if key in ("mean_pos", "median_pos"):
                raw[i, j] = fv
                annots[i][j] = f"{fv:.1f}"
            else:
                raw[i, j] = fv * 100
                annots[i][j] = f"{fv * 100:.1f}%"

    return row_labels, col_labels, raw, annots


def plot_positions_heatmap(results: list[dict], out_path: Path) -> None:
    row_labels, col_labels, raw, annots = _load_position_summary(results)
    n_rows, n_cols = raw.shape

    fig_w = max(7, n_cols * 1.4 + 2.5)
    fig_h = max(3.5, n_rows * 0.85 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _dark(fig, ax)

    # Separate colour norms per column group
    # Cols 0-1: absolute position (low = bad = red for attacker)
    # Cols 2-5: rate (high = good = green for attacker)
    display = np.zeros_like(raw)
    # Build a normalised 0-1 display matrix to colour uniformly
    for j in range(n_cols):
        col = raw[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        lo, hi = valid.min(), valid.max()
        invert = (j < 2) or (j >= 3)   # pos cols and rank-N cols: lower = better attack; ASR col: higher = better
        if invert:
            # invert so green = better for attacker
            if hi > lo:
                display[:, j] = np.where(np.isnan(col), np.nan,
                                         1.0 - (col - lo) / (hi - lo))
            else:
                display[:, j] = np.where(np.isnan(col), np.nan, 0.5)
        else:
            # ASR: high = good for attacker
            if hi > lo:
                display[:, j] = np.where(np.isnan(col), np.nan,
                                         (col - lo) / (hi - lo))
            else:
                display[:, j] = np.where(np.isnan(col), np.nan, 0.5)

    cmap = plt.get_cmap(CMAP_POS)
    rgba = np.zeros((*display.shape, 4))
    for i in range(n_rows):
        for j in range(n_cols):
            v = display[i, j]
            rgba[i, j] = cmap(float(v)) if not np.isnan(v) else (0.15, 0.15, 0.15, 1)

    ax.imshow(rgba, aspect="auto")

    # Separator before ALL_MODELS
    sep = n_rows - 1
    ax.axhline(sep - 0.5, color=FG, lw=1.1, alpha=0.4)

    for i in range(n_rows):
        for j in range(n_cols):
            txt = annots[i][j]
            if not txt:
                continue
            bg = rgba[i, j]
            tc = "black" if _lum(bg) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=tc, fontsize=8.5)

    # Column separators between pos cols and rate cols
    ax.axvline(1.5, color=FG, lw=0.7, alpha=0.3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=25, ha="right",
                       fontsize=9, color=FG)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9, color=FG)
    ax.tick_params(colors=FG, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Gallery Search — Position Summary per Model",
                 color=FG, **FONT_TITLE, pad=12)
    _subtitle(ax, "Mean/Median: higher = better attack  |  "
              "ASR = position > 1  |  green = better for attacker  |  "
              "colour scaled within each column")
    fig.tight_layout()
    _save(fig, out_path)


def plot_positions_table(results: list[dict], out_path: Path) -> None:
    row_labels, col_labels, raw, _ = _load_position_summary(results)
    n_rows, n_cols = raw.shape

    # Build clean annotations: position cols as-is, rate cols as 0-1
    annots = [[""] * n_cols for _ in range(n_rows)]
    col_keys = ["mean_pos", "median_pos", "attack_success_rate",
                "rank1_acc", "rank5_acc", "rank10_acc"]
    ri_map = {m: i for i, m in enumerate(row_labels)}
    for r in results:
        if r["group_by"] != "overall":
            continue
        i = ri_map.get(r["model"])
        if i is None:
            continue
        for j, key in enumerate(col_keys):
            v = r.get(key)
            if v is None:
                continue
            fv = float(v) if not isinstance(v, float) else v
            if np.isnan(fv):
                continue
            if key in ("mean_pos", "median_pos"):
                annots[i][j] = f"{fv:.2f}"
            else:
                annots[i][j] = f"{fv:.3f}"

    # Values: keep raw (positions as raw numbers, rates as 0-1 fractions)
    # But _plain_table just uses annots, so pass raw directly
    all_idx = row_labels.index("ALL_MODELS") if "ALL_MODELS" in row_labels else None

    _plain_table(
        title="Gallery Search — Position Summary per Model",
        subtitle="Mean/Median position (higher = better attack)  |  ASR and Rank-N as fractions (0–1)",
        row_labels=row_labels,
        col_labels=col_labels,
        values=raw,
        annots=annots,
        out_path=out_path,
        sep_before_rows=[all_idx] if all_idx is not None else None,
        bold_rows=[all_idx] if all_idx is not None else None,
        col_group_seps=[2],   # separator between position cols and rate cols
        row_label_header="Model",
    )
def _load_breakdown(results: list[dict], group_by: str,
                    metric_key: str = "rank1_acc"
                    ) -> tuple[list, list, np.ndarray, list]:
    """
    Rows = real models (not ALL_MODELS).
    Cols = group values for `group_by` (pose labels or pattern IDs).
    Cell = metric_key value (0-100 %).
    Annotation = "mean_pos / metric_key%".
    """
    model_labels = [m for m in MODEL_ORDER
                    if any(r["model"] == m and r["group_by"] == group_by
                           for r in results)]
    gvs = sorted(set(
        r["group_value"] for r in results
        if r["group_by"] == group_by and r["model"] != "ALL_MODELS"
    ))
    n_rows = len(model_labels)
    n_cols = len(gvs)
    values = np.full((n_rows, n_cols), np.nan)
    annots = [[""] * n_cols for _ in range(n_rows)]

    mi = {m: i for i, m in enumerate(model_labels)}
    gi = {g: j for j, g in enumerate(gvs)}

    for r in results:
        if r["group_by"] != group_by or r["model"] == "ALL_MODELS":
            continue
        i = mi.get(r["model"])
        j = gi.get(r["group_value"])
        if i is None or j is None:
            continue
        v_rate = r.get(metric_key)
        v_mean = r.get("mean_pos")
        if v_rate is not None and not (isinstance(v_rate, float) and np.isnan(v_rate)):
            pct = float(v_rate) * 100
            values[i, j] = pct
            mean_s = f"{float(v_mean):.1f}" if v_mean and not np.isnan(float(v_mean)) else "?"
            annots[i][j] = f"{pct:.1f}%\nμ={mean_s}"

    return model_labels, gvs, values, annots


def _plot_breakdown_panel(ax, model_labels, col_labels, values, annots,
                          title: str) -> plt.cm.ScalarMappable:
    n_rows, n_cols = values.shape
    im = ax.imshow(values, aspect="auto", cmap=CMAP_RATE, vmin=0, vmax=100)

    for i in range(n_rows):
        for j in range(n_cols):
            txt = annots[i][j]
            if not txt:
                continue
            v = values[i, j]
            tc = "black" if 25 < v < 75 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=tc, fontsize=7.5, linespacing=1.4)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right",
                       fontsize=9, color=FG)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=9, color=FG)
    ax.tick_params(colors=FG, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=8)
    return im


def plot_breakdown_heatmap(results: list[dict], out_path: Path) -> None:
    """Two-panel figure: left = pose breakdown, right = pattern breakdown."""
    ml_pose, gv_pose, val_pose, ann_pose = _load_breakdown(results, "pose")
    ml_pat,  gv_pat,  val_pat,  ann_pat  = _load_breakdown(results, "pattern")

    n_pose_cols = len(gv_pose)
    n_pat_cols  = len(gv_pat)
    total_cols  = n_pose_cols + n_pat_cols

    fig_w = max(10, total_cols * 1.4 + 3)
    fig_h = max(4,  max(len(ml_pose), len(ml_pat)) * 0.85 + 2.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                   gridspec_kw={"width_ratios": [n_pose_cols, n_pat_cols]})
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    im1 = _plot_breakdown_panel(ax1, ml_pose, gv_pose, val_pose, ann_pose,
                                "By Pose")
    im2 = _plot_breakdown_panel(ax2, ml_pat,  gv_pat,  val_pat,  ann_pat,
                                "By Pattern")

    fig.suptitle("Gallery Search — Attack Success Rate Breakdown "
                 "(Rank ≤ 1 Accuracy)",
                 color=FG, **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             "ASR % = position > 1  |  μ = mean gallery position  |  "
             "green = better for attacker",
             ha="center", fontsize=7.5, color=MUTED, style="italic")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], pad=0.02, fraction=0.02)
    cbar.set_label("Attack Success Rate (%)", color=FG, fontsize=9)
    _style_cbar(cbar)

    fig.tight_layout()
    _save(fig, out_path)


def _plot_breakdown_table_panel(ax, model_labels, col_labels, values, annots,
                                title: str, cmap, norm) -> None:
    n_rows, n_cols = values.shape
    ax.axis("off")
    ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=6)

    col_w = 1.0 / (n_cols + 1.0)
    row_h = 1.0 / (n_rows + 1.5)
    x0    = col_w * 1.0
    y_top = 0.88

    ax.text(0.01, y_top + row_h * 0.5, "Model", color=FG, **FONT_HEADER,
            va="center", transform=ax.transAxes)
    for j, lbl in enumerate(col_labels):
        xc = x0 + j * col_w + col_w / 2
        ax.text(xc, y_top + row_h * 0.5, lbl, color=FG, **FONT_HEADER,
                ha="center", va="center", transform=ax.transAxes, rotation=30)

    for i, m in enumerate(model_labels):
        yc = y_top - (i + 0.5) * row_h
        ax.text(0.01, yc, m, color=FG, fontsize=8.5, va="center",
                transform=ax.transAxes)
        for j in range(n_cols):
            xc = x0 + j * col_w
            v  = values[i, j]
            bg = cmap(norm(v)) if not np.isnan(v) else (0.15, 0.15, 0.15, 1)
            _cell_rect(ax, xc, yc, col_w, row_h, bg)
            txt = annots[i][j]
            if txt:
                ax.text(xc + col_w / 2, yc, txt,
                        color="black" if _lum(bg) > 0.5 else "white",
                        fontsize=7, ha="center", va="center",
                        transform=ax.transAxes, linespacing=1.3, zorder=3)


def plot_breakdown_table(results: list[dict], out_path: Path) -> None:
    ml_pose, gv_pose, val_pose_pct, ann_pose_pct = _load_breakdown(results, "pose")
    ml_pat,  gv_pat,  val_pat_pct,  ann_pat_pct  = _load_breakdown(results, "pattern")

    # Convert values to 0-1; rebuild annotations
    val_pose = val_pose_pct / 100.0
    val_pat  = val_pat_pct  / 100.0

    def _rebuild_annots(results, group_by, model_labels, gvs):
        mi = {m: i for i, m in enumerate(model_labels)}
        gi = {g: j for j, g in enumerate(gvs)}
        ann = [[""] * len(gvs) for _ in range(len(model_labels))]
        for r in results:
            if r["group_by"] != group_by or r["model"] == "ALL_MODELS":
                continue
            i = mi.get(r["model"])
            j = gi.get(r["group_value"])
            if i is None or j is None:
                continue
            v_rate = r.get("rank1_acc")
            v_mean = r.get("mean_pos")
            if v_rate is not None and not (isinstance(v_rate, float) and np.isnan(float(v_rate))):
                pct = float(v_rate)
                mean_s = f"{float(v_mean):.1f}" if v_mean and not np.isnan(float(v_mean)) else "?"
                ann[i][j] = "{:.3f}\nμ={}".format(pct, mean_s)
        return ann

    ann_pose = _rebuild_annots(results, "pose",    ml_pose, gv_pose)
    ann_pat  = _rebuild_annots(results, "pattern", ml_pat,  gv_pat)

    # Two separate tables saved side by side as one figure via subplots
    import matplotlib.pyplot as plt
    import numpy as np

    BG = "#0f1117"
    MUTED = "#888888"

    def _sub_table(ax, model_labels, col_labels, values, annots, subtitle):
        """Draw a plain table into an existing axes (no file save)."""
        n_rows_t = len(model_labels)
        n_cols_t = len(col_labels)
        FG = "white"
        HDR_BG = "#1a1e2e"
        ALT_BG = "#14161f"
        RULE   = "#444444"
        AMBER  = "#ffdd88"

        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Layout in normalised coords
        LBL_W = 0.30
        col_w = (1.0 - LBL_W) / max(n_cols_t, 1)
        hdr_h = 0.12
        row_h = (1.0 - hdr_h - 0.08) / max(n_rows_t, 1)

        hdr_y  = 1.0 - hdr_h / 2 - 0.04
        rule_y = 1.0 - hdr_h - 0.04

        # Header bg
        from matplotlib.patches import FancyBboxPatch as FBP
        ax.add_patch(FBP((0, rule_y), 1.0, hdr_h + 0.04,
                         boxstyle="square,pad=0", linewidth=0,
                         facecolor=HDR_BG, zorder=1, transform=ax.transAxes))

        ax.text(0.01, hdr_y, "Model", ha="left", va="center",
                fontsize=7.5, fontweight="bold", color="#00d4ff",
                transform=ax.transAxes, zorder=2)
        for j, lbl in enumerate(col_labels):
            cx = LBL_W + (j + 0.5) * col_w
            ax.text(cx, hdr_y, lbl, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="#00d4ff",
                    transform=ax.transAxes, zorder=2)

        ax.plot([0, 1], [rule_y, rule_y], color=RULE, lw=1.0,
                transform=ax.transAxes, zorder=3)
        ax.plot([LBL_W - 0.01, LBL_W - 0.01],
                [rule_y - n_rows_t * row_h, rule_y],
                color=RULE, lw=0.7, transform=ax.transAxes, zorder=3)

        for i, m in enumerate(model_labels):
            yc = rule_y - (i + 0.5) * row_h
            if i % 2 == 1:
                y0 = rule_y - (i + 1) * row_h
                ax.add_patch(FBP((0, y0), 1.0, row_h,
                                 boxstyle="square,pad=0", linewidth=0,
                                 facecolor=ALT_BG, zorder=0,
                                 transform=ax.transAxes))
            ax.text(0.01, yc, m, ha="left", va="center",
                    fontsize=7, color=FG, transform=ax.transAxes, zorder=2)
            for j in range(n_cols_t):
                cx = LBL_W + (j + 0.5) * col_w
                v   = values[i, j]
                txt = annots[i][j] if annots[i][j] else (f"{v:.3f}" if not np.isnan(v) else "—")
                ax.text(cx, yc, txt, ha="center", va="center",
                        fontsize=7, color=FG, transform=ax.transAxes,
                        zorder=2, linespacing=1.3)

        bot_y = rule_y - n_rows_t * row_h
        ax.plot([0, 1], [bot_y, bot_y], color=RULE, lw=0.5,
                transform=ax.transAxes, zorder=3)
        ax.set_title(subtitle, color=MUTED, fontsize=8, pad=4)

    n_pose_cols = max(len(gv_pose), 1)
    n_pat_cols  = max(len(gv_pat),  1)
    fig_w = 5 + (n_pose_cols + n_pat_cols) * 1.3
    fig_h = max(3.0, max(len(ml_pose), len(ml_pat)) * 0.38 + 1.8)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [n_pose_cols + 2, n_pat_cols + 2]})
    fig.patch.set_facecolor(BG)

    _sub_table(ax1, ml_pose, gv_pose, val_pose, ann_pose, "By Pose")
    _sub_table(ax2, ml_pat,  gv_pat,  val_pat,  ann_pat,  "By Pattern")

    fig.suptitle(
        "Gallery Search — Attack Success Rate Breakdown (Rank ≤ 1 Accuracy)",
        color="white", fontsize=11, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "Values: fraction (0–1)  +  μ = mean gallery position  |  lower fraction = better attack",
             ha="center", fontsize=7.5, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")
def _plain_table(title: str, subtitle: str,
                 row_labels: list[str], col_labels: list[str],
                 values: "np.ndarray",            # 2-D, 0-1 floats or NaN
                 annots: "list[list[str]]",        # same shape, display strings
                 out_path: "Path",
                 sep_before_rows: "list[int] | None" = None,
                 bold_rows: "list[int] | None" = None,
                 col_group_seps: "list[int] | None" = None,
                 row_label_header: str = "",
                 fmt: str = ".3f") -> None:
    """
    Render a clean dark-background table with no cell colours.

    Parameters
    ----------
    sep_before_rows  : row indices before which a horizontal rule is drawn
    bold_rows        : row indices whose label is rendered bold + amber
    col_group_seps   : column indices before which a vertical rule is drawn
    fmt              : number format when annots entry is empty (fallback)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_rows = len(row_labels)
    n_cols = len(col_labels)

    BG      = "#0f1117"
    FG      = "white"
    MUTED   = "#888888"
    AMBER   = "#ffdd88"
    RULE    = "#444444"
    HDR_BG  = "#1a1e2e"
    ALT_BG  = "#14161f"   # alternating row tint

    # ── figure sizing ─────────────────────────────────────────────────────────
    # Aim for ~1.6 cm per data column, ~2.8 cm for row-label column, 0.5 cm per row
    COL_W_IN  = 1.55          # inches per data column
    ROW_LBL_W = max(1.8, max(len(l) for l in row_labels) * 0.11 + 0.3)
    HDR_H_IN  = 0.55          # header row height
    ROW_H_IN  = 0.38          # data row height
    TOP_PAD   = 0.65          # title + subtitle
    BOT_PAD   = 0.15

    fig_w = ROW_LBL_W + n_cols * COL_W_IN + 0.3
    fig_h = TOP_PAD + HDR_H_IN + n_rows * ROW_H_IN + BOT_PAD

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # ── coordinate helpers ────────────────────────────────────────────────────
    def col_x(j: int) -> float:
        """Left edge of data column j."""
        return ROW_LBL_W + j * COL_W_IN

    def col_cx(j: int) -> float:
        return col_x(j) + COL_W_IN / 2

    def row_y(i: int) -> float:
        """Centre y of data row i (0 = top)."""
        return fig_h - TOP_PAD - HDR_H_IN - (i + 0.5) * ROW_H_IN

    hdr_y = fig_h - TOP_PAD - HDR_H_IN / 2

    # ── title & subtitle ──────────────────────────────────────────────────────
    ax.text(fig_w / 2, fig_h - 0.12, title,
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=FG, transform=ax.transData)
    ax.text(fig_w / 2, fig_h - 0.38, subtitle,
            ha="center", va="top", fontsize=7.5, color=MUTED,
            style="italic", transform=ax.transData)

    # ── header background ─────────────────────────────────────────────────────
    from matplotlib.patches import FancyBboxPatch as FBP
    hdr_rect = FBP((0, fig_h - TOP_PAD - HDR_H_IN),
                   fig_w, HDR_H_IN,
                   boxstyle="square,pad=0", linewidth=0,
                   facecolor=HDR_BG, zorder=1)
    ax.add_patch(hdr_rect)

    # ── alternating row backgrounds ───────────────────────────────────────────
    for i in range(n_rows):
        if i % 2 == 1:
            y0 = fig_h - TOP_PAD - HDR_H_IN - (i + 1) * ROW_H_IN
            row_rect = FBP((0, y0), fig_w, ROW_H_IN,
                           boxstyle="square,pad=0", linewidth=0,
                           facecolor=ALT_BG, zorder=0)
            ax.add_patch(row_rect)

    # ── header row label ──────────────────────────────────────────────────────
    ax.text(0.08, hdr_y, row_label_header,
            ha="left", va="center", fontsize=8.5, fontweight="bold",
            color="#00d4ff", transform=ax.transData, zorder=2)

    # ── column headers ────────────────────────────────────────────────────────
    for j, lbl in enumerate(col_labels):
        ax.text(col_cx(j), hdr_y, lbl,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#00d4ff", transform=ax.transData, zorder=2)

    # ── vertical separator after row-label column ─────────────────────────────
    sep_top = fig_h - TOP_PAD
    sep_bot = fig_h - TOP_PAD - HDR_H_IN - n_rows * ROW_H_IN
    ax.plot([ROW_LBL_W - 0.05, ROW_LBL_W - 0.05], [sep_bot, sep_top],
            color=RULE, lw=0.8, zorder=3)

    # ── optional column group separators ─────────────────────────────────────
    for j in (col_group_seps or []):
        x = col_x(j) - 0.04
        ax.plot([x, x], [sep_bot, sep_top], color=RULE, lw=0.8,
                ls="--", alpha=0.6, zorder=3)

    # ── data rows ─────────────────────────────────────────────────────────────
    for i, lbl in enumerate(row_labels):
        yc     = row_y(i)
        is_bold = bold_rows and i in bold_rows
        lc = AMBER if is_bold else FG
        fw = "bold" if is_bold else "normal"

        # row label
        ax.text(0.08, yc, lbl,
                ha="left", va="center", fontsize=8, color=lc,
                fontweight=fw, transform=ax.transData, zorder=2)

        for j in range(n_cols):
            v   = values[i, j]
            txt = annots[i][j] if annots[i][j] else (f"{v:{fmt}}" if not np.isnan(v) else "—")
            ax.text(col_cx(j), yc, txt,
                    ha="center", va="center", fontsize=8, color=FG,
                    transform=ax.transData, zorder=2)

    # ── horizontal rules ─────────────────────────────────────────────────────
    # Rule below header
    rule_y = fig_h - TOP_PAD - HDR_H_IN
    ax.plot([0, fig_w], [rule_y, rule_y], color=RULE, lw=1.0, zorder=3)
    # Rule at bottom
    bot_y = fig_h - TOP_PAD - HDR_H_IN - n_rows * ROW_H_IN
    ax.plot([0, fig_w], [bot_y, bot_y], color=RULE, lw=0.6, zorder=3)
    # Optional separator rules inside data
    for i in (sep_before_rows or []):
        y = fig_h - TOP_PAD - HDR_H_IN - i * ROW_H_IN
        ax.plot([0, fig_w], [y, y], color="#666666", lw=0.8,
                ls="--", zorder=3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Shared drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dark(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 1.01, text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, color=MUTED, style="italic")


def _style_cbar(cbar) -> None:
    cbar.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)


def _add_colorbar_axis(fig, cmap, norm, label: str,
                       ticks=None) -> None:
    cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    kw = {"ticks": ticks} if ticks is not None else {}
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", **kw)
    cbar.set_label(label, color=FG, fontsize=8)
    cbar.ax.xaxis.set_tick_params(color=FG, labelsize=7)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color=FG)


def _cell_rect(ax, xc, yc, col_w, row_h, bg) -> None:
    rect = FancyBboxPatch(
        (xc + 0.003, yc - row_h * 0.43),
        col_w - 0.006, row_h * 0.86,
        boxstyle="round,pad=0.005",
        linewidth=0, transform=ax.transAxes,
        facecolor=bg, zorder=2)
    ax.add_patch(rect)


def _lum(rgba) -> float:
    return 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]


def _save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  stdout summary (mirrors the console-output style of the other scripts)
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(results: list[dict], models_present: list[str]) -> None:
    shorts = [MODEL_SHORT.get(m, m) for m in models_present]

    # ── Overall position + ASR ────────────────────────────────────────────
    print("\n── Gallery Search — overall position stats per model ──")
    hdr = f"  {'model':<30}  {'n':>5}  {'mean':>7}  {'median':>7}  "
    hdr += "  ".join(f"R≤{k:2d}%" for k in [1, 2, 5, 10])
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for model in models_present + ["ALL_MODELS"]:
        r = next((x for x in results
                  if x["model"] == model and x["group_by"] == "overall"), None)
        if not r:
            continue
        n    = r["n_images"]
        mean = float(r["mean_pos"])   if r["mean_pos"]   else 0
        med  = float(r["median_pos"]) if r["median_pos"] else 0
        parts = [f"{float(r.get(f'rank{k}_acc', 0)) * 100:5.1f}%"
                 for k in [1, 2, 5, 10]]
        label = MODEL_SHORT.get(model, model) if model != "ALL_MODELS" else "All Models"
        print(f"  {label:<30}  {n:>5}  {mean:>7.2f}  {med:>7.1f}  " +
              "  ".join(parts))

    # ── Per-pose ──────────────────────────────────────────────────────────
    print("\n── Gallery Search — breakdown by pose ──")
    poses = sorted(set(r["group_value"] for r in results
                       if r["group_by"] == "pose" and r["model"] != "ALL_MODELS"))
    for pose in poses:
        print(f"  {pose}:")
        for model in models_present:
            r = next((x for x in results
                      if x["model"] == model and x["group_by"] == "pose"
                      and x["group_value"] == pose), None)
            if not r:
                continue
            parts = [f"{float(r.get(f'rank{k}_acc', 0)) * 100:5.1f}%"
                     for k in [1, 2, 5, 10]]
            label = MODEL_SHORT.get(model, model)
            mean  = float(r["mean_pos"]) if r["mean_pos"] else 0
            print(f"    {label:<28}  n={r['n_images']:>4}  mean={mean:.1f}  " +
                  "  ".join(parts))

    # ── Per-pattern ───────────────────────────────────────────────────────
    print("\n── Gallery Search — breakdown by pattern ──")
    pats = sorted(set(r["group_value"] for r in results
                      if r["group_by"] == "pattern" and r["model"] != "ALL_MODELS"))
    for pat in pats:
        print(f"  pattern {pat}:")
        for model in models_present:
            r = next((x for x in results
                      if x["model"] == model and x["group_by"] == "pattern"
                      and x["group_value"] == pat), None)
            if not r:
                continue
            parts = [f"{float(r.get(f'rank{k}_acc', 0)) * 100:5.1f}%"
                     for k in [1, 2, 5, 10]]
            label = MODEL_SHORT.get(model, model)
            mean  = float(r["mean_pos"]) if r["mean_pos"] else 0
            print(f"    {label:<28}  n={r['n_images']:>4}  mean={mean:.1f}  " +
                  "  ".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gallery search statistics and figures.")
    parser.add_argument("--db",  default=DEFAULT_DB)
    parser.add_argument("--out", default=str(_SCRIPT_DIR / "results"))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_results_dir(Path(args.out))

    con     = open_db(db_path)
    rows    = load_all_attack_rows(con)
    con.close()

    if not rows:
        print("No attack records found.", file=sys.stderr)
        sys.exit(1)

    records = _build_records(rows)
    if not records:
        print("No gallery_position values found in attack records.\n"
              "Run 'Recompute Positions' in the Gallery Search tab first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} (image, model) records with gallery positions.")

    # Derive gallery total for axis labels
    gallery_total = int(np.median([r["gallery_total"] for r in records
                                   if r["gallery_total"] > 0]) or 0)

    results, models_present = _compute_metrics(records)
    print(f"Computed metrics for {len(results)} (model, group, value) combinations.")

    # ── CSV ───────────────────────────────────────────────────────────────
    _write_csv(results, out_dir / "gallery_search_stats.csv")

    # ── stdout summary ────────────────────────────────────────────────────
    _print_summary(results, models_present)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures…")

    print("\n[1] Rank-N accuracy (overall)")
    plot_rank_n_heatmap(results, out_dir / "fig_gs_rank_n_heatmap.png",
                        gallery_total)
    plot_rank_n_table(results,  out_dir / "fig_gs_rank_n_table.png",
                      gallery_total)

    print("\n[2] Position summary (mean / median / ASR / Rank-1,5,10)")
    plot_positions_heatmap(results, out_dir / "fig_gs_positions_heatmap.png")
    plot_positions_table(results,   out_dir / "fig_gs_positions_table.png")

    print("\n[3] Breakdown by pose and pattern")
    plot_breakdown_heatmap(results, out_dir / "fig_gs_breakdown_heatmap.png")
    plot_breakdown_table(results,   out_dir / "fig_gs_breakdown_table.png")

    print(f"\n{'=' * 60}")
    print(f"Done.  Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
