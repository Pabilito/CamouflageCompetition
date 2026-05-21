"""
data_processing/visualize_results.py
─────────────────────────────────────────────────────────────────────────────
Generate figures for three analyses, each in two formats: heatmap and table.

Figures produced (saved to the same output directory as the CSVs):
  1. fig_overall_accuracy_heatmap.png   — heatmap
  2. fig_overall_accuracy_table.png     — table
  3. fig_cross_pattern_heatmap.png      — heatmap
  4. fig_cross_pattern_table.png        — table

Each figure covers:
  • "Overall accuracy per model"
      Rows: models  |  Cols: overall, pose groups, pattern groups
      Cell value: accuracy % + n_recognised/n_images

  • "Cross-pattern accuracy per model"
      Rows: pattern pair (+ ALL)  |  Cols: models
      Cell value: match rate % + n_matched/n_pairs

Usage
─────
    python data_processing/visualize_results.py [--out DIR]

The script reads the CSV files that were previously written to --out.

Defaults:
    --out  data_processing/results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib import ticker

# ── Palette ───────────────────────────────────────────────────────────────────
# Green = model recognises / matches  (bad for adversarial attack)
# Red   = attack succeeds / model fooled  (good for attacker)
# We use a diverging map centered at 50 %.

CMAP_ACCURACY  = "RdYlGn"          # high accuracy = green (model wins)
CMAP_ASR       = "RdYlGn_r"        # high attack-success rate = red (attacker wins)
CMAP_XPATT     = "RdYlGn"          # cross-pattern: high match rate = green

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold"}
FONT_LABEL  = {"fontsize": 9}
FONT_CELL   = {"fontsize": 8.5}
FONT_HEADER = {"fontsize": 9, "fontweight": "bold"}

_SCRIPT_DIR = Path(__file__).resolve().parent


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [WARN] Not found: {path}", file=sys.stderr)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ══════════════════════════════════════════════════════════════════════════════
#  1. OVERALL ACCURACY  (model_accuracy.csv)
# ══════════════════════════════════════════════════════════════════════════════

def _load_overall_accuracy(rows: list[dict]) -> tuple[list, list, np.ndarray, list]:
    """
    Returns (model_labels, group_labels, values_2d, annotation_2d).
    Rows = individual models + "Average" average row.
    Cols = overall / pose groups / pattern groups.
    """
    # Collect individual models in canonical order
    model_order = [m for m in MODEL_ORDER
                   if any(r["model"] == m for r in rows)]

    # Collect groups in stable order: overall first, then pose, then pattern
    group_order = []
    group_label_map: dict[str, str] = {}
    for gb_prefix, label_prefix in [("overall", ""), ("pose", "pose:"), ("pattern", "pat:")]:
        gvs = sorted(set(
            r["group_value"] for r in rows
            if r["group_by"] == gb_prefix and r["model"] != "ALL_MODELS"
        ))
        for gv in gvs:
            key = f"{gb_prefix}/{gv}"
            group_order.append(key)
            group_label_map[key] = f"{label_prefix}{gv}" if label_prefix else gv

    # Individual models + "Average" row at bottom
    n_models = len(model_order) + 1   # +1 for All Models
    n_groups = len(group_order)
    values   = np.full((n_models, n_groups), np.nan)
    annots   = [[""] * n_groups for _ in range(n_models)]

    m_idx = {m: i for i, m in enumerate(model_order)}
    g_idx = {g: i for i, g in enumerate(group_order)}
    all_row = len(model_order)  # index of the All Models row

    for r in rows:
        key = f"{r['group_by']}/{r['group_value']}"
        gi = g_idx.get(key)
        if gi is None or not r.get("accuracy"):
            continue
        acc_frac = float(r["accuracy"])   # 0-1
        ann_s = f"{acc_frac*100:.1f}%"
        if r["model"] == "ALL_MODELS":
            values[all_row, gi] = acc_frac
            annots[all_row][gi] = ann_s
        else:
            mi = m_idx.get(r["model"])
            if mi is not None:
                values[mi, gi] = acc_frac
                annots[mi][gi] = ann_s

    col_labels     = [group_label_map[g] for g in group_order]
    display_labels = [MODEL_SHORT.get(m, m) for m in model_order] + ["Average"]
    return display_labels, col_labels, values, annots


def plot_overall_accuracy_heatmap(rows: list[dict], out_path: Path) -> None:
    model_labels, col_labels, values, annots = _load_overall_accuracy(rows)
    n_rows, n_cols = values.shape

    fig_w = max(10, n_cols * 1.4)
    fig_h = max(4,  n_rows * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(values, aspect="auto", cmap=CMAP_ACCURACY, vmin=0, vmax=1)

    # Annotations
    for i in range(n_rows):
        for j in range(n_cols):
            txt = annots[i][j]
            if not txt:
                continue
            v = values[i, j]
            cmap_obj = plt.get_cmap(CMAP_ACCURACY)
            rgba = cmap_obj(v)
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            color = "black" if lum > 0.45 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right",
                       fontsize=9, color="white")
    ax.set_yticks(range(n_rows))
    yticklabels = [
        plt.Text(0, i, lbl, color="#ffdd88" if lbl == "Average" else "white",
                 fontsize=9, fontweight="bold" if lbl == "Average" else "normal")
        for i, lbl in enumerate(model_labels)
    ]
    ax.set_yticklabels(model_labels, fontsize=9, color="white")
    # Colour "Average" tick label amber
    for tick, lbl in zip(ax.get_yticklabels(), model_labels):
        tick.set_color("#ffdd88" if lbl == "Average" else "white")
        tick.set_fontweight("bold" if lbl == "Average" else "normal")
    ax.tick_params(colors="white", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Separator before All Models row
    ax.axhline(n_rows - 1.5, color="white", lw=0.8, alpha=0.4)
    # Column-type separators
    _add_col_separators(ax, col_labels, n_rows)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Accuracy (0–1)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Overall Accuracy per Model", color="white", **FONT_TITLE, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_overall_accuracy_table(rows: list[dict], out_path: Path) -> None:
    model_labels, col_labels, values_pct, _ = _load_overall_accuracy(rows)
    n_rows, n_cols = values_pct.shape

    values = values_pct  # already 0-1

    # Re-load raw counts for annotations
    m_idx = {m: i for i, m in enumerate(model_labels)}
    group_order, group_label_map = [], {}
    for gb_prefix, label_prefix in [("overall", ""), ("pose", "pose:"), ("pattern", "pat:")]:
        gvs = sorted(set(r["group_value"] for r in rows
                         if r["group_by"] == gb_prefix and r["model"] != "ALL_MODELS"))
        for gv in gvs:
            key = f"{gb_prefix}/{gv}"
            group_order.append(key)
            group_label_map[key] = f"{label_prefix}{gv}" if label_prefix else gv
    g_idx = {g: i for i, g in enumerate(group_order)}

    annots = [[""] * n_cols for _ in range(n_rows)]
    for r in rows:
        if r["model"] == "ALL_MODELS":
            continue
        key = f"{r['group_by']}/{r['group_value']}"
        mi = m_idx.get(r["model"])
        gi = g_idx.get(key)
        if mi is None or gi is None:
            continue
        if r["accuracy"]:
            acc = float(r["accuracy"])
            annots[mi][gi] = "{:.3f}".format(acc)

    # Column group separators: find where pose/pattern sections start
    col_group_seps = []
    prev_prefix = None
    for j, key in enumerate(group_order):
        prefix = key.split("/")[0]
        if prefix != prev_prefix and j > 0:
            col_group_seps.append(j)
        prev_prefix = prefix

    all_idx = len(model_labels) - 1  # All Models is last row
    _plain_table(
        title="Overall Accuracy per Model",
        row_labels=model_labels,
        col_labels=[group_label_map[g] for g in group_order],
        values=values,
        annots=annots,
        out_path=out_path,
        col_group_seps=col_group_seps,
        sep_before_rows=[all_idx],
        bold_rows=[all_idx],
        row_label_header="Model",
    )
def _load_score_statistics(rows: list[dict], model_order: list[str]
                           ) -> tuple[list, list, np.ndarray, list]:
    """
    Returns (group_labels, model_labels, values_2d, annots_2d).
    Rows = groups, Cols = models.
    Cell = attack-success rate (fooled/evaluated).
    """
    # Group order
    group_order = []
    for gb_prefix in ("overall", "pose", "pattern"):
        gvs = sorted(set(r["group_value"] for r in rows if r["group_by"] == gb_prefix))
        for gv in gvs:
            group_order.append((gb_prefix, gv))

    # Model column discovery: pick from MODEL_ORDER columns present in rows
    short_to_full: dict[str, str] = {}
    for r in rows:
        for key in r:
            if key.startswith("rate_"):
                short = key[5:]
                short_to_full[short] = short  # short names used in CSV
    # Preserve canonical model order where possible
    col_labels: list[str] = []
    for m in model_order:
        # Try matching by MODEL_SHORT suffix
        for short in list(short_to_full.keys()):
            if m.lower().replace("_", "") in short.lower().replace("_", "") or \
               short.lower().replace("_", "") in m.lower().replace("_", ""):
                if short not in col_labels:
                    col_labels.append(short)
                break
    # Fall back: just use all rate_ columns in order
    if not col_labels:
        col_labels = sorted(short_to_full.keys())

    n_rows_out = len(group_order)
    n_cols     = len(col_labels)
    values     = np.full((n_rows_out, n_cols), np.nan)
    annots     = [[""] * n_cols for _ in range(n_rows_out)]

    row_map  = {(gb, gv): i for i, (gb, gv) in enumerate(group_order)}

    for r in rows:
        ri = row_map.get((r["group_by"], r["group_value"]))
        if ri is None:
            continue
        for ci, short in enumerate(col_labels):
            rate_key    = f"rate_{short}"
            success_key = f"success_{short}"
            n_key       = f"n_{short}"
            rate_str    = r.get(rate_key, "")
            if not rate_str:
                continue
            asr = float(rate_str) * 100
            values[ri, ci] = asr
            n_s = r.get(success_key, "?")
            n_e = r.get(n_key, "?")
            annots[ri][ci] = f"{asr:.1f}%"

    group_labels = [f"{'pose:' if gb=='pose' else 'pat:' if gb=='pattern' else ''}{gv}"
                    for gb, gv in group_order]
    return group_labels, col_labels, values, annots


def plot_score_statistics_heatmap(rows: list[dict], model_order: list[str],
                                  out_path: Path) -> None:
    group_labels, col_labels, values, annots = _load_score_statistics(rows, model_order)
    n_rows, n_cols = values.shape

    fig_w = max(8, n_cols * 1.6)
    fig_h = max(4, n_rows * 0.85 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(values, aspect="auto", cmap=CMAP_ASR, vmin=0, vmax=100)

    for i in range(n_rows):
        for j in range(n_cols):
            txt = annots[i][j]
            if not txt:
                continue
            v = values[i, j]
            cmap_obj = plt.get_cmap(CMAP_ACCURACY)
            rgba = cmap_obj(v)
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            color = "black" if lum > 0.45 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right",
                       fontsize=9, color="white")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(group_labels, fontsize=9, color="white")
    ax.tick_params(colors="white", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _add_row_separators(ax, group_labels, n_cols)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Attack Success Rate (%)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Score Statistics — Attack Success Rate per Model",
                 color="white", **FONT_TITLE, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_score_statistics_table(rows: list[dict], model_order: list[str],
                                out_path: Path) -> None:
    group_labels, col_labels, values_pct, _ = _load_score_statistics(rows, model_order)
    n_rows, n_cols = values_pct.shape
    values = values_pct / 100.0

    # Re-build annotations with 0-1 values
    row_map = {lbl: i for i, lbl in enumerate(group_labels)}
    annots  = [[""] * n_cols for _ in range(n_rows)]
    for r in rows:
        gb, gv = r["group_by"], r["group_value"]
        glbl = f"{'pose:' if gb=='pose' else 'pat:' if gb=='pattern' else ''}{gv}"
        ri = row_map.get(glbl)
        if ri is None:
            continue
        for ci, short in enumerate(col_labels):
            rate_str = r.get(f"rate_{short}", "")
            if not rate_str:
                continue
            asr = float(rate_str)
            n_s = r.get(f"success_{short}", "?")
            n_e = r.get(f"n_{short}", "?")
            annots[ri][ci] = "{:.3f}".format(asr)

    # Separator before pattern section
    sep_rows = []
    prev_prefix = None
    for i, lbl in enumerate(group_labels):
        prefix = lbl.split(":")[0] if ":" in lbl else "overall"
        if prefix != prev_prefix and i > 0:
            sep_rows.append(i)
        prev_prefix = prefix

    _plain_table(
        title="Score Statistics — Attack Success Rate per Model",
        row_labels=group_labels,
        col_labels=col_labels,
        values=values,
        annots=annots,
        out_path=out_path,
        sep_before_rows=sep_rows,
        row_label_header="Group",
    )
def _load_cross_pattern(rows: list[dict]) -> tuple[list, list, np.ndarray, list]:
    """
    Returns (pair_labels, model_labels, values_2d, annots_2d).
    Rows = pattern pairs (+ ALL pair), Cols = individual models + "Average".
    Cell = match rate (stored as % for heatmap colouring).
    """
    # Build ordered pair list; move ALL to end
    pair_labels_seen: list[str] = []
    for r in rows:
        pp = r["pattern_pair"]
        if pp not in pair_labels_seen:
            pair_labels_seen.append(pp)
    if "ALL" in pair_labels_seen:
        pair_labels_seen.remove("ALL")
        pair_labels_seen.append("ALL")
    pair_labels = pair_labels_seen

    # Individual models in canonical order + "Average" column at end
    full_models_in_csv = {r["model"] for r in rows if r["model"] != "ALL"}
    model_full  = [m for m in MODEL_ORDER if m in full_models_in_csv]
    model_labels = [MODEL_SHORT.get(m, m) for m in model_full] + ["Average"]

    n_pairs  = len(pair_labels)
    n_models = len(model_labels)
    values   = np.full((n_pairs, n_models), np.nan)
    annots   = [[""] * n_models for _ in range(n_pairs)]

    pi_map   = {p: i for i, p in enumerate(pair_labels)}
    mi_map   = {full: j for j, full in enumerate(model_full)}
    all_col  = len(model_full)  # index of All Models column

    for r in rows:
        pi = pi_map.get(r["pattern_pair"])
        if pi is None:
            continue
        acc_str = r.get("accuracy", "")
        if not acc_str:
            continue
        acc = float(acc_str)
        if r["model"] == "ALL":
            values[pi, all_col] = acc
            annots[pi][all_col] = f"{acc:.3f}"
        else:
            mi = mi_map.get(r["model"])
            if mi is not None:
                values[pi, mi] = acc
                annots[pi][mi] = f"{acc:.3f}"

    return pair_labels, model_labels, values, annots


def plot_cross_pattern_heatmap(rows: list[dict], out_path: Path) -> None:
    pair_labels, model_labels, values, annots = _load_cross_pattern(rows)
    n_pairs, n_models = values.shape

    fig_w = max(8, n_models * 1.6)
    fig_h = max(4, n_pairs * 0.85 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(values, aspect="auto", cmap=CMAP_XPATT, vmin=0, vmax=1)

    for i in range(n_pairs):
        for j in range(n_models):
            txt = annots[i][j]
            if not txt:
                continue
            v = values[i, j]
            cmap_obj = plt.get_cmap(CMAP_ACCURACY)
            rgba = cmap_obj(v / 100.0)
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            color = "black" if lum > 0.45 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold")

    # Separator before ALL pair row
    if "ALL" in pair_labels:
        sep_y = pair_labels.index("ALL") - 0.5
        ax.axhline(sep_y, color="white", lw=0.8, alpha=0.4)

    # Separator before All Models column (last col)
    ax.axvline(n_models - 1.5, color="white", lw=0.8, alpha=0.4)

    ax.set_xticks(range(n_models))
    xticklabels = model_labels
    ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=9, color="white")
    for tick, lbl in zip(ax.get_xticklabels(), model_labels):
        tick.set_color("#ffdd88" if lbl == "Average" else "white")
        tick.set_fontweight("bold" if lbl == "Average" else "normal")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=9, color="white")
    ax.tick_params(colors="white", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Match Rate (0–1)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Cross-Pattern Accuracy per Model (per-model thresholds)",
                 color="white", **FONT_TITLE, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cross_pattern_table(rows: list[dict], out_path: Path) -> None:
    pair_labels, model_labels, values_pct, annots = _load_cross_pattern(rows)
    n_pairs, n_models = values_pct.shape
    values = values_pct  # already 0-1

    # Re-build annotations with 0-1 values

    all_idx = pair_labels.index("ALL") if "ALL" in pair_labels else None

    all_col_idx = len(model_labels) - 1  # All Models is last column
    _plain_table(
        title="Cross-Pattern Accuracy per Model",
        row_labels=pair_labels,
        col_labels=model_labels,
        values=values,
        annots=annots,
        out_path=out_path,
        sep_before_rows=[all_idx] if all_idx is not None else None,
        bold_rows=[all_idx] if all_idx is not None else None,
        col_group_seps=[all_col_idx],
        row_label_header="Pair",
    )
def _load_pattern_pose_accuracy(rows: list[dict]) -> tuple[list, list, np.ndarray, list]:
    """
    Returns (row_labels, col_labels, values_2d, annots_2d).
    Rows = Pattern × Pose  (group_by="pattern_pose", group_value="<pat>/<pose>").
    Cols = individual models + "Average".
    """
    # Collect all joint group_values present
    joint_rows = [r for r in rows if r["group_by"] == "pattern_pose"]
    if not joint_rows:
        return [], [], np.empty((0, 0)), []

    # Parse and sort row keys
    keys_seen: list[tuple[str, str]] = []
    for r in joint_rows:
        if r["model"] == "ALL_MODELS":
            continue
        parts = r["group_value"].split("/", 1)
        if len(parts) == 2:
            key = (parts[0], parts[1])
            if key not in keys_seen:
                keys_seen.append(key)

    def _pat_sort(k):
        try:
            return (0, int(k[0]), k[1])
        except ValueError:
            return (1, k[0], k[1])

    # Keep poses in canonical order within each pattern
    pose_order = ["frontal", "profile", "pitch"]
    sorted_keys: list[tuple[str, str]] = []
    patterns = sorted(set(p for p, _ in keys_seen), key=lambda p: (0, int(p)) if p.isdigit() else (1, p))
    for pat in patterns:
        poses_for_pat = [po for _, po in keys_seen if _ == pat]
        ordered_poses = [po for po in pose_order if po in poses_for_pat] + \
                        [po for po in poses_for_pat if po not in pose_order]
        for pose in ordered_poses:
            sorted_keys.append((pat, pose))

    # Build display labels; dashed separator at each new pattern
    row_labels: list[str] = []
    sep_before_rows: list[int] = []
    prev_pat = None
    for i, (pat, pose) in enumerate(sorted_keys):
        if pat != prev_pat and i > 0:
            sep_before_rows.append(i)
        pat_display = f"pat:{pat}" if pat != "unclassified" else "unclassified"
        row_labels.append(f"{pat_display} / {pose}")
        prev_pat = pat

    # Column order: individual models + "Average"
    model_order_present = [m for m in MODEL_ORDER if any(r["model"] == m for r in joint_rows)]
    col_labels = [MODEL_SHORT.get(m, m) for m in model_order_present] + ["Average"]
    avg_col = len(model_order_present)

    n_r = len(sorted_keys)
    n_c = len(col_labels)
    values = np.full((n_r, n_c), np.nan)
    annots: list[list[str]] = [[""] * n_c for _ in range(n_r)]

    ri_map = {k: i for i, k in enumerate(sorted_keys)}
    mi_map = {m: i for i, m in enumerate(model_order_present)}

    for r in joint_rows:
        if not r.get("accuracy"):
            continue
        parts = r["group_value"].split("/", 1)
        if len(parts) != 2:
            continue
        key = (parts[0], parts[1])
        ri = ri_map.get(key)
        if ri is None:
            continue
        acc = float(r["accuracy"])
        ann = f"{acc:.3f}"
        if r["model"] == "ALL_MODELS":
            values[ri, avg_col] = acc
            annots[ri][avg_col] = ann
        else:
            ci = mi_map.get(r["model"])
            if ci is not None:
                values[ri, ci] = acc
                annots[ri][ci] = ann

    return row_labels, col_labels, values, annots, sep_before_rows


def plot_pattern_pose_accuracy_table(rows: list[dict], out_path: Path) -> None:
    result = _load_pattern_pose_accuracy(rows)
    if not result[0]:
        print("  [WARN] No pattern_pose rows in model_accuracy.csv — skipping.", file=sys.stderr)
        return
    row_labels, col_labels, values, annots, sep_before_rows = result
    avg_col = len(col_labels) - 1
    _plain_table(
        title="Accuracy per Pattern × Pose",
        row_labels=row_labels,
        col_labels=col_labels,
        values=values,
        annots=annots,
        out_path=out_path,
        sep_before_rows=sep_before_rows,
        col_group_seps=[avg_col],
        row_label_header="Pattern / Pose",
    )


def _add_col_separators(ax: plt.Axes, col_labels: list[str], n_rows: int) -> None:
    """Add faint vertical lines between pose/pattern/overall groups."""
    current_prefix = None
    for j, lbl in enumerate(col_labels):
        prefix = lbl.split(":")[0] if ":" in lbl else "overall"
        if prefix != current_prefix and j > 0:
            ax.axvline(j - 0.5, color="white", lw=0.6, alpha=0.3)
        current_prefix = prefix


def _add_row_separators(ax: plt.Axes, row_labels: list[str], n_cols: int) -> None:
    """Add faint horizontal lines between overall/pose/pattern groups."""
    current_prefix = None
    for i, lbl in enumerate(row_labels):
        prefix = lbl.split(":")[0] if ":" in lbl else "overall"
        if prefix != current_prefix and i > 0:
            ax.axhline(i - 0.5, color="white", lw=0.6, alpha=0.3)
        current_prefix = prefix


def _add_colorbar_axis(fig: plt.Figure, cmap, norm, label: str) -> None:
    """Add a small horizontal colorbar at the bottom of a table figure."""
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label, color="white", fontsize=8)
    cbar.ax.xaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")


# ══════════════════════════════════════════════════════════════════════════════
#  Plain table renderer  (no cell colours, values in 0–1)
# ══════════════════════════════════════════════════════════════════════════════

def _plain_table(title: str,
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
    TOP_PAD   = 0.35          # title only
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

    # ── title ─────────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, fig_h - 0.12, title,
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=FG, transform=ax.transData)

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
#  Main
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ORDER = [
    "IResNet_R18_Glint360K",
    "InceptionResnetCASIA",
    "SwinFace",
    "TransFace_S_Glint360K",
    "edgeface_base",
]

MODEL_SHORT: dict[str, str] = {
    "IResNet_R18_Glint360K":  "IResNet18",
    "InceptionResnetCASIA":   "FaceNet",
    "SwinFace":               "SwinFace",
    "TransFace_S_Glint360K":  "TransFace",
    "edgeface_base":          "EdgeFace",
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heatmap and table figures from data-processing CSVs.")
    parser.add_argument(
        "--out", default=str(_SCRIPT_DIR / "results"),
        help="Directory containing the CSVs (and where figures will be saved).",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSVs from: {out_dir}")
    print("=" * 60)

    # ── 1. Overall accuracy ───────────────────────────────────────────
    acc_rows = _read_csv(out_dir / "model_accuracy.csv")
    if acc_rows:
        print("\n[1] Overall accuracy per model")
        plot_overall_accuracy_heatmap(
            acc_rows, out_dir / "fig_overall_accuracy_heatmap.png")
        plot_overall_accuracy_table(
            acc_rows, out_dir / "fig_overall_accuracy_table.png")
        print("\n[1b] Accuracy per pattern × pose")
        plot_pattern_pose_accuracy_table(
            acc_rows, out_dir / "fig_pattern_pose_accuracy_table.png")
    else:
        print("[1] model_accuracy.csv not found — skipping")

    # ── 2. Cross-pattern accuracy ─────────────────────────────────────
    xp_rows = _read_csv(out_dir / "cross_pattern_by_pair.csv")
    if xp_rows:
        print("\n[2] Cross-pattern accuracy per model")
        plot_cross_pattern_heatmap(
            xp_rows, out_dir / "fig_cross_pattern_heatmap.png")
        plot_cross_pattern_table(
            xp_rows, out_dir / "fig_cross_pattern_table.png")
    else:
        print("[3] cross_pattern_by_pair.csv not found — skipping")

    print("\n" + "=" * 60)
    print(f"Done.  Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
