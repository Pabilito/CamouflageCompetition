"""
data_processing/visualize_results.py
─────────────────────────────────────────────────────────────────────────────
Generate figures for three analyses, each in two formats: heatmap and table.

Figures produced (saved to the same output directory as the CSVs):
  1. fig_overall_accuracy_heatmap.png   — heatmap
  2. fig_overall_accuracy_table.png     — table
  3. fig_score_statistics_heatmap.png   — heatmap
  4. fig_score_statistics_table.png     — table
  5. fig_cross_pattern_heatmap.png      — heatmap
  6. fig_cross_pattern_table.png        — table

Each figure covers:
  • "Overall accuracy per model"
      Rows: models  |  Cols: overall, pose groups, pattern groups
      Cell value: accuracy % + n_recognised/n_images

  • "Score statistics (per-model thresholds)"
      Rows: group (overall / pose / pattern)  |  Cols: models
      Cell value: attack-success rate % + n_fooled/n_evaluated

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
    Rows = models, Cols = groups.
    Only "overall/all", "pose/*", "pattern/*" — not ALL_MODELS.
    """
    # Collect models in canonical order (excluding ALL_MODELS)
    model_order = []
    for r in rows:
        m = r["model"]
        if m != "ALL_MODELS" and m not in model_order:
            model_order.append(m)

    # Collect groups in a stable order: overall first, then pose, then pattern
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

    n_models = len(model_order)
    n_groups = len(group_order)
    values   = np.full((n_models, n_groups), np.nan)
    annots   = [[""] * n_groups for _ in range(n_models)]

    # Index lookup
    m_idx = {m: i for i, m in enumerate(model_order)}
    g_idx = {g: i for i, g in enumerate(group_order)}

    for r in rows:
        if r["model"] == "ALL_MODELS":
            continue
        key = f"{r['group_by']}/{r['group_value']}"
        mi = m_idx.get(r["model"])
        gi = g_idx.get(key)
        if mi is None or gi is None:
            continue
        if r["accuracy"]:
            acc = float(r["accuracy"]) * 100
            values[mi, gi] = acc
            annots[mi][gi] = f"{acc:.1f}%\n{r['n_recognised']}/{r['n_images']}"

    col_labels = [group_label_map[g] for g in group_order]
    return model_order, col_labels, values, annots


def plot_overall_accuracy_heatmap(rows: list[dict], out_path: Path) -> None:
    model_labels, col_labels, values, annots = _load_overall_accuracy(rows)
    n_rows, n_cols = values.shape

    fig_w = max(10, n_cols * 1.4)
    fig_h = max(4,  n_rows * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(values, aspect="auto", cmap=CMAP_ACCURACY, vmin=0, vmax=100)

    # Annotations
    for i in range(n_rows):
        for j in range(n_cols):
            txt = annots[i][j]
            if not txt:
                continue
            v = values[i, j]
            color = "black" if 25 < v < 75 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=7.5, linespacing=1.4)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right",
                       fontsize=9, color="white")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(model_labels, fontsize=9, color="white")
    ax.tick_params(colors="white", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Column-type separators
    _add_col_separators(ax, col_labels, n_rows)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Accuracy (%)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Overall Accuracy per Model", color="white", **FONT_TITLE, pad=12)
    _add_subtitle(ax, "green = model still recognises (higher = worse attack); "
                  "red = fooled", color="#aaa")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_overall_accuracy_table(rows: list[dict], out_path: Path) -> None:
    model_labels, col_labels, values_pct, _ = _load_overall_accuracy(rows)
    n_rows, n_cols = values_pct.shape

    # Convert to 0-1
    values = values_pct / 100.0

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
            annots[mi][gi] = "{:.3f}\n{}/{}".format(acc, r["n_recognised"], r["n_images"])

    # Column group separators: find where pose/pattern sections start
    col_group_seps = []
    prev_prefix = None
    for j, key in enumerate(group_order):
        prefix = key.split("/")[0]
        if prefix != prev_prefix and j > 0:
            col_group_seps.append(j)
        prev_prefix = prefix

    _plain_table(
        title="Overall Accuracy per Model",
        subtitle="accuracy (0–1)  +  recognised / total   |   columns: all · pose groups · pattern groups",
        row_labels=model_labels,
        col_labels=[group_label_map[g] for g in group_order],
        values=values,
        annots=annots,
        out_path=out_path,
        col_group_seps=col_group_seps,
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
            annots[ri][ci] = f"{asr:.1f}%\n{n_s}/{n_e}"

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
            color = "black" if 25 < v < 75 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=7.5, linespacing=1.4)

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
    _add_subtitle(ax, "red = model fooled (higher = better attack); "
                  "green = model recognises", color="#aaa")
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
            annots[ri][ci] = "{:.3f}\n{}/{}".format(asr, n_s, n_e)

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
        subtitle="attack success rate (0–1)  +  fooled / evaluated   |   rows: overall · pose · pattern",
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
    Rows = pattern pairs (including ALL), Cols = models (excluding ALL).
    Cell = match rate % + n_matched/n_pairs.
    """
    pair_labels  = []
    model_labels = []
    for r in rows:
        pp = r["pattern_pair"]
        m  = r["model"]
        if pp not in pair_labels:
            pair_labels.append(pp)
        if m != "ALL" and m not in model_labels:
            model_labels.append(m)

    # Move ALL to end of pairs
    if "ALL" in pair_labels:
        pair_labels.remove("ALL")
        pair_labels.append("ALL")

    n_pairs  = len(pair_labels)
    n_models = len(model_labels)
    values   = np.full((n_pairs, n_models), np.nan)
    annots   = [[""] * n_models for _ in range(n_pairs)]

    pi_map = {p: i for i, p in enumerate(pair_labels)}
    mi_map = {m: j for j, m in enumerate(model_labels)}

    for r in rows:
        if r["model"] == "ALL":
            continue
        pi = pi_map.get(r["pattern_pair"])
        mi = mi_map.get(r["model"])
        if pi is None or mi is None:
            continue
        acc_str = r.get("accuracy", "")
        if not acc_str:
            continue
        acc = float(acc_str) * 100
        values[pi, mi] = acc
        annots[pi][mi] = f"{acc:.1f}%\n{r['n_matched']}/{r['n_pairs']}"

    return pair_labels, model_labels, values, annots


def plot_cross_pattern_heatmap(rows: list[dict], out_path: Path) -> None:
    pair_labels, model_labels, values, annots = _load_cross_pattern(rows)
    n_pairs, n_models = values.shape

    fig_w = max(8, n_models * 1.6)
    fig_h = max(4, n_pairs * 0.85 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(values, aspect="auto", cmap=CMAP_XPATT, vmin=0, vmax=100)

    for i in range(n_pairs):
        for j in range(n_models):
            txt = annots[i][j]
            if not txt:
                continue
            v = values[i, j]
            color = "black" if 25 < v < 75 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=7.5, linespacing=1.4)

    # Bold / separator for ALL row
    if "ALL" in pair_labels:
        sep_y = pair_labels.index("ALL") - 0.5
        ax.axhline(sep_y, color="white", lw=1.2, alpha=0.5)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, rotation=35, ha="right",
                       fontsize=9, color="white")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=9, color="white")
    ax.tick_params(colors="white", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Match Rate (%)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Cross-Pattern Accuracy per Model (per-model thresholds)",
                 color="white", **FONT_TITLE, pad=12)
    _add_subtitle(ax, "match rate % + n_matched/n_pairs  |  "
                  "green = cross-pattern images match (same identity recognised)",
                  color="#aaa")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cross_pattern_table(rows: list[dict], out_path: Path) -> None:
    pair_labels, model_labels, values_pct, _ = _load_cross_pattern(rows)
    n_pairs, n_models = values_pct.shape
    values = values_pct / 100.0

    # Re-build annotations with 0-1 values
    pi_map = {p: i for i, p in enumerate(pair_labels)}
    mi_map = {m: j for j, m in enumerate(model_labels)}
    annots = [[""] * n_models for _ in range(n_pairs)]
    for r in rows:
        if r["model"] == "ALL":
            continue
        pi = pi_map.get(r["pattern_pair"])
        mi = mi_map.get(r["model"])
        if pi is None or mi is None:
            continue
        acc_str = r.get("accuracy", "")
        if acc_str:
            acc = float(acc_str)
            annots[pi][mi] = "{:.3f}\n{}/{}".format(acc, r["n_matched"], r["n_pairs"])

    all_idx = pair_labels.index("ALL") if "ALL" in pair_labels else None

    _plain_table(
        title="Cross-Pattern Accuracy per Model",
        subtitle="match rate (0–1)  +  n_matched / n_pairs   |   pattern-pair rows  ·  model columns",
        row_labels=pair_labels,
        col_labels=model_labels,
        values=values,
        annots=annots,
        out_path=out_path,
        sep_before_rows=[all_idx] if all_idx is not None else None,
        bold_rows=[all_idx] if all_idx is not None else None,
        row_label_header="Pair",
    )
def _add_subtitle(ax: plt.Axes, text: str, color: str = "#aaa") -> None:
    ax.text(0.5, 1.01, text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, color=color, style="italic")


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
#  Main
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ORDER = [
    "IResNet_R18_Glint360K",
    "InceptionResnetCASIA",
    "SwinFace",
    "TransFace_S_Glint360K",
    "edgeface_base",
]

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
    else:
        print("[1] model_accuracy.csv not found — skipping")

    # ── 2. Score statistics / attack success ──────────────────────────
    stat_rows = _read_csv(out_dir / "score_statistics.csv")
    if stat_rows:
        print("\n[2] Score statistics (attack success rate)")
        plot_score_statistics_heatmap(
            stat_rows, MODEL_ORDER,
            out_dir / "fig_score_statistics_heatmap.png")
        plot_score_statistics_table(
            stat_rows, MODEL_ORDER,
            out_dir / "fig_score_statistics_table.png")
    else:
        print("[2] score_statistics.csv not found — skipping")

    # ── 3. Cross-pattern accuracy ─────────────────────────────────────
    xp_rows = _read_csv(out_dir / "cross_pattern_by_pair.csv")
    if xp_rows:
        print("\n[3] Cross-pattern accuracy per model")
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
