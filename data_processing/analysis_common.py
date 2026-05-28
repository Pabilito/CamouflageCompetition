"""
data_processing/analysis_common.py
─────────────────────────────────────────────────────────────────────────────
Shared machinery for the metric-vs-recognition analyses
(analyze_color_vs_rank.py and analyze_patchiness_vs_rank.py).

Centralises three things both scripts need so they stay in lock-step:

  1. Correlation primitives.
     pearson()  — linear correlation, used here against the *raw similarity
                  score* (the continuous quantity), which is the more
                  meaningful Pearson target than rank.
     spearman() — rank correlation, used against *rank* (monotonic fit).

     IMPORTANT CONVENTION (changed): for every metric we report
        Pearson r   = corr(metric, similarity_score)      ← linear, vs score
        Spearman ρ  = corr(metric, rank)                  ← monotonic, vs rank
     Lower similarity_score = better attack, and rank 1 = lowest score, so a
     metric that makes attacks stronger shows up as NEGATIVE Pearson r
     (metric ↑ → score ↓) and POSITIVE Spearman ρ (metric ↑ → rank ↑).
     The two therefore generally carry opposite signs; that is expected.

  2. Per-model ranking retrieval.
     get_per_model_rankings(db) → {model_name: [ {image_path, score, rank}... ]}
     Built from Database.get_ranking(model_name=...), already sorted ASC by
     similarity, so enumerate() gives the per-model rank directly.
     Also returns an "AVG" pseudo-model from get_average_ranking().

  3. Output rendering.
     write_corr_table()  — merged PNG table: one block of columns per model.
     write_scatter_grid() — one scatter-grid PNG per model.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

AVG_KEY = "AVG"


# ── Correlation primitives ─────────────────────────────────────────────────

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m]))
    ry = np.argsort(np.argsort(y[m]))
    return pearson(rx.astype(np.float64), ry.astype(np.float64))


def correlate(vals: np.ndarray,
              score_arr: np.ndarray,
              rank_arr: np.ndarray) -> tuple[float, float]:
    """Return (Pearson r vs score, Spearman ρ vs rank) for one metric."""
    return pearson(vals, score_arr), spearman(vals, rank_arr)


# ── Per-model ranking retrieval ─────────────────────────────────────────────

def get_per_model_rankings(db) -> dict[str, list[dict]]:
    """Map model_name → list of {image_path, score, rank} sorted by score ASC.

    Includes an AVG pseudo-model (cross-model average) under AVG_KEY so the
    same code path produces the original averaged analysis too.
    """
    out: dict[str, list[dict]] = {}

    # Cross-model average (best attack first), same as the original scripts.
    avg = db.get_average_ranking(limit=10_000_000)
    out[AVG_KEY] = [
        {"image_path": e["image_path"],
         "score": float(e["avg_score"]),
         "rank": i}
        for i, e in enumerate(avg, start=1)
    ]

    # Discover model names from a large pooled ranking pull.
    pooled = db.get_ranking(limit=10_000_000)
    model_names = sorted({r.model_name for r in pooled})

    for mn in model_names:
        rows = db.get_ranking(model_name=mn, limit=10_000_000)  # ASC by score
        out[mn] = [
            {"image_path": r.image_path,
             "score": float(r.similarity_score),
             "rank": i}
            for i, r in enumerate(rows, start=1)
        ]
    return out


# ── Merged correlation table PNG (one column-block per model) ───────────────

def write_corr_table(per_model_corr: dict[str, list[tuple]],
                     metrics: list[tuple[str, str]],
                     out_path: Path,
                     title: str) -> None:
    """Render a merged correlation table: rows = metrics, columns grouped by
    model, each model contributing (Pearson r vs score, Spearman ρ vs rank).

    per_model_corr : {model_name: [(key, label, pearson_r, spearman_rho), ...]}
                     The metric order must match `metrics`.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[TABLE] matplotlib not installed — skipping correlation table.")
        return

    models = list(per_model_corr.keys())
    # Lookup: (model, key) → (pr, sp)
    lut: dict[tuple[str, str], tuple[float, float]] = {}
    for mn, rows in per_model_corr.items():
        for key, _label, pr, sp in rows:
            lut[(mn, key)] = (pr, sp)

    # Column layout: a metric column, a description column, then for each model
    # an "r" and "ρ" sub-column.
    label_w = 0.085
    desc_w  = 0.215
    avail   = 1.0 - label_w - desc_w
    per_model_w = avail / max(len(models), 1)
    sub_w = per_model_w / 2.0

    n_rows = len(metrics)
    fig_w = max(11.0, 6.0 + 2.0 * len(models))
    fig_h = 1.4 + n_rows * 0.38 + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    HDR_BG  = "#1F3864"
    SUB_BG  = "#2E5496"
    HDR_FG  = "white"
    ALT_BG  = "#F2F2F2"
    POS_BG  = "#D6F0D6"
    NEG_BG  = "#FCE4D6"
    NOTE_FG = "#595959"

    ROW_H   = 0.38 / fig_h
    GRP_H   = 0.40 / fig_h          # model-group header band
    HDR_H   = 0.40 / fig_h          # r / ρ sub-header band
    TOP_Y   = 1.0 - (0.50 / fig_h)

    def cell(x, y, w, h, text, bg=None, fg="black",
             bold=False, ha="center", fontsize=9.0):
        if bg:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y - h), w, h, boxstyle="square,pad=0",
                linewidth=0, color=bg, transform=ax.transAxes, zorder=1))
        ax.text(x + w * (0.06 if ha == "left" else 0.5), y - h / 2, text,
                transform=ax.transAxes, zorder=2, ha=ha, va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=fg, clip_on=True)

    # Title
    ax.text(0.5, 1.0 - 0.05 / fig_h, title,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color="#1F3864")

    # Group header row (model names spanning their two sub-columns)
    gy = TOP_Y
    cell(0.0, gy, label_w, GRP_H, "", bg=HDR_BG)
    cell(label_w, gy, desc_w, GRP_H, "", bg=HDR_BG)
    x = label_w + desc_w
    for mn in models:
        cell(x, gy, per_model_w, GRP_H, mn, bg=HDR_BG, fg=HDR_FG,
             bold=True, fontsize=9.0)
        x += per_model_w

    # Sub-header row (Metric / Description / r / ρ ...)
    sy = TOP_Y - GRP_H
    cell(0.0, sy, label_w, HDR_H, "Metric", bg=SUB_BG, fg=HDR_FG, bold=True)
    cell(label_w, sy, desc_w, HDR_H, "Description", bg=SUB_BG, fg=HDR_FG,
         bold=True, ha="left")
    x = label_w + desc_w
    for _mn in models:
        cell(x, sy, sub_w, HDR_H, "r·score", bg=SUB_BG, fg=HDR_FG,
             bold=True, fontsize=8.0)
        cell(x + sub_w, sy, sub_w, HDR_H, "ρ·rank", bg=SUB_BG, fg=HDR_FG,
             bold=True, fontsize=8.0)
        x += per_model_w

    def sign_bg(v, row_bg):
        if v != v:
            return row_bg
        return POS_BG if v >= 0 else NEG_BG

    # Data rows
    body_top = TOP_Y - GRP_H - HDR_H
    for ri, (key, label) in enumerate(metrics):
        y = body_top - ri * ROW_H
        row_bg = ALT_BG if ri % 2 else "white"
        cell(0.0, y, label_w, ROW_H, key, bg=row_bg, fg="#1F3864", bold=True)
        cell(label_w, y, desc_w, ROW_H, label.strip(), bg=row_bg, ha="left")
        x = label_w + desc_w
        for mn in models:
            pr, sp = lut.get((mn, key), (float("nan"), float("nan")))
            pr_s = f"{pr:+.3f}" if pr == pr else "—"
            sp_s = f"{sp:+.3f}" if sp == sp else "—"
            cell(x, y, sub_w, ROW_H, pr_s, bg=sign_bg(pr, row_bg), fontsize=8.5)
            cell(x + sub_w, y, sub_w, ROW_H, sp_s, bg=sign_bg(sp, row_bg),
                 fontsize=8.5)
            x += per_model_w

    # Grid lines
    def hline(y, color, lw):
        ax.plot([0, 1], [y, y], color=color, lw=lw, transform=ax.transAxes,
                zorder=3, solid_capstyle="butt")
    def vline(x, y0, y1, color, lw):
        ax.plot([x, x], [y0, y1], color=color, lw=lw, transform=ax.transAxes,
                zorder=3, solid_capstyle="butt")

    table_bottom = body_top - n_rows * ROW_H
    hline(TOP_Y, "#AAAAAA", 0.5)
    hline(body_top, HDR_BG, 1.2)
    for ri in range(n_rows + 1):
        hline(body_top - ri * ROW_H, "#CCCCCC", 0.4)
    # vertical separators
    vline(label_w, table_bottom, TOP_Y, "#CCCCCC", 0.4)
    vline(label_w + desc_w, table_bottom, TOP_Y, "#AAAAAA", 0.7)
    x = label_w + desc_w
    for _mn in models:
        x += per_model_w
        vline(x, table_bottom, TOP_Y, "#AAAAAA", 0.7)
    vline(0.0, table_bottom, TOP_Y, "#CCCCCC", 0.4)
    vline(1.0, table_bottom, TOP_Y, "#CCCCCC", 0.4)

    note_y = table_bottom - 0.16 / fig_h
    ax.text(0.0, note_y,
            "  r·score = Pearson(metric, similarity score)   "
            "ρ·rank = Spearman(metric, rank)   "
            "green ≥ 0, red < 0   |corr| < 0.15 → no real relationship.  "
            "Note: r and ρ usually carry OPPOSITE signs "
            "(lower score = better attack = lower rank number).",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=NOTE_FG, style="italic")

    plt.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[TABLE] wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


# ── Per-model scatter-grid PNG ──────────────────────────────────────────────

def write_scatter_grid(model_name: str,
                       rows: list[dict],
                       metrics: list[tuple[str, str]],
                       out_path: Path,
                       title_prefix: str) -> None:
    """One scatter-grid PNG for a single model. X axis = similarity score,
    Y axis = metric. Annotates Pearson r (vs score) and Spearman ρ (vs rank).
    `rows` must each contain the metric keys plus 'score' and 'rank'.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not installed — skipping plot output.")
        return

    score_arr = np.array([r["score"] for r in rows], dtype=np.float64)
    rank_arr  = np.array([r["rank"] for r in rows], dtype=np.float64)

    n_metrics = len(metrics)
    cols = 3
    rows_grid = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows_grid, cols,
                             figsize=(cols * 4.2, rows_grid * 3.0),
                             squeeze=False)
    fig.suptitle(
        f"{title_prefix} — model: {model_name}  ({len(rows)} images)\n"
        f"x = similarity score (lower = better attack)",
        fontsize=12, y=0.995)

    for i, (key, label) in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        m = np.isfinite(vals) & np.isfinite(score_arr)
        ax.scatter(score_arr[m], vals[m], s=6, alpha=0.45,
                   color="#1f77b4", linewidths=0)
        ax.set_xlabel("similarity score", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        if m.sum() >= 3 and vals[m].std() > 0 and score_arr[m].std() > 0:
            pr = pearson(vals, score_arr)
            sp = spearman(vals, rank_arr)
            coef = np.polyfit(score_arr[m], vals[m], 1)
            xs = np.array([score_arr[m].min(), score_arr[m].max()])
            ax.plot(xs, coef[0] * xs + coef[1],
                    color="#d62728", linewidth=1.0, alpha=0.7)
            ax.set_title(f"Pearson r = {pr:+.3f}   Spearman ρ = {sp:+.3f}",
                         fontsize=8)
        else:
            ax.set_title("(insufficient data)", fontsize=8)

    for j in range(n_metrics, rows_grid * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[PLOT] wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
