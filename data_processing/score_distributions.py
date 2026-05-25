"""
data_processing/score_distributions.py
─────────────────────────────────────────────────────────────────────────────
Generates similarity score distribution plots per model and saves them to
    data_processing/results/<model_name>/

Four plots are produced per model:

  Plot 1 – Mated vs Non-Mated (all attacks)
      Mated   : every control × every adversarial of the SAME identity
      Non-mated: equal-count random control × adversarial of a DIFFERENT identity

  Plot 2 – Pose breakdown (frontal / profile, mated + non-mated)
      Four distributions:
        mated   frontal   | mated   profile
        non-mated frontal | non-mated profile

  Plot 3 – Cross-pattern comparisons (mated + equal non-mated)
      All pairwise pattern combinations (1v2, 1v3, 2v3 …)
      Mated: control(id) × adversarial(id), pattern A vs pattern B
      Non-mated: equal number of randomly sampled cross-identity pairs

  Plot 4 – Overlay of Plot 1 and Plot 3
      Combines the mated/non-mated from Plot 1 plus every cross-pattern
      mated distribution from Plot 3 onto one figure.

Usage
─────
  python data_processing/score_distributions.py [--db DB] [--out DIR]

  --db   path to SQLite database (default: db.sqlite)
  --out  root output directory   (default: data_processing/results)

Requirements: numpy, matplotlib, scipy (optional – used for KDE smoothing)
"""

from __future__ import annotations

import argparse
import itertools
import random
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Try optional KDE ──────────────────────────────────────────────────────────
try:
    from scipy.stats import gaussian_kde
    HAS_KDE = True
except ImportError:
    HAS_KDE = False

# ── Try pyeer ─────────────────────────────────────────────────────────────────
try:
    from pyeer.eer_info import get_eer_stats
    from pyeer.report import generate_eer_report, export_error_rates
    HAS_PYEER = True
except ImportError:
    HAS_PYEER = False

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "mated":             "#1f77b4",   # strong blue
    "non_mated":         "#d62728",   # strong red
    # Plot 2 – four clearly distinct hues
    "mated_frontal":     "#1f77b4",   # blue
    "mated_profile":     "#2ca02c",   # green
    "non_mated_frontal": "#d62728",   # red
    "non_mated_profile": "#ff7f0e",   # orange
    # Patterns – maximally distinct colours for up to 5 patterns
    "pattern_1":  "#e41a1c",   # red
    "pattern_2":  "#377eb8",   # blue
    "pattern_3":  "#4daf4a",   # green
    "pattern_4":  "#984ea3",   # purple
    "pattern_5":  "#ff7f00",   # orange
}

ALPHA_FILL  = 0.12   # light fill under KDE curve
ALPHA_LINE  = 0.90
LINE_WIDTH  = 2.2
X_MIN       = -0.25  # left bound of all plots
X_MAX       =  1.0
BINS        = 40
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def get_model_names(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT model_name FROM attacks ORDER BY model_name"
    ).fetchall()
    return [r["model_name"] for r in rows]


def load_control_embeddings(con: sqlite3.Connection, model: str) -> dict[str, np.ndarray]:
    """Return {image_path: embedding} for control images of *model*."""
    rows = con.execute(
        "SELECT e.image_path, e.embedding "
        "FROM embeddings e "
        "INNER JOIN captures c ON c.image_path = e.image_path "
        "WHERE e.model_name = ?",
        (model,),
    ).fetchall()
    return {
        r["image_path"]: np.frombuffer(r["embedding"], dtype=np.float32).copy()
        for r in rows
    }


def load_adversarial_embeddings(con: sqlite3.Connection, model: str) -> dict[str, np.ndarray]:
    """Return {image_path: embedding} for adversarial images of *model*."""
    rows = con.execute(
        "SELECT DISTINCT e.image_path, e.embedding "
        "FROM embeddings e "
        "INNER JOIN attacks a ON a.image_path = e.image_path "
        "WHERE e.model_name = ?",
        (model,),
    ).fetchall()
    return {
        r["image_path"]: np.frombuffer(r["embedding"], dtype=np.float32).copy()
        for r in rows
    }


def load_attack_meta(con: sqlite3.Connection) -> dict[str, dict]:
    """Return {image_path: {identity_name, pattern_id, is_frontal, is_profile, is_pitch}}."""
    rows = con.execute(
        "SELECT DISTINCT image_path, identity_name, "
        "pattern_id, is_frontal, is_profile, is_pitch FROM attacks"
    ).fetchall()
    return {r["image_path"]: dict(r) for r in rows}


def load_capture_identity(con: sqlite3.Connection) -> dict[str, str]:
    """Return {image_path: identity_name} for control images."""
    rows = con.execute(
        "SELECT c.image_path, u.name AS identity_name "
        "FROM captures c INNER JOIN users u ON u.id = c.user_id"
    ).fetchall()
    return {r["image_path"]: r["identity_name"] for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity (vectors already L2-normed in DB)
# ─────────────────────────────────────────────────────────────────────────────

def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))


# ─────────────────────────────────────────────────────────────────────────────
# All-models aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_control_embeddings(
    con: sqlite3.Connection,
    models: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Return {model: {image_path: embedding}} for control images of all models."""
    return {m: load_control_embeddings(con, m) for m in models}


def load_all_adversarial_embeddings(
    con: sqlite3.Connection,
    models: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Return {model: {image_path: embedding}} for adversarial images of all models."""
    return {m: load_adversarial_embeddings(con, m) for m in models}


def avg_cosine(
    ctrl_path: str,
    adv_path:  str,
    ctrl_embs_all: dict[str, dict[str, np.ndarray]],
    adv_embs_all:  dict[str, dict[str, np.ndarray]],
) -> float | None:
    """
    Average cosine similarity across all models for a (ctrl, adv) pair.
    Returns None if no model has both embeddings.
    Mirrors the gallery-search aggregation in recognition_tab.py.
    """
    scores = []
    for m in ctrl_embs_all:
        c = ctrl_embs_all[m].get(ctrl_path)
        a = adv_embs_all[m].get(adv_path)
        if c is not None and a is not None:
            scores.append(cosine(c, a))
    return float(np.mean(scores)) if scores else None


def avg_cosine_adv_adv(
    path_a: str,
    path_b: str,
    adv_embs_all: dict[str, dict[str, np.ndarray]],
) -> float | None:
    """
    Average cosine similarity across all models for an (adv_A, adv_B) pair.
    Returns None if no model has both embeddings.
    """
    scores = []
    for m in adv_embs_all:
        a = adv_embs_all[m].get(path_a)
        b = adv_embs_all[m].get(path_b)
        if a is not None and b is not None:
            scores.append(cosine(a, b))
    return float(np.mean(scores)) if scores else None


def scores_from_pairs_all_models(
    pairs:          list[tuple[str, str]],
    ctrl_embs_all:  dict[str, dict[str, np.ndarray]],
    adv_embs_all:   dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Averaged-similarity scores for ctrl↔adv pairs across all models."""
    out = []
    for c, a in pairs:
        v = avg_cosine(c, a, ctrl_embs_all, adv_embs_all)
        if v is not None:
            out.append(v)
    return np.array(out, dtype=np.float32)


def scores_adv_adv_all_models(
    pairs:        list[tuple[str, str]],
    adv_embs_all: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Averaged-similarity scores for adv↔adv pairs across all models."""
    out = []
    for a, b in pairs:
        v = avg_cosine_adv_adv(a, b, adv_embs_all)
        if v is not None:
            out.append(v)
    return np.array(out, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pair-building helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_mated_pairs(
    ctrl_by_id: dict[str, list[str]],
    adv_by_id:  dict[str, list[str]],
) -> list[tuple[str, str]]:
    """All control × adversarial pairs for the SAME identity."""
    pairs: list[tuple[str, str]] = []
    for identity in set(ctrl_by_id) & set(adv_by_id):
        for c in ctrl_by_id[identity]:
            for a in adv_by_id[identity]:
                pairs.append((c, a))
    return pairs


def build_nonmated_pairs(
    ctrl_by_id:  dict[str, list[str]],
    adv_by_id:   dict[str, list[str]],
    n:           int,
    rng:         random.Random,
) -> list[tuple[str, str]]:
    """
    *n* random control × adversarial pairs where identities DIFFER.
    Sampling with replacement when the combinatorial space is smaller than n.
    """
    identities = list(set(ctrl_by_id) | set(adv_by_id))
    ctrl_ids   = list(ctrl_by_id.keys())
    adv_ids    = list(adv_by_id.keys())

    pairs: list[tuple[str, str]] = []
    attempts = 0
    while len(pairs) < n and attempts < n * 20:
        attempts += 1
        c_id = rng.choice(ctrl_ids)
        a_id = rng.choice(adv_ids)
        if c_id == a_id:
            continue
        c_path = rng.choice(ctrl_by_id[c_id])
        a_path = rng.choice(adv_by_id[a_id])
        pairs.append((c_path, a_path))
    return pairs


def scores_from_pairs(
    pairs:      list[tuple[str, str]],
    ctrl_emb:   dict[str, np.ndarray],
    adv_emb:    dict[str, np.ndarray],
) -> np.ndarray:
    scores = []
    for c, a in pairs:
        if c in ctrl_emb and a in adv_emb:
            scores.append(cosine(ctrl_emb[c], adv_emb[a]))
    return np.array(scores, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting primitives
# ─────────────────────────────────────────────────────────────────────────────

def _draw_dist(
    ax: plt.Axes,
    scores: np.ndarray,
    color: str,
    label: str,
    bins: int = BINS,
    linestyle: str = "-",
    fill: bool = True,
    show_n: bool = True,
) -> None:
    """Draw a KDE line, optionally with a light fill underneath."""
    if len(scores) == 0:
        return
    display_label = f"{label} (n={len(scores):,})" if show_n else label
    xs = np.linspace(X_MIN, X_MAX, 500)
    if HAS_KDE and len(scores) > 5:
        try:
            kde = gaussian_kde(scores, bw_method="scott")
            ys  = kde(xs)
            ax.plot(xs, ys, color=color, lw=LINE_WIDTH, alpha=ALPHA_LINE,
                    linestyle=linestyle, label=display_label)
            if fill:
                ax.fill_between(xs, ys, alpha=ALPHA_FILL, color=color)
            return
        except Exception:
            pass
    # Fallback: normalised step histogram when scipy is absent
    ax.hist(
        scores, bins=bins, range=(X_MIN, X_MAX),
        density=True, histtype="step",
        color=color, lw=LINE_WIDTH, alpha=ALPHA_LINE,
        linestyle=linestyle,
        label=display_label,
    )


def _style_ax(ax: plt.Axes, title: str, xlabel: str = "Cosine Similarity",
              legend_inside: bool = False) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    if legend_inside:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    else:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, framealpha=0.7)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_threshold_lines(
    ax: plt.Axes,
    gscores: np.ndarray,
    iscores: np.ndarray,
) -> None:
    """
    Compute EER, FMR100, FMR1000 thresholds via pyeer, draw vertical dashed
    lines, then rebuild the axis legend to include both the existing
    distribution entries and a clearly separated threshold section.
    Silently skips if pyeer is unavailable or scores are too few.
    """
    if not HAS_PYEER or len(gscores) < 5 or len(iscores) < 5:
        return
    try:
        # Standard convention: mated=genuine, nonmated=impostor.
        # EER will be high (attack works), thresholds are correctly positioned:
        # EER_th at crossing, FMR100_th at 99th pct of nonmated (right of EER).
        st = get_eer_stats(gscores.tolist(), iscores.tolist())

        # Acc@EER = 1 - EER; TPR at FMR operating points
        acc_eer    = 1.0 - st.eer
        tpr_fmr100 = 1.0 - st.fmr100
        tpr_fmr1k  = 1.0 - st.fmr1000

        threshold_defs = [
            (st.eer_th,     f"EER [{st.eer_th:.3f}] (Acc={acc_eer:.3f})",          "#333333"),
            (st.fmr100_th,  f"FMR100 [{st.fmr100_th:.3f}] (TPR={tpr_fmr100:.3f})", "#e67e22"),
            (st.fmr1000_th, f"FMR1000 [{st.fmr1000_th:.3f}] (TPR={tpr_fmr1k:.3f})","#8e44ad"),
        ]
        # Draw plain vertical lines
        for th, _, color in threshold_defs:
            ax.axvline(th, color=color, lw=1.2, linestyle="--", alpha=0.8)

        # Collect existing legend handles/labels
        existing_h, existing_l = ax.get_legend_handles_labels()

        # Build threshold+accuracy handles
        thresh_handles = [
            matplotlib.lines.Line2D([], [], color=c, lw=1.2,
                                    linestyle="--", label=lbl)
            for _, lbl, c in threshold_defs
        ]

        # Invisible spacer handle to act as a section divider
        spacer = matplotlib.lines.Line2D([], [], color="none", label="─ Thresholds ─")

        # Rebuild as one combined legend
        all_handles = existing_h + [spacer] + thresh_handles
        ax.legend(
            handles=all_handles,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            framealpha=0.7,
        )
    except Exception:
        pass


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _unify_ylim(axes_flat) -> None:
    """Set the same y-axis upper bound on all axes (max across all, bottom=0)."""
    ymax = max((ax.get_ylim()[1] for ax in axes_flat if ax.get_visible()), default=1.0)
    for ax in axes_flat:
        if ax.get_visible():
            ax.set_ylim(0, ymax)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model plot generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots_for_model(
    model:      str,
    con:        sqlite3.Connection,
    out_root:   Path,
    rng:        random.Random,
) -> None:
    print(f"\n[{model}]")

    out_dir = out_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    ctrl_emb   = load_control_embeddings(con, model)
    adv_emb    = load_adversarial_embeddings(con, model)
    atk_meta   = load_attack_meta(con)           # keyed by adv image_path
    ctrl_id    = load_capture_identity(con)      # keyed by ctrl image_path

    if not ctrl_emb:
        print(f"  No control embeddings for {model} – skipping.")
        return
    if not adv_emb:
        print(f"  No adversarial embeddings for {model} – skipping.")
        return

    # Group control paths by identity
    ctrl_by_id: dict[str, list[str]] = defaultdict(list)
    for path, ident in ctrl_id.items():
        if path in ctrl_emb:
            ctrl_by_id[ident].append(path)

    # Group adversarial paths by identity
    adv_by_id: dict[str, list[str]] = defaultdict(list)
    for path in adv_emb:
        meta = atk_meta.get(path, {})
        ident = meta.get("identity_name", "")
        if ident:
            adv_by_id[ident].append(path)

    # ── PLOT 1 ─────────────────────────────────────────────────────────────────
    print("  Plot 1 …", end=" ")
    mated_pairs    = build_mated_pairs(ctrl_by_id, adv_by_id)
    nonmated_pairs = build_nonmated_pairs(ctrl_by_id, adv_by_id, len(mated_pairs), rng)

    mated_scores    = scores_from_pairs(mated_pairs,    ctrl_emb, adv_emb)
    nonmated_scores = scores_from_pairs(nonmated_pairs, ctrl_emb, adv_emb)

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_dist(ax, mated_scores,    PALETTE["mated"],     "Mated")
    _draw_dist(ax, nonmated_scores, PALETTE["non_mated"], "Non-mated", linestyle=":")
    _style_ax(ax, f"{model}\nMated vs Non-Mated Score Distributions")
    _draw_threshold_lines(ax, mated_scores, nonmated_scores)
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot1_mated_vs_nonmated.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_dist(ax, mated_scores,    PALETTE["mated"],     "Mated",     show_n=False)
    _draw_dist(ax, nonmated_scores, PALETTE["non_mated"], "Non-mated", linestyle=":", show_n=False)
    _style_ax(ax, f"{model}\nMated vs Non-Mated Score Distributions", legend_inside=True)
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot1_mated_vs_nonmated_clean.png")
    print("done")

    # ── PLOT 2 ─────────────────────────────────────────────────────────────────
    print("  Plot 2 …", end=" ")

    def _filter_adv_by_pose(flag: str) -> dict[str, list[str]]:
        filtered: dict[str, list[str]] = defaultdict(list)
        for path, meta in atk_meta.items():
            if path in adv_emb and meta.get(flag, 0):
                ident = meta.get("identity_name", "")
                if ident:
                    filtered[ident].append(path)
        return filtered

    adv_frontal = _filter_adv_by_pose("is_frontal")
    adv_profile = _filter_adv_by_pose("is_profile")

    def _pose_mated_nonmated(adv_subset: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
        m_pairs  = build_mated_pairs(ctrl_by_id, adv_subset)
        nm_pairs = build_nonmated_pairs(ctrl_by_id, adv_subset, len(m_pairs), rng)
        return (
            scores_from_pairs(m_pairs,  ctrl_emb, adv_emb),
            scores_from_pairs(nm_pairs, ctrl_emb, adv_emb),
        )

    m_front,  nm_front  = _pose_mated_nonmated(adv_frontal)
    m_profil, nm_profil = _pose_mated_nonmated(adv_profile)

    # Frontal = strong blue, Profile = vivid orange; solid = mated, dotted = non-mated
    COL_FRONTAL = "#1f77b4"   # strong blue
    COL_PROFILE = "#d62728"   # strong red
    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_dist(ax, m_front,   COL_FRONTAL, "Mated – Frontal")
    _draw_dist(ax, nm_front,  COL_FRONTAL, "Non-mated – Frontal", linestyle=":")
    _draw_dist(ax, m_profil,  COL_PROFILE, "Mated – Profile")
    _draw_dist(ax, nm_profil, COL_PROFILE, "Non-mated – Profile", linestyle=":")
    _style_ax(ax, f"{model}\nPose-Stratified Score Distributions")
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot2_pose_breakdown.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_dist(ax, m_front,   COL_FRONTAL, "Mated – Frontal",     show_n=False)
    _draw_dist(ax, nm_front,  COL_FRONTAL, "Non-mated – Frontal", linestyle=":", show_n=False)
    _draw_dist(ax, m_profil,  COL_PROFILE, "Mated – Profile",     show_n=False)
    _draw_dist(ax, nm_profil, COL_PROFILE, "Non-mated – Profile", linestyle=":", show_n=False)
    _style_ax(ax, f"{model}\nPose-Stratified Score Distributions", legend_inside=True)
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot2_pose_breakdown_clean.png")
    print("done")

    # ── PLOT 3 ─────────────────────────────────────────────────────────────────
    # Also accumulate scores for biometric stats (keyed by combo tuple)
    pattern_combo_scores: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    print("  Plot 3 …", end=" ")

    # Collect available patterns (non-zero pattern_id)
    pattern_ids: set[int] = set()
    for meta in atk_meta.values():
        pid = meta.get("pattern_id", 0)
        if pid and pid > 0:
            pattern_ids.add(int(pid))
    pattern_ids_sorted = sorted(pattern_ids)

    # Group adversarial paths by (identity, pattern_id)
    adv_by_id_pat: dict[tuple[str, int], list[str]] = defaultdict(list)
    for path, meta in atk_meta.items():
        if path not in adv_emb:
            continue
        pid   = int(meta.get("pattern_id", 0))
        ident = meta.get("identity_name", "")
        if pid > 0 and ident:
            adv_by_id_pat[(ident, pid)].append(path)

    # adv_by_pat[pid] = {identity: [paths]}
    adv_by_pat: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (ident, pid), paths in adv_by_id_pat.items():
        adv_by_pat[pid][ident].extend(paths)

    combos = list(itertools.combinations(pattern_ids_sorted, 2))

    if not combos:
        print("no pattern combinations found – skipping.")
    else:
        # Determine subplot grid
        n_combos = len(combos)
        ncols    = min(3, n_combos)
        nrows    = (n_combos + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"{model}\nCross-Pattern Score Distributions (adversarial pA vs adversarial pB)",
            fontsize=11, fontweight="bold",
        )

        # Assign a unique colour per combo so each subplot is visually distinct
        COMBO_COLORS = [
            ("#e41a1c", "#377eb8"),  # red / blue
            ("#4daf4a", "#984ea3"),  # green / purple
            ("#ff7f00", "#a65628"),  # orange / brown
            ("#f781bf", "#999999"),  # pink / grey
            ("#e41a1c", "#4daf4a"),  # red / green
            ("#377eb8", "#ff7f00"),  # blue / orange
            ("#984ea3", "#a65628"),  # purple / brown
            ("#f781bf", "#e41a1c"),  # pink / red
            ("#4daf4a", "#377eb8"),  # green / blue
            ("#ff7f00", "#984ea3"),  # orange / purple
        ]
        NM_COLOR = "#555555"  # dark grey for non-mated across all subplots

        for idx, (pA, pB) in enumerate(combos):
            ax      = axes[idx // ncols][idx % ncols]
            col_pA, col_pB = COMBO_COLORS[idx % len(COMBO_COLORS)]

            adv_pA: dict[str, list[str]] = dict(adv_by_pat.get(pA, {}))
            adv_pB: dict[str, list[str]] = dict(adv_by_pat.get(pB, {}))

            # Mated: for each identity that has BOTH pA and pB images,
            # compare every pA adversarial vs every pB adversarial.
            shared_ids = set(adv_pA) & set(adv_pB)
            mated_pairs_p3: list[tuple[str, str]] = []
            for ident in shared_ids:
                for a in adv_pA[ident]:
                    for b in adv_pB[ident]:
                        mated_pairs_p3.append((a, b))

            # Non-mated: equal count, pA image from one identity vs
            # pB image from a DIFFERENT identity.
            n_nm = len(mated_pairs_p3)
            pA_ids = list(adv_pA.keys())
            pB_ids = list(adv_pB.keys())
            nonmated_pairs_p3: list[tuple[str, str]] = []
            attempts = 0
            while len(nonmated_pairs_p3) < n_nm and attempts < n_nm * 20:
                attempts += 1
                if not pA_ids or not pB_ids:
                    break
                id_a = rng.choice(pA_ids)
                id_b = rng.choice(pB_ids)
                if id_a == id_b:
                    continue
                a = rng.choice(adv_pA[id_a])
                b = rng.choice(adv_pB[id_b])
                nonmated_pairs_p3.append((a, b))

            # Scores — both images are adversarial, so look up in adv_emb
            def adv_adv_scores(pairs: list[tuple[str, str]]) -> np.ndarray:
                s = []
                for a, b in pairs:
                    if a in adv_emb and b in adv_emb:
                        s.append(cosine(adv_emb[a], adv_emb[b]))
                return np.array(s, dtype=np.float32)

            m_scores  = adv_adv_scores(mated_pairs_p3)
            nm_scores = adv_adv_scores(nonmated_pairs_p3)

            pattern_combo_scores[(pA, pB)] = (m_scores, nm_scores)
            _draw_dist(ax, m_scores,  col_pA, f"Mated P{pA}↔P{pB}",  bins=30)
            _draw_dist(ax, nm_scores, NM_COLOR, "Non-mated", bins=30, linestyle="--")
            _style_ax(ax, f"Pattern {pA} vs Pattern {pB}")
            _draw_threshold_lines(ax, m_scores, nm_scores)

        # Hide unused subplots
        for idx in range(n_combos, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        _unify_ylim([axes[r][c] for r in range(nrows) for c in range(ncols)])
        fig.tight_layout()
        _save(fig, out_dir / "plot3_cross_pattern.png")

        # ── Plot 3 clean variant ───────────────────────────────────────────────
        fig_c, axes_c = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        fig_c.suptitle(
            f"{model}\nCross-Pattern Score Distributions (adversarial pA vs adversarial pB)",
            fontsize=11, fontweight="bold",
        )
        for idx, (pA, pB) in enumerate(combos):
            col_pA, _ = COMBO_COLORS[idx % len(COMBO_COLORS)]
            ax_c = axes_c[idx // ncols][idx % ncols]
            m_sc, nm_sc = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax_c, m_sc,  col_pA,   f"Mated P{pA}↔P{pB}", bins=30, show_n=False)
            _draw_dist(ax_c, nm_sc, NM_COLOR, "Non-mated",           bins=30, linestyle="--", show_n=False)
            _style_ax(ax_c, f"Pattern {pA} vs Pattern {pB}", legend_inside=True)
        for idx in range(n_combos, nrows * ncols):
            axes_c[idx // ncols][idx % ncols].set_visible(False)
        _unify_ylim([axes_c[r][c] for r in range(nrows) for c in range(ncols)])
        fig_c.tight_layout()
        _save(fig_c, out_dir / "plot3_cross_pattern_clean.png")
        print("done")

    # ── PLOT 4 ─────────────────────────────────────────────────────────────────
    # Combines Plot 1 (ctrl vs adv, mated + non-mated) with Plot 3 cross-pattern
    # lines (adv pA vs adv pB, mated + non-mated per combo).
    print("  Plot 4 …", end=" ")
    fig, ax = plt.subplots(figsize=(9, 5))

    # Base layers from Plot 1 — same colour, solid mated, dotted non-mated
    COL_CTRL_ADV = PALETTE["mated"]   # blue for both
    _draw_dist(ax, mated_scores,    COL_CTRL_ADV, "Mated ctrl↔adv (all)",     bins=BINS, fill=False)
    _draw_dist(ax, nonmated_scores, COL_CTRL_ADV, "Non-mated ctrl↔adv (all)", bins=BINS, linestyle=":", fill=False)

    # Cross-pattern layers from Plot 3 — solid mated, dotted non-mated, same colour per combo
    if combos:
        COMBO_COLORS = [
            ("#e41a1c", "#377eb8"),
            ("#4daf4a", "#984ea3"),
            ("#ff7f00", "#a65628"),
            ("#f781bf", "#999999"),
            ("#e41a1c", "#4daf4a"),
            ("#377eb8", "#ff7f00"),
            ("#984ea3", "#a65628"),
            ("#f781bf", "#e41a1c"),
            ("#4daf4a", "#377eb8"),
            ("#ff7f00", "#984ea3"),
        ]

        for idx, (pA, pB) in enumerate(combos):
            col_pA, _ = COMBO_COLORS[idx % len(COMBO_COLORS)]

            adv_pA_: dict[str, list[str]] = dict(adv_by_pat.get(pA, {}))
            adv_pB_: dict[str, list[str]] = dict(adv_by_pat.get(pB, {}))

            shared_ids = set(adv_pA_) & set(adv_pB_)
            mated_pairs_p4: list[tuple[str, str]] = []
            for ident in shared_ids:
                for a in adv_pA_[ident]:
                    for b in adv_pB_[ident]:
                        mated_pairs_p4.append((a, b))

            n_nm = len(mated_pairs_p4)
            pA_ids = list(adv_pA_.keys())
            pB_ids = list(adv_pB_.keys())
            nonmated_pairs_p4: list[tuple[str, str]] = []
            attempts = 0
            while len(nonmated_pairs_p4) < n_nm and attempts < n_nm * 20:
                attempts += 1
                if not pA_ids or not pB_ids:
                    break
                id_a = rng.choice(pA_ids)
                id_b = rng.choice(pB_ids)
                if id_a == id_b:
                    continue
                a = rng.choice(adv_pA_[id_a])
                b = rng.choice(adv_pB_[id_b])
                nonmated_pairs_p4.append((a, b))

            def _adv_adv(pairs: list[tuple[str, str]]) -> np.ndarray:
                s = []
                for a, b in pairs:
                    if a in adv_emb and b in adv_emb:
                        s.append(cosine(adv_emb[a], adv_emb[b]))
                return np.array(s, dtype=np.float32)

            m_scores_p4  = _adv_adv(mated_pairs_p4)
            nm_scores_p4 = _adv_adv(nonmated_pairs_p4)

            _draw_dist(ax, m_scores_p4,  col_pA, f"Mated P{pA}↔P{pB}",     bins=30, fill=False)
            _draw_dist(ax, nm_scores_p4, col_pA, f"Non-mated P{pA}↔P{pB}", bins=30, linestyle=":", fill=False)

    _style_ax(ax, f"{model}\nCombined: ctrl↔adv (Plot 1) + cross-pattern adv↔adv (Plot 3)")
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot4_combined.png")

    # ── Plot 4 clean variant ───────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(9, 5))
    _draw_dist(ax_c, mated_scores,    COL_CTRL_ADV, "Mated ctrl↔adv (all)",     bins=BINS, fill=False, show_n=False)
    _draw_dist(ax_c, nonmated_scores, COL_CTRL_ADV, "Non-mated ctrl↔adv (all)", bins=BINS, linestyle=":", fill=False, show_n=False)
    if combos:
        for idx, (pA, pB) in enumerate(combos):
            col_p4, _ = COMBO_COLORS[idx % len(COMBO_COLORS)]
            m_sc_p4, nm_sc_p4 = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax_c, m_sc_p4,  col_p4, f"Mated P{pA}↔P{pB}",     bins=30, fill=False, show_n=False)
            _draw_dist(ax_c, nm_sc_p4, col_p4, f"Non-mated P{pA}↔P{pB}", bins=30, linestyle=":", fill=False, show_n=False)
    _style_ax(ax_c, f"{model}\nCombined: ctrl↔adv (Plot 1) + cross-pattern adv↔adv (Plot 3)",
              legend_inside=True)
    _unify_ylim([ax_c])
    fig_c.tight_layout()
    _save(fig_c, out_dir / "plot4_combined_clean.png")
    print("done")

    # ── PER-PATTERN CLEAN PLOTS ────────────────────────────────────────────────
    per_pattern_data = _generate_per_pattern_clean_plots(
        pattern_ids_sorted = pattern_ids_sorted,
        adv_by_pat         = adv_by_pat,
        ctrl_by_id         = ctrl_by_id,
        ctrl_emb           = ctrl_emb,
        adv_emb            = adv_emb,
        atk_meta           = atk_meta,
        rng                = rng,
        model_label        = model,
        out_dir            = out_dir,
        score_fn           = scores_from_pairs,
        score_adv_fn       = lambda pairs, adv: np.array(
            [cosine(adv[a], adv[b]) for a, b in pairs if a in adv and b in adv],
            dtype=np.float32,
        ),
    )

    # ── BIOMETRIC STATS ────────────────────────────────────────────────────────
    compute_biometric_stats(
        model          = model,
        out_dir        = out_dir,
        mated_scores   = mated_scores,
        nonmated_scores= nonmated_scores,
        pattern_combos = pattern_combo_scores,
    )

    # Return computed data for the combined overview figure
    return {
        "model":               model,
        "mated_scores":        mated_scores,
        "nonmated_scores":     nonmated_scores,
        "adv_frontal_scores":  (m_front, nm_front),
        "adv_profile_scores":  (m_profil, nm_profil),
        "pattern_combo_scores": pattern_combo_scores,
        "combos":              combos if combos else [],
        "per_pattern_data":    per_pattern_data,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern clean plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _generate_per_pattern_clean_plots(
    pattern_ids_sorted: list[int],
    adv_by_pat:         dict,          # pid → {identity: [paths]}
    ctrl_by_id:         dict,          # identity → [ctrl paths]
    ctrl_emb:           dict,          # path → embedding  (single-model or averaged via wrapper)
    adv_emb:            dict,          # path → embedding
    atk_meta:           dict,          # adv path → meta dict
    rng:                "random.Random",
    model_label:        str,
    out_dir:            "Path",
    score_fn,          # callable(pairs, ctrl_emb, adv_emb) → np.ndarray
    score_adv_fn,      # callable(pairs, adv_emb) → np.ndarray
) -> dict:
    """
    For each pattern in pattern_ids_sorted, produce clean variants of plots 1–4
    restricted to adversarial images belonging to that pattern only.

    Output filenames: plot{1-4}_*_pat{N}_clean.png  inside out_dir.

    Returns a dict keyed by pattern_id with the computed scores needed by the
    per-pattern cross-model overview figures:
        {pid: {
            "mated_scores":     np.ndarray,
            "nonmated_scores":  np.ndarray,
            "adv_frontal_scores": (m_front, nm_front),
            "adv_profile_scores": (m_profil, nm_profil),
            "pat_combo_scores": {(pA, pB): (m_sc, nm_sc)},
            "combos_pat":       [(pA, pB), ...],
        }}
    """
    # Accumulated return data
    per_pattern_data: dict = {}
    COL_FRONTAL = "#1f77b4"
    COL_PROFILE = "#d62728"
    COMBO_COLORS = [
        ("#e41a1c", "#377eb8"), ("#4daf4a", "#984ea3"),
        ("#ff7f00", "#a65628"), ("#f781bf", "#999999"),
        ("#e41a1c", "#4daf4a"), ("#377eb8", "#ff7f00"),
        ("#984ea3", "#a65628"), ("#f781bf", "#e41a1c"),
        ("#4daf4a", "#377eb8"), ("#ff7f00", "#984ea3"),
    ]
    NM_COLOR    = "#555555"
    COL_CTRL_ADV = PALETTE["mated"]

    for pid in pattern_ids_sorted:
        tag   = f"pat{pid}"
        label = f"Pattern {pid}"
        adv_by_id_pat: dict[str, list[str]] = dict(adv_by_pat.get(pid, {}))

        if not adv_by_id_pat:
            continue

        print(f"  Per-pattern clean plots [{label}] …", end=" ")

        # ── Plot 1 — mated vs non-mated ───────────────────────────────────────
        mated_pairs    = build_mated_pairs(ctrl_by_id, adv_by_id_pat)
        nonmated_pairs = build_nonmated_pairs(ctrl_by_id, adv_by_id_pat, len(mated_pairs), rng)
        m_scores  = score_fn(mated_pairs,    ctrl_emb, adv_emb)
        nm_scores = score_fn(nonmated_pairs, ctrl_emb, adv_emb)

        # Store plot-1 scores for cross-model overview
        per_pattern_data.setdefault(pid, {})["mated_scores"]    = m_scores
        per_pattern_data[pid]["nonmated_scores"] = nm_scores

        fig, ax = plt.subplots(figsize=(7, 4))
        _draw_dist(ax, m_scores,  PALETTE["mated"],     "Mated",     show_n=False)
        _draw_dist(ax, nm_scores, PALETTE["non_mated"], "Non-mated", linestyle=":", show_n=False)
        _style_ax(ax, f"{model_label} – {label}\nMated vs Non-Mated Score Distributions",
                  legend_inside=True)
        _unify_ylim([ax])
        fig.tight_layout()
        _save(fig, out_dir / f"plot1_mated_vs_nonmated_{tag}_clean.png")

        # ── Plot 2 — pose breakdown ───────────────────────────────────────────
        def _pat_pose_subset(flag: str) -> dict[str, list[str]]:
            out: dict[str, list[str]] = defaultdict(list)
            for path, paths_list in adv_by_id_pat.items():
                ident = path  # adv_by_id_pat key IS the identity
                for p in paths_list:
                    if atk_meta.get(p, {}).get(flag, 0):
                        out[ident].append(p)
            return out

        adv_frontal_pat = _pat_pose_subset("is_frontal")
        adv_profile_pat = _pat_pose_subset("is_profile")

        def _pose_scores(adv_subset):
            mp  = build_mated_pairs(ctrl_by_id, adv_subset)
            nmp = build_nonmated_pairs(ctrl_by_id, adv_subset, len(mp), rng)
            return score_fn(mp, ctrl_emb, adv_emb), score_fn(nmp, ctrl_emb, adv_emb)

        m_front,  nm_front  = _pose_scores(adv_frontal_pat)
        m_profil, nm_profil = _pose_scores(adv_profile_pat)

        # Store plot-2 scores for cross-model overview
        per_pattern_data[pid]["adv_frontal_scores"] = (m_front,  nm_front)
        per_pattern_data[pid]["adv_profile_scores"] = (m_profil, nm_profil)

        fig, ax = plt.subplots(figsize=(8, 4))
        _draw_dist(ax, m_front,   COL_FRONTAL, "Mated – Frontal",     show_n=False)
        _draw_dist(ax, nm_front,  COL_FRONTAL, "Non-mated – Frontal", linestyle=":", show_n=False)
        _draw_dist(ax, m_profil,  COL_PROFILE, "Mated – Profile",     show_n=False)
        _draw_dist(ax, nm_profil, COL_PROFILE, "Non-mated – Profile", linestyle=":", show_n=False)
        _style_ax(ax, f"{model_label} – {label}\nPose-Stratified Score Distributions",
                  legend_inside=True)
        _unify_ylim([ax])
        fig.tight_layout()
        _save(fig, out_dir / f"plot2_pose_breakdown_{tag}_clean.png")

        # ── Plot 3 — cross-pattern (this pattern vs every other) ──────────────
        other_pids = [p for p in pattern_ids_sorted if p != pid]
        combos_pat = [(pid, p) for p in other_pids]

        pat_combo_scores: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        # Store plot-3 combo list for cross-model overview (filled below)
        per_pattern_data[pid]["combos_pat"] = combos_pat

        if combos_pat:
            n_combos = len(combos_pat)
            ncols    = min(3, n_combos)
            nrows    = (n_combos + ncols - 1) // ncols
            fig_c, axes_c = plt.subplots(
                nrows, ncols,
                figsize=(5 * ncols, 4 * nrows),
                squeeze=False,
            )
            fig_c.suptitle(
                f"{model_label} – {label}\nCross-Pattern Score Distributions",
                fontsize=11, fontweight="bold",
            )

            for c_idx, (pA, pB) in enumerate(combos_pat):
                col_pA, _ = COMBO_COLORS[c_idx % len(COMBO_COLORS)]
                ax_c = axes_c[c_idx // ncols][c_idx % ncols]

                adv_pA = dict(adv_by_pat.get(pA, {}))
                adv_pB = dict(adv_by_pat.get(pB, {}))

                shared_ids = set(adv_pA) & set(adv_pB)
                mated_p: list[tuple[str, str]] = [
                    (a, b)
                    for ident in shared_ids
                    for a in adv_pA[ident]
                    for b in adv_pB[ident]
                ]
                pA_ids = list(adv_pA.keys())
                pB_ids = list(adv_pB.keys())
                nonmated_p: list[tuple[str, str]] = []
                attempts = 0
                while len(nonmated_p) < len(mated_p) and attempts < len(mated_p) * 20:
                    attempts += 1
                    if not pA_ids or not pB_ids:
                        break
                    id_a = rng.choice(pA_ids)
                    id_b = rng.choice(pB_ids)
                    if id_a == id_b:
                        continue
                    nonmated_p.append((rng.choice(adv_pA[id_a]), rng.choice(adv_pB[id_b])))

                m_sc  = score_adv_fn(mated_p,    adv_emb)
                nm_sc = score_adv_fn(nonmated_p, adv_emb)
                pat_combo_scores[(pA, pB)] = (m_sc, nm_sc)

                _draw_dist(ax_c, m_sc,  col_pA,   f"Mated P{pA}↔P{pB}", bins=30, show_n=False)
                _draw_dist(ax_c, nm_sc, NM_COLOR, "Non-mated",           bins=30, linestyle="--", show_n=False)
                _style_ax(ax_c, f"Pattern {pA} vs Pattern {pB}", legend_inside=True)

            for idx in range(n_combos, nrows * ncols):
                axes_c[idx // ncols][idx % ncols].set_visible(False)
            _unify_ylim([axes_c[r][c] for r in range(nrows) for c in range(ncols)])
            fig_c.tight_layout()
            _save(fig_c, out_dir / f"plot3_cross_pattern_{tag}_clean.png")

        # Store per-combo scores for cross-model overview
        per_pattern_data[pid]["pat_combo_scores"] = pat_combo_scores

        # ── Plot 4 — combined ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_dist(ax, m_scores,  COL_CTRL_ADV, "Mated ctrl↔adv",     bins=BINS, fill=False, show_n=False)
        _draw_dist(ax, nm_scores, COL_CTRL_ADV, "Non-mated ctrl↔adv", bins=BINS, linestyle=":", fill=False, show_n=False)
        for c_idx, (pA, pB) in enumerate(combos_pat):
            col_pA, _ = COMBO_COLORS[c_idx % len(COMBO_COLORS)]
            m_sc, nm_sc = pat_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax, m_sc,  col_pA, f"Mated P{pA}↔P{pB}",     bins=30, fill=False, show_n=False)
            _draw_dist(ax, nm_sc, col_pA, f"Non-mated P{pA}↔P{pB}", bins=30, linestyle=":", fill=False, show_n=False)
        _style_ax(ax, f"{model_label} – {label}\nCombined: ctrl↔adv + cross-pattern adv↔adv",
                  legend_inside=True)
        _unify_ylim([ax])
        fig.tight_layout()
        _save(fig, out_dir / f"plot4_combined_{tag}_clean.png")

        print("done")

    return per_pattern_data


# ─────────────────────────────────────────────────────────────────────────────
# All-models aggregated plot generation
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS_LABEL = "All Models (avg sim)"

def generate_plots_for_all_models(
    models:   list[str],
    con:      sqlite3.Connection,
    out_root: Path,
    rng:      random.Random,
) -> dict | None:
    """
    Generate the same four plots and biometric stats as generate_plots_for_model,
    but using scores averaged across all models for every pair — mirroring the
    gallery-search aggregation in the ranking tab.

    Output folder: out_root / "all_models"
    """
    label = ALL_MODELS_LABEL
    print(f"\n[{label}]")

    out_dir = out_root / "all_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data for every model ──────────────────────────────────────────────
    ctrl_embs_all = load_all_control_embeddings(con, models)
    adv_embs_all  = load_all_adversarial_embeddings(con, models)
    atk_meta      = load_attack_meta(con)
    ctrl_id       = load_capture_identity(con)

    # Use the union of all per-model path sets
    all_ctrl_paths: set[str] = set()
    for m_emb in ctrl_embs_all.values():
        all_ctrl_paths |= m_emb.keys()

    all_adv_paths: set[str] = set()
    for m_emb in adv_embs_all.values():
        all_adv_paths |= m_emb.keys()

    if not all_ctrl_paths:
        print(f"  No control embeddings across models – skipping.")
        return None
    if not all_adv_paths:
        print(f"  No adversarial embeddings across models – skipping.")
        return None

    # Group control paths by identity (union across models)
    ctrl_by_id: dict[str, list[str]] = defaultdict(list)
    for path, ident in ctrl_id.items():
        if path in all_ctrl_paths:
            ctrl_by_id[ident].append(path)

    # Group adversarial paths by identity (union across models)
    adv_by_id: dict[str, list[str]] = defaultdict(list)
    for path in all_adv_paths:
        meta  = atk_meta.get(path, {})
        ident = meta.get("identity_name", "")
        if ident:
            adv_by_id[ident].append(path)

    # ── PLOT 1 ─────────────────────────────────────────────────────────────────
    print("  Plot 1 …", end=" ")
    mated_pairs    = build_mated_pairs(ctrl_by_id, adv_by_id)
    nonmated_pairs = build_nonmated_pairs(ctrl_by_id, adv_by_id, len(mated_pairs), rng)

    mated_scores    = scores_from_pairs_all_models(mated_pairs,    ctrl_embs_all, adv_embs_all)
    nonmated_scores = scores_from_pairs_all_models(nonmated_pairs, ctrl_embs_all, adv_embs_all)

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_dist(ax, mated_scores,    PALETTE["mated"],     "Mated")
    _draw_dist(ax, nonmated_scores, PALETTE["non_mated"], "Non-mated", linestyle=":")
    _style_ax(ax, f"{label}\nMated vs Non-Mated Score Distributions")
    _draw_threshold_lines(ax, mated_scores, nonmated_scores)
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot1_mated_vs_nonmated.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    _draw_dist(ax, mated_scores,    PALETTE["mated"],     "Mated",     show_n=False)
    _draw_dist(ax, nonmated_scores, PALETTE["non_mated"], "Non-mated", linestyle=":", show_n=False)
    _style_ax(ax, f"{label}\nMated vs Non-Mated Score Distributions", legend_inside=True)
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot1_mated_vs_nonmated_clean.png")
    print("done")

    # ── PLOT 2 ─────────────────────────────────────────────────────────────────
    print("  Plot 2 …", end=" ")

    def _filter_adv_by_pose(flag: str) -> dict[str, list[str]]:
        filtered: dict[str, list[str]] = defaultdict(list)
        for path, meta in atk_meta.items():
            if path in all_adv_paths and meta.get(flag, 0):
                ident = meta.get("identity_name", "")
                if ident:
                    filtered[ident].append(path)
        return filtered

    adv_frontal = _filter_adv_by_pose("is_frontal")
    adv_profile = _filter_adv_by_pose("is_profile")

    def _pose_mated_nonmated(adv_subset: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
        m_pairs  = build_mated_pairs(ctrl_by_id, adv_subset)
        nm_pairs = build_nonmated_pairs(ctrl_by_id, adv_subset, len(m_pairs), rng)
        return (
            scores_from_pairs_all_models(m_pairs,  ctrl_embs_all, adv_embs_all),
            scores_from_pairs_all_models(nm_pairs, ctrl_embs_all, adv_embs_all),
        )

    m_front,  nm_front  = _pose_mated_nonmated(adv_frontal)
    m_profil, nm_profil = _pose_mated_nonmated(adv_profile)

    COL_FRONTAL = "#1f77b4"
    COL_PROFILE = "#d62728"
    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_dist(ax, m_front,   COL_FRONTAL, "Mated – Frontal")
    _draw_dist(ax, nm_front,  COL_FRONTAL, "Non-mated – Frontal", linestyle=":")
    _draw_dist(ax, m_profil,  COL_PROFILE, "Mated – Profile")
    _draw_dist(ax, nm_profil, COL_PROFILE, "Non-mated – Profile", linestyle=":")
    _style_ax(ax, f"{label}\nPose-Stratified Score Distributions")
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot2_pose_breakdown.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_dist(ax, m_front,   COL_FRONTAL, "Mated – Frontal",     show_n=False)
    _draw_dist(ax, nm_front,  COL_FRONTAL, "Non-mated – Frontal", linestyle=":", show_n=False)
    _draw_dist(ax, m_profil,  COL_PROFILE, "Mated – Profile",     show_n=False)
    _draw_dist(ax, nm_profil, COL_PROFILE, "Non-mated – Profile", linestyle=":", show_n=False)
    _style_ax(ax, f"{label}\nPose-Stratified Score Distributions", legend_inside=True)
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot2_pose_breakdown_clean.png")
    print("done")

    # ── PLOT 3 ─────────────────────────────────────────────────────────────────
    pattern_combo_scores: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    print("  Plot 3 …", end=" ")

    pattern_ids: set[int] = set()
    for meta in atk_meta.values():
        pid = meta.get("pattern_id", 0)
        if pid and pid > 0:
            pattern_ids.add(int(pid))
    pattern_ids_sorted = sorted(pattern_ids)

    adv_by_id_pat: dict[tuple[str, int], list[str]] = defaultdict(list)
    for path, meta in atk_meta.items():
        if path not in all_adv_paths:
            continue
        pid   = int(meta.get("pattern_id", 0))
        ident = meta.get("identity_name", "")
        if pid > 0 and ident:
            adv_by_id_pat[(ident, pid)].append(path)

    adv_by_pat: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (ident, pid), paths in adv_by_id_pat.items():
        adv_by_pat[pid][ident].extend(paths)

    combos = list(itertools.combinations(pattern_ids_sorted, 2))

    if not combos:
        print("no pattern combinations found – skipping.")
    else:
        n_combos = len(combos)
        ncols    = min(3, n_combos)
        nrows    = (n_combos + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"{label}\nCross-Pattern Score Distributions (adversarial pA vs adversarial pB)",
            fontsize=11, fontweight="bold",
        )

        COMBO_COLORS = [
            ("#e41a1c", "#377eb8"), ("#4daf4a", "#984ea3"),
            ("#ff7f00", "#a65628"), ("#f781bf", "#999999"),
            ("#e41a1c", "#4daf4a"), ("#377eb8", "#ff7f00"),
            ("#984ea3", "#a65628"), ("#f781bf", "#e41a1c"),
            ("#4daf4a", "#377eb8"), ("#ff7f00", "#984ea3"),
        ]
        NM_COLOR = "#555555"

        for idx, (pA, pB) in enumerate(combos):
            ax         = axes[idx // ncols][idx % ncols]
            col_pA, _  = COMBO_COLORS[idx % len(COMBO_COLORS)]

            adv_pA: dict[str, list[str]] = dict(adv_by_pat.get(pA, {}))
            adv_pB: dict[str, list[str]] = dict(adv_by_pat.get(pB, {}))

            shared_ids = set(adv_pA) & set(adv_pB)
            mated_pairs_p3: list[tuple[str, str]] = []
            for ident in shared_ids:
                for a in adv_pA[ident]:
                    for b in adv_pB[ident]:
                        mated_pairs_p3.append((a, b))

            n_nm    = len(mated_pairs_p3)
            pA_ids  = list(adv_pA.keys())
            pB_ids  = list(adv_pB.keys())
            nonmated_pairs_p3: list[tuple[str, str]] = []
            attempts = 0
            while len(nonmated_pairs_p3) < n_nm and attempts < n_nm * 20:
                attempts += 1
                if not pA_ids or not pB_ids:
                    break
                id_a = rng.choice(pA_ids)
                id_b = rng.choice(pB_ids)
                if id_a == id_b:
                    continue
                a = rng.choice(adv_pA[id_a])
                b = rng.choice(adv_pB[id_b])
                nonmated_pairs_p3.append((a, b))

            m_scores  = scores_adv_adv_all_models(mated_pairs_p3,    adv_embs_all)
            nm_scores = scores_adv_adv_all_models(nonmated_pairs_p3, adv_embs_all)

            pattern_combo_scores[(pA, pB)] = (m_scores, nm_scores)
            _draw_dist(ax, m_scores,  col_pA,   f"Mated P{pA}↔P{pB}", bins=30)
            _draw_dist(ax, nm_scores, NM_COLOR, "Non-mated",           bins=30, linestyle="--")
            _style_ax(ax, f"Pattern {pA} vs Pattern {pB}")
            _draw_threshold_lines(ax, m_scores, nm_scores)

        for idx in range(n_combos, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        _unify_ylim([axes[r][c] for r in range(nrows) for c in range(ncols)])
        fig.tight_layout()
        _save(fig, out_dir / "plot3_cross_pattern.png")

        # ── Plot 3 clean variant ───────────────────────────────────────────────
        fig_c, axes_c = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        fig_c.suptitle(
            f"{label}\nCross-Pattern Score Distributions (adversarial pA vs adversarial pB)",
            fontsize=11, fontweight="bold",
        )
        for idx, (pA, pB) in enumerate(combos):
            col_pA, _ = COMBO_COLORS[idx % len(COMBO_COLORS)]
            ax_c = axes_c[idx // ncols][idx % ncols]
            m_sc, nm_sc = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax_c, m_sc,  col_pA,   f"Mated P{pA}↔P{pB}", bins=30, show_n=False)
            _draw_dist(ax_c, nm_sc, NM_COLOR, "Non-mated",           bins=30, linestyle="--", show_n=False)
            _style_ax(ax_c, f"Pattern {pA} vs Pattern {pB}", legend_inside=True)
        for idx in range(n_combos, nrows * ncols):
            axes_c[idx // ncols][idx % ncols].set_visible(False)
        _unify_ylim([axes_c[r][c] for r in range(nrows) for c in range(ncols)])
        fig_c.tight_layout()
        _save(fig_c, out_dir / "plot3_cross_pattern_clean.png")
        print("done")

    # ── PLOT 4 ─────────────────────────────────────────────────────────────────
    print("  Plot 4 …", end=" ")
    fig, ax = plt.subplots(figsize=(9, 5))

    COL_CTRL_ADV = PALETTE["mated"]
    _draw_dist(ax, mated_scores,    COL_CTRL_ADV, "Mated ctrl↔adv (all)",     bins=BINS, fill=False)
    _draw_dist(ax, nonmated_scores, COL_CTRL_ADV, "Non-mated ctrl↔adv (all)", bins=BINS, linestyle=":", fill=False)

    if combos:
        COMBO_COLORS_P4 = [
            ("#e41a1c", "#377eb8"), ("#4daf4a", "#984ea3"),
            ("#ff7f00", "#a65628"), ("#f781bf", "#999999"),
            ("#e41a1c", "#4daf4a"), ("#377eb8", "#ff7f00"),
            ("#984ea3", "#a65628"), ("#f781bf", "#e41a1c"),
            ("#4daf4a", "#377eb8"), ("#ff7f00", "#984ea3"),
        ]
        for idx, (pA, pB) in enumerate(combos):
            col_pA, _ = COMBO_COLORS_P4[idx % len(COMBO_COLORS_P4)]
            m_sc, nm_sc = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax, m_sc,  col_pA, f"Mated P{pA}↔P{pB}",     bins=30, fill=False)
            _draw_dist(ax, nm_sc, col_pA, f"Non-mated P{pA}↔P{pB}", bins=30, linestyle=":", fill=False)

    _style_ax(ax, f"{label}\nCombined: ctrl↔adv (Plot 1) + cross-pattern adv↔adv (Plot 3)")
    _unify_ylim([ax])
    fig.tight_layout()
    _save(fig, out_dir / "plot4_combined.png")

    # ── Plot 4 clean variant ───────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(9, 5))
    _draw_dist(ax_c, mated_scores,    COL_CTRL_ADV, "Mated ctrl↔adv (all)",     bins=BINS, fill=False, show_n=False)
    _draw_dist(ax_c, nonmated_scores, COL_CTRL_ADV, "Non-mated ctrl↔adv (all)", bins=BINS, linestyle=":", fill=False, show_n=False)
    if combos:
        for idx, (pA, pB) in enumerate(combos):
            col_pA, _ = COMBO_COLORS_P4[idx % len(COMBO_COLORS_P4)]
            m_sc, nm_sc = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax_c, m_sc,  col_pA, f"Mated P{pA}↔P{pB}",     bins=30, fill=False, show_n=False)
            _draw_dist(ax_c, nm_sc, col_pA, f"Non-mated P{pA}↔P{pB}", bins=30, linestyle=":", fill=False, show_n=False)
    _style_ax(ax_c, f"{label}\nCombined: ctrl↔adv (Plot 1) + cross-pattern adv↔adv (Plot 3)",
              legend_inside=True)
    _unify_ylim([ax_c])
    fig_c.tight_layout()
    _save(fig_c, out_dir / "plot4_combined_clean.png")
    print("done")

    # ── PER-PATTERN CLEAN PLOTS ────────────────────────────────────────────────
    per_pattern_data = _generate_per_pattern_clean_plots(
        pattern_ids_sorted = pattern_ids_sorted,
        adv_by_pat         = adv_by_pat,
        ctrl_by_id         = ctrl_by_id,
        ctrl_emb           = ctrl_embs_all,   # passed through to score_fn lambda
        adv_emb            = adv_embs_all,    # passed through to score_fn / score_adv_fn
        atk_meta           = atk_meta,
        rng                = rng,
        model_label        = label,
        out_dir            = out_dir,
        score_fn           = lambda pairs, ctrl, adv: scores_from_pairs_all_models(pairs, ctrl, adv),
        score_adv_fn       = lambda pairs, adv: scores_adv_adv_all_models(pairs, adv),
    )

    # ── BIOMETRIC STATS ────────────────────────────────────────────────────────
    compute_biometric_stats(
        model           = label,
        out_dir         = out_dir,
        mated_scores    = mated_scores,
        nonmated_scores = nonmated_scores,
        pattern_combos  = pattern_combo_scores,
    )

    return {
        "model":                label,
        "mated_scores":         mated_scores,
        "nonmated_scores":      nonmated_scores,
        "adv_frontal_scores":   (m_front, nm_front),
        "adv_profile_scores":   (m_profil, nm_profil),
        "pattern_combo_scores": pattern_combo_scores,
        "combos":               combos if combos else [],
        "per_pattern_data":     per_pattern_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optimal-accuracy threshold table figures
# ─────────────────────────────────────────────────────────────────────────────

def _optimal_threshold_accuracy(
    mated: np.ndarray,
    nonmated: np.ndarray,
    n_thresholds: int = 400,
) -> tuple[float, float]:
    """
    Find the threshold t* that maximises balanced accuracy:
        Acc(t) = 0.5 * TPR(t) + 0.5 * TNR(t)
               = 0.5 * (mated >= t).mean() + 0.5 * (nonmated < t).mean()

    Returns (threshold, accuracy).  Falls back to (nan, nan) when either
    array is empty.
    """
    if len(mated) == 0 or len(nonmated) == 0:
        return float("nan"), float("nan")

    all_scores = np.concatenate([mated, nonmated])
    lo, hi     = float(all_scores.min()), float(all_scores.max())
    if lo == hi:
        return lo, float("nan")

    thresholds = np.linspace(lo, hi, n_thresholds)
    best_acc   = -1.0
    best_th    = thresholds[0]

    for t in thresholds:
        tpr = float((mated    >= t).mean())
        tnr = float((nonmated  < t).mean())
        acc = 0.5 * (tpr + tnr)
        if acc > best_acc:
            best_acc = acc
            best_th  = float(t)

    return best_th, best_acc


def generate_accuracy_threshold_tables(
    overview_list: list[dict],
    out_root: Path,
) -> None:
    """
    Produce two dark-styled table PNG figures saved to out_root:

    1. ``fig_accuracy_threshold_accuracy.png``
       Combined table: rows = "All patterns", "P1", "P2", … then "P1↔P2", …
       Separator line between the two sections.
       Columns: one per model entry in overview_list (last = "Avg (sim)").
       No "Average" column — Avg (sim) IS the all-models entry.
       Cells show balanced accuracy only.

    2. ``fig_accuracy_threshold_thresholds.png``
       Identical layout but cells show the optimal threshold value only.

    Visual style: dark navy background, cyan bold header text, alternating
    row stripes, white cell text, no chart title.
    """
    if not overview_list:
        print("  No model data — skipping accuracy threshold tables.")
        return

    # ── Palette (matches reference screenshot) ───────────────────────────────
    C_BG        = "#1a2332"   # figure background
    C_HEADER_BG = "#0d1b2a"   # header row background
    C_HEADER_FG = "#4dd9e8"   # header text (cyan)
    C_ROW_A     = "#1a2332"   # odd data row background
    C_ROW_B     = "#1e2d40"   # even data row background (slightly lighter)
    C_SEP_ROW   = "#243447"   # section-separator row background
    C_TEXT      = "#ffffff"   # regular cell text
    C_SEP_LINE  = "#3a5068"   # thin horizontal rule colour
    C_ROWLABEL  = "#4dd9e8"   # row-label text (cyan, bold)
    C_AVG_BG    = "#0d1b2a"   # Avg (sim) column background (same as header)
    C_AVG_FG    = "#f0c040"   # Avg (sim) column text (gold, to match ref)

    ROW_H   = 0.52   # inches per data row
    HDR_H   = 0.60   # inches for header row
    COL_W_LABEL = 1.30   # inches for the row-label column
    COL_W_DATA  = 1.45   # inches per data column

    # ── Build column labels (no extra "Average" column) ──────────────────────
    col_labels_raw = [md["model"] for md in overview_list]
    col_labels = [
        "Avg (sim)" if lbl == ALL_MODELS_LABEL else lbl.split("/")[-1]
        for lbl in col_labels_raw
    ]
    n_cols = len(col_labels)   # one per model/avg entry — no extra average

    # ── Collect row data ──────────────────────────────────────────────────────
    all_pids: list[int] = sorted({
        pid
        for md in overview_list
        for pid in md.get("per_pattern_data", {}).keys()
    })
    all_combos: list[tuple[int, int]] = sorted({
        combo
        for md in overview_list
        for combo in md.get("pattern_combo_scores", {}).keys()
    })

    # Section 1 rows
    sec1_labels: list[str] = ["All patterns"] + [f"P{pid}" for pid in all_pids]
    sec1_data:   list[list[tuple[float, float]]] = []

    row_all: list[tuple[float, float]] = []
    for md in overview_list:
        m  = md.get("mated_scores",    np.array([]))
        nm = md.get("nonmated_scores", np.array([]))
        row_all.append(_optimal_threshold_accuracy(m, nm))
    sec1_data.append(row_all)

    for pid in all_pids:
        row_pid: list[tuple[float, float]] = []
        for md in overview_list:
            ppd = md.get("per_pattern_data", {}).get(pid, {})
            m   = ppd.get("mated_scores",   np.array([]))
            nm  = ppd.get("nonmated_scores", np.array([]))
            row_pid.append(_optimal_threshold_accuracy(m, nm))
        sec1_data.append(row_pid)

    # Section 2 rows
    sec2_labels: list[str] = [f"P{pA}↔P{pB}" for pA, pB in all_combos]
    sec2_data:   list[list[tuple[float, float]]] = []

    for (pA, pB) in all_combos:
        row_combo: list[tuple[float, float]] = []
        for md in overview_list:
            m_sc, nm_sc = md.get("pattern_combo_scores", {}).get(
                (pA, pB), (np.array([]), np.array([]))
            )
            row_combo.append(_optimal_threshold_accuracy(m_sc, nm_sc))
        sec2_data.append(row_combo)

    # Combined row sequence with a separator marker between sections
    SEP = "__sep__"
    all_row_labels = (
        sec1_labels
        + ([SEP] if sec2_labels else [])
        + sec2_labels
    )
    all_row_data = (
        sec1_data
        + ([None] if sec2_labels else [])   # None marks separator row
        + sec2_data
    )

    n_data_rows = sum(1 for d in all_row_data if d is not None)
    n_sep_rows  = sum(1 for d in all_row_data if d is None)
    n_total_rows = len(all_row_data)

    # ── Generic dark-style renderer ───────────────────────────────────────────
    def _render_dark_table(
        cell_value_fn,   # callable(th, acc) → str
        out_path: Path,
    ) -> None:
        fig_w = COL_W_LABEL + COL_W_DATA * n_cols + 0.15
        fig_h = HDR_H + ROW_H * n_data_rows + (ROW_H * 0.35) * n_sep_rows + 0.15
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C_BG)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(C_BG)
        ax.set_axis_off()
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)

        total_w = COL_W_LABEL + COL_W_DATA * n_cols
        left    = (fig_w - total_w) / 2
        top     = fig_h - 0.075   # small top margin

        # Column x-positions (left edge of each column)
        col_xs = [left] + [left + COL_W_LABEL + COL_W_DATA * i for i in range(n_cols)]
        col_ws = [COL_W_LABEL] + [COL_W_DATA] * n_cols

        # ── Header row ────────────────────────────────────────────────────────
        y_hdr = top - HDR_H
        for ci, (cx, cw) in enumerate(zip(col_xs, col_ws)):
            bg = C_HEADER_BG
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, y_hdr), cw, HDR_H,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, clip_on=False,
            ))
            if ci == 0:
                label = "Row"
            else:
                label = col_labels[ci - 1]
            is_avg = (ci > 0 and col_labels[ci - 1] == "Avg (sim)")
            fg = C_AVG_FG if is_avg else C_HEADER_FG
            ax.text(
                cx + cw / 2, y_hdr + HDR_H / 2, label,
                ha="center", va="center", color=fg,
                fontsize=9, fontweight="bold",
            )

        # Thin bottom border under header
        ax.plot([left, left + total_w], [y_hdr, y_hdr],
                color=C_SEP_LINE, lw=0.8)

        # ── Data rows ─────────────────────────────────────────────────────────
        y_cursor = y_hdr
        data_row_idx = 0   # counts only real (non-separator) rows

        for ri, (row_label, row_data) in enumerate(zip(all_row_labels, all_row_data)):

            # Separator row
            if row_data is None:
                sep_h = ROW_H * 0.35
                y_cursor -= sep_h
                for cx, cw in zip(col_xs, col_ws):
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (cx, y_cursor), cw, sep_h,
                        boxstyle="square,pad=0", linewidth=0,
                        facecolor=C_SEP_ROW, clip_on=False,
                    ))
                ax.plot([left, left + total_w], [y_cursor + sep_h, y_cursor + sep_h],
                        color=C_SEP_LINE, lw=0.5, linestyle="--")
                ax.plot([left, left + total_w], [y_cursor, y_cursor],
                        color=C_SEP_LINE, lw=0.5, linestyle="--")
                continue

            row_h = ROW_H
            y_top = y_cursor
            y_cursor -= row_h

            row_bg = C_ROW_A if data_row_idx % 2 == 0 else C_ROW_B
            data_row_idx += 1

            # Row-label cell
            cx0, cw0 = col_xs[0], col_ws[0]
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx0, y_cursor), cw0, row_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=row_bg, clip_on=False,
            ))
            ax.text(
                cx0 + 0.12, y_cursor + row_h / 2, row_label,
                ha="left", va="center", color=C_ROWLABEL,
                fontsize=9, fontweight="bold",
            )

            # Data cells
            for ci, (th, acc) in enumerate(row_data):
                cx, cw = col_xs[ci + 1], col_ws[ci + 1]
                is_avg = col_labels[ci] == "Avg (sim)"
                cell_bg = C_AVG_BG if is_avg else row_bg
                ax.add_patch(mpatches.FancyBboxPatch(
                    (cx, y_cursor), cw, row_h,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor=cell_bg, clip_on=False,
                ))
                txt = cell_value_fn(th, acc)
                fg  = C_AVG_FG if is_avg else C_TEXT
                ax.text(
                    cx + cw / 2, y_cursor + row_h / 2, txt,
                    ha="center", va="center", color=fg,
                    fontsize=9,
                )

            # Thin row separator
            ax.plot([left, left + total_w], [y_cursor, y_cursor],
                    color=C_SEP_LINE, lw=0.4)

        # Outer border
        outer_h = top - y_cursor
        ax.add_patch(mpatches.FancyBboxPatch(
            (left, y_cursor), total_w, outer_h,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor=C_SEP_LINE, facecolor="none", clip_on=False,
        ))
        # Vertical separator after row-label column
        ax.plot(
            [left + COL_W_LABEL, left + COL_W_LABEL],
            [y_cursor, top],
            color=C_SEP_LINE, lw=0.8,
        )
        # Vertical separator before Avg (sim) column (last column)
        x_avg = col_xs[-1]
        ax.plot(
            [x_avg, x_avg],
            [y_cursor, top],
            color=C_SEP_LINE, lw=1.2, linestyle="--",
        )

        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=C_BG, edgecolor="none")
        plt.close(fig)
        print(f"  Saved → {out_path}")

    # ── Render accuracy table ─────────────────────────────────────────────────
    print("  Building accuracy table …", end=" ")
    _render_dark_table(
        cell_value_fn = lambda th, acc: "—" if np.isnan(acc) else f"{acc:.3f}",
        out_path      = out_root / "fig_accuracy_threshold_accuracy.png",
    )
    print("done")

    # ── Render threshold table ────────────────────────────────────────────────
    print("  Building threshold table …", end=" ")
    _render_dark_table(
        cell_value_fn = lambda th, acc: "—" if np.isnan(th) else f"{th:.3f}",
        out_path      = out_root / "fig_accuracy_threshold_thresholds.png",
    )
    print("done")




def compute_biometric_stats(
    model:    str,
    out_dir:  Path,
    # Plot-1 scores
    mated_scores:    np.ndarray,
    nonmated_scores: np.ndarray,
    # Plot-3 per-combo scores: {(pA, pB): (mated_arr, nonmated_arr)}
    pattern_combos:  dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Compute EER, FMR100, FMR1000 and export ROC rates using pyeer.
    Outputs per model_dir:
      biometric_stats.csv   – summary table (EER, FMR100, FMR1000, AUC, d')
      roc_<tag>.csv         – FMR / FNMR columns for each experiment
    """
    if not HAS_PYEER:
        print("  [biometric stats] pyeer not installed – skipping. "
              "Run: pip install pyeer")
        return

    import csv

    experiments: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Plot-1 experiment
    if len(mated_scores) > 0 and len(nonmated_scores) > 0:
        experiments.append(("ctrl_vs_adversarial", mated_scores, nonmated_scores))

    # Plot-3 per-pattern-combo experiments
    for (pA, pB), (m_sc, nm_sc) in sorted(pattern_combos.items()):
        if len(m_sc) > 0 and len(nm_sc) > 0:
            experiments.append((f"P{pA}_vs_P{pB}", m_sc, nm_sc))

    # Aggregate: pool all mated and all non-mated scores from every experiment
    # with equal weight (sample the same n from each so no single experiment
    # dominates due to differing pair counts).
    all_mated_parts   = [mated_scores]   + [m for _, (m, _) in sorted(pattern_combos.items()) if len(m) > 0]
    all_nonmated_parts= [nonmated_scores]+ [nm for _, (_, nm) in sorted(pattern_combos.items()) if len(nm) > 0]
    if all_mated_parts and all_nonmated_parts:
        min_m  = min(len(a) for a in all_mated_parts)
        min_nm = min(len(a) for a in all_nonmated_parts)
        rng_agg = random.Random(RANDOM_SEED)
        agg_mated   = np.concatenate([
            np.array(rng_agg.sample(list(a), min(min_m,  len(a))), dtype=np.float32)
            for a in all_mated_parts
        ])
        agg_nonmated = np.concatenate([
            np.array(rng_agg.sample(list(a), min(min_nm, len(a))), dtype=np.float32)
            for a in all_nonmated_parts
        ])
        if len(agg_mated) > 0 and len(agg_nonmated) > 0:
            experiments.append(("all_mated_vs_nonmated", agg_mated, agg_nonmated))

    if not experiments:
        print("  [biometric stats] no score pairs available – skipping.")
        return

    print(f"  Biometric stats ({len(experiments)} experiment(s)) …", end=" ")

    summary_rows: list[dict] = []

    for tag, gscores, iscores in experiments:
        try:
            # Standard: mated=genuine, nonmated=impostor
            stats = get_eer_stats(gscores.tolist(), iscores.tolist())
        except Exception as exc:
            print(f"\n  [WARN] pyeer failed for {tag}: {exc}")
            continue

        # TPR = 1 - FNMR at each operating point
        tpr_eer    = round(1.0 - stats.eer,    6)
        tpr_fmr100 = round(1.0 - stats.fmr100, 6)
        tpr_fmr1000= round(1.0 - stats.fmr1000,6)

        summary_rows.append({
            "experiment":      tag,
            "n_mated":         len(gscores),
            "n_nonmated":      len(iscores),
            "EER":             round(stats.eer,     6),
            "Acc@EER":         tpr_eer,   # Acc = 1 - EER
            "FNMR@FMR100":     round(stats.fmr100,  6),
            "TPR@FMR100":      tpr_fmr100,
            "FNMR@FMR1000":    round(stats.fmr1000, 6),
            "TPR@FMR1000":     tpr_fmr1000,
            "AUC":             round(stats.auc,     6),
            "EER_thresh":      round(stats.eer_th,  6),
        })

        # Export ROC (FMR vs 1-FNMR) as CSV
        roc_path = out_dir / f"roc_{tag}.csv"
        try:
            export_error_rates(stats.fmr, stats.fnmr, str(roc_path))
        except Exception as exc:
            print(f"\n  [WARN] could not export ROC for {tag}: {exc}")

    if not summary_rows:
        print("no results.")
        return

    # Write summary CSV
    summary_path = out_dir / "biometric_stats.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"done → {summary_path.name} + {len(summary_rows)} roc_*.csv file(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Biometric stats summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_biometric_summary(models: list[str], out_root: Path) -> None:
    """
    Read biometric_stats.csv and roc_*.csv from each model's subfolder and
    produce a single summary figure saved to out_root/biometric_summary.png.

    Layout: one row per model, two columns:
      Col 1 – grouped bar chart: EER / FMR100 / FMR1000 per experiment
      Col 2 – ROC curve (1-FNMR vs FMR) per experiment
    """
    import csv

    # Colour cycle for experiments (same order every model)
    EXP_COLORS = [
        "#1f77b4", "#d62728", "#2ca02c", "#984ea3",
        "#ff7f00", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    # ── collect data ──────────────────────────────────────────────────────────
    model_data: list[dict] = []   # one entry per model that has data

    # Include the all_models aggregated folder alongside per-model folders
    all_model_entries = list(models) + [ALL_MODELS_LABEL]

    for model in all_model_entries:
        folder_name = "all_models" if model == ALL_MODELS_LABEL else model
        model_dir = out_root / folder_name
        stats_path = model_dir / "biometric_stats.csv"
        if not stats_path.exists():
            continue

        rows: list[dict] = []
        with open(stats_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        # Load ROC files
        roc_data: dict[str, tuple[list, list]] = {}
        for row in rows:
            tag = row["experiment"]
            roc_path = model_dir / f"roc_{tag}.csv"
            if roc_path.exists():
                fmr_vals, fnmr_vals = [], []
                with open(roc_path, newline="") as f:
                    for r in csv.DictReader(f):
                        try:
                            fmr_vals.append(float(r["FMR"]))
                            fnmr_vals.append(float(r["FNMR"]))
                        except (ValueError, KeyError):
                            pass
                if fmr_vals:
                    roc_data[tag] = (fmr_vals, fnmr_vals)

        model_data.append({"model": model, "rows": rows, "roc": roc_data})

    if not model_data:
        print("  [biometric summary] no biometric_stats.csv files found – skipping.")
        return

    n_models = len(model_data)

    # Total rows = one per model/aggregation + one cross-model comparison row
    n_rows_total = n_models + 1

    fig, axes = plt.subplots(
        n_rows_total, 3,
        figsize=(20, 4.5 * n_rows_total),
        squeeze=False,
        gridspec_kw={"width_ratios": [3, 3, 4]},
    )
    fig.suptitle("Biometric Performance Summary", fontsize=13, fontweight="bold", y=1.01)

    # EER = error rate; FMR100/FMR1000 in CSV are FNMR at those operating points
    METRICS = ["EER", "FNMR@FMR100", "FNMR@FMR1000"]

    for row_idx, mdata in enumerate(model_data):
        model   = mdata["model"]
        rows    = mdata["rows"]
        roc     = mdata["roc"]
        exps    = [r["experiment"] for r in rows]
        colors  = [EXP_COLORS[i % len(EXP_COLORS)] for i in range(len(exps))]

        # ── Col 1: grouped bar chart ──────────────────────────────────────────
        ax_bar = axes[row_idx][0]
        n_exp     = len(exps)
        n_metrics = len(METRICS)
        x         = np.arange(n_exp)
        bar_w     = 0.22
        offsets   = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_w

        metric_colors  = {"EER": "#1f77b4", "FNMR@FMR100": "#ff7f0e", "FNMR@FMR1000": "#2ca02c"}
        metric_csv_key = {"EER": "EER", "FNMR@FMR100": "FNMR@FMR100", "FNMR@FMR1000": "FNMR@FMR1000"}

        for m_idx, metric in enumerate(METRICS):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[metric_csv_key[metric]]))
                except (ValueError, KeyError):
                    vals.append(0.0)
            ax_bar.bar(
                x + offsets[m_idx], vals,
                width=bar_w,
                label=metric,
                color=metric_colors[metric],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(
            [e.replace("_", " ") for e in exps],
            fontsize=7, rotation=20, ha="right",
        )
        ax_bar.set_ylabel("Error Rate", fontsize=8)
        ax_bar.set_ylim(bottom=0)
        ax_bar.set_title(f"{model} — EER / FNMR@FMR100 / FNMR@FMR1000", fontsize=9, fontweight="bold")
        ax_bar.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
                      borderaxespad=0, framealpha=0.7)
        ax_bar.grid(axis="y", lw=0.4, alpha=0.5)
        ax_bar.spines[["top", "right"]].set_visible(False)

        # ── Col 2: ROC curves ─────────────────────────────────────────────────
        ax_roc = axes[row_idx][1]

        # Chance line
        ax_roc.plot([0, 1], [0, 1], color="#cccccc", lw=1, linestyle="--", zorder=0)

        for i, exp in enumerate(exps):
            if exp not in roc:
                continue
            fmr_vals, fnmr_vals = roc[exp]
            tpr = [1.0 - v for v in fnmr_vals]   # TPR = 1 - FNMR
            ax_roc.plot(
                fmr_vals, tpr,
                color=colors[i], lw=1.8, alpha=0.85,
                label=exp.replace("_", " "),
            )

            # Mark EER point: x=EER (FMR=FNMR=EER at crossing), y=TPR@EER
            try:
                eer_fmr = float(rows[i]["EER"])
                tpr_eer = float(rows[i]["TPR@EER"])
                if eer_fmr <= 0.3:
                    ax_roc.scatter([eer_fmr], [tpr_eer],
                                   color=colors[i], s=40, zorder=5)
            except (ValueError, KeyError, IndexError):
                pass

        ax_roc.set_xlabel("FMR", fontsize=8)
        ax_roc.set_ylabel("1 − FNMR (TPR)", fontsize=8)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_title(f"{model} — ROC Curves", fontsize=9, fontweight="bold")
        ax_roc.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
                      borderaxespad=0, framealpha=0.7)
        ax_roc.grid(lw=0.4, alpha=0.4)
        ax_roc.spines[["top", "right"]].set_visible(False)

        # ── Col 3: accuracy table ─────────────────────────────────────────────
        ax_tbl = axes[row_idx][2]
        ax_tbl.axis("off")

        TABLE_COLS = ["Experiment", "Acc@EER", "TPR@FMR100", "TPR@FMR1000", "AUC"]
        table_data = []
        for r in rows:
            def _fmt(key):
                try:
                    return f"{float(r[key]):.3f}"
                except (KeyError, ValueError):
                    return "—"
            table_data.append([
                r["experiment"].replace("_", " "),
                _fmt("Acc@EER"),
                _fmt("TPR@FMR100"),
                _fmt("TPR@FMR1000"),
                _fmt("AUC"),
            ])

        tbl = ax_tbl.table(
            cellText=table_data,
            colLabels=TABLE_COLS,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.auto_set_column_width(col=list(range(len(TABLE_COLS))))

        # Style header row
        for col_idx in range(len(TABLE_COLS)):
            cell = tbl[0, col_idx]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

        # Alternate row shading; highlight last row (aggregate experiment)
        last_row = len(table_data)
        for r_idx in range(1, last_row + 1):
            if r_idx == last_row:
                bg = "#d4e6f1"
            else:
                bg = "#f0f4f8" if r_idx % 2 == 0 else "white"
            for col_idx in range(len(TABLE_COLS)):
                cell = tbl[r_idx, col_idx]
                cell.set_facecolor(bg)
                if r_idx == last_row:
                    cell.set_text_props(fontweight="bold")

        ax_tbl.set_title(f"{model} — Accuracy Table", fontsize=9, fontweight="bold")

    # ── Last row: cross-model comparison (raw EER / FNMR@FMR100 / FNMR@FMR1000) ──
    # One subplot per metric, spanning the full row; each subplot shows one bar
    # per model/aggregation (x-axis) grouped by experiment (hue).
    # The three axes are merged by hiding the middle two and using only the
    # first, or we draw three side-by-side charts in the three columns.
    COMP_METRICS = [
        ("EER",          "EER",          "#1f77b4"),
        ("FNMR@FMR100",  "FNMR@FMR100",  "#ff7f0e"),
        ("FNMR@FMR1000", "FNMR@FMR1000", "#2ca02c"),
    ]

    # Collect the union of experiment tags across all models (preserve order
    # of first occurrence so columns are consistent).
    seen_exps: list[str] = []
    for mdata in model_data:
        for r in mdata["rows"]:
            tag = r["experiment"]
            if tag not in seen_exps:
                seen_exps.append(tag)

    model_labels = [md["model"] for md in model_data]
    n_model_entries = len(model_labels)
    # Shorten long model names for the x-axis tick labels
    short_labels = [
        lbl if len(lbl) <= 22 else lbl[:20] + "…"
        for lbl in model_labels
    ]

    # Colour per experiment (consistent with EXP_COLORS above)
    exp_color_map = {
        exp: EXP_COLORS[i % len(EXP_COLORS)]
        for i, exp in enumerate(seen_exps)
    }

    comp_row = n_models   # index of the last (comparison) row

    for col_idx, (csv_key, metric_label, _bar_color) in enumerate(COMP_METRICS):
        ax = axes[comp_row][col_idx]

        n_exps     = len(seen_exps)
        bar_w      = 0.8 / max(n_exps, 1)
        x          = np.arange(n_model_entries)
        offsets    = np.linspace(
            -(n_exps - 1) / 2,
             (n_exps - 1) / 2,
            n_exps,
        ) * bar_w

        for e_idx, exp in enumerate(seen_exps):
            vals = []
            for mdata in model_data:
                # Find the row for this experiment in this model's CSV
                val = None
                for r in mdata["rows"]:
                    if r["experiment"] == exp:
                        try:
                            val = float(r[csv_key])
                        except (KeyError, ValueError):
                            val = None
                        break
                vals.append(val if val is not None else float("nan"))

            # Highlight the all_models bar with a hatched pattern
            hatches = [
                "//" if lbl == ALL_MODELS_LABEL else ""
                for lbl in model_labels
            ]
            for xi, (v, hatch) in enumerate(zip(vals, hatches)):
                ax.bar(
                    xi + offsets[e_idx],
                    v if not np.isnan(v) else 0,
                    width=bar_w,
                    label=exp.replace("_", " ") if xi == 0 else "_nolegend_",
                    color=exp_color_map[exp],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch=hatch,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=7, rotation=25, ha="right")
        ax.set_ylabel(metric_label, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"Cross-Model Comparison — {metric_label}\n"
            f"(bars = experiments; hatched = All Models aggregation)",
            fontsize=9, fontweight="bold",
        )
        ax.legend(
            fontsize=7, loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0, framealpha=0.7,
            title="Experiment", title_fontsize=7,
        )
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = out_root / "biometric_summary.png"
    _save(fig, out_path)
    print(f"  Biometric summary figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern cross-model overview figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_per_pattern_overview_figures(
    overview_list: list[dict],
    out_root: Path,
) -> None:
    """
    For every (plot_type, pattern_id) combination, produce a 2×3 overview figure
    that shows the same clean plot for every model side-by-side (5 individual
    models + 1 all-models average panel).

    The grid is always 2 rows × 3 columns regardless of how many models are in
    overview_list (up to 6 entries: 5 models + avg).  Extra entries beyond 6 are
    silently ignored; fewer than 6 leave the trailing cells blank.

    Output filenames saved to out_root:
        overview_plot1_mated_vs_nonmated_pat{N}_clean.png
        overview_plot2_pose_breakdown_pat{N}_clean.png
        overview_plot3_cross_pattern_pat{N}_clean.png
        overview_plot4_combined_pat{N}_clean.png

    Parameters
    ----------
    overview_list : list of model-data dicts (each entry must have a
        ``per_pattern_data`` key produced by _generate_per_pattern_clean_plots).
        The last entry is expected to be the all-models aggregate.
    out_root : root results directory where the overview files are written.
    """
    if not overview_list:
        print("  No model data — skipping per-pattern overview figures.")
        return

    # Collect the union of all pattern ids present across any model
    all_pattern_ids: set[int] = set()
    for md in overview_list:
        all_pattern_ids.update(md.get("per_pattern_data", {}).keys())
    if not all_pattern_ids:
        print("  No per-pattern data found — skipping per-pattern overview figures.")
        return

    NROWS, NCOLS = 2, 3   # fixed 2×3 grid
    MAX_PANELS   = NROWS * NCOLS   # 6

    # Colour constants (match _generate_per_pattern_clean_plots)
    COL_FRONTAL  = "#1f77b4"
    COL_PROFILE  = "#d62728"
    COMBO_COLORS = [
        ("#e41a1c", "#377eb8"), ("#4daf4a", "#984ea3"),
        ("#ff7f00", "#a65628"), ("#f781bf", "#999999"),
        ("#e41a1c", "#4daf4a"), ("#377eb8", "#ff7f00"),
        ("#984ea3", "#a65628"), ("#f781bf", "#e41a1c"),
        ("#4daf4a", "#377eb8"), ("#ff7f00", "#984ea3"),
    ]
    NM_COLOR     = "#555555"
    COL_CTRL_ADV = PALETTE["mated"]

    # ── Helper: draw a single plot-type into an existing axes ─────────────────
    def _draw_plot1(ax, pd, model_label):
        m  = pd.get("mated_scores",   np.array([]))
        nm = pd.get("nonmated_scores", np.array([]))
        _draw_dist(ax, m,  PALETTE["mated"],     "Mated",     show_n=False)
        _draw_dist(ax, nm, PALETTE["non_mated"], "Non-mated", linestyle=":", show_n=False)
        _style_ax(ax, f"{model_label}\nMated vs Non-Mated", legend_inside=True)
        _unify_ylim([ax])

    def _draw_plot2(ax, pd, model_label):
        m_f,  nm_f  = pd.get("adv_frontal_scores", (np.array([]), np.array([])))
        m_p,  nm_p  = pd.get("adv_profile_scores", (np.array([]), np.array([])))
        _draw_dist(ax, m_f,  COL_FRONTAL, "Mated – Frontal",     show_n=False)
        _draw_dist(ax, nm_f, COL_FRONTAL, "Non-mated – Frontal", linestyle=":", show_n=False)
        _draw_dist(ax, m_p,  COL_PROFILE, "Mated – Profile",     show_n=False)
        _draw_dist(ax, nm_p, COL_PROFILE, "Non-mated – Profile", linestyle=":", show_n=False)
        _style_ax(ax, f"{model_label}\nPose Breakdown", legend_inside=True)
        _unify_ylim([ax])

    def _draw_plot3(ax, pd, model_label, pid):
        combo_scores = pd.get("pat_combo_scores", {})
        combos_pat   = pd.get("combos_pat", [])
        for c_idx, (pA, pB) in enumerate(combos_pat):
            col_pA, _ = COMBO_COLORS[c_idx % len(COMBO_COLORS)]
            m_sc, nm_sc = combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax, m_sc,  col_pA,   f"Mated P{pA}↔P{pB}", bins=30, show_n=False)
            _draw_dist(ax, nm_sc, NM_COLOR, "Non-mated",           bins=30, linestyle="--", show_n=False)
        _style_ax(ax, f"{model_label}\nCross-Pattern (Pat {pid})", legend_inside=True)
        _unify_ylim([ax])

    def _draw_plot4(ax, pd, model_label, pid):
        m  = pd.get("mated_scores",   np.array([]))
        nm = pd.get("nonmated_scores", np.array([]))
        _draw_dist(ax, m,  COL_CTRL_ADV, "Mated ctrl↔adv",     bins=BINS, fill=False, show_n=False)
        _draw_dist(ax, nm, COL_CTRL_ADV, "Non-mated ctrl↔adv", bins=BINS, linestyle=":", fill=False, show_n=False)
        combo_scores = pd.get("pat_combo_scores", {})
        combos_pat   = pd.get("combos_pat", [])
        for c_idx, (pA, pB) in enumerate(combos_pat):
            col_pA, _ = COMBO_COLORS[c_idx % len(COMBO_COLORS)]
            m_sc, nm_sc = combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax, m_sc,  col_pA, f"Mated P{pA}↔P{pB}",     bins=30, fill=False, show_n=False)
            _draw_dist(ax, nm_sc, col_pA, f"Non-mated P{pA}↔P{pB}", bins=30, linestyle=":", fill=False, show_n=False)
        _style_ax(ax, f"{model_label}\nCombined (Pat {pid})", legend_inside=True)
        _unify_ylim([ax])

    PLOT_DEFS = [
        ("plot1_mated_vs_nonmated", _draw_plot1, "Mated vs Non-Mated"),
        ("plot2_pose_breakdown",    _draw_plot2, "Pose Breakdown"),
        ("plot3_cross_pattern",     _draw_plot3, "Cross-Pattern"),
        ("plot4_combined",          _draw_plot4, "Combined"),
    ]

    entries = overview_list[:MAX_PANELS]   # at most 6 panels

    for pid in sorted(all_pattern_ids):
        for plot_key, draw_fn, plot_title in PLOT_DEFS:
            out_name = f"overview_{plot_key}_pat{pid}_clean.png"
            out_path = out_root / out_name
            print(f"  Per-pattern overview [{plot_title}, Pat {pid}] …", end=" ")

            fig, axes = plt.subplots(
                NROWS, NCOLS,
                figsize=(6.5 * NCOLS, 4.5 * NROWS),
                squeeze=False,
            )
            fig.suptitle(
                f"All Models — {plot_title} | Pattern {pid}",
                fontsize=14, fontweight="bold", y=1.01,
            )

            for panel_idx, md in enumerate(entries):
                row = panel_idx // NCOLS
                col = panel_idx % NCOLS
                ax  = axes[row][col]
                ppd = md.get("per_pattern_data", {}).get(pid)

                # Short model label (last component after '/' or full name)
                raw_label = md["model"]
                short_label = raw_label.split("/")[-1]

                if ppd is None:
                    ax.text(0.5, 0.5, f"{short_label}\n(no data for Pat {pid})",
                            ha="center", va="center", transform=ax.transAxes, fontsize=9)
                    ax.set_visible(True)
                    continue

                # call the right draw helper; plot3/plot4 need pid too
                if plot_key in ("plot3_cross_pattern", "plot4_combined"):
                    draw_fn(ax, ppd, short_label, pid)
                else:
                    draw_fn(ax, ppd, short_label)

                # Highlight the all-models panel with a distinct border
                if md["model"] == ALL_MODELS_LABEL:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#e67e22")
                        spine.set_linewidth(2.0)

            # Hide any unused panels
            for panel_idx in range(len(entries), NROWS * NCOLS):
                axes[panel_idx // NCOLS][panel_idx % NCOLS].set_visible(False)

            fig.tight_layout()
            _save(fig, out_path)
            print("done")


# ─────────────────────────────────────────────────────────────────────────────
# Combined overview figure (all models × plots 1–4)
# ─────────────────────────────────────────────────────────────────────────────

def generate_combined_overview(
    model_data_list: list[dict],
    out_root: Path,
) -> None:
    """
    One big figure: rows = models, columns = Plot1 | Plot2 | Plot3_combo_0 |
    Plot3_combo_1 | … | Plot4.  Each Plot3 combo gets its own column so the
    layout matches the individual per-model Plot 3 subplots exactly.
    Saved to out_root/all_models_overview.png.
    """
    if not model_data_list:
        print("  No model data — skipping combined overview.")
        return

    # Determine combo list from first model that has combos
    combos: list[tuple[int, int]] = []
    for md in model_data_list:
        if md["combos"]:
            combos = md["combos"]
            break

    n_models  = len(model_data_list)
    n_combos  = len(combos)
    # Layout: Plot1 | Plot2 | combo_0 … combo_N | Plot4
    NCOLS     = 2 + n_combos + 1

    fig, axes = plt.subplots(
        n_models, NCOLS,
        figsize=(6.5 * NCOLS, 4.5 * n_models),
        squeeze=False,
    )
    fig.suptitle("All Models — Score Distribution Overview", fontsize=14,
                 fontweight="bold", y=1.01)

    COMBO_COLORS = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
        "#01665e", "#8c510a",
    ]
    NM_COLOR    = "#555555"
    COL_FRONTAL = "#1f77b4"
    COL_PROFILE = "#d62728"

    for row_idx, md in enumerate(model_data_list):
        model                = md["model"]
        mated_scores         = md["mated_scores"]
        nonmated_scores      = md["nonmated_scores"]
        m_front, nm_front    = md["adv_frontal_scores"]
        m_profil, nm_profil  = md["adv_profile_scores"]
        pattern_combo_scores = md["pattern_combo_scores"]

        col_idx = 0

        # ── Plot 1 ────────────────────────────────────────────────────────────
        ax = axes[row_idx][col_idx]; col_idx += 1
        _draw_dist(ax, mated_scores,    PALETTE["mated"],     "Mated")
        _draw_dist(ax, nonmated_scores, PALETTE["non_mated"], "Non-mated", linestyle=":")
        _style_ax(ax, f"{model}\nMated vs Non-Mated")
        _draw_threshold_lines(ax, mated_scores, nonmated_scores)
        _unify_ylim([ax])

        # ── Plot 2 ────────────────────────────────────────────────────────────
        ax = axes[row_idx][col_idx]; col_idx += 1
        _draw_dist(ax, m_front,   COL_FRONTAL, "Mated – Frontal")
        _draw_dist(ax, nm_front,  COL_FRONTAL, "Non-mated – Frontal", linestyle=":")
        _draw_dist(ax, m_profil,  COL_PROFILE, "Mated – Profile")
        _draw_dist(ax, nm_profil, COL_PROFILE, "Non-mated – Profile", linestyle=":")
        _style_ax(ax, f"{model}\nPose Breakdown")
        _unify_ylim([ax])

        # ── Plot 3 — one subplot per combo ────────────────────────────────────
        for c_idx, (pA, pB) in enumerate(combos):
            col_pA = COMBO_COLORS[c_idx % len(COMBO_COLORS)]
            ax = axes[row_idx][col_idx]; col_idx += 1
            m_sc, nm_sc = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax, m_sc,  col_pA,   f"Mated P{pA}\u21d4P{pB}")
            _draw_dist(ax, nm_sc, NM_COLOR, "Non-mated", linestyle="--")
            _style_ax(ax, f"{model}\nPattern {pA} vs {pB}")
            _draw_threshold_lines(ax, m_sc, nm_sc)
            _unify_ylim([ax])

        # ── Plot 4 ────────────────────────────────────────────────────────────
        ax = axes[row_idx][col_idx]; col_idx += 1
        COL_CTRL_ADV = PALETTE["mated"]
        _draw_dist(ax, mated_scores,    COL_CTRL_ADV, "Mated ctrl\u2194adv",     fill=False)
        _draw_dist(ax, nonmated_scores, COL_CTRL_ADV, "Non-mated ctrl\u2194adv", linestyle=":", fill=False)
        for c_idx, (pA, pB) in enumerate(combos):
            col = COMBO_COLORS[c_idx % len(COMBO_COLORS)]
            m_sc, nm_sc = pattern_combo_scores.get((pA, pB), (np.array([]), np.array([])))
            _draw_dist(ax, m_sc,  col, f"Mated P{pA}\u21d4P{pB}",     fill=False)
            _draw_dist(ax, nm_sc, col, f"Non-mated P{pA}\u21d4P{pB}", linestyle=":", fill=False)
        _style_ax(ax, f"{model}\nCombined")
        _unify_ylim([ax])

    fig.tight_layout()
    out_path = out_root / "all_models_overview.png"
    _save(fig, out_path)
    print(f"  Combined overview → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate similarity score distribution plots per model."
    )
    parser.add_argument("--db",      default="db.sqlite",
                        help="Path to the SQLite database (default: db.sqlite)")
    parser.add_argument("--out", default="data_processing/results",
                        help="Root output directory (default: data_processing/results)")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED,
                        help="Random seed for non-mated sampling")
    args = parser.parse_args()

    db_path  = args.db
    out_root = Path(args.out)
    rng      = random.Random(args.seed)

    if not Path(db_path).exists():
        sys.exit(f"ERROR: database not found at '{db_path}'")

    con     = _connect(db_path)
    models  = get_model_names(con)

    if not models:
        sys.exit("ERROR: no attack records found in the database.")

    print(f"Found {len(models)} model(s): {', '.join(models)}")
    print(f"Output root: {out_root.resolve()}")
    print(f"KDE smoothing: {'enabled (scipy)' if HAS_KDE else 'disabled (install scipy to enable)'}")

    model_data_list = []
    for model in models:
        data = generate_plots_for_model(model, con, out_root, rng)
        if data is not None:
            model_data_list.append(data)

    # ── All-models aggregated analysis ────────────────────────────────────────
    print(f"\nGenerating all-models aggregated plots …")
    all_models_data = generate_plots_for_all_models(models, con, out_root, rng)

    con.close()

    print("\nGenerating biometric summary figure …")
    plot_biometric_summary(models, out_root)

    print("\nGenerating combined overview figure …")
    # Append all-models entry at the bottom of the overview grid
    overview_list = model_data_list[:]
    if all_models_data is not None:
        overview_list.append(all_models_data)
    generate_combined_overview(overview_list, out_root)

    print("\nGenerating per-pattern cross-model overview figures …")
    generate_per_pattern_overview_figures(overview_list, out_root)

    print("\nGenerating optimal-accuracy threshold tables …")
    generate_accuracy_threshold_tables(overview_list, out_root)

    print("\nAll done.")


if __name__ == "__main__":
    main()
