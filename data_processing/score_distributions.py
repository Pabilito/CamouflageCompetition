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
) -> None:
    """Draw a KDE line, optionally with a light fill underneath."""
    if len(scores) == 0:
        return
    xs = np.linspace(X_MIN, X_MAX, 500)
    if HAS_KDE and len(scores) > 5:
        try:
            kde = gaussian_kde(scores, bw_method="scott")
            ys  = kde(xs)
            ax.plot(xs, ys, color=color, lw=LINE_WIDTH, alpha=ALPHA_LINE,
                    linestyle=linestyle, label=f"{label} (n={len(scores):,})")
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
        label=f"{label} (n={len(scores):,})",
    )


def _style_ax(ax: plt.Axes, title: str, xlabel: str = "Cosine Similarity") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
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
    print("done")

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
    }



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
    print("done")

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
    }


# ─────────────────────────────────────────────────────────────────────────────
# Biometric statistics (pyeer)
# ─────────────────────────────────────────────────────────────────────────────

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

    print("\nAll done.")


if __name__ == "__main__":
    main()
