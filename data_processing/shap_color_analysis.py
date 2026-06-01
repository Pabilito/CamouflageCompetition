# -*- coding: utf-8 -*-
"""
shap_color_analysis.py
------------------------------------------------------------------------------
SHAP-based feature importance for colour → similarity-score on color_vs_rank.csv.

Key design choices (learned from data diagnostics):
  1. GroupKFold(groups=identity) — standard KFold gives R² ≈ -13 because
     identities have different baseline scores.
  2. Within-identity centering — subtracts each identity's mean from both
     features and target so the model learns "within this person, how does
     colour shift affect score?" rather than confounding with identity.
  3. Ridge regression — outperforms GBR/RF/SVR on this small (n=365) dataset
     with weak signal; trees overfit, Ridge generalises.
  4. LinearExplainer — exact SHAP for linear models; no approximation.

Expected CV R² ≈ 0.08–0.09 (colour is a real but secondary predictor).

Usage
-----
  python shap_color_analysis.py --csv color_vs_rank.csv
  python shap_color_analysis.py --csv color_vs_rank.csv --model AVG --target rank
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "axes.titlecolor":   "black",
    "figure.edgecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -- Feature configuration ----------------------------------------------------

FEATURES = ["L", "a", "b", "C", "dE", "L_std", "dh", "H"]

FEATURE_LABELS = {
    "L":     "L*  (lightness)",
    "a":     "a*  (green→red)",
    "b":     "b*  (blue→yellow)",
    "C":     "C*  (chroma)",
    "dE":    "ΔE  (total colour dist)",
    "L_std": "L* std  (patch variance)",
    "dh":    "Δh  (hue shift)",
    "H":     "H  (HSV hue)",
}

RIDGE_ALPHA = 50

# Concept groups for STABLE attribution. Individual columns are collinear
# (empirically: L–dE r=-0.64, b–H r=-0.82, b–C r=+0.69), so per-column SHAP
# rankings are noisy — Ridge hands shared credit to whichever column wins by a
# hair. Summing |SHAP| within a concept group is reproducible across resamples.
# Edit these clusters freely; any feature not listed falls into its own group.
FEATURE_GROUPS = {
    "lightness / distance": ["L", "dE", "L_std"],
    "hue / chroma":         ["b", "C", "H", "dh"],
    "redness (a*)":         ["a"],
}

N_BOOTSTRAP = 1000   # resamples over identities for CI + importance stability

# -- Data helpers --------------------------------------------------------------

def load_and_filter(csv_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"model", "score", "rank", "identity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    if model_name not in df["model"].unique().tolist():
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available: {df['model'].unique().tolist()}"
        )

    df = df[df["model"] == model_name].copy()
    n_ids = df["identity"].nunique()
    print(f"  Rows: {len(df)}  |  Identities: {n_ids}  "
          f"(~{len(df)/n_ids:.0f} images each)")
    return df


def center_within_identity(df: pd.DataFrame, features: list[str],
                            target: str) -> pd.DataFrame:
    """Subtract per-identity mean from features and target."""
    df = df.copy()
    for f in features:
        df[f] = df.groupby("identity")[f].transform(lambda x: x - x.mean())
    df[f"{target}_centered"] = df.groupby("identity")[target].transform(
        lambda x: x - x.mean()
    )
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    available = [f for f in FEATURES if f in df.columns]
    skipped = [f for f in FEATURES if f not in df.columns]
    if skipped:
        print(f"  Warning: skipped (not in CSV): {skipped}")
    return available


def build_Xy(df, features, target):
    tgt_col = f"{target}_centered"
    sub = df[features + [tgt_col, "identity"]].dropna()
    X = sub[features].values.astype(np.float64)
    y = sub[tgt_col].values.astype(np.float64)
    groups = sub["identity"].values
    return X, y, groups, features


def print_diagnostics(X, y, groups, feature_names, df, target):
    raw_y = df[target].values
    print(f"\n  Raw target: mean={raw_y.mean():.4f}  std={raw_y.std():.4f}  "
          f"range=[{raw_y.min():.4f}, {raw_y.max():.4f}]")

    df_tmp = pd.DataFrame({"y": raw_y, "g": df['identity'].values})
    within_std = df_tmp.groupby("g")["y"].std().mean()
    print(f"  Within-identity σ = {within_std:.4f}  "
          f"(overall σ = {raw_y.std():.4f}, ratio = {within_std/raw_y.std():.2f})")

    print(f"  Centered target: mean={y.mean():.6f}  std={y.std():.4f}")

    print("\n  Feature → centered-target Spearman correlations:")
    for i, f in enumerate(feature_names):
        r, p = stats.spearmanr(X[:, i], y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        label = FEATURE_LABELS.get(f, f)
        print(f"    {label:<28}  ρ = {r:+.3f}  p = {p:.4f} {sig}")
    print()


# -- Reliability / robustness analysis ----------------------------------------

def pooled_oof_r2_with_ci(y, y_pred_cv, groups, n_boot=N_BOOTSTRAP, seed=0):
    """Pooled out-of-fold R² + identity-level bootstrap 95% CI.

    Far more stable than averaging per-fold R² over tiny held-out folds
    (that average's std is mostly an artifact of the split count). The CI
    comes from resampling whole identities, respecting the group structure.
    """
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    oof_r2 = float(r2_score(y, y_pred_cv))

    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    boot = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in chosen])
        if np.std(y[rows]) == 0:
            continue
        boot.append(r2_score(y[rows], y_pred_cv[rows]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return oof_r2, float(lo), float(hi)


def bootstrap_importance_stability(X, y, groups, feature_names,
                                   alpha=RIDGE_ALPHA, n_boot=N_BOOTSTRAP, seed=0):
    """How reliable is each feature's importance rank?

    Resamples identities, refits the standardized Ridge, and records each
    feature's |coef| rank (identical to the mean|SHAP| rank for a linear model
    on standardized inputs). Returns median rank and P(feature in top 3).
    A feature that is only 'important' in one fit drops out here.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    n_feat = len(feature_names)
    ranks = {f: [] for f in feature_names}

    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in chosen])
        yb = y[rows]
        if np.std(yb) == 0:
            continue
        pipe = Pipeline([("s", StandardScaler()),
                         ("r", Ridge(alpha=alpha))]).fit(X[rows], yb)
        coef = np.abs(pipe.named_steps["r"].coef_)
        order = np.argsort(coef)[::-1]
        for pos, fi in enumerate(order, 1):
            ranks[feature_names[fi]].append(pos)

    summary = []  # (feature, median_rank, p_top3)
    for f in feature_names:
        r = np.asarray(ranks[f])
        summary.append((f, float(np.median(r)), float((r <= 3).mean())))
    summary.sort(key=lambda t: (t[1], -t[2]))
    return summary


def grouped_shap_importance(shap_values, feature_names):
    """Sum mean|SHAP| within concept groups (collinearity-robust attribution).

    feature_names are the labelled column names; we map labels back to raw
    keys via FEATURE_LABELS to apply FEATURE_GROUPS.
    """
    label_to_key = {v: k for k, v in FEATURE_LABELS.items()}
    mean_abs = np.abs(shap_values.values).mean(axis=0)

    # raw-key -> mean|SHAP|
    key_imp = {}
    for name, imp in zip(feature_names, mean_abs):
        key = label_to_key.get(name, name)
        key_imp[key] = imp

    grouped, claimed = [], set()
    for gname, members in FEATURE_GROUPS.items():
        present = [k for k in members if k in key_imp]
        if not present:
            continue
        claimed.update(present)
        grouped.append((gname, sum(key_imp[k] for k in present), present))
    # leftovers each form their own group
    for k, imp in key_imp.items():
        if k not in claimed:
            grouped.append((FEATURE_LABELS.get(k, k), imp, [k]))
    grouped.sort(key=lambda t: t[1], reverse=True)
    return grouped


def cross_model_consistency(csv_path, features, target, alpha=RIDGE_ALPHA):
    """Rank features independently within each recognition model.

    Agreement across models is a robustness result that no single-model SHAP
    run can provide: a feature ranked #1 in all models is trustworthy even when
    R² is low. Returns {model: (oof_r2, ranked_feature_keys)}.
    """
    from sklearn.metrics import r2_score
    df_all = pd.read_csv(csv_path)
    out = {}
    for m in sorted(df_all["model"].unique()):
        sub = df_all[df_all["model"] == m].copy()
        feats = [f for f in features if f in sub.columns]
        for f in feats:
            sub[f] = sub.groupby("identity")[f].transform(lambda x: x - x.mean())
        tgt = sub.groupby("identity")[target].transform(lambda x: x - x.mean())
        keep = sub[feats + ["identity"]].assign(_y=tgt).dropna()
        X = keep[feats].values
        y = keep["_y"].values
        g = keep["identity"].values
        pipe = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=alpha))])
        y_oof = cross_val_predict(pipe, X, y, cv=GroupKFold(5), groups=g)
        r2 = float(r2_score(y, y_oof))
        coef = np.abs(pipe.fit(X, y).named_steps["r"].coef_)
        ranked = [feats[i] for i in np.argsort(coef)[::-1]]
        out[m] = (r2, ranked)
    return out


# -- Plotting ------------------------------------------------------------------

def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_beeswarm(shap_values, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    plt.sca(ax)
    shap.plots.beeswarm(shap_values, max_display=12, show=False, color_bar=True)
    ax.set_title("SHAP Beeswarm — feature impact on score (within-identity)",
                 color="black", fontsize=13, pad=10)
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
    _save(fig, out)


def plot_bar(shap_values, out):
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feat_names = list(shap_values.feature_names)
    order = np.argsort(mean_abs)[::-1]
    vals = mean_abs[order]
    names = [feat_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    ax.barh(range(len(names)), vals[::-1], color="#e0245e", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=11, color="black")
    ax.set_xlabel("Mean |SHAP value|", color="black", fontsize=11)
    ax.set_title("Mean |SHAP| — global feature importance (within-identity)",
                 color="black", fontsize=13, pad=10)
    ax.tick_params(colors="black")
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")
    fig.tight_layout()
    _save(fig, out)


def plot_dependence_top3(shap_values, feature_names, out):
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    n = min(3, len(feature_names))
    top = np.argsort(mean_abs)[::-1][:n]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1: axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle("SHAP Dependence — top features", color="black", fontsize=13)

    for ax, idx in zip(axes, top):
        feat = feature_names[idx]
        ax.set_facecolor("white"); plt.sca(ax)
        shap.plots.scatter(shap_values[:, feat], ax=ax, show=False,
                           color=shap_values)
        ax.set_title(feat, color="black", fontsize=11, pad=10)
        for s in ax.spines.values(): s.set_edgecolor("#cccccc")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black"); ax.yaxis.label.set_color("black")
        for child in fig.get_axes():
            if child is not ax and hasattr(child, "yaxis"):
                child.yaxis.label.set_color("black")
                child.tick_params(colors="black")

    fig.tight_layout(pad=2.0, w_pad=4.0)
    _save(fig, out)


def plot_heatmap(shap_values, out):
    sv = shap_values.values
    feat = list(shap_values.feature_names)
    n_feat = min(12, len(feat))

    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n_feat]
    sv_top = sv[:, top_idx]
    names = [feat[i] for i in top_idx]

    order = np.argsort(sv_top.sum(axis=1))
    sv_sorted = sv_top[order].T

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4],
                          width_ratios=[20, 1], hspace=0.05, wspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    fx = sv_top[order].sum(axis=1)
    ax_top.plot(fx, color="black", linewidth=0.8)
    ax_top.axhline(np.median(fx), color="#aaa", linewidth=0.7, linestyle="--")
    ax_top.set_xlim(0, len(fx) - 1)
    ax_top.set_ylabel("f(x)", fontsize=9, color="black")
    ax_top.tick_params(labelbottom=False, colors="black", labelsize=8)
    ax_top.set_facecolor("white")
    for s in ax_top.spines.values(): s.set_edgecolor("#cccccc")

    ax_hm = fig.add_subplot(gs[1, 0])
    vmax = np.abs(sv_sorted).max()
    im = ax_hm.imshow(sv_sorted, aspect="auto", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_hm.set_yticks(range(n_feat))
    ax_hm.set_yticklabels(names, fontsize=10, color="black")
    ax_hm.set_xlabel("Instances (sorted by f(x))", fontsize=10, color="black")
    ax_hm.tick_params(colors="black"); ax_hm.set_facecolor("white")
    for s in ax_hm.spines.values(): s.set_edgecolor("#cccccc")

    ax_cb = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("SHAP value", fontsize=9, color="black")
    cb.ax.tick_params(colors="black", labelsize=8)

    fig.suptitle("SHAP Heatmap — per-sample contributions (within-identity)",
                 color="black", fontsize=13)
    _save(fig, out)


def plot_lab_scatter(shap_values, X_df, feature_names, out):
    feats = [f for f in ["L", "a", "b", "C"] if f in feature_names]
    if not feats: return

    sv_arr = shap_values.values
    labeled_names = list(shap_values.feature_names)

    all_shap = np.concatenate([
        sv_arr[:, labeled_names.index(FEATURE_LABELS.get(f, f))] for f in feats
    ])
    ymax = np.abs(all_shap).max() * 1.05
    vmax = ymax

    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 5),
                             sharey=True)
    if len(feats) == 1: axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle("Centred Lab value vs SHAP contribution", color="black",
                 fontsize=13)

    for i, (ax, feat) in enumerate(zip(axes, feats)):
        ax.set_facecolor("white")
        label = FEATURE_LABELS.get(feat, feat)
        fi = labeled_names.index(label)
        x = X_df[feat].values
        y_sv = sv_arr[:, fi]
        sc = ax.scatter(x, y_sv, c=y_sv, cmap="RdYlGn", s=18, alpha=0.7,
                        edgecolors="none", vmin=-vmax, vmax=vmax)
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(f"{label}  (centred)", color="black")
        if i == 0: ax.set_ylabel("SHAP value", color="black")
        ax.set_title(f"{label.split('(')[0].strip()} impact",
                     color="black", fontsize=11)
        for s in ax.spines.values(): s.set_edgecolor("#cccccc")
        ax.tick_params(colors="black")

    fig.subplots_adjust(right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Score impact")
    cbar_ax.tick_params(colors="black"); cbar_ax.yaxis.label.set_color("black")
    _save(fig, out)


def plot_actual_vs_predicted(y_true, y_pred, oof_r2, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.scatter(y_true, y_pred, s=15, alpha=0.5, color="#e0245e",
               edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    m = (hi - lo) * 0.05
    ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo - m, hi + m); ax.set_ylim(lo - m, hi + m)
    r, _ = stats.spearmanr(y_true, y_pred)
    ax.set_xlabel("Actual (centred score)", fontsize=12)
    ax.set_ylabel("Predicted (GroupKFold CV)", fontsize=12)
    ax.set_title(f"Actual vs Predicted  (pooled OOF R²={oof_r2:.3f}, ρ={r:.3f})",
                 fontsize=13, pad=10)
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")
    ax.tick_params(colors="black"); ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, out)


# -- Report --------------------------------------------------------------------

def write_report(shap_values, feature_names, cv_r2, cv_std,
                 oof_r2, oof_ci, stability, grouped, consistency,
                 target, model_name, n_ids, n_samples, out):
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    sv = shap_values.values
    order = np.argsort(mean_abs)[::-1]
    stab = {name: (med, ptop3) for name, med, ptop3 in stability}

    lines = [
        "SHAP Colour Analysis — Summary Report",
        "=" * 60,
        f"  Model filter : {model_name}",
        f"  Regressor    : Ridge(alpha={RIDGE_ALPHA}) + StandardScaler",
        f"  CV strategy  : GroupKFold(n_splits=5, groups=identity)",
        f"  Centering    : within-identity (features & target)",
        f"  Target       : {target} (centred)",
        f"  Samples      : {n_samples}  ({n_ids} identities)",
        f"  Features     : {sv.shape[1]}",
        f"  Pooled OOF R² : {oof_r2:.3f}   95% CI [{oof_ci[0]:+.3f}, {oof_ci[1]:+.3f}]"
        f"   (bootstrap over identities, n={N_BOOTSTRAP})",
        f"  Per-fold R²   : {cv_r2:.3f} ± {cv_std:.3f}   (reference only — the ±"
        f" is largely a split-count artifact)",
        "",
    ]

    headline = oof_r2  # judge signal on the stable pooled number, not per-fold
    ci_excludes_zero = oof_ci[0] > 0
    if headline < 0.05:
        lines.append("  ⚠  Very weak signal: colour explains < 5% of"
                     " within-identity variance.")
    elif headline < 0.15:
        lines.append("  ℹ  Modest R²: colour has a weak but real effect on"
                     " score within each identity.")
        lines.append("     SHAP directions are meaningful; magnitudes are"
                     " small.")
    else:
        lines.append("  ✓  Reasonable R²: colour meaningfully predicts score"
                     " within identities.")
    lines.append("  " + ("✓  CI excludes 0 → effect is statistically credible."
                         if ci_excludes_zero else
                         "⚠  CI includes 0 → effect not distinguishable from noise."))

    lines += [
        "",
        "Feature Importance (mean |SHAP|) + bootstrap stability",
        "-" * 58,
        f"  {'feature':<28} {'|SHAP|':>8}  {'dir':>5}  {'medRank':>7}  {'P(top3)':>7}",
    ]
    for rank_i, fi in enumerate(order, 1):
        feat = feature_names[fi]
        imp = mean_abs[fi]
        pos_frac = (sv[:, fi] > 0).mean()
        direction = ("↑" if pos_frac > 0.55
                     else "↓" if pos_frac < 0.45 else "~")
        med, ptop3 = stab.get(feat, (float("nan"), float("nan")))
        flag = "" if ptop3 >= 0.75 else "  (unstable)" if ptop3 < 0.40 else ""
        lines.append(
            f"  {rank_i:2d}. {feat:<24} {imp:8.4f}  {direction:>5}  "
            f"{med:7.1f}  {ptop3:7.2f}{flag}"
        )
    lines.append("     medRank/P(top3) from refitting on bootstrapped identities;")
    lines.append("     trust only features that stay near the top across resamples.")

    lines += [
        "",
        "Grouped importance (collinearity-robust concept attribution)",
        "-" * 58,
    ]
    for gname, imp, members in grouped:
        lines.append(f"  {gname:<28} Σ|SHAP|={imp:.4f}   [{', '.join(members)}]")

    lines += [
        "",
        "Cross-model consistency (same analysis, each recognition model)",
        "-" * 58,
        f"  {'model':<24} {'OOF R²':>7}   top-3 features",
    ]
    for m, (r2, ranked) in consistency.items():
        marker = "  <- this report" if m == model_name else ""
        lines.append(f"  {m:<24} {r2:+7.3f}   {ranked[:3]}{marker}")
    # which feature is universally top?
    tops = [ranked[0] for _, ranked in consistency.values()]
    if tops and len(set(tops)) == 1:
        lines.append(f"  → '{tops[0]}' ranks #1 in ALL {len(tops)} models"
                     f" — robust headline finding.")

    lines += [
        "",
        "Methodology notes",
        "-" * 58,
        "  • Headline metric is POOLED out-of-fold R² (one R² over all CV",
        "    predictions) with a bootstrap CI over identities. Averaging",
        "    per-fold R² is unstable: its ± swings with the split count",
        "    (3/5/10 folds → ±0.05 / ±0.09 / ±0.18 on this data).",
        "  • Per-column SHAP rank is noisy under collinearity (L–dE, b–H,",
        "    b–C). Read the grouped importances and the bootstrap stability",
        "    columns, not the raw per-column order, for what's real.",
        "  • Within-identity centering removes between-identity confounds",
        "    (some faces are inherently easier to attack).",
        "  • Standard KFold gives R² ≈ -13 (identities have different",
        "    baselines); GroupKFold is mandatory.",
        "  • Cross-model agreement is the strongest robustness evidence",
        "    available at this signal level.",
        "",
        "Generated by shap_color_analysis.py",
    ]

    report = "\n".join(lines)
    out.write_text(report, encoding="utf-8")
    print(f"  [saved] {out}")
    print()
    print(report)


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data_processing/results/color_vs_rank.csv")
    ap.add_argument("--model", default="AVG")
    ap.add_argument("--target", default="score", choices=["score", "rank"])
    ap.add_argument("--out-dir", default="data_processing/results")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SHAP Colour Analysis  (GroupKFold + within-identity centering)")
    print(f"  CSV    : {csv_path}")
    print(f"  Model  : {args.model}")
    print(f"  Target : {args.target}")
    print(f"{'='*60}\n")

    # -- Load & centre --------------------------------------------------------
    df = load_and_filter(csv_path, args.model)
    features = select_features(df)
    print(f"  Features: {features}\n")

    df = center_within_identity(df, features, args.target)
    X, y, groups, feat_names = build_Xy(df, features, args.target)
    n_ids = len(np.unique(groups))
    X_df = pd.DataFrame(X, columns=feat_names)

    print_diagnostics(X, y, groups, feat_names, df, args.target)

    # Labelled columns for plots
    feature_labels = [FEATURE_LABELS.get(f, f) for f in feat_names]
    X_labeled = X_df.rename(
        columns={f: FEATURE_LABELS.get(f, f) for f in feat_names})

    # -- Fit ------------------------------------------------------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    gkf = GroupKFold(n_splits=5)

    print("  GroupKFold CV (Ridge on centred features) ...")
    cv_scores = cross_val_score(pipeline, X, y, cv=gkf, scoring="r2",
                                groups=groups)
    cv_r2 = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"  5-fold GroupKFold R² = {cv_r2:.3f}  (std={cv_std:.3f})")
    print(f"  Per-fold: {['%.3f' % s for s in cv_scores]}")

    y_pred_cv = cross_val_predict(pipeline, X, y, cv=gkf, groups=groups)

    # -- Reliable headline metric: pooled OOF R² + bootstrap CR over identities
    print("\n  Pooled out-of-fold R² + bootstrap CI (over identities) ...")
    oof_r2, oof_lo, oof_hi = pooled_oof_r2_with_ci(y, y_pred_cv, groups)
    print(f"  Pooled OOF R² = {oof_r2:.3f}   95% CI [{oof_lo:+.3f}, {oof_hi:+.3f}]")

    # Final fit on all data for SHAP
    pipeline.fit(X, y)
    model = pipeline.named_steps["ridge"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_names).rename(
        columns={f: FEATURE_LABELS.get(f, f) for f in feat_names})

    # -- SHAP -----------------------------------------------------------------
    print("\n  Computing SHAP (LinearExplainer) ...")
    explainer = shap.LinearExplainer(model, X_scaled_df)
    sv = explainer(X_scaled_df)

    # -- Robustness analyses --------------------------------------------------
    print("\n  Bootstrapping importance stability ...")
    stability = bootstrap_importance_stability(X, y, groups, feature_labels)
    print(f"  Most stable: {stability[0][0]} "
          f"(median rank {stability[0][1]:.1f}, P(top3)={stability[0][2]:.2f})")

    grouped = grouped_shap_importance(sv, feature_labels)

    print("  Cross-model consistency (all recognition models) ...")
    consistency = cross_model_consistency(csv_path, features, args.target)

    # -- Plots ----------------------------------------------------------------
    print("\n  Generating plots ...")
    plot_beeswarm(sv, out_dir / "shap_summary_beeswarm.png")
    plot_bar(sv, out_dir / "shap_bar.png")
    plot_dependence_top3(sv, feature_labels, out_dir / "shap_dependence_top3.png")
    plot_heatmap(sv, out_dir / "shap_heatmap.png")
    plot_lab_scatter(sv, X_df, feat_names, out_dir / "shap_lab_scatter.png")
    plot_actual_vs_predicted(y, y_pred_cv, oof_r2,
                             out_dir / "shap_actual_vs_pred.png")

    # -- Report ---------------------------------------------------------------
    write_report(sv, feature_labels, cv_r2, cv_std,
                 oof_r2, (oof_lo, oof_hi), stability, grouped, consistency,
                 args.target, args.model, n_ids, len(y),
                 out_dir / "shap_report.txt")


if __name__ == "__main__":
    main()
