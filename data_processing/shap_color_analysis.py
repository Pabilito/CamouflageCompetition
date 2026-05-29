# -*- coding: utf-8 -*-
"""
shap_color_analysis.py
------------------------------------------------------------------------------
SHAP-based feature importance analysis on color_vs_rank.csv.

Trains a GradientBoosting regressor to predict the similarity *score*
(or optionally *rank*) from the colour metrics, then uses SHAP to explain
which Lab combinations drive the prediction.

Outputs (written next to the CSV, or --out-dir):
  shap_summary_beeswarm.png   -- beeswarm: direction + magnitude per sample
  shap_bar.png                -- mean |SHAP|: global importance ranking
  shap_dependence_top3.png    -- dependence plots for the 3 most important feats
  shap_heatmap.png            -- sample-level heatmap (sorted by score)
  shap_interaction_Lab.png    -- L* x a* x b* interaction scatter (if all top)
  shap_report.txt             -- text summary: top features + key findings

Usage
-----
  python shap_color_analysis.py --csv color_vs_rank.csv
  python shap_color_analysis.py --csv color_vs_rank.csv --model AVG --target rank
  python shap_color_analysis.py --csv color_vs_rank.csv --out-dir results/shap/
"""

from __future__ import annotations

import argparse
import textwrap
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# -- Feature sets -------------------------------------------------------------

# Core Lab + derived colour metrics (always present in color_vs_rank.csv)
LAB_FEATURES = ["L", "a", "b"]
DELTA_FEATURES = ["dL", "da", "db", "dE", "dC"]
OTHER_FEATURES = ["C"]
ALL_FEATURES = LAB_FEATURES + OTHER_FEATURES + DELTA_FEATURES

FEATURE_LABELS = {
    "L":  "L*  (lightness)",
    "a":  "a*  (green/red tint)",
    "b":  "b*  (blue/yellow tint)",
    "C":  "C*  (chroma)",
    "dL": "DL* vs control",
    "da": "Da* vs control",
    "db": "Db* vs control",
    "dE": "DE* (L2 norm shift)",
    "dC": "DC* vs control",
}


# -- Helpers -------------------------------------------------------------------

def load_and_filter(csv_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"model", "score", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    models_present = df["model"].unique().tolist()
    if model_name not in models_present:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {models_present}"
        )

    df = df[df["model"] == model_name].copy()
    print(f"  Rows after filtering model='{model_name}': {len(df)}")
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: features not in CSV, skipped: {missing}")
    return available


def build_Xy(df: pd.DataFrame, features: list[str], target: str
             ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sub = df[features + [target]].dropna()
    X = sub[features].values.astype(np.float64)
    y = sub[target].values.astype(np.float64)
    return X, y, features


# -- Plotting ------------------------------------------------------------------

PALETTE = "white"   # light bg
ACCENT  = "#e94560"


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_beeswarm(shap_values, X_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.sca(ax)
    shap.plots.beeswarm(shap_values, max_display=12, show=False,
                        color_bar=True)
    ax.set_title("SHAP Beeswarm -- feature impact on score",
                 color="black", fontsize=13, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    _save(fig, out)


def plot_bar(shap_values, out: Path) -> None:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feat_names = shap_values.feature_names
    order = np.argsort(mean_abs)[::-1][:12]
    vals  = mean_abs[order]
    names = [feat_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    ax.barh(range(len(names)), vals[::-1], color="#e0245e", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=11, color="black")
    ax.set_xlabel("Mean |SHAP value|", color="black", fontsize=11)
    ax.set_title("Mean |SHAP| -- global feature importance",
                 color="black", fontsize=13, pad=10)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    fig.tight_layout()
    _save(fig, out)


def plot_dependence_top3(shap_values, X_df: pd.DataFrame,
                         feature_names: list[str], out: Path) -> None:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top3 = np.argsort(mean_abs)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("SHAP Dependence -- top 3 features", color="black",
                 fontsize=13, y=1.01)

    for ax, idx in zip(axes, top3):
        feat = feature_names[idx]
        label = FEATURE_LABELS.get(feat, feat)
        ax.set_facecolor("white")
        plt.sca(ax)
        shap.plots.scatter(shap_values[:, feat], ax=ax, show=False,
                           color=shap_values)
        ax.set_title(label, color="black", fontsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

    _save(fig, out)


def plot_heatmap(shap_values, out: Path) -> None:
    sv   = shap_values.values
    feat = shap_values.feature_names
    n_feat = min(12, len(feat))

    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_feat]
    sv_top   = sv[:, top_idx]
    names    = [feat[i] for i in top_idx]

    order     = np.argsort(sv_top.sum(axis=1))
    sv_sorted = sv_top[order].T

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1, 4],
                           width_ratios=[20, 1], hspace=0.05, wspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    fx = sv_top[order].sum(axis=1)
    ax_top.plot(fx, color="black", linewidth=0.8)
    ax_top.axhline(np.median(fx), color="#aaa", linewidth=0.7, linestyle="--")
    ax_top.set_xlim(0, len(fx) - 1)
    ax_top.set_ylabel("f(x)", fontsize=9, color="black")
    ax_top.tick_params(labelbottom=False, colors="black", labelsize=8)
    ax_top.set_facecolor("white")
    for spine in ax_top.spines.values():
        spine.set_edgecolor("#cccccc")

    ax_hm = fig.add_subplot(gs[1, 0])
    vmax  = np.abs(sv_sorted).max()
    im    = ax_hm.imshow(sv_sorted, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_hm.set_yticks(range(n_feat))
    ax_hm.set_yticklabels(names, fontsize=10, color="black")
    ax_hm.set_xlabel("Instances (sorted by f(x))", fontsize=10, color="black")
    ax_hm.tick_params(colors="black")
    ax_hm.set_facecolor("white")
    for spine in ax_hm.spines.values():
        spine.set_edgecolor("#cccccc")

    ax_cb = fig.add_subplot(gs[1, 1])
    cb    = fig.colorbar(im, cax=ax_cb)
    cb.set_label("SHAP value", fontsize=9, color="black")
    cb.ax.tick_params(colors="black", labelsize=8)

    fig.suptitle("SHAP Heatmap -- per-sample contributions",
                 color="black", fontsize=13)
    _save(fig, out)


def plot_lab_scatter(shap_values, X_df: pd.DataFrame,
                     feature_names: list[str], out: Path) -> None:
    """3-panel scatter: raw Lab value vs SHAP value, coloured by score."""
    lab = [f for f in ["L", "a", "b"] if f in feature_names]
    if not lab:
        print("  (skipping Lab scatter -- L/a/b not all available)")
        return

    fig, axes = plt.subplots(1, len(lab), figsize=(5 * len(lab), 5))
    if len(lab) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle("Lab raw value vs SHAP contribution", color="black",
                 fontsize=13, y=1.01)

    sv_arr = shap_values.values          # (n, nfeat)
    base_y = shap_values.base_values

    for ax, feat in zip(axes, lab):
        ax.set_facecolor("white")
        fi = feature_names.index(feat)
        x = X_df[feat].values
        y = sv_arr[:, fi]
        sc = ax.scatter(x, y, c=y, cmap="RdYlGn", s=18, alpha=0.7,
                        edgecolors="none")
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), color="black")
        ax.set_ylabel("SHAP value", color="black")
        ax.set_title(f"{feat} impact", color="black", fontsize=11)
        plt.colorbar(sc, ax=ax, label="SHAP ->score")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

    fig.tight_layout()
    _save(fig, out)


# -- Text report ---------------------------------------------------------------

def write_report(shap_values, feature_names: list[str],
                 cv_r2: float, target: str, model_name: str,
                 out: Path) -> None:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    sv = shap_values.values
    order = np.argsort(mean_abs)[::-1]

    lines = [
        "SHAP Colour Analysis -- Summary Report",
        "=" * 60,
        f"  Model filter : {model_name}",
        f"  Target       : {target}",
        f"  Samples      : {sv.shape[0]}",
        f"  CV R2        : {cv_r2:.3f}  "
        f"({'good fit' if cv_r2 > 0.3 else 'weak fit -- interpret with care'})",
        "",
        "Feature Importance (mean |SHAP|, descending)",
        "-" * 45,
    ]
    for rank_i, fi in enumerate(order, 1):
        feat  = feature_names[fi]
        label = FEATURE_LABELS.get(feat, feat)
        imp   = mean_abs[fi]
        # direction: positive SHAP -> higher score
        pos_frac = (sv[:, fi] > 0).mean()
        direction = "^ score" if pos_frac > 0.55 else (
                    "v score" if pos_frac < 0.45 else "mixed")
        lines.append(f"  {rank_i:2d}. {label:<28}  |SHAP|={imp:.4f}  {direction}")

    lines += [
        "",
        "Key findings",
        "-" * 45,
    ]

    top_feat = feature_names[order[0]]
    top_label = FEATURE_LABELS.get(top_feat, top_feat)
    lines.append(f"  * Most influential: {top_label}")

    lab_in_top3 = [feature_names[fi] for fi in order[:3] if feature_names[fi] in LAB_FEATURES]
    if lab_in_top3:
        lines.append(f"  * Raw Lab channels in top 3: {', '.join(lab_in_top3)}")

    delta_in_top3 = [feature_names[fi] for fi in order[:3] if feature_names[fi] in DELTA_FEATURES]
    if delta_in_top3:
        lines.append(f"  * Delta metrics in top 3: {', '.join(delta_in_top3)} "
                     "(deviation from control matters)")

    if cv_r2 < 0.1:
        lines.append("  * Low R2 suggests colour alone explains little of score variance.")
    elif cv_r2 > 0.5:
        lines.append("  * High R2: colour metrics are strong predictors of attack success.")

    lines += ["", "Generated by shap_color_analysis.py"]

    report = "\n".join(lines)
    out.write_text(report, encoding="utf-8")
    print(f"  [saved] {out}")
    print()
    print(report)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", default="data_processing/results/color_vs_rank.csv",
                    help="Path to color_vs_rank.csv (default: data_processing/results/color_vs_rank.csv)")
    ap.add_argument("--model", default="AVG",
                    help="Model name to filter on (default: AVG)")
    ap.add_argument("--target", default="score", choices=["score", "rank"],
                    help="Prediction target (default: score)")
    ap.add_argument("--out-dir", default="data_processing/results",
                    help="Output directory (default: data_processing/results)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SHAP Colour Analysis")
    print(f"  CSV    : {csv_path}")
    print(f"  Model  : {args.model}")
    print(f"  Target : {args.target}")
    print(f"  Out    : {out_dir}")
    print(f"{'='*60}\n")

    # -- Load data --------------------------------------------------------
    df = load_and_filter(csv_path, args.model)
    features = select_features(df)
    print(f"  Features selected: {features}\n")

    X, y, feat_names = build_Xy(df, features, args.target)
    X_df = pd.DataFrame(X, columns=feat_names)

    # Friendly labels for SHAP plots
    feature_labels = [FEATURE_LABELS.get(f, f) for f in feat_names]
    X_labeled = X_df.rename(columns={f: FEATURE_LABELS.get(f, f)
                                      for f in feat_names})

    # -- Fit model --------------------------------------------------------
    print("  Fitting GradientBoostingRegressor ...")
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        random_state=42,
    )
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_r2 = float(cv_scores.mean())
    print(f"  5-fold CV R2 = {cv_r2:.3f}  (std={cv_scores.std():.3f})")

    model.fit(X, y)

    # -- SHAP -------------------------------------------------------------
    print("  Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_labeled)   # Explanation object with feature labels

    # -- Plots -------------------------------------------------------------
    print("\n  Generating plots ...")
    plot_beeswarm(sv, X_labeled, out_dir / "shap_summary_beeswarm.png")
    plot_bar(sv, out_dir / "shap_bar.png")
    plot_dependence_top3(sv, X_labeled, feature_labels, out_dir / "shap_dependence_top3.png")
    plot_heatmap(sv, out_dir / "shap_heatmap.png")
    plot_lab_scatter(sv, X_df, feat_names, out_dir / "shap_lab_scatter.png")

    # -- Report ------------------------------------------------------------
    write_report(sv, feature_labels, cv_r2, args.target, args.model,
                 out_dir / "shap_report.txt")


if __name__ == "__main__":
    main()
