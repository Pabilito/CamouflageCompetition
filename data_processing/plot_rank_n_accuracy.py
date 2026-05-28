#!/usr/bin/env python3
"""
Plot rank-n (CMC) accuracy and print/save summary retrieval statistics.

Reads a CSV with an `all_positions` column (semicolon-separated 1-based ranks,
one value per query image for that identity). Each value is the position at
which a correct match was retrieved for one query.

Usage:
    python plot_rank_n_accuracy.py [path/to/csv]

Defaults to: data_processing/datagallery_search_with_lfw.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_RANK = 20

DEFAULT_CSV = "data_processing/results/gallery_search_with_lfw.csv"
OUT_DIR = "data_processing"


def parse_positions(series: pd.Series) -> np.ndarray:
    """Flatten the semicolon-separated all_positions column into one array of ints."""
    positions = []
    for cell in series.dropna():
        for tok in str(cell).split(";"):
            tok = tok.strip()
            if tok:
                try:
                    positions.append(int(float(tok)))
                except ValueError:
                    pass
    return np.asarray(positions, dtype=int)


def cmc_curve(positions: np.ndarray, max_rank: int) -> np.ndarray:
    """Cumulative match characteristic: fraction of queries matched at rank <= n."""
    n = len(positions)
    ranks = np.arange(1, max_rank + 1)
    return np.array([(positions <= r).sum() / n for r in ranks])


def compute_stats(positions: np.ndarray) -> dict:
    """Aggregate retrieval statistics across all query samples."""
    n = len(positions)
    rr = 1.0 / positions  # reciprocal rank per query
    return {
        "n_samples": n,
        "MRR": rr.mean(),
        "mean_position": positions.mean(),
        "median_position": float(np.median(positions)),
        "std_position": positions.std(ddof=0),
        "min_position": int(positions.min()),
        "max_position": int(positions.max()),
        "rank1_accuracy": (positions <= 1).mean(),
        "rank5_accuracy": (positions <= 5).mean(),
        "rank10_accuracy": (positions <= 10).mean(),
        "rank20_accuracy": (positions <= 20).mean(),
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        sys.exit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "all_positions" not in df.columns:
        sys.exit("Column 'all_positions' not found in CSV.")

    positions = parse_positions(df["all_positions"])
    if positions.size == 0:
        sys.exit("No valid positions parsed from 'all_positions'.")

    out_dir = os.path.dirname(os.path.abspath(csv_path)) or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ---- Statistics table ----
    stats = compute_stats(positions)
    stats_df = pd.DataFrame(
        {"metric": list(stats.keys()), "value": list(stats.values())}
    )
    stats_df["value"] = stats_df["value"].apply(
        lambda v: f"{v:.4f}" if isinstance(v, float) else v
    )
    print("\n=== Retrieval statistics (across all query samples) ===")
    print(stats_df.to_string(index=False))

    stats_csv = os.path.join(out_dir, "rank_n_statistics.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nSaved statistics -> {stats_csv}")

    # ---- CMC / rank-n accuracy table ----
    cmc = cmc_curve(positions, MAX_RANK)
    ranks = np.arange(1, MAX_RANK + 1)
    cmc_df = pd.DataFrame({"rank": ranks, "accuracy": cmc})
    cmc_csv = os.path.join(out_dir, "rank_n_accuracy.csv")
    cmc_df.to_csv(cmc_csv, index=False)
    print(f"Saved rank-n accuracy table -> {cmc_csv}")

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#bbbbbb",
        "axes.linewidth": 0.8,
    })

    accent = "#2563eb"
    y = cmc * 100

    def render(annotate: bool, filename: str):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(ranks, y, color=accent, linewidth=2.4, zorder=3,
                solid_capstyle="round")
        ax.plot(ranks, y, marker="o", linestyle="none", color="white",
                markeredgecolor=accent, markeredgewidth=1.6, markersize=6, zorder=4)

        ax.set_xlabel("Rank (n)", fontsize=12, labelpad=8, color="#333333")
        ax.set_ylabel("Rank-n accuracy (%)", fontsize=12, labelpad=8, color="#333333")
        ax.set_xticks(ranks)
        ax.set_xlim(0.5, MAX_RANK + 0.5)
        ax.set_ylim(60, 100)
        ax.tick_params(colors="#555555", labelsize=10, length=0)

        # only horizontal gridlines, light and behind the curve
        ax.grid(True, axis="y", linestyle="-", linewidth=0.7, color="#ececec", zorder=0)
        ax.set_axisbelow(True)

        # remove top/right spines for a cleaner frame
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cccccc")

        if annotate:
            for r, val in zip(ranks, y):
                ax.annotate(f"{val:.1f}%", (r, val), textcoords="offset points",
                            xytext=(0, 9), ha="center", va="bottom", rotation=90,
                            fontsize=7.5, color="#333333", zorder=5)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, filename)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot -> {plot_path}")

    render(annotate=False, filename="rank_n_accuracy.png")
    render(annotate=True, filename="rank_n_accuracy_labeled.png")


if __name__ == "__main__":
    main()
