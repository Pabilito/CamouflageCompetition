"""
data_processing/analyze_color_vs_rank.py
─────────────────────────────────────────────────────────────────────────────
Sort every adversarial image in the DB by mean similarity (across all
recognition models, ascending — rank 1 = lowest similarity = hardest to
recognize), then compute a small battery of perceptual-colour metrics on
the face region of each image relative to the *first* control image of
the same identity, and check how each metric correlates with the rank.

What gets computed, per adversarial image
─────────────────────────────────────────
Mean colour over the elliptical face mask of the 112×112 aligned crop:

    L*, a*, b*               CIE Lab components
    ΔL, Δa, Δb               component-wise delta vs first control
    C*       = √(a² + b²)    chroma (colorfulness)
    ΔC*                      chroma delta vs control
    ΔE*      = √(ΔL² + Δa² + Δb²)
                              total perceptual colour shift
    Saturation               mean HSV S (0–255)
    ΔSaturation              vs control
    L*_std                   std-dev of L* across face pixels
                              (heuristic for "how patchy" the face is)

Sort is ASCENDING average-similarity so the BEST adversarial image gets
rank 1 (this matches how Database.get_average_ranking is documented).

Outputs
───────
  --csv  data_processing/color_vs_rank.csv
  --plot data_processing/color_vs_rank.png

Console: a table of Pearson and Spearman correlation coefficients between
each metric and rank.  Spearman is the more honest one for "does this
predict ranking" since rank is itself a rank.

Usage
─────
    python data_processing/analyze_color_vs_rank.py
        [--db db.sqlite]
        [--data-dir data]
        [--csv data_processing/color_vs_rank.csv]
        [--plot data_processing/color_vs_rank.png]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.database import Database     # noqa: E402


# ── Face mask + Lab math (mirrors recognition_tab) ─────────────────────────

def face_ellipse_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry = h / 2.0
    rx = w / 2.0 * 0.82
    return ((yy - cy) ** 2) / (ry * ry) + ((xx - cx) ** 2) / (rx * rx) <= 1.0


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 H×W×3 → CIE Lab float H×W×3 (D65)."""
    rgb = rgb_uint8.astype(np.float64) / 255.0
    mask = rgb > 0.04045
    lin = np.where(mask,
                   ((rgb + 0.055) / 1.055) ** 2.4,
                   rgb / 12.92)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ M.T
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883
    eps   = (6 / 29) ** 3
    kappa = (29 / 6) ** 2 / 3
    f = np.where(xyz > eps,
                 np.cbrt(xyz),
                 kappa * xyz + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def load_aligned_112(path: Path) -> Image.Image | None:
    """Load an image and normalise to 112×112 so the face ellipse lines up."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    if img.size != (112, 112):
        img = _center_crop_square(img).resize((112, 112), Image.LANCZOS)
    return img


# ── Per-image metrics ──────────────────────────────────────────────────────

def compute_metrics(img: Image.Image) -> dict | None:
    """Return all per-image colour metrics, computed over the face ellipse.

    Returns None if the image can't be processed.  All values are float;
    None values are not produced — every metric is always populated.
    """
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = face_ellipse_mask(h, w)
    n = int(mask.sum())
    if n == 0:
        return None

    lab = rgb_to_lab(rgb)
    face_lab = lab[mask]                       # (n, 3)
    L_arr, a_arr, b_arr = face_lab[:, 0], face_lab[:, 1], face_lab[:, 2]
    L = float(L_arr.mean())
    a = float(a_arr.mean())
    b = float(b_arr.mean())
    L_std = float(L_arr.std())
    C = math.sqrt(a * a + b * b)
    h_deg = math.degrees(math.atan2(b, a))     # -180..180

    # HSV — PIL gives 0–255 per channel
    hsv = np.array(img.convert("HSV"))
    face_hsv = hsv[mask]
    H = float(face_hsv[:, 0].mean())
    S = float(face_hsv[:, 1].mean())
    V = float(face_hsv[:, 2].mean())

    # Plain RGB mean for reference / sanity
    face_rgb = rgb[mask].astype(np.float64)
    R = float(face_rgb[:, 0].mean())
    G = float(face_rgb[:, 1].mean())
    B = float(face_rgb[:, 2].mean())

    return {
        "L": L, "a": a, "b": b, "L_std": L_std,
        "C": C, "h_deg": h_deg,
        "H": H, "S": S, "V": V,
        "R": R, "G": G, "B": B,
    }


def signed_hue_delta(h1: float, h2: float) -> float:
    """Shortest signed angular difference h1 - h2, wrapped to (-180, 180]."""
    d = (h1 - h2) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


# ── PNG correlation table ──────────────────────────────────────────────────

def _write_corr_table(corr_rows: list, out_path: "Path") -> None:
    """Render the correlation results as a styled PNG table figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import to_rgba
    except ImportError:
        print("[TABLE] matplotlib not installed — skipping correlation table.")
        return

    headers = ["Metric", "Description", "Pearson r", "Spearman ρ", "|ρ|"]
    col_widths = [0.10, 0.36, 0.16, 0.16, 0.12]  # fractions of figure width

    rows_data = []
    for key, label, pr, sp in corr_rows:
        abs_sp = abs(sp) if not (sp != sp) else float("nan")  # isnan
        rows_data.append([
            key,
            label.strip(),
            f"{pr:+.3f}" if pr == pr else "—",
            f"{sp:+.3f}" if sp == sp else "—",
            f"{abs_sp:.3f}" if abs_sp == abs_sp else "—",
        ])

    n_rows = len(rows_data)
    fig_h  = 1.0 + n_rows * 0.38 + 0.6   # title + rows + note
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axis("off")

    # ── Colour palette ────────────────────────────────────────────────────
    HDR_BG  = "#1F3864"
    HDR_FG  = "white"
    ALT_BG  = "#F2F2F2"
    POS_BG  = "#D6F0D6"
    NEG_BG  = "#FCE4D6"
    NOTE_FG = "#595959"

    ROW_H  = 0.38 / fig_h        # normalised row height
    HDR_H  = 0.46 / fig_h
    TOP_Y  = 1.0 - (0.55 / fig_h)   # top of header row (axes-fraction)

    def draw_cell(ax, x, y, w, h, text, bg=None, fg="black",
                  bold=False, ha="center", fontsize=9.5):
        if bg:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y - h), w, h, boxstyle="square,pad=0",
                linewidth=0, color=bg, transform=ax.transAxes, zorder=1))
        ax.text(x + w * (0.06 if ha == "left" else 0.5),
                y - h / 2, text,
                transform=ax.transAxes, zorder=2,
                ha=ha, va="center",
                fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color=fg, clip_on=True)

    # ── Title ─────────────────────────────────────────────────────────────
    title_y = 1.0 - 0.05 / fig_h
    ax.text(0.5, title_y,
            "Colour-Metric × Rank Correlations",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color="#1F3864")

    # ── Header ────────────────────────────────────────────────────────────
    x = 0.0
    for i, (h, cw) in enumerate(zip(headers, col_widths)):
        draw_cell(ax, x, TOP_Y, cw, HDR_H, h,
                  bg=HDR_BG, fg=HDR_FG, bold=True, fontsize=9.5)
        x += cw

    # ── Data rows ─────────────────────────────────────────────────────────
    for ri, row in enumerate(rows_data):
        y = TOP_Y - HDR_H - ri * ROW_H
        x = 0.0
        _, _, pr_s, sp_s, _ = row
        # sign-based tint for numeric columns
        def sign_bg(s):
            if s.startswith("+"):  return POS_BG
            if s.startswith("-"):  return NEG_BG
            return ALT_BG if ri % 2 else "white"
        row_bg = ALT_BG if ri % 2 else "white"

        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            if ci == 0:
                bg, fg, bold, ha = row_bg, "#1F3864", True, "center"
            elif ci == 1:
                bg, fg, bold, ha = row_bg, "black", False, "left"
            elif ci in (2, 3):
                bg = sign_bg(cell)
                fg, bold, ha = "black", False, "center"
            else:
                bg, fg, bold, ha = row_bg, "black", False, "center"
            draw_cell(ax, x, y, cw, ROW_H, cell,
                      bg=bg, fg=fg, bold=bold, ha=ha, fontsize=9.5)
            x += cw

    # ── Grid lines ────────────────────────────────────────────────────────
    total_rows = n_rows
    # horizontal lines (in axes coordinates via ax.plot)
    def hline(y, color, lw):
        ax.plot([0, 1], [y, y], color=color, linewidth=lw,
                transform=ax.transAxes, zorder=3, solid_capstyle="butt")
    def vline(x, y0, y1, color, lw):
        ax.plot([x, x], [y0, y1], color=color, linewidth=lw,
                transform=ax.transAxes, zorder=3, solid_capstyle="butt")

    hline(TOP_Y,           "#AAAAAA", 0.5)
    hline(TOP_Y - HDR_H,  HDR_BG,    1.5)
    for ri in range(total_rows + 1):
        hline(TOP_Y - HDR_H - ri * ROW_H, "#CCCCCC", 0.4)
    x = 0.0
    table_bottom = TOP_Y - HDR_H - total_rows * ROW_H
    for cw in col_widths:
        vline(x, table_bottom, TOP_Y, "#CCCCCC", 0.4)
        x += cw
    vline(1.0, table_bottom, TOP_Y, "#CCCCCC", 0.4)

    # ── Legend / note ─────────────────────────────────────────────────────
    note_y = table_bottom - 0.18 / fig_h
    ax.text(0.0, note_y,
            "  green = positive ρ   red = negative ρ   "
            "|ρ| < 0.15 → no real relationship   "
            "Spearman ρ is the primary signal",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=NOTE_FG, style="italic")

    plt.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[TABLE] wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="db.sqlite",
                   help="Path to the SQLite DB (default: db.sqlite).")
    p.add_argument("--data-dir", default="data",
                   help="App data directory (default: data). Used to "
                        "locate data/users/<identity>/ control images.")
    p.add_argument("--csv", default="data_processing/results/color_vs_rank.csv",
                   help="Output CSV path.")
    p.add_argument("--plot", default="data_processing/results/color_vs_rank.png",
                   help="Output plot path. Use '' to skip plotting.")
    p.add_argument("--corr-table", default="data_processing/results/color_vs_rank_correlations.png",
                   help="Output PNG correlation table figure. Use '' to skip.")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2

    db = Database(str(db_path))

    # ── Rank all adversarial images by mean similarity (asc) ──────────────
    print(f"[DB] querying average ranking from {db_path}…")
    ranking = db.get_average_ranking(limit=10_000_000)
    if not ranking:
        print("ERROR: no adversarial images in the DB.", file=sys.stderr)
        return 2
    print(f"[DB] {len(ranking)} adversarial images ranked")

    # ── Per-identity control reference (mean face Lab of FIRST capture) ───
    # We resolve the identity from the image path (its parent folder name),
    # then look up that user's earliest capture and use its mean face Lab.
    ref_cache: dict[str, dict] = {}   # identity → {"L","a","b","C","h_deg","H","S","V","R","G","B"}

    def get_reference(identity: str) -> dict | None:
        if identity in ref_cache:
            return ref_cache[identity] or None
        user = db.get_user_by_name(identity)
        if user is None:
            ref_cache[identity] = {}     # sentinel for "no user"
            return None
        captures = db.list_captures(user.id)
        if not captures:
            ref_cache[identity] = {}
            return None
        first = captures[0]              # earliest by created_at
        ctrl_img = load_aligned_112(Path(first.image_path))
        if ctrl_img is None:
            ref_cache[identity] = {}
            return None
        m = compute_metrics(ctrl_img)
        if m is None:
            ref_cache[identity] = {}
            return None
        ref_cache[identity] = m
        return m

    # ── Process every adversarial image ───────────────────────────────────
    rows: list[dict] = []
    n_skipped = 0

    for rank, entry in enumerate(ranking, start=1):
        img_path_str = entry["image_path"]
        avg_score    = float(entry["avg_score"])
        n_models     = int(entry["n_models"])

        img_path = Path(img_path_str)
        if not img_path.exists():
            n_skipped += 1
            continue

        identity = img_path.parent.name
        adv_img  = load_aligned_112(img_path)
        if adv_img is None:
            n_skipped += 1
            continue

        adv = compute_metrics(adv_img)
        if adv is None:
            n_skipped += 1
            continue

        ref = get_reference(identity)
        if ref is None:
            # Image exists but the identity has no usable first control.
            # Keep the row but mark deltas as nan so they're excluded from
            # delta-based correlations rather than skewing them with zeros.
            dL = da = db_ = dC = dE = dh = dS = dV = dR = dG = dB = float("nan")
            ref_L = ref_a = ref_b = float("nan")
        else:
            dL  = adv["L"] - ref["L"]
            da  = adv["a"] - ref["a"]
            db_ = adv["b"] - ref["b"]
            dC  = adv["C"] - ref["C"]
            dh  = signed_hue_delta(adv["h_deg"], ref["h_deg"])
            dS  = adv["S"] - ref["S"]
            dV  = adv["V"] - ref["V"]
            dR  = adv["R"] - ref["R"]
            dG  = adv["G"] - ref["G"]
            dB  = adv["B"] - ref["B"]
            dE  = math.sqrt(dL * dL + da * da + db_ * db_)
            ref_L, ref_a, ref_b = ref["L"], ref["a"], ref["b"]

        rows.append({
            "rank":         rank,
            "avg_score":    avg_score,
            "n_models":     n_models,
            "identity":     identity,
            "image_path":   img_path_str,
            "L":            adv["L"],
            "a":            adv["a"],
            "b":            adv["b"],
            "L_std":        adv["L_std"],
            "C":            adv["C"],
            "h_deg":        adv["h_deg"],
            "S":            adv["S"],
            "V":            adv["V"],
            "R":            adv["R"],
            "G":            adv["G"],
            "B":            adv["B"],
            "ref_L":        ref_L,
            "ref_a":        ref_a,
            "ref_b":        ref_b,
            "dL":           dL,
            "da":           da,
            "db":           db_,
            "dC":           dC,
            "dh":           dh,
            "dE":           dE,
            "dS":           dS,
            "dV":           dV,
            "dR":           dR,
            "dG":           dG,
            "dB":           dB,
        })

    if not rows:
        print("ERROR: no usable rows after filtering.", file=sys.stderr)
        return 2
    print(f"[ANALYSE] processed {len(rows)} images, skipped {n_skipped}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[CSV] wrote {csv_path}  ({csv_path.stat().st_size/1024:.1f} KB)")

    # ── Correlations ──────────────────────────────────────────────────────
    # Spearman = rank-based; uses np.argsort twice for ranks (handles ties
    # passably).  Pearson = standard linear correlation. We report both
    # against the rank field, but Spearman is the more honest "predictive"
    # signal.
    rank_arr = np.array([r["rank"] for r in rows], dtype=np.float64)

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

    # Metric columns to evaluate.  Order also determines the plot grid order.
    METRICS = [
        ("L",     "L*  (lightness)"),
        ("a",     "a*  (−green / +red)"),
        ("b",     "b*  (−blue / +yellow)"),
        ("dL",    "ΔL*  vs first control"),
        ("da",    "Δa*  vs first control"),
        ("db",    "Δb*  vs first control"),
        ("C",     "C*  (chroma)"),
        ("dC",    "ΔC*  vs first control"),
        ("dE",    "ΔE*  (total Lab shift)"),
        ("S",     "Saturation (HSV S)"),
        ("dS",    "ΔSaturation vs control"),
        ("L_std", "L* std-dev across face"),
    ]

    corr_rows = []
    for key, _label in METRICS:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        pr = pearson(vals, rank_arr)
        sp = spearman(vals, rank_arr)
        corr_rows.append((key, _label, pr, sp))

    print()
    print(f"{'Metric':<10}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'|ρ|':>6}")
    print("-" * 44)
    for key, _label, pr, sp in corr_rows:
        print(f"{key:<10}  {pr:>+10.3f}  {sp:>+11.3f}  {abs(sp):>6.3f}")

    print()
    print("Interpretation: |ρ| close to 1 means the metric is a strong "
          "monotonic predictor of rank.  Positive ρ → metric grows with "
          "rank (worse images have HIGHER values); negative ρ → metric "
          "shrinks with rank (worse images have LOWER values).  "
          "|ρ| < 0.15 typically means no real relationship.")

    # ── Excel correlation table ───────────────────────────────────────────
    if args.corr_table:
        _write_corr_table(corr_rows, Path(args.corr_table))

    # ── Plot ──────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[PLOT] matplotlib not installed — skipping plot output.")
            return 0

        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        n_metrics = len(METRICS)
        cols = 3
        rows_grid = math.ceil(n_metrics / cols)
        fig, axes = plt.subplots(rows_grid, cols,
                                 figsize=(cols * 4.2, rows_grid * 3.0),
                                 squeeze=False)
        fig.suptitle(
            f"Colour metrics vs adversarial rank  ({len(rows)} images)\n"
            f"rank 1 = lowest mean similarity (hardest to recognize)",
            fontsize=12, y=0.995)

        for i, (key, label) in enumerate(METRICS):
            ax = axes[i // cols][i % cols]
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            ranks = rank_arr
            m = np.isfinite(vals)
            ax.scatter(ranks[m], vals[m], s=6, alpha=0.45,
                       color="#1f77b4", linewidths=0)
            ax.set_xlabel("rank", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

            # Linear fit line + correlation annotation
            if m.sum() >= 3 and vals[m].std() > 0:
                pr = pearson(vals, ranks)
                sp = spearman(vals, ranks)
                # Best-fit line (least-squares) — visual aid only
                coef = np.polyfit(ranks[m], vals[m], 1)
                xs = np.array([ranks[m].min(), ranks[m].max()])
                ax.plot(xs, coef[0] * xs + coef[1],
                        color="#d62728", linewidth=1.0, alpha=0.7)
                ax.set_title(f"Pearson r = {pr:+.3f}   "
                             f"Spearman ρ = {sp:+.3f}",
                             fontsize=8)
            else:
                ax.set_title("(insufficient data)", fontsize=8)

        # Hide any leftover empty subplots
        for j in range(n_metrics, rows_grid * cols):
            axes[j // cols][j % cols].axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(plot_path, dpi=140)
        print(f"[PLOT] wrote {plot_path}  "
              f"({plot_path.stat().st_size/1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
