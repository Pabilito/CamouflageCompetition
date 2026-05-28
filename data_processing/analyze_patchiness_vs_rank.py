"""
data_processing/analyze_patchiness_vs_rank.py
─────────────────────────────────────────────────────────────────────────────
Sort every adversarial image in the DB by mean similarity (across all
recognition models, ascending — rank 1 = lowest similarity = hardest to
recognize), then measure how "patchy" the face region of each image is —
i.e. the *frequency of colour transitions* across the face — and check how
each patchiness metric correlates with the rank.

Same shape as analyze_color_vs_rank.py: same DB ranking, same 112×112
elliptical face mask, same low-quality saved crops (the .jpg path stored in
the DB — NOT the *_hq.png companion), same Pearson + Spearman table/plot.

Why several metrics
───────────────────
"Frequency of colour transitions / patchiness" has no single canonical
definition, so we compute a battery of complementary measures and correlate
each one independently. They fall into three families:

  EDGE-DENSITY  (how often neighbouring pixels differ in colour)
    edge_density       fraction of face pixels whose local colour gradient
                       exceeds a threshold (Sobel on CIE-Lab, ΔE per step)
    grad_mean          mean per-pixel colour-gradient magnitude (ΔE/px)
    grad_p90           90th-percentile gradient magnitude (strong edges)

  LOCAL-VARIANCE / TEXTURE ENERGY  (how busy the face is)
    local_var          mean local variance of L* in a 3×3 window
    lab_energy         mean local colour variance (L*+a*+b*) in 3×3 window
    laplacian_var      variance of the Laplacian of L* (focus/“busyness”)

  REGION-COUNT / SEGMENTATION  (how many distinct colour patches)
    n_regions          number of connected colour regions after coarse
                       Lab quantisation (patch count, normalised by area)
    region_density     n_regions per 1000 face pixels
    color_entropy      Shannon entropy of the quantised-colour histogram
                       (bits; high = many colours used evenly = patchy)

Delta-vs-control (ΔE-style) is added ONLY for the global scalar texture
energies where a "more/less patchy than your own control face" comparison
is meaningful: d_edge_density, d_grad_mean, d_lab_energy, d_color_entropy.
Region counts are absolute by nature, so no delta is produced for them.

Sort is ASCENDING average-similarity so the BEST adversarial image gets
rank 1 (matches Database.get_average_ranking).

Outputs
───────
  --csv         data_processing/results/patchiness_vs_rank.csv
  --plot        data_processing/results/patchiness_vs_rank.png
  --corr-table  data_processing/results/patchiness_vs_rank_correlations.png

Console: a table of Pearson and Spearman correlation coefficients between
each metric and rank.  Spearman is the more honest one for "does this
predict ranking" since rank is itself a rank.

Usage
─────
    python data_processing/analyze_patchiness_vs_rank.py
        [--db db.sqlite]
        [--data-dir data]
        [--csv data_processing/results/patchiness_vs_rank.csv]
        [--plot data_processing/results/patchiness_vs_rank.png]
        [--corr-table data_processing/results/patchiness_vs_rank_correlations.png]
        [--edge-thresh 8.0]
        [--quant 16]
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


# ── Face mask + Lab math (mirrors analyze_color_vs_rank / recognition_tab) ──

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


# ── Small numpy helpers (no SciPy dependency) ──────────────────────────────

def _sobel(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients of a 2-D float array via manual 3×3 convolution.

    Edges are handled with edge-replication padding so output matches the
    input shape.  Returns (gx, gy).
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    p = np.pad(channel, 1, mode="edge")
    gx = np.zeros_like(channel)
    gy = np.zeros_like(channel)
    for dy in range(3):
        for dx in range(3):
            win = p[dy:dy + channel.shape[0], dx:dx + channel.shape[1]]
            gx += kx[dy, dx] * win
            gy += ky[dy, dx] * win
    return gx, gy


def _local_window_stats(channel: np.ndarray) -> np.ndarray:
    """Per-pixel variance of `channel` in a 3×3 neighbourhood (edge-padded)."""
    p = np.pad(channel, 1, mode="edge")
    stack = []
    for dy in range(3):
        for dx in range(3):
            stack.append(p[dy:dy + channel.shape[0], dx:dx + channel.shape[1]])
    stack = np.stack(stack, axis=0)            # (9, H, W)
    return stack.var(axis=0)                   # population variance per pixel


def _laplacian(channel: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian of a 2-D array (edge-padded)."""
    p = np.pad(channel, 1, mode="edge")
    up    = p[0:-2, 1:-1]
    down  = p[2:,   1:-1]
    left  = p[1:-1, 0:-2]
    right = p[1:-1, 2:]
    return (up + down + left + right) - 4.0 * channel


def _count_regions(labels_2d: np.ndarray, mask: np.ndarray) -> int:
    """Count connected components of equal-label, mask-restricted pixels.

    4-connectivity, iterative union-find flood over the masked region.
    `labels_2d` is an int array of quantised-colour labels.
    """
    h, w = labels_2d.shape
    visited = np.zeros((h, w), dtype=bool)
    n = 0
    # Iterative stack-based flood fill (avoids recursion depth limits).
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            n += 1
            lab = labels_2d[y, x]
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w \
                            and not visited[ny, nx] and mask[ny, nx] \
                            and labels_2d[ny, nx] == lab:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return n


# ── Per-image patchiness metrics ───────────────────────────────────────────

def compute_metrics(img: Image.Image,
                    edge_thresh: float = 8.0,
                    quant: int = 16) -> dict | None:
    """Return all per-image patchiness metrics over the face ellipse.

    Parameters
    ----------
    edge_thresh : ΔE-per-pixel-step gradient magnitude above which a pixel is
                  counted as an "edge" (a colour transition) for edge_density.
    quant       : number of quantisation levels per Lab axis used for the
                  region-count / colour-entropy family.

    Returns None if the image can't be processed.
    """
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = face_ellipse_mask(h, w)
    n = int(mask.sum())
    if n == 0:
        return None

    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # ── EDGE-DENSITY family ──────────────────────────────────────────────
    # Per-channel Sobel, combined into a Lab colour-gradient magnitude
    # (ΔE per spatial step). This is the "frequency of colour transitions".
    gLx, gLy = _sobel(L)
    gax, gay = _sobel(a)
    gbx, gby = _sobel(b)
    grad = np.sqrt(gLx**2 + gLy**2 + gax**2 + gay**2 + gbx**2 + gby**2)
    grad_face = grad[mask]

    grad_mean    = float(grad_face.mean())
    grad_p90     = float(np.percentile(grad_face, 90))
    edge_density = float((grad_face > edge_thresh).mean())   # 0..1

    # ── LOCAL-VARIANCE / TEXTURE-ENERGY family ───────────────────────────
    local_var = float(_local_window_stats(L)[mask].mean())
    lab_energy = float((
        _local_window_stats(L) +
        _local_window_stats(a) +
        _local_window_stats(b)
    )[mask].mean())
    laplacian_var = float(_laplacian(L)[mask].var())

    # ── REGION-COUNT / SEGMENTATION family ───────────────────────────────
    # Coarsely quantise Lab into `quant` levels per axis, label each pixel by
    # its quantised colour, then count connected same-colour patches inside
    # the mask. More patches over the same area = patchier.
    qL = np.clip((L / (100.0 / quant)).astype(np.int64), 0, quant - 1)
    qa = np.clip(((a + 128.0) / (256.0 / quant)).astype(np.int64), 0, quant - 1)
    qb = np.clip(((b + 128.0) / (256.0 / quant)).astype(np.int64), 0, quant - 1)
    labels = (qL * quant + qa) * quant + qb     # unique int per colour cell

    n_regions = int(_count_regions(labels, mask))
    region_density = float(n_regions / n * 1000.0)   # regions per 1000 px

    # Colour entropy over the quantised palette inside the mask
    flat = labels[mask]
    _, counts = np.unique(flat, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    color_entropy = float(-(p * np.log2(p)).sum())   # bits

    return {
        "edge_density":  edge_density,
        "grad_mean":     grad_mean,
        "grad_p90":      grad_p90,
        "local_var":     local_var,
        "lab_energy":    lab_energy,
        "laplacian_var": laplacian_var,
        "n_regions":     float(n_regions),
        "region_density": region_density,
        "color_entropy": color_entropy,
    }


# ── PNG correlation table (mirrors analyze_color_vs_rank) ──────────────────

def _write_corr_table(corr_rows: list, out_path: "Path") -> None:
    """Render the correlation results as a styled PNG table figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[TABLE] matplotlib not installed — skipping correlation table.")
        return

    headers = ["Metric", "Description", "Pearson r", "Spearman ρ", "|ρ|"]
    col_widths = [0.14, 0.40, 0.15, 0.16, 0.11]

    rows_data = []
    for key, label, pr, sp in corr_rows:
        abs_sp = abs(sp) if not (sp != sp) else float("nan")
        rows_data.append([
            key,
            label.strip(),
            f"{pr:+.3f}" if pr == pr else "—",
            f"{sp:+.3f}" if sp == sp else "—",
            f"{abs_sp:.3f}" if abs_sp == abs_sp else "—",
        ])

    n_rows = len(rows_data)
    fig_h  = 1.0 + n_rows * 0.38 + 0.6
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.axis("off")

    HDR_BG  = "#1F3864"
    HDR_FG  = "white"
    ALT_BG  = "#F2F2F2"
    POS_BG  = "#D6F0D6"
    NEG_BG  = "#FCE4D6"
    NOTE_FG = "#595959"

    ROW_H  = 0.38 / fig_h
    HDR_H  = 0.46 / fig_h
    TOP_Y  = 1.0 - (0.55 / fig_h)

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

    title_y = 1.0 - 0.05 / fig_h
    ax.text(0.5, title_y,
            "Patchiness-Metric × Rank Correlations",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color="#1F3864")

    x = 0.0
    for h, cw in zip(headers, col_widths):
        draw_cell(ax, x, TOP_Y, cw, HDR_H, h,
                  bg=HDR_BG, fg=HDR_FG, bold=True, fontsize=9.5)
        x += cw

    for ri, row in enumerate(rows_data):
        y = TOP_Y - HDR_H - ri * ROW_H
        x = 0.0
        row_bg = ALT_BG if ri % 2 else "white"

        def sign_bg(s):
            if s.startswith("+"):  return POS_BG
            if s.startswith("-"):  return NEG_BG
            return row_bg

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

    def hline(y, color, lw):
        ax.plot([0, 1], [y, y], color=color, linewidth=lw,
                transform=ax.transAxes, zorder=3, solid_capstyle="butt")
    def vline(x, y0, y1, color, lw):
        ax.plot([x, x], [y0, y1], color=color, linewidth=lw,
                transform=ax.transAxes, zorder=3, solid_capstyle="butt")

    hline(TOP_Y,          "#AAAAAA", 0.5)
    hline(TOP_Y - HDR_H,  HDR_BG,    1.5)
    for ri in range(n_rows + 1):
        hline(TOP_Y - HDR_H - ri * ROW_H, "#CCCCCC", 0.4)
    x = 0.0
    table_bottom = TOP_Y - HDR_H - n_rows * ROW_H
    for cw in col_widths:
        vline(x, table_bottom, TOP_Y, "#CCCCCC", 0.4)
        x += cw
    vline(1.0, table_bottom, TOP_Y, "#CCCCCC", 0.4)

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
    p.add_argument("--csv", default="data_processing/results/patchiness_vs_rank.csv",
                   help="Output CSV path.")
    p.add_argument("--plot", default="data_processing/results/patchiness_vs_rank.png",
                   help="Output plot path. Use '' to skip plotting.")
    p.add_argument("--corr-table", default="data_processing/results/patchiness_vs_rank_correlations.png",
                   help="Output PNG correlation table figure. Use '' to skip.")
    p.add_argument("--edge-thresh", type=float, default=8.0,
                   help="Lab colour-gradient magnitude (ΔE/step) above which "
                        "a pixel counts as a colour transition (default 8.0).")
    p.add_argument("--quant", type=int, default=16,
                   help="Quantisation levels per Lab axis for region/entropy "
                        "metrics (default 16).")
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

    # ── Per-identity control reference (FIRST capture, low-quality crop) ──
    ref_cache: dict[str, dict] = {}

    def get_reference(identity: str) -> dict | None:
        if identity in ref_cache:
            return ref_cache[identity] or None
        user = db.get_user_by_name(identity)
        if user is None:
            ref_cache[identity] = {}
            return None
        captures = db.list_captures(user.id)
        if not captures:
            ref_cache[identity] = {}
            return None
        first = captures[0]
        ctrl_img = load_aligned_112(Path(first.image_path))
        if ctrl_img is None:
            ref_cache[identity] = {}
            return None
        m = compute_metrics(ctrl_img, args.edge_thresh, args.quant)
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

        adv = compute_metrics(adv_img, args.edge_thresh, args.quant)
        if adv is None:
            n_skipped += 1
            continue

        ref = get_reference(identity)
        if ref is None:
            d_edge_density = d_grad_mean = d_lab_energy = d_color_entropy = float("nan")
        else:
            d_edge_density  = adv["edge_density"]  - ref["edge_density"]
            d_grad_mean     = adv["grad_mean"]     - ref["grad_mean"]
            d_lab_energy    = adv["lab_energy"]    - ref["lab_energy"]
            d_color_entropy = adv["color_entropy"] - ref["color_entropy"]

        rows.append({
            "rank":           rank,
            "avg_score":      avg_score,
            "n_models":       n_models,
            "identity":       identity,
            "image_path":     img_path_str,
            "edge_density":   adv["edge_density"],
            "grad_mean":      adv["grad_mean"],
            "grad_p90":       adv["grad_p90"],
            "local_var":      adv["local_var"],
            "lab_energy":     adv["lab_energy"],
            "laplacian_var":  adv["laplacian_var"],
            "n_regions":      adv["n_regions"],
            "region_density": adv["region_density"],
            "color_entropy":  adv["color_entropy"],
            "d_edge_density":  d_edge_density,
            "d_grad_mean":     d_grad_mean,
            "d_lab_energy":    d_lab_energy,
            "d_color_entropy": d_color_entropy,
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

    # Metric columns to evaluate. Order also determines the plot grid order.
    METRICS = [
        ("edge_density",   "Edge density (colour-transition px frac)"),
        ("grad_mean",      "Mean Lab gradient (ΔE / px)"),
        ("grad_p90",       "90th-pct Lab gradient (strong edges)"),
        ("local_var",      "Local L* variance (3×3)"),
        ("lab_energy",     "Local Lab variance energy (3×3)"),
        ("laplacian_var",  "Laplacian-of-L* variance (busyness)"),
        ("n_regions",      "Connected colour regions (count)"),
        ("region_density", "Regions per 1000 face px"),
        ("color_entropy",  "Quantised-colour entropy (bits)"),
        ("d_edge_density", "ΔEdge density vs first control"),
        ("d_grad_mean",    "ΔMean gradient vs first control"),
        ("d_lab_energy",   "ΔLab energy vs first control"),
        ("d_color_entropy", "ΔColour entropy vs first control"),
    ]

    corr_rows = []
    for key, _label in METRICS:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        pr = pearson(vals, rank_arr)
        sp = spearman(vals, rank_arr)
        corr_rows.append((key, _label, pr, sp))

    print()
    print(f"{'Metric':<16}  {'Pearson r':>10}  {'Spearman ρ':>11}  {'|ρ|':>6}")
    print("-" * 50)
    for key, _label, pr, sp in corr_rows:
        pr_s = f"{pr:>+10.3f}" if pr == pr else f"{'—':>10}"
        sp_s = f"{sp:>+11.3f}" if sp == sp else f"{'—':>11}"
        ab_s = f"{abs(sp):>6.3f}" if sp == sp else f"{'—':>6}"
        print(f"{key:<16}  {pr_s}  {sp_s}  {ab_s}")

    print()
    print("Interpretation: |ρ| close to 1 means the metric is a strong "
          "monotonic predictor of rank.  Positive ρ → metric grows with "
          "rank (worse-recognised images are PATCHIER); negative ρ → metric "
          "shrinks with rank (worse-recognised images are SMOOTHER).  "
          "|ρ| < 0.15 typically means no real relationship.")

    # ── Correlation table PNG ─────────────────────────────────────────────
    if args.corr_table:
        _write_corr_table(corr_rows, Path(args.corr_table))

    # ── Scatter plot grid ─────────────────────────────────────────────────
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
            f"Patchiness metrics vs adversarial rank  ({len(rows)} images)\n"
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

            if m.sum() >= 3 and vals[m].std() > 0:
                pr = pearson(vals, ranks)
                sp = spearman(vals, ranks)
                coef = np.polyfit(ranks[m], vals[m], 1)
                xs = np.array([ranks[m].min(), ranks[m].max()])
                ax.plot(xs, coef[0] * xs + coef[1],
                        color="#d62728", linewidth=1.0, alpha=0.7)
                ax.set_title(f"Pearson r = {pr:+.3f}   "
                             f"Spearman ρ = {sp:+.3f}",
                             fontsize=8)
            else:
                ax.set_title("(insufficient data)", fontsize=8)

        for j in range(n_metrics, rows_grid * cols):
            axes[j // cols][j % cols].axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(plot_path, dpi=140)
        print(f"[PLOT] wrote {plot_path}  "
              f"({plot_path.stat().st_size/1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
