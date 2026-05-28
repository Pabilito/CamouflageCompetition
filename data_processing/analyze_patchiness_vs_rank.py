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
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from core.database import Database     # noqa: E402
from analysis_common import (          # noqa: E402
    AVG_KEY, correlate, get_per_model_rankings,
    write_corr_table, write_scatter_grid,
)


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


def _hf_energy_fraction(channel: np.ndarray, cutoff: float = 0.25) -> float:
    """Fraction of 2-D FFT power above a radial spatial-frequency cutoff.

    `channel` is mean-subtracted, FFT'd, and power is summed in two radial
    bands: |f| > cutoff (high) vs all non-DC. Returns high / total in [0,1].
    cutoff is a fraction of the Nyquist radius (0.25 → outer 75% of spectrum).
    """
    c = channel - channel.mean()
    F = np.fft.fftshift(np.fft.fft2(c))
    power = np.abs(F) ** 2
    h, w = channel.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)  # 0 at DC, ~1 at edge
    total = power.sum() - power[int(cy), int(cx)]               # exclude DC
    if total <= 0:
        return 0.0
    high = power[r > cutoff].sum()
    return float(high / total)


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

    # ── SYMMETRY family ──────────────────────────────────────────────────
    # Left-right mirror difference of the face itself. Flip horizontally and
    # compare in Lab over the overlap of the mask and its mirror. Higher =
    # more left/right asymmetric (one-sided pattern). Reported as mean ΔE.
    lab_flip = lab[:, ::-1, :]
    mask_flip = mask[:, ::-1]
    both = mask & mask_flip
    if both.any():
        d = lab - lab_flip
        dE_sym = np.sqrt((d ** 2).sum(axis=-1))
        asymmetry = float(dE_sym[both].mean())          # mean mirror ΔE
        asymmetry_max = float(np.percentile(dE_sym[both], 90))
    else:
        asymmetry = asymmetry_max = 0.0

    # ── HIGH-FREQUENCY SPECTRAL ENERGY (whole face region) ───────────────
    # FFT high-band power fraction, averaged over Lab channels. The mask is
    # applied by zeroing outside the ellipse before transforming so only the
    # face region contributes; mean-subtraction inside the helper removes DC.
    Lm = np.where(mask, L, L[mask].mean())
    am = np.where(mask, a, a[mask].mean())
    bm = np.where(mask, b, b[mask].mean())
    hf_energy = float(np.mean([
        _hf_energy_fraction(Lm),
        _hf_energy_fraction(am),
        _hf_energy_fraction(bm),
    ]))

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
        "asymmetry":     asymmetry,
        "asymmetry_max": asymmetry_max,
        "hf_energy":     hf_energy,
    }


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
                   help="Output CSV path (long form, one row per image×model).")
    p.add_argument("--plot-dir", default="data_processing/results",
                   help="Directory for per-model scatter-grid PNGs "
                        "(patchiness_vs_rank_<model>.png). Use '' to skip plots.")
    p.add_argument("--corr-table", default="data_processing/results/patchiness_vs_rank_correlations.png",
                   help="Output merged PNG correlation table. Use '' to skip.")
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

    # ── Metric definitions (order drives table rows & plot grid) ──────────
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
        ("asymmetry",      "L-R mirror asymmetry (mean ΔE)"),
        ("asymmetry_max",  "L-R mirror asymmetry (90th-pct ΔE)"),
        ("hf_energy",      "High-freq spectral energy fraction"),
        ("d_edge_density", "ΔEdge density vs first control"),
        ("d_grad_mean",    "ΔMean gradient vs first control"),
        ("d_lab_energy",   "ΔLab energy vs first control"),
        ("d_color_entropy", "ΔColour entropy vs first control"),
    ]
    METRIC_KEYS = [k for k, _ in METRICS]

    # ── Per-image metric cache (compute once, reuse across all models) ────
    metric_cache: dict[str, dict | None] = {}

    def metrics_for_image(img_path_str: str) -> dict | None:
        if img_path_str in metric_cache:
            return metric_cache[img_path_str]
        img_path = Path(img_path_str)
        if not img_path.exists():
            metric_cache[img_path_str] = None
            return None
        adv_img = load_aligned_112(img_path)
        if adv_img is None:
            metric_cache[img_path_str] = None
            return None
        adv = compute_metrics(adv_img, args.edge_thresh, args.quant)
        if adv is None:
            metric_cache[img_path_str] = None
            return None

        identity = img_path.parent.name
        ref = get_reference(identity)
        if ref is None:
            d_edge_density = d_grad_mean = d_lab_energy = d_color_entropy = float("nan")
        else:
            d_edge_density  = adv["edge_density"]  - ref["edge_density"]
            d_grad_mean     = adv["grad_mean"]     - ref["grad_mean"]
            d_lab_energy    = adv["lab_energy"]    - ref["lab_energy"]
            d_color_entropy = adv["color_entropy"] - ref["color_entropy"]

        rec = dict(adv)
        rec.update({
            "identity":        identity,
            "d_edge_density":  d_edge_density,
            "d_grad_mean":     d_grad_mean,
            "d_lab_energy":    d_lab_energy,
            "d_color_entropy": d_color_entropy,
        })
        metric_cache[img_path_str] = rec
        return rec

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

    # ── Pull per-model (+AVG) rankings ────────────────────────────────────
    print(f"[DB] querying per-model rankings from {db_path}…")
    rankings = get_per_model_rankings(db)
    if not rankings or all(len(v) == 0 for v in rankings.values()):
        print("ERROR: no adversarial images in the DB.", file=sys.stderr)
        return 2
    print(f"[DB] models: {', '.join(rankings.keys())}")

    # ── Build per-model joined rows (metrics + score + rank) ──────────────
    per_model_rows: dict[str, list[dict]] = {}
    n_skipped_total = 0
    for model_name, ranking in rankings.items():
        rows: list[dict] = []
        skipped = 0
        for entry in ranking:
            rec = metrics_for_image(entry["image_path"])
            if rec is None:
                skipped += 1
                continue
            row = {
                "model":      model_name,
                "rank":       entry["rank"],
                "score":      entry["score"],
                "identity":   rec["identity"],
                "image_path": entry["image_path"],
            }
            for k in METRIC_KEYS:
                row[k] = rec[k]
            rows.append(row)
        per_model_rows[model_name] = rows
        n_skipped_total += skipped
        print(f"[ANALYSE] {model_name:<28} {len(rows):>5} images "
              f"(skipped {skipped})")

    if all(len(r) == 0 for r in per_model_rows.values()):
        print("ERROR: no usable rows after filtering.", file=sys.stderr)
        return 2

    # ── Write long-form CSV (one row per image×model) ─────────────────────
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["model", "rank", "score", "identity", "image_path"]
                  + METRIC_KEYS)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model_name, rows in per_model_rows.items():
            for r in rows:
                w.writerow(r)
    print(f"[CSV] wrote {csv_path}  ({csv_path.stat().st_size/1024:.1f} KB)")

    # ── Correlations per model ────────────────────────────────────────────
    # Pearson r vs similarity score; Spearman ρ vs rank.
    per_model_corr: dict[str, list[tuple]] = {}
    for model_name, rows in per_model_rows.items():
        if not rows:
            per_model_corr[model_name] = [
                (k, lbl, float("nan"), float("nan")) for k, lbl in METRICS]
            continue
        score_arr = np.array([r["score"] for r in rows], dtype=np.float64)
        rank_arr  = np.array([r["rank"] for r in rows], dtype=np.float64)
        corr_rows = []
        for key, label in METRICS:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            pr, sp = correlate(vals, score_arr, rank_arr)
            corr_rows.append((key, label, pr, sp))
        per_model_corr[model_name] = corr_rows

    # ── Console summary (AVG model) ───────────────────────────────────────
    avg_corr = per_model_corr.get(AVG_KEY, [])
    print()
    print(f"=== {AVG_KEY} (cross-model average) ===")
    print(f"{'Metric':<16}  {'Pearson r (score)':>18}  {'Spearman ρ (rank)':>18}")
    print("-" * 58)
    for key, _label, pr, sp in avg_corr:
        pr_s = f"{pr:>+18.3f}" if pr == pr else f"{'—':>18}"
        sp_s = f"{sp:>+18.3f}" if sp == sp else f"{'—':>18}"
        print(f"{key:<16}  {pr_s}  {sp_s}")
    print()
    print("Pearson r is vs the raw similarity score (linear); Spearman ρ is "
          "vs rank (monotonic). They usually have OPPOSITE signs because "
          "lower score = better attack = lower rank number. |corr| < 0.15 "
          "≈ no real relationship. See the merged table for all models.")

    # ── Merged correlation table PNG ──────────────────────────────────────
    if args.corr_table:
        write_corr_table(
            per_model_corr, METRICS, Path(args.corr_table),
            title="Patchiness-Metric Correlations (per model)")

    # ── Per-model scatter-grid PNGs ───────────────────────────────────────
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        for model_name, rows in per_model_rows.items():
            if not rows:
                continue
            safe = model_name.replace("/", "_").replace(" ", "_")
            out = plot_dir / f"patchiness_vs_rank_{safe}.png"
            write_scatter_grid(model_name, rows, METRICS, out,
                               title_prefix="Patchiness metrics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
