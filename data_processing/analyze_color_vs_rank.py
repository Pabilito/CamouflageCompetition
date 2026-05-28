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
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from core.database import Database     # noqa: E402
from analysis_common import (          # noqa: E402
    AVG_KEY, correlate, get_per_model_rankings,
    write_corr_table, write_scatter_grid,
)


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
                   help="Output CSV path (long form, one row per image×model).")
    p.add_argument("--plot-dir", default="data_processing/results",
                   help="Directory for per-model scatter-grid PNGs "
                        "(color_vs_rank_<model>.png). Use '' to skip plots.")
    p.add_argument("--corr-table", default="data_processing/results/color_vs_rank_correlations.png",
                   help="Output merged PNG correlation table. Use '' to skip.")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2

    db = Database(str(db_path))

    # ── Metric definitions (order drives table rows & plot grid) ──────────
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
    METRIC_KEYS = [k for k, _ in METRICS]

    # ── Per-identity control reference (mean face Lab of FIRST capture) ───
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
        m = compute_metrics(ctrl_img)
        if m is None:
            ref_cache[identity] = {}
            return None
        ref_cache[identity] = m
        return m

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
        adv = compute_metrics(adv_img)
        if adv is None:
            metric_cache[img_path_str] = None
            return None

        identity = img_path.parent.name
        ref = get_reference(identity)
        if ref is None:
            dL = da = db_ = dC = dE = dh = dS = dV = dR = dG = dB = float("nan")
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

        rec = dict(adv)
        rec.update({
            "identity": identity,
            "dL": dL, "da": da, "db": db_, "dC": dC, "dh": dh,
            "dE": dE, "dS": dS, "dV": dV, "dR": dR, "dG": dG, "dB": dB,
        })
        metric_cache[img_path_str] = rec
        return rec

    # ── Pull per-model (+AVG) rankings ────────────────────────────────────
    print(f"[DB] querying per-model rankings from {db_path}…")
    rankings = get_per_model_rankings(db)
    if not rankings or all(len(v) == 0 for v in rankings.values()):
        print("ERROR: no adversarial images in the DB.", file=sys.stderr)
        return 2
    print(f"[DB] models: {', '.join(rankings.keys())}")

    # ── Build per-model joined rows (metrics + score + rank) ──────────────
    # CSV carries the full metric record (incl. raw RGB / extra deltas) so it
    # stays a superset of the old single-model CSV.
    EXTRA_CSV_KEYS = ["h_deg", "H", "V", "R", "G", "B",
                      "dh", "dV", "dR", "dG", "dB"]
    per_model_rows: dict[str, list[dict]] = {}
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
            for k in METRIC_KEYS + EXTRA_CSV_KEYS:
                row[k] = rec.get(k, float("nan"))
            rows.append(row)
        per_model_rows[model_name] = rows
        print(f"[ANALYSE] {model_name:<28} {len(rows):>5} images "
              f"(skipped {skipped})")

    if all(len(r) == 0 for r in per_model_rows.values()):
        print("ERROR: no usable rows after filtering.", file=sys.stderr)
        return 2

    # ── Write long-form CSV (one row per image×model) ─────────────────────
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["model", "rank", "score", "identity", "image_path"]
                  + METRIC_KEYS + EXTRA_CSV_KEYS)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model_name, rows in per_model_rows.items():
            for r in rows:
                w.writerow(r)
    print(f"[CSV] wrote {csv_path}  ({csv_path.stat().st_size/1024:.1f} KB)")

    # ── Correlations per model: Pearson vs score, Spearman vs rank ────────
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
    print(f"{'Metric':<10}  {'Pearson r (score)':>18}  {'Spearman ρ (rank)':>18}")
    print("-" * 52)
    for key, _label, pr, sp in avg_corr:
        pr_s = f"{pr:>+18.3f}" if pr == pr else f"{'—':>18}"
        sp_s = f"{sp:>+18.3f}" if sp == sp else f"{'—':>18}"
        print(f"{key:<10}  {pr_s}  {sp_s}")
    print()
    print("Pearson r is vs the raw similarity score (linear); Spearman ρ is "
          "vs rank (monotonic). They usually have OPPOSITE signs because "
          "lower score = better attack = lower rank number. |corr| < 0.15 "
          "≈ no real relationship. See the merged table for all models.")

    # ── Merged correlation table PNG ──────────────────────────────────────
    if args.corr_table:
        write_corr_table(
            per_model_corr, METRICS, Path(args.corr_table),
            title="Colour-Metric Correlations (per model)")

    # ── Per-model scatter-grid PNGs ───────────────────────────────────────
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        for model_name, rows in per_model_rows.items():
            if not rows:
                continue
            safe = model_name.replace("/", "_").replace(" ", "_")
            out = plot_dir / f"color_vs_rank_{safe}.png"
            write_scatter_grid(model_name, rows, METRICS, out,
                               title_prefix="Colour metrics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
