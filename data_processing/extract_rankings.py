"""
data_processing/extract_rankings.py
────────────────────────────────────────────────────────────────────────────
Export ranking data from the SQLite database to CSV files.

Produces two files:
  1. current_standings.csv  — one row per identity (attacker), with the best
     average similarity score, per-model scores for the best image, and
     timestamp.  Mirrors the "Current Standings" sub-tab.

  2. gallery_search.csv    — one row per identity, with gallery-search
     position statistics (best / worst / avg position, number of images).
     Mirrors the "Gallery Search" sub-tab.

Usage
─────
    python data_processing/extract_rankings.py [--db DB_PATH] [--out DIR]

Defaults:
    --db   db.sqlite          (same default as Database)
    --out  data_processing/results  (sibling of this script)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import DEFAULT_DB, RESULTS_DIR, setup_path, ensure_results_dir
setup_path()  # add project root to sys.path for core.database

from core.database import Database


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model_names_from_db(db: Database) -> list[str]:
    """Return an ordered list of distinct model names present in the attacks
    table.  Sorted alphabetically so the column order is deterministic."""
    with db._connect() as con:
        rows = con.execute(
            "SELECT DISTINCT model_name FROM attacks ORDER BY model_name"
        ).fetchall()
    return [r["model_name"] for r in rows]


# ── Current Standings export ─────────────────────────────────────────────────

def export_current_standings(db: Database, out_path: Path) -> int:
    """Write current_standings.csv and return the number of data rows."""

    leaderboard = db.get_user_leaderboard()
    all_rows = db.get_ranking(model_name=None, limit=10_000)
    model_names = _model_names_from_db(db)

    # Build per-image score map and metadata
    img_scores: dict[str, dict[str, float]] = {}
    img_meta: dict[str, dict] = {}
    for r in all_rows:
        if r.image_path not in img_scores:
            img_scores[r.image_path] = {}
            img_meta[r.image_path] = {
                "attacker": r.attacker_name,
                "created_at": r.created_at,
            }
        img_scores[r.image_path][r.model_name] = r.similarity_score

    # Per-image average
    img_avg: dict[str, float] = {
        p: sum(s.values()) / len(s)
        for p, s in img_scores.items() if s
    }

    # Per-identity best image (lowest avg)
    identity_best_img: dict[str, str] = {}
    for entry in leaderboard:
        identity = entry["identity_name"]
        candidates = [
            p for p, m in img_meta.items()
            if m["attacker"] == identity
        ]
        if candidates:
            identity_best_img[identity] = min(
                candidates, key=lambda p: img_avg.get(p, 1.0)
            )

    # Build rows
    fieldnames = ["rank", "identity", "best_avg_score"]
    for mn in model_names:
        fieldnames.append(f"score_{mn}")
    fieldnames += ["best_image_path", "timestamp"]

    csv_rows: list[dict] = []
    for rank, entry in enumerate(leaderboard, start=1):
        identity = entry["identity_name"]
        best_avg = entry["best_avg_score"]
        path = identity_best_img.get(identity, "")
        scores = img_scores.get(path, {})
        ts = img_meta.get(path, {}).get("created_at", "")

        row = {
            "rank": rank,
            "identity": identity,
            "best_avg_score": f"{best_avg:.6f}",
        }
        for mn in model_names:
            s = scores.get(mn)
            row[f"score_{mn}"] = f"{s:.6f}" if s is not None else ""
        row["best_image_path"] = path
        row["timestamp"] = ts
        csv_rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return len(csv_rows)


# ── Gallery Search export ────────────────────────────────────────────────────

def export_gallery_search(db: Database, out_path: Path) -> int:
    """Write gallery_search.csv and return the number of data rows."""

    gs_rows = db.get_gallery_search_data()

    # Group by identity
    identity_images: dict[str, list[dict]] = {}
    for r in gs_rows:
        ident = r["identity_name"]
        if not ident:
            continue
        pos = r["gallery_position"]
        if pos is None:
            continue
        identity_images.setdefault(ident, []).append(r)

    # Build cache exactly like ranking_tab._refresh_gallery_search
    cache: list[dict] = []
    for identity, images in identity_images.items():
        positions = [img["gallery_position"] for img in images]
        gallery_total_vals = [
            img["gallery_total"] for img in images
            if img.get("gallery_total") is not None
        ]
        best_pos = max(positions)
        worst_pos = min(positions)
        avg_pos = sum(positions) / len(positions)

        best_img_r = max(images, key=lambda x: x["gallery_position"])

        cache.append({
            "identity": identity,
            "n_images": len(images),
            "best_pos": best_pos,
            "worst_pos": worst_pos,
            "avg_pos": avg_pos,
            "gallery_total": gallery_total_vals[0] if gallery_total_vals else "",
            "best_image_path": best_img_r["image_path"],
            "best_matched_image": best_img_r.get("matched_image_path", ""),
        })

    # Sort: highest best_pos first (= best attack), then by avg_pos descending
    cache.sort(key=lambda x: (x["best_pos"], x["avg_pos"]), reverse=True)

    fieldnames = [
        "rank", "identity", "n_images",
        "best_position", "worst_position", "avg_position",
        "gallery_total",
        "best_image_path", "best_matched_image",
    ]

    csv_rows: list[dict] = []
    for rank, entry in enumerate(cache, start=1):
        csv_rows.append({
            "rank": rank,
            "identity": entry["identity"],
            "n_images": entry["n_images"],
            "best_position": entry["best_pos"],
            "worst_position": entry["worst_pos"],
            "avg_position": f"{entry['avg_pos']:.1f}",
            "gallery_total": entry["gallery_total"],
            "best_image_path": entry["best_image_path"],
            "best_matched_image": entry["best_matched_image"],
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return len(csv_rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ranking data from the database to CSV files."
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help="Path to the SQLite database (default: db.sqlite)",
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR),
        help="Output directory for CSV files (default: data_processing/results/)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_results_dir(Path(args.out))

    db = Database(str(db_path))

    # ── Current Standings ─────────────────────────────────────────────────
    standings_path = out_dir / "current_standings.csv"
    n_standings = export_current_standings(db, standings_path)
    print(f"Wrote {standings_path}  ({n_standings} rows)")

    # ── Gallery Search ────────────────────────────────────────────────────
    gs_path = out_dir / "gallery_search.csv"
    n_gs = export_gallery_search(db, gs_path)
    print(f"Wrote {gs_path}  ({n_gs} rows)")


if __name__ == "__main__":
    main()
