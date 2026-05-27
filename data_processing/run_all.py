#!/usr/bin/env python3
"""
data_processing/run_all.py
─────────────────────────────────────────────────────────────────────────────
Master script that runs every data-processing sub-script in sequence.

Each sub-script is invoked as a separate process so that failures in one
do not prevent the others from running.

Usage
─────
    python data_processing/run_all.py [--db DB_PATH] [--out DIR]

Defaults:
    --db   db.sqlite
    --out  data_processing/results/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

# Sub-scripts in execution order.
# Each entry is (filename, description, needs_db).
# needs_db=False → only --out is passed (e.g. visualize_results.py reads CSVs,
# not the database directly).
SCRIPTS: list[tuple[str, str, bool]] = [
    ("extract_rankings.py",         "Ranking exports (standings + gallery search)",  True),
    ("model_accuracy.py",           "Model accuracy (overall / pose / pattern)",     True),
    ("score_statistics.py",         "Score statistics (model-independent)",          True),
    ("cross_pattern_similarity.py", "Cross-pattern adversarial similarity",          True),
    ("gallery_search_stats.py",     "Gallery search statistics and figures",         True),
    ("score_distributions.py",      "Similarity score distribution plots per model", True),
    ("visualize_results.py",        "Visualisations for accuracy / score / pattern", False),
    ("analyze_color_vs_rank.py",    "Analyze colors",                                True),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all data-processing scripts.")
    parser.add_argument("--db",  default="db.sqlite",
                        help="Path to the SQLite database (default: db.sqlite)")
    parser.add_argument("--out", default=str(_SCRIPT_DIR / "results"),
                        help="Output directory (default: data_processing/results/)")
    args = parser.parse_args()

    db_arg  = args.db
    out_arg = args.out
    python  = sys.executable

    print("=" * 70)
    print("  DATA PROCESSING — running all scripts")
    print(f"  DB:  {db_arg}")
    print(f"  OUT: {out_arg}")
    print("=" * 70)

    passed  = 0
    failed  = 0
    t_total = time.time()

    for script_name, description, needs_db in SCRIPTS:
        script_path = _SCRIPT_DIR / script_name
        if not script_path.exists():
            print(f"\n⚠  {script_name} not found — skipping")
            failed += 1
            continue

        print(f"\n{'─' * 70}")
        print(f"▶  {description}")
        print(f"   {script_name}")
        print(f"{'─' * 70}")

        cmd = [python, str(script_path), "--out", out_arg]
        if needs_db:
            cmd += ["--db", db_arg]

        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(_SCRIPT_DIR.parent),   # run from project root
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"✓  Done in {elapsed:.1f}s")
            passed += 1
        else:
            print(f"✗  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            failed += 1

    total_time = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  FINISHED  —  {passed} passed, {failed} failed  "
          f"({total_time:.1f}s total)")
    print(f"  Results in: {out_arg}")
    print(f"{'=' * 70}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
