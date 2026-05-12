"""
enrich_spreadsheet.py
---------------------
Main spreadsheet data is hardcoded in MAIN_DATA below.
Loads per-identity log files (data/logs/1.csv, 2.csv, ...) and appends one column
pair per unique model found in the logs:
  <ModelName>_score   - the score value (float)
  <ModelName>_match   - the match result (True / False)

Usage (from project root or from data_processing/):
    python data_processing/enrich_spreadsheet.py --logs data/logs --output enriched.xlsx

Arguments:
    --logs    Directory containing per-identity log CSV files (default: ../data/logs)
    --output  Output file path; .xlsx or .csv (default: ../enriched.xlsx)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Hardcoded main spreadsheet data
# Columns: Identity, Picture, Front, Horizontal Rotation, Vertical Rotation, Detected, Valid
# ---------------------------------------------------------------------------
MAIN_DATA = [
    (1,  "000", 1, 0, 0, 1, 1),
    (1,  "001", 0, 1, 0, 1, 1),
    (1,  "002", 0, 1, 0, 1, 1),
    (1,  "003", 1, 0, 0, 1, 1),
    (1,  "004", 0, 0, 1, 1, 1),
    (1,  "005", 0, 1, 0, 1, 1),
    (2,  "000", 1, 0, 0, 1, 1),
    (2,  "001", 1, 0, 0, 1, 1),
    (2,  "002", 0, 1, 0, 1, 1),
    (2,  "003", 1, 0, 0, 1, 1),
    (2,  "004", 0, 1, 0, 1, 1),
    (2,  "005", 0, 1, 0, 1, 1),
    (3,  "000", 1, 0, 0, 1, 1),
    (3,  "001", 0, 1, 0, 1, 1),
    (3,  "002", 0, 1, 0, 1, 1),
    (3,  "003", 1, 0, 0, 1, 1),
    (3,  "004", 0, 1, 0, 1, 1),
    (3,  "005", 0, 1, 0, 1, 1),
    (4,  "000", 1, 0, 0, 0, 1),
    (4,  "001", 0, 1, 0, 0, 1),
    (4,  "002", 0, 1, 0, 0, 1),
    (4,  "003", 1, 0, 0, 1, 1),
    (4,  "004", 1, 0, 0, 1, 1),
    (5,  "000", 1, 0, 0, 1, 1),
    (5,  "001", 0, 1, 0, 1, 1),
    (5,  "002", 0, 1, 0, 1, 1),
    (5,  "003", 1, 0, 0, 1, 1),
    (5,  "004", 0, 1, 0, 1, 1),
    (5,  "005", 0, 1, 0, 1, 1),
    (6,  "000", 1, 0, 0, 1, 1),
    (6,  "001", 0, 1, 0, 1, 1),
    (6,  "002", 0, 1, 0, 1, 1),
    (6,  "003", 1, 0, 0, 1, 1),
    (6,  "004", 0, 1, 0, 1, 1),
    (6,  "005", 0, 1, 0, 1, 1),
    (7,  "000", 1, 0, 0, 1, 1),
    (7,  "001", 0, 1, 0, 1, 1),
    (7,  "002", 0, 1, 0, 1, 1),
    (7,  "003", 1, 0, 0, 1, 1),
    (7,  "004", 1, 0, 0, 1, 1),
    (7,  "005", 0, 1, 0, 1, 1),
    (8,  "000", 1, 0, 0, 1, 1),
    (8,  "001", 0, 1, 0, 1, 1),
    (8,  "002", 0, 0, 1, 1, 1),
    (8,  "003", 1, 0, 0, 1, 1),
    (8,  "004", 0, 1, 0, 1, 1),
    (8,  "005", 0, 1, 1, 1, 1),
    (9,  "000", 1, 0, 0, 1, 1),
    (9,  "001", 1, 0, 0, 1, 1),
    (9,  "002", 1, 0, 0, 1, 1),
    (9,  "003", 1, 0, 0, 1, 1),
    (9,  "004", 0, 1, 0, 1, 1),
    (9,  "005", 0, 1, 0, 1, 1),
    (10, "000", 1, 0, 0, 1, 1),
    (10, "001", 0, 1, 0, 1, 1),
    (10, "002", 0, 1, 0, 1, 1),
    (10, "003", 1, 0, 0, 1, 1),
    (10, "004", 0, 1, 0, 1, 1),
    (10, "005", 0, 0, 1, 1, 1),
    (11, "000", 1, 0, 0, 1, 1),
    (11, "001", 1, 0, 0, 1, 1),
    (11, "002", 0, 1, 0, 1, 1),
    (11, "003", 1, 0, 0, 1, 0),
    (11, "004", 0, 1, 0, 1, 0),
    (11, "005", 0, 1, 0, 1, 0),
    (12, "000", 1, 0, 0, 1, 1),
    (12, "001", 1, 0, 0, 1, 1),
    (12, "002", 1, 0, 0, 1, 1),
    (12, "003", 0, 1, 0, 1, 1),
    (12, "004", 0, 1, 0, 1, 1),
    (13, "000", 1, 0, 0, 1, 1),
    (13, "001", 0, 1, 0, 1, 1),
    (13, "002", 0, 1, 0, 1, 1),
    (13, "003", 1, 0, 0, 1, 1),
    (13, "004", 0, 1, 0, 1, 1),
    (13, "005", 0, 1, 0, 1, 1),
    (14, "000", 1, 0, 0, 1, 1),
    (14, "001", 1, 0, 0, 1, 1),
    (14, "002", 0, 1, 0, 1, 1),
    (14, "003", 1, 0, 0, 1, 1),
    (14, "004", 0, 1, 0, 1, 1),
    (14, "005", 0, 1, 0, 1, 1),
    (15, "000", 1, 0, 0, 1, 1),
    (15, "001", 0, 1, 0, 1, 1),
    (15, "002", 0, 0, 1, 1, 1),
    (15, "003", 1, 0, 0, 1, 1),
    (15, "004", 0, 1, 0, 1, 1),
    (15, "005", 0, 0, 1, 1, 1),
    (16, "000", 1, 0, 0, 1, 1),
    (16, "001", 0, 1, 0, 1, 1),
    (16, "002", 0, 1, 0, 1, 1),
    (16, "003", 0, 1, 0, 1, 1),
    (16, "004", 1, 0, 0, 1, 1),
    (16, "005", 0, 1, 0, 1, 1),
    (16, "006", 0, 1, 0, 1, 1),
    (17, "000", 1, 0, 0, 1, 1),
    (17, "001", 0, 1, 0, 1, 1),
    (17, "002", 0, 1, 0, 1, 1),
    (17, "003", 1, 0, 0, 1, 0),
    (17, "004", 0, 1, 0, 1, 0),
    (17, "005", 0, 1, 0, 1, 0),
    (18, "000", 1, 0, 0, 1, 1),
    (18, "001", 0, 1, 0, 1, 1),
    (18, "002", 0, 1, 0, 1, 1),
    (18, "003", 1, 0, 0, 1, 1),
    (18, "004", 0, 1, 0, 1, 1),
    (18, "005", 0, 1, 0, 1, 1),
    (19, "000", 1, 0, 0, 1, 1),
    (19, "001", 0, 1, 0, 1, 1),
    (19, "002", 0, 1, 0, 1, 1),
    (19, "003", 1, 0, 0, 1, 0),
    (19, "004", 0, 1, 0, 1, 0),
    (19, "005", 0, 1, 0, 1, 0),
    (20, "000", 1, 0, 0, 1, 1),
    (20, "001", 0, 1, 0, 1, 1),
    (20, "002", 0, 1, 0, 1, 1),
    (20, "003", 1, 0, 0, 1, 1),
    (20, "004", 0, 0, 1, 1, 1),
    (21, "000", 1, 0, 0, 1, 1),
    (21, "001", 0, 1, 0, 1, 1),
    (21, "002", 0, 0, 1, 1, 1),
    (21, "003", 1, 0, 0, 1, 0),
    (21, "004", 0, 1, 0, 1, 0),
    (21, "005", 0, 1, 1, 1, 0),
    (22, "000", 1, 0, 0, 1, 1),
    (22, "001", 0, 1, 0, 1, 1),
    (22, "002", 0, 1, 0, 1, 1),
    (22, "003", 1, 0, 0, 1, 1),
    (22, "004", 0, 1, 0, 1, 1),
    (22, "005", 0, 1, 0, 1, 1),
    (23, "000", 1, 0, 0, 1, 1),
    (23, "001", 0, 1, 0, 1, 1),
    (23, "002", 0, 1, 0, 1, 1),
    (23, "003", 1, 0, 0, 1, 1),
    (23, "004", 0, 1, 0, 1, 1),
    (23, "005", 0, 1, 0, 1, 1),
    (24, "000", 1, 0, 0, 1, 1),
    (24, "001", 0, 0, 1, 1, 1),
    (24, "002", 0, 1, 0, 1, 1),
    (24, "003", 1, 0, 0, 1, 1),
    (24, "004", 0, 1, 0, 1, 1),
    (24, "005", 0, 1, 0, 1, 1),
    (25, "000", 1, 0, 0, 1, 1),
    (25, "001", 0, 1, 0, 1, 1),
    (25, "002", 0, 1, 0, 1, 1),
    (25, "003", 1, 0, 0, 1, 1),
    (25, "004", 0, 1, 0, 1, 1),
    (25, "005", 0, 1, 0, 1, 1),
    (26, "000", 1, 0, 0, 1, 1),
    (26, "001", 0, 1, 0, 1, 1),
    (26, "002", 0, 1, 0, 1, 1),
    (26, "003", 1, 0, 0, 1, 1),
    (26, "004", 0, 1, 0, 1, 1),
    (26, "005", 0, 1, 0, 0, 1),
    (27, "000", 1, 0, 0, 1, 1),
    (27, "001", 0, 1, 0, 1, 1),
    (27, "002", 0, 0, 1, 1, 1),
    (27, "003", 1, 0, 0, 1, 0),
    (27, "004", 0, 1, 0, 1, 0),
    (27, "005", 0, 0, 1, 1, 0),
    (28, "000", 1, 0, 0, 1, 1),
    (28, "001", 0, 1, 0, 1, 1),
    (28, "002", 0, 1, 0, 1, 1),
    (28, "003", 1, 0, 0, 1, 1),
    (28, "004", 0, 1, 0, 1, 1),
    (28, "005", 0, 1, 0, 1, 1),
    (29, "000", 1, 0, 0, 1, 1),
    (29, "001", 0, 1, 0, 1, 1),
    (29, "002", 0, 1, 0, 1, 1),
    (29, "003", 1, 0, 0, 1, 1),
    (29, "004", 0, 1, 0, 1, 1),
    (29, "005", 0, 1, 0, 1, 1),
    (30, "000", 1, 0, 0, 0, 1),
    (30, "001", 0, 1, 0, 0, 1),
    (30, "002", 0, 1, 0, 0, 1),
    (30, "003", 1, 0, 0, 0, 1),
    (30, "004", 1, 0, 0, 0, 1),
    (30, "005", 0, 1, 0, 0, 1),
]

MAIN_COLUMNS = [
    "Identity", "Picture", "Front",
    "Horizontal Rotation", "Vertical Rotation",
    "Detected", "Valid"
]


def load_logs(log_dir):
    """Load all per-identity log CSV files from log_dir into a single DataFrame."""
    frames = []
    for fname in os.listdir(log_dir):
        if fname.endswith(".csv"):
            path = os.path.join(log_dir, fname)
            try:
                frames.append(pd.read_csv(path))
            except Exception as e:
                print(f"  Warning: could not read {path}: {e}", file=sys.stderr)

    if not frames:
        sys.exit(f"No CSV log files found in '{log_dir}'.")

    logs = pd.concat(frames, ignore_index=True)
    logs.columns = logs.columns.str.strip()

    required = {"identity", "frame_path", "model", "score", "match"}
    missing = required - set(logs.columns)
    if missing:
        sys.exit(f"Log files are missing required columns: {missing}")

    # Extract picture number from frame_path
    # e.g. data\adversarial\4\img_001.jpg  ->  "001"
    logs["pic_num"] = (
        logs["frame_path"]
        .str.replace("\\", "/", regex=False)
        .str.extract(r"img_(\d+)\.jpg", expand=False)
    )

    logs["identity"] = pd.to_numeric(logs["identity"], errors="coerce").astype("Int64")
    return logs


def enrich(log_dir, output_path):
    # Build main DataFrame from hardcoded data
    main = pd.DataFrame(MAIN_DATA, columns=MAIN_COLUMNS)
    main["_identity"] = main["Identity"].astype("Int64")
    main["_pic_num"]  = main["Picture"].str.zfill(3)

    # Load logs
    logs = load_logs(log_dir)
    logs["pic_num"] = logs["pic_num"].str.strip().str.zfill(3)

    models = sorted(logs["model"].dropna().unique())
    print(f"Models found: {models}")

    # Pivot: one row per (identity, pic_num, model) -> score + match
    logs_dedup = logs.drop_duplicates(subset=["identity", "pic_num", "model"])
    pivot = logs_dedup.pivot(
        index=["identity", "pic_num"],
        columns="model",
        values=["score", "match"]
    )
    # Flatten column names: (score, ModelA) -> ModelA_score
    pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    pivot.rename(columns={"identity": "_identity", "pic_num": "_pic_num"}, inplace=True)

    # Merge
    # outer join: keeps log rows even if not in hardcoded data
    enriched = main.merge(pivot, on=["_identity", "_pic_num"], how="outer")
    enriched.drop(columns=["_identity", "_pic_num"], inplace=True)

    # Save
    if output_path.endswith(".xlsx"):
        enriched.to_excel(output_path, index=False)
    else:
        enriched.to_csv(output_path, index=False)

    added_cols = [c for c in enriched.columns if any(m in c for m in models)]
    print(f"Done. {len(enriched)} rows written to '{output_path}'.")
    print(f"New columns added: {added_cols}")


def main():
    parser = argparse.ArgumentParser(description="Enrich hardcoded spreadsheet with log model scores.")
    parser.add_argument("--logs",   default=str(_PROJECT_ROOT / "data" / "logs"), help="Directory with per-identity log CSV files")
    parser.add_argument("--output", default=str(_SCRIPT_DIR / "enriched.xlsx"), help="Output file path (.xlsx or .csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.logs):
        sys.exit(f"Logs directory not found: '{args.logs}'")

    enrich(args.logs, args.output)


if __name__ == "__main__":
    main()
