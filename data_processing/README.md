# Data Processing Scripts

Offline analysis scripts. All paths resolve relative to the project root automatically, so scripts can be run from any working directory.

```bash
# From project root
python data_processing/cross_similarity.py
python data_processing/enrich_spreadsheet.py
python data_processing/rank_correlation.py
```

## cross_similarity.py

Computes pairwise cosine similarity between adversarial images across two pattern groups (A↔B) for each identity. Reads embeddings and attack records from the SQLite database, prints colour-coded terminal tables (mean/min/max similarity and accuracy per model), and exports results.

**Inputs:** `db.sqlite` (embeddings + attacks tables)

**Outputs** (written to `data_processing/`):

| File | Description |
|---|---|
| `across_results.csv` | Per-pair similarity scores with match verdicts |
| `across_summary.txt` | Plain-text copy of the terminal summary tables |

Hardcoded settings at the bottom of the file control the comparison mode (`across`), output filenames, and per-identity pattern splits.

## enrich_spreadsheet.py

Merges per-identity log CSV files (from `data/logs/`) into a single spreadsheet alongside hardcoded image metadata (pose flags, detection status). Adds one `<Model>_score` and one `<Model>_match` column per model found in the logs.

```bash
python data_processing/enrich_spreadsheet.py [--logs <dir>] [--output <file>]
```

| Argument | Default | Description |
|---|---|---|
| `--logs` | `data/logs` (project root) | Directory with per-identity log CSV files |
| `--output` | `data_processing/enriched.xlsx` | Output path; `.xlsx` or `.csv` |

## rank_correlation.py

Computes Spearman rank correlation and mean absolute rank error between three ranking columns (A, B, C) over 30 identities. All data is hardcoded — no external files or database access required.

**Output:** printed to terminal only.
