# data_processing/

Scripts for extracting, analysing, and exporting experimental results from the
SQLite database.  All scripts share a common configuration module (`config.py`)
and write their output to `data_processing/results/` by default.

## Quick start

```bash
# Run everything at once from the project root:
python data_processing/run_all.py

# Or with explicit paths:
python data_processing/run_all.py --db path/to/db.sqlite --out path/to/output/
```

## Scripts

| Script | Description | Output file(s) |
|--------|-------------|----------------|
| `run_all.py` | **Master script** — runs every sub-script below in sequence | — |
| `extract_rankings.py` | Export the Current Standings and Gallery Search ranking tables | `current_standings.csv`, `gallery_search.csv` |
| `model_accuracy.py` | Per-model recognition accuracy, broken down by overall / pose / pattern | `model_accuracy.csv` |
| `score_statistics.py` | Model-independent score statistics (mean, std, quartiles) per pose and pattern | `score_statistics.csv` |
| `cross_pattern_similarity.py` | Adversarial ↔ adversarial cosine similarity across pattern groups (A ↔ B) | `cross_pattern_pairs.csv`, `cross_pattern_summary.csv` |

## Shared configuration

`config.py` contains:

- **Per-model LFW decision thresholds** for the 5 active models
- **Pattern split map** — how many images belong to pattern A vs B per identity
- **Pose label helpers** — mapping `is_frontal` / `is_profile` / `is_pitch` flags
- **DB access helpers** — lightweight SQLite functions that avoid importing PyTorch

## Output files

All CSV files are written to `data_processing/results/` (created automatically).

### current_standings.csv
One row per identity (attacker), ranked by best average similarity score.
Includes per-model scores for the identity's best image.

### gallery_search.csv
One row per identity with gallery-search position statistics (best / worst /
average position, image count).

### model_accuracy.csv
One row per `(model, group_by, group_value)` combination.  `group_by` is one
of `overall`, `pose`, or `pattern`.  Includes count, accuracy, mean, std, min,
max, and median similarity scores.

### score_statistics.csv
Model-independent statistics.  Each image's score is first averaged across all
models, then aggregated by pose and pattern.  Includes mean, std, quartiles,
and attack success rate.

### cross_pattern_pairs.csv
Raw pairwise results: one row per `(identity, model, image_a, image_b)`
comparison between a pattern-A and a pattern-B adversarial image.

### cross_pattern_summary.csv
Aggregated cross-pattern statistics per `(identity, model)`, plus per-model
totals and a grand total row.

## CLI options

Every script accepts the same two flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `db.sqlite` | Path to the SQLite database |
| `--out` | `data_processing/results/` | Output directory for CSV files |

## Dependencies

- Python 3.10+
- `numpy` (used by all scripts for statistics and embedding operations)
- The project's `core/database.py` (used only by `extract_rankings.py`)
