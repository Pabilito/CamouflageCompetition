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

`run_all.py` executes all sub-scripts in order and summarises pass/fail counts.
Scripts that do not require the database (e.g. `visualize_results.py`) are
invoked with `--out` only; the rest receive both `--db` and `--out`.

> **Note:** `gallery_search_stats.py` recomputes gallery positions directly
> from the embeddings table at analysis time — no prior UI step is required.
> The "All Models (avg sim)" row replicates the averaged-similarity ranking
> used in the Gallery Search tab of the UI.

## Scripts

| Script | Description | Needs DB | Output file(s) |
|--------|-------------|:--------:|----------------|
| `run_all.py` | **Master script** — runs every sub-script below in sequence | — | — |
| `extract_rankings.py` | Export the Current Standings and Gallery Search ranking tables | ✓ | `current_standings.csv`, `gallery_search.csv` |
| `model_accuracy.py` | Per-model recognition accuracy, broken down by overall / pose / pattern | ✓ | `model_accuracy.csv` |
| `score_statistics.py` | Model-independent score statistics (mean, std, quartiles) per pose and pattern | ✓ | `score_statistics.csv` |
| `cross_pattern_similarity.py` | Adversarial ↔ adversarial cosine similarity across pattern groups (A ↔ B) | ✓ | `cross_pattern_pairs.csv`, `cross_pattern_summary.csv`, `cross_pattern_by_pair.csv` |
| `gallery_search_stats.py` | Gallery search Rank-N accuracy, MRR, and position statistics per model. Produces two "All Models" rows: per-model pooled average and averaged-similarity method (as used in the UI). All rate values on 0–1 scale; heatmap cells show value + pos/total counts. | ✓ | `gallery_search_stats.csv` + 3 PNGs |
| `score_distributions.py` | Similarity score distribution plots (mated vs non-mated, pose breakdown, cross-pattern, combined) + biometric statistics (EER, FMR100, FMR1000, TPR, AUC) saved per model under `results/<model_name>/`; combined overview and biometric summary saved to `results/` | ✓ | 4 PNGs + `biometric_stats.csv` + `roc_*.csv` per model; `biometric_summary.png`, `all_models_overview.png` |
| `visualize_results.py` | Heatmap and table figures for accuracy, score statistics, and cross-pattern results | — | 6 PNGs |

## Shared configuration

`config.py` contains:

- **Per-model LFW decision thresholds** for the 5 active models
- **Pattern split map** — how many images belong to pattern A vs B per identity
- **Pose label helpers** — mapping `is_frontal` / `is_profile` / `is_pitch` flags
- **DB access helpers** — lightweight SQLite functions that avoid importing PyTorch

## Output files

All CSV and PNG files are written to `data_processing/results/` (created automatically).

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
and per-model attack success rate.

### cross_pattern_pairs.csv
Raw pairwise results: one row per `(identity, model, image_a, image_b)`
comparison between a pattern-A and a pattern-B adversarial image.

### cross_pattern_summary.csv
Aggregated cross-pattern statistics per `(identity, model)`, plus per-model
totals and a grand total row.

### cross_pattern_by_pair.csv
Aggregated statistics per `(pattern_pair, model)`, e.g. `1↔2`, `1↔3`.
Includes per-pattern-pair totals and a grand total row.

### gallery_search_stats.csv
One row per `(model, group_by, group_value)`.  Per-model positions are
recomputed at analysis time from the embeddings table: for each
`(adversarial image, model)` pair, every gallery identity is scored by max
cosine similarity over its control images; identities are ranked descending
and the true identity's 1-based rank is recorded.

Columns: mean / std / median / min / max gallery position, MRR, attack
success rate, and Rank-N accuracy for N = 1 … 10.
An `ALL_MODELS` aggregate row pools positions across all five models.

**MRR** = mean(1 / gallery_position).  Lower MRR = better attack (true
identity ranks further down the gallery).

**All rate columns** (Rank-N, MRR) are stored and displayed on a 0–1 scale.

### Figure files

`visualize_results.py` produces three pairs of heatmap + table PNGs:

| File | Contents |
|------|----------|
| `fig_overall_accuracy_heatmap.png` | Model recognition accuracy by group (colour-coded) |
| `fig_overall_accuracy_table.png` | Same data as a clean table (values 0–1) |
| `fig_score_statistics_heatmap.png` | Attack success rate per model × group (colour-coded) |
| `fig_score_statistics_table.png` | Same data as a clean table (values 0–1) |
| `fig_cross_pattern_heatmap.png` | Cross-pattern match rate per model (colour-coded) |
| `fig_cross_pattern_table.png` | Same data as a clean table (values 0–1) |

`gallery_search_stats.py` produces three files:

| File | Contents |
|------|----------|
| `fig_gs_rank_n_heatmap.png` | Rank ≤ N accuracy per model, N = 1 … 10 (colour-coded; cells show value + k/n counts) |
| `fig_gs_rank_n_table.png` | Same data as a clean table (values 0–1) |
| `fig_gs_positions_table.png` | Mean / median position + MRR + Rank-1/5/10 per model (table, 0–1); includes both "All Models" rows: per-model pool and averaged-similarity (UI method) |

`score_distributions.py` produces the following output.

**Per-model files** saved to `results/<model_name>/`:

| File | Contents |
|------|----------|
| `plot1_mated_vs_nonmated.png` | Mated (ctrl × same-identity adversarial) vs equal-count random non-mated distributions; EER / FMR100 / FMR1000 threshold lines with Acc and TPR annotated in the legend |
| `plot2_pose_breakdown.png` | Four distributions: mated frontal, mated profile, non-mated frontal, non-mated profile; colour-coded by pose (blue = frontal, red = profile), solid = mated, dotted = non-mated |
| `plot3_cross_pattern.png` | All pairwise adversarial pattern combinations (1↔2, 1↔3, 2↔3 …); each subplot compares adv-pA vs adv-pB mated (same identity) and an equal-count non-mated baseline; threshold lines annotated |
| `plot4_combined.png` | Overlay of ctrl↔adv distributions (Plot 1) and all cross-pattern adv↔adv distributions (Plot 3); lines only, no fill |
| `biometric_stats.csv` | One row per experiment (`ctrl_vs_adversarial`, `P1_vs_P2`, …, `all_mated_vs_nonmated`): EER, Acc@EER, FNMR@FMR100, TPR@FMR100, FNMR@FMR1000, TPR@FMR1000, AUC, EER threshold |
| `roc_<tag>.csv` | FMR / FNMR columns for each experiment (ROC / DET curve data), exported by pyeer |

**Global summary files** saved to `results/`:

| File | Contents |
|------|----------|
| `biometric_summary.png` | One row per model: grouped bar chart (EER / FNMR@FMR100 / FNMR@FMR1000), ROC curves with EER operating point marked, accuracy table (Acc@EER, TPR@FMR100, TPR@FMR1000, AUC) |
| `all_models_overview.png` | Combined grid (models × plots): Plot 1 \| Plot 2 \| one column per cross-pattern combo \| Plot 4; all axes share a unified y-axis scale per row for direct cross-model comparison |

All plots use KDE smoothing (requires `scipy`; step-histogram fallback if absent). Non-mated pairs are sampled with a fixed seed (`--seed`) for reproducibility. Biometric statistics require `pyeer` (`pip install pyeer`).

**Colour convention (heatmaps):** green = model still recognises the attacker
(worse attack); red = model fooled (better attack).  Rank-N accuracy uses an
inverted scale because lower recognition rate is better for the attacker.

## CLI options

Most scripts accept the same two flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `db.sqlite` | Path to the SQLite database |
| `--out` | `data_processing/results/` | Output directory for CSV / PNG files |

`visualize_results.py` accepts `--out` only (it reads the CSVs written by the
other scripts, not the database directly).

## Dependencies

- Python 3.10+
- `numpy` — statistics and embedding operations (all scripts)
- `matplotlib` — figure generation (`gallery_search_stats.py`, `visualize_results.py`, `score_distributions.py`)
- `scipy` — KDE smoothing in `score_distributions.py` (optional; step-histogram fallback if absent)
- `pyeer` — biometric statistics in `score_distributions.py` (optional; `pip install pyeer`)
- `core/database.py` — used only by `extract_rankings.py`
