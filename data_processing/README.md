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
| `gallery_search_stats.py` | Gallery search Rank-N accuracy, MRR, and position statistics per model and per pattern/cross-pattern pair. Produces two "All Models" rows: per-model pooled average and averaged-similarity method (as used in the UI). All rate values on 0–1 scale. | ✓ | `gallery_search_stats.csv` + 9 PNGs |
| `score_distributions.py` | Similarity score distribution plots (mated vs non-mated, pose breakdown, cross-pattern, combined) + biometric statistics (EER, FMR100, FMR1000, TPR, AUC) + optimal-accuracy decision threshold tables, all saved per model and in combined overviews | ✓ | per-model PNGs + CSVs; global summary PNGs; per-pattern overview PNGs; threshold table PNGs |
| `analyze_color_vs_rank.py` | Ranks adversarial images by mean similarity (hardest-to-recognise = rank 1), computes face-region perceptual colour metrics relative to the identity's first control image, and reports Pearson/Spearman correlations between those metrics and ranking position | ✓ | `color_vs_rank.csv`, `color_vs_rank.png` |
| `analyze_patchiness_vs_rank.py` | Same ranking as `analyze_color_vs_rank.py`; measures face-region *patchiness* (frequency of colour transitions) using three metric families — edge-density, local-variance/texture-energy, and region-count/segmentation — and reports Pearson/Spearman correlations between each metric and rank. Delta values relative to the identity's first control image are included for the scalar texture metrics. | ✓ | `patchiness_vs_rank.csv`, `patchiness_vs_rank.png`, `patchiness_vs_rank_correlations.png` |
| `avg_similarity_by_identity.py` | Per-pattern scatter plots of average similarity score (across all models) per adversarial image, with identities sorted by a fixed numeric scheme and vertical cut-off lines marking participant groups. Also produces a combined all-patterns overlay and a grouped boxplot comparing score distributions across participant groups and patterns. | ✓ | `fig_avg_similarity_pat{N}.png`, `fig_avg_similarity_all_patterns.png`, `fig_boxplot_by_group.png` |
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

### color_vs_rank.csv

One row per adversarial image, ordered by average similarity rank
(rank 1 = lowest mean similarity = hardest to recognise).

For each image, the script computes face-region colour statistics from the
aligned 112×112 crop using the same elliptical face mask used elsewhere in
the project. Metrics include:

- Mean CIE Lab components (`L`, `a`, `b`)
- Chroma (`C`) and hue angle (`h_deg`)
- Mean HSV values (`H`, `S`, `V`)
- Mean RGB values (`R`, `G`, `B`)
- Lightness variation across the face (`L_std`)
- Deltas relative to the identity's first control image:
  `dL`, `da`, `db`, `dC`, `dE`, `dh`, `dS`, `dV`, `dR`, `dG`, `dB`

Also includes the adversarial image rank, mean similarity score, model count,
identity, image path, and reference-control Lab values.

The script prints Pearson (`r`) and Spearman (`ρ`) correlations between each
metric and rank. Spearman correlation is generally the more meaningful measure
because ranking itself is ordinal.

### patchiness_vs_rank.csv

One row per adversarial image, ordered by average similarity rank
(rank 1 = lowest mean similarity = hardest to recognise).

Patchiness metrics computed from the aligned 112×112 crop using the same
elliptical face mask as `analyze_color_vs_rank.py`:

- **Edge-density:** `edge_density` (fraction of face pixels above gradient threshold), `grad_mean` (mean Lab gradient magnitude, ΔE/px), `grad_p90` (90th-percentile gradient)
- **Local-variance / texture energy:** `local_var` (mean local L\* variance in 3×3 window), `lab_energy` (mean local Lab variance in 3×3 window), `laplacian_var` (variance of the Laplacian of L\*)
- **Region-count / segmentation:** `n_regions` (connected colour regions after coarse Lab quantisation), `region_density` (regions per 1 000 face pixels), `color_entropy` (Shannon entropy of quantised-colour histogram, bits)
- **Deltas vs first control image:** `d_edge_density`, `d_grad_mean`, `d_lab_energy`, `d_color_entropy`

Also includes rank, mean similarity score, model count, identity, and image path.

### avg_similarity output files

`fig_avg_similarity_pat{N}.png` — one file per pattern. Each plot shows one dot per adversarial image (x = identity sorted by numeric position, y = average similarity across all models), with frontal/profile/unknown poses colour-coded, per-identity mean lines, group cut-off markers, and the average model threshold.

`fig_avg_similarity_all_patterns.png` — all patterns overlaid in a single figure for cross-pattern comparison.

`fig_boxplot_by_group.png` — grouped boxplots with participant groups on the x-axis and one box per pattern per group, showing the distribution of per-image average similarity scores.

---

## Figure files

### analyze_color_vs_rank.py — 1 PNG

| File | Contents |
|------|----------|
| `color_vs_rank.png` | Scatter plots of adversarial rank versus each colour metric (`L`, `a`, `b`, `dL`, `da`, `db`, `C`, `dC`, `dE`, `S`, `dS`, `L_std`) with least-squares trend lines and Pearson/Spearman correlation annotations |

---

### analyze_patchiness_vs_rank.py — 2 PNGs

| File | Contents |
|------|----------|
| `patchiness_vs_rank.png` | 3-column scatter-plot grid (one panel per metric) of adversarial rank versus each patchiness metric with least-squares trend lines and Pearson/Spearman annotations |
| `patchiness_vs_rank_correlations.png` | Summary table of Pearson `r` and Spearman `ρ` for every metric, sorted by `|ρ|` |

---

### avg_similarity_by_identity.py — 2+ PNGs

| File | Contents |
|------|----------|
| `fig_avg_similarity_pat{N}.png` | Per-pattern scatter plot: one dot per adversarial image, x = identity (sorted by numeric position), y = average similarity across all models. Frontal/profile/unknown poses are colour-coded; per-identity mean lines, group cut-off markers, and average model threshold are overlaid. |
| `fig_avg_similarity_all_patterns.png` | All patterns overlaid in a single figure for cross-pattern comparison, using the same layout and markers as the per-pattern plots |
| `fig_boxplot_by_group.png` | Grouped boxplots comparing score distributions across participant groups (x-axis) and patterns (one box per pattern per group) |

---

### visualize_results.py — 6 PNGs

| File | Contents |
|------|----------|
| `fig_overall_accuracy_heatmap.png` | Model recognition accuracy by group (colour-coded) |
| `fig_overall_accuracy_table.png` | Same data as a clean table (values 0–1) |
| `fig_score_statistics_heatmap.png` | Attack success rate per model × group (colour-coded) |
| `fig_score_statistics_table.png` | Same data as a clean table (values 0–1) |
| `fig_cross_pattern_heatmap.png` | Cross-pattern match rate per model (colour-coded) |
| `fig_cross_pattern_table.png` | Same data as a clean table (values 0–1) |

---

### gallery_search_stats.py — 9 PNGs

**Overall figures** saved to `results/`:

| File | Contents |
|------|----------|
| `fig_gs_rank_n_heatmap.png` | Rank ≤ N accuracy per model, N = 1 … 10 (colour-coded; cells show value + k/n counts) |
| `fig_gs_rank_n_table.png` | Same data as a clean dark-background table (values 0–1) |
| `fig_gs_positions_table.png` | Mean / median position + MRR + Rank-1/5/10 per model; includes both "All Models" rows: per-model pool and averaged-similarity (UI method) |

**Per-pattern & cross-pattern figures** saved to `results/`:

| File | Contents |
|------|----------|
| `fig_gs_pattern_positions_table.png` | One sub-table per model side by side. Rows: Section 1 — P1, P2, … (adversarial queries against the **control gallery**); Section 2 — `P{Q} query → P{G} gallery` pairs where both query and gallery are adversarial images (pattern G used as gallery, pattern Q as query). Gallery uses every identity that has a pattern-G adversarial image; query images are pattern-Q adversarials whose identity is also present in the gallery. Columns per sub-table: Mean Pos, Median Pos, MRR, Rank ≤ 1/5/10. Caption shows gallery identity count and dataset image count. |
| `fig_gs_pattern_metric_mean_pos.png` | Mean gallery position for every pattern row × all models in one table |
| `fig_gs_pattern_metric_median_pos.png` | Median gallery position — same layout |
| `fig_gs_pattern_metric_mrr.png` | MRR — same layout |
| `fig_gs_pattern_metric_rank1.png` | Rank ≤ 1 accuracy — same layout |
| `fig_gs_pattern_metric_rank5.png` | Rank ≤ 5 accuracy — same layout |
| `fig_gs_pattern_metric_rank10.png` | Rank ≤ 10 accuracy — same layout |

**Section 1 rows** (`P1`, `P2`, …): standard gallery search — adversarial image queried against the **control (capture) gallery**.  Lower Mean Pos / higher Rank-N = model still recognises the attacker.

**Section 2 rows** (`P{Q} query → P{G} gallery`): adversarial gallery search — adversarial images of pattern G replace the control images as the gallery; query images are adversarial images of pattern Q.  The gallery contains every identity that has a pattern-G adversarial image (so gallery size is determined solely by G, not by the query pattern).  Query images are restricted to identities that also exist in the gallery — unrankable queries are dropped rather than counted as misses.  Each unordered pair `(pA, pB)` produces two rows (one per query direction) so you can directly compare how well the model matches cross-pattern adversarials in both directions; the two directions can have different gallery sizes when one pattern has more identities than the other.  A low position means the model finds the pattern-G adversarial similar to the pattern-Q adversarial of the same identity.  If no cross-pattern records are available (e.g. only one pattern in the data), Section 2 is omitted entirely with a console warning.

**`All (avg sim)` column** (gold, dashed separator): cosine similarities averaged across all models per gallery identity before ranking — replicates the UI Gallery Search method.

---

### score_distributions.py — per-model files + global summaries + threshold tables

**Per-model files** saved to `results/<model_name>/`:

| File | Contents |
|------|----------|
| `plot1_mated_vs_nonmated.png` | Mated (ctrl × same-identity adversarial) vs equal-count random non-mated distributions; EER / FMR100 / FMR1000 threshold lines with Acc and TPR annotated |
| `plot1_mated_vs_nonmated_clean.png` | Same without sample counts or threshold lines; legend inside |
| `plot1_mated_vs_nonmated_pat{N}_clean.png` | Plot 1 restricted to adversarials of pattern N |
| `plot2_pose_breakdown.png` | Four distributions: mated frontal, mated profile, non-mated frontal, non-mated profile |
| `plot2_pose_breakdown_clean.png` | Clean variant (legend inside, no threshold lines) |
| `plot2_pose_breakdown_pat{N}_clean.png` | Plot 2 restricted to pattern N |
| `plot3_cross_pattern.png` | All pairwise adversarial pattern combinations (1↔2, 1↔3, 2↔3 …); each subplot: adv-pA vs adv-pB mated (same identity) + equal-count non-mated; threshold lines annotated |
| `plot3_cross_pattern_clean.png` | Clean variant |
| `plot3_cross_pattern_pat{N}_clean.png` | Cross-pattern subplots anchored to pattern N (N vs every other pattern) |
| `plot4_combined.png` | Overlay of ctrl↔adv distributions (Plot 1) and all cross-pattern adv↔adv distributions (Plot 3); lines only, no fill |
| `plot4_combined_clean.png` | Clean variant |
| `plot4_combined_pat{N}_clean.png` | Combined plot restricted to pattern N |
| `biometric_stats.csv` | One row per experiment (`ctrl_vs_adversarial`, `P1_vs_P2`, …, `all_mated_vs_nonmated`): EER, Acc@EER, FNMR@FMR100, TPR@FMR100, FNMR@FMR1000, TPR@FMR1000, AUC, EER threshold |
| `roc_<tag>.csv` | FMR / FNMR columns for each experiment (ROC / DET curve data) |

**Global summary files** saved to `results/`:

| File | Contents |
|------|----------|
| `biometric_summary.png` | One row per model: grouped bar chart (EER / FNMR@FMR100 / FNMR@FMR1000), ROC curves with EER operating point marked, accuracy table (Acc@EER, TPR@FMR100, TPR@FMR1000, AUC) |
| `all_models_overview.png` | Combined grid (models × plots): Plot 1 \| Plot 2 \| one column per cross-pattern combo \| Plot 4; all axes share a unified y-axis scale per row |

**Per-pattern cross-model overview figures** saved to `results/`:

| File | Contents |
|------|----------|
| `overview_plot1_mated_vs_nonmated_pat{N}_clean.png` | 2 × 3 grid — one panel per model (5 individual + All Models avg); each panel shows plot1 clean for pattern N |
| `overview_plot2_pose_breakdown_pat{N}_clean.png` | Same grid layout for the pose-breakdown plot |
| `overview_plot3_cross_pattern_pat{N}_clean.png` | Same grid layout for the cross-pattern plot anchored to pattern N |
| `overview_plot4_combined_pat{N}_clean.png` | Same grid layout for the combined plot |

**Optimal-accuracy decision threshold tables** saved to `results/`:

| File | Contents |
|------|----------|
| `fig_accuracy_threshold_accuracy.png` | Rows = `No pattern (ctrl↔ctrl)`, `All patterns`, P1, P2, … then P1↔P2, P1↔P3, … (adv↔adv, Plot 3 pairs); columns = one per model + All (avg sim); cells show **optimal balanced accuracy** at the best swept threshold |
| `fig_accuracy_threshold_thresholds.png` | Identical layout; cells show the **optimal threshold value** t* that produced the accuracy above |
| `fig_accuracy_fixed_threshold.png` | Same row/column layout; cells show `acc (TPR/TNR)` evaluated at each model's **hardcoded LFW threshold** from `config.py`; legend at the bottom defines each term |

The first row, `No pattern (ctrl↔ctrl)`, evaluates all control images against each other (mated = same identity, non-mated = different identity) — a clean baseline with no adversarial perturbation.

Threshold optimisation sweeps 400 candidates and selects t* = argmax [0.5 · TPR(t) + 0.5 · TNR(t)].  Does **not** require `pyeer`.  The fixed-threshold table uses `MODEL_THRESHOLDS` from `config.py`; the `Avg (sim)` column uses `DEFAULT_THRESHOLD`.

All distribution plots use KDE smoothing (requires `scipy`; step-histogram fallback if absent).  Non-mated pairs are sampled with a fixed seed (`--seed`) for reproducibility.  Biometric statistics (EER, FMR100, AUC) require `pyeer` (`pip install pyeer`).

**Colour convention (heatmaps):** green = model still recognises the attacker
(worse attack); red = model fooled (better attack).  Rank-N accuracy uses an
inverted scale because lower recognition rate is better for the attacker.

---

## CLI options

Most scripts accept the same two flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `db.sqlite` | Path to the SQLite database |
| `--out` | `data_processing/results/` | Output directory for CSV / PNG files |

`visualize_results.py` accepts `--out` only (it reads the CSVs written by the
other scripts, not the database directly).

`score_distributions.py` additionally accepts:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `42` | Random seed for non-mated pair sampling |
| `--regen` | off | Overwrite existing output files; by default any file already on disk is skipped |

`analyze_color_vs_rank.py` additionally accepts:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Root application data directory used to locate control images |
| `--csv` | `data_processing/results/color_vs_rank.csv` | Output CSV path |
| `--plot` | `data_processing/results/color_vs_rank.png` | Output plot path (`''` disables plot generation) |

`analyze_patchiness_vs_rank.py` additionally accepts:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Root application data directory used to locate control images |
| `--csv` | `data_processing/results/patchiness_vs_rank.csv` | Output CSV path |
| `--plot` | `data_processing/results/patchiness_vs_rank.png` | Scatter-plot grid output path (`''` disables) |
| `--corr-table` | `data_processing/results/patchiness_vs_rank_correlations.png` | Correlation summary table PNG (`''` disables) |
| `--edge-thresh` | `8.0` | ΔE threshold for counting a pixel as an edge in `edge_density` |
| `--quant` | `16` | Number of Lab quantisation levels per channel for region-count and entropy metrics |

## Dependencies

- Python 3.10+
- `numpy` — statistics and embedding operations (all scripts)
- `Pillow` — image loading and colour-space analysis (`analyze_color_vs_rank.py`, `analyze_patchiness_vs_rank.py`)
- `matplotlib` — figure generation (`gallery_search_stats.py`, `visualize_results.py`, `score_distributions.py`, `analyze_color_vs_rank.py`, `analyze_patchiness_vs_rank.py`, `avg_similarity_by_identity.py`)
- `scipy` — KDE smoothing in `score_distributions.py` (optional; step-histogram fallback if absent)
- `pyeer` — biometric statistics in `score_distributions.py` (optional; `pip install pyeer`)
- `core/database.py` — used only by `extract_rankings.py`