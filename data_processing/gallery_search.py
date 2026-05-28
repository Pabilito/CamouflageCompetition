"""
data_processing/gallery_search.py
─────────────────────────────────────────────────────────────────────────────
Standalone replication of the recognition tab's "Gallery Search" subtab,
restricted to the in-app control gallery (data/users/<identity>/...) — no
external gallery datasets, because those make the live computation slow.

For every adversarial image in the `attacks` table, compute:
  • per-model best similarity to each control identity
  • cross-model averaged best similarity per identity
  • ranked identity list (descending), and the target identity's rank
  • the top-1 match across all identities

Output: data_processing/results/gallery_search.csv with one row per
adversarial image.

All embeddings (control + adversarial) are read from the SQLite `embeddings`
table — no model loading, no image decoding. This is the same fast path the
recognition tab uses for batch recomputation.

Run from the project root:
    python -m data_processing.gallery_search
or directly:
    python data_processing/gallery_search.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_DB,
    MODEL_ORDER,
    PROJECT_ROOT,
    ensure_results_dir,
    load_all_attack_rows,
    load_all_embeddings,
    open_db,
    setup_path,
)

setup_path()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _natural_sort_key(text: str):
    """Mirror core.database.natural_sort_key just enough to sort identity dirs."""
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", text)]


def build_control_gallery(
    data_dir: Path,
    project_root: Path,
    emb_cache: dict[str, dict[str, np.ndarray]],
    model_names: list[str],
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Walk data/users/<identity>/ and look up each image's embedding in the
    DB-loaded cache. No model inference — anything missing from the cache is
    skipped (and reported).

    The DB stores image_path as a PROJECT-RELATIVE string with the OS-native
    separator (e.g. 'data\\users\\1\\img_000.jpg' on Windows). We replicate
    that exact form when looking up keys.

    Returns: model_name → identity_name → list of embeddings
    Mirrors recognition_tab._gallery_paths but stores just embeddings (the
    paths aren't needed for ranking, only for matrix packing).
    """
    users_dir = data_dir / "users"
    if not users_dir.exists():
        print(f"[ERROR] {users_dir} does not exist.", file=sys.stderr)
        return {mn: {} for mn in model_names}

    gallery: dict[str, dict[str, list[np.ndarray]]] = {mn: {} for mn in model_names}
    missing_count = 0

    user_dirs = sorted(
        (d for d in users_dir.iterdir() if d.is_dir()),
        key=lambda p: _natural_sort_key(p.name),
    )

    for udir in user_dirs:
        identity = udir.name
        # Same image discovery as recognition_tab._reload_gallery: jpg + png,
        # excluding the *_hq.png companion files.
        img_paths = sorted(udir.glob("*.jpg")) + sorted(udir.glob("*.png"))
        img_paths = [p for p in img_paths if not p.stem.endswith("_hq")]
        if not img_paths:
            continue

        for mn in model_names:
            model_cache = emb_cache.get(mn, {})
            embs: list[np.ndarray] = []
            for p in img_paths:
                # DB keys are project-relative with native separators —
                # str(p.relative_to(project_root)) reproduces that exactly.
                key = str(p.relative_to(project_root))
                emb = model_cache.get(key)
                if emb is None:
                    missing_count += 1
                    continue
                embs.append(emb)
            if embs:
                gallery[mn][identity] = embs

    if missing_count:
        print(f"[WARN] {missing_count} control embeddings missing from DB cache "
              f"(run the app once to backfill).", file=sys.stderr)

    return gallery


def pack_gallery(
    gallery: dict[str, dict[str, list[np.ndarray]]],
    model_names: list[str],
) -> dict[str, tuple[list[str], list[int], np.ndarray]]:
    """Pack per-model gallery into (id_names, boundaries, matrix) tuples for
    one-shot matmul scoring. Same shape as recognition_tab._build_merged_gallery.

    boundaries[i] is the exclusive end-row of identity i in the matrix, so
    identity i occupies rows [boundaries[i-1]:boundaries[i]].
    """
    packed: dict[str, tuple[list[str], list[int], np.ndarray]] = {}
    for mn in model_names:
        identities = gallery.get(mn, {})
        if not identities:
            continue
        id_names: list[str] = []
        boundaries: list[int] = []
        all_embs: list[np.ndarray] = []
        for id_name in sorted(identities.keys(), key=_natural_sort_key):
            embs = identities[id_name]
            if not embs:
                continue
            id_names.append(id_name)
            all_embs.extend(embs)
            boundaries.append(len(all_embs))
        if all_embs:
            packed[mn] = (id_names, boundaries, np.stack(all_embs).astype("float32"))
    return packed


def best_per_identity(
    query: np.ndarray,
    id_names: list[str],
    boundaries: list[int],
    gallery_mat: np.ndarray,
) -> dict[str, float]:
    """Max cosine-similarity of `query` against each identity's vectors.
    Vectorised exactly like recognition_tab._recompute_gallery_positions.
    """
    qnorm = float(np.linalg.norm(query))
    if qnorm < 1e-10:
        return {name: -2.0 for name in id_names}
    norms = np.linalg.norm(gallery_mat, axis=1)
    sims = (gallery_mat @ query) / (norms * qnorm + 1e-10)

    out: dict[str, float] = {}
    prev = 0
    for i, b in enumerate(boundaries):
        out[id_names[i]] = float(np.max(sims[prev:b]))
        prev = b
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    db_path = PROJECT_ROOT / DEFAULT_DB
    if not db_path.exists():
        print(f"[ERROR] Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    data_dir = PROJECT_ROOT / "data"

    con = open_db(db_path)
    try:
        print(f"[INFO] Reading from {db_path}")
        attacks = load_all_attack_rows(con)
        emb_cache = load_all_embeddings(con)
    finally:
        con.close()

    # Use the canonical model order, restricted to models actually present in
    # the embeddings table. (Skipping a missing model is safer than crashing
    # if the user hasn't run all five.)
    model_names = [mn for mn in MODEL_ORDER if mn in emb_cache]
    if not model_names:
        print("[ERROR] No embeddings found in DB.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Models present: {', '.join(model_names)}")

    # Build & pack the in-app control gallery.
    gallery = build_control_gallery(data_dir, PROJECT_ROOT, emb_cache, model_names)
    packed = pack_gallery(gallery, model_names)

    total_identities = max((len(packed[mn][0]) for mn in packed), default=0)
    print(f"[INFO] Control gallery: {total_identities} identities, "
          f"{sum(packed[mn][2].shape[0] for mn in packed)} embedding rows "
          f"across {len(packed)} models")

    if not packed:
        print("[ERROR] Empty control gallery — no identities to rank against.",
              file=sys.stderr)
        sys.exit(1)

    # ── Deduplicate adversarial images ────────────────────────────────────────
    # The attacks table has one row per (image, model). For gallery search we
    # want one ranking per image, so collapse on image_path.
    by_image: dict[str, dict] = {}
    for atk in attacks:
        path = atk["image_path"]
        if not path:
            continue
        if path not in by_image:
            by_image[path] = {
                "image_path":   path,
                "attacker":     atk["attacker_name"],
                "identity":     atk["identity_name"] or atk["attacker_name"],
                "pattern_id":   atk["pattern_id"],
                "is_frontal":   atk["is_frontal"],
                "is_profile":   atk["is_profile"],
                "is_pitch":     atk["is_pitch"],
                "camera_name":  atk["camera_name"],
            }
    print(f"[INFO] Adversarial images to rank: {len(by_image)}")

    # ── Score each image ──────────────────────────────────────────────────────
    rows_out: list[dict] = []
    skipped_no_emb = 0
    skipped_no_target = 0

    for path, meta in by_image.items():
        target = meta["identity"]

        # Per-identity sum/count across the models we successfully scored.
        id_score_sums:   dict[str, float] = {}
        id_score_counts: dict[str, int]   = {}
        per_model_target_score: dict[str, float | None] = {mn: None for mn in model_names}

        models_used = 0
        for mn in model_names:
            if mn not in packed:
                continue
            query = emb_cache.get(mn, {}).get(path)
            if query is None:
                continue  # this image has no embedding for this model — skip

            id_names, boundaries, gallery_mat = packed[mn]
            per_id = best_per_identity(query, id_names, boundaries, gallery_mat)
            models_used += 1

            for name, best in per_id.items():
                id_score_sums[name]   = id_score_sums.get(name, 0.0) + best
                id_score_counts[name] = id_score_counts.get(name, 0) + 1

            if target in per_id:
                per_model_target_score[mn] = per_id[target]

        if not id_score_sums:
            skipped_no_emb += 1
            continue

        # Average per-identity best across models, sort desc.
        ranked = sorted(
            ((name, id_score_sums[name] / id_score_counts[name])
             for name in id_score_sums),
            key=lambda x: x[1],
            reverse=True,
        )
        total_ids  = len(ranked)
        top1_name  = ranked[0][0]
        top1_score = ranked[0][1]
        target_rank: int | None = next(
            (i + 1 for i, (name, _) in enumerate(ranked) if name == target),
            None,
        )
        target_avg_score: float | None = next(
            (s for name, s in ranked if name == target), None,
        )
        if target_rank is None:
            skipped_no_target += 1

        row = {
            "image_path":       path,
            "attacker":         meta["attacker"],
            "target_identity":  target,
            "gallery_position": target_rank if target_rank is not None else "",
            "gallery_total":    total_ids,
            "target_score_avg": f"{target_avg_score:.6f}" if target_avg_score is not None else "",
            "top1_name":        top1_name,
            "top1_score":       f"{top1_score:.6f}",
            "models_used":      models_used,
            "pattern_id":       meta["pattern_id"],
            "is_frontal":       meta["is_frontal"],
            "is_profile":       meta["is_profile"],
            "is_pitch":         meta["is_pitch"],
            "camera_name":      meta["camera_name"] or "",
        }
        # Per-model best similarity to the target (handy for downstream analysis).
        for mn in model_names:
            s = per_model_target_score[mn]
            row[f"target_score_{mn}"] = f"{s:.6f}" if s is not None else ""
        rows_out.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_dir = ensure_results_dir()
    out_path = out_dir / "gallery_search.csv"

    fieldnames = [
        "image_path", "attacker", "target_identity",
        "gallery_position", "gallery_total",
        "target_score_avg", "top1_name", "top1_score",
        "models_used",
        "pattern_id", "is_frontal", "is_profile", "is_pitch", "camera_name",
    ] + [f"target_score_{mn}" for mn in model_names]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort: best (lowest) rank first, then by attacker. Images without a
        # rankable target go to the bottom.
        rows_out.sort(key=lambda r: (
            r["gallery_position"] if isinstance(r["gallery_position"], int) else 1e9,
            r["attacker"],
        ))
        writer.writerows(rows_out)

    elapsed = time.time() - t0
    print(f"[INFO] Wrote {len(rows_out)} rows → {out_path}")
    if skipped_no_emb:
        print(f"[INFO] Skipped {skipped_no_emb} images with no embeddings in the DB.")
    if skipped_no_target:
        print(f"[INFO] {skipped_no_target} images had a target identity not "
              f"present in the control gallery — written with empty rank.")
    print(f"[INFO] Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
