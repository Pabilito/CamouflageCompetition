#!/usr/bin/env python3
"""
cross_similarity.py
───────────────────
Hardcoded config:
  --data-dir data
  --mode     across   (A↔B cross-pattern pairs only)
  --csv      across_results.csv

Run simply with:
  python cross_similarity.py
"""

import csv
import itertools
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── LFW decision thresholds ───────────────────────────────────────────────────
THRESHOLDS_LFW: dict[str, float] = {
    "IResNet_R18_Glint360K":    0.2411,
    "IResNet_R50_Glint360K":    0.2393,
    "IResNet_R100_Glint360K":   0.3309,
    "InceptionResnetCASIA":     0.4535,
    "InceptionResnetV1_CASIA":  0.4535,
    "InceptionResnetV1_VGG":    0.4474,
    "IDiffFace_CF50":           0.2694,
    "SFace_IResNet50":          0.2306,
    "SwinFace":                 0.2174,
    "TransFace_S_Glint360K":    0.1640,
    "edgeface_base":            0.2066,
}
DEFAULT_THRESHOLD = 0.2565

# ── Pattern split map ─────────────────────────────────────────────────────────
# (n_pattern_A, n_pattern_B) — images sorted by filename before splitting.
# All identities not listed here use DEFAULT_SPLIT = (3, 3).
PATTERN_SPLIT: dict[str, tuple[int, int]] = {
    "4":  (2, 3),
    "12": (2, 3),
    "16": (4, 3),
    "20": (2, 3),
    "30": (3, 2),
}
DEFAULT_SPLIT = (3, 3)

# Only process identities 1-30; -1 is explicitly excluded
VALID_IDENTITIES = {str(i) for i in range(1, 31)}


# ── DB helper ─────────────────────────────────────────────────────────────────

class _DB:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row

    def identities(self) -> list[str]:
        """All identity names in the DB that fall within 1-30, sorted numerically."""
        rows = self._conn.execute(
            "SELECT DISTINCT identity_name FROM attacks "
            "WHERE identity_name IS NOT NULL AND identity_name != '' "
            "ORDER BY identity_name"
        ).fetchall()
        valid = [r["identity_name"] for r in rows if r["identity_name"] in VALID_IDENTITIES]
        valid.sort(key=lambda x: int(x) if x.isdigit() else x)
        return valid

    def adv_image_paths(self, identity: str,
                        model: Optional[str] = None) -> list[str]:
        """Distinct adversarial image paths for an identity, sorted by filename."""
        sql = (
            "SELECT DISTINCT image_path FROM attacks "
            "WHERE identity_name = ? "
        )
        params: list = [identity]
        if model:
            sql += "AND model_name = ? "
            params.append(model)
        rows = self._conn.execute(sql, params).fetchall()
        paths = [r["image_path"] for r in rows]
        paths.sort(key=lambda p: Path(p).name)
        return paths

    def models_for_identity(self, identity: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT model_name FROM attacks WHERE identity_name = ? "
            "ORDER BY model_name",
            (identity,),
        ).fetchall()
        return [r["model_name"] for r in rows]

    def get_embedding(self, image_path: str,
                      model_name: str) -> Optional[np.ndarray]:
        """Return cached embedding from the DB, or None on miss."""
        row = self._conn.execute(
            "SELECT embedding FROM embeddings "
            "WHERE image_path=? AND model_name=?",
            (image_path, model_name),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()

    def load_all_embeddings(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Load the entire embeddings table in one query.
        Returns {model_name: {image_path: embedding_array}}.
        """
        rows = self._conn.execute(
            "SELECT model_name, image_path, embedding FROM embeddings"
        ).fetchall()
        cache: dict[str, dict[str, np.ndarray]] = {}
        for row in rows:
            model = row["model_name"]
            path  = row["image_path"]
            emb   = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            cache.setdefault(model, {})[path] = emb
        return cache

    def set_embedding(self, image_path: str, model_name: str,
                      embedding: np.ndarray) -> None:
        """Store an embedding in the DB cache."""
        from datetime import datetime, timezone
        blob = embedding.astype("float32").tobytes()
        now  = datetime.now(tz=timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(image_path, model_name, embedding, created_at) VALUES (?,?,?,?)",
            (image_path, model_name, blob, now),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ── Pattern splitter ──────────────────────────────────────────────────────────

def split_patterns(identity: str,
                   paths: list[str]) -> tuple[list[str], list[str]]:
    """
    Return (pattern_A_paths, pattern_B_paths).
    Uses PATTERN_SPLIT for exceptions, DEFAULT_SPLIT otherwise.
    Raises ValueError if there are not enough images.
    """
    n_a, n_b = PATTERN_SPLIT.get(identity, DEFAULT_SPLIT)
    needed = n_a + n_b
    if len(paths) < needed:
        raise ValueError(
            f"Identity '{identity}': expected {needed} images "
            f"(A={n_a}, B={n_b}) but only {len(paths)} found."
        )
    if len(paths) > needed:
        print(
            f"  [WARN] Identity '{identity}': {len(paths)} images found; "
            f"using first {needed} (A={n_a}, B={n_b}).",
            file=sys.stderr,
        )
    return paths[:n_a], paths[n_a : n_a + n_b]


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are L2-normalised, so dot product == cosine similarity.
    return float(np.dot(a, b))


def _get_embedding(emb_cache: dict, model_name: str,
                   path: str) -> Optional[np.ndarray]:
    """
    Look up a pre-computed embedding from the in-memory cache.
    Warns and returns None if an entry is missing.
    """
    emb = emb_cache.get(model_name, {}).get(path)
    if emb is None:
        print(f"  [WARN] No cached embedding for '{Path(path).name}' "
              f"/ {model_name} — skipping pair.", file=sys.stderr)
    return emb


# ── Core comparison ───────────────────────────────────────────────────────────

def _pairs_for_mode(
    pat_a: list[str], pat_b: list[str], mode: str
) -> list[tuple[str, str, str]]:
    """
    Return list of (path_x, path_y, pair_type).
    pair_type is one of: 'within_A', 'within_B', 'across'.
    """
    pairs: list[tuple[str, str, str]] = []
    if mode in ("within", "both"):
        for x, y in itertools.combinations(pat_a, 2):
            pairs.append((x, y, "within_A"))
        for x, y in itertools.combinations(pat_b, 2):
            pairs.append((x, y, "within_B"))
    if mode in ("across", "both"):
        for x in pat_a:
            for y in pat_b:
                pairs.append((x, y, "across"))
    return pairs


def compare_identity(
    db: _DB,
    emb_cache: dict,
    identity: str,
    model_filter: Optional[str],
    mode: str,
    matched_only: bool,
) -> list[dict]:
    models = [model_filter] if model_filter else db.models_for_identity(identity)
    if not models:
        print(f"  [INFO] No model records for '{identity}'.")
        return []

    all_paths = db.adv_image_paths(identity)
    try:
        pat_a, pat_b = split_patterns(identity, all_paths)
    except ValueError as exc:
        print(f"  [SKIP] {exc}", file=sys.stderr)
        return []

    n_a, n_b = PATTERN_SPLIT.get(identity, DEFAULT_SPLIT)
    print(
        f"  Pattern A ({len(pat_a)}): {[Path(p).name for p in pat_a]}\n"
        f"  Pattern B ({len(pat_b)}): {[Path(p).name for p in pat_b]}"
    )

    raw_pairs = _pairs_for_mode(pat_a, pat_b, mode)
    results: list[dict] = []

    for model_name in models:
        threshold = THRESHOLDS_LFW.get(model_name, DEFAULT_THRESHOLD)

        # Look up pre-computed embeddings from the in-memory cache
        unique_paths = list({p for p, _, _ in raw_pairs} | {q for _, q, _ in raw_pairs})
        local_cache: dict[str, Optional[np.ndarray]] = {
            p: _get_embedding(emb_cache, model_name, p) for p in unique_paths
        }

        for path_a, path_b, pair_type in raw_pairs:
            ea = local_cache[path_a]
            eb = local_cache[path_b]
            if ea is None or eb is None:
                continue

            sim     = _cosine(ea, eb)
            matched = sim >= threshold

            if matched_only and not matched:
                continue

            results.append({
                "identity":   identity,
                "model":      model_name,
                "threshold":  threshold,
                "pair_type":  pair_type,
                "image_a":    Path(path_a).name,
                "image_b":    Path(path_b).name,
                "similarity": sim,
                "matched":    matched,
                "path_a":     path_a,
                "path_b":     path_b,
            })

    return results


# ── Display ───────────────────────────────────────────────────────────────────

_G = "\033[92m"   # green
_R = "\033[91m"   # red
_Y = "\033[93m"   # yellow
_C = "\033[96m"   # cyan
_M = "\033[95m"   # magenta
_W = "\033[97m"   # white
_B = "\033[1m"    # bold
_X = "\033[0m"    # reset

_PAIR_COLOUR = {"within_A": _C, "within_B": _M, "across": _W}
_PAIR_LABEL  = {"within_A": "A↔A (within)", "within_B": "B↔B (within)", "across": "A↔B (across)"}


def _colour_sim(sim: float, threshold: float) -> str:
    diff = sim - threshold
    col  = _G if diff >= 0.05 else (_Y if diff >= 0 else _R)
    return f"{col}{sim:.6f}{_X}"


def _print_results(results: list[dict]) -> None:
    if not results:
        print("  No pairs to display.")
        return

    # Group: identity → model → pair_type → rows
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in results:
        grouped[r["identity"]][r["model"]][r["pair_type"]].append(r)

    for identity in sorted(grouped, key=lambda x: int(x) if x.isdigit() else x):
        print(f"\n{_B}{_C}━━━  Identity: {identity}  ━━━{_X}")

        for model_name in sorted(grouped[identity]):
            threshold = list(grouped[identity][model_name].values())[0][0]["threshold"]
            print(f"\n  {_B}Model: {model_name}{_X}  (threshold = {threshold})")

            for pair_type in ("within_A", "within_B", "across"):
                pairs = grouped[identity][model_name].get(pair_type)
                if not pairs:
                    continue
                pc   = _PAIR_COLOUR[pair_type]
                pl   = _PAIR_LABEL[pair_type]
                sims = [p["similarity"] for p in pairs]

                print(f"\n    {pc}{_B}[{pl}]{_X}")
                print(f"    {_B}{'Image A':<28}  {'Image B':<28}  "
                      f"{'Similarity':>12}  Match{_X}")
                print(f"    {'─'*28}  {'─'*28}  {'─'*12}  ─────")

                for p in sorted(pairs, key=lambda x: -x["similarity"]):
                    match_str = (f"{_G}✓ yes{_X}" if p["matched"] else f"{_R}✗ no{_X}")
                    print(
                        f"    {p['image_a']:<28}  {p['image_b']:<28}  "
                        f"{_colour_sim(p['similarity'], threshold):>12}  {match_str}"
                    )

                if len(sims) > 1:
                    print(
                        f"    → mean={np.mean(sims):.6f}  "
                        f"min={np.min(sims):.6f}  "
                        f"max={np.max(sims):.6f}  "
                        f"std={np.std(sims):.6f}"
                    )


# Short display names for the 5 models, in a fixed order
_MODEL_SHORT: dict[str, str] = {
    "IResNet_R18_Glint360K":  "IResNet18",
    "InceptionResnetCASIA":   "InceptionResnet",
    "SwinFace":               "SwinFace",
    "TransFace_S_Glint360K":  "TransFace",
    "edgeface_base":          "EdgeFace",
}
_MODEL_ORDER = list(_MODEL_SHORT.keys())
_COL_W = 15   # width of each model column


import re as _re

def _strip_ansi(text: str) -> str:
    """Remove ANSI colour codes for plain-text file output."""
    return _re.sub(r"\[[0-9;]*m", "", text)


def _fmt_cell(val: Optional[float], thr: float) -> str:
    if val is None:
        return f"{'—':^{_COL_W}}"
    col = _G if val >= thr else _R
    return f"{col}{val:^{_COL_W}.6f}{_X}"


def _print_summary(results: list[dict]) -> None:
    """
    Three similarity tables (mean / min / max) + accuracy table.
    Each table: one row per identity, one column per model + Avg column.
    Followed by a per-model and grand overall accuracy footer.
    Saves a plain-text copy to SUMMARY_OUT.
    """
    if not results:
        return

    import io as _io, sys as _sys

    # ── Collect raw data ───────────────────────────────────────────────────
    # sim_data  : identity → model → [similarity floats]
    # match_data: identity → model → [bool]
    sim_data:   dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    match_data: dict[str, dict[str, list[bool]]]  = defaultdict(lambda: defaultdict(list))
    for r in results:
        sim_data[r["identity"]][r["model"]].append(r["similarity"])
        match_data[r["identity"]][r["model"]].append(r["matched"])

    identities     = sorted(sim_data, key=lambda x: int(x) if x.isdigit() else x)
    models_present = [m for m in _MODEL_ORDER if any(m in sim_data[i] for i in identities)]
    short_names    = [_MODEL_SHORT.get(m, m) for m in models_present]

    id_w  = 6
    sep   = "  "
    AVG_W = _COL_W

    # header columns: one per model + Avg
    h_model_cols = sep.join(f"{s:^{_COL_W}}" for s in short_names)
    h_avg        = f"{'Avg':^{AVG_W}}"
    full_header  = h_model_cols + sep + h_avg
    div          = "─" * (id_w + 2 + len(full_header))
    w            = len(div) + 2

    # ── Helper: colour a cell ──────────────────────────────────────────────
    def _sim_cell(val, thr):
        if val is None:
            return f"{'—':^{_COL_W}}"
        col = _G if val >= thr else _R
        return f"{col}{val:^{_COL_W}.6f}{_X}"

    def _acc_cell(val):
        """val = fraction [0,1]. Green ≥ 0.5, yellow ≥ 0.25, red < 0.25."""
        if val is None:
            return f"{'—':^{_COL_W}}"
        pct = val * 100
        col = _G if val >= 0.5 else (_Y if val >= 0.25 else _R)
        return f"{col}{pct:^{_COL_W}.1f}%{_X}"

    def _avg_sim_cell(vals_per_model):
        """Average of per-model means — plain white."""
        flat = [v for vs in vals_per_model if vs for v in vs]
        if not flat:
            return f"{'—':^{AVG_W}}"
        return f"{_W}{float(np.mean(flat)):^{AVG_W}.6f}{_X}"

    def _avg_acc_cell(bools_per_model):
        flat = [b for bs in bools_per_model if bs for b in bs]
        if not flat:
            return f"{'—':^{AVG_W}}"
        v = sum(flat) / len(flat)
        col = _G if v >= 0.5 else (_Y if v >= 0.25 else _R)
        return f"{col}{v*100:^{AVG_W}.1f}%{_X}"

    # ── Generic table printer ──────────────────────────────────────────────
    def _print_table(title: str, row_fn) -> None:
        """row_fn(identity) → (list_of_model_cells, avg_cell)"""
        print(f"\n{_B}{title}{_X}")
        print(f"  {_B}{'ID':<{id_w}}{sep}{full_header}{_X}")
        print("  " + div)
        for identity in identities:
            model_cells, avg_cell = row_fn(identity)
            print(f"  {identity:<{id_w}}{sep}"
                  + sep.join(model_cells)
                  + sep + avg_cell)

    # ── Row builders ──────────────────────────────────────────────────────
    def _sim_row(agg):
        def _build(identity):
            cells, vals_for_avg = [], []
            for m in models_present:
                vs  = sim_data[identity].get(m)
                thr = THRESHOLDS_LFW.get(m, DEFAULT_THRESHOLD)
                v   = agg(vs) if vs else None
                cells.append(_sim_cell(v, thr))
                vals_for_avg.append(vs)
            avg = _avg_sim_cell(vals_for_avg)
            return cells, avg
        return _build

    def _acc_row(identity):
        cells, bools_for_avg = [], []
        for m in models_present:
            bs = match_data[identity].get(m)
            if bs:
                v = sum(bs) / len(bs)
                cells.append(_acc_cell(v))
            else:
                cells.append(_acc_cell(None))
            bools_for_avg.append(bs)
        avg = _avg_acc_cell(bools_for_avg)
        return cells, avg

    # ── Footer: per-model accuracy + grand total ───────────────────────────
    def _print_footer() -> None:
        print(f"\n{_B}  Overall accuracy per model  (% pairs above threshold):{_X}")
        print("  " + div)

        model_accs = []
        footer_cells = []
        for m in models_present:
            all_bools = [b for i in identities for b in match_data[i].get(m, [])]
            if all_bools:
                acc = sum(all_bools) / len(all_bools)
                model_accs.append(acc)
                footer_cells.append(_acc_cell(acc))
            else:
                model_accs.append(None)
                footer_cells.append(_acc_cell(None))

        grand = sum(b for i in identities for m in models_present
                    for b in match_data[i].get(m, []))
        total = sum(len(match_data[i].get(m, [])) for i in identities
                    for m in models_present)
        grand_acc = grand / total if total else None

        avg_cell = _acc_cell(grand_acc)
        print(f"  {'ALL':<{id_w}}{sep}"
              + sep.join(footer_cells)
              + sep + avg_cell)

        if grand_acc is not None:
            col = _G if grand_acc >= 0.5 else (_Y if grand_acc >= 0.25 else _R)
            print(f"\n  {_B}Grand overall accuracy: "
                  f"{col}{grand_acc*100:.1f}%{_X}  "
                  f"({grand}/{total} pairs above threshold)")

    # ── Print everything ───────────────────────────────────────────────────
    def _print_all() -> None:
        print(f"\n{_B}{'━' * w}{_X}")
        print(f"{_B}SUMMARY  —  A↔B cosine similarity  "
              f"(green ≥ threshold = fooling model / recognised,  red = not fooled){_X}")
        print(f"{_B}{'━' * w}{_X}")

        _print_table("MEAN  similarity",  _sim_row(lambda v: float(np.mean(v))))
        _print_table("MIN   similarity",  _sim_row(lambda v: float(np.min(v))))
        _print_table("MAX   similarity",  _sim_row(lambda v: float(np.max(v))))

        print(f"\n{_B}{'━' * w}{_X}")
        print(f"{_B}ACCURACY  —  % of A↔B pairs above decision threshold per identity{_X}")
        print(f"{_B}{'━' * w}{_X}")
        _print_table("ACCURACY  (% matched pairs)", lambda i: _acc_row(i))
        _print_footer()

    # Print to terminal
    _print_all()

    # Save plain-text copy
    buf = _io.StringIO()
    _orig_stdout = _sys.stdout
    _sys.stdout = buf
    _print_all()
    _sys.stdout = _orig_stdout
    Path(SUMMARY_OUT).write_text(_strip_ansi(buf.getvalue()), encoding="utf-8")
    print(f"\n[✓] Summary saved → {SUMMARY_OUT}")


def _export_csv(results: list[dict], out_path: str) -> None:
    fields = [
        "identity", "model", "threshold", "pair_type",
        "image_a", "image_b", "similarity", "matched",
        "path_a", "path_b",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\n[✓] Results exported → {out_path}")


# ── Hardcoded config ──────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DATA_DIR = _PROJECT_ROOT / "data"
DB_PATH  = _PROJECT_ROOT / "db.sqlite"
MODE     = "across"           # A↔B cross-pattern pairs only
CSV_OUT      = str(_SCRIPT_DIR / "across_results.csv")    # output CSV filename
SUMMARY_OUT  = str(_SCRIPT_DIR / "across_summary.txt")    # output summary filename


def _find_db(search_dir: Path) -> Path:
    candidates = list(search_dir.glob("*.db")) + list(search_dir.glob("*.sqlite"))
    if not candidates:
        sys.exit(f"[ERROR] No .db file found under '{search_dir}'.")
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if c.stem == "attacks":
            return c
    print(f"[WARN] Multiple .db files found; using '{candidates[0]}'.", file=sys.stderr)
    return candidates[0]


def main() -> None:
    db_path = DB_PATH
    if not db_path.exists():
        sys.exit(f"[ERROR] Database not found: {db_path}")

    db = _DB(db_path)

    # ── Select identities 1-30 (excludes -1) ──────────────────────────────
    identities = db.identities()
    if not identities:
        sys.exit("[ERROR] No valid identities (1-30) found in the database.")
    print(f"[•] Found {len(identities)} identit(y/ies): {', '.join(identities)}\n")

    # ── Run comparisons ────────────────────────────────────────────────────
    print("[•] Loading all embeddings from DB into memory…")
    emb_cache = db.load_all_embeddings()
    n_models = len(emb_cache)
    n_entries = sum(len(v) for v in emb_cache.values())
    print(f"[✓] Loaded {n_entries} embeddings across {n_models} model(s).\n")

    all_results: list[dict] = []
    for identity in identities:
        print(f"[•] Identity '{identity}'")
        res = compare_identity(
            db=db,
            emb_cache=emb_cache,
            identity=identity,
            model_filter=None,
            mode=MODE,
            matched_only=False,
        )
        all_results.extend(res)
        print()

    db.close()

    # ── Output ─────────────────────────────────────────────────────────────
    _print_results(all_results)
    _print_summary(all_results)
    _export_csv(all_results, CSV_OUT)

    if all_results:
        sims    = [r["similarity"] for r in all_results]
        matched = sum(1 for r in all_results if r["matched"])
        print(
            f"\nTotal pairs: {len(all_results)}  |  "
            f"Matched: {matched} ({matched / len(all_results):.1%})  |  "
            f"Grand-mean similarity: {np.mean(sims):.6f}"
        )


if __name__ == "__main__":
    main()
