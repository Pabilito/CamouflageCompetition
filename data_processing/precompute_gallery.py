"""
data_processing/precompute_gallery.py
─────────────────────────────────────
Offline pre-computation of external-gallery embeddings.

Run this on a beefy machine to align faces and compute embeddings for a folder
of images organised as:

    <folder>/
        identity_A/
            img_001.jpg
            img_002.jpg
        identity_B/
            ...

The script produces a single .npz file that can be loaded back into the app
via the "📥 Load Precomputed" button in the Gallery Search tab.  The file
contains:

    model_names         (S, )        — one entry per face-recognition model
    identities          (N, )        — one entry per accepted image
    paths               (N, )        — original image paths (for thumbnails)
    emb__<model_name>   (N, D)       — float32 embedding matrix for that model
    meta                json string  — display name, source folder, counts, …

The list of models is whatever EmbeddingEngine loads from --model-dir, which
must match the set of models the app is using when the file is loaded.

Example
───────
    python data_processing/precompute_gallery.py /data/CelebA \\
        --output /data/CelebA_gallery.npz \\
        --workers 8 \\
        --model-dir Pretrained_Models

Resume an interrupted run (skips images already embedded in the output file):

    python data_processing/precompute_gallery.py /data/CelebA \\
        --output /data/CelebA_gallery.npz --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# The script expects to be run from the project root (so `core.*` is importable).
# It also accepts being run directly from the data_processing/ folder.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.embeddings import EmbeddingEngine     # noqa: E402
from core.face_aligner import FaceAligner       # noqa: E402
from load_model import load_app_models          # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FILE_VERSION = 1


# ── Discovery ──────────────────────────────────────────────────────────────

def discover_images(folder: Path) -> dict[str, list[Path]]:
    """Return {identity_name: [image_path, …]} for every subfolder of `folder`.

    Mirrors the in-app discovery logic in ranking_tab._gs_add_dataset:
    one level of subfolders, common image extensions only, '_hq' companion
    files excluded.
    """
    identity_images: dict[str, list[Path]] = {}
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir():
            continue
        imgs = sorted(
            p for p in sub.iterdir()
            if p.is_file()
            and p.suffix.lower() in _IMG_EXTS
            and "_hq" not in p.stem
        )
        if imgs:
            identity_images[sub.name] = imgs
    return identity_images


# ── Resume support ─────────────────────────────────────────────────────────

def load_existing(out_path: Path) -> tuple[set[str], dict[str, list], list[str], list[str]]:
    """If `out_path` exists, return (done_paths, embeddings_per_model,
    identities_so_far, paths_so_far) so processing can resume.

    embeddings_per_model maps model_name → list of (D,) float32 arrays in the
    same order as identities_so_far / paths_so_far.
    """
    if not out_path.exists():
        return set(), {}, [], []

    with np.load(out_path, allow_pickle=True) as data:
        identities = [str(x) for x in data["identities"]]
        paths      = [str(x) for x in data["paths"]]
        model_names = [str(x) for x in data["model_names"]]
        emb_per_model: dict[str, list] = {}
        for mn in model_names:
            mat = data[f"emb__{mn}"]
            emb_per_model[mn] = [mat[i] for i in range(mat.shape[0])]

    return set(paths), emb_per_model, identities, paths


# ── Output ─────────────────────────────────────────────────────────────────

def write_output(out_path: Path,
                 model_names: list[str],
                 identities: list[str],
                 paths: list[str],
                 emb_per_model: dict[str, list[np.ndarray]],
                 meta: dict) -> None:
    """Atomically write the .npz file (write-then-rename)."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    arrays = {
        "model_names": np.array(model_names, dtype=object),
        "identities":  np.array(identities,  dtype=object),
        "paths":       np.array(paths,       dtype=object),
        "meta":        np.array(json.dumps(meta), dtype=object),
    }
    # Allow_pickle = True is needed for object arrays (variable-length strings).
    # Numpy now stores them as pickled blobs, but the consumer (the app)
    # re-loads with allow_pickle=True so this is fine.

    for mn in model_names:
        embs = emb_per_model.get(mn, [])
        if embs:
            arrays[f"emb__{mn}"] = np.stack(embs).astype("float32")
        else:
            arrays[f"emb__{mn}"] = np.zeros((0, 0), dtype="float32")

    np.savez(tmp_path, **arrays)
    tmp_path.replace(out_path)


# ── Worker ─────────────────────────────────────────────────────────────────

def _process_one(args: tuple[str, Path],
                 aligner: FaceAligner,
                 engine: EmbeddingEngine,
                 cancel_flag: threading.Event):
    """Align one image and embed it across every loaded model.

    Returns (rows, identity) where rows is dict {model_name: emb (D,) float32}
    or None on failure / cancellation.  identity is always returned so the
    progress callback can show it.
    """
    if cancel_flag.is_set():
        return None, args[0]
    identity, img_path = args

    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        return None, identity

    try:
        aligned = aligner.align(pil_img)
    except Exception:
        aligned = None
    if aligned is None:
        return None, identity

    face_arr = np.array(aligned.image.convert("RGB"))
    out: dict[str, np.ndarray] = {}
    for model_name in engine.model_names:
        try:
            emb = engine.embed(model_name, face_arr)
            out[model_name] = emb.astype("float32")
        except Exception as e:
            print(f"[EMBED] {model_name} {img_path.name}: {e}", flush=True)
    return (out if out else None), identity


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compute external gallery embeddings for the Face "
                    "Camouflage Competition app.")
    parser.add_argument("folder", type=Path,
                        help="Root folder. Direct subfolders are treated as "
                             "identities; their images are embedded.")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output .npz file path.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Thread-pool size (default 4).")
    parser.add_argument("--model-dir", type=str, default="Pretrained_Models",
                        help="Folder with .pt/.pth weight files. Forwarded "
                             "to EmbeddingEngine. Default: Pretrained_Models.")
    parser.add_argument("--display-name", type=str, default=None,
                        help="Display name for the dataset (default: folder "
                             "basename).")
    parser.add_argument("--resume", action="store_true",
                        help="If the output file already exists, skip images "
                             "whose paths are already recorded in it.")
    parser.add_argument("--flush-every", type=int, default=200,
                        help="Persist progress to disk every N accepted "
                             "images (default 200). 0 disables intermediate "
                             "flushes.")
    args = parser.parse_args()

    folder: Path = args.folder.expanduser().resolve()
    out_path: Path = args.output.expanduser().resolve()
    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.resume:
        print(f"ERROR: {out_path} already exists. Use --resume to extend it, "
              f"or pick a different output path.", file=sys.stderr)
        return 2

    # ── Discover images ───────────────────────────────────────────────────
    print(f"[SCAN] {folder}")
    identity_images = discover_images(folder)
    if not identity_images:
        print("ERROR: no identity subfolders with images found.", file=sys.stderr)
        return 2

    total_imgs = sum(len(v) for v in identity_images.values())
    print(f"[SCAN] {len(identity_images)} identities, {total_imgs} images")

    # ── Load models + aligner ─────────────────────────────────────────────
    print(f"[INIT] loading models from {args.model_dir} …")
    bundles = load_app_models(model_dir=args.model_dir)
    engine = EmbeddingEngine(bundles)
    print(f"[INIT] models: {engine.model_names}")
    print(f"[INIT] loading FaceAligner …")
    aligner = FaceAligner()

    # ── Resume support ────────────────────────────────────────────────────
    done_paths, emb_per_model, identities_acc, paths_acc = (set(), {}, [], [])
    if args.resume and out_path.exists():
        done_paths, emb_per_model, identities_acc, paths_acc = load_existing(out_path)
        print(f"[RESUME] {len(paths_acc)} images already in {out_path.name}")
        # Verify model compatibility with the resumed file
        existing_models = set(emb_per_model.keys())
        current_models  = set(engine.model_names)
        if existing_models != current_models:
            missing = current_models - existing_models
            extra   = existing_models - current_models
            print(f"ERROR: model mismatch with existing file.\n"
                  f"  missing in file: {sorted(missing) or '(none)'}\n"
                  f"  extra in file:   {sorted(extra) or '(none)'}\n"
                  f"Either delete {out_path.name} or run with the original "
                  f"--model-dir.", file=sys.stderr)
            return 2

    for mn in engine.model_names:
        emb_per_model.setdefault(mn, [])

    # ── Build work list ───────────────────────────────────────────────────
    work: list[tuple[str, Path]] = []
    for identity in sorted(identity_images.keys()):
        for img_path in identity_images[identity]:
            if str(img_path) in done_paths:
                continue
            work.append((identity, img_path))

    skipped  = total_imgs - len(work)
    if not work:
        print("[DONE] nothing to do — every image is already embedded.")
        return 0
    print(f"[WORK] {len(work)} new images to process ({skipped} already done)")

    cancel_flag = threading.Event()
    lock        = threading.Lock()
    processed   = 0
    accepted    = 0
    failed      = 0
    started     = time.time()
    last_flush  = 0  # accepted-count at the time of the last flush

    def _flush(final: bool = False) -> None:
        nonlocal last_flush
        meta = {
            "version":       _FILE_VERSION,
            "display_name":  args.display_name or folder.name,
            "source_folder": str(folder),
            "model_names":   list(engine.model_names),
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "total_scanned": total_imgs,
            "accepted":      len(paths_acc),
            "in_progress":   not final,
        }
        write_output(out_path, list(engine.model_names),
                     identities_acc, paths_acc, emb_per_model, meta)
        last_flush = accepted

    # ── Process ───────────────────────────────────────────────────────────
    print(f"[RUN] {args.workers} workers")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, item, aligner, engine, cancel_flag): item
                       for item in work}
            for future in concurrent.futures.as_completed(futures):
                try:
                    rows, identity = future.result()
                except Exception as e:
                    print(f"[ERR] worker raised: {e}", flush=True)
                    rows, identity = None, "?"

                with lock:
                    processed += 1
                    img_path = futures[future][1]
                    if rows:
                        accepted += 1
                        identities_acc.append(identity)
                        paths_acc.append(str(img_path))
                        for mn in engine.model_names:
                            # Every accepted image must have an embedding for
                            # every model, otherwise the columns get misaligned.
                            # If a model failed for this image, drop the whole
                            # image — same policy the in-app importer applies
                            # implicitly (all-or-nothing rows).
                            if mn not in rows:
                                # Roll back this image's partial accept
                                identities_acc.pop()
                                paths_acc.pop()
                                for already in engine.model_names:
                                    if already == mn:
                                        break
                                    emb_per_model[already].pop()
                                accepted -= 1
                                failed += 1
                                break
                            emb_per_model[mn].append(rows[mn])
                    else:
                        failed += 1

                    if processed % 25 == 0 or processed == len(work):
                        rate = processed / max(1e-6, time.time() - started)
                        eta_s = (len(work) - processed) / rate if rate > 0 else 0
                        print(f"[{processed:>6}/{len(work)}]  accepted={accepted}  "
                              f"failed={failed}  {rate:5.1f} img/s  "
                              f"ETA {eta_s/60:5.1f} min  identity={identity!r}",
                              flush=True)

                    if (args.flush_every > 0
                            and accepted - last_flush >= args.flush_every):
                        _flush(final=False)

    except KeyboardInterrupt:
        print("\n[CANCEL] Ctrl-C received — flushing partial results …")
        cancel_flag.set()

    # ── Final flush ───────────────────────────────────────────────────────
    _flush(final=True)
    elapsed = time.time() - started
    print(f"[DONE] {accepted} accepted, {failed} failed/skipped, "
          f"{elapsed/60:.1f} min wall.")
    print(f"[DONE] wrote {out_path}  "
          f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
