"""
data_processing/config.py
─────────────────────────────────────────────────────────────────────────────
Shared constants, thresholds, and helpers for all data-processing scripts.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent
DEFAULT_DB    = "db.sqlite"
RESULTS_DIR   = SCRIPT_DIR / "results"

# ── Per-model LFW decision thresholds ─────────────────────────────────────────
# A pair is considered a "match" (model recognises the person) when
# cosine_similarity ≥ threshold.  Below threshold = attack fools the model.

MODEL_THRESHOLDS: dict[str, float] = {
    "IResNet_R18_Glint360K":   0.2411,
    "InceptionResnetCASIA":    0.4535,
    "SwinFace":                0.2174,
    "TransFace_S_Glint360K":   0.1640,
    "edgeface_base":           0.2066,
}
DEFAULT_THRESHOLD = 0.2565

# Canonical display order for the five models
MODEL_ORDER: list[str] = [
    "IResNet_R18_Glint360K",
    "InceptionResnetCASIA",
    "SwinFace",
    "TransFace_S_Glint360K",
    "edgeface_base",
]

MODEL_SHORT: dict[str, str] = {
    "IResNet_R18_Glint360K":  "IResNet18",
    "InceptionResnetCASIA":   "InceptionResnet",
    "SwinFace":               "SwinFace",
    "TransFace_S_Glint360K":  "TransFace",
    "edgeface_base":          "EdgeFace",
}

# ── Pose labels ───────────────────────────────────────────────────────────────
# Each image has exactly one of these flags set (or none = "unknown").

POSE_COLUMNS = ("is_frontal", "is_profile", "is_pitch")

POSE_LABELS: dict[str, str] = {
    "is_frontal": "frontal",
    "is_profile": "profile",
    "is_pitch":   "pitch",
}

def pose_label(row: dict) -> str:
    """Return a human-readable pose label from a dict with is_frontal/is_profile/is_pitch."""
    for col, label in POSE_LABELS.items():
        if row.get(col):
            return label
    return "unknown"


# ── Threshold helper ──────────────────────────────────────────────────────────

def threshold_for(model_name: str) -> float:
    return MODEL_THRESHOLDS.get(model_name, DEFAULT_THRESHOLD)


def is_match(score: float, model_name: str) -> bool:
    """True when the model recognises the person (score ≥ threshold)."""
    return score >= threshold_for(model_name)


def is_attack_success(score: float, model_name: str) -> bool:
    """True when the attack fools the model (score < threshold)."""
    return score < threshold_for(model_name)


# ── Lightweight DB access (no core.database dependency) ───────────────────────

def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a read-only connection with Row factory."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def model_names_from_db(con: sqlite3.Connection) -> list[str]:
    """Return ordered model names present in the attacks table."""
    rows = con.execute(
        "SELECT DISTINCT model_name FROM attacks ORDER BY model_name"
    ).fetchall()
    return [r["model_name"] for r in rows]


def load_all_attack_rows(con: sqlite3.Connection) -> list[dict]:
    """Load every attack row as a dict."""
    rows = con.execute(
        "SELECT id, attacker_name, model_name, similarity_score, "
        "image_path, identity_name, matched_image_path, created_at, "
        "pattern_id, is_frontal, is_profile, is_pitch, camera_name, "
        "gallery_position, gallery_total "
        "FROM attacks"
    ).fetchall()
    return [dict(r) for r in rows]


def load_all_embeddings(con: sqlite3.Connection) -> dict[str, dict[str, np.ndarray]]:
    """Load the entire embeddings table.
    Returns {model_name: {image_path: embedding_array}}."""
    rows = con.execute(
        "SELECT model_name, image_path, embedding FROM embeddings"
    ).fetchall()
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        cache.setdefault(row["model_name"], {})[row["image_path"]] = emb
    return cache


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors (= dot product)."""
    return float(np.dot(a, b))


# ── Ensure output directory exists ────────────────────────────────────────────

def ensure_results_dir(out_dir: Path | None = None) -> Path:
    d = out_dir or RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Add project root to sys.path ─────────────────────────────────────────────

def setup_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
