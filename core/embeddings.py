"""
core/embeddings.py
─────────────────────────────────────────────────────────────────────────────
Wraps the five ModelBundles into a single EmbeddingEngine that the UI can
call without knowing anything about PyTorch or model internals.

Public API
──────────
  engine = EmbeddingEngine(bundles, device)

  # Compute an embedding for one face crop (H×W×3 uint8 numpy array, RGB):
  vec = engine.embed(model_name, face_crop_rgb)   # → np.ndarray (L2-normed)

  # Compare two embeddings:
  score = engine.cosine_similarity(vec1, vec2)     # → float in [-1, 1]

  # Identify against a gallery {name: [embedding, ...]}:
  match, score = engine.identify(model_name, face_crop_rgb, gallery)

  # Precompute gallery embeddings from saved face images on disk:
  gallery = engine.build_gallery(model_name, user_image_dir)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from load_model import ModelBundle


class EmbeddingEngine:
    """
    Manages multiple ModelBundles and exposes a uniform embedding API.
    Each model gets its own torchvision transform (respecting per-model
    resolution and normalisation stats).
    """

    def __init__(
        self,
        bundles: list[ModelBundle],
        device: torch.device | None = None,
    ) -> None:
        self.device   = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._bundles: dict[str, ModelBundle] = {b.name: b for b in bundles}
        self._transforms: dict[str, transforms.Compose] = {
            b.name: _make_transform(b.height, b.width, b.mean, b.std)
            for b in bundles
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_names(self) -> list[str]:
        return list(self._bundles.keys())

    # ── Core embedding ────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, model_name: str, face_rgb: np.ndarray) -> np.ndarray:
        """
        Parameters
        ──────────
        model_name : one of the five canonical model names
        face_rgb   : H×W×3 uint8 NumPy array in RGB order

        Returns
        ───────
        1-D float32 NumPy array, L2-normalised
        """
        bundle    = self._bundles[model_name]
        tfm       = self._transforms[model_name]
        pil_img   = Image.fromarray(face_rgb)
        tensor    = tfm(pil_img).unsqueeze(0).to(self.device)
        embedding = bundle.model(tensor)
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy()

    # ── Similarity helpers ────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised embedding vectors."""
        return float(np.dot(v1, v2))

    # ── Gallery operations ────────────────────────────────────────────────────

    def build_gallery(
        self,
        model_name: str,
        user_image_dir: str | os.PathLike,
    ) -> dict[str, list[np.ndarray]]:
        """
        Scan user_image_dir for sub-folders (one per registered user) and
        compute embeddings for every image found.

        Expected layout:
            user_image_dir/
                alice/
                    img_000.jpg
                    img_001.jpg
                bob/
                    img_000.jpg

        Returns
        ───────
        {user_name: [embedding_array, ...]}
        """
        gallery: dict[str, list[np.ndarray]] = {}
        root = Path(user_image_dir)
        for user_dir in sorted(root.iterdir()):
            if not user_dir.is_dir():
                continue
            embeddings = []
            for img_path in sorted(user_dir.glob("*.jpg")) + sorted(user_dir.glob("*.png")):
                try:
                    face_rgb = np.array(Image.open(img_path).convert("RGB"))
                    embeddings.append(self.embed(model_name, face_rgb))
                except Exception as exc:
                    print(f"[WARN] Could not embed {img_path}: {exc}")
            if embeddings:
                gallery[user_dir.name] = embeddings
        return gallery

    def identify(
        self,
        model_name: str,
        face_rgb: np.ndarray,
        gallery: dict[str, list[np.ndarray]],
    ) -> tuple[str, float]:
        """
        Find the closest match in the gallery using mean-embedding per user.

        Returns
        ───────
        (best_match_name, cosine_similarity_score)
        Returns ("unknown", -1.0) if the gallery is empty.
        """
        if not gallery:
            return "unknown", -1.0

        query = self.embed(model_name, face_rgb)
        best_name  = "unknown"
        best_score = -2.0

        for user_name, embeddings in gallery.items():
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            score = self.cosine_similarity(query, mean_emb)
            if score > best_score:
                best_score = score
                best_name  = user_name

        return best_name, float(best_score)


# ── Transform factory ─────────────────────────────────────────────────────────

def _make_transform(
    height: int,
    width: int,
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
