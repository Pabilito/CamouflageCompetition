"""
core/face_aligner.py
─────────────────────────────────────────────────────────────────────────────
Face detection + 5-landmark alignment for face recognition preprocessing.

Uses MTCNN (facenet-pytorch) for detection and landmark prediction, then
applies a similarity transform to warp the face to a canonical 112×112 pose
(the ArcFace / InsightFace reference template).

Public API
──────────
  aligner = FaceAligner(device)
  result  = aligner.align(pil_image_rgb)   # → AlignResult

  result.success      : bool
  result.image        : PIL Image (112×112 aligned face) or None
  result.error        : str description if failed
  result.bbox         : (x1,y1,x2,y2) of detected face, or None
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


# ── ArcFace / InsightFace 5-landmark reference template (112×112) ────────────
# Order: left-eye, right-eye, nose, left-mouth, right-mouth
_ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_OUT_SIZE = 112   # output face crop size


@dataclass
class AlignResult:
    success: bool
    image:   Image.Image | None   # aligned face if success
    error:   str                  # non-empty if not success
    bbox:    tuple | None         # (x1,y1,x2,y2) in original image coords


class FaceAligner:
    """
    Wraps MTCNN detection + similarity-transform alignment.
    Lazy-loads the MTCNN model on first use.
    """

    def __init__(self, device=None) -> None:
        import torch
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = None   # loaded on first call

    def _get_mtcnn(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                image_size=_OUT_SIZE,
                margin=0,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                keep_all=False,
                device=self._device,
            )
        return self._mtcnn

    def align(self, image: Image.Image) -> AlignResult:
        """
        Detect the largest face in `image` and return an aligned 112×112 crop.

        Parameters
        ──────────
        image : RGB PIL Image (any size)

        Returns
        ───────
        AlignResult — check .success before using .image
        """
        try:
            import torch
            mtcnn = self._get_mtcnn()

            # detect_landmarks returns (boxes, probs, landmarks)
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return AlignResult(
                    success=False, image=None,
                    error="No face detected in the image.",
                    bbox=None)

            # Pick the highest-confidence face
            best = int(np.argmax(probs))
            box  = boxes[best]
            lmk  = landmarks[best]   # shape (5, 2)
            bbox = tuple(int(v) for v in box)

            # Apply similarity transform aligning to ArcFace reference template
            aligned = _similarity_transform(image, lmk, _OUT_SIZE)

            return AlignResult(success=True, image=aligned,
                               error="", bbox=bbox)

        except Exception as exc:
            return AlignResult(
                success=False, image=None,
                error=f"Detector error: {exc}",
                bbox=None)


# ── Similarity transform ──────────────────────────────────────────────────────

def _similarity_transform(image: Image.Image,
                           src_pts: np.ndarray,
                           out_size: int) -> Image.Image:
    """
    Warp `image` so that `src_pts` (5×2 float32) maps onto the ArcFace
    reference landmarks at scale `out_size`.

    Uses a closed-form similarity transform (rotation + uniform scale + translation)
    computed via the Umeyama algorithm.
    """
    dst_pts = _ARCFACE_REF * (out_size / 112.0)
    M       = _umeyama(src_pts.astype(np.float64), dst_pts.astype(np.float64),
                       estimate_scale=True)

    # PIL affine transform expects the *inverse* 2×3 matrix in row-major order
    # PIL uses (a,b,c,d,e,f) where x' = ax+by+c, y' = dx+ey+f
    # M is 2×3 [R|t], so apply inverse
    import numpy as np
    M3 = np.vstack([M, [0, 0, 1]])
    Mi = np.linalg.inv(M3)[:2]   # 2×3 inverse
    coeffs = (Mi[0, 0], Mi[0, 1], Mi[0, 2],
              Mi[1, 0], Mi[1, 1], Mi[1, 2])

    warped = image.transform(
        (out_size, out_size),
        Image.AFFINE,
        coeffs,
        resample=Image.BICUBIC,
    )
    return warped


def _umeyama(src: np.ndarray, dst: np.ndarray,
             estimate_scale: bool = True) -> np.ndarray:
    """
    Least-squares similarity transform between corresponding point sets.
    Returns a 2×3 matrix [R * scale | t].
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    assert src.shape == dst.shape
    n, m = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = (src_c ** 2).sum() / n
    cov       = dst_c.T @ src_c / n

    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[m - 1, m - 1] = -1

    R = U @ D @ Vt
    scale = (S @ D) / sigma_src if estimate_scale else 1.0
    t     = mu_dst - scale * R @ mu_src

    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = scale * R
    M[:2,  2] = t
    return M
