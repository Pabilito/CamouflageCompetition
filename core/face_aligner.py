"""
core/face_aligner.py
─────────────────────────────────────────────────────────────────────────────
Face detection + landmark-based roll correction + crop for face recognition.

Pipeline
────────
1. MTCNN detects all face bounding boxes and 5 landmarks (eyes, nose, mouth).
2. The largest face (by bbox area) is selected, with an optional confidence
   filter to suppress weak background detections.
3. Roll angle is computed from the eye line and the full image is rotated
   around the face centre so the eyes are horizontal — no black corners
   because we rotate the whole frame, not just the crop.
4. The rotated image is cropped with 10% padding and made square.
5. Resized to 112×112 via LANCZOS.

Public API
──────────
  aligner = FaceAligner(device)
  result  = aligner.align(pil_image_rgb,
                          thresholds=[0.6, 0.7, 0.7],   # optional
                          select_largest=True)          # optional

  result.success    : bool
  result.image      : PIL Image (112×112) or None
  result.error      : str — non-empty if not success
  result.bbox       : (x1,y1,x2,y2) in original image coords, or None
  result.confidence : float in [0, 1] — MTCNN confidence for the picked face,
                      or None on failure
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image

_OUT_SIZE = 112

# MTCNN's three-stage cascade thresholds (P-Net, R-Net, O-Net). All three must
# be passed for a detection to survive. These match the upstream defaults.
_DEFAULT_THRESHOLDS = [0.6, 0.7, 0.7]

# When picking among multiple detections, suppress those below this prob so
# a tiny low-confidence background blob doesn't beat the real subject by
# area alone. If nothing survives the filter, we fall back to the single
# highest-confidence detection.
_CONFIDENCE_FILTER = 0.9


@dataclass
class AlignResult:
    success:    bool
    image:      Image.Image | None
    error:      str
    bbox:       tuple | None
    confidence: float | None = None


class FaceAligner:

    def __init__(self, device=None) -> None:
        import torch
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = None

    def _get_mtcnn(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                image_size=_OUT_SIZE,
                margin=0,
                min_face_size=40,
                thresholds=list(_DEFAULT_THRESHOLDS),
                factor=0.709,
                post_process=False,
                keep_all=True,            # return all faces, we pick later
                device=self._device,
            )
        return self._mtcnn

    def align(
        self,
        image: Image.Image,
        thresholds: list[float] | None = None,
        select_largest: bool = True,
    ) -> AlignResult:
        """
        Detect face, correct roll via eye-line angle, crop square, resize 112×112.

        Parameters
        ──────────
        image          : input PIL image (RGB)
        thresholds     : optional override for MTCNN's three-stage thresholds.
                         None → use defaults [0.6, 0.7, 0.7]. Lower values
                         make detection more permissive (useful for adversarial
                         camouflage that may cause weak detections).
        select_largest : if True (default), pick the largest face by bbox area
                         among detections with prob ≥ 0.9. If no detection
                         passes the confidence filter, fall back to the single
                         highest-confidence detection. If False, always pick
                         the highest-confidence detection.
        """
        try:
            mtcnn = self._get_mtcnn()
            # MTCNN reads self.thresholds at call time, so we can mutate it
            # cheaply to switch between strict and relaxed modes without
            # rebuilding the network.
            mtcnn.thresholds = list(thresholds) if thresholds is not None \
                               else list(_DEFAULT_THRESHOLDS)

            # landmarks shape: (N, 5, 2) — left-eye, right-eye, nose, mouth×2
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return AlignResult(success=False, image=None,
                                   error="No face detected in the image.",
                                   bbox=None, confidence=None)

            # ── Pick which face to use ────────────────────────────────────────
            # Some entries can be None when keep_all=True returns nothing —
            # filter those out first.
            valid = [i for i in range(len(boxes))
                     if boxes[i] is not None and probs[i] is not None]
            if not valid:
                return AlignResult(success=False, image=None,
                                   error="No face detected in the image.",
                                   bbox=None, confidence=None)

            if select_largest:
                # Filter by confidence first to suppress weak background blobs
                confident = [i for i in valid if probs[i] >= _CONFIDENCE_FILTER]
                pool = confident if confident else valid
                # Pick largest by bbox area
                def _area(i):
                    b = boxes[i]
                    return float(b[2] - b[0]) * float(b[3] - b[1])
                best = max(pool, key=_area)
            else:
                best = max(valid, key=lambda i: probs[i])

            box  = boxes[best].astype(np.float64)
            lmk  = landmarks[best].astype(np.float64)   # (5, 2)
            bbox = tuple(int(v) for v in box)
            conf = float(probs[best])

            # ── Step 1: compute roll from left-eye → right-eye vector ─────────
            left_eye  = lmk[0]   # (x, y)
            right_eye = lmk[1]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            # atan2(dy, dx) in image coords (Y down):
            # positive dy = right eye lower = CW tilt → need CCW correction
            # PIL.rotate positive = CCW, so pass angle_deg directly (no negation)
            angle_deg = math.degrees(math.atan2(dy, dx))

            # ── Step 2: rotate the full image around the face centre ──────────
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            rotated = image.rotate(
                angle_deg,         # positive = CCW = corrects CW tilt
                resample=Image.BICUBIC,
                center=(cx, cy),
                expand=False,
                fillcolor=None,
            )

            # ── Step 3: update box to rotated coords ──────────────────────────
            w = box[2] - box[0]
            h = box[3] - box[1]
            side = max(w, h)

            # ── Step 4: padded square crop from the rotated image ─────────────
            pad = side * 0.10   # 10% padding
            iw, ih = rotated.size
            rx1 = max(0,  int(cx - side / 2 - pad))
            ry1 = max(0,  int(cy - side / 2 - pad))
            rx2 = min(iw, int(cx + side / 2 + pad))
            ry2 = min(ih, int(cy + side / 2 + pad))

            # Force square
            cw = rx2 - rx1
            ch = ry2 - ry1
            if cw > ch:
                diff = cw - ch
                ry1 = max(0,  ry1 - diff // 2)
                ry2 = min(ih, ry1 + cw)
            elif ch > cw:
                diff = ch - cw
                rx1 = max(0,  rx1 - diff // 2)
                rx2 = min(iw, rx1 + ch)

            crop    = rotated.crop((rx1, ry1, rx2, ry2))
            aligned = crop.resize((_OUT_SIZE, _OUT_SIZE), Image.LANCZOS)

            return AlignResult(success=True, image=aligned, error="",
                               bbox=bbox, confidence=conf)

        except Exception as exc:
            return AlignResult(success=False, image=None,
                               error=f"Detector error: {exc}",
                               bbox=None, confidence=None)
