"""
core/face_aligner.py
─────────────────────────────────────────────────────────────────────────────
Face detection + landmark-based roll correction + crop for face recognition.

Pipeline
────────
1. MTCNN detects the face bounding box and 5 landmarks (eyes, nose, mouth).
2. Roll angle is computed from the eye line and the full image is rotated
   around the face centre so the eyes are horizontal — no black corners
   because we rotate the whole frame, not just the crop.
3. The rotated image is cropped with 20% padding and made square.
4. Resized to 112×112 via LANCZOS.

Public API
──────────
  aligner = FaceAligner(device)
  result  = aligner.align(pil_image_rgb)   # → AlignResult

  result.success : bool
  result.image   : PIL Image (112×112) or None
  result.error   : str — non-empty if not success
  result.bbox    : (x1,y1,x2,y2) in original image coords, or None
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image

_OUT_SIZE = 112


@dataclass
class AlignResult:
    success: bool
    image:   Image.Image | None
    error:   str
    bbox:    tuple | None


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
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                keep_all=False,
                device=self._device,
            )
        return self._mtcnn

    def align(self, image: Image.Image) -> AlignResult:
        """
        Detect face, correct roll via eye-line angle, crop square, resize 112×112.
        """
        try:
            mtcnn = self._get_mtcnn()

            # landmarks shape: (N, 5, 2) — left-eye, right-eye, nose, mouth×2
            boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return AlignResult(success=False, image=None,
                                   error="No face detected in the image.",
                                   bbox=None)

            best  = int(np.argmax(probs))
            box   = boxes[best].astype(np.float64)
            lmk   = landmarks[best].astype(np.float64)   # (5, 2)
            bbox  = tuple(int(v) for v in box)

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

            return AlignResult(success=True, image=aligned, error="", bbox=bbox)

        except Exception as exc:
            return AlignResult(success=False, image=None,
                               error=f"Detector error: {exc}", bbox=None)
