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
        Detect the largest face, crop the bounding box (with padding),
        and resize to 112×112. No rotation applied.
        """
        try:
            mtcnn = self._get_mtcnn()

            boxes, probs = mtcnn.detect(image, landmarks=False)

            if boxes is None or len(boxes) == 0:
                return AlignResult(
                    success=False, image=None,
                    error="No face detected in the image.",
                    bbox=None)

            best = int(np.argmax(probs))
            box  = boxes[best]   # (x1, y1, x2, y2)
            bbox = tuple(int(v) for v in box)

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            # 20% padding on each side to include some context around the face
            pad   = max(w, h) * 0.20
            iw, ih = image.size
            cx1 = max(0,  int(x1 - pad))
            cy1 = max(0,  int(y1 - pad))
            cx2 = min(iw, int(x2 + pad))
            cy2 = min(ih, int(y2 + pad))

            # Make it square by expanding the shorter side
            cw = cx2 - cx1
            ch = cy2 - cy1
            if cw > ch:
                diff = cw - ch
                cy1 = max(0,  cy1 - diff // 2)
                cy2 = min(ih, cy1 + cw)
            elif ch > cw:
                diff = ch - cw
                cx1 = max(0,  cx1 - diff // 2)
                cx2 = min(iw, cx1 + ch)

            crop = image.crop((cx1, cy1, cx2, cy2))
            aligned = crop.resize((_OUT_SIZE, _OUT_SIZE), Image.LANCZOS)

            return AlignResult(success=True, image=aligned,
                               error="", bbox=bbox)

        except Exception as exc:
            return AlignResult(
                success=False, image=None,
                error=f"Detector error: {exc}",
                bbox=None)


