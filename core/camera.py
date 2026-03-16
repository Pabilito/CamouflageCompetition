"""
core/camera.py
─────────────────────────────────────────────────────────────────────────────
Thin OpenCV wrapper that the UI can use to drive the webcam.

Usage
─────
  cam = Camera(index=0)
  cam.open()

  frame_rgb = cam.read_frame()   # H×W×3 uint8 NumPy RGB array, or None
  cam.save_frame(frame_rgb, path)
  cam.close()

The Camera class is safe to use as a context manager:

  with Camera() as cam:
      frame = cam.read_frame()
"""

import os
from pathlib import Path

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap: cv2.VideoCapture | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self._index}. "
                "Check that no other process is using the webcam."
            )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── Frame I/O ─────────────────────────────────────────────────────────────

    def read_frame(self) -> np.ndarray | None:
        """
        Read one frame from the webcam.

        Returns
        ───────
        H×W×3 uint8 NumPy array in **RGB** order, or None on failure.
        """
        if not self.is_open:
            return None
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_frame(frame_rgb: np.ndarray, path: str | os.PathLike) -> None:
        """
        Save an RGB frame to disk as a JPEG.
        Creates parent directories if they do not exist.

        Uses PIL instead of cv2.imwrite so that Unicode paths (e.g. paths
        containing accented characters on Windows) are handled correctly.
        """
        from PIL import Image
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame_rgb).save(str(dest), format="JPEG", quality=95)

    @staticmethod
    def next_image_path(user_dir: str | os.PathLike) -> Path:
        """
        Return an auto-incremented path like user_dir/img_007.jpg
        without overwriting any existing file.
        """
        folder = Path(user_dir)
        folder.mkdir(parents=True, exist_ok=True)
        idx = 0
        while True:
            p = folder / f"img_{idx:03d}.jpg"
            if not p.exists():
                return p
            idx += 1
