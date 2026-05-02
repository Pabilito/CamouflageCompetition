"""
ui/manual_align.py — Manual face-alignment dialog.

When MTCNN fails or its landmarks are obviously wrong, the user can open this
dialog to recreate the alignment by hand: place the five MTCNN landmarks
(left eye, right eye, nose, left-mouth corner, right-mouth corner) on the
full untouched original image, then preview and accept the resulting 112×112
aligned face.

The geometry exactly matches `core.face_aligner.FaceAligner`:

    1.  Compute eye angle from the two eye landmarks.
    2.  Rotate the full frame (and the landmarks) around the *face centre*
        — defined as the mean of the five landmarks — so the eyes are
        horizontal.
    3.  Build the tight bounding box of the rotated landmarks, expand it to
        a square, pad by 10 %, and crop.
    4.  Resize the crop to 112×112 with LANCZOS.

Computational notes:
  * The full-resolution rotated image is recomputed only when needed
    (after the user releases a marker), and only then is the LANCZOS
    output regenerated.
  * While the user is dragging, the preview is computed from a smaller
    cached working copy with BILINEAR resampling, so motion stays smooth
    even on multi-megapixel originals.

Public entry point:

    open_manual_aligner(parent, original_image, on_done, initial_landmarks=None)

`on_done(aligned_image | None)` is called exactly once: with the 112×112
PIL.Image when the user clicks "Apply", or with None if they cancel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

# ── Geometry constants — must match core/face_aligner.py ─────────────────────

OUTPUT_SIZE = 112        # final aligned crop side in pixels
PAD_RATIO   = 0.10       # 10 % padding around the landmark bounding box

# Five landmarks, in the same order MTCNN returns them. Names match the
# anatomical points MTCNN itself locates, so a manual placement is
# directly comparable to an automatic detection.
LANDMARK_NAMES = (
    "Left eye (pupil centre)",
    "Right eye (pupil centre)",
    "Nose tip",
    "Left mouth corner",
    "Right mouth corner",
)
# Short forms used in the instruction strip — fit the line width better.
LANDMARK_SHORT = (
    "left pupil",
    "right pupil",
    "tip of the nose",
    "left corner of the mouth",
    "right corner of the mouth",
)
LANDMARK_COLOURS = ("#4cc9f0", "#90ee90", "#ffd166", "#f72585", "#ff9f43")
NUM_LANDMARKS = 5

# Indices into the landmark list — keep in sync with LANDMARK_NAMES.
IDX_LEFT_EYE     = 0
IDX_RIGHT_EYE    = 1
IDX_NOSE         = 2
IDX_LEFT_MOUTH   = 3
IDX_RIGHT_MOUTH  = 4

# UI sizing
_DIALOG_TITLE      = "Manual Alignment"
_PREVIEW_DISPLAY   = 224     # the 112×112 result is upscaled 2× for visibility
_MARKER_RADIUS     = 7
_HIT_RADIUS        = 14      # how close a click needs to be to grab a marker


# ── Alignment maths ──────────────────────────────────────────────────────────
#
# This block mirrors core/face_aligner.FaceAligner.align() as closely as
# possible given that we have only the 5 landmarks, not MTCNN's bounding box:
#
#   * Production uses MTCNN's bbox centre as the rotation/crop centre, and
#     side = max(bbox_w, bbox_h) as the crop side. Crop spans
#     [centre - side/2 - pad, centre + side/2 + pad] with pad = 0.10·side, so
#     the total crop side is side·1.20.
#   * Production rotates with expand=False (rotated canvas is the same size
#     as the original) and BICUBIC.
#
# Without an MTCNN bbox we estimate `side` from the inter-eye distance using
# a calibrated multiplier — typical frontal MTCNN bboxes run ~2.5× the
# inter-eye distance on the larger axis. The user can scale this up or down
# with a slider in the dialog when their face has unusual proportions or
# they want a tighter / looser crop.

# Default ratio: side ≈ INTER_EYE_TO_SIDE_DEFAULT × inter_eye_distance.
INTER_EYE_TO_SIDE_DEFAULT = 2.5
INTER_EYE_TO_SIDE_MIN     = 1.5    # tightest the user can dial
INTER_EYE_TO_SIDE_MAX     = 3.5    # loosest the user can dial

# Offsets that nudge the crop centre, expressed as a fraction of the
# unpadded crop side and applied IN THE ROTATED FRAME (so "+y" is always
# "down toward the chin in the upright face", regardless of head tilt).
# The default vertical offset is slightly negative because the landmark
# centroid sits between the nose and the mouth — that's noticeably below
# the face's true centre, so the un-offset crop ends up with too much
# space below the chin and not enough above the eyes. Biasing the centre
# upward by ~10 % of the side compensates for that.
OFFSET_X_DEFAULT = 0.00
OFFSET_Y_DEFAULT = -0.10
OFFSET_MIN       = -0.30   # ±30 % of side covers any realistic recentring
OFFSET_MAX       = +0.30


@dataclass
class AlignmentGeom:
    """All the intermediate quantities needed to compute the aligned crop and
    to draw an informative overlay on the original image."""
    centre:        tuple[float, float]       # crop centre, in rotated-image coords
    angle_deg:     float                     # rotation applied to the image
    crop_corners_orig: list[tuple[float, float]]  # 4 corners of the crop, in
                                                  # original-image coords (a
                                                  # tilted square — used for
                                                  # the on-canvas overlay)
    side_unpadded: float                     # bbox side estimate (no padding)
    side_padded:   float                     # final crop side (= side · 1.20)
    rotation_centre: tuple[float, float]     # always the landmark centroid;
                                             # what we pass to PIL.rotate(center=…)


def _compute_geometry(landmarks: list[tuple[float, float]],
                      *,
                      side_multiplier: float = INTER_EYE_TO_SIDE_DEFAULT,
                      offset_x: float = OFFSET_X_DEFAULT,
                      offset_y: float = OFFSET_Y_DEFAULT
                      ) -> AlignmentGeom:
    """Pure-Python crop maths — no PIL ops, cheap to call on every drag tick.

    `side_multiplier` controls how big the crop is relative to the inter-eye
    distance. The default mirrors typical MTCNN bbox proportions.

    `offset_x`, `offset_y` shift the crop centre in the ROTATED (upright)
    frame, expressed as fractions of `side_unpadded`. +x = right in the
    upright face, +y = down. Defaults bias the crop slightly upward to
    compensate for the landmark centroid sitting below the true face centre.
    """
    centroid_x = sum(p[0] for p in landmarks) / NUM_LANDMARKS
    centroid_y = sum(p[1] for p in landmarks) / NUM_LANDMARKS

    le = landmarks[IDX_LEFT_EYE]
    re = landmarks[IDX_RIGHT_EYE]
    angle_rad = math.atan2(re[1] - le[1], re[0] - le[0])
    angle_deg = math.degrees(angle_rad)

    # Inter-eye distance in original (un-rotated) coords. The eye-to-eye
    # vector has the same length before and after rotation — we don't need
    # to rotate first.
    inter_eye = math.hypot(re[0] - le[0], re[1] - le[1])
    if inter_eye < 1e-3:
        # Degenerate placement (eyes coincident); fall back to landmark span.
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        inter_eye = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    side_unpadded = inter_eye * side_multiplier
    side_padded   = side_unpadded * (1.0 + 2.0 * PAD_RATIO)
    half          = side_padded / 2.0

    # PIL.rotate(center=…, expand=False) keeps the rotation centre fixed:
    # the pixel at the rotation centre stays at the same coordinates in the
    # rotated image. We always rotate around the centroid, so in the rotated
    # image the centroid is at (centroid_x, centroid_y).
    #
    # The user's offset is given in the rotated/upright frame (so "down"
    # always means "toward the chin", regardless of head tilt). In rotated-
    # image pixel coordinates, that's just a direct addition to the centroid:
    rot_cx = centroid_x + offset_x * side_unpadded
    rot_cy = centroid_y + offset_y * side_unpadded

    # The crop square in the rotated image is axis-aligned around (rot_cx,
    # rot_cy). To draw it as a tilted polygon on the original image we
    # back-project its four corners: rotate them around the centroid by
    # +angle_rad (the same direction the original image is tilted relative
    # to the upright face).
    cos_p = math.cos(angle_rad)
    sin_p = math.sin(angle_rad)
    rel_corners = [(-half, -half), (+half, -half),
                   (+half, +half), (-half, +half)]
    crop_corners_orig = []
    for rx, ry in rel_corners:
        # 1) corner in rotated-image absolute coords
        rcx = rot_cx + rx
        rcy = rot_cy + ry
        # 2) translate so centroid is the origin, rotate by +angle, translate back
        dxr = rcx - centroid_x
        dyr = rcy - centroid_y
        ox = centroid_x + cos_p * dxr - sin_p * dyr
        oy = centroid_y + sin_p * dxr + cos_p * dyr
        crop_corners_orig.append((ox, oy))

    return AlignmentGeom(
        centre=(rot_cx, rot_cy),         # crop centre in ROTATED-image coords
        angle_deg=angle_deg,
        crop_corners_orig=crop_corners_orig,
        side_unpadded=side_unpadded,
        side_padded=side_padded,
        rotation_centre=(centroid_x, centroid_y),
    )


def _align_image(image: Image.Image,
                 landmarks: list[tuple[float, float]],
                 *,
                 side_multiplier: float = INTER_EYE_TO_SIDE_DEFAULT,
                 offset_x: float = OFFSET_X_DEFAULT,
                 offset_y: float = OFFSET_Y_DEFAULT,
                 fast: bool = False) -> Image.Image:
    """Produce the final 112×112 aligned face.

    Mirrors core/face_aligner.FaceAligner.align(): rotate the full image
    around (cx, cy) with expand=False, then crop a square of `side_padded`
    centred on (cx, cy), then resize to 112×112. The crop centre includes
    the user's offset (see _compute_geometry).

    `fast=True` swaps in cheaper resampling for the live preview during
    landmark dragging; `fast=False` is the production-quality path.

    Note: PIL's `rotate()` accepts NEAREST / BILINEAR / BICUBIC only — it
    rejects LANCZOS, which is reserved for `resize()`. We use BICUBIC for
    rotation (matches the production aligner) and LANCZOS for the final
    downscale.
    """
    geom = _compute_geometry(landmarks,
                             side_multiplier=side_multiplier,
                             offset_x=offset_x,
                             offset_y=offset_y)
    if fast:
        rotate_filter = Image.BILINEAR
        resize_filter = Image.BILINEAR
    else:
        rotate_filter = Image.BICUBIC
        resize_filter = Image.LANCZOS

    cx, cy = geom.centre                # crop centre, in rotated-image coords
    rcx, rcy = geom.rotation_centre     # = landmark centroid, the rotation pivot
    rotated = image.rotate(
        geom.angle_deg,
        resample=rotate_filter,
        center=(rcx, rcy),
        expand=False,
    )
    half = geom.side_padded / 2.0
    iw, ih = rotated.size
    # Match production: clamp to image bounds, then re-square if clamping
    # produced a non-square box. Out-of-bounds pixels are not filled here —
    # production never lets the crop go past the image edge either.
    rx1 = max(0,  int(round(cx - half)))
    ry1 = max(0,  int(round(cy - half)))
    rx2 = min(iw, int(round(cx + half)))
    ry2 = min(ih, int(round(cy + half)))
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

    crop = rotated.crop((rx1, ry1, rx2, ry2))
    return crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), resize_filter)


# ── Dialog ───────────────────────────────────────────────────────────────────

class ManualAlignDialog(ctk.CTkToplevel):
    """Modal dialog for hand-placing the five MTCNN landmarks."""

    def __init__(self, parent, original: Image.Image,
                 on_done: Callable[[Image.Image | None], None],
                 initial_landmarks: list[tuple[float, float]] | None = None):
        super().__init__(parent)
        self.title(_DIALOG_TITLE)
        self.configure(fg_color="#0d0d1a")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._original = original
        self._on_done  = on_done

        # Landmarks in *original-image* pixel coordinates. `None` until placed.
        self._landmarks: list[tuple[float, float] | None] = (
            list(initial_landmarks) + [None] * (NUM_LANDMARKS - len(initial_landmarks))
            if initial_landmarks else [None] * NUM_LANDMARKS
        )
        self._next_idx        = self._first_unfilled()
        self._dragging_idx:   int | None = None
        self._image_scale     = 1.0   # display→original scale factor
        self._image_offset    = (0, 0)
        self._preview_photo   = None
        self._canvas_photo    = None
        self._canvas_image_id = None
        self._is_dragging     = False
        # User-adjustable crop scale (multiplier on the inter-eye distance).
        # Default mirrors typical MTCNN bbox proportions; the slider lets the
        # user tighten or loosen the crop when their face has unusual ratios.
        self._scale_value     = INTER_EYE_TO_SIDE_DEFAULT
        # Crop-centre offsets in the rotated/upright frame, as a fraction of
        # `side_unpadded`. Default vertical bias is slightly negative so the
        # crop sits a little higher than the landmark centroid (which is
        # below the true face centre).
        self._offset_x_value  = OFFSET_X_DEFAULT
        self._offset_y_value  = OFFSET_Y_DEFAULT
        self._scale_settle_after: str | None = None  # debounce handle

        # Pick a sensible window size: ~80 % of screen, capped.
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        win_w = min(int(sw * 0.85), 1280)
        win_h = min(int(sh * 0.85), 820)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(720, 500)

        self._build_ui()
        # Defer initial render until the canvas has a real size.
        self.after(50, self._initial_render)

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=4)
        self.columnconfigure(1, weight=0, minsize=280)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # Top instruction strip spans both columns.
        top = ctk.CTkFrame(self, fg_color="#1e1e2e", corner_radius=0, height=44)
        top.grid(row=0, column=0, columnspan=2, sticky="nsew")
        top.grid_propagate(False)
        top.columnconfigure(0, weight=1)
        self._instruction = ctk.CTkLabel(
            top, text="", font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#ffd166", anchor="w")
        self._instruction.grid(row=0, column=0, padx=14, pady=10, sticky="w")

        # Left: image canvas.
        canvas_frame = ctk.CTkFrame(self, fg_color="#0d0d1a", corner_radius=0)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(canvas_frame, bg="#0d0d1a", highlightthickness=0,
                                 cursor="crosshair")
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Right: landmark legend + preview + buttons.
        side = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=8)
        side.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        side.columnconfigure(0, weight=1)

        ctk.CTkLabel(side, text="Landmarks",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        # One row per landmark with its colour swatch and "(set)/(unset)" tag.
        self._legend_rows: list[tuple[ctk.CTkLabel, ctk.CTkLabel]] = []
        for i, (name, colour) in enumerate(zip(LANDMARK_NAMES, LANDMARK_COLOURS)):
            row = ctk.CTkFrame(side, fg_color="transparent")
            row.grid(row=1 + i, column=0, padx=12, pady=2, sticky="ew")
            row.columnconfigure(1, weight=1)
            swatch = tk.Canvas(row, width=14, height=14, bg="#1a1a2e",
                               highlightthickness=0)
            swatch.create_oval(2, 2, 12, 12, fill=colour, outline="")
            swatch.grid(row=0, column=0, padx=(0, 8))
            name_lbl = ctk.CTkLabel(row, text=f"{i + 1}. {name}",
                                    font=ctk.CTkFont(size=11),
                                    text_color="#ddd", anchor="w")
            name_lbl.grid(row=0, column=1, sticky="w")
            state_lbl = ctk.CTkLabel(row, text="—", text_color="#666",
                                     font=ctk.CTkFont(size=10), anchor="e")
            state_lbl.grid(row=0, column=2, sticky="e")
            self._legend_rows.append((name_lbl, state_lbl))

        ctk.CTkFrame(side, height=1, fg_color="#333").grid(
            row=1 + NUM_LANDMARKS, column=0, sticky="ew", padx=12, pady=8)

        # ── Crop-scale slider ────────────────────────────────────────────────
        # Lets the user widen or tighten the crop without moving the
        # landmarks. The default mirrors typical MTCNN bbox proportions; the
        # min/max give roughly "eyes-only" up to "full head" crops.
        scale_header = ctk.CTkFrame(side, fg_color="transparent")
        scale_header.grid(row=2 + NUM_LANDMARKS, column=0,
                          padx=12, pady=(2, 0), sticky="ew")
        scale_header.columnconfigure(0, weight=1)
        ctk.CTkLabel(scale_header, text="Crop scale",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#00d4ff", anchor="w").grid(
            row=0, column=0, sticky="w")
        self._scale_value_lbl = ctk.CTkLabel(
            scale_header,
            text=f"{self._scale_value:.2f}× eye-dist",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#00d4ff", anchor="e")
        self._scale_value_lbl.grid(row=0, column=1, sticky="e")

        self._scale_slider = ctk.CTkSlider(
            side,
            from_=INTER_EYE_TO_SIDE_MIN,
            to=INTER_EYE_TO_SIDE_MAX,
            number_of_steps=int((INTER_EYE_TO_SIDE_MAX
                                 - INTER_EYE_TO_SIDE_MIN) * 20),  # 0.05 step
            command=self._on_scale_changed)
        self._scale_slider.set(self._scale_value)
        self._scale_slider.grid(row=3 + NUM_LANDMARKS, column=0,
                                padx=12, pady=(2, 8), sticky="ew")

        # ── Crop-centre offset sliders ───────────────────────────────────────
        # Two sliders that nudge the crop centre in the upright (rotated)
        # frame, expressed as a fraction of the crop side. The vertical one
        # has a non-zero default so the un-touched preview already sits at a
        # plausible centre rather than too low.
        offset_header = ctk.CTkFrame(side, fg_color="transparent")
        offset_header.grid(row=4 + NUM_LANDMARKS, column=0,
                           padx=12, pady=(2, 0), sticky="ew")
        offset_header.columnconfigure(0, weight=1)
        ctk.CTkLabel(offset_header, text="Crop offset",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#00d4ff", anchor="w").grid(
            row=0, column=0, sticky="w")
        self._offset_value_lbl = ctk.CTkLabel(
            offset_header,
            text=self._format_offset_label(),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#00d4ff", anchor="e")
        self._offset_value_lbl.grid(row=0, column=1, sticky="e")

        # Horizontal offset (left ↔ right in the upright face).
        h_row = ctk.CTkFrame(side, fg_color="transparent")
        h_row.grid(row=5 + NUM_LANDMARKS, column=0,
                   padx=12, pady=(2, 2), sticky="ew")
        h_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(h_row, text="↔",
                     font=ctk.CTkFont(size=12), text_color="#888",
                     width=18).grid(row=0, column=0, sticky="w")
        self._offset_x_slider = ctk.CTkSlider(
            h_row, from_=OFFSET_MIN, to=OFFSET_MAX,
            number_of_steps=int((OFFSET_MAX - OFFSET_MIN) * 100),  # 0.01 step
            command=self._on_offset_x_changed)
        self._offset_x_slider.set(self._offset_x_value)
        self._offset_x_slider.grid(row=0, column=1, sticky="ew", padx=4)

        # Vertical offset (up ↔ down in the upright face).
        v_row = ctk.CTkFrame(side, fg_color="transparent")
        v_row.grid(row=6 + NUM_LANDMARKS, column=0,
                   padx=12, pady=(2, 8), sticky="ew")
        v_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(v_row, text="↕",
                     font=ctk.CTkFont(size=12), text_color="#888",
                     width=18).grid(row=0, column=0, sticky="w")
        self._offset_y_slider = ctk.CTkSlider(
            v_row, from_=OFFSET_MIN, to=OFFSET_MAX,
            number_of_steps=int((OFFSET_MAX - OFFSET_MIN) * 100),
            command=self._on_offset_y_changed)
        self._offset_y_slider.set(self._offset_y_value)
        self._offset_y_slider.grid(row=0, column=1, sticky="ew", padx=4)

        ctk.CTkLabel(side, text="Aligned preview (112×112)",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#00d4ff").grid(
            row=7 + NUM_LANDMARKS, column=0, padx=12, pady=(2, 4), sticky="w")

        # Preview canvas — fixed size so layout doesn't jump as the user drags.
        self._preview_canvas = tk.Canvas(
            side, width=_PREVIEW_DISPLAY, height=_PREVIEW_DISPLAY,
            bg="#0d0d1a", highlightthickness=1, highlightbackground="#333")
        self._preview_canvas.grid(row=8 + NUM_LANDMARKS, column=0,
                                  padx=12, pady=(0, 8))
        self._preview_canvas.create_text(
            _PREVIEW_DISPLAY / 2, _PREVIEW_DISPLAY / 2,
            text="Place all 5 landmarks\nto see preview",
            fill="#555", font=("Segoe UI", 10), justify="center")

        # Buttons strip at the bottom of the side panel.
        btns = ctk.CTkFrame(side, fg_color="transparent")
        btns.grid(row=9 + NUM_LANDMARKS, column=0, padx=12, pady=(4, 12),
                  sticky="ew")
        btns.columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btns, text="↺  Reset", fg_color="#3a3a5a",
                      hover_color="#4a4a6a",
                      command=self._reset).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ctk.CTkButton(btns, text="✗  Cancel", fg_color="#5a0000",
                      hover_color="#8b0000",
                      command=self._cancel).grid(
            row=1, column=0, sticky="ew", padx=(0, 4))
        self._apply_btn = ctk.CTkButton(
            btns, text="✓  Apply", fg_color="#004d00", hover_color="#007700",
            state="disabled", command=self._apply)
        self._apply_btn.grid(row=1, column=1, sticky="ew", padx=(4, 0))

        # Bottom hint strip.
        hint = ctk.CTkLabel(
            self,
            text=("Click to place each landmark in order.   "
                  "Drag any marker to refine.   "
                  "Adjust scale & offset to retune the crop.   "
                  "Esc = cancel."),
            font=ctk.CTkFont(size=10), text_color="#888")
        hint.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self.bind("<Escape>", lambda _e: self._cancel())

        self._refresh_legend()
        self._refresh_instruction()

    # ── Image rendering ──────────────────────────────────────────────────────

    def _initial_render(self) -> None:
        # Force a Configure-driven render now that the canvas has a real size.
        self._render_canvas()

    def _on_canvas_resize(self, _event) -> None:
        self._render_canvas()

    def _render_canvas(self) -> None:
        """Fit the original image into the canvas, recompute scale/offset, and
        redraw the markers and overlay."""
        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())
        if cw < 20 or ch < 20:
            return
        iw, ih = self._original.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        # Use NEAREST when the image is being downscaled significantly — it's
        # 5–10× faster than LANCZOS and quality doesn't matter for landmark
        # placement. Use BILINEAR for mild scaling.
        resample = Image.BILINEAR if scale > 0.5 else Image.NEAREST
        display = self._original.resize((nw, nh), resample)
        photo = ImageTk.PhotoImage(display)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self._image_scale  = scale
        self._image_offset = (ox, oy)
        self._canvas_photo = photo  # keep alive

        self._canvas.delete("all")
        self._canvas.create_image(ox, oy, anchor="nw", image=photo,
                                  tags=("bg",))
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        """Draw the markers, the connecting lines, and the crop preview box."""
        self._canvas.delete("overlay")
        ox, oy = self._image_offset
        s = self._image_scale

        def to_canvas(px: float, py: float) -> tuple[float, float]:
            return ox + px * s, oy + py * s

        # If all 5 are placed, draw the crop square in original-image space
        # so the user can see exactly which region will be saved as the
        # 112×112 face. This is the same square produced at the end of the
        # alignment pipeline, just back-projected through the rotation so it
        # appears tilted on the un-rotated image.
        if all(p is not None for p in self._landmarks):
            geom = _compute_geometry(
                self._landmarks,                  # type: ignore[arg-type]
                side_multiplier=self._scale_value,
                offset_x=self._offset_x_value,
                offset_y=self._offset_y_value)
            pts_canvas = []
            for (x, y) in geom.crop_corners_orig:
                cx_, cy_ = to_canvas(x, y)
                pts_canvas.extend([cx_, cy_])
            self._canvas.create_polygon(
                *pts_canvas, outline="#00d4ff", fill="", width=2,
                tags=("overlay",))

        # Connecting lines: eyes-line and eye-mouth quadrilateral, drawn faint
        # so they don't dominate the image.
        if (self._landmarks[IDX_LEFT_EYE] is not None
                and self._landmarks[IDX_RIGHT_EYE] is not None):
            x1, y1 = to_canvas(*self._landmarks[IDX_LEFT_EYE])
            x2, y2 = to_canvas(*self._landmarks[IDX_RIGHT_EYE])
            self._canvas.create_line(
                x1, y1, x2, y2, fill="#888", width=1, dash=(4, 3),
                tags=("overlay",))

        # Markers themselves — drawn last so they sit on top.
        for i, pt in enumerate(self._landmarks):
            if pt is None:
                continue
            x, y = to_canvas(*pt)
            colour = LANDMARK_COLOURS[i]
            r = _MARKER_RADIUS
            self._canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline=colour, width=2, fill="",
                tags=("overlay", f"marker_{i}"))
            # Crosshair through the centre — much more precise than just a dot.
            self._canvas.create_line(x - r - 3, y, x + r + 3, y,
                                     fill=colour, width=1, tags=("overlay",))
            self._canvas.create_line(x, y - r - 3, x, y + r + 3,
                                     fill=colour, width=1, tags=("overlay",))
            self._canvas.create_text(
                x + r + 4, y - r - 4, anchor="sw", text=str(i + 1),
                fill=colour, font=("Segoe UI", 9, "bold"),
                tags=("overlay",))

    # ── Click / drag handling ────────────────────────────────────────────────

    def _canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        ox, oy = self._image_offset
        s = self._image_scale or 1.0
        return (cx - ox) / s, (cy - oy) / s

    def _hit_test(self, cx: float, cy: float) -> int | None:
        """Return the index of the nearest existing marker within hit radius,
        or None if none qualifies."""
        best_idx, best_d2 = None, _HIT_RADIUS * _HIT_RADIUS
        for i, pt in enumerate(self._landmarks):
            if pt is None:
                continue
            ox, oy = self._image_offset
            s = self._image_scale
            mx, my = ox + pt[0] * s, oy + pt[1] * s
            d2 = (mx - cx) ** 2 + (my - cy) ** 2
            if d2 <= best_d2:
                best_d2, best_idx = d2, i
        return best_idx

    def _inside_image(self, ix: float, iy: float) -> bool:
        w, h = self._original.size
        return 0 <= ix < w and 0 <= iy < h

    def _on_canvas_click(self, event) -> None:
        # If a marker is under the cursor, start dragging it. Otherwise,
        # if there are still unplaced markers, drop the next one here.
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self._dragging_idx = hit
            self._is_dragging  = True
            return
        if self._next_idx is None:
            return  # all placed and click was off-target — nothing to do
        ix, iy = self._canvas_to_image(event.x, event.y)
        if not self._inside_image(ix, iy):
            return
        self._landmarks[self._next_idx] = (ix, iy)
        self._next_idx = self._first_unfilled()
        self._refresh_legend()
        self._refresh_instruction()
        self._draw_overlay()
        self._update_preview(during_drag=False)

    def _on_canvas_drag(self, event) -> None:
        if self._dragging_idx is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        # Clamp to image bounds so users can't drag a landmark off the canvas.
        w, h = self._original.size
        ix = max(0.0, min(w - 1.0, ix))
        iy = max(0.0, min(h - 1.0, iy))
        self._landmarks[self._dragging_idx] = (ix, iy)
        self._draw_overlay()
        self._update_preview(during_drag=True)

    def _on_canvas_release(self, _event) -> None:
        if self._dragging_idx is None:
            return
        self._dragging_idx = None
        self._is_dragging  = False
        # Final, high-quality preview now that the user has settled.
        self._update_preview(during_drag=False)

    # ── State / preview ──────────────────────────────────────────────────────

    def _first_unfilled(self) -> int | None:
        for i, pt in enumerate(self._landmarks):
            if pt is None:
                return i
        return None

    def _refresh_instruction(self) -> None:
        if self._next_idx is None:
            self._instruction.configure(
                text="✓  All landmarks placed — drag any marker to refine, "
                     "or adjust the crop scale, then click Apply.",
                text_color="#00ff9f")
        else:
            short  = LANDMARK_SHORT[self._next_idx]
            colour = LANDMARK_COLOURS[self._next_idx]
            # 'Left' / 'Right' here are from the VIEWER's perspective — i.e.
            # left side of the picture, which is the subject's right side if
            # they face the camera. This matches the MTCNN convention.
            self._instruction.configure(
                text=f"Step {self._next_idx + 1}/{NUM_LANDMARKS}: "
                     f"click the {short}    "
                     f"(left / right = your view of the picture)",
                text_color=colour)

    def _refresh_legend(self) -> None:
        for i, (_, state_lbl) in enumerate(self._legend_rows):
            if self._landmarks[i] is None:
                state_lbl.configure(text="unset", text_color="#666")
            else:
                state_lbl.configure(text="set", text_color="#00ff9f")
        # Apply button enabled iff all five are placed.
        if all(p is not None for p in self._landmarks):
            self._apply_btn.configure(state="normal")
        else:
            self._apply_btn.configure(state="disabled")

    def _format_offset_label(self) -> str:
        """Compact textual readout: '0.00 / -0.10' (x then y, both as
        fractions of the unpadded crop side)."""
        return (f"{self._offset_x_value:+.2f} / "
                f"{self._offset_y_value:+.2f}")

    def _live_repreview(self) -> None:
        """Update the preview at 'fast' quality, then schedule a single
        high-quality re-render once the slider settles. Shared by all three
        sliders (scale, offset-x, offset-y)."""
        if not all(p is not None for p in self._landmarks):
            return
        self._update_preview(during_drag=True)
        if self._scale_settle_after is not None:
            try:
                self.after_cancel(self._scale_settle_after)
            except Exception:
                pass
        self._scale_settle_after = self.after(
            150, lambda: self._update_preview(during_drag=False))

    def _on_scale_changed(self, value) -> None:
        self._scale_value = round(float(value), 2)
        self._scale_value_lbl.configure(
            text=f"{self._scale_value:.2f}× eye-dist")
        self._draw_overlay()           # tilted square on the canvas resizes
        self._live_repreview()

    def _on_offset_x_changed(self, value) -> None:
        self._offset_x_value = round(float(value), 2)
        self._offset_value_lbl.configure(text=self._format_offset_label())
        self._draw_overlay()
        self._live_repreview()

    def _on_offset_y_changed(self, value) -> None:
        self._offset_y_value = round(float(value), 2)
        self._offset_value_lbl.configure(text=self._format_offset_label())
        self._draw_overlay()
        self._live_repreview()

    def _update_preview(self, *, during_drag: bool) -> None:
        if not all(p is not None for p in self._landmarks):
            return
        try:
            aligned = _align_image(self._original,
                                   self._landmarks,  # type: ignore[arg-type]
                                   side_multiplier=self._scale_value,
                                   offset_x=self._offset_x_value,
                                   offset_y=self._offset_y_value,
                                   fast=during_drag)
            # Upscale to the preview canvas size so the user can actually see
            # what's going on. NEAREST during drag for speed, BILINEAR on
            # release for a sharper image.
            disp_resample = Image.NEAREST if during_drag else Image.BILINEAR
            display = aligned.resize((_PREVIEW_DISPLAY, _PREVIEW_DISPLAY),
                                     disp_resample)
            photo = ImageTk.PhotoImage(display)
            self._preview_photo = photo
            self._preview_canvas.delete("all")
            self._preview_canvas.create_image(0, 0, anchor="nw", image=photo)
        except Exception as exc:                                  # pragma: no cover
            # Geometric edge cases (e.g. all five landmarks coincident) can
            # produce a degenerate crop_box. Don't crash the dialog.
            self._preview_canvas.delete("all")
            self._preview_canvas.create_text(
                _PREVIEW_DISPLAY / 2, _PREVIEW_DISPLAY / 2,
                text=f"Preview error:\n{exc}",
                fill="#ff6b6b", font=("Segoe UI", 10), justify="center")

    # ── Buttons ──────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._landmarks    = [None] * NUM_LANDMARKS
        self._next_idx     = 0
        self._dragging_idx = None
        # Reset all sliders to their defaults so the dialog feels fresh.
        self._scale_value    = INTER_EYE_TO_SIDE_DEFAULT
        self._offset_x_value = OFFSET_X_DEFAULT
        self._offset_y_value = OFFSET_Y_DEFAULT
        self._scale_slider.set(self._scale_value)
        self._offset_x_slider.set(self._offset_x_value)
        self._offset_y_slider.set(self._offset_y_value)
        self._scale_value_lbl.configure(
            text=f"{self._scale_value:.2f}× eye-dist")
        self._offset_value_lbl.configure(text=self._format_offset_label())
        self._preview_canvas.delete("all")
        self._preview_canvas.create_text(
            _PREVIEW_DISPLAY / 2, _PREVIEW_DISPLAY / 2,
            text="Place all 5 landmarks\nto see preview",
            fill="#555", font=("Segoe UI", 10), justify="center")
        self._refresh_legend()
        self._refresh_instruction()
        self._draw_overlay()

    def _apply(self) -> None:
        if not all(p is not None for p in self._landmarks):
            return
        try:
            aligned = _align_image(self._original,
                                   self._landmarks,  # type: ignore[arg-type]
                                   side_multiplier=self._scale_value,
                                   offset_x=self._offset_x_value,
                                   offset_y=self._offset_y_value,
                                   fast=False)
        except Exception as exc:
            self._instruction.configure(
                text=f"⚠  Could not produce aligned crop: {exc}",
                text_color="#ff6b6b")
            return
        self._finish(aligned)

    def _cancel(self) -> None:
        self._finish(None)

    def _finish(self, result: Image.Image | None) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        cb, self._on_done = self._on_done, None
        self.destroy()
        if cb is not None:
            cb(result)


# ── Convenience wrapper ──────────────────────────────────────────────────────

def open_manual_aligner(parent,
                        original_image: Image.Image,
                        on_done: Callable[[Image.Image | None], None],
                        initial_landmarks: list[tuple[float, float]] | None = None
                        ) -> ManualAlignDialog:
    """Open the manual alignment dialog. `on_done` is called with the 112×112
    aligned PIL.Image on success, or None on cancel."""
    return ManualAlignDialog(parent, original_image, on_done,
                             initial_landmarks=initial_landmarks)
