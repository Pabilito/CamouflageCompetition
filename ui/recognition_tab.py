"""
ui/recognition_tab.py  —  Adversarial Pictures tab

Layout mirrors the Control Pictures tab:
  Left panel  — identity dropdown (for mated comparison) + scrollable thumbnail
                strip of saved adversarial images (clickable to enlarge)
  Right panel — live camera preview (with blue crop square) + model score bars

Adversarial images are saved to data/adversarial/<identity>/ on accept.
"""

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

from core.camera import Camera
from core.database import Database
from core.embeddings import EmbeddingEngine
from core.face_aligner import FaceAligner

_PREVIEW_SIZE = 420
_THUMB_SIZE   = 72
_BAR_H        = 14
_CROP_COLOUR  = "#00d4ff"

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

# LFW recognition accuracy (%) for each model, normalised to [0, 1] for the bar
ACCURACY_LFW: dict[str, float] = {
    "InceptionResnetCASIA":     90.18284 / 100,
    "InceptionResnetV1_CASIA":  90.18284 / 100,
    "IResNet_R18_Glint360K":    96.48238 / 100,
    "SwinFace":                 92.02094 / 100,
    "TransFace_S_Glint360K":    97.47524 / 100,
    "edgeface_base":            97.17714 / 100,
}


def _score_to_bar(score: float) -> float:
    return max(0.0, min(1.0, (score + 1) / 2))


def _score_colour(score: float, threshold: float | None) -> str:
    if threshold is None:
        return "#00ff9f" if score >= 0.5 else "#ffd166" if score >= 0.3 else "#ff6b6b"
    diff = score - threshold
    return "#00ff9f" if diff >= 0.05 else "#ffd166" if diff >= 0 else "#ff6b6b"


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h  = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _hq_path(image_path: str | Path) -> Path:
    """Return the companion high-quality PNG path for a given saved image path.

    Example: data/adversarial/alice/img_007.jpg → data/adversarial/alice/img_007_hq.png
    """
    p = Path(image_path)
    return p.with_name(p.stem + "_hq.png")


class RecognitionTab(ctk.CTkFrame):
    def __init__(self, parent, engine: EmbeddingEngine, db: Database,
                 data_dir: str, aligner=None):
        super().__init__(parent, fg_color="transparent")
        self._engine        = engine
        self._db            = db
        self._data_dir      = data_dir
        self._camera        = Camera(index=0)
        self._aligner       = aligner or FaceAligner()
        self._running       = False
        self._thread_gen    = 0
        self._latest_frame  = None
        self._galleries: dict = {}
        self._gallery_paths: dict = {}
        self._gallery_ready = False
        self._gallery_mtime: float = 0.0
        self._gallery_loading = False   # guard against concurrent loads
        self._photo_ref     = None
        self._score_widgets: dict = {}
        self._pending_frame:  Image.Image | None = None
        self._raw_frame:      Image.Image | None = None
        self._original_frame: Image.Image | None = None   # full untouched original (for HQ companion)
        self._pending_source: str = "camera"   # "camera" or "upload" — drives retake behaviour
        self._thumb_refs: list = []
        self._ranking_tab = None   # set by main.py after construction
        self._current_scores: dict[str, float | None] = {}  # last known score per model
        self._review_image: Image.Image | None = None       # image shown in review canvas

        self._build_ui()

    # ── Tab hooks ─────────────────────────────────────────────────────────────

    def on_tab_shown(self) -> None:
        user_dir = Path(self._data_dir) / "users"
        if self._gallery_ready and user_dir.exists():
            try:
                mtime = user_dir.stat().st_mtime
                if mtime == self._gallery_mtime:
                    return
            except OSError:
                pass
        self._reload_gallery()

    def on_tab_hidden(self) -> None:
        """Stop the camera when leaving this tab."""
        if self._running:
            self._camera.close()
            self._running = False
            self._thread_gen += 1
            self._cam_btn.configure(text="▶  Start Camera")
            self._capture_btn.configure(state="disabled")
            self._canvas.delete("all")
            self._status.configure(text="Camera stopped (tab switched).", text_color="#888")

    def _on_canvas_resize(self, event) -> None:
        size = min(event.width, event.height)
        if size > 10:
            self._canvas_size = size
        # If a review image is being shown (camera not live), redraw it at the new size
        if not self._running and self._review_image is not None:
            self._show_review_image(self._review_image)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1, minsize=260)
        self.columnconfigure(1, weight=4)
        self.rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════════════════════════════════
        # LEFT panel — identity + saved adversarial thumbnails
        # ═══════════════════════════════════════════════════════════════════════
        left = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Compare Against",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        self._identity_var = ctk.StringVar(value="—")
        self._identity_menu = ctk.CTkOptionMenu(
            left, variable=self._identity_var,
            values=["—"], width=210,
            command=self._on_identity_changed)
        self._identity_menu.grid(row=1, column=0, padx=14, pady=4, sticky="ew")

        # Gallery loading indicator
        self._gallery_status = ctk.CTkLabel(
            left,
            text="⏳  Loading…",
            font=ctk.CTkFont(size=10), text_color="#ffd166")
        self._gallery_status.grid(row=2, column=0, padx=14, pady=(2, 6), sticky="w")

        ctk.CTkFrame(left, height=1, fg_color="#333").grid(
            row=3, column=0, sticky="ew", padx=14, pady=4)

        ctk.CTkLabel(left, text="Saved Adversarial Images",
                     font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=4, column=0, padx=14, sticky="w")

        self._thumb_frame = ctk.CTkScrollableFrame(
            left, height=340, fg_color="transparent", label_text="")
        self._thumb_frame.grid(row=5, column=0, padx=10, pady=(4, 10), sticky="nsew")
        left.rowconfigure(5, weight=1)

        self._adv_count_label = ctk.CTkLabel(
            left, text="", font=ctk.CTkFont(size=11), text_color="#aaa")
        self._adv_count_label.grid(row=6, column=0, padx=14, pady=(0, 4), sticky="w")

        # Best score badge
        best_row = ctk.CTkFrame(left, fg_color="transparent")
        best_row.grid(row=7, column=0, padx=14, pady=(0, 12), sticky="ew")
        ctk.CTkLabel(best_row, text="Your best score:",
                     font=ctk.CTkFont(size=11), text_color="#888").pack(side="left")
        self._lbl_left_best = ctk.CTkLabel(
            best_row, text="—",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#aaa")
        self._lbl_left_best.pack(side="left", padx=(6, 0))

        # ═══════════════════════════════════════════════════════════════════════
        # RIGHT panel — camera preview + scores
        # ═══════════════════════════════════════════════════════════════════════
        right_outer = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        right_outer.grid(row=0, column=1, sticky="nsew")
        right_outer.columnconfigure(0, weight=1)
        right_outer.columnconfigure(1, weight=1)
        right_outer.rowconfigure(0, weight=1)

        # ── Camera column ──────────────────────────────────────────────────────
        cam_col = ctk.CTkFrame(right_outer, fg_color="transparent")
        cam_col.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)
        cam_col.columnconfigure(0, weight=1)
        cam_col.rowconfigure(2, weight=1)

        ctk.CTkLabel(cam_col, text="Live Camera",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, pady=(0, 4), sticky="w")
        ctk.CTkLabel(cam_col, text="Blue square = saved crop region",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=1, column=0, sticky="w")

        self._canvas_size = _PREVIEW_SIZE
        self._canvas = tk.Canvas(cam_col, bg="#0d0d1a", highlightthickness=0)
        self._canvas.grid(row=2, column=0, pady=(4, 4), sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        cam_btns = ctk.CTkFrame(cam_col, fg_color="transparent")
        cam_btns.grid(row=3, column=0, pady=4)
        self._cam_btn = ctk.CTkButton(
            cam_btns, text="▶  Start Camera", width=155, command=self._toggle_camera)
        self._cam_btn.pack(side="left", padx=4)
        self._capture_btn = ctk.CTkButton(
            cam_btns, text="📷  Capture", width=130,
            fg_color="#005f73", hover_color="#0a9396",
            state="disabled", command=self._capture_frame)
        self._capture_btn.pack(side="left", padx=4)
        ctk.CTkButton(
            cam_btns, text="📁  Upload Image", width=145,
            fg_color="#2a2a4a", hover_color="#3a3a6a",
            command=self._upload_image).pack(side="left", padx=4)

        # Detection-threshold slider (session-only).
        # The displayed value is MTCNN's O-Net (final-stage) confidence
        # threshold — i.e. detections must score ≥ this value to be returned.
        # Default 0.63 matches MTCNN's stock 0.7 multiplied by 0.9, slightly
        # more permissive than upstream defaults so adversarial faces still
        # detect. Internally we keep the original 6:7 ratio for P-Net.
        thr_frame = ctk.CTkFrame(cam_col, fg_color="transparent")
        thr_frame.grid(row=4, column=0, pady=(2, 2), sticky="ew")
        thr_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(thr_frame, text="Detection threshold:",
                     font=ctk.CTkFont(size=10), text_color="#888").grid(
            row=0, column=0, padx=(8, 6), sticky="w")
        self._thr_value = 0.63   # default O-Net threshold
        # Range 0.35–0.70 with 0.01 granularity → 35 steps.
        self._thr_slider = ctk.CTkSlider(
            thr_frame, from_=0.35, to=0.70, number_of_steps=35,
            command=self._on_threshold_changed)
        self._thr_slider.set(self._thr_value)
        self._thr_slider.grid(row=0, column=1, padx=4, sticky="ew")
        self._thr_value_lbl = ctk.CTkLabel(
            thr_frame, text=f"{self._thr_value:.2f}",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#00d4ff", width=44)
        self._thr_value_lbl.grid(row=0, column=2, padx=(2, 8))

        # Normal review buttons (shown after successful alignment)
        self._review_btns = ctk.CTkFrame(cam_col, fg_color="transparent")
        self._review_btns.grid(row=5, column=0, pady=4)
        self._review_btns.grid_remove()
        ctk.CTkLabel(self._review_btns, text="Accept aligned face?",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#ffd166").pack(side="left", padx=6)
        ctk.CTkButton(self._review_btns, text="✓  Accept & Log", width=130,
                      fg_color="#004d00", hover_color="#007700",
                      command=self._accept_frame).pack(side="left", padx=4)
        ctk.CTkButton(self._review_btns, text="✎  Realign", width=100,
                      fg_color="#7a5c00", hover_color="#a87b00",
                      text_color="#fff8dc",
                      command=self._open_realign).pack(side="left", padx=4)
        ctk.CTkButton(self._review_btns, text="✗  Retake", width=100,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._retake_frame).pack(side="left", padx=4)

        # Detection-failed fallback buttons
        self._fail_btns = ctk.CTkFrame(cam_col, fg_color="transparent")
        self._fail_btns.grid(row=5, column=0, pady=4)
        self._fail_btns.grid_remove()
        ctk.CTkLabel(self._fail_btns, text="⚠  No face detected",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#ff6b6b").pack(side="left", padx=6)
        ctk.CTkButton(self._fail_btns, text="💾  Save & Log Unprocessed", width=190,
                      fg_color="#5a3a00", hover_color="#8b5a00",
                      command=self._accept_unprocessed).pack(side="left", padx=4)
        ctk.CTkButton(self._fail_btns, text="✎  Realign", width=100,
                      fg_color="#7a5c00", hover_color="#a87b00",
                      text_color="#fff8dc",
                      command=self._open_realign).pack(side="left", padx=4)
        ctk.CTkButton(self._fail_btns, text="✗  Retake", width=100,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._retake_frame).pack(side="left", padx=4)

        self._status = ctk.CTkLabel(
            cam_col, text="Start camera to begin.",
            font=ctk.CTkFont(size=11), text_color="#888")
        self._status.grid(row=6, column=0, pady=(0, 2))

        # Persistent detection-confidence readout (separate from _status,
        # so it survives status overwrites like "Running inference...").
        self._confidence_lbl = ctk.CTkLabel(
            cam_col, text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#888")
        self._confidence_lbl.grid(row=7, column=0, pady=(0, 4))

        # ── Image Summary (shown after accept) ────────────────────────────────
        self._img_summary_frame = ctk.CTkFrame(
            cam_col, corner_radius=8, fg_color=("#0d1b2a", "#0d1b2a"))
        self._img_summary_frame.grid(row=8, column=0, sticky="ew", pady=(4, 6))
        self._img_summary_frame.columnconfigure(1, weight=1)
        self._img_summary_frame.grid_remove()

        def _img_stat(label: str, row: int):
            ctk.CTkLabel(self._img_summary_frame, text=label,
                         font=ctk.CTkFont(size=11), text_color="#888",
                         anchor="w").grid(row=row, column=0, padx=(12, 6), pady=3, sticky="w")
            val = ctk.CTkLabel(self._img_summary_frame, text="—",
                               font=ctk.CTkFont(size=12, weight="bold"),
                               text_color="#aaa", anchor="e")
            val.grid(row=row, column=1, padx=(0, 12), pady=3, sticky="e")
            return val

        ctk.CTkLabel(self._img_summary_frame, text="Image Summary",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, columnspan=2, padx=12, pady=(8, 2), sticky="w")

        self._lbl_img_name      = _img_stat("Attacker:",          1)
        self._lbl_avg_score     = _img_stat("Avg score:",         2)
        self._lbl_rank          = _img_stat("Image rank:",        3)

        ctk.CTkFrame(self._img_summary_frame, height=4, fg_color="transparent").grid(
            row=4, column=0, columnspan=2)

        # ── Scores column ──────────────────────────────────────────────────────
        scores_col = ctk.CTkScrollableFrame(
            right_outer, fg_color="transparent", label_text="")
        scores_col.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        scores_col.columnconfigure(0, weight=1)

        ctk.CTkLabel(scores_col, text="Similarity Scores",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, pady=(0, 6), sticky="w")

        for i, name in enumerate(self._engine.model_names):
            self._build_model_row(scores_col, name, row=i + 1)

        # ── Session Summary ────────────────────────────────────────────────────
        n = len(self._engine.model_names)

        sep = ctk.CTkFrame(scores_col, height=1, fg_color="#333")
        sep.grid(row=n + 1, column=0, sticky="ew", pady=(10, 6))

        summary_frame = ctk.CTkFrame(
            scores_col, corner_radius=8, fg_color=("#0d1b2a", "#0d1b2a"))
        summary_frame.grid(row=n + 2, column=0, sticky="ew", pady=(0, 4))
        summary_frame.columnconfigure(1, weight=1)

        def _stat_row(parent, label: str, row: int):
            ctk.CTkLabel(parent, text=label,
                         font=ctk.CTkFont(size=11), text_color="#888",
                         anchor="w").grid(row=row, column=0, padx=(12, 6), pady=3, sticky="w")
            val = ctk.CTkLabel(parent, text="—",
                               font=ctk.CTkFont(size=12, weight="bold"),
                               text_color="#aaa", anchor="e")
            val.grid(row=row, column=1, padx=(0, 12), pady=3, sticky="e")
            return val

        ctk.CTkLabel(summary_frame, text="Session Summary",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, columnspan=2, padx=12, pady=(8, 2), sticky="w")

        self._lbl_global_best   = _stat_row(summary_frame, "Current best score:", 1)
        self._lbl_your_best     = _stat_row(summary_frame, "Your best score:",    2)
        self._lbl_your_place    = _stat_row(summary_frame, "Your current place:", 3)
        self._lbl_best_score    = self._lbl_your_best   # alias for existing code

        ctk.CTkFrame(summary_frame, height=4, fg_color="transparent").grid(
            row=4, column=0, columnspan=2)

        self._save_note = ctk.CTkLabel(
            scores_col, text="", font=ctk.CTkFont(size=10), text_color="#666")
        self._save_note.grid(row=n + 3, column=0, pady=(4, 0), sticky="w")

    # ── Model score rows ──────────────────────────────────────────────────────

    def _build_model_row(self, parent, model_name: str, row: int) -> None:
        threshold = THRESHOLDS_LFW.get(model_name)
        accuracy  = ACCURACY_LFW.get(model_name)

        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=("#16213e", "#16213e"))
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)   # whole frame expands

        # ── Row 0: model name | score | verdict | threshold ────────────────────
        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        # col 0: model name (expands)  col 1: score  col 2: verdict  col 3: threshold
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=0)
        top.columnconfigure(3, weight=0)

        ctk.CTkLabel(top, text=model_name,
                     font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
                     text_color="#e0e0e0", anchor="w").grid(
            row=0, column=0, sticky="w")

        score_lbl = ctk.CTkLabel(top, text="—",
                                 font=ctk.CTkFont(size=12, weight="bold"),
                                 text_color="#aaa", width=72, anchor="e")
        score_lbl.grid(row=0, column=1, padx=(8, 4))

        verdict_lbl = ctk.CTkLabel(top, text="",
                                   font=ctk.CTkFont(size=10),
                                   text_color="#555", width=80, anchor="w")
        verdict_lbl.grid(row=0, column=2, padx=(0, 4))

        if threshold is not None:
            ctk.CTkLabel(top, text=f"Threshold: {threshold:.4f}",
                         font=ctk.CTkFont(size=10),
                         text_color="#ff4444", anchor="e").grid(
                row=0, column=3, padx=(4, 0))

        # ── Row 1: similarity bar ──────────────────────────────────────────────
        sim_canvas = tk.Canvas(frame, height=_BAR_H + 10, bg="#16213e",
                               highlightthickness=0, bd=0)
        sim_canvas.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        sim_canvas.bind("<Configure>",
                        lambda e, c=sim_canvas, n=model_name, t=threshold:
                            self._redraw_bar_for_model(c, n, t))

        # ── Row 2: accuracy bar ────────────────────────────────────────────────
        if accuracy is not None:
            acc_row = ctk.CTkFrame(frame, fg_color="transparent")
            acc_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
            acc_row.columnconfigure(1, weight=1)

            ctk.CTkLabel(acc_row, text="Recognition accuracy:",
                         font=ctk.CTkFont(size=9), text_color="#666",
                         anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 6))

            acc_canvas = tk.Canvas(acc_row, height=6, bg="#16213e",
                                   highlightthickness=0, bd=0)
            acc_canvas.grid(row=0, column=1, sticky="ew", padx=(0, 4))
            acc_canvas.bind("<Configure>",
                            lambda e, c=acc_canvas, a=accuracy:
                                self._draw_accuracy_bar(c, a))

            ctk.CTkLabel(acc_row, text=f"{accuracy:.3f}",
                         font=ctk.CTkFont(size=9, weight="bold"),
                         text_color="#6699cc", anchor="e").grid(
                row=0, column=2, sticky="e")

        ctk.CTkFrame(frame, height=4, fg_color="transparent").grid(
            row=3, column=0)

        self._score_widgets[model_name] = (score_lbl, verdict_lbl, sim_canvas, threshold)

    def _redraw_bar_for_model(self, canvas: tk.Canvas, model_name: str,
                               threshold: float | None) -> None:
        """Redraw the sim bar using the last known score — called on canvas resize."""
        score = self._current_scores.get(model_name)
        if score is None:
            self._draw_bar(canvas, 0.0, "#333", threshold)
        else:
            colour = _score_colour(score, threshold)
            self._draw_bar(canvas, _score_to_bar(score), colour, threshold)

    # ── Bar drawing ───────────────────────────────────────────────────────────

    def _draw_bar(self, canvas: tk.Canvas, fill_ratio: float,
                  colour: str, threshold: float | None) -> None:
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 2:
            return
        h = _BAR_H
        canvas.create_rectangle(0, 0, w, h, fill="#2a2a3e", outline="")
        fill_w = int(w * fill_ratio)
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill=colour, outline="")
        if threshold is not None:
            tick_x = int(w * _score_to_bar(threshold))
            canvas.create_line(tick_x, 0, tick_x, h, fill="#ff2222", width=2)
            canvas.create_polygon(
                tick_x - 4, h + 1, tick_x + 4, h + 1, tick_x, h + 7,
                fill="#ff2222", outline="")

    def _draw_accuracy_bar(self, canvas: tk.Canvas, accuracy: float) -> None:
        """Static filled bar showing model recognition accuracy (0–1)."""
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 2:
            return
        h = 6
        canvas.create_rectangle(0, 0, w, h, fill="#1a2a3a", outline="")
        fill_w = int(w * accuracy)
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill="#4477aa", outline="")

    # ── Gallery loading ───────────────────────────────────────────────────────

    def add_control_image(self, user_name: str, image_path: str) -> None:
        """
        Incrementally add one new control image to the in-memory gallery.
        Called by CaptureTab after a new image is accepted, so the gallery
        stays in sync without a full reload.
        """
        if not self._gallery_ready:
            return  # full load still in progress; it will pick this image up
        # Defensive: never treat the HQ companion as a separate control image.
        if Path(image_path).stem.endswith("_hq"):
            return

        def _embed():
            try:
                face_rgb = np.array(Image.open(image_path).convert("RGB"))
            except Exception:
                return

            for model_name in self._engine.model_names:
                emb = self._engine.embed(model_name, face_rgb)
                # Store in DB cache for future gallery loads
                self._db.set_embedding(image_path, model_name, emb)
                # Update mean-embedding gallery
                existing = self._galleries.setdefault(model_name, {})
                existing.setdefault(user_name, []).append(emb)
                # Update per-path gallery
                paths = self._gallery_paths.setdefault(model_name, {})
                paths.setdefault(user_name, []).append((image_path, emb))

            # Refresh identity dropdown in case this is a brand-new user
            all_users = sorted({u for g in self._galleries.values() for u in g})
            def _ui():
                current = self._identity_var.get()
                self._identity_menu.configure(values=all_users)
                if current not in all_users and all_users:
                    self._identity_var.set(all_users[0])
                self._gallery_status.configure(
                    text=f"✓  {len(all_users)} identit"
                         f"{'y' if len(all_users) == 1 else 'ies'} loaded.",
                    text_color="#00ff9f")
            self.after(0, _ui)

        threading.Thread(target=_embed, daemon=True).start()

    def refresh_after_clear(self) -> None:
        """Reset all gallery state and UI after a DB/file wipe."""
        self._galleries.clear()
        self._gallery_paths.clear()
        self._gallery_ready   = False
        self._gallery_loading = False
        self._gallery_mtime   = 0.0
        self._identity_menu.configure(values=["—"])
        self._identity_var.set("—")
        self._gallery_status.configure(
            text="⏳  Gallery not loaded yet — you can still capture images.",
            text_color="#ffd166")
        self._adv_count_label.configure(text="")
        self._lbl_left_best.configure(text="—", text_color="#aaa")
        self._img_summary_frame.grid_remove()
        for w in self._thumb_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        for lbl in (self._lbl_global_best, self._lbl_your_best,
                    self._lbl_your_place, self._lbl_left_best):
            lbl.configure(text="—", text_color="#aaa")
        # Trigger a fresh reload in the background
        self.after(100, self._reload_gallery)

    def _reload_gallery(self) -> None:
        if self._gallery_loading:
            return   # already loading, don't start a second thread
        user_dir = Path(self._data_dir) / "users"
        if not user_dir.exists():
            self._galleries.clear()
            self._gallery_paths.clear()
            self._gallery_ready = True
            self._gallery_loading = False
            self._identity_menu.configure(values=["—"])
            self._identity_var.set("—")
            self._gallery_status.configure(
                text="No control images yet. Add some in the first tab.",
                text_color="#888")
            self._refresh_thumbnails()
            self._refresh_best_score()
            return

        self._gallery_ready   = False
        self._gallery_loading = True
        self._gallery_status.configure(
            text="⏳  Loading — you can still capture images.", text_color="#ffd166")

        def _load():
            try:
                galleries     = {mn: {} for mn in self._engine.model_names}
                gallery_paths = {mn: {} for mn in self._engine.model_names}

                user_dirs = [d for d in sorted(Path(user_dir).iterdir())
                             if d.is_dir()]

                for model_name in self._engine.model_names:
                    # One DB query fetches ALL cached embeddings for this model
                    cache = self._db.get_all_embeddings_for_model(model_name)

                    # Purge any stale embeddings for *_hq.png companion files
                    # (e.g. from earlier app versions that mistakenly embedded
                    # the HQ companions). Cheap one-shot cleanup.
                    for cached_path in list(cache.keys()):
                        if Path(cached_path).stem.endswith("_hq"):
                            self._db.delete_embeddings_for_path(cached_path)
                            del cache[cached_path]

                    for udir in user_dirs:
                        img_paths = (sorted(udir.glob("*.jpg")) +
                                     sorted(udir.glob("*.png")))
                        # Skip *_hq.png companion files — they are an untouched
                        # copy of the same frame, not a separate control image.
                        img_paths = [p for p in img_paths
                                     if not p.stem.endswith("_hq")]
                        if not img_paths:
                            continue
                        identity = udir.name
                        pairs = []
                        embs  = []

                        for p in img_paths:
                            path_str = str(p)
                            emb = cache.get(path_str)
                            if emb is None:
                                # Cache miss — compute and store
                                try:
                                    emb = self._engine.embed(
                                        model_name,
                                        np.array(Image.open(p).convert("RGB")))
                                    self._db.set_embedding(path_str, model_name, emb)
                                    cache[path_str] = emb   # update local cache too
                                except Exception:
                                    continue
                            pairs.append((path_str, emb))
                            embs.append(emb)

                        if embs:
                            galleries[model_name][identity]     = embs
                            gallery_paths[model_name][identity] = pairs

                # Atomic assign
                self._galleries    = galleries
                self._gallery_paths = gallery_paths
                all_users = sorted({u for g in galleries.values() for u in g})
                self._gallery_ready   = True
                self._gallery_loading = False
                try:
                    self._gallery_mtime = Path(user_dir).stat().st_mtime
                except OSError:
                    self._gallery_mtime = 0.0

                def _done():
                    if all_users:
                        self._identity_menu.configure(values=all_users)
                        if self._identity_var.get() not in all_users:
                            self._identity_var.set(all_users[0])
                        self._gallery_status.configure(
                            text=f"✓  {len(all_users)} identit"
                                 f"{'y' if len(all_users) == 1 else 'ies'} loaded.",
                            text_color="#00ff9f")
                        self._refresh_thumbnails()
                        self._refresh_best_score()
                    else:
                        self._gallery_status.configure(
                            text="⚠  No control images found.", text_color="#ff6b6b")

                self.after(0, _done)

            except Exception as exc:
                self._gallery_loading = False
                self.after(0, self._gallery_status.configure,
                           {"text": f"⚠  Load error: {exc}", "text_color": "#ff6b6b"})

        threading.Thread(target=_load, daemon=True).start()

    def _on_identity_changed(self, _name: str) -> None:
        for score_lbl, verdict_lbl, canvas, threshold in self._score_widgets.values():
            score_lbl.configure(text="—", text_color="#aaa")
            verdict_lbl.configure(text="")
            self.after(0, self._draw_bar, canvas, 0.0, "#333", threshold)
        self._img_summary_frame.grid_remove()
        self._refresh_thumbnails()
        self._refresh_best_score()

    # ── Summary panel helpers ─────────────────────────────────────────────────

    def _refresh_best_score(self) -> None:
        """Refresh all static summary stats for the selected identity."""
        identity = self._identity_var.get()

        # Reset all labels
        for lbl in (self._lbl_global_best, self._lbl_your_best,
                    self._lbl_your_place, self._lbl_left_best):
            lbl.configure(text="—", text_color="#aaa")
        if identity == "—":
            self._lbl_avg_score.configure(text="—", text_color="#aaa")
            self._lbl_rank.configure(text="—", text_color="#aaa")
            self._img_summary_frame.grid_remove()
            return

        # Current best score = top-1 image globally
        global_best = self._db.get_global_best_avg_score()
        if global_best is not None:
            col = _score_colour(global_best, None)
            self._lbl_global_best.configure(text=f"{global_best:.4f}", text_color=col)

        # Your best score = best image for this identity
        your_best = self._db.get_best_avg_score_for_identity(identity)
        if your_best is not None:
            col = _score_colour(your_best, None)
            self._lbl_your_best.configure(text=f"{your_best:.4f}", text_color=col)
            self._lbl_left_best.configure(text=f"{your_best:.4f}", text_color=col)
        else:
            self._lbl_your_best.configure(text="no data", text_color="#555")
            self._lbl_left_best.configure(text="no data", text_color="#555")

        # Your current place = rank among all identities by best image
        leaderboard = self._db.get_user_leaderboard()
        total_users = len(leaderboard)
        your_place  = next(
            (i + 1 for i, e in enumerate(leaderboard)
             if e["identity_name"] == identity),
            None,
        )
        if your_place is not None and total_users > 0:
            place_col = "#00ff9f" if your_place == 1 else (
                "#ffd166" if your_place <= max(2, total_users // 4) else "#ff6b6b")
            self._lbl_your_place.configure(
                text=f"#{your_place} / {total_users}", text_color=place_col)

    def _update_summary(self, scores: list[float], save_path: str) -> None:
        """Called after inference: populate and show the image summary box."""
        if not scores:
            return
        avg = sum(scores) / len(scores)
        col = _score_colour(avg, None)

        identity = self._identity_var.get()
        self._lbl_img_name.configure(text=identity, text_color="#e0e0e0")
        self._lbl_avg_score.configure(text=f"{avg:.4f}", text_color=col)

        # Image rank among all individual images
        ranking = self._db.get_average_ranking()
        rank  = next(
            (i + 1 for i, e in enumerate(ranking) if e["image_path"] == save_path),
            None,
        )
        total = len(ranking)
        if rank is not None:
            rank_col = "#00ff9f" if rank <= max(1, total // 4) else (
                "#ffd166" if rank <= total // 2 else "#ff6b6b")
            self._lbl_rank.configure(text=f"#{rank} / {total}", text_color=rank_col)

        self._img_summary_frame.grid()   # make visible

        # Refresh session summary stats
        self._refresh_best_score()

    # ── Adversarial thumbnail strip ───────────────────────────────────────────

    def _refresh_thumbnails(self, preserve_scroll: bool = False) -> None:
        # Capture current scroll position before destroying children, so we
        # can put it back after rebuild. CTkScrollableFrame exposes its inner
        # tk.Canvas as ._parent_canvas; yview() returns (top, bottom) as
        # fractions of the total scrollable height.
        saved_top = None
        if preserve_scroll:
            try:
                saved_top = self._thumb_frame._parent_canvas.yview()[0]
            except Exception:
                saved_top = None

        for w in self._thumb_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        identity = self._identity_var.get()
        if identity == "—":
            self._adv_count_label.configure(text="")
            return

        adv_dir = Path(self._data_dir) / "adversarial" / identity
        if not adv_dir.exists():
            self._adv_count_label.configure(text="No images yet.")
            return

        # Build a quick path → avg_score lookup from the DB
        all_ranking = self._db.get_average_ranking()
        score_map: dict[str, float] = {
            e["image_path"]: e["avg_score"] for e in all_ranking
        }

        images = sorted(adv_dir.glob("*.jpg")) + sorted(adv_dir.glob("*.png"))
        # Hide *_hq.png companions — they are not separate adversarial images.
        images = [p for p in images if not p.stem.endswith("_hq")]
        for i, p in enumerate(images):
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail((_THUMB_SIZE, _THUMB_SIZE))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                       size=(_THUMB_SIZE, _THUMB_SIZE))

                # Wrapper frame holds thumbnail + score label
                cell = ctk.CTkFrame(self._thumb_frame, fg_color="transparent")
                cell.grid(row=i // 3, column=i % 3, padx=3, pady=3)

                lbl = ctk.CTkLabel(cell, image=ctk_img, text="", cursor="hand2")
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, path=str(p): self._open_viewer(path))
                lbl.bind("<Button-3>", lambda e, path=str(p):
                         self._thumb_context_menu(e, path))

                # Score badge below the thumbnail
                avg = score_map.get(str(p))
                if avg is not None:
                    score_col = _score_colour(avg, None)
                    score_text = f"{avg:.3f}"
                else:
                    score_col  = "#555"
                    score_text = "—"
                ctk.CTkLabel(cell, text=score_text,
                             font=ctk.CTkFont(size=9, weight="bold"),
                             text_color=score_col).pack(pady=(1, 0))

                self._thumb_refs.append(ctk_img)
            except Exception:
                pass

        count = len(images)
        self._adv_count_label.configure(
            text=f"{count} image(s) saved for '{identity}'")

        # Restore scroll position. The canvas needs to settle its new
        # scrollregion after the rebuild before yview_moveto will work,
        # so force an idle pass first.
        if saved_top is not None:
            try:
                canvas = self._thumb_frame._parent_canvas
                canvas.update_idletasks()
                canvas.yview_moveto(saved_top)
            except Exception:
                pass

    def _thumb_context_menu(self, event, image_path: str) -> None:
        menu = tk.Menu(self, tearoff=0, bg="#1e1e2e", fg="white",
                       activebackground="#333", activeforeground="white",
                       bd=0, font=("Segoe UI", 10))
        menu.add_command(label="🔍  View full size",
                         command=lambda: self._open_viewer(image_path))
        hq = _hq_path(image_path)
        hq_state = "normal" if hq.exists() else "disabled"
        menu.add_command(label="🖼  View original (high quality)",
                         state=hq_state,
                         command=lambda: self._open_viewer(str(hq)))
        menu.add_command(label="🔄  Recompute scores",
                         command=lambda: self._recompute_scores(image_path))
        menu.add_separator()
        menu.add_command(label="🗑  Delete this image",
                         command=lambda: self._delete_adversarial(image_path))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _recompute_scores(self, image_path: str) -> None:
        """Delete existing DB records for this image and re-run inference."""
        identity = self._identity_var.get()
        if not self._gallery_ready or identity == "—":
            self._status.configure(
                text="⚠  Load gallery first before recomputing scores.",
                text_color="#ff6b6b")
            return
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self._status.configure(text=f"⚠  Cannot open image: {exc}",
                                   text_color="#ff6b6b")
            return

        # Wipe existing records so _run_inference writes fresh ones
        self._db.delete_attacks_by_image(image_path)
        self._status.configure(text="Recomputing scores…", text_color="#aaa")
        self._run_inference(img, Path(image_path), identity)

    def _delete_adversarial(self, image_path: str) -> None:
        import os
        # Remove DB attack records linked to this image
        self._db.delete_attacks_by_image(image_path)
        # Delete file from disk
        try:
            os.remove(image_path)
        except OSError:
            pass
        # Also remove the HQ companion if present.
        hq = _hq_path(image_path)
        if hq.exists():
            try:
                hq.unlink()
            except OSError:
                pass
        self._refresh_thumbnails(preserve_scroll=True)
        self._refresh_best_score()

    def _open_viewer(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return
        popup = tk.Toplevel(self)
        popup.title(Path(path).name)
        popup.configure(bg="#0d0d1a")
        sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
        max_w, max_h = int(sw * 0.8), int(sh * 0.8)
        # Upscale small images (e.g. 112x112 aligned faces) to a readable size
        iw, ih = img.size
        scale = max(max_w / iw, max_h / ih, 1.0)   # at least 1× (never shrink below original)
        scale = min(scale, min(max_w / iw, max_h / ih))  # but cap at screen limit
        nw, nh = int(iw * scale), int(ih * scale)
        nw, nh = min(nw, max_w), min(nh, max_h)
        img = img.resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(popup, image=tk_img, bg="#0d0d1a")
        lbl.image = tk_img
        lbl.pack(padx=10, pady=10)
        tk.Label(popup, text=path, fg="#555", bg="#0d0d1a",
                 font=("Courier New", 9), wraplength=int(sw * 0.8)).pack(pady=(0, 6))
        popup.bind("<Escape>", lambda e: popup.destroy())
        lbl.bind("<Button-1>", lambda e: popup.destroy())
        popup.update_idletasks()
        popup.geometry(
            f"+{(sw - popup.winfo_width()) // 2}+{(sh - popup.winfo_height()) // 2}")

    # ── Camera ────────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._running:
            # Close capture device FIRST so read_frame() unblocks immediately.
            self._camera.close()
            self._running = False
            self._thread_gen += 1
            self._cam_btn.configure(text="▶  Start Camera")
            self._capture_btn.configure(state="disabled")
            self._canvas.delete("all")
            self._status.configure(text="Camera stopped.", text_color="#888")
        else:
            self._camera.close()
            self._running = True
            self._thread_gen += 1
            my_gen = self._thread_gen
            self._camera.open()
            self._cam_btn.configure(text="⏹  Stop Camera")
            self._capture_btn.configure(state="normal")
            self._review_btns.grid_remove()
            self._pending_frame  = None
            self._original_frame = None
            self._pending_source = "camera"
            self._review_image   = None
            self._set_confidence_label(None)
            self._status.configure(text="Camera running.", text_color="#888")
            threading.Thread(target=self._camera_loop, args=(my_gen,), daemon=True).start()

    def _camera_loop(self, gen: int) -> None:
        while self._running and gen == self._thread_gen:
            frame = self._camera.read_frame()
            if frame is None:
                continue
            self._latest_frame = frame
            img = Image.fromarray(frame)
            w, h  = img.size
            side  = min(w, h)
            scale = self._canvas_size / max(w, h)
            disp_w, disp_h = int(w * scale), int(h * scale)
            img_disp  = img.resize((disp_w, disp_h), Image.BILINEAR)
            crop_side = int(side * scale)
            cx = (disp_w - crop_side) // 2
            cy = (disp_h - crop_side) // 2
            tk_img = ImageTk.PhotoImage(img_disp)

            def _upd(tk=tk_img, dw=disp_w, dh=disp_h, g=gen,
                     cx=cx, cy=cy, cs=crop_side):
                if g != self._thread_gen:
                    return
                ox = (self._canvas_size - dw) // 2
                oy = (self._canvas_size - dh) // 2
                self._canvas.delete("all")
                self._canvas.create_image(ox, oy, anchor="nw", image=tk)
                self._canvas.create_rectangle(
                    ox + cx, oy + cy, ox + cx + cs, oy + cy + cs,
                    outline=_CROP_COLOUR, width=2)
                self._photo_ref = tk

            self.after(0, _upd)

    # ── Upload ────────────────────────────────────────────────────────────────

    def _upload_image(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            raw = Image.open(path).convert("RGB")
        except Exception as exc:
            self._status.configure(text=f"⚠  Cannot open image: {exc}",
                                   text_color="#ff6b6b")
            return
        # Stop camera if running
        if self._running:
            self._camera.close()
            self._running = False
            self._thread_gen += 1
            self._cam_btn.configure(text="▶  Start Camera")
            self._capture_btn.configure(state="disabled")
        self._original_frame = raw                # full untouched original
        self._raw_frame = _center_crop_square(raw)
        self._pending_source = "upload"
        self._status.configure(text="Detecting face in uploaded image…", text_color="#aaa")
        threading.Thread(target=self._detect_and_review, args=(raw,), daemon=True).start()

    # ── Capture → Detect → Review → Accept ───────────────────────────────────

    def _capture_frame(self) -> None:
        frame = self._latest_frame
        if frame is None:
            return
        self._camera.close()
        self._running = False
        self._thread_gen += 1
        self._cam_btn.configure(text="▶  Start Camera")
        self._capture_btn.configure(state="disabled")
        self._status.configure(text="Detecting face…", text_color="#aaa")

        raw = Image.fromarray(frame)
        self._original_frame = raw                # full untouched frame
        self._raw_frame = _center_crop_square(raw)
        self._pending_source = "camera"
        threading.Thread(target=self._detect_and_review,
                         args=(raw,), daemon=True).start()

    def _detect_and_review(self, raw: Image.Image) -> None:
        result = self._aligner.align(
            raw,
            thresholds=self._current_thresholds(),
            select_largest=True,
        )
        if result.success:
            self.after(0, self._show_aligned, result.image, result.confidence)
        else:
            self.after(0, self._show_detection_failed, result.error)

    def _show_aligned(self, aligned: Image.Image,
                      confidence: float | None = None) -> None:
        self._pending_frame = aligned
        self._show_review_image(aligned)
        self._fail_btns.grid_remove()
        self._review_btns.grid()
        self._status.configure(
            text="Face detected & aligned — accept or retake.", text_color="#00ff9f")
        self._set_confidence_label(confidence)

    def _show_detection_failed(self, error: str) -> None:
        self._pending_frame = self._raw_frame
        self._show_review_image(self._raw_frame)
        self._review_btns.grid_remove()
        self._fail_btns.grid()
        self._status.configure(text=f"⚠  {error}", text_color="#ff6b6b")
        self._set_confidence_label(None, failed=True)

    # ── Detection threshold slider ────────────────────────────────────────────

    def _current_thresholds(self) -> list[float]:
        """Map the slider's value (= O-Net threshold) to MTCNN's three-stage
        thresholds, preserving the original 6:7 ratio from the upstream
        defaults of [0.6, 0.7, 0.7]."""
        s = self._thr_value
        return [s * (6.0 / 7.0), s, s]

    def _on_threshold_changed(self, value) -> None:
        # Slider callback: snap to 0.01 granularity, update label, store for
        # the next align() call. Detection on the current frame is not
        # re-run — the user must click Capture / Upload / Realign.
        self._thr_value = round(float(value), 2)
        self._thr_value_lbl.configure(text=f"{self._thr_value:.2f}")

    def _set_confidence_label(self, confidence: float | None,
                              failed: bool = False) -> None:
        if confidence is None:
            if failed:
                self._confidence_lbl.configure(
                    text=f"Detection confidence: —  (no face at threshold ≥ {self._thr_value:.2f})",
                    text_color="#ff6b6b")
            else:
                self._confidence_lbl.configure(text="", text_color="#888")
            return
        # The displayed confidence is on the same [0, 1] scale as the slider
        # value. By construction it's always ≥ slider value (MTCNN's O-Net
        # gate). Colour by headroom above that gate.
        headroom = confidence - self._thr_value
        if headroom >= 0.20:
            col = "#00ff9f"
        elif headroom >= 0.05:
            col = "#ffd166"
        elif headroom >= 0.0:
            col = "#ff9f43"
        else:
            col = "#ff6b6b"   # shouldn't happen, defensive
        self._confidence_lbl.configure(
            text=f"Detection confidence: {confidence:.3f}  "
                 f"(threshold ≥ {self._thr_value:.2f})",
            text_color=col)

    def _open_realign(self) -> None:
        """Placeholder — manual realignment UI to be wired up."""
        self._status.configure(
            text="Manual realignment is not yet wired up.",
            text_color="#ffd166")

    def _show_review_image(self, img: Image.Image) -> None:
        self._review_image = img   # remember for canvas resize redraws
        # Use the canvas's actual rendered size so the image fills it whether
        # the window is small or large.  winfo_width/height return real pixel
        # dimensions after the widget has been drawn; fall back to the tracked
        # _canvas_size if the canvas hasn't been realised yet.
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:          # not yet realised
            cw = ch = self._canvas_size
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)   # fit inside canvas, preserving aspect ratio
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        display = img.resize((nw, nh), Image.LANCZOS)
        tk_img  = ImageTk.PhotoImage(display)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self._canvas.delete("all")
        self._canvas.create_image(ox, oy, anchor="nw", image=tk_img)
        self._photo_ref = tk_img

    def _run_inference(self, img: Image.Image, save_path: Path,
                       identity: str) -> None:
        """Shared inference logic for both accept paths."""
        if not self._gallery_ready or identity == "—":
            self._status.configure(
                text=f"✓  Saved {save_path.name}  (gallery not ready — scores skipped)",
                text_color="#ffd166")
            self.after(50, lambda: self._refresh_thumbnails(preserve_scroll=True))
            return

        def _infer():
            results = []
            scores_by_model:  dict[str, float] = {}
            closest_by_model: dict[str, str]   = {}

            # Accumulate per-control-path scores across all models as we go
            path_scores: dict[str, list[float]] = {}

            for model_name in self._engine.model_names:
                pairs = self._gallery_paths.get(model_name, {}).get(identity, [])
                if not pairs:
                    continue

                query = self._engine.embed(model_name, np.array(img.convert("RGB")))

                # Worst-case: score against the closest individual control image
                best_path, best_score = "", -2.0
                for path, emb in pairs:
                    s = self._engine.cosine_similarity(query, emb)
                    path_scores.setdefault(path, []).append(s)
                    if s > best_score:
                        best_score, best_path = s, path

                threshold = THRESHOLDS_LFW.get(model_name)
                scores_by_model[model_name]  = best_score
                closest_by_model[model_name] = best_path

                results.append({
                    "model": model_name, "score": best_score,
                    "threshold": threshold,
                    "match": (best_score >= threshold) if threshold is not None else None,
                    "closest_image": best_path,
                    "frame_path": str(save_path),
                })

            # Ranking closest = control image with highest average similarity
            # across all models — computed from scores already accumulated above
            if path_scores:
                ranking_closest = max(
                    path_scores,
                    key=lambda p: sum(path_scores[p]) / len(path_scores[p]))
            else:
                ranking_closest = ""

            # Write DB records — all with the same ranking_closest for consistency
            for r in results:
                self._db.add_attack(
                    attacker_name=identity,
                    model_name=r["model"],
                    similarity_score=r["score"],
                    image_path=str(save_path),
                    identity_name=identity,
                    matched_image_path=ranking_closest,
                )

            def _update_all():
                for r in results:
                    self._update_row(r["model"], r["score"])
                for model_name in self._engine.model_names:
                    if model_name not in scores_by_model:
                        self._update_row(model_name, None)
                all_scores = [r["score"] for r in results]
                self._update_summary(all_scores, str(save_path))
                self._status.configure(
                    text=f"✓  Saved & logged → {save_path.name}",
                    text_color="#00ff9f")
                self._refresh_thumbnails(preserve_scroll=True)
                if self._ranking_tab is not None:
                    self._ranking_tab._refresh()

            self._write_log(identity, save_path, results)
            self.after(0, _update_all)

        threading.Thread(target=_infer, daemon=True).start()

    def _save_hq_companion(self, save_path: Path) -> None:
        """Write the full untouched original next to save_path as a PNG, if available."""
        if self._original_frame is None:
            return
        try:
            self._original_frame.save(str(_hq_path(save_path)), format="PNG")
        except Exception as exc:
            print(f"[WARN] Could not save HQ companion: {exc}")

    def _accept_frame(self) -> None:
        img = self._pending_frame
        if img is None:
            return
        identity = self._identity_var.get()
        adv_dir  = Path(self._data_dir) / "adversarial" / (
            identity if identity != "—" else "unknown")
        save_path = Camera.next_image_path(adv_dir)
        img.save(str(save_path), format="JPEG", quality=95)
        self._save_hq_companion(save_path)
        # Re-load the saved JPEG so the embedding is computed from the exact
        # bytes that live on disk — otherwise a later "Recompute scores" would
        # read those bytes and produce slightly different scores due to JPEG
        # round-trip noise.
        try:
            scoring_img = Image.open(save_path).convert("RGB")
        except Exception:
            scoring_img = img   # fall back to in-memory if reload fails
        self._review_btns.grid_remove()
        self._fail_btns.grid_remove()
        self._status.configure(text="Running inference…", text_color="#aaa")
        self._pending_frame  = None
        self._raw_frame      = None
        self._original_frame = None
        self._run_inference(scoring_img, save_path, identity)

    def _accept_unprocessed(self) -> None:
        img = self._raw_frame
        if img is None:
            return
        identity = self._identity_var.get()
        adv_dir  = Path(self._data_dir) / "adversarial" / (
            identity if identity != "—" else "unknown")
        save_path = Camera.next_image_path(adv_dir)
        img.save(str(save_path), format="JPEG", quality=95)
        self._save_hq_companion(save_path)
        # Re-load the saved JPEG (see _accept_frame for rationale).
        try:
            scoring_img = Image.open(save_path).convert("RGB")
        except Exception:
            scoring_img = img
        self._review_btns.grid_remove()
        self._fail_btns.grid_remove()
        self._status.configure(text="Running inference…", text_color="#aaa")
        self._pending_frame  = None
        self._raw_frame      = None
        self._original_frame = None
        self._run_inference(scoring_img, save_path, identity)

    def _retake_frame(self) -> None:
        # Remember whether the discarded frame came from the camera or from
        # an uploaded file — the retake action should match the original
        # source.
        was_upload = (self._pending_source == "upload")

        self._pending_frame  = None
        self._raw_frame      = None
        self._original_frame = None
        self._review_btns.grid_remove()
        self._fail_btns.grid_remove()
        self._set_confidence_label(None)

        if was_upload:
            # Re-open the file picker; don't touch the camera state. If the
            # user cancels the dialog, _upload_image returns silently — set
            # an idle status so the UI doesn't look stuck mid-action.
            self._status.configure(text="Idle. Upload an image or start the camera.",
                                   text_color="#888")
            self._upload_image()
            return

        # Camera path: restart the live feed.
        self._camera.close()
        self._running = True
        self._thread_gen += 1
        my_gen = self._thread_gen
        self._camera.open()
        self._cam_btn.configure(text="⏹  Stop Camera")
        self._capture_btn.configure(state="normal")
        self._status.configure(text="Camera running.", text_color="#888")
        threading.Thread(target=self._camera_loop, args=(my_gen,), daemon=True).start()


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_row(self, model_name: str, score: float | None) -> None:
        score_lbl, verdict_lbl, canvas, threshold = self._score_widgets[model_name]
        self._current_scores[model_name] = score   # persist for canvas resize redraws
        if score is None:
            score_lbl.configure(text="no data", text_color="#555")
            verdict_lbl.configure(text="")
            self._draw_bar(canvas, 0.0, "#333", threshold)
            return
        colour = _score_colour(score, threshold)
        score_lbl.configure(text=f"{score:.4f}", text_color=colour)
        if threshold is not None:
            verdict_lbl.configure(
                text="✓ match" if score >= threshold else "✗ no match",
                text_color="#00ff9f" if score >= threshold else "#ff6b6b")
        self._draw_bar(canvas, _score_to_bar(score), colour, threshold)

    def _find_closest_image(self, model_name: str, query_emb: np.ndarray,
                            identity: str) -> str:
        pairs = self._gallery_paths.get(model_name, {}).get(identity, [])
        if not pairs:
            return ""
        best_path, best_score = "", -2.0
        for path, emb in pairs:
            s = self._engine.cosine_similarity(query_emb, emb)
            if s > best_score:
                best_score, best_path = s, path
        return best_path

    def _write_log(self, identity: str, frame_path: Path,
                   results: list[dict]) -> None:
        log_dir  = Path(self._data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{identity}.csv"
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        write_header = not log_file.exists()
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "identity", "frame_path",
                "model", "score", "threshold", "match", "closest_image"])
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow({
                    "timestamp":     ts,
                    "identity":      identity,
                    "frame_path":    str(frame_path),
                    "model":         r["model"],
                    "score":         f"{r['score']:.6f}",
                    "threshold":     r["threshold"] if r["threshold"] is not None else "",
                    "match":         r["match"] if r["match"] is not None else "",
                    "closest_image": r["closest_image"],
                })
        self.after(0, self._save_note.configure,
                   {"text": f"📄 log → {log_file.name}"})

    def destroy(self) -> None:
        self._running = False
        self._thread_gen += 1
        self._camera.close()
        super().destroy()
