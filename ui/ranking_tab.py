"""
ui/ranking_tab.py  —  Ranking tab

Two sub-tabs:
  • Current Standings  — one row per identity, their best (lowest) avg score
  • All Images         — every scored image with model filter dropdown
"""

import csv
import json
import threading
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from core.database import Database, AttackRow
from core.embeddings import EmbeddingEngine
from core.face_aligner import FaceAligner

_THUMB      = 56
_ALL_MODELS = "All models (average)"
_THRESHOLD  = 0.2565
_PAGE_SIZE  = 20


def _hq_path(image_path: str) -> str:
    """Return the companion high-quality PNG path for a saved image.

    Example: data/adversarial/alice/img_007.jpg → …/img_007_hq.png
    Returns the path as a string; the file may or may not exist.
    """
    p = Path(image_path)
    return str(p.with_name(p.stem + "_hq.png"))

# Pattern badge colours (must match recognition_tab)
_PATTERN_COLOURS = {
    0: "#555", 1: "#e06c75", 2: "#e5c07b", 3: "#61afef",
    4: "#c678dd", 5: "#56b6c2",
}


def _tag_badge_text(tags: dict) -> str:
    """Compact badge: 'P2 F·Y' or '∅'."""
    pid = tags.get("pattern_id", 0)
    parts = []
    if pid > 0:
        parts.append(f"P{pid}")
    pose = []
    if tags.get("is_frontal"):
        pose.append("F")
    if tags.get("is_profile"):
        pose.append("Y")
    if tags.get("is_pitch"):
        pose.append("P")
    if pose:
        parts.append("·".join(pose))
    return " ".join(parts) if parts else "∅"

# Per-model LFW decision thresholds (match = score ≥ threshold)
_MODEL_THRESHOLDS: dict[str, float] = {
    "IResNet_R18_Glint360K":    0.2411,
    "InceptionResnetCASIA":     0.4535,
    "SwinFace":                 0.2174,
    "TransFace_S_Glint360K":    0.1640,
    "edgeface_base":            0.2066,
}

def _model_score_col(score: float, model_name: str) -> str:
    """Green = below threshold (fooling model), red = at/above (recognised)."""
    t = _MODEL_THRESHOLDS.get(model_name, _THRESHOLD)
    return "#00ff9f" if score < t else "#ff6b6b"

# Medal colours for top 3
_MEDAL = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}

# Column name → grid index  (shared by both sub-tabs)
_C_STANDINGS = {"rank": 0, "attacker": 1, "score": 2, "attack": 3, "timestamp": 4}
_C_ALL       = {"rank": 0, "attacker": 1, "score": 2,
                "attack": 3, "closest": 4, "timestamp": 5, "delete": 6}


def _sc_col(score: float) -> str:
    return "#00ff9f" if score < _THRESHOLD else "#ff6b6b"


def _scroll_to_top(scrollable_frame) -> None:
    """Scroll a CTkScrollableFrame to its top.

    The frame wraps a tk.Canvas exposed as ._parent_canvas. The canvas needs
    its scrollregion settled (after fresh widgets were grid-packed inside)
    before yview_moveto behaves correctly, so we force an idle pass first.
    """
    try:
        canvas = scrollable_frame._parent_canvas
        canvas.update_idletasks()
        canvas.yview_moveto(0.0)
    except Exception:
        pass


def _capture_scroll(scrollable_frame) -> float | None:
    """Read current top fraction of a CTkScrollableFrame, or None on failure."""
    try:
        return scrollable_frame._parent_canvas.yview()[0]
    except Exception:
        return None


def _restore_scroll(scrollable_frame, top: float | None) -> None:
    """Restore scroll to a previously captured top fraction."""
    if top is None:
        return
    try:
        canvas = scrollable_frame._parent_canvas
        canvas.update_idletasks()
        canvas.yview_moveto(top)
    except Exception:
        pass


class RankingTab(ctk.CTkFrame):
    def __init__(self, parent, db: Database, data_dir: str,
                 engine: EmbeddingEngine, capture_tab=None, recognition_tab=None,
                 aligner=None):
        super().__init__(parent, fg_color="transparent")
        self._db              = db
        self._data_dir        = data_dir
        self._engine          = engine
        self._capture_tab     = capture_tab
        self._recognition_tab = recognition_tab
        self._aligner         = aligner or FaceAligner()
        self._thumb_refs: list = []
        self._podium_window: "PodiumWindow | None" = None

        # ── Shared font cache (avoid allocating hundreds of CTkFont per render) ──
        self._font_11       = ctk.CTkFont(size=11)
        self._font_11_bold  = ctk.CTkFont(size=11, weight="bold")
        self._font_12       = ctk.CTkFont(size=12)
        self._font_7        = ctk.CTkFont(size=7)

        # ── Staleness tracking — skip redundant refreshes on tab switch ──────
        self._db_dirty: bool = True   # set True when data may have changed

        # ── Standings state ───────────────────────────────────────────────────
        self._st_cache: list[dict] = []    # full sorted leaderboard
        self._st_page:  int        = 0     # current 0-based page index

        # ── All Images state ──────────────────────────────────────────────────
        self._all_cache: list[dict] = []
        self._all_cache_model: str  = ""
        self._all_page: int         = 0

        # ── Gallery Search state ──────────────────────────────────────────────
        self._gs_cache: list[dict] = []
        self._gs_page:  int        = 0
        self._gs_import_running: bool = False
        self._gs_gallery_stats: dict  = {}
        self._gs_cancel_flag: "threading.Event | None" = None   # set to cancel import

        self._build_ui()
        self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        toolbar.columnconfigure(5, weight=1)

        ctk.CTkLabel(toolbar, text="Adversarial Attack Ranking",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        ctk.CTkButton(toolbar, text="⟳  Refresh", width=100,
                      command=lambda: self._refresh(force=True)).grid(row=0, column=1, padx=(20, 0))

        ctk.CTkButton(toolbar, text="🗑  Clear All", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._confirm_clear).grid(row=0, column=2, padx=(8, 0))

        ctk.CTkButton(toolbar, text="🏆  Podium View", width=130,
                      fg_color="#1a3a1a", hover_color="#2a5a2a",
                      command=self._open_podium).grid(row=0, column=3, padx=(8, 0))

        self._btn_export_csv = ctk.CTkButton(
            toolbar, text="⬇  Export CSV", width=130,
            fg_color="#1a2a3a", hover_color="#2a4a6a",
            command=self._export_current_tab_csv)
        self._btn_export_csv.grid(row=0, column=4, padx=(8, 0))


        ctk.CTkLabel(toolbar,
                     text=f"Avg. similarity threshold: {_THRESHOLD}  •  "
                          "click thumbnails to enlarge  •  🗑 to delete row",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=0, column=5, padx=(16, 0), sticky="w")

        # ── Sub-tab view ──────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(self, corner_radius=6)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 4))
        self._tabs.add("Current Standings")
        self._tabs.add("All Images")
        self._tabs.add("Gallery Search")
        self._tabs.set("Current Standings")
        self._tabs.configure(command=self._on_subtab_change)

        # ── Current Standings sub-tab ─────────────────────────────────────────
        st = self._tabs.tab("Current Standings")
        st.columnconfigure(0, weight=1)
        st.rowconfigure(0, weight=1)

        self._standings_body = ctk.CTkScrollableFrame(
            st, fg_color=("#12122a", "#12122a"), corner_radius=6, label_text="")
        self._standings_body.grid(row=0, column=0, sticky="nsew")
        for i in range(10):   # 10 cols: rank, attacker, avg, s1-s5, attack, ts
            self._standings_body.columnconfigure(i, weight=1)

        self._standings_summary = ctk.CTkLabel(
            st, text="", font=ctk.CTkFont(size=11), text_color="#888")
        self._standings_summary.grid(row=1, column=0, sticky="w", pady=(2, 4))

        # ── Standings pagination bar ──────────────────────────────────────────
        st_pag = ctk.CTkFrame(st, fg_color="transparent")
        st_pag.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        st_pag.columnconfigure(1, weight=1)

        self._st_btn_prev = ctk.CTkButton(
            st_pag, text="◀  Prev", width=90,
            command=self._st_page_prev)
        self._st_btn_prev.grid(row=0, column=0, padx=(0, 8))

        self._st_lbl_page = ctk.CTkLabel(
            st_pag, text="", font=ctk.CTkFont(size=11), text_color="#888",
            anchor="center")
        self._st_lbl_page.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(st_pag, text="Go to page:",
                     font=ctk.CTkFont(size=11),
                     text_color="#888").grid(row=0, column=2, padx=(0, 4))
        self._st_entry_page = ctk.CTkEntry(st_pag, width=48,
                                           font=ctk.CTkFont(size=11),
                                           justify="center")
        self._st_entry_page.grid(row=0, column=3, padx=(0, 4))
        self._st_entry_page.bind("<Return>", lambda e: self._st_page_goto())
        ctk.CTkButton(st_pag, text="Go", width=40,
                      command=self._st_page_goto).grid(row=0, column=4, padx=(0, 12))

        self._st_btn_next = ctk.CTkButton(
            st_pag, text="Next  ▶", width=90,
            command=self._st_page_next)
        self._st_btn_next.grid(row=0, column=5, padx=(8, 0))

        # ── All Images sub-tab ────────────────────────────────────────────────
        ai = self._tabs.tab("All Images")
        ai.columnconfigure(0, weight=1)
        ai.rowconfigure(1, weight=1)

        # Model filter row inside All Images tab
        filter_row = ctk.CTkFrame(ai, fg_color="transparent")
        filter_row.grid(row=0, column=0, sticky="ew", pady=(4, 4))
        ctk.CTkLabel(filter_row, text="Model:",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 6))
        model_options = [_ALL_MODELS] + self._engine.model_names
        self._model_var = ctk.StringVar(value=_ALL_MODELS)
        ctk.CTkOptionMenu(
            filter_row, variable=self._model_var,
            values=model_options, width=220,
            command=lambda _: self._refresh_all_images(),
        ).pack(side="left")

        self._all_body = ctk.CTkScrollableFrame(
            ai, fg_color=("#12122a", "#12122a"), corner_radius=6, label_text="")
        self._all_body.grid(row=1, column=0, sticky="nsew")
        for i in range(len(_C_ALL)):
            self._all_body.columnconfigure(i, weight=1)

        self._all_summary = ctk.CTkLabel(
            ai, text="", font=ctk.CTkFont(size=11), text_color="#888")
        self._all_summary.grid(row=2, column=0, sticky="w", pady=(2, 4))

        # ── Pagination bar ────────────────────────────────────────────────────
        pag = ctk.CTkFrame(ai, fg_color="transparent")
        pag.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        pag.columnconfigure(1, weight=1)   # centre label expands

        self._btn_prev = ctk.CTkButton(
            pag, text="◀  Prev", width=90,
            command=self._page_prev)
        self._btn_prev.grid(row=0, column=0, padx=(0, 8))

        self._lbl_page = ctk.CTkLabel(
            pag, text="", font=ctk.CTkFont(size=11), text_color="#888",
            anchor="center")
        self._lbl_page.grid(row=0, column=1, sticky="ew")

        # Go-to-page controls (Entry + Go button)
        ctk.CTkLabel(pag, text="Go to page:",
                     font=ctk.CTkFont(size=11),
                     text_color="#888").grid(row=0, column=2, padx=(0, 4))
        self._entry_page = ctk.CTkEntry(pag, width=48,
                                        font=ctk.CTkFont(size=11),
                                        justify="center")
        self._entry_page.grid(row=0, column=3, padx=(0, 4))
        self._entry_page.bind("<Return>", lambda e: self._page_goto())
        ctk.CTkButton(pag, text="Go", width=40,
                      command=self._page_goto).grid(row=0, column=4, padx=(0, 12))

        self._btn_next = ctk.CTkButton(
            pag, text="Next  ▶", width=90,
            command=self._page_next)
        self._btn_next.grid(row=0, column=5, padx=(8, 0))

        # ── Gallery Search sub-tab ─────────────────────────────────────────────
        gs = self._tabs.tab("Gallery Search")
        gs.columnconfigure(0, weight=1)
        gs.columnconfigure(1, weight=0)   # right panel — fixed width
        gs.rowconfigure(0, weight=1)

        # ── Left: results area ────────────────────────────────────────────────
        gs_left = ctk.CTkFrame(gs, fg_color="transparent")
        gs_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        gs_left.columnconfigure(0, weight=1)
        gs_left.rowconfigure(0, weight=1)

        self._gs_body = ctk.CTkScrollableFrame(
            gs_left, fg_color=("#12122a", "#12122a"), corner_radius=6, label_text="")
        self._gs_body.grid(row=0, column=0, sticky="nsew")

        self._gs_summary = ctk.CTkLabel(
            gs_left, text="", font=ctk.CTkFont(size=11), text_color="#888")
        self._gs_summary.grid(row=1, column=0, sticky="w", pady=(2, 4))

        gs_pag = ctk.CTkFrame(gs_left, fg_color="transparent")
        gs_pag.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        gs_pag.columnconfigure(1, weight=1)

        self._gs_btn_prev = ctk.CTkButton(
            gs_pag, text="◀  Prev", width=90,
            command=self._gs_page_prev)
        self._gs_btn_prev.grid(row=0, column=0, padx=(0, 8))

        self._gs_lbl_page = ctk.CTkLabel(
            gs_pag, text="", font=ctk.CTkFont(size=11), text_color="#888",
            anchor="center")
        self._gs_lbl_page.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(gs_pag, text="Go to page:",
                     font=ctk.CTkFont(size=11),
                     text_color="#888").grid(row=0, column=2, padx=(0, 4))
        self._gs_entry_page = ctk.CTkEntry(gs_pag, width=48,
                                           font=ctk.CTkFont(size=11),
                                           justify="center")
        self._gs_entry_page.grid(row=0, column=3, padx=(0, 4))
        self._gs_entry_page.bind("<Return>", lambda e: self._gs_page_goto())
        ctk.CTkButton(gs_pag, text="Go", width=40,
                      command=self._gs_page_goto).grid(row=0, column=4, padx=(0, 12))

        self._gs_btn_next = ctk.CTkButton(
            gs_pag, text="Next  ▶", width=90,
            command=self._gs_page_next)
        self._gs_btn_next.grid(row=0, column=5, padx=(8, 0))

        # ── Right: external gallery dataset panel ─────────────────────────────
        gs_right = ctk.CTkFrame(gs, fg_color="#0d1a2e", corner_radius=8,
                                border_width=1, border_color="#1e3a5f")
        gs_right.grid(row=0, column=1, sticky="ns", padx=(0, 0), pady=0)
        gs_right.columnconfigure(0, weight=1)

        ctk.CTkLabel(gs_right, text="External Gallery",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=12, pady=(10, 2), sticky="w")
        ctk.CTkLabel(gs_right, text="Identity = subfolder name.",
                     font=ctk.CTkFont(size=10), text_color="#555",
                     anchor="w").grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        self._gs_btn_add = ctk.CTkButton(
            gs_right, text="＋  Add Folder", width=190,
            command=self._gs_add_dataset)
        self._gs_btn_add.grid(row=2, column=0, padx=12, pady=(0, 4))

        # Load a precomputed embeddings file produced by
        # data_processing/precompute_gallery.py on an external machine.
        self._gs_btn_load = ctk.CTkButton(
            gs_right, text="📥  Load Precomputed", width=190,
            fg_color="#2a3a5a", hover_color="#3a4a7a",
            command=self._gs_load_precomputed)
        self._gs_btn_load.grid(row=3, column=0, padx=12, pady=(0, 4))

        self._gs_btn_recompute = ctk.CTkButton(
            gs_right, text="↻  Recompute Positions", width=190,
            fg_color="#1a3a1a", hover_color="#2a5a2a",
            command=self._gs_trigger_recompute)
        self._gs_btn_recompute.grid(row=4, column=0, padx=12, pady=(0, 10))

        # Dataset list (scrollable)
        self._gs_dataset_frame = ctk.CTkScrollableFrame(
            gs_right, fg_color="transparent", width=210, height=280)
        self._gs_dataset_frame.grid(row=5, column=0, padx=6, pady=(0, 6), sticky="nsew")
        self._gs_dataset_frame.columnconfigure(0, weight=1)
        gs_right.rowconfigure(5, weight=1)

        # Progress area
        self._gs_progress_label = ctk.CTkLabel(
            gs_right, text="", font=ctk.CTkFont(size=10), text_color="#888",
            wraplength=200, justify="left")
        self._gs_progress_label.grid(row=6, column=0, padx=12, pady=(4, 2), sticky="w")

        self._gs_progress_bar = ctk.CTkProgressBar(gs_right, width=200)
        self._gs_progress_bar.grid(row=7, column=0, padx=12, pady=(0, 10))
        self._gs_progress_bar.set(0)
        self._gs_progress_bar.grid_remove()   # hidden until import starts

        # Populate dataset list from DB on startup
        self._gs_refresh_dataset_list()

    def _refresh(self, force: bool = False) -> None:
        if not force and not self._db_dirty:
            # Cache is still valid — just re-render current page (instant)
            current = self._tabs.get()
            if current == "Current Standings" and self._st_cache:
                self._render_standings_page()
            elif current == "Gallery Search" and self._gs_cache:
                self._render_gs_page()
            elif self._all_cache_model == self._model_var.get() and self._all_cache:
                self._render_all_page()
            self._notify_podium()
            return

        self._db_dirty = False
        self._thumb_refs.clear()
        current = self._tabs.get()
        if current == "Current Standings":
            self._st_cache = []          # invalidate so DB is re-queried
            self._refresh_standings()
        elif current == "Gallery Search":
            self._gs_cache = []
            self._refresh_gallery_search()
        else:
            self._all_cache_model = ""   # invalidate so DB is re-queried
            self._refresh_all_images()
        self._notify_podium()

    def mark_dirty(self) -> None:
        """Called by other tabs after scoring / deleting to signal stale data."""
        self._db_dirty = True

    def _on_subtab_change(self) -> None:
        current = self._tabs.get()
        if current == "Current Standings":
            if self._st_cache and not self._db_dirty:
                self._render_standings_page()   # instant — reuse cache
            else:
                self._db_dirty = False
                self._st_cache = []
                self._refresh_standings()       # first visit or stale
        elif current == "Gallery Search":
            self._db_dirty = False
            self._gs_cache = []
            self._refresh_gallery_search()
        else:
            if (self._all_cache_model == self._model_var.get()
                    and self._all_cache and not self._db_dirty):
                self._render_all_page()         # instant — reuse cache
            else:
                self._db_dirty = False
                self._all_cache_model = ""
                self._refresh_all_images()
        self._notify_podium()

    # ── Tab hooks ─────────────────────────────────────────────────────────────

    def on_tab_shown(self) -> None:
        if self._db_dirty:
            self._refresh(force=True)
        # If not dirty, do nothing — the cached page is already displayed.

    # ── Current Standings ─────────────────────────────────────────────────────

    def _refresh_standings(self) -> None:
        """Rebuild cache from DB in a background thread, then render page 0."""
        # Show a loading indicator immediately
        body = self._standings_body
        for w in body.winfo_children():
            w.destroy()
        ctk.CTkLabel(body, text="Loading…",
                     font=self._font_12, text_color="#555").grid(
            row=0, column=0, columnspan=10, pady=30)

        def _query():
            leaderboard  = self._db.get_user_leaderboard()
            model_names  = self._engine.model_names
            all_rows     = self._db.get_ranking(model_name=None, limit=10_000)
            return leaderboard, model_names, all_rows

        def _on_done(leaderboard, model_names, all_rows):
            # image_path → {model_name: score}  /  metadata
            img_scores: dict[str, dict[str, float]] = {}
            img_meta:   dict[str, dict] = {}
            for r in all_rows:
                if r.image_path not in img_scores:
                    img_scores[r.image_path] = {}
                    img_meta[r.image_path]   = {
                        "attacker":    r.attacker_name,
                        "created_at":  r.created_at,
                        "tags": {"pattern_id": r.pattern_id,
                                 "is_frontal": r.is_frontal,
                                 "is_profile": r.is_profile,
                                 "is_pitch":   r.is_pitch},
                    }
                img_scores[r.image_path][r.model_name] = r.similarity_score

            img_avg: dict[str, float] = {
                p: sum(s.values()) / len(s)
                for p, s in img_scores.items() if s
            }

            # Per-identity best image
            identity_best_img: dict[str, str] = {}
            for entry in leaderboard:
                identity   = entry["identity_name"]
                candidates = [p for p, m in img_meta.items()
                              if m["attacker"] == identity]
                if candidates:
                    identity_best_img[identity] = min(
                        candidates, key=lambda p: img_avg.get(p, 1.0))

            # Per-model column minimums
            col_min: dict[str, float] = {}
            for entry in leaderboard:
                path = identity_best_img.get(entry["identity_name"], "")
                if not path:
                    continue
                for mn in model_names:
                    s = img_scores.get(path, {}).get(mn)
                    if s is not None and (mn not in col_min or s < col_min[mn]):
                        col_min[mn] = s

            # Build flat cache list — one dict per participant
            self._st_cache = [
                {
                    "identity":   e["identity_name"],
                    "best_avg":   e["best_avg_score"],
                    "path":       identity_best_img.get(e["identity_name"], ""),
                    "scores":     img_scores.get(
                                      identity_best_img.get(e["identity_name"], ""), {}),
                    "ts":         img_meta.get(
                                      identity_best_img.get(e["identity_name"], ""),
                                      {}).get("created_at", ""),
                    "tags":       img_meta.get(
                                      identity_best_img.get(e["identity_name"], ""),
                                      {}).get("tags", {}),
                    "model_names": model_names,
                    "col_min":    col_min,
                }
                for e in leaderboard
            ]
            self._st_page = 0
            self._render_standings_page()

        def _bg():
            try:
                leaderboard, model_names, all_rows = _query()
            except Exception:
                return
            try:
                if self.winfo_exists():
                    self.after(0, lambda: _on_done(leaderboard, model_names, all_rows))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_bg, daemon=True).start()

    # ── Standings pagination helpers ──────────────────────────────────────────

    def _st_page_prev(self) -> None:
        if self._st_page > 0:
            self._st_page -= 1
            self._render_standings_page()
            _scroll_to_top(self._standings_body)

    def _st_page_next(self) -> None:
        total_pages = max(1, -(-len(self._st_cache) // _PAGE_SIZE))
        if self._st_page < total_pages - 1:
            self._st_page += 1
            self._render_standings_page()
            _scroll_to_top(self._standings_body)

    def _st_page_goto(self) -> None:
        total_pages = max(1, -(-len(self._st_cache) // _PAGE_SIZE))
        try:
            target = int(self._st_entry_page.get().strip()) - 1
        except ValueError:
            self._st_entry_page.delete(0, "end")
            return
        self._st_page = max(0, min(target, total_pages - 1))
        self._st_entry_page.delete(0, "end")
        self._render_standings_page()
        _scroll_to_top(self._standings_body)

    def _render_standings_page(self) -> None:
        """Render the current page slice from self._st_cache. No DB query."""
        body    = self._standings_body
        _N_COLS = 10

        for w in body.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        for i in range(_N_COLS):
            body.columnconfigure(i, weight=1)

        total = len(self._st_cache)
        if total == 0:
            ctk.CTkLabel(body, text="No records yet.",
                         font=self._font_12, text_color="#555").grid(
                row=1, column=0, columnspan=_N_COLS, pady=30)
            self._standings_summary.configure(text="0 participants.")
            self._st_lbl_page.configure(text="")
            self._st_btn_prev.configure(state="disabled")
            self._st_btn_next.configure(state="disabled")
            return

        total_pages = max(1, -(-total // _PAGE_SIZE))
        self._st_page = max(0, min(self._st_page, total_pages - 1))
        start     = self._st_page * _PAGE_SIZE
        end       = min(start + _PAGE_SIZE, total)
        page_data = self._st_cache[start:end]

        # ── Header ────────────────────────────────────────────────────────────
        HDR_BG  = "#16213e"
        AVG_BG  = "#0d2a1a"
        BEST_BG = "#1a1a0a"
        model_names = self._st_cache[0]["model_names"] if self._st_cache else []

        def hdr(col: int, text: str, bg: str = HDR_BG) -> None:
            ctk.CTkLabel(body, text=text,
                         font=self._font_11_bold,
                         text_color="#00d4ff", fg_color=bg,
                         anchor="center").grid(
                row=0, column=col, padx=2, pady=(4, 4), sticky="ew")

        hdr(0, "Rank")
        hdr(1, "Attacker")
        hdr(2, "Best Avg ↓", bg=AVG_BG)
        for si, mn in enumerate(model_names):
            hdr(3 + si, f"Model {si+1}")
        hdr(8, "Best Attack")
        hdr(9, "Timestamp")

        # ── Rows ──────────────────────────────────────────────────────────────
        for local_idx, entry in enumerate(page_data):
            global_rank = start + local_idx + 1
            grid_row    = local_idx + 1
            identity    = entry["identity"]
            best_avg    = entry["best_avg"]
            path        = entry["path"]
            scores      = entry["scores"]
            ts          = entry["ts"]
            col_min     = entry["col_min"]
            medal       = _MEDAL.get(global_rank)
            txt_col     = medal if medal else "#e0e0e0"

            def cell(col: int, text: str, txt: str = "#ccc",
                     bg: str = "transparent", r: int = grid_row) -> None:
                ctk.CTkLabel(body, text=text, text_color=txt,
                             font=self._font_11, anchor="center",
                             fg_color=bg).grid(
                    row=r, column=col, padx=2, pady=2, sticky="ew")

            cell(0, f"#{global_rank}", txt=txt_col)
            cell(1, identity,          txt=txt_col)
            cell(2, f"{best_avg:.4f}", txt=_sc_col(best_avg), bg=AVG_BG)

            for si, mn in enumerate(model_names):
                s = scores.get(mn)
                if s is None:
                    cell(3 + si, "—", txt="#555")
                else:
                    is_best = (col_min.get(mn) is not None and
                               abs(s - col_min[mn]) < 1e-9)
                    cell(3 + si, f"{s:.4f}",
                         txt=_model_score_col(s, mn),
                         bg=BEST_BG if is_best else "transparent")

            self._add_thumb(body, path, grid_row, col=8, label="best attack",
                            tags=entry.get("tags"))
            cell(9, ts[:19].replace("T", " ") if ts else "—", txt="#666")

        self._standings_summary.configure(
            text=f"{total} participant(s) — ranked by best avg score ASC  •  "
                 f"Model 1–{len(model_names)} = per-model similarity  •  "
                 f"showing {start + 1}–{end}")
        self._st_lbl_page.configure(
            text=f"Page {self._st_page + 1} / {total_pages}")
        self._st_btn_prev.configure(
            state="normal" if self._st_page > 0 else "disabled")
        self._st_btn_next.configure(
            state="normal" if self._st_page < total_pages - 1 else "disabled")

    # ── All Images ────────────────────────────────────────────────────────────

    def _refresh_all_images(self) -> None:
        """
        Called when the model dropdown changes OR on an explicit Refresh.
        Rebuilds the cache for the selected model in a background thread,
        resets to page 0, renders.
        """
        selected = self._model_var.get()

        # Show loading indicator immediately
        body = self._all_body
        for w in body.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        ctk.CTkLabel(body, text="Loading…",
                     font=self._font_12, text_color="#555").grid(
            row=0, column=0, columnspan=7, pady=30)

        def _query():
            if selected == _ALL_MODELS:
                all_rows = self._db.get_ranking(model_name=None, limit=10_000)
                groups: dict = {}
                for r in all_rows:
                    groups.setdefault((r.image_path, r.attacker_name), []).append(r)
                averaged = []
                for (img_path, attacker), rows in groups.items():
                    avg     = sum(r.similarity_score for r in rows) / len(rows)
                    closest = next((r.matched_image_path for r in rows
                                    if r.matched_image_path), "")
                    first   = rows[0]
                    averaged.append({"attacker": attacker, "avg_score": avg,
                                     "image_path": img_path, "closest_path": closest,
                                     "created_at": first.created_at,
                                     "tags": {"pattern_id": first.pattern_id,
                                              "is_frontal": first.is_frontal,
                                              "is_profile": first.is_profile,
                                              "is_pitch":   first.is_pitch}})
                averaged.sort(key=lambda x: x["avg_score"])
                return averaged
            else:
                rows = self._db.get_ranking(model_name=selected, limit=10_000)
                return [
                    {"attacker": r.attacker_name, "avg_score": r.similarity_score,
                     "image_path": r.image_path, "closest_path": r.matched_image_path,
                     "created_at": r.created_at,
                     "tags": {"pattern_id": r.pattern_id,
                              "is_frontal": r.is_frontal,
                              "is_profile": r.is_profile,
                              "is_pitch":   r.is_pitch}}
                    for r in rows
                ]

        def _on_done(cache):
            self._all_cache = cache
            self._all_cache_model = selected
            self._all_page = 0
            self._render_all_page()

        def _bg():
            try:
                cache = _query()
            except Exception:
                return
            try:
                if self.winfo_exists():
                    self.after(0, lambda: _on_done(cache))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_bg, daemon=True).start()

    # ── Pagination helpers ────────────────────────────────────────────────────

    def _page_prev(self) -> None:
        if self._all_page > 0:
            self._all_page -= 1
            self._render_all_page()
            _scroll_to_top(self._all_body)

    def _page_next(self) -> None:
        total_pages = max(1, -(-len(self._all_cache) // _PAGE_SIZE))  # ceil div
        if self._all_page < total_pages - 1:
            self._all_page += 1
            self._render_all_page()
            _scroll_to_top(self._all_body)

    def _page_goto(self) -> None:
        total_pages = max(1, -(-len(self._all_cache) // _PAGE_SIZE))
        try:
            target = int(self._entry_page.get().strip()) - 1  # convert to 0-based
        except ValueError:
            self._entry_page.delete(0, "end")
            return
        target = max(0, min(target, total_pages - 1))
        self._all_page = target
        self._entry_page.delete(0, "end")
        self._render_all_page()
        _scroll_to_top(self._all_body)

    def _draw_all_header(self, body, is_average: bool) -> None:
        score_label = "Avg Similarity ↓" if is_average else "Similarity ↓"
        labels = ["Rank", "Attacker", score_label, "Attack",
                  "Closest", "Timestamp", ""]
        for col, text in enumerate(labels):
            ctk.CTkLabel(body, text=text,
                         font=self._font_11_bold,
                         text_color="#00d4ff", fg_color="#16213e",
                         anchor="center").grid(
                row=0, column=col, padx=2, pady=(4, 4), sticky="ew")

    def _render_all_page(self) -> None:
        """
        Destroy existing rows and render only the current page slice.
        No DB query — works entirely from self._all_cache.
        """
        body = self._all_body
        # Clear old widgets
        for w in body.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        is_average = (self._all_cache_model == _ALL_MODELS)
        self._draw_all_header(body, is_average)

        data  = self._all_cache
        total = len(data)

        if total == 0:
            ctk.CTkLabel(body, text="No records yet.",
                         font=self._font_12, text_color="#555").grid(
                row=1, column=0, columnspan=7, pady=30)
            self._all_summary.configure(text="0 records.")
            self._lbl_page.configure(text="")
            self._btn_prev.configure(state="disabled")
            self._btn_next.configure(state="disabled")
            return

        total_pages = max(1, -(-total // _PAGE_SIZE))
        self._all_page = max(0, min(self._all_page, total_pages - 1))

        start = self._all_page * _PAGE_SIZE
        end   = min(start + _PAGE_SIZE, total)
        page_data = data[start:end]

        # Global rank = position in the full sorted list (1-based)
        for local_idx, entry in enumerate(page_data):
            global_rank = start + local_idx + 1
            grid_row    = local_idx + 1   # header is row 0
            self._all_row_cells(body, grid_row, global_rank,
                                entry["attacker"], entry["avg_score"],
                                entry["image_path"], entry["closest_path"],
                                entry["created_at"],
                                tags=entry.get("tags"))

        # Update summary and pagination controls
        label = "average similarity" if is_average else "similarity"
        self._all_summary.configure(
            text=f"{total} image(s) — {label} ASC  •  "
                 f"showing {start + 1}–{end}")

        self._lbl_page.configure(
            text=f"Page {self._all_page + 1} / {total_pages}")
        self._btn_prev.configure(
            state="normal" if self._all_page > 0 else "disabled")
        self._btn_next.configure(
            state="normal" if self._all_page < total_pages - 1 else "disabled")

    def _all_row_cells(self, body, grid_row: int, rank: int, attacker: str,
                       score: float, img_path: str, closest_path: str,
                       ts: str, tags: dict | None = None) -> None:
        medal   = _MEDAL.get(rank)
        txt_col = medal if medal else "#e0e0e0"

        def cell(col_name: str, text: str, color: str = "#ccc") -> None:
            ctk.CTkLabel(body, text=text, text_color=color,
                         font=self._font_11, anchor="center").grid(
                row=grid_row, column=_C_ALL[col_name], padx=4, pady=3, sticky="ew")

        cell("rank",      f"#{rank}",                 color=txt_col)
        cell("attacker",  attacker,                   color=txt_col)
        cell("score",     f"{score:.4f}",            color=_sc_col(score))
        cell("timestamp", ts[:19].replace("T", " "), color="#666")

        self._add_thumb(body, img_path,      grid_row, _C_ALL["attack"],  "attack frame",
                        tags=tags)
        self._add_thumb(body, closest_path,  grid_row, _C_ALL["closest"], "closest match")
        self._add_delete_btn(body, img_path, grid_row, _C_ALL["delete"])

    # ── Gallery Search ─────────────────────────────────────────────────────

    # ── Dataset panel helpers ──────────────────────────────────────────────

    def _gs_refresh_dataset_list(self) -> None:
        """Rebuild the list of external gallery datasets shown in the right panel."""
        frame = self._gs_dataset_frame
        for w in frame.winfo_children():
            w.destroy()

        datasets = self._db.get_ext_gallery_datasets()
        if not datasets:
            ctk.CTkLabel(frame, text="No datasets loaded.",
                         font=ctk.CTkFont(size=10), text_color="#444").grid(
                row=0, column=0, padx=6, pady=6, sticky="w")
            return

        for i, ds in enumerate(datasets):
            row_frame = ctk.CTkFrame(frame, fg_color="#111830", corner_radius=6)
            row_frame.grid(row=i, column=0, sticky="ew", pady=(0, 4), padx=2)
            row_frame.columnconfigure(0, weight=1)

            name_lbl = ctk.CTkLabel(
                row_frame, text=ds["display_name"],
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="#c0d8ff", anchor="w", wraplength=160)
            name_lbl.grid(row=0, column=0, padx=8, pady=(6, 1), sticky="ew")

            count_lbl = ctk.CTkLabel(
                row_frame,
                text=f"{ds['image_count']} image{'s' if ds['image_count'] != 1 else ''}",
                font=ctk.CTkFont(size=9), text_color="#556", anchor="w")
            count_lbl.grid(row=1, column=0, padx=8, pady=(0, 6), sticky="ew")

            ctk.CTkButton(
                row_frame, text="✕", width=26, height=22,
                fg_color="#3a0000", hover_color="#700000",
                font=ctk.CTkFont(size=10),
                command=lambda did=ds["id"]: self._gs_remove_dataset(did),
            ).grid(row=0, column=1, rowspan=2, padx=(4, 6), pady=4)

    def _gs_add_dataset(self) -> None:
        """Open a folder-picker and import all images as an external gallery dataset."""
        if self._gs_import_running:
            return
        folder = filedialog.askdirectory(title="Select gallery folder (subfolders = identities)")
        if not folder:
            return
        folder = Path(folder)

        # Discover image files grouped by identity (one level of subfolders)
        _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        identity_images: dict[str, list[Path]] = {}
        for sub in sorted(folder.iterdir()):
            if sub.is_dir():
                imgs = sorted(
                    p for p in sub.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS
                    and "_hq" not in p.stem
                )
                if imgs:
                    identity_images[sub.name] = imgs

        if not identity_images:
            self._gs_progress_label.configure(
                text="No identity subfolders with images found.", text_color="#ff6b6b")
            return

        total_imgs = sum(len(v) for v in identity_images.values())
        display_name = folder.name

        # Register in DB (get or create)
        dataset_id = self._db.add_ext_gallery_dataset(str(folder), display_name)
        already_done = self._db.get_ext_gallery_existing_paths(dataset_id)

        self._gs_import_running = True
        self._gs_cancel_flag = threading.Event()
        cancel_flag = self._gs_cancel_flag
        self._gs_btn_add.configure(text="✕  Cancel", fg_color="#5a0000",
                                   hover_color="#8b0000",
                                   command=self._gs_cancel_import)
        self._gs_btn_load.configure(state="disabled")
        self._gs_btn_recompute.configure(state="disabled")
        self._gs_progress_bar.grid()
        self._gs_progress_bar.set(0)
        self._gs_progress_label.configure(
            text=f"Importing {display_name}…  0 / {total_imgs}", text_color="#888")

        def _bg() -> None:
            import numpy as np
            import concurrent.futures
            import threading as _threading

            # ── Pre-filter: skip images already embedded ──────────────────────
            work: list[tuple[str, Path]] = []
            skipped = 0
            for identity, paths in identity_images.items():
                for img_path in paths:
                    if str(img_path) in already_done:
                        skipped += 1
                    else:
                        work.append((identity, img_path))

            total_work = len(work)
            print(f"[EXT GALLERY] Starting import: {display_name}")
            print(f"[EXT GALLERY] {total_imgs} images total, "
                  f"{skipped} already embedded, {total_work} to process")
            for identity in sorted(identity_images.keys()):
                print(f"[EXT GALLERY]   identity: {identity!r}  "
                      f"({len(identity_images[identity])} images)")

            lock       = _threading.Lock()
            batch: list[tuple] = []
            BATCH_SIZE = 500
            processed  = skipped
            accepted   = skipped

            def _post_progress(frac, p, a, identity):
                try:
                    if self.winfo_exists():
                        self.after(0, lambda f=frac, pp=p, aa=a, id=identity:
                            self._gs_import_progress(f, pp, total_imgs, aa, id))
                except Exception:
                    pass

            def _flush_batch():
                if batch:
                    try:
                        self._db.bulk_upsert_ext_gallery_embeddings(list(batch))
                    except Exception as e:
                        print(f"[EXT GALLERY] DB write error: {e}")
                    batch.clear()

            def _process_one(args):
                if cancel_flag.is_set():
                    return None, args[0]
                identity, img_path = args
                path_str = str(img_path)
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                except Exception:
                    return None, identity
                try:
                    aligned = self._aligner.align(pil_img)
                except Exception:
                    aligned = None
                if aligned is None:
                    return None, identity
                face_arr = np.array(aligned.image.convert("RGB"))
                rows = []
                for model_name in self._engine.model_names:
                    try:
                        emb  = self._engine.embed(model_name, face_arr)
                        blob = emb.astype("float32").tobytes()
                        rows.append((dataset_id, path_str, identity,
                                     model_name, blob))
                    except Exception as e:
                        print(f"[EXT GALLERY] embed error {model_name} "
                              f"{img_path.name}: {e}")
                return rows if rows else None, identity

            cancelled = False
            N_WORKERS = 4
            with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                futures = {pool.submit(_process_one, item): item for item in work}
                for future in concurrent.futures.as_completed(futures):
                    if cancel_flag.is_set():
                        cancelled = True
                        # Cancel pending futures (already-running ones finish naturally)
                        for f in futures:
                            f.cancel()
                        break
                    rows, identity = future.result()
                    with lock:
                        processed += 1
                        if rows:
                            accepted += 1
                            batch.extend(rows)
                            if len(batch) >= BATCH_SIZE:
                                _flush_batch()
                        frac = processed / total_imgs
                        # Post UI update for every accepted image; throttle skips
                        if rows or processed % 10 == 0:
                            _post_progress(frac, processed, accepted, identity)

            with lock:
                _flush_batch()

            try:
                real_count = self._db.count_ext_gallery_embeddings(dataset_id)
                self._db.update_ext_gallery_image_count(dataset_id, real_count)
                status = "Cancelled" if cancelled else "Import done"
                print(f"[EXT GALLERY] {status}: {accepted} accepted, "
                      f"{real_count} embeddings stored")
            except Exception as e:
                print(f"[EXT GALLERY] Count error: {e}")

            try:
                if self.winfo_exists():
                    self.after(0, lambda: self._gs_import_done(cancelled=cancelled))
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _gs_import_progress(self, frac: float, done: int,
                            total: int, accepted: int,
                            identity: str = "") -> None:
        try:
            self._gs_progress_bar.set(frac)
            id_hint = f"  [{identity}]" if identity else ""
            self._gs_progress_label.configure(
                text=f"Importing…  {done} / {total}  ({accepted} accepted){id_hint}",
                text_color="#888")
        except Exception:
            pass

    def _gs_cancel_import(self) -> None:
        """Signal the background import thread to stop after its current batch."""
        if self._gs_cancel_flag is not None:
            self._gs_cancel_flag.set()
        self._gs_btn_add.configure(state="disabled", text="Cancelling…")
        self._gs_progress_label.configure(text="Cancelling…", text_color="#ffd166")

    def _gs_import_done(self, cancelled: bool = False) -> None:
        self._gs_import_running = False
        self._gs_cancel_flag    = None
        try:
            self._gs_progress_bar.set(1.0)
            if cancelled:
                self._gs_progress_label.configure(
                    text="Import cancelled — partial data kept.", text_color="#ffd166")
            else:
                self._gs_progress_label.configure(
                    text="Import complete.", text_color="#00ff9f")
            # Restore Add Folder button
            self._gs_btn_add.configure(
                text="＋  Add Folder", state="normal",
                fg_color=["#3a7ebf", "#1f538d"],   # CTk default button colours
                hover_color=["#325882", "#14375e"],
                command=self._gs_add_dataset)
            self._gs_btn_load.configure(state="normal")
            self._gs_btn_recompute.configure(state="normal")
            self._gs_refresh_dataset_list()
        except Exception:
            pass

    # ── Precomputed embeddings loader ─────────────────────────────────────
    #
    # The companion script data_processing/precompute_gallery.py walks a
    # folder, aligns + embeds every image, and writes a single .npz file
    # holding identities, paths and per-model embedding matrices.  This
    # handler loads such a file back into the database as a new external
    # gallery dataset — no alignment or embedding happens here, so it's
    # cheap regardless of dataset size.
    def _gs_load_precomputed(self) -> None:
        """Open a .npz file produced by precompute_gallery.py and insert
        its embeddings into the external gallery."""
        if self._gs_import_running:
            return

        file_path = filedialog.askopenfilename(
            title="Select precomputed embeddings file",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        )
        if not file_path:
            return
        file_path = Path(file_path)

        import numpy as np

        # ── Load + validate ───────────────────────────────────────────────
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as exc:
            messagebox.showerror(
                "Load failed",
                f"Could not open {file_path.name}:\n{exc}",
            )
            return

        try:
            file_models = [str(x) for x in data["model_names"]]
            identities  = [str(x) for x in data["identities"]]
            file_paths  = [str(x) for x in data["paths"]]
            meta_raw    = data["meta"].item() if "meta" in data.files else "{}"
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
        except Exception as exc:
            messagebox.showerror(
                "Load failed",
                f"{file_path.name} is missing required arrays:\n{exc}",
            )
            data.close()
            return

        # Strict model-set check — user-selected behaviour.
        engine_models = set(self._engine.model_names)
        file_model_set = set(file_models)
        if engine_models != file_model_set:
            missing = sorted(engine_models - file_model_set)
            extra   = sorted(file_model_set - engine_models)
            lines = ["Model mismatch — cannot load."]
            if missing:
                lines.append(f"Missing in file: {', '.join(missing)}")
            if extra:
                lines.append(f"Extra in file: {', '.join(extra)}")
            lines.append("\nThe file must contain embeddings for exactly "
                         "the models currently loaded.")
            messagebox.showerror("Model mismatch", "\n".join(lines))
            data.close()
            return

        n = len(identities)
        if n == 0 or len(file_paths) != n:
            messagebox.showerror(
                "Load failed",
                f"{file_path.name} contains no usable rows "
                f"(identities={n}, paths={len(file_paths)}).",
            )
            data.close()
            return

        # Per-model embedding matrices
        emb_matrices: dict[str, np.ndarray] = {}
        for mn in self._engine.model_names:
            key = f"emb__{mn}"
            if key not in data.files:
                messagebox.showerror(
                    "Load failed",
                    f"{file_path.name} is missing array {key!r}.",
                )
                data.close()
                return
            mat = np.asarray(data[key])
            if mat.shape[0] != n:
                messagebox.showerror(
                    "Load failed",
                    f"{file_path.name}: embedding count for {mn} "
                    f"({mat.shape[0]}) does not match identity count ({n}).",
                )
                data.close()
                return
            emb_matrices[mn] = mat.astype("float32", copy=False)
        data.close()

        # ── Optional path remap ───────────────────────────────────────────
        # Paths inside the file refer to wherever the script ran.  If the
        # user has a local copy of the images, they can map the recorded
        # prefix → a local prefix so thumbnails resolve.  Both empty fields
        # = keep paths as-is.
        sample_path = file_paths[0] if file_paths else ""
        prefixes = self._gs_ask_path_remap(sample_path)
        if prefixes is None:                         # cancelled
            return
        src_prefix, dst_prefix = prefixes
        if src_prefix:
            remapped: list[str] = []
            for p in file_paths:
                if p.startswith(src_prefix):
                    remapped.append(dst_prefix + p[len(src_prefix):])
                else:
                    remapped.append(p)
            file_paths = remapped

        # ── Register dataset + insert embeddings ──────────────────────────
        display_name = meta.get("display_name") or file_path.stem
        source_label = meta.get("source_folder") or str(file_path)
        dataset_id = self._db.add_ext_gallery_dataset(source_label, display_name)
        already_done = self._db.get_ext_gallery_existing_paths(dataset_id)

        self._gs_import_running = True
        self._gs_btn_add.configure(state="disabled")
        self._gs_btn_load.configure(state="disabled", text="Loading…")
        self._gs_btn_recompute.configure(state="disabled")
        self._gs_progress_bar.grid()
        self._gs_progress_bar.set(0)
        self._gs_progress_label.configure(
            text=f"Loading {display_name}…  0 / {n}", text_color="#888")

        model_names = list(self._engine.model_names)

        def _bg() -> None:
            inserted = 0
            skipped  = 0
            BATCH_SIZE = 500
            batch: list[tuple] = []

            def _flush():
                if not batch:
                    return
                try:
                    self._db.bulk_upsert_ext_gallery_embeddings(list(batch))
                except Exception as e:
                    print(f"[LOAD PRECOMPUTED] DB write error: {e}")
                batch.clear()

            def _post_progress(done: int):
                try:
                    if self.winfo_exists():
                        frac = done / max(1, n)
                        self.after(0, lambda d=done, f=frac:
                            self._gs_load_progress(f, d, n))
                except Exception:
                    pass

            for i in range(n):
                path_str = file_paths[i]
                identity = identities[i]
                if path_str in already_done:
                    skipped += 1
                else:
                    for mn in model_names:
                        blob = emb_matrices[mn][i].tobytes()
                        batch.append(
                            (dataset_id, path_str, identity, mn, blob))
                    inserted += 1
                    if len(batch) >= BATCH_SIZE:
                        _flush()

                if (i + 1) % 200 == 0 or (i + 1) == n:
                    _post_progress(i + 1)

            _flush()

            try:
                real_count = self._db.count_ext_gallery_embeddings(dataset_id)
                self._db.update_ext_gallery_image_count(dataset_id, real_count)
                print(f"[LOAD PRECOMPUTED] {display_name}: "
                      f"inserted={inserted}, skipped={skipped}, "
                      f"db_total={real_count}")
            except Exception as e:
                print(f"[LOAD PRECOMPUTED] Count error: {e}")

            try:
                if self.winfo_exists():
                    self.after(0, lambda: self._gs_load_done(
                        inserted=inserted, skipped=skipped))
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _gs_load_progress(self, frac: float, done: int, total: int) -> None:
        try:
            self._gs_progress_bar.set(frac)
            self._gs_progress_label.configure(
                text=f"Loading…  {done} / {total}", text_color="#888")
        except Exception:
            pass

    def _gs_load_done(self, inserted: int, skipped: int) -> None:
        self._gs_import_running = False
        try:
            self._gs_progress_bar.set(1.0)
            msg = f"Loaded {inserted} image{'s' if inserted != 1 else ''}."
            if skipped:
                msg += f"  ({skipped} already present, skipped.)"
            self._gs_progress_label.configure(text=msg, text_color="#00ff9f")
            self._gs_btn_add.configure(state="normal")
            self._gs_btn_load.configure(text="📥  Load Precomputed",
                                        state="normal")
            self._gs_btn_recompute.configure(state="normal")
            self._gs_refresh_dataset_list()
        except Exception:
            pass

    def _gs_ask_path_remap(self,
                           sample_path: str) -> tuple[str, str] | None:
        """Modal dialog asking the user for an optional path-prefix remap.

        Returns (src_prefix, dst_prefix) — both may be empty strings meaning
        "use paths verbatim".  Returns None if the user cancels.
        """
        dlg = ctk.CTkToplevel(self)
        dlg.title("Path remapping (optional)")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()
        dlg.resizable(False, False)

        result: dict = {}

        ctk.CTkLabel(
            dlg,
            text="Image paths in the file refer to the machine where "
                 "embeddings were computed.",
            font=ctk.CTkFont(size=11), wraplength=460, justify="left",
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 4), sticky="w")
        ctk.CTkLabel(
            dlg,
            text="If you have a local copy of the dataset, map the original "
                 "prefix to your local prefix so thumbnails resolve. "
                 "Leave both empty to keep paths unchanged.",
            font=ctk.CTkFont(size=10), text_color="#888",
            wraplength=460, justify="left",
        ).grid(row=1, column=0, columnspan=2, padx=14, pady=(0, 10), sticky="w")

        ctk.CTkLabel(
            dlg, text="Example path from file:",
            font=ctk.CTkFont(size=10), text_color="#888",
        ).grid(row=2, column=0, padx=14, pady=(0, 2), sticky="w")
        ctk.CTkLabel(
            dlg, text=sample_path or "(none)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#c0d8ff", wraplength=460, justify="left",
        ).grid(row=3, column=0, columnspan=2,
               padx=14, pady=(0, 12), sticky="w")

        ctk.CTkLabel(dlg, text="Replace this prefix:",
                     font=ctk.CTkFont(size=11)).grid(
            row=4, column=0, padx=14, pady=(0, 2), sticky="w")
        src_entry = ctk.CTkEntry(dlg, width=460)
        src_entry.grid(row=5, column=0, columnspan=2,
                       padx=14, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(dlg, text="With this prefix:",
                     font=ctk.CTkFont(size=11)).grid(
            row=6, column=0, padx=14, pady=(0, 2), sticky="w")
        dst_entry = ctk.CTkEntry(dlg, width=460)
        dst_entry.grid(row=7, column=0, columnspan=2,
                       padx=14, pady=(0, 14), sticky="ew")

        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.grid(row=8, column=0, columnspan=2, padx=14, pady=(0, 14), sticky="e")

        def _ok():
            result["src"] = src_entry.get()
            result["dst"] = dst_entry.get()
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        ctk.CTkButton(btns, text="Cancel", width=90, fg_color="#444",
                      hover_color="#666", command=_cancel).pack(
            side="right", padx=(8, 0))
        ctk.CTkButton(btns, text="Load", width=90,
                      command=_ok).pack(side="right")

        dlg.update_idletasks()
        # Centre on the main window
        try:
            pw = self.winfo_toplevel()
            x = pw.winfo_rootx() + (pw.winfo_width() - dlg.winfo_width()) // 2
            y = pw.winfo_rooty() + (pw.winfo_height() - dlg.winfo_height()) // 3
            dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

        self.wait_window(dlg)

        if not result:
            return None
        return result["src"], result["dst"]

    def _gs_remove_dataset(self, dataset_id: int) -> None:
        # Find dataset name for the confirmation message
        datasets = self._db.get_ext_gallery_datasets()
        ds = next((d for d in datasets if d["id"] == dataset_id), None)
        name  = ds["display_name"] if ds else "this dataset"
        count = ds["image_count"]  if ds else 0

        confirmed = tk.messagebox.askyesno(
            title="Remove dataset",
            message=f"Remove \"{name}\" ({count} image{'s' if count != 1 else ''})?\n\n"
                    "All stored embeddings for this dataset will be deleted.\n"
                    "This cannot be undone.",
            icon="warning",
        )
        if not confirmed:
            return

        self._db.remove_ext_gallery_dataset(dataset_id)
        self._gs_refresh_dataset_list()
        self._gs_progress_label.configure(text="Dataset removed.", text_color="#888")

    def _gs_trigger_recompute(self) -> None:
        """Ask the recognition tab to batch-recompute gallery positions using
        the current merged gallery (control + external datasets), then refresh."""
        if self._recognition_tab is None:
            self._gs_progress_label.configure(
                text="Recognition tab not available.", text_color="#ff6b6b")
            return
        self._gs_progress_label.configure(
            text="Recomputing positions…", text_color="#888")

        def _bg():
            try:
                self._recognition_tab._recompute_gallery_positions()
            except Exception as exc:
                try:
                    if self.winfo_exists():
                        self.after(0, lambda: self._gs_progress_label.configure(
                            text=f"Error: {exc}", text_color="#ff6b6b"))
                except Exception:
                    pass
                return
            try:
                if self.winfo_exists():
                    self.after(0, self._gs_recompute_done)
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _gs_recompute_done(self) -> None:
        self._gs_progress_label.configure(
            text="Positions updated.", text_color="#00ff9f")
        self._gs_cache = []
        self._refresh_gallery_search()

    # ── Gallery search data refresh ────────────────────────────────────────

    def _refresh_gallery_search(self) -> None:
        """Load stored gallery search positions from the DB."""
        body = self._gs_body
        for w in body.winfo_children():
            w.destroy()
        ctk.CTkLabel(body, text="Loading gallery search data…",
                     font=self._font_12, text_color="#555").grid(
            row=0, column=0, columnspan=10, pady=30)

        def _query():
            rows = self._db.get_gallery_search_data()

            # Group by identity
            identity_images: dict[str, list[dict]] = {}
            for r in rows:
                ident = r["identity_name"]
                if not ident:
                    continue
                pos = r["gallery_position"]
                if pos is None:
                    continue   # not yet computed
                identity_images.setdefault(ident, []).append(r)

            # Build a lookup: folder_path_prefix → dataset display_name
            # so we can detect whether a ref path lives inside an ext dataset folder.
            try:
                ext_datasets = self._db.get_ext_gallery_datasets()
            except Exception:
                ext_datasets = []
            ext_folder_map: list[tuple[str, str]] = [
                (str(Path(d["folder_path"])), d["display_name"])
                for d in ext_datasets
            ]

            def _ext_info(path: str) -> dict | None:
                """Return ext dataset info dict if path lives in an ext folder, else None."""
                if not path:
                    return None
                ref_p = Path(path)
                for folder, ds_name in ext_folder_map:
                    try:
                        ref_p.relative_to(folder)
                        return {
                            "dataset":  ds_name,
                            "folder":   folder,
                            "identity": ref_p.parent.name,
                            "filename": ref_p.name,
                        }
                    except ValueError:
                        continue
                return None

            # Pre-load CONTROL-ONLY embeddings (captures table) for the top-1 lookup.
            # Using get_all_embeddings_for_model would include adversarial images too.
            # ctrl_embs: only control (captures) images — used as gallery candidates
            # adv_embs:  only adversarial images — used as query embeddings
            model_names = self._engine.model_names
            ctrl_embs: dict[str, dict[str, object]] = {}
            adv_embs:  dict[str, dict[str, object]] = {}
            for mn in model_names:
                ctrl_embs[mn] = self._db.get_control_embeddings_for_model(mn)
                adv_embs[mn]  = self._db.get_adversarial_embeddings_for_model(mn)

            def _find_top1_any(adv_path: str) -> tuple[str, str] | None:
                """Return (identity_name, image_path) of the closest CONTROL or EXT
                gallery image to the adversarial image at adv_path, across all
                identities. Query embedding comes from adv_embs; gallery candidates
                come from ctrl_embs + ext gallery — never adversarial images."""
                import numpy as np

                id_scores: dict[str, list[float]] = {}

                for mn in model_names:
                    # Query = the adversarial image's embedding
                    query = adv_embs.get(mn, {}).get(adv_path)
                    if query is None:
                        continue
                    qnorm = np.linalg.norm(query)
                    if qnorm < 1e-10:
                        continue

                    # Gallery candidates = control images grouped by identity
                    ctrl_by_id: dict[str, dict[str, object]] = {}
                    for p, emb in ctrl_embs[mn].items():
                        ctrl_by_id.setdefault(Path(p).parent.name, {})[p] = emb

                    for id_name, embs in ctrl_by_id.items():
                        best = max(
                            float(np.dot(query, emb) /
                                  (qnorm * np.linalg.norm(emb) + 1e-10))
                            for emb in embs.values()
                        )
                        id_scores.setdefault(id_name, []).append(best)

                    # Ext gallery datasets
                    for ds in ext_datasets:
                        try:
                            ext_rows = self._db.get_ext_gallery_embeddings_for_dataset_model(
                                ds["id"], mn)
                        except Exception:
                            continue
                        ext_by_id: dict[str, list] = {}
                        for id_name, emb in ext_rows:
                            ext_by_id.setdefault(id_name, []).append(emb)
                        for id_name, embs in ext_by_id.items():
                            best = max(
                                float(np.dot(query, emb) /
                                      (qnorm * np.linalg.norm(emb) + 1e-10))
                                for emb in embs
                            )
                            id_scores.setdefault(id_name, []).append(best)

                if not id_scores:
                    return None

                # Average across models, pick best identity
                ranked = sorted(
                    ((name, sum(s) / len(s)) for name, s in id_scores.items()),
                    key=lambda x: x[1], reverse=True,
                )
                best_id = ranked[0][0]

                # Find the specific closest image for best_id
                best_score = -2.0
                best_img   = ""
                for mn in model_names:
                    query = adv_embs.get(mn, {}).get(adv_path)
                    if query is None:
                        continue
                    qnorm = np.linalg.norm(query)
                    if qnorm < 1e-10:
                        continue
                    # Search control images
                    for p, emb in ctrl_embs[mn].items():
                        if Path(p).parent.name != best_id:
                            continue
                        s = float(np.dot(query, emb) /
                                  (qnorm * np.linalg.norm(emb) + 1e-10))
                        if s > best_score:
                            best_score = s
                            best_img   = p
                    # Search ext gallery images (with actual paths)
                    for ds in ext_datasets:
                        try:
                            ext_rows = self._db.get_ext_gallery_embeddings_with_path(
                                ds["id"], mn)
                        except Exception:
                            continue
                        for img_path, id_name, emb in ext_rows:
                            if id_name != best_id:
                                continue
                            s = float(np.dot(query, emb) /
                                      (qnorm * np.linalg.norm(emb) + 1e-10))
                            if s > best_score:
                                best_score = s
                                best_img   = img_path   # real path now

                return (best_id, best_img) if best_id else None

            cache = []
            for identity, images in identity_images.items():
                positions  = [img["gallery_position"] for img in images]
                best_pos   = max(positions)   # highest rank = best attack
                worst_pos  = min(positions)   # lowest rank = worst attack
                avg_pos    = sum(positions) / len(positions)

                # Highest gallery position = best attack (model ranked target identity
                # furthest down — most other identities scored higher similarity).
                # Lowest gallery position = worst attack (model still recognises target).
                best_img_r  = max(images, key=lambda x: x["gallery_position"])
                worst_img_r = min(images, key=lambda x: x["gallery_position"])

                ref_path      = best_img_r["matched_image_path"] or ""
                ref_ext_info  = _ext_info(ref_path)

                # Top-1 any-identity match for the BEST attack image — this is the
                # identity the model confused the attacker with when the disguise worked.
                top1_result = _find_top1_any(best_img_r["image_path"])
                if top1_result:
                    top1_id, top1_img = top1_result
                    top1_ext = _ext_info(top1_img)   # None if control, dict if ext dataset
                else:
                    top1_id  = ""
                    top1_img = ""
                    top1_ext = None

                cache.append({
                    "identity":        identity,
                    "images":          images,
                    "positions":       positions,
                    "best_pos":        best_pos,
                    "worst_pos":       worst_pos,
                    "avg_pos":         avg_pos,
                    "best_img":        best_img_r["image_path"],
                    "best_ref":        ref_path,
                    "best_ref_ext":    ref_ext_info,
                    "worst_top1_id":   top1_id,
                    "worst_top1_img":  top1_img,
                    "worst_top1_ext":  top1_ext,
                    "n_images":        len(images),
                })

            cache.sort(key=lambda x: (x["best_pos"], x["avg_pos"]), reverse=True)

            # Gallery size stats: control group (captures table) + ext datasets
            try:
                ctrl_stats  = self._db.get_control_gallery_stats()   # {identities, images}
            except Exception:
                ctrl_stats  = {"identities": 0, "images": 0}
            try:
                ext_images   = sum(d["image_count"] for d in ext_datasets)
                ext_ids_sets: set[str] = set()
                for d in ext_datasets:
                    try:
                        ids = self._db.get_ext_gallery_identity_names(d["id"])
                        ext_ids_sets.update(ids)
                    except Exception:
                        pass
                ext_identities = len(ext_ids_sets)
            except Exception:
                ext_images     = 0
                ext_identities = 0

            gallery_stats = {
                "ctrl_identities": ctrl_stats.get("identities", 0),
                "ctrl_images":     ctrl_stats.get("images", 0),
                "ext_identities":  ext_identities,
                "ext_images":      ext_images,
                "ext_datasets":    len(ext_datasets) if ext_datasets else 0,
            }
            return cache, gallery_stats

        def _on_done(cache, gallery_stats):
            self._gs_cache = cache
            self._gs_gallery_stats = gallery_stats
            self._gs_page = 0
            self._render_gs_page()

        def _bg():
            try:
                cache, gallery_stats = _query()
            except Exception:
                import traceback
                traceback.print_exc()
                cache, gallery_stats = [], {}
            try:
                if self.winfo_exists():
                    self.after(0, lambda: _on_done(cache, gallery_stats))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_bg, daemon=True).start()

    # ── Gallery Search pagination ─────────────────────────────────────────

    def _gs_page_prev(self) -> None:
        if self._gs_page > 0:
            self._gs_page -= 1
            self._render_gs_page()
            _scroll_to_top(self._gs_body)

    def _gs_page_next(self) -> None:
        total_pages = max(1, -(-len(self._gs_cache) // _PAGE_SIZE))
        if self._gs_page < total_pages - 1:
            self._gs_page += 1
            self._render_gs_page()
            _scroll_to_top(self._gs_body)

    def _gs_page_goto(self) -> None:
        total_pages = max(1, -(-len(self._gs_cache) // _PAGE_SIZE))
        try:
            target = int(self._gs_entry_page.get().strip()) - 1
        except ValueError:
            self._gs_entry_page.delete(0, "end")
            return
        target = max(0, min(target, total_pages - 1))
        self._gs_page = target
        self._gs_entry_page.delete(0, "end")
        self._render_gs_page()
        _scroll_to_top(self._gs_body)

    @staticmethod
    def _gs_pos_colour(pos: int) -> str:
        """Colour-code a gallery search position.
        High position = model confused attacker with many others = best attack (green).
        Position 1 = model still recognises target = worst attack (red)."""
        if pos == 1:
            return "#ff6b6b"
        elif pos <= 5:
            return "#ffd166"
        else:
            return "#00ff9f"

    def _render_gs_page(self) -> None:
        body = self._gs_body
        for w in body.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        # Column layout:
        # 0=Rank  1=Identity  2=Positions  3=Best  4=Avg
        # 5=Best Attack  6=Closest (same id)  7=Closest (any id)
        _GS_COLS = 8
        for i in range(_GS_COLS):
            body.columnconfigure(i, weight=1)

        data  = self._gs_cache
        total = len(data)

        if total == 0:
            ctk.CTkLabel(body, text="No gallery search data yet.\n"
                         "Score adversarial images first, then refresh.",
                         font=self._font_12, text_color="#555").grid(
                row=0, column=0, columnspan=_GS_COLS, pady=30)
            gs = self._gs_gallery_stats
            if gs:
                ctrl_part = (f"{gs.get('ctrl_images',0)} ctrl / "
                             f"{gs.get('ctrl_identities',0)} id")
                ext_part  = (f"{gs.get('ext_images',0)} ext / "
                             f"{gs.get('ext_identities',0)} id")
                self._gs_summary.configure(
                    text=f"0 attackers  •  Gallery: {ctrl_part}  +  {ext_part}")
            else:
                self._gs_summary.configure(text="0 identities.")
            self._gs_lbl_page.configure(text="")
            self._gs_btn_prev.configure(state="disabled")
            self._gs_btn_next.configure(state="disabled")
            return

        total_pages = max(1, -(-total // _PAGE_SIZE))
        self._gs_page = max(0, min(self._gs_page, total_pages - 1))
        start = self._gs_page * _PAGE_SIZE
        end   = min(start + _PAGE_SIZE, total)
        page_data = data[start:end]

        # Header
        HDR_BG = "#16213e"
        headers = ["Rank", "Identity", "Positions (per image)",
                   "Best", "Avg",
                   "Best Attack", "Closest (same id)", "Closest (any id)"]
        for col, text in enumerate(headers):
            ctk.CTkLabel(body, text=text,
                         font=self._font_11_bold,
                         text_color="#00d4ff", fg_color=HDR_BG,
                         anchor="center").grid(
                row=0, column=col, padx=2, pady=(4, 4), sticky="ew")

        # Rows
        for local_idx, entry in enumerate(page_data):
            global_rank = start + local_idx + 1
            grid_row    = local_idx + 1
            medal       = _MEDAL.get(global_rank)
            txt_col     = medal if medal else "#e0e0e0"

            # Rank
            ctk.CTkLabel(body, text=f"#{global_rank}", text_color=txt_col,
                         font=self._font_11, anchor="center").grid(
                row=grid_row, column=0, padx=4, pady=3, sticky="ew")

            # Identity
            ctk.CTkLabel(body, text=entry["identity"], text_color=txt_col,
                         font=self._font_11, anchor="center").grid(
                row=grid_row, column=1, padx=4, pady=3, sticky="ew")

            # Positions — show 5 best (lowest) and 5 worst (highest); ellipsis if more
            pos_frame = ctk.CTkFrame(body, fg_color="transparent")
            pos_frame.grid(row=grid_row, column=2, padx=4, pady=3)
            all_pos   = sorted(entry["positions"])          # ascending = best first
            n         = len(all_pos)
            _N        = 5
            if n <= _N * 2:
                display = [(p, False) for p in all_pos]    # show all, no ellipsis
            else:
                best  = [(p, False) for p in all_pos[:_N]]
                worst = [(p, False) for p in all_pos[n - _N:]]
                display = best + [(None, True)] + worst     # None = ellipsis sentinel
            for item, is_ellipsis in display:
                if is_ellipsis:
                    ctk.CTkLabel(pos_frame, text="···",
                                 font=self._font_11,
                                 text_color="#555",
                                 width=18, anchor="center").pack(side="left", padx=1)
                else:
                    ctk.CTkLabel(pos_frame, text=str(item),
                                 font=self._font_11,
                                 text_color=self._gs_pos_colour(item),
                                 width=28, anchor="center").pack(side="left", padx=1)

            # Best
            best_pos = entry["best_pos"]
            ctk.CTkLabel(body, text=str(best_pos),
                         text_color=self._gs_pos_colour(best_pos),
                         font=self._font_11_bold, anchor="center").grid(
                row=grid_row, column=3, padx=4, pady=3, sticky="ew")

            # Avg
            avg_pos = entry["avg_pos"]
            ctk.CTkLabel(body, text=f"{avg_pos:.1f}",
                         text_color=self._gs_pos_colour(round(avg_pos)),
                         font=self._font_11, anchor="center").grid(
                row=grid_row, column=4, padx=4, pady=3, sticky="ew")

            # Best Attack thumbnail (highest gallery position = hardest to identify)
            self._add_thumb(body, entry["best_img"], grid_row, col=5,
                            label="best attack")

            # Closest (same identity) — reference for the best attack
            self._add_gs_ref_cell(body, entry["best_ref"],
                                  entry.get("best_ref_ext"), grid_row, col=6)

            # Closest (any identity) — top-1 gallery match for the worst attack
            top1_img = entry.get("worst_top1_img", "")
            top1_ext = entry.get("worst_top1_ext")
            self._add_gs_ref_cell(body, top1_img, top1_ext, grid_row, col=7)

        gs = self._gs_gallery_stats
        ctrl_part = (f"{gs.get('ctrl_images',0)} ctrl img / "
                     f"{gs.get('ctrl_identities',0)} identities")
        ext_part  = (f"{gs.get('ext_images',0)} ext img / "
                     f"{gs.get('ext_identities',0)} identities "
                     f"({gs.get('ext_datasets',0)} dataset"
                     f"{'s' if gs.get('ext_datasets',0) != 1 else ''})")
        total_gallery_ids = gs.get('ctrl_identities', 0) + gs.get('ext_identities', 0)
        self._gs_summary.configure(
            text=f"{total} attacker{'s' if total != 1 else ''} ranked  •  "
                 f"Gallery: {ctrl_part}  +  {ext_part}  "
                 f"= {total_gallery_ids} total identities  •  "
                 f"best position = highest gallery rank  •  "
                 f"showing {start + 1}–{end}")
        self._gs_lbl_page.configure(
            text=f"Page {self._gs_page + 1} / {total_pages}")
        self._gs_btn_prev.configure(
            state="normal" if self._gs_page > 0 else "disabled")
        self._gs_btn_next.configure(
            state="normal" if self._gs_page < total_pages - 1 else "disabled")

    def _add_gs_ref_cell(self, body, path: str, ext_info: dict | None,
                         grid_row: int, col: int,
                         override_label: str = "") -> None:
        """Render a reference cell in the Gallery Search table.

        For control images (ext_info is None) show the usual thumbnail.
        For external dataset images, load the image from its real path and show
        it alongside the dataset name, identity and filename — all centred.
        """
        if ext_info:
            # Coloured border frame signals this is an external dataset image
            frame = ctk.CTkFrame(body, fg_color="transparent",
                                 border_width=2, border_color="#00d4ff",
                                 corner_radius=6)
            frame.grid(row=grid_row, column=col, padx=6, pady=3)

            # Dataset name — shown as fallback if image cannot be loaded;
            # hidden once the image loads successfully (frame border stays as indicator)
            lbl_dataset  = ctk.CTkLabel(frame,
                             text=ext_info["dataset"],
                             font=self._font_11_bold,
                             text_color="#00d4ff",
                             anchor="center")
            lbl_dataset.pack(anchor="center", padx=6, pady=(4, 0))
            lbl_identity = ctk.CTkLabel(frame,
                             text=ext_info["identity"],
                             font=self._font_11,
                             text_color="#aaa",
                             anchor="center")
            lbl_identity.pack(anchor="center")
            lbl_filename = ctk.CTkLabel(frame,
                             text=ext_info["filename"],
                             font=ctk.CTkFont(family="Courier New", size=9),
                             text_color="#555",
                             anchor="center")
            lbl_filename.pack(anchor="center", pady=(0, 4))

            # Load the actual image if path exists on disk
            if path and Path(path).exists():
                placeholder = ctk.CTkLabel(frame, text="…", text_color="#444",
                                           font=self._font_11,
                                           width=_THUMB, height=_THUMB)
                placeholder.pack(anchor="center", pady=4, before=lbl_dataset)

                def _load(p=path, pl=placeholder,
                          ld=lbl_dataset, li=lbl_identity, lf=lbl_filename,
                          info=ext_info):
                    try:
                        img = Image.open(p).convert("RGB")
                        img.thumbnail((_THUMB, _THUMB))
                        out = Image.new("RGB", (_THUMB, _THUMB), (30, 30, 42))
                        out.paste(img, ((_THUMB - img.width) // 2,
                                        (_THUMB - img.height) // 2))
                    except Exception:
                        return   # keep text labels visible as fallback

                    def _apply(pil_img=out, pl=pl,
                               ld=ld, li=li, lf=lf, info=info):
                        try:
                            if not pl.winfo_exists():
                                return
                        except Exception:
                            return
                        ctk_img = ctk.CTkImage(light_image=pil_img,
                                               dark_image=pil_img,
                                               size=(_THUMB, _THUMB))
                        self._thumb_refs.append(ctk_img)
                        pl.configure(image=ctk_img, text="",
                                     cursor="hand2", width=_THUMB, height=_THUMB)
                        pl.bind("<Button-1>",
                                lambda e, pp=p: self._open_viewer(pp, "closest any-id"))

                        # Image loaded — hide all text, info moves to right-click
                        try:
                            ld.pack_forget()
                            li.pack_forget()
                            lf.pack_forget()
                        except Exception:
                            pass

                        def _show_context(event, pp=p, info=info):
                            menu = tk.Menu(pl, tearoff=0)
                            menu.add_command(
                                label=f"Dataset:   {info['dataset']}",
                                state="disabled")
                            menu.add_command(
                                label=f"Identity:  {info['identity']}",
                                state="disabled")
                            menu.add_command(
                                label=f"File:      {info['filename']}",
                                state="disabled")
                            menu.add_separator()
                            menu.add_command(
                                label="Copy path",
                                command=lambda: (
                                    pl.clipboard_clear(),
                                    pl.clipboard_append(pp)
                                ))
                            menu.add_command(
                                label="View full size",
                                command=lambda: self._open_viewer(pp, "closest any-id"))
                            try:
                                menu.tk_popup(event.x_root, event.y_root)
                            finally:
                                menu.grab_release()

                        pl.bind("<Button-3>", _show_context)

                    try:
                        if self.winfo_exists():
                            self.after(0, _apply)
                    except Exception:
                        pass
                threading.Thread(target=_load, daemon=True).start()
        else:
            if override_label:
                frame = ctk.CTkFrame(body, fg_color="transparent")
                frame.grid(row=grid_row, column=col, padx=4, pady=3)
                ctk.CTkLabel(frame,
                             text=override_label,
                             font=self._font_11_bold,
                             text_color="#e0e0e0",
                             anchor="center").pack(anchor="center")
                if path:
                    placeholder = ctk.CTkLabel(frame, text="…", text_color="#444",
                                               font=self._font_11,
                                               width=_THUMB, height=_THUMB)
                    placeholder.pack(anchor="center")

                    def _load(p=path, pl=placeholder):
                        try:
                            img = Image.open(p).convert("RGB")
                            img.thumbnail((_THUMB, _THUMB))
                            out = Image.new("RGB", (_THUMB, _THUMB), (30, 30, 42))
                            out.paste(img, ((_THUMB - img.width) // 2,
                                            (_THUMB - img.height) // 2))
                        except Exception:
                            return
                        def _apply(pil_img=out, pl=pl):
                            try:
                                if not pl.winfo_exists():
                                    return
                            except Exception:
                                return
                            ctk_img = ctk.CTkImage(light_image=pil_img,
                                                   dark_image=pil_img,
                                                   size=(_THUMB, _THUMB))
                            self._thumb_refs.append(ctk_img)
                            pl.configure(image=ctk_img, text="",
                                         cursor="hand2", width=_THUMB, height=_THUMB)
                            pl.bind("<Button-1>",
                                    lambda e, pp=p: self._open_viewer(pp, "closest any-id"))
                        try:
                            if self.winfo_exists():
                                self.after(0, _apply)
                        except Exception:
                            pass
                    threading.Thread(target=_load, daemon=True).start()
            else:
                self._add_thumb(body, path, grid_row, col, label="closest match")

    # ── CSV export ────────────────────────────────────────────────────────────
    #
    # Exports the FULL dataset of the currently-active sub-tab (not just the
    # 20 rows on the current page) to a user-chosen CSV file. Columns mirror
    # what the active tab displays on screen.
    #
    # The export runs against the in-memory cache (self._st_cache /
    # self._all_cache / self._gs_cache), so:
    #   • it's instant — no DB round-trip,
    #   • it reflects exactly what the user is looking at (same filter, same
    #     ordering, same per-model scores).
    # If the cache is empty (user hasn't visited this tab since startup, or
    # data was just cleared), the export is refused with a friendly hint to
    # refresh first.

    def _export_current_tab_csv(self) -> None:
        """Pick the right exporter based on the active sub-tab."""
        current = self._tabs.get()
        if current == "Current Standings":
            self._export_standings_csv()
        elif current == "All Images":
            self._export_all_images_csv()
        elif current == "Gallery Search":
            self._export_gallery_search_csv()
        else:
            messagebox.showinfo("Export CSV",
                                "Unknown tab — nothing to export.")

    @staticmethod
    def _default_csv_name(prefix: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}.csv"

    def _ask_save_path(self, default_name: str) -> str | None:
        """Show a Save-As dialog; return the chosen path or None on cancel."""
        return filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        ) or None

    def _write_csv(self, path: str, header: list[str],
                   rows: list[list]) -> tuple[bool, str]:
        """Write rows to `path`. Returns (ok, message)."""
        try:
            # utf-8-sig writes a BOM so Excel opens the file with the right encoding.
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.writer(fh, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
            return True, f"Exported {len(rows)} row(s) to {Path(path).name}"
        except Exception as exc:
            return False, f"Export failed: {exc}"

    def _notify_export(self, ok: bool, msg: str, path: str = "") -> None:
        """Show a non-blocking confirmation; on success include the folder."""
        if ok:
            full = f"{msg}\n\nLocation:\n{path}" if path else msg
            messagebox.showinfo("Export CSV", full)
        else:
            messagebox.showerror("Export CSV", msg)

    # ── Per-tab exporters ─────────────────────────────────────────────────────

    def _export_standings_csv(self) -> None:
        """Export the Current Standings tab (full leaderboard, all participants)."""
        if not self._st_cache:
            messagebox.showinfo(
                "Export CSV",
                "Nothing to export yet — the Current Standings cache is empty.\n"
                "Click ⟳ Refresh first, then try again.")
            return

        model_names = self._st_cache[0].get("model_names", [])
        header = (["rank", "attacker", "best_avg"]
                  + list(model_names)
                  + ["best_attack_path", "timestamp"])

        rows: list[list] = []
        for i, entry in enumerate(self._st_cache, start=1):
            scores = entry.get("scores", {})
            row = [
                i,
                entry.get("identity", ""),
                f"{entry.get('best_avg', 0.0):.4f}",
            ]
            for mn in model_names:
                s = scores.get(mn)
                row.append(f"{s:.4f}" if s is not None else "")
            ts = entry.get("ts", "") or ""
            row.append(entry.get("path", ""))
            # Normalise timestamp to "YYYY-MM-DD HH:MM:SS" like the UI
            row.append(ts[:19].replace("T", " ") if ts else "")
            rows.append(row)

        default_name = self._default_csv_name("standings")
        save_path = self._ask_save_path(default_name)
        if not save_path:
            return

        def _bg():
            ok, msg = self._write_csv(save_path, header, rows)
            try:
                if self.winfo_exists():
                    self.after(0, lambda: self._notify_export(ok, msg, save_path))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _export_all_images_csv(self) -> None:
        """Export the All Images tab (every scored image, current model filter)."""
        if not self._all_cache:
            messagebox.showinfo(
                "Export CSV",
                "Nothing to export yet — the All Images cache is empty.\n"
                "Click ⟳ Refresh first, then try again.")
            return

        is_average  = (self._all_cache_model == _ALL_MODELS)
        model_label = self._all_cache_model or _ALL_MODELS
        score_col   = "avg_similarity" if is_average else "similarity"

        header = [
            "rank", "attacker", "model", score_col,
            "attack_path", "closest_path", "timestamp",
            "pattern_id", "is_frontal", "is_profile", "is_pitch",
        ]

        rows: list[list] = []
        for i, entry in enumerate(self._all_cache, start=1):
            tags = entry.get("tags") or {}
            ts = entry.get("created_at", "") or ""
            rows.append([
                i,
                entry.get("attacker", ""),
                model_label,
                f"{entry.get('avg_score', 0.0):.4f}",
                entry.get("image_path", ""),
                entry.get("closest_path", "") or "",
                ts[:19].replace("T", " ") if ts else "",
                tags.get("pattern_id", 0),
                int(bool(tags.get("is_frontal"))),
                int(bool(tags.get("is_profile"))),
                int(bool(tags.get("is_pitch"))),
            ])

        # Include the model filter in the default filename so multiple exports
        # for different models don't overwrite each other.
        safe_model = (model_label.replace(" ", "_")
                                  .replace("(", "")
                                  .replace(")", "")
                                  .replace("/", "_"))
        default_name = self._default_csv_name(f"all_images_{safe_model}")
        save_path = self._ask_save_path(default_name)
        if not save_path:
            return

        def _bg():
            ok, msg = self._write_csv(save_path, header, rows)
            try:
                if self.winfo_exists():
                    self.after(0, lambda: self._notify_export(ok, msg, save_path))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _export_gallery_search_csv(self) -> None:
        """Export the Gallery Search tab (one row per attacker identity)."""
        if not self._gs_cache:
            messagebox.showinfo(
                "Export CSV",
                "Nothing to export yet — the Gallery Search cache is empty.\n"
                "Click ⟳ Refresh first, then try again.")
            return

        header = [
            "rank", "identity", "n_images",
            "best_position", "worst_position", "avg_position",
            "all_positions",
            "best_attack_path",
            "closest_same_id_path",
            "closest_any_id_path",
            "closest_any_id_identity",
            "closest_any_id_dataset",
        ]

        rows: list[list] = []
        for i, entry in enumerate(self._gs_cache, start=1):
            positions = entry.get("positions", []) or []
            # Sort positions ascending in the cell so it reads naturally
            all_pos_str = ";".join(str(p) for p in sorted(positions))
            top1_ext    = entry.get("worst_top1_ext") or {}
            rows.append([
                i,
                entry.get("identity", ""),
                entry.get("n_images", len(positions)),
                entry.get("best_pos", ""),
                entry.get("worst_pos", ""),
                f"{entry.get('avg_pos', 0.0):.2f}",
                all_pos_str,
                entry.get("best_img", "") or "",
                entry.get("best_ref", "") or "",
                entry.get("worst_top1_img", "") or "",
                entry.get("worst_top1_id", "") or "",
                top1_ext.get("dataset", "") if top1_ext else "",
            ])

        default_name = self._default_csv_name("gallery_search")
        save_path = self._ask_save_path(default_name)
        if not save_path:
            return

        def _bg():
            ok, msg = self._write_csv(save_path, header, rows)
            try:
                if self.winfo_exists():
                    self.after(0, lambda: self._notify_export(ok, msg, save_path))
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=_bg, daemon=True).start()

    # ── Thumbnail loader ──────────────────────────────────────────────────────

    def _add_thumb(self, body, path: str, grid_row: int, col: int,
                   label: str, tags: dict | None = None) -> None:
        if not path:
            ctk.CTkLabel(body, text="—", text_color="#555",
                         font=self._font_11).grid(
                row=grid_row, column=col, padx=4, pady=2)
            return

        # Container frame: thumbnail image + optional tag badge to the right
        container = ctk.CTkFrame(body, fg_color="transparent")
        container.grid(row=grid_row, column=col, padx=4, pady=2)

        # Place a lightweight placeholder immediately so the row renders at
        # the right height without blocking on disk I/O.
        placeholder = ctk.CTkLabel(container, text="…", text_color="#444",
                                   font=self._font_11,
                                   width=_THUMB, height=_THUMB)
        placeholder.pack(side="left")

        # Tag badge (drawn immediately — no I/O needed)
        if tags:
            pid = tags.get("pattern_id", 0)
            badge = _tag_badge_text(tags)
            badge_col = _PATTERN_COLOURS.get(pid, "#555")
            ctk.CTkLabel(container, text=badge,
                         font=self._font_7,
                         text_color=badge_col).pack(side="left", padx=(3, 0))

        def _load() -> None:
            # Only PIL work here (disk I/O + resize) — no Tk/CTk objects
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((_THUMB, _THUMB))
                out = Image.new("RGB", (_THUMB, _THUMB), (30, 30, 42))
                out.paste(img, ((_THUMB - img.width) // 2,
                                (_THUMB - img.height) // 2))
            except Exception:
                return
            # Hand the PIL image to the main thread for CTkImage creation.
            # If the widget (or the whole app) was destroyed while we were
            # loading, .after() will raise — drop the result silently.
            try:
                if not self.winfo_exists():
                    return
                self.after(0, lambda: _apply(out))
            except (RuntimeError, tk.TclError):
                return

        def _apply(pil_img) -> None:
            # All Tk/CTk object creation must happen on the main thread
            try:
                if not placeholder.winfo_exists():
                    return
            except Exception:
                return
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                   size=(_THUMB, _THUMB))
            self._thumb_refs.append(ctk_img)
            placeholder.configure(image=ctk_img, text="",
                                  cursor="hand2", width=_THUMB, height=_THUMB)
            placeholder.bind("<Button-1>",
                             lambda e, p=path, t=label: self._open_viewer(p, t))

        threading.Thread(target=_load, daemon=True).start()

    def _add_delete_btn(self, body, image_path: str, grid_row: int,
                        col: int) -> None:
        ctk.CTkButton(
            body, text="🗑", width=32, height=28,
            fg_color="#3a0000", hover_color="#700000",
            font=self._font_12,
            command=lambda p=image_path: self._delete_row(p),
        ).grid(row=grid_row, column=col, padx=4, pady=2)

    def _delete_row(self, image_path: str) -> None:
        # Remember current page indices and scroll position so the refresh
        # doesn't snap us back to page 1 / top of body.
        # _render_*_page() already clamps the page to the new total, so if
        # the deletion shrank the page count we'll fall back to the new
        # last page automatically — in which case we scroll to the top of
        # that new page rather than restoring the old scroll fraction.
        saved_st_page    = self._st_page
        saved_all_page   = self._all_page
        saved_gs_page    = self._gs_page
        saved_st_scroll  = _capture_scroll(self._standings_body)
        saved_all_scroll = _capture_scroll(self._all_body)
        saved_gs_scroll  = _capture_scroll(self._gs_body)

        self._db.delete_attacks_by_image(image_path)
        self._db_dirty = True
        self._refresh(force=True)

        self._st_page  = saved_st_page
        self._all_page = saved_all_page
        self._gs_page  = saved_gs_page

        # Re-render the visible sub-tab at the restored page, then either
        # scroll to top (if the clamp had to drop us to an earlier page) or
        # restore the previous scroll fraction (still on the same page).
        current = self._tabs.get()
        if current == "Current Standings":
            self._render_standings_page()
            if self._st_page == saved_st_page:
                _restore_scroll(self._standings_body, saved_st_scroll)
            else:
                _scroll_to_top(self._standings_body)
        elif current == "Gallery Search":
            self._render_gs_page()
            if self._gs_page == saved_gs_page:
                _restore_scroll(self._gs_body, saved_gs_scroll)
            else:
                _scroll_to_top(self._gs_body)
        else:
            self._render_all_page()
            if self._all_page == saved_all_page:
                _restore_scroll(self._all_body, saved_all_scroll)
            else:
                _scroll_to_top(self._all_body)

        if self._recognition_tab is not None:
            self._recognition_tab._refresh_thumbnails()
            self._recognition_tab._refresh_best_score()

    # ── Image viewer ──────────────────────────────────────────────────────────

    def _open_viewer(self, path: str, title: str) -> None:
        # Prefer the lossless HQ companion (e.g. img_007_hq.png) when it exists,
        # so the viewer always shows the full-resolution original rather than the
        # 112×112 aligned crop.
        hq = _hq_path(path)
        display_path = hq if Path(hq).exists() else path
        try:
            img = Image.open(display_path).convert("RGB")
        except Exception:
            return
        popup = tk.Toplevel(self)
        popup.title(f"{title}  —  {Path(display_path).name}")
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
        tk.Label(popup, text=display_path, fg="#555", bg="#0d0d1a",
                 font=("Courier New", 9), wraplength=int(sw * 0.8)).pack(pady=(0, 6))
        popup.bind("<Escape>", lambda e: popup.destroy())
        lbl.bind("<Button-1>", lambda e: popup.destroy())
        popup.update_idletasks()
        popup.geometry(
            f"+{(sw - popup.winfo_width()) // 2}+{(sh - popup.winfo_height()) // 2}")

    # ── Clear ─────────────────────────────────────────────────────────────────

    # ── Podium window ─────────────────────────────────────────────────────────

    def _open_podium(self) -> None:
        """Open (or focus) the live podium window."""
        if self._podium_window is not None and self._podium_window.winfo_exists():
            self._podium_window.lift()
            self._podium_window.focus_force()
            return
        self._podium_window = PodiumWindow(self, db=self._db,
                                           engine=self._engine)

    def _get_top3_data(self) -> list[dict]:
        """Return up to 3 dicts: {rank, name, score, adv_path, ctrl_path}."""
        leaderboard = self._db.get_user_leaderboard()
        if not leaderboard:
            return []

        all_rows = self._db.get_ranking(model_name=None, limit=10_000)
        img_scores: dict[str, dict[str, float]] = {}
        img_meta:   dict[str, dict]             = {}
        for r in all_rows:
            if r.image_path not in img_scores:
                img_scores[r.image_path] = {}
                img_meta[r.image_path]   = {
                    "attacker": r.attacker_name,
                    "closest":  r.matched_image_path or ""}
            img_scores[r.image_path][r.model_name] = r.similarity_score

        img_avg: dict[str, float] = {}
        for path, scores in img_scores.items():
            if scores:
                img_avg[path] = sum(scores.values()) / len(scores)

        result = []
        for rank, entry in enumerate(leaderboard[:3], start=1):
            identity   = entry["identity_name"]
            best_avg   = entry["best_avg_score"]
            candidates = [p for p, m in img_meta.items()
                          if m["attacker"] == identity]
            adv_path  = ""
            ctrl_path = ""
            if candidates:
                best_path = min(candidates, key=lambda p: img_avg.get(p, 1.0))
                adv_path  = best_path
                ctrl_path = img_meta[best_path]["closest"]
            # Prefer high-quality companion images for the podium display
            adv_hq  = _hq_path(adv_path)  if adv_path  else ""
            ctrl_hq = _hq_path(ctrl_path) if ctrl_path else ""
            result.append({"rank": rank, "name": identity,
                            "score": best_avg,
                            "adv_path": adv_path,
                            "ctrl_path": ctrl_path,
                            "adv_hq_path":  adv_hq  if adv_hq  and Path(adv_hq).exists()  else "",
                            "ctrl_hq_path": ctrl_hq if ctrl_hq and Path(ctrl_hq).exists() else ""})
        return result

    def _notify_podium(self) -> None:
        """Push fresh top-3 data to the podium window if it is open."""
        if (self._podium_window is not None
                and self._podium_window.winfo_exists()):
            self._podium_window.update_data(self._get_top3_data())

    def _confirm_clear(self) -> None:
        import tkinter.messagebox as mb
        import shutil

        choice = _ClearDialog(self).result
        if choice is None:
            return

        if choice == "rankings":
            deleted = self._db.clear_attacks()
            self._db_dirty = True
            self._refresh(force=True)
            if self._recognition_tab is not None:
                self._recognition_tab._refresh_thumbnails()
                self._recognition_tab._refresh_best_score()

        elif choice == "all":
            ok = mb.askyesno(
                "Delete Everything",
                "Permanently delete:\n"
                "  • All attack, capture and user DB records\n"
                "  • All images (users/, comparisons/, attacks/, adversarial/)\n\n"
                "Cannot be undone. Continue?",
                icon="warning")
            if not ok:
                return
            counts = self._db.clear_all_data()
            for folder in ("users", "comparisons", "attacks", "adversarial"):
                p = Path(self._data_dir) / folder
                if p.exists():
                    shutil.rmtree(p)
            self._db_dirty = True
            self._refresh(force=True)
            if self._capture_tab is not None:
                self._capture_tab.refresh_after_clear()
            if self._recognition_tab is not None:
                self._recognition_tab.refresh_after_clear()



# ── Podium Window ─────────────────────────────────────────────────────────────

_PODIUM_BG    = "#0d0d1a"
_PODIUM_PANEL = "#12122a"
_MEDAL_BG     = {1: "#2a2200", 2: "#1e1e1e", 3: "#1a1000"}
_MEDAL_BORDER = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
_MEDAL_LABEL  = {1: "🥇  1st Place", 2: "🥈  2nd Place", 3: "🥉  3rd Place"}
_PODIUM_ORDER = [1, 2, 3]

_IMG_MIN = 40
_IMG_MAX = 500

# Non-image chrome height per slot (px) — tightened paddings (padx=10 everywhere):
#   medal label  (pady 8+3,  font 9-15px)  = 30
#   "Adversarial" caption (pady 2+4)       = 18
#   "Reference"   caption (pady 2+3)       = 17
#   name label   (pady 4+1,  font 9-15px)  = 24
#   score label  (pady 0+10, font 8-12px)  = 26
#   slot border + top/bottom pad           = 16
_CHROME_H = 30 + 18 + 17 + 24 + 26 + 16   # 131 px
_CHROME_W = 20   # left+right inner padding (padx=10 each side)


class PodiumWindow(ctk.CTkToplevel):
    """Standalone resizable window — live top-3 podium."""

    def __init__(self, ranking_tab: "RankingTab", db: Database,
                 engine: "EmbeddingEngine"):
        super().__init__(ranking_tab)
        self._ranking_tab = ranking_tab
        self._db          = db
        self._engine      = engine
        self._data: list[dict] = []
        self._resize_job       = None

        self.title("🏆  Live Podium — Top 3")
        self.geometry("960x700")
        self.minsize(480, 420)
        self.configure(fg_color=_PODIUM_BG)

        self._build_ui()
        # Load data after the window has been fully mapped & measured
        self.after(150, lambda: self.update_data(ranking_tab._get_top3_data()))
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Fixed-height header
        hdr = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=0, height=44)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="🏆  LIVE PODIUM",
                     font=ctk.CTkFont(family="Courier New", size=15, weight="bold"),
                     text_color="#FFD700").pack(side="left", padx=16, pady=8)

        # Outer frame — we bind <Configure> HERE (on the container, not the slots)
        # so we measure the available area before telling slots how big to be.
        self._arena = ctk.CTkFrame(self, fg_color="transparent")
        self._arena.pack(fill="both", expand=True, padx=12, pady=10)
        self._arena.bind("<Configure>", self._on_arena_resize)

        for c in range(3):
            self._arena.columnconfigure(c, weight=1, uniform="pod")
        self._arena.rowconfigure(0, weight=1)

        self._slots: dict[int, "_PodiumSlot"] = {}
        for col, rank in enumerate(_PODIUM_ORDER):
            slot = _PodiumSlot(self._arena, rank=rank)
            slot.grid(row=0, column=col, sticky="nsew", padx=6, pady=4)
            self._slots[rank] = slot

    # ── Data update ───────────────────────────────────────────────────────────

    def update_data(self, data: list[dict]) -> None:
        self._data = data
        img_size   = self._calc_img_size()
        by_rank    = {d["rank"]: d for d in data}
        for rank in (1, 2, 3):
            self._slots[rank].apply_data(by_rank.get(rank), img_size)
        pass

    # ── Sizing — driven by the ARENA, not by slot content ────────────────────

    def _calc_img_size(self) -> int:
        """
        Compute image size from the arena's actual pixel dimensions.
        The arena is the container that holds the three slots; its size is set
        by the window manager and does NOT depend on image content, so there is
        no feedback loop.
        """
        aw = self._arena.winfo_width()
        ah = self._arena.winfo_height()

        if aw < 30 or ah < 30:   # not yet mapped — use window geometry fallback
            try:
                ww = self.winfo_width()
                wh = self.winfo_height()
            except Exception:
                ww, wh = 960, 700
            aw = max(30, ww - 24)
            ah = max(30, wh - 44 - 20)  # subtract header + arena padding

        # Width: fit inside one column (3 equal cols with ~12 px gaps)
        col_w  = (aw - 36) // 3
        size_w = max(_IMG_MIN, min(_IMG_MAX, col_w - _CHROME_W))

        # Height: subtract all non-image chrome, then divide by 2 (two images stacked)
        avail_h = max(0, ah - _CHROME_H - 8)   # 8 px arena top/bottom pad
        size_h  = max(_IMG_MIN, min(_IMG_MAX, avail_h // 2))

        return min(size_w, size_h)

    def _on_arena_resize(self, event) -> None:
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(80, self._do_resize)

    def _do_resize(self) -> None:
        self._resize_job = None
        if self._data:
            self.update_data(self._data)


class _PodiumSlot(ctk.CTkFrame):
    """
    One podium column.  Accepts explicit img_size from PodiumWindow so it never
    drives its own size — breaking the content-→-size feedback loop.
    """

    def __init__(self, parent, rank: int):
        super().__init__(parent,
                         fg_color=_MEDAL_BG.get(rank, _PODIUM_PANEL),
                         corner_radius=12,
                         border_width=2,
                         border_color=_MEDAL_BORDER.get(rank, "#333"))
        self._img_refs: list = []
        self.columnconfigure(0, weight=1)
        # Chrome rows are fixed height; image rows share remaining space equally
        for r in (0, 2, 4, 5, 6):
            self.rowconfigure(r, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=1)

        # Medal label — ref kept for adaptive font updates
        self._medal_lbl = ctk.CTkLabel(self,
                     text=_MEDAL_LABEL.get(rank, f"#{rank}"),
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=_MEDAL_BORDER.get(rank, "#ccc"),
                     wraplength=1)
        self._medal_lbl.grid(row=0, column=0, pady=(8, 3), padx=10, sticky="ew")

        # Adversarial image
        self._adv_lbl = ctk.CTkLabel(self, text="", fg_color="#0d0d1a",
                                     corner_radius=6)
        self._adv_lbl.grid(row=1, column=0, padx=8, pady=(0, 0), sticky="n")
        self._adv_cap = ctk.CTkLabel(self, text="Adversarial",
                     font=ctk.CTkFont(size=9), text_color="#666")
        self._adv_cap.grid(row=2, column=0, pady=(2, 4), padx=10, sticky="ew")

        # Reference image
        self._ctrl_lbl = ctk.CTkLabel(self, text="", fg_color="#0d0d1a",
                                      corner_radius=6)
        self._ctrl_lbl.grid(row=3, column=0, padx=8, pady=(0, 0), sticky="n")
        self._ref_cap = ctk.CTkLabel(self, text="Reference",
                     font=ctk.CTkFont(size=9), text_color="#666")
        self._ref_cap.grid(row=4, column=0, pady=(2, 3), padx=10, sticky="ew")

        # Name — wraplength set dynamically to prevent border overflow
        self._name_lbl = ctk.CTkLabel(self, text="—",
                                      font=ctk.CTkFont(size=13, weight="bold"),
                                      text_color="#e0e0e0",
                                      wraplength=1)
        self._name_lbl.grid(row=5, column=0, pady=(4, 1), padx=10, sticky="ew")

        # Score — wraplength set dynamically
        self._score_lbl = ctk.CTkLabel(self, text="",
                                       font=ctk.CTkFont(family="Courier New", size=11),
                                       text_color="#aaa",
                                       wraplength=1)
        self._score_lbl.grid(row=6, column=0, pady=(0, 10), padx=10, sticky="ew")

    def apply_data(self, entry: "dict | None", img_size: int) -> None:
        self._img_refs.clear()

        # ── Derive font sizes from img_size so everything scales with the window
        # img_size ~40..500; map to readable font range
        medal_fs = max(9,  min(15, img_size // 10 + 5))
        cap_fs   = max(7,  min(11, img_size // 14 + 4))
        name_fs  = max(9,  min(15, img_size // 10 + 4))
        score_fs = max(8,  min(12, img_size // 12 + 4))

        # wraplength = slot inner width ≈ img_size (images fill the column)
        wrap = max(40, img_size - 4)

        self._medal_lbl.configure(
            font=ctk.CTkFont(size=medal_fs, weight="bold"), wraplength=wrap)
        self._adv_cap.configure(font=ctk.CTkFont(size=cap_fs))
        self._ref_cap.configure(font=ctk.CTkFont(size=cap_fs))
        self._name_lbl.configure(
            font=ctk.CTkFont(size=name_fs, weight="bold"),  wraplength=wrap)
        self._score_lbl.configure(
            font=ctk.CTkFont(family="Courier New", size=score_fs), wraplength=wrap)

        if entry:
            score     = entry["score"]
            # Prefer high-quality companion images when available
            adv_src   = entry.get("adv_hq_path")  or entry.get("adv_path",  "")
            ctrl_src  = entry.get("ctrl_hq_path") or entry.get("ctrl_path", "")
            adv_img   = self._load_img(adv_src,  img_size)
            ctrl_img  = self._load_img(ctrl_src, img_size)
            score_col = "#00ff9f" if score < _THRESHOLD else "#ff6b6b"
            self._name_lbl.configure( text=entry["name"],              text_color="#e0e0e0")
            self._score_lbl.configure(text=f"Avg Similarity: {score:.4f}", text_color=score_col)
        else:
            adv_img = ctrl_img = None
            self._name_lbl.configure( text="—", text_color="#444")
            self._score_lbl.configure(text="",  text_color="#444")

        ph = self._placeholder(img_size)
        fi = adv_img  or ph
        fc = ctrl_img or ph
        self._img_refs.extend([fi, fc])
        self._adv_lbl.configure( image=fi, text="", width=img_size, height=img_size)
        self._ctrl_lbl.configure(image=fc, text="", width=img_size, height=img_size)

    @staticmethod
    def _load_img(path: str, size: int) -> "ctk.CTkImage | None":
        if not path:
            return None
        try:
            img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
            return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        except Exception:
            return None

    @staticmethod
    def _placeholder(size: int) -> ctk.CTkImage:
        img = Image.new("RGB", (size, size), "#1e1e2e")
        return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))


# ── Clear dialog ──────────────────────────────────────────────────────────────

class _ClearDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Clear Data")
        self.geometry("380x200")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ctk.CTkLabel(self, text="What would you like to delete?",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(20, 4))
        ctk.CTkLabel(self,
                     text="'Rankings only' keeps saved images.\n"
                          "'Delete Everything' removes all DB records\nand all images from disk.",
                     font=ctk.CTkFont(size=11), text_color="#aaa",
                     justify="center").pack(pady=(0, 16))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack()
        ctk.CTkButton(row, text="Rankings Only", width=130,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=lambda: self._pick("rankings")).pack(side="left", padx=6)
        ctk.CTkButton(row, text="Delete Everything", width=150,
                      fg_color="#3a0000", hover_color="#700000",
                      command=lambda: self._pick("all")).pack(side="left", padx=6)
        ctk.CTkButton(self, text="Cancel", width=100, fg_color="transparent",
                      border_width=1, command=self.destroy).pack(pady=(12, 0))
        self.wait_window()

    def _pick(self, value: str) -> None:
        self.result = value
        self.destroy()
