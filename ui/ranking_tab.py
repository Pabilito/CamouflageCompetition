"""
ui/ranking_tab.py  —  Ranking tab

Two sub-tabs:
  • Current Standings  — one row per identity, their best (lowest) avg score
  • All Images         — every scored image with model filter dropdown
"""

import threading
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.database import Database, AttackRow
from core.embeddings import EmbeddingEngine

_THUMB      = 56
_ALL_MODELS = "All models (average)"
_THRESHOLD  = 0.2565
_PAGE_SIZE  = 30

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


class RankingTab(ctk.CTkFrame):
    def __init__(self, parent, db: Database, data_dir: str,
                 engine: EmbeddingEngine, capture_tab=None, recognition_tab=None):
        super().__init__(parent, fg_color="transparent")
        self._db              = db
        self._data_dir        = data_dir
        self._engine          = engine
        self._capture_tab     = capture_tab
        self._recognition_tab = recognition_tab
        self._thumb_refs: list = []
        self._podium_window: "PodiumWindow | None" = None

        # ── Standings state ───────────────────────────────────────────────────
        self._st_cache: list[dict] = []    # full sorted leaderboard
        self._st_page:  int        = 0     # current 0-based page index

        # ── All Images state ──────────────────────────────────────────────────
        # Cached sorted lists; rebuilt only on explicit refresh or model change.
        # Switching pages just slices from the cache — no DB query, no image I/O.
        self._all_cache: list[dict] = []   # current model's full sorted list
        self._all_cache_model: str  = ""   # which model the cache belongs to
        self._all_page: int         = 0    # current 0-based page index

        self._build_ui()
        self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        toolbar.columnconfigure(4, weight=1)

        ctk.CTkLabel(toolbar, text="Adversarial Attack Ranking",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        ctk.CTkButton(toolbar, text="⟳  Refresh", width=100,
                      command=self._refresh).grid(row=0, column=1, padx=(20, 0))

        ctk.CTkButton(toolbar, text="🗑  Clear All", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._confirm_clear).grid(row=0, column=2, padx=(8, 0))

        ctk.CTkButton(toolbar, text="🏆  Podium View", width=130,
                      fg_color="#1a3a1a", hover_color="#2a5a2a",
                      command=self._open_podium).grid(row=0, column=3, padx=(8, 0))


        ctk.CTkLabel(toolbar,
                     text=f"Avg. similarity threshold: {_THRESHOLD}  •  "
                          "click thumbnails to enlarge  •  🗑 to delete row",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=0, column=4, padx=(16, 0), sticky="w")

        # ── Sub-tab view ──────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(self, corner_radius=6)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 4))
        self._tabs.add("Current Standings")
        self._tabs.add("All Images")
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

    def _refresh(self) -> None:
        self._thumb_refs.clear()
        current = self._tabs.get()
        if current == "Current Standings":
            self._st_cache = []          # invalidate so DB is re-queried
            self._refresh_standings()
        else:
            self._all_cache_model = ""   # invalidate so DB is re-queried
            self._refresh_all_images()
        self._notify_podium()

    def _on_subtab_change(self) -> None:
        current = self._tabs.get()
        if current == "Current Standings":
            if self._st_cache:
                self._render_standings_page()   # instant — reuse cache
            else:
                self._refresh_standings()       # first visit
        else:
            if self._all_cache_model == self._model_var.get():
                self._render_all_page()         # instant — reuse cache
            else:
                self._refresh_all_images()
        self._notify_podium()

    # ── Tab hooks ─────────────────────────────────────────────────────────────

    def on_tab_shown(self) -> None:
        self._refresh()

    # ── Current Standings ─────────────────────────────────────────────────────

    def _refresh_standings(self) -> None:
        """Rebuild cache from DB, then render page 0."""
        leaderboard  = self._db.get_user_leaderboard()
        model_names  = self._engine.model_names
        all_rows     = self._db.get_ranking(model_name=None, limit=10_000)

        # image_path → {model_name: score}  /  metadata
        img_scores: dict[str, dict[str, float]] = {}
        img_meta:   dict[str, dict] = {}
        for r in all_rows:
            if r.image_path not in img_scores:
                img_scores[r.image_path] = {}
                img_meta[r.image_path]   = {
                    "attacker":    r.attacker_name,
                    "created_at":  r.created_at}
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
                "model_names": model_names,
                "col_min":    col_min,
            }
            for e in leaderboard
        ]
        self._st_page = 0
        self._render_standings_page()

    # ── Standings pagination helpers ──────────────────────────────────────────

    def _st_page_prev(self) -> None:
        if self._st_page > 0:
            self._st_page -= 1
            self._render_standings_page()

    def _st_page_next(self) -> None:
        total_pages = max(1, -(-len(self._st_cache) // _PAGE_SIZE))
        if self._st_page < total_pages - 1:
            self._st_page += 1
            self._render_standings_page()

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
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
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
                         font=ctk.CTkFont(size=11, weight="bold"),
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
                             font=ctk.CTkFont(size=11), anchor="center",
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

            self._add_thumb(body, path, grid_row, col=8, label="best attack")
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
        Rebuilds the cache for the selected model, resets to page 0, renders.
        """
        selected = self._model_var.get()

        # ── Rebuild cache ─────────────────────────────────────────────────────
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
                averaged.append({"attacker": attacker, "avg_score": avg,
                                 "image_path": img_path, "closest_path": closest,
                                 "created_at": rows[0].created_at})
            averaged.sort(key=lambda x: x["avg_score"])
            self._all_cache = averaged
        else:
            rows = self._db.get_ranking(model_name=selected, limit=10_000)
            self._all_cache = [
                {"attacker": r.attacker_name, "avg_score": r.similarity_score,
                 "image_path": r.image_path, "closest_path": r.matched_image_path,
                 "created_at": r.created_at}
                for r in rows
            ]

        self._all_cache_model = selected
        self._all_page = 0
        self._render_all_page()

    # ── Pagination helpers ────────────────────────────────────────────────────

    def _page_prev(self) -> None:
        if self._all_page > 0:
            self._all_page -= 1
            self._render_all_page()

    def _page_next(self) -> None:
        total_pages = max(1, -(-len(self._all_cache) // _PAGE_SIZE))  # ceil div
        if self._all_page < total_pages - 1:
            self._all_page += 1
            self._render_all_page()

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

    def _draw_all_header(self, body, is_average: bool) -> None:
        score_label = "Avg Similarity ↓" if is_average else "Similarity ↓"
        labels = ["Rank", "Attacker", score_label, "Attack",
                  "Closest", "Timestamp", ""]
        for col, text in enumerate(labels):
            ctk.CTkLabel(body, text=text,
                         font=ctk.CTkFont(size=11, weight="bold"),
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
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
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
                                entry["created_at"])

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
                       ts: str) -> None:
        medal   = _MEDAL.get(rank)
        txt_col = medal if medal else "#e0e0e0"

        def cell(col_name: str, text: str, color: str = "#ccc") -> None:
            ctk.CTkLabel(body, text=text, text_color=color,
                         font=ctk.CTkFont(size=11), anchor="center").grid(
                row=grid_row, column=_C_ALL[col_name], padx=4, pady=3, sticky="ew")

        cell("rank",      f"#{rank}",                 color=txt_col)
        cell("attacker",  attacker,                   color=txt_col)
        cell("score",     f"{score:.4f}",            color=_sc_col(score))
        cell("timestamp", ts[:19].replace("T", " "), color="#666")

        self._add_thumb(body, img_path,      grid_row, _C_ALL["attack"],  "attack frame")
        self._add_thumb(body, closest_path,  grid_row, _C_ALL["closest"], "closest match")
        self._add_delete_btn(body, img_path, grid_row, _C_ALL["delete"])

    # ── Cell helpers ──────────────────────────────────────────────────────────

    def _add_thumb(self, body, path: str, grid_row: int, col: int,
                   label: str) -> None:
        if not path:
            ctk.CTkLabel(body, text="—", text_color="#555",
                         font=ctk.CTkFont(size=11)).grid(
                row=grid_row, column=col, padx=4, pady=2)
            return

        # Place a lightweight placeholder immediately so the row renders at
        # the right height without blocking on disk I/O.
        placeholder = ctk.CTkLabel(body, text="…", text_color="#444",
                                   font=ctk.CTkFont(size=11),
                                   width=_THUMB, height=_THUMB)
        placeholder.grid(row=grid_row, column=col, padx=4, pady=2)

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
            # Hand the PIL image to the main thread for CTkImage creation
            self.after(0, lambda: _apply(out))

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
            font=ctk.CTkFont(size=12),
            command=lambda p=image_path: self._delete_row(p),
        ).grid(row=grid_row, column=col, padx=4, pady=2)

    def _delete_row(self, image_path: str) -> None:
        self._db.delete_attacks_by_image(image_path)
        self._refresh()
        if self._recognition_tab is not None:
            self._recognition_tab._refresh_thumbnails()
            self._recognition_tab._refresh_best_score()

    # ── Image viewer ──────────────────────────────────────────────────────────

    def _open_viewer(self, path: str, title: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return
        popup = tk.Toplevel(self)
        popup.title(f"{title}  —  {Path(path).name}")
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
            result.append({"rank": rank, "name": identity,
                            "score": best_avg,
                            "adv_path": adv_path,
                            "ctrl_path": ctrl_path})
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
            self._refresh()
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
            self._refresh()
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
            adv_img   = self._load_img(entry.get("adv_path",  ""), img_size)
            ctrl_img  = self._load_img(entry.get("ctrl_path", ""), img_size)
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
