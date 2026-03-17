"""
ui/ranking_tab.py  —  Ranking tab

Default: average similarity across all models, sorted ASC (↓ = lower = better attack).
Dropdown to filter by individual model.
Columns scale with window width.
Each row has a 🗑 button to delete that image's records.
Thumbnails clickable to enlarge.
"""

from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.database import Database, AttackRow
from core.embeddings import EmbeddingEngine

_THUMB      = 56
_ALL_MODELS = "All models (average)"

# Column indices — shared between header and rows
# 0: Rank   1: Attacker   2: Score   3: Attack   4: Closest   5: Timestamp   6: Delete
_COL_RANK      = 0
_COL_ATTACKER  = 1
_COL_SCORE     = 2
_COL_ATTACK    = 3
_COL_CLOSEST   = 4
_COL_TIMESTAMP = 5
_COL_DELETE    = 6


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

        self._build_ui()
        self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        toolbar.columnconfigure(4, weight=1)

        ctk.CTkLabel(toolbar, text="Adversarial Attack Ranking",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(toolbar, text="Model:",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=1, padx=(20, 4))

        model_options = [_ALL_MODELS] + self._engine.model_names
        self._model_var = ctk.StringVar(value=_ALL_MODELS)
        ctk.CTkOptionMenu(
            toolbar, variable=self._model_var,
            values=model_options, width=220,
            command=lambda _: self._refresh(),
        ).grid(row=0, column=2)

        ctk.CTkButton(toolbar, text="⟳  Refresh", width=100,
                      command=self._refresh).grid(row=0, column=3, padx=(10, 0))

        ctk.CTkButton(toolbar, text="🗑  Clear All", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._confirm_clear).grid(row=0, column=4, padx=(8, 0))

        ctk.CTkLabel(toolbar, text="click thumbnails to enlarge  •  🗑 to delete row",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=0, column=5, padx=(16, 0))

        # ── Table header ──────────────────────────────────────────────────────
        self._hdr_frame = ctk.CTkFrame(self, fg_color=("#16213e", "#16213e"), corner_radius=6)
        self._hdr_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 2))
        # Header columns mirror body columns — attacker expands
        self._hdr_frame.columnconfigure(_COL_ATTACKER, weight=1)
        self._build_header(is_average=True)

        # ── Scrollable body ───────────────────────────────────────────────────
        self._body = ctk.CTkScrollableFrame(
            self, fg_color=("#12122a", "#12122a"), corner_radius=6, label_text="")
        self._body.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 4))
        # Make attacker column expand so rows scale with window width
        self._body.columnconfigure(_COL_ATTACKER, weight=1)

        # ── Summary ───────────────────────────────────────────────────────────
        self._summary = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=11), text_color="#888")
        self._summary.grid(row=3, column=0, sticky="w", padx=14, pady=(0, 8))

    def _build_header(self, is_average: bool) -> None:
        for w in self._hdr_frame.winfo_children():
            w.destroy()

        score_label = "Avg Similarity ↓" if is_average else "Similarity ↓"

        _header_label(self._hdr_frame, "Rank",        _COL_RANK,      50,  expand=False)
        _header_label(self._hdr_frame, "Attacker",    _COL_ATTACKER,  None, expand=True)
        _header_label(self._hdr_frame, score_label,   _COL_SCORE,     110, expand=False)
        _header_label(self._hdr_frame, "Attack",      _COL_ATTACK,    66,  expand=False)
        _header_label(self._hdr_frame, "Closest",     _COL_CLOSEST,   66,  expand=False)
        _header_label(self._hdr_frame, "Timestamp",   _COL_TIMESTAMP, 160, expand=False)
        _header_label(self._hdr_frame, "",             _COL_DELETE,    36,  expand=False)

    # ── Tab hook ──────────────────────────────────────────────────────────────

    def on_tab_shown(self) -> None:
        self._refresh()

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        for w in self._body.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        model_filter = self._model_var.get()
        is_average   = (model_filter == _ALL_MODELS)
        self._build_header(is_average=is_average)

        if is_average:
            self._refresh_average()
        else:
            self._refresh_model(model_filter)

    def _refresh_average(self) -> None:
        all_rows = self._db.get_ranking(model_name=None, limit=10_000)
        if not all_rows:
            ctk.CTkLabel(self._body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
                row=0, column=0, columnspan=7, pady=30)
            self._summary.configure(text="0 records.")
            return

        groups: dict[tuple, list] = {}
        for r in all_rows:
            key = (r.image_path, r.attacker_name)
            groups.setdefault(key, []).append(r)

        averaged = []
        for (img_path, attacker), rows in groups.items():
            avg = sum(r.similarity_score for r in rows) / len(rows)
            closest = next((r.matched_image_path for r in rows
                            if r.matched_image_path), "")
            averaged.append({
                "attacker":     attacker,
                "avg_score":    avg,
                "image_path":   img_path,
                "closest_path": closest,
                "created_at":   rows[0].created_at,
            })
        averaged.sort(key=lambda x: x["avg_score"])

        for rank, entry in enumerate(averaged, start=1):
            self._add_avg_row(rank, entry)

        self._summary.configure(
            text=f"{len(averaged)} image(s) — average similarity across all models ASC")

    def _refresh_model(self, model_name: str) -> None:
        rows = self._db.get_ranking(model_name=model_name, limit=200)
        if not rows:
            ctk.CTkLabel(self._body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
                row=0, column=0, columnspan=7, pady=30)
            self._summary.configure(text="0 records.")
            return
        for rank, row in enumerate(rows, start=1):
            self._add_model_row(rank, row)
        self._summary.configure(
            text=f"{len(rows)} record(s) — sorted by similarity ASC")

    # ── Row builders ──────────────────────────────────────────────────────────

    def _add_avg_row(self, rank: int, entry: dict) -> None:
        r  = rank - 1
        sc = entry["avg_score"]
        sc_col = "#00ff9f" if sc < 0.35 else ("#ffd166" if sc < 0.55 else "#ff6b6b")
        self._row_cells(
            r,
            rank=rank,
            attacker=entry["attacker"],
            score=sc,
            sc_col=sc_col,
            img_path=entry["image_path"],
            closest_path=entry["closest_path"],
            ts=entry["created_at"],
        )

    def _add_model_row(self, rank: int, row: AttackRow) -> None:
        r  = rank - 1
        sc = row.similarity_score
        sc_col = "#00ff9f" if sc < 0.35 else ("#ffd166" if sc < 0.55 else "#ff6b6b")
        self._row_cells(
            r,
            rank=rank,
            attacker=row.attacker_name,
            score=sc,
            sc_col=sc_col,
            img_path=row.image_path,
            closest_path=row.matched_image_path,
            ts=row.created_at,
        )

    def _row_cells(self, r: int, rank: int, attacker: str, score: float,
                   sc_col: str, img_path: str, closest_path: str, ts: str) -> None:
        def cell(col, text, color="#ccc", expand=False):
            kw = dict(text=text, text_color=color,
                      font=ctk.CTkFont(size=11), anchor="center")
            lbl = ctk.CTkLabel(self._body, **kw)
            sticky = "ew" if expand else ""
            lbl.grid(row=r, column=col, padx=4, pady=3, sticky=sticky)

        cell(_COL_RANK,      f"#{rank}",                              color="#888")
        cell(_COL_ATTACKER,  attacker,                                color="#e0e0e0", expand=True)
        cell(_COL_SCORE,     f"{score:.4f}",                         color=sc_col)
        cell(_COL_TIMESTAMP, ts[:19].replace("T", " "),              color="#666")
        self._add_thumb(img_path,      r, col=_COL_ATTACK,  label="attack frame")
        self._add_thumb(closest_path,  r, col=_COL_CLOSEST, label="closest match")
        self._add_delete_btn(img_path, r, col=_COL_DELETE)

    # ── Cell helpers ──────────────────────────────────────────────────────────

    def _add_thumb(self, path: str, grid_row: int, col: int, label: str) -> None:
        if not path:
            ctk.CTkLabel(self._body, text="—", text_color="#555",
                         font=ctk.CTkFont(size=11)).grid(
                row=grid_row, column=col, padx=4, pady=2)
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((_THUMB, _THUMB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(_THUMB, _THUMB))
            lbl = ctk.CTkLabel(self._body, image=ctk_img, text="", cursor="hand2")
            lbl.grid(row=grid_row, column=col, padx=4, pady=2)
            lbl.bind("<Button-1>", lambda e, p=path, t=label: self._open_viewer(p, t))
            self._thumb_refs.append(ctk_img)
        except Exception:
            ctk.CTkLabel(self._body, text="?", text_color="#555",
                         font=ctk.CTkFont(size=11)).grid(
                row=grid_row, column=col, padx=4, pady=2)

    def _add_delete_btn(self, image_path: str, grid_row: int, col: int) -> None:
        ctk.CTkButton(
            self._body, text="🗑", width=32, height=28,
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
        img.thumbnail((int(sw * 0.8), int(sh * 0.8)), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(popup, image=tk_img, bg="#0d0d1a")
        lbl.image = tk_img
        lbl.pack(padx=10, pady=10)
        tk.Label(popup, text=path, fg="#555", bg="#0d0d1a",
                 font=("Courier New", 9), wraplength=int(sw * 0.8)).pack(pady=(0, 6))
        popup.bind("<Escape>", lambda e: popup.destroy())
        lbl.bind("<Button-1>", lambda e: popup.destroy())
        popup.update_idletasks()
        popup.geometry(f"+{(sw - popup.winfo_width()) // 2}+{(sh - popup.winfo_height()) // 2}")

    # ── Clear all ─────────────────────────────────────────────────────────────

    def _confirm_clear(self) -> None:
        import tkinter.messagebox as mb
        import shutil

        choice = _ClearDialog(self).result
        if choice is None:
            return

        if choice == "rankings":
            deleted = self._db.clear_attacks()
            self._refresh()
            self._summary.configure(text=f"✓  Deleted {deleted} attack record(s).")
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
            self._summary.configure(
                text=f"✓  Cleared {counts['attacks']} attacks, "
                     f"{counts['captures']} captures, {counts['users']} users.")
            if self._capture_tab is not None:
                self._capture_tab.refresh_after_clear()
            if self._recognition_tab is not None:
                self._recognition_tab.refresh_after_clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header_label(parent, text: str, col: int, width: int | None,
                  expand: bool = False) -> None:
    kw = dict(text=text, font=ctk.CTkFont(size=11, weight="bold"),
              text_color="#00d4ff", anchor="center")
    if width:
        kw["width"] = width
    lbl = ctk.CTkLabel(parent, **kw)
    sticky = "ew" if expand else ""
    lbl.grid(row=0, column=col, padx=6, pady=6, sticky=sticky)


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
