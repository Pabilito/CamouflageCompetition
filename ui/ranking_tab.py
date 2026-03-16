"""
ui/ranking_tab.py
─────────────────────────────────────────────────────────────────────────────
Adversarial attack leaderboard.  Click any thumbnail to open a full-size
image viewer popup.

Columns: Rank | Attacker | vs Identity | Model | Score | Attack | Closest | Timestamp
"""

from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.database import Database, AttackRow
from core.embeddings import EmbeddingEngine

_THUMB    = 56
_ALL_MODELS = "All models"


class RankingTab(ctk.CTkFrame):
    def __init__(self, parent, db: Database, data_dir: str, engine: EmbeddingEngine):
        super().__init__(parent, fg_color="transparent")
        self._db       = db
        self._data_dir = data_dir
        self._engine   = engine
        self._thumb_refs: list = []

        self._build_ui()
        self._refresh()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        toolbar.columnconfigure(2, weight=1)

        ctk.CTkLabel(toolbar, text="Adversarial Attack Ranking",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(toolbar, text="Filter:",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=1, padx=(20, 6))

        model_options = [_ALL_MODELS] + self._engine.model_names
        self._model_var = ctk.StringVar(value=_ALL_MODELS)
        ctk.CTkOptionMenu(
            toolbar, variable=self._model_var,
            values=model_options, width=200,
            command=lambda _: self._refresh(),
        ).grid(row=0, column=2, sticky="w")

        ctk.CTkButton(
            toolbar, text="⟳  Refresh", width=100,
            command=self._refresh,
        ).grid(row=0, column=3, padx=(10, 0))

        ctk.CTkButton(
            toolbar, text="🗑  Clear All", width=110,
            fg_color="#5a0000", hover_color="#8b0000",
            command=self._confirm_clear,
        ).grid(row=0, column=4, padx=(8, 0))

        ctk.CTkLabel(toolbar, text="click thumbnails to enlarge",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=0, column=5, padx=(16, 0))

        # ── Table header ──────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=("#16213e", "#16213e"), corner_radius=6)
        hdr.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 2))
        _header_label(hdr, "Rank",        0, width=50)
        _header_label(hdr, "Attacker",    1, width=130)
        _header_label(hdr, "vs Identity", 2, width=110)
        _header_label(hdr, "Model",       3, width=180)
        _header_label(hdr, "Score ↑",     4, width=80)
        _header_label(hdr, "Attack ↗",    5, width=66)
        _header_label(hdr, "Closest ↗",   6, width=66)
        _header_label(hdr, "Timestamp",   7, width=160)

        # ── Scrollable table body ─────────────────────────────────────────────
        self._body = ctk.CTkScrollableFrame(
            self, fg_color=("#12122a", "#12122a"),
            corner_radius=6, label_text="")
        self._body.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 12))

        # ── Summary footer ────────────────────────────────────────────────────
        self._summary = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=11), text_color="#888")
        self._summary.grid(row=3, column=0, sticky="w", padx=14, pady=(0, 8))

    # ── Clear records ─────────────────────────────────────────────────────────

    def _confirm_clear(self) -> None:
        """Offer to clear just rankings, or everything including images on disk."""
        import tkinter.messagebox as mb
        import shutil

        choice = _ClearDialog(self).result   # "rankings", "all", or None
        if choice is None:
            return

        if choice == "rankings":
            deleted = self._db.clear_attacks()
            self._refresh()
            self._summary.configure(text=f"✓  Deleted {deleted} attack record(s).")

        elif choice == "all":
            ok = mb.askyesno(
                "Delete Everything",
                "This will permanently delete:\n"
                "  • All attack records (DB)\n"
                "  • All user records (DB)\n"
                "  • All saved images on disk\n"
                "    (users/, comparisons/, attacks/)\n\n"
                "This CANNOT be undone. Continue?",
                icon="warning",
            )
            if not ok:
                return
            counts = self._db.clear_all_data()
            # Delete image folders
            deleted_dirs = []
            for folder in ("users", "comparisons", "attacks"):
                p = Path(self._data_dir) / folder
                if p.exists():
                    shutil.rmtree(p)
                    deleted_dirs.append(folder)
            self._refresh()
            self._summary.configure(
                text=f"✓  Cleared {counts['attacks']} attacks, "
                     f"{counts['captures']} captures, "
                     f"{counts['users']} users. "
                     f"Deleted folders: {', '.join(deleted_dirs) or 'none'}."
            )

    # ── Data refresh ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        for w in self._body.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        model_filter = self._model_var.get()
        rows = self._db.get_ranking(
            model_name=None if model_filter == _ALL_MODELS else model_filter,
            limit=100,
        )

        if not rows:
            ctk.CTkLabel(self._body, text="No attack records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
                row=0, column=0, pady=30)
            self._summary.configure(text="0 records.")
            return

        for rank, row in enumerate(rows, start=1):
            self._add_table_row(rank, row)

        self._summary.configure(
            text=f"{len(rows)} record(s) — sorted by similarity ASC  •  click thumbnails to enlarge")

    def _add_table_row(self, rank: int, row: AttackRow) -> None:
        r = rank - 1

        def cell(col, text, color="#ccc", width=None, anchor="w"):
            kw = dict(text=text, text_color=color,
                      font=ctk.CTkFont(size=11), anchor=anchor)
            if width:
                kw["width"] = width
            lbl = ctk.CTkLabel(self._body, **kw)
            lbl.grid(row=r, column=col, padx=4, pady=3, sticky="w")

        sc = row.similarity_score
        score_col = "#00ff9f" if sc < 0.35 else ("#ffd166" if sc < 0.55 else "#ff6b6b")

        cell(0, f"#{rank}",                 color="#888",    width=50,  anchor="e")
        cell(1, row.attacker_name,           color="#e0e0e0", width=130)
        cell(2, row.identity_name or "—",    color="#a8dadc", width=110)
        cell(3, row.model_name,              color="#a0c4ff", width=180)
        cell(4, f"{sc:.4f}",                color=score_col, width=80,  anchor="e")
        cell(7, row.created_at[:19].replace("T", " "), color="#666", width=160)

        self._add_thumb(row.image_path,         r, col=5, label="attack frame")
        self._add_thumb(row.matched_image_path, r, col=6, label="closest match")

    def _add_thumb(self, path: str, grid_row: int, col: int, label: str) -> None:
        if not path:
            ctk.CTkLabel(self._body, text="—", text_color="#555",
                         font=ctk.CTkFont(size=11), width=_THUMB).grid(
                row=grid_row, column=col, padx=4, pady=2)
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((_THUMB, _THUMB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(_THUMB, _THUMB))
            lbl = ctk.CTkLabel(self._body, image=ctk_img, text="",
                               cursor="hand2")
            lbl.grid(row=grid_row, column=col, padx=4, pady=2)
            lbl.bind("<Button-1>", lambda e, p=path, t=label: self._open_viewer(p, t))
            self._thumb_refs.append(ctk_img)
        except Exception:
            ctk.CTkLabel(self._body, text="?", text_color="#555",
                         font=ctk.CTkFont(size=11), width=_THUMB).grid(
                row=grid_row, column=col, padx=4, pady=2)

    # ── Full-size image viewer popup ──────────────────────────────────────────

    def _open_viewer(self, path: str, title: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            return

        popup = tk.Toplevel(self)
        popup.title(f"{title}  —  {Path(path).name}")
        popup.configure(bg="#0d0d1a")
        popup.resizable(True, True)

        # Fit to 80% of screen
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        max_w, max_h = int(sw * 0.8), int(sh * 0.8)
        img.thumbnail((max_w, max_h), Image.LANCZOS)

        # Use plain ImageTk inside a tk.Label so it's not destroyed by ctk GC
        tk_img = ImageTk.PhotoImage(img)

        container = tk.Frame(popup, bg="#0d0d1a")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        img_lbl = tk.Label(container, image=tk_img, bg="#0d0d1a")
        img_lbl.image = tk_img   # keep reference
        img_lbl.pack()

        # File path label
        tk.Label(
            popup,
            text=path,
            fg="#555", bg="#0d0d1a",
            font=("Courier New", 9),
            wraplength=max_w,
        ).pack(pady=(0, 6))

        # Close on Escape or click
        popup.bind("<Escape>", lambda e: popup.destroy())
        img_lbl.bind("<Button-1>", lambda e: popup.destroy())

        popup.update_idletasks()
        # Centre the popup on screen
        pw = popup.winfo_width()
        ph = popup.winfo_height()
        popup.geometry(f"+{(sw - pw) // 2}+{(sh - ph) // 2}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header_label(parent, text: str, col: int, width: int) -> None:
    ctk.CTkLabel(
        parent, text=text, width=width,
        font=ctk.CTkFont(size=11, weight="bold"),
        text_color="#00d4ff", anchor="w",
    ).grid(row=0, column=col, padx=6, pady=6, sticky="w")


# ── Clear dialog ──────────────────────────────────────────────────────────────

class _ClearDialog(ctk.CTkToplevel):
    """
    Modal dialog with two delete options:
      • Clear Rankings only  (attack DB records)
      • Delete Everything    (DB + all images on disk)
    Sets self.result to "rankings", "all", or None on cancel.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Clear Data")
        self.geometry("380x200")
        self.resizable(False, False)
        self.grab_set()          # modal
        self.result = None

        ctk.CTkLabel(
            self, text="What would you like to delete?",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(pady=(20, 4))

        ctk.CTkLabel(
            self,
            text="'Rankings only' keeps saved images.\n"
                 "'Delete Everything' removes all DB records\nand all images from disk.",
            font=ctk.CTkFont(size=11), text_color="#aaa",
            justify="center",
        ).pack(pady=(0, 16))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack()

        ctk.CTkButton(
            btn_row, text="Rankings Only", width=130,
            fg_color="#5a0000", hover_color="#8b0000",
            command=self._pick_rankings,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            btn_row, text="Delete Everything", width=150,
            fg_color="#3a0000", hover_color="#700000",
            command=self._pick_all,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            self, text="Cancel", width=100, fg_color="transparent",
            border_width=1, command=self.destroy,
        ).pack(pady=(12, 0))

        self.wait_window()

    def _pick_rankings(self):
        self.result = "rankings"
        self.destroy()

    def _pick_all(self):
        self.result = "all"
        self.destroy()
