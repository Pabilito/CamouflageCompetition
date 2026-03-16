"""
ui/ranking_tab.py  —  Ranking tab

Default view: average similarity across all models, one row per image.
Sub-tabs: one tab per model showing per-model scores.
Clear All: calls capture_tab.refresh_after_clear() so the first tab syncs.
"""

from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.database import Database, AttackRow
from core.embeddings import EmbeddingEngine

_THUMB      = 56
_ALL_MODELS = "All models (average)"


class RankingTab(ctk.CTkFrame):
    def __init__(self, parent, db: Database, data_dir: str,
                 engine: EmbeddingEngine, capture_tab=None):
        super().__init__(parent, fg_color="transparent")
        self._db          = db
        self._data_dir    = data_dir
        self._engine      = engine
        self._capture_tab = capture_tab   # set by main.py after construction
        self._thumb_refs: list = []

        self._build_ui()
        self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        toolbar.columnconfigure(3, weight=1)

        ctk.CTkLabel(toolbar, text="Adversarial Attack Ranking",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        ctk.CTkButton(toolbar, text="⟳  Refresh", width=100,
                      command=self._refresh).grid(row=0, column=1, padx=(16, 0))

        ctk.CTkButton(toolbar, text="🗑  Clear All", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._confirm_clear).grid(row=0, column=2, padx=(8, 0))

        ctk.CTkLabel(toolbar, text="click thumbnails to enlarge",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=0, column=4, padx=(16, 0))

        # Sub-tab view: Average + one per model
        self._tabs = ctk.CTkTabview(self, corner_radius=6)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))

        tab_names = [_ALL_MODELS] + self._engine.model_names
        for name in tab_names:
            self._tabs.add(name)

        # Each tab gets its own scrollable body + summary label
        self._bodies: dict[str, ctk.CTkScrollableFrame] = {}
        self._summaries: dict[str, ctk.CTkLabel] = {}
        for name in tab_names:
            tab = self._tabs.tab(name)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

            hdr = ctk.CTkFrame(tab, fg_color=("#16213e", "#16213e"), corner_radius=6)
            hdr.grid(row=0, column=0, sticky="ew", pady=(0, 2))
            if name == _ALL_MODELS:
                _header_label(hdr, "Rank",        0, 50)
                _header_label(hdr, "Attacker",    1, 130)
                _header_label(hdr, "vs Identity", 2, 110)
                _header_label(hdr, "Avg Score ↑", 3, 90)
                _header_label(hdr, "Attack ↗",    4, 66)
                _header_label(hdr, "Timestamp",   5, 160)
            else:
                _header_label(hdr, "Rank",        0, 50)
                _header_label(hdr, "Attacker",    1, 130)
                _header_label(hdr, "vs Identity", 2, 110)
                _header_label(hdr, "Score ↑",     3, 80)
                _header_label(hdr, "Attack ↗",    4, 66)
                _header_label(hdr, "Closest ↗",   5, 66)
                _header_label(hdr, "Timestamp",   6, 160)

            body = ctk.CTkScrollableFrame(
                tab, fg_color=("#12122a", "#12122a"), corner_radius=6, label_text="")
            body.grid(row=1, column=0, sticky="nsew")
            tab.rowconfigure(1, weight=1)
            self._bodies[name] = body

            summary = ctk.CTkLabel(tab, text="", font=ctk.CTkFont(size=11), text_color="#888")
            summary.grid(row=2, column=0, sticky="w", pady=(2, 4))
            self._summaries[name] = summary

        self._tabs.configure(command=self._refresh)

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._thumb_refs.clear()
        current_tab = self._tabs.get()
        if current_tab == _ALL_MODELS:
            self._refresh_average()
        else:
            self._refresh_model(current_tab)

    def _refresh_average(self) -> None:
        body    = self._bodies[_ALL_MODELS]
        summary = self._summaries[_ALL_MODELS]
        for w in body.winfo_children():
            w.destroy()

        # Collect all rows, group by (image_path, identity) → avg score
        all_rows = self._db.get_ranking(model_name=None, limit=10_000)
        if not all_rows:
            ctk.CTkLabel(body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(pady=30)
            summary.configure(text="0 records.")
            return

        groups: dict[tuple, list] = {}
        for r in all_rows:
            key = (r.image_path, r.attacker_name, r.identity_name)
            groups.setdefault(key, []).append(r)

        averaged = []
        for (img_path, attacker, identity), rows in groups.items():
            avg = sum(r.similarity_score for r in rows) / len(rows)
            averaged.append({
                "attacker": attacker,
                "identity": identity,
                "avg_score": avg,
                "image_path": img_path,
                "created_at": rows[0].created_at,
                "n_models": len(rows),
            })

        averaged.sort(key=lambda x: x["avg_score"])

        for rank, entry in enumerate(averaged, start=1):
            self._add_avg_row(body, rank, entry)

        summary.configure(
            text=f"{len(averaged)} image(s) — average similarity across all models, ASC  •  click thumbnails to enlarge")

    def _refresh_model(self, model_name: str) -> None:
        body    = self._bodies[model_name]
        summary = self._summaries[model_name]
        for w in body.winfo_children():
            w.destroy()

        rows = self._db.get_ranking(model_name=model_name, limit=100)
        if not rows:
            ctk.CTkLabel(body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(pady=30)
            summary.configure(text="0 records.")
            return

        for rank, row in enumerate(rows, start=1):
            self._add_model_row(body, rank, row)

        summary.configure(
            text=f"{len(rows)} record(s) — sorted by similarity ASC  •  click thumbnails to enlarge")

    # ── Row builders ──────────────────────────────────────────────────────────

    def _add_avg_row(self, body, rank: int, entry: dict) -> None:
        r = rank - 1
        sc = entry["avg_score"]
        score_col = "#00ff9f" if sc < 0.35 else ("#ffd166" if sc < 0.55 else "#ff6b6b")

        def cell(col, text, color="#ccc", width=None, anchor="w"):
            kw = dict(text=text, text_color=color, font=ctk.CTkFont(size=11), anchor=anchor)
            if width:
                kw["width"] = width
            ctk.CTkLabel(body, **kw).grid(row=r, column=col, padx=4, pady=3, sticky="w")

        cell(0, f"#{rank}",                  color="#888",    width=50,  anchor="e")
        cell(1, entry["attacker"],            color="#e0e0e0", width=130)
        cell(2, entry["identity"] or "—",     color="#a8dadc", width=110)
        cell(3, f"{sc:.4f}",                 color=score_col, width=90,  anchor="e")
        cell(5, entry["created_at"][:19].replace("T", " "), color="#666", width=160)
        self._add_thumb(body, entry["image_path"], r, col=4, label="attack frame")

    def _add_model_row(self, body, rank: int, row: AttackRow) -> None:
        r = rank - 1
        sc = row.similarity_score
        score_col = "#00ff9f" if sc < 0.35 else ("#ffd166" if sc < 0.55 else "#ff6b6b")

        def cell(col, text, color="#ccc", width=None, anchor="w"):
            kw = dict(text=text, text_color=color, font=ctk.CTkFont(size=11), anchor=anchor)
            if width:
                kw["width"] = width
            ctk.CTkLabel(body, **kw).grid(row=r, column=col, padx=4, pady=3, sticky="w")

        cell(0, f"#{rank}",                   color="#888",    width=50,  anchor="e")
        cell(1, row.attacker_name,             color="#e0e0e0", width=130)
        cell(2, row.identity_name or "—",      color="#a8dadc", width=110)
        cell(3, f"{sc:.4f}",                  color=score_col, width=80,  anchor="e")
        cell(6, row.created_at[:19].replace("T", " "), color="#666", width=160)
        self._add_thumb(body, row.image_path,         r, col=4, label="attack frame")
        self._add_thumb(body, row.matched_image_path, r, col=5, label="closest match")

    def _add_thumb(self, body, path: str, grid_row: int, col: int, label: str) -> None:
        if not path:
            ctk.CTkLabel(body, text="—", text_color="#555",
                         font=ctk.CTkFont(size=11), width=_THUMB).grid(
                row=grid_row, column=col, padx=4, pady=2)
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((_THUMB, _THUMB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(_THUMB, _THUMB))
            lbl = ctk.CTkLabel(body, image=ctk_img, text="", cursor="hand2")
            lbl.grid(row=grid_row, column=col, padx=4, pady=2)
            lbl.bind("<Button-1>", lambda e, p=path, t=label: self._open_viewer(p, t))
            self._thumb_refs.append(ctk_img)
        except Exception:
            ctk.CTkLabel(body, text="?", text_color="#555",
                         font=ctk.CTkFont(size=11), width=_THUMB).grid(
                row=grid_row, column=col, padx=4, pady=2)

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

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _confirm_clear(self) -> None:
        import tkinter.messagebox as mb
        import shutil

        choice = _ClearDialog(self).result
        if choice is None:
            return

        if choice == "rankings":
            deleted = self._db.clear_attacks()
            self._refresh()
            self._summaries[self._tabs.get()].configure(
                text=f"✓  Deleted {deleted} attack record(s).")

        elif choice == "all":
            ok = mb.askyesno(
                "Delete Everything",
                "Permanently delete:\n"
                "  • All attack, capture and user DB records\n"
                "  • All images (users/, comparisons/, attacks/)\n\n"
                "Cannot be undone. Continue?",
                icon="warning")
            if not ok:
                return
            counts = self._db.clear_all_data()
            deleted_dirs = []
            for folder in ("users", "comparisons", "attacks"):
                p = Path(self._data_dir) / folder
                if p.exists():
                    import shutil as _sh; _sh.rmtree(p)
                    deleted_dirs.append(folder)
            # Refresh this tab
            self._refresh()
            self._summaries[self._tabs.get()].configure(
                text=f"✓  Cleared {counts['attacks']} attacks, "
                     f"{counts['captures']} captures, "
                     f"{counts['users']} users.")
            # Refresh capture tab so user list clears
            if self._capture_tab is not None:
                self._capture_tab.refresh_after_clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header_label(parent, text: str, col: int, width: int) -> None:
    ctk.CTkLabel(parent, text=text, width=width,
                 font=ctk.CTkFont(size=11, weight="bold"),
                 text_color="#00d4ff", anchor="w").grid(
        row=0, column=col, padx=6, pady=6, sticky="w")


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
