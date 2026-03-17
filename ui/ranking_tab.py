"""
ui/ranking_tab.py  —  Ranking tab

Two sub-tabs:
  • Current Standings  — one row per identity, their best (lowest) avg score
  • All Images         — every scored image with model filter dropdown
"""

from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.database import Database, AttackRow
from core.embeddings import EmbeddingEngine

_THUMB      = 56
_ALL_MODELS = "All models (average)"
_THRESHOLD  = 0.2565

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

        self._build_ui()
        self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        toolbar.columnconfigure(3, weight=1)

        ctk.CTkLabel(toolbar, text="Adversarial Attack Ranking",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        ctk.CTkButton(toolbar, text="⟳  Refresh", width=100,
                      command=self._refresh).grid(row=0, column=1, padx=(20, 0))

        ctk.CTkButton(toolbar, text="🗑  Clear All", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._confirm_clear).grid(row=0, column=2, padx=(8, 0))

        ctk.CTkLabel(toolbar,
                     text=f"Avg. similarity threshold: {_THRESHOLD}  •  "
                          "click thumbnails to enlarge  •  🗑 to delete row",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=0, column=3, padx=(16, 0), sticky="w")

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

    # ── Sub-tab switch ────────────────────────────────────────────────────────

    def _on_subtab_change(self) -> None:
        self._refresh()

    # ── Tab hooks ─────────────────────────────────────────────────────────────

    def on_tab_shown(self) -> None:
        self._refresh()

    # ── Refresh dispatcher ────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._thumb_refs.clear()
        current = self._tabs.get()
        if current == "Current Standings":
            self._refresh_standings()
        else:
            self._refresh_all_images()

    # ── Current Standings ─────────────────────────────────────────────────────

    def _refresh_standings(self) -> None:
        body = self._standings_body
        for w in body.winfo_children():
            w.destroy()

        # Reconfigure columns for the wider standings table:
        # 0:Rank 1:Attacker 2:BestAvg 3..7:S1-S5 8:Attack 9:Timestamp
        _N_COLS = 10
        for i in range(_N_COLS):
            body.columnconfigure(i, weight=1)

        leaderboard = self._db.get_user_leaderboard()
        if not leaderboard:
            ctk.CTkLabel(body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
                row=1, column=0, columnspan=_N_COLS, pady=30)
            self._standings_summary.configure(text="0 participants.")
            return

        model_names = self._engine.model_names  # ordered list of 5 models

        # ── Collect all DB rows grouped by image_path ─────────────────────────
        all_rows = self._db.get_ranking(model_name=None, limit=10_000)

        # image_path → {model_name: score}
        img_scores: dict[str, dict[str, float]] = {}
        img_meta:   dict[str, dict] = {}
        for r in all_rows:
            if r.image_path not in img_scores:
                img_scores[r.image_path] = {}
                img_meta[r.image_path]   = {
                    "attacker": r.attacker_name,
                    "created_at": r.created_at}
            img_scores[r.image_path][r.model_name] = r.similarity_score

        # For each image compute avg across available models
        img_avg: dict[str, float] = {}
        for path, scores in img_scores.items():
            if scores:
                img_avg[path] = sum(scores.values()) / len(scores)

        # Per-identity: find the image_path with lowest avg
        identity_best_img: dict[str, str] = {}   # identity → best image_path
        for entry in leaderboard:
            identity = entry["identity_name"]
            # Find all images for this identity
            candidates = [p for p, m in img_meta.items()
                          if m["attacker"] == identity]
            if candidates:
                best = min(candidates, key=lambda p: img_avg.get(p, 1.0))
                identity_best_img[identity] = best

        # ── Find per-model column minimums for "best in column" highlight ──────
        col_min: dict[str, float] = {}
        for entry in leaderboard:
            identity = entry["identity_name"]
            path = identity_best_img.get(identity, "")
            if not path:
                continue
            for mn in model_names:
                s = img_scores.get(path, {}).get(mn)
                if s is not None:
                    if mn not in col_min or s < col_min[mn]:
                        col_min[mn] = s

        # ── Draw header ───────────────────────────────────────────────────────
        HDR_BG   = "#16213e"
        AVG_BG   = "#0d2a1a"   # dark green tint → ranking basis
        BEST_BG  = "#1a1a0a"   # dark gold tint  → best in column

        def hdr(col: int, text: str, bg: str = HDR_BG,
                tooltip: str = "") -> None:
            lbl = ctk.CTkLabel(body, text=text,
                               font=ctk.CTkFont(size=11, weight="bold"),
                               text_color="#00d4ff", fg_color=bg,
                               anchor="center")
            lbl.grid(row=0, column=col, padx=2, pady=(4, 4), sticky="ew")

        hdr(0, "Rank")
        hdr(1, "Attacker")
        hdr(2, "Best Avg ↓", bg=AVG_BG)   # ranking basis — green tint
        for si, mn in enumerate(model_names):
            hdr(3 + si, f"Model {si+1}")
        hdr(8, "Best Attack")
        hdr(9, "Timestamp")

        # ── Draw rows ─────────────────────────────────────────────────────────
        for rank, entry in enumerate(leaderboard, start=1):
            identity = entry["identity_name"]
            best_avg = entry["best_avg_score"]
            medal    = _MEDAL.get(rank)
            txt_col  = medal if medal else "#e0e0e0"

            path     = identity_best_img.get(identity, "")
            scores   = img_scores.get(path, {})
            ts       = img_meta.get(path, {}).get("created_at", "")
            grid_row = rank   # data rows start at 1 (header is row 0)

            def cell(col: int, text: str, txt: str = "#ccc",
                     bg: str = "transparent", r: int = grid_row) -> None:
                ctk.CTkLabel(body, text=text, text_color=txt,
                             font=ctk.CTkFont(size=11), anchor="center",
                             fg_color=bg).grid(
                    row=r, column=col, padx=2, pady=2, sticky="ew")

            cell(0, f"#{rank}", txt=txt_col)
            cell(1, identity,   txt=txt_col)
            # Best Avg — always green-tinted background as ranking basis
            cell(2, f"{best_avg:.4f}", txt=_sc_col(best_avg), bg=AVG_BG)

            # Model 1–5 per-model scores
            for si, mn in enumerate(model_names):
                s = scores.get(mn)
                if s is None:
                    cell(3 + si, "—", txt="#555")
                else:
                    is_best = (col_min.get(mn) is not None and
                               abs(s - col_min[mn]) < 1e-9)
                    bg = BEST_BG if is_best else "transparent"
                    cell(3 + si, f"{s:.4f}", txt=_model_score_col(s, mn), bg=bg)

            self._add_thumb(body, path, grid_row, col=8, label="best attack")
            cell(9, ts[:19].replace("T", " ") if ts else "—", txt="#666")

        total = len(leaderboard)
        self._standings_summary.configure(
            text=f"{total} participant(s) — ranked by best avg score ASC  •  "
                 f"Model 1–{len(model_names)} = per-model similarity  •  "
                 f"highlighted = ranking basis / column best")

    # ── All Images ────────────────────────────────────────────────────────────

    def _refresh_all_images(self) -> None:
        body = self._all_body
        for w in body.winfo_children():
            w.destroy()

        is_average = (self._model_var.get() == _ALL_MODELS)
        self._draw_all_header(body, is_average)

        if is_average:
            self._populate_average(body)
        else:
            self._populate_model(body, self._model_var.get())

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

    def _populate_average(self, body) -> None:
        all_rows = self._db.get_ranking(model_name=None, limit=10_000)
        if not all_rows:
            ctk.CTkLabel(body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
                row=1, column=0, columnspan=7, pady=30)
            self._all_summary.configure(text="0 records.")
            return

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

        for rank, entry in enumerate(averaged, start=1):
            self._all_row_cells(body, rank, rank, entry["attacker"],
                                entry["avg_score"], entry["image_path"],
                                entry["closest_path"], entry["created_at"])

        self._all_summary.configure(
            text=f"{len(averaged)} image(s) — average similarity ASC")

    def _populate_model(self, body, model_name: str) -> None:
        rows = self._db.get_ranking(model_name=model_name, limit=200)
        if not rows:
            ctk.CTkLabel(body, text="No records yet.",
                         font=ctk.CTkFont(size=12), text_color="#555").grid(
                row=1, column=0, columnspan=7, pady=30)
            self._all_summary.configure(text="0 records.")
            return

        for rank, row in enumerate(rows, start=1):
            self._all_row_cells(body, rank, rank, row.attacker_name,
                                row.similarity_score, row.image_path,
                                row.matched_image_path, row.created_at)

        self._all_summary.configure(text=f"{len(rows)} record(s) — similarity ASC")

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
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((_THUMB, _THUMB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(_THUMB, _THUMB))
            lbl = ctk.CTkLabel(body, image=ctk_img, text="", cursor="hand2")
            lbl.grid(row=grid_row, column=col, padx=4, pady=2)
            lbl.bind("<Button-1>", lambda e, p=path, t=label: self._open_viewer(p, t))
            self._thumb_refs.append(ctk_img)
        except Exception:
            ctk.CTkLabel(body, text="?", text_color="#555",
                         font=ctk.CTkFont(size=11)).grid(
                row=grid_row, column=col, padx=4, pady=2)

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
