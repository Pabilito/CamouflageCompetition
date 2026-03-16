"""
ui/recognition_tab.py
─────────────────────────────────────────────────────────────────────────────
Recognise tab: live camera feed + per-model similarity scores vs a selected
identity.  Every "Identify & Log" click auto-saves a frame + logs scores to
DB and CSV — no separate manual log button.

Features
────────
  • Canvas progress bar with red ▼ tick at LFW decision threshold per model
  • ✓ match / ✗ no match verdict per model
  • Auto-save frame to data/comparisons/<identity>/ on every Identify click
  • Auto-log all 5 model scores to DB + data/logs/<identity>.csv
  • Closest gallery image stored per model entry
"""

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image
import numpy as np

from core.camera import Camera
from core.database import Database
from core.embeddings import EmbeddingEngine

_PREVIEW_W = 480
_PREVIEW_H = 360
_BAR_H     = 14

THRESHOLDS_LFW: dict[str, float] = {
    "IResNet_R18_Glint360K":    0.2411,
    "IResNet_R50_Glint360K":    0.2393,
    "IResNet_R100_Glint360K":   0.3309,
    "InceptionResnetCASIA":     0.4535,   # name as returned by load_model.py
    "InceptionResnetV1_CASIA":  0.4535,   # legacy alias
    "InceptionResnetV1_VGG":    0.4474,
    "IDiffFace_CF50":           0.2694,
    "SFace_IResNet50":          0.2306,
    "SwinFace":                 0.2174,
    "TransFace_S_Glint360K":    0.1640,
    "edgeface_base":            0.2066,
}


def _score_to_bar(score: float) -> float:
    return max(0.0, min(1.0, (score + 1) / 2))


def _threshold_to_bar(threshold: float) -> float:
    return _score_to_bar(threshold)


def _score_colour(score: float, threshold: float | None) -> str:
    if threshold is None:
        return "#00ff9f" if score >= 0.5 else "#ffd166" if score >= 0.3 else "#ff6b6b"
    diff = score - threshold
    if diff >= 0.05:
        return "#00ff9f"
    if diff >= 0:
        return "#ffd166"
    return "#ff6b6b"


class RecognitionTab(ctk.CTkFrame):
    def __init__(self, parent, engine: EmbeddingEngine, db: Database, data_dir: str):
        super().__init__(parent, fg_color="transparent")
        self._engine   = engine
        self._db       = db
        self._data_dir = data_dir
        self._camera   = Camera(index=0)
        self._running  = False
        self._thread_gen = 0
        self._latest_frame = None
        self._galleries: dict = {}
        self._gallery_paths: dict = {}
        self._photo_ref = None
        self._score_widgets: dict = {}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # ── Left: camera ──────────────────────────────────────────────────────
        left = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Live Camera",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        self._preview = ctk.CTkLabel(left, text="", width=_PREVIEW_W, height=_PREVIEW_H)
        self._preview.grid(row=1, column=0, padx=14, pady=4)

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.grid(row=2, column=0, pady=8)

        self._cam_btn = ctk.CTkButton(
            btn_row, text="▶  Start Camera", width=160,
            command=self._toggle_camera)
        self._cam_btn.pack(side="left", padx=6)

        self._run_btn = ctk.CTkButton(
            btn_row, text="🔍  Identify & Log", width=150,
            fg_color="#005f73", hover_color="#0a9396",
            state="disabled", command=self._run_and_log)
        self._run_btn.pack(side="left", padx=6)

        self._status = ctk.CTkLabel(
            left, text="Start camera, select identity and load gallery.",
            font=ctk.CTkFont(size=11), text_color="#888")
        self._status.grid(row=3, column=0, pady=(0, 10))

        # ── Right: identity + scores ───────────────────────────────────────────
        right = ctk.CTkScrollableFrame(
            self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"), label_text="")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        id_row = ctk.CTkFrame(right, fg_color="transparent")
        id_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        id_row.columnconfigure(1, weight=1)

        ctk.CTkLabel(id_row, text="Compare against:",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#00d4ff").grid(row=0, column=0, sticky="w")

        self._identity_var = ctk.StringVar(value="—")
        self._identity_menu = ctk.CTkOptionMenu(
            id_row, variable=self._identity_var,
            values=["—"], width=180,
            command=self._on_identity_changed,
        )
        self._identity_menu.grid(row=0, column=1, padx=8, sticky="ew")

        for i, name in enumerate(self._engine.model_names):
            self._build_model_row(right, name, row=i + 1)

        self._save_note = ctk.CTkLabel(
            right, text="", font=ctk.CTkFont(size=10), text_color="#666")
        self._save_note.grid(
            row=len(self._engine.model_names) + 1, column=0,
            padx=14, pady=(6, 10), sticky="w")

    def _build_model_row(self, parent, model_name: str, row: int) -> None:
        threshold = THRESHOLDS_LFW.get(model_name)
        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=("#16213e", "#16213e"))
        frame.grid(row=row, column=0, sticky="ew", padx=14, pady=4)
        frame.columnconfigure(0, weight=1)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        top.columnconfigure(0, weight=1)

        ctk.CTkLabel(top, text=model_name,
                     font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
                     text_color="#e0e0e0", anchor="w").grid(row=0, column=0, sticky="w")

        score_lbl = ctk.CTkLabel(top, text="—",
                                 font=ctk.CTkFont(size=12, weight="bold"),
                                 text_color="#aaa", width=72, anchor="e")
        score_lbl.grid(row=0, column=1, padx=(6, 6))

        verdict_lbl = ctk.CTkLabel(top, text="",
                                   font=ctk.CTkFont(size=10),
                                   text_color="#555", width=90, anchor="w")
        verdict_lbl.grid(row=0, column=2)

        if threshold is not None:
            ctk.CTkLabel(top, text=f"Decision Threshold: {threshold:.4f}",
                         font=ctk.CTkFont(size=10),
                         text_color="#ff4444", anchor="e").grid(row=0, column=3, padx=(8, 0))

        canvas = tk.Canvas(frame, height=_BAR_H + 10, bg="#16213e",
                           highlightthickness=0, bd=0)
        canvas.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        canvas.bind("<Configure>",
                    lambda e, c=canvas, t=threshold: self._draw_bar(c, 0.0, "#333", t))

        self._score_widgets[model_name] = (score_lbl, verdict_lbl, canvas, threshold)

    # ── Tab visibility hook ───────────────────────────────────────────────────

    def on_tab_shown(self) -> None:
        """Called by main.py whenever the Recognise tab becomes active.
        Silently reloads the gallery so it stays in sync with new captures."""
        self._reload_gallery()

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
            tick_x = int(w * _threshold_to_bar(threshold))
            canvas.create_line(tick_x, 0, tick_x, h, fill="#ff2222", width=2)
            canvas.create_polygon(
                tick_x - 4, h + 1,
                tick_x + 4, h + 1,
                tick_x,     h + 7,
                fill="#ff2222", outline="",
            )

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _reload_gallery(self) -> None:
        user_dir = Path(self._data_dir) / "users"
        if not user_dir.exists():
            self._status.configure(text="⚠  No user data directory found.", text_color="#ff6b6b")
            return
        self._status.configure(text="Loading gallery…", text_color="#aaa")

        def _load():
            for model_name in self._engine.model_names:
                self._galleries[model_name] = self._engine.build_gallery(model_name, user_dir)

            self._gallery_paths = {}
            for model_name in self._engine.model_names:
                self._gallery_paths[model_name] = {}
                for user_dir_entry in sorted(Path(user_dir).iterdir()):
                    if not user_dir_entry.is_dir():
                        continue
                    pairs = []
                    for img_path in (sorted(user_dir_entry.glob("*.jpg")) +
                                     sorted(user_dir_entry.glob("*.png"))):
                        try:
                            face_rgb = np.array(Image.open(img_path).convert("RGB"))
                            emb = self._engine.embed(model_name, face_rgb)
                            pairs.append((str(img_path), emb))
                        except Exception:
                            pass
                    if pairs:
                        self._gallery_paths[model_name][user_dir_entry.name] = pairs

            all_users = sorted({u for g in self._galleries.values() for u in g})

            def _update_menu():
                if all_users:
                    self._identity_menu.configure(values=all_users)
                    if self._identity_var.get() not in all_users:
                        self._identity_var.set(all_users[0])
                    self._status.configure(
                        text=f"✓  {len(all_users)} identit{'y' if len(all_users)==1 else 'ies'} loaded.",
                        text_color="#00ff9f")
                else:
                    self._status.configure(text="⚠  No user images found.", text_color="#ff6b6b")

            self.after(0, _update_menu)

        threading.Thread(target=_load, daemon=True).start()

    def _on_identity_changed(self, _name: str) -> None:
        for score_lbl, verdict_lbl, canvas, threshold in self._score_widgets.values():
            score_lbl.configure(text="—", text_color="#aaa")
            verdict_lbl.configure(text="")
            self.after(0, self._draw_bar, canvas, 0.0, "#333", threshold)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._running:
            self._running = False
            self._thread_gen += 1
            self._camera.close()
            self._cam_btn.configure(text="▶  Start Camera")
            self._run_btn.configure(state="disabled")
        else:
            self._running = True
            self._thread_gen += 1
            my_gen = self._thread_gen
            self._camera.open()
            self._cam_btn.configure(text="⏹  Stop Camera")
            self._run_btn.configure(state="normal")
            threading.Thread(
                target=self._camera_loop, args=(my_gen,), daemon=True
            ).start()

    def _camera_loop(self, gen: int) -> None:
        while self._running and gen == self._thread_gen:
            frame = self._camera.read_frame()
            if frame is None:
                continue
            self._latest_frame = frame
            img = Image.fromarray(frame)
            img.thumbnail((_PREVIEW_W, _PREVIEW_H))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(img.width, img.height))
            self._photo_ref = ctk_img
            self.after(0, lambda c=ctk_img, g=gen: (
                self._preview.configure(image=c, text="")
                if g == self._thread_gen else None
            ))

    # ── Identify + auto-log ───────────────────────────────────────────────────

    def _run_and_log(self) -> None:
        frame = self._latest_frame
        if frame is None:
            return
        identity = self._identity_var.get()
        if identity == "—" or not self._galleries:
            self._status.configure(
                text="⚠  Load gallery and select an identity first.", text_color="#ff6b6b")
            return

        self._status.configure(text=f"Comparing against '{identity}'…", text_color="#aaa")
        self._run_btn.configure(state="disabled")

        def _infer():
            comp_dir  = Path(self._data_dir) / "comparisons" / identity
            save_path = Camera.next_image_path(comp_dir)
            Camera.save_frame(frame, save_path)

            results: list[dict] = []
            for model_name in self._engine.model_names:
                embeddings = self._galleries.get(model_name, {}).get(identity)
                if not embeddings:
                    self.after(0, self._update_row, model_name, None)
                    continue

                query    = self._engine.embed(model_name, frame)
                mean_emb = np.mean(embeddings, axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
                score    = self._engine.cosine_similarity(query, mean_emb)
                closest  = self._find_closest_image(model_name, query, identity)
                threshold = THRESHOLDS_LFW.get(model_name)

                self._db.add_attack(
                    attacker_name=identity,
                    model_name=model_name,
                    similarity_score=score,
                    image_path=str(save_path),
                    identity_name=identity,
                    matched_image_path=closest,
                )
                results.append({
                    "model": model_name, "score": score,
                    "threshold": threshold,
                    "match": (score >= threshold) if threshold is not None else None,
                    "closest_image": closest,
                    "frame_path": str(save_path),
                })
                self.after(0, self._update_row, model_name, score)

            self._write_log(identity, save_path, results)
            self.after(0, self._status.configure,
                       {"text": f"✓  Saved & logged → {save_path.name}",
                        "text_color": "#00ff9f"})
            self.after(0, self._run_btn.configure, {"state": "normal"})

        threading.Thread(target=_infer, daemon=True).start()

    def _write_log(self, identity: str, frame_path: Path, results: list[dict]) -> None:
        log_dir  = Path(self._data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{identity}.csv"
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        write_header = not log_file.exists()
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "identity", "frame_path",
                "model", "score", "threshold", "match", "closest_image",
            ])
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

    # ── Row update ────────────────────────────────────────────────────────────

    def _update_row(self, model_name: str, score: float | None) -> None:
        score_lbl, verdict_lbl, canvas, threshold = self._score_widgets[model_name]
        if score is None:
            score_lbl.configure(text="no data", text_color="#555")
            verdict_lbl.configure(text="")
            self._draw_bar(canvas, 0.0, "#333", threshold)
            return
        colour = _score_colour(score, threshold)
        score_lbl.configure(text=f"{score:.4f}", text_color=colour)
        if threshold is not None:
            if score >= threshold:
                verdict_lbl.configure(text="✓ match", text_color="#00ff9f")
            else:
                verdict_lbl.configure(text="✗ no match", text_color="#ff6b6b")
        self._draw_bar(canvas, _score_to_bar(score), colour, threshold)

    # ── Closest gallery image ─────────────────────────────────────────────────

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

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._running = False
        self._thread_gen += 1
        self._camera.close()
        super().destroy()
