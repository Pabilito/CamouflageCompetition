"""
ui/capture_tab.py
─────────────────────────────────────────────────────────────────────────────
Control Pictures tab: register users and capture square-cropped face images.

Changes vs previous version
────────────────────────────
  • Live preview shows a square crop overlay (the region that gets saved)
  • Captured images are center-cropped to a square before saving
  • Camera restart bug fixed: uses a generation counter so old threads
    self-terminate cleanly before the new one starts writing to the UI
"""

import os
import threading
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw

from core.camera import Camera
from core.database import Database
from core.embeddings import EmbeddingEngine

_PREVIEW_SIZE = 420   # square preview canvas size (pixels)
_THUMB_SIZE   = 72
_CROP_COLOUR  = "#00d4ff"   # overlay border colour


def _center_crop_square(img: Image.Image) -> Image.Image:
    """Return a square PIL image cropped from the centre."""
    w, h = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


class CaptureTab(ctk.CTkFrame):
    def __init__(self, parent, engine: EmbeddingEngine, db: Database, data_dir: str):
        super().__init__(parent, fg_color="transparent")
        self._engine        = engine
        self._db            = db
        self._data_dir      = data_dir
        self._camera        = Camera(index=0)
        self._running       = False
        self._thread_gen    = 0          # incremented on each start; lets old threads die
        self._current_user_id: int | None = None
        self._latest_frame  = None
        self._photo_ref     = None

        self._build_ui()
        self._refresh_user_list()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1, minsize=260)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────────────
        left = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Registered Users",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        self._user_var = ctk.StringVar(value="")
        self._user_menu = ctk.CTkOptionMenu(
            left, variable=self._user_var,
            values=[], width=210,
            command=self._on_user_selected,
        )
        self._user_menu.grid(row=1, column=0, padx=14, pady=4, sticky="ew")

        # New user row
        new_row = ctk.CTkFrame(left, fg_color="transparent")
        new_row.grid(row=2, column=0, padx=14, pady=4, sticky="ew")
        new_row.columnconfigure(0, weight=1)

        self._new_name_var = ctk.StringVar()
        ctk.CTkEntry(new_row, textvariable=self._new_name_var,
                     placeholder_text="New user name…",
                     width=140).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(new_row, text="Add", width=56,
                      command=self._add_user).grid(row=0, column=1)

        self._count_label = ctk.CTkLabel(
            left, text="No user selected",
            font=ctk.CTkFont(size=11), text_color="#aaa")
        self._count_label.grid(row=3, column=0, padx=14, pady=(8, 4), sticky="w")

        ctk.CTkFrame(left, height=1, fg_color="#333").grid(
            row=4, column=0, sticky="ew", padx=14, pady=6)

        ctk.CTkLabel(left, text="Saved Images",
                     font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=5, column=0, padx=14, sticky="w")

        self._thumb_frame = ctk.CTkScrollableFrame(
            left, height=340, fg_color="transparent", label_text="")
        self._thumb_frame.grid(row=6, column=0, padx=10, pady=(4, 10), sticky="nsew")
        left.rowconfigure(6, weight=1)

        ctk.CTkButton(
            left, text="Delete User", fg_color="#8b0000",
            hover_color="#c00000", height=28,
            command=self._delete_user,
        ).grid(row=7, column=0, padx=14, pady=(0, 12), sticky="ew")

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Camera Preview",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 2), sticky="w")

        ctk.CTkLabel(right,
                     text="Blue square = saved crop region",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=1, column=0, padx=14, sticky="w")

        # Canvas for the live preview + crop overlay
        self._canvas = tk.Canvas(
            right,
            width=_PREVIEW_SIZE, height=_PREVIEW_SIZE,
            bg="#0d0d1a", highlightthickness=0,
        )
        self._canvas.grid(row=2, column=0, padx=14, pady=(4, 4))
        self._canvas_img_id  = None
        self._canvas_rect_id = None

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=3, column=0, pady=8)

        self._cam_btn = ctk.CTkButton(
            btn_row, text="▶  Start Camera", width=160,
            command=self._toggle_camera)
        self._cam_btn.pack(side="left", padx=8)

        self._capture_btn = ctk.CTkButton(
            btn_row, text="📷  Capture", width=140,
            fg_color="#005f73", hover_color="#0a9396",
            state="disabled", command=self._capture_frame)
        self._capture_btn.pack(side="left", padx=8)

        self._status = ctk.CTkLabel(
            right, text="Start the camera to begin.",
            font=ctk.CTkFont(size=11), text_color="#888")
        self._status.grid(row=4, column=0, pady=(0, 12))

    # ── User management ───────────────────────────────────────────────────────

    def _refresh_user_list(self) -> None:
        users = self._db.list_users()
        names = [u.name for u in users]
        self._user_menu.configure(values=names if names else ["(none)"])
        if names:
            self._user_var.set(names[0])
            self._on_user_selected(names[0])
        else:
            self._user_var.set("")
            self._count_label.configure(text="No users yet.")

    def _on_user_selected(self, name: str) -> None:
        row = self._db.get_user_by_name(name)
        if row is None:
            return
        self._current_user_id = row.id
        count = self._db.capture_count(row.id)
        self._count_label.configure(text=f"{count} image(s) saved for '{name}'")
        self._refresh_thumbnails(row.id)

    def _add_user(self) -> None:
        name = self._new_name_var.get().strip()
        if not name:
            return
        try:
            self._db.add_user(name)
        except Exception:
            pass
        self._new_name_var.set("")
        self._refresh_user_list()
        self._user_var.set(name)
        self._on_user_selected(name)

    def _delete_user(self) -> None:
        if self._current_user_id is None:
            return
        self._db.delete_user(self._current_user_id)
        self._current_user_id = None
        self._refresh_user_list()
        self._clear_thumbnails()

    # ── Thumbnails ────────────────────────────────────────────────────────────

    def _refresh_thumbnails(self, user_id: int) -> None:
        self._clear_thumbnails()
        captures = self._db.list_captures(user_id)
        self._thumb_refs = []
        for i, cap in enumerate(captures):
            try:
                img = Image.open(cap.image_path).convert("RGB")
                img.thumbnail((_THUMB_SIZE, _THUMB_SIZE))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                       size=(_THUMB_SIZE, _THUMB_SIZE))
                lbl = ctk.CTkLabel(self._thumb_frame, image=ctk_img, text="")
                lbl.grid(row=i // 3, column=i % 3, padx=3, pady=3)
                self._thumb_refs.append(ctk_img)
            except Exception:
                pass

    def _clear_thumbnails(self) -> None:
        for w in self._thumb_frame.winfo_children():
            w.destroy()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._running:
            # Stop: increment generation so the running thread self-exits
            self._running = False
            self._thread_gen += 1
            self._camera.close()
            self._cam_btn.configure(text="▶  Start Camera")
            self._capture_btn.configure(state="disabled")
            self._canvas.delete("all")
            self._status.configure(text="Camera stopped.", text_color="#888")
        else:
            self._running = True
            self._thread_gen += 1
            my_gen = self._thread_gen
            self._camera.open()
            self._cam_btn.configure(text="⏹  Stop Camera")
            self._capture_btn.configure(state="normal")
            self._status.configure(text="Camera running.", text_color="#888")
            threading.Thread(
                target=self._camera_loop, args=(my_gen,), daemon=True
            ).start()

    def _camera_loop(self, gen: int) -> None:
        """Runs in a background thread. Exits immediately if gen is stale."""
        while self._running and gen == self._thread_gen:
            frame = self._camera.read_frame()
            if frame is None:
                continue
            self._latest_frame = frame

            # Build preview: full frame scaled to fit canvas, then crop overlay
            img = Image.fromarray(frame)
            w, h = img.size
            side = min(w, h)

            # Scale full frame to fit _PREVIEW_SIZE, keeping aspect ratio
            scale = _PREVIEW_SIZE / max(w, h)
            disp_w = int(w * scale)
            disp_h = int(h * scale)
            img_disp = img.resize((disp_w, disp_h), Image.BILINEAR)

            # Compute crop rect in display coords
            crop_side = int(side * scale)
            cx = (disp_w - crop_side) // 2
            cy = (disp_h - crop_side) // 2

            from PIL import ImageTk
            tk_img = ImageTk.PhotoImage(img_disp)

            def _update(tk=tk_img, dw=disp_w, dh=disp_h,
                        cx=cx, cy=cy, cs=crop_side):
                if gen != self._thread_gen:
                    return
                # Centre image on canvas
                ox = (_PREVIEW_SIZE - dw) // 2
                oy = (_PREVIEW_SIZE - dh) // 2
                self._canvas.delete("all")
                self._canvas.create_image(ox, oy, anchor="nw", image=tk)
                # Crop overlay
                self._canvas.create_rectangle(
                    ox + cx, oy + cy,
                    ox + cx + cs, oy + cy + cs,
                    outline=_CROP_COLOUR, width=2,
                )
                # Keep reference alive
                self._photo_ref = tk

            self.after(0, _update)

    # ── Capture ───────────────────────────────────────────────────────────────

    def _capture_frame(self) -> None:
        if self._current_user_id is None:
            self._status.configure(text="⚠  Select a user first.", text_color="#ff6b6b")
            return
        frame = self._latest_frame
        if frame is None:
            return
        user_row = self._db.get_user_by_name(self._user_var.get())
        if user_row is None:
            return

        # Center-crop to square before saving
        img = Image.fromarray(frame)
        img = _center_crop_square(img)

        user_dir  = Path(self._data_dir) / "users" / user_row.name
        save_path = Camera.next_image_path(user_dir)
        img.save(str(save_path), format="JPEG", quality=95)

        self._db.add_capture(user_row.id, str(save_path))
        count = self._db.capture_count(user_row.id)
        self._count_label.configure(text=f"{count} image(s) saved for '{user_row.name}'")
        self._status.configure(text=f"✓  Saved {save_path.name}", text_color="#00ff9f")
        self._refresh_thumbnails(user_row.id)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._running = False
        self._thread_gen += 1
        self._camera.close()
        super().destroy()
