"""
ui/capture_tab.py  —  Control Pictures tab
"""

import threading
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from core.camera import Camera
from core.database import Database
from core.embeddings import EmbeddingEngine

_PREVIEW_SIZE = 420
_THUMB_SIZE   = 72
_CROP_COLOUR  = "#00d4ff"


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h  = img.size
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
        self._thread_gen    = 0
        self._current_user_id: int | None = None
        self._latest_frame  = None
        self._photo_ref     = None
        # pending review state
        self._pending_frame: Image.Image | None = None

        self._build_ui()
        self._refresh_user_list()

    def refresh_after_clear(self) -> None:
        """Called externally (e.g. after DB wipe) to re-sync the user list."""
        self._refresh_user_list()
        self._clear_thumbnails()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1, minsize=260)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────────────
        left = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Registered Users",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        self._user_var = ctk.StringVar(value="")
        self._user_menu = ctk.CTkOptionMenu(
            left, variable=self._user_var, values=[], width=210,
            command=self._on_user_selected)
        self._user_menu.grid(row=1, column=0, padx=14, pady=4, sticky="ew")

        new_row = ctk.CTkFrame(left, fg_color="transparent")
        new_row.grid(row=2, column=0, padx=14, pady=4, sticky="ew")
        new_row.columnconfigure(0, weight=1)
        self._new_name_var = ctk.StringVar()
        ctk.CTkEntry(new_row, textvariable=self._new_name_var,
                     placeholder_text="New user name…", width=140).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(new_row, text="Add", width=56,
                      command=self._add_user).grid(row=0, column=1)

        self._count_label = ctk.CTkLabel(
            left, text="No user selected", font=ctk.CTkFont(size=11), text_color="#aaa")
        self._count_label.grid(row=3, column=0, padx=14, pady=(8, 4), sticky="w")

        ctk.CTkFrame(left, height=1, fg_color="#333").grid(
            row=4, column=0, sticky="ew", padx=14, pady=6)

        ctk.CTkLabel(left, text="Saved Images",
                     font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=5, column=0, padx=14, sticky="w")

        self._thumb_frame = ctk.CTkScrollableFrame(
            left, height=300, fg_color="transparent", label_text="")
        self._thumb_frame.grid(row=6, column=0, padx=10, pady=(4, 10), sticky="nsew")
        left.rowconfigure(6, weight=1)

        ctk.CTkButton(left, text="Delete User", fg_color="#8b0000",
                      hover_color="#c00000", height=28,
                      command=self._delete_user).grid(
            row=7, column=0, padx=14, pady=(0, 12), sticky="ew")

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Camera Preview",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 2), sticky="w")
        ctk.CTkLabel(right, text="Blue square = saved crop region",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=1, column=0, padx=14, sticky="w")

        self._canvas = tk.Canvas(right, width=_PREVIEW_SIZE, height=_PREVIEW_SIZE,
                                 bg="#0d0d1a", highlightthickness=0)
        self._canvas.grid(row=2, column=0, padx=14, pady=(4, 4))

        # Live controls
        self._live_btns = ctk.CTkFrame(right, fg_color="transparent")
        self._live_btns.grid(row=3, column=0, pady=6)

        self._cam_btn = ctk.CTkButton(
            self._live_btns, text="▶  Start Camera", width=160,
            command=self._toggle_camera)
        self._cam_btn.pack(side="left", padx=8)

        self._capture_btn = ctk.CTkButton(
            self._live_btns, text="📷  Capture", width=140,
            fg_color="#005f73", hover_color="#0a9396",
            state="disabled", command=self._capture_frame)
        self._capture_btn.pack(side="left", padx=8)

        # Review controls (hidden until a frame is captured)
        self._review_frame = ctk.CTkFrame(right, fg_color="transparent")
        self._review_frame.grid(row=4, column=0, pady=4)
        self._review_frame.grid_remove()

        ctk.CTkLabel(self._review_frame,
                     text="Accept this image?",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#ffd166").pack(side="left", padx=10)

        ctk.CTkButton(self._review_frame, text="✓  Accept", width=110,
                      fg_color="#004d00", hover_color="#007700",
                      command=self._accept_frame).pack(side="left", padx=6)
        ctk.CTkButton(self._review_frame, text="✗  Retake", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._retake_frame).pack(side="left", padx=6)

        self._status = ctk.CTkLabel(
            right, text="Start the camera to begin.",
            font=ctk.CTkFont(size=11), text_color="#888")
        self._status.grid(row=5, column=0, pady=(0, 12))

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
            self._clear_thumbnails()

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
                lbl = ctk.CTkLabel(self._thumb_frame, image=ctk_img, text="",
                                   cursor="hand2")
                lbl.grid(row=i // 3, column=i % 3, padx=3, pady=3)
                lbl.bind("<Button-1>",
                         lambda e, p=cap.image_path: self._open_viewer(p))
                lbl.bind("<Button-3>",
                         lambda e, p=cap.image_path, uid=user_id:
                             self._thumb_context_menu(e, p, uid))
                self._thumb_refs.append(ctk_img)
            except Exception:
                pass

    def _thumb_context_menu(self, event, image_path: str, user_id: int) -> None:
        menu = tk.Menu(self, tearoff=0, bg="#1e1e2e", fg="white",
                       activebackground="#333", activeforeground="white",
                       bd=0, font=("Segoe UI", 10))
        menu.add_command(label="🔍  View full size",
                         command=lambda: self._open_viewer(image_path))
        menu.add_separator()
        menu.add_command(label="🗑  Delete this image",
                         command=lambda: self._delete_capture(image_path, user_id))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _delete_capture(self, image_path: str, user_id: int) -> None:
        import os
        self._db.delete_capture_by_path(image_path)
        try:
            os.remove(image_path)
        except OSError:
            pass
        self._refresh_thumbnails(user_id)
        count = self._db.capture_count(user_id)
        self._count_label.configure(
            text=f"{count} image(s) saved for '{self._user_var.get()}'")

    def _open_viewer(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return
        popup = tk.Toplevel(self)
        popup.title(Path(path).name)
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

    def _clear_thumbnails(self) -> None:
        for w in self._thumb_frame.winfo_children():
            w.destroy()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._running:
            # Close capture device FIRST — this unblocks read_frame() immediately.
            # Then invalidate the thread so the loop exits cleanly.
            self._camera.close()
            self._running = False
            self._thread_gen += 1
            self._cam_btn.configure(text="▶  Start Camera")
            self._capture_btn.configure(state="disabled")
            self._canvas.delete("all")
            self._status.configure(text="Camera stopped.", text_color="#888")
        else:
            self._camera.close()   # ensure fully released before re-opening
            self._running = True
            self._thread_gen += 1
            my_gen = self._thread_gen
            self._camera.open()
            self._cam_btn.configure(text="⏹  Stop Camera")
            self._capture_btn.configure(state="normal")
            self._status.configure(text="Camera running.", text_color="#888")
            self._review_frame.grid_remove()
            self._pending_frame = None
            threading.Thread(target=self._camera_loop, args=(my_gen,), daemon=True).start()

    def _camera_loop(self, gen: int) -> None:
        while self._running and gen == self._thread_gen:
            frame = self._camera.read_frame()
            if frame is None:
                continue
            self._latest_frame = frame
            img = Image.fromarray(frame)
            w, h = img.size
            side  = min(w, h)
            scale = _PREVIEW_SIZE / max(w, h)
            disp_w, disp_h = int(w * scale), int(h * scale)
            img_disp  = img.resize((disp_w, disp_h), Image.BILINEAR)
            crop_side = int(side * scale)
            cx = (disp_w - crop_side) // 2
            cy = (disp_h - crop_side) // 2
            tk_img = ImageTk.PhotoImage(img_disp)

            def _update(tk=tk_img, dw=disp_w, dh=disp_h, cx=cx, cy=cy, cs=crop_side):
                if gen != self._thread_gen:
                    return
                ox = (_PREVIEW_SIZE - dw) // 2
                oy = (_PREVIEW_SIZE - dh) // 2
                self._canvas.delete("all")
                self._canvas.create_image(ox, oy, anchor="nw", image=tk)
                self._canvas.create_rectangle(
                    ox + cx, oy + cy, ox + cx + cs, oy + cy + cs,
                    outline=_CROP_COLOUR, width=2)
                self._photo_ref = tk

            self.after(0, _update)

    # ── Capture → Review → Accept / Retake ───────────────────────────────────

    def _capture_frame(self) -> None:
        if self._current_user_id is None:
            self._status.configure(text="⚠  Select a user first.", text_color="#ff6b6b")
            return
        frame = self._latest_frame
        if frame is None:
            return
        # Close camera so the button state is consistent — Stop Camera can't
        # be clicked while reviewing, and restarting works cleanly after.
        self._camera.close()
        self._running = False
        self._thread_gen += 1
        self._cam_btn.configure(text="▶  Start Camera")
        self._capture_btn.configure(state="disabled")
        img = Image.fromarray(frame)
        self._pending_frame = _center_crop_square(img)
        self._show_review_image(self._pending_frame)
        self._review_frame.grid()
        self._status.configure(text="Review the image — accept or retake.", text_color="#ffd166")

    def _show_review_image(self, img: Image.Image) -> None:
        """Render img (square) centred on the canvas."""
        display = img.copy()
        display.thumbnail((_PREVIEW_SIZE, _PREVIEW_SIZE), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(display)
        ox = (_PREVIEW_SIZE - display.width) // 2
        oy = (_PREVIEW_SIZE - display.height) // 2
        self._canvas.delete("all")
        self._canvas.create_image(ox, oy, anchor="nw", image=tk_img)
        self._photo_ref = tk_img   # keep alive

    def _accept_frame(self) -> None:
        if self._pending_frame is None:
            return
        user_row = self._db.get_user_by_name(self._user_var.get())
        if user_row is None:
            return
        user_dir  = Path(self._data_dir) / "users" / user_row.name
        save_path = Camera.next_image_path(user_dir)
        self._pending_frame.save(str(save_path), format="JPEG", quality=95)
        self._db.add_capture(user_row.id, str(save_path))
        count = self._db.capture_count(user_row.id)
        self._count_label.configure(text=f"{count} image(s) saved for '{user_row.name}'")
        self._status.configure(text=f"✓  Saved {save_path.name}", text_color="#00ff9f")
        self._refresh_thumbnails(user_row.id)
        self._pending_frame = None
        self._review_frame.grid_remove()

    def _retake_frame(self) -> None:
        self._pending_frame = None
        self._review_frame.grid_remove()
        self._camera.close()
        self._running = True
        self._thread_gen += 1
        my_gen = self._thread_gen
        self._camera.open()
        self._cam_btn.configure(text="⏹  Stop Camera")
        self._capture_btn.configure(state="normal")
        self._status.configure(text="Camera running.", text_color="#888")
        threading.Thread(target=self._camera_loop, args=(my_gen,), daemon=True).start()


    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._running = False
        self._thread_gen += 1
        self._camera.close()
        super().destroy()
