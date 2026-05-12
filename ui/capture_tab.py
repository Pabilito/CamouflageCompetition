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
from core.face_aligner import FaceAligner
from ui.manual_align import open_manual_aligner

_PREVIEW_SIZE = 420
_THUMB_SIZE   = 72
_CROP_COLOUR  = "#00d4ff"


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h  = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _hq_path(image_path: str | Path) -> Path:
    """Return the companion high-quality PNG path for a given saved image path.

    Example: data/users/alice/img_007.jpg → data/users/alice/img_007_hq.png
    """
    p = Path(image_path)
    return p.with_name(p.stem + "_hq.png")


class CaptureTab(ctk.CTkFrame):
    def __init__(self, parent, engine: EmbeddingEngine, db: Database,
                 data_dir: str, aligner=None, camera_source=None):
        super().__init__(parent, fg_color="transparent")
        self._engine        = engine
        self._db            = db
        self._data_dir      = data_dir
        self._camera_source = camera_source     # may be None for legacy callers
        self._camera        = Camera(
            index=camera_source.current_index if camera_source else 0)
        self._aligner       = aligner or FaceAligner()
        self._running       = False
        self._thread_gen    = 0
        self._current_user_id: int | None = None
        self._latest_frame  = None
        self._photo_ref     = None
        self._pending_frame:  Image.Image | None = None   # aligned or fallback
        self._raw_frame:      Image.Image | None = None   # centre-cropped fallback
        self._original_frame: Image.Image | None = None   # full untouched original (for HQ companion)
        self._pending_source: str = "camera"   # "camera" or "upload" — drives retake behaviour
        self._on_image_accepted = None
        self._review_image: Image.Image | None = None    # image shown in review canvas

        # Live face-detection overlay state.  A background thread runs MTCNN
        # on the latest frame every ~150 ms and stores the results here; the
        # camera-loop draw callback reads them to paint red bounding boxes.
        self._live_boxes: list[tuple[tuple[int,int,int,int], float]] = []
        self._detect_lock = threading.Lock()

        self._build_ui()
        self._refresh_user_list()

        # Subscribe to camera-source changes so picking a device in the
        # other tab also updates this tab's dropdown and live camera.
        if self._camera_source is not None:
            self._camera_source.subscribe(self._on_camera_source_changed)

    def refresh_after_clear(self) -> None:
        """Called externally (e.g. after DB wipe) to re-sync the user list."""
        self._refresh_user_list()
        self._clear_thumbnails()

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

        # Batch-add row: create N identities named i<next>, i<next+1>, …
        batch_row = ctk.CTkFrame(left, fg_color="transparent")
        batch_row.grid(row=3, column=0, padx=14, pady=4, sticky="ew")
        batch_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(batch_row, text="Batch:", font=ctk.CTkFont(size=11),
                     text_color="#888").grid(row=0, column=0, padx=(0, 4), sticky="w")
        self._batch_count_var = ctk.StringVar(value="5")
        ctk.CTkEntry(batch_row, textvariable=self._batch_count_var,
                     width=48, justify="center").grid(row=0, column=1, sticky="w")
        ctk.CTkButton(batch_row, text="Add N Users", width=100,
                      command=self._batch_add_users).grid(
            row=0, column=2, padx=(4, 0))

        self._count_label = ctk.CTkLabel(
            left, text="No user selected", font=ctk.CTkFont(size=11), text_color="#aaa")
        self._count_label.grid(row=4, column=0, padx=14, pady=(8, 4), sticky="w")

        ctk.CTkFrame(left, height=1, fg_color="#333").grid(
            row=5, column=0, sticky="ew", padx=14, pady=6)

        ctk.CTkLabel(left, text="Saved Images",
                     font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=6, column=0, padx=14, sticky="w")

        self._thumb_frame = ctk.CTkScrollableFrame(
            left, height=300, fg_color="transparent", label_text="")
        self._thumb_frame.grid(row=7, column=0, padx=10, pady=(4, 10), sticky="nsew")
        left.rowconfigure(7, weight=1)

        ctk.CTkButton(left, text="Delete User", fg_color="#8b0000",
                      hover_color="#c00000", height=28,
                      command=self._delete_user).grid(
            row=8, column=0, padx=14, pady=(0, 12), sticky="ew")

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e1e2e", "#1e1e2e"))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        right.rowconfigure(2, weight=1)

        ctk.CTkLabel(right, text="Camera Preview",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#00d4ff").grid(
            row=0, column=0, padx=14, pady=(14, 2), sticky="w")
        ctk.CTkLabel(right, text="Blue square = saved crop region",
                     font=ctk.CTkFont(size=10), text_color="#555").grid(
            row=1, column=0, padx=14, sticky="w")

        self._canvas_size = _PREVIEW_SIZE
        self._canvas = tk.Canvas(right, bg="#0d0d1a", highlightthickness=0)
        self._canvas.grid(row=2, column=0, padx=14, pady=(4, 4), sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # ── Camera-source selector (device dropdown + refresh) ───────────────
        # Lets the user switch between the laptop's built-in webcam and any
        # other DirectShow source — notably "Phone Link" virtual cameras
        # exposed by Android phones running the Phone Link app.
        self._build_camera_source_row(right, row=3)

        # Live controls
        self._live_btns = ctk.CTkFrame(right, fg_color="transparent")
        self._live_btns.grid(row=4, column=0, pady=6)

        self._cam_btn = ctk.CTkButton(
            self._live_btns, text="▶  Start Camera", width=160,
            command=self._toggle_camera)
        self._cam_btn.pack(side="left", padx=8)

        self._capture_btn = ctk.CTkButton(
            self._live_btns, text="📷  Capture", width=140,
            fg_color="#005f73", hover_color="#0a9396",
            state="disabled", command=self._capture_frame)
        self._capture_btn.pack(side="left", padx=8)

        ctk.CTkButton(
            self._live_btns, text="📁  Upload Image", width=150,
            fg_color="#2a2a4a", hover_color="#3a3a6a",
            command=self._upload_image).pack(side="left", padx=8)

        # Normal review controls (shown after successful alignment)
        self._review_frame = ctk.CTkFrame(right, fg_color="transparent")
        self._review_frame.grid(row=5, column=0, pady=4)
        self._review_frame.grid_remove()

        ctk.CTkLabel(self._review_frame,
                     text="Accept aligned face?",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#ffd166").pack(side="left", padx=10)
        ctk.CTkButton(self._review_frame, text="✓  Accept", width=110,
                      fg_color="#004d00", hover_color="#007700",
                      command=self._accept_frame).pack(side="left", padx=6)
        ctk.CTkButton(self._review_frame, text="✎  Realign", width=110,
                      fg_color="#7a5c00", hover_color="#a87b00",
                      text_color="#fff8dc",
                      command=self._open_realign).pack(side="left", padx=6)
        ctk.CTkButton(self._review_frame, text="✗  Retake", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._retake_frame).pack(side="left", padx=6)

        # Detection-failed fallback controls
        self._fail_frame = ctk.CTkFrame(right, fg_color="transparent")
        self._fail_frame.grid(row=5, column=0, pady=4)
        self._fail_frame.grid_remove()

        ctk.CTkLabel(self._fail_frame,
                     text="⚠  No face detected",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#ff6b6b").pack(side="left", padx=8)
        ctk.CTkButton(self._fail_frame, text="💾  Save Unprocessed", width=160,
                      fg_color="#5a3a00", hover_color="#8b5a00",
                      command=self._accept_unprocessed).pack(side="left", padx=4)
        ctk.CTkButton(self._fail_frame, text="✎  Realign", width=110,
                      fg_color="#7a5c00", hover_color="#a87b00",
                      text_color="#fff8dc",
                      command=self._open_realign).pack(side="left", padx=4)
        ctk.CTkButton(self._fail_frame, text="✗  Retake", width=110,
                      fg_color="#5a0000", hover_color="#8b0000",
                      command=self._retake_frame).pack(side="left", padx=4)

        self._status = ctk.CTkLabel(
            right, text="Start the camera to begin.",
            font=ctk.CTkFont(size=11), text_color="#888")
        self._status.grid(row=6, column=0, pady=(0, 12))

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

    def _batch_add_users(self) -> None:
        """Create N identities named i<next>, i<next+1>, … where <next> is
        one past the highest existing i-prefixed numeric identity."""
        import re
        try:
            n = int(self._batch_count_var.get())
        except ValueError:
            return
        if n <= 0:
            return

        # Find the highest existing "i<number>" identity
        existing = self._db.list_users()
        max_idx = 0
        for u in existing:
            m = re.fullmatch(r"i(\d+)", u.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        last_name = ""
        for k in range(1, n + 1):
            name = f"i{max_idx + k}"
            try:
                self._db.add_user(name)
                last_name = name
            except Exception:
                pass

        self._refresh_user_list()
        if last_name:
            self._user_var.set(last_name)
            self._on_user_selected(last_name)

    def _delete_user(self) -> None:
        if self._current_user_id is None:
            return
        # Resolve the user's folder before we drop the DB row, so we can
        # also remove their saved control images (and HQ companions) from disk.
        import shutil
        name = self._user_var.get()
        user_dir = Path(self._data_dir) / "users" / name
        self._db.delete_user(self._current_user_id)
        if user_dir.exists():
            try:
                shutil.rmtree(user_dir)
            except OSError as exc:
                print(f"[WARN] Could not remove {user_dir}: {exc}")
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
        hq = _hq_path(image_path)
        hq_state = "normal" if hq.exists() else "disabled"
        menu.add_command(label="🖼  View original (high quality)",
                         state=hq_state,
                         command=lambda: self._open_viewer(str(hq)))
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
        # Also remove the HQ companion if present.
        hq = _hq_path(image_path)
        if hq.exists():
            try:
                hq.unlink()
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
            self._pending_frame  = None
            self._original_frame = None
            self._pending_source = "camera"
            self._review_image   = None
            with self._detect_lock:
                self._live_boxes = []
            threading.Thread(target=self._camera_loop, args=(my_gen,), daemon=True).start()
            threading.Thread(target=self._detection_loop, args=(my_gen,), daemon=True).start()

    def _detection_loop(self, gen: int) -> None:
        """Background thread: run MTCNN on the latest frame every ~150 ms and
        store the detected bounding boxes for the camera-loop draw callback."""
        import time
        while self._running and gen == self._thread_gen:
            frame = self._latest_frame
            if frame is not None:
                pil = Image.fromarray(frame)
                boxes = self._aligner.detect_boxes(pil)
                with self._detect_lock:
                    self._live_boxes = boxes
            time.sleep(0.15)

    def _camera_loop(self, gen: int) -> None:
        while self._running and gen == self._thread_gen:
            frame = self._camera.read_frame()
            if frame is None:
                continue
            self._latest_frame = frame
            img = Image.fromarray(frame)
            w, h = img.size
            side  = min(w, h)
            scale = self._canvas_size / max(w, h)
            disp_w, disp_h = int(w * scale), int(h * scale)
            img_disp  = img.resize((disp_w, disp_h), Image.BILINEAR)
            crop_side = int(side * scale)
            cx = (disp_w - crop_side) // 2
            cy = (disp_h - crop_side) // 2
            tk_img = ImageTk.PhotoImage(img_disp)

            with self._detect_lock:
                boxes_snapshot = list(self._live_boxes)

            def _update(tk=tk_img, dw=disp_w, dh=disp_h, cx=cx, cy=cy,
                        cs=crop_side, sc=scale, boxes=boxes_snapshot):
                if gen != self._thread_gen:
                    return
                ox = (self._canvas_size - dw) // 2
                oy = (self._canvas_size - dh) // 2
                self._canvas.delete("all")
                self._canvas.create_image(ox, oy, anchor="nw", image=tk)
                self._canvas.create_rectangle(
                    ox + cx, oy + cy, ox + cx + cs, oy + cy + cs,
                    outline=_CROP_COLOUR, width=2)
                # Draw live face-detection bounding boxes
                for (bx1, by1, bx2, by2), conf in boxes:
                    dx1 = ox + int(bx1 * sc)
                    dy1 = oy + int(by1 * sc)
                    dx2 = ox + int(bx2 * sc)
                    dy2 = oy + int(by2 * sc)
                    col = "#00ff9f" if conf >= 0.9 else "#ffd166" if conf >= 0.7 else "#ff6b6b"
                    self._canvas.create_rectangle(
                        dx1, dy1, dx2, dy2, outline=col, width=2)
                    self._canvas.create_text(
                        dx1 + 3, dy1 - 2, anchor="sw",
                        text=f"{conf:.2f}", fill=col,
                        font=("Courier New", 9, "bold"))
                self._photo_ref = tk

            self.after(0, _update)

    # ── Upload ────────────────────────────────────────────────────────────────

    def _upload_image(self) -> None:
        from tkinter import filedialog
        if self._current_user_id is None:
            self._status.configure(text="⚠  Select a user first.", text_color="#ff6b6b")
            return
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

    # ── Capture → Detect → Review → Accept / Retake ──────────────────────────

    def _capture_frame(self) -> None:
        if self._current_user_id is None:
            self._status.configure(text="⚠  Select a user first.", text_color="#ff6b6b")
            return
        frame = self._latest_frame
        if frame is None:
            return
        # Stop camera immediately
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

        # Run face detection in background so UI stays responsive
        threading.Thread(target=self._detect_and_review,
                         args=(raw,), daemon=True).start()

    def _detect_and_review(self, raw: Image.Image) -> None:
        result = self._aligner.align(raw)
        if result.success:
            self.after(0, self._show_aligned, result.image)
        else:
            # Show the centre-cropped raw frame with the failure panel
            self.after(0, self._show_detection_failed, result.error)

    def _show_aligned(self, aligned: Image.Image) -> None:
        self._pending_frame = aligned
        self._show_review_image(aligned)
        self._fail_frame.grid_remove()
        self._review_frame.grid()
        self._status.configure(text="Face detected & aligned — accept or retake.",
                               text_color="#00ff9f")

    def _show_detection_failed(self, error: str) -> None:
        self._pending_frame = self._raw_frame   # fallback = centre crop
        self._show_review_image(self._raw_frame)
        self._review_frame.grid_remove()
        self._fail_frame.grid()
        self._status.configure(text=f"⚠  {error}", text_color="#ff6b6b")

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

    def _save_and_finish(self, img: Image.Image) -> None:
        """Shared save logic for both accept paths."""
        user_row = self._db.get_user_by_name(self._user_var.get())
        if user_row is None:
            return
        user_dir  = Path(self._data_dir) / "users" / user_row.name
        save_path = Camera.next_image_path(user_dir)
        img.save(str(save_path), format="JPEG", quality=95)
        # Save the full untouched original as a lossless PNG companion.
        if self._original_frame is not None:
            try:
                self._original_frame.save(str(_hq_path(save_path)), format="PNG")
            except Exception as exc:
                print(f"[WARN] Could not save HQ companion: {exc}")
        cam_name = self._current_camera_name()
        self._db.add_capture(user_row.id, str(save_path), camera_name=cam_name)
        count = self._db.capture_count(user_row.id)
        self._count_label.configure(text=f"{count} image(s) saved for '{user_row.name}'")
        self._status.configure(text=f"✓  Saved {save_path.name}", text_color="#00ff9f")
        self._refresh_thumbnails(user_row.id)
        self._pending_frame  = None
        self._raw_frame      = None
        self._original_frame = None
        self._review_image   = None
        self._review_frame.grid_remove()
        self._fail_frame.grid_remove()
        if self._on_image_accepted is not None:
            self._on_image_accepted(user_row.name, str(save_path))

    def _accept_frame(self) -> None:
        """Accept the aligned (or pending) frame."""
        if self._pending_frame is not None:
            self._save_and_finish(self._pending_frame)

    def _accept_unprocessed(self) -> None:
        """Save the raw centre-cropped frame when detection failed."""
        if self._raw_frame is not None:
            self._save_and_finish(self._raw_frame)

    def _retake_frame(self) -> None:
        was_upload = (self._pending_source == "upload")

        self._pending_frame  = None
        self._raw_frame      = None
        self._original_frame = None
        self._review_image   = None
        self._review_frame.grid_remove()
        self._fail_frame.grid_remove()

        if was_upload:
            self._status.configure(text="Idle. Upload an image or start the camera.",
                                   text_color="#888")
            self._upload_image()
            return

        self._camera.close()
        self._running = True
        self._thread_gen += 1
        my_gen = self._thread_gen
        self._camera.open()
        self._cam_btn.configure(text="⏹  Stop Camera")
        self._capture_btn.configure(state="normal")
        self._status.configure(text="Camera running.", text_color="#888")
        with self._detect_lock:
            self._live_boxes = []
        threading.Thread(target=self._camera_loop, args=(my_gen,), daemon=True).start()
        threading.Thread(target=self._detection_loop, args=(my_gen,), daemon=True).start()

    def _open_realign(self) -> None:
        """Launch the manual alignment dialog on the original full-resolution
        frame. On success the result is dropped straight into the normal
        review/accept flow."""
        if self._original_frame is None:
            self._status.configure(
                text="⚠  No image to realign. Capture or upload first.",
                text_color="#ff6b6b")
            return
        self._status.configure(
            text="Manual alignment — place the 5 landmarks.",
            text_color="#ffd166")
        open_manual_aligner(
            self.winfo_toplevel(), self._original_frame, self._on_manual_done)

    def _on_manual_done(self, aligned: Image.Image | None) -> None:
        if aligned is None:
            # User cancelled — restore prior status hint without changing the
            # pending frame, so they can still Accept the (poor) auto-align
            # or Retake.
            self._status.configure(
                text="Manual alignment cancelled.", text_color="#888")
            return
        # Treat the manual result the same as a successful auto-align: it
        # becomes the pending frame, the review panel is shown, the failure
        # panel (if visible) is hidden.
        self._pending_frame = aligned
        self._show_review_image(aligned)
        self._fail_frame.grid_remove()
        self._review_frame.grid()
        self._status.configure(
            text="Manually aligned — accept or retake.", text_color="#00ff9f")

    # ── Camera-source selector ────────────────────────────────────────────────

    def _current_camera_name(self) -> str:
        """Return the active camera device name, or 'unknown' / 'upload'."""
        if self._pending_source == "upload":
            return "upload"
        if self._camera_source is not None:
            return self._camera_source.current_name
        return "unknown"

    def _build_camera_source_row(self, parent, *, row: int) -> None:
        """Compact row above the live buttons: device dropdown + refresh.
        Disabled if no camera_source was supplied (legacy mode)."""
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=row, column=0, padx=14, pady=(2, 0), sticky="ew")
        bar.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            bar, text="Source:",
            font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=0, column=0, padx=(0, 6), sticky="w")

        if self._camera_source is None:
            # Legacy mode: show an inert "Camera 0" label rather than a
            # dropdown, so the layout still looks right.
            ctk.CTkLabel(bar, text="Camera 0 (default)",
                         font=ctk.CTkFont(size=11),
                         text_color="#aaa").grid(row=0, column=1, sticky="w")
            return

        names = self._camera_source.names
        self._device_var = ctk.StringVar(value=self._camera_source.current_name)
        self._device_menu = ctk.CTkOptionMenu(
            bar, variable=self._device_var, values=names,
            command=self._on_device_picked, width=260)
        self._device_menu.grid(row=0, column=1, padx=(0, 6), sticky="ew")

        ctk.CTkButton(
            bar, text="🔄", width=32,
            fg_color="#2a2a4a", hover_color="#3a3a6a",
            command=self._on_devices_refresh).grid(row=0, column=2, sticky="e")

    def _on_device_picked(self, name: str) -> None:
        """User picked a different camera in the dropdown. Update the shared
        source — that fires the listener (incl. our own) which actually
        rebinds the Camera and restarts the live feed if it was running."""
        if self._camera_source is None:
            return
        self._camera_source.set_by_name(name)

    def _on_devices_refresh(self) -> None:
        """Re-enumerate the device list (e.g. after plugging in Phone Link).
        ``re_enumerate`` notifies listeners with the (possibly unchanged)
        current device, and we use that to refresh both dropdowns."""
        if self._camera_source is None:
            return
        self._camera_source.re_enumerate()
        # set_by_name only fires when the value changes, but we want the
        # dropdown options to refresh too — do that here directly.
        self._device_menu.configure(values=self._camera_source.names)
        self._device_var.set(self._camera_source.current_name)
        self._status.configure(
            text=f"Re-enumerated cameras: "
                 f"{len(self._camera_source.devices)} device(s) found.",
            text_color="#00d4ff")

    def _on_camera_source_changed(self, device) -> None:
        """Listener: the shared selection changed (here or in the other
        tab). Stop the live feed, point our Camera at the new index, and
        restart if it had been running."""
        was_running = self._running
        if was_running:
            self._camera.close()
            self._running = False
            self._thread_gen += 1
            self._cam_btn.configure(text="▶  Start Camera")
            self._capture_btn.configure(state="disabled")
        self._camera.set_index(device.index)
        # Update the dropdown selection in case the change came from elsewhere.
        if hasattr(self, "_device_var"):
            self._device_var.set(device.name)
        if was_running:
            # Resume on the new device.
            self._camera.open()
            self._running = True
            self._thread_gen += 1
            my_gen = self._thread_gen
            self._cam_btn.configure(text="⏹  Stop Camera")
            self._capture_btn.configure(state="normal")
            self._status.configure(
                text=f"Switched to: {device.name}", text_color="#00d4ff")
            with self._detect_lock:
                self._live_boxes = []
            threading.Thread(target=self._camera_loop,
                             args=(my_gen,), daemon=True).start()
            threading.Thread(target=self._detection_loop,
                             args=(my_gen,), daemon=True).start()
        else:
            self._status.configure(
                text=f"Camera source: {device.name}", text_color="#888")


    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._running = False
        self._thread_gen += 1
        self._camera.close()
        if self._camera_source is not None:
            self._camera_source.unsubscribe(self._on_camera_source_changed)
        super().destroy()
