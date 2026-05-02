"""
core/camera_source.py
─────────────────────────────────────────────────────────────────────────────
Camera-device enumeration and a shared "current selection" object that both
tabs subscribe to.

Why a selector object?
──────────────────────
Both tabs (Capture, Recognise) own their own ``Camera`` instance and manage
its lifecycle independently — they open/close the camera as you switch
tabs. A single shared ``Camera`` instance would fight that pattern. So
instead we share just the *selection* (which device index to use) and have
each tab rebuild its ``Camera`` whenever the selection changes.

How indices work
────────────────
On Windows we list devices via DirectShow (``pygrabber``) — that gives us
real device names like "Integrated Camera" or
"Pixel 7 (Phone Link Camera)". The position in this list maps directly to
``cv2.VideoCapture(index, cv2.CAP_DSHOW)``. On other platforms or when
``pygrabber`` is missing we fall back to probing indices 0..7 and showing
generic names like "Camera 0".

Persistence
───────────
The current selection is saved to a small JSON file *by device name* (not
by index, because indices can shift between sessions when devices are
plugged or unplugged). On load, we look up the saved name in the current
device list — if not present (e.g. phone disconnected since last run),
we fall back to index 0.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ── Device record ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraDevice:
    """One enumerated camera device."""
    index: int       # what to pass to cv2.VideoCapture(...)
    name:  str       # human-readable label for the dropdown


# ── Enumeration ──────────────────────────────────────────────────────────────

def _enumerate_windows_dshow() -> list[CameraDevice]:
    """Enumerate DirectShow video-input devices on Windows.

    Returns an empty list (and prints a warning) if ``pygrabber`` isn't
    installed — the caller will then fall back to probing.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("[WARN] pygrabber not installed — falling back to generic "
              "camera names. Install with: pip install pygrabber")
        return []
    try:
        names = FilterGraph().get_input_devices()
    except Exception as exc:                                  # pragma: no cover
        print(f"[WARN] DirectShow enumeration failed: {exc}")
        return []
    return [CameraDevice(index=i, name=name) for i, name in enumerate(names)]


def _enumerate_by_probing(max_index: int = 7) -> list[CameraDevice]:
    """Cross-platform fallback: try opening each index up to ``max_index``
    and list those that respond. Slow on Windows (~0.5–1 s per probed index)
    so we only use this when DirectShow enumeration is unavailable."""
    try:
        import cv2
    except ImportError:                                       # pragma: no cover
        return []
    devices: list[CameraDevice] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            devices.append(CameraDevice(index=i, name=f"Camera {i}"))
            cap.release()
    return devices


def enumerate_devices() -> list[CameraDevice]:
    """Return the list of currently available camera devices, best-effort.

    Always returns at least one entry (a synthetic ``Camera 0`` placeholder)
    so the UI can populate its dropdown even when nothing is detected — the
    user might have plugged a camera in *after* the app started, and the
    "Refresh" button can re-enumerate later.
    """
    if platform.system() == "Windows":
        devices = _enumerate_windows_dshow()
        if devices:
            return devices
        # Fall through to probing if DirectShow path failed.
    devices = _enumerate_by_probing()
    if devices:
        return devices
    return [CameraDevice(index=0, name="Camera 0 (default)")]


# ── Capture-backend hint ─────────────────────────────────────────────────────

def preferred_backend():
    """Return the cv2 capture-backend constant we want to use, or 0 (auto)
    when no preference applies. On Windows, DirectShow is more reliable
    than Media Foundation for virtual cameras like Phone Link."""
    try:
        import cv2
    except ImportError:                                       # pragma: no cover
        return 0
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return 0   # auto


# ── Selector ─────────────────────────────────────────────────────────────────

class CameraSource:
    """Shared camera-selection state. Both tabs hold a reference to the same
    instance and subscribe to changes; when the user picks a different
    device in either tab, both tabs receive the notification and rebuild
    their own ``Camera`` accordingly.

    Threading: all methods are intended to be called from the Tk main
    thread. Callbacks are invoked synchronously inside ``set_index``.
    """

    def __init__(self, devices: list[CameraDevice] | None = None) -> None:
        self._devices: list[CameraDevice] = devices or enumerate_devices()
        self._index: int = 0
        self._listeners: list[Callable[[CameraDevice], None]] = []

    # ── Querying ─────────────────────────────────────────────────────────────

    @property
    def devices(self) -> list[CameraDevice]:
        return list(self._devices)

    @property
    def names(self) -> list[str]:
        return [d.name for d in self._devices]

    @property
    def current(self) -> CameraDevice:
        # Defensive: clamp in case the device list shrank under us.
        idx = max(0, min(self._index, len(self._devices) - 1))
        return self._devices[idx] if self._devices else CameraDevice(0, "Camera 0")

    @property
    def current_index(self) -> int:
        return self.current.index

    @property
    def current_name(self) -> str:
        return self.current.name

    # ── Mutation ─────────────────────────────────────────────────────────────

    def set_by_name(self, name: str) -> bool:
        """Select the device with the given display name. Returns True if
        the name was found and the selection changed; False otherwise."""
        for i, d in enumerate(self._devices):
            if d.name == name:
                if i != self._index:
                    self._index = i
                    self._notify()
                return True
        return False

    def re_enumerate(self) -> None:
        """Refresh the device list. Tries to preserve the current selection
        by name; if the previously-selected device is gone, falls back to
        index 0."""
        prev_name = self.current_name if self._devices else None
        self._devices = enumerate_devices()
        new_idx = 0
        if prev_name is not None:
            for i, d in enumerate(self._devices):
                if d.name == prev_name:
                    new_idx = i
                    break
        self._index = new_idx
        self._notify()

    # ── Subscription ─────────────────────────────────────────────────────────

    def subscribe(self, callback: Callable[[CameraDevice], None]) -> None:
        """Register a listener for selection changes. The callback receives
        the new ``CameraDevice``. Safe to call multiple times with the same
        callback (we deduplicate)."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[CameraDevice], None]) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify(self) -> None:
        current = self.current
        for cb in list(self._listeners):
            try:
                cb(current)
            except Exception as exc:                          # pragma: no cover
                # A misbehaving listener must not block the others.
                print(f"[WARN] CameraSource listener raised: {exc}",
                      file=sys.stderr)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_to(self, path: str | Path) -> None:
        """Write the current selection (by name) to disk. Best-effort: any
        I/O error is logged and swallowed so a config-write failure can
        never crash the app."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"last_device_name": self.current_name}),
                         encoding="utf-8")
        except OSError as exc:
            print(f"[WARN] Could not save camera selection to {path}: {exc}")

    def load_from(self, path: str | Path) -> None:
        """Restore a previously-saved selection from disk by looking up the
        saved name in the current device list. Missing file or unknown
        name → silently keep the default (index 0)."""
        try:
            p = Path(path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            saved = data.get("last_device_name")
            if isinstance(saved, str):
                self.set_by_name(saved)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Could not load camera selection from {path}: {exc}")
