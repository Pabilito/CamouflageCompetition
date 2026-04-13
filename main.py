"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point for the facial recognition desktop app.

  python main.py [--model-dir PATH] [--db PATH] [--data-dir PATH]

Loads all five models, opens the SQLite database, then launches the
CustomTkinter window with three tabs:
  • Capture   – register users + save face images from webcam
  • Recognise – live similarity scores across all 5 models
  • Ranking   – adversarial-attack leaderboard
"""

import argparse
import sys
import torch
import customtkinter as ctk

from load_model import load_app_models
from core.database import Database
from core.embeddings import EmbeddingEngine
from core.face_aligner import FaceAligner
from ui.capture_tab import CaptureTab
from ui.recognition_tab import RecognitionTab
from ui.ranking_tab import RankingTab


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face Recognition App")
    p.add_argument("--model-dir", default="Pretrained_Models",
                   help="Folder containing .pt / .pth weight files")
    p.add_argument("--db",        default="db.sqlite",
                   help="Path to SQLite database file")
    p.add_argument("--data-dir",  default="data",
                   help="Root folder for user images and attack images")
    return p.parse_args()


# ── App window ────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self, engine: EmbeddingEngine, db: Database,
                 data_dir: str, aligner: FaceAligner) -> None:
        super().__init__()

        self.title("Face Recognition Competition")
        self.geometry("1100x720")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._engine   = engine
        self._db       = db
        self._data_dir = data_dir
        self._aligner  = aligner

        self._build_ui()

    def _build_ui(self) -> None:
        self._ranking_popped = False   # track pop-out state
        self._ranking_window: ctk.CTkToplevel | None = None

        # ── Header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, height=52, corner_radius=0,
                              fg_color=("#1a1a2e", "#1a1a2e"))
        header.pack(fill="x", side="top")
        ctk.CTkLabel(
            header, text=" RMA Face Camouflage Competition",
            font=ctk.CTkFont(family="Courier New", size=16, weight="bold"),
            text_color="#00d4ff",
        ).pack(side="left", padx=16, pady=10)

        self._popout_btn = ctk.CTkButton(
            header, text="⎋  Pop out Ranking", width=160,
            fg_color="#2a2a4a", hover_color="#3a3a6a",
            command=self._toggle_ranking_popout,
        )
        self._popout_btn.pack(side="right", padx=12, pady=10)

        # ── Tab view ──────────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(self, corner_radius=8)
        self._tabs.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        for name in ("Control Pictures", "Adversarial Pictures", "Ranking"):
            self._tabs.add(name)

        self._capture_tab = CaptureTab(
            self._tabs.tab("Control Pictures"),
            engine=self._engine,
            db=self._db,
            data_dir=self._data_dir,
            aligner=self._aligner,
        )
        self._capture_tab.pack(fill="both", expand=True)

        self._recognition_tab = RecognitionTab(
            self._tabs.tab("Adversarial Pictures"),
            engine=self._engine,
            db=self._db,
            data_dir=self._data_dir,
            aligner=self._aligner,
        )
        self._recognition_tab.pack(fill="both", expand=True)

        self._ranking_tab = RankingTab(
            self._tabs.tab("Ranking"),
            db=self._db,
            data_dir=self._data_dir,
            engine=self._engine,
            capture_tab=self._capture_tab,
            recognition_tab=self._recognition_tab,
        )
        self._ranking_tab.pack(fill="both", expand=True)

        # Auto-reload on tab switch; stop camera when leaving a camera tab
        def _on_tab_change():
            tab = self._tabs.get()
            if tab == "Adversarial Pictures":
                self._capture_tab.on_tab_hidden()
                self._recognition_tab.on_tab_shown()
            elif tab == "Ranking":
                self._capture_tab.on_tab_hidden()
                self._recognition_tab.on_tab_hidden()
                self._ranking_tab.on_tab_shown()
            elif tab == "Control Pictures":
                self._recognition_tab.on_tab_hidden()

        self._tabs.configure(command=_on_tab_change)

        # Pre-load gallery on startup so Adversarial Pictures is ready immediately
        self.after(200, self._recognition_tab.on_tab_shown)

        # Incremental gallery update: when a control image is accepted,
        # add it directly to the recognition tab's in-memory gallery
        self._capture_tab._on_image_accepted = \
            self._recognition_tab.add_control_image

        # Let recognition tab refresh the ranking after scoring
        self._recognition_tab._ranking_tab = self._ranking_tab

    # ── Ranking pop-out / dock ────────────────────────────────────────────────

    def _toggle_ranking_popout(self) -> None:
        if self._ranking_popped:
            self._dock_ranking()
        else:
            self._popout_ranking()

    def _popout_ranking(self) -> None:
        """Detach RankingTab from the tabview into its own Toplevel window."""
        # Hide the Ranking tab in the main tabview
        self._tabs.delete("Ranking")

        # Create the standalone window
        win = ctk.CTkToplevel(self)
        win.title("Ranking — RMA Face Camouflage Competition")
        win.geometry("1000x680")
        win.minsize(700, 400)
        win.configure(fg_color=("#0d0d1a", "#0d0d1a"))
        self._ranking_window = win

        # Reparent the RankingTab widget into the new window.
        # Tkinter doesn't support true widget re-parenting, so we rebuild a
        # fresh RankingTab inside the new window and destroy the old one.
        container = ctk.CTkFrame(win, fg_color="transparent")
        container.pack(fill="both", expand=True)
        old = self._ranking_tab
        self._ranking_tab = RankingTab(
            container,
            db=self._db,
            data_dir=self._data_dir,
            engine=self._engine,
            capture_tab=self._capture_tab,
            recognition_tab=self._recognition_tab,
        )
        self._ranking_tab.pack(fill="both", expand=True)
        self._recognition_tab._ranking_tab = self._ranking_tab
        old.destroy()

        self._ranking_popped = True
        self._popout_btn.configure(text="⬅  Dock Ranking")

        # If user closes the window manually, auto-dock
        win.protocol("WM_DELETE_WINDOW", self._dock_ranking)

    def _dock_ranking(self) -> None:
        """Re-attach RankingTab back into the main tabview."""
        # Destroy the popped-out window and its RankingTab
        old = self._ranking_tab
        if self._ranking_window is not None:
            self._ranking_window.protocol("WM_DELETE_WINDOW", lambda: None)
            self._ranking_window.destroy()
            self._ranking_window = None

        # Re-add the Ranking tab to the main tabview
        self._tabs.add("Ranking")

        # Build a fresh RankingTab inside the tabview tab
        self._ranking_tab = RankingTab(
            self._tabs.tab("Ranking"),
            db=self._db,
            data_dir=self._data_dir,
            engine=self._engine,
            capture_tab=self._capture_tab,
            recognition_tab=self._recognition_tab,
        )
        self._ranking_tab.pack(fill="both", expand=True)
        self._recognition_tab._ranking_tab = self._ranking_tab
        old.destroy()

        self._ranking_popped = False
        self._popout_btn.configure(text="⎋  Pop out Ranking")

    def on_close(self) -> None:
        if self._ranking_window is not None:
            self._ranking_window.destroy()
        self.destroy()


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading models — this may take a moment...")
    try:
        bundles = load_app_models(
            model_dir=args.model_dir,
            device=device,
            skip_on_error=False,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load models: {exc}")
        sys.exit(1)

    engine = EmbeddingEngine(bundles, device=device)
    db     = Database(args.db)

    print("[INFO] Loading face aligner (MTCNN)...")
    aligner = FaceAligner(device=device)
    aligner._get_mtcnn()   # force-load now so first capture has no delay
    print("[INFO] Face aligner ready.")

    app = App(engine=engine, db=db, data_dir=args.data_dir, aligner=aligner)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
