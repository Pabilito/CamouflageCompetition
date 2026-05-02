# RMA Face Camouflage Competition

A desktop application for evaluating adversarial face camouflage attacks against multiple face recognition models. Participants attempt to fool recognition systems while being scored in real time.

## Project Structure

```
project/
├── main.py                        # Entry point
├── load_model.py                  # Model loading
├── README.md
│
├── core/
│   ├── camera.py                  # OpenCV webcam wrapper
│   ├── camera_source.py           # Camera device enumeration & shared selection
│   ├── database.py                # SQLite layer
│   ├── embeddings.py              # Embedding management
│   └── face_aligner.py            # MTCNN face detection + alignment correction + crop
│
├── ui/
│   ├── capture_tab.py             # "Control Pictures" tab
│   ├── recognition_tab.py         # "Adversarial Pictures" tab
│   ├── ranking_tab.py             # "Ranking" tab
│   └── manual_align.py            # Manual face-alignment dialog
│
├── Pretrained_Models/             # Weight files
│   ├── iresnet.py                 # IResNet architecture definition
│   ├── vit_trans.py               # TransFace ViT architecture definition
│   ├── swin_subnets.py            # SwinFace architecture definition
│   ├── <IResNet_R18_...>.pth      # IResNet-18 weights
│   ├── <TransFace_S_...>.pt       # TransFace-S weights
│   └── <SwinFace_...>.pt          # SwinFace weights
│
└── data/                          # Created automatically at runtime
    ├── users/                     # Control pictures per identity
    │   └── <name>/img_000.jpg …
    ├── adversarial/               # Adversarial attack images
    │   └── <name>/img_000.jpg …
    ├── logs/                      # Per-identity CSV score logs
    └── camera.json                # Persisted camera device selection
```

## Models

| Name | Source | Input |
|---|---|---|
| `InceptionResnetCASIA` | `facenet-pytorch` (auto-download) | 160×160 |
| `edgeface_base` | `torch.hub` otroshi/edgeface (auto-download) | 112×112 |
| `IResNet_R18_Glint360K` | `.pth` file in `Pretrained_Models/` | 112×112 |
| `TransFace_S_Glint360K` | `.pt` file in `Pretrained_Models/` | 112×112 |
| `SwinFace` | `.pt` checkpoint in `Pretrained_Models/` | 112×112 |

The two library models download automatically on first run. The three file-based models require weight files placed in `Pretrained_Models/`. The file names must contain the substrings `IResNet_R18`, `TransFace_S`, and `SwinFace` respectively.

## Installation

```bash
conda env create -f environment.yml
conda activate CompetitionEnv
```


## Running the App

```bash
python main.py
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `Pretrained_Models` | Folder containing `.pt`/`.pth` weight files |
| `--db` | `db.sqlite` | SQLite database path |
| `--data-dir` | `data` | Root folder for images and logs |

Example:

```bash
python main.py --model-dir /path/to/weights --db competition.sqlite --data-dir /path/to/data
```


## How It Works

### Startup

On launch the app:
1. Loads all models into memory
2. Loads MTCNN face detector
3. Opens the SQLite database (creates it if missing)
4. Enumerates available camera devices (DirectShow on Windows, probe-based fallback elsewhere)
5. Restores the last-used camera selection from `data/camera.json`
6. Starts the UI and begins pre-loading gallery embeddings in the background

### Camera Device Selection

Both tabs share a `CameraSource` object that manages the active camera device. A dropdown and refresh button appear above the camera controls in each tab, allowing the user to switch between the laptop's built-in webcam and other sources (e.g. "Phone Link" virtual cameras from Android phones). Changing the device in one tab automatically updates the other.

On Windows, devices are enumerated via DirectShow (`pygrabber`), giving real device names. On other platforms the app probes indices 0–7 and shows generic names. The current selection is persisted by device name to `data/camera.json` so it survives restarts even if indices shift.

### Image Processing Pipeline

Every image, whether captured from webcam or uploaded, goes through the same pipeline before being saved or scored:

1. **Detection** — MTCNN locates the face and 5 landmarks (eyes, nose, mouth corners)
2. **Angle correction** — the full frame is rotated around the face centre so eyes are horizontal
3. **Crop** — a square region with 10% padding is cropped around the face
4. **Resize** — cropped to 112×112 (160×160 for InceptionResnet) via LANCZOS

If detection fails, the user can choose to save the raw centre-cropped image, or open the **manual alignment** dialog to place the five landmarks by hand.

#### Manual Alignment

When MTCNN fails or produces poor results, a "Realign" button opens a full-screen dialog where the user places the five MTCNN landmarks (left eye, right eye, nose, left mouth corner, right mouth corner) directly on the original image. The dialog provides:

- Click-to-place and drag-to-refine landmark markers with colour-coded crosshairs
- A live 112×112 preview that updates as landmarks are moved
- A **crop scale slider** to widen or tighten the crop (default mirrors typical MTCNN bbox proportions)
- **Crop offset sliders** (horizontal and vertical) to nudge the crop centre in the upright frame
- A tilted crop-preview polygon drawn on the original image showing exactly which region will be saved

The alignment geometry matches the production pipeline: rotate around the landmark centroid, crop a padded square, resize to 112×112 with LANCZOS.

#### High-Quality Companions

When a control or adversarial image is saved, the full untouched original frame is also written alongside it as a lossless PNG companion (e.g. `img_007_hq.png` next to `img_007.jpg`). These companions can be viewed via the right-click context menu ("View original (high quality)") but are excluded from embedding computation and gallery loading.

### Scoring

Each adversarial image is compared against every individual control picture of the target identity. The score per model is the **highest cosine similarity** found across all control images (worst case for the attacker). Scores are in `[-1, 1]` where higher = more similar = harder to fool.

The **closest control image** displayed in the ranking is the one with the highest average similarity across all models (not per-model closest).

| Model | Source |
|---|---|
| TransFace_S_Glint360K | https://github.com/DanJun6737/TransFace |
| SwinFace | https://github.com/lxq1000/SwinFace |
| edgeface_base | https://github.com/otroshi/edgeface |
| IResNet_R18_Glint360K | https://github.com/deepinsight/insightface/tree/master/model_zoo |
| InceptionResnetCASIA | https://github.com/timesler/facenet-pytorch |

`score ≥ threshold` → recognised (attack failed for this model)
`score < threshold` → fooled (attack succeeded for this model)

The **ranking metric** is the average across all 5 models of the worst-case scores. Lower average = better camouflage.

### Detection Threshold

The Adversarial Pictures tab includes an adjustable **detection threshold** slider (range 0.35–0.70, default 0.63). This controls MTCNN's O-Net confidence gate — lowering it makes face detection more permissive, which can help when adversarial camouflage causes weak detections. The detection confidence of each accepted face is displayed below the camera preview.

### Embedding Cache

Embeddings for control pictures are stored in the `embeddings` table of the SQLite database. On subsequent gallery loads, embeddings are fetched in bulk rather than recomputed, making gallery loading nearly instant after the first run.

When a new control image is accepted in the Capture tab, it is incrementally added to the in-memory gallery and embedded across all models without triggering a full gallery reload.

Stale embeddings (e.g. from `_hq.png` companion files mistakenly embedded in earlier versions) are automatically purged during gallery loading.


## Tab Overview

### Control Pictures

Register identities and capture their reference (control) images. Each accepted image is:
- Face-detected, roll-corrected, and cropped to 112×112
- Saved to `data/users/<name>/img_NNN.jpg` with a lossless HQ companion
- Added to the DB and immediately embedded into the in-memory gallery

Supports both webcam capture and file upload. If face detection fails, the user can save the unprocessed centre crop or open the manual alignment dialog. Right-click any thumbnail to view full size, view the HQ original, or delete.

### Adversarial Pictures

Capture or upload adversarial attack images and score them against a selected identity. After accepting an image:
- Saved to `data/adversarial/<name>/img_NNN.jpg` with a lossless HQ companion
- Scored against all control images of the selected identity across all 5 models
- Results written to `data/logs/<name>.csv` and to the `attacks` DB table
- Ranking tab refreshed automatically

The tab displays:
- **Similarity score bars** per model with LFW thresholds and match/no-match verdicts
- **Recognition accuracy bars** showing each model's 5-benchmark (LFW, CPLFW, CFPFP, CALFW, AgeDB30) average accuracy
- **Detection confidence** readout after each face detection
- **Image Summary** panel showing the image's average score and rank among all images
- **Session Summary** panel showing the global best score, your best score, and your current leaderboard position
- **Thumbnail strip** with score badges below each adversarial image

Right-click thumbnails to view full size, view the HQ original, recompute scores, or delete.

### Ranking

The ranking tab can be **popped out** into its own standalone window via the "Pop out Ranking" button in the header, and docked back at any time.

Two sub-tabs:

- **Current Standings** — one row per identity showing their best (lowest average) image score, with individual per-model similarity scores. The lowest score in each model column is highlighted. Top 3 highlighted in gold/silver/bronze. Paginated (30 rows per page) with go-to-page navigation.

- **All Images** — every scored image, filterable by model or showing the cross-model average. Displays attack and closest-match thumbnails (loaded asynchronously). Individual rows can be deleted. Paginated with go-to-page navigation.

#### Podium View

A "Podium View" button opens a resizable standalone window displaying the top 3 participants with their adversarial and reference images. The podium updates live whenever rankings change, and all elements (images, fonts, labels) scale responsively with the window size.

#### Clear Data

The "Clear All" button opens a dialog with two options:
- **Rankings Only** — deletes all attack records from the database but preserves saved images on disk
- **Delete Everything** — removes all DB records (users, captures, attacks, embeddings) and all image directories from disk (requires confirmation)


## Database Schema

| Table | Purpose |
|---|---|
| `users` | Registered identity names |
| `captures` | Control picture paths linked to users |
| `attacks` | Per-model similarity scores for each adversarial image |
| `embeddings` | Cached face embeddings (image_path × model_name → float32 BLOB) |
