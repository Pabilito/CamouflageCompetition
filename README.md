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
│   ├── database.py                # SQLite layer
│   ├── embeddings.py              # Embedding management
│   └── face_aligner.py            # MTCNN face detection + alignment correction + crop
│
├── ui/
│   ├── capture_tab.py             # "Control Pictures" tab
│   ├── recognition_tab.py         # "Adversarial Pictures" tab
│   └── ranking_tab.py             # "Ranking" tab
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
    └── logs/                      # Per-identity CSV score logs
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
4. Starts the UI and begins pre-loading gallery embeddings in the background

### Image Processing Pipeline

Every image, whether captured from webcam or uploaded, goes through the same pipeline before being saved or scored:

1. **Detection** — MTCNN locates the face and 5 landmarks (eyes, nose, mouth corners)
2. **Angle correction** — the full frame is rotated around the face centre so eyes are horizontal
3. **Crop** — a square region with 10% padding is cropped around the face
4. **Resize** — cropped to 112×112 (160×160 for InceptionResnet) via LANCZOS

If detection fails, the user can choose to save the raw centre-cropped image instead.

### Scoring

Each adversarial image is compared against every individual control picture of the target identity. The score per model is the **highest cosine similarity** found across all control images (worst case for the attacker). Scores are in `[-1, 1]` where higher = more similar = harder to fool.

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

### Embedding Cache

Embeddings for control pictures are stored in the `embeddings` table of the SQLite database. On subsequent gallery loads, embeddings are fetched in bulk rather than recomputed, making gallery loading nearly instant after the first run.


## Tab Overview

### Control Pictures

Register identities and capture their reference (control) images. Each accepted image is:
- Face-detected, roll-corrected, and cropped to 112×112
- Saved to `data/users/<name>/img_NNN.jpg`
- Added to the DB and immediately embedded into the in-memory gallery

Supports both webcam capture and file upload. Right-click any thumbnail to view full size or delete.

### Adversarial Pictures

Capture or upload adversarial attack images and score them against a selected identity. After accepting an image:
- Saved to `data/adversarial/<name>/img_NNN.jpg`
- Scored against all control images of the selected identity across all 5 models
- Results written to `data/logs/<name>.csv` and to the `attacks` DB table
- Ranking tab refreshed automatically

Right-click thumbnails to view, recompute scores, or delete. The Session Summary panel shows your best score, overall rank, and leaderboard position.

### Ranking

Two sub-tabs:

**Current Standings** - one row per identity showing their best (lowest average) image score across all 5 models, with individual per-model scores (Model 1–5). Top 3 highlighted in gold/silver/bronze.

**All Images** - every scored image, filterable by model, with attack and closest-match thumbnails. Individual rows can be deleted.


## Database Schema

| Table | Purpose |
|---|---|
| `users` | Registered identity names |
| `captures` | Control picture paths linked to users |
| `attacks` | Per-model similarity scores for each adversarial image |
| `embeddings` | Cached face embeddings (image_path × model_name → float32 BLOB) |

