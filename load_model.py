"""
load_model.py
─────────────────────────────────────────────────────────────────────────────
Loads the five face-recognition models used by the desktop app and returns
a uniform ModelBundle named tuple:

    ModelBundle(model, name, height, width, mean, std)

Supported models
────────────────
  Library models (no local file required):
    InceptionResnetCASIA   – facenet_pytorch, pretrained on CASIA-WebFace
    edgeface_base          – torch.hub (otroshi/edgeface)

  File-based models (place weights in Pretrained_Models/):
    IResNet_R18_Glint360K  – IResNet-18 (.pth state dict)
    TransFace_S_Glint360K  – TransFace-S ViT (.pt state dict)
    SwinFace               – SwinFace multi-branch (.pt checkpoint)

Usage
─────
  from load_model import load_app_models, APP_MODEL_SPECS, ModelBundle

  # Load all five at once (fine if VRAM allows):
  bundles = load_app_models(model_dir="Pretrained_Models", device=device)

  # Memory-safe: one at a time (recommended for GPU use):
  for spec in APP_MODEL_SPECS:
      bundle = load_model_by_spec(spec, model_dir="Pretrained_Models", device=device)
      # ... use bundle ...
      release_model(bundle)
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

import Pretrained_Models.iresnet as iresnet
import Pretrained_Models.vit_trans as trans
from Pretrained_Models import swin_subnets


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    model:  nn.Module
    name:   str
    height: int
    width:  int
    mean:   list[float]
    std:    list[float]


# ── Shared defaults ───────────────────────────────────────────────────────────

_MEAN = [0.5, 0.5, 0.5]
_STD  = [0.5, 0.5, 0.5]
_HW   = 112  # all five models use 112×112 except InceptionResnet (160×160)


# ── Model spec descriptors ────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Lightweight descriptor – no weights loaded yet."""
    name:      str
    is_library: bool   # True → no local file needed


# Canonical load order (also used by the UI to list available models).
APP_MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("InceptionResnetCASIA",  is_library=True),
    ModelSpec("edgeface_base",         is_library=True),
    ModelSpec("IResNet_R18_Glint360K", is_library=False),
    ModelSpec("TransFace_S_Glint360K", is_library=False),
    ModelSpec("SwinFace",              is_library=False),
]


# ── Individual loaders ────────────────────────────────────────────────────────

def _load_inception_casia(device: torch.device) -> ModelBundle:
    """InceptionResnetV1 pretrained on CASIA-WebFace (facenet_pytorch)."""
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="casia-webface").to(device).eval()
    print("[LOADED] InceptionResnetCASIA")
    # FaceNet expects 160×160 RGB input
    return ModelBundle(model, "InceptionResnetCASIA", 160, 160, _MEAN, _STD)


def _load_edgeface_base(device: torch.device) -> ModelBundle:
    """EdgeFace-base via torch.hub (otroshi/edgeface)."""
    model = torch.hub.load(
        "otroshi/edgeface", "edgeface_base",
        source="github", pretrained=True,
    ).to(device).eval()
    print("[LOADED] edgeface_base")
    return ModelBundle(model, "edgeface_base", _HW, _HW, _MEAN, _STD)


def _load_iresnet_r18(path: str, device: torch.device) -> ModelBundle:
    """IResNet-18 from a .pth state-dict file."""
    model = iresnet.iresnet18()
    state = torch.load(path, map_location=device, weights_only=False)
    # Guard: some checkpoints are already serialised Module objects
    if isinstance(state, nn.Module):
        model = state
    else:
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("[LOADED] IResNet_R18_Glint360K")
    return ModelBundle(model, "IResNet_R18_Glint360K", _HW, _HW, _MEAN, _STD)


def _load_transface_s(path: str, device: torch.device) -> ModelBundle:
    """TransFace-S (embed_dim=512, depth=12, num_heads=8) from a .pt file."""
    model = trans.VisionTransformer(
        img_size=112, patch_size=9, num_classes=512,
        embed_dim=512, depth=12, num_heads=8,
        drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0,
    )
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=False),
        strict=True,
    )
    model.to(device).eval()
    print("[LOADED] TransFace_S_Glint360K")
    return ModelBundle(model, "TransFace_S_Glint360K", _HW, _HW, _MEAN, _STD)


def _load_swinface(path: str, device: torch.device) -> ModelBundle:
    """SwinFace multi-branch model from a .pt checkpoint."""
    cfg   = swin_subnets.SwinFaceCfg()
    model = swin_subnets.build_model(cfg)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.backbone.load_state_dict(ckpt["state_dict_backbone"])
    model.fam.load_state_dict(ckpt["state_dict_fam"])
    model.tss.load_state_dict(ckpt["state_dict_tss"])
    model.om.load_state_dict(ckpt["state_dict_om"])
    model.set_output_type("Recognition")
    model.to(device).eval()
    print("[LOADED] SwinFace")
    return ModelBundle(model, "SwinFace", _HW, _HW, _MEAN, _STD)


# ── File-path resolver ────────────────────────────────────────────────────────

_FILE_PATTERNS: dict[str, tuple[str, ...]] = {
    # canonical name → accepted filename prefixes / substrings
    "IResNet_R18_Glint360K": ("IResNet_R18",),
    "TransFace_S_Glint360K": ("TransFace_S",),
    "SwinFace":              ("SwinFace",),
}

def _find_weight_file(name: str, model_dir: str) -> str:
    """
    Return the first .pt / .pth file in model_dir whose stem contains one
    of the registered substrings for the given model name.
    Raises FileNotFoundError if nothing matches.
    """
    patterns = _FILE_PATTERNS[name]
    for fname in sorted(os.listdir(model_dir)):
        stem = os.path.splitext(fname)[0]
        ext  = os.path.splitext(fname)[1].lower()
        if ext in (".pt", ".pth") and any(p in stem for p in patterns):
            return os.path.join(model_dir, fname)
    raise FileNotFoundError(
        f"No weight file found for '{name}' in '{model_dir}'. "
        f"Expected a .pt/.pth file containing one of: {patterns}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_model_by_spec(
    spec: ModelSpec,
    device: torch.device,
    model_dir: str = "Pretrained_Models",
) -> ModelBundle:
    """
    Load a single model from a ModelSpec.
    For library models model_dir is not used.
    Raises on failure (no silent swallowing — let the caller decide).
    """
    name = spec.name

    if name == "InceptionResnetCASIA":
        return _load_inception_casia(device)

    if name == "edgeface_base":
        return _load_edgeface_base(device)

    path = _find_weight_file(name, model_dir)

    if name == "IResNet_R18_Glint360K":
        return _load_iresnet_r18(path, device)

    if name == "TransFace_S_Glint360K":
        return _load_transface_s(path, device)

    if name == "SwinFace":
        return _load_swinface(path, device)

    raise ValueError(f"Unknown model name: '{name}'")


def load_app_models(
    model_dir: str = "Pretrained_Models",
    device: torch.device | None = None,
    skip_on_error: bool = False,
) -> list[ModelBundle]:
    """
    Load all five app models and return them as a list of ModelBundles.

    Parameters
    ──────────
    model_dir    : folder containing the three .pt/.pth weight files
    device       : torch device; defaults to CUDA if available, else CPU
    skip_on_error: if True, print a warning and continue on failure;
                   if False (default), re-raise immediately

    Returns
    ───────
    list[ModelBundle]  – one entry per successfully loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundles: list[ModelBundle] = []
    for spec in APP_MODEL_SPECS:
        try:
            bundles.append(load_model_by_spec(spec, device, model_dir))
        except Exception as exc:
            if skip_on_error:
                print(f"[WARN] Skipped {spec.name}: {exc}")
            else:
                raise
    return bundles


def release_model(bundle: ModelBundle) -> None:
    """
    Move model off GPU, delete it, and free the CUDA cache.
    Call this immediately after you are done with a bundle.
    """
    bundle.model.cpu()
    del bundle.model
    torch.cuda.empty_cache()
