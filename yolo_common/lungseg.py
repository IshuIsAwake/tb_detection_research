"""
lungseg.py — Lung-field masks for TBX11K images (the copy-paste prerequisite).

TBX11K ships NO lung masks, so the synthetic-lesion copy-paste (cp_exp) needs a
segmenter to know where the lungs are before it can paste a lesion inside one.
We use a pretrained CXR anatomy segmenter (torchxrayvision's ChestX-Det PSPNet,
whose targets include Left/Right Lung) rather than training our own — gated on a
measured IoU check against the 31 Montgomery ground-truth masks we do have
(see cp_exp/exp1_lungseg_validate.py). If that check fails we fall back to training a
U-Net; until then this is the single source of lung masks for the project.

Inference-only and lazy: `torchxrayvision`/`torch` are imported inside
`load_model()`, so this module imports fine (and the mask/geometry helpers work)
even before `pip install torchxrayvision`.

    from yolo_common import lungseg
    mask = lungseg.segment_lungs("data/TBX11K/imgs/health/h0001.png")  # bool [512,512]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from yolo_common import settings as S

IMG = 512
MONTGOMERY_DIR = S.REPO_ROOT / "data_external" / "montgomery"

_MODEL = None  # cached pretrained segmenter


# ── pretrained segmenter ──────────────────────────────────────────────────────

def load_model(device: str | None = None):
    """Instantiate the pretrained lung segmenter once (cached). Downloads weights
    on first call (needs internet). Raises a clear message if xrv is missing."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "[lungseg] torchxrayvision is not installed. Run:\n"
            "    pip install torchxrayvision"
        ) from e
    dev = device or ("cuda" if (S.DEVICE != "cpu" and torch.cuda.is_available()) else "cpu")
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval().to(dev)
    _MODEL = model
    return model


def _lung_channel_indices(model) -> list[int]:
    """Indices of the Left/Right Lung output channels (robust to target order)."""
    idx = [i for i, t in enumerate(model.targets) if "Lung" in t and "Hilus" not in t]
    if not idx:
        raise SystemExit(f"[lungseg] no Lung channel in model.targets: {model.targets}")
    return idx


def _to_gray512(img) -> np.ndarray:
    """Any input (path / PIL / ndarray, gray or RGB, any size) → float32 [512,512] 0..255."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("L").resize((IMG, IMG)))
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        if arr.shape != (IMG, IMG):
            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((IMG, IMG)))
    return arr.astype(np.float32)


def segment_lungs(img, device: str | None = None, thr: float = 0.5) -> np.ndarray:
    """Predict the lung-field mask for one CXR. Returns bool [512,512] (L∪R)."""
    import torch
    import torchxrayvision as xrv

    model = load_model(device)
    dev = next(model.parameters()).device
    gray = _to_gray512(img)
    x = xrv.datasets.normalize(gray, 255)  # 0..255 → [-1024, 1024]
    t = torch.from_numpy(x)[None, None].float().to(dev)  # [1,1,512,512]
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0]  # [C,512,512]
    lungs = prob[_lung_channel_indices(model)] > thr
    return lungs.any(dim=0).cpu().numpy().astype(bool)


# ── Montgomery ground-truth masks (validation anchor) ─────────────────────────

def montgomery_ids() -> list[str]:
    """Stems that carry BOTH a left and right GT mask in data_external/."""
    l, r = MONTGOMERY_DIR / "leftMask", MONTGOMERY_DIR / "rightMask"
    if not (l.exists() and r.exists()):
        return []
    return sorted({p.stem for p in l.glob("*.png")} & {p.stem for p in r.glob("*.png")})


def gt_montgomery_mask(name: str, size: int = IMG) -> np.ndarray | None:
    """Union of the L+R Montgomery GT masks, resized to `size`, bool. None if absent.
    Matches exp9_overlays.load_lung_mask exactly (resize→threshold>127→union)."""
    l = MONTGOMERY_DIR / "leftMask" / f"{name}.png"
    r = MONTGOMERY_DIR / "rightMask" / f"{name}.png"
    if not (l.exists() and r.exists()):
        return None
    ml = np.array(Image.open(l).convert("L").resize((size, size))) > 127
    mr = np.array(Image.open(r).convert("L").resize((size, size))) > 127
    return ml | mr


# ── mask overlap metrics ──────────────────────────────────────────────────────

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / s if s else 1.0
