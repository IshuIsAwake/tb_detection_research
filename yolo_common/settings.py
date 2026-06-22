"""
settings.py — Single source of truth for paths and knobs.

No logic beyond reading env overrides and a couple of pure helpers. The
read-only original data, the (rebuildable) generated tree, and the per-run
results all resolve from REPO_ROOT so the harness works from any CWD.

Knobs are env-overridable with plain, *deliberately untuned* defaults — this
is a diagnostic baseline harness, not a tuned model. See the exp1 spec.

    EPOCHS=100 BATCH=16 SEED=0 CONF_THRESHOLDS=0.10,0.25,0.50
    DEVICE=0 (first GPU; "cpu" to force CPU)
    TB_SMOKE=1   → 1-2 epochs on a tiny subset, full plumbing, no real train
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Roots ─────────────────────────────────────────────────────────────────────
# yolo_common/settings.py → parents[1] = repo root (tb/)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    """Env-overridable path: `TB_DATA_ROOT=/mnt/disk/TBX11K ...`."""
    v = os.environ.get(var)
    return Path(v).expanduser() if v else default


# Original TBX11K data — READ-ONLY. Never written by this harness.
# Override with TB_DATA_ROOT (e.g. data living on another disk).
DATA_ROOT = _env_path("TB_DATA_ROOT", REPO_ROOT / "data" / "TBX11K")
IMGS_ROOT = DATA_ROOT / "imgs"
TB_IMAGES = IMGS_ROOT / "tb"          # 800 imgs; 799 carry boxes
HEALTH_IMAGES = IMGS_ROOT / "health"  # 3800 negatives
SICK_IMAGES = IMGS_ROOT / "sick"      # 3800 negatives (abnormal but non-TB)

# COCO annotations at 512x512 coords are the SOURCE OF TRUTH for boxes.
# The VOC XML uses ORIGINAL clinical resolution (~2840px) and is NOT usable
# directly — do not convert from XML. See CLAUDE.md.
ANN_JSON = DATA_ROOT / "annotations" / "json" / "TBX11K_trainval_only_tb.json"

# ── Generated, shared, rebuildable (safe to delete) ───────────────────────────
# Override with TB_GEN_ROOT.
GEN_ROOT = _env_path("TB_GEN_ROOT", REPO_ROOT / "yolo_datasets")
LABELS_ALL_DIR = GEN_ROOT / "labels_all"      # canonical YOLO .txt per TB positive
CONVERT_REPORT = GEN_ROOT / "convert_report.json"
SPLITS_JSON = GEN_ROOT / "splits.json"        # the frozen split — built once
DET_TREE = GEN_ROOT / "det"                   # materialised train/val YOLO tree
DATA_YAML = DET_TREE / "data.yaml"

# ── Per-run results (per experiment) ──────────────────────────────────────────
# Override with TB_RESULTS_ROOT.
RESULTS_ROOT = _env_path("TB_RESULTS_ROOT", REPO_ROOT / "yolo_experiments" / "results")

# ── Class map (COCO category_id → YOLO class id) ──────────────────────────────
# COCO: 1=ActiveTuberculosis, 2=ObsoletePulmonaryTuberculosis,
#       3=PulmonaryTuberculosis (does not appear in trainval boxes).
# YOLO: 0=Active, 1=Obsolete.  Verified, not assumed — convert.py re-checks.
CLASS_NAMES = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]
COCO_CAT_TO_YOLO = {1: 0, 2: 1}
NUM_CLASSES = len(CLASS_NAMES)

# ── Split (frozen reference for EVERY future experiment) ──────────────────────
SEED = int(os.environ.get("SEED", "0"))
SPLIT_TRAIN_FRAC = 0.70
SPLIT_VAL_FRAC = 0.15
SPLIT_TEST_FRAC = 0.15            # → test_positives (the blackbox TB slice)
N_BLACKBOX_NEG_PER_GROUP = 120   # 120 sick + 120 healthy in the sealed test set

# ── Training knobs (env-overridable; plain defaults — NOT tuned) ──────────────
# Every knob below is overridable from the terminal, e.g.:
#   MODEL=yolov8s.pt EPOCHS=60 BATCH=8 FREEZE=10 PATIENCE=20 \
#       python yolo_experiments/exp1_yolo_baseline.py --imgsz 512
# CLI flags on the exp script take precedence over these env values.
MODEL = os.environ.get("MODEL", "yolov8n.pt")   # MODEL=yolov8s.pt to bump capacity
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH = int(os.environ.get("BATCH", "16"))
DEVICE = os.environ.get("DEVICE", "0")     # "0" = first GPU; "cpu" to force CPU
WORKERS = int(os.environ.get("WORKERS", "8"))
AUG_LEVEL = os.environ.get("AUG_LEVEL", "off")  # see yolo_common/aug.py
PATIENCE = int(os.environ.get("PATIENCE", "100"))  # early-stop patience (epochs)

# Backbone freeze: number of leading layers to freeze (regulariser for small
# data). None/0 = full fine-tune (the exp1 floor). FREEZE=10 freezes the
# yolov8 backbone; partial freeze adapts the domain while curbing overfit.
_freeze = os.environ.get("FREEZE", "").strip().lower()
FREEZE = None if _freeze in ("", "none", "0", "false") else int(_freeze)

# Base LR. None → Ultralytics auto (optimizer=auto picks it). LR0=0.01 to pin.
_lr0 = os.environ.get("LR0", "").strip()
LR0 = float(_lr0) if _lr0 else None

# Ultralytics' AMP self-check downloads a helper model that 404s in some envs;
# also AMP is a confound for a deterministic floor. Off by design.
AMP = False

# ── Negatives-in-training (exp2) ──────────────────────────────────────────────
# How many background (no-TB) images to add to TRAINING, as a multiple of the
# train-positive count, and what fraction of them are sick (non-TB pathology)
# vs healthy. Drawn fixed-seed from the non-reserved pool (the 240 blackbox
# negatives are always excluded — see splits.py / leakage rule). Both are
# env-overridable so the ratio can be swept:
#   NEG_PER_POS=3 NEG_SICK_FRAC=0.5 python yolo_experiments/exp2_negatives.py --imgsz 512
NEG_PER_POS = float(os.environ.get("NEG_PER_POS", "1.0"))
NEG_SICK_FRAC = float(os.environ.get("NEG_SICK_FRAC", "0.5"))

# ── Evaluation knobs ──────────────────────────────────────────────────────────
CONF_THRESHOLDS = [
    float(x) for x in os.environ.get("CONF_THRESHOLDS", "0.10,0.25,0.50").split(",")
]
# Untuned recall ceiling reported alongside the fixed thresholds — catches a
# model that is actually fine but hidden by a high default conf. (Lesson from
# the Oral_cancer harness; not a tuning knob.)
RECALL_CEILING_CONF = 0.001
LOC_CONF = 0.25          # localisation pass operates on conf >= this
IOU_MATCH = 0.50         # greedy-match acceptance threshold

# GT box area buckets (fraction of 512x512 image area), for by-size recall.
SIZE_BUCKETS = (("small", 0.0, 0.01), ("medium", 0.01, 0.05), ("large", 0.05, 1.01))

# ── Smoke mode ────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("TB_SMOKE", "0") == "1"
SMOKE_EPOCHS = 2
SMOKE_SUBSET = 16        # positives used in a smoke train/val/test

IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def list_images(directory: Path) -> list[Path]:
    """All images directly in ``directory`` (non-recursive), sorted by name."""
    files: list[Path] = []
    for ext in IMG_EXTS:
        files.extend(directory.glob(f"*{ext}"))
    uniq = {p.resolve(): p for p in files}
    return sorted(uniq.values(), key=lambda p: p.name)


def effective_epochs() -> int:
    return SMOKE_EPOCHS if SMOKE else EPOCHS
