"""
vindr_pretrain.py — Supervised YOLOv8n pretrain on VinDr-CXR (Kaggle 2xT4).

WHY: produce a domain-matched backbone (vindr_best.pt) to initialise the
TBX11K detector via the harness MODEL= flag (zero harness change). The output
we care about is the *weights*, NOT VinDr mAP — see the notes below.

This is SELF-CONTAINED on purpose: it does not import yolo_common (that package
isn't available on Kaggle). It mirrors two locked decisions from exp1-4:
  - architecture = yolov8n  (MUST match downstream or weights won't load)
  - augmentation = our "mosaic" finalist  (exact values from yolo_common/aug.py)

RESOLUTION-PORTABLE: IMGSZ is an env knob and the run name carries it, so the
SAME script makes both backbones — IMGSZ=512 (default, for the 512 TBX11K runs)
and IMGSZ=1024 (for the Kaggle 1024@16 final, exp7/8). A square->square resize
preserves NORMALISED coords, so the converter needs no change between the two;
just attach the matching-resolution VinDr dataset and lower BATCH at 1024 (4x the
pixels). See kaggle/README.md for the 1024 dataset slug + batch guidance.

READ BEFORE JUDGING THE RESULT
------------------------------
VinDr is a brutal *detection* benchmark (14 classes, 3 disagreeing radiologists,
~70% "No finding"). Kaggle competition WINNERS scored ~0.31 mAP@0.4 with heavy
full-res ensembles. A single yolov8n @ 512 will land ~0.10-0.20 mAP50 and that
is NORMAL — it is NOT a weak backbone. Do not tune it up.
  - Sanity bar (must clear): loss decreases and the easy common classes
    (Cardiomegaly, Aortic enlargement, Pleural effusion) get non-zero AP.
  - Failure signal: near-zero AP EVERYWHERE + flat loss => a data bug (almost
    always garbage boxes). The overlay gate below exists to catch that early.
  - The real judge is exp6 (VinDr-init vs COCO-init on TBX11K Active mAP50),
    not anything printed here.

DATASET CONTRACT (the "512x512 square" choice)
----------------------------------------------
Attach a VinDr dataset (recommended: awsaf49/vinbigdata-512-image-dataset) where:
  - images are 512x512 squares, named <image_id>.png/.jpg
  - a CSV with columns: image_id, class_id, x_min, y_min, x_max, y_max
    ("No finding" rows have class_id 14 and NaN coords).
We normalise boxes to YOLO format, preferring the CSV's `width`/`height`
columns (the awsaf49 case: original coords + original dims) and falling back to
each image's real pixel size otherwise. A square->square resize preserves
normalised coords, so both paths agree — NO original-DICOM lookup, NO rescale
guesswork. The overlay gate halts if the coords don't actually fit the image.

Run order in a Kaggle notebook (GPU T4 x2, Internet ON for the first run so
Ultralytics can fetch yolov8n.pt): one cell ->  %run vindr_pretrain.py
Or paste the file into a cell. Knobs are env vars (see CONFIG).
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# ── CONFIG (all env-overridable; plain defaults) ──────────────────────────────
IMGSZ    = int(os.environ.get("IMGSZ", "512"))
EPOCHS   = int(os.environ.get("EPOCHS", "80"))      # early-stop usually ends sooner
BATCH    = int(os.environ.get("BATCH", "64"))       # 2xT4 @ 512, yolov8n; /GPU count
PATIENCE = int(os.environ.get("PATIENCE", "20"))    # convergence guard, not a hard cap
DEVICE   = os.environ.get("DEVICE", "0,1")          # 2xT4 DDP; set "0" if DDP errors
MODEL    = os.environ.get("MODEL", "yolov8n.pt")    # MUST stay yolov8n for transfer
AUG      = os.environ.get("AUG", "mosaic")          # mosaic | mosaic_mixup
WORKERS  = int(os.environ.get("WORKERS", "4"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.08"))  # VinDr val for best.pt / early-stop
SEED     = int(os.environ.get("SEED", "0"))
SMOKE    = os.environ.get("SMOKE", "0") == "1"      # tiny subset, 2 epochs, plumbing only

INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
WORK_ROOT  = Path(os.environ.get("WORK_ROOT", "/kaggle/working"))
PROJECT    = WORK_ROOT / "vindr_runs"
DATA_DIR   = WORK_ROOT / "vindr_yolo"               # materialised YOLO tree

# VinDr finding classes 0..13. Class 14 "No finding" is NOT a class — those
# images become negatives (empty label => background). Order is the canonical
# VinBigData class_id order; do not reorder.
VINDR_CLASSES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]
NO_FINDING_ID = 14

# Our augmentation levels, copied EXACTLY from yolo_common/aug.py.
# hsv_* = 0 (X-ray intensity is diagnostic), geometry only + mosaic; close_mosaic
# turns mosaic off for the last epochs so training ends on real images.
# NOTE: aug barely moves a *pretrain* backbone (the downstream fine-tune aug is
# what's controlled). AUG=mosaic_mixup is offered because mixup looks promising
# downstream; the 512 backbone used plain mosaic, a negligible difference here.
AUG_MOSAIC = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=10.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0,
    erasing=0.0, close_mosaic=10,
)
AUG_MIXUP = {**AUG_MOSAIC, "mixup": 0.1}            # the "mosaic_mixup" level
AUG_KWARGS = {"mosaic": AUG_MOSAIC, "mosaic_mixup": AUG_MIXUP}[AUG]

IMG_EXTS = (".png", ".jpg", ".jpeg")


# ── Input discovery (don't hardcode a dataset slug) ───────────────────────────
def _find_boxes_csv() -> Path:
    """The attached CSV that holds per-box rows (image_id + x_min..y_max)."""
    need = {"image_id", "class_id", "x_min", "y_min", "x_max", "y_max"}
    cands = []
    for p in INPUT_ROOT.rglob("*.csv"):
        try:
            cols = set(pd.read_csv(p, nrows=1).columns)
        except Exception:
            continue
        if need.issubset(cols):
            cands.append(p)
    if not cands:
        sys.exit(
            "HALT: no boxes CSV with columns "
            f"{sorted(need)} found under {INPUT_ROOT}. Attach a VinDr 512 "
            "dataset whose train CSV carries boxes in 512px space."
        )
    # Prefer a 'train' csv if several match.
    cands.sort(key=lambda p: (("train" not in p.name.lower()), len(str(p))))
    return cands[0]


def _index_images() -> dict[str, Path]:
    """image_id -> path for every image under INPUT_ROOT (first hit wins)."""
    idx: dict[str, Path] = {}
    for p in INPUT_ROOT.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.stem not in idx:
            idx[p.stem] = p
    if not idx:
        sys.exit(f"HALT: no images found under {INPUT_ROOT}.")
    return idx


# ── Convert: CSV (512px boxes) -> normalised YOLO labels ──────────────────────
def build_labels(df: pd.DataFrame, img_index: dict[str, Path]) -> dict[str, list[str]]:
    """
    Returns image_id -> list of YOLO label lines. Images present in img_index but
    with no finding box get an empty list (kept as a background negative).
    Normalisation uses each image's REAL pixel size (read from the file), so the
    CSV's coordinate space just has to match the image it came from.
    """
    findings = df[df["class_id"] != NO_FINDING_ID].copy()
    findings = findings.dropna(subset=["x_min", "y_min", "x_max", "y_max"])

    # Normalisation reference. Two valid dataset shapes (both end up identical
    # because a square->square resize preserves NORMALISED coords):
    #   (a) CSV carries `width`/`height` columns = the ORIGINAL dims, coords in
    #       original space (awsaf49 datasets). Normalise by those columns.
    #   (b) No such columns => coords assumed already in the image's own pixel
    #       space. Normalise by the image's real pixel size.
    use_csv_dims = {"width", "height"}.issubset(findings.columns)
    print(f"[convert] normalising by {'CSV width/height columns' if use_csv_dims else 'image pixel size'}")

    # Keep ONLY images that appear in the train CSV. "No finding" images are in
    # the CSV (class_id 14) and stay as real negatives; images absent from the
    # CSV are the unlabeled TEST split — they must NOT become fake negatives.
    labeled_ids = set(df["image_id"].unique()) & set(img_index)
    n_drop = len(img_index) - len(labeled_ids)
    if n_drop:
        print(f"[convert] dropping {n_drop} images absent from train CSV "
              f"(unlabeled test split — NOT negatives)")

    sizes: dict[str, tuple[int, int]] = {}  # image pixel-size cache (case b)
    labels: dict[str, list[str]] = {iid: [] for iid in labeled_ids}

    skipped = 0
    for row in findings.itertuples(index=False):
        iid = row.image_id
        path = img_index.get(iid)
        if path is None:
            continue  # box for an image we don't have (e.g. test split)
        if use_csv_dims and not (pd.isna(row.width) or pd.isna(row.height)):
            w, h = float(row.width), float(row.height)
        else:
            if iid not in sizes:
                with Image.open(path) as im:
                    sizes[iid] = im.size  # (w, h)
            w, h = sizes[iid]
        x0, y0, x1, y1 = float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max)
        if x1 <= x0 or y1 <= y0:
            skipped += 1
            continue
        xc = ((x0 + x1) / 2) / w
        yc = ((y0 + y1) / 2) / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        # Clamp tiny float spill; hard-fail if a box is wildly out of frame
        # (that would mean the CSV is NOT in this image's coordinate space).
        if not (-0.01 <= xc <= 1.01 and -0.01 <= yc <= 1.01 and 0 < bw <= 1.02 and 0 < bh <= 1.02):
            sys.exit(
                f"HALT: box out of [0,1] for {iid}: xc={xc:.3f} yc={yc:.3f} "
                f"bw={bw:.3f} bh={bh:.3f}. The CSV coords do NOT match the image "
                f"size {w}x{h} — wrong dataset pairing (coordinate landmine)."
            )
        xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
        bw, bh = min(bw, 1.0), min(bh, 1.0)
        labels[iid].append(f"{int(row.class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if skipped:
        print(f"[convert] skipped {skipped} degenerate boxes (x1<=x0 or y1<=y0)")
    n_pos = sum(1 for v in labels.values() if v)
    print(f"[convert] {len(labels)} images | {n_pos} with boxes | "
          f"{len(labels) - n_pos} negatives (No finding)")
    return labels


# ── Overlay gate: the whole ballgame. Prove boxes land on anatomy ─────────────
def overlay_gate(labels: dict[str, list[str]], img_index: dict[str, Path], n: int = 8):
    from PIL import ImageDraw
    out = WORK_ROOT / "overlay_check"
    out.mkdir(exist_ok=True)
    with_boxes = [iid for iid, v in labels.items() if v]
    if not with_boxes:
        sys.exit("HALT: no image has any box — conversion produced nothing.")
    random.Random(SEED).shuffle(with_boxes)
    for iid in with_boxes[:n]:
        path = img_index[iid]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for line in labels[iid]:
            c, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            d.rectangle([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], outline=(255, 0, 0), width=3)
        im.save(out / f"{iid}.png")
    print(f"[overlay] wrote {min(n, len(with_boxes))} overlays to {out} — "
          f"EYEBALL THESE before trusting the run.")


# ── Materialise a YOLO tree (symlinked images + label txt) ────────────────────
def materialise(labels: dict[str, list[str]], img_index: dict[str, Path]) -> Path:
    # Rebuild from scratch — the tree is fully derived, and leaving stale
    # symlinks from a previous (e.g. smoke) run risks train/val leakage.
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    ids = list(labels.keys())
    random.Random(SEED).shuffle(ids)
    if SMOKE:
        ids = ids[:200]
    n_val = max(1, int(len(ids) * VAL_FRAC))
    val_ids, train_ids = set(ids[:n_val]), ids[n_val:]

    for split, split_ids in (("train", train_ids), ("val", val_ids)):
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for iid in split_ids:
            src = img_index[iid]
            link = img_dir / f"{iid}{src.suffix}"
            if not link.exists():
                os.symlink(src, link)
            (lbl_dir / f"{iid}.txt").write_text("\n".join(labels[iid]))

    yaml_path = DATA_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(VINDR_CLASSES))
    yaml_path.write_text(
        f"path: {DATA_DIR}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(VINDR_CLASSES)}\nnames:\n{names}\n"
    )
    print(f"[materialise] train={len(train_ids)} val={len(val_ids)} -> {yaml_path}")
    return yaml_path


# ── Train ─────────────────────────────────────────────────────────────────────
def train(yaml_path: Path):
    from ultralytics import YOLO
    run_name = f"vindr_yolov8n_{AUG}{IMGSZ}"   # carries aug + resolution; never collide
    model = YOLO(MODEL)
    model.train(
        data=str(yaml_path),
        imgsz=IMGSZ,
        epochs=2 if SMOKE else EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        project=str(PROJECT),
        name=run_name,
        exist_ok=True,
        pretrained=True,      # start from COCO yolov8n.pt, then learn CXR
        optimizer="auto",
        **AUG_KWARGS,
    )
    best = PROJECT / run_name / "weights" / "best.pt"
    print(f"\nDONE. Backbone weights: {best}")
    print("Download best.pt and bring it into the repo (see kaggle/README.md).")


def main():
    csv_path = _find_boxes_csv()
    print(f"[input] boxes CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    img_index = _index_images()
    print(f"[input] {len(img_index)} images indexed under {INPUT_ROOT}")

    labels = build_labels(df, img_index)
    overlay_gate(labels, img_index)
    yaml_path = materialise(labels, img_index)
    train(yaml_path)


if __name__ == "__main__":
    main()
