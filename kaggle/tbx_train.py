"""
tbx_train.py — Self-contained TBX11K detector TRAINING on Kaggle / Colab.

WHY: the local 6 GB card can't train 1024@16 (or yolov8s). This offloads the
*training* to a 2×T4 (Kaggle) or a Colab GPU and emits best.pt. The SEALED-TEST
EVAL STAYS LOCAL — run yolo_common/metrics.py on the downloaded best.pt so every
number is byte-identical to exp1-6 and directly comparable. This script does NOT
evaluate; it only trains.

SELF-CONTAINED on purpose (yolo_common isn't on Kaggle). It mirrors the harness
exactly where it matters:
  - uses the FROZEN splits.json you upload — never regenerates a split.
  - positives-only train/val tree (the locked detector rule; negatives are the
    classifier's job — see exp2).
  - aug = mosaic / mosaic_mixup copied verbatim from yolo_common/aug.py.
  - class map 0=ActiveTuberculosis, 1=ObsoletePulmonaryTuberculosis.
The tree mirrors materialise(): images/{train,val} + labels/{train,val} so
Ultralytics' images->labels path inference resolves.

WHAT TO UPLOAD (one private Kaggle dataset, or Colab Drive folder)
    splits.json                         the frozen split (authority)
    tb/<stem>.png                       the TB positive images (512² is fine —
                                        imgsz upscales for 1024 runs, same as the
                                        local 1024 runs did)
    labels_all/<stem>.txt               the converted YOLO labels per positive
    Only the train_ids + val_ids positives are actually needed; uploading the
    whole tb/ folder + labels_all/ is simplest. The blackbox/test images are NOT
    needed here (eval is local).
    Assemble locally with:
        mkdir -p upload && cp yolo_datasets/splits.json upload/
        cp -r data/TBX11K/imgs/tb upload/tb
        cp -r yolo_datasets/labels_all upload/labels_all
    then upload `upload/` as a Kaggle dataset (or to Drive for Colab).

RUN (Kaggle notebook: GPU T4×2, Internet ON first run for yolov8*.pt)
    !pip -q install ultralytics
    import os
    os.environ.update(MODEL="yolov8n.pt", IMGSZ="1024", BATCH="16",
                      AUG="mosaic", SEED="0")          # one run; vary SEED for multi-seed
    %run tbx_train.py
    # then download tbx_runs/<run_name>/weights/best.pt and eval LOCALLY.

KNOBS (env)  MODEL IMGSZ BATCH SEED AUG EPOCHS PATIENCE DEVICE WORKERS  + SMOKE=1
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# ── CONFIG (env-overridable; defaults = the locked exp4/exp6 recipe) ──────────
MODEL    = os.environ.get("MODEL", "yolov8n.pt")     # yolov8n.pt | yolov8s.pt
IMGSZ    = int(os.environ.get("IMGSZ", "1024"))
BATCH    = int(os.environ.get("BATCH", "16"))
EPOCHS   = int(os.environ.get("EPOCHS", "200"))
PATIENCE = int(os.environ.get("PATIENCE", "100"))
DEVICE   = os.environ.get("DEVICE", "0,1")           # 2×T4 DDP; "0" for single GPU/Colab
WORKERS  = int(os.environ.get("WORKERS", "4"))
SEED     = int(os.environ.get("SEED", "0"))
AUG      = os.environ.get("AUG", "mosaic")           # mosaic | mosaic_mixup
SMOKE    = os.environ.get("SMOKE", "0") == "1"
RESUME   = os.environ.get("RESUME", "0") == "1"      # resume from last.pt (Colab disconnects)
RUN_NAME = os.environ.get("RUN_NAME", "")            # override the derived run dir name

INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
WORK_ROOT  = Path(os.environ.get("WORK_ROOT", "/kaggle/working"))
# PROJECT_ROOT can point at Drive (Colab) so checkpoints survive a disconnect;
# WORK_ROOT stays local/fast for the materialised symlink tree.
PROJECT    = Path(os.environ.get("PROJECT_ROOT", str(WORK_ROOT / "tbx_runs")))
DATA_DIR   = WORK_ROOT / "tbx_yolo"                   # materialised positives-only tree

CLASS_NAMES = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]

# Aug levels — copied EXACTLY from yolo_common/aug.py (mosaic finalist + mixup).
_GEO = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=10.0, translate=0.1, scale=0.5,
            shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0)
AUG_LEVELS = {
    "mosaic":       {**_GEO, "mosaic": 1.0, "close_mosaic": 10},
    "mosaic_mixup": {**_GEO, "mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10},
}
IMG_EXTS = (".png", ".jpg", ".jpeg")


# ── Input discovery (don't hardcode a dataset slug) ───────────────────────────
def _find(name: str, *, is_dir=False) -> Path:
    for p in INPUT_ROOT.rglob(name):
        if (p.is_dir() if is_dir else p.is_file()):
            return p
    sys.exit(f"HALT: could not find {name!r} under {INPUT_ROOT}. Upload the "
             f"TBX11K bundle (splits.json + tb/ + labels_all/). See the docstring.")


def _index(dir_path: Path, exts) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in dir_path.rglob("*"):
        if p.suffix.lower() in exts and p.stem not in idx:
            idx[p.stem] = p
    return idx


# ── Materialise the FROZEN positives-only train/val tree ──────────────────────
def materialise(split: dict, img_index: dict[str, Path], lbl_index: dict[str, Path]) -> Path:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)                       # fully derived; rebuild clean

    train_ids = list(split["train_ids"])
    val_ids = list(split["val_ids"])
    if SMOKE:
        train_ids, val_ids = train_ids[:16], val_ids[:4]

    missing = []
    for subset, ids in (("train", train_ids), ("val", val_ids)):
        img_dir = DATA_DIR / "images" / subset
        lbl_dir = DATA_DIR / "labels" / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem in ids:
            src_img, src_lbl = img_index.get(stem), lbl_index.get(stem)
            if src_img is None or src_lbl is None:
                missing.append(stem)
                continue
            link = img_dir / f"{stem}{src_img.suffix}"
            if not link.exists():
                os.symlink(src_img, link)
            (lbl_dir / f"{stem}.txt").write_text(src_lbl.read_text())
    if missing:
        sys.exit(f"HALT: {len(missing)} split positives missing from the upload "
                 f"(e.g. {missing[:5]}). The frozen split references images/labels "
                 f"that aren't in the attached dataset.")

    yaml_path = DATA_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {DATA_DIR}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n")
    print(f"[materialise] FROZEN split seed={split['seed']} | "
          f"train={len(train_ids)} val={len(val_ids)} positives -> {yaml_path}")
    return yaml_path


def train(yaml_path: Path):
    from ultralytics import YOLO
    if AUG not in AUG_LEVELS:
        sys.exit(f"HALT: AUG={AUG!r} not in {sorted(AUG_LEVELS)}")
    arch = Path(MODEL).stem                            # yolov8n / yolov8s / vindr...
    run_name = RUN_NAME or f"tbx_{arch}_{AUG}_{IMGSZ}_b{BATCH}_s{SEED}"
    last_ckpt = PROJECT / run_name / "weights" / "last.pt"
    resume = RESUME and last_ckpt.exists()
    print(f"\n████ {run_name}  (model={MODEL}, imgsz={IMGSZ}, batch={BATCH}, "
          f"aug={AUG}, seed={SEED}, epochs={'2(smoke)' if SMOKE else EPOCHS}, "
          f"resume={resume}) ████\n")
    if resume:
        # Ultralytics reads data/aug/epochs from the checkpoint's own args; we
        # only point it at last.pt and say resume. The materialised tree was
        # rebuilt at the SAME WORK_ROOT path, so the stored data.yaml resolves.
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
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
            deterministic=True,
            amp=False,                                 # off by design (exp1-6)
            project=str(PROJECT),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            **AUG_LEVELS[AUG],
        )
    best = PROJECT / run_name / "weights" / "best.pt"
    (PROJECT / run_name / "DONE").write_text("ok\n")   # completion sentinel (Colab skip/resume logic)
    print(f"\nDONE. Weights: {best}")
    print("Download best.pt, then EVAL LOCALLY on the sealed test (do NOT eval here):")
    print(f"  python yolo_experiments/eval_weights.py --weights <best.pt> --imgsz {IMGSZ} "
          f"--name {run_name}")


def main():
    split = json.loads(_find("splits.json").read_text())
    tb_dir = _find("tb", is_dir=True)
    lbl_dir = _find("labels_all", is_dir=True)
    img_index = _index(tb_dir, IMG_EXTS)
    lbl_index = _index(lbl_dir, (".txt",))
    print(f"[input] splits.json seed={split['seed']} | {len(img_index)} tb images | "
          f"{len(lbl_index)} labels")
    yaml_path = materialise(split, img_index, lbl_index)
    train(yaml_path)


if __name__ == "__main__":
    main()
