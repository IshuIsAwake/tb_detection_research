"""
vindr_retinanet.py — the MIDDLE rung of the SSL chain:
                     COCO -> BYOL(NIH 112k) -> **VinDr** -> RetinaNet -> TBX11K.

WHY THIS EXISTS
    kaggle/vindr_pretrain.py does this rung for YOLO. There was no torchvision
    equivalent, so `BYOL -> VinDr -> RetinaNet` had no middle step at all. This is it.

    BYOL on unlabelled NIH gives an anatomy prior but has never seen a box. VinDr-CXR
    (15k images, 14 finding classes) teaches that prior to LOCALISE on chest X-rays,
    in-domain, before the tiny 559-image TBX11K fine-tune. It is the same staging
    BarlowTwins-CXR used (NIH SSL -> VinDr supervised), which is the closest published
    prior art to this arm.

    ⚠ CONFOUND, MEASURE IT — DO NOT ASSUME IT HELPS. VinDr supervises 14 *generic*
    abnormality classes. Our RetinaNet's known weakness is firing on sick-but-not-TB
    lungs (29 of 32 false alarms at 0.90 sensitivity; AUROC-vs-sick 0.9029, 2nd worst
    on the board). Training it to box ANY abnormality may sharpen box regression while
    making TB-vs-sick discrimination WORSE. When the TBX rung lands, read BOTH the
    Active mAP50 and the matched-sensitivity / AUROC-vs-sick table, not mAP50 alone.

SELF-CONTAINED, like the other kaggle/ scripts — `yolo_common` is not available on
Kaggle. The one exception is `vindr_pretrain`, which sits in this same directory and
is import-safe (module level is only env reads + constants, no heavy imports, main
guarded). We reuse its CSV discovery, image indexing and coordinate normalisation
rather than copy-pasting them: that logic carries a hard-won halt-on-mismatch gate for
the "CSV coords are not in this image's space" landmine.

    ⚠ The train loop below is deliberately a lean twin of `yolo_common/tv_detect.train`.
    It is NOT shared because tv_detect imports yolo_common (settings/splits/metrics),
    which does not exist on Kaggle — the same reason vindr_pretrain.py is standalone.
    Keep the two in sync by hand if the recipe changes.

OUTPUT
    `vindr_retinanet_<tag>.pt` holding the FULL detector state_dict with
    format="torchvision_detector". The TBX side (yolo_common/tv_detect.py) takes the
    **backbone = body + FPN** from it and rebuilds the heads for our 2 classes, since
    VinDr has 14. That transfers strictly more than the BYOL artifact did (which was
    the trunk only) — the FPN has now been trained to localise on CXR.

USAGE (Kaggle; attach awsaf49/vinbigdata-512-image-dataset)
    pip install --no-deps lightly     # not needed here, but the chain's earlier rung uses it

    BACKBONE_WEIGHTS=/kaggle/input/<ds>/byol_resnet50_coco_n112120_ep30.pt \
    IMGSZ=512 BATCH=8 EPOCHS=30 LR=0.01 FREEZE_BN=1 \
    python kaggle/vindr_retinanet.py

    Controls: IMGSZ BATCH ACCUM EPOCHS LR PATIENCE VAL_FRAC SEED WORKERS DEVICE
              NEGATIVES(1/0) MAX_IMAGES SMOKE BACKBONE_WEIGHTS FREEZE_BN BN_RECALIB

    ⚠ FREEZE_BN=1 is STRONGLY recommended at small batch. retinanet_resnet50_fpn_v2
    has 53 REAL BatchNorm2d layers (the *v1* model is the one with FrozenBatchNorm),
    so at batch 8 every normalisation uses 8 samples and gradient accumulation does
    NOT fix it — accumulation fixes gradients, BN is per-forward. Worse, the BYOL
    backbone arrives with BN statistics learned at batch 96; a small-batch fine-tune
    overwrites them with noise and can erase the transfer we are trying to measure.

MODEL SELECTION
    Validation LOSS (torchvision detection models return their loss dict in train
    mode), not AP. This is a PRETRAINING rung — the locked lesson from the YOLO arm is
    "do not judge the backbone by VinDr mAP; the real judge is the downstream TBX run".
    A loss-based stop is enough to pick a checkpoint and costs no AP implementation to
    get subtly wrong. Both `best` and `last` are saved so a noisy val curve is recoverable.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import vindr_pretrain` works

# ── Config (env-overridable; house style is every knob on the terminal) ───────
IMGSZ      = int(os.environ.get("IMGSZ", "512"))
BATCH      = int(os.environ.get("BATCH", "8"))
ACCUM      = int(os.environ.get("ACCUM", "2"))        # effective batch = BATCH * ACCUM
EPOCHS     = int(os.environ.get("EPOCHS", "30"))
LR         = float(os.environ.get("LR", "0.01"))
PATIENCE   = int(os.environ.get("PATIENCE", "8"))     # 0 = off
VAL_FRAC   = float(os.environ.get("VAL_FRAC", "0.08"))
SEED       = int(os.environ.get("SEED", "0"))
WORKERS    = int(os.environ.get("WORKERS", "4"))
DEVICE     = os.environ.get("DEVICE", "cuda:0")
NEGATIVES  = os.environ.get("NEGATIVES", "1") == "1"  # keep the 10,606 "No finding" imgs
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))   # >0 = subsample (debug / time-box)
SMOKE      = os.environ.get("SMOKE", "0") == "1"
WARMUP_EP  = int(os.environ.get("WARMUP_EP", "1"))

BACKBONE_WEIGHTS = os.environ.get("BACKBONE_WEIGHTS", "").strip()
FREEZE_BN        = os.environ.get("FREEZE_BN", "1") == "1"
BN_RECALIB       = int(os.environ.get("BN_RECALIB", "50"))   # batches; 0 = off

WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working"))
OUT_DIR   = WORK_ROOT / "vindr_retinanet"
OUT_NAME  = os.environ.get("OUT_NAME", "vindr_retinanet")

# VinDr: 14 finding classes (id 14 = "No finding"). torchvision reserves 0 for
# background, so a CSV class_id c maps to label c+1.
NUM_CLASSES = 14 + 1

# ⚠⚠ EVERY KNOB, RESOLVED, INTO EVERY CHECKPOINT. Read this before changing it.
#
# This is the fix for the most expensive class of bug this project has: a Kaggle
# treatment/control pair that silently disagreed on one setting. Config here arrives as
# hand-typed environment variables in a notebook cell, once per run, and nothing about
# the launch is recoverable afterwards — /kaggle/working does not persist and the logs
# go with it. The old `meta` saved batch/lr/epochs/freeze_bn/seed/n_train but NOT
# BN_RECALIB, which was the one knob that differed between vindr_retinanet_best.pt and
# vindr_retinanet_cocoinit_best.pt. Consequence: the COCO control trained 22 epochs on
# untouched torchvision BatchNorm statistics while the BYOL arm got 50 batches of
# in-domain re-estimation, the two were treated as a matched pair for a day, and the
# mismatch was only found by counting `num_batches_tracked` (864384 = the shipped
# torchvision value, vs 50) long after both runs were paid for.
#
# So: dump the whole resolved namespace, not a hand-picked subset. A knob nobody
# thought to record is exactly the knob that will differ. `tv_detect` prints this back
# out when it loads the checkpoint, so an asymmetry is visible at load time.
RUN_CONFIG = {
    "imgsz": IMGSZ, "batch": BATCH, "accum": ACCUM, "eff_batch": BATCH * ACCUM,
    "epochs": EPOCHS, "lr": LR, "patience": PATIENCE, "val_frac": VAL_FRAC,
    "seed": SEED, "workers": WORKERS, "device": DEVICE, "negatives": NEGATIVES,
    "max_images": MAX_IMAGES, "smoke": SMOKE, "warmup_ep": WARMUP_EP,
    "backbone_weights": BACKBONE_WEIGHTS or None,
    "freeze_bn": FREEZE_BN, "bn_recalib": BN_RECALIB,
    "num_classes": NUM_CLASSES, "out_name": OUT_NAME,
}

GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "0"))   # 0 = off; see tv_detect.py


def seed_everything(seed: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


# ── Data ──────────────────────────────────────────────────────────────────────
def load_vindr_items() -> list[tuple[Path, list[tuple[int, float, float, float, float]]]]:
    """(image_path, [(class_id, xc, yc, bw, bh) normalised]) for every VinDr image.

    Reuses vindr_pretrain's CSV discovery + normalisation, which already resolves the
    "CSV coords are in ORIGINAL dims, images are 512px" landmine and HALTS if the two
    do not agree. Boxes come back as normalised YOLO strings; we parse them back and
    denormalise to absolute xyxy below. The round trip is exact to 6 decimals
    (5e-4 px at 512), so nothing is lost.
    """
    import pandas as pd
    import vindr_pretrain as VP

    csv_path = VP._find_boxes_csv()
    print(f"[data] boxes CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    img_index = VP._index_images()
    print(f"[data] indexed {len(img_index)} images")
    labels = VP.build_labels(df, img_index)          # {image_id: ["cls xc yc bw bh", ...]}

    items = []
    for iid, rows in labels.items():
        p = img_index.get(iid)
        if p is None:
            continue
        boxes = []
        for r in rows:
            c, xc, yc, bw, bh = r.split()
            boxes.append((int(c), float(xc), float(yc), float(bw), float(bh)))
        if not boxes and not NEGATIVES:
            continue
        items.append((p, boxes))
    n_pos = sum(1 for _, b in items if b)
    print(f"[data] {len(items)} images ({n_pos} with boxes, {len(items)-n_pos} negatives; "
          f"NEGATIVES={'on' if NEGATIVES else 'off'})")
    if not items:
        raise SystemExit("HALT: no VinDr items — is the dataset attached?")
    return items


class VinDrDetDataset:
    """torchvision-detection dataset: returns (CHW float tensor in [0,1], target dict).

    Augmentation is hflip only, deliberately. This rung exists to teach the backbone
    to localise on CXR; the heavy geometric recipe belongs to the downstream TBX run
    where it is already ablated. Keeping it light also keeps this comparable to the
    YOLO VinDr pretrain, whose locked note was "aug barely matters for a *pretrain*".
    """

    def __init__(self, items, imgsz: int, train: bool):
        self.items, self.imgsz, self.train = items, imgsz, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        path, boxes = self.items[i]
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        if (w0, h0) != (self.imgsz, self.imgsz):
            img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        S = self.imgsz

        xyxy, labels = [], []
        for c, xc, yc, bw, bh in boxes:
            x0, y0 = (xc - bw / 2) * S, (yc - bh / 2) * S
            x1, y1 = (xc + bw / 2) * S, (yc + bh / 2) * S
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(S), x1), min(float(S), y1)
            if x1 - x0 < 1 or y1 - y0 < 1:          # degenerate after clipping
                continue
            xyxy.append([x0, y0, x1, y1])
            labels.append(c + 1)                     # 0 is background in torchvision

        import numpy as np
        arr = np.asarray(img, dtype="float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if self.train and random.random() < 0.5:     # hflip, box-aware
            t = torch.flip(t, dims=[2])
            xyxy = [[S - b[2], b[1], S - b[0], b[3]] for b in xyxy]

        target = {
            "boxes": torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return t, target


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# ── Model ─────────────────────────────────────────────────────────────────────
def load_byol_backbone(model, path: str) -> dict:
    """Load a BYOL ResNet-50 trunk into backbone.body. HALTS on an unusable init.

    Mirrors yolo_common/tv_detect.load_backbone_weights, and for the same reasons:
    each check corresponds to a failure that has already cost this project a run —
    an all-NaN backbone, a backbone byte-identical to its init because the encoder was
    frozen, and a silent strict=False load that matches nothing.
    ⚠ `num_batches_tracked` is EXCLUDED from the drift: it is a step counter, and a
    frozen encoder still ticks it, so including it makes this guard pass the exact bug
    it exists to catch.
    """
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"HALT: {path} has no usable state_dict.")

    bad = [k for k, v in sd.items()
           if torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(f"HALT: {path} has {len(bad)} non-finite tensors — a collapsed "
                         f"SSL run. Refusing to train on it.")

    body = model.backbone.body
    before = {k: v.detach().clone() for k, v in body.state_dict().items()}
    missing, unexpected = body.load_state_dict(sd, strict=False)
    matched = len(body.state_dict()) - len(missing)
    if matched == 0:
        raise SystemExit(f"HALT: {path} matched 0 tensors in backbone.body — wrong format "
                         f"(expected plain resnet50 trunk keys conv1/bn1/layer1..layer4).")
    after = body.state_dict()
    drift = sum(float((after[k].double() - before[k].double()).pow(2).sum())
                for k in before
                if before[k].is_floating_point() and "num_batches_tracked" not in k
                and not (k.endswith("running_mean") or k.endswith("running_var"))) ** 0.5
    if drift == 0.0:
        raise SystemExit(f"HALT: {path} is byte-identical to the COCO backbone — the SSL "
                         f"pretrain never moved the encoder. Not an SSL result.")
    print(f"[backbone] {Path(path).name}: {matched} loaded, {len(missing)} missing, "
          f"{len(unexpected)} unexpected, L2-vs-COCO(weights) {drift:.3f} ✓")
    return {"path": str(path), "matched": matched, "l2_vs_coco": round(drift, 3),
            "byol_meta": blob.get("byol_meta") if isinstance(blob, dict) else None}


def freeze_backbone_bn(model) -> int:
    import torch.nn as nn
    n = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
            n += 1
    return n


def recalibrate_backbone_bn(model, ds, device, n_batches: int, batch: int = 4) -> int:
    """Re-estimate backbone BN stats at THIS resolution before freezing them.

    BYOL pretrains at 256 (512 cannot finish a Kaggle session for ResNet-50); this rung
    trains at 512. BN running statistics are resolution-sensitive, so freezing the
    256-derived ones at 512 would freeze them at the wrong values for the whole run.

    It also repairs a SECOND, larger mismatch. byol_pretrain.py pins "INPUT REGIME =
    raw [0,1], NO ImageNet mean/std" — a decision made for the yolov8n arm, because
    Ultralytics feeds its backbone [0,1]. That rationale did not survive the
    ResNet-50/torchvision pivot: GeneralizedRCNNTransform DOES apply ImageNet mean/std.
    conv1 is bias-free, so the shift is affine and a correct BN re-estimate absorbs it.

    ⚠ WHICH IS WHY THE FORWARD BELOW GOES THROUGH `model.transform`. Calling
    `model.backbone(x)` directly skips the transform — i.e. skips the very
    normalisation — so BN would be calibrated on [0,1] while training feeds
    normalised pixels. Measured (2026-08-01, 16 real TBX images): bn1 running_var
    0.4237 vs 8.03 (19x off, exactly 1/0.225^2) and FPN activations ~1000x too large
    (std 7096 vs 6.3, absmax 70784 — past fp16's 65504, and autocast is on). Using the
    model's own transform also means the mean/std constants here cannot drift from it.
    """
    import torch
    import torch.nn as nn
    bns = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns or n_batches <= 0:
        return 0
    saved_momentum = [m.momentum for m in bns]   # restored below
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                      # cumulative average over this pass
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     num_workers=0, collate_fn=collate)
    was = model.training
    model.train()
    seen = 0
    with torch.no_grad():
        for i, (imgs, _t) in enumerate(dl):
            if i >= n_batches:
                break
            x, _ = model.transform([im.to(device) for im in imgs])   # normalise + resize
            model.backbone(x.tensors)
            seen += len(imgs)
    model.train(was)
    # Restore momentum: `None` means "cumulative average since the last reset", correct
    # for THIS pass and wrong for training, where it would make the statistics ossify as
    # 1/n shrinks. Masked while FREEZE_BN=1 (BN never updates), live at FREEZE_BN=0.
    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom
    print(f"[bn] recalibrated {len(bns)} BatchNorm layers on {seen} images @ {IMGSZ}px "
          f"(through model.transform — ImageNet-normalised, as training feeds it)")
    return seen


def build_model():
    import torch.nn as nn
    from functools import partial
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        score_thresh=0.001, detections_per_img=100,
        min_size=IMGSZ, max_size=IMGSZ)
    in_ch = model.backbone.out_channels
    n_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_ch, n_anchors, NUM_CLASSES, norm_layer=partial(nn.GroupNorm, 32))
    return model


@__import__("contextlib").contextmanager
def _bn_eval(model):
    """BN in eval for a validation pass, so measuring never mutates running stats."""
    import torch.nn as nn
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d) and m.training]
    for m in bns:
        m.eval()
    try:
        yield
    finally:
        for m in bns:
            m.train()


def val_loss(model, dl, device) -> float:
    """Mean total loss on val. torchvision detectors only return losses in train mode,
    so we stay in train mode but hold BN in eval (see _bn_eval) — otherwise the act of
    validating would move the running statistics we are trying to preserve."""
    import torch
    was = model.training
    model.train()
    tot, n = 0.0, 0
    with torch.no_grad(), _bn_eval(model):
        for imgs, targets in dl:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(imgs, targets)
            tot += float(sum(losses.values()))
            n += 1
    model.train(was)
    return tot / max(1, n)


def save_ckpt(model, path: Path, meta: dict) -> None:
    import torch
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = [k for k, v in sd.items()
           if torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(f"HALT: refusing to write {path.name} — {len(bad)} non-finite tensors.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "retinanet", "format": "torchvision_detector",
                "num_classes": NUM_CLASSES, "imgsz": IMGSZ,
                "state_dict": sd, "vindr_meta": meta,
                "date": datetime.now(timezone.utc).isoformat()}, path)
    print(f"[save] {path}  ({len(sd)} tensors)")


def main() -> None:
    import torch

    seed_everything(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[cfg] imgsz={IMGSZ} batch={BATCH} accum={ACCUM} (eff {BATCH*ACCUM}) "
          f"epochs={EPOCHS} lr={LR} device={device} smoke={SMOKE}")

    items = load_vindr_items()
    if MAX_IMAGES:
        items = items[:MAX_IMAGES]
    if SMOKE:
        items = items[:24]
    # ⚠ SORT BEFORE SHUFFLE — the shuffle is seeded but the INPUT ORDER was not.
    # `items` is built by iterating `build_labels`' dict, which is keyed off a set of
    # image ids, so its order varies with string hashing between processes.
    # `seed_everything` writes PYTHONHASHSEED, but that is a no-op after the interpreter
    # has already started, so it never pinned anything. Seeding a shuffle of a
    # differently-ordered list gives a different split every run — meaning the treatment
    # and control rungs did not even train on the same 13,800 images. Sorting by path
    # makes the split a pure function of SEED, so a re-run reproduces it and a paired
    # run matches it.
    items.sort(key=lambda t: str(t[0]))
    random.Random(SEED).shuffle(items)
    n_val = max(1, int(len(items) * VAL_FRAC))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"[split] train={len(train_items)} val={len(val_items)} (VAL_FRAC={VAL_FRAC}, "
          f"seed={SEED}, deterministic: sorted before shuffle)")

    ds_tr = VinDrDetDataset(train_items, IMGSZ, train=True)
    ds_va = VinDrDetDataset(val_items, IMGSZ, train=False)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                                        num_workers=WORKERS, collate_fn=collate,
                                        pin_memory=(device.type == "cuda"))
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                                        num_workers=WORKERS, collate_fn=collate)

    model = build_model().to(device)
    binit = load_byol_backbone(model, BACKBONE_WEIGHTS) if BACKBONE_WEIGHTS else None
    if not BACKBONE_WEIGHTS:
        print("[backbone] ⚠ no BACKBONE_WEIGHTS — this is the COCO-init CONTROL arm, "
              "not the BYOL rung.")
    # Order matters: recalibrate at this resolution BEFORE freezing, or we freeze stale stats.
    # ⚠ NOT gated on BACKBONE_WEIGHTS (fixed 2026-08-01). If the statistics are about to be
    # FROZEN for the whole run, they must be estimated on the data and resolution about to be
    # trained on — whatever the init. Gating on the init instead gave the BYOL arm VinDr@512
    # stats while the COCO control kept frozen COCO natural-image stats, so any BYOL win would
    # be partly the recalibration. Same class of asymmetry as the bug this call had inside it.
    n_recal = 0
    if BN_RECALIB and FREEZE_BN:
        n_recal = recalibrate_backbone_bn(model, ds_tr, device, BN_RECALIB)
    else:
        # Say it out loud. The silent version of this branch is what produced an
        # unmatched pair; a control that skipped recalibration must announce itself.
        print(f"[bn] ⚠ NO recalibration (BN_RECALIB={BN_RECALIB}, FREEZE_BN={FREEZE_BN}) "
              f"— this backbone keeps the BN statistics it arrived with. If the paired "
              f"run DID recalibrate, these two are not comparable.", flush=True)
    n_bn = freeze_backbone_bn(model) if FREEZE_BN else 0
    if FREEZE_BN:
        print(f"[bn] froze {n_bn} backbone BatchNorm layers — required at batch {BATCH}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    epochs = 2 if SMOKE else EPOCHS
    warm_steps = max(1, WARMUP_EP * len(dl_tr))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best, best_ep, gstep = float("inf"), -1, 0
    t_start = time.time()

    for ep in range(epochs):
        model.train()
        if FREEZE_BN:
            freeze_backbone_bn(model)          # model.train() re-enables BN every epoch
        cos_lr = 0.5 * LR * (1 + math.cos(math.pi * ep / max(1, epochs)))
        run, nb, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)

        for i, (imgs, targets) in enumerate(dl_tr):
            lr_now = cos_lr * min(1.0, (gstep + 1) / warm_steps)
            for g in opt.param_groups:
                g["lr"] = lr_now
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = sum(model(imgs, targets).values())
            if not torch.isfinite(loss):
                raise SystemExit(f"HALT: non-finite loss at ep{ep+1} step{i} — stopping "
                                 f"before it writes a poisoned checkpoint.")
            scaler.scale(loss / ACCUM).backward()
            if (i + 1) % ACCUM == 0 or (i + 1) == len(dl_tr):
                # Clip on UNSCALED grads: one exploding batch otherwise makes GradScaler
                # skip the step and halve the scale, and a recurrence decays the scale to
                # subnormal — after which every gradient underflows to zero and the model
                # is frozen while the loop still reports "training". That cost the TBX rung
                # 31 dead epochs on 2026-08-02. The `or (i+1)==len(dl_tr)` also flushes the
                # final partial accumulation window, which used to be silently dropped.
                if GRAD_CLIP_NORM > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += float(loss.detach()); nb += 1; gstep += 1
            if i % 50 == 0:
                print(f"  · ep{ep+1} {i+1}/{len(dl_tr)}  loss={float(loss):.4f}  "
                      f"lr={lr_now:.2e}", flush=True)
            if ep == 0 and i == 29:
                per = (time.time() - t0) / 30
                tot_h = per * len(dl_tr) * epochs / 3600
                print(f"\n[eta] {per*1000:.0f} ms/step · {per*len(dl_tr)/60:.1f} min/epoch "
                      f"· {epochs} epochs → ~{tot_h:.1f} h", flush=True)
                if tot_h > 11:
                    print("[eta] ⚠⚠ WILL NOT FINISH a 12 h Kaggle session — lower EPOCHS "
                          "or IMGSZ, or set MAX_IMAGES.", flush=True)
                print()

        vl = val_loss(model, dl_va, device)
        print(f"[ep {ep+1:03d}/{epochs}] train_loss={run/max(1,nb):.4f}  val_loss={vl:.4f}  "
              f"lr={cos_lr:.2e}  {time.time()-t0:.0f}s", flush=True)

        # `config` is the FULL resolved namespace — see RUN_CONFIG's note. Never trim it
        # back to a hand-picked subset; the knob nobody records is the one that differs.
        # `bn_recalib_images` is what ACTUALLY ran, not what was requested.
        meta = {"epoch": ep + 1, "of": epochs, "val_loss": round(vl, 5),
                "train_loss": round(run / max(1, nb), 5), "backbone_init": binit,
                "config": RUN_CONFIG, "bn_recalib_images": n_recal,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "freeze_bn": FREEZE_BN, "imgsz": IMGSZ, "batch": BATCH, "accum": ACCUM,
                "lr": LR, "seed": SEED, "negatives": NEGATIVES,
                "n_train": len(train_items), "n_val": len(val_items)}
        save_ckpt(model, OUT_DIR / f"{OUT_NAME}_last.pt", meta)
        if vl < best:
            best, best_ep = vl, ep + 1
            save_ckpt(model, OUT_DIR / f"{OUT_NAME}_best.pt", {**meta, "note": "best-by-val-loss"})
        elif PATIENCE and (ep + 1 - best_ep) >= PATIENCE:
            print(f"[stop] val_loss flat for {PATIENCE} epochs (best {best:.4f} @ ep{best_ep})")
            break

    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"best_val_loss": best, "best_epoch": best_ep, "epochs_ran": ep + 1,
         "hours": round((time.time() - t_start) / 3600, 2),
         "backbone_init": binit, "freeze_bn": FREEZE_BN}, indent=2))
    print(f"\nDONE. best val_loss {best:.4f} @ ep{best_ep} → {OUT_DIR/f'{OUT_NAME}_best.pt'}")
    print("Download it, place at weights/pretrain/vindr_from_byol_best.pt, then the TBX rung:")
    print("  BACKBONE_WEIGHTS=weights/pretrain/vindr_from_byol_best.pt FREEZE_BN=1 \\")
    print("  python yolo_experiments/retinanet.py --name retinanet_byolvindr_s0")


if __name__ == "__main__":
    main()
