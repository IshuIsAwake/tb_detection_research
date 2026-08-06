"""
tv_detect.py — Shared torchvision detection harness (RetinaNet / Faster R-CNN)
               scored on the SAME frozen split + sealed-360 as the YOLO champion.

WHY THIS EXISTS
    Our YOLO detector is locked at Active mAP50 0.745 on the sealed split. The
    question this module answers: does a DIFFERENT architecture family — a
    one-stage RetinaNet (SymFormer's own leaderboard baseline was RetinaNet+P2T)
    or a two-stage Faster R-CNN — do better on the SAME data? CodaLab is dead, so
    the only comparison available is our own sealed split; this keeps it exactly
    apples-to-apples with the 0.745 by reusing yolo_common/splits.py (the frozen
    split), yolo_common/metrics.py (the model-agnostic localisation / screening /
    by-size / confusion passes), and an AP@0.5 computed here from raw predictions.

    The two experiment entry points are separate thin files by request:
        yolo_experiments/retinanet.py     (arch="retinanet")
        yolo_experiments/faster_rcnn.py   (arch="fasterrcnn")
    Both just call run(arch, args); all logic lives here (repo no-copy-paste rule).

COMPARABILITY NOTE
    The YOLO 0.745 was produced by Ultralytics' own AP. This module computes AP
    with a self-contained COCO-style 101-point interpolation at IoU 0.5 (mAP50)
    and averaged 0.5:0.05:0.95 (mAP50-95). To compare on IDENTICAL AP code, the
    champion can be re-scored via `ap_from_per_image(metrics.collect(yolo, ...))`
    — see rescore_yolo() at the bottom. Expect small (~±0.01) shifts vs the
    Ultralytics digit; that gap is the AP-implementation difference, not a model
    difference, and it applies equally to all three models here.

FRAMEWORK CAVEATS (stated, not hidden)
    - Default aug `geo` MATCHES YOLO's `_GEO` level (box-aware affine: degrees=10,
      translate=0.1, scale=[0.5,1.5], fliplr=0.5) so the run is comparable to our
      YOLO `geo` runs (≈0.685 Active mAP50) — that isolates the ARCHITECTURE at a
      matched recipe. A ResNet-50-FPN (~30-40M params) on 559 train positives is a
      big model on tiny data — COCO-pretrained, so it is transfer, but expect
      variance.
    - ⚠ `--aug mosaic|mosaic_mixup` now EXIST here (added 2026-08-03). torchvision
      ships neither, so both are implemented in TBDetDataset — see the mosaic block
      for why mosaic cannot be a v2 transform and why the composite must be cropped
      back to imgsz. They are the SHAPE of Ultralytics' versions, not a port, so
      they compare validly to `geo` HERE but do not reproduce YOLO's mosaic cell.
    - Detection models resize to min_size/max_size internally; we pin both to the
      image size (512) so inputs match the YOLO 512 runs and fit 6 GB.

GPU RULE
    A full train is the user's to launch. Smoke locally first:
        TB_SMOKE=1 python yolo_experiments/retinanet.py
        TB_SMOKE=1 python yolo_experiments/faster_rcnn.py
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from yolo_common import metrics, settings as S, splits


# ── single-class (lesion-finder) mode ─────────────────────────────────────────
# Set ONCE by run() from --single-class. When False every code path below is
# bit-identical to the 2-class one that produced the 13 runs on the board.
#
# Why a module flag and not settings.CLASS_NAMES: convert.py re-verifies the
# global class map against the COCO JSON and halts on mismatch, and the YOLO
# label files on disk are 2-class. So the merge is a RUNTIME transform applied
# after reading, never a change to the frozen data or the global vocabulary.
#
# ⚠ A single-class run reports its headline under the key "lesion", NOT "active".
# The alias would have been free, but a lesion number sitting in metrics.json
# under "active" is indistinguishable from a 2-class Active number and would go
# straight onto the board next to 0.7580. Loud KeyError beats a silent mix-up.
SINGLE_CLASS = False


def _fg(cls: int) -> int:
    """GT class -> foreground index. Single-class merges Active+Obsolete into 0."""
    return 0 if SINGLE_CLASS else cls


def class_keys() -> list[str]:
    """Per-class report keys, in output order."""
    return ["lesion"] if SINGLE_CLASS else ["active", "obsolete"]


def headline_key() -> str:
    """The class whose mAP50 is the run's headline / model-selection signal."""
    return "lesion" if SINGLE_CLASS else "active"


# ── device ────────────────────────────────────────────────────────────────────
def resolve_device(dev: str | None):
    import torch
    d = dev if dev is not None else S.DEVICE
    if d in ("cpu", "cpu:0"):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0" if d in ("0", "cuda", "cuda:0", "") else f"cuda:{d}")


# ── dataset ───────────────────────────────────────────────────────────────────
_GEO_AUGS = ("geo", "mosaic", "mosaic_mixup")   # levels that carry the YOLO _GEO affine


def _build_aug(aug: str, translate_div: float = 1.0):
    """Box-aware train transform (torchvision v2). `geo` MATCHES YOLO's
    yolo_common/aug.py `_GEO` level so a torchvision run is comparable to our
    YOLO `geo` runs: degrees=10, translate=0.1, scale=[0.5,1.5], fliplr=0.5, gray
    (114) border fill, NO shear/perspective/photometric (X-ray intensity is
    diagnostic; upright pose). `hflip` = flip only; `off` = none/eval.

    `mosaic` / `mosaic_mixup` get the SAME affine — they are `geo` plus a
    Dataset-level composite step, exactly as YOLO's aug.py builds them
    (`{**_GEO, "mosaic": 1.0, ...}`). See the mosaic block below.

    ⚠ `translate_div` exists only for the mosaic path. v2.RandomAffine takes
    `translate` as a FRACTION OF THE INPUT, and the mosaic path feeds it a 2S canvas
    that is then cropped back to S. At translate=0.1 that would shift by 0.1*2S =
    twice the intended pixels, so mosaic would silently run a stronger geometry than
    `geo` and the two would no longer be one-knob comparable. Passing 2.0 restores
    0.1*S. Defaults to 1.0, so the `geo` path is untouched."""
    from torchvision.transforms import v2
    if aug == "off":
        return None
    ops = []
    if aug in _GEO_AUGS:
        ops.append(v2.RandomAffine(degrees=10,
                                   translate=(0.1 / translate_div, 0.1 / translate_div),
                                   scale=(0.5, 1.5), fill=114))       # == YOLO _GEO
    ops.append(v2.RandomHorizontalFlip(p=0.5))                        # fliplr=0.5
    ops += [v2.ClampBoundingBoxes(), v2.SanitizeBoundingBoxes()]      # drop boxes warped out
    return v2.Compose(ops)


# ── mosaic / mixup ────────────────────────────────────────────────────────────
# ⚠ MOSAIC CANNOT BE A v2 TRANSFORM. A torchvision v2 op sees ONE sample; mosaic
# needs FOUR. So it lives in the Dataset. mixup alone could have gone in _build_aug,
# but it is kept here so the two read as one recipe and share _sample().
#
# WHY THIS EXISTS (2026-08-03). Every torchvision run on this arm is aug='geo' —
# there has never been an augmentation ablation here, while on the YOLO arm
# mosaic_mixup beat geo 0.734 vs 0.682 in COCO units (+0.052): the largest single
# lever that arm found, and ~2x the ±0.025 significance bar. With the initialisation
# axis closed (four pretraining chains all landing at 0.754 ±0.013), augmentation is
# one of the three levers left.
#
# ⚠⚠ ORDER IS LOAD-BEARING: CANVAS → AFFINE → CROP. Ultralytics fuses the two with
# random_perspective(border=-imgsz//2): the warp runs on the 2S canvas and its OUTPUT
# is the central S window. Doing it the other way round (crop to S, then affine) is
# the obvious implementation and it is WRONG — measured 2026-08-03, and visible at a
# glance in a rendered grid. At the scale=0.5 end of the affine the already-cropped
# 512 tile shrinks to ~256 centred in a 512 frame and the rest fills with grey 114:
# roughly a quarter of the sample is padding, and every lesion in it is half-size.
# That is precisely the small-box damage the crop exists to prevent. Warping the 2S
# canvas first means scale<1 pulls MORE chest into frame instead of padding it.
#
# ⚠ THE CROP MUST HAPPEN IN THE DATASET, and this is the trap. build_model pins
# min_size == max_size == imgsz, so GeneralizedRCNNTransform resizes whatever it is
# handed down to 512. Returning the 1024 canvas would therefore shrink every chest to
# 256 and halve every lesion — silently attacking small-box recall, already this
# arm's weakest bucket. Cropping back to S keeps lesions at native scale.
#
# ⚠ STILL NOT A BIT-EXACT PORT. Ultralytics warps with a single fused perspective
# matrix and filters candidates inside it; here the affine is torchvision's v2 op and
# the candidate filter runs after the crop. `mosaic` is therefore comparable to `geo`
# HERE — do not read it as reproducing YOLO's mosaic cell.
#
# ⚠ NO close_mosaic. YOLO's champion disabled mosaic for its last 10 epochs so
# training ends on real images. That needs an epoch-aware Dataset, i.e. a change to
# train() — deliberately NOT made while the freeze ladder is running on this file.
# Recipe difference vs the champion, stated rather than hidden.
_MOSAIC_WH_THR = 2.0      # drop a cropped box thinner/shorter than this many px...
_MOSAIC_AREA_THR = 0.10   # ...or with under 10% of its original area left. These are
                          # Ultralytics' box_candidates defaults, and they are
                          # load-bearing: without them a 2-px sliver of a lesion at a
                          # crop edge becomes a full-confidence GT box. That is pure
                          # label noise injected exactly into the small-box regime.
_MIXUP_P = 0.10           # == the champion's `mixup: 0.1`
_MIXUP_BETA = 32.0        # Beta(32,32) ≈ 0.5 ± 0.06 — a true double exposure, not a
                          # near-copy of one image with a ghost of the other.


class TBDetDataset:
    """Positives from the frozen split → (image[0..1] CHW float, target dict).

    target = {"boxes": FloatTensor[N,4] xyxy pixel, "labels": Int64[N] in 1..2}.
    YOLO class 0/1 (Active/Obsolete) → torchvision label +1 (0 = background).
    Train aug is box-aware via _build_aug (see it for the YOLO-`geo` match);
    `mosaic`/`mosaic_mixup` additionally compose samples here (see the block above).
    """

    def __init__(self, stems: list[str], train: bool, aug: str = "geo"):
        self.stems = list(stems)
        self.train = train
        self.aug = aug
        # Gated on `train` too: recalibrate_backbone_bn() and every eval path build
        # this with train=False, and a BN re-estimate must see REAL images.
        self.mosaic = bool(train) and aug in ("mosaic", "mosaic_mixup")
        self.mixup = bool(train) and aug == "mosaic_mixup"
        # The affine runs on the 2S mosaic canvas, so its translate fraction is
        # halved to keep the pixel shift equal to the `geo` runs' — see _build_aug.
        self.tf = _build_aug(aug, translate_div=2.0 if self.mosaic else 1.0) \
            if train else None

    def __len__(self):
        return len(self.stems)

    def _load_raw(self, i):
        """(uint8 CHW image, boxes xyxy float32 [N,4], labels int64 [N]) — NO aug.

        Split out of __getitem__ so mosaic can pull SIBLING samples; that is the
        whole reason mosaic cannot be a v2 transform."""
        import torch
        from torchvision.io import read_image, ImageReadMode

        stem = self.stems[i]
        img = read_image(str(splits.tb_image_path(stem)), ImageReadMode.RGB)   # uint8 [3,H,W]
        _, h, w = img.shape
        boxes, labels = [], []
        for cls, (x1, y1, x2, y2) in metrics._gt_boxes(splits.label_path(stem), w, h):
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(_fg(cls) + 1)                         # +1: 0 = background
        return (img,
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                torch.tensor(labels, dtype=torch.int64))

    @staticmethod
    def _crop_boxes(boxes, labels, x0, y0, w, h, area0):
        """Shift boxes into the crop window, clip to it, then apply the
        box_candidates filter (see _MOSAIC_WH_THR / _MOSAIC_AREA_THR)."""
        if not len(boxes):
            return boxes, labels
        b = boxes.clone()
        b[:, 0::2] = (b[:, 0::2] - x0).clamp(0, w)
        b[:, 1::2] = (b[:, 1::2] - y0).clamp(0, h)
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        keep = ((bw > _MOSAIC_WH_THR) & (bh > _MOSAIC_WH_THR)
                & ((bw * bh) / area0.clamp(min=1e-6) > _MOSAIC_AREA_THR))
        return b[keep], labels[keep]

    def _mosaic_canvas(self, i):
        """Ultralytics' mosaic4 layout: a 2S canvas with a JITTERED join point, four
        chests butted against it, each clipped by the canvas edge, boxes offset to
        match. Returns the 2S canvas — the affine runs on THIS and _sample crops the
        central S window afterwards (see the block comment: that order is the point).

        Randomness lives in the join point, exactly as upstream, so the crop itself is
        deterministic. Tile size comes from the PRIMARY image (every TBX11K image is
        512x512), so this needs no imgsz argument and cannot drift from --imgsz.
        """
        import torch

        img0, b0, l0 = self._load_raw(i)
        _, H, W = img0.shape
        cy = random.randint(H // 2, 3 * H // 2)         # the join point; Ultralytics'
        cx = random.randint(W // 2, 3 * W // 2)         # uniform(-border, 2s+border)
        canvas = torch.full((3, 2 * H, 2 * W), 114, dtype=img0.dtype)   # == the affine fill
        boxes, labels = [], []
        idxs = [i] + [random.randrange(len(self.stems)) for _ in range(3)]
        for k, j in enumerate(idxs):
            img, bb, ll = (img0, b0, l0) if k == 0 else self._load_raw(j)
            h, w = img.shape[1], img.shape[2]
            if k == 0:                                  # top-left: its BR corner at (cx,cy)
                xa1, ya1, xa2, ya2 = max(cx - w, 0), max(cy - h, 0), cx, cy
                xb1, yb1 = w - (xa2 - xa1), h - (ya2 - ya1)
            elif k == 1:                                # top-right: its BL corner
                xa1, ya1, xa2, ya2 = cx, max(cy - h, 0), min(cx + w, 2 * W), cy
                xb1, yb1 = 0, h - (ya2 - ya1)
            elif k == 2:                                # bottom-left: its TR corner
                xa1, ya1, xa2, ya2 = max(cx - w, 0), cy, cx, min(cy + h, 2 * H)
                xb1, yb1 = w - (xa2 - xa1), 0
            else:                                       # bottom-right: its TL corner
                xa1, ya1, xa2, ya2 = cx, cy, min(cx + w, 2 * W), min(cy + h, 2 * H)
                xb1, yb1 = 0, 0
            canvas[:, ya1:ya2, xa1:xa2] = img[:, yb1:yb1 + (ya2 - ya1),
                                              xb1:xb1 + (xa2 - xa1)]
            if len(bb):
                bb = bb.clone()
                bb[:, 0::2] += xa1 - xb1                # padw
                bb[:, 1::2] += ya1 - yb1                # padh
                boxes.append(bb)
                labels.append(ll)
        if not boxes:                                   # cannot happen on positives-only
            return img0, b0, l0
        boxes, labels = torch.cat(boxes), torch.cat(labels)
        # A tile clipped by the canvas edge takes part of its lesion with it — same
        # filter as the final crop, so a sliver never survives as a full-weight GT box.
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes, labels = self._crop_boxes(boxes, labels, 0, 0, 2 * W, 2 * H, area0)
        return canvas, boxes, labels

    def _sample(self, i):
        """One fully-augmented sample: (float[0..1] CHW, boxes, labels).

        mosaic path: 2S canvas → affine/flip → central S crop. Plain path: image →
        affine/flip. mixup composes TWO of these, which is why it is a separate step
        in __getitem__."""
        import torch
        from torchvision import tv_tensors

        img, boxes_t, labels_t = (self._mosaic_canvas(i) if self.mosaic
                                  else self._load_raw(i))
        _, h, w = img.shape
        if self.tf is not None and len(boxes_t):
            bb = tv_tensors.BoundingBoxes(boxes_t, format="XYXY", canvas_size=(h, w))
            img, out = self.tf(tv_tensors.Image(img), {"boxes": bb, "labels": labels_t})
            boxes_t = out["boxes"].as_subclass(torch.Tensor).float().reshape(-1, 4)
            labels_t = out["labels"].as_subclass(torch.Tensor)
        if self.mosaic:
            # Central S window of the WARPED 2S canvas == Ultralytics' border=-S//2.
            # An empty result is fine and is upstream's behaviour too: RetinaNet reads
            # a zero-box target as a pure-background sample (verified — finite loss).
            y0, x0 = h // 4, w // 4
            area0 = ((boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
                     if len(boxes_t) else None)
            img = img[:, y0:y0 + h // 2, x0:x0 + w // 2]
            boxes_t, labels_t = self._crop_boxes(boxes_t, labels_t, x0, y0,
                                                 w // 2, h // 2, area0)
        return img.as_subclass(torch.Tensor).float() / 255.0, boxes_t, labels_t

    def __getitem__(self, i):
        import torch

        img, boxes_t, labels_t = self._sample(i)
        if self.mixup and random.random() < _MIXUP_P:
            # Double exposure of two (mosaic) chests. Boxes are CONCATENATED, not
            # blended: both sets of lesions are genuinely present in the pixels, at
            # reduced contrast. That is the entire regularisation — the detector must
            # fire on a lesion it can only half see.
            img2, b2, l2 = self._sample(random.randrange(len(self.stems)))
            if img2.shape == img.shape:
                lam = float(np.random.beta(_MIXUP_BETA, _MIXUP_BETA))
                img = lam * img + (1.0 - lam) * img2
                boxes_t = torch.cat([boxes_t, b2])
                labels_t = torch.cat([labels_t, l2])
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([i])}
        return img, target


def _collate(batch):
    return tuple(zip(*batch))


# ── model ─────────────────────────────────────────────────────────────────────
def load_backbone_weights(model, path) -> dict:
    """Load a pretrained backbone. HALTS rather than training on a bad init.

    TWO ACCEPTED FORMATS, and they transfer different amounts — the report says which:

      * `torchvision_resnet_trunk` (kaggle/byol_pretrain.py ARCH=resnet50)
        → the ResNet-50 TRUNK only, into `backbone.body`. FPN, heads and anchors stay
          COCO-pretrained, so this isolates ONE variable: the backbone. It is NOT a
          pure SSL model — say so when reporting, exactly as the yolov8n arm had to.

      * `torchvision_detector` (kaggle/vindr_retinanet.py)
        → trunk **and FPN**, into `backbone` — the BYOL→VinDr rung. The FPN has been
          trained to localise on CXR, so strictly more transfers than above. The heads
          are NOT taken: VinDr has 14 classes, we have 2, so they are rebuilt.

    Four checks, each one a bug that has already cost this project a run:
      * finite      — the 2026-07-27 yolov8n run saved an all-NaN backbone.
      * differs     — the 2026-07-26 run saved a backbone byte-identical to its init
                      because the encoder was frozen; two runs reported as "SSL".
      * key match   — a silent strict=False load that matches nothing trains a COCO
                      model while you believe it is BYOL.
      * counters    — `num_batches_tracked` is EXCLUDED from the drift below. It is a
                      step counter, and a FROZEN encoder still ticks it, so including
                      it made this guard pass the exact failure it exists to catch
                      (caught 2026-07-29). BN running stats are excluded for the same
                      reason: "the encoder learned" must not be satisfiable by
                      statistics drifting while the weights sit still.
    """
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"HALT: {path} has no usable state_dict.")
    fmt = blob.get("format", "torchvision_resnet_trunk") if isinstance(blob, dict) else \
        "torchvision_resnet_trunk"

    bad = [k for k, v in sd.items()
           if torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(f"HALT: backbone init {path} has {len(bad)} non-finite tensors "
                         f"(first: {bad[:3]}) — a collapsed SSL run. Refusing to train on it.")

    if fmt == "torchvision_detector":
        # Full detector checkpoint: lift the `backbone.*` subtree (body + FPN) only.
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not sd:
            raise SystemExit(f"HALT: {path} declares format=torchvision_detector but has no "
                             f"`backbone.*` keys.")
        target = model.backbone
    else:
        target = model.backbone.body

    body = target
    before = {k: v.detach().clone() for k, v in body.state_dict().items()}
    missing, unexpected = body.load_state_dict(sd, strict=False)
    matched = len(body.state_dict()) - len(missing)
    if matched == 0:
        raise SystemExit(
            f"HALT: {path} (format={fmt}) matched 0 tensors — wrong format. Expected either "
            f"plain resnet50 trunk keys (conv1/bn1/layer1..layer4) or a detector checkpoint "
            f"carrying `backbone.*`.")
    after = body.state_dict()
    # Weights only — `num_batches_tracked` is a step counter, and a frozen encoder
    # still ticks it (see the matching note in byol_pretrain.save_resnet50_init).
    drift = sum(float((after[k].double() - before[k].double()).pow(2).sum())
                for k in before
                if before[k].is_floating_point() and "num_batches_tracked" not in k
                and not (k.endswith("running_mean") or k.endswith("running_var"))) ** 0.5
    if drift == 0.0:
        raise SystemExit(f"HALT: {path} is byte-identical to the COCO backbone — the SSL "
                         f"pretrain never moved the encoder. Do not report this as SSL.")
    scope = "body+FPN" if fmt == "torchvision_detector" else "body (trunk only)"
    rep = {"path": str(path), "format": fmt, "scope": scope, "matched": matched,
           "missing": len(missing), "unexpected": len(unexpected),
           "l2_vs_coco": round(drift, 3),
           "byol_meta": blob.get("byol_meta") if isinstance(blob, dict) else None,
           "vindr_meta": blob.get("vindr_meta") if isinstance(blob, dict) else None}
    print(f"[backbone] {Path(path).name} [{fmt}] → {scope}: {matched} tensors loaded, "
          f"{len(missing)} missing, {len(unexpected)} unexpected, "
          f"L2-vs-COCO(weights) {drift:.3f} ✓")
    # ⚠ ECHO THE UPSTREAM RUN'S CONFIG. A pretraining rung's settings are otherwise
    # invisible here, and two checkpoints that differ on one knob look identical at the
    # load line. That is exactly how vindr_retinanet_best.pt (BN recalibrated on 50
    # batches) and vindr_retinanet_cocoinit_best.pt (never recalibrated) were paired.
    # Printed at load so the mismatch is on screen BEFORE the GPU hours are spent.
    vm = rep.get("vindr_meta") or {}
    up = vm.get("config")
    if up:
        keys = ("bn_recalib", "freeze_bn", "imgsz", "batch", "accum", "lr", "epochs", "seed")
        print("[backbone]   upstream config: "
              + "  ".join(f"{k}={up.get(k)}" for k in keys if k in up))
        if vm.get("bn_recalib_images") is not None:
            print(f"[backbone]   upstream bn_recalib_images={vm['bn_recalib_images']} "
                  f"(0 => that rung did NOT re-estimate BN; check the paired run matches)")
    elif fmt == "torchvision_detector":
        print("[backbone]   ⚠ no upstream `config` in this checkpoint — it predates the "
              "provenance fix (2026-08-02). Verify the paired run's settings by hand: "
              "backbone.body.bn1.num_batches_tracked == 864384 means BN was NEVER "
              "recalibrated (that is the stock torchvision value).")
    return rep


def build_model(arch: str, imgsz: int, score_thresh: float = 0.001, pretrained: bool = True,
                backbone_weights=None, trainable_layers=None, num_fg_classes: int | None = None):
    """COCO-pretrained RetinaNet / Faster R-CNN, head rebuilt for our 2 classes.
    min_size == max_size == imgsz so inputs stay at 512 (matches YOLO, fits 6 GB).
    A low score_thresh keeps the full PR curve for AP.

    `backbone_weights` swaps in a BYOL-pretrained ResNet-50 trunk (see
    load_backbone_weights); everything else stays COCO.

    `trainable_layers` is torchvision's ResNet freeze ladder — the equivalent of
    YOLO's `freeze=` (exp6). It counts trainable stages from the TOP:

        5  train everything                       == YOLO freeze=none
        4  freeze conv1+bn1
        3  freeze conv1+bn1, layer1               <- torchvision DEFAULT when
                                                     pretrained weights are passed,
                                                     so it is what retinanet_s0
                                                     (0.739) and fasterrcnn_s0
                                                     (0.726) were measured at
        2  freeze through layer2
        1  train layer4 only
        0  freeze the whole ResNet body

    ⚠ It gates the ResNet BODY ONLY — the FPN and the detection heads always train.
    So `0` is a frozen-TRUNK arm, not a linear probe. That is the right shape for
    asking "are the BYOL features good enough to detect from without rewriting
    them", but do not report it as a linear probe.

    ⚠ Independent of `freeze_bn`, which freezes BatchNorm *statistics* at every
    depth and leaves conv trainable. The two compose; a run can set both.

    None keeps torchvision's default (3 with pretrained weights, 5 without)."""
    import torch.nn as nn
    from functools import partial
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)

    # num_fg_classes=None follows the SINGLE_CLASS module flag (the training path).
    # load_model passes it EXPLICITLY, inferred from the checkpoint, so an external
    # scorer can load a single-class run without setting the global first.
    n_fg = num_fg_classes if num_fg_classes is not None else (1 if SINGLE_CLASS else S.NUM_CLASSES)
    num_classes = n_fg + 1                                          # + background
    common = dict(min_size=imgsz, max_size=imgsz)
    if trainable_layers is not None:
        if not 0 <= int(trainable_layers) <= 5:
            raise ValueError(f"trainable_layers must be 0..5, got {trainable_layers!r}")
        common["trainable_backbone_layers"] = int(trainable_layers)

    if arch == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        w = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        model = retinanet_resnet50_fpn_v2(weights=w, score_thresh=score_thresh,
                                          detections_per_img=100, **common)
        in_ch = model.backbone.out_channels
        n_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_ch, n_anchors, num_classes, norm_layer=partial(nn.GroupNorm, 32))
    elif arch == "fasterrcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=score_thresh,
                                           box_detections_per_img=100, **common)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r} (retinanet | fasterrcnn)")
    if backbone_weights:
        model._backbone_init = load_backbone_weights(model, backbone_weights)
    return model


# ── AP@0.5 (self-contained, COCO-style 101-point) ─────────────────────────────
def _ap_one_class(preds, gts_by_img, iou_thr):
    """preds: [(img_id, conf, box)] for ONE class across all images.
    gts_by_img: {img_id: [box,...]} for ONE class. Returns (ap, prec[], rec[])."""
    npos = sum(len(v) for v in gts_by_img.values())
    if npos == 0:
        return None
    preds = sorted(preds, key=lambda x: -x[1])
    matched = {img: set() for img in gts_by_img}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img, _conf, box) in enumerate(preds):
        gts = gts_by_img.get(img, [])
        best_v, best_j = iou_thr, -1
        seen = matched.get(img, set())
        for j, g in enumerate(gts):
            if j in seen:
                continue
            v = metrics.iou(box, g)
            if v >= best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            seen.add(best_j)
            matched.setdefault(img, seen)
            tp[i] = 1
        else:
            fp[i] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / npos
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101
    return ap, prec, rec


def _best_pr(preds, gts_by_img, iou_thr=0.5):
    """Precision/recall at the max-F1 operating point (Ultralytics-style headline)."""
    r = _ap_one_class(preds, gts_by_img, iou_thr)
    if r is None:
        return 0.0, 0.0
    _, prec, rec = r
    if len(prec) == 0:
        return 0.0, 0.0
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    return float(prec[k]), float(rec[k])


def ap_from_per_image(per_image, iou50_95=True) -> dict:
    """Model-agnostic detection AP on the TB positives from per-image dicts
    ({group, gts:[(cls,box)], preds:[(cls,conf,box)]}). Same output shape as
    metrics.detection_ap so it drops into the locked schema. Class-aware
    (Active=0, Obsolete=1). Works for torchvision OR (via metrics.collect) YOLO."""
    tb = [im for im in per_image if im["group"] == "tb"]
    thrs = list(np.arange(0.5, 0.96, 0.05)) if iou50_95 else [0.5]

    def cls_block(c):
        preds = [(idx, cf, bx) for idx, im in enumerate(tb)
                 for (pc, cf, bx) in im["preds"] if pc == c]
        gts_by_img = {idx: [bx for (gc, bx) in im["gts"] if gc == c]
                      for idx, im in enumerate(tb)}
        r50 = _ap_one_class(preds, gts_by_img, 0.5)
        if r50 is None:
            return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}, None
        ap50 = r50[0]
        aps = []
        for t in thrs:
            rr = _ap_one_class(preds, gts_by_img, float(t))
            aps.append(rr[0] if rr else 0.0)
        p, rec = _best_pr(preds, gts_by_img, 0.5)
        return {"mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(np.mean(aps)), 4),
                "precision": round(p, 4), "recall": round(rec, 4)}, ap50

    keys = class_keys()
    blocks, aps = {}, []
    for i, name in enumerate(keys):
        blk, ap = cls_block(i)
        blocks[name] = blk
        aps.append(ap)
    present = [a for a in aps if a is not None]
    overall_map50 = round(float(np.mean(present)), 4) if present else 0.0
    overall = {
        "mAP50": overall_map50,
        "mAP50_95": round(float(np.mean([b["mAP50_95"] for b in blocks.values()])), 4),
        "precision": round(float(np.mean([b["precision"] for b in blocks.values()])), 4),
        "recall": round(float(np.mean([b["recall"] for b in blocks.values()])), 4),
    }
    return {"overall": overall, **blocks}


# ── inference → per-image dicts (mirrors metrics.collect for torchvision) ──────
def infer_per_image(model, items, device, conf_floor=S.RECALL_CEILING_CONF,
                    batch: int = 1, single_class: bool | None = None) -> list[dict]:
    """Per-image {group, gts, preds} dicts for a torchvision detector.

    ⚠ `batch` DEFAULTS TO 1, AND REPORTED NUMBERS MUST KEEP IT THERE. Measured
    2026-08-02 on the sealed 360: batch=1 reproduces the recorded headline EXACTLY
    (0.7563 / 0.7475) while batch=8 gives 0.7562 / 0.7474.

    The 1e-4 is NOT a padding bug — that was tested directly and is dead. Every
    TBX11K image is 512x512, build_model pins min_size == max_size == 512, and
    512 % 32 == 0, so GeneralizedRCNNTransform rescales by 1.0 and pads 0 px; the
    per-image input tensors are bit-identical batched vs singleton. What moves is
    cuDNN's kernel choice at a different batch shape, worth ~1.3e-3 on a sigmoid
    score after 50 layers of fp32 accumulation.

    That is scientifically irrelevant against the +-0.025 bar, but "re-running the
    eval reproduces the recorded number exactly" is a property worth more than the
    ~3 s it costs on 360 images — the sealed eval is the reported result, and every
    cross-run comparison in RESULTS.md leans on it.

    ⚠ A 1.3e-3 score shift CAN flip a borderline box across a fixed conf threshold,
    so the screening counts at 0.10/0.25/0.50 are the fragile part, not AP. Those
    counts are already known to be noise-dominated (see the objective memo) — do not
    make it worse by batching the pass that produces them.

    Where batching IS the right call: the per-epoch validation (see _val_active_map50).
    It is a model-selection signal, never reported, and at 54 ms/image it was ~18% of
    every epoch. Also drop to 1 if a future dataset mixes image sizes, which would
    reintroduce the padding dependency this docstring rules out.
    """
    import torch
    from torchvision.io import read_image, ImageReadMode

    # Per-MODEL, not global: ap_compare scores single-class and 2-class checkpoints in
    # one pass, so the GT treatment has to follow the checkpoint being scored.
    merge_gt = SINGLE_CLASS if single_class is None else single_class
    model.eval()
    per_image = []
    with torch.no_grad():
        for i in range(0, len(items), max(1, batch)):
            chunk = items[i:i + max(1, batch)]
            xs, meta = [], []
            for img_path, label_path, group in chunk:
                img = read_image(str(img_path), ImageReadMode.RGB)
                _, h, w = img.shape
                xs.append((img.float() / 255.0).to(device))
                meta.append((label_path, group, w, h))
            outs = model(xs)
            for out, (label_path, group, w, h) in zip(outs, meta):
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                preds = [(int(l) - 1, float(s),
                          (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                         for b, s, l in zip(boxes, scores, labels) if s >= conf_floor]
                gts = metrics._gt_boxes(label_path, w, h)
                if merge_gt:
                    # The label files on disk stay 2-class; merge the EVAL ground
                    # truth too, or the 37 sealed Obsolete boxes silently leave the
                    # denominator and lesion recall is scored against 142 not 179.
                    # ⚠ Merge to 0 DIRECTLY — do not route this through _fg(), which
                    # reads the SINGLE_CLASS *global*. An external scorer (ap_compare)
                    # never sets that global, so _fg() returned c unchanged and this
                    # whole branch was a silent no-op that produced plausible-looking
                    # but wrong AP. merge_gt is already the decision; honour it here.
                    gts = [(0, b) for _c, b in gts]
                per_image.append({
                    "group": group,
                    "gts": gts,
                    "preds": preds,
                })
    return per_image


def evaluate_tv(model, split, items, device) -> dict:
    """The locked `metrics` sub-dict, torchvision-side. Detection AP from raw
    preds (ap_from_per_image); localisation/screening/by_size/confusion reuse the
    model-agnostic metrics.py passes verbatim."""
    per_image = infer_per_image(model, items, device)
    return {
        "detection": ap_from_per_image(per_image),
        "localization": metrics.localization(per_image, class_keys=class_keys()),
        "screening": metrics.screening(per_image, list(S.CONF_THRESHOLDS) + [S.RECALL_CEILING_CONF]),
        "by_size": metrics.by_size(per_image),
        "confusion_matrix": metrics.confusion_matrix(
            per_image, class_names=["Lesion"] if SINGLE_CLASS else None),
    }


# ── train ─────────────────────────────────────────────────────────────────────
def _val_active_map50(model, val_stems, device) -> float:
    """Per-epoch model-selection signal — NOT a reported number, so it takes the
    batched inference path (see infer_per_image: batch>1 costs ~1e-4 of AP to cuDNN
    kernel choice, and buys back ~18% of every epoch on 119 images). The sealed eval
    stays at batch=1 so the headline reproduces exactly."""
    items = [(splits.tb_image_path(s), splits.label_path(s), "tb") for s in val_stems]
    per_image = infer_per_image(model, items, device, batch=8)
    return ap_from_per_image(per_image, iou50_95=False)[headline_key()]["mAP50"]


def freeze_backbone_bn(model) -> int:
    """Put every backbone BatchNorm in eval mode: use the pretrained running stats,
    stop updating them. Returns how many layers were frozen.

    ⚠ WHY THIS EXISTS (bug found 2026-07-28). This module's `train()` docstring used
    to claim "the pretrained backbone uses FrozenBatchNorm ... so nothing here depends
    on batch statistics", and justified `--batch 2` with it. That is true of
    `retinanet_resnet50_fpn` (v1: 53 FrozenBatchNorm2d) but FALSE of the v2 models we
    actually build: `retinanet_resnet50_fpn_v2` has **53 real BatchNorm2d**. So the
    retinanet_s0 / fasterrcnn_s0 runs normalised on **2-sample batch statistics** and
    wrote those into the running estimates. `accum=8` does NOT help — accumulation
    fixes the gradient batch, but BN is computed per forward pass.

    Two consequences:
      * it is a plausible contributor to RetinaNet's val bouncing +-0.02 in a
        0.74-0.78 band (already noted in RESULTS.md as "best-by-val is partly luck");
      * it would WRECK the BYOL chain — SSL learns BN statistics at batch 64+, and a
        batch-2 fine-tune immediately overwrites them with 2-sample noise, which could
        erase the very transfer we are trying to measure.

    Freezing backbone BN at small batch is the standard fix (it is what v1 ships and
    what Detectron2 does by default). Kept behind a flag, defaulting OFF, so the
    existing recorded numbers stay reproducible.
    """
    import torch.nn as nn
    n = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
            n += 1
    return n


def recalibrate_backbone_bn(model, stems, imgsz, device, n_batches=50, batch=4) -> int:
    """Re-estimate backbone BN running statistics at the TARGET resolution.

    ⚠ WHY. BYOL pretrains at 256 (512 cannot finish a Kaggle session — ResNet-50 is
    ~18x the yolov8n backbone's FLOPs), but the detector fine-tunes at 512. BatchNorm
    running statistics are RESOLUTION-SENSITIVE: activation means/variances at 256 are
    not those at 512. Normally the fine-tune would slowly re-learn them — but combined
    with `freeze_bn` (which we need at batch 2) they would be FROZEN AT THE WRONG
    VALUES for the whole run, quietly costing exactly the transfer we are trying to
    measure. This closes that gap: forward-only passes at the target resolution, in
    train mode so BN updates its running estimates, under no_grad so nothing else
    moves. Then freeze. Standard practice ("BN re-estimation"), and cheap.

    It also repairs a SECOND, larger mismatch. kaggle/byol_pretrain.py pins "INPUT
    REGIME = raw [0,1], NO ImageNet mean/std" — decided for the yolov8n arm, because
    Ultralytics feeds its backbone [0,1]. That rationale did not survive the
    ResNet-50/torchvision pivot: GeneralizedRCNNTransform DOES apply ImageNet mean/std.
    conv1 is bias-free, so the shift is affine and a correct BN re-estimate absorbs it.

    ⚠ WHICH IS WHY THE FORWARD BELOW GOES THROUGH `model.transform` (fixed 2026-08-01).
    It used to call `model.backbone(x)` directly, which skips the transform — i.e.
    skips the very normalisation — so BN was calibrated on [0,1] while training feeds
    normalised pixels. Measured on 16 real TBX images: bn1 running_var 0.4237 vs 8.03
    (19x off, exactly 1/0.225^2) and FPN activations ~1000x too large (std 7096 vs 6.3,
    absmax 70784 — past fp16's 65504, and `use_amp` is on for every CUDA run). Using
    the model's own transform also means the mean/std constants cannot drift from it.

    ⚠⚠ This ran ONLY on the BYOL arm (it is gated on `backbone_weights` in train()),
    so `retinanet_byol_s0` carried the defect and `retinanet_cocoinit_fzbn_s0` did not
    — the matched pair was not matched on this axis. Both must be re-run to read the
    BYOL result.

    Only meaningful when a custom backbone is loaded AND freeze_bn is on.
    """
    import torch
    import torch.nn as nn
    bns = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        return 0
    saved_momentum = [m.momentum for m in bns]   # restored below — see the note at the end
    for m in bns:                      # reset so old-resolution stats do not linger
        m.reset_running_stats()
        m.momentum = None              # None => cumulative average over the pass
    ds = TBDetDataset(stems, train=False, aug="off")
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     num_workers=0, collate_fn=_collate)
    was_training = model.training
    model.train()
    seen = 0
    with torch.no_grad():
        for i, (imgs, _t) in enumerate(dl):
            if i >= n_batches:
                break
            x, _ = model.transform([im.to(device) for im in imgs])   # normalise + resize
            model.backbone(x.tensors)
            seen += len(imgs)
    model.train(was_training)
    # ⚠ RESTORE momentum. `momentum=None` means "cumulative average over everything seen
    # since the last reset", which is right for THIS pass and wrong for training: left in
    # place, every subsequent forward would be folded into the same running average with
    # weight 1/n, so the statistics would ossify as n grows. Harmless while freeze_bn is
    # on (BN sits in eval mode and never updates), which is why it went unnoticed — but
    # it silently changes the meaning of --bn-recalib WITHOUT --freeze-bn.
    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom
    print(f"[bn] recalibrated {len(bns)} backbone BatchNorm layers on {seen} images "
          f"@ imgsz={imgsz} through model.transform (ImageNet-normalised, as training "
          f"feeds it — see recalibrate_backbone_bn)")
    return seen


# ── divergence guards ────────────────────────────────────────────────────────
# WHY THESE EXIST (retinanet_cocovindr_s0, 2026-08-02). Its loss went non-finite in
# epoch 2 and the run then trained 31 more epochs, saved best.pt, ran the sealed eval
# and wrote a metrics.json reading "Active mAP50 = 0.005" — shaped exactly like a real
# cell of the 2x2 it belonged to. Nothing flagged it. Proof it learned nothing: the
# weight L2 of best.pt (ep2) and last.pt (ep32) differ in the SIXTH significant digit
# (490.247 vs 490.248) while every parameter stayed finite. That is the GradScaler
# death spiral: non-finite grads -> scaler.step() skipped -> scale halved -> repeat
# until the scale is subnormal, after which every gradient underflows to zero and the
# model is frozen while the loop reports it as "training". scaler.update() only grows
# the scale back after 2000 CONSECUTIVE clean steps, which never arrive.
#
# load_backbone_weights() already halts on a non-finite backbone. There was no matching
# guard on a non-finite LOSS, so the arm's cheapest failure mode was also its quietest.
# ⚠ CLIPPING IS OFF BY DEFAULT, AND THAT IS DELIBERATE — do not "turn it on for safety".
# Measured on a smoke run at norm 10.0: it bound on 7 of 16 optimizer steps. So it is
# NOT a no-op on healthy training — it changes the optimisation path. Every recorded
# number in this arm (0.7563 / 0.7475 / 0.5911 / 0.6857) was produced WITHOUT it, and
# the runs those numbers must be compared against are the ones being launched now.
# Enabling it by default would fix one asymmetry by introducing another, which is the
# exact mistake this whole session exists to undo.
#
# The NaN guard below is the real protection and it is safe to make unconditional,
# because it only ever acts on values that are already non-finite: a healthy run is
# bit-identical with or without it. Clipping PREVENTS divergence (a modelling choice);
# the guard DETECTS it (a safety property). Only the second belongs in a default.
# Turn clipping on deliberately, for a run that is diverging, via GRAD_CLIP_NORM=10.
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "0"))   # 0 = off
_NAN_EPOCH_FRAC = 0.25    # halt if >=25% of an epoch's batches are non-finite. Below
                          # this, skip the batch and continue: a one-off bad batch must
                          # not kill a 2-hour run, but a death spiral must not survive
                          # one epoch.


def train(arch, train_stems, val_stems, *, imgsz, epochs, batch, lr, device, aug,
          workers, out_dir, accum=8, patience=30, backbone_weights=None,
          freeze_bn=False, bn_recalib=0, trainable_layers=None,
          warmup_steps=0, amp_dtype="fp16") -> dict:
    """SGD + warmup + cosine LR, gradient accumulation to a real effective batch,
    best-by-val-Active-mAP50 with early stopping (like YOLO's patience).

    ⚠ Batch is NOT free here. The v2 backbone has real BatchNorm2d (53 layers), so at
    `--batch 2` every normalisation uses 2 samples and `accum` does not fix it — see
    freeze_backbone_bn(). Pass `freeze_bn=True` (or raise the batch) for any run whose
    backbone statistics matter, which includes every BYOL-initialised run."""
    import math
    import torch

    model = build_model(arch, imgsz, backbone_weights=backbone_weights,
                        trainable_layers=trainable_layers).to(device)
    # Order matters: recalibrate BEFORE freezing, or we freeze the stale stats.
    # ⚠ NOT gated on `backbone_weights` (fixed 2026-08-02). It used to be, which meant a
    # COCO-init run SILENTLY IGNORED --bn-recalib: the flag was accepted and discarded.
    # That made every COCO-vs-BYOL pair unmatched on BN adaptation, and it is fatal at
    # --trainable-layers 0, where the trunk is frozen so BN re-estimation is the ONLY way
    # the trunk can adapt to the target domain — the control was denied the sole
    # adaptation mechanism available to it while the BYOL arm received it. Proven from
    # the run logs: byol_tl0_s0 printed "[bn] recalibrated 53 ...", cocoinit_tl0_s0 did
    # not. If the stats are about to be FROZEN for a whole run they must be estimated on
    # the data about to be trained on, whatever the init, or the control is handicapped.
    bn_recalib_ran = 0
    if bn_recalib:
        bn_recalib_ran = recalibrate_backbone_bn(model, train_stems, imgsz, device,
                                                 n_batches=bn_recalib)
    n_bn = freeze_backbone_bn(model) if freeze_bn else 0
    if freeze_bn:
        print(f"[bn] froze {n_bn} backbone BatchNorm2d layers (running stats kept, "
              f"not updated) — required at batch {batch}")
    # Report what will actually receive gradients. A freeze arm that silently trained
    # everything would be indistinguishable from the control in the results table.
    body_tr = sum(p.numel() for p in model.backbone.body.parameters() if p.requires_grad)
    body_fz = sum(p.numel() for p in model.backbone.body.parameters() if not p.requires_grad)
    # FULLY frozen only. With freeze_bn on, every stage is *partly* frozen (its BN affine),
    # which would make the ladder rung unreadable from the log if we listed partials.
    _stage = {}
    for n, p in model.backbone.body.named_parameters():
        s = n.split(".")[0]
        _stage[s] = _stage.get(s, True) and not p.requires_grad
    frozen_stages = sorted(s for s, allfz in _stage.items() if allfz)
    print(f"[freeze] trainable_layers={trainable_layers if trainable_layers is not None else 'default(3)'}"
          f"  body trainable={body_tr:,} frozen={body_fz:,} "
          f"({body_tr / max(1, body_tr + body_fz):.1%} trainable)"
          f"  fully frozen stages: {frozen_stages or 'none'}")
    ds_tr = TBDetDataset(train_stems, train=True, aug=aug)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True,
                                        num_workers=workers, collate_fn=_collate,
                                        pin_memory=(device.type == "cuda"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    # ⚠ AMP DTYPE IS LOAD-BEARING ON THIS ARM — see the mosaic NaN diagnosis
    # (2026-08-05). fp16 tops out at 65504, and with --freeze-bn the trunk has NO
    # renormalisation left (all 53 body BatchNorms are eval-mode with frozen affine),
    # so any conv-weight growth in layer3/layer4 propagates multiplicatively into the
    # FPN and heads. Measured on the crashed run's ep2 checkpoint: peak activation
    # 107062 = 163% of the fp16 ceiling while the SAME forward in fp32 gives a healthy
    # loss of 1.37. That inf is what the NaN guard was catching.
    # bf16 has fp32's exponent range (~3.4e38), so the cliff disappears; it needs no
    # GradScaler. Ampere (the 3050) supports it natively. Default stays fp16 so every
    # recorded run reproduces.
    use_amp = device.type == "cuda" and amp_dtype != "off"
    _AMP_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
    amp_torch_dtype = _AMP_DT.get(amp_dtype, torch.float16)
    # GradScaler exists to keep fp16 GRADIENTS off the underflow floor. bf16 gradients
    # have fp32 range, so scaling is unnecessary — and an enabled scaler would add the
    # skip-step/halve-scale death-spiral path for no benefit.
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_torch_dtype is torch.float16))
    if device.type == "cuda":
        print(f"[amp] autocast={amp_dtype}"
              + ("" if amp_dtype == "off" else
                 f" (grad_scaler={'on' if scaler.is_enabled() else 'off'})"))

    steps = len(dl_tr)
    # ⚠⚠ WARMUP IS COUNTED IN OPTIMIZER STEPS, NOT DATALOADER ITERATIONS.
    # The old line was `warm_iters = min(500, steps)` compared against `gstep`, which
    # ticks once per MICRO-batch. `steps` is micro-batches per epoch (70 at batch 8,
    # 280 at batch 2), so the min() never selected 500 and the warmup was always
    # exactly one epoch = steps/accum = **35 optimizer steps**, at both batch sizes.
    # That formula is torchvision's reference recipe (`min(1000, len(data_loader)-1)`),
    # but it is a COCO-SIZED-EPOCH heuristic: on 118k images one epoch is thousands of
    # steps, on our 559 it is 35. So we were reaching lr=0.01 at effective batch 16 in
    # 35 steps where detectron2's RetinaNet R50-FPN recipe — same lr, same effective
    # batch — takes 1000. That is the divergence trigger: `geo` survives it, mosaic
    # does not, and which runs die is stochastic because this arm never seeds.
    # warmup_steps=0 keeps the legacy behaviour so the existing board reproduces.
    warm_iters = (warmup_steps * accum) if warmup_steps else (min(500, steps) if steps > 1 else 0)
    warm_opt_steps = warm_iters // max(1, accum)
    print(f"[sched] lr={lr} eff_batch={batch * accum}  warmup={warm_opt_steps} optimizer "
          f"steps (~{warm_opt_steps * accum / max(1, steps):.1f} epochs)"
          + ("  [legacy: one epoch of micro-batches]" if not warmup_steps else ""))
    best_map, best_ep, since = -1.0, -1, 0
    best_pt, last_pt = out_dir / "best.pt", out_dir / "last.pt"
    eff_batch = batch * accum
    gstep = 0
    # ⚠ The learning CURVE, not just its argmax. Added 2026-08-02 after the
    # cocoinit/byol pair finished: the per-epoch line was printed to stdout and
    # NOWHERE else, so when the box rebooted the only surviving record of either
    # run was best_epoch. That erased the one question the pair raised — whether
    # the BYOL arm starts ahead and then stagnates (representation washed out by
    # fine-tuning 98.8% of the trunk on 559 images) or simply never leads.
    # Appended per epoch rather than written at the end, so a run that is killed,
    # OOMs, or hits a reboot still leaves everything it had measured.
    history, hist_path = [], out_dir / "history.jsonl"
    hist_path.unlink(missing_ok=True)
    nan_batches = 0                       # cumulative, reported in metrics.json
    clipped_steps, opt_steps = 0, 0       # did GRAD_CLIP_NORM ever bind? see below
    empty_groups = 0                      # accumulation groups that lost EVERY batch to
                                          # the NaN guard, so no step was taken
    micro = 0                             # finite batches accumulated in the open group
    for ep in range(epochs):
        cos_lr = 0.5 * lr * (1 + math.cos(math.pi * ep / max(1, epochs)))   # cosine per epoch
        model.train()
        if freeze_bn:
            freeze_backbone_bn(model)   # model.train() re-enables BN — re-freeze every epoch
        opt.zero_grad()
        run_loss, nb, t0 = 0.0, 0, time.time()
        n_nan = 0
        for it, (imgs, targets) in enumerate(dl_tr):
            cur_lr = lr * (0.01 + 0.99 * gstep / max(1, warm_iters)) if gstep < warm_iters else cos_lr
            for g in opt.param_groups:
                g["lr"] = cur_lr
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_torch_dtype):
                loss = sum(model(imgs, targets).values()) / accum
            # ⚠ NaN GUARD — see the block comment above `_NAN_EPOCH_FRAC`.
            # ⚠⚠ THE SKIP MUST NOT SKIP THE ACCUMULATION BOUNDARY (fixed 2026-08-04).
            # This used to be `continue`, which jumped past the whole block below —
            # including `opt.zero_grad()`. So whenever a non-finite batch happened to
            # land ON a boundary, the group's partial gradient was never cleared: the
            # next group accumulated on TOP of it and stepped with ~2x the gradient
            # while still dividing by `accum`. That is a positive feedback loop —
            # a skip inflates the next step, the inflated step causes more skips — and
            # it is what turned the mosaic runs' first NaN into 20.7% -> 100% of an
            # epoch in one epoch. Now: never backward a poisoned batch, but ALWAYS
            # close the group.
            # ✅ Provably a no-op for every number already on the board: the branch is
            # dead unless a batch is non-finite, and all 13 aug='geo' runs recorded
            # nan_batches_skipped == 0. Healthy runs take the identical path.
            lv = float(loss.item())
            finite = math.isfinite(lv)
            if finite:
                scaler.scale(loss).backward()
                micro += 1                    # this group has a real gradient in it
            else:
                n_nan += 1
                nan_batches += 1              # no backward: keep the poison out of .grad
            if (it + 1) % accum == 0 or (it + 1) == steps:         # step every `accum` batches
                # Clip on UNSCALED grads. Without this a single exploding batch produces
                # inf grads, GradScaler skips the step and halves the scale, and if that
                # recurs the scale decays to subnormal — every later gradient underflows
                # to zero and the model is frozen while training "continues". That is
                # exactly how retinanet_cocovindr_s0 burned 31 epochs (2026-08-02).
                if micro:                     # nothing accumulated => nothing to step on
                    if GRAD_CLIP_NORM > 0:
                        # Count how often the clip actually BINDS, so "was this run
                        # comparable to the earlier arm?" is answerable from metrics.json
                        # instead of assumed. See the note on GRAD_CLIP_NORM.
                        scaler.unscale_(opt)
                        total_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                        if float(total_norm) > GRAD_CLIP_NORM:
                            clipped_steps += 1
                    scaler.step(opt)
                    scaler.update()
                    opt_steps += 1
                else:
                    empty_groups += 1
                opt.zero_grad()               # ALWAYS: never carry a partial group over
                micro = 0
            if finite:
                run_loss += lv * accum
                nb += 1
            gstep += 1
        if n_nan:
            frac = n_nan / max(1, steps)
            print(f"[nan] ⚠ {n_nan}/{steps} batches ({frac:.1%}) had a non-finite loss "
                  f"this epoch and were SKIPPED.", flush=True)
            if frac >= _NAN_EPOCH_FRAC:
                raise SystemExit(
                    f"HALT: {frac:.1%} of epoch {ep+1}'s batches produced a non-finite "
                    f"loss (threshold {_NAN_EPOCH_FRAC:.0%}). Training has diverged — "
                    f"refusing to burn GPU hours on a model that cannot learn, and "
                    f"refusing to emit a metrics.json that would look like a real result. "
                    f"Check the backbone init's BN statistics, then lower --lr.")
        vmap = _val_active_map50(model, val_stems, device)
        torch.save({"arch": arch, "imgsz": imgsz, "single_class": SINGLE_CLASS,
                    "num_fg_classes": 1 if SINGLE_CLASS else S.NUM_CLASSES,
                    "state_dict": model.state_dict()},last_pt)
        flag = ""
        if vmap > best_map:
            best_map, best_ep, since = vmap, ep, 0
            torch.save({"arch": arch, "imgsz": imgsz, "single_class": SINGLE_CLASS,
                    "num_fg_classes": 1 if SINGLE_CLASS else S.NUM_CLASSES,
                    "state_dict": model.state_dict()},best_pt)
            flag = "  ← best"
        else:
            since += 1
        row = {"epoch": ep + 1, "loss": round(run_loss / max(1, nb), 5),
               "val_active_map50": round(float(vmap), 5),
               "lr": float(opt.param_groups[0]["lr"]),
               "sec": round(time.time() - t0, 1), "best": bool(flag)}
        history.append(row)
        with hist_path.open("a") as fh:                 # append: survives a kill
            fh.write(json.dumps(row) + "\n")
        print(f"[ep {ep+1:>3}/{epochs}] loss={run_loss/max(1,nb):.4f} "
              f"val_active_mAP50={vmap:.4f}  lr={opt.param_groups[0]['lr']:.2e} "
              f"eff_batch={eff_batch}  {time.time()-t0:.0f}s{flag}", flush=True)
        if patience and since >= patience:
            print(f"[early-stop] no val improvement for {patience} epochs "
                  f"(best {best_map:.4f} @ ep{best_ep+1})", flush=True)
            break

    return {"best_pt": best_pt, "last_pt": last_pt, "best_val_map50": best_map,
            "best_epoch": best_ep + 1, "epochs_ran": ep + 1, "epochs_cap": epochs,
            "history": history, "history_file": str(hist_path),
            "batch": batch, "accum": accum, "eff_batch": eff_batch, "lr": lr,
            "backbone_init": getattr(model, "_backbone_init", None),
            "backbone_bn_frozen": n_bn,
            # ⚠ PERSIST THE BN KNOB. It was previously recorded NOWHERE — the only trace
            # that recalibration ran was a stdout line that is simply absent when it
            # does not, so "did the control get the same treatment?" was unanswerable
            # from metrics.json. That is how the COCO/BYOL pairs stayed unmatched for a
            # day. `bn_recalib_images` is what ACTUALLY happened, not what was asked for.
            "bn_recalib_requested": bn_recalib, "bn_recalib_images": bn_recalib_ran,
            "grad_clip_norm": GRAD_CLIP_NORM, "nan_batches_skipped": nan_batches,
            "grad_clipped_steps": clipped_steps, "optimizer_steps": opt_steps,
            "empty_accum_groups": empty_groups,
            # In OPTIMIZER steps. The legacy value is 35 at every batch size — see the
            # warm_iters note. Recorded so a re-run's schedule is auditable from
            # metrics.json instead of re-derived from batch/accum.
            "warmup_opt_steps": warm_opt_steps, "amp_dtype": amp_dtype,
            "body_trainable_params": body_tr, "body_frozen_params": body_fz}


def _ckpt_num_fg_classes(ckpt, probe) -> int:
    """Foreground-class count a checkpoint was TRAINED with, read off the head.

    Needed because build_model otherwise follows the SINGLE_CLASS module flag, which
    is False in any external scorer (ap_compare imports this module without calling
    run()). A single-class checkpoint would then be loaded into a 2-class head and
    die on a size mismatch. Reading the shape also works for the four single-class
    runs that were saved before `single_class` was recorded in the checkpoint."""
    sd = ckpt["state_dict"]
    if "head.classification_head.cls_logits.weight" in sd:              # retinanet
        n_anchors = probe.head.classification_head.num_anchors
        return sd["head.classification_head.cls_logits.weight"].shape[0] // n_anchors - 1
    if "roi_heads.box_predictor.cls_score.weight" in sd:                # fasterrcnn
        return sd["roi_heads.box_predictor.cls_score.weight"].shape[0] - 1
    raise KeyError("cannot infer class count — unrecognised detector head")


def load_model(best_pt: Path, imgsz: int, device):
    import torch
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt.get("imgsz", imgsz), pretrained=False)
    n_fg = ckpt.get("num_fg_classes") or _ckpt_num_fg_classes(ckpt, model)
    if n_fg != S.NUM_CLASSES:
        model = build_model(ckpt["arch"], ckpt.get("imgsz", imgsz), pretrained=False,
                            num_fg_classes=n_fg)
    model.load_state_dict(ckpt["state_dict"])
    # Read by per_image_for so a single-class model's GROUND TRUTH is merged too —
    # without it the 37 Obsolete boxes stay class 1, unmatchable by a model that only
    # emits class 0, and recall is silently scored against 142 boxes instead of 179.
    model._single_class = (n_fg == 1)
    return model.to(device)


# ── environment / report ──────────────────────────────────────────────────────
def _environment() -> dict:
    info = {"timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        import torch, torchvision
        info["torch"] = torch.__version__
        info["torchvision"] = torchvision.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["torch"] = info["torchvision"] = info["gpu"] = "?"
    return info


def _assert_labels_present(stems: list[str]) -> None:
    """HALT if the YOLO label files are missing — training on zero boxes is SILENT.

    ⚠ This is the failure that cost a Kaggle session on 2026-08-06. A positives-only
    detector whose targets are all empty does not crash and does not trip the NaN
    guard: focal loss with no positives is simply tiny. It reads as
    `loss=0.0336 → 0.0014, val_active_mAP50=0.0000` and runs the full 120 epochs to
    a complete, entirely worthless metrics.json. A healthy run starts near loss 1.1.

    The cause was `splits.build_or_load()` returning early on a machine given a
    pre-built splits.json, so `convert.build_or_load()` never ran. That is fixed at
    the source; this stays as the tripwire, because "no boxes" has many possible
    causes (wrong TB_GEN_ROOT, a half-copied dataset) and only one symptom.
    """
    missing = [s for s in stems if not splits.label_path(s).exists()]
    if not missing:
        return
    raise SystemExit(
        f"HALT: {len(missing)}/{len(stems)} training images have NO label file.\n"
        f"  first missing: {splits.label_path(missing[0])}\n"
        f"Training would run to completion on empty targets — near-zero loss, "
        f"mAP 0.000, no crash — and write a metrics.json that looks like a result.\n"
        f"Labels live in TB_GEN_ROOT/labels_all/ and are generated from the COCO "
        f"JSON. Check TB_GEN_ROOT is writable and TB_DATA_ROOT points at the "
        f"TBX11K root, then re-run: python -c \"import sys;sys.path.insert(0,'.');"
        f"from yolo_common import convert;print(convert.build_or_load()['n_images'])\"")


# ── which images to train on / evaluate against ───────────────────────────────
def _resolve_split(args, smoke: bool):
    """(split, train_stems, val_stems, eval_items, fold_meta) for this run.

    Two modes, and they answer different questions:

    * DEFAULT — the frozen `splits.json` and its sealed 360 blackbox (121 tb +
      120 sick + 120 healthy). One fixed test set; the negatives are present, so
      the screening / false-alarm block is meaningful.
    * `--fold i` — one outer fold of the rotating k-way partition built by
      `yolo_common/kfold.py`. Across the k runs every one of the ~799 positives
      is tested exactly once, so the test set is the whole positive set instead
      of 121 images.

    ⚠ Fold test sets are POSITIVES-ONLY (kfold.py:170), so under `--fold` the
      screening rows are all-zero by construction, not by merit. False-alarm
      numbers only exist on the sealed 360 — never quote them from a fold run.
    ⚠ This does NOT touch splits.json. The folds are a separate partition with
      their own frozen folds.json; the frozen split stays byte-identical.
    """
    if getattr(args, "fold", None) is None:
        split = splits.build_or_load()
        train_stems = list(split["train_ids"])
        val_stems = list(split["val_ids"])
        items = splits.blackbox(split)
        if smoke:
            train_stems = train_stems[:S.SMOKE_SUBSET]
            val_stems = val_stems[:4]
            items = ([it for it in items if it[2] == "tb"][:6]
                     + [it for it in items if it[2] == "sick"][:4]
                     + [it for it in items if it[2] == "healthy"][:4])
        return split, train_stems, val_stems, items, None

    from yolo_common import kfold
    k = args.k
    if not (0 <= args.fold < k):
        raise SystemExit(f"--fold must be in [0, {k}); got {args.fold}")
    folds_data = kfold.build_or_load_folds(k, S.SEED)
    part = kfold.fold_partition(folds_data, args.fold, seed=S.SEED,
                                inner_val_frac=args.inner_val_frac, smoke=smoke)
    # split_like carries test_positive_ids only — that is all evaluate_tv() and
    # the AP pass need. No YOLO tree is materialised: the torchvision Dataset
    # takes stem lists directly, so materialise_fold() is the YOLO path's problem.
    split, items = kfold.fold_eval_inputs(part)
    meta = {"fold": args.fold, "k": k, "folds_file": str(kfold.FOLDS_JSON),
            "fold_seed": S.SEED, "inner_val_frac": args.inner_val_frac,
            "fold_sizes": folds_data["fold_sizes"],
            "n_positives_all_folds": folds_data["n_positives"],
            "eval_scope": "rotating test fold, POSITIVES-ONLY (screening rows are "
                          "zero by construction — use the sealed 360 for those)"}
    return split, part["train"], part["inner_val"], items, meta


# ── top-level run (called by the two thin experiment scripts) ─────────────────
def run(arch: str, args) -> dict:
    global SINGLE_CLASS
    SINGLE_CLASS = bool(getattr(args, "single_class", False))
    device = resolve_device(args.device)
    # ⚠ the fold MUST be in the default tag: without it all k folds default to
    # the same name and each one silently overwrites the last (this module's
    # scripts have already lost a recorded run that way — see retinanet.py).
    fold_sfx = "" if getattr(args, "fold", None) is None else f"_fold{args.fold}"
    tag = args.name or f"{arch}{fold_sfx}_s{S.SEED}"
    smoke = S.SMOKE

    print(f"\n████ {tag}  (arch={arch}, imgsz={args.imgsz}, batch={args.batch}, "
          f"epochs={args.epochs}, aug={args.aug}, lr={args.lr}, device={device}, "
          f"smoke={smoke}, single_class={SINGLE_CLASS}) ████\n")
    if SINGLE_CLASS:
        print("  [single-class] Active+Obsolete merged → 'Lesion'; head has 1 foreground "
              "class; eval GT merged too (179 boxes, not 142). Reports under key 'lesion'.\n")

    split, train_stems, val_stems, items, fold_meta = _resolve_split(args, smoke)
    _assert_labels_present(train_stems)
    if fold_meta:
        print(f"  [kfold] fold {fold_meta['fold']}/{fold_meta['k'] - 1}  "
              f"sizes={fold_meta['fold_sizes']} (Σ={fold_meta['n_positives_all_folds']} "
              f"positives, each tested exactly once across the {fold_meta['k']} runs)")
        print("  [kfold] test fold is POSITIVES-ONLY → screening/false-alarm rows "
              "are zero by construction, not by merit.")
    print(f"  train_positives={len(train_stems)} val_positives={len(val_stems)} "
          f"{'fold_test_items' if fold_meta else 'sealed_eval_items'}={len(items)}\n")

    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"══ train {arch} ══")
    epochs = S.SMOKE_EPOCHS if smoke else args.epochs
    accum = 1 if smoke else args.accum
    patience = 0 if smoke else args.patience
    tr = train(arch, train_stems, val_stems, imgsz=args.imgsz, epochs=epochs,
               batch=args.batch, lr=args.lr, device=device, aug=args.aug,
               workers=args.workers, out_dir=out_dir, accum=accum, patience=patience,
               backbone_weights=getattr(args, "backbone_weights", None),
               freeze_bn=getattr(args, "freeze_bn", False),
               bn_recalib=getattr(args, "bn_recalib", 0),
               trainable_layers=getattr(args, "trainable_layers", None),
               warmup_steps=getattr(args, "warmup_steps", 0),
               amp_dtype=getattr(args, "amp_dtype", "fp16"))
    print(f"  best.pt → {tr['best_pt']}  (best val Active mAP50 {tr['best_val_map50']:.4f} "
          f"@ ep{tr['best_epoch']}; ran {tr['epochs_ran']}/{tr['epochs_cap']}, "
          f"eff_batch={tr['eff_batch']})\n")

    print(f"══ evaluate on {'the rotating test fold' if fold_meta else 'the sealed 360'} ══")
    t0 = time.time()
    model = load_model(tr["best_pt"], args.imgsz, device)
    metric_block = evaluate_tv(model, split, items, device)
    eval_sec = round(time.time() - t0, 1)

    report = {
        "experiment": tag,
        "config": {"model": arch, "imgsz": args.imgsz, "seed": S.SEED,
                   "arm": "torchvision-detector-posonly", "arch": arch,
                   "epochs_cap": epochs, "epochs_ran": tr["epochs_ran"],
                   "best_epoch": tr["best_epoch"], "patience": patience,
                   "batch": args.batch, "accum": accum, "eff_batch": tr["eff_batch"],
                   "lr": args.lr, "aug": args.aug,
                   # True => detection block is keyed "lesion" and scored on 179
                   # merged GT boxes. NEVER put such a run's mAP50 on the Active board.
                   "single_class": SINGLE_CLASS,
                   "augmentation": args.aug != "off", "amp": device.type == "cuda",
                   # ⚠ Both added 2026-08-05 with legacy defaults. A run WITHOUT these
                   # keys is fp16 + a 35-optimizer-step warmup; do not assume otherwise.
                   "amp_dtype": tr.get("amp_dtype"),
                   "warmup_opt_steps": tr.get("warmup_opt_steps"),
                   "pretrained": "COCO_V1", "min_max_size": args.imgsz,
                   # Provenance for the SSL arm: which backbone, and whether its BN
                   # statistics survived the fine-tune (they do NOT at batch 2 unless
                   # freeze_bn is on — see freeze_backbone_bn()).
                   "backbone_init": (tr.get("backbone_init") or "COCO_V1"),
                   "freeze_bn": getattr(args, "freeze_bn", False),
                   # See the note on the same keys in train()'s return: without these,
                   # "was the control treated identically?" cannot be answered from a
                   # results folder, only from a stdout line that is absent on the arm
                   # that skipped it. bn_recalib_images == 0 means it did NOT run.
                   "bn_recalib_requested": tr.get("bn_recalib_requested"),
                   "bn_recalib_images": tr.get("bn_recalib_images"),
                   "grad_clip_norm": tr.get("grad_clip_norm"),
                   "grad_clipped_steps": tr.get("grad_clipped_steps"),
                   "optimizer_steps": tr.get("optimizer_steps"),
                   "nan_batches_skipped": tr.get("nan_batches_skipped"),
                   # >0 means a whole accumulation group was lost to the NaN guard, so
                   # that step never happened. Reads as 0 on every healthy run.
                   "empty_accum_groups": tr.get("empty_accum_groups"),
                   # None == torchvision default (3 with pretrained weights). Recorded
                   # explicitly so a freeze-ladder row can never be confused with the
                   # default; retinanet_s0/fasterrcnn_s0 predate the flag and were at 3.
                   "trainable_layers": getattr(args, "trainable_layers", None),
                   "body_trainable_params": tr.get("body_trainable_params"),
                   "body_frozen_params": tr.get("body_frozen_params"),
                   "negatives_in_training": False,
                   "ap_note": "self-contained COCO-style AP@0.5 (see tv_detect.py); "
                              "re-score YOLO via rescore_yolo() for identical-code compare",
                   "conf_thresholds": S.CONF_THRESHOLDS},
        "dataset": {"train_positives": len(train_stems), "val_positives": len(val_stems),
                    # ⚠ Exactly one of `blackbox` / `kfold` is present, and which one
                    # says what the headline AP was scored on. `kfold` => the rotating
                    # POSITIVES-ONLY test fold, so the screening block is meaningless
                    # there; aggregate those runs with exp5_kfold.py --aggregate, and
                    # never table a fold number beside a sealed-360 number.
                    **({"kfold": fold_meta} if fold_meta else
                       {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                        "blackbox": {"tb": split["counts"]["test_positives"],
                                     "sick_non_tb": split["counts"]["blackbox_sick"],
                                     "healthy": split["counts"]["blackbox_healthy"]}}),
                    "class_map": ({"0": "Lesion (Active+Obsolete merged)"} if SINGLE_CLASS
                                  else {"0": "ActiveTuberculosis",
                                        "1": "ObsoletePulmonaryTuberculosis"})},
        "environment": _environment(),
        # The learning curve, so a finished run can be re-read without its stdout.
        # `history` duplicates history.jsonl on purpose: the jsonl survives a killed
        # run, this survives a deleted scratch dir.
        "training": {f"best_val_{headline_key()}_map50": round(float(tr["best_val_map50"]), 5),
                     "best_epoch": tr["best_epoch"], "epochs_ran": tr["epochs_ran"],
                     "history": tr.get("history", [])},
        "metrics": metric_block,
        "timing": {"train_sec": None, "eval_sec": eval_sec},
        "artifacts": {"weights": str(tr["best_pt"])},
        "reference": {"yolo_geo_active_mAP50": 0.685,
                      "yolo_champion_mosaic_mixup_active_mAP50": 0.745,
                      "note": "compare to YOLO geo (0.685) at aug=geo — matched recipe; "
                              "0.745 champion used mosaic_mixup (a separate arm). Both are "
                              "Ultralytics AP; small AP-impl gap vs this AP code expected"},
    }
    metrics.save_confusion_png(metric_block["confusion_matrix"],
                               out_dir / "confusion_matrix_test.png")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.txt").write_text(metrics.summary_text(report))
    print(f"\n→ {out_dir}/metrics.json\n→ {out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())
    ours = metric_block["detection"][headline_key()]["mAP50"]
    if fold_meta:
        # Every board anchor below is a sealed-360 number. A fold is a DIFFERENT
        # test set (and a different size), so printing them side by side would
        # invite exactly the comparison that is invalid. One fold alone means
        # little anyway — the reportable quantity is the mean±std over all k.
        print(f"\nFOLD {fold_meta['fold']}/{fold_meta['k'] - 1} ({arch}, aug={args.aug}, "
              f"{'SINGLE-CLASS' if SINGLE_CLASS else '2-class'}): "
              f"{headline_key()} mAP50 = {ours:.3f} on {len(items)} held-out positives.  "
              f"⚠ NOT comparable to any sealed-360 cell on the board — different test "
              f"set. Run all {fold_meta['k']} folds and report mean±std.")
    elif SINGLE_CLASS:
        # ⚠ NOT comparable to any Active mAP50 on the board — different GT (179
        # lesion boxes vs 142 Active). The anchors are the 2-class models scored
        # class-agnostically on the same 179 (ap_compare_screen_v2 preds, NMS@0.65).
        print(f"\nCOMPARE ({arch}, aug={args.aug}, SINGLE-CLASS): Lesion mAP50 = {ours:.3f}  "
              f"vs 2-class models merged post-hoc: rn_cocoinit 0.810 · rn_mosaicmix 0.841 · "
              f"rn_cocovindr 0.851 · rn_byolvindr 0.846.  "
              f"⚠ Do NOT compare this to Active mAP50 (0.758 etc) — different denominator.")
    else:
        anchor = "YOLO geo 0.685" if args.aug == "geo" else "YOLO champion(mosaic_mixup) 0.745"
        print(f"\nCOMPARE ({arch}, aug={args.aug}): Active mAP50 = {ours:.3f} (this AP code)  vs  "
              f"{anchor} (Ultralytics AP).  [champion 0.745 used mosaic_mixup = separate arm]")
    return report


def add_common_args(ap):
    """Shared CLI for the two thin scripts."""
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=120, help="CAP — early-stops on val (see --patience)")
    ap.add_argument("--patience", type=int, default=30, help="stop if val Active mAP50 flat for N epochs (0=off)")
    ap.add_argument("--batch", type=int, default=2, help="6 GB fits ~2 for ResNet50-FPN@512; drop to 1 on OOM")
    ap.add_argument("--accum", type=int, default=8, help="grad-accum steps → effective batch = batch×accum (default 16)")
    ap.add_argument("--lr", type=float, default=0.01, help="SGD lr for the effective batch (warmup+cosine)")
    ap.add_argument("--aug", choices=["geo", "hflip", "off", "mosaic", "mosaic_mixup"],
                    default="geo",
                    help="geo = matches YOLO's _GEO level (comparable to our YOLO geo runs). "
                         "mosaic = geo + a 4-image composite built in the Dataset; "
                         "mosaic_mixup adds a p=0.1 double exposure — the recipe that beat "
                         "geo by +0.052 (COCO units) on the YOLO arm. See the mosaic block "
                         "in tv_detect.py: same shape as Ultralytics', NOT a port, and with "
                         "no close_mosaic.")
    ap.add_argument("--single-class", action="store_true",
                    default=os.environ.get("SINGLE_CLASS", "") == "1",
                    help="LESION-FINDER mode: merge ActiveTuberculosis + "
                         "ObsoletePulmonaryTuberculosis into one 'Lesion' class, in the "
                         "training targets AND the eval GT. The head gets 1 foreground "
                         "class instead of 2. For the detector's role as stage 2 of a "
                         "detector->crop-classifier pipeline, where active-vs-obsolete is "
                         "someone else's job. ⚠ Reports under the key 'lesion', and its "
                         "mAP50 is scored on 179 GT boxes — NOT comparable to the 142-box "
                         "Active mAP50 on the board. Env: SINGLE_CLASS=1")
    ap.add_argument("--warmup-steps", type=int,
                    default=int(os.environ.get("WARMUP_STEPS", "0")),
                    help="linear LR warmup length in OPTIMIZER steps (not micro-batches). "
                         "0 = legacy: one epoch of micro-batches, which on 559 images is "
                         "only 35 optimizer steps at any batch size — detectron2's "
                         "RetinaNet R50-FPN uses 1000 at the same lr and effective batch. "
                         "That 35-step ramp is the mosaic-NaN divergence trigger "
                         "(2026-08-05); 500 is the recommended value. Env: WARMUP_STEPS")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16", "off"],
                    default=os.environ.get("AMP_DTYPE", "fp16"),
                    help="autocast dtype. fp16 (default, what every recorded run used) "
                         "overflows at 65504 — with --freeze-bn the trunk cannot "
                         "renormalise, and the crashed mosaic runs peaked at 107062 in "
                         "fp32 while their fp32 LOSS was healthy. bf16 has fp32's exponent "
                         "range and needs no GradScaler. Env: AMP_DTYPE")
    ap.add_argument("--fold", type=int, default=(int(os.environ["KFOLD_FOLD"])
                                                 if os.environ.get("KFOLD_FOLD") else None),
                    help="run one outer fold of the rotating k-way CV over ALL ~799 "
                         "positives (yolo_common/kfold.py) instead of the frozen split. "
                         "Across the k folds every positive is tested exactly once, so "
                         "the test set is 799 images rather than the sealed 121 — that is "
                         "the point of it. ⚠ Fold test sets are POSITIVES-ONLY, so "
                         "screening/false-alarm rows are zero by construction and only "
                         "the sealed 360 can produce them. ⚠ A fold number is NOT "
                         "comparable to a sealed-360 cell. Env: KFOLD_FOLD")
    ap.add_argument("--k", type=int, default=int(os.environ.get("KFOLD_K", "5")),
                    help="folds in the partition. Must match across the k runs — "
                         "kfold.py halts if an existing folds.json disagrees on (seed, k). "
                         "Env: KFOLD_K")
    ap.add_argument("--inner-val-frac", type=float,
                    default=float(os.environ.get("KFOLD_INNER_VAL_FRAC", "0.15")),
                    help="stratified slice of the k-1 training folds held out to select "
                         "best.pt. Selection only — never a reported number. "
                         "Env: KFOLD_INNER_VAL_FRAC")
    ap.add_argument("--device", default=None, help="0 | cpu (default settings.DEVICE)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None, help="run/output name (default <arch>_s<seed>)")
    ap.add_argument("--backbone-weights", default=os.environ.get("BACKBONE_WEIGHTS") or None,
                    help="BYOL-pretrained ResNet-50 trunk from kaggle/byol_pretrain.py "
                         "(ARCH=resnet50). FPN + heads stay COCO. Env: BACKBONE_WEIGHTS")
    ap.add_argument("--freeze-bn", action="store_true",
                    default=os.environ.get("FREEZE_BN", "") == "1",
                    help="put backbone BatchNorm in eval mode (keep running stats, stop "
                         "updating them). STRONGLY recommended at --batch 2: the v2 backbone "
                         "has 53 REAL BatchNorm2d and accum does not fix batch stats. "
                         "Off by default so the recorded s0 runs stay reproducible. Env: FREEZE_BN=1")
    ap.add_argument("--bn-recalib", type=int, default=int(os.environ.get("BN_RECALIB", "0")),
                    help="N forward-only batches to re-estimate backbone BN stats at THIS "
                         "imgsz before freezing. Needed when the backbone was BYOL-pretrained "
                         "at a different resolution (256) than the detector trains at (512) — "
                         "BN stats are resolution-sensitive. 50 is plenty. Env: BN_RECALIB")
    _tl = os.environ.get("TRAINABLE_LAYERS", "").strip()
    ap.add_argument("--trainable-layers", type=int, default=(int(_tl) if _tl else None),
                    choices=[0, 1, 2, 3, 4, 5],
                    help="ResNet freeze ladder — the RetinaNet equivalent of YOLO's freeze= "
                         "(exp6). Trainable stages counted from the TOP: 5=train everything "
                         "(== YOLO freeze=none), 4=freeze conv1, 3=freeze conv1+layer1 "
                         "(torchvision DEFAULT, what retinanet_s0 0.739 used), 2=freeze "
                         "through layer2, 1=layer4 only, 0=freeze the whole ResNet body. "
                         "FPN + heads ALWAYS train, so 0 is a frozen-trunk arm, not a linear "
                         "probe. Independent of --freeze-bn. Env: TRAINABLE_LAYERS")
    return ap


# ── equal-footing helpers (one AP code + one prediction budget) ───────────────
# WHY: two confounds make a raw YOLO-vs-torchvision AP compare invalid.
#   (a) different AP implementations (Ultralytics vs ap_from_per_image above);
#   (b) different prediction BUDGETS — at the 0.001 conf floor RetinaNet emits
#       100 boxes/img (its detections_per_img cap) vs YOLO's ~8-18. 101-point
#       interpolated AP takes the max precision to the right, so extra low-conf
#       true positives can only RAISE AP50. More boxes = free AP.
# These helpers fix both: run every model through per_image_for() → the same
# ap_from_per_image(), and equalise the budget with cap_per_image().

def tb_positive_items(split) -> list[tuple]:
    """The sealed TB positives as metrics-style (img, label, group) items.
    AP is only ever computed on these, so a compare run needs nothing else."""
    return [(splits.tb_image_path(s), splits.label_path(s), "tb")
            for s in split["test_positive_ids"]]


def per_image_for(kind: str, weights, items, imgsz: int = 512, device=None) -> list[dict]:
    """Per-image {group,gts,preds} dicts for EITHER family, same schema.
    kind: 'yolo' (Ultralytics) | 'retinanet' | 'fasterrcnn' | 'tv' (auto from ckpt)."""
    if kind == "yolo":
        from ultralytics import YOLO
        return metrics.collect(YOLO(str(weights)), items, imgsz)
    if kind in ("tv", "retinanet", "fasterrcnn"):
        dev = resolve_device(device)
        model = load_model(weights, imgsz, dev)
        return infer_per_image(model, items, dev,
                               single_class=getattr(model, "_single_class", False))
    raise ValueError(f"unknown kind {kind!r} (yolo | retinanet | fasterrcnn | tv)")


def cap_per_image(per_image, k: int | None) -> list[dict]:
    """Keep only the k highest-confidence preds per image — class-agnostic,
    exactly what a detector's max_det / detections_per_img does. k=None → as-is."""
    if k is None:
        return per_image
    return [{**im, "preds": sorted(im["preds"], key=lambda p: -p[1])[:k]}
            for im in per_image]


def renms_per_image(per_image, iou_thr: float | None) -> list[dict]:
    """Re-apply greedy per-class NMS at a COMMON IoU threshold. Ultralytics
    predicts at NMS IoU 0.7, torchvision at 0.5 — so duplicate suppression
    differs. NOTE: NMS can only be TIGHTENED after the fact (suppressed boxes
    are gone), so pass a threshold <= every model's own (0.5 here) and read it
    as 'both are now at most 0.5-NMS'd'. None → unchanged."""
    if iou_thr is None:
        return per_image
    out = []
    for im in per_image:
        kept = []
        for cls in sorted({p[0] for p in im["preds"]}):
            for p in sorted([q for q in im["preds"] if q[0] == cls], key=lambda q: -q[1]):
                if all(metrics.iou(p[2], q[2]) < iou_thr for q in kept if q[0] == cls):
                    kept.append(p)
        out.append({**im, "preds": sorted(kept, key=lambda p: -p[1])})
    return out


def avg_preds(per_image, group: str = "tb") -> float:
    """Mean predictions per image — the budget number the cap equalises."""
    sel = [im for im in per_image if im["group"] == group] or per_image
    return round(float(np.mean([len(im["preds"]) for im in sel])), 2)


def recall_ceiling(per_image, cls: int = 0, iou_thr: float = 0.5) -> float:
    """Fraction of GT boxes of `cls` covered by ANY pred of that class at
    IoU>=thr, ignoring confidence. The recall AP could reach with perfect
    ranking — so it says whether a cap deleted real detections or just noise.
    (Per-GT coverage, not a bipartite matching: two GTs may share one pred.)"""
    npos = hit = 0
    for im in per_image:
        preds = [p for p in im["preds"] if p[0] == cls]
        for _, g in [(c, b) for (c, b) in im["gts"] if c == cls]:
            npos += 1
            hit += any(metrics.iou(p[2], g) >= iou_thr for p in preds)
    return round(hit / npos, 4) if npos else 0.0


def dump_per_image(per_image, path) -> None:
    """Cache raw preds so inference (GPU) runs ONCE and every re-analysis
    (caps, NMS, AP variants) is free CPU work afterwards."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(per_image))


def load_per_image(path) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    return [{"group": im["group"],
             "gts": [(int(c), tuple(b)) for c, b in im["gts"]],
             "preds": [(int(c), float(s), tuple(b)) for c, s, b in im["preds"]]}
            for im in raw]


# ── optional: re-score a YOLO champion with THIS AP code (identical comparison) ─
def rescore_yolo(best_pt: str, imgsz: int = 512) -> dict:
    """Run a YOLO best.pt over the sealed TB positives and score with
    ap_from_per_image — so YOLO and the torchvision models use the SAME AP
    implementation. Handy one-off; the full matrix is yolo_experiments/ap_compare.py."""
    items = tb_positive_items(splits.build_or_load())
    return ap_from_per_image(per_image_for("yolo", best_pt, items, imgsz))
