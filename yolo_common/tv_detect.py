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
      YOLO `geo` runs (≈0.685 Active mAP50). torchvision has NO mosaic/mixup, so
      the mosaic_mixup champion (0.745) is a SEPARATE arm — this isolates the
      ARCHITECTURE at a matched geo recipe. A ResNet-50-FPN (~30-40M params) on
      559 train positives is a big model on tiny data — COCO-pretrained, so it is
      transfer, but expect variance.
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
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from yolo_common import metrics, settings as S, splits


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
def _build_aug(aug: str):
    """Box-aware train transform (torchvision v2). `geo` MATCHES YOLO's
    yolo_common/aug.py `_GEO` level so a torchvision run is comparable to our
    YOLO `geo` runs: degrees=10, translate=0.1, scale=[0.5,1.5], fliplr=0.5, gray
    (114) border fill, NO shear/perspective/photometric (X-ray intensity is
    diagnostic; upright pose). `hflip` = flip only; `off` = none/eval."""
    from torchvision.transforms import v2
    if aug == "off":
        return None
    ops = []
    if aug == "geo":
        ops.append(v2.RandomAffine(degrees=10, translate=(0.1, 0.1),
                                   scale=(0.5, 1.5), fill=114))       # == YOLO _GEO
    ops.append(v2.RandomHorizontalFlip(p=0.5))                        # fliplr=0.5
    ops += [v2.ClampBoundingBoxes(), v2.SanitizeBoundingBoxes()]      # drop boxes warped out
    return v2.Compose(ops)


class TBDetDataset:
    """Positives from the frozen split → (image[0..1] CHW float, target dict).

    target = {"boxes": FloatTensor[N,4] xyxy pixel, "labels": Int64[N] in 1..2}.
    YOLO class 0/1 (Active/Obsolete) → torchvision label +1 (0 = background).
    Train aug is box-aware via _build_aug (see it for the YOLO-`geo` match).
    """

    def __init__(self, stems: list[str], train: bool, aug: str = "geo"):
        self.stems = list(stems)
        self.train = train
        self.aug = aug
        self.tf = _build_aug(aug) if train else None

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        import torch
        from torchvision.io import read_image, ImageReadMode
        from torchvision import tv_tensors

        stem = self.stems[i]
        img = read_image(str(splits.tb_image_path(stem)), ImageReadMode.RGB)   # uint8 [3,H,W]
        _, h, w = img.shape

        boxes, labels = [], []
        for cls, (x1, y1, x2, y2) in metrics._gt_boxes(splits.label_path(stem), w, h):
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)                              # +1: 0 = background
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        if self.tf is not None and len(boxes):
            bb = tv_tensors.BoundingBoxes(boxes_t, format="XYXY", canvas_size=(h, w))
            img, out = self.tf(tv_tensors.Image(img), {"boxes": bb, "labels": labels_t})
            boxes_t = out["boxes"].as_subclass(torch.Tensor).float().reshape(-1, 4)
            labels_t = out["labels"].as_subclass(torch.Tensor)

        img = img.as_subclass(torch.Tensor).float() / 255.0
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
    return rep


def build_model(arch: str, imgsz: int, score_thresh: float = 0.001, pretrained: bool = True,
                backbone_weights=None, trainable_layers=None):
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

    num_classes = S.NUM_CLASSES + 1                                 # + background
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

    active, ap_a = cls_block(0)
    obsolete, ap_o = cls_block(1)
    present = [a for a in (ap_a, ap_o) if a is not None]
    overall_map50 = round(float(np.mean(present)), 4) if present else 0.0
    overall = {
        "mAP50": overall_map50,
        "mAP50_95": round(float(np.mean([active["mAP50_95"], obsolete["mAP50_95"]])), 4),
        "precision": round(float(np.mean([active["precision"], obsolete["precision"]])), 4),
        "recall": round(float(np.mean([active["recall"], obsolete["recall"]])), 4),
    }
    return {"overall": overall, "active": active, "obsolete": obsolete}


# ── inference → per-image dicts (mirrors metrics.collect for torchvision) ──────
def infer_per_image(model, items, device, conf_floor=S.RECALL_CEILING_CONF,
                    batch: int = 1) -> list[dict]:
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
                per_image.append({
                    "group": group,
                    "gts": metrics._gt_boxes(label_path, w, h),
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
        "localization": metrics.localization(per_image),
        "screening": metrics.screening(per_image, list(S.CONF_THRESHOLDS) + [S.RECALL_CEILING_CONF]),
        "by_size": metrics.by_size(per_image),
        "confusion_matrix": metrics.confusion_matrix(per_image),
    }


# ── train ─────────────────────────────────────────────────────────────────────
def _val_active_map50(model, val_stems, device) -> float:
    """Per-epoch model-selection signal — NOT a reported number, so it takes the
    batched inference path (see infer_per_image: batch>1 costs ~1e-4 of AP to cuDNN
    kernel choice, and buys back ~18% of every epoch on 119 images). The sealed eval
    stays at batch=1 so the headline reproduces exactly."""
    items = [(splits.tb_image_path(s), splits.label_path(s), "tb") for s in val_stems]
    per_image = infer_per_image(model, items, device, batch=8)
    return ap_from_per_image(per_image, iou50_95=False)["active"]["mAP50"]


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
    print(f"[bn] recalibrated {len(bns)} backbone BatchNorm layers on {seen} images "
          f"@ imgsz={imgsz} through model.transform (ImageNet-normalised, as training "
          f"feeds it — see recalibrate_backbone_bn)")
    return seen


def train(arch, train_stems, val_stems, *, imgsz, epochs, batch, lr, device, aug,
          workers, out_dir, accum=8, patience=30, backbone_weights=None,
          freeze_bn=False, bn_recalib=0, trainable_layers=None) -> dict:
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
    if bn_recalib and backbone_weights:
        recalibrate_backbone_bn(model, train_stems, imgsz, device, n_batches=bn_recalib)
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
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps = len(dl_tr)
    warm_iters = min(500, steps) if steps > 1 else 0          # ~1 epoch (or 500 iters) warmup
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
    for ep in range(epochs):
        cos_lr = 0.5 * lr * (1 + math.cos(math.pi * ep / max(1, epochs)))   # cosine per epoch
        model.train()
        if freeze_bn:
            freeze_backbone_bn(model)   # model.train() re-enables BN — re-freeze every epoch
        opt.zero_grad()
        run_loss, nb, t0 = 0.0, 0, time.time()
        for it, (imgs, targets) in enumerate(dl_tr):
            cur_lr = lr * (0.01 + 0.99 * gstep / max(1, warm_iters)) if gstep < warm_iters else cos_lr
            for g in opt.param_groups:
                g["lr"] = cur_lr
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = sum(model(imgs, targets).values()) / accum
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0 or (it + 1) == steps:         # step every `accum` batches
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            run_loss += float(loss.item()) * accum
            nb += 1
            gstep += 1
        vmap = _val_active_map50(model, val_stems, device)
        torch.save({"arch": arch, "imgsz": imgsz, "state_dict": model.state_dict()}, last_pt)
        flag = ""
        if vmap > best_map:
            best_map, best_ep, since = vmap, ep, 0
            torch.save({"arch": arch, "imgsz": imgsz, "state_dict": model.state_dict()}, best_pt)
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
            "body_trainable_params": body_tr, "body_frozen_params": body_fz}


def load_model(best_pt: Path, imgsz: int, device):
    import torch
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt.get("imgsz", imgsz), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
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


# ── top-level run (called by the two thin experiment scripts) ─────────────────
def run(arch: str, args) -> dict:
    device = resolve_device(args.device)
    tag = args.name or f"{arch}_s{S.SEED}"
    smoke = S.SMOKE

    print(f"\n████ {tag}  (arch={arch}, imgsz={args.imgsz}, batch={args.batch}, "
          f"epochs={args.epochs}, aug={args.aug}, lr={args.lr}, device={device}, "
          f"smoke={smoke}) ████\n")

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
    print(f"  train_positives={len(train_stems)} val_positives={len(val_stems)} "
          f"sealed_eval_items={len(items)}\n")

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
               trainable_layers=getattr(args, "trainable_layers", None))
    print(f"  best.pt → {tr['best_pt']}  (best val Active mAP50 {tr['best_val_map50']:.4f} "
          f"@ ep{tr['best_epoch']}; ran {tr['epochs_ran']}/{tr['epochs_cap']}, "
          f"eff_batch={tr['eff_batch']})\n")

    print("══ evaluate on the sealed 360 ══")
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
                   "augmentation": args.aug != "off", "amp": device.type == "cuda",
                   "pretrained": "COCO_V1", "min_max_size": args.imgsz,
                   # Provenance for the SSL arm: which backbone, and whether its BN
                   # statistics survived the fine-tune (they do NOT at batch 2 unless
                   # freeze_bn is on — see freeze_backbone_bn()).
                   "backbone_init": (tr.get("backbone_init") or "COCO_V1"),
                   "freeze_bn": getattr(args, "freeze_bn", False),
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
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": len(train_stems), "val_positives": len(val_stems),
                    "blackbox": {"tb": split["counts"]["test_positives"],
                                 "sick_non_tb": split["counts"]["blackbox_sick"],
                                 "healthy": split["counts"]["blackbox_healthy"]},
                    "class_map": {"0": "ActiveTuberculosis", "1": "ObsoletePulmonaryTuberculosis"}},
        "environment": _environment(),
        # The learning curve, so a finished run can be re-read without its stdout.
        # `history` duplicates history.jsonl on purpose: the jsonl survives a killed
        # run, this survives a deleted scratch dir.
        "training": {"best_val_active_map50": round(float(tr["best_val_map50"]), 5),
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
    ours = metric_block["detection"]["active"]["mAP50"]
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
    ap.add_argument("--aug", choices=["geo", "hflip", "off"], default="geo",
                    help="geo = matches YOLO's _GEO level (comparable to our YOLO geo runs); "
                         "mosaic/mixup is a SEPARATE arm (torchvision lacks it)")
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
        return infer_per_image(load_model(weights, imgsz, dev), items, dev)
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
