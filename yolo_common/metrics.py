"""
metrics.py — One honest diagnostic evaluation over the sealed 360.

Does NOT rely on Ultralytics' built-in val alone. Two passes:

  1. Detection AP (threshold-free): Ultralytics `model.val()` on the 120 test
     positives → mAP50, mAP50-95, precision, recall — overall + per class.
  2. Custom pass over the full 360 (120 TB + 120 sick + 120 healthy):
       • localisation  — preds at conf>=LOC_CONF greedily matched to GT, kept at
                         IoU>=0.5; mean IoU / IoP / IoG over matched pairs.
       • screening     — at each of CONF_THRESHOLDS (+ an untuned recall ceiling):
                         TB detection rate, healthy & sick false-alarm rates,
                         avg preds/img per group, raw "X/120" counts.
       • by_size       — GT boxes bucketed by area fraction (small <1%, medium
                         1-5%, large >5%); recall per bucket.

Match rule is pluggable (MatchRule). Headline = IoU>=0.5 (radiologist boxes in
TBX11K are tight, unlike the loose oral-cancer annotations, so IoU is the right
gate here — a deliberate, stated choice). IoP/IoG are reported alongside.

Returns the `metrics` sub-dict in the locked exp1 schema. The caller wraps it
with experiment/config/dataset/environment/timing/artifacts and writes
metrics.json + summary.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yolo_common import settings as S


# ── geometry ──────────────────────────────────────────────────────────────────

def _inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _area(x) -> float:
    return max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])


def iou(a, b) -> float:
    u = _area(a) + _area(b) - _inter(a, b)
    return _inter(a, b) / u if u > 0 else 0.0


def iop(p, g) -> float:           # intersection / prediction area  (over-prediction)
    a = _area(p)
    return _inter(p, g) / a if a > 0 else 0.0


def iog(p, g) -> float:           # intersection / GT area          (coverage)
    a = _area(g)
    return _inter(p, g) / a if a > 0 else 0.0


class MatchRule:
    __slots__ = ("name", "key", "ok")

    def __init__(self, name, key, ok):
        self.name, self.key, self.ok = name, key, ok


IOU_RULE = MatchRule("iou>=0.5", iou, lambda p, g: iou(p, g) >= S.IOU_MATCH)


# ── prediction / GT extraction ────────────────────────────────────────────────

def _gt_boxes(label_path: Path | None, w: int, h: int) -> list[tuple[int, tuple]]:
    if label_path is None or not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def _preds(res) -> list[tuple[int, float, tuple]]:
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        for b, cf, cl in zip(xyxy, conf, cls):
            out.append((int(cl), float(cf),
                        (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
    return out


def collect(model, items, imgsz) -> list[dict]:
    """Run the model once per image at the recall ceiling; cache all preds.

    Returns per-image dicts: {group, gts:[(cls,box)], preds:[(cls,conf,box)]}.
    Filtering by conf happens in-memory afterwards — model runs once per image.
    """
    per_image = []
    for img_path, label_path, group in items:
        res = model.predict(str(img_path), imgsz=imgsz, device=S.DEVICE,
                            conf=S.RECALL_CEILING_CONF, verbose=False)[0]
        h, w = res.orig_shape
        per_image.append({
            "group": group,
            "gts": _gt_boxes(label_path, w, h),
            "preds": _preds(res),
        })
    return per_image


# ── matching core ─────────────────────────────────────────────────────────────

def _match(gts, preds, conf, rule=IOU_RULE):
    """Greedy highest-conf-first match within class. Returns (pairs, matched_gt)
    where pairs = [(pred_box, gt_box, cls)], matched_gt = set of gt indices."""
    preds = sorted([p for p in preds if p[1] >= conf], key=lambda x: -x[1])
    pairs, matched = [], set()
    used = set()
    for cls, _, pbox in preds:
        best_j, best_v = -1, -1.0
        for j, (gc, gbox) in enumerate(gts):
            if j in used or gc != cls:
                continue
            v = rule.key(pbox, gbox)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0 and rule.ok(pbox, gts[best_j][1]):
            used.add(best_j)
            matched.add(best_j)
            pairs.append((pbox, gts[best_j][1], cls))
    return pairs, matched


# ── localisation ──────────────────────────────────────────────────────────────

def _loc_block(pairs) -> dict:
    if not pairs:
        return {"iou": 0.0, "iop": 0.0, "iog": 0.0, "n_matched": 0}
    return {
        "iou": round(float(np.mean([iou(p, g) for p, g, _ in pairs])), 4),
        "iop": round(float(np.mean([iop(p, g) for p, g, _ in pairs])), 4),
        "iog": round(float(np.mean([iog(p, g) for p, g, _ in pairs])), 4),
        "n_matched": len(pairs),
    }


def localization(per_image, conf=S.LOC_CONF) -> dict:
    all_pairs, by_cls = [], {0: [], 1: []}
    for im in per_image:
        if im["group"] != "tb":
            continue
        pairs, _ = _match(im["gts"], im["preds"], conf)
        all_pairs += pairs
        for pr in pairs:
            by_cls[pr[2]].append(pr)
    return {
        "overall": _loc_block(all_pairs),
        "active": _loc_block(by_cls[0]),
        "obsolete": _loc_block(by_cls[1]),
    }


# ── screening ─────────────────────────────────────────────────────────────────

def _group_imgs(per_image, group):
    return [im for im in per_image if im["group"] == group]


def screening(per_image, confs) -> dict:
    groups = {g: _group_imgs(per_image, g) for g in ("tb", "sick", "healthy")}
    out = {}
    for conf in confs:
        def rate(group):
            imgs = groups[group]
            flagged = sum(1 for im in imgs
                          if any(p[1] >= conf for p in im["preds"]))
            return flagged, len(imgs)

        def avg_preds(group):
            imgs = groups[group]
            if not imgs:
                return 0.0
            return round(float(np.mean([sum(1 for p in im["preds"] if p[1] >= conf)
                                        for im in imgs])), 3)

        tb_f, tb_n = rate("tb")
        h_f, h_n = rate("healthy")
        s_f, s_n = rate("sick")
        out[f"{conf:.2f}"] = {
            "tb_detect_rate": round(tb_f / tb_n, 4) if tb_n else 0.0,
            "tb_flagged": f"{tb_f}/{tb_n}",
            "healthy_false_alarm": round(h_f / h_n, 4) if h_n else 0.0,
            "healthy_flagged": f"{h_f}/{h_n}",
            "sick_false_alarm": round(s_f / s_n, 4) if s_n else 0.0,
            "sick_flagged": f"{s_f}/{s_n}",
            "avg_preds_per_img": {"tb": avg_preds("tb"),
                                  "healthy": avg_preds("healthy"),
                                  "sick": avg_preds("sick")},
        }
    return out


# ── by lesion size ────────────────────────────────────────────────────────────

def by_size(per_image, conf=S.LOC_CONF, img_area=512 * 512) -> dict:
    buckets = {name: {"matched": 0, "n": 0} for name, _, _ in S.SIZE_BUCKETS}

    def bucket_of(area_frac):
        for name, lo, hi in S.SIZE_BUCKETS:
            if lo <= area_frac < hi:
                return name
        return S.SIZE_BUCKETS[-1][0]

    for im in per_image:
        if im["group"] != "tb":
            continue
        _, matched = _match(im["gts"], im["preds"], conf)
        for j, (_, gbox) in enumerate(im["gts"]):
            name = bucket_of(_area(gbox) / img_area)
            buckets[name]["n"] += 1
            if j in matched:
                buckets[name]["matched"] += 1

    return {name: {"recall": round(b["matched"] / b["n"], 4) if b["n"] else 0.0,
                   "n": b["n"]}
            for name, b in buckets.items()}


# ── confusion matrix on the TEST positives ────────────────────────────────────

def confusion_matrix(per_image, conf=S.LOC_CONF, iou_thr=S.IOU_MATCH) -> dict:
    """Detection confusion matrix on the sealed TEST positives, mirroring the
    Ultralytics layout (rows = PREDICTED, cols = TRUE, last index = background)
    — but on the blackbox, not val. Matching is CLASS-AGNOSTIC (highest-conf
    pred → best-IoU unused GT regardless of class) so cross-class confusion
    (Active↔Obsolete) shows up in the off-diagonal cells.

      matrix[pred][true_bg] = false positive (pred matched no GT)
      matrix[bg][true_cls]  = false negative (GT missed)
      matrix[bg][bg]        = undefined for detection (no true negatives)
    """
    n = S.NUM_CLASSES
    BG = n
    M = [[0] * (n + 1) for _ in range(n + 1)]
    for im in per_image:
        if im["group"] != "tb":
            continue
        gts = im["gts"]
        preds = sorted([p for p in im["preds"] if p[1] >= conf], key=lambda x: -x[1])
        used = set()
        for pcls, _, pbox in preds:
            best_j, best_v = -1, -1.0
            for j, (_, gbox) in enumerate(gts):
                if j in used:
                    continue
                v = iou(pbox, gbox)
                if v > best_v:
                    best_v, best_j = v, j
            if best_j >= 0 and best_v >= iou_thr:
                used.add(best_j)
                M[pcls][gts[best_j][0]] += 1        # matched: pred-class vs true-class
            else:
                M[pcls][BG] += 1                     # FP → true = background
        for j, (gc, _) in enumerate(gts):
            if j not in used:
                M[BG][gc] += 1                       # FN → pred = background
    return {"labels": S.CLASS_NAMES + ["background"], "matrix": M,
            "conf": conf, "iou": iou_thr, "axes": "rows=predicted, cols=true"}


def save_confusion_png(cm: dict, path) -> None:
    """Render the TEST confusion matrix to a PNG (rows=pred, cols=true)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array(cm["matrix"], dtype=float)
    labels = cm["labels"]
    bg = len(labels) - 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title(f"Confusion Matrix — TEST (conf>={cm['conf']}, IoU>={cm['iou']})")
    thr = M.max() / 2 if M.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == bg and j == bg:
                continue
            ax.text(j, i, int(M[i][j]), ha="center", va="center",
                    color="white" if M[i][j] > thr else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


# ── detection AP via Ultralytics val ──────────────────────────────────────────

def detection_ap(model, imgsz, split, root=None) -> dict:
    """Build a temp val tree on the test positives and run model.val().

    `root` lets a caller (e.g. exp5 k-fold) point at a fold-local tree; default
    is the shared `_eval_testpos`. The val dirs are cleared first so a varying
    set of test stems (different folds) can't leave stale links behind."""
    from yolo_common import splits as SP
    import shutil

    root = root or (S.GEN_ROOT / "_eval_testpos")
    for sub in ("images/val", "labels/val"):
        d = root / sub
        if d.exists():
            shutil.rmtree(d)
    for stem in split["test_positive_ids"]:
        SP._link(SP.tb_image_path(stem), root / "images" / "val" / f"{stem}.png")
        SP._link(SP.label_path(stem), root / "labels" / "val" / f"{stem}.txt")
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root.resolve()}\ntrain: images/val\nval: images/val\n"
        f"nc: {S.NUM_CLASSES}\nnames: {S.CLASS_NAMES}\n"
    )
    m = model.val(data=str(yaml), imgsz=imgsz, device=S.DEVICE, verbose=False,
                  plots=False, split="val")

    def cls_block(i):
        try:
            p, r, ap50, ap = m.class_result(i)
            return {"mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4),
                    "precision": round(float(p), 4), "recall": round(float(r), 4)}
        except Exception:
            return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "overall": {"mAP50": round(float(m.box.map50), 4),
                    "mAP50_95": round(float(m.box.map), 4),
                    "precision": round(float(m.box.mp), 4),
                    "recall": round(float(m.box.mr), 4)},
        "active": cls_block(0),
        "obsolete": cls_block(1),
    }


# ── top-level ─────────────────────────────────────────────────────────────────

def evaluate(model, imgsz, split, items, eval_root=None) -> dict:
    """Return the locked `metrics` sub-dict (detection/localization/screening/by_size).

    `eval_root` is forwarded to detection_ap so a k-fold caller can isolate the
    temp val tree per fold; None keeps the shared default."""
    per_image = collect(model, items, imgsz)
    confs = list(S.CONF_THRESHOLDS) + [S.RECALL_CEILING_CONF]
    return {
        "detection": detection_ap(model, imgsz, split, root=eval_root),
        "localization": localization(per_image),
        "screening": screening(per_image, confs),
        "by_size": by_size(per_image),
        "confusion_matrix": confusion_matrix(per_image),
    }


# ── human-readable summary ────────────────────────────────────────────────────

def summary_text(report: dict) -> str:
    m = report["metrics"]
    L = [f"# {report['experiment']}  ({report['config']['model']} @ "
         f"imgsz={report['config']['imgsz']})",
         f"# headline match rule: {IOU_RULE.name}  |  aug={report['config']['augmentation']}"
         f"  amp={report['config']['amp']}", ""]

    d = m["detection"]
    L += ["== detection (AP on 120 test positives) =="]
    for k in ("overall", "active", "obsolete"):
        b = d[k]
        L.append(f"  {k:<9} mAP50={b['mAP50']:.3f} mAP50-95={b['mAP50_95']:.3f} "
                 f"P={b['precision']:.3f} R={b['recall']:.3f}")

    loc = m["localization"]
    L += ["", "== localisation on matched pairs (conf>=0.25, IoU>=0.5) =="]
    for k in ("overall", "active", "obsolete"):
        b = loc[k]
        L.append(f"  {k:<9} IoU={b['iou']:.3f} IoP={b['iop']:.3f} IoG={b['iog']:.3f} "
                 f"(n_matched={b['n_matched']})")

    L += ["", "== screening (image flagged if >=1 pred at conf) =="]
    for conf in sorted(m["screening"], key=float):
        s = m["screening"][conf]
        L.append(f"  conf={conf}: TB {s['tb_flagged']} (det {s['tb_detect_rate']:.3f}) | "
                 f"healthy FA {s['healthy_flagged']} ({s['healthy_false_alarm']:.3f}) | "
                 f"sick FA {s['sick_flagged']} ({s['sick_false_alarm']:.3f})")
        ap = s["avg_preds_per_img"]
        L.append(f"            avg preds/img  tb={ap['tb']} healthy={ap['healthy']} "
                 f"sick={ap['sick']}")

    bs = m["by_size"]
    L += ["", "== recall by GT lesion size (small<1%, medium 1-5%, large>5% img area) =="]
    for k in ("small", "medium", "large"):
        L.append(f"  {k:<7} recall={bs[k]['recall']:.3f}  (n={bs[k]['n']})")

    cm = m.get("confusion_matrix")
    if cm:
        M, labs = cm["matrix"], cm["labels"]
        bg = len(labs) - 1
        short = [l[:9] for l in labs]
        L += ["", f"== confusion matrix on TEST positives (rows=pred, cols=true, "
                  f"conf>={cm['conf']}, IoU>={cm['iou']}) =="]
        L.append("  pred\\true  " + " ".join(f"{s:>9}" for s in short))
        for i, row in enumerate(M):
            cells = ["        -" if (i == bg and j == bg) else f"{v:>9}"
                     for j, v in enumerate(row)]
            L.append(f"  {short[i]:<9} " + " ".join(cells))

    L += ["",
          "NOTE: a positives-only model is EXPECTED to over-fire on negatives, so",
          "high false-alarm is the diagnostic, not a bug. The key signal is the",
          "healthy-vs-sick false-alarm GAP: a larger gap means the model is",
          "confusing other pathology (sick) with TB.", ""]
    return "\n".join(L)
