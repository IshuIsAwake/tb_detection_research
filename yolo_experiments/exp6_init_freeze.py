"""
exp6_init_freeze.py — Does the VinDr-pretrained backbone help, and how deep
                      should we freeze it? Positives-only, single-seed SCREEN.

QUESTION
    exp1-5 fixed the *training recipe* (mosaic @ 512 @ batch 16, positives-only,
    full COCO→TBX11K fine-tune). exp6 changes the *initialisation*: instead of
    COCO weights (yolov8n.pt), start from the VinDr-CXR supervised backbone
    (a domain-matched chest-X-ray detector — see weights/pretrain/), and sweep
    how much of that backbone we freeze during TBX11K fine-tuning.

THE MATRIX (10 cells, single seed — this is a SCREEN, not a verdict)
    init  = VinDr-CXR  (weights/pretrain/vindr_supervised_best.pt)
    freeze ∈ {none, 4, 8, 10, 13}          # see the freeze map below
    aug    ∈ {mosaic, mosaic_mixup}        # both carried; mixup was a TIE under
                                           # COCO-init (exp4) — re-checked here in
                                           # case a domain backbone changes it.
    → 5 freeze × 2 aug = 10 runs, all at 512 @ batch 16, 200 epochs.

    The COCO-init full-FT reference is NOT re-run — exp3 already has it
    multi-seed at this exact config:
        COCO mosaic       512@16 → 0.707 ± 0.024  (exp3, 3 seeds)
        COCO mosaic_mixup 512@16 → 0.726 ± 0.025  (exp3, 3 seeds)
    The VinDr freeze=none cells drop straight onto those bands → that comparison
    answers "does domain pretrain beat COCO?".

FREEZE MAP (yolov8n — verified against the loaded model, do not guess)
    Ultralytics freeze=N freezes module indices 0..N-1. The backbone is modules
    0-9 (ends at SPPF[9]); the neck/head is 10-22. CRUCIAL: modules 10 (Upsample)
    and 11 (Concat) have ZERO params — the first trainable module past the
    backbone is C2f[12]. So freeze=10, 11, 12 all freeze the SAME weights.
        freeze=4   stem + first C2f stage  (early backbone, ~P1-P2)
        freeze=8   most of backbone, leaves C2f[8]+SPPF[9] trainable
        freeze=10  ENTIRE backbone (the clean boundary)
        freeze=13  backbone + first neck C2f[12]   (NOT 12 — that == 10)

DISCIPLINE (why this is only a screen)
    Single-seed run-to-run noise (±~0.025 Active mAP50, the exp5 CV bar) is as
    large as the gap between adjacent freeze depths. So these 10 cells reveal the
    SHAPE — does freezing help at all, where's the cliff — but cannot rank two
    cells within 0.025 of each other. The winning (init × freeze × aug) cell gets
    multi-seed CONFIRMED afterwards (re-run with SEED=1, SEED=2), then carried to
    Kaggle for the 1024 @ batch 16 final.

USAGE  (one cell per invocation; run the matrix sequentially — cells share the
        generated positives-only tree dir)
    for aug in mosaic mosaic_mixup; do
      for fz in none 4 8 10 13; do
        python yolo_experiments/exp6_init_freeze.py --aug-level $aug --freeze $fz
      done
    done
    python yolo_experiments/exp6_init_freeze.py --aggregate   # comparison table

    SEED env selects the seed (default 0). The frozen split is loaded, never
    rebuilt. Override the init with --model (default = the VinDr weights).
    Default run name: exp6_vindr_<aug>_fz<freeze>_s<seed>.
    TB_SMOKE=1 python yolo_experiments/exp6_init_freeze.py --aug-level mosaic --freeze 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → yolo_common
from yolo_common import aug as AUG, metrics, settings as S, splits, train_eval
from yolo_common.inventory import inventory_or_halt

# Domain-matched init produced by the VinDr-CXR supervised pretrain (Kaggle).
VINDR_WEIGHTS = S.REPO_ROOT / "weights" / "pretrain" / "vindr_supervised_best.pt"
# exp3 COCO-init full-FT references at this exact config (mean ± std, 3 seeds),
# printed by --aggregate so the VinDr cells are read against the right baseline.
COCO_REFERENCE = {"mosaic": (0.707, 0.024), "mosaic_mixup": (0.726, 0.025)}


def environment() -> dict:
    info = {"timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": "n/a (not a git repo)"}
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = "?"
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["torch"], info["gpu"] = "?", "?"
    return info


def parse_freeze(v: str) -> int | None:
    """'none'/'0'/'' → None (full FT); otherwise an int layer count."""
    v = (v or "").strip().lower()
    return None if v in ("", "none", "0", "false") else int(v)


def cell_tag(aug_level: str, freeze: int | None, seed: int) -> str:
    fz = "none" if freeze is None else str(freeze)
    return f"exp6_vindr_{aug_level}_fz{fz}_s{seed}"


def run_cell(args) -> None:
    model = args.model or str(VINDR_WEIGHTS)
    freeze = parse_freeze(args.freeze)
    seed = S.SEED
    tag = args.name or cell_tag(args.aug_level, freeze, seed)

    if not Path(model).exists():
        raise SystemExit(f"init weights not found: {model}\n"
                         f"  expected the VinDr pretrain at {VINDR_WEIGHTS}")

    print(f"\n████ {tag}  (model={Path(model).name}, freeze={freeze}, "
          f"imgsz={args.imgsz}, batch={args.batch}, epochs={args.epochs}, "
          f"aug={args.aug_level}, seed={seed}, smoke={S.SMOKE}) ████")
    print(f"  aug: {AUG.describe(args.aug_level)}\n")

    inv = inventory_or_halt()
    split = splits.build_or_load()
    print(f"══ Step 1 — split (seed={split['seed']}) + positives-only tree ══")
    print(f"  {json.dumps(split['counts'])}")
    data_yaml = splits.materialise(smoke=S.SMOKE)     # positives only (no negatives)
    print(f"  positives-only · data.yaml → {data_yaml}\n")

    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"══ Step 2 — train {Path(model).name} @ imgsz={args.imgsz}, "
          f"aug={args.aug_level}, freeze={freeze} ══")
    tr = train_eval.train(data_yaml, args.imgsz, project=out_dir, name="ultralytics",
                          model=model, epochs=args.epochs, batch=args.batch,
                          device=args.device, aug_level=args.aug_level,
                          freeze=freeze, patience=args.patience,
                          lr0=args.lr0, workers=args.workers)
    print(f"  best.pt → {tr['best_pt']}  ({tr['train_sec']}s)\n")

    print("══ Step 3 — evaluate on the sealed 360 ══")
    t0 = time.time()
    yolo = train_eval.load_best(tr["best_pt"])
    items = splits.blackbox(split)
    metric_block = metrics.evaluate(yolo, args.imgsz, split, items)
    eval_sec = round(time.time() - t0, 1)

    report = {
        "experiment": tag,
        "config": {**tr["resolved"], "imgsz": args.imgsz, "seed": seed,
                   "arm": "init-freeze-posonly", "init": Path(model).name,
                   "augmentation": tr["resolved"]["aug_level"] != "off", "amp": S.AMP,
                   "aug_kwargs": AUG.AUG_LEVELS[args.aug_level],
                   "negatives_in_training": False,
                   "conf_thresholds": S.CONF_THRESHOLDS},
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": split["counts"]["train_positives"],
                    "val_positives": split["counts"]["val_positives"],
                    "blackbox": {"tb": split["counts"]["test_positives"],
                                 "sick_non_tb": split["counts"]["blackbox_sick"],
                                 "healthy": split["counts"]["blackbox_healthy"]},
                    "box_counts": {"active": inv["box_counts"]["ActiveTuberculosis"],
                                   "obsolete": inv["box_counts"]["ObsoletePulmonaryTuberculosis"]},
                    "class_map": {"0": "ActiveTuberculosis", "1": "ObsoletePulmonaryTuberculosis"}},
        "environment": environment(),
        "metrics": metric_block,
        "timing": {"train_sec": tr["train_sec"], "eval_sec": eval_sec},
        "artifacts": {"weights": str(tr["best_pt"]),
                      "pr_curve": str(tr["run_dir"] / "PR_curve.png"),
                      "results_csv": str(tr["run_dir"] / "results.csv"),
                      "confusion_matrix_val": str(tr["run_dir"] / "confusion_matrix.png"),
                      "confusion_matrix_test": str(out_dir / "confusion_matrix_test.png")},
    }
    metrics.save_confusion_png(metric_block["confusion_matrix"],
                               out_dir / "confusion_matrix_test.png")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.txt").write_text(metrics.summary_text(report))
    (out_dir / "config_snapshot.json").write_text(json.dumps(report["config"], indent=2))

    print(f"\n→ {out_dir}/metrics.json")
    print(f"→ {out_dir}/summary.txt\n")
    print((out_dir / "summary.txt").read_text())


def aggregate(args) -> None:
    """Tabulate every exp6 cell found on disk, sorted by Active mAP50, against
    the exp3 COCO-init references. Single-seed — read SHAPE, not fine ranking."""
    seed = S.SEED
    print(f"══ exp6 aggregate — VinDr-init × freeze, seed={seed} ══")
    cells = []
    for aug in ("mosaic", "mosaic_mixup"):
        for fz in (None, 4, 8, 10, 13):
            p = S.RESULTS_ROOT / cell_tag(aug, fz, seed) / "metrics.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())["metrics"]
            cells.append({
                "aug": aug, "freeze": ("none" if fz is None else fz),
                "active_mAP50": m["detection"]["active"]["mAP50"],
                "overall_mAP50": m["detection"]["overall"]["mAP50"],
                "obsolete_mAP50": m["detection"]["obsolete"]["mAP50"],
                "active_recall": m["detection"]["active"]["recall"],
                "loc_iou": m["localization"]["active"]["iou"],
                "medium_recall": m["by_size"]["medium"]["recall"],
            })
    if not cells:
        raise SystemExit(f"no exp6 cells found under {S.RESULTS_ROOT}/exp6_vindr_*_s{seed}/ "
                         f"— run the matrix first.")

    ranked = sorted(cells, key=lambda c: c["active_mAP50"], reverse=True)
    lines = [
        f"# exp6 — VinDr-init × freeze (positives-only, mosaic/mixup @ 512 @ batch 16, "
        f"200ep, seed={seed})",
        f"# SCREEN: single-seed; gaps < ~0.025 (exp5 CV bar) are TIES — confirm the "
        f"winner multi-seed.",
        f"# COCO-init full-FT reference (exp3, 3 seeds):  "
        f"mosaic {COCO_REFERENCE['mosaic'][0]:.3f}±{COCO_REFERENCE['mosaic'][1]:.3f}   "
        f"mosaic_mixup {COCO_REFERENCE['mosaic_mixup'][0]:.3f}±{COCO_REFERENCE['mosaic_mixup'][1]:.3f}",
        "",
        f"  {'aug':<13}{'freeze':<8}{'ActmAP50':<10}{'vsCOCO':<9}"
        f"{'OvmAP50':<9}{'ObsmAP50':<10}{'ActRec':<8}{'locIoU':<8}{'medRec'}",
    ]
    for c in ranked:
        ref = COCO_REFERENCE[c["aug"]][0]
        delta = c["active_mAP50"] - ref
        lines.append(
            f"  {c['aug']:<13}{str(c['freeze']):<8}{c['active_mAP50']:<10.3f}"
            f"{delta:+.3f}   {c['overall_mAP50']:<9.3f}{c['obsolete_mAP50']:<10.3f}"
            f"{c['active_recall']:<8.3f}{c['loc_iou']:<8.3f}{c['medium_recall']:.3f}")
    best = ranked[0]
    lines += [
        "",
        f"  BEST (this seed): {best['aug']} freeze={best['freeze']} → "
        f"Active mAP50 {best['active_mAP50']:.3f} "
        f"({best['active_mAP50'] - COCO_REFERENCE[best['aug']][0]:+.3f} vs COCO). "
        f"Confirm multi-seed before carrying to Kaggle 1024@16.",
    ]
    text = "\n".join(lines) + "\n"

    out_dir = S.RESULTS_ROOT / f"exp6_init_freeze_s{seed}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"experiment": f"exp6_init_freeze_s{seed}", "seed": seed,
         "coco_reference": COCO_REFERENCE, "cells": ranked,
         "environment": environment()}, indent=2))
    (out_dir / "summary.txt").write_text(text)
    print(f"\n→ {out_dir}/summary.json\n→ {out_dir}/summary.txt\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="exp6: VinDr-init × freeze-depth screen (positives-only, "
                    "512@16). One cell per invocation, then --aggregate. SEED env "
                    "selects the seed.")
    ap.add_argument("--aggregate", action="store_true",
                    help="tabulate all exp6 cells on disk vs the COCO reference")
    ap.add_argument("--aug-level", choices=sorted(AUG.AUG_LEVELS),
                    help="mosaic or mosaic_mixup (the two carried augs)")
    ap.add_argument("--freeze", default="none",
                    help="none | 4 | 8 | 10 | 13  (see freeze map in the docstring)")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--model", default=None,
                    help="init weights (default = the VinDr pretrain)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args)
        return
    if not args.aug_level:
        ap.error("--aug-level is required for a training cell "
                 "(or pass --aggregate to tabulate)")
    run_cell(args)


if __name__ == "__main__":
    main()
