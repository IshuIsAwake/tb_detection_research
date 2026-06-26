"""
exp3_augmentation.py — augmentation SCREEN (positives-only, the single variable).

PURPOSE
    exp1 showed the floor overfits by ~epoch 12 and the bottleneck is recall on
    small/medium lesions. Augmentation is the biggest untouched lever for BOTH.
    exp3 holds the exp1 floor fixed (yolov8n, positives-only, AMP off,
    deterministic, frozen split, same sealed 360) and changes exactly ONE thing:
    the augmentation level. It is a SCREEN — single seed, fast 512 @ batch 16 —
    to find the direction; finalists are confirmed multi-seed at 1024 in exp4.

ABLATION (one file = one ablation; the variable is --aug-level)
    off         no-aug control (re-runs the floor at 512@16 for a clean baseline)
    geo         realistic geometry: degrees/translate/scale/fliplr
    geo_photo   geo + brightness (hsv_v); hue/sat are no-ops on grayscale
    mosaic      geo + mosaic (YOLO's strongest regulariser), close_mosaic 10
    mosaic_mixup  mosaic + mixup — optional follow-up if mosaic earns it
    default     Ultralytics kitchen sink (incl. albumentations Blur/CLAHE/...)
    See yolo_common/aug.py for the exact knob values.

WHY 200 EPOCHS
    Augmentation delays overfitting, so the floor's 100 epochs would UNDERTRAIN
    the aug arms and handicap them. best.pt (peak val) makes the longer schedule
    safe — the `off` arm just peaks early and best.pt keeps that peak.

RUN (screen all arms at 512 @ batch 16; ~11-20 min each on a 6GB card)
    python yolo_experiments/exp3_augmentation.py --aug-level off
    python yolo_experiments/exp3_augmentation.py --aug-level geo
    python yolo_experiments/exp3_augmentation.py --aug-level geo_photo
    python yolo_experiments/exp3_augmentation.py --aug-level mosaic
    python yolo_experiments/exp3_augmentation.py --aug-level default
    TB_SMOKE=1 python yolo_experiments/exp3_augmentation.py --aug-level geo  # plumbing

    Default run name: exp3_<aug-level>_<imgsz>  (override with --name).

READING  (yolo_experiments/results/<run>/)
    Headline:   detection.active.mAP50   (lesion-finding on positives)
    Guardrail:  by_size.small / .medium recall (did it attack the real wall?)
    Guardrail:  localization.overall IoU  (real finds, not spray)
    Overfit:    the Ultralytics run dir results.csv — does val mAP50 now peak
                LATER than ~epoch 12, with a smaller train/val gap?
    Compare every arm against the `off` run — same blackbox, same schema.
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="exp3: augmentation screen (positives-only, one variable = "
                    "--aug-level). Knobs mirror exp1; defaults are the 512@16 "
                    "screen. Every knob is also an env var (settings.py).")
    ap.add_argument("--aug-level", required=True, choices=sorted(AUG.AUG_LEVELS),
                    help="the swept variable — see yolo_common/aug.py")
    ap.add_argument("--imgsz", type=int, default=512, help="screen default 512")
    ap.add_argument("--batch", type=int, default=16, help="screen default 16")
    ap.add_argument("--epochs", type=int, default=200,
                    help="aug delays overfit; 200 + best.pt (default 200)")
    # exp1-style knobs (None → settings/env default)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    model = args.model or S.MODEL
    tag = args.name or f"exp3_{args.aug_level}_{args.imgsz}"
    print(f"\n████ {tag}  (model={model}, imgsz={args.imgsz}, batch={args.batch}, "
          f"epochs={args.epochs}, aug={args.aug_level}, smoke={S.SMOKE}) ████")
    print(f"  aug: {AUG.describe(args.aug_level)}\n")

    conv_rep = inventory_or_halt()

    # Step 1 — frozen split (build once) + materialised positives-only tree.
    split = splits.build_or_load()
    print(f"══ Step 1 — split (seed={split['seed']}) ══")
    print(f"  {json.dumps(split['counts'])}")
    data_yaml = splits.materialise(smoke=S.SMOKE)     # positives only (no negatives)
    print(f"  data.yaml → {data_yaml}\n")

    # Step 2 — train (positives only, aug = the variable, amp OFF, deterministic).
    out_dir = S.RESULTS_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"══ Step 2 — train {model} @ imgsz={args.imgsz}, aug={args.aug_level} ══")
    tr = train_eval.train(data_yaml, args.imgsz, project=out_dir, name="ultralytics",
                          model=args.model, epochs=args.epochs, batch=args.batch,
                          device=args.device, aug_level=args.aug_level,
                          freeze=args.freeze, patience=args.patience,
                          lr0=args.lr0, workers=args.workers)
    print(f"  best.pt → {tr['best_pt']}  ({tr['train_sec']}s)\n")

    # Step 3 — diagnostic eval on the same sealed 360.
    print("══ Step 3 — evaluate on the sealed 360 ══")
    t0 = time.time()
    yolo = train_eval.load_best(tr["best_pt"])
    items = splits.blackbox(split)
    metric_block = metrics.evaluate(yolo, args.imgsz, split, items)
    eval_sec = round(time.time() - t0, 1)

    # Step 4 — assemble metrics.json (exp1 schema) + summary.txt + snapshot.
    report = {
        "experiment": tag,
        "config": {**tr["resolved"], "imgsz": args.imgsz, "seed": S.SEED,
                   "augmentation": tr["resolved"]["aug_level"] != "off", "amp": S.AMP,
                   "aug_kwargs": AUG.AUG_LEVELS[args.aug_level],
                   "conf_thresholds": S.CONF_THRESHOLDS},
        "dataset": {"splits_file": str(S.SPLITS_JSON), "split_seed": split["seed"],
                    "train_positives": split["counts"]["train_positives"],
                    "val_positives": split["counts"]["val_positives"],
                    "blackbox": {"tb": split["counts"]["test_positives"],
                                 "sick_non_tb": split["counts"]["blackbox_sick"],
                                 "healthy": split["counts"]["blackbox_healthy"]},
                    "box_counts": {"active": conv_rep["box_counts"]["ActiveTuberculosis"],
                                   "obsolete": conv_rep["box_counts"]["ObsoletePulmonaryTuberculosis"]},
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


if __name__ == "__main__":
    main()
